"""Stage E2 (Module 7 Part 3, Phase 2B): general visitor measurement backend.

A staged, resumable, UI-free workflow.  Every input comes from the caller (visitor calibration + baseline, four videos, annotations,
confirmations).  This module reads NO original-room data: it does not import ``stage_c_integration`` / ``stage_e1_integration`` /
``validation`` (those load the original annotations / ground truth), never opens the Phase 1 calibration, manifest, annotation seeds,
manual reference frames or saved poses, and holds no ground truth.

Reused UNCHANGED (imports): calibration adapter (A), video validation + TemporaryVideoWorkspace (B), automatic frame selection and
frame extraction (C), annotation backend (D), and the Phase 1 geometry (features.match_set, pose.build_dataset / estimate_pose,
geometry.*, measure.view_points / register_views / measure_wall).

One deliberate orchestration difference (no Phase 1 file is changed): ``measure.reconstruct_geometry`` calls ``geometry.fit_floor_plane``
with its default ``y_min = 2600`` px, which is only meaningful for 3840-px-tall frames.  ``fit_floor_plane`` already takes ``y_min``;
``_reconstruct_set`` mirrors ``reconstruct_geometry`` step for step and passes ``2600 * H / 3840`` (identical at 3840 rows).  A check reports
when the height differs from 3840: the scaled floor region is an unvalidated assumption.

Workflow (each step returns a JSON-serialisable dict; ``export_state`` / ``resume`` allow stopping between steps)
    1 validate_calibration   2 validate_videos   3 select_frames (-> frames + annotation instructions)
    5/6 submit_annotations (refine + validate -> annotation review incl. confirmation requests)
    8 submit_confirmations   9/10 measure (-> dimensions + diagnostics)

Hard limits: ``JobLimits`` time and memory limits are COOPERATIVE -- checked between steps and per frame set, never inside a numeric call;
the memory figure is the process peak RSS (monotonic, process-wide).  They bound the work, they do not guarantee a ceiling.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from . import features, geometry as G, measure as M, pose as P, visitor_annotations as V, visitor_calibration as VC, visitor_frame_selection as FS, visitor_video_validation as VV
from .frames import extract_frames
from .pipeline import to_jsonable
from .visitor_annotations import Status

SCHEMA = "csc8830.module7.visitor_measurement.v1"
TOP_VARIANT = 0
REQUIRED_LINES = V.REQUIRED_LINES
OPTIONAL_LINES = V.OPTIONAL_LINES
ALL_LINES = V.ALL_PHASE1_LINES
VIEW_KEYS = ("AL", "PL", "AR", "PR")
PHASE1_FLOOR_ROW, PHASE1_FRAME_HEIGHT = 2600, 3840        # Phase 1 default row threshold for the floor region and the frame height it was written for
STAGES = ("created", "calibration_validated", "videos_validated", "frames_selected", "annotations_reviewed", "measured")
ACCURACY_NOTE = "Checks establish input consistency and internal geometric consistency only; low residuals do not show that the physical dimensions are accurate. No uncertainty interval is claimed."
CALIBRATION_FIELDS = ("K1", "D1", "K2", "D2", "image_size_wh", "capture_mode", "baseline_mm")
VIDEO_FIELDS = ("placement_1_left", "placement_1_right", "placement_2_left", "placement_2_right")
VIDEO_VIEW = dict(zip(VIDEO_FIELDS, VIEW_KEYS))            # placement_1_left = AL, placement_1_right = PL, placement_2_left = AR, placement_2_right = PR
ANNOTATION_FIELDS = V.REQUIRED_FIELDS + ("frame_fingerprint", "annotation_id")
CONFIRMATION_FIELDS = ("scope", "set_index", "line_name", "annotation_id", "frame_index", "frame_fingerprint", "annotation_digest", "refined_line_digest", "confirmed_codes", "confirmed", "confirmed_by")


# ------------------------------------------------------------------ configuration
@dataclass(frozen=True)
class JobLimits:
    """Cooperative resource limits (see module docstring).  ``None`` disables a limit."""
    video: VV.VideoLimits = VV.DEFAULT_VIDEO_LIMITS          # file size, duration, resolution, fps
    max_frame_sets: int = 8
    max_frame_pixels: int = 40_000_000                       # per decoded frame
    max_processing_s: float | None = 1800.0
    max_working_memory_mb: float | None = 8000.0             # process peak RSS, checked between steps
    max_concurrent_jobs: int = 4                             # per process (class-level slots)


@dataclass(frozen=True)
class QualityLimits:
    """Generic engineering defaults, configurable.  None was fitted to the original room's dimensions or ground truth."""
    min_stereo_verified_warn: int = 100                     # frame-selection evidence (preview matches)
    min_cross_tracks_warn: int = 50
    pose_inlier_fraction_warn: float = 0.90
    residual_rms_px_warn: float = 0.75                      # epipolar / reprojection rms of gated points (the gate itself is 1.5 px)
    gated_fraction_warn: float = 0.80
    gated_fraction_fail: float = 0.30
    min_gated_points_fail: int = 50
    floor_rms_mm_warn: float = 10.0
    floor_rms_mm_fail: float = 40.0
    floor_tilt_deg_warn: float = 45.0                       # angle between the floor normal and the left camera's vertical axis
    floor_camera_height_mm: tuple = (200.0, 5000.0)         # |floor offset| outside this range -> WARNING
    vertical_error_deg_warn: float = 5.0
    vertical_error_deg_fail: float = 15.0
    skew_mm_warn: float = 30.0
    skew_mm_fail: float = 150.0
    corner_ceiling_angle_deg_warn: float = 35.0
    registration_min_inliers_fail: int = 20
    registration_min_inliers_warn: int = 40
    registration_min_inlier_fraction_fail: float = 0.5
    registration_min_inlier_fraction_warn: float = 0.8
    registration_rms_mm_warn: float = 10.0
    registration_rms_mm_fail: float = M.REG_THR_MM
    registration_bbox_fraction_warn: float = 0.15           # bounding box of the registration pairs / image area
    registration_bbox_fraction_fail: float = 0.03
    registration_min_cells_warn: int = 3                    # of a 3x3 grid
    pose_yaw_range_warn_deg: float = 0.03                   # across leave-one-set-out poses (Stage C: 0.02-0.04 deg moved dimensions 15-30 mm)
    pose_dimension_range_rel_warn: float = 0.005
    set_spread_rel_warn: float = 0.02                       # relative sd across frame sets (descriptive, NOT an uncertainty)
    max_dimension_mm: float = 100_000.0


# ------------------------------------------------------------------ small helpers
def _peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1 << 20) if sys.platform == "darwin" else r / 1024, 1)
    except Exception:                                         # pragma: no cover
        return None


def issue(stage: str, code: str, message: str, **extra) -> dict:
    return dict(stage=stage, code=code, message=message, **extra)


def check(name, status: Status, message, value=None, threshold=None, required=True) -> dict:
    return dict(name=name, status=status.value, required=required, value=value, threshold=threshold, message=message)


def _canon(o): return json.dumps(to_jsonable(o), sort_keys=True, separators=(",", ":"), default=str)
def annotation_digest(raw: Mapping) -> str: return hashlib.sha256(_canon(raw).encode()).hexdigest()


def line_digest(line: Mapping | None) -> str | None:
    if line is None: return None
    return hashlib.sha256(_canon(dict(kind=line["kind"], a=f"{float(line['a']):.6f}", b=f"{float(line['b']):.6f}", span=[f"{float(x):.4f}" for x in line["span"]])).encode()).hexdigest()


def lines_digest(lines: Mapping) -> str: return hashlib.sha256(_canon({n: line_digest(l) for n, l in sorted(lines.items())}).encode()).hexdigest()


class JobLimitExceeded(RuntimeError):
    pass


class _Stop(Exception):
    """Internal: stop the current step with a structured failure."""
    def __init__(self, failure): self.failure = failure


# ------------------------------------------------------------------ confirmation contract
@dataclass(frozen=True)
class Confirmation:
    """The visitor's explicit acceptance of specific warnings.  Bound to the exact annotation, refined line, frame fingerprint and codes.
    scope 'annotation': one boundary annotation.  scope 'set_issue': one set-level warning (``line_name`` = ``CODE:subject``), bound to the
    exact set of refined required lines.  ``confirmed`` must be literally True (a False record is a declined confirmation)."""
    scope: str
    set_index: int
    line_name: str
    annotation_id: str
    frame_index: int | None
    frame_fingerprint: str | None
    annotation_digest: str
    refined_line_digest: str
    confirmed_codes: tuple
    confirmed: bool
    confirmed_by: str
    reason: str = ""

    def to_dict(self): return to_jsonable(asdict(self))

    @classmethod
    def from_request(cls, request: Mapping, confirmed_by: str, reason: str = "", confirmed: bool = True) -> "Confirmation":
        """Build the record answering a ``confirmation_request`` returned by the review step (the explicit act is the call itself)."""
        return cls(scope=request["scope"], set_index=request["set_index"], line_name=request["line_name"], annotation_id=request["annotation_id"], frame_index=request["frame_index"],
                   frame_fingerprint=request["frame_fingerprint"], annotation_digest=request["annotation_digest"], refined_line_digest=request["refined_line_digest"],
                   confirmed_codes=tuple(request["warning_codes"]), confirmed=confirmed, confirmed_by=confirmed_by, reason=reason)


def parse_confirmation(d) -> tuple:
    """-> (Confirmation | None, problems).  Missing identity information is rejected, never defaulted."""
    if isinstance(d, Confirmation): return d, []
    if not isinstance(d, Mapping): return None, ["a confirmation must be a mapping"]
    missing = [k for k in CONFIRMATION_FIELDS if k not in d or (d[k] is None and k not in ("frame_index", "frame_fingerprint"))]
    if missing: return None, [f"missing required field(s): {', '.join(missing)}"]
    bad = []
    if d["scope"] not in ("annotation", "set_issue"): bad.append("scope must be 'annotation' or 'set_issue'")
    if not isinstance(d["confirmed"], bool): bad.append("confirmed must be a boolean")
    if d["scope"] == "annotation" and (not d["frame_fingerprint"] or d["frame_index"] is None): bad.append("an annotation confirmation needs frame_index and frame_fingerprint")
    if not isinstance(d["confirmed_codes"], (list, tuple)) or not all(isinstance(c, str) for c in d["confirmed_codes"]): bad.append("confirmed_codes must be a list of warning codes")
    if not isinstance(d["set_index"], int) or isinstance(d["set_index"], bool): bad.append("set_index must be an integer")
    if bad: return None, bad
    return Confirmation(scope=d["scope"], set_index=d["set_index"], line_name=str(d["line_name"]), annotation_id=str(d["annotation_id"]), frame_index=d["frame_index"], frame_fingerprint=d["frame_fingerprint"],
                        annotation_digest=str(d["annotation_digest"]), refined_line_digest=str(d["refined_line_digest"]), confirmed_codes=tuple(d["confirmed_codes"]), confirmed=d["confirmed"],
                        confirmed_by=str(d["confirmed_by"]), reason=str(d.get("reason", ""))), []


def confirmation_problems(conf: Confirmation, ctx: Mapping) -> list:
    bad = [f"{k} differs from the current value (the annotation / refined line / frame changed, or the confirmation belongs to something else)"
           for k in ("scope", "set_index", "line_name", "annotation_id", "frame_index", "frame_fingerprint", "annotation_digest", "refined_line_digest") if getattr(conf, k) != ctx[k]]
    if set(conf.confirmed_codes) != set(ctx["warning_codes"]): bad.append(f"confirmed codes {sorted(conf.confirmed_codes)} != current warning codes {sorted(ctx['warning_codes'])}")
    if conf.confirmed is not True: bad.append("the visitor declined this confirmation")
    return bad


# ------------------------------------------------------------------ calibration input contract
def parse_calibration_input(d) -> tuple:
    """-> (stage-A model dict | None, baseline_mm | None, failures).  Capture-mode provenance is REQUIRED for both cameras."""
    if not isinstance(d, Mapping): return None, None, [issue("calibration", "INPUT_NOT_A_MAPPING", "the calibration input must be a mapping")]
    missing = [k for k in CALIBRATION_FIELDS if d.get(k) is None]
    if missing: return None, None, [issue("calibration", "INPUT_CONTRACT_MISSING_FIELD", f"missing required calibration field(s): {', '.join(missing)}", fields=missing)]
    cm = d["capture_mode"]
    if not (isinstance(cm, Mapping) and isinstance(cm.get("left"), str) and isinstance(cm.get("right"), str) and cm["left"].strip() and cm["right"].strip()):
        return None, None, [issue("calibration", "CAPTURE_MODE_PROVENANCE_REQUIRED", "capture_mode must declare 'left' and 'right' capture modes (e.g. 'video'); it is never inferred")]
    sz = d["image_size_wh"]
    try: sz = list(sz); ok = len(sz) == 2
    except TypeError: ok = False
    if not ok: return None, None, [issue("calibration", "INVALID_IMAGE_SIZE", "image_size_wh must be [width, height]")]
    model = dict(K1=d["K1"], D1=d["D1"], K2=d["K2"], D2=d["D2"], image_size_left=sz, image_size_right=sz, camera=dict(left=dict(capture_mode=cm["left"]), right=dict(capture_mode=cm["right"])))
    return model, d["baseline_mm"], []


def parse_video_input(d) -> tuple:
    if not isinstance(d, Mapping): return None, [issue("videos", "INPUT_NOT_A_MAPPING", "the video input must be a mapping")]
    missing = [k for k in VIDEO_FIELDS if d.get(k) is None or (isinstance(d.get(k), str) and not d[k])]
    if missing: return None, [issue("videos", "INPUT_CONTRACT_MISSING_FIELD", f"missing video(s): {', '.join(missing)}", fields=missing)]
    return {k: d[k] for k in VIDEO_FIELDS}, []


def annotation_contract_problems(raw) -> list:
    """Identity information every annotation must carry (Stage D checks the rest)."""
    if not isinstance(raw, Mapping): return ["an annotation must be a mapping"]
    miss = [k for k in ANNOTATION_FIELDS if raw.get(k) is None or raw.get(k) == ""]
    return [f"missing required field(s): {', '.join(miss)}"] if miss else []


# ------------------------------------------------------------------ Stage D evaluation + gate
def _identity(raw, res, fps, idx, size_by_view, cal_size, set_index) -> dict:
    """Independent audit against the SERVER's frame: set / frame index / placement / camera / size / fingerprint.  Client fingerprints are never trusted."""
    md = res.metadata; vk = md.get("view_key")
    if vk is None or vk not in fps: return dict(ok=False, reasons=["annotation could not be associated with a served frame"], checks={})
    fp = raw.get("frame_fingerprint") if isinstance(raw, Mapping) else None
    chk = dict(set_index=md["set_index"] == set_index, frame_index=md["frame_index"] == idx.get(vk), placement_camera=V.VIEW_KEY.get((md["placement"], md["camera"])) == vk,
               image_size=tuple(md["image_wh"]) == size_by_view[vk], calibration_size=tuple(cal_size) == size_by_view[vk], fingerprint_present=bool(fp), fingerprint_matches=bool(fp) and fp == fps[vk])
    msg = dict(set_index="wrong set", frame_index="annotation frame index differs from the served frame", placement_camera="placement/camera does not match the view", image_size="annotation image size differs from the served frame",
               calibration_size="frame size differs from the calibration size", fingerprint_present="no frame_fingerprint", fingerprint_matches="frame fingerprint differs from the server's frame")
    reasons = [msg[k] for k, ok in chk.items() if not ok]
    return dict(ok=not reasons, reasons=reasons, checks=chk, frame_fingerprint=fps[vk], annotation_fingerprint=fp)


def evaluate_set(raw_list: Sequence, frames: Mapping, fps: Mapping, idx: Mapping, cal, set_index: int, limits: V.AnnotationLimits) -> dict:
    """Stage D on one set (no pose: annotation stereo consistency stays NOT_EVALUATED).  ``fps`` are the server's fingerprints of ``frames``."""
    raw_list = list(raw_list); sr = V.process_annotation_set(raw_list, frames, cal, limits, set_index=set_index, expected_frame_indices=idx, pose=None)
    size = {k: (int(f.shape[1]), int(f.shape[0])) for k, f in frames.items()}; entries = []
    for i, (raw, res) in enumerate(zip(raw_list, sr.annotations)):
        entries.append(dict(index=i, raw=raw, result=res, line_name=res.line_name, status=res.status, warning_codes=sorted(set(res.warning_codes)), failure_codes=res.failure_codes,
                            identity=_identity(raw, res, fps, idx, size, cal.image_size, set_index), annotation_digest=annotation_digest(raw), refined_line=res.downstream_line, refined_line_digest=line_digest(res.downstream_line)))
    return dict(set_index=set_index, set_result=sr, entries=entries)


def _group(ev) -> dict:
    g = {}
    for e in ev["entries"]:
        if e["line_name"]: g.setdefault(e["line_name"], []).append(e)
    return g


def _ann_ctx(ev, e) -> dict:
    return dict(scope="annotation", set_index=ev["set_index"], line_name=e["line_name"], annotation_id=e["result"].annotation_id, frame_index=e["result"].metadata.get("frame_index"),
                frame_fingerprint=e["identity"].get("annotation_fingerprint"), annotation_digest=e["annotation_digest"], refined_line_digest=e["refined_line_digest"], warning_codes=e["warning_codes"])


def gate_set(ev: dict, confirmations: Sequence[Confirmation]) -> dict:
    """What may reach the Phase 1 geometry.  PASS -> allowed; WARNING -> only with an exactly matching explicit confirmation; FAIL -> blocked
    (a confirmation cannot override it); NOT_EVALUATED annotation status -> blocked.  Returns usable lines, per-line decisions, failures,
    warnings and the confirmation requests still needed."""
    sr = ev["set_result"]; si = ev["set_index"]; by_name = _group(ev); decisions, usable, failures, warnings, applied, requests = {}, {}, [], [], [], []
    confs = list(confirmations)

    def find(ctx):
        cands = [c for c in confs if c.scope == ctx["scope"] and c.set_index == ctx["set_index"] and c.line_name == ctx["line_name"]]
        if not cands: return None, ["no confirmation recorded for this annotation"]
        probs = [confirmation_problems(c, ctx) for c in cands]
        for c, p in zip(cands, probs):
            if not p: return c, []
        return None, probs[-1]

    for nm in ALL_LINES:
        required = nm in REQUIRED_LINES; es = by_name.get(nm, [])
        d = dict(line=nm, required=required, annotated=bool(es), stage_d_status=None, decision=None, reasons=[], warning_codes=[], failure_codes=[], checks_not_evaluated=[])
        if not es: d.update(decision="NOT_ANNOTATED", stage_d_status=Status.NOT_EVALUATED.value, reasons=["no annotation supplied"])
        elif len(es) > 1: d.update(decision="BLOCKED_DUPLICATE", stage_d_status=Status.FAIL.value, reasons=[f"{len(es)} annotations for the same boundary; none is used"])
        else:
            e = es[0]; res = e["result"]
            d.update(stage_d_status=e["status"].value, warning_codes=e["warning_codes"], failure_codes=e["failure_codes"], identity=e["identity"], annotation_id=res.annotation_id, frame_index=res.metadata.get("frame_index"),
                     frame_fingerprint=e["identity"].get("annotation_fingerprint"), annotation_digest=e["annotation_digest"], refined_line_digest=e["refined_line_digest"],
                     checks_not_evaluated=["stereo_consistency"] if res.stereo_consistency is Status.NOT_EVALUATED and res.line_name.split("_")[1] == "corner" else [])
            if not e["identity"]["ok"]: d.update(decision="BLOCKED_IDENTITY", reasons=e["identity"]["reasons"])
            elif e["status"] is Status.FAIL: d.update(decision="BLOCKED_FAIL", reasons=[f"blocking failure(s): {e['failure_codes']}"])
            elif e["status"] is Status.NOT_EVALUATED: d.update(decision="BLOCKED_NOT_EVALUATED", reasons=["annotation status NOT_EVALUATED is not treated as PASS"])
            elif e["refined_line"] is None: d.update(decision="BLOCKED_NO_REFINED_LINE", reasons=["no refined line available; the unrefined click line is never used"])
            elif e["status"] is Status.PASS: d.update(decision="ALLOWED"); usable[nm] = e["refined_line"]
            else:
                ctx = _ann_ctx(ev, e); conf, probs = find(ctx)
                if conf is None:
                    d.update(decision="BLOCKED_UNCONFIRMED_WARNING", confirmation_valid=False, reasons=probs if probs and isinstance(probs[0], str) else [x for p in probs for x in p])
                    requests.append(dict(ctx, required=required, message=f"{nm}: " + "; ".join(i.message for i in res.issues if i.status is Status.WARNING), previous_confirmation_problems=d["reasons"]))
                else: d.update(decision="ALLOWED_CONFIRMED", confirmation_valid=True, reasons=[f"warning(s) {e['warning_codes']} explicitly confirmed by {conf.confirmed_by!r}"]); usable[nm] = e["refined_line"]; applied.append(conf.to_dict())
            if e["status"] is Status.FAIL and any(c.scope == "annotation" and c.set_index == si and c.line_name == nm for c in confs): d["confirmation_ignored"] = "a blocking failure cannot be overridden by a confirmation"
        decisions[nm] = d; ok = d["decision"] in ("ALLOWED", "ALLOWED_CONFIRMED")
        if required and not ok: failures.append(issue("annotations", "REQUIRED_ANNOTATION_MISSING" if d["decision"] == "NOT_ANNOTATED" else "REQUIRED_ANNOTATION_BLOCKED",
                                                       f"set {si} required line {nm}: {d['decision']}: {'; '.join(d['reasons'])}", line=nm, set_index=si, decision=d["decision"]))
        if not required and es and not ok: warnings.append(issue("annotations", "OPTIONAL_LINE_NOT_USED", f"optional line {nm} not used ({d['decision']}); the six-line measurement does not need it", line=nm, set_index=si))
    for e in ev["entries"]:
        if e["line_name"] not in ALL_LINES: warnings.append(issue("annotations", "UNASSOCIATED_ANNOTATION", f"annotation {e['result'].annotation_id!r} could not be associated with a boundary and is ignored", set_index=si, failure_codes=e["failure_codes"]))
    for i in sr.issues:
        if i.status is Status.FAIL and i.subject not in OPTIONAL_LINES and i.code not in (V.Code.MISSING_REQUIRED, V.Code.REQUIRED_INVALID):
            failures.append(issue("annotations", i.code, f"set {si} set-level failure: {i.message}", set_index=si, subject=i.subject))
    req_usable = {n: usable[n] for n in REQUIRED_LINES if n in usable}
    for i in sr.issues:
        if i.status is Status.WARNING and i.subject not in OPTIONAL_LINES and i.code not in (V.Code.OPTIONAL_REJECTED, V.Code.UNKNOWN_LINE):
            ctx = dict(scope="set_issue", set_index=si, line_name=f"{i.code}:{i.subject}", annotation_id=f"set{si}:{i.code}:{i.subject}", frame_index=None, frame_fingerprint=None, annotation_digest="",
                       refined_line_digest=lines_digest(req_usable), warning_codes=[i.code]); conf, probs = find(ctx)
            if conf is None:
                requests.append(dict(ctx, required=True, message=i.message)); failures.append(issue("annotations", "SET_WARNING_UNCONFIRMED", f"set {si}: warning {i.code} ({i.subject}) needs explicit confirmation: {i.message}", set_index=si))
            else: applied.append(conf.to_dict()); warnings.append(issue("annotations", i.code, f"confirmed set-level warning: {i.message}", set_index=si, confirmed=True))
        elif i.status is Status.WARNING: warnings.append(issue("annotations", i.code, i.message, set_index=si, subject=i.subject))
    return dict(set_index=si, ok=not failures, usable_lines=usable, decisions=decisions, failures=failures, warnings=warnings, confirmations_applied=applied, confirmation_requests=requests,
                stage_d_set_status=sr.status.value, stage_d_failure_codes=sr.failure_codes, reconstruction=to_jsonable(sr.reconstruction), completeness=to_jsonable(sr.completeness),
                stereo_consistency={k: v.get("status") for k, v in sr.stereo_consistency.items()}, stereo_consistency_note="NOT_EVALUATED before pose estimation; evaluated after pose as 'stereo_consistency_post_pose'")


# ------------------------------------------------------------------ geometry (Phase 1 functions; floor row scaled to the frame height)
class _GeomError(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


def floor_row_threshold(height_px: int) -> float:
    return PHASE1_FLOOR_ROW * height_px / PHASE1_FRAME_HEIGHT


def _gate(cal, R, T, a, b):
    X = G.triangulate(cal, R, T, a, b); e1, e2 = G.reprojection_errors(cal, R, T, X, a, b); d = G.epipolar_distances(cal, R, T, a, b)
    g = (X[:, 2] > M.DEPTH_RANGE_MM[0]) & (X[:, 2] < M.DEPTH_RANGE_MM[1]) & (e1 < M.GATE_PX) & (e2 < M.GATE_PX) & (d < M.GATE_PX)
    return X, g, e1, e2, d


def _reconstruct_set(cal, R, T, F, rng, y_min, image_wh):
    """Mirror of ``measure.reconstruct_geometry`` (same Phase 1 calls, same order) plus residual / coverage diagnostics; floor row threshold explicit."""
    diag = {}; pts = {}
    for v, (a, b, mt) in (("view1", ("A1", "B1", "M1")), ("view2", ("A2", "B2", "M2"))):
        try: X, g, e1, e2, d = _gate(cal, R, T, F[a], F[b])
        except Exception as ex: raise _GeomError("TRIANGULATION_FAILED", f"{v}: {type(ex).__name__}: {ex}")
        n = int(g.sum())
        if n < 10: raise _GeomError("TRIANGULATION_FAILED", f"{v}: only {n} of {len(X)} stereo matches triangulate within the depth range and the {M.GATE_PX} px gates")
        floor_sel = g & (F[a][:, 1] > y_min)
        diag[v] = dict(n_stereo_matches=int(len(X)), n_gated_points=n, gated_fraction=n / max(len(X), 1), reprojection_rms_px=float(np.sqrt(np.mean(np.r_[e1[g], e2[g]] ** 2))), epipolar_rms_px=float(np.sqrt(np.mean(d[g] ** 2))),
                       median_depth_mm=float(np.median(X[g, 2])), n_floor_candidates=int(floor_sel.sum()), floor_plane_rms_mm=None)
        pts[v] = (X[g], F[a][g], F[mt][g])
    out = {}
    for v, a_key in (("view1", "A1"), ("view2", "A2")):
        X, a, mt = pts[v]
        try:
            fl = G.fit_floor_plane(X, a, y_min=y_min)
        except Exception as ex: raise _GeomError("FLOOR_PLANE_FAILED", f"{v}: floor plane could not be fitted ({type(ex).__name__}: {ex}); the lower part of the frame (rows > {y_min:.0f}) must show the floor")
        if diag[v]["n_floor_candidates"] < 10 or not all(np.isfinite(np.asarray(fl[0], float))) or not np.isfinite(fl[1]):
            raise _GeomError("FLOOR_PLANE_FAILED", f"{v}: only {diag[v]['n_floor_candidates']} floor points below row {y_min:.0f}; the floor plane is not determined")
        sel = a[:, 1] > y_min; diag[v]["floor_plane_rms_mm"] = float(np.sqrt(np.mean((X[sel] @ fl[0] + fl[1]) ** 2))) if sel.sum() >= 10 else None
        ang = float(np.degrees(np.arccos(min(1.0, abs(float(fl[0] @ np.array([0.0, 1.0, 0.0])))))))
        diag[v].update(floor_tilt_deg=ang, floor_offset_mm=float(fl[1])); out[v] = fl
    (X1, a1, i1), (X2, a2, i2) = pts["view1"], pts["view2"]
    try: R0, t0, rms, ni, nt = M.register_views(X1, i1[:, 0], X2, i2[:, 0], F["MX"], rng)
    except Exception as ex: raise _GeomError("REGISTRATION_FAILED", f"{type(ex).__name__}: {ex}")
    d1 = {k: i for i, k in enumerate(i1[:, 0])}; d2 = {k: i for i, k in enumerate(i2[:, 0])}; pr = [d1[q] for q, w in F["MX"] if q in d1 and w in d2]
    W_, H_ = image_wh; xy = a1[pr] if pr else np.empty((0, 2))
    cov = dict(n_pairs=int(len(xy)), bbox_fraction=float((np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])) / (W_ * H_)) if len(xy) > 1 else 0.0,
               cells_occupied=int(len({(min(int(x / W_ * 3), 2), min(int(y / H_ * 3), 2)) for x, y in xy})), cells_total=9,
               note="coverage of ALL registration pairs in the placement-1 left image (inlier locations are not exposed by the Phase 1 registration)")
    return dict(f1=out["view1"], f2=out["view2"], R0=R0, t0=t0, registration=dict(rms_mm=rms, n_inliers=ni, n_pairs=nt), n_points=(len(X1), len(X2))), diag, cov


def _measure_sets(cal, p5, match_sets, line_sets, image_wh, seed=3):
    """Mirror of ``measure.measure_all_sets`` (top_variant 0).  Failures are recorded with a code, never hidden."""
    R, T = G.pose_from_params(p5, cal.baseline_mm); rows = []; y_min = floor_row_threshold(image_wh[1])
    for i, (F, L) in enumerate(zip(match_sets, line_sets)):
        try:
            g, diag, cov = _reconstruct_set(cal, R, T, F, np.random.default_rng(seed + i), y_min, image_wh)
            try: o = M.measure_wall(cal, R, T, g, L, TOP_VARIANT)
            except Exception as ex: raise _GeomError("WALL_RECONSTRUCTION_FAILED", f"{type(ex).__name__}: {ex}")
            rows.append(dict(ok=True, width_mm=o["width_mm"], height_mm=o["height_mm"], H1=o["V1"]["H"], H2=o["V2"]["H"], vertical_error_deg=(o["V1"]["vertical_error_deg"], o["V2"]["vertical_error_deg"]),
                             skew_mm=(o["V1"]["skew"], o["V2"]["skew"]), registration=g["registration"], geometry=diag, coverage=cov, bottom_corners=dict(V1=o["V1"]["B"], V2=o["V2"]["B"])))
        except _GeomError as e: rows.append(dict(ok=False, code=e.code, error=str(e), width_mm=float("nan"), height_mm=float("nan")))
        except Exception as e: rows.append(dict(ok=False, code="GEOMETRY_FAILED", error=f"{type(e).__name__}: {e}", width_mm=float("nan"), height_mm=float("nan")))
    return rows


# ------------------------------------------------------------------ the job
class VisitorMeasurementJob:
    """One visitor measurement.  Owns an isolated ``TemporaryVideoWorkspace`` (unless one is supplied); ``close()`` removes only that directory."""
    _slots = 0
    _slot_lock = threading.Lock()

    def __init__(self, limits: JobLimits = JobLimits(), quality: QualityLimits = QualityLimits(), annotation_limits: V.AnnotationLimits = V.DEFAULT_LIMITS,
                 selection_config: FS.SelectionConfig | None = None, workspace: VV.TemporaryVideoWorkspace | None = None):
        with VisitorMeasurementJob._slot_lock:
            if VisitorMeasurementJob._slots >= limits.max_concurrent_jobs: raise JobLimitExceeded(f"{VisitorMeasurementJob._slots} jobs are already active (limit {limits.max_concurrent_jobs})")
            VisitorMeasurementJob._slots += 1
        self._slot_held = True; self.limits, self.quality, self.annotation_limits, self.selection_config = limits, quality, annotation_limits, selection_config
        self._owns_ws = workspace is None; self.ws = workspace or VV.TemporaryVideoWorkspace(prefix="visitor_job_"); self.t_start = time.perf_counter()
        self.stage = "created"; self.timings = {}; self.cal_model = None; self.baseline_mm = None; self.cal = None; self.compat = None; self.video_validation = None
        self.video_paths = {}; self.selection = None; self.n_sets = None; self.frames = []; self.annotations = []; self.confirmations = []; self.rejected_confirmations = []
        self._eval_key = None; self._evals = None; self.last_result = None; self._rejected_records = []; self._vr = None; self._adapted = None

    # ---- lifecycle
    def close(self):
        if self._owns_ws: self.ws.close()
        if self._slot_held:
            with VisitorMeasurementJob._slot_lock: VisitorMeasurementJob._slots -= 1
            self._slot_held = False

    def __enter__(self): return self
    def __exit__(self, *exc): self.close(); return False

    @classmethod
    def active_jobs(cls) -> int: return cls._slots

    # ---- plumbing
    def _checkpoint(self, where):
        el = time.perf_counter() - self.t_start; lim = self.limits
        if lim.max_processing_s is not None and el > lim.max_processing_s: raise _Stop(issue("limits", "TIME_LIMIT_EXCEEDED", f"processing time {el:.0f} s exceeded the {lim.max_processing_s:.0f} s limit (checked at '{where}'; limits are cooperative)"))
        pk = _peak_rss_mb()
        if lim.max_working_memory_mb is not None and pk is not None and pk > lim.max_working_memory_mb: raise _Stop(issue("limits", "MEMORY_LIMIT_EXCEEDED", f"process peak RSS {pk:.0f} MB exceeded the {lim.max_working_memory_mb:.0f} MB limit (checked at '{where}'; limits are cooperative)"))

    def _need(self, stage):
        if STAGES.index(self.stage) < STAGES.index(stage): raise _Stop(issue("workflow", "STAGE_OUT_OF_ORDER", f"this step needs the '{stage}' stage first; the job is at '{self.stage}'", required_next_action=self._next_action()))

    def _next_action(self):
        return {"created": "submit the calibration (validate_calibration)", "calibration_validated": "submit the four videos (validate_videos)", "videos_validated": "select frames (select_frames)",
                "frames_selected": "annotate the displayed frames and submit them (submit_annotations)", "annotations_reviewed": "confirm the listed warnings (submit_confirmations) or fix blocked annotations, then measure",
                "measured": "done"}[self.stage]

    def _result(self, step, status, failures=(), warnings=(), **sections):
        failures = list(failures)
        res = dict(schema=SCHEMA, step=step, stage=self.stage, overall_status=status, can_proceed=status != "FAIL" and not failures, required_next_action=self._next_action() if not failures and status != "FAIL" else (failures[0].get("required_next_action") or "fix the listed failures and repeat this step"),
                   failures=failures, warnings=list(warnings), accuracy_note=ACCURACY_NOTE, top_variant=TOP_VARIANT, ground_truth_used=False, timings_s={k: round(v, 3) for k, v in self.timings.items()},
                   runtime_s=round(time.perf_counter() - self.t_start, 3), peak_rss_mb=_peak_rss_mb(), limits_note="time / memory limits are cooperative checkpoints, not hard ceilings", **sections)
        res = to_jsonable(res); self.last_result = res; return res

    def _fail_result(self, step, failure, **sections): return self._result(step, "FAIL", [failure] if isinstance(failure, dict) else failure, **sections)

    def _timed(self, name, t0): self.timings[name] = self.timings.get(name, 0.0) + time.perf_counter() - t0

    def _invalidate_from(self, stage):
        """Discard everything produced by ``stage`` and later steps (a changed earlier input invalidates later work); lower the job stage accordingly."""
        k = STAGES.index(stage)
        if k <= STAGES.index("videos_validated"): self.video_paths = {}; self.video_validation = None; self._vr = None
        if k <= STAGES.index("frames_selected"): self.selection = None; self.frames = []; self.n_sets = None
        if k <= STAGES.index("annotations_reviewed"): self.annotations = []; self.confirmations = []; self.rejected_confirmations = []; self._rejected_records = []; self._evals = None; self._eval_key = None
        if STAGES.index(self.stage) >= k: self.stage = STAGES[k - 1]

    def _frame_path(self, set_index, vk): return Path(self.ws.path) / "frames" / f"set{set_index:02d}_{vk}.png"

    def _load_set(self, set_index):
        """Server-side frames of one set from the workspace, re-verified against the stored fingerprints (tamper / corruption check)."""
        fr, fps = {}, {}
        for vk in VIEW_KEYS:
            rec = self.frames[set_index]["views"][vk]; im = cv2.imread(str(self._frame_path(set_index, vk)), cv2.IMREAD_GRAYSCALE)
            if im is None: raise _Stop(issue("frames", "FRAME_MISSING", f"set {set_index} {vk}: the stored frame is missing", set_index=set_index))
            fp = V.frame_fingerprint(im)
            if fp != rec["frame_fingerprint"]: raise _Stop(issue("frames", "FRAME_INTEGRITY_FAILED", f"set {set_index} {vk}: the stored frame no longer matches its fingerprint", set_index=set_index))
            fr[vk], fps[vk] = im, fp
        return fr, fps

    # ---- step 1
    def validate_calibration(self, calibration: Mapping) -> dict:
        t0 = time.perf_counter()
        try:
            model, baseline, fails = parse_calibration_input(calibration)
            if fails: return self._fail_result("validate_calibration", fails, calibration=None)
            ad = VC.adapt_visitor_calibration(model, baseline, name="visitor_stereo")
            rec = ad.to_dict(); rec.update(accuracy_verified=False, note=VC.ACCURACY_NOTE, baseline_source="visitor-entered physical baseline")
            if not ad.ok: return self._fail_result("validate_calibration", issue("calibration", "CALIBRATION_INVALID", "; ".join(i.message for i in ad.errors), codes=[i.code for i in ad.errors]), calibration=rec)
            self._invalidate_from("videos_validated"); self.cal_model, self.baseline_mm, self.cal, self._adapted = model, baseline, ad.calibration, ad; self.stage = "calibration_validated"
            return self._result("validate_calibration", "WARNING" if ad.warnings else "PASS", warnings=[issue("calibration", i.code, i.message) for i in ad.warnings], calibration=rec)
        except _Stop as s: return self._fail_result("validate_calibration", s.failure)
        finally: self._timed("validate_calibration", t0)

    # ---- step 2
    def validate_videos(self, videos: Mapping) -> dict:
        t0 = time.perf_counter()
        try:
            self._need("calibration_validated"); parsed, fails = parse_video_input(videos)
            if fails: return self._fail_result("validate_videos", fails, videos=None)
            paths = {}
            for k, v in parsed.items():                                           # bytes are staged in the isolated workspace; paths are only ever read
                if isinstance(v, (bytes, bytearray)): paths[k] = str(self.ws.write(bytes(v)))
                elif isinstance(v, (str, Path)): paths[k] = str(v)
                else: return self._fail_result("validate_videos", issue("videos", "INVALID_VIDEO_INPUT", f"{k}: expected a path or bytes"), videos=None)
            vr = VV.validate_visitor_videos(paths["placement_1_left"], paths["placement_1_right"], paths["placement_2_left"], paths["placement_2_right"], self._adapted, self.limits.video)
            rec = vr.to_dict(); compat = vr.calibration_compatibility; fails = []
            if not vr.can_proceed: fails.append(issue("video_validation", "VIDEO_VALIDATION_BLOCKING", "; ".join(f"[{i.code}] {i.message}" for i in vr.errors), codes=[i.code for i in vr.errors]))
            if compat is None or compat.status.value != "COMPATIBLE_METADATA": fails.append(issue("video_validation", "CALIBRATION_VIDEO_INCOMPATIBLE", f"calibration/video compatibility is {None if compat is None else compat.status.value}; it must be COMPATIBLE_METADATA (video-mode calibration at the video resolution and orientation)"))
            for r in vr.videos.values():
                if r.width and r.height and r.width * r.height > self.limits.max_frame_pixels: fails.append(issue("videos", "FRAME_TOO_LARGE", f"{r.label}: {r.width}x{r.height} exceeds the {self.limits.max_frame_pixels} pixel limit"))
            self._checkpoint("validate_videos")
            if fails: return self._fail_result("validate_videos", fails, video_diagnostics=rec)
            self._invalidate_from("videos_validated"); self.video_paths = paths; self.video_validation = rec; self.stage = "videos_validated"; self._vr = vr
            return self._result("validate_videos", "WARNING" if vr.warnings else "PASS", warnings=[issue("video_validation", i.code, i.message) for i in vr.warnings], video_diagnostics=rec)
        except _Stop as s: return self._fail_result("validate_videos", s.failure)
        finally: self._timed("validate_videos", t0)

    # ---- steps 3-4
    def select_frames(self, n_sets: int = 3) -> dict:
        t0 = time.perf_counter()
        try:
            self._need("videos_validated")
            if not isinstance(n_sets, int) or isinstance(n_sets, bool) or not 1 <= n_sets <= self.limits.max_frame_sets: return self._fail_result("select_frames", issue("frame_selection", "INVALID_FRAME_SET_COUNT", f"n_sets must be an integer from 1 to {self.limits.max_frame_sets}"))
            wh = tuple(self.cal.image_size); est_mb = n_sets * 4 * wh[0] * wh[1] / 1e6
            if self.limits.max_working_memory_mb is not None and est_mb > self.limits.max_working_memory_mb / 2: return self._fail_result("select_frames", issue("limits", "MEMORY_ESTIMATE_EXCEEDED", f"extracting {n_sets} sets needs about {est_mb:.0f} MB of frames; the limit allows {self.limits.max_working_memory_mb / 2:.0f}"))
            vp = [self.video_paths[k] for k in VIDEO_FIELDS]
            sel = FS.select_frame_sets(*vp, n_sets=n_sets, config=self.selection_config, validation=self._vr); sel_d = sel.to_dict()
            if not sel.can_proceed: return self._fail_result("select_frames", [issue("frame_selection", i["code"], i["message"], subject=i.get("subject")) for i in sel_d["errors"]] or issue("frame_selection", "SELECTION_FAILED", "selection failed"), frame_selection=sel_d)
            self._checkpoint("select_frames")
            ext = FS.extract_selected_frames(sel, *vp); Path(self.ws.path, "frames").mkdir(exist_ok=True); frames = []
            for i, (s, e) in enumerate(zip(sel.sets, ext)):
                idx = dict(AL=s.placement_1.left_frame_index, PL=s.placement_1.right_frame_index, AR=s.placement_2.left_frame_index, PR=s.placement_2.right_frame_index)
                src = dict(AL=e["placement_1"]["left"], PL=e["placement_1"]["right"], AR=e["placement_2"]["left"], PR=e["placement_2"]["right"]); views = {}
                for vk in VIEW_KEYS:
                    im = src[vk]
                    if (im.shape[1], im.shape[0]) != wh: return self._fail_result("select_frames", issue("frames", "FRAME_SIZE_MISMATCH", f"set {i} {vk}: decoded {im.shape[1]}x{im.shape[0]} but the calibration is {wh[0]}x{wh[1]}"), frame_selection=sel_d)
                    p = self._frame_path(i, vk); cv2.imwrite(str(p), im, [cv2.IMWRITE_PNG_COMPRESSION, 1]); fp = V.frame_fingerprint(im)
                    back = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if back is None or V.frame_fingerprint(back) != fp: return self._fail_result("select_frames", issue("frames", "FRAME_STORAGE_FAILED", f"set {i} {vk}: the stored frame does not round-trip"))
                    pl, cam = V.VIEW_PLACEMENT[vk]; views[vk] = dict(set_index=i, view_key=vk, placement=pl, camera=cam, frame_index=int(idx[vk]), image=dict(width=wh[0], height=wh[1]), frame_fingerprint=fp, preview_png=str(p))
                frames.append(dict(set_index=i, views=views)); src.clear()
            del ext
            self._invalidate_from("frames_selected"); self.selection = sel_d; self.n_sets = n_sets; self.frames = frames; self.stage = "frames_selected"
            q = self._selection_check(sel_d)
            return self._result("select_frames", "WARNING" if q["status"] == "WARNING" else "PASS", warnings=[issue("frame_selection", i["code"], i["message"]) for i in sel_d["warnings"]], frame_selection=sel_d, selection_check=q,
                                selected_frames=frames, annotation_instructions=self.annotation_instructions())
        except _Stop as s: return self._fail_result("select_frames", s.failure)
        finally: self._timed("select_frames", t0)

    def _selection_check(self, sel_d):
        sets = sel_d["sets"]; ql = self.quality
        sv = min(min(s["placement_1"]["stereo_verified_matches"], s["placement_2"]["stereo_verified_matches"]) for s in sets); tr = min(s["cross_placement"]["four_view_tracks"] for s in sets)
        codes = {w["code"] for w in sel_d["warnings"]}; msgs = []
        if sv < ql.min_stereo_verified_warn: msgs.append(f"weakest stereo pair has {sv} verified preview matches (< {ql.min_stereo_verified_warn})")
        if tr < ql.min_cross_tracks_warn: msgs.append(f"weakest cross-placement set has {tr} four-view tracks (< {ql.min_cross_tracks_warn})")
        for c in ("COVERAGE_RELAXED", "OFFSET_NOT_IDENTIFIABLE"):
            if c in codes: msgs.append(c)
        return check("frame_selection_quality", Status.WARNING if msgs else Status.PASS, "; ".join(msgs) or "selection evidence above the warning thresholds (preview-resolution counts)", value=dict(min_stereo_verified=sv, min_four_view_tracks=tr), required=False)

    def annotation_instructions(self) -> dict:
        """What the visitor must annotate on which displayed frame."""
        need = {"AL": ["corner"], "PL": ["corner", "ceilF"], "AR": ["corner"], "PR": ["corner", "ceilF"]}; opt = {"PL": ["ceilS"], "PR": ["ceilS"]}
        text = dict(corner="Two points on the vertical wall-corner edge, as far apart as the visible edge allows.", ceilF="Two points on the ceiling / front-wall edge (near-horizontal).", ceilS="Optional: two points on the ceiling / side-wall edge (oblique); not needed for the measurement.")
        return dict(coordinate_spaces=list(V.COORDINATE_SPACES), required_fields=list(ANNOTATION_FIELDS), hidden_corner_note="A wall corner need not be visible; annotate only the visible part of each edge.",
                    boundary_help=text, per_frame=[dict(set_index=f["set_index"], view_key=vk, placement=v["placement"], camera=v["camera"], frame_index=v["frame_index"], frame_fingerprint=v["frame_fingerprint"], image=v["image"],
                                                        preview_png=v["preview_png"], required_boundaries=need[vk], optional_boundaries=opt.get(vk, []))
                                                   for f in self.frames for vk, v in f["views"].items()], required_lines=list(REQUIRED_LINES), optional_lines=list(OPTIONAL_LINES))

    # ---- steps 5-7
    def submit_annotations(self, annotations: Sequence[Mapping]) -> dict:
        t0 = time.perf_counter()
        try:
            self._need("frames_selected"); items = list(annotations); bad = []; by_set = {i: [] for i in range(self.n_sets)}
            for k, a in enumerate(items):
                probs = annotation_contract_problems(a)
                si = a.get("set_index") if isinstance(a, Mapping) else None
                if not probs and not (isinstance(si, int) and not isinstance(si, bool) and 0 <= si < self.n_sets): probs = [f"set_index {si!r} is not one of the {self.n_sets} selected sets"]
                if probs: bad.append(issue("annotations", "INVALID_ANNOTATION_RECORD", f"annotation #{k}: {'; '.join(probs)}", index=k)); continue
                by_set[si].append(a)
            self._invalidate_from("annotations_reviewed"); self.annotations = [a for v in by_set.values() for a in v]; self._rejected_records = bad
            return self._review(step="submit_annotations")
        except _Stop as s: return self._fail_result("submit_annotations", s.failure)
        finally: self._timed("submit_annotations", t0)

    def _evaluate(self):
        key = hashlib.sha256(_canon(self.annotations).encode()).hexdigest()
        if self._evals is not None and self._eval_key == key: return self._evals
        evs = []
        for i in range(self.n_sets):
            fr, fps = self._load_set(i); idx = {vk: self.frames[i]["views"][vk]["frame_index"] for vk in VIEW_KEYS}
            evs.append(evaluate_set([a for a in self.annotations if a["set_index"] == i], fr, fps, idx, self.cal, i, self.annotation_limits)); del fr; self._checkpoint(f"annotations set {i}")
        self._evals, self._eval_key = evs, key; return evs

    def _review(self, step):
        evs = self._evaluate(); gates = [gate_set(ev, self.confirmations) for ev in evs]
        failures = list(getattr(self, "_rejected_records", [])); warnings = []; reqs = []
        for g in gates: failures += g["failures"]; warnings += g["warnings"]; reqs += g["confirmation_requests"]
        unconf = [f for f in failures if f.get("decision") == "BLOCKED_UNCONFIRMED_WARNING" or f["code"] == "SET_WARNING_UNCONFIRMED"]
        hard = [f for f in failures if f not in unconf]
        self._gates = gates; self.stage = "annotations_reviewed"
        optional_reqs = [r for r in reqs if not r["required"]]; reqs = [r for r in reqs if r["required"]]
        status = "FAIL" if hard else "WARNING" if (reqs or warnings or unconf) else "PASS"
        ready = not failures
        review = dict(per_set=[dict(set_index=g["set_index"], ok=g["ok"], stage_d_set_status=g["stage_d_set_status"], annotation_statuses={n: d["stage_d_status"] for n, d in g["decisions"].items()}, decisions=g["decisions"],
                                    stereo_consistency=g["stereo_consistency"], stereo_consistency_note=g["stereo_consistency_note"], reconstruction=g["reconstruction"], completeness=g["completeness"], confirmations_applied=g["confirmations_applied"]) for g in gates],
                      confirmation_requests=reqs, optional_confirmation_requests=optional_reqs, confirmation_state=dict(pending=len(reqs), applied=sum(len(g["confirmations_applied"]) for g in gates), rejected_records=self.rejected_confirmations),
                      measurement_ready=ready, readiness_note="annotation validation status, post-pose stereo consistency and overall measurement readiness are separate; annotation stereo consistency is NOT_EVALUATED until a pose exists")
        res = self._result(step, status, failures=hard if hard else [], warnings=warnings + [dict(f, stage="annotations") for f in unconf], annotation_review=review)
        res["can_proceed"] = ready; res["required_next_action"] = ("measure" if ready else f"confirm {len(reqs)} warning(s) with submit_confirmations" if reqs and not hard else "fix the blocked annotations and resubmit")
        return res

    # ---- step 8
    def submit_confirmations(self, confirmations: Sequence) -> dict:
        t0 = time.perf_counter()
        try:
            self._need("annotations_reviewed"); parsed, rejected = [], []
            for k, c in enumerate(confirmations):
                conf, probs = parse_confirmation(c)
                if conf is None: rejected.append(dict(index=k, problems=probs))
                else: parsed.append(conf)
            self.confirmations = parsed; self.rejected_confirmations = rejected
            return self._review(step="submit_confirmations")
        except _Stop as s: return self._fail_result("submit_confirmations", s.failure)
        finally: self._timed("submit_confirmations", t0)

    # ---- steps 9-10
    def measure(self) -> dict:
        t0 = time.perf_counter()
        try:
            self._need("annotations_reviewed"); evs = self._evaluate(); gates = [gate_set(ev, self.confirmations) for ev in evs]; failures = []
            for g in gates: failures += g["failures"]
            failures += list(getattr(self, "_rejected_records", []))
            if failures: return self._fail_result("measure", failures, annotation_review=dict(blocked=True, per_set=[dict(set_index=g["set_index"], decisions=g["decisions"]) for g in gates]), dimensions=None)
            wh = tuple(self.cal.image_size); match_sets = []; tm = time.perf_counter()
            for i in range(self.n_sets):
                fr, fps = self._load_set(i); match_sets.append(features.match_set(fr)); del fr; self._checkpoint(f"matching set {i}")
            self.timings["matching"] = time.perf_counter() - tm
            line_sets = [dict(g["usable_lines"]) for g in gates]; tp = time.perf_counter()
            D = P.build_dataset(self.cal, match_sets); pJ, pA, prep, _ = P.estimate_pose(D); self.timings["pose_estimation"] = time.perf_counter() - tp; self._checkpoint("pose")
            fl = prep["method_J"]["flags"]
            pose = dict(yaw_deg=G.yaw_deg(pJ), rodrigues_deg=np.degrees(pJ[:3]), baseline_mm=float(self.baseline_mm), inlier_fraction=fl["inlier_fraction"], robust_scale_px=fl["scale_px"], converged=fl["converged"],
                        degenerate=fl["degenerate"], n_inlier=prep["method_J"]["n_inlier"], n_residuals=prep["method_J"]["n_residuals"], n_tracks=prep["method_J"]["n_tracks"], n_stereo_matches=int(len(D["n1"])))
            if fl["degenerate"]: return self._fail_result("measure", issue("pose", "POSE_DEGENERATE", f"the stereo pose is degenerate (flags {fl}); the cameras may not have been rigid, the scene not static, or the calibration is wrong for these videos"), pose_diagnostics=pose, dimensions=None)
            tg = time.perf_counter(); rows = _measure_sets(self.cal, pJ, match_sets, line_sets, wh); self.timings["geometry"] = time.perf_counter() - tg
            bad = [dict(frame_set=i, code=r["code"], error=r["error"]) for i, r in enumerate(rows) if not r["ok"]]
            if bad: return self._fail_result("measure", [issue("geometry", b["code"], f"frame set {b['frame_set']}: {b['error']}", frame_set=b["frame_set"]) for b in bad], pose_diagnostics=pose, dimensions=None)
            stability = self._pose_stability(match_sets, line_sets, pJ, rows, wh); self._checkpoint("pose stability")
            stereo_post = self._stereo_post_pose(pJ, line_sets)
            checks = self._quality_checks(pose, rows, stability, stereo_post, gates, wh)
            W = np.array([r["width_mm"] for r in rows]); H = np.array([r["height_mm"] for r in rows]); Wm, Hm = float(W.mean()), float(H.mean())
            checks.append(self._dimension_check(W, H, Wm, Hm)); self.stage = "measured"
            cfail = [c for c in checks if c["status"] == "FAIL" or (c["required"] and c["status"] == "NOT_EVALUATED")]; cwarn = [c for c in checks if c["status"] == "WARNING"]
            fs = [issue("quality", f"CHECK_{c['name'].upper()}", c["message"] + ("" if c["status"] == "FAIL" else " (required check could not be evaluated)")) for c in cfail]
            rel = [issue("reliability", f"CHECK_{c['name'].upper()}", c["message"]) for c in cwarn]
            diag = dict(pose_diagnostics=pose, pose_stability=stability, stereo_consistency_post_pose=stereo_post, quality_checks=checks, registration=[dict(frame_set=i, **r["registration"], inlier_fraction=r["registration"]["n_inliers"] / max(r["registration"]["n_pairs"], 1), coverage=r["coverage"]) for i, r in enumerate(rows)],
                        geometry=[dict(frame_set=i, **r["geometry"]) for i, r in enumerate(rows)], per_set=[dict(frame_set=i, width_mm=r["width_mm"], height_mm=r["height_mm"], H1=r["H1"], H2=r["H2"], vertical_error_deg=r["vertical_error_deg"], skew_mm=r["skew_mm"]) for i, r in enumerate(rows)],
                        annotation_review=dict(statuses=[{n: d["stage_d_status"] for n, d in g["decisions"].items()} for g in gates], confirmations_applied=[c for g in gates for c in g["confirmations_applied"]],
                                               rejected=[dict(set_index=g["set_index"], line=n, decision=d["decision"]) for g in gates for n, d in g["decisions"].items() if d["annotated"] and d["decision"] not in ("ALLOWED", "ALLOWED_CONFIRMED")]),
                        frame_indices=[{vk: f["views"][vk]["frame_index"] for vk in VIEW_KEYS} for f in self.frames], calibration=dict(image_size_wh=list(wh), baseline_mm=self.baseline_mm, accuracy_verified=False))
            if fs: return self._fail_result("measure", fs, warnings=rel, dimensions=None, unaccepted_estimate_for_diagnostics_only=dict(width_mm=Wm, height_mm=Hm), **diag)
            dims = dict(width_mm=Wm, height_mm=Hm, width_per_set_mm=W.tolist(), height_per_set_mm=H.tolist(), top_variant=TOP_VARIANT,
                        width_sd_across_sets_mm=float(W.std(ddof=1)) if len(W) > 1 else None, height_sd_across_sets_mm=float(H.std(ddof=1)) if len(H) > 1 else None,
                        note="sd across sets is descriptive (one shared pose); it is not an uncertainty interval")
            return self._result("measure", "WARNING" if cwarn or any(c["status"] == "NOT_EVALUATED" for c in checks) else "PASS", warnings=rel, dimensions=dims, **diag)
        except _Stop as s: return self._fail_result("measure", s.failure, dimensions=None)
        except Exception as e: return self._fail_result("measure", issue("internal", "INTERNAL_ERROR", f"{type(e).__name__}: {e}"), dimensions=None)
        finally: self._timed("measure", t0)

    # ---- diagnostics
    def _pose_stability(self, match_sets, line_sets, pJ, rows, wh):
        n = len(match_sets); ql = self.quality
        if n < 3: return dict(status="NOT_EVALUATED", reason=f"needs at least 3 frame sets for leave-one-set-out poses (have {n})", note="not a confidence interval")
        subs = []; W0, H0 = float(np.mean([r["width_mm"] for r in rows])), float(np.mean([r["height_mm"] for r in rows]))
        for j in range(n):
            keep = [k for k in range(n) if k != j]
            try:
                Dk = P.build_dataset(self.cal, [match_sets[k] for k in keep]); pk, _, rep, _ = P.estimate_pose(Dk)
                if rep["method_J"]["flags"]["degenerate"]: subs.append(dict(left_out_set=j, degenerate=True)); continue
                rk = _measure_sets(self.cal, pk, match_sets, line_sets, wh); okk = [r for r in rk if r["ok"]]
                subs.append(dict(left_out_set=j, degenerate=False, yaw_deg=G.yaw_deg(pk), width_mm=float(np.mean([r["width_mm"] for r in okk])) if len(okk) == n else None, height_mm=float(np.mean([r["height_mm"] for r in okk])) if len(okk) == n else None))
            except Exception as e: subs.append(dict(left_out_set=j, degenerate=True, error=f"{type(e).__name__}: {e}"))
        good = [s for s in subs if not s["degenerate"] and s.get("width_mm") is not None]
        if len(good) < 2: return dict(status="WARNING", reason="fewer than two leave-one-set-out poses could be evaluated", subsets=subs, note="not a confidence interval")
        yaws = [G.yaw_deg(pJ)] + [s["yaw_deg"] for s in good]; Ws = [W0] + [s["width_mm"] for s in good]; Hs = [H0] + [s["height_mm"] for s in good]
        yr, wr, hr = float(np.ptp(yaws)), float(np.ptp(Ws)), float(np.ptp(Hs)); rel = max(wr / W0, hr / H0); msgs = []
        if yr > ql.pose_yaw_range_warn_deg: msgs.append(f"yaw varies by {yr:.3f} deg across leave-one-set-out poses (> {ql.pose_yaw_range_warn_deg})")
        if rel > ql.pose_dimension_range_rel_warn: msgs.append(f"dimensions vary by up to {100 * rel:.2f} % ({wr:.1f} mm width, {hr:.1f} mm height) across leave-one-set-out poses (> {100 * ql.pose_dimension_range_rel_warn:.2f} %)")
        if any(s["degenerate"] for s in subs): msgs.append("some leave-one-set-out poses were degenerate")
        return dict(status="WARNING" if msgs else "PASS", messages=msgs, yaw_range_deg=yr, width_range_mm=wr, height_range_mm=hr, full_pose_yaw_deg=yaws[0], subsets=subs,
                    note="spread across leave-one-set-out poses is a stability diagnostic, not a calibrated confidence interval; the reported dimensions use the pose from all sets")

    def _stereo_post_pose(self, pJ, line_sets):
        R, T = G.pose_from_params(pJ, self.baseline_mm); out = []; sts = []
        for i, ls in enumerate(line_sets):
            for v in ("V1", "V2"):
                cl, cp = M.VIEW_LINES[v][:2]
                try: st, iss, info = V.stereo_consistency(self.cal, R, T, ls[cl], ls[cp], self.annotation_limits)
                except Exception as e: st, iss, info = Status.NOT_EVALUATED, [], dict(error=str(e))
                out.append(dict(set_index=i, view=v, status=st.value, issues=[x.to_dict() for x in iss], **info)); sts.append(st)
        return dict(status=(V.worst(sts) if sts else Status.NOT_EVALUATED).value, per_view=out, note=V.STEREO_NOTE)

    def _quality_checks(self, pose, rows, stab, stereo, gates, wh):
        ql = self.quality; cs = []; S = Status
        cs.append(check("calibration_compatibility", S.PASS, "calibration declared video-mode and matches the video frame size/orientation (metadata only; accuracy not verified)", value=dict(image_size_wh=list(wh))))
        cs.append(check("pose_inlier_fraction", S.WARNING if pose["inlier_fraction"] < ql.pose_inlier_fraction_warn else S.PASS, f"pose inlier fraction {pose['inlier_fraction']:.3f}", pose["inlier_fraction"], ql.pose_inlier_fraction_warn))
        ep = [r["geometry"][v]["epipolar_rms_px"] for r in rows for v in ("view1", "view2")]; rp = [r["geometry"][v]["reprojection_rms_px"] for r in rows for v in ("view1", "view2")]
        cs.append(check("residuals", S.WARNING if max(ep + rp) > ql.residual_rms_px_warn else S.PASS, f"worst epipolar rms {max(ep):.2f} px, worst reprojection rms {max(rp):.2f} px of gated points (low residuals do not imply accurate dimensions)", dict(epipolar_rms_px=max(ep), reprojection_rms_px=max(rp)), ql.residual_rms_px_warn))
        gf = min(r["geometry"][v]["gated_fraction"] for r in rows for v in ("view1", "view2")); ng = min(r["geometry"][v]["n_gated_points"] for r in rows for v in ("view1", "view2"))
        cs.append(check("triangulation_depth_validity", S.FAIL if (gf < ql.gated_fraction_fail or ng < ql.min_gated_points_fail) else S.WARNING if gf < ql.gated_fraction_warn else S.PASS, f"smallest gated fraction {gf:.2f}, fewest gated points {ng}", dict(gated_fraction=gf, n_gated=ng), dict(warn=ql.gated_fraction_warn, fail=ql.gated_fraction_fail)))
        fr = [r["geometry"][v] for r in rows for v in ("view1", "view2")]; rms = [f["floor_plane_rms_mm"] for f in fr]
        if any(x is None for x in rms): cs.append(check("floor_plane_plausibility", S.NOT_EVALUATED, "floor-plane residual could not be evaluated (too few floor points); show the floor in the lower part of the frame"))
        else:
            tilt = max(f["floor_tilt_deg"] for f in fr); off = [abs(f["floor_offset_mm"]) for f in fr]; lo, hi = ql.floor_camera_height_mm; msgs = []
            st = S.FAIL if max(rms) > ql.floor_rms_mm_fail else S.WARNING if max(rms) > ql.floor_rms_mm_warn else S.PASS
            if tilt > ql.floor_tilt_deg_warn: msgs.append(f"floor normal tilted {tilt:.0f} deg from the camera vertical"); st = V.worst([st, S.WARNING]) if st is not S.FAIL else st
            if min(off) < lo or max(off) > hi: msgs.append(f"camera height above the floor {min(off):.0f}-{max(off):.0f} mm is outside {lo:.0f}-{hi:.0f} mm"); st = V.worst([st, S.WARNING]) if st is not S.FAIL else st
            cs.append(check("floor_plane_plausibility", st, f"floor rms {max(rms):.1f} mm; " + ("; ".join(msgs) or "tilt and camera height plausible"), dict(rms_mm=max(rms), tilt_deg=tilt, camera_height_mm=[min(off), max(off)]), dict(rms_warn=ql.floor_rms_mm_warn, rms_fail=ql.floor_rms_mm_fail)))
        cs.append(check("floor_region_assumption", S.PASS if wh[1] == PHASE1_FRAME_HEIGHT else S.WARNING, f"floor region = rows > {floor_row_threshold(wh[1]):.0f} px (Phase 1 value 2600 px scaled by height / 3840)" + ("" if wh[1] == PHASE1_FRAME_HEIGHT else "; the scaled floor region is unvalidated at this resolution"), floor_row_threshold(wh[1]), required=False))
        ve = max(max(r["vertical_error_deg"]) for r in rows); sk = max(max(r["skew_mm"]) for r in rows)
        ang = [d.get("corner_ceilF_angle_deg") for g in gates for d in (g["reconstruction"] or {}).values() if isinstance(d, dict) and d.get("corner_ceilF_angle_deg") is not None]
        st = S.FAIL if (ve > ql.vertical_error_deg_fail or sk > ql.skew_mm_fail) else S.WARNING if (ve > ql.vertical_error_deg_warn or sk > ql.skew_mm_warn or (ang and min(ang) < ql.corner_ceiling_angle_deg_warn)) else S.PASS
        cs.append(check("boundary_conditioning", st, f"worst corner verticality error {ve:.1f} deg, worst ray/vertical skew {sk:.1f} mm" + (f", smallest corner/ceiling angle {min(ang):.0f} deg" if ang else ""), dict(vertical_error_deg=ve, skew_mm=sk, min_corner_ceilF_angle_deg=min(ang) if ang else None)))
        ex = [d.get("extrapolation_ratio") for g in gates for d in (g["reconstruction"] or {}).values() if isinstance(d, dict) and d.get("extrapolation_ratio") is not None]
        cs.append(check("hidden_corner_extrapolation", S.NOT_EVALUATED if not ex else S.WARNING if max(ex) > self.annotation_limits.max_extrapolation_ratio else S.PASS,
                        "extrapolation not evaluated" if not ex else f"hidden corner extrapolated up to {max(ex):.1f}x the annotated corner-edge length (a hidden corner is expected; large extrapolation is less reliable)", max(ex) if ex else None, self.annotation_limits.max_extrapolation_ratio, required=False))
        reg = [r["registration"] for r in rows]; ni = min(r["n_inliers"] for r in reg); fr_ = min(r["n_inliers"] / max(r["n_pairs"], 1) for r in reg); rr = max(r["rms_mm"] for r in reg)
        cs.append(check("registration_inlier_count", S.FAIL if ni < ql.registration_min_inliers_fail else S.WARNING if ni < ql.registration_min_inliers_warn else S.PASS, f"fewest registration inliers {ni}", ni, dict(fail=ql.registration_min_inliers_fail, warn=ql.registration_min_inliers_warn)))
        cs.append(check("registration_inlier_fraction", S.FAIL if fr_ < ql.registration_min_inlier_fraction_fail else S.WARNING if fr_ < ql.registration_min_inlier_fraction_warn else S.PASS, f"lowest registration inlier fraction {fr_:.2f}", fr_, dict(fail=ql.registration_min_inlier_fraction_fail, warn=ql.registration_min_inlier_fraction_warn)))
        cs.append(check("registration_rms", S.FAIL if (not np.isfinite(rr) or rr > ql.registration_rms_mm_fail) else S.WARNING if rr > ql.registration_rms_mm_warn else S.PASS, f"worst registration rms {rr:.1f} mm", rr, dict(warn=ql.registration_rms_mm_warn, fail=ql.registration_rms_mm_fail)))
        bb = min(r["coverage"]["bbox_fraction"] for r in rows); cells = min(r["coverage"]["cells_occupied"] for r in rows)
        cs.append(check("registration_spatial_coverage", S.FAIL if bb < ql.registration_bbox_fraction_fail else S.WARNING if (bb < ql.registration_bbox_fraction_warn or cells < ql.registration_min_cells_warn) else S.PASS,
                        f"registration pairs span {100 * bb:.0f} % of the image area and {cells} of 9 grid cells (all pairs, not only inliers)", dict(bbox_fraction=bb, cells=cells), dict(warn=ql.registration_bbox_fraction_warn, fail=ql.registration_bbox_fraction_fail), required=False))
        cs.append(check("pose_stability", S(stab["status"]), "; ".join(stab.get("messages", [])) or stab.get("reason") or "leave-one-set-out poses agree within the warning thresholds (stability diagnostic, not a confidence interval)", required=False))
        sp = stereo["status"]; cs.append(check("stereo_consistency_post_pose", S(sp) if sp != "NOT_EVALUATED" else S.NOT_EVALUATED, "weak test: planes are compatible with one common 3D line in front of both cameras; it does not show both annotations mark the same wall edge", required=False))
        return cs

    def _dimension_check(self, W, H, Wm, Hm):
        ql = self.quality; ok = np.isfinite(W).all() and np.isfinite(H).all() and (W > 0).all() and (H > 0).all() and 0 < Wm < ql.max_dimension_mm and 0 < Hm < ql.max_dimension_mm
        if not ok: return check("dimension_validity", Status.FAIL, f"width {Wm}, height {Hm} mm are not finite, positive and < {ql.max_dimension_mm} mm")
        sp = max(W.std(ddof=1) / Wm, H.std(ddof=1) / Hm) if len(W) > 1 else 0.0
        return check("dimension_validity", Status.WARNING if sp > ql.set_spread_rel_warn else Status.PASS, f"dimensions finite and positive; relative spread across sets {100 * sp:.2f} % (descriptive only)", sp, ql.set_spread_rel_warn)

    # ---- stop / resume
    def export_state(self) -> dict:
        """JSON-serialisable state: resume with ``VisitorMeasurementJob.resume`` over the SAME workspace directory.  Frames stay in the workspace;
        nothing here holds secrets or arrays."""
        return to_jsonable(dict(schema=SCHEMA, stage=self.stage, cal_model=self.cal_model, baseline_mm=self.baseline_mm, video_paths=self.video_paths, video_validation=self.video_validation, selection=self.selection,
                                n_sets=self.n_sets, frames=self.frames, annotations=self.annotations, confirmations=[c.to_dict() for c in self.confirmations], rejected_confirmations=self.rejected_confirmations,
                                workspace=str(self.ws.path)))

    @classmethod
    def resume(cls, state: Mapping, workspace: VV.TemporaryVideoWorkspace, **kw) -> "VisitorMeasurementJob":
        if state.get("schema") != SCHEMA: raise ValueError(f"unsupported job state schema {state.get('schema')!r}")
        if str(workspace.path) != state.get("workspace"): raise ValueError("the supplied workspace is not the one this state was exported from")
        job = cls(workspace=workspace, **kw); job._owns_ws = False
        try:
            if state["cal_model"] is not None:
                ad = VC.adapt_visitor_calibration(state["cal_model"], state["baseline_mm"], name="visitor_stereo")
                if not ad.ok: raise ValueError("stored calibration is no longer valid")
                job.cal_model, job.baseline_mm, job.cal, job._adapted = state["cal_model"], state["baseline_mm"], ad.calibration, ad
            job.video_paths, job.video_validation, job.selection, job.n_sets, job.frames = state["video_paths"], state["video_validation"], state["selection"], state["n_sets"], state["frames"]
            if job.video_paths:
                job._vr = VV.validate_visitor_videos(*[job.video_paths[k] for k in VIDEO_FIELDS], job._adapted, job.limits.video)
            job.annotations = list(state["annotations"]); job.confirmations = [parse_confirmation(c)[0] for c in state["confirmations"] if parse_confirmation(c)[0] is not None]; job.rejected_confirmations = state["rejected_confirmations"]; job.stage = state["stage"]
            job._rejected_records = []
        except Exception:
            job.close(); raise
        return job


# ------------------------------------------------------------------ convenience: whole workflow with callbacks (tests / regression; a UI would call the steps)
def run_workflow(calibration, videos, n_sets, annotate, confirm=None, job_kwargs=None) -> dict:
    """``annotate(job_result_of_select_frames, job) -> annotations``; ``confirm(review_result) -> confirmations`` (explicit; None = no confirmations).
    Returns dict(steps=[...], final=<measure result or the first blocking result>)."""
    steps = []
    with VisitorMeasurementJob(**(job_kwargs or {})) as job:
        for name, call in (("validate_calibration", lambda: job.validate_calibration(calibration)), ("validate_videos", lambda: job.validate_videos(videos)), ("select_frames", lambda: job.select_frames(n_sets))):
            r = call(); steps.append(r)
            if not r["can_proceed"]: return dict(steps=steps, final=r)
        r = job.submit_annotations(annotate(steps[-1], job)); steps.append(r)
        if r["annotation_review"]["confirmation_requests"] and confirm is not None:
            r = job.submit_confirmations(confirm(r)); steps.append(r)
        if not r["can_proceed"]: return dict(steps=steps, final=r)
        r = job.measure(); steps.append(r); return dict(steps=steps, final=r)
