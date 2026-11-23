"""Stage F6 (Module 7 Part 3, Phase 2B): simplified CLASSICAL visitor workflow -- presentation-independent core (NO Streamlit import).

Orchestrates the EXISTING, unchanged stages; it defines no new geometry, no new line fitting and no new confirmation format:

  Stage A  visitor_calibration           (through VisitorMeasurementJob.validate_calibration)   -- IMPORTS a previously computed calibration.
  Stage B  visitor_video_validation      (through VisitorMeasurementJob.validate_videos)
  Stage C  visitor_frame_selection       (through VisitorMeasurementJob.select_frames)
  Stage F2 stage_f2_boundary_detection   classical gradient / line-search candidate detector (OpenCV + NumPy only)
  Stage D  visitor_annotations           refinement of the chosen candidate (or of the visitor's two manual points)
  Stage E2 visitor_measurement           identity / digest / confirmation gate, Phase 1 geometry, dimensions (top_variant = 0)
  Stage F1 annotation_ui/core.py         manual correction (clicks, zoom / pan mapping, refinement, confirmation)

Visitor-facing rules enforced here
  * F2's top candidate is only a PROPOSAL.  Nothing reaches Stage E2 until the visitor explicitly confirms that boundary.
  * A confirmation is bound to (set, boundary, source, candidate rank, annotation id + digest, refined-line digest, frame index, frame fingerprint).
    Changing the candidate / the manual points / the frame set removes it; ``export_for_measurement`` re-verifies every confirmation and refuses stale ones.
  * Stage D / E2 PASS-WARNING-FAIL is respected: FAIL can never be confirmed; WARNING needs the exact Stage E2 ``Confirmation``.
  * Actual dimensions are OPTIONAL, live in ``ActualDimensions`` and are used only by ``validation_report`` AFTER an estimate exists.  They are never
    passed to the job, the detector, the pose, the triangulation or the dimension computation.
  * Calibration is IMPORTED (K, distortion, baseline).  This workflow does not calibrate from checkerboard images (see README_F6.md).
No machine learning of any kind is imported here.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

MODULE = Path(__file__).resolve().parents[1]
if str(MODULE) not in sys.path: sys.path.insert(0, str(MODULE))

from room_measurement import stage_f2_boundary_detection as F2, visitor_annotations as V, visitor_measurement as VM          # noqa: E402
from room_measurement.pipeline import to_jsonable                                                                            # noqa: E402


def _load_f1_core():
    """The existing Stage F1 core, loaded by path under a private name (its generic module name ``core`` is not reserved here)."""
    path = MODULE / "annotation_ui" / "core.py"; spec = importlib.util.spec_from_file_location("f1_annotation_core", path)
    mod = importlib.util.module_from_spec(spec); sys.modules.setdefault("f1_annotation_core", mod); spec.loader.exec_module(mod); return mod


f1core = sys.modules.get("f1_annotation_core") or _load_f1_core()

SCHEMA = "csc8830.module7.stage_f6_visitor_workflow.v1"
SOURCE_CANDIDATE, SOURCE_MANUAL = "candidate", "manual"
VISITOR_ID = "f6_visitor"
DEFAULT_SETS, MIN_SETS, MAX_SETS = 2, 2, 3          # two sets is the smallest configuration the Stage E2 regression exercised
BOUNDARY_ORDER = tuple(VM.REQUIRED_LINES)            # AL_corner PL_corner PL_ceilF AR_corner PR_corner PR_ceilF (verified Phase 1 order)
STEPS = ("Calibration", "Upload Videos", "Review Boundaries", "View Results")


# ------------------------------------------------------------------ plain-language labels (technical ids stay in diagnostics)
LABELS = {
    "AL_corner": "Left wall corner — placement 1", "PL_corner": "Right-camera wall corner — placement 1", "PL_ceilF": "Wall-ceiling edge — placement 1",
    "AR_corner": "Left wall corner — placement 2", "PR_corner": "Right-camera wall corner — placement 2", "PR_ceilF": "Wall-ceiling edge — placement 2",
}
INSTRUCTIONS = {
    "corner": "Check that the line follows the tall vertical edge where two walls meet. The bottom of the corner may be hidden — that is fine.",
    "ceilF": "Check that the line follows the edge where the front wall meets the ceiling. Door frames, shelves and lights are not wall edges.",
}
CONFUSER_TEXT = {"ISOLATED": "a single clear edge", "BAND": "a thin strip (trim, door frame or shadow)", "SHOULDER": "close to a much stronger edge"}
DETECTION_TEXT = {"DETECTED": "Suggested line — please check it", "UNCERTAIN": "Several possible lines — please choose carefully", "NOT_DETECTED": "No line found — please mark it yourself"}


def label_of(line: str) -> str: return LABELS[line]
def camera_text(line: str) -> str:
    a = f1core.association(line); cam = "LEFT camera" if a["camera"] == "left" else "RIGHT camera"
    return f"{cam} · {a['placement'].replace('_', ' ')}"


# ------------------------------------------------------------------ units and optional actual dimensions (validation only)
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}
DISPLAY_UNITS = ("mm", "m", "ft")


def to_mm(value, unit: str) -> float:
    """A positive finite length in ``unit`` -> millimetres.  Booleans, NaN, inf, zero, negatives and unknown units are rejected."""
    if unit not in UNIT_TO_MM: raise ValueError(f"unknown unit {unit!r}; choose one of {', '.join(UNIT_TO_MM)}")
    if isinstance(value, bool): raise ValueError("a length must be a number")
    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        try: value = float(s)
        except ValueError: raise ValueError(f"{value!r} is not a number") from None
    try: v = float(value)
    except (TypeError, ValueError): raise ValueError("a length must be a number") from None
    if not math.isfinite(v) or v <= 0: raise ValueError("a length must be a positive, finite number")
    return v * UNIT_TO_MM[unit]


def from_mm(mm: float, unit: str) -> float:
    if unit not in UNIT_TO_MM: raise ValueError(f"unknown unit {unit!r}")
    return float(mm) / UNIT_TO_MM[unit]


def format_length(mm: float, unit: str) -> str:
    v = from_mm(mm, unit); return f"{v:,.0f} mm" if unit == "mm" else f"{v:,.3f} {unit}" if unit == "m" else f"{v:,.2f} {unit}"


@dataclass(frozen=True)
class ActualDimensions:
    """Visitor-entered ground truth.  Optional per dimension.  Validation only: nothing in the estimation path accepts it."""
    width_mm: float | None = None
    height_mm: float | None = None
    unit: str | None = None

    @property
    def any(self) -> bool: return self.width_mm is not None or self.height_mm is not None


def _blank(v) -> bool: return v is None or (isinstance(v, str) and not v.strip())


def parse_actual_dimensions(width, height, unit: str = "mm") -> ActualDimensions:
    """Blank fields mean 'not provided'.  Anything else must be a positive finite number (ValueError otherwise); converted to millimetres."""
    if _blank(width) and _blank(height): return ActualDimensions(None, None, None)
    return ActualDimensions(None if _blank(width) else to_mm(width, unit), None if _blank(height) else to_mm(height, unit), unit)


def compare_dimension(estimated_mm: float, actual_mm: float) -> dict:
    """Signed error (estimate - actual) and the ABSOLUTE percentage error  |estimate - actual| / actual x 100.  Both stay separate and labelled."""
    for n, v in (("estimated", estimated_mm), ("actual", actual_mm)):
        if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(v) or v <= 0: raise ValueError(f"{n} dimension must be a positive finite number of mm")
    e, a = float(estimated_mm), float(actual_mm); signed = e - a
    return dict(estimated_mm=e, actual_mm=a, signed_error_mm=signed, absolute_error_mm=abs(signed), signed_percent_error=signed / a * 100.0, absolute_percent_error=abs(signed) / a * 100.0)


def validation_report(dimensions: Mapping | None, actual: ActualDimensions | None) -> dict:
    """Comparison after the fact.  No estimate -> nothing; no actual value for a dimension -> estimate only and NO percentage for it."""
    out = dict(ground_truth_used_in_estimation=False, width=None, height=None, note="Actual dimensions are used only to compute this comparison, after the estimate exists.")
    if not dimensions or dimensions.get("width_mm") is None: out["available"] = False; return out
    out["available"] = True
    for k in ("width", "height"):
        est = dimensions[f"{k}_mm"]; act = None if actual is None else getattr(actual, f"{k}_mm")
        out[k] = dict(estimated_mm=float(est), actual_mm=None, comparison=None) if act is None else dict(estimated_mm=float(est), actual_mm=act, comparison=compare_dimension(est, act))
    return out


# ------------------------------------------------------------------ calibration import (truthful: nothing is estimated here)
def calibration_contract_from_phase1_json(d: Mapping, capture_mode: str | None, baseline_mm=None) -> dict:
    """Map the saved Phase 1 / Recipe A calibration file layout onto the Stage A input contract.  Nothing is invented: the capture mode must be DECLARED by the
    caller (the file does not record it) and the baseline comes from the file (``physical_baseline_mm``) or from the caller."""
    try: left, right = d["left"], d["right"]
    except (KeyError, TypeError): raise ValueError("not a recognised calibration file: expected 'left' and 'right' camera entries or the Stage A contract (K1, D1, K2, D2, ...)") from None
    base = d.get("physical_baseline_mm") if baseline_mm is None else baseline_mm
    out = dict(K1=left.get("K"), D1=left.get("distortion_k1_k2_p1_p2_k3"), K2=right.get("K"), D2=right.get("distortion_k1_k2_p1_p2_k3"), image_size_wh=d.get("image_size_wh"), baseline_mm=base)
    if capture_mode: out["capture_mode"] = dict(left=capture_mode, right=capture_mode)
    return out


def parse_calibration_json(data: bytes | str, capture_mode: str | None = None, baseline_mm=None) -> tuple:
    """-> (stage-A style dict | None, [problems]).  Accepts the Stage A contract directly or the saved Phase 1 layout."""
    try: d = json.loads(data)
    except (ValueError, TypeError): return None, ["the file is not valid JSON"]
    if not isinstance(d, dict): return None, ["the calibration JSON must be an object"]
    if all(k in d for k in ("K1", "D1", "K2", "D2")):
        d = dict(d)
        if baseline_mm is not None and d.get("baseline_mm") is None: d["baseline_mm"] = baseline_mm
        if capture_mode and not d.get("capture_mode"): d["capture_mode"] = dict(left=capture_mode, right=capture_mode)
        return d, []
    try: return calibration_contract_from_phase1_json(d, capture_mode, baseline_mm), []
    except ValueError as e: return None, [str(e)]


# ------------------------------------------------------------------ per-boundary review state
@dataclass
class BoundaryReview:
    set_index: int
    line_name: str
    source: str = SOURCE_CANDIDATE
    rank: int | None = None                  # F2 candidate rank (1 = F2's top), or None
    revision: int = 0                        # bumped on EVERY change of the selection
    raw: dict | None = None                  # Stage D payload actually evaluated
    result: dict | None = None               # JSON-able Stage D / E2 gate summary
    confirmation: object | None = None       # visitor_measurement.Confirmation (WARNING only)
    acceptance: dict | None = None           # the visitor's explicit confirmation record (bound identity)
    _ev: dict | None = None                  # Stage D evaluation (never persisted)

    @property
    def view_key(self) -> str: return self.line_name[:2]


def _summarize(ev: dict, line: str, confirmation) -> dict:
    """Stage E2 gate over one Stage D evaluation -> the summary the UI shows (same fields as the F1 prototype's summary)."""
    g = VM.gate_set(ev, [confirmation] if confirmation is not None else []); d = g["decisions"][line]
    e = [x for x in ev["entries"] if x["line_name"] == line][0]; r = e["result"]
    req = [q for q in g["confirmation_requests"] if q["line_name"] == line and q["scope"] == "annotation"]
    return dict(status=e["status"].value, decision=d["decision"], decision_reasons=d["reasons"], warning_codes=e["warning_codes"], failure_codes=e["failure_codes"],
                issues=[i.to_dict() for i in r.issues if i.status in (V.Status.WARNING, V.Status.FAIL)], identity_ok=e["identity"]["ok"], identity_reasons=e["identity"]["reasons"],
                initial_line=r.initial_line, refined_line=r.refined_line, refined_downstream=r.downstream_line is not None,
                diagnostics={k: r.diagnostics.get(k) for k in ("displacement_max_px", "angle_change_deg", "fit_rms_px", "n_edge_points", "inlier_fraction", "edge_strength", "competitor_ratio", "competitor_offset_px", "pass_used")},
                annotation_id=r.annotation_id, annotation_digest=e["annotation_digest"], refined_line_digest=e["refined_line_digest"], frame_fingerprint=e["identity"].get("annotation_fingerprint"),
                confirmation_request=req[0] if req else None, confirmation_valid=d.get("confirmation_valid"), checks_not_evaluated=d.get("checks_not_evaluated", []))


class F6Error(RuntimeError):
    pass


# ------------------------------------------------------------------ the workflow session
class F6Session:
    """One visitor, one isolated Stage E2 job (own temporary workspace).  Nothing is shared between sessions."""

    def __init__(self, n_sets: int = DEFAULT_SETS, job_kwargs: Mapping | None = None):
        if not isinstance(n_sets, int) or isinstance(n_sets, bool) or not MIN_SETS <= n_sets <= MAX_SETS: raise ValueError(f"n_sets must be an integer from {MIN_SETS} to {MAX_SETS}")
        self.session_id = uuid.uuid4().hex[:12]; self.n_sets = n_sets; self.job = VM.VisitorMeasurementJob(**(job_kwargs or {}))
        self._finalizer = weakref.finalize(self, self.job.close)
        self.detections: dict = {}; self.detection_meta: dict = {}; self.reviews: dict = {}; self._manual = None
        self.set_confirmations: dict = {}; self.measurement: dict | None = None; self.calibration_provenance: dict | None = None; self.actual = ActualDimensions(); self.log: list = []; self.closed = False

    # ---- lifecycle
    def close(self):
        """Removes only this session's own workspace (Stage B ``TemporaryVideoWorkspace``); videos given by path are never touched."""
        if self.closed: return
        self.closed = True; self._manual = None; self.detections.clear(); self.reviews.clear(); self._finalizer()

    def workspace_path(self) -> Path | None:
        try: return Path(self.job.ws.path)
        except Exception: return None

    def set_frame_set_count(self, n_sets: int):
        if not isinstance(n_sets, int) or isinstance(n_sets, bool) or not MIN_SETS <= n_sets <= MAX_SETS: raise ValueError(f"n_sets must be an integer from {MIN_SETS} to {MAX_SETS}")
        if self.job.stage in ("frames_selected", "annotations_reviewed", "measured") and n_sets != self.n_sets: self._reset_downstream(); self.job._invalidate_from("frames_selected")
        self.n_sets = n_sets

    # ---- steps 1-2: calibration, videos
    def import_calibration(self, calibration: Mapping) -> dict:
        self._reset_downstream(); r = self.job.validate_calibration(calibration); self._note("calibration", r); return r

    def submit_videos(self, videos: Mapping) -> dict:
        """``videos``: placement_1_left / placement_1_right / placement_2_left / placement_2_right -> bytes or a read-only path (Stage B)."""
        self._reset_downstream(); r = self.job.validate_videos(videos); self._note("videos", r); return r

    def select_frames(self) -> dict:
        """Stage C automatic frame selection, then Stage F2 candidate generation on every selected set."""
        self._reset_downstream(); r = self.job.select_frames(self.n_sets); self._note("frames", r)
        if r["can_proceed"] and self.job.stage == "frames_selected": self.detect_boundaries()
        return r

    # ---- frames
    def frame_record(self, set_index: int, view_key: str) -> dict: return self.job.frames[set_index]["views"][view_key]

    def load_gray(self, set_index: int, view_key: str) -> np.ndarray:
        """The exact stored frame, re-fingerprinted on every load (mismatch -> FrameIntegrityError; nothing is substituted)."""
        rec = self.frame_record(set_index, view_key); g = cv2.imread(rec["preview_png"], cv2.IMREAD_GRAYSCALE)
        if g is None: raise f1core.FrameIntegrityError(f"frame for set {set_index} {view_key} is missing")
        if V.frame_fingerprint(g) != rec["frame_fingerprint"]: raise f1core.FrameIntegrityError(f"frame for set {set_index} {view_key} no longer matches its fingerprint")
        return g

    def _frames_of_set(self, si):
        fr = {vk: self.load_gray(si, vk) for vk in VM.VIEW_KEYS}
        return fr, {vk: self.frame_record(si, vk)["frame_fingerprint"] for vk in VM.VIEW_KEYS}, {vk: self.frame_record(si, vk)["frame_index"] for vk in VM.VIEW_KEYS}

    # ---- step 3: automatic boundary suggestions (Stage F2, classical)
    def detect_boundaries(self):
        if self.job.stage != "frames_selected": raise F6Error("frames must be selected before boundaries can be suggested")
        self.detections.clear(); self.reviews.clear(); self.detection_meta = {}; t0 = time.perf_counter()
        for si in range(self.job.n_sets):
            fr, fps, idx = self._frames_of_set(si); res = F2.detect_set(fr, self.job.cal, set_index=si, frame_indices=idx, fingerprints=fps); res.pop("_prepared", None); del fr
            self.detection_meta[si] = dict(timing_s=res["timing_s"], n_candidates=res["n_candidates"], missing_views=res["missing_views"])
            for ln in BOUNDARY_ORDER:
                br = res["boundaries"].get(ln)
                if br is None: continue
                self.detections[(si, ln)] = br
                self.reviews[(si, ln)] = BoundaryReview(si, ln, SOURCE_CANDIDATE, br.selected_rank if br.candidates and br.status != "NOT_DETECTED" else None)
        self.detection_meta["total_s"] = time.perf_counter() - t0

    def candidates(self, si: int, line: str) -> list: return list(self.detections[(si, line)].candidates) if (si, line) in self.detections else []
    def detection(self, si: int, line: str): return self.detections.get((si, line))
    def review(self, si: int, line: str) -> BoundaryReview:
        if (si, line) not in self.reviews: raise F6Error(f"no boundary {line} in set {si}")
        return self.reviews[(si, line)]

    # ---- candidate switching / confirmation
    def _invalidate(self, rv: BoundaryReview):
        rv.revision += 1; rv.raw = None; rv.result = None; rv.confirmation = None; rv.acceptance = None; rv._ev = None

    def select_candidate(self, si: int, line: str, rank: int):
        rv = self.review(si, line); cands = self.candidates(si, line)
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= len(cands): raise ValueError(f"candidate rank must be 1..{len(cands)}")
        rv.source, rv.rank = SOURCE_CANDIDATE, rank; self._invalidate(rv)

    def use_manual(self, si: int, line: str):
        rv = self.review(si, line); rv.source, rv.rank = SOURCE_MANUAL, None; self._invalidate(rv)

    def evaluate(self, si: int, line: str) -> dict | None:
        """Stage D refinement + Stage E2 gate of the CURRENT selection (cached until the selection changes).  Manual boundaries are evaluated by F1's session."""
        rv = self.review(si, line)
        if rv.source == SOURCE_MANUAL: return self._sync_manual(rv)
        if rv.rank is None: return None
        if rv.result is not None: return rv.result
        br = self.detections[(si, line)]; vk = rv.view_key; gray = self.load_gray(si, vk)
        raw = F2.candidate_to_annotation(br, rv.rank, set_index=si, annotation_id=f"f6:{self.session_id}:set{si}:{line}:cand{rv.rank}:r{rv.revision}")
        fp = V.frame_fingerprint(gray); ev = VM.evaluate_set([raw], {vk: gray}, {vk: fp}, {vk: br.frame_index}, self.job.cal, si, self.job.annotation_limits)
        rv.raw, rv._ev = raw, ev; rv.result = _summarize(ev, line, rv.confirmation); return rv.result

    def confirm(self, si: int, line: str) -> bool:
        """The visitor's explicit confirmation of the CURRENT selection.  PASS -> accepted.  WARNING -> accepted only with the exact Stage E2 confirmation.
        FAIL / blocked -> False (never overridable)."""
        rv = self.review(si, line); res = self.evaluate(si, line)
        if res is None: return False
        if rv.source == SOURCE_MANUAL: return self._confirm_manual(rv)
        if res["decision"] == "BLOCKED_UNCONFIRMED_WARNING" and res["confirmation_request"]:
            rv.confirmation = VM.Confirmation.from_request(res["confirmation_request"], VISITOR_ID, "the visitor checked the highlighted line by eye"); rv.result = _summarize(rv._ev, line, rv.confirmation); res = rv.result
        if res["decision"] not in ("ALLOWED", "ALLOWED_CONFIRMED"): return False
        rv.acceptance = self._acceptance_record(rv); return True

    def reject(self, si: int, line: str): self._invalidate(self.review(si, line))

    def _acceptance_record(self, rv: BoundaryReview) -> dict:
        r = rv.result; br = self.detections.get((rv.set_index, rv.line_name)); rec = self.frame_record(rv.set_index, rv.view_key); a = f1core.association(rv.line_name)
        cand = None if rv.source != SOURCE_CANDIDATE or br is None else {k: br.candidates[rv.rank - 1].get(k) for k in ("rank", "rank_score", "confuser_class", "visible_length_px", "support_fraction")}
        return dict(set_index=rv.set_index, line_name=rv.line_name, boundary_type=a["boundary_type"], camera=a["camera"], placement=a["placement"], view_key=rv.view_key, source=rv.source, rank=rv.rank, revision=rv.revision,
                    annotation_id=r["annotation_id"], annotation_digest=r["annotation_digest"], refined_line_digest=r["refined_line_digest"], frame_index=rec["frame_index"], frame_fingerprint=rec["frame_fingerprint"],
                    coordinate_space=(rv.raw or {}).get("coordinate_space"), stage_d_status=r["status"], decision=r["decision"], warning_codes=list(r["warning_codes"]),
                    candidate_provenance=cand, f2_status=None if br is None else br.status, confirmed_by=VISITOR_ID, confirmed_at=time.time())

    # ---- manual correction (Stage F1 session over this job's stored frames)
    def manual_session(self):
        if self._manual is None:
            recs = {(si, vk): f1core.FrameRecord(si, vk, self.frame_record(si, vk)["frame_index"], self.frame_record(si, vk)["frame_fingerprint"], tuple(self.job.cal.image_size), self.frame_record(si, vk)["preview_png"])
                    for si in range(self.job.n_sets) for vk in VM.VIEW_KEYS}
            self._manual = f1core.AnnotationSession(recs, self.job.cal, self.job.annotation_limits, job_id=self.session_id)
        return self._manual

    def manual_focus(self, si: int, line: str):
        s = self.manual_session(); s.select_set(si); s.goto(s.lines.index(line)); return s

    def _sync_manual(self, rv: BoundaryReview) -> dict | None:
        st = self.manual_session().state(rv.line_name, rv.set_index); rv.raw, rv.result, rv.confirmation, rv._ev = st.raw, st.result, st.confirmation, st._ev
        if rv.acceptance is not None and (st.revision != rv.acceptance.get("manual_revision") or st.result is None): rv.acceptance = None
        return st.result

    def _confirm_manual(self, rv: BoundaryReview) -> bool:
        s = self.manual_focus(rv.set_index, rv.line_name); st = s.state()
        if s.can_confirm(): s.confirm(VISITOR_ID, "the visitor checked the highlighted line by eye")
        self._sync_manual(rv)
        if not (st.accepted and st.result and st.result["decision"] in ("ALLOWED", "ALLOWED_CONFIRMED")): return False
        rv.acceptance = dict(self._acceptance_record(rv), manual_revision=st.revision); return True

    # ---- review status / stale-confirmation protection
    def acceptance_problems(self, si: int, line: str) -> list:
        """Why the stored confirmation is NOT valid for the current selection ([] = valid).  Re-derived from current state each time."""
        rv = self.review(si, line); acc = rv.acceptance
        if acc is None: return ["not confirmed by the visitor"]
        res = self.evaluate(si, line); rv = self.review(si, line); acc = rv.acceptance
        if acc is None or res is None or rv.raw is None: return ["not confirmed by the visitor"]
        rec = self.frame_record(si, rv.view_key); bad = []
        for k, cur in (("set_index", si), ("line_name", line), ("source", rv.source), ("rank", rv.rank), ("annotation_id", res["annotation_id"]), ("annotation_digest", VM.annotation_digest(rv.raw)),
                       ("refined_line_digest", res["refined_line_digest"]), ("frame_index", rec["frame_index"]), ("frame_fingerprint", rec["frame_fingerprint"])):
            if acc.get(k) != cur: bad.append(f"{k} differs from the current value")
        if rv.source == SOURCE_CANDIDATE and acc.get("revision") != rv.revision: bad.append("the selection changed after it was confirmed")
        if res["decision"] not in ("ALLOWED", "ALLOWED_CONFIRMED"): bad.append(f"the current decision is {res['decision']}")
        return bad

    def is_confirmed(self, si: int, line: str) -> bool: return not self.acceptance_problems(si, line)

    def status(self, si: int, line: str) -> str:
        """NO_SUGGESTION / PROPOSED / CONFIRMED / NEEDS_CONFIRMATION / BLOCKED  (a PROPOSED boundary is not yet usable)."""
        rv = self.review(si, line)
        if rv.source == SOURCE_CANDIDATE and rv.rank is None: return "NO_SUGGESTION"
        res = rv.result if rv.source == SOURCE_CANDIDATE else self._sync_manual(rv)
        if res is None: return "PROPOSED" if rv.source == SOURCE_CANDIDATE else "NO_SUGGESTION"
        if self.is_confirmed(si, line): return "CONFIRMED"
        return {"ALLOWED": "PROPOSED", "ALLOWED_CONFIRMED": "PROPOSED", "BLOCKED_UNCONFIRMED_WARNING": "NEEDS_CONFIRMATION"}.get(res["decision"], "BLOCKED")

    def progress(self) -> dict:
        total = len(self.reviews); done = sum(1 for k in self.reviews if self.is_confirmed(*k)); return dict(confirmed=done, total=total, all_confirmed=total > 0 and done == total)

    def export_for_measurement(self) -> tuple:
        """(annotations, confirmations) for Stage E2 -- ONLY boundaries whose visitor confirmation is still valid.  Raises if any is missing / stale / blocked."""
        anns, confs, bad = [], [], []
        want = [(si, ln) for si in range(self.job.n_sets) for ln in BOUNDARY_ORDER]
        for si, ln in want:
            if (si, ln) not in self.reviews: bad.append(f"set {si} {ln}: no review state"); continue
            p = self.acceptance_problems(si, ln)
            if p: bad.append(f"set {si} {label_of(ln)}: {'; '.join(p)}"); continue
            rv = self.reviews[(si, ln)]; anns.append(rv.raw)
            if rv.confirmation is not None: confs.append(rv.confirmation)
        if bad: raise F6Error("cannot measure: " + " | ".join(bad))
        return anns, confs

    # ---- step 6: measurement (Stage E2, unchanged)
    def pending_set_confirmations(self) -> list:
        """Set-level Stage E2 warnings the visitor still has to confirm explicitly (after annotations were submitted)."""
        rv = (self.job.last_result or {}).get("annotation_review") or {}; have = {c.line_name for c in self.set_confirmations.values()}
        return [q for q in rv.get("confirmation_requests", []) if q["scope"] == "set_issue" and q["line_name"] not in have]

    def confirm_set_issue(self, request: Mapping):
        c = VM.Confirmation.from_request(request, VISITOR_ID, "the visitor accepted this set-level warning"); self.set_confirmations[(c.set_index, c.line_name)] = c

    def measure(self) -> dict:
        """Submit the confirmed annotations and run the unchanged Stage E2 measurement.  Ground truth is not an input.  On failure NO dimensions are returned."""
        if self.job.stage not in ("frames_selected", "annotations_reviewed", "measured"): raise F6Error("select frames and review the boundaries first")
        anns, confs = self.export_for_measurement()
        r = self.job.submit_annotations(anns)
        if not r["can_proceed"] and r["failures"]: self.measurement = r; return r
        r = self.job.submit_confirmations(confs + list(self.set_confirmations.values()))      # Stage E2 rejects any confirmation whose digests no longer match
        if not r.get("can_proceed"): self.measurement = r; return r
        r = self.job.measure(); self.measurement = r; self._note("measure", r); return r

    def dimensions(self) -> dict | None:
        m = self.measurement; return None if not m or m.get("step") != "measure" or not m.get("can_proceed") else m.get("dimensions")

    # ---- optional actual dimensions (validation only)
    def set_actual_dimensions(self, width, height, unit: str = "mm") -> ActualDimensions:
        self.actual = parse_actual_dimensions(width, height, unit); return self.actual

    def validation(self) -> dict: return validation_report(self.dimensions(), self.actual)

    # ---- plumbing
    def _reset_downstream(self):
        self.detections.clear(); self.reviews.clear(); self.detection_meta = {}; self._manual = None; self.set_confirmations.clear(); self.measurement = None

    def _note(self, step, r): self.log.append(dict(step=step, status=r.get("overall_status"), can_proceed=r.get("can_proceed"), t=time.time()))

    def summary_for_export(self) -> dict:
        return to_jsonable(dict(schema=SCHEMA, session_id=self.session_id, top_variant=VM.TOP_VARIANT, n_sets=self.n_sets, ground_truth_used_in_estimation=False, no_machine_learning=True, calibration_provenance=self.calibration_provenance,
                                detections={f"set{si}:{ln}": dict(status=b.status, selected_rank=b.selected_rank, candidates=b.candidates, diagnostics=b.diagnostics, reasons=b.reasons) for (si, ln), b in self.detections.items()},
                                acceptances=[rv.acceptance for rv in self.reviews.values() if rv.acceptance], dimensions=self.dimensions(), validation=self.validation() if self.measurement else None))


# ------------------------------------------------------------------ rendering helper (shared by the page and the screenshot script)
def overlay_image(sess: F6Session, si: int, line: str, max_wh=(560, 900)) -> np.ndarray:
    """Full frame with the proposed line (amber, dashed) and the Stage D refined line (green) -- both mapped into the distorted frame the visitor sees."""
    rv = sess.review(si, line); gray = sess.load_gray(si, rv.view_key); full = (gray.shape[1], gray.shape[0]); view = f1core.make_view(full, f1core.ViewState(), max_wh=max_wh)
    a = f1core.association(line); K, D = (sess.job.cal.K_left, sess.job.cal.D_left) if a["camera"] == "left" else (sess.job.cal.K_right, sess.job.cal.D_right)
    res = sess.evaluate(si, line); clicked = refined = None
    if rv.source == SOURCE_CANDIDATE and rv.rank is not None:
        cl = sess.candidates(si, line)[rv.rank - 1]["line"]; clicked = f1core.line_polyline(dict(cl, span=tuple(cl["span"])), K, D)
    if res and res.get("refined_line"): refined = f1core.line_polyline(res["refined_line"], K, D)
    elif res and res.get("initial_line") and clicked is None: clicked = f1core.line_polyline(res["initial_line"], K, D)
    pts = sess.manual_session().state(line, si).points if rv.source == SOURCE_MANUAL else ()
    return f1core.render_view(gray, view, pts, clicked, refined)
