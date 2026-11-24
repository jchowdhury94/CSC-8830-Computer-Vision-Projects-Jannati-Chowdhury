"""Three-viewpoint Wall B: input checks, sandboxed run and result assembly (no Streamlit import).

Nothing about resources is new: the run goes through the SAME supervised worker machinery as the two-viewpoint reconstruction (admission through the shared memory budget, wall-clock timeout,
process-tree memory ceiling, worker-side peak-RSS watchdog, process-tree kill, workspace deletion) with the SAME limits object (300 s, 3072 MB request, 1024 MB minimum grant).  Only the worker
program differs (``three_view_worker``).  That machinery lives in the reconstruction backend, which only ``f8_measurement`` may name: it is handed to ``prepare`` / ``run`` as the first argument
``RR`` and this module never imports it.  The runner is reached ONLY through ``f8_measurement.MeasurementService.start`` after the processing gate.

``prepare`` is the typed input check (like the two-viewpoint input check): it collects every problem, corrects and substitutes nothing, and builds the staging request from the visitor's own
EXTRACTED frames (the very frames the boundaries were marked on, verified by fingerprint), never from a re-decoded video.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(MODULE.parent), str(HERE), str(MODULE / "f8_chunk_poc")):   # MODULE.parent: the shared stereo_measure package
    if _p not in sys.path:
        sys.path.insert(0, _p)

import room_frames as RF                                                             # noqa: E402
import room_videos as RV                                                             # noqa: E402
import three_view_result as TVR                                                      # noqa: E402
import three_view_workflow as TVW                                                    # noqa: E402
from room_measurement import three_view_wall_b as T                                  # noqa: E402

WORKER = HERE / "three_view_worker.py"
SLOTS = RV.ROOM_UPLOAD_SLOTS_3

# Phase D.3: admission minimum for a three-viewpoint worker.  The shared default (1024 MB) is far below what this worker needs, so a container with, say, 1.5 GB free would START the
# job and have it stopped mid-run.  MEASURED on Streamlit Community Cloud (guarded diagnostic, synthetic images, one frame set at 2160 x 3840, six detections + five matches): worker peak RSS
# 2418.1 MB.  2450 MB is that figure rounded up (the diagnostic's own worker cap).  It is a measurement of GENERATED images, not of real recordings, and three sets in one process were
# not measured; the value is configurable (server environment ``F8_THREE_VIEW_MIN_MB``, 1024-8192) and may only RAISE a limit that the operator already set.
THREE_VIEW_MIN_TREE_MB = 2450.0
THREE_VIEW_MIN_ENV, THREE_VIEW_MIN_RANGE_MB = "F8_THREE_VIEW_MIN_MB", (1024.0, 8192.0)


def three_view_min_mb(env=None) -> float:
    import math
    import os
    env = os.environ if env is None else env
    try:
        v = float(str(env[THREE_VIEW_MIN_ENV]).strip())
    except (KeyError, ValueError):
        return THREE_VIEW_MIN_TREE_MB
    return v if math.isfinite(v) and THREE_VIEW_MIN_RANGE_MB[0] <= v <= THREE_VIEW_MIN_RANGE_MB[1] else THREE_VIEW_MIN_TREE_MB      # an invalid value is ignored, never clamped


def three_view_limits(RR, limits=None):
    """``limits`` (default: the trusted ``ReconstructionLimits``) with the admission minimum raised to ``three_view_min_mb()``.  Never lowers a minimum that is already higher, and a
    request ceiling below the minimum (``max_tree_rss_mb``) still wins, exactly as in ``resource_budget.reserve``.  The wall timeout and the CPU-time limit are NOT touched."""
    limits = limits if limits is not None else RR.ReconstructionLimits()
    floor = three_view_min_mb()
    return limits if limits.min_tree_rss_mb >= floor else dataclasses.replace(limits, min_tree_rss_mb=floor)


def _resource_wording(fail: dict, info: dict | None) -> dict:
    """Clearer visitor wording for a refusal to START (nothing ran).  The typed code and the technical message are unchanged."""
    code, busy_with = fail.get("code"), fail.get("busy_with")
    if code == "INSUFFICIENT_MEMORY":
        need, free = fail.get("minimum_mb"), fail.get("available_mb")
        lim = ((info or {}).get("budget") or {}).get("limitation")
        fail["visitor_message"] = ("This server does not have enough free memory right now to measure three viewpoints"
                                   + (f" (about {need:.0f} MB are needed and {free:.0f} MB are free)" if isinstance(need, (int, float)) and isinstance(free, (int, float)) else "")
                                   + ". Nothing was started and nothing was concluded about your room; this is not a problem with your videos."
                                   + (" The server could not determine its own memory size, so it declines large jobs to be safe." if lim else "") + " Try again in a few minutes.")
    elif code == "WORKER_BUSY" and busy_with in ("light", "heavy", "light_limit"):
        fail["visitor_message"] = "The server is busy with another visitor's videos or measurement. Nothing was started. Try again in a minute."
    return fail


@dataclass
class StagedRequest:
    """The duck-typed request the shared staging code needs (``owner``, ``frames``, ``calibration``, ``line_sets``)."""
    owner: str
    calibration: dict
    frames: tuple
    line_sets: tuple = ()


@dataclass
class Prepared:
    request: StagedRequest | None
    failures: list
    meta: dict
    annotations: dict | None = None
    keys: tuple = ()


# Reused failure codes whose stock wording speaks of four videos, five frame sets or six boundaries: this method's own sentence is attached instead (the code is unchanged).
_THREE_VIEW_WORDING = {
    "TOO_FEW_USABLE_SETS": "Fewer than two frame sets have all four vertical wall corners marked. Mark the four vertical corners (left camera and right camera at the left end and at the right end of the wall) in at least two frame sets.",
    "UPLOAD_EXPIRED": "An uploaded video is no longer available because it expired. Upload the six room videos again.",
    "FRAMES_NOT_EXTRACTED": "The frame selection has not been confirmed, or a chosen frame has not been read yet. Go back and confirm the frames.",
    "ANNOTATION_MISSING": "A wall corner line is missing. Mark the four vertical wall corners in each frame set you want to use.",
    "ANNOTATION_INVALID": "A wall corner line cannot be used. Mark it again.",
    "VIDEO_REPLACED": "A room video was replaced after the frames were chosen. Choose the frames and mark the wall corners again.",
    "CALIBRATION_CHANGED": "The calibration changed after the frames were chosen. Choose the frames and mark the wall corners again.",
}


def _failure(RR, code, msg, *, stage, **kw) -> dict:
    f = RR.failure(code, msg, stage=stage, **kw); c = str(getattr(code, "value", code))
    if c in _THREE_VIEW_WORDING:
        f["visitor_message"] = _THREE_VIEW_WORDING[c]
    return f


def _ok_stage(code: str) -> str:
    return "annotations" if code.startswith("ANNOTATION") else "frames" if code.startswith("FRAME") or code == "TOO_FEW_USABLE_SETS" else "videos" if code in ("UPLOAD_EXPIRED", "VIDEO_REPLACED", "VIDEO_HASH_MISMATCH", "CALIBRATION_VIDEO_SIZE_MISMATCH", "JOB_TOO_LARGE") else "calibration"


def prepare(RR, wf, owner: str, contract, cal, sources: dict, limits=None, min_usable: int | None = None) -> Prepared:
    """Verify EVERY input of one visitor's three-viewpoint Wall B and build the staging request.  Collects all typed problems instead of stopping at the first."""
    limits = limits or RR.ReconstructionLimits()
    from room_measurement import three_view_quality as Q
    min_usable = Q.DEFAULT_THRESHOLDS.min_usable_frames if min_usable is None else min_usable
    fails: list = []
    add = lambda code, msg, stage=None, **kw: fails.append(_failure(RR, code, msg, stage=stage or _ok_stage(str(getattr(code, 'value', code))), **kw))      # noqa: E731
    meta: dict = {}
    try:
        wf.assert_owner(owner)
    except RF.WorkflowError:
        return Prepared(None, [RR.failure(RR.Code.OWNERSHIP_MISMATCH, "the supplied session does not own this workflow", stage="ownership")], meta)
    if not isinstance(wf, TVW.ThreeViewWorkflow):
        return Prepared(None, [RR.failure(RR.Code.REQUEST_INVALID, "this wall does not use the three-viewpoint workflow", stage="request")], meta)
    if contract is None or cal is None:
        return Prepared(None, [RR.failure(RR.Code.CALIBRATION_NOT_ACCEPTED, "no accepted calibration contract was supplied", stage="calibration")], meta)
    try:
        params = RR.calibration_params(contract, cal); T.calibration_from_contract(dict(K1=params["K_left"], D1=params["D_left"], K2=params["K_right"], D2=params["D_right"],
                                                                                         image_size_wh=params["image_size_wh"], baseline_mm=params["baseline_mm"]))
    except Exception as ex:
        return Prepared(None, [RR.failure(RR.Code.CALIBRATION_INVALID, f"{type(ex).__name__}: {ex}", stage="calibration")], meta)
    if not (params["baseline_mm"] > 0):
        add(RR.Code.BASELINE_INVALID, f"baseline {params['baseline_mm']!r} mm")
    if not (RR._arr_eq(cal.K_left, params["K_left"]) and RR._arr_eq(cal.K_right, params["K_right"]) and RR._arr_eq(cal.D_left, params["D_left"]) and RR._arr_eq(cal.D_right, params["D_right"])
            and tuple(int(v) for v in cal.image_size) == tuple(params["image_size_wh"]) and float(cal.baseline_mm) == params["baseline_mm"]):
        add(RR.Code.CALIBRATION_INCONSISTENT, "the calibration object and the accepted contract differ (K, D, image size or baseline)")
    if params["calibration_id"] != wf.binding.calibration:
        add(RR.Code.CALIBRATION_CHANGED, "the accepted calibration is not the one the frames were chosen under")
    wh = tuple(params["image_size_wh"]); vids = {}; bound = {s: (h, u) for s, h, u in wf.binding.videos}
    for slot in SLOTS:
        label = RV.SLOT_INFO[slot][2]; cur = sources.get(slot) if isinstance(sources, dict) else None
        if cur is None:
            add(RR.Code.UPLOAD_EXPIRED, f"the {label} is missing or expired", slot=slot); continue
        if slot not in bound or (cur.sha256, cur.upload_id) != bound[slot]:
            add(RR.Code.VIDEO_REPLACED, f"the {label} is not the upload the frames were chosen from", slot=slot)
        p = Path(cur.path)
        try:
            ok_file = p.is_file() and not p.is_symlink()
        except OSError:
            ok_file = False
        if not ok_file:
            add(RR.Code.UPLOAD_EXPIRED, f"the stored file of the {label} is gone", slot=slot); continue
        if tuple(cur.wh) != wh:
            add(RR.Code.CALIBRATION_VIDEO_SIZE_MISMATCH, f"the {label} is {cur.wh[0]} x {cur.wh[1]} px but the accepted calibration is {wh[0]} x {wh[1]} px (intrinsics are never rescaled)", slot=slot)
        if limits.verify_video_hashes:
            try:
                if p.stat().st_size > limits.job_limits.video.max_file_bytes:
                    add(RR.Code.JOB_TOO_LARGE, f"the {label} exceeds the room-video size limit", slot=slot); continue
                if RR.sha256_stream(p) != cur.sha256:
                    add(RR.Code.VIDEO_HASH_MISMATCH, f"the stored {label} no longer matches its recorded SHA-256", slot=slot)
            except OSError:
                add(RR.Code.UPLOAD_EXPIRED, f"the stored file of the {label} could not be read", slot=slot)
        vids[slot] = dict(sha256=cur.sha256, frame_count=int(cur.frame_count))
    if not wf.confirmed:
        add(RR.Code.FRAMES_NOT_EXTRACTED, "the frame selection has not been confirmed")
    if wf.n_sets > limits.job_limits.max_frame_sets:
        add(RR.Code.JOB_TOO_LARGE, f"{wf.n_sets} frame sets exceed the limit of {limits.job_limits.max_frame_sets}")
    if wh[0] * wh[1] > limits.job_limits.max_frame_pixels:
        add(RR.Code.JOB_TOO_LARGE, f"frames of {wh[0]} x {wh[1]} px exceed the per-frame limit")
    usable = wf.usable_sets(); excluded = [i for i in range(wf.n_sets) if i not in usable]
    if len(usable) < min_usable:
        add(RR.Code.TOO_FEW_USABLE_SETS, f"{len(usable)} of {wf.n_sets} frame sets have all four vertical corners marked; at least {min_usable} are needed")
    ws_root = Path(wf.ws.path).resolve(); frames, key_list = [], []
    for i in usable:
        fs = {}
        for slot in SLOTS:
            rec = wf.full.get((i, slot))
            if rec is None or rec["index"] != wf.sets[i][slot]:
                add(RR.Code.FRAMES_NOT_EXTRACTED, f"frame set {i + 1}, {slot}: frame {wf.sets[i][slot]} has not been extracted", frame_set=i, view=slot); continue
            src = Path(rec["path"])
            try:
                inside = ws_root in src.resolve().parents and src.is_file() and not src.is_symlink()
            except OSError:
                inside = False
            if not inside:
                add(RR.Code.FRAME_STALE, f"frame set {i + 1}, {slot}: the extracted frame is not in this visitor's workspace", frame_set=i, view=slot); continue
            try:
                g = wf.frame_gray(i, slot); shape_ok = (g.shape[1], g.shape[0]) == wh; del g          # deep: re-read and re-fingerprint (a mismatch raises)
            except RF.WorkflowError as ex:
                add(RR.Code.FRAME_STALE, f"frame set {i + 1}, {slot}: {ex}", frame_set=i, view=slot); continue
            if not shape_ok:
                add(RR.Code.FRAME_SIZE_MISMATCH, f"frame set {i + 1}, {slot} does not have the calibrated size {wh}", frame_set=i, view=slot); continue
            fs[slot] = dict(path=str(src), index=int(rec["index"]), fingerprint=rec["fingerprint"])
        frames.append(fs); key_list.append(i)
    doc = None
    if not fails:
        try:
            doc = wf.annotation_document(only_usable=True); T.check_annotations_complete(T.parse_annotations(doc, wh), key_list)
        except T.ThreeViewError as e:
            add(RR.Code.ANNOTATION_INVALID if not e.code.endswith("MISSING") else RR.Code.ANNOTATION_MISSING, e.message)
    meta = dict(calibration=dict(calibration_id=params["calibration_id"], baseline_mm=params["baseline_mm"], image_size_wh=list(wh), capture_mode=params.get("capture_mode")), videos=vids,
                frame_sets={str(i): dict(wf.sets[i]) for i in key_list}, source_sets=list(key_list), excluded_sets=list(excluded))
    if fails:
        return Prepared(None, fails, meta)
    return Prepared(StagedRequest(RF.owner_digest(owner), params, tuple(frames)), [], meta, doc, tuple(key_list))


def _config_dict(config: "T.ThreeViewConfig") -> dict:
    d = dataclasses.asdict(config); d.pop("thresholds", None); return d


def run(RR, wf, owner: str, contract, cal, sources: dict, limits=None, *, config: "T.ThreeViewConfig | None" = None, worker_cmd=None, workspace_root=None, cancel=None, coordinator=None,
        retention=None) -> dict:
    """Check every input, then run the three-viewpoint reconstruction once in the sandbox.  Always returns a result dict (SUCCESS, QUESTIONABLE or FAILED); never raises.
    ``retention`` protects the uploads for the whole call and is always released.  ``config`` defaults to the fast matcher; ``ThreeViewConfig.original_matcher()`` selects the original one."""
    limits = three_view_limits(RR, limits); config = config or T.ThreeViewConfig(); t0 = time.monotonic(); meta: dict = {}
    try:
        if retention is not None:
            f = retention.acquire(sources, max_s=limits.timeout_s + 180.0)
            if f is not None:
                return TVR.failed(meta, [f], seconds=time.monotonic() - t0)
        if limits.max_tree_rss_mb < limits.min_tree_rss_mb:                            # Phase D.3: a ceiling below the three-viewpoint minimum would otherwise be admitted with a grant the worker cannot live in
            f = RR.failure(RR.Code.INSUFFICIENT_MEMORY, f"the configured memory ceiling ({limits.max_tree_rss_mb:.0f} MB) is below the {limits.min_tree_rss_mb:.0f} MB this measurement needs; nothing was started",
                           stage="admission", minimum_mb=limits.min_tree_rss_mb, ceiling_mb=limits.max_tree_rss_mb)
            f["visitor_message"] = ("This server is set up with less memory than measuring three viewpoints needs. Nothing was started and nothing was concluded about your room; "
                                    "this is not a problem with your videos.")
            return TVR.failed(meta, [f], seconds=time.monotonic() - t0)
        try:
            prep = prepare(RR, wf, owner, contract, cal, sources, limits)
        except Exception as ex:
            return TVR.failed(meta, [RR.failure(RR.A.FailureReason.INTERNAL_ERROR, f"{type(ex).__name__}: {ex}", stage="prepare")], seconds=time.monotonic() - t0)
        meta = prep.meta
        if prep.request is None:
            return TVR.failed(meta, prep.failures, seconds=time.monotonic() - t0)
        extra = dict(annotations=prep.annotations, keys=list(prep.keys), config=_config_dict(config))

        def cmd(job):                                                                  # called by the shared supervisor after it staged the frames: add this method's own input file
            (Path(job) / "three_view.json").write_text(json.dumps(extra, allow_nan=False))
            return list(worker_cmd(job)) if worker_cmd else [sys.executable, "-I", str(WORKER), str(job)]
        out, fail, info = RR._run_worker(prep.request, limits, worker_cmd=cmd, workspace_root=workspace_root, cancel=cancel, coordinator=coordinator, retention=retention)
        if fail is not None:
            return TVR.failed(meta, [_resource_wording(fail, info)], process_info=info, seconds=time.monotonic() - t0)
        if not (isinstance(out, dict) and out.get("schema") == "f8.three_view_worker/1" and (out.get("result") is not None or out.get("failures"))):
            return TVR.failed(meta, [RR.failure(RR.Code.WORKER_OUTPUT_INVALID, "the worker result is not the expected document", stage="worker")], process_info=info, seconds=time.monotonic() - t0)
        return TVR.assemble(meta, worker_out=out, process_info=info, seconds=time.monotonic() - t0)
    except Exception as ex:                                                            # nothing may escape as an untyped exception
        return TVR.failed(meta, [RR.failure(RR.A.FailureReason.INTERNAL_ERROR, f"{type(ex).__name__}: {ex}", stage="internal")], seconds=time.monotonic() - t0)
    finally:
        if retention is not None:
            retention.release()
