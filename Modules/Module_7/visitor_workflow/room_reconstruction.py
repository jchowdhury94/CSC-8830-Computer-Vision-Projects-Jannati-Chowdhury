"""F8.31C-1: visitor Method J BACKEND (no Streamlit import, wired to nothing live, ``ROOM_PROCESSING_ENABLED`` is not consulted and not changed).

Same calibration algorithm.  Same Method J reconstruction.  Different, visitor-provided data.

    accepted visitor calibration + four uploaded room videos + confirmed frame sets + accepted wall-boundary lines
        -> ``prepare`` (typed input checks, nothing is corrected silently)
        -> a private job directory (frames copied byte for byte, request file)
        -> ``reconstruction_worker`` in a SANDBOXED subprocess (isolated interpreter, scrubbed environment, own process group,
           wall-clock timeout, process-tree memory guard, process-tree kill, job directory deleted)
              original ``features.match_set`` per frame set (SIFT, ratio test, mutual check; unchanged parameters)
              ``visitor_method_j_adapter.evaluate``  (``pose.build_dataset`` / ``pose.estimate_pose`` / the original geometry and ``measure_wall``)
        -> ``assemble_result``: a JSON-safe result that keeps four things apart:
              1. visitor input problems          (``failure_class = visitor_input``)
              2. an original Method J rejection  (``failure_class = method_j``)
              3. worker / resource failures      (``failure_class = worker_resource``)
              4. advisory quality warnings       (``warnings``; never a failure, never change a number)

What this module does NOT do: no mathematics (everything numerical is the original Method J, called through the validated adapter); no calibration
(the calibration is read from the accepted visitor contract and never modified); no Recipe A value, no historical baseline, no original annotation, no
ground truth, no historical file is read; no resizing, cropping, rotation or resampling of any frame; no confidence interval, no percentage error,
no Monte Carlo; no UI.  A FAILED result carries no dimensions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(HERE), str(MODULE / "f8_chunk_poc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import room_frames as RF                                                             # noqa: E402
import room_videos as RV                                                             # noqa: E402
import resource_budget as RB                                                          # noqa: E402  (F8.31C-1C: container-aware budget shared with the calibration runner)
import visitor_messages as VM                                                        # noqa: E402  (F8.31C-1C: visitor wording, data only)
from monitor import read_cgroup, snapshot_processes, tree_pids                       # noqa: E402  (the existing process-tree sampler)
from room_measurement import visitor_method_j_adapter as A                            # noqa: E402
from room_measurement.visitor_measurement import JobLimits                            # noqa: E402

SCHEMA_REQUEST = "f8.visitor_reconstruction_request/1"
SCHEMA_RESULT = "f8.visitor_reconstruction/1"
WORKER = HERE / "reconstruction_worker.py"
MiB = 1024 * 1024
SUCCESS, QUESTIONABLE, FAILED = A.SUCCESS, A.QUESTIONABLE, A.FAILED
REQUIRED_FRAME_SETS = RF.DEFAULT_SETS                  # the original methodology: five frame sets
REQUIRED_LINES = tuple(RF.REQUIRED_LINES)              # AL_corner PL_corner PL_ceilF AR_corner PR_corner PR_ceilF (top_variant 0), identical to the adapter's
VIEW_KEYS = tuple(RF.VIEW_KEYS)                        # AL PL AR PR
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")


# ------------------------------------------------------------------ typed failures
class FailureClass(str, Enum):
    VISITOR_INPUT = "visitor_input"          # an input is missing, stale, expired, replaced, inconsistent or invalid; the visitor can act on it
    METHOD_J = "method_j"                    # the ORIGINAL Method J rejected the data (or could not match features) after the inputs were accepted
    WORKER_RESOURCE = "worker_resource"      # timeout, memory limit, crash, busy, storage: nothing was concluded about the data


class Code(str, Enum):
    """Failure codes raised by this layer (the adapter's own ``FailureReason`` codes pass through unchanged)."""
    OWNERSHIP_MISMATCH = "OWNERSHIP_MISMATCH"
    CALIBRATION_NOT_ACCEPTED = "CALIBRATION_NOT_ACCEPTED"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CALIBRATION_CHANGED = "CALIBRATION_CHANGED"
    CALIBRATION_INCONSISTENT = "CALIBRATION_INCONSISTENT"
    BASELINE_INVALID = "BASELINE_INVALID"
    CALIBRATION_VIDEO_SIZE_MISMATCH = "CALIBRATION_VIDEO_SIZE_MISMATCH"
    UPLOAD_EXPIRED = "UPLOAD_EXPIRED"
    VIDEO_REPLACED = "VIDEO_REPLACED"
    VIDEO_HASH_MISMATCH = "VIDEO_HASH_MISMATCH"
    FRAME_SET_COUNT = "FRAME_SET_COUNT"
    TOO_FEW_USABLE_SETS = "TOO_FEW_USABLE_SETS"
    FRAME_SET_INCOMPLETE = "FRAME_SET_INCOMPLETE"
    FRAMES_NOT_EXTRACTED = "FRAMES_NOT_EXTRACTED"
    FRAME_STALE = "FRAME_STALE"
    FRAME_SIZE_MISMATCH = "FRAME_SIZE_MISMATCH"
    ANNOTATION_MISSING = "ANNOTATION_MISSING"
    ANNOTATION_STALE = "ANNOTATION_STALE"
    ANNOTATION_INVALID = "ANNOTATION_INVALID"
    REQUEST_INVALID = "REQUEST_INVALID"
    FEATURE_MATCHING_FAILED = "FEATURE_MATCHING_FAILED"      # the original match_set raised (too few correspondences): the original pipeline's own failure
    WORKER_TIMEOUT = "WORKER_TIMEOUT"
    WORKER_MEMORY_LIMIT = "WORKER_MEMORY_LIMIT"
    WORKER_MEMORY_UNMEASURABLE = "WORKER_MEMORY_UNMEASURABLE"
    WORKER_CRASHED = "WORKER_CRASHED"
    WORKER_OUTPUT_INVALID = "WORKER_OUTPUT_INVALID"
    WORKER_CANCELLED = "WORKER_CANCELLED"
    WORKER_BUSY = "WORKER_BUSY"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"              # F8.31C-1C: the shared memory budget could not grant the minimum; nothing was started (distinct from BUSY)
    JOB_TOO_LARGE = "JOB_TOO_LARGE"
    WORKSPACE_ERROR = "WORKSPACE_ERROR"


_VISITOR_INPUT = {c for c in Code if c.name in {
    "OWNERSHIP_MISMATCH", "CALIBRATION_NOT_ACCEPTED", "CALIBRATION_INVALID", "CALIBRATION_CHANGED", "CALIBRATION_INCONSISTENT", "BASELINE_INVALID", "CALIBRATION_VIDEO_SIZE_MISMATCH",
    "UPLOAD_EXPIRED", "VIDEO_REPLACED", "VIDEO_HASH_MISMATCH", "FRAME_SET_COUNT", "TOO_FEW_USABLE_SETS", "FRAME_SET_INCOMPLETE", "FRAMES_NOT_EXTRACTED", "FRAME_STALE", "FRAME_SIZE_MISMATCH",
    "ANNOTATION_MISSING", "ANNOTATION_STALE", "ANNOTATION_INVALID", "REQUEST_INVALID"}}
_WORKER = {Code.WORKER_TIMEOUT, Code.WORKER_MEMORY_LIMIT, Code.WORKER_MEMORY_UNMEASURABLE, Code.WORKER_CRASHED, Code.WORKER_OUTPUT_INVALID, Code.WORKER_CANCELLED, Code.WORKER_BUSY,
           Code.INSUFFICIENT_MEMORY, Code.JOB_TOO_LARGE, Code.WORKSPACE_ERROR}
_ADAPTER_INPUT = {A.FailureReason.INPUT_NO_FRAME_SETS.value, A.FailureReason.INPUT_MALFORMED_FRAME_SET.value, A.FailureReason.INPUT_LINES_MISSING.value,
                  A.FailureReason.INPUT_IMAGE_SIZE_MISMATCH.value, A.FailureReason.INPUT_INVALID_CALIBRATION.value, A.FailureReason.LINE_RETARGET_FAILED.value,
                  A.FailureReason.BASELINE_CHANGED_UNLABELLED.value}
_ADAPTER_INTERNAL = {A.FailureReason.INTERNAL_ERROR.value}


def failure_class_of(code: str) -> str:
    """Which of the three failure families a typed code belongs to (advisory warnings are not failures and have no class)."""
    code = str(code)
    if code in {c.value for c in _VISITOR_INPUT} or code in _ADAPTER_INPUT:
        return FailureClass.VISITOR_INPUT.value
    if code in {c.value for c in _WORKER} or code in _ADAPTER_INTERNAL:
        return FailureClass.WORKER_RESOURCE.value
    return FailureClass.METHOD_J.value                     # FEATURE_MATCHING_FAILED, DATASET_BUILD_FAILED, POSE_*, *_FAILED geometry codes, NO_USABLE_FRAME_SETS, ...


# F8.31C-1C: the wording lives in ``visitor_messages`` (data only).  Every code of this layer and of the adapter has an entry; an unknown code gets the generic
# wording, never the technical message.  The technical ``message`` is kept beside it on every record.
VISITOR_MESSAGES = {code: VM.failure_text(code) for code in VM.FAILURE_TEXT}


def failure(code, message: str, *, stage: str, **extra) -> dict:
    """One typed failure record: ``code``, ``failure_class``, ``stage``, a technical ``message`` and a visitor-readable ``visitor_message``."""
    code = code.value if isinstance(code, Enum) else str(code)
    return dict(code=code, failure_class=failure_class_of(code), stage=stage, message=message, visitor_message=VM.failure_text(code), **extra)


# ------------------------------------------------------------------ limits
DEFAULT_CPU_LIMIT_S = 1800.0                  # the long-standing worker CPU-time limit (all threads).  Unchanged by Phase D.3.
CPU_LIMIT_ENV, CPU_LIMIT_RANGE_S = "F8_RECON_CPU_S", (60.0, 7200.0)


def default_cpu_limit_s(env=None) -> float:
    """The CPU-time limit handed to a reconstruction worker.  ``F8_RECON_CPU_S`` (60-7200) set in the SERVER's environment overrides the default 1800 s; an invalid value is ignored, never
    clamped.  (Until now the worker read this variable itself, but its environment is scrubbed, so the override could never reach it: the limit was fixed at 1800 s in practice.)"""
    env = os.environ if env is None else env
    try:
        v = float(str(env[CPU_LIMIT_ENV]).strip())
    except (KeyError, ValueError):
        return DEFAULT_CPU_LIMIT_S
    return v if math.isfinite(v) and CPU_LIMIT_RANGE_S[0] <= v <= CPU_LIMIT_RANGE_S[1] else DEFAULT_CPU_LIMIT_S


@dataclass(frozen=True)
class ReconstructionLimits:
    """Every bound of one reconstruction job.  ``job_limits`` is ``JobLimits(video=room_video_limits())`` (the room file-size ceiling, never F6's 200 MB)."""
    timeout_s: float = 300.0                  # wall clock of the worker (a previous original-data reconstruction took about 27 s)
    cpu_limit_s: float = field(default_factory=default_cpu_limit_s)   # Phase D.3: explicit CPU-time limit (all threads) written into the worker's request; 1800 s unless the server sets F8_RECON_CPU_S
    max_tree_rss_mb: float = 3072.0           # the ceiling this job would LIKE for its worker tree (a previous original-data run peaked near 2.45 GB; this is that figure plus headroom, a PLANNING value, not a Cloud measurement).  F8.31C-1C: it is a REQUEST to the
                                              # shared memory budget, not an entitlement: the ceiling actually enforced is the grant (<= this, <= what the container budget has left)
    min_tree_rss_mb: float = 1024.0           # F8.31C-1C: the job is not started with a smaller grant (a resource-safety planning value, not a threshold on any measurement;
                                              # the synthetic 1080 x 1920 worker peaked near 844 MB, the original 2160 x 3840 run near 2.45 GB, neither measured on Cloud)
    watchdog_s: float = 0.05                  # F8.31C-1C: the worker checks its OWN peak RSS this often (finer than the parent's poll); None-equivalent 0 disables
    adaptive_poll_fraction: float = 0.60      # above this fraction of the ceiling the parent polls at ``fast_poll_s``
    fast_poll_s: float = 0.05
    container_fraction: float = 0.90          # stop when the container's own anonymous memory exceeds this fraction of the container limit (when it can be read)
    heartbeat_s: float = 5.0                  # how often the upload-retention lease is renewed while the job runs
    poll_s: float = 0.25
    kill_grace_s: float = 3.0
    blind_polls_max: int = 20                 # consecutive polls without a usable RSS reading: the memory limit cannot be enforced, so stop (fail closed)
    max_job_bytes: int = 1024 * MiB           # staged frames + request on disk
    max_result_bytes: int = 8 * MiB
    required_frame_sets: int | None = REQUIRED_FRAME_SETS      # None: any number the F8.31B workflow allows (2 to JobLimits.max_frame_sets)
    min_usable_sets: int | None = None                         # F8.31D-6: None = STRICT (every selected set needs all six accepted lines, the earlier behaviour).  An integer = a set whose six
                                                               # boundaries were not all refined is SKIPPED and reported (the original recorded failed sets and measured the rest); fewer than
                                                               # this many usable sets is a failure.  Selected frames are still checked for ALL sets.
    verify_video_hashes: bool = True          # stream-hash every stored video (bounded chunks) before processing
    job_limits: JobLimits = field(default_factory=lambda: JobLimits(video=RV.room_video_limits()))


# ------------------------------------------------------------------ small helpers
def json_safe(o):
    """JSON-safe copy: numpy to Python, non-finite floats to ``None``, tuples to lists."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, Enum):
        return o.value
    return o


def sha256_stream(path, chunk=4 * MiB) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def serialize_line(line: dict) -> dict:
    """A fitted line (``lines.fit_lines`` shape: kind, a, b, n, N, rms, span, pts) as JSON.  Floats round-trip exactly; the dtype of ``pts`` is kept."""
    pts = np.asarray(line["pts"])
    return dict(kind=str(line["kind"]), a=float(line["a"]), b=float(line["b"]), n=int(line["n"]), N=int(line["N"]), rms=float(line["rms"]), span=[float(line["span"][0]), float(line["span"][1])],
                pts=pts.tolist(), pts_dtype=str(pts.dtype))


def line_digest(line: dict) -> str:
    """A digest of the fitted line's parameters (kind, a, b, span), for provenance only."""
    blob = json.dumps(dict(kind=line["kind"], a=f"{float(line['a']):.6f}", b=f"{float(line['b']):.6f}", span=[f"{float(x):.4f}" for x in line["span"]]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def deserialize_line(d: dict) -> dict:
    return dict(kind=d["kind"], a=float(d["a"]), b=float(d["b"]), n=int(d["n"]), N=int(d["N"]), rms=float(d["rms"]), span=(float(d["span"][0]), float(d["span"][1])),
                pts=np.asarray(d["pts"], dtype=np.dtype(d.get("pts_dtype", "float64"))).reshape(-1, 2))


def calibration_params(contract: dict, cal) -> dict:
    """The calibration exactly as the visitor accepted it (JSON-safe).  ``K1``/``D1`` are the LEFT camera, ``K2``/``D2`` the RIGHT camera."""
    return dict(K_left=np.asarray(contract["K1"], np.float64).tolist(), D_left=np.asarray(contract["D1"], np.float64).reshape(-1).tolist(),
                K_right=np.asarray(contract["K2"], np.float64).tolist(), D_right=np.asarray(contract["D2"], np.float64).reshape(-1).tolist(),
                image_size_wh=[int(v) for v in contract["image_size_wh"]], baseline_mm=float(contract["baseline_mm"]), baseline_unit="mm", camera_order=list(RF.CAMERA_ORDER),
                distortion_model=RF.DISTORTION_MODEL, capture_mode=dict(contract.get("capture_mode") or {}), calibration_id=RF.calibration_id(contract))


def context_from_params(p: dict) -> A.VisitorCalibrationContext:
    return A.VisitorCalibrationContext.from_params(dict(K_left=p["K_left"], D_left=p["D_left"], K_right=p["K_right"], D_right=p["D_right"], baseline_mm=p["baseline_mm"]),
                                                   tuple(p["image_size_wh"]), label="visitor")


# ------------------------------------------------------------------ the request
@dataclass(frozen=True)
class ReconstructionRequest:
    """A validated, self-contained description of one reconstruction.  Holds no pixels.  Built only by ``prepare`` (never from a default calibration)."""
    owner: str                      # RF.owner_digest of the visitor session
    calibration: dict               # ``calibration_params``
    frames: tuple                   # per frame set: {view key: dict(path, index, fingerprint)}
    line_sets: tuple                # per frame set: {line name: serialized line}
    videos: dict                    # slot -> dict(sha256, upload_id, size_bytes, frame_count, fps, wh)
    annotations: tuple              # provenance of every accepted boundary (no pixels)
    frame_table: tuple              # per set: {view key: original frame index}
    source_sets: tuple = ()         # F8.31D-6: the visitor-side set number (0-based) of each included set; sets that were skipped are listed in ``skipped_sets``
    skipped_sets: tuple = ()

    @property
    def n_sets(self) -> int:
        return len(self.frames)

    def structural_problems(self) -> list:
        out = []
        if not self.frames or len(self.frames) != len(self.line_sets):
            out.append("frame sets and line sets differ in number")
        for i, f in enumerate(self.frames):
            if set(f) != set(VIEW_KEYS):
                out.append(f"frame set {i}: needs exactly the views {list(VIEW_KEYS)}")
        for i, ls in enumerate(self.line_sets):
            if set(REQUIRED_LINES) - set(ls):
                out.append(f"frame set {i}: missing lines {sorted(set(REQUIRED_LINES) - set(ls))}")
            if set(ls) - set(REQUIRED_LINES):
                out.append(f"frame set {i}: unexpected lines {sorted(set(ls) - set(REQUIRED_LINES))}")
        return out


@dataclass
class Prepared:
    request: ReconstructionRequest | None
    failures: list                  # typed records; empty iff ``request`` is set
    checks: dict = field(default_factory=dict)


def _arr_eq(a, b) -> bool:
    try:
        return np.array_equal(np.asarray(a, np.float64), np.asarray(b, np.float64))
    except Exception:
        return False


def prepare(wf: RF.RoomFrameWorkflow, owner: str, contract, cal, sources: dict, limits: ReconstructionLimits | None = None) -> Prepared:
    """Verify EVERY input of one visitor and build the request.  Collects all typed problems instead of stopping at the first; corrects and substitutes nothing.

    ``wf``: the visitor's F8.31B workflow; ``owner``: the session token; ``contract`` / ``cal``: the accepted calibration (the Stage A contract and the F6
    calibration object); ``sources``: slot -> ``RF.SourceVideo`` as the upload store reports them RIGHT NOW (None for an expired or missing upload).
    Nothing is mutated here (``wf.sync`` is the caller's job)."""
    limits = limits or ReconstructionLimits(); fails: list = []
    add = lambda code, msg, stage, **kw: fails.append(failure(code, msg, stage=stage, **kw))                       # noqa: E731
    try:
        wf.assert_owner(owner)
    except RF.WorkflowError:
        return Prepared(None, [failure(Code.OWNERSHIP_MISMATCH, "the supplied session does not own this workflow", stage="ownership")])
    # ---- calibration
    if contract is None or cal is None:
        return Prepared(None, [failure(Code.CALIBRATION_NOT_ACCEPTED, "no accepted calibration contract was supplied", stage="calibration")])
    try:
        params = calibration_params(contract, cal); ctx = context_from_params(params); ctx.to_stereo_calibration()
    except Exception as ex:
        return Prepared(None, [failure(Code.CALIBRATION_INVALID, f"{type(ex).__name__}: {ex}", stage="calibration")])
    if not (math.isfinite(params["baseline_mm"]) and params["baseline_mm"] > 0):
        add(Code.BASELINE_INVALID, f"baseline {params['baseline_mm']!r} mm", "calibration")
    if not (_arr_eq(cal.K_left, params["K_left"]) and _arr_eq(cal.K_right, params["K_right"]) and _arr_eq(cal.D_left, params["D_left"]) and _arr_eq(cal.D_right, params["D_right"])
            and tuple(int(v) for v in cal.image_size) == tuple(params["image_size_wh"]) and float(cal.baseline_mm) == params["baseline_mm"]):
        add(Code.CALIBRATION_INCONSISTENT, "the calibration object and the accepted contract differ (K, D, image size or baseline)", "calibration")
    if params["calibration_id"] != wf.binding.calibration:
        add(Code.CALIBRATION_CHANGED, "the accepted calibration is not the one the frames were chosen under", "calibration")
    wh = tuple(params["image_size_wh"])
    # ---- videos
    vids = {}
    for slot in RV.ROOM_UPLOAD_SLOTS:
        label = RV.SLOT_INFO[slot][2]; cur = sources.get(slot) if isinstance(sources, dict) else None; prev = {s: (h, u) for s, h, u in wf.binding.videos}.get(slot)
        if cur is None:
            add(Code.UPLOAD_EXPIRED, f"the {label} is missing or expired", "videos", slot=slot); continue
        if prev is None or (cur.sha256, cur.upload_id) != prev:
            add(Code.VIDEO_REPLACED, f"the {label} is not the upload the frames were chosen from", "videos", slot=slot)
        p = Path(cur.path)
        try:
            ok_file = p.is_file() and not p.is_symlink()
        except OSError:
            ok_file = False
        if not ok_file:
            add(Code.UPLOAD_EXPIRED, f"the stored file of the {label} is gone", "videos", slot=slot); continue
        if tuple(cur.wh) != wh:
            add(Code.CALIBRATION_VIDEO_SIZE_MISMATCH, f"the {label} is {cur.wh[0]} x {cur.wh[1]} px but the accepted calibration is {wh[0]} x {wh[1]} px (intrinsics are never rescaled)", "videos", slot=slot)
        if limits.verify_video_hashes:
            try:
                if p.stat().st_size > limits.job_limits.video.max_file_bytes:
                    add(Code.JOB_TOO_LARGE, f"the {label} exceeds the room-video size limit", "videos", slot=slot); continue
                if sha256_stream(p) != cur.sha256:
                    add(Code.VIDEO_HASH_MISMATCH, f"the stored {label} no longer matches its recorded SHA-256", "videos", slot=slot)
            except OSError:
                add(Code.UPLOAD_EXPIRED, f"the stored file of the {label} could not be read", "videos", slot=slot)
        vids[slot] = dict(sha256=cur.sha256, upload_id=cur.upload_id, size_bytes=int(cur.size_bytes), frame_count=int(cur.frame_count), fps=float(cur.fps), wh=[int(cur.wh[0]), int(cur.wh[1])])
    # ---- frame sets
    n = wf.n_sets
    if limits.required_frame_sets is not None and n != limits.required_frame_sets:
        add(Code.FRAME_SET_COUNT, f"{n} frame sets were prepared; the original methodology uses {limits.required_frame_sets}", "frames")
    if n > limits.job_limits.max_frame_sets:
        add(Code.JOB_TOO_LARGE, f"{n} frame sets exceed the limit of {limits.job_limits.max_frame_sets}", "frames")
    if wh[0] * wh[1] > limits.job_limits.max_frame_pixels:
        add(Code.JOB_TOO_LARGE, f"frames of {wh[0]} x {wh[1]} px exceed the per-frame limit", "frames")
    ws_root = Path(wf.ws.path).resolve(); frames, table = [], []
    for i in range(n):
        fs, tb = {}, {}
        for vp in RF.VIEWPOINTS:
            pair = wf.pairs.get((vp, i))
            if pair is None or not pair.get("confirmed"):
                add(Code.FRAME_SET_INCOMPLETE, f"viewpoint {vp}, frame set {i + 1}: the frame pair is not chosen and confirmed", "frames", frame_set=i, viewpoint=vp); continue
            for side in RF.SIDES:
                vk = RF.VIEW_KEY[(vp, side)]; rec = wf.full.get((i, vk))
                if rec is None or rec["index"] != pair[side]:
                    add(Code.FRAMES_NOT_EXTRACTED, f"frame set {i + 1}, {vk}: the confirmed frame {pair[side]} has not been extracted", "frames", frame_set=i, view=vk); continue
                src = Path(rec["path"])
                try:
                    inside = ws_root in src.resolve().parents and src.is_file() and not src.is_symlink()
                except OSError:
                    inside = False
                if not inside:
                    add(Code.FRAME_STALE, f"frame set {i + 1}, {vk}: the extracted frame is not in this visitor's workspace", "frames", frame_set=i, view=vk); continue
                try:
                    g = wf.frame_gray(i, vk)                                              # deep: re-read and re-fingerprint (a mismatch raises)
                    shape_ok = (g.shape[1], g.shape[0]) == wh; del g
                except RF.WorkflowError as ex:
                    add(Code.FRAME_STALE, f"frame set {i + 1}, {vk}: {ex}", "frames", frame_set=i, view=vk); continue
                if not shape_ok:
                    add(Code.FRAME_SIZE_MISMATCH, f"frame set {i + 1}, {vk} does not have the calibrated size {wh}", "frames", frame_set=i, view=vk); continue
                fs[vk] = dict(path=str(src), index=int(rec["index"]), fingerprint=rec["fingerprint"]); tb[vk] = int(rec["index"])
        frames.append(fs); table.append(tb)
    # ---- annotations: exactly the six top_variant = 0 boundaries, accepted, on the frame that is still the selected one
    lines, prov, keep, skipped = [], [], [], []
    for i in range(n):
        ls = {}; pv = []
        if limits.min_usable_sets is not None and any(not getattr(wf.ann.get((i, nm)), "accepted", False) for nm in REQUIRED_LINES):
            skipped.append(i); continue                                   # F8.31D-6: reported and skipped, like a failed set in the original; never silently measured
        for name in REQUIRED_LINES:
            st = wf.ann.get((i, name)); vk = name[:2]
            if st is None or not st.accepted:
                add(Code.ANNOTATION_MISSING, f"frame set {i + 1}, {name}: not marked and accepted", "annotations", frame_set=i, line=name); continue
            f = wf.full.get((i, vk))
            if f is None or st.frame_fingerprint != f["fingerprint"] or st.frame_index != f["index"]:
                add(Code.ANNOTATION_STALE, f"frame set {i + 1}, {name}: made on a frame that is no longer the selected one", "annotations", frame_set=i, line=name); continue
            res = st.result
            if res is None or not res.downstream_permitted or res.downstream_line is None:
                add(Code.ANNOTATION_INVALID, f"frame set {i + 1}, {name}: the refined line is not usable", "annotations", frame_set=i, line=name); continue
            if len(st.points) != 2 or any(not (0 <= x <= wh[0] - 1 and 0 <= y <= wh[1] - 1) for x, y in st.points):
                add(Code.ANNOTATION_INVALID, f"frame set {i + 1}, {name}: the clicked points are missing or outside the {wh[0]} x {wh[1]} px frame", "annotations", frame_set=i, line=name); continue
            try:
                ls[name] = serialize_line(res.downstream_line)
                pv.append(dict(frame_set=i, line=name, points_distorted_full_px=[[float(x), float(y)] for x, y in st.points], status=res.status.value, accepted_warning_codes=list(st.acknowledged_codes),
                               warnings_acknowledged_automatically=bool(getattr(st, "auto", False)),
                               frame_index=int(f["index"]), frame_fingerprint=f["fingerprint"], refined_line_digest=line_digest(res.downstream_line)))
            except Exception as ex:
                add(Code.ANNOTATION_INVALID, f"frame set {i + 1}, {name}: {type(ex).__name__}: {ex}", "annotations", frame_set=i, line=name)
        lines.append(ls); prov.extend(pv); keep.append(i)
    if limits.min_usable_sets is not None and len(keep) < limits.min_usable_sets:
        add(Code.TOO_FEW_USABLE_SETS, f"{len(keep)} of {n} frame sets have all six boundaries refined; at least {limits.min_usable_sets} are needed", "annotations")
    if fails:
        return Prepared(None, fails, checks=dict(n_sets=n, image_size_wh=list(wh)))
    req = ReconstructionRequest(owner=RF.owner_digest(owner), calibration=params, frames=tuple(frames[i] for i in keep), line_sets=tuple(lines), videos=vids, annotations=tuple(prov),
                                frame_table=tuple(table[i] for i in keep), source_sets=tuple(keep), skipped_sets=tuple(skipped))
    bad = req.structural_problems()
    if bad:
        return Prepared(None, [failure(Code.REQUEST_INVALID, "; ".join(bad), stage="request")])
    return Prepared(req, [], checks=dict(n_sets=n, image_size_wh=list(wh)))


# ------------------------------------------------------------------ the sandbox
# F8.31C-1C: the old in-process ``BoundedSemaphore(1)`` is gone.  A module-level lock only coordinates the threads of ONE process and knew nothing about the calibration
# runner; the single reconstruction slot and the memory budget now live in ``resource_budget`` (cross-process, shared with calibration).


class ReconstructionWorkspace:
    """A private (0o700) directory for ONE job; removed as a whole.  Never shared between jobs or visitors."""

    def __init__(self, root=None):
        self.path = Path(tempfile.mkdtemp(prefix="f8_recon_", dir=None if root is None else str(root)))
        os.chmod(self.path, 0o700)

    def close(self):
        shutil.rmtree(self.path, ignore_errors=True)


def _scrubbed_env() -> dict:
    return {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}                # the passcode and every other secret are NOT passed


def _kill_tree(proc, grace_s: float):
    """SIGTERM the process group, then SIGKILL it and every descendant that is still visible (even one that left the group); always reaps the child."""
    if proc is None:
        return
    try:
        table = snapshot_processes(); pids = [p for p in tree_pids(table, proc.pid) if p != proc.pid] if table else []
    except Exception:
        pids = []
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for p in pids:
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError):
                pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _stage(request: ReconstructionRequest, job_dir: Path, limits: ReconstructionLimits, worker_limits: dict | None = None) -> dict:
    """Copy the extracted frames byte for byte into the job directory and write the request file.  Returns the request document (paths relative to the job directory)."""
    total = 0; frames = []
    (job_dir / "frames").mkdir(mode=0o700)
    for i, fs in enumerate(request.frames):
        row = {}
        for vk, rec in fs.items():
            dst = job_dir / "frames" / f"s{i}_{vk}.png"; shutil.copyfile(rec["path"], dst); total += dst.stat().st_size
            if total > limits.max_job_bytes:
                raise OSError("job too large")
            row[vk] = dict(file=f"frames/{dst.name}", index=rec["index"], fingerprint=rec["fingerprint"])
        frames.append(row)
    doc = dict(schema=SCHEMA_REQUEST, calibration=request.calibration, frames=frames, line_sets=[dict(ls) for ls in request.line_sets])
    if worker_limits:
        doc["limits"] = dict(worker_limits)                                       # F8.31C-1C: the granted ceiling for the worker's own watchdog (not used by any computation)
    (job_dir / "request.json").write_text(json.dumps(doc, allow_nan=False)); return doc


WATCHDOG_EXIT_CODE = 86                           # the worker's own peak-RSS watchdog stopped it (see reconstruction_worker)


def _run_worker(request: ReconstructionRequest, limits: ReconstructionLimits, *, worker_cmd=None, workspace_root=None, cancel=None, coordinator=None, retention=None):
    """-> (worker output dict | None, failure | None, process info).  Never raises; always removes the job directory, the process tree and the memory reservation.

    F8.31C-1C: admission goes through the application-wide ``resource_budget`` coordinator (shared with the calibration runner, cross-process, no queue).  BUSY (the single
    reconstruction slot is taken) and INSUFFICIENT_MEMORY (the budget cannot grant ``min_tree_rss_mb``) are different outcomes.  The ceiling enforced below is the GRANT."""
    info = dict(tree_peak_rss_mb=None, tree_cpu_s=None, cpu_limit_s=float(limits.cpu_limit_s), rss_checks=0, rss_fresh_checks=0, blind_polls=0, exit_code=None, wall_seconds=None, limit_mb=None, requested_limit_mb=limits.max_tree_rss_mb,
                timeout_s=limits.timeout_s, budget=None, container_guard=dict(active=False, limit_mb=None, peak_used_mb=None), enforcement=None)
    coord = coordinator if coordinator is not None else RB.app_coordinator()
    try:
        adm = coord.reserve("reconstruction", request.owner, limits.max_tree_rss_mb, min(limits.min_tree_rss_mb, limits.max_tree_rss_mb))
    except Exception as ex:                                                    # an unusable coordinator must not become an unbudgeted run: fail closed
        return None, failure(Code.INSUFFICIENT_MEMORY, f"the memory budget could not be consulted ({type(ex).__name__}); nothing was started", stage="admission"), info
    d = adm.detail; b = d.get("budget") or {}
    info["budget"] = dict(outcome=adm.outcome, total_mb=b.get("total_mb"), usable_mb=b.get("usable_mb"), reserve_mb=b.get("reserve_mb"), source=b.get("source"), limitation=b.get("limitation"),
                          parent_rss_mb=d.get("parent_rss_mb"), reserved_mb_before=d.get("reserved_mb"), available_mb_before=d.get("available_mb"), granted_mb=adm.granted_mb,
                          cross_process=d.get("cross_process"))
    if adm.outcome == RB.BUSY:
        return None, failure(Code.WORKER_BUSY, str(d.get("reason") or "another reconstruction holds the only slot"), stage="admission", busy_with=d.get("busy_with")), info
    if not adm.ok:
        return None, failure(Code.INSUFFICIENT_MEMORY, str(d.get("reason")), stage="admission", available_mb=d.get("available_mb"), minimum_mb=limits.min_tree_rss_mb), info
    ceiling = float(adm.granted_mb); info["limit_mb"] = ceiling
    container = (b.get("container") or {}) if b.get("container", {}).get("reliable") else {}
    cg_limit = container.get("limit_mb")
    info["container_guard"] = dict(active=bool(cg_limit), limit_mb=cg_limit, fraction=limits.container_fraction, peak_used_mb=None)
    info["enforcement"] = dict(kind="sampled process-tree RSS (adaptive poll) + worker-side peak-RSS watchdog + container anonymous-memory check; the container OOM killer is the only hard limit",
                               hard_os_limit=False, poll_s=limits.poll_s, fast_poll_s=limits.fast_poll_s, watchdog_s=limits.watchdog_s)
    ws = None; proc = None
    try:
        try:
            ws = ReconstructionWorkspace(workspace_root); job = ws.path / "job"; job.mkdir(mode=0o700)
            _stage(request, job, limits, dict(max_rss_mb=ceiling, watchdog_s=float(limits.watchdog_s), exit_code=WATCHDOG_EXIT_CODE, cpu_limit_s=float(limits.cpu_limit_s)))
        except OSError as ex:
            code = Code.JOB_TOO_LARGE if "too large" in str(ex) else Code.WORKSPACE_ERROR
            return None, failure(code, f"{type(ex).__name__}: {ex}", stage="staging"), info
        argv = list(worker_cmd(job)) if worker_cmd else [sys.executable, "-I", str(WORKER), str(job)]
        t0 = time.monotonic(); err = open(job / "worker.stderr", "wb")
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, env=_scrubbed_env(), start_new_session=True)
        except OSError as ex:
            return None, failure(Code.WORKER_CRASHED, f"the worker could not be started: {type(ex).__name__}", stage="worker"), info
        finally:
            err.close()
        verdict = None; wait_s = limits.poll_s; last_hb = time.monotonic()
        while True:
            try:
                proc.wait(timeout=wait_s); break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                verdict = failure(Code.WORKER_CANCELLED, "cancelled by the caller", stage="worker"); break
            if retention is not None and time.monotonic() - last_hb >= limits.heartbeat_s:       # keep the uploads protected; learn about a session lock / lost lease
                last_hb = time.monotonic()
                try:
                    stop = retention.heartbeat()
                except Exception:
                    stop = True
                if stop:
                    verdict = failure(Code.WORKER_CANCELLED, "the upload lease ended (session locked, cancelled or expired)", stage="worker"); break
            if time.monotonic() - t0 > limits.timeout_s:
                verdict = failure(Code.WORKER_TIMEOUT, f"the worker exceeded {limits.timeout_s:g} s and its process tree was killed", stage="worker", timeout_s=limits.timeout_s); break
            info["rss_checks"] += 1
            try:
                table = snapshot_processes(); rss = sum((table[p].get("rss_mb") or 0.0) for p in tree_pids(table, proc.pid) if p in table) if table else None
                cpu = sum((table[p].get("cpu_s") or 0.0) for p in tree_pids(table, proc.pid) if p in table) if table else None       # diagnostics only: cumulative CPU of THIS worker's process tree, from the same snapshot
            except Exception:
                rss = cpu = None
            if cpu is not None and cpu > 0:
                info["tree_cpu_s"] = max(info["tree_cpu_s"] or 0.0, cpu)
            if rss is None or rss <= 0:
                info["blind_polls"] += 1
                if info["blind_polls"] >= limits.blind_polls_max:
                    verdict = failure(Code.WORKER_MEMORY_UNMEASURABLE, "the process memory could not be measured, so the configured limit cannot be enforced", stage="worker"); break
            else:
                info["blind_polls"] = 0; info["rss_fresh_checks"] += 1; info["tree_peak_rss_mb"] = max(info["tree_peak_rss_mb"] or 0.0, rss)
                if rss > ceiling:
                    verdict = failure(Code.WORKER_MEMORY_LIMIT, f"process-tree RSS {rss:.0f} MB exceeded the granted {ceiling:.0f} MB and the tree was killed", stage="worker", limit_mb=ceiling); break
                wait_s = limits.fast_poll_s if rss > limits.adaptive_poll_fraction * ceiling else limits.poll_s       # poll faster when close to the ceiling
            if cg_limit:
                try:
                    used = RB.container_used_mb(read_cgroup())
                except Exception:
                    used = None
                if used is not None:
                    cgi = info["container_guard"]; cgi["peak_used_mb"] = max(cgi["peak_used_mb"] or 0.0, used)
                    if used > limits.container_fraction * cg_limit:
                        verdict = failure(Code.WORKER_MEMORY_LIMIT, f"container memory {used:.0f} MB exceeded {limits.container_fraction:.0%} of its {cg_limit:.0f} MB limit and the worker tree was killed",
                                          stage="worker", limit_mb=cg_limit); break
        info["wall_seconds"] = round(time.monotonic() - t0, 3)
        if verdict is not None:
            _kill_tree(proc, limits.kill_grace_s); info["exit_code"] = proc.returncode
            return None, verdict, info
        info["exit_code"] = proc.returncode
        if getattr(signal, "SIGXCPU", None) is not None and proc.returncode == -int(signal.SIGXCPU):      # the operating system stopped it at its CPU-time limit (no such signal: never matches)
            return None, failure(Code.WORKER_CRASHED, f"the worker used more than its {limits.cpu_limit_s:g} CPU-second limit and was stopped by the operating system (SIGXCPU)", stage="worker",
                                 exit_code=proc.returncode, cpu_limit_s=float(limits.cpu_limit_s)), info
        if proc.returncode == WATCHDOG_EXIT_CODE:                              # the worker stopped itself: its own peak RSS passed the granted ceiling
            return None, failure(Code.WORKER_MEMORY_LIMIT, f"the worker's own peak-RSS watchdog stopped it above the granted {ceiling:.0f} MB", stage="worker", limit_mb=ceiling), info
        res_path = job / "result.json"
        if proc.returncode != 0 or not res_path.is_file() or res_path.is_symlink():
            return None, failure(Code.WORKER_CRASHED, f"the worker exited with status {proc.returncode} and no result", stage="worker", exit_code=proc.returncode), info
        if res_path.stat().st_size > limits.max_result_bytes:
            return None, failure(Code.WORKER_OUTPUT_INVALID, "the worker result exceeds the size limit", stage="worker"), info
        try:
            out = json.loads(res_path.read_text())
        except ValueError:
            return None, failure(Code.WORKER_OUTPUT_INVALID, "the worker result is not valid JSON", stage="worker"), info
        peak = out.get("peak_rss_mb") if isinstance(out, dict) else None          # the kernel's high-water mark: cannot miss a spike between polls (checked after the fact)
        if isinstance(peak, (int, float)) and not isinstance(peak, bool) and math.isfinite(peak):
            info["worker_self_peak_rss_mb"] = float(peak)
            if peak > ceiling:
                return None, failure(Code.WORKER_MEMORY_LIMIT, f"the worker's own peak RSS {peak:.0f} MB exceeded the granted {ceiling:.0f} MB (found after the run; the result was discarded)", stage="worker", limit_mb=ceiling), info
        return out, None, info
    finally:
        try:
            if proc is not None:
                _kill_tree(proc, limits.kill_grace_s)                              # idempotent: a finished worker is only reaped
            if ws is not None:
                ws.close()
        finally:
            adm.reservation.release()                                              # always: a crash, timeout or exception never leaves a reservation


# ------------------------------------------------------------------ the result contract
def _validate_worker_output(out) -> str | None:
    """Return a problem description, or None when the worker's document has the shape the assembler relies on."""
    try:
        if not isinstance(out, dict) or out.get("schema") != "f8.reconstruction_worker/1":
            return "unexpected worker document"
        ev = out.get("adapter")
        if ev is None:                                                         # the worker stopped before Method J: it must say why
            return None if isinstance(out.get("failures"), list) and out["failures"] else "no adapter result and no failure"
        if not isinstance(ev, dict) or ev.get("outcome") not in (SUCCESS, QUESTIONABLE, FAILED) or not isinstance(ev.get("failures"), list) or not isinstance(ev.get("warnings"), list):
            return "malformed adapter result"
        if ev["outcome"] != FAILED:
            for k in ("width_mm", "height_mm"):
                v = ev.get(k)
                if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
                    return f"{k} is not a positive finite number"
        return None
    except Exception as ex:
        return f"{type(ex).__name__}"


def _provenance(request: ReconstructionRequest | None, calibration_override=None) -> dict:
    if request is None:
        return dict(ground_truth_used=False, recipe_a_used=False, historical_baseline_used=False)
    c = request.calibration
    return dict(
        calibration=dict(calibration_id=c["calibration_id"], image_size_wh=list(c["image_size_wh"]), baseline_mm=c["baseline_mm"], baseline_unit="mm", baseline_source="visitor measured, preserved as supplied",
                         camera_order=list(c["camera_order"]), distortion_model=c["distortion_model"], capture_mode=dict(c["capture_mode"]),
                         calibration_digest=context_from_params(c).digest()),
        videos={s: dict(v) for s, v in sorted(request.videos.items())}, frame_indices=[dict(t) for t in request.frame_table], n_frame_sets=request.n_sets, source_frame_sets=list(request.source_sets), skipped_frame_sets=list(request.skipped_sets),
        frame_fingerprints=[{vk: f["fingerprint"] for vk, f in fs.items()} for fs in request.frames], annotations=[dict(a) for a in request.annotations],
        coordinate_space=RF.COORDINATE_SPACE, owner_digest=request.owner, method="Method J (original) via visitor_method_j_adapter", top_variant=0,
        required_lines=list(REQUIRED_LINES), ground_truth_used=False, recipe_a_used=False, historical_baseline_used=False)


def _per_frame_set(request: ReconstructionRequest | None, out: dict | None, ev: dict | None, ok_result: bool) -> list:
    n = request.n_sets if request is not None else 0; rows = []
    wsets = {s.get("frame_set"): s for s in (out or {}).get("sets", []) if isinstance(s, dict)}
    good = {p.get("frame_set"): p for p in ((ev or {}).get("diagnostics", {}).get("per_set") or [])} if ok_result else {}
    bad = {f.get("frame_set"): f for f in ((ev or {}).get("diagnostics", {}).get("set_failures") or [])}
    for i in range(n):
        w = wsets.get(i, {}); row = dict(frame_set=i, source_frame_set=(request.source_sets[i] if i < len(request.source_sets) else i), frame_indices=dict(request.frame_table[i]), features=dict(w.get("features", {})))
        if i in good:
            g = good[i]
            row.update(status="reconstructed", width_mm=g.get("width_mm"), height_mm=g.get("height_mm"), registration=g.get("registration"), vertical_error_deg=g.get("vertical_error_deg"), skew_mm=g.get("skew_mm"))
        elif i in bad:
            row.update(status="failed", failure_code=bad[i].get("code"), failure_class=failure_class_of(bad[i].get("code")), width_mm=None, height_mm=None)
        else:
            row.update(status="not_reconstructed", width_mm=None, height_mm=None)
        rows.append(row)
    return rows


def _visitor_notice(status: str, failures: list) -> str:
    """One sentence for the visitor.  It follows the decision (status / failures); it never changes it, and a FAILED result never reads like a measurement."""
    if status == FAILED:
        return (failures[0].get("visitor_message") if failures else VM.failure_text("INTERNAL_ERROR")) or VM.failure_text("INTERNAL_ERROR")
    if status == QUESTIONABLE:
        return f"{VM.QUESTIONABLE_SUMMARY} {VM.NO_CLAIMS}"
    return VM.NO_CLAIMS


def assemble_result(request: ReconstructionRequest | None, *, worker_out=None, failures=None, process_info=None, seconds: float = 0.0) -> dict:
    """The backend result contract (JSON-safe).  ``failures`` are failures of THIS layer (input checks, worker); ``worker_out`` is the worker's validated document.
    A FAILED result never carries a dimension, a per-set dimension or any other number that could be read as a measurement."""
    failures = [dict(f) for f in (failures or [])]; warnings, notes = [], []
    ev = (worker_out or {}).get("adapter") if isinstance(worker_out, dict) else None
    diag = {}; width = height = None; status = FAILED
    if worker_out is not None and not failures:
        failures += [failure(f["code"], f.get("message", ""), stage=f.get("stage", "worker"), **{k: v for k, v in f.items() if k not in ("code", "message", "stage", "failure_class", "visitor_message")})
                     for f in (worker_out.get("failures") or [])]
    if ev is not None and not failures:
        failures += [failure(f["code"], f.get("message", ""), stage=f.get("stage", "method_j"), **{k: v for k, v in f.items() if k not in ("code", "message", "stage")}) for f in ev.get("failures", [])]
        warnings = [dict(w, advisory=True, visitor_explanation=VM.warning_parts(w.get("code"))) for w in ev.get("warnings", [])]; notes = list(ev.get("notes", [])) + [dict(n) for n in ev.get("diagnostic_notes", [])]
        diag = {k: v for k, v in (ev.get("diagnostics") or {}).items()}
        if ev["outcome"] != FAILED and not failures:
            status, width, height = ev["outcome"], ev.get("width_mm"), ev.get("height_mm")
    ok_result = status != FAILED and not failures
    if not ok_result:
        status, width, height = FAILED, None, None
        diag = {k: v for k, v in diag.items() if k not in ("per_set", "geometry_consistency_informational")}      # no intermediate dimension is exposed as a result
    classes = sorted({f["failure_class"] for f in failures})
    res = dict(
        schema=SCHEMA_RESULT, status=status, width_mm=width, height_mm=height, units="mm", partial=bool(ev.get("partial")) if ev and ok_result else False,
        dimensions_accepted=ok_result, per_frame_set=_per_frame_set(request, worker_out, ev, ok_result), failures=failures, failure_classes=classes,
        failure_class=classes[0] if len(classes) == 1 else ("mixed" if classes else None), warnings=warnings, notes=notes,
        diagnostics=dict(pose=diag.get("pose"), registration=diag.get("registration"), set_failures=diag.get("set_failures", []), geometry_consistency_informational=diag.get("geometry_consistency_informational"),
                         original_pipeline_decision=diag.get("original_pipeline_decision"), n_sets=diag.get("n_sets"), n_sets_ok=diag.get("n_sets_ok"),
                         calibration_use=dict(lines_retargeted=bool((ev or {}).get("calibration", {}).get("lines_retargeted")), digest=(ev or {}).get("calibration", {}).get("digest"))),
        processing=dict(seconds=round(float(seconds), 3), worker=dict(**(process_info or {})), stage_seconds=(worker_out or {}).get("timings") if isinstance(worker_out, dict) else None,
                        worker_self_peak_rss_mb=(worker_out or {}).get("peak_rss_mb") if isinstance(worker_out, dict) else None),
        provenance=_provenance(request),
        uncertainty=dict(status="not_computed", note="no confidence interval or calibrated uncertainty is claimed; no percentage error is computed"),
        ground_truth_used=False, note=(ev or {}).get("note") if ok_result else "no valid reconstruction; no dimensions are reported",
        visitor_notice=_visitor_notice(status, failures))
    return json.loads(json.dumps(json_safe(res), allow_nan=False))


# ------------------------------------------------------------------ resolution policy (reporting only)
def resolution_policy(limits: ReconstructionLimits | None = None) -> dict:
    """F8.31C-1C: what is validated, what is only exercised and what is merely allowed.  Describes behaviour; decides nothing.  Nothing is ever resized: the calibration, every video and
    every frame must have the same size (``CALIBRATION_VIDEO_SIZE_MISMATCH`` / ``FRAME_SIZE_MISMATCH``), and the only size rule is the explicit resource bound of ``JobLimits``."""
    jl = (limits or ReconstructionLimits()).job_limits
    return dict(initial_deployment_target_wh=[2160, 3840], target_is_a_guarantee=False, validated_on_real_recordings=[list(w) for w in RV.VALIDATED_FRAME_SIZES],
                exercised_with_synthetic_renders_only=[[1080, 1920]], other_sizes="allowed but unvalidated (a warning, never a rejection)",
                rejected_only_when=f"pixels per frame > {jl.max_frame_pixels} or a side > {jl.video.max_side_px} (resource bound), or the size differs from the calibration",
                resized_cropped_or_resampled=False, intrinsics_rescaled=False, pixel_unit_thresholds_rescaled=False, floor_row_cut_rule="2600 * H / 3840 (adapter, unchanged, validated at 3840 rows only)")


# ------------------------------------------------------------------ upload retention while a job runs
class StoreRetention:
    """F8.31C-1C: a BOUNDED job lease on the visitor's room-video ``ChunkStore`` uploads, held only while one reconstruction is using them.

    ``acquire`` re-checks ownership, that each required upload is still the one the frames were chosen from (``upload_id`` AND SHA-256, so an expired, deleted or replaced
    upload can never become valid again) and still intact, then protects those uploads from the sweeper, from deletion and from replacement.  ``heartbeat`` renews it and
    reports whether the job must stop (the session was locked, the lease expired).  ``release`` always ends the protection and restarts the ordinary retention clock: nothing is
    kept longer than before, abandoned uploads are still cleaned up, and the lease has a hard deadline of its own (``max_s``), so a crash can only protect them that long."""

    def __init__(self, store, owner: str, slots=None):
        self.store, self.owner = store, owner
        self.slots = tuple(slots or RV.ROOM_UPLOAD_SLOTS)
        self.lease_id = None

    def acquire(self, sources: dict, max_s: float):
        """-> None on success, else a typed failure record (nothing is held)."""
        expect = {}
        for slot in self.slots:
            cur = sources.get(slot) if isinstance(sources, dict) else None
            if cur is None:
                return failure(Code.UPLOAD_EXPIRED, f"the upload of slot {slot} is missing or expired", stage="retention", slot=slot)
            expect[slot] = (cur.upload_id, cur.sha256)
        try:
            self.lease_id = self.store.acquire_job_lease(self.owner, expect, max_s)["lease_id"]
        except Exception as ex:
            code = getattr(ex, "code", "")
            if code == "replaced":
                return failure(Code.VIDEO_REPLACED, "a room video is not the upload the frames were chosen from", stage="retention")
            if code in ("job_running", "busy"):
                return failure(Code.WORKER_BUSY, "these uploads are already in use by another job", stage="retention")
            return failure(Code.UPLOAD_EXPIRED, f"the uploads could not be protected ({code or type(ex).__name__})", stage="retention")
        return None

    def heartbeat(self) -> bool:
        """True when the job must stop."""
        if self.lease_id is None:
            return True
        try:
            return bool(self.store.heartbeat(self.owner, self.lease_id).get("cancel_requested"))
        except Exception:
            return True                                                       # lease expired / session released: stop

    def release(self):
        lid, self.lease_id = self.lease_id, None
        if lid is not None:
            try:
                self.store.release_job_lease(self.owner, lid)
            except Exception:
                pass


# ------------------------------------------------------------------ the entry points
def run_request(request: ReconstructionRequest, limits: ReconstructionLimits | None = None, *, worker_cmd=None, workspace_root=None, cancel=None, coordinator=None,
                retention=None) -> dict:
    """Run one prepared request in the sandbox and return the result contract.  Never raises."""
    limits = limits or ReconstructionLimits(); t0 = time.monotonic()
    try:
        out, fail, info = _run_worker(request, limits, worker_cmd=worker_cmd, workspace_root=workspace_root, cancel=cancel, coordinator=coordinator, retention=retention)
        if fail is not None:
            return assemble_result(request, failures=[fail], process_info=info, seconds=time.monotonic() - t0)
        problem = _validate_worker_output(out)
        if problem:
            return assemble_result(request, failures=[failure(Code.WORKER_OUTPUT_INVALID, problem, stage="worker")], process_info=info, seconds=time.monotonic() - t0)
        return assemble_result(request, worker_out=out, process_info=info, seconds=time.monotonic() - t0)
    except Exception as ex:                                                    # nothing may escape as an untyped exception
        return assemble_result(request, failures=[failure(A.FailureReason.INTERNAL_ERROR, f"{type(ex).__name__}: {ex}", stage="internal")], seconds=time.monotonic() - t0)


def run_visitor_reconstruction(wf: RF.RoomFrameWorkflow, owner: str, contract, cal, sources: dict, limits: ReconstructionLimits | None = None, *, worker_cmd=None,
                               workspace_root=None, cancel=None, coordinator=None, retention=None) -> dict:
    """Check every input, then run Method J once on this visitor's data.  Always returns a result dict (SUCCESS, QUESTIONABLE or FAILED); never raises.
    ``retention`` (a ``StoreRetention``) protects the uploads for the whole call and is always released.  Not called by any page or service (F8.31C-1C does not enable it)."""
    limits = limits or ReconstructionLimits(); t0 = time.monotonic()
    try:
        if retention is not None:
            f = retention.acquire(sources, max_s=limits.timeout_s + 180.0)       # worker time + staging / hashing allowance; a hard bound, never a keep-alive
            if f is not None:
                return assemble_result(None, failures=[f], seconds=time.monotonic() - t0)
        try:
            prep = prepare(wf, owner, contract, cal, sources, limits)
        except Exception as ex:
            return assemble_result(None, failures=[failure(A.FailureReason.INTERNAL_ERROR, f"{type(ex).__name__}: {ex}", stage="prepare")], seconds=time.monotonic() - t0)
        if prep.request is None:
            return assemble_result(None, failures=prep.failures, seconds=time.monotonic() - t0)
        return run_request(prep.request, limits, worker_cmd=worker_cmd, workspace_root=workspace_root, cancel=cancel, coordinator=coordinator, retention=retention)
    finally:
        if retention is not None:
            retention.release()
