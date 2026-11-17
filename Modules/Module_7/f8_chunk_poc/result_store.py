"""Stage F8.10: durable-enough storage for calibration RESULTS, independent of Streamlit ``session_state`` (NO Streamlit import; standard library only).

What this is, and is not
------------------------
* A finished (or failed) job writes ONE structured JSON document, atomically (temp file in the same directory -> flush -> fsync -> ``os.replace`` -> directory
  fsync), BEFORE the temporary videos are deleted. A reader therefore sees either the whole document or none of it.
* It survives a browser disconnect, a page refresh, a new Streamlit session and Streamlit reruns, because it lives in a server-side directory and is found
  by a long random RECOVERY CODE (128 bit, shown to the person who started the job and kept in the page URL as ``?job=<code>``). Only a hash of the code
  names the directory, so the code itself is never written to disk.
* It is NOT permanent. On Streamlit Community Cloud the local disk is ephemeral: an app restart, redeploy, reboot or resource-limit restart erases it.
  Results are also deleted after ``ttl_s`` (default 24 h) and the folder is bounded. The DOWNLOAD button is the only durable copy: download it.
* Original video content is never stored: only metadata, SHA-256 digests, frame indices and calibration numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path

MiB = 1024 * 1024
SCHEMA = "f8.calibration_result/1"
_CODE_RE = re.compile(r"[A-Za-z0-9_-]{16,64}")
RESULT_NAME, STATUS_NAME = "result.json", "status.json"


class ResultError(Exception):
    pass


def new_code() -> str:
    """A fresh unguessable recovery code (about 22 URL-safe characters, 128 bit)."""
    return secrets.token_urlsafe(16)


def valid_code(code) -> bool:
    return isinstance(code, str) and _CODE_RE.fullmatch(code) is not None


def job_id(code: str) -> str:
    """Public, non-secret identifier derived from a code (safe to put in a file or a log)."""
    return hashlib.sha256(code.encode()).hexdigest()[:12]


def _clean(o, depth=0):
    """JSON-safe copy: NaN/inf -> None, unknown types -> short str, bounded depth. Never raises."""
    if depth > 12:
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v, depth + 1) for v in o]
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)[:200]


def write_atomic(path, data: bytes, mode: int = 0o600) -> None:
    """All-or-nothing file write: a crash leaves the old file (or none) and never a half-written one."""
    path = Path(path); tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:                                                          # make the rename itself durable (not available everywhere)
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------ the document
def build_result_document(res: dict, *, code: str, settings: dict | None = None, now: float | None = None) -> dict:
    """The structured, bounded result written for every outcome. ``res`` is the supervisor's result dict. Detections (per-frame corner dumps) and
    server paths are NOT copied; selected poses are stored as frame indices."""
    res = res or {}; cal = res.get("calibration") or {}; model = res.get("model") or {}; ass = res.get("assessment") or {}
    metrics = ass.get("metrics") or {}; vids = ((res.get("validation") or {}).get("videos")) or {}
    rs = res.get("resources") or {}; settings = settings or {}
    side = lambda k: dict(K=model.get("K" + k), distortion=model.get("D" + k), image_size_wh=model.get("image_size_left" if k == "1" else "image_size_right"),
                          rms_px=model.get("rms_left" if k == "1" else "rms_right"), distortion_assessment=model.get("distortion_left" if k == "1" else "distortion_right"))
    doc = dict(
        schema=SCHEMA, job_id=job_id(code), saved_at_unix=round(now if now is not None else time.time(), 3), mode=res.get("mode"), outcome=res.get("outcome"),
        ok=bool(res.get("ok")), error=res.get("error"),
        settings=dict(cols=settings.get("cols"), rows=settings.get("rows"), square_mm=settings.get("square_mm"), measured_baseline_mm=settings.get("baseline_mm")),
        quality=dict(level=ass.get("level"), headline=ass.get("headline"), problems=ass.get("problems"), concerns=ass.get("concerns"), advice=ass.get("advice"),
                     stereo_rms_px=model.get("stereo_rms", metrics.get("rms_stereo_px")), epipolar_median_px=metrics.get("epipolar_median_px"), metrics=metrics or None),
        cameras=dict(left=side("1"), right=side("2")),
        extrinsics=dict(R=model.get("R"), T=model.get("T"), baseline_mm=model.get("baseline_mm"), measured_baseline_mm=metrics.get("measured_baseline_mm")),
        selected_poses=dict(count=cal.get("pairs_used"), discarded=cal.get("pairs_discarded"), frame_pairs_left_right=cal.get("pair_frames"), ordering=cal.get("ordering")),
        synchronization=dict(sync=cal.get("sync"), pairing=cal.get("pairing"), fps=cal.get("fps")),
        detections_found=cal.get("detections"), contract=res.get("contract"),
        videos={s: {k: (vids.get(s) or {}).get(k) for k in ("container", "codec", "width", "height", "fps", "frame_count", "duration_s")} for s in ("left", "right")},
        runtime=dict(total_wall_s=res.get("total_wall_s"), worker_cpu_s=res.get("worker_cpu_s"), worker_peak_rss_mb=res.get("peak_rss_mb"), stage_timings=res.get("stage_timings"),
                     supervisor_total_s=rs.get("total_wall_s"), timeout=res.get("timeout")),
        measurements=dict(peak=rs.get("peak"), cgroup=rs.get("cgroup"), cpu=rs.get("cpu"), available=rs.get("available"), stages=rs.get("stages"),
                          memory_guard=res.get("memory_guard")),
        benchmark=res.get("benchmark_report"), environment=res.get("environment"), worker_env=res.get("env"), f7=res.get("f7"),
        cleanup=res.get("cleanup"),
        note="Contains calibration numbers, frame indices, timings and SHA-256 digests only; no video content and no file paths. Local Cloud storage is ephemeral: keep this download.")
    return _clean(doc)


class ResultStore:
    """Directory of results and job-status files, keyed by recovery-code hash. Thread-safe; bounded by age, count and size."""

    def __init__(self, root=None, *, ttl_s: float = 86400.0, max_results: int = 20, max_bytes: int = 8 * MiB, clock=time.time):
        if root is None:
            root = os.environ.get("F8_RESULTS_DIR") or Path(tempfile.gettempdir()) / f"f8_results_{os.getuid() if hasattr(os, 'getuid') else 0}"
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self.ttl_s, self.max_results, self.max_bytes, self._clock = ttl_s, max_results, max_bytes, clock
        self._lock = threading.Lock()

    def _dir(self, code) -> Path:
        if not valid_code(code):
            raise ResultError("That recovery code is not valid.")
        return self.root / ("job_" + hashlib.sha256(code.encode()).hexdigest()[:32])

    def _write(self, code, name, obj, limit) -> dict:
        data = json.dumps(_clean(obj), indent=1, sort_keys=False, allow_nan=False).encode()
        if len(data) > limit:
            raise ResultError(f"The document is larger than {limit} bytes.")
        d = self._dir(code)
        with self._lock:
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_atomic(d / name, data)
        return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def save_result(self, code: str, doc: dict) -> dict:
        """Atomically write the result document. Returns dict(bytes, sha256). Raises ResultError (nothing partial is left) on failure."""
        try:
            return self._write(code, RESULT_NAME, doc, self.max_bytes)
        except OSError as e:
            raise ResultError(f"The result could not be written ({type(e).__name__}).") from e

    def result_bytes(self, code: str) -> bytes | None:
        try:
            return (self._dir(code) / RESULT_NAME).read_bytes()
        except (OSError, ResultError):
            return None

    def load_result(self, code: str) -> dict | None:
        b = self.result_bytes(code)
        try:
            return json.loads(b) if b else None
        except ValueError:
            return None

    def write_status(self, code: str, status: dict) -> None:
        """Small progress record read by other sessions. Best effort: a failure never disturbs the job."""
        try:
            self._write(code, STATUS_NAME, dict(status, updated_unix=round(self._clock(), 3)), 16 * 1024)
        except (OSError, ResultError):
            pass

    def read_status(self, code: str) -> dict | None:
        try:
            st = json.loads((self._dir(code) / STATUS_NAME).read_text())
        except (OSError, ValueError, ResultError):
            return None
        if isinstance(st, dict) and isinstance(st.get("updated_unix"), (int, float)):
            st["age_s"] = round(max(self._clock() - st["updated_unix"], 0.0), 1)
        return st if isinstance(st, dict) else None

    def sweep(self) -> int:
        """Delete result folders older than ``ttl_s``, then the oldest beyond ``max_results``. Only ``job_*`` folders are touched."""
        n = 0
        with self._lock:
            try:
                dirs = [d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("job_")]
            except OSError:
                return 0
            aged = []
            for d in dirs:
                try:
                    aged.append((max((p.stat().st_mtime for p in d.iterdir()), default=d.stat().st_mtime), d))
                except OSError:
                    continue
            aged.sort(key=lambda x: x[0]); now = self._clock()
            keep = []
            for t, d in aged:
                if now - t > self.ttl_s:
                    shutil.rmtree(d, ignore_errors=True); n += 1
                else:
                    keep.append(d)
            for d in keep[:max(len(keep) - self.max_results, 0)]:
                shutil.rmtree(d, ignore_errors=True); n += 1
        return n
