"""Stage F8.31B: SANDBOXED, bounded frame inspection / extraction for visitor room videos (NO Streamlit import).

The same isolation model as ``f8_chunk_poc/video_check.py``: OpenCV runs in a SEPARATE PROCESS (isolated interpreter ``-I``, scrubbed environment, own process
group) with a wall-clock timeout, a CPU-time cap and (Linux) an address-space cap; on timeout the whole process group is killed.  The Streamlit process never
decodes a visitor video.  A video is decoded as a STREAM (``grab`` / ``retrieve``): a complete video is never loaded into memory, and at most one frame is
held at a time (plus the small per-frame sharpness numbers).

Two operations (both read-only on the video; both write only inside a private, per-job output directory that the caller owns):

  ``scan``     one sequential pass.  Optionally the Laplacian-variance sharpness of EVERY frame (used for the original experiment's frame-selection rule) and
               reduced-size PREVIEW images (display only) of the requested indices.
  ``extract``  one sequential pass.  The requested indices as FULL-RESOLUTION grayscale PNG files (lossless; same convention as Phase 1: 0-based sequential
               decoding, BGR2GRAY, no seeking, no crop / resize / rotation / resampling) and, optionally, previews.

Limits come from ONE object, ``FrameJobLimits.from_job_limits(JobLimits(video=room_video_limits()))``: the file-size ceiling is the room limit (never F6's
default 200 MB) and the per-frame pixel ceiling is the F6 job's ``max_frame_pixels``.  They are enforced twice: by the parent BEFORE the worker starts (file
size) and by the worker itself (decoded size, frame count).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

MiB = 1024 * 1024
MAX_PREVIEW_LONG_SIDE = 1600


@dataclass(frozen=True)
class FrameJobLimits:
    max_file_bytes: int                      # the ROOM limit (``room_video_limits().max_file_bytes``), not F6's 200 MB default
    max_side_px: int = 16_384                # F6 Stage B (``VideoLimits.max_side_px``)
    max_frame_pixels: int = 40_000_000       # F6 job (``JobLimits.max_frame_pixels``)
    max_scan_frames: int = 14_400            # sequential frames one job may decode (120 s at 120 fps)
    max_indices: int = 24                    # frames one job may write
    timeout_s: float = 150.0                 # wall clock for the whole worker
    worker_extra_memory_mb: int = 3072       # Linux: address space allowed on top of what the worker maps after importing OpenCV
    max_output_bytes: int = 600 * MiB        # bytes one job may leave on disk (checked after the job; the job directory is deleted on excess)
    preview_long_side: int = 900

    @classmethod
    def from_job_limits(cls, job_limits, **overrides) -> "FrameJobLimits":
        """``job_limits``: a ``visitor_measurement.JobLimits`` (its ``video`` is a ``VideoLimits``)."""
        v = job_limits.video
        base = dict(max_file_bytes=int(v.max_file_bytes), max_side_px=int(v.max_side_px), max_frame_pixels=int(job_limits.max_frame_pixels))
        base.update(overrides)
        return cls(**base)


class FrameJobError(RuntimeError):
    """A visitor-safe failure (never a path or a traceback)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


# ------------------------------------------------------------------ private workspace (one per visitor; removed as a whole)
class FrameWorkspace:
    """A private (0o700) directory for ONE visitor's previews and full-resolution frames.  Every job writes into its own new subdirectory, so no two jobs,
    and no two visitors, ever share an output path.  ``close`` removes everything; it is safe to call twice."""

    def __init__(self, root: Path | str | None = None, prefix: str = "f8_frames_"):
        self.path = Path(tempfile.mkdtemp(prefix=prefix, dir=None if root is None else str(root)))
        os.chmod(self.path, 0o700)
        self.closed = False

    def new_job_dir(self) -> Path:
        if self.closed:
            raise FrameJobError("workspace_closed", "This session's frame workspace has been closed.")
        d = Path(tempfile.mkdtemp(prefix="job_", dir=str(self.path)))
        os.chmod(d, 0o700)
        return d

    def drop(self, path: Path | str):
        """Delete one job directory (only if it lies inside this workspace)."""
        p = Path(path).resolve()
        if self.path.resolve() in p.parents:
            shutil.rmtree(p, ignore_errors=True)

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file()) if self.path.exists() else 0

    def close(self):
        if not self.closed:
            self.closed = True
            shutil.rmtree(self.path, ignore_errors=True)


# ------------------------------------------------------------------ the worker (runs in the subprocess)
def _worker_main(request_json: str) -> int:
    """Prints exactly ONE JSON line.  Reads the video, writes only into ``req['outdir']``."""
    req = json.loads(request_json)
    lim = FrameJobLimits(**req["limits"])
    out = dict(opened=False)
    try:
        import resource
        cpu = int(lim.timeout_s) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu)); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        resource = None
    import cv2
    import numpy as np
    if resource is not None and sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status") as fh:
                vm = next(int(line.split()[1]) for line in fh if line.startswith("VmSize:")) * 1024
            cap_b = vm + lim.worker_extra_memory_mb * MiB; resource.setrlimit(resource.RLIMIT_AS, (cap_b, cap_b))
        except Exception:
            pass
    outdir = Path(req["outdir"])
    want_full = sorted({int(i) for i in req.get("full_indices", [])})
    want_prev = sorted({int(i) for i in req.get("preview_indices", [])})
    all_sharp = bool(req.get("sharpness_all"))
    if len(set(want_full) | set(want_prev)) > lim.max_indices:
        out["error"] = "too_many_indices"; print(json.dumps(out)); return 0
    last_needed = max(want_full + want_prev + [-1])
    cap = cv2.VideoCapture(req["path"])
    try:
        if not cap.isOpened():
            print(json.dumps(out)); return 0
        out["opened"] = True
        pw, ph = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if pw > lim.max_side_px or ph > lim.max_side_px or pw * ph > lim.max_frame_pixels:
            out["error"] = "dimensions"; print(json.dumps(out)); return 0
        sharp, written, wh = [], {}, None
        pos = 0
        stop_at = lim.max_scan_frames - 1 if all_sharp else min(last_needed, lim.max_scan_frames - 1)
        while pos <= stop_at:
            if not cap.grab():
                break
            need_img = all_sharp or pos in want_full or pos in want_prev
            if need_img:
                ok, f = cap.retrieve()
                if not ok or f is None or not f.size:
                    sharp.append(None) if all_sharp else None
                    pos += 1; continue
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                if wh is None:
                    wh = [int(g.shape[1]), int(g.shape[0])]
                    if wh[0] > lim.max_side_px or wh[1] > lim.max_side_px or wh[0] * wh[1] > lim.max_frame_pixels:
                        out["error"] = "decoded_dimensions"; out["decoded_wh"] = wh; print(json.dumps(out)); return 0
                if all_sharp:
                    sharp.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
                if pos in want_full:
                    p = outdir / f"full_{pos}.png"
                    if not cv2.imwrite(str(p), g, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        out["error"] = "write_failed"; print(json.dumps(out)); return 0
                    written[f"full_{pos}"] = dict(path=str(p), wh=[int(g.shape[1]), int(g.shape[0])])
                if pos in want_prev:
                    s = min(1.0, min(lim.preview_long_side, MAX_PREVIEW_LONG_SIDE) / max(g.shape))
                    pv = g if s >= 1 else cv2.resize(g, (max(1, round(g.shape[1] * s)), max(1, round(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
                    p = outdir / f"prev_{pos}.png"
                    if not cv2.imwrite(str(p), pv, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        out["error"] = "write_failed"; print(json.dumps(out)); return 0
                    written[f"prev_{pos}"] = dict(path=str(p), wh=[int(pv.shape[1]), int(pv.shape[0])], full_wh=[int(g.shape[1]), int(g.shape[0])])
                del g, f
            pos += 1
        out.update(decoded_frames=pos, decoded_wh=wh, written=written)
        if all_sharp:
            out["sharpness"] = sharp
        print(json.dumps(out)); return 0
    finally:
        cap.release()


# ------------------------------------------------------------------ the parent API
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")


def _scrubbed_env():
    return {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}


def _spawn(request: dict, limits: FrameJobLimits, worker_cmd=None):
    cmd = list(worker_cmd) if worker_cmd else [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", json.dumps(request)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_scrubbed_env(), start_new_session=True)
    try:
        stdout, _ = proc.communicate(timeout=limits.timeout_s)
        return proc.returncode, stdout, time.perf_counter() - t0, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return None, b"", time.perf_counter() - t0, True


def _budget():
    """The shared resource budget (``f8_chunk_poc/resource_budget``), imported only in the PARENT process: the decode worker itself must stay light."""
    mod = Path(__file__).resolve().parent.parent / "f8_chunk_poc"
    if str(mod) not in sys.path:
        sys.path.insert(0, str(mod))
    import resource_budget
    return resource_budget


def run_frame_job(workspace: FrameWorkspace, video_path, limits: FrameJobLimits, *, full_indices=(), preview_indices=(), sharpness_all=False, worker_cmd=None,
                  coordinator=None, admit: bool = True, wait_s: float = 3.0) -> dict:
    """Phase D.3: admission, then the unchanged job.  A decode worker is memory-hungry (previously unbudgeted) and must not run next to a calibration or reconstruction, nor in unlimited
    numbers: it takes a ``frames`` slot of the container-wide budget (``resource_budget``; shared by every process of the app, released on success, failure, timeout or cancel, and by the
    kernel if the process dies).  Refusal raises ``FrameJobError('busy')`` with a visitor-safe message; nothing is started and nothing is left behind.  ``wait_s`` lets a SHORT job ahead of
    this one finish, no longer.  ``admit=False`` is for tests only; ``coordinator`` defaults to the application-wide one."""
    if not admit:
        return _run_frame_job(workspace, video_path, limits, full_indices=full_indices, preview_indices=preview_indices, sharpness_all=sharpness_all, worker_cmd=worker_cmd)
    RB = _budget()
    try:
        reservation = RB.acquire_light("frames", "frame-job", coordinator, wait_s=wait_s)
    except RB.ResourceBusy as busy:
        raise FrameJobError("busy", busy.message) from None
    try:
        return _run_frame_job(workspace, video_path, limits, full_indices=full_indices, preview_indices=preview_indices, sharpness_all=sharpness_all, worker_cmd=worker_cmd)
    finally:
        reservation.release()


def _run_frame_job(workspace: FrameWorkspace, video_path, limits: FrameJobLimits, *, full_indices=(), preview_indices=(), sharpness_all=False, worker_cmd=None) -> dict:
    """Run one bounded job and return ``dict(job_dir, written, decoded_frames, decoded_wh, sharpness, seconds)``.  Raises ``FrameJobError`` (visitor-safe message)
    and leaves no output behind when anything fails.  The video must be an existing regular file (not a symlink) no larger than ``limits.max_file_bytes``."""
    p = Path(video_path)
    try:
        if not p.is_file() or p.is_symlink():
            raise FrameJobError("video_missing", "The video file is no longer available. It may have expired: upload it again.")
        size = p.stat().st_size
    except OSError:
        raise FrameJobError("video_missing", "The video file is no longer available. It may have expired: upload it again.") from None
    if size > limits.max_file_bytes:
        raise FrameJobError("file_too_large", f"The file is {size / MiB:.0f} MB; the limit for a room video is {limits.max_file_bytes / MiB:.0f} MB.")
    full_indices, preview_indices = sorted({int(i) for i in full_indices}), sorted({int(i) for i in preview_indices})
    if any(i < 0 for i in full_indices + preview_indices):
        raise FrameJobError("bad_index", "Frame numbers start at 0.")
    if len(set(full_indices) | set(preview_indices)) > limits.max_indices:
        raise FrameJobError("too_many_frames", f"At most {limits.max_indices} frames can be read in one step.")
    if not (full_indices or preview_indices or sharpness_all):
        raise FrameJobError("nothing_to_do", "No frame was requested.")
    job_dir = workspace.new_job_dir()
    req = dict(path=str(p), outdir=str(job_dir), full_indices=full_indices, preview_indices=preview_indices, sharpness_all=bool(sharpness_all), limits=asdict(limits))
    try:
        rc, stdout, dt, timed_out = _spawn(req, limits, worker_cmd)
        if timed_out:
            raise FrameJobError("timeout", f"Reading the video took longer than {limits.timeout_s:g} s and was stopped. Choose frames nearer the start, or use a shorter video.")
        if rc != 0:
            raise FrameJobError("worker_failed", "The video decoder stopped unexpectedly (the file may be corrupt).")
        try:
            out = json.loads(stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise FrameJobError("worker_failed", "The video decoder returned nothing usable.") from None
        if not out.get("opened"):
            raise FrameJobError("not_decodable", "The video could not be opened for reading.")
        err = out.get("error")
        if err:
            msg = {"dimensions": "The video's frame size exceeds the supported limits.", "decoded_dimensions": "The decoded frame size exceeds the supported limits.",
                   "too_many_indices": f"At most {limits.max_indices} frames can be read in one step.", "write_failed": "A frame could not be written."}.get(err, "The frames could not be read.")
            raise FrameJobError(err, msg)
        written = out.get("written", {})
        wanted = {f"full_{i}" for i in full_indices} | {f"prev_{i}" for i in preview_indices}
        missing = sorted(wanted - set(written))
        if missing:
            raise FrameJobError("frame_not_decodable", f"The video has no decodable frame {missing[0].split('_')[1]} (it has {out.get('decoded_frames')} frames). Choose an earlier frame.")
        for rec in written.values():                                       # the worker's paths must lie inside this job's directory
            if job_dir.resolve() not in Path(rec["path"]).resolve().parents:
                raise FrameJobError("worker_failed", "The decoder produced an unexpected output.")
        if sum(f.stat().st_size for f in job_dir.rglob("*") if f.is_file()) > limits.max_output_bytes:
            raise FrameJobError("output_too_large", "The extracted frames exceeded the temporary-storage limit for one step.")
        return dict(job_dir=str(job_dir), written=written, decoded_frames=int(out.get("decoded_frames") or 0), decoded_wh=out.get("decoded_wh"), sharpness=out.get("sharpness"), seconds=round(dt, 2))
    except BaseException:
        workspace.drop(job_dir)                                            # no partial output survives a failure, a timeout or a cancel
        raise


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        sys.exit(_worker_main(sys.argv[2]))
    sys.exit("usage: frame_worker.py --worker <request-json>")
