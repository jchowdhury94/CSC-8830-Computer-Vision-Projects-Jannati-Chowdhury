"""Three-viewpoint Wall B: the SANDBOXED worker (no Streamlit import).  Runs as ``python -I three_view_worker.py <job_dir>`` under the same parent supervision as the two-viewpoint
worker (timeout, process-tree memory ceiling, worker-side peak-RSS watchdog, process-tree kill, job directory deleted).

It reads ONLY the job directory the parent staged -- ``request.json`` (calibration, frame table; written by the shared staging code), ``three_view.json`` (annotations, configuration) and
``frames/*.png`` (the visitor's own extracted frames, copied byte for byte) -- and writes ONLY ``result.json``.  It never reads a video.  No frame is resized, cropped, rotated or resampled; a frame
whose size or fingerprint is not the recorded one stops the job.  All mathematics is ``room_measurement.three_view_wall_b`` (a port of the validated offline reconstruction, built on the
unchanged Method J primitives); nothing numerical lives here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
SCHEMA = "f8.three_view_worker/1"


def _peak_rss_mb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(v) / (1024 * 1024) if sys.platform == "darwin" else float(v) / 1024
    except Exception:
        return None


DEFAULT_CPU_LIMIT_S = 1800                                           # unchanged; the supervisor now sends the limit it configured (``ReconstructionLimits.cpu_limit_s``)
CPU_LIMIT_RANGE_S = (60, 7200)


def cpu_limit_from(limits: dict | None, env=None) -> int:
    """The CPU-time limit (seconds, all threads) this worker applies to itself.  Order: the value the supervisor wrote into the request (``limits['cpu_limit_s']``), then the
    ``F8_RECON_CPU_S`` environment variable (only present when this worker is run by hand: the supervisor scrubs the environment), then 1800.  A value outside 60-7200 or not a
    finite number is ignored, never clamped.  Pure: applies nothing."""
    import math
    env = os.environ if env is None else env
    for raw in ((limits or {}).get("cpu_limit_s"), env.get("F8_RECON_CPU_S")):
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and CPU_LIMIT_RANGE_S[0] <= v <= CPU_LIMIT_RANGE_S[1]:
            return int(v)
    return DEFAULT_CPU_LIMIT_S


def _start_watchdog(limits: dict):
    """A daemon thread that checks this process's OWN peak RSS (the kernel's high-water mark, so a spike between two checks cannot be missed) every ``watchdog_s`` seconds and stops the process with
    ``exit_code`` the moment it passes the ceiling the parent granted.  It reads one number and changes nothing in the computation; it supplements the parent's sampling.  No-op when the
    platform cannot report peak RSS or no ceiling was granted."""
    try:
        import threading
        ceiling = float(limits["max_rss_mb"]); every = float(limits.get("watchdog_s", 0.05)); code = int(limits.get("exit_code", 86))
        if not (ceiling > 0 and every > 0) or _peak_rss_mb() is None:
            return None

        def loop():
            while True:
                v = _peak_rss_mb()
                if v is not None and v > ceiling:
                    os._exit(code)
                time.sleep(every)
        t = threading.Thread(target=loop, daemon=True, name="three-view-rss-watchdog"); t.start(); return t
    except Exception:
        return None


def _clean(o):
    """JSON-safe copy: non-finite floats become None (the schema is written with allow_nan=False)."""
    import math
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "tolist"):
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def _fail(out: dict, code: str, message: str, stage: str, **extra) -> dict:
    out.setdefault("failures", []).append(dict(code=code, message=message, stage=stage, **extra)); out["result"] = None; return out


def run_job(job_dir) -> dict:
    """Process the staged job and return the worker document.  Never raises for a data problem: it returns a typed failure."""
    for p in (str(MODULE), str(MODULE.parent), str(HERE)):   # MODULE.parent: the shared stereo_measure package
        if p not in sys.path:
            sys.path.insert(0, p)
    import cv2
    from room_measurement import three_view_wall_b as T
    from room_measurement import visitor_annotations as V
    job = Path(job_dir); t_all = time.perf_counter(); out = dict(schema=SCHEMA, failures=[], result=None, timings={}, peak_rss_mb=None)
    req = json.loads((job / "request.json").read_text()); tv = json.loads((job / "three_view.json").read_text())
    c = req["calibration"]
    try:
        cal = T.calibration_from_contract(dict(K1=c["K_left"], D1=c["D_left"], K2=c["K_right"], D2=c["D_right"], image_size_wh=c["image_size_wh"], baseline_mm=c["baseline_mm"]))
        ann = T.parse_annotations(tv["annotations"], cal.image_size)
    except T.ThreeViewError as e:
        return _fail(out, "ANNOTATION_INVALID" if e.code.startswith("ANNOTATION") else "CALIBRATION_INVALID", e.message, "request")
    keys = [int(k) for k in tv["keys"]]; table = {k: req["frames"][n] for n, k in enumerate(keys)}; W, H = cal.image_size; decode_s = [0.0]

    def gray_of(key, v, side):
        t0 = time.perf_counter(); slot = f"p{v}_{'left' if side == 'L' else 'right'}"; rec = table[key][slot]; path = (job / rec["file"]).resolve()
        if job.resolve() not in path.parents or not path.is_file() or path.is_symlink():
            raise T.ThreeViewError("FRAME_STALE", f"frame set {key}, {slot}: the staged frame is missing")
        g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise T.ThreeViewError("FRAME_STALE", f"frame set {key}, {slot}: the staged frame could not be read")
        if (g.shape[1], g.shape[0]) != (W, H):
            raise T.ThreeViewError("FRAME_SIZE_MISMATCH", f"frame set {key}, {slot}: {g.shape[1]} x {g.shape[0]} px but the calibration is {W} x {H} px (nothing is resized)")
        if V.frame_fingerprint(g) != rec["fingerprint"]:
            raise T.ThreeViewError("FRAME_STALE", f"frame set {key}, {slot}: the staged frame does not match its recorded fingerprint")
        decode_s[0] += time.perf_counter() - t0
        return g

    cfg = T.ThreeViewConfig(**tv.get("config", {}))
    try:
        res = T.measure_from_grays(cal, gray_of, keys, ann, cfg)
    except T.ThreeViewError as e:
        return _fail(out, e.code, e.message, "frames" if e.code.startswith(("FRAME", "RESOLUTION")) else "annotations" if e.code.startswith("ANNOTATION") else "worker")
    out["result"] = res; out["timings"] = dict(decode_and_verify=decode_s[0], **(res.get("timing_s") or {}), total=time.perf_counter() - t_all); out["peak_rss_mb"] = _peak_rss_mb()
    out["matcher"] = "fast" if cfg.uses_fast_matcher() else "original"
    return out


def main(argv) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: three_view_worker.py <job_dir>\n"); return 2
    job = Path(argv[1]); limits = {}
    try:
        limits = (json.loads((job / "request.json").read_text()) or {}).get("limits") or {}
        _start_watchdog(limits)
    except Exception:
        pass
    try:
        import resource
        cpu = cpu_limit_from(limits)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu)); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
    try:
        out = run_job(job)
    except MemoryError:
        return 3
    except Exception as ex:                                          # a defect must surface as a typed failure, never as a half-written result
        out = dict(schema=SCHEMA, failures=[dict(code="INTERNAL_ERROR", message=f"{type(ex).__name__}: {ex}", stage="worker")], result=None, timings={}, peak_rss_mb=_peak_rss_mb())
    tmp = job / "result.json.tmp"
    tmp.write_text(json.dumps(_clean(out), default=lambda o: o.item() if hasattr(o, "item") else str(o), allow_nan=False))
    os.replace(tmp, job / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
