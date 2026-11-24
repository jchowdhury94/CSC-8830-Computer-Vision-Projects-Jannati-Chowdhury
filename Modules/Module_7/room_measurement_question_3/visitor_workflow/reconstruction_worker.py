"""F8.31C-1: the SANDBOXED reconstruction worker (no Streamlit import).  Runs as ``python -I reconstruction_worker.py <job_dir>``.

It reads ONLY the job directory the parent staged (``request.json`` and ``frames/*.png``, copied byte for byte from the visitor's own extracted frames),
and writes ONLY ``result.json`` in it.  It never reads a video, never decodes more than one frame set at a time, and never touches any other path.

Per frame set, in the original order and with the original functions and parameters:

    ``features.match_set(frames)``          SIFT (12000 features, contrast 0.04), ratio test 0.75 (stereo) / 0.85 (cross-view), mutual check, seed 0
    ``visitor_method_j_adapter.evaluate``   ``pose.build_dataset`` (``features.build_tracks``) -> ``pose.estimate_pose`` -> original floor / registration /
                                            ``measure_wall`` geometry -> the validated acceptance and failure classification

No frame is resized, cropped, rotated or resampled; a frame whose size or fingerprint is not the recorded one stops the job.  Nothing numerical lives here.
``run_job`` is importable so that tests can call it in-process with a MOCKED ``match_set`` (clearly labelled as such); the subprocess entry always uses the real one.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
SCHEMA = "f8.reconstruction_worker/1"
VIEW_KEYS = ("AL", "PL", "AR", "PR")
REQUIRED_LINES = ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF")


def _peak_rss_mb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(v) / (1024 * 1024) if sys.platform == "darwin" else float(v) / 1024
    except Exception:
        return None


def _fail(out: dict, code: str, message: str, stage: str, **extra) -> dict:
    out.setdefault("failures", []).append(dict(code=code, message=message, stage=stage, **extra)); out["adapter"] = None; return out


def run_job(job_dir, *, match_set=None, evaluate=None) -> dict:
    """Process the staged job and return the worker document (also written by ``main``).  Never raises for a data problem: it returns a typed failure."""
    for p in (str(MODULE), str(MODULE.parent)):   # MODULE.parent: the shared stereo_measure package
        if p not in sys.path:
            sys.path.insert(0, p)
    import cv2
    import numpy as np
    from room_measurement import features
    from room_measurement import visitor_annotations as V
    from room_measurement import visitor_method_j_adapter as A
    match_set = match_set or features.match_set                      # the ORIGINAL matcher with its original defaults
    evaluate = evaluate or A.evaluate
    job = Path(job_dir); t_all = time.perf_counter(); timings = dict(decode_and_verify=0.0, feature_matching=0.0, method_j=0.0)
    out = dict(schema=SCHEMA, failures=[], sets=[], adapter=None, timings=timings, peak_rss_mb=None)
    req = json.loads((job / "request.json").read_text())
    cal = req["calibration"]
    ctx = A.VisitorCalibrationContext.from_params(dict(K_left=cal["K_left"], D_left=cal["D_left"], K_right=cal["K_right"], D_right=cal["D_right"], baseline_mm=cal["baseline_mm"]),
                                                  tuple(cal["image_size_wh"]), label="visitor")
    W, H = (int(v) for v in cal["image_size_wh"]); match_sets = []
    for i, fs in enumerate(req["frames"]):
        t0 = time.perf_counter(); frames = {}
        for vk in VIEW_KEYS:
            rec = fs[vk]; path = (job / rec["file"]).resolve()
            if job.resolve() not in path.parents or not path.is_file() or path.is_symlink():
                return _fail(out, "FRAME_STALE", f"frame set {i}, {vk}: the staged frame is missing", "decode", frame_set=i, view=vk)
            g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if g is None:
                return _fail(out, "FRAME_STALE", f"frame set {i}, {vk}: the staged frame could not be read", "decode", frame_set=i, view=vk)
            if (g.shape[1], g.shape[0]) != (W, H):
                return _fail(out, "FRAME_SIZE_MISMATCH", f"frame set {i}, {vk}: {g.shape[1]} x {g.shape[0]} px but the calibration is {W} x {H} px (nothing is resized)", "decode", frame_set=i, view=vk)
            if V.frame_fingerprint(g) != rec["fingerprint"]:
                return _fail(out, "FRAME_STALE", f"frame set {i}, {vk}: the staged frame does not match its recorded fingerprint", "decode", frame_set=i, view=vk)
            frames[vk] = g
        timings["decode_and_verify"] += time.perf_counter() - t0; t0 = time.perf_counter()
        try:
            F = match_set(frames)                                    # original feature extraction + matching; raises (as the original does) when nothing matches
            tracks = features.build_tracks(F)
        except Exception as ex:                                      # the original pipeline's own failure behaviour, typed
            return _fail(out, "FEATURE_MATCHING_FAILED", f"frame set {i}: {type(ex).__name__}: {ex}", "features", frame_set=i)
        timings["feature_matching"] += time.perf_counter() - t0
        out["sets"].append(dict(frame_set=i, features=dict(n_stereo_matches_view1=int(len(F["A1"])), n_stereo_matches_view2=int(len(F["A2"])), n_cross_view_matches=int(len(F["MX"])), n_four_view_tracks=int(len(tracks)))))
        match_sets.append(F); del frames
    line_sets = []
    for ls in req["line_sets"]:
        line_sets.append({n: dict(kind=d["kind"], a=float(d["a"]), b=float(d["b"]), n=int(d["n"]), N=int(d["N"]), rms=float(d["rms"]), span=(float(d["span"][0]), float(d["span"][1])),
                                  pts=np.asarray(d["pts"], dtype=np.dtype(d.get("pts_dtype", "float64"))).reshape(-1, 2)) for n, d in ls.items()})
    t0 = time.perf_counter()
    inputs = A.ReconstructionInputs(tuple(match_sets), tuple(line_sets), ctx)         # lines were fitted under THIS calibration (the visitor's), so nothing is retargeted
    out["adapter"] = evaluate(inputs, ctx)
    timings["method_j"] = time.perf_counter() - t0; timings["total"] = time.perf_counter() - t_all; out["peak_rss_mb"] = _peak_rss_mb()
    return out


def _start_watchdog(limits: dict):
    """F8.31C-1C: a daemon thread that checks this process's OWN peak RSS (the kernel's high-water mark, so a spike between two checks cannot be missed) every
    ``watchdog_s`` seconds and stops the process with ``exit_code`` the moment it passes the ceiling the parent granted.  It reads one number and changes nothing in the
    computation; it supplements, and does not replace, the parent's sampling.  No-op when the platform cannot report peak RSS or no ceiling was granted."""
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
        t = threading.Thread(target=loop, daemon=True, name="recon-rss-watchdog"); t.start(); return t
    except Exception:
        return None


def main(argv) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: reconstruction_worker.py <job_dir>\n"); return 2
    job = Path(argv[1])
    try:
        _start_watchdog((json.loads((job / "request.json").read_text()) or {}).get("limits") or {})
    except Exception:
        pass
    try:
        import resource
        cpu = int(float(os.environ.get("F8_RECON_CPU_S", "1800")))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu)); resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
    try:
        out = run_job(job)
    except MemoryError:
        return 3
    except Exception as ex:                                          # a defect must surface as a typed failure, never as a half-written result
        out = dict(schema=SCHEMA, failures=[dict(code="INTERNAL_ERROR", message=f"{type(ex).__name__}: {ex}", stage="worker")], sets=[], adapter=None, timings={}, peak_rss_mb=_peak_rss_mb())
    tmp = job / "result.json.tmp"
    tmp.write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o), allow_nan=False))
    os.replace(tmp, job / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
