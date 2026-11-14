"""Stage F8.5: the calibration WORKER process (run as ``python -I calibration_worker.py <spec.json>``; never imported by the app).

It reads two verified video PATHS, takes shared file locks on them (so nothing can delete them while it reads), and runs the EXISTING F7 functions
unchanged, sequentially:  extract_detections(left) -> extract_detections(right) -> calibrate_from_detections -> assess (-> contract_from_model).
No video is copied into memory; F7 decodes sequentially and keeps only corner coordinates and a motion value per frame.
It writes ``progress.json`` while running and ``result.json`` at the end (both small, atomic). It exits on its own if its parent disappears.

Two modes (``spec["mode"]``):  "calibrate" (default; the full F7 pipeline)  and  "benchmark" (F8.7: ONLY ``extract_detections`` with its existing
``max_frames`` argument on the first N frames of each video, then stop; nothing is solved or assessed). Both modes treat the videos as strictly
READ-ONLY: they are opened for reading only, must live under ``spec["allowed_root"]`` (the upload store), and are re-hashed (SHA-256) against the
digest verified at upload time before they are used; the benchmark re-hashes them again at the end to prove they were not modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                          # -I removes the script directory from sys.path; this is our own trusted code

import filelock                                                        # noqa: E402

MiB = 1024 * 1024
MAX_DUMP = 4000                                                        # detections per camera written to result.json (bounded)


def _peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / MiB if sys.platform == "darwin" else r / 1024, 1)


def jsonable(o, depth=0):
    """numpy -> plain JSON; NaN/inf -> None; bounded depth. Never raises."""
    import numpy as np
    if depth > 8:
        return None
    if isinstance(o, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v, depth + 1) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist(), depth + 1)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)[:200]


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


class _Done(Exception):
    """Internal: leave the try-block after a benchmark (not an error)."""


def run_benchmark(K, spec, left_path, right_path, pattern, progress, stages, expected, t_start):
    """The F8.7 Cloud benchmark. Calls the UNMODIFIED ``K.extract_detections(path, pattern, K.STRIDE, cb, max_frames=N)`` on the first N frames of each video
    (sequentially, left then right) and reports counts and timings. No calibration is solved. The caller has already hard-capped N."""
    n = int(spec["max_frames"]); counts = spec.get("frame_counts") or {}
    out = dict(max_frames_per_video=n, stride=K.STRIDE, pattern=list(pattern), videos={})
    cpu0 = cpu_seconds(); acc = 0                                                                  # acc: frames finished in earlier videos
    for side, path, lo, hi in (("left", left_path, 0.05, 0.50), ("right", right_path, 0.50, 0.95)):
        total = int(counts.get(side) or 0)
        target = min(n, total) if total else n

        def cb(f, side=side, lo=lo, hi=hi, total=total, target=target, acc=acc):                  # F7 reports i / (frames in the whole video)
            done = min(f * total if total else f * target, target)
            progress("detect_" + side, lo + (hi - lo) * done / max(target, 1), frames=int(acc + done))
        t0, c0 = time.perf_counter(), cpu_seconds(); progress("detect_" + side, lo, True)
        d = K.extract_detections(path, pattern, K.STRIDE, cb, max_frames=n)
        wall, cpu = time.perf_counter() - t0, cpu_seconds() - c0
        frames, dets = int(d["n_frames"]), len(d["detections"]); attempts = (frames + K.STRIDE - 1) // K.STRIDE
        out["videos"][side] = dict(frames_processed=frames, detection_attempts=attempts, detections=dets, detection_rate=round(dets / attempts, 4) if attempts else None,
                                   wall_s=round(wall, 3), cpu_s=round(cpu, 3), frames_per_s=round(frames / wall, 2) if wall > 0 else None,
                                   ms_per_attempt=round(1000 * wall / attempts, 1) if attempts else None, size=jsonable(d["size"]), fps=jsonable(d["fps"]),
                                   video_frame_count=total or None)
        stages["detect_" + side] = dict(wall_s=round(wall, 3), peak_rss_mb=_peak_rss_mb(), frames=frames, detections=dets)
        acc += frames
    V = out["videos"].values()
    out.update(frames_processed=sum(v["frames_processed"] for v in V), detection_attempts=sum(v["detection_attempts"] for v in V), detections=sum(v["detections"] for v in V),
               detect_wall_s=round(sum(v["wall_s"] for v in V), 3), detect_cpu_s=round(cpu_seconds() - cpu0, 3))
    out["frames_per_s"] = round(out["frames_processed"] / out["detect_wall_s"], 2) if out["detect_wall_s"] > 0 else None
    t0 = time.perf_counter(); progress("verify_after", 0.97, True)                                  # prove the benchmark left both files byte-for-byte unchanged
    out["videos_unchanged"] = all(sha256_file(p) == expected.get(s) for s, p in (("left", left_path), ("right", right_path))) if expected else None
    stages["verify_after"] = dict(wall_s=round(time.perf_counter() - t0, 3))
    return out


def cpu_seconds():
    """User+system CPU of this process and its (reaped) children, from the kernel's own accounting."""
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return round(a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime, 3)


def inside(path, root):
    """True when ``path`` resolves (symlinks followed) to a location under ``root``."""
    try:
        p, r = os.path.realpath(path), os.path.realpath(root)
        return p == r or p.startswith(r.rstrip(os.sep) + os.sep)
    except (OSError, ValueError):
        return False


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    out_dir = Path(spec["out_dir"]); t_start = time.perf_counter(); stages = {}; state = dict(stage="start", last=0.0)
    parent = spec.get("parent_pid")
    left_path, right_path = spec["left_path"], spec["right_path"]

    def clean(msg):                                                    # never leak server paths into results
        s = str(msg)
        for p, name in ((left_path, "<LEFT>"), (right_path, "<RIGHT>"), (str(out_dir), "<JOB>")):
            s = s.replace(p, name)
        return s[:600]

    def progress(stage, frac=0.0, force=False, **extra):
        now = time.perf_counter()
        if parent is not None and os.getppid() != parent:             # orphaned: the supervisor is gone, stop at once
            os._exit(4)
        if force or stage != state["stage"] or now - state["last"] >= 1.0:
            state.update(stage=stage, last=now)
            write_json(out_dir / "progress.json", dict(stage=stage, fraction=round(float(frac), 4), t=round(now - t_start, 2), pid=os.getpid(), peak_rss_mb=_peak_rss_mb(), **extra))

    result = dict(ok=False, error=None, stage_timings={}, env={}, f7={})
    try:
        mode = spec.get("mode", "calibrate")
        if mode not in ("calibrate", "benchmark"):
            raise ValueError("unknown worker mode")
        root = spec.get("allowed_root")
        if root and not all(inside(p, root) for p in (left_path, right_path)):                      # only the store's own copies may ever be read
            raise PermissionError("a video is outside the upload store")
        locks = filelock.acquire_shared([left_path, right_path])      # kept referenced until exit
        progress("import", 0.0, True)
        import cv2
        sys.path.insert(0, spec["f7_dir"])                             # trusted, configured by the supervisor
        import f7_calibration as K                                     # the EXISTING F7 module, unchanged

        def info(mod):
            f = getattr(mod, "__file__", None)
            return dict(file=Path(f).name if f else None, sha256=sha256_file(f) if f else None)
        build = cv2.getBuildInformation()
        result["env"] = dict(cv2=cv2.__version__, threads=cv2.getNumThreads(), cpus=cv2.getNumberOfCPUs(), python=sys.version.split()[0],
                             ffmpeg=next((l.strip()[:80] for l in build.splitlines() if "FFMPEG" in l), None), pid=os.getpid())
        result["f7"] = dict(stride=K.STRIDE, max_pairs=K.MAX_PAIRS, offset_search_s=K.OFFSET_SEARCH_S, f7_calibration=info(K), visitor_calibration=info(K.VC),
                            visitor_video_validation=info(K.VV), user_calibration=info(K.uc))
        pattern = (int(spec["cols"]), int(spec["rows"])); settings = dict(cols=pattern[0], rows=pattern[1], square_mm=float(spec["square_mm"]))
        baseline = spec.get("baseline_mm")

        def stage_cb(name, weight_lo, weight_hi):
            return lambda f: progress(name, weight_lo + (weight_hi - weight_lo) * float(f))

        expected = spec.get("expected_sha256") or {}
        t0 = time.perf_counter(); progress("verify", 0.0, True)
        for side, path in (("left", left_path), ("right", right_path)):                            # same bytes that were verified at upload time?
            want = expected.get(side)
            if want and sha256_file(path) != want:
                raise ValueError(f"the {side} video no longer matches its verified SHA-256")
        stages["verify"] = dict(wall_s=round(time.perf_counter() - t0, 3), checked=bool(expected))
        if mode == "benchmark":
            result.update(benchmark=run_benchmark(K, spec, left_path, right_path, pattern, progress, stages, expected, t_start))
            result.update(ok=True)
            progress("done", 1.0, True)
            raise _Done()

        dets = {}
        for side, path, lo, hi in (("left", left_path, 0.02, 0.50), ("right", right_path, 0.50, 0.98)):
            t0 = time.perf_counter(); progress("detect_" + side, lo, True)
            dets[side] = K.extract_detections(path, pattern, K.STRIDE, stage_cb("detect_" + side, lo, hi))        # sequential decode; frames are not kept
            stages["detect_" + side] = dict(wall_s=round(time.perf_counter() - t0, 3), peak_rss_mb=_peak_rss_mb(), frames=dets[side]["n_frames"], detections=len(dets[side]["detections"]))
        t0 = time.perf_counter(); progress("solve", 0.98, True)
        res = K.calibrate_from_detections(dets["left"], dets["right"], settings)
        stages["solve"] = dict(wall_s=round(time.perf_counter() - t0, 3), peak_rss_mb=_peak_rss_mb())
        t0 = time.perf_counter(); progress("assess", 0.99, True)
        res["runtime_s"] = round(time.perf_counter() - t_start, 3)
        assessment = K.assess(res, baseline)
        contract = None
        if res.get("ok") and assessment["level"] != K.FAILED:
            contract = K.contract_from_model(res["model"], baseline)
        stages["assess"] = dict(wall_s=round(time.perf_counter() - t0, 3), peak_rss_mb=_peak_rss_mb())

        model = res.get("model")
        summary = None
        if model is not None:
            summary = dict(K1=model["K1"], D1=model["D1"], K2=model["K2"], D2=model["D2"], R=model["R"], T=model["T"], baseline_mm=model["baseline_mm"],
                           stereo_rms=model["stereo_rms"], rms_left=model["rms_left"], rms_right=model["rms_right"], image_size_left=model.get("image_size_left"),
                           image_size_right=model.get("image_size_right"), distortion_left=model.get("distortion_left"), distortion_right=model.get("distortion_right"))
        dump = None
        if spec.get("dump_detections", True):
            dump = {s: dict(size=dets[s]["size"], fps=dets[s]["fps"], n_frames=dets[s]["n_frames"],
                            detections=[dict(frame=d["frame"], square_px=d["square_px"], sharp=d["sharp"], corners=[[round(float(x), 4), round(float(y), 4)] for x, y in d["corners"]])
                                        for d in dets[s]["detections"][:MAX_DUMP]],
                            motion=[round(float(v), 4) for v in dets[s]["motion"]]) for s in ("left", "right")}
        slim = {k: v for k, v in res.items() if k not in ("model",)}
        result.update(ok=bool(res.get("ok")), error=clean(res.get("error")) if res.get("error") else None, calibration=jsonable(slim), model=jsonable(summary),
                      assessment=jsonable(assessment), contract=jsonable(contract), detections=jsonable(dump), sizes=dict(left=jsonable(dets["left"]["size"]), right=jsonable(dets["right"]["size"])))
        progress("done", 1.0, True)
    except _Done:
        code = 0
    except BaseException as e:                                         # the worker must always leave a result
        result.update(ok=False, error=f"{type(e).__name__}: {clean(e)}", traceback=clean(traceback.format_exc())[-1500:])
        code = 3
    else:
        code = 0
    result["stage_timings"] = stages; result["total_wall_s"] = round(time.perf_counter() - t_start, 3); result["peak_rss_mb"] = _peak_rss_mb()
    result["worker_cpu_s"] = cpu_seconds()
    write_json(out_dir / "result.json", jsonable(result))
    return code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("internal worker: python -I calibration_worker.py <spec.json>", file=sys.stderr); sys.exit(2)
    sys.exit(main(sys.argv[1]))
