"""Run the Method J room measurement (Module 7 Part 3, Phase 1).

    python run_room_measurement.py [--out DIR] [--quick] [--skip-uncertainty] [--overwrite]

1. estimates width/height from the protected wall videos (no ground truth is touched)  -> estimation_result.json
2. only afterwards compares with the ground truth                                       -> validation.json
Videos are read-only.  Existing outputs are never overwritten unless --overwrite is given.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_measurement import pipeline  # noqa: E402
from room_measurement.frames import DEFAULT_VIDEO_DIR  # noqa: E402

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs" / "room_measurement")); ap.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    ap.add_argument("--quick", action="store_true", help="fewer uncertainty replicates (faster, noisier interval)"); ap.add_argument("--skip-uncertainty", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv); out = Path(a.out); est_p, val_p = out / "estimation_result.json", out / "validation.json"
    if (est_p.exists() or val_p.exists()) and not a.overwrite: sys.exit(f"{out} already has results; use --overwrite or another --out")
    kw = dict(n_bootstrap=12, n_intrinsics=12, n_boundary=100) if a.quick else None
    res = pipeline.run_estimation(video_dir=a.video_dir, with_uncertainty=not a.skip_uncertainty, uncertainty_kwargs=kw)
    out.mkdir(parents=True, exist_ok=True); est_p.write_text(json.dumps(res, indent=1) + "\n")
    from room_measurement import validation   # imported only now: estimation is finished and saved
    val = validation.compare_to_ground_truth(res); val_p.write_text(json.dumps(pipeline.to_jsonable(val), indent=1) + "\n")
    s = res["summary"]; print(f"\nestimate: width {s['width_mm']:.0f} mm, height {s['height_mm']:.0f} mm   (saved {est_p})")
    if "uncertainty" in res:
        u = res["uncertainty"]["total"]; print(f"  95% interval: width {u['width']['interval_95_mm'][0]:.0f}..{u['width']['interval_95_mm'][1]:.0f}, height {u['height']['interval_95_mm'][0]:.0f}..{u['height']['interval_95_mm'][1]:.0f}")
    for k in ("width_mm", "height_mm"): v = val[k]; print(f"validation {k}: error {v['error_mm']:+.0f} mm ({v['error_percent']:+.2f}%)")

if __name__ == "__main__":
    main()
