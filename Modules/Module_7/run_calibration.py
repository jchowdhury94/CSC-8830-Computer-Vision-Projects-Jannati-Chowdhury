"""Reproduce the Model B calibration from the 30 corrected checkerboard corner pairs and compare it with the stored model.

    python run_calibration.py [--save PATH]

By default nothing is written: the fresh calibration is only compared with calibration/model_b/stereo_calibration_30pairs.npz.
"""
import argparse
from pathlib import Path

import numpy as np

from stereo_measure import calibrate_model_b, load_model_b

KEYS = ("K1", "D1", "K2", "D2", "R", "T", "E", "F")


def compare(fresh, stored):
    diffs = {k: float(np.abs(fresh[k] - stored[k]).max()) for k in KEYS}
    diffs.update({k: abs(fresh[k] - stored[k]) for k in ("stereo_rms", "rms_left", "rms_right", "baseline_mm")})
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", type=Path, help="also write the fresh calibration to this .npz path")
    args = ap.parse_args()
    fresh, stored = calibrate_model_b(), load_model_b()
    print(f"fresh: stereo rms {fresh['stereo_rms']:.6f} px, baseline {fresh['baseline_mm']:.4f} mm, {len(fresh['pairs'])} pairs")
    for k, v in compare(fresh, stored).items():
        print(f"  max |fresh - stored| {k:11s} {v:.3e}")
    if args.save:
        np.savez(args.save, K_left=fresh["K1"], D_left=fresh["D1"], K_right=fresh["K2"], D_right=fresh["D2"], R=fresh["R"], T=fresh["T"],
                 E=fresh["E"], F=fresh["F"], rms=fresh["stereo_rms"], baseline_mm=fresh["baseline_mm"], rms_left=fresh["rms_left"],
                 rms_right=fresh["rms_right"], image_size=np.array(fresh["image_size"]), flags=fresh["flags"], pairs=np.array(fresh["pairs"]))


if __name__ == "__main__":
    main()
