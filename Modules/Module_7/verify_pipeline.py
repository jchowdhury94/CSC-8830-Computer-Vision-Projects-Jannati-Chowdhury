"""Verify the standalone Model B pipeline against the saved reference results (reference/expected_model_b_results.json).

    python verify_pipeline.py [--skip-calibration] [--full]

Checks: Model B integrity, availability of the 30 corrected-corner pairs and the saved selections / display images, triangulation /
Z-depth / width / height of the three notebook pairs, positive depth in both cameras, agreement of the saved outputs/ tables with a
fresh measurement, calibration reproducibility, and (via an audit hook on file opens) that no Model A / old-experiment file is
ever read. Exit code 1 if any check fails.

The original photographs are git-ignored, so by default their absence is reported as SKIP and nothing depends on them. --full
is the local mode: it additionally requires all 60 calibration photographs and the 6 notebook photographs to be present.
"""
import argparse
import csv
import hashlib
import json
import sys

FORBIDDEN = ("20pairs", "experiment_225mm", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9",
             "stereo_calibration.npz", "camera_calibration")
opened = set()


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], str):
        opened.add(args[0])


sys.addaudithook(_audit)

import numpy as np  # noqa: E402

from run_calibration import compare  # noqa: E402
from stereo_measure import calibrate_model_b, load_model_b, measure_pair  # noqa: E402
from stereo_measure.config import (CALIB_PAIR_IDS, CALIB_PHOTO_DIR, CALIB_PREFIX, CORNERS_DIR, IMAGE_SIZE, MODEL_B_NPZ,  # noqa: E402
                                   NOTEBOOK_PAIR_IDS, OUTPUT_DIR, REFERENCE_JSON, ROOT, SELECTION_DIR)
from stereo_measure.calibration import load_corners  # noqa: E402
from stereo_measure.measurement import left_photo, right_photo  # noqa: E402

# the rounded values from the project brief: pair -> (mean Z, width, height) to 1 decimal
BRIEF = {1: (789.4, 301.9, 212.0), 2: (779.0, 300.5, 208.9), 3: (767.0, 299.9, 213.1)}
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--full", action="store_true", help="also require the git-ignored original photographs (local use)")
    args = ap.parse_args()
    ref = json.load(open(REFERENCE_JSON))

    sha = hashlib.sha256(MODEL_B_NPZ.read_bytes()).hexdigest()
    check("Model B file unchanged (SHA-256)", sha == ref["model_b_npz_sha256"], sha[:16])
    model = load_model_b()
    check("Model B loads standalone", model["image_size"] == IMAGE_SIZE and model["pairs"] == CALIB_PAIR_IDS,
          f"baseline {model['baseline_mm']:.4f} mm, stereo rms {model['stereo_rms']:.4f} px, pairs {len(model['pairs'])}")

    shapes = [load_corners(s, i).shape for s in ("left", "right") for i in CALIB_PAIR_IDS if (CORNERS_DIR / f"{s}_{i:02d}_corners.csv").is_file()]
    check("corrected corners present for pairs 1-30 (48 x 2 each)", len(shapes) == 60 and set(shapes) == {(48, 2)}, f"{len(shapes)}/60 files")
    sel = [SELECTION_DIR / f"pair_{i}_corners.json" for i in NOTEBOOK_PAIR_IDS]
    check("3 saved corner selections present", all(p.is_file() for p in sel))
    disp = [ROOT / "display" / f"pair_{i}_{s}.jpg" for i in NOTEBOOK_PAIR_IDS for s in ("left", "right")]
    check("6 display images used by the app present", all(p.is_file() for p in disp), f"{sum(p.is_file() for p in disp)}/6 files")

    # original photographs are git-ignored: required only with --full, otherwise reported and skipped (never a failure)
    photos = [CALIB_PHOTO_DIR / s / f"{CALIB_PREFIX[s]}_{i}.jpeg" for s in ("left", "right") for i in CALIB_PAIR_IDS]
    nb = [f for i in NOTEBOOK_PAIR_IDS for f in (left_photo(i), right_photo(i))]
    for name, files in (("all 30 calibration photo pairs (60 files)", photos), ("3 notebook photo pairs (6 files)", nb)):
        n = sum(p.is_file() for p in files)
        if args.full:
            check(f"{name} present", n == len(files), f"{n}/{len(files)} files")
        else:
            print(f"[SKIP] {name}: {n}/{len(files)} original photographs present; not required without --full")

    for i in NOTEBOOK_PAIR_IDS:
        r, e = measure_pair(model, i), ref["pairs"][str(i)]
        s, tol = r["summary"], e["tolerance_mm"]
        check(f"pair {i}: positive finite depth in both cameras", s["depth_positive_both_cameras"] and s["all_finite"],
              f"left Z {r['X'][:, 2].min():.1f}-{r['X'][:, 2].max():.1f}, right Z min {r['Xr'][:, 2].min():.1f} mm")
        dz = max(abs(r["X"][k, 2] - e["corner_Z_mm"][c]) for k, c in enumerate(("TL", "TR", "BR", "BL")))
        check(f"pair {i}: corner Z-depths reproduce saved ({e['precision']})", dz <= tol, f"max diff {dz:.2e} mm")
        check(f"pair {i}: mean Z reproduces saved", abs(s["mean_Z_depth_mm"] - e["mean_Z_mm"]) <= tol, f"{s['mean_Z_depth_mm']:.4f} vs {e['mean_Z_mm']:.4f}")
        # summary keys: top_width, ..., width_mm; reference keys all carry the _mm suffix
        de = max(abs(s[k] - e[k if k.endswith("_mm") else k + "_mm"])
                 for k in ("top_width", "bottom_width", "left_height", "right_height", "width_mm", "height_mm"))
        check(f"pair {i}: edge lengths, width, height reproduce saved", de <= tol, f"max diff {de:.2e} mm")
        bz, bw, bh = BRIEF[i]
        check(f"pair {i}: matches brief values to 1 decimal", (round(s["mean_Z_depth_mm"], 1), round(s["width_mm"], 1), round(s["height_mm"], 1)) == (bz, bw, bh),
              f"Z {s['mean_Z_depth_mm']:.1f}, W {s['width_mm']:.1f}, H {s['height_mm']:.1f}")
        check(f"pair {i}: ground-truth errors consistent", abs(s["width_err_mm"] - (s["width_mm"] - ref['ground_truth_mm']['width'])) < 1e-9 and
              abs(s["height_err_mm"] - (s["height_mm"] - ref['ground_truth_mm']['height'])) < 1e-9,
              f"width {s['width_err_mm']:+.2f} mm ({s['width_err_pct']:+.2f}%), height {s['height_err_mm']:+.2f} mm ({s['height_err_pct']:+.2f}%)")

    # the saved outputs/ tables must agree with a fresh measurement from the saved selections (CSVs are rounded to 4 decimals)
    fresh = {i: measure_pair(model, i) for i in NOTEBOOK_PAIR_IDS}
    with open(OUTPUT_DIR / "pair_summary.csv", newline="") as f:
        rows = {int(r["pair"]): r for r in csv.DictReader(f)}
    d = max(abs(float(rows[i][k]) - fresh[i]["summary"][k]) for i in NOTEBOOK_PAIR_IDS
            for k in ("mean_Z_depth_mm", "min_Z_mm", "max_Z_mm", "top_width", "bottom_width", "left_height", "right_height", "width_mm", "height_mm"))
    check("outputs/pair_summary.csv matches a fresh measurement", set(rows) == set(NOTEBOOK_PAIR_IDS) and d < 1e-4, f"max diff {d:.2e} mm")
    with open(OUTPUT_DIR / "corner_coordinates_and_depths.csv", newline="") as f:
        crow = list(csv.DictReader(f))
    d = max(abs(float(r["Z_depth_mm"]) - fresh[int(r["pair"])]["X"][("TL", "TR", "BR", "BL").index(r["corner"]), 2]) for r in crow)
    check("outputs/corner_coordinates_and_depths.csv matches a fresh measurement", len(crow) == 12 and d < 1e-4, f"{len(crow)} rows, max Z diff {d:.2e} mm")
    js = json.load(open(OUTPUT_DIR / "measurement_summary.json"))
    d = max(abs(p[k] - fresh[p["pair"]]["summary"][k]) for p in js["pairs"] for k in ("mean_Z_depth_mm", "width_mm", "height_mm"))
    check("outputs/measurement_summary.json matches a fresh measurement", len(js["pairs"]) == 3 and d < 1e-6, f"max diff {d:.2e} mm")
    figs = [OUTPUT_DIR / "visualizations" / f for f in ("depth_topdown.png", "error_vs_ground_truth.png", "pair_1_measurement.jpg", "pair_2_measurement.jpg", "pair_3_measurement.jpg")]
    check("saved visualization files present", all(p.is_file() for p in figs))

    if not args.skip_calibration:
        fresh = calibrate_model_b()
        worst = max(compare(fresh, model).values())
        # K is ~4400 px, so 1e-5 absolute is ~2e-9 relative: optimizer-level floating-point noise, not a different calibration
        check("calibration reproducible from the 30 corrected-corner pairs", worst < 1e-5, f"max |fresh - stored| = {worst:.2e}")
        dm = max(abs(measure_pair(fresh, i)["summary"][k] - measure_pair(model, i)["summary"][k])
                 for i in NOTEBOOK_PAIR_IDS for k in ("mean_Z_depth_mm", "width_mm", "height_mm"))
        check("fresh calibration gives the same Z / width / height", dm < 1e-4, f"max diff {dm:.2e} mm")

    bad = sorted(p for p in opened if any(f in p for f in FORBIDDEN))
    check("Model A / old experiment files never opened", not bad, ", ".join(bad))
    inside = sorted(p for p in opened if str(ROOT) in p and p.endswith((".npz", ".csv", ".json")))
    print("data files read:", len(inside), "| distinct dirs:", sorted({p.replace(str(ROOT) + '/', '').rsplit('/', 1)[0] for p in inside}))
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
