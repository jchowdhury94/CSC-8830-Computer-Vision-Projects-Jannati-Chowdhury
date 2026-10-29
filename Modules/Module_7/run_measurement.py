"""Measure the notebook in the three stereo pairs with the stored Model B calibration and write the final results.

    python run_measurement.py [--out DIR] [--overwrite]

Writes corner_coordinates_and_depths.csv, edge_length_calculations.csv, pair_summary.csv, measurement_summary.json and
visualizations/ to DIR (default outputs/). Nothing is recalibrated or rescaled.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from stereo_measure import load_model_b, measure_pair
from stereo_measure.config import GT_HEIGHT_MM, GT_WIDTH_MM, NOTEBOOK_PAIR_IDS, OUTPUT_DIR
from stereo_measure.visualization import annotated_pair, depth_topdown, error_chart


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        sys.exit(f"{args.out} is not empty; pass --overwrite to replace the results")
    (args.out / "visualizations").mkdir(parents=True, exist_ok=True)

    model = load_model_b()
    results = [measure_pair(model, i) for i in NOTEBOOK_PAIR_IDS]
    if not all(r["summary"]["depth_positive_both_cameras"] for r in results):
        sys.exit("a triangulated point is non-finite or behind a camera")

    write_csv(args.out / "corner_coordinates_and_depths.csv", [c for r in results for c in r["corners"]])
    write_csv(args.out / "edge_length_calculations.csv", [e for r in results for e in r["edges"]])
    write_csv(args.out / "pair_summary.csv", [r["summary"] for r in results])
    json.dump(dict(model="Model B: 30-pair stereo calibration", frame="left camera (iPhone Air), mm, X right, Y down, Z forward",
                   ground_truth_mm=dict(width=GT_WIDTH_MM, height=GT_HEIGHT_MM), baseline_mm=model["baseline_mm"],
                   pairs=[r["summary"] for r in results]), open(args.out / "measurement_summary.json", "w"), indent=2)

    vis = args.out / "visualizations"
    for r in results:
        cv2.imwrite(str(vis / f"pair_{r['pair']}_measurement.jpg"), annotated_pair(r), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(vis / "depth_topdown.png"), depth_topdown(results))
    cv2.imwrite(str(vis / "error_vs_ground_truth.png"), error_chart([r["summary"] for r in results]))

    print(f"Model B baseline {model['baseline_mm']:.2f} mm | truth {GT_WIDTH_MM:.0f} x {GT_HEIGHT_MM:.0f} mm")
    for r in results:
        s = r["summary"]
        print(f"pair {s['pair']}: mean Z {s['mean_Z_depth_mm']:.1f} mm | width {s['width_mm']:.1f} ({s['width_err_mm']:+.1f}) "
              f"| height {s['height_mm']:.1f} ({s['height_err_mm']:+.1f})")


if __name__ == "__main__":
    main()
