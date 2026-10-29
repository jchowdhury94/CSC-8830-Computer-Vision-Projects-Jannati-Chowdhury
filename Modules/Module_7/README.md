# Module 7 – Calibrated stereo measurement of a notebook (Model B)

Two phones (left: iPhone Air, right: iPhone 17 Pro Max) are calibrated as a stereo pair from 30 checkerboard photo pairs
(8×6 internal corners, 20 mm squares). The calibrated pair then measures a notebook (305 × 215 mm including the red spine) from
three stereo photographs. Classical computer vision only (OpenCV); nothing is rescaled to match the known size.

## Method
1. **Calibration (Model B)** – `cv2.calibrateCamera` per camera (k1, k2 only; `FIX_K3 | ZERO_TANGENT_DIST`), then
   `cv2.stereoCalibrateExtended` with `FIX_INTRINSIC` on all 30 pairs → K, D for each camera and the stereo R, T (baseline 226.09 mm).
2. **Corners** – four approved corners (TL, TR, BR, BL) per photo, original-resolution pixels (`selections/`).
3. **Triangulation** – each pixel is undistorted with its own K, D; the two viewing rays are intersected with
   P1 = [I | 0], P2 = [R | T] (`cv2.triangulatePoints`, no rectification) → X, Y, Z in mm in the left-camera frame.
4. **Z-depth** – the Z coordinate of each triangulated corner (distance along the left optical axis). It is an output of the intersection.
5. **Size** – 3D Euclidean edge lengths √(dX²+dY²+dZ²); width = mean of top and bottom edges, height = mean of left and right edges.
6. **Validation** – errors against 305 × 215 mm, used only for reporting.

## Final results (Model B)
| Pair | Mean Z (mm) | Width (mm) | Height (mm) | Width err | Height err |
|---|---|---|---|---|---|
| 1 | 789.4 | 301.9 | 212.0 | −3.1 mm (−1.01%) | −3.0 mm (−1.37%) |
| 2 | 779.0 | 300.5 | 208.9 | −4.5 mm (−1.49%) | −6.1 mm (−2.82%) |
| 3 | 767.0 | 299.9 | 213.1 | −5.1 mm (−1.66%) | −1.9 mm (−0.88%) |

Pairs 1 and 2 use the original approved corners; pair 3 uses the refined bottom-right corner (Stage 10G). All points have positive depth in both cameras.
The measurements run about 1–3% short; the reasons were not isolated here.

## Layout
```
stereo_measure/        calibration.py  triangulation.py  measurement.py  visualization.py  config.py
run_measurement.py     triangulate the 3 notebook pairs with Model B → outputs/
run_calibration.py     re-derive Model B from the corrected corners and compare with the stored model
verify_pipeline.py     check everything against reference/expected_model_b_results.json
calibration/model_b/   stereo_calibration_30pairs.npz (the stored Model B parameters)
calibration/corrected_corners/   {left,right}_NN_corners.csv for pairs 1–30
selections/            final approved notebook corners (pair_1/2/3_corners.json)
reference/             saved results used for verification
data/calibration/      30 original calibration photo pairs (air_N.jpeg / pro_N.jpeg)
data/object_measurement/   3 original notebook photo pairs
outputs/               CSV/JSON results and visualizations
```

## Run
```
python run_measurement.py --overwrite   # results and figures
python run_calibration.py               # reproducibility check (differences ~1e-6, floating-point level)
python verify_pipeline.py               # all checks
```
Requires `numpy` and `opencv-python` (see the repository `requirements.txt`).
