# Module 7 – Calibrated Stereo Measurement

Module 7 measures the Z-depth, width and height of a rectangular object from two photographs taken by a calibrated left/right
camera pair. It uses classical computer vision only (OpenCV): checkerboard stereo calibration, undistortion, linear triangulation
(`cv2.triangulatePoints`) and 3D Euclidean edge lengths. Nothing is rescaled to a known object size; the only metric scale comes from
the checkerboard square size entered during calibration.

The module has two sides:

- **An interactive Streamlit page** (`Pages/app_module7.py`) in which a visitor loads or computes their own stereo calibration and
  measures a rectangle in their own photographs.
- **The author's saved experiment (Model B):** an iPhone Air (left) and an iPhone 17 Pro Max (right) calibrated from 30
  checkerboard pairs (8×6 inner corners, 20 mm squares) and used to measure a 305 × 215 mm notebook (including the red spine) in three
  stereo pairs. Its method, results and verification scripts are described below.

**Visitor measurements never use Model B.** The app does not fall back to the author's calibration: without a loaded or computed
visitor calibration, measurement is unavailable. Model B is used only by the saved-experiment view, the command-line scripts and
regression tests.

## Using the app

The page is registered in `home.py` as **Module 7 - Stereo Vision** (`/module-7`). Its home screen offers two parts:

| Part | Status |
|---|---|
| **Part 1 – Calibrated Stereo Object Measurement** | Implemented (this document). |
| **Part 3 – Room Measurement from Video** | Placeholder only; no implementation yet. |

A **Module 7 Home** button returns to the part selector. Part 1 is organised as calibration, then object photographs, then corner
selection, then results.

### 1. Calibration: two workflows

Two large buttons choose how to obtain a calibration; selecting one runs nothing.

- **Load Saved Calibration** – load a `.npz` file exported earlier (see [Calibration NPZ files](#calibration-npz-files)).
- **Calibrate New Camera Setup** – compute a calibration from checkerboard photographs.

**Calibrate New Camera Setup**

1. *Checkerboard settings.* Enter the **inner corners** (columns and rows; the points where four squares meet, so a board of 9 × 7
   squares has 8 × 6 inner corners; 3 to 40 each) and the **square size in mm** (0.5 to 1000). The square size sets the metric scale of
   every later 3D measurement. Changing the settings clears photos analysed under the old ones, with a warning.
2. *Upload stereo pairs.* Choose the number of pairs (8 to 50, default 8) and upload a left and a right JPEG or PNG for each. Each photo is
   analysed as it is uploaded and the board is detected and refined to sub-pixel accuracy. A pair can be removed (**Remove … photo**,
   **Remove all calibration photos**) or excluded from, and later included again in, the calibration. Lowering the pair count hides the
   last slots but keeps their photos.
3. *Run.* **Run Stereo Calibration** needs at least **8 valid pairs**. Guidance in the app recommends 15–30 pairs, with the board tilted,
   at varied distances, and covering the corners and edges of the frame.

Photographs are analysed within the session only and are not saved.

**Photo validation.** Each file must be a decodable JPEG or PNG of at most 30 MB and 50 megapixels, with both sides at least 480 px.
EXIF orientation is applied. Board detection must give a regular grid: corner rows and columns must be near-straight, sub-pixel
refinement shifts must be small, and the corner ordering must be unambiguous. A board entered with swapped rows and columns is
reported as such. Left and right corner orderings are matched per pair using the epipolar geometry, and pairs that cannot be matched
confidently are flagged rather than silently used.

**Quality checks.** After calibration the app shows the per-camera and stereo reprojection RMS, the epipolar residual, the baseline,
per-pair residuals with outlier flags, the recovered board distances, and, for each camera, the distortion model chosen by
cross-validation (`radial2`, `radial3` or `full`, simplest model within 10% of the best held-out error). Warnings are heuristic and
never block the result. They cover RMS or epipolar errors that are large for the image resolution, an implausible principal point or
fx/fy ratio, a baseline below 1% of the board distance, reconstructed square sizes more than 2% from the entered size, too few tilted
views, poor frame coverage, little variation in distance, and fewer than 15 pairs.

### Calibration NPZ files

A finished calibration can be saved with **Download Calibration (.npz)** and loaded later with **Load Saved Calibration (.npz)**. The
file (format version 1, defined in `stereo_measure/calibration_io.py`) holds the intrinsics, distortion coefficients, `R`, `T`, image
sizes, checkerboard settings, quality metrics, warnings and camera metadata. It is small (the loader accepts at most 1 MB).

The loader is deliberately strict:

- It never unpickles data (`allow_pickle=False`) and inspects the zip container before reading it.
- Every field must have the expected dtype, shape and range, and distortion coefficients must obey their declared model.
- Derived geometry (baseline, essential and fundamental matrices) is **recomputed** from `K`, `D`, `R`, `T`; a stored baseline that
  disagrees with `|T|` is rejected.

**Limitation:** a file that passes validation is well-formed and self-consistent. That does not prove the calibration is accurate or
authentic, and a loaded calibration is marked as not authenticity-verified. A loaded file also does not carry the original photographs,
so the per-pair views of a freshly computed calibration are not available for it.

### 2. Object photographs

With a calibration active, upload a left and a right photograph of the object (JPEG or PNG). Each is checked against the calibration:

- An **image size** that differs from the calibration is an error, because the intrinsics are valid only at the calibrated resolution.
- **Camera metadata** (model, lens, focal length, orientation, from EXIF) that differs from the calibration is a *warning*. Matching
  metadata does not prove the physical rig is unchanged, and swapped left/right photos are flagged when the metadata suggests them.

### 3. Corner selection and refinement

Select the four object corners in this order on **each** image: **top-left, top-right, bottom-right, bottom-left**. Start with the left
image, then the right. Once the photographs are accepted, an instruction dialog for the current side explains the order; **Start Corner Selection** in it begins the selection (reopen it with **Show corner-selection instructions**). The optional **Actual Size** fields (width and height in mm) let a visitor enter the object's real size so the results also show the error against it; if left blank, only the estimate is reported.

1. Click near the corner on the photograph.
2. A **zoomed view** of that region opens. Drag the marker to the exact corner; releasing the pointer records the position, mapped back to
   original-resolution pixel coordinates (`stereo_measure/corner_drag.py` and its tiny static frontend in `corner_drag_frontend/`).
3. **Accept Corner** keeps it; **Cancel** discards the pending corner and any unaccepted drag. **Undo last point**, **Reset left**,
   **Reset right** and **Reset both** correct earlier choices.
4. After four accepted corners, review the markers and press **Confirm Left Image Selections**, then repeat for the right image and press
   **Confirm Right Image Selections**. Accepting a left corner again withdraws earlier confirmations.

### 4. Depth, width and height

**Calculate Dimensions** becomes available after both sides are confirmed. For each corner pair the app undoes each camera's lens
distortion, triangulates in the left-camera frame (X right, Y down, Z forward, in mm), and reports:

- the **Z-depth** of each corner: its Z coordinate, i.e. the distance along the left optical axis (not the straight-line distance);
- the four **3D edge lengths** √(dX² + dY² + dZ²), the **width** (mean of top and bottom edges) and the **height** (mean of left and
  right edges);
- the reconstructed 3D coordinates (left-camera frame) and annotated stereo images, plus the errors against the optional actual size.

Non-blocking warnings flag a non-convex quadrilateral, very unequal opposite edges (ratio above 1.25), a large epipolar distance, or depth
far outside the range of the calibration's board distances. Their thresholds scale with the calibration's own residuals and image size,
and are heuristics, not guarantees. The size reported is only as accurate as the calibration, the corner picks and the camera geometry.

### 5. Results, Measure Another Image and Reset All

- **Measure Another Image** discards the current object photographs, corners and results but **keeps the active calibration**.
- **Reset All** returns the whole visitor workflow (calibration photos, settings, loaded or computed calibration, uploads, selections) to
  its first-visit state. It affects the visitor session only; Model B, the saved selections and the experiment view are never modified.
- **Clear Calibration** removes the active calibration and the uploads, selections and results that depend on it.

### 6. My Experiment & Results

The **View My Experiment & Results** button shows the author's saved Model B experiment: its setup, calibration quality, the saved
corner selections, depths, edge lengths, errors against the known 305 × 215 mm size, and figures (with **Notebook stereo pair** 1–3 to
choose from and **Close experiment** to return). It reads only files that are committed to Git –
`calibration/model_b/`, `selections/`, `outputs/`, `display/` and `reference/` – and does **not** need the original photographs, so it
works on Streamlit Community Cloud.

## Model B method

1. **Calibration** – `cv2.calibrateCamera` per camera (k1, k2 only; `FIX_K3 | ZERO_TANGENT_DIST`), then
   `cv2.stereoCalibrateExtended` with `FIX_INTRINSIC` on all 30 pairs → K, D for each camera and the stereo R, T (baseline 226.09 mm).
2. **Corners** – four approved corners (TL, TR, BR, BL) per photo, original-resolution pixels (`selections/`).
3. **Triangulation** – each pixel is undistorted with its own K, D; the two viewing rays are intersected with
   P1 = [I | 0], P2 = [R | T] (`cv2.triangulatePoints`, no rectification) → X, Y, Z in mm in the left-camera frame.
4. **Z-depth** – the Z coordinate of each triangulated corner (distance along the left optical axis). It is an output of the intersection.
5. **Size** – 3D Euclidean edge lengths √(dX²+dY²+dZ²); width = mean of top and bottom edges, height = mean of left and right edges.
6. **Validation** – errors against 305 × 215 mm, used only for reporting.

The visitor workflow applies the same triangulation and size computation to the visitor's own calibration.

## Final results (Model B)

| Pair | Mean Z (mm) | Width (mm) | Height (mm) | Width err | Height err |
|---|---|---|---|---|---|
| 1 | 789.4 | 301.9 | 212.0 | −3.1 mm (−1.01%) | −3.0 mm (−1.37%) |
| 2 | 779.0 | 300.5 | 208.9 | −4.5 mm (−1.49%) | −6.1 mm (−2.82%) |
| 3 | 767.0 | 299.9 | 213.1 | −5.1 mm (−1.66%) | −1.9 mm (−0.88%) |

Pairs 1 and 2 use the original approved corners; pair 3 uses a refined bottom-right corner. All points have positive depth in both cameras.
The measurements run about 1–3% short; the reasons were not isolated here.

## Project structure

```text
Modules/Module_7/
├── README.md
├── stereo_measure/                  package used by the app, the scripts and the tests
│   ├── config.py                    paths and constants (pattern, square size, image size, pair ids)
│   ├── calibration.py               Model B calibration from the corrected corners
│   ├── triangulation.py             undistortion, triangulation, right-camera coordinates
│   ├── measurement.py               notebook measurement for the saved pairs (Model B)
│   ├── live.py                      measurement from user-supplied corners with any calibration, warning thresholds
│   ├── user_calibration.py          visitor calibration: photo validation, board detection, pair matching, quality checks
│   ├── calibration_wizard.py        pure-Python state and rules behind the calibration wizard
│   ├── calibration_io.py            validated NPZ export and import
│   ├── selection.py                 upload checks, coordinate mapping, zoom windows, overlays
│   ├── corner_drag.py               draggable-marker Streamlit component (Python side)
│   ├── corner_drag_frontend/        index.html for that component (no build step)
│   └── visualization.py             figures for the saved results
├── run_measurement.py               measure the 3 notebook pairs with Model B → outputs/
├── run_calibration.py               re-derive Model B from the corrected corners and compare with the stored model
├── verify_pipeline.py               check everything against reference/expected_model_b_results.json
├── make_display_images.py           create the reduced display/ images from the original photographs
├── calibration/
│   ├── model_b/                     stereo_calibration_30pairs.npz (stored Model B parameters)
│   └── corrected_corners/           {left,right}_NN_corners.csv for pairs 1–30
├── selections/                      final approved notebook corners (pair_1/2/3_corners.json)
├── reference/                       saved results used for verification
├── display/                         reduced images shown by the app (committed)
├── outputs/                         CSV/JSON results and visualizations
└── data/                            original photographs (git-ignored; see below)
    ├── calibration/{left,right}/        air_N.jpeg / pro_N.jpeg, N = 1–30
    └── object_measurement/{left,right}/ air_imageN.jpeg / pro_imageN.jpeg, N = 1–3
```

The page itself is `Pages/app_module7.py`; the tests are `tests/test_module7_*.py` in the repository root.

## Tracked and ignored data

| Kept in Git (needed to run the app and the checks) | Git-ignored (kept locally) |
|---|---|
| `calibration/model_b/`, `calibration/corrected_corners/` | `data/calibration/{left,right}/*.jpeg` (60 original checkerboard photographs) |
| `selections/`, `reference/`, `outputs/` | `data/object_measurement/{left,right}/*.jpeg` (6 original notebook photographs, about 22 MB) |
| `display/` (reduced images used by the app) | `results/corner_detection/` and `results/stage1_5_validation/` images |
| `.gitkeep` placeholders in `data/calibration/{left,right}/` | |

The deployed app and the default checks need only the left-hand column. The original photographs are required only to regenerate
`display/` (`make_display_images.py`), to re-run board detection from scratch, and for the tests marked as needing them.

## Running the scripts

From `Modules/Module_7/` (use the repository's Python 3.12 environment):

```bash
python run_measurement.py --overwrite       # re-measure the 3 pairs; rewrites outputs/ (refuses to overwrite without --overwrite)
python run_calibration.py                   # reproducibility check against the stored Model B (differences ~1e-6, floating-point level)
python run_calibration.py --save PATH.npz   # also write the fresh calibration
python verify_pipeline.py                   # all checks, using only tracked files
python verify_pipeline.py --skip-calibration  # as above but without recomputing the calibration (faster)
python verify_pipeline.py --full            # local use: additionally require the 60 + 6 original photographs
```

`verify_pipeline.py` checks that the Model B file is unchanged (SHA-256), that the 30 corrected-corner pairs, selections and display
images are present, that each notebook pair reproduces the saved depths, edge lengths, width and height, that the saved `outputs/`
tables match a fresh measurement, and (unless skipped) that recalibrating from the corrected corners reproduces Model B. Missing
original photographs are reported as `SKIP` and never fail the default run; `--full` turns them into required checks. The exit code is 1
if any check fails.

## Testing

From the repository root:

```bash
python -m unittest discover -s tests
```

`pytest tests` also works if pytest is installed. The suite covers calibration, NPZ I/O, the wizard, corner mapping and dragging, the
measurement logic, reset behaviour and the Streamlit page (via `streamlit.testing`). Tests that need the git-ignored original
photographs are skipped, with a message, when those files are absent, so the suite runs from a fresh clone of the tracked files and runs
in full when the photographs are present.

## Python and deployment

Use Python 3.12 (see `.python-version` and the comments in the root `requirements.txt`; select Python 3.12 in the Streamlit Community
Cloud app settings). From the repository root:

```bash
SAM2_BUILD_CUDA=0 python -m pip install -r requirements.txt
python -m streamlit run home.py
```

Module 7 itself needs `numpy`, `opencv-python-headless`, `pandas`, `Pillow`, `streamlit` and `streamlit-image-coordinates`, all in the
root `requirements.txt` (the other packages there serve Module 4). The deployed app runs entirely from files committed to Git; it does not
need the original photographs and keeps visitor photographs and calibrations in the session only (nothing is stored by the app unless the visitor downloads a calibration `.npz`).
