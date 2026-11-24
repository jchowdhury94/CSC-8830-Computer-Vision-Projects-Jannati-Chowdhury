# Module 7 – Stereo Vision

## 1. Overview

Module 7 uses a calibrated pair of cameras to measure real-world sizes from stereo images and stereo video. Both questions use **classical
computer vision only** (OpenCV and NumPy); no machine-learning or deep-learning model is used.

| Question | Task | Directory |
|---|---|---|
| **Question 1 – Calibrated Stereo Object Measurement** | Estimate the Z-depth, width and height of a rectangular object (a notebook) from two calibrated photographs. | [`object_measurement_question_1/`](object_measurement_question_1/) |
| **Question 3 – Room Measurement Using Stereo Videos** | Estimate a room's width, length and height from stereo videos, in two stages: Wall A (width and height) and Wall B (length). | [`room_measurement_question_3/`](room_measurement_question_3/) |

Both experiments use the same two phones, an iPhone Air (left camera) and an iPhone 17 Pro Max (right camera), and both rely on the shared
stereo calibration package [`stereo_measure/`](stereo_measure/).

## 2. Directory Structure

```text
Modules/Module_7/
├── README.md
├── .gitignore                          rules that keep the private photographs and recordings out of Git
├── stereo_measure/                     SHARED stereo calibration and measurement package
├── object_measurement_question_1/      Question 1
│   ├── calibration/                    model_b/ (stored calibration), corrected_corners/ (30 checkerboard pairs)
│   ├── selections/                     approved notebook corners (pair_1/2/3_corners.json)
│   ├── display/                        reduced photographs shown by the website
│   ├── outputs/                        saved measurement tables and visualizations
│   ├── reference/                      reference values used by verify_pipeline.py
│   ├── data/                           original photographs (private, not in Git)
│   └── run_calibration.py, run_measurement.py, verify_pipeline.py, make_display_images.py
└── room_measurement_question_3/        Question 3
    ├── room_measurement/               estimation, validation and visitor-measurement code (see its README)
    ├── visitor_workflow/               standalone visitor application (upload, frames, annotation, measurement)
    ├── f8_chunk_poc/                   chunked video upload, calibration worker and resource budget (used by the visitor application)
    ├── annotation_ui/                  boundary-annotation helper
    ├── calibration/room_video/         room-video calibration file (both phones, 2160 × 3840)
    ├── outputs/room_measurement/       accepted results, display images and the ten playback videos
    ├── data/                           original room recordings (private, not in Git)
    └── run_room_measurement.py, run_f8_integrated_app.py
```

**Shared package.** `stereo_measure/` holds the calibration, undistortion, triangulation and corner-measurement code. Question 1 uses it
directly, the Question 3 visitor workflow reuses its calibration routines, and Module 2's visitor calibration imports its single-photo
helpers. It therefore stays at the Module 7 root. Its paths (`stereo_measure/config.py`) point at `object_measurement_question_1/`.

Outside this directory: `Pages/app_module7.py` is the Streamlit page; `home_navigation.py` and `select_cards.py` provide the shared
navigation row and selection cards; `run_local.py` and `local_mode.py` are the local launcher; tests are in `tests/` (repository root) and in
the `tests/` folders of `room_measurement_question_3/`.

## 3. Using the Application

Open **Module 7 – Stereo Vision** (`/module-7`). Every page has a top row with a Back arrow, **Home** and a breadcrumb.

```text
Module 7
├── View My Experiments & Results
│   ├── Calibrated Stereo Object Measurement      the author's Question 1 experiment and results
│   └── Room Measurement Using Stereo Videos      the author's Question 3 experiment and results
└── Run Your Own Experiment
    ├── Calibrated Stereo Object Measurement      calibrate (or load a calibration) and measure an object in your own photographs
    └── Room Measurement Using Stereo Videos
        ├── Setup Instructions                    inline, collapsed by default: how to run the experiment on your computer
        └── Run Your Experiment (Local Only)      opens the standalone application (see section 7)
            └── Camera Calibration
                ├── Calibrate My Cameras          perform a new stereo camera calibration
                └── Upload Saved Calibration      use an existing camera calibration file
```

**Local-only requirement.** The public website shows the saved results of both experiments and the Setup Instructions, but it does not
process visitor videos: full-resolution (4K) stereo videos need more memory than the free hosting plan provides. **Run Your Experiment (Local
Only)** is therefore disabled on the public website and enabled only when the website is started with `python run_local.py` on your own
computer. It then opens the standalone room-measurement application **in the same browser tab**; that application's Back arrow, **Home** and
breadcrumb links return to the corresponding website pages.

**One workflow after calibration.** **Calibrate My Cameras** and **Upload Saved Calibration** are two ways of obtaining a calibration. Both lead
to the same review-and-accept step, and after a calibration is accepted both paths continue through exactly the same room-measurement workflow:
video upload, frame selection, wall-boundary annotation, Wall A, Wall B and the final results.

## 4. Question 1 – Calibrated Stereo Object Measurement

### 4.1 Objective and approach

Question 1 measures the Z-depth, width and height of a rectangular object from two photographs taken by a calibrated left/right camera pair.
It uses checkerboard stereo calibration, lens-distortion removal, linear triangulation (`cv2.triangulatePoints`) and 3D Euclidean edge
lengths. Nothing is rescaled to a known object size; the only metric scale comes from the checkerboard square size used in calibration.

Question 1 has two sides:

- **The author's saved experiment ("Model B").** An iPhone Air (left) and an iPhone 17 Pro Max (right) were calibrated from 30 checkerboard
  pairs (8 × 6 inner corners, 20 mm squares) and used to measure a 305 × 215 mm notebook (including the red spine) in three stereo pairs.
  It is shown under **View My Experiments & Results**.
- **An interactive visitor workflow** under **Run Your Own Experiment**, in which a visitor loads or computes their own stereo calibration and
  measures a rectangle in their own photographs. **Visitor measurements never use Model B**: without a loaded or computed calibration,
  measurement is unavailable.

### 4.2 Method (Model B)

1. **Calibration.** `cv2.calibrateCamera` per camera (k1, k2 only; `FIX_K3 | ZERO_TANGENT_DIST`), then `cv2.stereoCalibrateExtended` with
   `FIX_INTRINSIC` on all 30 pairs. This gives K and D for each camera and the stereo R and T (baseline 226.09 mm).
2. **Corners.** Four approved corners (top-left, top-right, bottom-right, bottom-left) per photograph, in original-resolution pixels
   (`selections/`).
3. **Triangulation.** Each pixel is undistorted with its camera's K and D; the two viewing rays are intersected with P1 = [I | 0] and
   P2 = [R | T] (no rectification), giving X, Y, Z in millimetres in the left-camera frame.
4. **Z-depth.** The Z coordinate of each triangulated corner (distance along the left optical axis).
5. **Size.** Euclidean edge lengths √(dX² + dY² + dZ²); width is the mean of the top and bottom edges, height the mean of the left and right
   edges.
6. **Validation.** Errors against 305 × 215 mm, used only for reporting.

### 4.3 Results (Model B)

| Pair | Mean Z (mm) | Width (mm) | Height (mm) | Width error | Height error |
|---|---|---|---|---|---|
| 1 | 789.4 | 301.9 | 212.0 | −3.1 mm (−1.01%) | −3.0 mm (−1.37%) |
| 2 | 779.0 | 300.5 | 208.9 | −4.5 mm (−1.49%) | −6.1 mm (−2.82%) |
| 3 | 767.0 | 299.9 | 213.1 | −5.1 mm (−1.66%) | −1.9 mm (−0.88%) |

Pairs 1 and 2 use the original approved corners; pair 3 uses a refined bottom-right corner. All points have positive depth in both cameras.
All six dimensions are shorter than the known size, by between 0.9% and 2.8%; the reasons were not isolated in this experiment. The saved
figures, tables, corner selections and the calibration reference are shown on the results page, which reads only files committed to Git
(`calibration/model_b/`, `selections/`, `outputs/`, `display/`, `reference/`) and does not need the original photographs.

### 4.4 Visitor workflow

The visitor workflow follows five steps; selecting a calibration option runs nothing by itself.

1. **Calibration.** *Load Saved Calibration* loads a `.npz` file exported earlier; *Calibrate New Camera Setup* computes one from checkerboard
   photographs. For a new calibration, enter the checkerboard's inner corners (3–40 each) and square size (0.5–1000 mm), upload 8–50 left/right
   pairs (JPEG or PNG; at least 8 valid pairs are needed, 15–30 are recommended, with the board tilted and at varied distances), and run the
   calibration. Photographs are analysed in the session only and are not saved.
2. **Object photographs.** Upload a left and a right photograph. An image size that differs from the calibration is an error; differing camera
   metadata is a warning.
3. **Corner selection.** Select the four corners in the order top-left, top-right, bottom-right, bottom-left on each image. Each click opens a
   zoomed view in which the marker is dragged to the exact corner; corners are mapped back to original-resolution pixels. Optional actual-size
   fields allow the result to report an error.
4. **Depth, width and height.** The app undoes each camera's lens distortion, triangulates in the left-camera frame and reports the corner
   Z-depths, the four edge lengths, the width and height, and the 3D coordinates. Non-blocking warnings flag a non-convex quadrilateral, very
   unequal opposite edges, a large epipolar distance, or depth outside the calibration's range.
5. **Results and reset.** *Measure Another Image* keeps the calibration; *Reset All* returns the visitor session to its first-visit state;
   *Clear Calibration* removes the calibration and everything that depends on it. None of these modify Model B or the saved experiment.

**Calibration quality checks.** A new calibration reports per-camera and stereo reprojection RMS, the epipolar residual, the baseline,
per-pair residuals with outlier flags, recovered board distances and the distortion model chosen by cross-validation. Warnings are heuristic
and never block the result.

**Calibration NPZ files.** A finished calibration can be downloaded as `.npz` (format defined in `stereo_measure/calibration_io.py`; at most
1 MB when loaded). The loader never unpickles data, checks every field's type, shape and range, and recomputes the derived geometry from K, D,
R and T. A file that passes validation is well-formed and self-consistent; that does not prove the calibration is accurate or authentic.

## 5. Question 3 – Room Measurement Using Stereo Videos

### 5.1 Objective and experimental setup

Question 3 is **one room-measurement experiment conducted in two stages**: **Wall A** estimates the room's width and height from two
viewpoints (four videos), and **Wall B** estimates its length from three viewpoints (six videos). Together the room was recorded from five
viewpoints, giving ten video recordings that form five stereo pairs. It is an approximate, educational experiment, not a precision
room-measurement system.

| | |
|---|---|
| Left camera | iPhone Air, 1× rear camera |
| Right camera | iPhone 17 Pro Max, 1× rear camera |
| Video | 2160 × 3840 (portrait), 30 fps |
| Physical baseline | 225 mm, held fixed in the estimator |
| Calibration | `room_measurement_question_3/calibration/room_video/room_video_recipe_a_calibration.json`: intrinsics and 5-term distortion of both phones at 2160 × 3840, fitted from checkerboard corners (8 × 6 inner corners, 20 mm squares) in calibration videos. The stereo extrinsics stored in that file are kept for provenance only; the rig pose is re-estimated from the wall videos. |

### 5.2 Methodology

The website's **How It Works — Methodology** section presents the same eleven steps with their equations.

1. **Stereo Video Acquisition and Frame Selection.** Frames are extracted from the recordings; each frame set holds a left and a right image
   for every viewpoint. *Wall A:* five frame sets from different parts of the videos, chosen for sharpness with the Laplacian-variance measure.
   *Wall B:* frames 60, 150 and 240 of all six videos, giving three frame sets.
2. **Camera Model and Calibration.** Both cameras are calibrated from checkerboard videos (focal lengths, principal points, lens distortion);
   the cameras' relative pose is a rotation R and translation T with the measured 225 mm baseline, under the projection model s·x̃ = K(RX + T).
3. **SIFT Feature Detection and Matching.** SIFT features are matched with Lowe's ratio test and mutual matching. *Wall A:* ratio 0.75 between
   stereo images and 0.85 between viewpoints, giving 805 four-image feature tracks. *Wall B:* ratio 0.75 between stereo images and 0.90 between
   neighbouring viewpoints.
4. **Epipolar Geometry and Stereo Pose Estimation.** The essential matrix E = [T]ₓR relates corresponding points. An eight-point estimate is
   refined with Sampson distance and a Tukey biweight, with the baseline fixed. *Wall A:* one shared stereo pose is estimated and refined from
   18,600 stereo matches and the four-view tracks (robust residual scale 0.27 px, 95.4% inliers). *Wall B:* a separate stereo pose is estimated
   for each of the three viewpoints.
5. **3D Triangulation.** Matched points are triangulated with the Direct Linear Transform; only points with depth between 0.8 m and 20 m and
   reprojection and epipolar errors below 1.5 pixels are kept. These points support the floor plane and the registration; they are not used
   directly as wall corners.
6. **Floor-Plane Estimation.** Reconstructed points near the bottom of each image give an initial plane by principal component analysis,
   refined by iteratively reweighted least squares: nᵀX + d = 0 with ‖n‖ = 1.
7. **Wall-Boundary Annotation and Line Fitting.** *Wall A:* approximate vertical wall edges and ceiling edges are marked manually; image
   gradients and robust line fitting refine them to sub-pixel precision. *Wall B:* two points are marked manually along each required vertical
   wall edge and define the boundary lines directly; the middle viewpoint is used for registration and needs no annotations.
8. **Geometric Corner Estimation.** Each boundary line is back-projected to a plane; the intersection of the left- and right-camera planes
   gives the vertical wall edge, and its intersection with the floor plane gives the bottom corner. *Wall A:* bottom and top corners are
   estimated (the top corner from the viewing ray through the ceiling intersection and the vertical edge). *Wall B:* only the two bottom end
   corners are needed. Partially hidden corners are thereby estimated without being seen in both cameras.
9. **Multi-View Registration Using RANSAC.** Each viewpoint's points are in their own coordinate system. RANSAC (1,500 iterations, three-point
   samples, 40 mm inlier threshold) finds reliable matching 3D points and singular value decomposition gives the rotation and translation that
   align the viewpoints. *Wall A:* viewpoint 1 is aligned with viewpoint 2. *Wall B:* viewpoints 1 and 3 are aligned with viewpoint 2.
10. **Room Dimension Estimation.** In the common frame, dimensions are Euclidean distances. *Wall A:* the width is the distance between the two
    registered bottom corners and the height is the mean of the two corner heights. *Wall B:* the length is the distance between the two end
    corners after alignment with the middle viewpoint. Final values average the successful frame sets.
11. **Measurement Uncertainty and Consistency.** *Wall A:* four independent sources (frame selection and matching, the pose-estimation
    method, boundary placement and, as an assumed perturbation, the camera intrinsics) are combined in quadrature, σ = √(Σσₖ²), and reported as
    an approximate 95% interval, estimate ± 1.96σ. *Wall B:* the three frame sets show frame-to-frame consistency (a spread of about 2.1 cm);
    because they share one set of annotations this is not a measurement uncertainty, and **no confidence interval is calculated for Wall B**.

The estimation code is in `room_measurement_question_3/room_measurement/` (see its [README](room_measurement_question_3/room_measurement/README.md)).
No measured room dimension or camera height enters any estimation step; the tape measurements are compared with the estimates only afterwards.

### 5.3 Results

The estimates were written first and compared with the tape-measured dimensions afterwards. The actual width, length and height were measured
with a tape measure.

| Dimension | Estimated | Actual | Absolute error | Percentage error |
|---|---:|---:|---:|---:|
| Width (Wall A) | 287.96 cm | 290.00 cm | 2.04 cm | 0.70% |
| Length (Wall B) | 380.23 cm | 393.70 cm | 13.47 cm | 3.42% |
| Height (Wall A) | 238.02 cm | 240.00 cm | 1.98 cm | 0.82% |

Overall (n = 3): **MAE 5.83 cm, MAPE 1.65%, RMSE 7.95 cm.** The approximate room dimensions are **288 × 380 × 238 cm** (width × length × height).
All three estimates are slightly shorter than the tape measurements.

- **Wall A uncertainty.** The approximate 95% intervals are 265.6–310.3 cm for the width and 218.8–257.3 cm for the height, and both tape
  measurements lie inside them. The camera-intrinsics term dominates (1σ of 10.7 cm in width and 9.4 cm in height, against 1.9, 1.9 and 3.0 cm
  for the three other width sources and 1.7, 1.6 and 1.7 cm for height); its perturbation sizes were assumed, not measured. Systematic annotation
  errors, additional lens-model errors and camera movement between unsynchronised frames are not included.
- **Wall B consistency.** The three frame sets (frames 60, 150 and 240) gave 379.09, 381.20 and 380.40 cm (mean 380.23 cm). The saved values
  are the regression references of the test suite, which reproduces them to within ±5 mm per frame set; that tolerance measures algorithmic
  reproducibility and is not a claim about accuracy in the room.

### 5.4 Limitations

- The left (iPhone Air) camera cannot see the ceiling corners, so the top corners are inferred from the right camera's rays only; one bottom
  corner is partly hidden and is inferred from the stereo-derived vertical line and the floor plane. The inference assumes vertical walls and a
  planar floor.
- The camera intrinsics dominate the uncertainty; an alternative calibration fit of the same cameras moved the Wall A width by about −9%
  (recorded in `estimation_result.json`). The relative stereo yaw is only weakly observable from epipolar geometry, so a low epipolar residual
  does not by itself prove a correct metric reconstruction.
- Results depend on boundary-annotation precision, and the frame sets of one rig placement are correlated captures, not independent
  experiments.
- Accuracy was validated on one room with one pair of phones. Other recordings may give inaccurate results or be refused with an explanation.

### 5.5 Visitor-Workflow Validation (Separate from the Accepted Results)

This subsection is **not part of the accepted experimental results** above. The author ran both walls once through the standalone visitor
application on the same recordings and the room-video calibration, entering the boundaries manually through the application's frame view. That
single, manually recorded run gave about 285.4 × 235.6 cm for Wall A and 377.99, 388.79 and 383.92 cm (mean 383.6 cm) for the Wall B frame
sets, against 287.96 × 238.02 cm and 379.09, 381.20 and 380.40 cm (mean 380.23 cm) in the accepted results. The run was not scripted, so
**these application figures cannot be independently reproduced from the tracked files**; they are recorded only as a transparency check of the
visitor workflow.

Replaying the saved Wall B annotation coordinates through the same code reproduces the accepted Wall B values (the opt-in real-recording test in Section 8 checks this to within ±5 mm per frame set). Boundary-annotation precision is a plausible contributor to the differences observed in the manually recorded visitor run. However, the exact cause and magnitude of those differences were not independently established.

### 5.6 Visitor workflow (local only)

The standalone application runs on your own computer, after you accept a calibration (either option in section 3):

1. **Videos.** Upload four videos for Wall A (two viewpoints) or six for Wall B (three viewpoints), one per left/right camera slot. Supported:
   MP4/M4V, MOV, Matroska/WebM or AVI (checked by content); 1–120 s and 1–480 fps; at most 1024 MB each (`F8_ROOM_MAX_FILE_MB` can change the
   limit, up to 2048). Videos must have exactly the frame size and orientation of the accepted calibration; they are never resized or rotated.
2. **Frames and boundaries.** Choose the frame sets and mark the wall boundaries on them.
3. **Calculate.** *Calculate Wall Dimensions* measures Wall A; *Continue to Wall B — Room Length* and *Calculate Room Length* measure Wall B. The
   final page shows the combined room dimensions only when both walls have an accepted result.

Recording advice (from the author's experiment, not enforced): the same cameras and settings as in calibration (resolution, orientation, frame
rate, zoom, focus, stabilisation), a fixed rig, a still room, a wall with visible corners and a floor with visible texture.

**Resources.** Each measurement runs in a separate worker, stopped after 300 s of run time or 1,800 CPU-seconds, and it starts only if enough
memory is free (at least 1,024 MB for two viewpoints, 2,450 MB for three). Keep about 4 GB free; the Wall A calculation used about 2.0 GB at its
peak on the author's recordings. The memory budget can be capped with `F8_APP_MEMORY_CEILING_MB`. Extracted frames are kept in a marked
temporary folder (roughly 60–100 MB per measurement) and removed by **New Measurement**, when the session ends or the application stops, or at
the next start after a crash. Computing a calibration from your own checkerboard videos (**Calibrate My Cameras**) is disabled unless you
start the application with an explicit memory limit, for example
`python Modules/Module_7/room_measurement_question_3/run_f8_integrated_app.py --enable-full-calibration --max-tree-rss-mb=3000`
(256–65536, in the `=` form); **Upload Saved Calibration** needs no such option.

## 6. Experimental Data and Privacy

| In Git (needed to run the website and the default checks) | Private, kept locally and excluded by `.gitignore` |
|---|---|
| `object_measurement_question_1/calibration/`, `selections/`, `display/`, `outputs/`, `reference/` | `object_measurement_question_1/data/`: the 60 original checkerboard photographs and the 6 original notebook photographs |
| `room_measurement_question_3/calibration/room_video/` | `room_measurement_question_3/data/`: the original 2160 × 3840 room recordings and the hand-made annotation files |
| `room_measurement_question_3/outputs/room_measurement/`: accepted results, display images and the ten 720 × 1280 playback videos (about 10 MB) | `room_measurement_question_3/outputs/stage_*/` and similar intermediate evidence |
| Source code, reproducibility scripts and regression tests required for the published experiments and application | Caches, virtual environments, operating-system metadata |

The original photographs and recordings are required only to reproduce the experiments from scratch (section 8). The website and the visitor
workflow run without them; the visitor workflow uses the visitor's own videos and keeps photographs and calibrations in the session only.

## 7. Installation and Local Execution

All commands run from the repository root. Python environment used for development and testing: **Python 3.13.7 on macOS (Apple silicon)**,
in a virtual environment.

```bash
git clone https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury.git
cd CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury
python -m venv .venv                       # Windows PowerShell: py -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r Modules/Module_7/room_measurement_question_3/visitor_workflow/requirements.txt
python run_local.py                        # website http://127.0.0.1:8501, room-measurement application http://127.0.0.1:8640
```

`run_local.py` starts two separate Streamlit applications, both bound to `127.0.0.1`: the website (`home.py`) and the room-measurement
application (`room_measurement_question_3/run_f8_integrated_app.py`). Then open the website, choose **Module 7 > Run Your Own Experiment >
Room Measurement Using Stereo Videos** and press **Run Your Experiment (Local Only)**. Ctrl+C stops both. If a port is busy nothing is
started and the port is named (`--website-port` and `--visitor-port` choose others). Local mode is enabled only by this launcher
(`CSC8830_LOCAL_MODE=1`), so `python -m streamlit run home.py` alone behaves like the public website, with the card disabled.

**Requirements files.**

| File | For |
|---|---|
| `requirements.txt` (repository root) | The whole website, including the other modules (Module 4 adds PyTorch and SAM 2). |
| `Modules/Module_7/room_measurement_question_3/visitor_workflow/requirements.txt` | The standalone room-measurement application and the Module 7 tests: Streamlit 1.64.0, NumPy 2.5.3, OpenCV (headless) 5.0.0.93, pandas 3.0.5, Pillow 12.3.0, streamlit-image-coordinates 0.4.1 and their pinned support packages; classical computer vision only. |

**Python versions.** The tests and the website were run with Python 3.13.7. `.python-version` lists 3.12 and the root `requirements.txt`
mentions selecting 3.12 for Streamlit Community Cloud, but this repository has not been tested with those versions, so only 3.13.7 is
verified. The Python version of the public deployment is a setting in Streamlit Community Cloud and is not recorded in the repository.

**Troubleshooting.** *Run Your Experiment (Local Only) is disabled*: you are on the public website, or the application is not running; run
`python run_local.py` and press **Check again**. *A video is rejected*: the message names the failed check (size, length, frame rate, or a size
or orientation different from the calibration). *Not enough memory*: close other large programs so about 4 GB is free, then press **Try again**.

## 8. Reproducibility

The saved results are tracked in Git and the website displays them without recomputation. Reproducing an experiment from scratch needs the
**private original photographs and recordings**, which are intentionally excluded from GitHub. Without them, only the checks that use tracked
files can run. Commands:

**Question 1** (from `Modules/Module_7/object_measurement_question_1/`). To reproduce Question 1 with your own photographs, first create the
folders `data/calibration/left/`, `data/calibration/right/`, `data/object_measurement/left/` and `data/object_measurement/right/` under that
directory and add the photographs there, named `air_N.jpeg` / `pro_N.jpeg` (calibration, N = 1–30) and `air_imageN.jpeg` / `pro_imageN.jpeg`
(notebook, N = 1–3); they are not in Git:

```bash
python verify_pipeline.py                 # checks the stored calibration (SHA-256), the saved corner selections and display images, and
                                          # that each notebook pair reproduces the saved depths, edge lengths, width and height; uses tracked files
python verify_pipeline.py --full          # additionally requires the 60 + 6 original photographs
python run_calibration.py                 # re-derives Model B from the 30 corrected-corner pairs and compares it with the stored model (~1e-6)
python run_measurement.py --out DIR       # re-measures the three pairs into DIR (refuses to write into a non-empty DIR without --overwrite)
python make_display_images.py             # rebuilds display/ from the original photographs
```

**Question 3, Wall A** (needs the private wall recordings in `room_measurement_question_3/data/room_measurement/calibration_videos/`):

```bash
python Modules/Module_7/room_measurement_question_3/run_room_measurement.py --out DIR            # full run including uncertainty
python Modules/Module_7/room_measurement_question_3/run_room_measurement.py --out DIR --quick    # fewer uncertainty replicates
```

It writes `estimation_result.json` first and only then compares with the measured size (`validation.json`). Existing results are not
overwritten without `--overwrite`.

**Question 3, Wall B** (needs the private recordings and annotation file in `room_measurement_question_3/data/room_measurement/wall_b_three_viewpoints/`).
Wall B is reproduced by the opt-in regression tests, which replay the saved annotations through the production code path and compare each
frame set with the accepted values:

```bash
M7_REAL_VIDEO_TESTS=1 python -m pytest \
  Modules/Module_7/room_measurement_question_3/visitor_workflow/tests/test_f8_35_real_three_view.py \
  Modules/Module_7/room_measurement_question_3/room_measurement/tests/test_three_view_real_data.py
```

## 9. Testing

The tests use pytest (several rely on its fixtures). From the repository root, with the visitor requirements installed plus `pytest`:

```bash
python -m pip install pytest
python -m pytest tests                                                          # website and Module 7 page tests
python -m pytest Modules/Module_7/room_measurement_question_3/visitor_workflow/tests
python -m pytest Modules/Module_7/room_measurement_question_3/room_measurement/tests
python -m pytest Modules/Module_7/room_measurement_question_3/f8_chunk_poc/tests
```

The suites cover Question 1 (calibration, NPZ input/output, the wizard, corner mapping and dragging, measurement logic, reset behaviour and the
Streamlit page), Question 3 (the visitor workflow, workers, memory budget, temporary-file cleanup, estimation code on synthetic data), the main-website navigation and the launcher. Tests that need the private photographs or recordings are skipped when those
files are absent. The expensive real-recording tests run only with `M7_REAL_VIDEO_TESTS=1`. A few integrity tests use `git status` to confirm that the
validated algorithm sources are unchanged, so they require a Git checkout with a clean working tree.

## 10. Deployment and Limitations

- **Public website.** The main site (`home.py`, with `Pages/app_module7.py`) is deployed on Streamlit Community Cloud at
  <https://jannati-chowdhury-computer-vision-projects.streamlit.app/> using the root `requirements.txt`. It runs entirely from files committed to
  Git, keeps visitor photographs and calibrations in the session only, and needs none of the private data. It shows both experiments' results,
  playback videos and visualizations, and the Setup Instructions.
- **Question 3 visitor workflow.** It is intentionally local-only and is not part of the public deployment (section 3).
- **Scientific limitations.** See sections 4.3 and 5.4: the experiments are approximate and educational, were validated on one setup, and the
  reported uncertainty covers only the sources listed in section 5.3.
