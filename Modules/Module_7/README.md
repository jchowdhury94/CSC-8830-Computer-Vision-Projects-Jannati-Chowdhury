# Module 7 – Stereo Vision

Module 7 has two questions, both implemented with classical computer vision only (no machine learning or deep learning):

- **Question 1 – Calibrated Stereo Object Measurement** (this section and the sections up to [Testing](#testing)).
- **Question 3 – Room Measurement Using Stereo Videos** (see [Question 3](#question-3--room-measurement-using-stereo-videos) and
  [Run the visitor application locally](#run-the-visitor-application-locally)).

## Question 1 – Calibrated Stereo Object Measurement

Question 1 measures the Z-depth, width and height of a rectangular object from two photographs taken by a calibrated left/right
camera pair. It uses OpenCV: checkerboard stereo calibration, undistortion, linear triangulation
(`cv2.triangulatePoints`) and 3D Euclidean edge lengths. Nothing is rescaled to a known object size; the only metric scale comes from
the checkerboard square size entered during calibration.

Question 1 has two sides:

- **An interactive Streamlit page** (`Pages/app_module7.py`) in which a visitor loads or computes their own stereo calibration and
  measures a rectangle in their own photographs.
- **The author's saved experiment (Model B):** an iPhone Air (left) and an iPhone 17 Pro Max (right) calibrated from 30
  checkerboard pairs (8×6 inner corners, 20 mm squares) and used to measure a 305 × 215 mm notebook (including the red spine) in three
  stereo pairs. Its method, results and verification scripts are described below.

**Visitor measurements never use Model B.** The app does not fall back to the author's calibration: without a loaded or computed
visitor calibration, measurement is unavailable. Model B is used only by the saved-experiment view, the command-line scripts and
regression tests.

## Using the app

The page is registered in `home.py` as **Module 7 - Stereo Vision** (`/module-7`). Its home screen offers two parts, which the page
labels Part 1 and Part 3 (the questions of the same number):

| Part (page label) | Content |
|---|---|
| **Part 1 – Calibrated Stereo Object Measurement** (Question 1) | Documented in this section. |
| **Part 3 – Room Measurement Using Stereo Videos** (Question 3) | The author's saved room experiment is shown on the website. **Measure a Room Using Your Own Stereo Videos** shows two buttons, **Setup Instructions** and **Run Your Experiment** (enabled only locally): the visitor workflow runs in a separate local application started with `python run_local.py` (see [Question 3](#question-3--room-measurement-using-stereo-videos)). |

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
All six dimensions are short of the known size, by between 0.9% and 2.8%. The reasons were not isolated in this experiment.

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
├── outputs/                         CSV/JSON results and visualizations (outputs/room_measurement/: Question 3 saved experiment)
├── room_measurement/                Question 3: estimation, validation and visitor measurement code (see Question 3)
├── visitor_workflow/                Question 3: the standalone visitor application and its worker processes
├── f8_chunk_poc/                    Question 3: chunked video upload, calibration worker and resource budget
├── annotation_ui/                   Question 3: boundary-annotation helper
├── run_room_measurement.py          Question 3: re-run the saved Wall A experiment
├── run_f8_integrated_app.py         Question 3: start the visitor application on its own
└── data/                            original photographs and recordings (git-ignored; see below)
    ├── calibration/{left,right}/        air_N.jpeg / pro_N.jpeg, N = 1–30
    ├── object_measurement/{left,right}/ air_imageN.jpeg / pro_imageN.jpeg, N = 1–3
    └── room_measurement/                original room recordings (Question 3)
```

Question 1 is the `stereo_measure/` package and the top-level scripts, `calibration/model_b/`, `selections/`, `reference/` and `display/`;
Question 3 is the directories marked above together with `calibration/room_video/`. The repository root also holds `run_local.py` and
`local_mode.py` (the local launcher). The page itself is `Pages/app_module7.py`; the tests are `tests/test_module7_*.py` in the repository
root and the `tests/` folders inside `Modules/Module_7/`.

## Tracked and ignored data

| Kept in Git (needed to run the app and the checks) | Git-ignored (kept locally) |
|---|---|
| `calibration/model_b/`, `calibration/corrected_corners/` | `data/calibration/{left,right}/*.jpeg` (60 original checkerboard photographs) |
| `selections/`, `reference/`, `outputs/` | `data/object_measurement/{left,right}/*.jpeg` (6 original notebook photographs, about 22 MB) |
| `display/` (reduced images used by the app) | `results/corner_detection/` and `results/stage1_5_validation/` images |
| `.gitkeep` placeholders in `data/calibration/{left,right}/` | |
| `calibration/room_video/room_video_recipe_a_calibration.json` (Question 3 calibration) | `data/room_measurement/` (original 2160 × 3840 room recordings) |
| `outputs/room_measurement/` (Question 3 results, display images and 720 × 1280 reduced display videos, about 4 MB) | |

The deployed app and the default checks need only the left-hand column. The original photographs are required only to regenerate
`display/` (`make_display_images.py`), to re-run board detection from scratch, and for the tests marked as needing them. The original room
recordings are required only to re-run the Question 3 experiment from scratch and for the tests that read them.

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

The tests use pytest (several rely on pytest fixtures, so `python -m unittest discover` would run only part of the suite). From the
repository root:

```bash
python -m pip install pytest
python -m pytest Modules/Module_7 tests/test_module7_*.py     # complete Module 7 suite, roughly 25 minutes
python -m pytest tests/test_module7_stereo.py                 # for example, a single file
```

The suite covers Question 1 (calibration, NPZ I/O, the wizard, corner mapping and dragging, the measurement logic, reset behaviour and the
Streamlit page via `streamlit.testing`) and Question 3 (the visitor workflow, workers, memory budget and temporary-file cleanup, and
synthetic-data checks of the estimation code). Tests that need the git-ignored original photographs or room recordings are skipped, with a
message, when those files are absent, so the suite runs from a fresh clone of the tracked files. The expensive regression tests on the private Wall B
recordings run only when the recordings are present and `M7_REAL_VIDEO_TESTS=1` is set. The Module 7 suite needs only
`visitor_workflow/requirements.txt` and pytest; the tests of the other modules (`python -m pytest tests`) need the root `requirements.txt`.

## Question 3 – Room Measurement Using Stereo Videos

Question 3 estimates the dimensions of a room from stereo videos, using classical computer vision only (OpenCV and NumPy: SIFT features,
robust epipolar geometry, triangulation, plane and line fitting, rigid registration). No machine-learning or deep-learning model is used
anywhere in it. It has two measurements:

- **Wall A – room width and height**, from two viewpoints (four videos: left and right camera at each viewpoint).
- **Wall B – room length**, from three viewpoints (six videos).

It is an approximate, educational experiment, not a precision room-measurement system.

### Setup of the author's experiment

| | |
|---|---|
| Left camera | iPhone Air, 1× rear camera |
| Right camera | iPhone 17 Pro Max, 1× rear camera |
| Video | 2160 × 3840 (portrait), 30 fps |
| Physical baseline | 225 mm, held fixed in the estimator |
| Calibration | `calibration/room_video/room_video_recipe_a_calibration.json` ("Recipe A"): intrinsics and distortion of both phones at 2160 × 3840, fitted from checkerboard corners in calibration videos. The stereo extrinsics stored in that file are kept for provenance only; the rig pose is re-estimated from the wall videos. |

### Method

The estimation is described step by step in [`room_measurement/README.md`](room_measurement/README.md). In summary: SIFT features are
matched between the left and right videos and across viewpoints; the shared stereo pose is estimated by robust (Tukey) epipolar and
four-view track reprojection costs with the baseline fixed; the floor plane is fitted robustly to triangulated points; the wall boundaries
are located from the **manually placed annotation lines** (the visitor marks the wall's boundaries and corners; automatic sub-pixel edge
search then refines the lines); corners are inferred from the intersection of the vertical-edge planes with the floor and ceiling lines; and
the viewpoints are registered rigidly (RANSAC) on cross-view matches. Wall B repeats the stereo pose, floor plane and corner steps for each
of the three viewpoints and measures the 3D distance between the two outer bottom corners after registering them into the middle viewpoint's frame
(`room_measurement/three_view_wall_b.py`). No measured room dimension or camera height enters any fit.

### Accepted experimental results

*Where the results are:* the Wall A files `outputs/room_measurement/estimation_result.json` and `validation.json` (with the display images and reduced videos beside them), and the Wall B file [`outputs/room_measurement/wall_b_accepted_results.json`](outputs/room_measurement/wall_b_accepted_results.json). The original recordings are excluded from Git, are not part of the repository and must be supplied by anyone who wants to re-run the experiment; the Wall B file records their SHA-256 hashes so that the exact inputs can be identified.

**Wall A (the author's experiment, shown under *View My Experiment & Results* on the website).** The estimate was written first and compared
with the measured size afterwards (`outputs/room_measurement/estimation_result.json` and `validation.json`):

| | Estimate | Measured size | Error | 95 % interval |
|---|---|---|---|---|
| Room width | 2879.6 mm (287.96 cm) | 2900 mm | −20.4 mm (−0.70 %) | 2655.7 – 3103.5 mm |
| Room height | 2380.2 mm (238.02 cm) | 2400 mm | −19.8 mm (−0.82 %) | 2187.7 – 2572.8 mm |

Both measured sizes lie inside the interval. The interval is dominated by the uncertainty of the camera intrinsics (about 107 mm of the 114 mm
1σ in width and 94 of 98 mm in height), whose assumed perturbation sizes were not measured. The small spread between frame sets (a few mm) is an
artefact of sharing one pose and one set of annotations and is not the measurement uncertainty.

**Wall B (the author's offline experiment).** The three frame sets (frames 60, 150 and 240 of the three-viewpoint recordings) gave room lengths of
**3790.9 mm, 3812.0 mm and 3804.0 mm** (379.09, 381.20 and 380.40 cm; mean 380.2 cm), computed from the author's hand-placed annotations. These
values are recorded in [`outputs/room_measurement/wall_b_accepted_results.json`](outputs/room_measurement/wall_b_accepted_results.json) and used as regression references in the test suite (tolerance ±5 mm); the website's saved-experiment view shows Wall A only. The Wall B
application reports no confidence interval and no percentage error.

### Later verification of the visitor application

After the application was completed, both walls were run once through the visitor application in a browser, with the author's recordings and
the Recipe A calibration, and the boundaries entered through the application's frame view. These numbers are a check of the application,
**not** the accepted experimental results above:

| | Visitor application | Accepted experiment |
|---|---|---|
| Wall A width × height | about 285.4 × 235.6 cm | 287.96 × 238.02 cm |
| Wall B length, frames 60 / 150 / 240 | 377.99 / 388.79 / 383.92 cm (mean 383.6 cm) | 379.09 / 381.20 / 380.40 cm (mean 380.2 cm) |

The Wall A differences (about 2.6 cm in width and 2.4 cm in height) are small compared with that experiment's stated uncertainty. The Wall B
frame sets differ by up to 7.6 cm from the references. This is the **sensitivity to annotation precision**: replaying the author's exact
annotation coordinates through the same code reproduces the references exactly, while coordinates snapped to the application's click grid
reproduce the application's numbers to within about 1 mm per frame set. A change of a few pixels in a boundary line therefore moves the
result by centimetres at this wall distance.

### Limitations

- The left (iPhone Air) camera cannot see the ceiling corners, so top corners are inferred from the right camera's rays only; one bottom corner
  is obstructed and is inferred from the stereo-derived vertical line and the floor plane. The inference assumes vertical walls and a planar floor.
- The camera intrinsics dominate the uncertainty; a different calibration fit moves the width by several percent. A relative stereo yaw of 0.1° is
  worth about 80 mm of width and is only weakly observable, so a low epipolar residual does not by itself prove a correct metric reconstruction.
- The result depends on the annotation precision (see above). Accuracy was validated on one room with one pair of phones; other recordings may give
  inaccurate results or be refused with an explanation.

### Website and local application

The deployed Streamlit website does **not** process visitor videos: full-resolution (4K) stereo videos need more memory than the free hosting plan
provides. The website shows the author's saved Wall A experiment, a **Setup Instructions** page, and a **Run Your Experiment** button that is disabled
on the public website. The upload, frame selection, annotation and measurement of a visitor's own videos run only in the standalone local application
described next.

## Run the visitor application locally

The website (`home.py`) never accepts videos and starts no measurement worker: the free hosting plan cannot reliably process 4K stereo videos. On the website, **Measure a Room Using Your Own Stereo Videos** opens a page with two buttons, the same on the public website and locally: **Setup Instructions** (always enabled) and **Run Your Experiment** (disabled on the public website with the note "Available when running locally. See Setup Instructions."; enabled locally while the visitor application is running, where it opens that application in a new tab and starts no processing by itself). The complete visitor workflow (video upload with chunked transfer, frame selection, boundary annotation, Wall A and Wall B measurement, results) is the standalone Streamlit application `visitor_workflow/app_f8_integrated.py`. One command starts the website and the visitor application together:

```bash
git clone https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury.git
cd CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury
python3.12 -m venv .venv                 # Windows PowerShell: py -3.12 -m venv .venv
source .venv/bin/activate                # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r Modules/Module_7/visitor_workflow/requirements.txt
python run_local.py                      # website http://127.0.0.1:8501, visitor application http://127.0.0.1:8640
```

`run_local.py` (repository root) starts two separate Streamlit applications bound to `127.0.0.1`: `home.py` and `Modules/Module_7/run_f8_integrated_app.py <port>`. It refuses to start when a port is busy (naming it; `--website-port` and `--visitor-port` choose others), reports a failed start-up, and stops both applications and their child processes on Ctrl+C or SIGTERM. Local mode is enabled only by the launcher, through `CSC8830_LOCAL_MODE=1` in the environment of the website it starts (see `local_mode.py`); the visitor application's address is always `http://127.0.0.1:<port>`. Running `python -m streamlit run home.py` yourself therefore behaves like the public website, with **Run Your Experiment** disabled. The visitor application alone: `python Modules/Module_7/run_f8_integrated_app.py [port]`.

Tested: the macOS (Apple silicon) commands above, from a fresh copy of the repository in a new virtual environment with Python 3.13.7 (`python run_local.py` started both applications, both answered their health checks, and Ctrl+C stopped both). The pinned packages need Python 3.12 or newer (`.python-version` says 3.12; the application was tested with 3.13.7, and 3.12 itself has not been tested); the Windows and Linux commands follow standard Python conventions and have not been tested. No FFmpeg or system packages are needed (OpenCV bundles its own decoder), and no secrets or private files are required; the validated calibration file `calibration/room_video/room_video_recipe_a_calibration.json` is part of the repository.

Both walls (Wall A and Wall B) are enabled in this application. The measurement runs in a separate worker that is stopped after 300 seconds of run time or 1800 CPU-seconds, and it starts only if enough memory is free; if a limit is reached, the application names it.

**Memory (what to expect on your own computer).** Nothing needs to be configured. When no container limit applies (a Mac, Windows, or a Linux computer that is not inside a limited container) the application uses your computer's physical memory as its budget, keeps 10 % of it (at least 256 MB) in reserve, and checks again at every start that enough memory is free *right now*, because other programs may be using part of it. On the author's recordings the Wall A calculation used about 2.0 GB at its peak; the application requires at least 1,024 MB for two viewpoints and 2,450 MB for three (Wall B), and refuses to start, with a clear message, when less is free. As a guide, keep **at least 4 GB of memory free** (a computer with 8 GB or more is comfortable) and close other large programs first. Inside a container with a readable memory limit, that limit is used instead (as on Streamlit Community Cloud). If the limit of an environment cannot be read at all, a conservative 2,048 MB budget applies; the optional `F8_APP_MEMORY_CEILING_MB=<MB>` (512 to 262144) then states the memory the application may use, and anywhere else it can only *lower* the budget, never raise it above what the computer has (for example `F8_APP_MEMORY_CEILING_MB=4096 python run_local.py`).

**Temporary files.** While a measurement is open, the extracted frames (roughly 60 to 100 MB per measurement) are kept in your system's temporary folder in a `f8_frames_...` folder that the application marks as its own. It is removed when you press **New Measurement** or start over, when your session ends (the application releases an abandoned session the next time anyone opens the page), when the application stops, and, if the application was killed, at the next start (only folders that carry the application's own marker and whose owner process no longer exists are removed). Folders of that name that were created by earlier versions have no marker and are never touched; you may delete them yourself. Uploaded videos are removed when the application stops.

**Calibration.** The default route is **Upload Saved Calibration** with a calibration file made for your own cameras (the repository's validated file is accepted with capture mode *video* but only fits a matching rig). Computing a calibration from your own checkerboard videos is disabled unless you start the application with an explicit memory limit for the calibration worker, for example `python Modules/Module_7/run_f8_integrated_app.py --enable-full-calibration --max-tree-rss-mb=3000`. The worker peaked at about 770 MB on the author's 3.6-minute recordings, so 2500-3000 MB is a reasonable limit; it must be between 256 and 65536 and given in the `=` form.

**Recording requirements.** Enforced by the application: four videos for Wall A (two viewpoints) or six for Wall B (three viewpoints), one per left/right camera slot; MP4/M4V, MOV, Matroska/WebM or AVI (checked by file content); 1-120 seconds and 1-480 fps; at most 1024 MB per video (`F8_ROOM_MAX_FILE_MB` can lower or raise it, to at most 2048); and exactly the frame size and orientation of the accepted calibration, since videos are never resized or rotated. A left/right frame-rate difference over 1 fps is flagged. Recommended, from the author's validated experiment, and not enforced: the same cameras and settings (resolution, orientation, frame rate, zoom, focus, stabilization) as in calibration, a fixed rig, a still room, a wall with visible corners and a textured floor; the validated recordings are 2160 x 3840 portrait. H.265/HEVC videos pass the file check, but whether they decode depends on your OpenCV build. These checks do not guarantee an accurate result, and accuracy was validated on one room only.

**Tests.** From the repository root, with `pytest` installed: `python -m pytest Modules/Module_7 tests/test_module7_*.py` (see [Testing](#testing)).

**Troubleshooting.** *Command not found*: install Python 3.12 and activate the virtual environment. *A package fails to install*: `python -m pip install --upgrade pip`, then retry. *Port in use*: `run_local.py` names the busy port; pass `--website-port` / `--visitor-port`. *Run Your Experiment greyed out*: you are on the public website, or the visitor application is not running; use `python run_local.py` and press **Check again**. *A video is rejected*: the message names the check (size, length, frame rate, or a size or orientation different from the calibration). *Out of memory or a stopped calculation* ("Not enough memory"): close other large programs so that at least 4 GB is free, then press **Try again**; the message names the limit that was reached (see **Memory** above). *Calibration button disabled*: start with `--enable-full-calibration --max-tree-rss-mb=<MB>` or use **Upload Saved Calibration**.

## Python and deployment

Use Python 3.12 (see `.python-version` and the comments in the root `requirements.txt`; select Python 3.12 in the Streamlit Community
Cloud app settings). From the repository root:

```bash
SAM2_BUILD_CUDA=0 python -m pip install -r requirements.txt
python -m streamlit run home.py
```

Question 1 and the website pages of Question 3 need `numpy`, `opencv-python-headless`, `pandas`, `Pillow`, `streamlit` and `streamlit-image-coordinates`, all in the
root `requirements.txt` (the other packages there serve Module 4). The deployed app runs entirely from files committed to Git; it does not
need the original photographs and keeps visitor photographs and calibrations in the session only (nothing is stored by the app unless the visitor downloads a calibration `.npz`). The local Question 3 visitor application has its own pinned requirements file, `visitor_workflow/requirements.txt` (classical computer-vision packages only).
