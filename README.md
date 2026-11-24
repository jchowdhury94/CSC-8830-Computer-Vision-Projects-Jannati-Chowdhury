# CSC 8830 - Computer Vision Projects

**Student:** Jannati Chowdhury

**University:** Georgia State University

**Semester:** Fall 2026

**Instructor:** Dr. Ashwin Ashok

**Public Web Interface:**

[https://jannati-chowdhury-computer-vision-projects.streamlit.app](https://jannati-chowdhury-computer-vision-projects.streamlit.app/)

**GitHub Repository:**

[https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury](https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury)

This repository contains Streamlit-based projects completed for CSC 8830 Computer Vision during Fall 2026. The projects cover Modules 2, 3, 4, 6, and 7. The public Streamlit website is the easiest way to view the projects without local installation. Choose a module on the home page to explore its interface and results.

## Current Modules

### Module 2 - 2D Object Measurement

- **Camera calibration:** OpenCV detects and refines checkerboard corners to estimate camera intrinsics and lens distortion. The setup uses an 8 × 6 inner-corner pattern with 20 mm squares.
- **Calibration parameters and results:** `Modules/Module_2/camera_calibration.npz` stores the camera matrix, distortion coefficients, RMS reprojection error, and calibration setup metadata. The Streamlit calibration reference displays these values, including focal lengths and the principal point.
- **Image-based 2D object measurement:** The interface corrects lens distortion in an uploaded image and uses four selected object corners, calibrated focal lengths, and an entered camera-to-object distance to estimate width and height. It reports absolute and percentage errors against entered actual dimensions.
- **Streamlit interface:** `Pages/app_module2.py` provides image upload, corner selection, measurement controls, calibration reference, and validation results. Existing validation trials are stored in `Modules/Module_2/measurement_results.csv`; replacing a trial requires explicit confirmation in the interface.

Select corners in this order: top-left, top-right, bottom-right, bottom-left. Distances and dimensions are entered in centimeters. The method assumes the measured object is approximately parallel to the camera image plane. Measurement images should correspond to the calibrated camera and use the calibration image resolution because the saved focal lengths are not rescaled.

Calibration photographs, measurement photographs, and supporting scripts remain in `Modules/Module_2/`. Its command-line support scripts use paths relative to that directory. The saved calibration is already included; regenerating it is not required to launch the website.

### Module 3 - Image Filtering

- **Spatial Gaussian filtering:** Builds a normalized Gaussian kernel and applies explicit spatial convolution with reflected image boundaries.
- **Frequency-domain Gaussian filtering:** Uses the Fourier transform to multiply the image and kernel spectra, then applies the inverse transform. Padding and cropping match the spatial method's boundaries and output size.
- **Numerical comparison and convolution theorem validation:** Compares the unrounded spatial and frequency results using maximum absolute difference and mean absolute difference (MAE). Agreement near floating-point precision numerically validates the convolution theorem for the implemented filtering methods.
- **Streamlit interface:** `Pages/app_module3.py` supports image upload, kernel size and sigma controls, kernel display, filtered-image previews, and comparison of both results. Filtering functions are implemented in `Modules/Module_3/image_filtering.py`.

Representative Module 3 validation results:

| Parameter or metric | Value |
| --- | --- |
| Gaussian kernel | 9 × 9 |
| Sigma | 2.0 |
| Kernel sum | 1.000000000000 |
| Maximum Absolute Difference | 5.115908 × 10^-13 |
| Mean Absolute Difference (MAE) | 8.668103 × 10^-14 |

These representative differences are extremely close to zero and consistent with floating-point roundoff. Exact comparison values depend on the input image and filter settings.

### Module 4 - Edge Detection, Boundary Detection, and Features

`Pages/app_module4.py` provides two human-boundary workflows:

- **Question 1 - RGB Human Boundary Detection:** Upload a regular RGB image and select a rectangle around the person. Large RGB images are automatically resized to a shared processing resolution for memory-safe execution; Classical and SAM2 receive the same prepared RGB image and rectangle coordinates. Either method can be run first. Selecting Classical Method reveals its processing settings before application. The Classical pipeline uses Gaussian preprocessing, Canny edge detection for visualization, GrabCut segmentation, optional morphological cleanup, and largest external contour extraction for the boundary overlay. SAM 2.1 Hiera Tiny uses the same prepared image with a box-only prompt. Method-specific reset controls allow either method to be rerun while preserving the shared image and rectangle selection and without unnecessarily rerunning the other method. When both current results are available, masks and boundaries can be compared side by side. SAM2's predicted mask-quality score is not ground-truth accuracy.
- **Question 2 - Thermal Human Boundary Detection:** Upload a thermal pseudocolor image. The Classical pipeline is HSV conversion → warm-color thresholding → 3×3 elliptical morphological closing → connected-component analysis → largest foreground component → external contour extraction → human-boundary overlay. The Classical boundary is bright magenta. Large thermal images are automatically resized to a shared processing resolution before processing. Classical and SAM2 both use the same prepared thermal image. SAM2 uses a user-selected rectangle as a box-only prompt. This rectangle affects only SAM2; the Classical thermal method does not use it. Either method can be started first, with contextual actions leading to side-by-side mask and boundary comparisons.
- **Thermal comparison metric:** IoU is the single numerical comparison metric: foreground intersection divided by foreground union, calculated directly from the Classical and SAM2 masks at their shared processing resolution. IoU measures the overlap/agreement between the Classical and SAM2 segmentation masks. It is not a ground-truth accuracy metric. The interface displays four decimal places, or N/A when both masks are empty.
- **Model checkpoint:** Both workflows use SAM 2.1 Hiera Tiny. The checkpoint is not stored in Git. The runtime obtains and caches the official checkpoint when required, or reuses verified local weights.

### Module 6 - Optical Flow and Structure from Motion

`Pages/app_module6.py` presents saved results; the scripts and outputs are in `Modules/Module_6/`.

- **Part A - Optical flow and Lucas-Kanade validation:** Sparse pyramidal Lucas-Kanade tracking of manually selected features in two videos (horizontal and diagonal motion), with a manual Lucas-Kanade estimate from the optical-flow constraint and a worked bilinear-interpolation example as validation.
- **Part B - Structure from Motion by factorization:** Eight manually matched points on a book cover in four views form a measurement matrix. Rank-3 factorization (Tomasi-Kanade, orthographic model) gives an unscaled 3D reconstruction, which is scaled to the measured cover size and validated, with its limitations stated. Step notes are in `Modules/Module_6/Part_B/README*.md`.

The original videos and high-resolution photographs are git-ignored; the page uses the saved outputs.

### Module 7 - Stereo Vision

Module 7 measures real-world sizes with a calibrated stereo pair of phones (an iPhone Air and an iPhone 17 Pro Max), using classical computer vision only (no machine learning or deep learning). It has two questions:

- **Question 1 - Calibrated Stereo Object Measurement.** Z-depth, width and height of an object from two calibrated photographs (OpenCV checkerboard stereo calibration, undistortion and triangulation). Visitors can load a saved calibration (`.npz`) or compute one from their own checkerboard photographs, select the object's corners with a draggable zoom view, and calculate the dimensions. The page also shows the author's saved experiment: a notebook measured in three stereo pairs with a calibration from 30 checkerboard pairs. Visitor measurements never use the author's calibration.
- **Question 3 - Room Measurement Using Stereo Videos.** One experiment in two stages: Wall A (width and height from two viewpoints) and Wall B (length from three viewpoints). The estimated room is approximately **288 x 380 x 238 cm** (width x length x height); against tape measurements the mean absolute percentage error is 1.65 %. The website shows the experiment's methodology, results, playback videos and visualizations. Measuring a room from your own stereo videos runs only in a standalone application on your own computer, started with `python run_local.py` (see [Run the Module 7 room-measurement application locally](#run-the-module-7-room-measurement-application-locally)); the public website does not process visitor videos.

Module 7 is organized by question: `Modules/Module_7/object_measurement_question_1/` (Question 1), `Modules/Module_7/room_measurement_question_3/` (Question 3) and the shared stereo calibration package `Modules/Module_7/stereo_measure/`. See [Modules/Module_7/README.md](Modules/Module_7/README.md) for the methods, workflows, results, data policy and reproduction commands.

## Project Structure

```text
CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury/
├── README.md
├── home.py
├── home_navigation.py
├── select_cards.py
├── run_local.py
├── local_mode.py
├── requirements.txt
├── .gitignore
├── Pages/
│   ├── app_module2.py
│   ├── app_module3.py
│   ├── app_module4.py
│   ├── app_module6.py
│   └── app_module7.py
├── Modules/
│   ├── Module_2/
│   ├── Module_3/
│   ├── Module_4/
│   │   ├── human_boundary.py
│   │   ├── rgb_preprocessing.py
│   │   ├── thermal_boundary.py
│   │   ├── thermal_preprocessing.py
│   │   ├── sam2_comparison.py
│   │   ├── sam2_runtime.py
│   │   └── requirements-sam2.txt
│   ├── Module_6/
│   │   ├── Part_A/
│   │   └── Part_B/
│   └── Module_7/
│       ├── README.md
│       ├── stereo_measure/                  shared stereo calibration package
│       ├── object_measurement_question_1/   Question 1: calibration, results, reproducibility scripts
│       └── room_measurement_question_3/     Question 3: estimator, visitor application, results
└── tests/
```

`home.py` is the single Streamlit entry point and registers each module page. `home_navigation.py` and `select_cards.py` hold the shared navigation row and selection cards, and `run_local.py` with `local_mode.py` start the local room-measurement application. This root README documents the combined course website.

### Adding Future Modules

1. Add implementation, data, and support files under `Modules/Module_N/`.
2. Add the module's Streamlit page at `Pages/app_moduleN.py`.
3. Register the new page in the existing `st.navigation` list in `home.py`, using the same `st.Page` pattern as the existing modules.
4. Add dependencies to `requirements.txt` only when needed.
5. Add tests under `tests/`.
6. Add a short section for the module to this root README. A module with detailed methods or workflows may have its own `Modules/Module_N/README.md`, which the root section should link to rather than repeat.

Existing modules do not need to be reorganized when future modules are added.

## Local Installation and Run Instructions

The project was developed and tested with Python 3.13.7 on macOS (Apple silicon); other Python versions have not been verified. Open a terminal in the project root. On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
SAM2_BUILD_CUDA=0 python -m pip install -r requirements.txt
python -m streamlit run home.py
```

Open the local URL printed by Streamlit and choose a module on the home page. For access without local installation, use the [public Streamlit website](https://jannati-chowdhury-computer-vision-projects.streamlit.app/), which is the main deployment of this project.

## Run the Module 7 room-measurement application locally

The Question 3 visitor workflow (measuring a room from your own stereo videos) is **local-only**: processing full-resolution (4K) stereo videos needs more memory than the free hosting plan provides, so the public website shows only the saved results and the Setup Instructions. To run it on your own computer, install the lighter Module 7 dependencies (no PyTorch or SAM 2) and start both applications with one command:

```bash
git clone https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury.git
cd CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury
python -m venv .venv                       # Windows PowerShell: py -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r Modules/Module_7/room_measurement_question_3/visitor_workflow/requirements.txt
python run_local.py                        # website http://127.0.0.1:8501, room-measurement application http://127.0.0.1:8640
```

Then open the website and choose **Module 7 > Run Your Own Experiment > Room Measurement Using Stereo Videos**. **Setup Instructions** expands inline with these steps; **Run Your Experiment (Local Only)** opens the application in the same browser tab, where you choose **Calibrate My Cameras** or **Upload Saved Calibration** and continue through the same workflow. Ctrl+C stops both applications. You need your own stereo videos and a calibration for your own cameras: the author's original recordings are private and are not in the repository. Requirements, video limits, memory, troubleshooting and the reproduction commands for the author's experiments are in [Modules/Module_7/README.md](Modules/Module_7/README.md).

## Running the Tests

The tests use pytest, which is the supported way to run them (several tests rely on pytest fixtures). From the project root, with the root `requirements.txt` installed:

```bash
python -m pip install pytest
python -m pytest tests                       # website, all modules
```

The Module 7 package suites and the opt-in real-recording regression tests are listed in [Modules/Module_7/README.md](Modules/Module_7/README.md#9-testing). Tests that need the original photographs or room recordings, which are git-ignored and not part of the repository, are skipped automatically when those files are absent, so the suite runs from a fresh clone. A few integrity tests use `git status` to confirm that validated algorithm sources are unchanged and therefore need a Git checkout with a clean working tree.
