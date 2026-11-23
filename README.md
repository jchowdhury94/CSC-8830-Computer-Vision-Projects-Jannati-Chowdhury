# CSC 8830 - Computer Vision Projects

**Student:** Jannati Chowdhury

**University:** Georgia State University

**Semester:** Fall 2026

**Instructor:** Dr. Ashwin Ashok

**Public Web Interface:**

[https://jannati-chowdhury-computer-vision-projects.streamlit.app](https://jannati-chowdhury-computer-vision-projects.streamlit.app/)

**GitHub Repository:**

[https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury](https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury)

This repository contains Streamlit-based projects completed for CSC 8830 Computer Vision. Modules 2, 3, 4, 6, and 7 are currently available, and additional modules will be added throughout the course. The public Streamlit website is the easiest way to view the projects without local installation. Select a module from the sidebar to explore its interface and results.

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

Both questions use classical computer vision only (no machine learning or deep learning).

**Question 1 - Calibrated Stereo Object Measurement.** Calibrated stereo measurement of an object's Z-depth, width and height from two photographs, using classical OpenCV. A visitor can load a saved calibration (`.npz`) or compute one from their own checkerboard photographs, select the object's corners on both images with a draggable zoom view, and calculate the dimensions. The page also shows the author's saved experiment: an iPhone Air and iPhone 17 Pro Max calibrated from 30 checkerboard pairs, used to measure a notebook. Visitor measurements never use the author's calibration, and the original photographs are not needed to run the page.

**Question 3 - Room Measurement Using Stereo Videos** (Part 3 on the Module 7 page). The website shows the author's saved room experiment (Wall A: room width and height from two viewpoints). The website does not process visitor videos: measuring a room from your own stereo videos (Wall A, and Wall B: room length from three viewpoints) runs in a separate standalone application on your own computer, started with `python run_local.py` (see [Run the Module 7 room-measurement application locally](#run-the-module-7-room-measurement-application-locally)).

See [Modules/Module_7/README.md](Modules/Module_7/README.md) for the method, workflows, results, data files and verification commands.

## Project Structure

```text
CSC_8830_Projects_Jannati_Chowdhury/
├── README.md
├── home.py
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
│       ├── stereo_measure/              Question 1
│       ├── calibration/, selections/, reference/, display/, outputs/
│       ├── run_measurement.py, run_calibration.py, verify_pipeline.py
│       └── room_measurement/, visitor_workflow/, f8_chunk_poc/, annotation_ui/   Question 3
└── tests/
```

`home.py` is the single Streamlit entry point and registers each module page. This root README contains the documentation for the combined course website.

### Adding Future Modules

1. Add implementation, data, and support files under `Modules/Module_N/`.
2. Add the module's Streamlit page at `Pages/app_moduleN.py`.
3. Register the new page in the existing `st.navigation` list in `home.py`, using the same `st.Page` pattern as the existing modules.
4. Add dependencies to `requirements.txt` only when needed.
5. Add tests under `tests/`.
6. Add a short section for the module to this root README. A module with detailed methods or workflows may have its own `Modules/Module_N/README.md`, which the root section should link to rather than repeat.

Existing modules do not need to be reorganized when future modules are added.

## Local Installation and Run Instructions

With Python 3.12 installed, open a terminal in the project root. On macOS or Linux, run:

```bash
python -m venv .venv
source .venv/bin/activate
SAM2_BUILD_CUDA=0 python -m pip install -r requirements.txt
python -m streamlit run home.py
```

Open the local URL printed by Streamlit, then choose a module from the sidebar. For access without local installation, use the [public Streamlit website](https://jannati-chowdhury-computer-vision-projects.streamlit.app/).

## Run the Module 7 room-measurement application locally

The public website does not accept video uploads: processing full-resolution (4K) stereo videos needs more memory than the free hosting plan provides. On the website, **Module 7 > Room Measurement > Measure a Room Using Your Own Stereo Videos** shows two buttons: **Setup Instructions** (enabled) and **Run Your Experiment** (visibly disabled on the public website: "Available when running locally. See Setup Instructions."). Running locally, the same page enables **Run Your Experiment**, which opens the standalone visitor application (upload, frame selection, annotation, Wall A: room width and height, Wall B: room length) in a new browser tab.

1. **Get the code** (Git, or download the repository as a ZIP from GitHub and open a terminal in the extracted folder):

   ```bash
   git clone https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury.git
   cd CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury
   ```

2. **Create and activate a virtual environment** with Python 3.12 or newer (the pinned NumPy requires 3.12; the application was tested with 3.13.7 on macOS, and 3.12 itself has not been tested). If your command is `python3` or `python3.13`, use it in place of `python3.12`. macOS or Linux:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install the dependencies** (classical-CV packages only; this does not need PyTorch or SAM 2, which only Module 4 uses):

   ```bash
   python -m pip install -r Modules/Module_7/visitor_workflow/requirements.txt
   ```

4. **Start everything with one command:**

   ```bash
   python run_local.py
   ```

   `run_local.py` starts two separate Streamlit applications, both reachable only from your own computer (127.0.0.1): the website (`home.py`) at <http://127.0.0.1:8501> and the visitor application (`Modules/Module_7/run_f8_integrated_app.py`) at <http://127.0.0.1:8640>. Open the website, go to **Module 7 > Room Measurement > Measure a Room Using Your Own Stereo Videos** and press **Run Your Experiment**. Press Ctrl+C in the terminal to stop both. If a port is busy, nothing is started and the busy port is named; choose other ports with `python run_local.py --website-port 8502 --visitor-port 8641`. Local mode is switched on only by this launcher (it sets `CSC8830_LOCAL_MODE=1` for the website it starts), so `python -m streamlit run home.py` alone always behaves like the public website. The visitor application can still be started alone with `python Modules/Module_7/run_f8_integrated_app.py [port]`.

5. **Measure a wall.** In the visitor application, accept a stereo calibration (**Upload Saved Calibration**), upload the stereo videos, choose the frames, mark the wall boundaries and press **Calculate Wall Dimensions** (Wall A: two viewpoints, four videos). Then **Continue to Wall B - Room Length** (three viewpoints, six videos) and press **Calculate Room Length**. Computing a new calibration from checkerboard videos is off by default and needs `--enable-full-calibration --max-tree-rss-mb=3000` on the standalone launcher (see `Modules/Module_7/README.md`).

To view all modules of the website without the visitor application, install the root `requirements.txt` and run `python -m streamlit run home.py` (see above); there **Run Your Experiment** stays disabled.

**Videos and calibration are supplied by you:** the author's original recordings are private and are not in the repository, so you need your own stereo videos and a calibration for your own cameras (see below). The saved results of the author's experiments are tracked in the repository (see *Where the results are* in `Modules/Module_7/README.md`).

**Requirements enforced by the application:** four videos (Wall A) or six (Wall B) from the left and right cameras, one per slot; MP4/M4V, MOV, Matroska/WebM or AVI; 1-120 seconds and 1-480 fps per video; at most 1024 MB per video; and exactly the frame size and orientation of the accepted calibration (videos are never resized or rotated). **Recommendations from the author's validated experiment (not enforced):** the same two cameras and settings (resolution, orientation, frame rate, zoom, focus, stabilization) as in calibration, a fixed rig, a still room, a wall with visible corners and a floor with visible texture. The validated recordings are 2160 x 3840 portrait. H.265/HEVC files pass the file check, but decoding depends on your computer's OpenCV build. Arbitrary videos may give inaccurate results or be refused; accuracy was validated on one room only.

**Memory and temporary files:** nothing has to be configured. On a Mac, Windows or an unlimited Linux computer the application uses your computer's physical memory as its budget (keeping 10 % in reserve) and checks at every start that enough memory is free right now. Keep **at least 4 GB free** (the Wall A calculation used about 2.0 GB at its peak on the author's recordings; the application requires 1,024 MB for two viewpoints and 2,450 MB for three) and close other large programs first. Extracted frames live in a marked `f8_frames_...` folder in the system temporary folder and are removed by **New Measurement**, when the session ends, when the application stops, or at the next start after a crash; folders from earlier versions are never touched. Details and the optional `F8_APP_MEMORY_CEILING_MB` setting are in `Modules/Module_7/README.md`.

**Troubleshooting:** *command not found* - install Python 3.12 and activate the virtual environment; *a package fails to install* - run `python -m pip install --upgrade pip` and retry; *port in use* - `run_local.py` names the busy port; pass other ports with `--website-port` and `--visitor-port`; *Run Your Experiment greyed out* - you are on the public website, or the visitor application is not running: start `python run_local.py` and press **Check again**; *a video is rejected* - the message names the failed check (size, length, frame rate, or a different size or orientation than the calibration); *out of memory or a stopped calculation* ("Not enough memory") - close other large programs so that at least 4 GB is free and press **Try again**.

## Running the Tests

The tests use pytest, which is the supported way to run them (several tests rely on pytest fixtures, so `python -m unittest discover` finds only part of the suite). From the project root, with the root `requirements.txt` installed:

```bash
python -m pip install pytest
python -m pytest tests                              # all modules, about 1.5 minutes plus the Module 7 tests below
python -m pytest Modules/Module_7 tests/test_module7_*.py   # complete Module 7 suite (roughly 25 minutes)
```

Tests that need the original photographs or room recordings, which are git-ignored and not part of the repository, are skipped automatically when those files are absent, so the suite runs from a fresh clone (and runs those extra checks when the private files are present). The Module 7 suite alone needs only `Modules/Module_7/visitor_workflow/requirements.txt` plus pytest.
