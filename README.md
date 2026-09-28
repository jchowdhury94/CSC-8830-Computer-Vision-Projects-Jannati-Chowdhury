# CSC 8830 - Computer Vision Projects

**Student:** Jannati Chowdhury

**University:** Georgia State University

**Semester:** Fall 2026

**Instructor:** Dr. Ashwin Ashok

**Public Web Interface:**

[https://jannati-chowdhury-computer-vision-projects.streamlit.app](https://jannati-chowdhury-computer-vision-projects.streamlit.app)

**GitHub Repository:**

[https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury](https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury)

This repository contains Streamlit-based projects completed for CSC 8830 Computer Vision. Modules 2, 3, and Module 4 Question 1 are currently available, and additional modules will be added throughout the course. The public Streamlit website is the easiest way to view the projects without local installation. Select a module from the sidebar to explore its interface and results.

## Current Modules

### Module 2 - Camera Calibration and 2D Object Measurement

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

- **Question 1 - RGB Human Boundary Detection:** `Pages/app_module4.py` accepts an RGB image and lets the user select a rectangle around the person. It runs a classical computer-vision boundary-detection workflow and SAM 2.1 as the required comparison, with side-by-side masks and boundaries.
- **Classical method:** Gaussian preprocessing smooths the image, Canny edge detection provides edge visualization, GrabCut segments the foreground using the selected rectangle, and contour extraction selects the largest external contour for the human boundary. Optional morphological cleanup is available.
- **SAM2 comparison:** SAM 2.1 Hiera Tiny uses the same uploaded image and user-selected rectangle with a box-only prompt and no point prompts. Run SAM2 generates results dynamically; Compare Results uses the current session results. The displayed predicted mask-quality score is not measured accuracy.
- **Model checkpoint:** The checkpoint is not stored in Git. The deployment runtime obtains and caches the official checkpoint when needed, or reuses verified local weights.
- **Question 2 - Thermal Image:** Currently a placeholder.
- **Question 3 - Frequency Domain Analysis:** Currently a placeholder.

## Project Structure

```text
CSC_8830_Projects_Jannati_Chowdhury/
├── README.md
├── home.py
├── requirements.txt
├── .gitignore
├── Pages/
│   ├── app_module2.py
│   ├── app_module3.py
│   └── app_module4.py
└── Modules/
    ├── Module_2/
    ├── Module_3/
    └── Module_4/
        ├── human_boundary.py
        ├── sam2_comparison.py
        ├── sam2_runtime.py
        └── requirements-sam2.txt
```

`home.py` is the single Streamlit entry point and registers each module page. This root README contains the documentation for the combined course website.

### Adding Future Modules

1. Add implementation, data, and support files under `Modules/Module_N/`.
2. Add the module's Streamlit page at `Pages/app_moduleN.py`.
3. Register the new page in the existing `st.navigation` list in `home.py`, using the same `st.Page` pattern as the existing modules.
4. Add dependencies to `requirements.txt` only when needed.

Existing modules do not need to be reorganized when future modules are added. Extend this root README to document new modules instead of creating separate module README files.

## Local Installation and Run Instructions

With Python 3.12 installed, open a terminal in the project root. On macOS or Linux, run:

```bash
python -m venv .venv
source .venv/bin/activate
SAM2_BUILD_CUDA=0 python -m pip install -r requirements.txt
python -m streamlit run home.py
```

Open the local URL printed by Streamlit, then choose a module from the sidebar. For access without local installation, use the [public Streamlit website](https://jannati-chowdhury-computer-vision-projects.streamlit.app).
