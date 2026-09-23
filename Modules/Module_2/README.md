# CSC 8830 Computer Vision - Module 2

## Overview

This project calibrates a smartphone camera using a printed checkerboard, saves the camera intrinsic parameters and distortion coefficients, and corrects lens distortion before estimating the real-world width and height of planar objects using perspective projection. It uses Python, OpenCV, NumPy, and an interactive Streamlit web application. The measurement method was validated with 20 object measurements at distances greater than 2 meters.

## Camera Calibration

The calibration setup uses one printed checkerboard:

- Physical checkerboard: **9 × 7 squares**
- Detected inner corners: **8 × 6**
- Square size: **20 mm × 20 mm**
- Multiple photographs of the same checkerboard captured from different positions and orientations
- **18 photographs** with successfully detected checkerboard corners

`calibration.py` detects and refines checkerboard corners, uses OpenCV to estimate the camera matrix and distortion coefficients, and saves the parameters and setup metadata to `camera_calibration.npz`.

The current calibration produced an overall mean reprojection error of **2.358 pixels** (reported calibration result) and an RMS reprojection error of **3.0875 pixels**. The script prints both errors; the NPZ stores the RMS error but not the overall mean error.

## 2D Object Measurement

1. Enter the object name.
2. Enter the actual width and height in centimeters.
3. Enter the camera-to-object distance in centimeters.
4. Upload the measurement image.
5. The application corrects lens distortion using the saved calibration.
6. Select four object corners in order: **top-left (TL), top-right (TR), bottom-right (BR), bottom-left (BL)**. Selected points are visually numbered on the image.
7. The application estimates width and height using calibrated focal lengths and perspective-projection geometry, averaging opposite edge lengths.
8. The application displays estimated dimensions, absolute errors, and percentage errors.

The calculation converts image-coordinate differences using focal lengths `fx` and `fy` and the entered distance. It assumes the measured corners are approximately at the same depth, so the planar object should be approximately parallel to the camera image plane.

## Validation

The validation dataset contains **5 different objects**, measured at **4 distances per object**, for **20 total measurements**. The distances are **210 cm, 220 cm, 230 cm, and 250 cm**, all greater than 2 meters.

| Statistic | Width | Height |
| --- | --- | --- |
| Mean Absolute Error | 0.60 cm | 0.61 cm |
| Mean Percentage Error | 1.75% | 2.19% |
| Standard Deviation of Percentage Error | 0.91% | 1.25% |
| RMSE | 0.68 cm | 0.70 cm |

Detailed trials are stored in `measurement_results.csv`. These statistics describe the current validation dataset and are calculated by `analyze_results.py`.

## Web Application

`Pages/app_module2.py` provides the Module 2 Streamlit interface as part of the larger CSC 8830 multipage application, launched through the root-level `home.py`, with:

- **Camera Calibration Reference**, including the calibration setup, camera matrix, focal lengths, principal point, distortion coefficients, and RMS error
- Image upload and four-corner selection with numbered visual markers
- Live dimension estimation and absolute and percentage error calculation
- Input and corner reset controls
- Validation results and an expandable validation measurement table
- **Update Validation Trial** to replace a selected existing CSV trial after confirmation
- **Discard Result** to clear the temporary result and selected corners

## Project Files

Module 2 supporting files are located in `Modules/Module_2/`. Paths in the following table are relative to that directory:

| File or directory | Purpose |
| --- | --- |
| `README.md` | Module 2 assignment documentation and usage instructions. |
| `calibration.py` | Detects checkerboard corners, calibrates the camera, reports reprojection errors, and saves calibration data. |
| `measurement.py` | Interactive terminal and OpenCV measurement tool; appends approved measurements to the CSV. |
| `analyze_results.py` | Reads the validation CSV and reports aggregate error statistics. |
| `camera_calibration.npz` | Saved camera matrix, distortion coefficients, RMS error, and calibration setup metadata. |
| `measurement_results.csv` | Validation dataset containing distances, actual and estimated dimensions, and errors. |
| `calibration_images/` | Checkerboard photographs used for calibration. |
| `measurement_images/` | Object photographs used for measurement and validation. |

The web interface files are located relative to the project root, `CSC_8830_Projects_Jannati_chowdhury/`:

| File | Purpose |
| --- | --- |
| `home.py` | Entry point and navigation for the complete CSC 8830 multipage Streamlit application. |
| `Pages/app_module2.py` | Module 2 Streamlit measurement interface, calibration reference, and validation controls. |

## Requirements

Python 3 with the following packages:

- `numpy`
- `opencv-python`
- `streamlit`
- `streamlit-image-coordinates`

Other imports (`csv`, `glob`, `os`, `pathlib`, and `tempfile`) belong to the Python standard library.

## Installation

From the project root, `CSC_8830_Projects_Jannati_chowdhury/`, on macOS, create and activate a virtual environment, then install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python streamlit streamlit-image-coordinates
```

## Running the Project

For the three supporting scripts below, keep the root-level virtual environment active and change from the project root to `Modules/Module_2/`:

```bash
cd Modules/Module_2
```

Run these scripts from that directory so their relative image and data paths resolve correctly.

**Camera calibration:** Run when calibration parameters need to be generated or regenerated. This writes `camera_calibration.npz` using the JPEG photographs in `calibration_images/`. A saved calibration is already included.

```bash
python calibration.py
```

**Standalone measurement:** Follow the terminal prompts, enter an image filename from `measurement_images/`, and select four corners in the OpenCV window. Approving a result appends it to `measurement_results.csv`.

```bash
python measurement.py
```

**Error analysis:** Calculate statistics from the saved validation dataset.

```bash
python analyze_results.py
```

**Streamlit web application:** Launch the complete application from the project root, `CSC_8830_Projects_Jannati_chowdhury/`. If you are in `Modules/Module_2/` after running the supporting scripts, return to the root first:

```bash
cd ../..
```

From the project root, run:

```bash
.venv/bin/python -m streamlit run home.py
```

Open the displayed local URL and select **Module 2 - 2D Object Measurement** through the Streamlit navigation. The standalone measurement script is not required before launching the app.

## Streamlit Measurement Instructions

1. From the project root, `CSC_8830_Projects_Jannati_chowdhury/`, start the app with `.venv/bin/python -m streamlit run home.py`, open the displayed local URL, and select **Module 2 - 2D Object Measurement** through the Streamlit navigation.
2. Enter the object name and positive actual width and height in centimeters.
3. Enter a positive camera-to-object distance in centimeters.
4. Upload a JPG, JPEG, or PNG image.
5. Select **TL, TR, BR, BL** in order; use **Reset Points** to repeat the selection.
6. Click **Calculate Measurement**.
7. Review estimated dimensions and absolute and percentage errors.
8. View **Camera Calibration Reference**, **Validation Results**, or **View Validation Measurements** as needed.

Live measurements are **not automatically written to the validation CSV**. To save a result, select an existing **Validation Trial**, check the replacement confirmation, and click **Update Validation Trial**. This replaces that trial and refreshes the validation results. **Discard Result** clears the live result and corners without changing the CSV.

## Notes

- Calibration must correspond to the camera used for measurement. Measurement images should use the same resolution as the calibration images because the code uses the saved focal lengths without resolution scaling.
- Camera-to-object distance is entered in centimeters.
- Object dimensions are entered and reported in centimeters.
- Corner selection accuracy can affect measurement accuracy.
- `measurement_results.csv` contains the validation dataset; saving or replacing trials changes the reported validation statistics.
