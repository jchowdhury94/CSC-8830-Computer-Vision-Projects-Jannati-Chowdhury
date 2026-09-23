import csv
import os
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates


# Locates Module 2 data before and after moving this file into Pages.
APP_DIR = Path(__file__).resolve().parent
MODULE_DIR = (
    APP_DIR.parent / "Modules" / "Module_2"
    if APP_DIR.name == "Pages"
    else APP_DIR
)

# Loads the saved camera calibration
calibration = np.load(MODULE_DIR / "camera_calibration.npz")

camera_matrix = calibration["camera_matrix"]
distortion_coefficients = calibration["distortion_coefficients"]
rms_error = calibration["rms_error"]

# Gets the calibrated focal lengths
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]


# Calculates the real-world length between two image points
def calculate_length(point1, point2, distance):
    u1, v1 = point1
    u2, v2 = point2
    delta_x = (u2 - u1) / fx
    delta_y = (v2 - v1) / fy
    return distance * np.sqrt(delta_x ** 2 + delta_y ** 2)


CSV_COLUMNS = [
    "Trial", "Image", "Distance_cm", "Actual_Width_cm", "Estimated_Width_cm",
    "Width_Error_cm", "Width_Error_Percent", "Actual_Height_cm",
    "Estimated_Height_cm", "Height_Error_cm", "Height_Error_Percent"
]
RESULTS_PATH = MODULE_DIR / "measurement_results.csv"


# Reads and checks the existing validation trials
def read_validation_rows():
    with RESULTS_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("The validation CSV columns do not match the expected format.")
        rows = list(reader)
    for row in rows:
        for column in CSV_COLUMNS:
            if column != "Image":
                if not np.isfinite(float(row[column])):
                    raise ValueError("The validation CSV contains invalid numeric values.")
    return rows


# Replaces one existing trial using an atomic file replacement
def update_validation_trial(trial, result):
    rows = read_validation_rows()
    matches = [index for index, row in enumerate(rows) if row["Trial"] == trial]
    if len(matches) != 1:
        raise ValueError("The selected trial must exist exactly once in the validation CSV.")
    rows[matches[0]] = {"Trial": trial, **result}
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=RESULTS_PATH.parent, delete=False
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, RESULTS_PATH)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


# Clears the temporary result and its confirmation
def clear_live_result():
    st.session_state.pop("live_result", None)
    st.session_state.replace_trial_confirmed = False


# Clears confirmation when the selected trial changes
def clear_trial_confirmation():
    st.session_state.replace_trial_confirmed = False


# Discards the result and resets the selected corners
def discard_result():
    reset_points()
    st.session_state.result_notice = "Live result discarded."


# Clears the corners and starts a fresh click component
def reset_points():
    clear_live_result()
    st.session_state.selected_points = []
    st.session_state.click_generation += 1


# Resets the typed object and measurement information
def reset_inputs():
    clear_live_result()
    st.session_state.object_name = ""
    st.session_state.actual_width = None
    st.session_state.actual_height = None
    st.session_state.distance = None


# Configures the webpage
st.set_page_config(
    page_title="2D Object Measurement",
    page_icon="📐",
    layout="centered"
)

# Displays the webpage title
st.title("2D Object Measurement")

# Displays a short description
st.write(
    "Measures the real-world width and height of an object "
    "using camera calibration and perspective projection."
)

# Displays the camera calibration reference
with st.expander("Camera Calibration Reference"):
    st.write(
        "These parameters were obtained from camera calibration and are used "
        "to correct lens distortion and convert image measurements into "
        "real-world dimensions."
    )

    st.write("**Camera Matrix (3×3)**")
    st.table([[f"{value:.2f}" for value in row] for row in camera_matrix])

    # Gets the calibrated principal point
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    st.write("**Focal Length:**")
    st.write(f"fx (pixels): {fx:.2f}")
    st.write(f"fy (pixels): {fy:.2f}")

    st.write("**Principal Point:**")
    st.write(f"cx (pixels): {cx:.2f}")
    st.write(f"cy (pixels): {cy:.2f}")

    st.write("**Distortion Coefficients**")
    st.table({
        "Coefficient": ["k1", "k2", "p1", "p2", "k3"],
        "Value": [f"{value:.5f}" for value in distortion_coefficients.flatten()]
    })
    st.write(f"RMS Reprojection Error (pixels): {float(rms_error):.4f}")

    # Displays the saved calibration setup
    st.subheader("Calibration Setup")
    st.write(
        "The camera was calibrated using multiple photographs of the same "
        "printed checkerboard captured from different positions and orientations."
    )
    metadata_fields = (
        "checkerboard_columns", "checkerboard_rows",
        "square_size_mm", "successful_images"
    )
    if all(field in calibration for field in metadata_fields):
        checkerboard_columns = int(calibration["checkerboard_columns"])
        checkerboard_rows = int(calibration["checkerboard_rows"])
        square_size_mm = float(calibration["square_size_mm"])

        # Calculates the physical checkerboard square counts
        checkerboard_squares_columns = checkerboard_columns + 1
        checkerboard_squares_rows = checkerboard_rows + 1
        st.write(
            f"Physical Checkerboard: {checkerboard_squares_columns} "
            f"× {checkerboard_squares_rows} squares"
        )
        st.write(
            f"Detected Inner Corners: {checkerboard_columns} "
            f"× {checkerboard_rows}"
        )
        st.write(f"Each Square Size: {square_size_mm:g} mm × {square_size_mm:g} mm")
        st.write(f"Successful Calibration Photos: {int(calibration['successful_images'])}")
    else:
        st.info(
            "Calibration setup metadata is not available. "
            "Run calibration.py again to update the calibration file."
        )

st.subheader("Object Information")

# Gets the object information
object_name = st.text_input("Object Name", key="object_name", on_change=clear_live_result)

actual_width = st.number_input(
    "Actual Width (cm)",
    value=None,
    placeholder="0.00",
    min_value=0.0,
    step=0.1,
    key="actual_width",
    on_change=clear_live_result
)

actual_height = st.number_input(
    "Actual Height (cm)",
    value=None,
    placeholder="0.00",
    min_value=0.0,
    step=0.1,
    key="actual_height",
    on_change=clear_live_result
)

st.subheader("Measurement Information")

# Gets the camera-to-object distance
distance = st.number_input(
    "Camera-to-Object Distance (cm)",
    value=None,
    placeholder="0.00",
    min_value=0.0,
    step=1.0,
    key="distance",
    on_change=clear_live_result
)

st.button("Reset Inputs", on_click=reset_inputs)

# Stores the selected points
if "selected_points" not in st.session_state:
    st.session_state.selected_points = []
    st.session_state.click_generation = 0

# Gets the measurement image
uploaded_image = st.file_uploader(
    "Upload Measurement Image",
    type=["jpg", "jpeg", "png"],
    on_change=reset_points
)

if uploaded_image is not None:

    # Reads the uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_image.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Corrects lens distortion
    undistorted = cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients
    )

    # Converts the image from BGR to RGB
    undistorted_rgb = cv2.cvtColor(
        undistorted,
        cv2.COLOR_BGR2RGB
    )

    st.subheader("Select Object Corners")

    st.write(
        "Click the four corners of the object in this exact order:"
    )

    st.write("1. **Top-left**")
    st.write("2. **Top-right**")
    st.write("3. **Bottom-right**")
    st.write("4. **Bottom-left**")

    st.info(
        "Click as close as possible to each outer corner of the object."
    )

    # Defines the display width
    display_width = 500

    # Calculates the display scale
    original_width = undistorted_rgb.shape[1]
    display_scale = display_width / original_width

    # Resizes the image for display
    display_height = int(
        undistorted_rgb.shape[0] * display_scale
    )

    display_image = cv2.resize(
        undistorted_rgb,
        (display_width, display_height)
    )

    corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
    points = st.session_state.selected_points

    st.button("Reset Points", on_click=reset_points)

    # Displays the next corner and confirms the selected corners
    if len(points) < 4:
        st.info(
            f"Select point {len(points) + 1} of 4: "
            f"{corner_names[len(points)].upper()}"
        )
    else:
        st.success("All 4 corners selected.")

    for corner_name in corner_names[:len(points)]:
        st.write(f"✓ {corner_name} selected")

    # Draws numbered markers at the selected display positions
    for point_number, (original_x, original_y) in enumerate(points, start=1):
        display_x = round(original_x * display_scale)
        display_y = round(original_y * display_scale)
        marker_position = (display_x, display_y)
        cv2.circle(display_image, marker_position, 7, (0, 0, 0), -1)
        cv2.circle(display_image, marker_position, 5, (255, 255, 0), -1)

        # Keeps the label inside the displayed image
        label = str(point_number)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        label_position = (
            max(2, min(display_x + 10, display_width - label_width - 3)),
            max(label_height + 3, min(display_y - 10, display_height - baseline - 3))
        )
        cv2.putText(
            display_image, label, label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            display_image, label, label_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
        )

    # Displays the resized clickable image
    coordinates = streamlit_image_coordinates(
        display_image,
        key=f"measurement_image_{st.session_state.click_generation}"
    )

    # Converts display coordinates to original coordinates
    if coordinates is not None and len(points) < 4:
        original_x = coordinates["x"] / display_scale
        original_y = coordinates["y"] / display_scale
        points.append((original_x, original_y))

        # Prevents the consumed click from being reused on reruns
        st.session_state.click_generation += 1
        st.rerun()

    if st.button("Calculate Measurement", disabled=len(points) != 4):
        if any(value is None or value <= 0 for value in (distance, actual_width, actual_height)):
            st.error("Enter a positive distance, actual width, and actual height.")
        else:
            top_left, top_right, bottom_right, bottom_left = points

            # Calculates the estimated width
            top_width = calculate_length(top_left, top_right, distance)
            bottom_width = calculate_length(bottom_left, bottom_right, distance)
            estimated_width = (top_width + bottom_width) / 2

            # Calculates the estimated height
            left_height = calculate_length(top_left, bottom_left, distance)
            right_height = calculate_length(top_right, bottom_right, distance)
            estimated_height = (left_height + right_height) / 2

            # Calculates the absolute and percentage errors
            width_error = abs(estimated_width - actual_width)
            height_error = abs(estimated_height - actual_height)
            width_error_percent = width_error / actual_width * 100
            height_error_percent = height_error / actual_height * 100

            # Stores a temporary snapshot of the completed measurement
            clear_live_result()
            st.session_state.live_result = {
                "Image": uploaded_image.name,
                "Distance_cm": distance,
                "Actual_Width_cm": actual_width,
                "Estimated_Width_cm": estimated_width,
                "Width_Error_cm": width_error,
                "Width_Error_Percent": width_error_percent,
                "Actual_Height_cm": actual_height,
                "Estimated_Height_cm": estimated_height,
                "Height_Error_cm": height_error,
                "Height_Error_Percent": height_error_percent
            }

    if "result_notice" in st.session_state:
        st.success(st.session_state.pop("result_notice"))

    if "live_result" in st.session_state:
        result = st.session_state.live_result

        # Displays the measurement results
        st.subheader("Measurement Results")
        width_column, height_column = st.columns(2)
        with width_column:
            st.metric("Estimated Width (cm)", f"{result['Estimated_Width_cm']:.2f}")
            st.metric("Actual Width (cm)", f"{result['Actual_Width_cm']:.2f}")
            st.metric("Width Error (cm)", f"{result['Width_Error_cm']:.2f}")
            st.metric("Width Error (%)", f"{result['Width_Error_Percent']:.2f}")
        with height_column:
            st.metric("Estimated Height (cm)", f"{result['Estimated_Height_cm']:.2f}")
            st.metric("Actual Height (cm)", f"{result['Actual_Height_cm']:.2f}")
            st.metric("Height Error (cm)", f"{result['Height_Error_cm']:.2f}")
            st.metric("Height Error (%)", f"{result['Height_Error_Percent']:.2f}")

        st.subheader("Save Measurement")
        try:
            existing_rows = read_validation_rows()
        except (OSError, ValueError, csv.Error, TypeError) as error:
            st.error(f"Validation trials could not be loaded: {error}")
        else:
            if existing_rows:
                selected_trial = st.selectbox(
                    "Validation Trial", [row["Trial"] for row in existing_rows],
                    key="selected_validation_trial", on_change=clear_trial_confirmation
                )
                st.write(f"Validation Trial {selected_trial} will be replaced.")
                confirmed = st.checkbox(
                    "I understand this will replace the selected validation trial.",
                    key="replace_trial_confirmed"
                )
                if st.button("Update Validation Trial", disabled=not confirmed) and confirmed:
                    try:
                        update_validation_trial(selected_trial, result)
                    except (OSError, ValueError, csv.Error, TypeError) as error:
                        st.error(f"Validation trial could not be updated: {error}")
                    else:
                        st.session_state.pop("live_result", None)
                        st.session_state.result_notice = (
                            f"Validation Trial {selected_trial} updated successfully."
                        )
                        st.rerun()
            else:
                st.info("There are no existing validation trials to replace.")

        st.button("Discard Result", on_click=discard_result)


# Displays the validation results
st.divider()
st.subheader("Validation Results")

# Reads the validation results
try:
    validation_rows = read_validation_rows()
except FileNotFoundError:
    st.error("Validation results could not be loaded: measurement_results.csv was not found.")
except (OSError, ValueError, csv.Error, TypeError) as error:
    st.error(f"Validation results could not be loaded: {error}")
else:
    measurement_count = len(validation_rows)
    st.write(
        f"The measurement method was validated using {measurement_count} object "
        "measurements taken at camera-to-object distances greater than 2 meters."
    )
    st.write(f"Number of validation measurements: {measurement_count}")

    if not validation_rows:
        st.info("The validation CSV contains no measurements.")
    else:
        # Calculates the error statistics
        statistics = {
            "Statistic": [
                "Mean Absolute Error",
                "Mean Percentage Error",
                "Standard Deviation of Percentage Error",
                "RMSE"
            ]
        }
        for dimension in ("Width", "Height"):
            errors = np.array([
                float(row[f"{dimension}_Error_cm"]) for row in validation_rows
            ])
            percent_errors = np.array([
                float(row[f"{dimension}_Error_Percent"]) for row in validation_rows
            ])
            statistics[dimension] = [
                f"{np.mean(errors):.2f} cm",
                f"{np.mean(percent_errors):.2f} %",
                f"{np.std(percent_errors):.2f} %",
                f"{np.sqrt(np.mean(errors ** 2)):.2f} cm"
            ]

        st.table(statistics)

    # Displays the individual validation measurements
    with st.expander("View Validation Measurements"):
        st.dataframe(validation_rows, hide_index=True)
