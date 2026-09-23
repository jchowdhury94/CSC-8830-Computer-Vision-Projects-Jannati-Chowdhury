import cv2
import glob
import numpy as np
import os

# Defines the number of inner checkerboard corners
CHECKERBOARD = (8, 6)

# Defines the physical size of each checkerboard square in millimeters
SQUARE_SIZE = 20.0

# Defines the criteria for refining corner locations
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# Creates the real-world coordinates of the checkerboard corners
object_points_template = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

object_points_template[:, :2] = (
    np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
    .T
    .reshape(-1, 2)
)

# Converts the checkerboard coordinates to millimeters
object_points_template *= SQUARE_SIZE

# Stores calibration data
object_points = []
image_points = []
successful_image_names = []

# Finds all calibration images
images = glob.glob("calibration_images/*.jpeg")

print("Number of images found:", len(images))

# Processes each calibration image
for image_path in images:

    # Reads the image
    image = cv2.imread(image_path)

    # Converts the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Finds the checkerboard corners
    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD
    )

    if found:

        # Refines the detected corner locations
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        # Stores the real-world coordinates
        object_points.append(object_points_template.copy())

        # Stores the detected image coordinates
        image_points.append(corners_refined)

        # Stores the successful image name
        successful_image_names.append(os.path.basename(image_path))

        print(image_path, "-> Corners detected ✓")

    else:
        print(image_path, "-> NOT detected ✗")


# Calibrates the camera
rms_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = (
    cv2.calibrateCamera(
        object_points,
        image_points,
        gray.shape[::-1],
        None,
        None
    )
)


# Displays the calibration results
print("\nSuccessful calibration images:", len(object_points))

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion Coefficients:")
print(distortion_coefficients)

print("\nCalibration RMS Error:")
print(rms_error)


# Stores all corner errors
all_errors = []

print("\nMean Reprojection Error Per Image:")

# Calculates the reprojection error for each image
for i in range(len(object_points)):

    # Projects the real-world points back onto the image
    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rotation_vectors[i],
        translation_vectors[i],
        camera_matrix,
        distortion_coefficients
    )

    # Reshapes the detected and projected points
    detected = image_points[i].reshape(-1, 2)
    projected = projected_points.reshape(-1, 2)

    # Calculates the pixel error for each corner
    distances = np.linalg.norm(
        detected - projected,
        axis=1
    )

    # Stores all corner errors
    all_errors.extend(distances)

    # Calculates the mean error for the image
    image_mean_error = np.mean(distances)

    print(
        successful_image_names[i],
        "->",
        round(image_mean_error, 4),
        "pixels"
    )


# Converts the errors to a NumPy array
all_errors = np.array(all_errors)

# Calculates overall error statistics
mean_reprojection_error = np.mean(all_errors)
calculated_rms_error = np.sqrt(np.mean(all_errors ** 2))

print("\nOverall Mean Reprojection Error:")
print(round(mean_reprojection_error, 4), "pixels")

print("\nCalculated RMS Reprojection Error:")
print(round(calculated_rms_error, 4), "pixels")


# Saves the calibration parameters
np.savez(
    "camera_calibration.npz",
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    rms_error=rms_error,
    checkerboard_columns=CHECKERBOARD[0],
    checkerboard_rows=CHECKERBOARD[1],
    square_size_mm=SQUARE_SIZE,
    successful_images=len(object_points)
)

print("\nCalibration saved to camera_calibration.npz")
