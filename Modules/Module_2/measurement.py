import cv2
import numpy as np
import csv
import os

# Defines the folder containing measurement images
IMAGE_FOLDER = "measurement_images"

# Defines the file containing measurement results
RESULTS_FILE = "measurement_results.csv"

# Loads the saved camera calibration
calibration = np.load("camera_calibration.npz")

camera_matrix = calibration["camera_matrix"]
distortion_coefficients = calibration["distortion_coefficients"]

# Gets the calibrated focal lengths
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]


# Calculates the real-world length between two image points
def calculate_length(point1, point2, distance):

    u1, v1 = point1
    u2, v2 = point2

    delta_x = (u2 - u1) / fx
    delta_y = (v2 - v1) / fy

    length = distance * np.sqrt(
        delta_x ** 2 + delta_y ** 2
    )

    return length


# Gets the next trial number from the existing CSV file
def get_next_trial_number():

    if not os.path.exists(RESULTS_FILE):
        return 1

    with open(RESULTS_FILE, "r") as file:
        reader = csv.reader(file)
        rows = list(reader)

    if len(rows) <= 1:
        return 1

    return int(rows[-1][0]) + 1


# Saves an approved measurement to the CSV file
def save_measurement(
    image_path,
    distance,
    actual_width,
    estimated_width,
    width_error,
    width_error_percent,
    actual_height,
    estimated_height,
    height_error,
    height_error_percent
):

    file_exists = os.path.exists(RESULTS_FILE)
    trial_number = get_next_trial_number()

    with open(
        RESULTS_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        # Writes the column names only if the file is new
        if not file_exists:

            writer.writerow([
                "Trial",
                "Image",
                "Distance_cm",
                "Actual_Width_cm",
                "Estimated_Width_cm",
                "Width_Error_cm",
                "Width_Error_Percent",
                "Actual_Height_cm",
                "Estimated_Height_cm",
                "Height_Error_cm",
                "Height_Error_Percent"
            ])

        # Writes the approved measurement
        writer.writerow([
            trial_number,
            image_path,
            round(distance, 2),
            round(actual_width, 2),
            round(estimated_width, 2),
            round(width_error, 2),
            round(width_error_percent, 2),
            round(actual_height, 2),
            round(estimated_height, 2),
            round(height_error, 2),
            round(height_error_percent, 2)
        ])

    return trial_number


# Gets information for a new object
def get_object_information():

    while True:

        print("\n--- New Object ---")

        object_name = input(
            "\nEnter object name: "
        )

        actual_width = float(
            input("Enter actual width in cm: ")
        )

        actual_height = float(
            input("Enter actual height in cm: ")
        )

        print("\nObject:", object_name)
        print("Width:", actual_width, "cm")
        print("Height:", actual_height, "cm")

        correct = input(
            "\nIs this information correct? (y/n): "
        ).lower()

        if correct == "y":
            return (
                object_name,
                actual_width,
                actual_height
            )

        print("\nRe-enter the object information.")


# Gets the image and distance information
def get_measurement_information():

    while True:

        print("\n--- Distance Measurement ---")

        image_filename = input(
            "\nEnter image filename: "
        )

        distance = float(
            input(
                "Enter camera-to-object distance in cm: "
            )
        )

        print("\nImage:", image_filename)
        print("Distance:", distance, "cm")

        correct = input(
            "\nIs this information correct? (y/n): "
        ).lower()

        if correct == "y":

            image_path = os.path.join(
                IMAGE_FOLDER,
                image_filename
            )

            return (
                image_filename,
                image_path,
                distance
            )

        print("\nRe-enter the image and distance.")


# Measures an object in an image
def measure_image(
    image_filename,
    image_path,
    distance,
    object_name,
    actual_width,
    actual_height
):

    # Reads the image
    image = cv2.imread(image_path)

    # Checks that the image was loaded
    if image is None:
        print("\nImage could not be loaded.")
        return None

    # Corrects lens distortion
    undistorted = cv2.undistort(
        image,
        camera_matrix,
        distortion_coefficients
    )

    while True:

        # Stores the selected points
        points = []

        # Defines the display scale
        display_scale = 0.25

        # Resizes the image for display
        display_image = cv2.resize(
            undistorted,
            None,
            fx=display_scale,
            fy=display_scale
        )

        # Handles mouse clicks
        def select_point(event, x, y, flags, param):

            if (
                event == cv2.EVENT_LBUTTONDOWN
                and len(points) < 4
            ):

                # Converts displayed coordinates to full-resolution coordinates
                original_x = x / display_scale
                original_y = y / display_scale

                points.append(
                    (original_x, original_y)
                )

                # Draws the selected point
                cv2.circle(
                    display_image,
                    (x, y),
                    6,
                    (0, 0, 255),
                    -1
                )

                # Displays the point number
                cv2.putText(
                    display_image,
                    str(len(points)),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "Select Object",
                    display_image
                )

        # Creates the image window
        cv2.namedWindow("Select Object")

        # Connects mouse clicks to the selection function
        cv2.setMouseCallback(
            "Select Object",
            select_point
        )

        print(
            "\nOpening",
            image_filename + "..."
        )

        print("\nClick the four corners in this order:")
        print("1. TOP-LEFT")
        print("2. TOP-RIGHT")
        print("3. BOTTOM-RIGHT")
        print("4. BOTTOM-LEFT")

        # Displays the image until four points are selected
        while True:

            cv2.imshow(
                "Select Object",
                display_image
            )

            key = cv2.waitKey(1)

            # Stops after four points are selected
            if len(points) == 4:
                break

            # Allows the window to close with the Escape key
            if key == 27:
                break

        # Closes the image window
        cv2.destroyWindow("Select Object")

        # Processes the window-close event
        for _ in range(5):
            cv2.waitKey(1)

        # Checks that four points were selected
        if len(points) != 4:
            print("\nFour points were not selected.")
            return None

        top_left = points[0]
        top_right = points[1]
        bottom_right = points[2]
        bottom_left = points[3]

        # Calculates the width using the top edge
        top_width = calculate_length(
            top_left,
            top_right,
            distance
        )

        # Calculates the width using the bottom edge
        bottom_width = calculate_length(
            bottom_left,
            bottom_right,
            distance
        )

        # Averages the two width measurements
        estimated_width = (
            top_width + bottom_width
        ) / 2

        # Calculates the height using the left edge
        left_height = calculate_length(
            top_left,
            bottom_left,
            distance
        )

        # Calculates the height using the right edge
        right_height = calculate_length(
            top_right,
            bottom_right,
            distance
        )

        # Averages the two height measurements
        estimated_height = (
            left_height + right_height
        ) / 2

        # Calculates the absolute errors
        width_error = abs(
            estimated_width - actual_width
        )

        height_error = abs(
            estimated_height - actual_height
        )

        # Calculates the percentage errors
        width_error_percent = (
            width_error / actual_width
        ) * 100

        height_error_percent = (
            height_error / actual_height
        ) * 100

        # Displays the measurement results
        print("\n--- Measurement Results ---")

        print("\nObject:")
        print(object_name)

        print("\nDistance:")
        print(round(distance, 2), "cm")

        print("\nEstimated Width:")
        print(round(estimated_width, 2), "cm")

        print("\nActual Width:")
        print(round(actual_width, 2), "cm")

        print("\nWidth Error:")
        print(round(width_error, 2), "cm")

        print("\nWidth Percentage Error:")
        print(
            round(width_error_percent, 2),
            "%"
        )

        print("\nEstimated Height:")
        print(round(estimated_height, 2), "cm")

        print("\nActual Height:")
        print(round(actual_height, 2), "cm")

        print("\nHeight Error:")
        print(round(height_error, 2), "cm")

        print("\nHeight Percentage Error:")
        print(
            round(height_error_percent, 2),
            "%"
        )

        # Asks whether the measurement should be saved
        approve = input(
            "\nApprove and save this measurement? (y/n): "
        ).lower()

        if approve == "y":

            trial_number = save_measurement(
                image_path,
                distance,
                actual_width,
                estimated_width,
                width_error,
                width_error_percent,
                actual_height,
                estimated_height,
                height_error,
                height_error_percent
            )

            print(
                "\nTrial",
                trial_number,
                "saved to measurement_results.csv."
            )

            return True

        # Allows the same image to be measured again
        redo = input(
            "\nRedo corner selection for this image? (y/n): "
        ).lower()

        if redo != "y":
            return False


# Runs the measurement program
while True:

    # Gets the current object's information
    (
        object_name,
        actual_width,
        actual_height
    ) = get_object_information()

    # Processes distance measurements for the current object
    while True:

        (
            image_filename,
            image_path,
            distance
        ) = get_measurement_information()

        measure_image(
            image_filename,
            image_path,
            distance,
            object_name,
            actual_width,
            actual_height
        )

        another_distance = input(
            "\nAdd another distance measurement for "
            + object_name
            + "? (y/n): "
        ).lower()

        if another_distance != "y":
            break

    # Asks whether a different object should be measured
    print("\nWhat would you like to do?")
    print("1. Start a new object")
    print("2. Exit program")

    choice = input("\nEnter choice: ")

    if choice != "1":
        print("\nMeasurement program finished.")
        break