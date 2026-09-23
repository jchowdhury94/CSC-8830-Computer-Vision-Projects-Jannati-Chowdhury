import csv
import numpy as np

# Defines the measurement results file
RESULTS_FILE = "measurement_results.csv"

# Stores the error values
width_errors = []
height_errors = []
width_percent_errors = []
height_percent_errors = []

# Reads the measurement results
with open(RESULTS_FILE, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        width_errors.append(
            float(row["Width_Error_cm"])
        )

        height_errors.append(
            float(row["Height_Error_cm"])
        )

        width_percent_errors.append(
            float(row["Width_Error_Percent"])
        )

        height_percent_errors.append(
            float(row["Height_Error_Percent"])
        )

# Converts the lists to NumPy arrays
width_errors = np.array(width_errors)
height_errors = np.array(height_errors)

width_percent_errors = np.array(
    width_percent_errors
)

height_percent_errors = np.array(
    height_percent_errors
)

# Calculates mean absolute errors
mean_width_error = np.mean(width_errors)
mean_height_error = np.mean(height_errors)

# Calculates mean percentage errors
mean_width_percent_error = np.mean(
    width_percent_errors
)

mean_height_percent_error = np.mean(
    height_percent_errors
)

# Calculates standard deviations of percentage errors
std_width_percent_error = np.std(
    width_percent_errors
)

std_height_percent_error = np.std(
    height_percent_errors
)

# Calculates RMSE
rmse_width = np.sqrt(
    np.mean(width_errors ** 2)
)

rmse_height = np.sqrt(
    np.mean(height_errors ** 2)
)

# Displays the results
print("\n--- Error Statistics ---")

print("\nNumber of measurements:")
print(len(width_errors))

print("\nWidth Statistics")
print(
    "Mean Absolute Error:",
    round(mean_width_error, 2),
    "cm"
)
print(
    "Mean Percentage Error:",
    round(mean_width_percent_error, 2),
    "%"
)
print(
    "Standard Deviation:",
    round(std_width_percent_error, 2),
    "%"
)
print(
    "RMSE:",
    round(rmse_width, 2),
    "cm"
)

print("\nHeight Statistics")
print(
    "Mean Absolute Error:",
    round(mean_height_error, 2),
    "cm"
)
print(
    "Mean Percentage Error:",
    round(mean_height_percent_error, 2),
    "%"
)
print(
    "Standard Deviation:",
    round(std_height_percent_error, 2),
    "%"
)
print(
    "RMSE:",
    round(rmse_height, 2),
    "cm"
)