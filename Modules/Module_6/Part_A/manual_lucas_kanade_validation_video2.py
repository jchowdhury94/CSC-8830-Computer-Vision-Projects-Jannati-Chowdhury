"""Validate Video 2 with the same direct local LK calculation as Video 1."""

import csv
from pathlib import Path

import cv2
import numpy as np


PART_A = Path(__file__).resolve().parent
OUTPUT = PART_A / "output"
CSV_FRAME = 483
FEATURE_ID = 12
WINDOW_SIZE = 21
RESULTS_PATH = OUTPUT / "video_2_manual_lk_validation.txt"


def main():
    old_image = cv2.imread(str(OUTPUT / "video_2_validation_frame_old.png"))
    new_image = cv2.imread(str(OUTPUT / "video_2_validation_frame_new.png"))
    if old_image is None or new_image is None:
        raise RuntimeError("Cannot load both original validation frames.")
    if old_image.shape != new_image.shape:
        raise RuntimeError("The validation frames must have identical dimensions.")
    if WINDOW_SIZE < 3 or WINDOW_SIZE % 2 != 1:
        raise RuntimeError("WINDOW_SIZE must be an odd integer of at least 3.")

    with (OUTPUT / "video_2_tracking.csv").open(newline="", encoding="utf-8") as file:
        matches = [row for row in csv.DictReader(file)
                   if int(row["frame"]) == CSV_FRAME and int(row["feature_id"]) == FEATURE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one target CSV row; found {len(matches)}.")
    row = matches[0]
    # Before estimating flow, the CSV supplies only the old feature location.
    old_x, old_y = float(row["old_x"]), float(row["old_y"])
    if not np.isfinite([old_x, old_y]).all():
        raise RuntimeError("The old feature location is not finite.")
    center_x, center_y = round(old_x), round(old_y)
    radius = WINDOW_SIZE // 2
    height, width = old_image.shape[:2]
    if not (radius <= center_x < width - radius and radius <= center_y < height - radius):
        raise RuntimeError("The full calculation window does not fit inside the image.")

    # Brightness constancy: I(x,y,t) = I(x+dx,y+dy,t+dt).
    # Grayscale provides the scalar intensity needed by this equation; color
    # channels are not required. Float64 avoids unsigned subtraction/clipping.
    old_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    average = (old_gray + new_gray) / 2.0
    # Ix and Iy are 3x3 Sobel derivatives of the temporal average image.
    # A unit intensity ramp gives an unscaled Sobel response of 8, so scale=1/8
    # puts these derivatives in intensity-per-pixel units for displacement in px.
    ix = cv2.Sobel(average, cv2.CV_64F, 1, 0, ksize=3, scale=1.0 / 8.0)
    iy = cv2.Sobel(average, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8.0)
    # It is the intensity difference over one consecutive-frame interval.
    it = new_gray - old_gray
    window = (slice(center_y - radius, center_y + radius + 1),
              slice(center_x - radius, center_x + radius + 1))
    # Linearized brightness constancy: Ix*u + Iy*v + It = 0.
    # Local LK assumes neighboring pixels share approximately the same (u,v),
    # giving many equations in two unknowns: A [u v]^T approximately equals b.
    a = np.column_stack((ix[window].ravel(), iy[window].ravel()))
    b = -it[window].ravel()
    motion, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    manual_u, manual_v = motion
    manual_magnitude = np.hypot(manual_u, manual_v)
    # Round only the window center; preserve the original subpixel coordinate
    # when predicting the new feature position.
    predicted_x, predicted_y = old_x + manual_u, old_y + manual_v

    # A corner/textured region varies in two directions, yielding two nonzero
    # eigenvalues and better conditioning than a flat region or a simple edge.
    ata = a.T @ a
    eigenvalues = np.linalg.eigvalsh(ata)
    condition_number = np.linalg.cond(ata)

    # Only AFTER the independent intensity-based estimate, read CSV motion and
    # the OpenCV destination for comparison. They never enter A or b.
    csv_u, csv_v = float(row["u"]), float(row["v"])
    csv_new_x, csv_new_y = float(row["new_x"]), float(row["new_y"])
    csv_magnitude = float(row["magnitude"])
    if not np.isfinite([csv_u, csv_v, csv_new_x, csv_new_y, csv_magnitude]).all():
        raise RuntimeError("The CSV comparison values are not finite.")
    u_error, v_error = manual_u - csv_u, manual_v - csv_v
    position_error = np.hypot(predicted_x - csv_new_x, predicted_y - csv_new_y)
    actual_magnitude = np.hypot(csv_u, csv_v)
    # Image axes: +x points right, +y points down. Positive angles from +x
    # therefore turn toward downward motion. Wrap differences to [0, 180].
    manual_angle = np.degrees(np.arctan2(manual_v, manual_u))
    opencv_angle = np.degrees(np.arctan2(csv_v, csv_u))
    angle_difference = abs((manual_angle - opencv_angle + 180.0) % 360.0 - 180.0)
    relative_motion_error = np.hypot(u_error, v_error) / max(actual_magnitude, 1e-12)
    warnings = []
    if rank < 2:
        warnings.append("Warning: the local system is rank deficient; motion is not uniquely constrained.")
    if not np.isfinite(condition_number) or condition_number > 1e6:
        warnings.append("Warning: the normal matrix is poorly conditioned.")
    # Interpret direction and magnitude separately rather than hiding a discrepancy.
    direction_agrees = manual_u > 0 and manual_v > 0 and csv_u > 0 and csv_v > 0
    interpretation = (
        "Both methods support rightward/downward diagonal motion; assess numerical agreement using the reported errors."
        if direction_agrees else
        "The estimated direction does not fully agree with tracking; retain the discrepancy."
    )
    lines = [
        f"Feature ID: {FEATURE_ID}; CSV frame: {CSV_FRAME}",
        f"Original frame pair: {CSV_FRAME - 1} -> {CSV_FRAME} (one-based)",
        f"OpenCV frame indices: {CSV_FRAME - 2} -> {CSV_FRAME - 1} (zero-based)",
        f"Old coordinate: ({old_x:.8f}, {old_y:.8f})",
        f"Window: {WINDOW_SIZE} x {WINDOW_SIZE}; integer center: ({center_x}, {center_y})",
        "Derivatives: 3x3 Sobel on average grayscale, scale=1/8; It=new-old; intensities in [0,1].",
        f"Number of equations: {len(b)}; A shape: {a.shape}; b shape: {b.shape}",
        "A^T A:\n" + np.array2string(ata, precision=12),
        "Eigenvalues: " + np.array2string(eigenvalues, precision=12),
        f"Rank: {rank}; condition number of A^T A: {condition_number:.12f}",
        f"Manual u: {manual_u:.12f} px",
        f"Manual v: {manual_v:.12f} px",
        f"Manual magnitude: {manual_magnitude:.12f} px",
        f"Manual direction angle: {manual_angle:.12f} degrees",
        f"Predicted new coordinate: ({predicted_x:.12f}, {predicted_y:.12f})",
        f"OpenCV u: {csv_u:.8f} px; OpenCV v: {csv_v:.8f} px",
        f"OpenCV magnitude from CSV: {csv_magnitude:.8f} px",
        f"OpenCV direction angle: {opencv_angle:.12f} degrees",
        f"OpenCV new coordinate: ({csv_new_x:.8f}, {csv_new_y:.8f})",
        f"u error: {u_error:.12f} px; v error: {v_error:.12f} px",
        f"Position error: {position_error:.12f} px",
        f"Absolute angle difference: {angle_difference:.12f} degrees",
        f"Relative motion-vector error: {relative_motion_error:.6%}",
        interpretation,
        ("The local feature provides intensity variation in two independent directions (full-rank system)."
         if rank == 2 else "The local feature does not constrain both motion components independently."),
        f"Magnitude difference relative to OpenCV: {(manual_magnitude / max(actual_magnitude, 1e-12) - 1):.6%}",
        "OpenCV tracking is iterative and pyramidal; this is a single-level, one-step local linear least-squares estimate.",
        *warnings,
    ]
    report = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Results file: {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, cv2.error, np.linalg.LinAlgError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
