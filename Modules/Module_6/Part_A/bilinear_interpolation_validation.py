"""A worked bilinear intensity example from Video 2's verified feature."""

import csv
import math
from pathlib import Path

import cv2
import numpy as np


OUTPUT = Path(__file__).resolve().parent / "output"
CSV_FRAME = 483
FEATURE_ID = 12
IMAGE_PATH = OUTPUT / "video_2_bilinear_interpolation.png"
TEXT_PATH = OUTPUT / "video_2_bilinear_interpolation.txt"


def text(image, label, origin, color=(255, 255, 255), scale=0.55):
    (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 3, y - height - 3),
                  (x + width + 3, y + baseline + 3), (25, 25, 25), -1)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    frame = cv2.imread(str(OUTPUT / "video_2_validation_frame_old.png"))
    if frame is None:
        raise RuntimeError("Cannot load the original Video 2 old validation frame.")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    with (OUTPUT / "video_2_tracking.csv").open(newline="", encoding="utf-8") as file:
        matches = [r for r in csv.DictReader(file)
                   if int(r["frame"]) == CSV_FRAME and int(r["feature_id"]) == FEATURE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching CSV row; found {len(matches)}.")
    row = matches[0]
    x, y = float(row["old_x"]), float(row["old_y"])
    if not math.isfinite(x) or not math.isfinite(y):
        raise RuntimeError("Feature coordinate is not finite.")
    # Images store samples at integer coordinates; flow can locate a feature
    # between samples. Array indexing is gray[y, x], not gray[x, y].
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = x0 + 1, y0 + 1
    height, width = gray.shape
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise RuntimeError("The four surrounding pixels are not all within the image.")
    i11, i21 = int(gray[y0, x0]), int(gray[y0, x1])
    i12, i22 = int(gray[y1, x0]), int(gray[y1, x1])
    dx, dy = x - x0, y - y0
    # Bilinear interpolation combines four samples. Closer samples receive
    # greater weights, and the weights sum to one (a convex combination).
    w11 = (1 - dx) * (1 - dy)
    w21 = dx * (1 - dy)
    w12 = (1 - dx) * dy
    w22 = dx * dy
    weight_sum = w11 + w21 + w12 + w22
    if not math.isclose(weight_sum, 1.0, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Bilinear weights do not sum to one.")
    manual = w11 * i11 + w21 * i21 + w12 * i12 + w22 * i22
    # Independent verification comes AFTER the explicit manual calculation.
    # Float32 sampling preserves fractional intensity rather than rounding to
    # uint8. OpenCV's float32 center can slightly round the CSV coordinate.
    verification = float(cv2.getRectSubPix(gray.astype(np.float32), (1, 1), (x, y))[0, 0])
    difference = abs(manual - verification)
    if not math.isfinite(verification):
        raise RuntimeError("OpenCV returned a nonfinite verification intensity.")
    # Sampling between pixels lets iterative LK evaluate intensities at its
    # fractional feature positions; it does not change the stored pixel values.
    lines = [
        f"Feature ID: {FEATURE_ID}; CSV frame: {CSV_FRAME}",
        f"Original old frame: {CSV_FRAME - 1} (one-based); OpenCV index: {CSV_FRAME - 2}",
        f"P = ({row['old_x']}, {row['old_y']})",
        f"Q11 = ({x0}, {y0}); I11 = {i11}",
        f"Q21 = ({x1}, {y0}); I21 = {i21}",
        f"Q12 = ({x0}, {y1}); I12 = {i12}",
        f"Q22 = ({x1}, {y1}); I22 = {i22}",
        f"dx = {dx:.12f}; dy = {dy:.12f}",
        f"w11 = {w11:.12f}; w21 = {w21:.12f}",
        f"w12 = {w12:.12f}; w22 = {w22:.12f}",
        f"Sum of weights = {weight_sum:.12f}",
        "I(x,y) = (1-dx)(1-dy)I11 + dx(1-dy)I21 + (1-dx)dy I12 + dx dy I22",
        f"I(x,y) = {w11:.12f}({i11}) + {w21:.12f}({i21}) + {w12:.12f}({i12}) + {w22:.12f}({i22})",
        f"Manual interpolated intensity = {manual:.12f}",
        f"OpenCV interpolated intensity = {verification:.12f}",
        f"Absolute difference = {difference:.12f}",
        "Intensities use the 0-255 grayscale range. Small verification differences can arise from float32 rounding.",
    ]

    # Show a recognizable original crop alongside a nearest-neighbor detail.
    # Pixel sample centers map to the centers of the enlarged pixel blocks.
    visual = np.full((820, 1180, 3), 25, dtype=np.uint8)
    left, right = max(0, x0 - 70), min(width, x0 + 71)
    top, bottom = max(0, y0 - 70), min(height, y0 + 71)
    crop = frame[top:bottom, left:right]
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    visual[95:95 + enlarged.shape[0], 20:20 + enlarged.shape[1]] = enlarged
    p_crop = (round(20 + (x - left + 0.5) * 3 - 0.5),
              round(95 + (y - top + 0.5) * 3 - 0.5))
    cv2.circle(visual, p_crop, 8, (0, 255, 255), 2, cv2.LINE_AA)
    text(visual, "P: Feature 12", (p_crop[0] + 12, p_crop[1] - 12), (0, 255, 255))
    text(visual, "Video 2: bilinear interpolation at a tracked subpixel point", (20, 32), scale=0.7)
    text(visual, "Original color crop (nearest-neighbor x3)", (20, 72))
    text(visual, "Grayscale pixel neighborhood (nearest-neighbor x160)", (510, 72))

    # Four-by-four neighborhood gives space around the four sample locations.
    detail_left, detail_top = max(0, x0 - 1), max(0, y0 - 1)
    detail = gray[detail_top:min(height, y1 + 2), detail_left:min(width, x1 + 2)]
    scale = 160
    detail = cv2.resize(detail, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    detail = cv2.cvtColor(detail, cv2.COLOR_GRAY2BGR)
    visual[95:95 + detail.shape[0], 510:510 + detail.shape[1]] = detail

    def location(px, py):
        return (round(510 + (px - detail_left + 0.5) * scale - 0.5),
                round(95 + (py - detail_top + 0.5) * scale - 0.5))

    q11, q21, q12, q22 = [location(px, py) for px, py in
                          ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    for a, b in ((q11, q21), (q21, q22), (q22, q12), (q12, q11)):
        cv2.line(visual, a, b, (0, 255, 255), 1, cv2.LINE_AA)
    for name, intensity, center, offset in (
        ("Q11", i11, q11, (-110, -25)), ("Q21", i21, q21, (12, -25)),
        ("Q12", i12, q12, (-110, 38)), ("Q22", i22, q22, (12, 38)),
    ):
        cv2.circle(visual, center, 5, (0, 255, 255), -1, cv2.LINE_AA)
        text(visual, f"{name}: {intensity}", (center[0] + offset[0], center[1] + offset[1]), (0, 255, 255))
    p_detail = location(x, y)
    cv2.drawMarker(visual, p_detail, (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
    text(visual, "P", (p_detail[0] + 10, p_detail[1] + 28), (255, 0, 255))
    text(visual, f"P = ({x:.8f}, {y:.8f}); dx = {dx:.8f}; dy = {dy:.8f}", (20, 762))
    text(visual, f"Manual intensity: {manual:.8f}; OpenCV: {verification:.8f}; difference: {difference:.8f}", (20, 796))
    if not cv2.imwrite(str(IMAGE_PATH), visual):
        raise RuntimeError(f"Cannot save visual: {IMAGE_PATH}")
    report = "\n".join(lines) + "\n"
    TEXT_PATH.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Visual: {IMAGE_PATH}")
    print(f"Results: {TEXT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, cv2.error) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
