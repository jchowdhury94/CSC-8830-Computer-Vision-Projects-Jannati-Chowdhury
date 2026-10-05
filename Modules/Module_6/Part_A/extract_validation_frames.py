"""Extract original Video 1 frames for a known CSV tracking transition."""

import csv
import math
from pathlib import Path

import cv2
import numpy as np


PART_A = Path(__file__).resolve().parent
OUTPUT = PART_A / "output"
VIDEO = PART_A / "input" / "video_1_horizontal.MOV"
TRACKING_CSV = OUTPUT / "video_1_tracking.csv"
CSV_FRAME = 485
FEATURE_ID = 14
COLOR = (0, 255, 255)  # Consistent yellow annotations in BGR order.


def draw_text(image, text, origin, scale=0.6):
    """A small dark background keeps yellow text readable."""
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 3, y - height - 3),
                  (x + width + 3, y + baseline + 3), (0, 0, 0), -1)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                scale, COLOR, 1, cv2.LINE_AA)


def annotate(frame, x, y):
    annotated = frame.copy()
    # Round only for raster drawing; retain CSV floats for all numerical checks.
    center = (round(x), round(y))
    cv2.circle(annotated, center, 12, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(annotated, center, 12, COLOR, 2, cv2.LINE_AA)
    labels = (f"Feature {FEATURE_ID}", f"({x:.8f}, {y:.8f})")
    text_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
                     for label in labels)
    height, width = frame.shape[:2]
    label_x = max(8, min(center[0] + 20, width - text_width - 8))
    label_y = max(24, min(center[1] - 28, height - 36))
    for label, baseline in zip(labels, (label_y, label_y + 26)):
        draw_text(annotated, label, (label_x, baseline))
    return annotated


def main():
    # Each CSV row identifies one successfully tracked feature transition.
    with TRACKING_CSV.open(newline="", encoding="utf-8") as csv_file:
        matches = [row for row in csv.DictReader(csv_file)
                   if int(row["frame"]) == CSV_FRAME and int(row["feature_id"]) == FEATURE_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one matching CSV row; found {len(matches)}.")
    row = matches[0]
    values = {key: float(row[key]) for key in
              ("old_x", "old_y", "new_x", "new_y", "u", "v", "magnitude")}
    if not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("The CSV row contains nonfinite coordinates or motion values.")
    old_x, old_y = values["old_x"], values["old_y"]
    new_x, new_y = values["new_x"], values["new_y"]
    # optical_flow.py starts at frame 1 and increments before writing a transition.
    # CSV frame N means old one-based frame N-1 -> new one-based frame N.
    # Thus their zero-based OpenCV indices are N-2 and N-1.
    old_index, new_index = CSV_FRAME - 2, CSV_FRAME - 1
    if old_index < 0 or new_index != old_index + 1:
        raise RuntimeError("The requested transition does not identify consecutive frames.")
    calculated = (new_x - old_x, new_y - old_y, math.hypot(new_x - old_x, new_y - old_y))
    for key, expected in zip(("u", "v", "magnitude"), calculated):
        # Eight-decimal CSV rounding can introduce tiny differences on recomputation.
        if not math.isclose(values[key], expected, rel_tol=0, abs_tol=2e-8):
            raise RuntimeError(f"CSV {key} is inconsistent with the coordinates.")

    capture = cv2.VideoCapture(str(VIDEO))
    old_frame = new_frame = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open original video: {VIDEO}")
        # Decode sequentially from the start to avoid approximate codec seeking.
        for index in range(new_index + 1):
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Cannot decode original video frame index {index}.")
            if index == old_index:
                old_frame = frame.copy()
            if index == new_index:
                new_frame = frame.copy()
    finally:
        capture.release()

    for frame, x, y in ((old_frame, old_x, old_y), (new_frame, new_x, new_y)):
        height, width = frame.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            raise RuntimeError("A feature coordinate lies outside its original frame.")
    old_annotated = annotate(old_frame, old_x, old_y)
    new_annotated = annotate(new_frame, new_x, new_y)

    # Keep both original images at full resolution; add title and summary bands.
    height, width = old_frame.shape[:2]
    comparison = np.zeros((height + 160, width * 2, 3), dtype=np.uint8)
    comparison[50:50 + height, :width] = old_annotated
    comparison[50:50 + height, width:] = new_annotated
    draw_text(comparison, f"Old Frame: {old_index + 1} (index {old_index})", (16, 32), 0.8)
    draw_text(comparison, f"New Frame: {new_index + 1} (index {new_index})", (width + 16, 32), 0.8)
    summary = (
        f"Feature ID: {FEATURE_ID} | CSV frame: {CSV_FRAME}",
        f"Old: ({row['old_x']}, {row['old_y']}) | New: ({row['new_x']}, {row['new_y']})",
        f"u: {row['u']} px | v: {row['v']} px | magnitude: {row['magnitude']} px",
    )
    for line_number, text in enumerate(summary):
        draw_text(comparison, text, (16, height + 80 + 28 * line_number))

    images = {
        "video_1_validation_frame_old.png": old_frame,
        "video_1_validation_frame_new.png": new_frame,
        "video_1_validation_frame_old_annotated.png": old_annotated,
        "video_1_validation_frame_new_annotated.png": new_annotated,
        "video_1_validation_comparison.png": comparison,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, image in images.items():
        path = OUTPUT / filename
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Cannot save image: {path}")

    print(f"CSV frame: {CSV_FRAME}; feature ID: {FEATURE_ID}")
    print(f"Original frames: {old_index + 1} -> {new_index + 1} (one-based)")
    print(f"OpenCV indices: {old_index} -> {new_index} (zero-based, consecutive)")
    print(f"Old coordinate: ({row['old_x']}, {row['old_y']})")
    print(f"New coordinate: ({row['new_x']}, {row['new_y']})")
    print(f"u: {row['u']}; v: {row['v']}; magnitude: {row['magnitude']} pixels")
    print("Checks passed: in-frame coordinates; CSV motion agrees within rounding precision.")
    print("Inspect the clean and annotated images to verify feature identity visually.")
    for filename in images:
        print(OUTPUT / filename)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, cv2.error) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
