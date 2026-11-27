"""Add motion arrows to the Part A consecutive-frame validation images.

Run from the repository root:
    .venv/bin/python -m Modules.Module_6.Part_A.annotate_validation_frames

The right-hand frame is redrawn from the saved clean frame and the tracking CSV: the feature marker
at its measured position, the OpenCV displacement as an arrow, and an enlarged view of both. The
existing images are not changed; video_<n>_validation_comparison_arrows.png is written.

Inputs (in output/): video_<n>_validation_comparison.png, video_<n>_validation_frame_new.png and
video_<n>_tracking.csv. Requires NumPy and OpenCV (repository requirements.txt).
"""

import csv

import cv2
import numpy as np

from Modules.Module_6.Part_A.extract_validation_frames_video2 import COLOR, draw_text
from Modules.Module_6.Part_A.optical_flow import (
    ARROW_MAGNIFICATION, ARROW_THICKNESS, DOT_RADIUS, PART_A, draw_motion_arrows,
)

OUTPUT = PART_A / "output"
VALIDATIONS = {1: (485, 14), 2: (483, 12)}  # video: (CSV frame, feature id)
TITLE_BAND = 50  # height of the title band above the frames in the comparison image
CROP_WIDTH, CROP_HEIGHT, ZOOM = 90, 70, 8
INSET_X, INSET_Y = 40, 40  # top-left of the enlarged view, over blank wall in both videos
BLACK = (0, 0, 0)
# Heavier than one pixel in the enlarged view, in proportion to its larger arrow, so that it reads when scaled down.
INSET_ARROW_THICKNESS, INSET_DOT_RADIUS = 3, 6


def load_row(number):
    frame_number, feature_id = VALIDATIONS[number]
    with (OUTPUT / f"video_{number}_tracking.csv").open(newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle)
                   if int(row["frame"]) == frame_number and int(row["feature_id"]) == feature_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one CSV row for video {number}; found {len(matches)}.")
    return matches[0]


def outlined_arrow(image, old, new, thickness, zoom=1.0):
    """The video arrow with a dark outline, so that it stays visible on light backgrounds."""
    draw_motion_arrows(image, [old], [new], thickness + 2, BLACK, zoom)
    draw_motion_arrows(image, [old], [new], thickness, zoom=zoom)


def marker(image, center, radius):
    """A filled feature dot with a thin dark outline."""
    cv2.circle(image, center, radius + 1, BLACK, -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, COLOR, -1, cv2.LINE_AA)


def enlarged_view(clean, old, new):
    """Crop around the current position, enlarged; the arrow is enlarged by the same factor."""
    left, top = int(round(new[0])) - CROP_WIDTH // 2, int(round(new[1])) - CROP_HEIGHT // 2
    view = cv2.resize(clean[top:top + CROP_HEIGHT, left:left + CROP_WIDTH], None, fx=ZOOM, fy=ZOOM,
                      interpolation=cv2.INTER_CUBIC)

    def to_view(point):  # pixel centers map to the centers of the enlarged pixels
        return (point - (left, top) + 0.5) * ZOOM - 0.5

    current = to_view(new)
    outlined_arrow(view, to_view(old), current, INSET_ARROW_THICKNESS, ZOOM)
    center = tuple(int(c) for c in np.rint(current))
    marker(view, center, INSET_DOT_RADIUS)
    draw_text(view, f"{ZOOM}x enlarged view", (14, 36), 0.9)
    draw_text(view, f"Dot: measured position  Arrow: displacement x{ARROW_MAGNIFICATION}",
              (14, view.shape[0] - 18), 0.8)
    return view, (left, top)


def annotate_frame(clean, row, feature_id):
    old = np.array([float(row["old_x"]), float(row["old_y"])])
    new = np.array([float(row["new_x"]), float(row["new_y"])])
    frame = clean.copy()
    center = (round(new[0]), round(new[1]))

    view, (left, top) = enlarged_view(clean, old, new)
    height, width = view.shape[:2]
    # Outline the enlarged region and join it to the view.
    cv2.rectangle(frame, (left, top), (left + CROP_WIDTH, top + CROP_HEIGHT), BLACK, 4, cv2.LINE_AA)
    cv2.rectangle(frame, (left, top), (left + CROP_WIDTH, top + CROP_HEIGHT), COLOR, 2, cv2.LINE_AA)
    for color, thickness in ((BLACK, 4), (COLOR, 2)):
        cv2.line(frame, (left, top), (INSET_X + width // 2, INSET_Y + height), color, thickness, cv2.LINE_AA)
    frame[INSET_Y:INSET_Y + height, INSET_X:INSET_X + width] = view
    cv2.rectangle(frame, (INSET_X, INSET_Y), (INSET_X + width, INSET_Y + height), COLOR, 3)

    outlined_arrow(frame, old, new, ARROW_THICKNESS)
    marker(frame, center, DOT_RADIUS)  # drawn last: the measured position stays on top of the arrow
    # Labels sit above the enlarged region, clear of the arrow.
    labels = (f"Feature {feature_id}", f"({new[0]:.8f}, {new[1]:.8f})")
    for label, baseline in zip(labels, (center[1] - 74, center[1] - 48)):
        draw_text(frame, label, (center[0] + 20, baseline))
    return frame


def render(number):
    row = load_row(number)
    comparison = cv2.imread(str(OUTPUT / f"video_{number}_validation_comparison.png"))
    clean = cv2.imread(str(OUTPUT / f"video_{number}_validation_frame_new.png"))
    if comparison is None or clean is None:
        raise RuntimeError(f"Cannot load the saved validation images for video {number}.")
    height, width = clean.shape[:2]
    comparison[TITLE_BAND:TITLE_BAND + height, width:] = annotate_frame(clean, row, VALIDATIONS[number][1])
    path = OUTPUT / f"video_{number}_validation_comparison_arrows.png"
    if not cv2.imwrite(str(path), comparison):
        raise RuntimeError(f"Cannot save image: {path}")
    print(f"{path.name}: u = {row['u']} px, v = {row['v']} px")


if __name__ == "__main__":
    for video in VALIDATIONS:
        render(video)
