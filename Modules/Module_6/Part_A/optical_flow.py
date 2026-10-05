"""Track manually selected soda-can features with sparse Lucas-Kanade flow.

Run from the repository root:
    .venv/bin/python Modules/Module_6/Part_A/optical_flow.py --video 1
    .venv/bin/python Modules/Module_6/Part_A/optical_flow.py --video 2

Frame numbers are one-based; CSV frame 2 describes motion from frame 1 to 2.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PART_A = Path(__file__).resolve().parent
INPUT_VIDEOS = {1: "video_1_horizontal.MOV", 2: "video_2_diagonal.MOV"}

FEATURE_PARAMS = dict(maxCorners=50, qualityLevel=0.3, minDistance=7, blockSize=7)
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
)
TRACK_COLOR = (0, 255, 255)  # Yellow in OpenCV's BGR color order.
CSV_COLUMNS = ["frame", "feature_id", "old_x", "old_y", "new_x", "new_y", "u", "v", "magnitude"]


def annotate(frame, frame_number, points):
    """Add compact markers and readable frame/feature counts."""
    for point in points.reshape(-1, 2):
        cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, TRACK_COLOR, -1)
    labels = (f"Frame: {frame_number}", f"Tracked features: {len(points)}")
    for label, y in zip(labels, (28, 56)):
        cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACK_COLOR, 1)


def main():
    parser = argparse.ArgumentParser(description="Track soda-can features in one Part A video.")
    parser.add_argument("--video", type=int, choices=(1, 2), default=1,
                        help="video to process: 1 (horizontal, default) or 2 (diagonal)")
    args = parser.parse_args()
    input_video = PART_A / "input" / INPUT_VIDEOS[args.video]
    output_video = PART_A / "output" / f"video_{args.video}_optical_flow.mp4"
    output_csv = PART_A / "output" / f"video_{args.video}_tracking.csv"
    capture = cv2.VideoCapture(str(input_video))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_video}")
        success, first_frame = capture.read()
        if not success or first_frame is None:
            raise RuntimeError("Cannot read the video's first frame.")

        height, width = first_frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            print("Warning: input FPS is unavailable; using 30 FPS.", flush=True)
            fps = 30.0

        print("Draw a rectangle around the soda can on the FIRST FRAME.", flush=True)
        print("Press ENTER or SPACE to confirm; press C to cancel.", flush=True)
        window = "Select soda-can ROI"
        try:
            x, y, roi_width, roi_height = cv2.selectROI(
                window, first_frame, showCrosshair=True, fromCenter=False
            )
        finally:
            cv2.destroyAllWindows()
        if (roi_width <= 0 or roi_height <= 0 or x < 0 or y < 0
                or x + roi_width > width or y + roi_height > height):
            raise RuntimeError("ROI selection was cancelled or invalid. Select a nonempty box inside the frame.")

        # Grayscale supplies intensity values for corner detection and tracking.
        previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        # The mask limits initial feature detection to the manually selected can.
        mask = np.zeros_like(previous_gray)
        mask[y:y + roi_height, x:x + roi_width] = 255
        # Shi-Tomasi finds strong corners whose local intensity varies in two directions.
        points = cv2.goodFeaturesToTrack(previous_gray, mask=mask, **FEATURE_PARAMS)
        if points is None or len(points) == 0:
            raise RuntimeError("No features detected inside the ROI. Try a clearer ROI or adjust FEATURE_PARAMS.")
        initial_count = len(points)
        feature_ids = np.arange(initial_count)

        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video writer: {output_video}")

        frame_number = 1
        annotate(first_frame, frame_number, points)
        writer.write(first_frame)
        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_COLUMNS)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame_number += 1
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if len(points):
                    # Pyramidal LK follows the same corners between consecutive frames,
                    # using several image scales to handle larger pixel movements.
                    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        previous_gray, current_gray, points, None, **LK_PARAMS
                    )
                    if next_points is None or status is None:
                        valid = np.zeros(len(points), dtype=bool)
                        next_points = np.empty_like(points)
                    else:
                        valid = (status.reshape(-1) == 1) & np.isfinite(next_points.reshape(-1, 2)).all(axis=1)
                    old_points = points.reshape(-1, 2)[valid]
                    new_points = next_points.reshape(-1, 2)[valid]
                    feature_ids = feature_ids[valid]
                    for feature_id, old, new in zip(feature_ids, old_points, new_points):
                        # u is horizontal displacement; v is vertical displacement.
                        # Image x increases rightward and image y increases downward.
                        u, v = new.astype(np.float64) - old.astype(np.float64)
                        values = [*old, *new, u, v, np.hypot(u, v)]
                        csv_writer.writerow([frame_number, int(feature_id), *[f"{value:.8f}" for value in values]])
                        cv2.line(frame, tuple(np.rint(old).astype(int)),
                                 tuple(np.rint(new).astype(int)), TRACK_COLOR, 1, cv2.LINE_AA)
                    # Remove failed tracks permanently; never detect replacement features.
                    points = new_points.reshape(-1, 1, 2)
                    if len(points) == 0:
                        print(f"Warning: all features lost at frame {frame_number}; saving remaining frames with zero tracks.", flush=True)
                annotate(frame, frame_number, points)
                writer.write(frame)
                previous_gray = current_gray

        # Finalize the MP4 before reporting its location.
        writer.release()
        print(f"Total frames processed: {frame_number}")
        print(f"Initial detected features: {initial_count}")
        print(f"Features still tracked at end: {len(points)}")
        print(f"Output video: {output_video}")
        print(f"Tracking CSV: {output_csv}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, cv2.error, OSError) as error:
        print(f"Error: {error}", flush=True)
        raise SystemExit(1)
