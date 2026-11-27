"""Redraw the Part A optical flow videos with motion arrows from the saved tracking data.

Run from the repository root:
    .venv/bin/python -m Modules.Module_6.Part_A.render_arrow_visualization

Tracking is not rerun: every dot and arrow comes from video_<n>_tracking.csv, drawn over the
original recording. Only video_<n>_optical_flow_arrows.mp4 is written.

Inputs: input/video_<n>_*.MOV (kept locally) and output/video_<n>_tracking.csv.
Requires NumPy, OpenCV and imageio-ffmpeg (repository requirements.txt).
"""

import csv
from collections import defaultdict

import cv2
import numpy as np

from Modules.Module_6.Part_A.optical_flow import (
    DOT_RADIUS, INPUT_VIDEOS, PART_A, annotate, draw_motion_arrows,
)
from Modules.Module_6.Part_A.visitor_tracking import H264Writer


def load_tracks(path):
    """Return {frame number: (old points, new points)} from a saved tracking CSV."""
    rows = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["frame"])].append([float(row[key]) for key in ("old_x", "old_y", "new_x", "new_y")])
    return {number: (np.float32(values)[:, :2], np.float32(values)[:, 2:]) for number, values in rows.items()}


def render(number):
    tracks = load_tracks(PART_A / "output" / f"video_{number}_tracking.csv")
    source = PART_A / "input" / INPUT_VIDEOS[number]
    output = PART_A / "output" / f"video_{number}_optical_flow_arrows.mp4"
    capture = cv2.VideoCapture(str(source))
    writer = None
    frame_number = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open input video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            if frame_number == 1:
                writer = H264Writer(output, frame.shape[1], frame.shape[0], fps)
                points = tracks[2][0]  # initial feature positions
            else:
                old, points = tracks.get(frame_number, (np.empty((0, 2)), np.empty((0, 2))))
                draw_motion_arrows(frame, old, points)
            annotate(frame, frame_number, points, dot_radius=DOT_RADIUS)
            writer.write(frame)
        if frame_number < max(tracks):
            raise RuntimeError(f"{source.name} ended at frame {frame_number}, before the tracking data.")
        writer.close()
    finally:
        capture.release()
        if writer is not None:
            writer.abort()
    print(f"{output.name}: {frame_number} frames at {fps:.3f} fps")


if __name__ == "__main__":
    for video in INPUT_VIDEOS:
        render(video)
