"""Load the saved Part B experiment shown on the results page.

The verified correspondence coordinates in output/revised/ are the only input. Every displayed quantity is
computed from them by the shared sfm_pipeline.reconstruct; nothing is written here.

Imported by Pages/app_module6.py. Inputs: output/revised/part_b_correspondences.csv and the measured cover
dimensions in correspondence_selection.json. Requires NumPy and OpenCV.
"""
from pathlib import Path
import csv
import json

import numpy as np

from . import sfm_pipeline as sfm

BASE = Path(__file__).resolve().parent
REVISED = BASE / "output" / "revised"
PHOTOS = BASE / "submission_images"
PHOTO_NAMES = ("IMG_7596.jpeg", "IMG_7597.jpeg", "IMG_7598.jpeg", "IMG_7603.jpeg")


def load_tracks():
    """The saved (4, 8, 2) verified coordinates in full-resolution upright pixels."""
    with (REVISED / "part_b_correspondences.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if tuple(r["point_id"] for r in rows) != sfm.POINT_IDS:
        raise ValueError("Unexpected correspondence order")
    return np.array([[[float(r[f"view{v}_x"]), float(r[f"view{v}_y"])] for r in rows] for v in range(1, 5)])


def load():
    selection = json.loads((BASE / "correspondence_selection.json").read_text(encoding="utf-8"))
    width = float(selection["book_dimensions_cm"]["width"])
    height = float(selection["book_dimensions_cm"]["height"])
    result = sfm.reconstruct(load_tracks(), width, height)
    return dict(
        width=width, height=height, result=result, cm=result.cm, book=result.frame["book"], scale=result.scale,
        mean_width=result.unscaled_lengths["mean_width"], mean_height=result.unscaled_lengths["mean_height"],
        measurements=result.measurements,
    )
