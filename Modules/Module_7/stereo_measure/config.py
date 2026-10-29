"""Paths and constants for the Module 7 Model B (30-pair) stereo measurement pipeline.

Left camera = iPhone Air, right camera = iPhone 17 Pro Max. All 3D coordinates are in the LEFT camera frame, millimetres,
OpenCV axes (x right, y down, z forward); the right camera is X_R = R @ X_L + T.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CALIB_PHOTO_DIR = ROOT / "data" / "calibration"            # left/air_N.jpeg, right/pro_N.jpeg (N = 1..30)
NOTEBOOK_PHOTO_DIR = ROOT / "data" / "object_measurement"  # left/air_imageN.jpeg, right/pro_imageN.jpeg (N = 1..3)
CORNERS_DIR = ROOT / "calibration" / "corrected_corners"   # {left,right}_NN_corners.csv, 48 checkerboard corners per photo
MODEL_B_NPZ = ROOT / "calibration" / "model_b" / "stereo_calibration_30pairs.npz"
SELECTION_DIR = ROOT / "selections"                        # pair_{1,2,3}_corners.json: final approved notebook corners
REFERENCE_JSON = ROOT / "reference" / "expected_model_b_results.json"
OUTPUT_DIR = ROOT / "outputs"

PATTERN = (8, 6)             # internal checkerboard corners (cols, rows)
SQUARE_MM = 20.0
IMAGE_SIZE = (4284, 5712)    # (width, height) after EXIF orientation
CALIB_PAIR_IDS = list(range(1, 31))
CALIB_PREFIX = {"left": "air", "right": "pro"}

NOTEBOOK_PAIR_IDS = (1, 2, 3)
CORNER_NAMES = ("TL", "TR", "BR", "BL")
# edge name -> (start corner index, end corner index) into CORNER_NAMES
EDGES = {"top_width": (0, 1), "bottom_width": (3, 2), "left_height": (0, 3), "right_height": (1, 2)}
GT_WIDTH_MM, GT_HEIGHT_MM = 305.0, 215.0   # notebook including red spine; used only to report errors, never to scale anything
