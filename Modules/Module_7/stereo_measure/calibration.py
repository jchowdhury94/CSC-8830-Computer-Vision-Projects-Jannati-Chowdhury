"""Model B: load the stored 30-pair stereo calibration, or reproduce it from the corrected checkerboard corners."""
import cv2
import numpy as np

from .config import (CALIB_PAIR_IDS, CORNERS_DIR, IMAGE_SIZE, MODEL_B_NPZ, PATTERN, SQUARE_MM)

# k1, k2 radial distortion only (no k3, no tangential); stereo step keeps the per-camera intrinsics fixed
MONO_FLAGS = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
STEREO_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-7)


def object_points():
    """Checkerboard frame in mm, row-major to match OpenCV corner order (x along 8 columns, y along 6 rows)."""
    g = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    g[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    return g


def load_corners(side, pair_id):
    return np.loadtxt(CORNERS_DIR / f"{side}_{pair_id:02d}_corners.csv", delimiter=",").astype(np.float32)


def load_model_b(path=MODEL_B_NPZ):
    """Stored Model B as a dict: K1, D1 (left), K2, D2 (right), R, T, F, plus rms/baseline for reference."""
    z = np.load(path)
    return dict(K1=z["K_left"], D1=z["D_left"], K2=z["K_right"], D2=z["D_right"], R=z["R"], T=z["T"], E=z["E"], F=z["F"],
                stereo_rms=float(z["rms"]), rms_left=float(z["rms_left"]), rms_right=float(z["rms_right"]),
                baseline_mm=float(z["baseline_mm"]), image_size=tuple(int(v) for v in z["image_size"]),
                flags=int(z["flags"]), pairs=[int(v) for v in z["pairs"]])


def calibrate_model_b():
    """Recalibrate from the 30 corrected corner pairs (nothing is stored). Same procedure that produced the stored Model B."""
    left = [load_corners("left", i) for i in CALIB_PAIR_IDS]
    right = [load_corners("right", i) for i in CALIB_PAIR_IDS]
    obj = [object_points()] * len(CALIB_PAIR_IDS)
    rms_l, K1, D1, _, _ = cv2.calibrateCamera(obj, left, IMAGE_SIZE, None, None, flags=MONO_FLAGS)
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(obj, right, IMAGE_SIZE, None, None, flags=MONO_FLAGS)
    out = cv2.stereoCalibrateExtended(obj, left, right, K1, D1, K2, D2, IMAGE_SIZE, None, None,
                                      flags=cv2.CALIB_FIX_INTRINSIC, criteria=STEREO_CRITERIA)
    rms, _, _, _, _, R, T, E, F, _, _, _ = out
    return dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F, stereo_rms=float(rms), rms_left=float(rms_l),
                rms_right=float(rms_r), baseline_mm=float(np.linalg.norm(T)), image_size=IMAGE_SIZE,
                flags=int(MONO_FLAGS), pairs=list(CALIB_PAIR_IDS))
