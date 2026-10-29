"""General (un-rectified) stereo triangulation with Model B."""
import cv2
import numpy as np

P1_NORMALISED = np.hstack([np.eye(3), np.zeros((3, 1))])


def triangulate(model, left_px, right_px):
    """Matching pixel points -> 3D points (N, 3) in mm, LEFT camera frame; column 2 is the Z-depth along the left optical axis.

    Each pixel is undistorted with its own camera's K, D into normalised coordinates, then the two viewing rays are
    intersected with P1 = [I | 0] and P2 = [R | T]. Z is an output of that intersection; no distance is assumed.
    """
    x1 = cv2.undistortPoints(np.asarray(left_px, np.float64).reshape(-1, 1, 2), model["K1"], model["D1"]).reshape(-1, 2)
    x2 = cv2.undistortPoints(np.asarray(right_px, np.float64).reshape(-1, 1, 2), model["K2"], model["D2"]).reshape(-1, 2)
    P2 = np.hstack([model["R"], model["T"].reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1_NORMALISED, P2, x1.T, x2.T)
    with np.errstate(all="ignore"):
        return (Xh[:3] / Xh[3]).T


def to_right_camera(model, X_left):
    """Same points in the right camera frame (X_R = R X_L + T); column 2 is the depth seen by the right camera."""
    return X_left @ model["R"].T + model["T"].ravel()
