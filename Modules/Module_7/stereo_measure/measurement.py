"""Notebook measurement: corners -> Z-depths -> edge lengths -> width/height -> error against the 305 x 215 mm truth."""
import json

import numpy as np

from .config import (CORNER_NAMES, EDGES, GT_HEIGHT_MM, GT_WIDTH_MM, NOTEBOOK_PHOTO_DIR, SELECTION_DIR)
from .triangulation import to_right_camera, triangulate


def left_photo(pair_id):
    return NOTEBOOK_PHOTO_DIR / "left" / f"air_image{pair_id}.jpeg"


def right_photo(pair_id):
    return NOTEBOOK_PHOTO_DIR / "right" / f"pro_image{pair_id}.jpeg"


def load_selection(pair_id):
    """Final approved corners for a notebook pair -> (left (4, 2), right (4, 2)) float64 pixels, order TL, TR, BR, BL."""
    with open(SELECTION_DIR / f"pair_{pair_id}_corners.json") as f:
        d = json.load(f)
    assert d["confirmed"] is True and d["corner_order"] == list(CORNER_NAMES)
    return np.array(d["left_px"], np.float64), np.array(d["right_px"], np.float64)


def measure_pair(model, pair_id):
    """Triangulate one notebook pair and return corner coordinates/depths, edge lengths, width, height and errors."""
    left_px, right_px = load_selection(pair_id)
    X = triangulate(model, left_px, right_px)                       # mm, left-camera frame
    Xr = to_right_camera(model, X)
    finite = bool(np.isfinite(X).all())
    depth_positive = bool(finite and (X[:, 2] > 0).all() and (Xr[:, 2] > 0).all())

    corners = [dict(pair=pair_id, corner=c, left_px_x=left_px[k, 0], left_px_y=left_px[k, 1],
                    right_px_x=right_px[k, 0], right_px_y=right_px[k, 1], X_mm=X[k, 0], Y_mm=X[k, 1],
                    Z_depth_mm=X[k, 2], dist_from_left_camera_mm=float(np.linalg.norm(X[k])), Z_right_cam_mm=Xr[k, 2])
               for k, c in enumerate(CORNER_NAMES)]

    edges, length = [], {}
    for name, (a, b) in EDGES.items():
        d = X[b] - X[a]
        length[name] = float(np.sqrt(d @ d))                         # sqrt(dX^2 + dY^2 + dZ^2)
        edges.append(dict(pair=pair_id, edge=name, endpoints=f"{CORNER_NAMES[a]}-{CORNER_NAMES[b]}", dX_mm=d[0], dY_mm=d[1],
                          dZ_mm=d[2], length_3D_mm=length[name], length_ignoring_dZ_mm=float(np.hypot(d[0], d[1])),
                          truth_mm=GT_WIDTH_MM if "width" in name else GT_HEIGHT_MM))

    width = (length["top_width"] + length["bottom_width"]) / 2
    height = (length["left_height"] + length["right_height"]) / 2
    summary = dict(pair=pair_id, mean_Z_depth_mm=float(X[:, 2].mean()), min_Z_mm=float(X[:, 2].min()), max_Z_mm=float(X[:, 2].max()),
                   **length, width_mm=width, height_mm=height,
                   width_err_mm=width - GT_WIDTH_MM, width_err_pct=100 * (width - GT_WIDTH_MM) / GT_WIDTH_MM,
                   height_err_mm=height - GT_HEIGHT_MM, height_err_pct=100 * (height - GT_HEIGHT_MM) / GT_HEIGHT_MM,
                   all_finite=finite, depth_positive_both_cameras=depth_positive)
    return dict(pair=pair_id, left_px=left_px, right_px=right_px, X=X, Xr=Xr, corners=corners, edges=edges, summary=summary)
