"""Measure a rectangle from user-supplied corner pairs with any stereo calibration model (no files read).

The mathematics is that of measurement.measure_pair: undistortion and triangulation (triangulation.py), 3D Euclidean edge lengths.
Only the warning thresholds depend on the calibration: a visitor calibration supplies its own epipolar residuals and board-distance
range; a model without that information (Model B, used for regression) keeps the fixed defaults below. Thresholds only ever
produce warnings, never errors.
"""
import numpy as np

from .config import CORNER_NAMES, EDGES
from .triangulation import to_right_camera, triangulate
from .user_calibration import _epipolar_px

MIN_EDGE_MM = 1.0
OPPOSITE_EDGE_WARN_RATIO = 1.25   # a rectangle's opposite edges should be close in length
# Defaults for models that carry no calibration metrics (Model B): fixed values chosen for that setup.
DEPTH_WARN_MM = (200.0, 3000.0)   # far outside the ~700-850 mm range Model B was validated at
EPIPOLAR_WARN_PX = 150.0          # the saved notebook corners reach ~51 px with Model B (stereo RMS 4.9 px)
# Calibration-derived thresholds (heuristics): scale with the calibration's own epipolar residual and board distances.
EPIPOLAR_FACTOR = 3.0             # warn above 3x the worst calibration epipolar residual ...
EPIPOLAR_FLOOR_FRACTION = 0.004   # ... but never below 0.4% of the longest image side (hand-picked corners are less exact)
DEPTH_RANGE_FACTORS = (0.5, 2.0)  # warn outside [0.5 x nearest, 2 x farthest] calibration board distance


def convex_quad(pts):
    """True if the 4 points, in order, form a convex quadrilateral (all turns the same way, none degenerate)."""
    p = np.asarray(pts, np.float64)
    cross = []
    for i in range(4):
        a, b = p[(i + 1) % 4] - p[i], p[(i + 2) % 4] - p[(i + 1) % 4]
        cross.append(a[0] * b[1] - a[1] * b[0])        # z of the 2D cross product
    return bool(all(c > 0 for c in cross) or all(c < 0 for c in cross))


def epipolar_distance_px(model, left_px, right_px):
    """Distance in right-image pixels from each right point to the epipolar line of its left point (F from Model B)."""
    F = model["F"]
    l = np.c_[np.asarray(left_px, np.float64), np.ones(len(left_px))] @ F.T       # lines a x + b y + c = 0 in the right image
    r = np.c_[np.asarray(right_px, np.float64), np.ones(len(right_px))]
    return np.abs((l * r).sum(1)) / np.hypot(l[:, 0], l[:, 1])


def thresholds_from_calibration(model):
    """Warning thresholds for `model`: derived from its epipolar residuals and board-distance range when it carries them
    (visitor calibrations, computed or loaded), otherwise the fixed defaults. Returns a dict (also reported in the result):
    epipolar_warn_px, epipolar_basis ('raw_right_image' | 'calibration_residual'), depth_range_mm (lo, hi) or None,
    depth_basis ('default' | 'calibration_board_distance') and source ('default' | 'calibration').
    """
    th = dict(source="default", epipolar_warn_px=EPIPOLAR_WARN_PX, epipolar_basis="raw_right_image",
              depth_range_mm=DEPTH_WARN_MM, depth_basis="default")
    stats = (model.get("metrics") or {}).get("epipolar_px")
    sizes = [model.get(k) for k in ("image_size_left", "image_size_right") if model.get(k)] or [model.get("image_size")]
    sizes = [size for size in sizes if size]
    if stats and sizes:
        max_side = max(max(int(v) for v in size) for size in sizes)
        th.update(source="calibration", epipolar_basis="calibration_residual",
                  epipolar_warn_px=max(EPIPOLAR_FLOOR_FRACTION * max_side, EPIPOLAR_FACTOR * float(stats["max"])))
    distance = (model.get("board") or {}).get("distance_mm")
    if distance and 0 < distance["min"] <= distance["max"]:
        th.update(source="calibration", depth_basis="calibration_board_distance",
                  depth_range_mm=(DEPTH_RANGE_FACTORS[0] * float(distance["min"]), DEPTH_RANGE_FACTORS[1] * float(distance["max"])))
    return th


def _resolve_thresholds(model, thresholds):
    th = thresholds_from_calibration(model)
    if thresholds is None:
        return th
    if not isinstance(thresholds, dict) or set(thresholds) - {"epipolar_warn_px", "epipolar_basis", "depth_range_mm"}:
        raise ValueError("thresholds must be a dict with keys from: epipolar_warn_px, epipolar_basis, depth_range_mm")
    th["source"] = "custom"
    if "epipolar_warn_px" in thresholds:
        v = thresholds["epipolar_warn_px"]
        if v is not None and not (np.isfinite(v) and v > 0):
            raise ValueError("epipolar_warn_px must be a positive number or None")
        th["epipolar_warn_px"] = v
    if "epipolar_basis" in thresholds:
        if thresholds["epipolar_basis"] not in ("raw_right_image", "calibration_residual"):
            raise ValueError("epipolar_basis must be 'raw_right_image' or 'calibration_residual'")
        th["epipolar_basis"] = thresholds["epipolar_basis"]
    if "depth_range_mm" in thresholds:
        r = thresholds["depth_range_mm"]
        if r is not None and not (len(r) == 2 and np.isfinite(r).all() and 0 < r[0] < r[1]):
            raise ValueError("depth_range_mm must be (low, high) with 0 < low < high, or None")
        th["depth_range_mm"] = None if r is None else (float(r[0]), float(r[1]))
        th["depth_basis"] = "custom"
    return th


def epipolar_residual_px(model, left_px, right_px):
    """Symmetric epipolar distance of each correspondence after removing lens distortion, in pixels. This is the same quantity
    the calibration reports for its own board corners, so the two are directly comparable."""
    return _epipolar_px(model["F"], model["K1"], model["D1"], model["K2"], model["D2"], np.asarray(left_px, np.float64),
                        np.asarray(right_px, np.float64))


def measure_points(model, left_px, right_px, truth=None, thresholds=None):
    """left_px, right_px: (4, 2) original-resolution pixels ordered TL, TR, BR, BL. truth: optional (width_mm, height_mm), either may be None.

    thresholds: optional dict overriding any of epipolar_warn_px, epipolar_basis, depth_range_mm; by default they come from
    thresholds_from_calibration(model). They affect warnings only.

    Returns a dict with X (mm, left-camera frame), Xr, corners, edges, summary, epipolar_px (distance in the right image, raw
    pixels), epipolar_residual_px (symmetric, after undistortion; comparable with the calibration's own metric), thresholds (as
    applied), errors (blocking) and warnings. Dimensions are only reported (summary['valid']) when errors is empty.
    """
    th = _resolve_thresholds(model, thresholds)
    left_px = np.asarray(left_px, np.float64).reshape(4, 2)
    right_px = np.asarray(right_px, np.float64).reshape(4, 2)
    X = triangulate(model, left_px, right_px)
    Xr = to_right_camera(model, X)
    errors, warnings = [], []

    if not (convex_quad(left_px) and convex_quad(right_px)):
        errors.append("The corners do not form a convex quadrilateral in order TL, TR, BR, BL in both images. "
                      "Check the selection order and that the same physical corners were chosen in both images.")
    finite = bool(np.isfinite(X).all())
    if not finite:
        errors.append("Triangulation produced non-finite coordinates.")
    elif not ((X[:, 2] > 0).all() and (Xr[:, 2] > 0).all()):
        errors.append("At least one corner triangulates behind a camera, so the left/right correspondences are inconsistent.")

    length = {}
    corners, edges = [], []
    if finite:
        for name, (a, b) in EDGES.items():
            d = X[b] - X[a]
            length[name] = float(np.sqrt(d @ d))                      # sqrt(dX^2 + dY^2 + dZ^2)
            edges.append(dict(edge=name, endpoints=f"{CORNER_NAMES[a]}-{CORNER_NAMES[b]}", dX_mm=d[0], dY_mm=d[1], dZ_mm=d[2],
                              length_3D_mm=length[name]))
        corners = [dict(corner=c, left_px_x=left_px[k, 0], left_px_y=left_px[k, 1], right_px_x=right_px[k, 0], right_px_y=right_px[k, 1],
                        X_mm=X[k, 0], Y_mm=X[k, 1], Z_depth_mm=X[k, 2], dist_from_left_camera_mm=float(np.linalg.norm(X[k])),
                        Z_right_cam_mm=Xr[k, 2]) for k, c in enumerate(CORNER_NAMES)]
        if min(length.values()) < MIN_EDGE_MM:
            errors.append("An object edge is shorter than 1 mm, so the geometry is degenerate.")

    epi = epipolar_distance_px(model, left_px, right_px)
    epi_res = epipolar_residual_px(model, left_px, right_px)
    if th["epipolar_warn_px"] is not None:
        if th["epipolar_basis"] == "calibration_residual":
            if epi_res.max() > th["epipolar_warn_px"]:
                warnings.append(f"A corner pair is {epi_res.max():.0f} px from its epipolar line after lens-distortion correction, above "
                                f"the {th['epipolar_warn_px']:.0f} px guide derived from this calibration's own epipolar residuals "
                                "(a heuristic); the corners may not match the same physical points. The measurement was still calculated.")
        elif epi.max() > th["epipolar_warn_px"]:
            warnings.append(f"A right-image corner is {epi.max():.0f} px from the epipolar line of its left-image corner "
                            f"(threshold {th['epipolar_warn_px']:.0f} px); the corners may not match the same physical points.")

    summary = dict(valid=not errors)
    if finite and not errors:
        width = (length["top_width"] + length["bottom_width"]) / 2
        height = (length["left_height"] + length["right_height"]) / 2
        z = X[:, 2]
        summary.update(mean_Z_depth_mm=float(z.mean()), min_Z_mm=float(z.min()), max_Z_mm=float(z.max()), **length,
                       width_mm=width, height_mm=height)
        for a, b, label in (("top_width", "bottom_width", "top and bottom widths"), ("left_height", "right_height", "left and right heights")):
            if max(length[a], length[b]) / min(length[a], length[b]) > OPPOSITE_EDGE_WARN_RATIO:
                warnings.append(f"The {label} differ by more than {100 * (OPPOSITE_EDGE_WARN_RATIO - 1):.0f}%; "
                                "the object may not be rectangular or a corner may be misplaced.")
        lo_hi = th["depth_range_mm"]
        if lo_hi is not None and (z.mean() < lo_hi[0] or z.mean() > lo_hi[1]):
            if th["depth_basis"] == "calibration_board_distance":
                near, far = lo_hi[0] / DEPTH_RANGE_FACTORS[0], lo_hi[1] / DEPTH_RANGE_FACTORS[1]
                warnings.append(f"Mean Z-depth {z.mean():.0f} mm is more than a factor of 2 outside the board distances used in "
                                f"calibration ({near:.0f}-{far:.0f} mm); the calibration has not been checked at this distance, so "
                                "the estimate is less reliable. The measurement was still calculated.")
            elif th["depth_basis"] == "default":
                warnings.append(f"Mean Z-depth {z.mean():.0f} mm is far outside the ~700-850 mm range this calibration was tested at.")
            else:
                warnings.append(f"Mean Z-depth {z.mean():.0f} mm is outside the allowed range {lo_hi[0]:.0f}-{lo_hi[1]:.0f} mm.")
        width_truth, height_truth = truth if truth is not None else (None, None)
        for key, estimate, actual in (("width", width, width_truth), ("height", height, height_truth)):
            if actual is not None and actual > 0:
                summary[f"{key}_err_mm"] = estimate - actual                       # signed: estimate - actual
                summary[f"{key}_abs_err_mm"] = abs(estimate - actual)
                summary[f"{key}_err_pct"] = 100 * (estimate - actual) / actual
    return dict(left_px=left_px, right_px=right_px, X=X, Xr=Xr, corners=corners, edges=edges, summary=summary,
                epipolar_px=epi, epipolar_residual_px=epi_res, thresholds=th, errors=errors, warnings=warnings)
