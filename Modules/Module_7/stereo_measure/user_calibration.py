"""Visitor stereo calibration core: checkerboard photos -> corners -> verified left/right correspondences -> K, D, R, T.

Independent of Model B and of the Streamlit app: nothing here reads or writes the stored calibration, and nothing is cached.
The solve reuses the verified Model B recipe (cv2.calibrateCamera per camera, then cv2.stereoCalibrateExtended with
CALIB_FIX_INTRINSIC) with a distortion model chosen per camera by held-out validation.

Pipeline
    process_calibration_photo(bytes, pattern)  one photo -> validated size, board corners (sub-pixel), thumbnail, status
    resolve_pair_orderings(pairs, pattern)     decide, per pair, how the right corners correspond to the left corners
    calibrate_stereo(left, right, ...)         calibration result dict (same keys as load_model_b) + metrics + warnings

Corner order convention: OpenCV row-major, ``pattern = (columns, rows)`` of INNER corners; board x runs along the columns.
All 3D units are millimetres and the world frame is the checkerboard (z = 0 plane).
"""
import hashlib
import io

import cv2
import numpy as np
from PIL import Image

from .calibration import MONO_FLAGS, STEREO_CRITERIA
from .triangulation import to_right_camera, triangulate

MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 50_000_000
MIN_SIDE_PX = 480
MIN_VALID_PAIRS = 8
RECOMMENDED_PAIRS = (15, 30)
DETECT_MAX_SIDE = 2800            # board detection runs on a copy whose long side is at most this; refinement is full-size
THUMBNAIL_SIDE = 320
ACCEPTED_FORMATS = ("JPEG", "PNG")

# Distortion models, simplest first. radial2 is the Model B model (k1, k2; no k3, no tangential terms).
DISTORTION_MODELS = {"radial2": MONO_FLAGS, "radial3": cv2.CALIB_ZERO_TANGENT_DIST, "full": 0}
CV_FOLDS = 4
CV_SIMPLICITY_TOLERANCE = 0.10    # heuristic: keep the simplest model whose held-out RMS is within 10% of the best

# Heuristic thresholds. They guide warnings only; none of them is a measure of object-size error.
GRID_RESIDUAL_MAX = 0.20          # max deviation of a detected corner row/column from a straight line, in squares
SUBPIX_SHIFT_MAX = 0.25           # max sub-pixel refinement shift, in squares
ORDER_RATIO_MIN = 3.0             # best candidate ordering must beat the runner-up by this factor to be accepted
MIN_PAIRS_TO_VERIFY = 3           # one planar board leaves the epipolar geometry undetermined, so it cannot verify anything
ORDER_SCORE_MAX = 0.02            # ... and agree with the epipolar geometry to this fraction of the long image side
RMS_MONO_WARN_PER_4000PX = 1.5
RMS_STEREO_WARN_PER_4000PX = 2.0
EPIPOLAR_WARN_PER_4000PX = 2.0
OUTLIER_FACTOR = 3.0

_ORIENT_OPS = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
               5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
               8: Image.Transpose.ROTATE_90}


class CalibrationPhotoError(ValueError):
    """A calibration photograph cannot be used (unreadable, too large, wrong format, ...)."""


class CalibrationFailed(RuntimeError):
    """OpenCV could not compute a calibration from the supplied corners."""


# ---------------------------------------------------------------- one photograph

def board_object_points(pattern, square_mm):
    """Inner corners of the board in mm (z = 0), row-major to match OpenCV's corner order."""
    cols, rows = pattern
    g = np.zeros((cols * rows, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_mm)
    return g


def _check_pattern(pattern):
    try:
        cols, rows = (int(v) for v in pattern)
    except (TypeError, ValueError) as error:
        raise ValueError("pattern must be (columns, rows) of inner corners") from error
    if cols < 3 or rows < 3 or cols != pattern[0] or rows != pattern[1]:
        raise ValueError("The checkerboard needs at least 3 x 3 inner corners (whole numbers).")
    return cols, rows


def inspect_photo(data, min_side=MIN_SIDE_PX):
    """Header-only validation. Returns (PIL image, exif orientation, oriented (w, h), exif summary); raises CalibrationPhotoError.

    min_side: smallest accepted image side in pixels (pass 1 to skip the check, as the measurement path does before comparing
    against the exact size its calibration expects).
    """
    if not data:
        raise CalibrationPhotoError("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise CalibrationPhotoError(f"The file is {len(data) / 2**20:.1f} MB; the limit is {MAX_FILE_BYTES // 2**20} MB per photo.")
    try:
        im = Image.open(io.BytesIO(data))
        fmt, size, exif = im.format, im.size, im.getexif()
    except Exception as error:   # PIL raises many types for corrupt, unsupported or decompression-bomb files
        raise CalibrationPhotoError(f"The file could not be read as an image ({type(error).__name__}).") from error
    if fmt not in ACCEPTED_FORMATS:
        raise CalibrationPhotoError(f"Unsupported format {fmt}. Use a JPG, JPEG or PNG file.")
    try:
        orientation = int(exif.get(274, 1))
    except (TypeError, ValueError):
        orientation = 1
    oriented = (size[1], size[0]) if orientation in (5, 6, 7, 8) else size
    megapixels = oriented[0] * oriented[1] / 1e6
    if oriented[0] * oriented[1] > MAX_PIXELS:
        raise CalibrationPhotoError(f"The image is {megapixels:.0f} megapixels; the limit is {MAX_PIXELS // 1_000_000} megapixels.")
    if min(oriented) < min_side:
        raise CalibrationPhotoError(f"The image is {oriented[0]} x {oriented[1]} px, too small for calibration (minimum side {min_side} px).")
    sub = exif.get_ifd(0x8769)
    summary = {"make": str(exif.get(271, "")).strip(), "model": str(exif.get(272, "")).strip(),
               "lens_model": str(sub.get(0xA434, "")).strip(), "orientation": orientation}
    for key, tag in (("focal_length_mm", 0x920A), ("focal_length_35mm", 0xA405)):
        try:
            summary[key] = float(sub[tag]) if tag in sub else None
        except (TypeError, ValueError):
            summary[key] = None
    return im, orientation, oriented, summary


def decode_gray(im, orientation, expected_size):
    """Grayscale pixels in the EXIF-oriented frame (orientation applied explicitly; never resized)."""
    if im.format == "JPEG":
        im.draft("L", im.size)                      # decode luminance only (about 3x less memory); scale stays 1
    g = im if im.mode == "L" else im.convert("L")
    op = _ORIENT_OPS.get(orientation)
    if op is not None:
        g = g.transpose(op)
    arr = np.ascontiguousarray(np.asarray(g))
    if (arr.shape[1], arr.shape[0]) != tuple(expected_size):
        raise CalibrationPhotoError("The decoded image size does not match its header.")
    return arr


def _grid_residual(corners, pattern, square_px):
    """Largest deviation of any detected corner row or column from a straight line, in squares."""
    cols, rows = pattern
    grid = corners.reshape(rows, cols, 2)
    worst = 0.0
    for line in list(grid) + list(grid.transpose(1, 0, 2)):
        centred = line - line.mean(0)
        normal = np.linalg.svd(centred, full_matrices=False)[2][1]
        worst = max(worst, float(np.abs(centred @ normal).max()))
    return worst / square_px


def detect_board(gray, pattern):
    """Find and refine the inner corners. Returns a dict: status, message, corners (N, 2) float64 or None, diagnostics.

    status: ok | not_found | swapped_pattern | irregular. Detection runs on a downscaled copy (long side <= DETECT_MAX_SIDE); the
    corners are mapped back and refined with cornerSubPix on the full-size image.
    """
    cols, rows = _check_pattern(pattern)
    h, w = gray.shape
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    found, used, swapped = None, None, False
    for sc in ([scale, 1.0] if scale < 1.0 else [1.0]):
        small = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else gray
        sx, sy = small.shape[1] / w, small.shape[0] / h
        ok, c = cv2.findChessboardCorners(small, (cols, rows), flags)
        if ok:
            c = c.reshape(-1, 2).astype(np.float64)
            found, used = np.column_stack([(c[:, 0] + 0.5) / sx - 0.5, (c[:, 1] + 0.5) / sy - 0.5]), sc
            break
        if sc == scale and cols != rows:
            swapped = bool(cv2.findChessboardCorners(small, (rows, cols), flags)[0])
    if found is None:
        if swapped:
            return dict(status="swapped_pattern", corners=None, detect_scale=scale, square_px=None, grid_residual=None,
                        subpix_max_shift=None, message=f"A {rows} x {cols} board was found instead of {cols} x {rows}. "
                                                       f"Swap the columns and rows values.")
        return dict(status="not_found", corners=None, detect_scale=scale, square_px=None, grid_residual=None, subpix_max_shift=None,
                    message=f"No {cols} x {rows} inner-corner checkerboard was found. Make sure the whole board is visible, in "
                            "focus and evenly lit, with a white border around it, and that the corner counts are correct.")
    grid = found.reshape(rows, cols, 2)
    square_px = float(np.median(np.r_[np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                                      np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()]))
    win = int(np.clip(round(square_px / 4), 3, 12))
    refined = found.astype(np.float32).reshape(-1, 1, 2).copy()
    cv2.cornerSubPix(gray, refined, (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 1e-3))
    refined = refined.reshape(-1, 2).astype(np.float64)
    shift = float(np.linalg.norm(refined - found, axis=1).max())
    residual = _grid_residual(refined, pattern, square_px)
    out = dict(status="ok", message="Checkerboard detected.", corners=refined, detect_scale=used, square_px=square_px,
               grid_residual=residual, subpix_max_shift=shift)
    if residual > GRID_RESIDUAL_MAX or shift > SUBPIX_SHIFT_MAX * square_px:
        out.update(status="irregular", corners=None,
                   message=f"A board was found but its corner grid looks distorted (row/column deviation {residual:.2f} squares, "
                           f"refinement shift {shift:.1f} px), so it was not trusted. Retake the photo with the board flat and sharp.")
    return out


def _thumbnail(gray, corners, pattern):
    sc = THUMBNAIL_SIDE / max(gray.shape)
    small = cv2.cvtColor(cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    if corners is not None:
        pts = (corners * sc).astype(np.float32).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(small, tuple(int(v) for v in pattern), pts, True)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def process_calibration_photo(data, pattern):
    """One uploaded calibration photograph -> a small result dict; the image itself is not retained.

    Keys: status (ok | rejected | not_found | swapped_pattern | irregular), message, size (w, h) after EXIF orientation or None,
    exif, digest (sha256), corners ((N, 2) float64 in oriented-image pixels, or None unless ok), thumbnail (JPEG bytes), square_px,
    grid_residual, subpix_max_shift.
    """
    _check_pattern(pattern)
    digest = hashlib.sha256(data or b"").hexdigest()
    try:
        im, orientation, size, exif = inspect_photo(data)
        gray = decode_gray(im, orientation, size)
    except CalibrationPhotoError as error:
        return dict(status="rejected", message=str(error), size=None, exif=None, digest=digest, corners=None, thumbnail=None,
                    square_px=None, grid_residual=None, subpix_max_shift=None)
    except (OSError, ValueError, MemoryError) as error:
        return dict(status="rejected", message=f"The image could not be decoded ({type(error).__name__}).", size=None, exif=None,
                    digest=digest, corners=None, thumbnail=None, square_px=None, grid_residual=None, subpix_max_shift=None)
    det = detect_board(gray, pattern)
    return dict(status=det["status"], message=det["message"], size=size, exif=exif, digest=digest, corners=det["corners"],
                thumbnail=_thumbnail(gray, det["corners"], pattern), square_px=det["square_px"],
                grid_residual=det["grid_residual"], subpix_max_shift=det["subpix_max_shift"])


def camera_info_from_photos(photos):
    """Summarize the EXIF of one camera's calibration photos (process_calibration_photo results) for later comparison.

    Returns (info, notes). info holds the most common make, model, lens_model, orientation, focal_length_mm and
    focal_length_35mm (omitted when no photo carries them) plus n_photos; values are strings or numbers, so it can be stored in the
    calibration file as is. notes lists plain-language remarks about photos that disagree with each other or lack metadata.
    """
    exifs = [p["exif"] for p in photos if p.get("exif")]
    info, notes = {"n_photos": len(photos)}, []
    if len(exifs) < len(photos):
        notes.append(f"{len(photos) - len(exifs)} of {len(photos)} calibration photos have no readable metadata.")
    for key, label in (("make", "camera make"), ("model", "camera model"), ("lens_model", "lens"), ("orientation", "EXIF orientation"),
                       ("focal_length_mm", "focal length"), ("focal_length_35mm", "35 mm-equivalent focal length")):
        values = [e.get(key) for e in exifs if e.get(key) not in (None, "")]
        if not values:
            continue
        common = max(set(values), key=values.count)
        info[key] = common
        if len(set(values)) > 1:
            notes.append(f"The calibration photos disagree on the {label} ({len(set(values))} different values); using the most common, "
                         f"{common!r}.")
    return info, notes


# ---------------------------------------------------------------- left/right correspondence

def _candidate_orderings(pattern):
    """Index permutations of the RIGHT corners that a board rotation could produce (mirror images are physically impossible)."""
    cols, rows = pattern
    idx = np.arange(cols * rows).reshape(rows, cols)
    cands = {"as_detected": idx.ravel(), "rotated_180": idx[::-1, ::-1].ravel()}
    if cols == rows:
        cands["rotated_90"] = np.rot90(idx, 1).ravel()
        cands["rotated_270"] = np.rot90(idx, 3).ravel()
    return cands


def _inplane_rotation_deg(left, right):
    """Rotation (degrees) of the left->right planar homography at the board centre; ~0 for two similarly oriented cameras."""
    H, _ = cv2.findHomography(left, right, 0)
    if H is None or not np.isfinite(H).all():
        return 180.0
    c = left.mean(0)
    w = H[2, 0] * c[0] + H[2, 1] * c[1] + H[2, 2]
    x = (H[0, 0] * c[0] + H[0, 1] * c[1] + H[0, 2]) / w
    y = (H[1, 0] * c[0] + H[1, 1] * c[1] + H[1, 2]) / w
    j00, j01 = (H[0, 0] - x * H[2, 0]) / w, (H[0, 1] - x * H[2, 1]) / w
    j10, j11 = (H[1, 0] - y * H[2, 0]) / w, (H[1, 1] - y * H[2, 1]) / w
    return float(np.degrees(np.arctan2(j10 - j01, j00 + j11)))


def _sampson_px(F, left, right):
    """First-order geometric (Sampson) distance of each correspondence from the epipolar constraint, in pixels."""
    x1 = np.c_[left, np.ones(len(left))]
    x2 = np.c_[right, np.ones(len(right))]
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    num = (x2 * Fx1).sum(1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-18))


def _robust_fundamental(left, right, threshold, n_pairs):
    if n_pairs < MIN_PAIRS_TO_VERIFY or len(left) < 8:
        return None
    F, _ = cv2.findFundamentalMat(left, right, cv2.FM_RANSAC, threshold, 0.999, 10000)
    if F is None:
        return None
    F = np.asarray(F, np.float64)
    return F[:3] if F.shape[0] >= 3 and np.isfinite(F).all() else None


def resolve_pair_orderings(pairs, pattern, max_side):
    """Decide how each pair's right corners correspond to its left corners, using geometry rather than corner-vector direction.

    pairs: list of (left (N, 2), right (N, 2)) as detected. max_side: longest image side in pixels (scales thresholds).

    A detected board can be numbered in more than one way (an even x even grid looks the same after a 180 degree rotation), and a
    planar homography cannot tell those apart. The epipolar geometry of the rig can: a robust fundamental matrix F is fitted to all
    pairs under the current assignment, and every candidate ordering of each pair is scored by its median Sampson distance under F.
    A pair is accepted only if its best ordering fits F and beats the runner-up by ORDER_RATIO_MIN; otherwise it is flagged
    ``ambiguous`` or ``inconsistent`` and left exactly as detected. Nothing is changed silently: every pair reports its action.

    Returns dict(pairs=[dict(status, action, right_corners, scores, ratio, message)], F, ransac_threshold_px, iterations).
    status: ok | ambiguous | inconsistent | unverifiable.
    """
    _check_pattern(pattern)
    cands = _candidate_orderings(pattern)
    names = list(cands)
    lefts = [np.asarray(l, np.float64).reshape(-1, 2) for l, _ in pairs]
    rights = [np.asarray(r, np.float64).reshape(-1, 2) for _, r in pairs]
    thr = max(2.0, 0.0012 * max_side)
    max_score = ORDER_SCORE_MAX * max_side

    # initial hypothesis: least in-plane rotation of the left->right homography
    choice = [min(names, key=lambda n: abs(_inplane_rotation_deg(lefts[i], rights[i][cands[n]]))) for i in range(len(pairs))]
    F, iterations = None, 0
    for iterations in range(1, 5):
        L = np.vstack(lefts)
        R = np.vstack([rights[i][cands[choice[i]]] for i in range(len(pairs))])
        F = _robust_fundamental(L, R, thr, len(pairs))
        if F is None:
            break
        best = [min(names, key=lambda n: np.median(_sampson_px(F, lefts[i], rights[i][cands[n]]))) for i in range(len(pairs))]
        if best == choice:
            break
        choice = best

    out = []
    for i in range(len(pairs)):
        if F is None:
            out.append(dict(status="unverifiable", action="as_detected", right_corners=rights[i], scores={}, ratio=None,
                            message="The correspondence could not be verified: too few pairs, or no consistent epipolar geometry."))
            continue
        scores = {n: float(np.median(_sampson_px(F, lefts[i], rights[i][cands[n]]))) for n in names}
        order = sorted(names, key=scores.get)
        best_n, second = order[0], order[1]
        ratio = scores[second] / max(scores[best_n], 1e-9)
        info = dict(scores=scores, ratio=float(ratio))
        if scores[best_n] > max_score:
            out.append(dict(status="inconsistent", action="as_detected", right_corners=rights[i], **info,
                            message=f"Neither corner ordering agrees with the epipolar geometry of the other pairs (best median "
                                    f"error {scores[best_n]:.0f} px). A corner may be mis-detected, or the cameras moved between photos."))
        elif ratio < ORDER_RATIO_MIN:
            out.append(dict(status="ambiguous", action="as_detected", right_corners=rights[i], **info,
                            message=f"The left/right corner correspondence is ambiguous: the best ordering ({best_n}) is only "
                                    f"{ratio:.1f}x better than the alternative. Retake this pair with the board tilted so its "
                                    "orientation is unambiguous, or exclude it."))
        else:
            out.append(dict(status="ok", action=best_n, right_corners=rights[i][cands[best_n]], **info,
                            message="Corner correspondence verified" +
                                    ("" if best_n == "as_detected" else f"; the right corners were renumbered ({best_n}) to match the left") + "."))
    return dict(pairs=out, F=F, ransac_threshold_px=thr, iterations=iterations)


# ---------------------------------------------------------------- calibration

def _mono(obj, corners_list, size, flags):
    objs = [obj] * len(corners_list)
    imgs = [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corners_list]
    try:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, tuple(int(v) for v in size), None, None, flags=flags)
    except cv2.error as error:
        raise CalibrationFailed(f"Single-camera calibration failed: {error.msg.strip() if hasattr(error, 'msg') else error}") from error
    if not (np.isfinite(K).all() and np.isfinite(D).all()):
        raise CalibrationFailed("Single-camera calibration produced non-finite parameters.")
    return float(rms), K, D, rvecs, tvecs


def _view_errors(obj, corners_list, K, D, rvecs, tvecs):
    errs = []
    for c, rv, tv in zip(corners_list, rvecs, tvecs):
        proj = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
        errs.append(float(np.sqrt(((proj - np.asarray(c).reshape(-1, 2)) ** 2).sum(1).mean())))
    return errs


def _heldout_rms(obj, train, test, size, flags):
    """Calibrate on `train`, then fit only the board pose of each held-out view and report its reprojection RMS."""
    _, K, D, _, _ = _mono(obj, train, size, flags)
    sq, n = 0.0, 0
    for c in test:
        c32 = np.asarray(c, np.float32).reshape(-1, 1, 2)
        ok, rv, tv = cv2.solvePnP(obj, c32, K, D)
        if not ok:
            return float("inf")
        proj = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
        sq += float(((proj - c32.reshape(-1, 2)) ** 2).sum())
        n += len(proj)
    return float(np.sqrt(sq / (2 * n)))


def select_distortion_model(obj, corners_list, size, requested="auto"):
    """Choose the distortion model for ONE camera. Returns (name, table).

    'auto' compares radial2 (Model B's), radial3 and full by held-out reprojection RMS (CV_FOLDS-fold over the board views) and keeps
    the simplest model within CV_SIMPLICITY_TOLERANCE of the best, so extra coefficients are used only when they generalize.
    """
    if requested != "auto":
        if requested not in DISTORTION_MODELS:
            raise ValueError(f"Unknown distortion model {requested!r}; choose one of {sorted(DISTORTION_MODELS)} or 'auto'.")
        return requested, []
    n = len(corners_list)
    folds = [list(range(f, n, CV_FOLDS)) for f in range(min(CV_FOLDS, n // 2))]
    table = []
    for name, flags in DISTORTION_MODELS.items():
        sq_total, count = 0.0, 0
        try:
            for held in folds:
                train = [c for i, c in enumerate(corners_list) if i not in held]
                test = [corners_list[i] for i in held]
                rms = _heldout_rms(obj, train, test, size, flags)
                sq_total += rms ** 2 * len(test)
                count += len(test)
            cv_rms = float(np.sqrt(sq_total / count))
        except (CalibrationFailed, cv2.error):
            cv_rms = float("inf")
        if not np.isfinite(cv_rms):
            cv_rms = float("inf")
        table.append(dict(model=name, cv_rms=cv_rms))
    best = min(t["cv_rms"] for t in table)
    if not np.isfinite(best):
        raise CalibrationFailed("No distortion model could be fitted to the board views.")
    chosen = next(t["model"] for t in table if t["cv_rms"] <= best * (1 + CV_SIMPLICITY_TOLERANCE))
    for t in table:
        t["chosen"] = t["model"] == chosen
    return chosen, table


def _undistort_pixels(pts, K, D):
    return cv2.undistortPoints(np.asarray(pts, np.float64).reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)


def _epipolar_px(F, K1, D1, K2, D2, left, right):
    """Symmetric point-to-epipolar-line distance of each correspondence after removing lens distortion (pixels)."""
    a, b = _undistort_pixels(left, K1, D1), _undistort_pixels(right, K2, D2)
    x1, x2 = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    l2, l1 = x1 @ F.T, x2 @ F
    num = np.abs((x2 * l2).sum(1))
    return 0.5 * (num / np.hypot(l2[:, 0], l2[:, 1]) + num / np.hypot(l1[:, 0], l1[:, 1]))


def _rectification_px(K1, D1, K2, D2, R, T, size, left, right):
    """Vertical misalignment of corresponding points after stereo rectification, in pixels at the rectified focal length."""
    try:
        R1, R2, P1, _, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, tuple(int(v) for v in size), R, T, flags=0, alpha=0)
        n1 = cv2.undistortPoints(np.asarray(left, np.float64).reshape(-1, 1, 2), K1, D1).reshape(-1, 2)
        n2 = cv2.undistortPoints(np.asarray(right, np.float64).reshape(-1, 1, 2), K2, D2).reshape(-1, 2)
        r1 = np.c_[n1, np.ones(len(n1))] @ R1.T
        r2 = np.c_[n2, np.ones(len(n2))] @ R2.T
        dy = np.abs(r1[:, 1] / r1[:, 2] - r2[:, 1] / r2[:, 2]) * float(P1[1, 1])
    except cv2.error:
        return None
    return dy if np.isfinite(dy).all() else None


def _stats(a):
    a = np.asarray(a, np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)), max=float(a.max()))


def _coverage(corners_list, size):
    """Image coverage of the board views: fraction of the frame inside the board hulls, and how many of 3 x 3 cells hold a corner."""
    w, h = size
    s = 1 / 16
    mask = np.zeros((max(1, int(h * s)), max(1, int(w * s))), np.uint8)
    cells = set()
    for c in corners_list:
        c = np.asarray(c).reshape(-1, 2)
        cv2.fillConvexPoly(mask, cv2.convexHull((c * s).astype(np.int32)), 1)
        cells.update(zip(np.clip((c[:, 0] / w * 3).astype(int), 0, 2), np.clip((c[:, 1] / h * 3).astype(int), 0, 2)))
    return dict(area_fraction=float(mask.mean()), cells_used=len(cells))


def _tilts_deg(rvecs):
    return [float(np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(rv)[0][2, 2]))))) for rv in rvecs]


def _warn(code, message, heuristic=True):
    return dict(code=code, message=message, heuristic=heuristic)


def calibrate_stereo(left_corners, right_corners, pattern, square_mm, left_size, right_size, distortion="auto", pair_ids=None):
    """Calibrate a stereo rig from corresponding board corners (already verified and ordered; see resolve_pair_orderings).

    left_corners/right_corners: sequences of (N, 2) arrays, one per valid pair. left_size/right_size: (w, h) of that camera's
    EXIF-oriented images (may differ between cameras). distortion: 'auto' | 'radial2' | 'radial3' | 'full'.

    Raises ValueError for unusable arguments (including fewer than MIN_VALID_PAIRS pairs) and CalibrationFailed when OpenCV cannot
    solve. Otherwise returns a dict with the load_model_b keys (K1, D1, K2, D2, R, T, E, F, stereo_rms, rms_left, rms_right,
    baseline_mm, image_size, flags, pairs) plus: valid, errors (mathematically invalid -> do not use), warnings (suspicious, may be
    fine), metrics, per_pair, distortion_left/right, distortion_cv, board, image_size_left/right, pattern, square_mm.
    """
    cols, rows = _check_pattern(pattern)
    n = len(left_corners)
    if n != len(right_corners):
        raise ValueError("left_corners and right_corners must have the same length.")
    if n < MIN_VALID_PAIRS:
        raise ValueError(f"At least {MIN_VALID_PAIRS} valid stereo pairs are required; got {n}.")
    if not (np.isfinite(square_mm) and square_mm > 0):
        raise ValueError("The checkerboard square size must be a positive number of millimetres.")
    pair_ids = list(range(1, n + 1)) if pair_ids is None else list(pair_ids)
    obj = board_object_points(pattern, square_mm)
    left = [np.asarray(c, np.float64).reshape(-1, 2) for c in left_corners]
    right = [np.asarray(c, np.float64).reshape(-1, 2) for c in right_corners]
    if any(c.shape != (cols * rows, 2) or not np.isfinite(c).all() for c in left + right):
        raise ValueError(f"Every corner set must be a finite {cols * rows} x 2 array.")
    left_size, right_size = tuple(int(v) for v in left_size), tuple(int(v) for v in right_size)

    name_l, table_l = select_distortion_model(obj, left, left_size, distortion)
    name_r, table_r = select_distortion_model(obj, right, right_size, distortion)
    rms_l, K1, D1, rv_l, tv_l = _mono(obj, left, left_size, DISTORTION_MODELS[name_l])
    rms_r, K2, D2, rv_r, tv_r = _mono(obj, right, right_size, DISTORTION_MODELS[name_r])
    objs = [obj] * n
    try:
        out = cv2.stereoCalibrateExtended(objs, [c.astype(np.float32).reshape(-1, 1, 2) for c in left],
                                          [c.astype(np.float32).reshape(-1, 1, 2) for c in right], K1, D1, K2, D2, left_size, None,
                                          None, flags=cv2.CALIB_FIX_INTRINSIC, criteria=STEREO_CRITERIA)
    except cv2.error as error:
        raise CalibrationFailed(f"Stereo calibration failed: {error.msg.strip() if hasattr(error, 'msg') else error}") from error
    stereo_rms, _, _, _, _, R, T, E, F, rv_s, tv_s, view_err = out
    T = np.asarray(T, np.float64).reshape(3, 1)
    model = dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F, stereo_rms=float(stereo_rms), rms_left=rms_l, rms_right=rms_r,
                 baseline_mm=float(np.linalg.norm(T)), image_size=left_size, flags=int(cv2.CALIB_FIX_INTRINSIC),
                 pairs=pair_ids)

    errors, warnings = [], []
    arrays = dict(K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, F=F)
    if not all(np.isfinite(v).all() for v in arrays.values()):
        errors.append("The calibration contains non-finite values.")
    if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6):
        errors.append("The stereo rotation matrix R is not a proper rotation.")
    for tag, K in (("left", K1), ("right", K2)):
        if not (K[0, 0] > 0 and K[1, 1] > 0):
            errors.append(f"The {tag} camera has a non-positive focal length.")
    if model["baseline_mm"] <= 1e-6:
        errors.append("The estimated baseline is zero, so depth cannot be triangulated.")

    # positive depth + reconstructed board scale, from the calibration corners themselves
    ratios, neg = [], 0
    if not errors:
        for l, r in zip(left, right):
            X = triangulate(model, l, r)
            Xr = to_right_camera(model, X)
            neg += int((~np.isfinite(X)).any(1).sum() + (X[:, 2] <= 0).sum() + (Xr[:, 2] <= 0).sum())
            g = X.reshape(rows, cols, 3)
            d = np.r_[np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(), np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()]
            ratios.append(np.median(d) / square_mm)
        if neg:
            errors.append(f"{neg} calibration points triangulate behind a camera, so the left/right geometry is inconsistent.")

    epi = np.concatenate([_epipolar_px(F, K1, D1, K2, D2, l, r) for l, r in zip(left, right)]) if not errors else None
    rect = None
    if not errors:
        parts = [_rectification_px(K1, D1, K2, D2, R, T, left_size, l, r) for l, r in zip(left, right)]
        rect = np.concatenate(parts) if all(p is not None for p in parts) else None
    metrics = dict(rms_left=rms_l, rms_right=rms_r, rms_stereo=float(stereo_rms), baseline_mm=model["baseline_mm"],
                   baseline_direction=(T.ravel() / max(model["baseline_mm"], 1e-12)).tolist(),
                   epipolar_px=_stats(epi) if epi is not None else None,
                   rectification_vertical_px=_stats(rect) if rect is not None else None,
                   board_square_scale=_stats(ratios) if ratios else None)

    dist_z = [float(tv.ravel()[2]) for tv in tv_s]
    tilts_l, tilts_r = _tilts_deg(rv_l), _tilts_deg(rv_r)
    board = dict(distance_mm=dict(min=min(dist_z), max=max(dist_z)), tilt_deg_left=_stats(tilts_l), tilt_deg_right=_stats(tilts_r),
                 coverage_left=_coverage(left, left_size), coverage_right=_coverage(right, right_size))
    ve_l, ve_r = _view_errors(obj, left, K1, D1, rv_l, tv_l), _view_errors(obj, right, K2, D2, rv_r, tv_r)
    view_err = np.asarray(view_err, np.float64).reshape(n, 2)
    scale_l, scale_r = max(left_size) / 4000, max(right_size) / 4000
    scale = max(scale_l, scale_r)
    med = np.median(view_err, axis=0)
    per_pair = []
    for i in range(n):
        outlier = bool((view_err[i] > OUTLIER_FACTOR * np.maximum(med, 0.25 * np.array([scale_l, scale_r]))).any())
        per_pair.append(dict(pair=pair_ids[i], mono_left_px=ve_l[i], mono_right_px=ve_r[i], stereo_left_px=float(view_err[i, 0]),
                             stereo_right_px=float(view_err[i, 1]), epipolar_mean_px=float(_epipolar_px(F, K1, D1, K2, D2, left[i], right[i]).mean())
                             if not errors else None, outlier=outlier))
        if outlier:
            warnings.append(_warn("pair_outlier", f"Pair {pair_ids[i]} fits much worse than the others (stereo error "
                                                  f"{view_err[i].max():.2f} px vs median {med.max():.2f} px). It may contain a mis-detected "
                                                  f"corner or a blurred photo; consider replacing or excluding it."))

    # suspicious-but-possibly-valid conditions: warnings only, all thresholds are heuristics
    for tag, rms, sc in (("left", rms_l, scale_l), ("right", rms_r, scale_r)):
        if rms > RMS_MONO_WARN_PER_4000PX * sc:
            warnings.append(_warn("rms_mono", f"The {tag} single-camera reprojection RMS is {rms:.2f} px, above the heuristic guide of "
                                              f"{RMS_MONO_WARN_PER_4000PX * sc:.1f} px for this resolution."))
    if stereo_rms > RMS_STEREO_WARN_PER_4000PX * scale:
        warnings.append(_warn("rms_stereo", f"The stereo reprojection RMS is {stereo_rms:.2f} px, above the heuristic guide of "
                                            f"{RMS_STEREO_WARN_PER_4000PX * scale:.1f} px for this resolution."))
    if epi is not None and np.median(epi) > EPIPOLAR_WARN_PER_4000PX * scale:
        warnings.append(_warn("epipolar", f"The median epipolar distance is {np.median(epi):.2f} px, above the heuristic guide of "
                                          f"{EPIPOLAR_WARN_PER_4000PX * scale:.1f} px."))
    for tag, K, size in (("left", K1, left_size), ("right", K2, right_size)):
        off = np.array([K[0, 2] / size[0] - 0.5, K[1, 2] / size[1] - 0.5])
        if not (0 <= K[0, 2] <= size[0] and 0 <= K[1, 2] <= size[1]):
            warnings.append(_warn("principal_point_outside", f"The {tag} principal point ({K[0, 2]:.0f}, {K[1, 2]:.0f}) lies outside the "
                                                             f"{size[0]} x {size[1]} image. This is unusual; the calibration photos may "
                                                             "not constrain the camera well.", heuristic=False))
        elif np.abs(off).max() > 0.15:
            warnings.append(_warn("principal_point_offset", f"The {tag} principal point is far from the image centre "
                                                            f"({100 * off[0]:+.0f}%, {100 * off[1]:+.0f}% of the image size); unusual for phone cameras."))
        if not 0.8 < K[0, 0] / K[1, 1] < 1.25:
            warnings.append(_warn("focal_ratio", f"The {tag} fx/fy ratio is {K[0, 0] / K[1, 1]:.2f}; pixels are normally square."))
    mid_depth = 0.5 * (board["distance_mm"]["min"] + board["distance_mm"]["max"])
    if model["baseline_mm"] / mid_depth < 0.01:
        warnings.append(_warn("short_baseline", f"The baseline ({model['baseline_mm']:.1f} mm) is under 1% of the board distance "
                                                f"({mid_depth:.0f} mm); depth from triangulation will be very sensitive to pixel errors."))
    if ratios and abs(metrics["board_square_scale"]["median"] - 1) > 0.02:
        warnings.append(_warn("board_scale", f"Reconstructed board squares average {metrics['board_square_scale']['median']:.3f}x the "
                                             "entered square size; check the square size and the left/right correspondence."))
    if sum(t >= 15 for t in tilts_l) < 3 or sum(t >= 15 for t in tilts_r) < 3:
        warnings.append(_warn("tilt_diversity", "Fewer than 3 photos per camera show the board tilted by 15 degrees or more; tilted "
                                                "views improve focal-length and distortion estimates."))
    if min(board["coverage_left"]["cells_used"], board["coverage_right"]["cells_used"]) < 6:
        warnings.append(_warn("coverage", "The board did not appear in enough regions of the frame (fewer than 6 of 9); place it "
                                          "near the edges and corners in some photos."))
    if board["distance_mm"]["max"] / board["distance_mm"]["min"] < 1.2:
        warnings.append(_warn("distance_spread", "All photos were taken at nearly the same distance; vary the board distance."))
    if n < RECOMMENDED_PAIRS[0]:
        warnings.append(_warn("few_pairs", f"Only {n} valid pairs were used; {RECOMMENDED_PAIRS[0]}-{RECOMMENDED_PAIRS[1]} are recommended.",
                              heuristic=False))

    model.update(valid=not errors, errors=errors, warnings=warnings, metrics=metrics, per_pair=per_pair,
                 distortion_left=name_l, distortion_right=name_r, distortion_cv=dict(left=table_l, right=table_r), board=board,
                 image_size_left=left_size, image_size_right=right_size, pattern=(cols, rows), square_mm=float(square_mm))
    return model
