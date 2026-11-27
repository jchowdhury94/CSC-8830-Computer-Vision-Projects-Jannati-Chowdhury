"""Shared Part B structure-from-motion pipeline (used by the saved experiment, the results page and the visitor page).

This module is imported, not run. The saved experiment is run by automatic_landmark_experiment.py.
Requires NumPy and OpenCV (headless), listed in the repository requirements.txt.
Inputs: four decoded photographs, corner or point coordinates and the object's measured width and height.
Outputs: Python objects only (Correspondences, Reconstruction); no files are written.

Correspondences (find_correspondences):
    P1-P4 corners selected in photo 1 -> SIFT features inside that boundary -> ratio-test matches to each
    other photo -> RANSAC homography. P5-P8 are four interior features matched in all four photos, at their
    observed image positions; P1-P4 in photos 2-4 are estimated through the homographies. Automatic
    positions only initialize the points: every point is verified or corrected by a person, and only the
    verified coordinates are reconstructed.

Reconstruction of exactly eight points in four views (Tomasi-Kanade, orthographic camera):
    measurement matrix -> per-view centering -> rank-3 SVD factorization -> metric upgrade (L = Q Q^T,
    least squares, Cholesky; never corrected if not positive definite) -> alignment with the object's
    width/height/normal -> one least-squares isotropic scale from the measured width and height ->
    edge lengths and percentage errors.

Point and view order everywhere: P1 top-left, P2 top-right, P3 bottom-right, P4 bottom-left corner,
P5-P8 landmarks; measurement-matrix rows V1_x, V1_y, ..., V4_y. Image coordinates are full-resolution
upright pixels (EXIF orientation applied), origin upper-left, +x right, +y down.
"""
import math
from dataclasses import dataclass, field

import numpy as np

try:  # package import (Streamlit page, tests) or plain script import (experiment script run from this folder)
    from . import visitor_reconstruction as images
except ImportError:  # pragma: no cover - exercised by running the experiment script directly
    import visitor_reconstruction as images

VIEW_COUNT = 4
POINT_COUNT = 8
POINT_IDS = tuple(f"P{i}" for i in range(1, POINT_COUNT + 1))
POINT_ROLES = ("Top-left corner", "Top-right corner", "Bottom-right corner", "Bottom-left corner",
               "Landmark", "Landmark", "Landmark", "Landmark")
# (key, label, first point index, second point index)
EDGES = (("top_width", "Top edge", 0, 1), ("bottom_width", "Bottom edge", 3, 2),
         ("left_height", "Left edge", 0, 3), ("right_height", "Right edge", 1, 2),
         ("diagonal_P1_P3", "Diagonal P1–P3", 0, 2), ("diagonal_P2_P4", "Diagonal P2–P4", 1, 3))

# Correspondence settings (working images have a long side of images.WORKING_LONG_SIDE pixels).
HOMOGRAPHY_THRESHOLD_PX = 4.0  # RANSAC reprojection threshold, working pixels
MIN_HOMOGRAPHY_INLIERS = 30
MIN_INLIER_COVERAGE = 0.25  # convex hull of the inliers / area of the P1-P4 boundary in photo 1
MIN_BOUNDARY_FRACTION = 0.01  # P1-P4 boundary area / photo 1 area
DEGENERATE_SIGMA3_RATIO = 1e-4

# Automatic interior landmarks, in object coordinates where the P1-P4 boundary is the unit square.
LANDMARK_MARGIN = 0.05  # minimum distance from the boundary edges
LANDMARK_SEPARATION = 0.2  # minimum distance between two landmarks
LANDMARK_QUADRANTS = ((0, 0), (1, 0), (1, 1), (0, 1))  # P5-P8: top-left, top-right, bottom-right, bottom-left

PREDICTED, OUTSIDE, UNAVAILABLE = "predicted", "outside", "unavailable"


class PointError(images.ReconstructionError):
    """Invalid point selections; str(error) is safe to show to a visitor."""


# --------------------------------------------------------------------------- coordinates

def _axis_scales(image):
    """Working pixels per original pixel along x and y (they can differ slightly after rounding)."""
    return np.array([image.size[0] / image.original_size[0], image.size[1] / image.original_size[1]])


def to_working(points, image):
    """Original upright pixel coordinates -> working-image coordinates (pixel-center convention)."""
    return (np.asarray(points, float) + 0.5) * _axis_scales(image) - 0.5


def to_original(points, image):
    """Working-image coordinates -> original upright pixel coordinates (inverse of to_working)."""
    return (np.asarray(points, float) + 0.5) / _axis_scales(image) - 0.5


def display_to_original(x, y, shown_width, shown_height, image, origin=(0.0, 0.0), crop_size=None):
    """A click at (x, y) on a displayed image (optionally a working-image crop) -> original pixel coordinates.

    origin/crop_size describe the shown region in working pixels; by default the whole working image is shown.
    The result is clipped to the original image.
    """
    if shown_width <= 0 or shown_height <= 0:
        raise ValueError("The image is not ready. Please click again.")
    crop = image.size if crop_size is None else crop_size
    working = np.array([origin[0] + (x + 0.5) * crop[0] / shown_width - 0.5,
                        origin[1] + (y + 0.5) * crop[1] / shown_height - 0.5])
    point = to_original(working, image)
    return np.clip(point, 0, np.array(image.original_size, float) - 1)


def inside(points, size):
    """Boolean mask of finite (x, y) points inside an image of size (width, height)."""
    points = np.asarray(points, float).reshape(-1, 2)
    finite = np.isfinite(points).all(axis=1)
    safe = np.where(np.isfinite(points), points, -1.0)
    return finite & (safe[:, 0] >= 0) & (safe[:, 1] >= 0) & (safe[:, 0] <= size[0] - 1) & (safe[:, 1] <= size[1] - 1)


def check_corners(corners, size):
    """Validate the P1-P4 corners in photo 1 (original pixels). Raises PointError."""
    corners = np.asarray(corners, float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise PointError("Select the four corners, P1–P4, in photo 1.")
    if not inside(corners, size).all():
        raise PointError("Every point in photo 1 must lie inside the photo.")
    if len(np.unique(np.rint(corners), axis=0)) < 4:
        raise PointError("Two points in photo 1 are at the same position. Each point must be a different location.")
    corners = corners.astype(np.float32)
    import cv2
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        raise PointError("P1–P4 must be the top-left, top-right, bottom-right and bottom-left corners, in that "
                         "order, forming the object's outline.")
    if abs(cv2.contourArea(corners)) < MIN_BOUNDARY_FRACTION * size[0] * size[1]:
        raise PointError("The object outline P1–P4 is too small. Photograph the object larger in the frame.")


def check_reference_points(points, size):
    """Validate P1-P8 in photo 1 (original pixels). Raises PointError."""
    points = np.asarray(points, float)
    if points.shape != (POINT_COUNT, 2) or not np.isfinite(points).all():
        raise PointError("Select all eight points, P1–P8, in photo 1.")
    if not inside(points, size).all():
        raise PointError("Every point in photo 1 must lie inside the photo.")
    if len(np.unique(np.rint(points), axis=0)) < POINT_COUNT:
        raise PointError("Two points in photo 1 are at the same position. Each point must be a different location.")
    check_corners(points[:4], size)


# --------------------------------------------------------------------------- correspondence estimation

@dataclass
class ViewPrediction:
    view: int  # 1-based photo number (2-4)
    points: np.ndarray  # (N, 2) estimated original pixel coordinates; NaN where no estimate is available
    status: tuple  # per point: PREDICTED (has a position), OUTSIDE (estimated outside the photo) or UNAVAILABLE
    reliable: bool  # homography passed every reliability check
    matches: int  # ratio-test matches with photo 1
    inliers: int  # RANSAC homography inliers
    coverage: float  # inlier convex-hull area / P1-P4 boundary area in photo 1
    homography: np.ndarray  # 3 x 3, working photo 1 -> working photo `view` (None if not estimated)
    message: str
    inlier_matches: dict = field(default_factory=dict)  # photo-1 feature index -> (x, y) here, working pixels


def _homography(points_a, points_b):
    import cv2
    if len(points_a) < 4:
        return None, np.zeros(len(points_a), bool)
    with images._RANSAC_LOCK:
        cv2.setRNGSeed(0)
        matrix, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, HOMOGRAPHY_THRESHOLD_PX,
                                          maxIters=5000, confidence=0.999)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None, np.zeros(len(points_a), bool)
    return matrix, mask.ravel().astype(bool)


def _project(matrix, points):
    """Apply a homography; points that map to or behind infinity become NaN."""
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    w = homogeneous[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = homogeneous[:, :2] / w
    projected[(w[:, 0] <= 1e-12) | ~np.isfinite(projected).all(axis=1)] = np.nan
    return projected


def predict_view(reference, reference_features, boundary, reference_points_working, image, view):
    """Match one photo with photo 1's in-boundary features and map the reference points into it through the
    homography. Never raises for poor matches."""
    import cv2
    count = len(reference_points_working)

    def failed(message, matches=0, inliers=0, coverage=0.0, matrix=None):
        return ViewPrediction(view, np.full((count, 2), np.nan), (UNAVAILABLE,) * count, False,
                              matches, inliers, coverage, matrix, message)

    try:
        features = images.detect_features(image.rgb)
    except images.InsufficientMatchesError:
        return failed(f"Photo {view} has too few distinctive features. Select its points manually.")
    pairs = images.match_descriptors(reference_features.descriptors, features.descriptors)
    if len(pairs) < MIN_HOMOGRAPHY_INLIERS:
        return failed(f"Photo {view} shares too few matching features with photo 1 ({len(pairs)}). "
                      "Select its points manually.", len(pairs))
    a = reference_features.points[pairs[:, 0]]
    b = features.points[pairs[:, 1]]
    matrix, mask = _homography(a, b)
    inliers = int(mask.sum())
    coverage = (float(cv2.contourArea(cv2.convexHull(a[mask].astype(np.float32))) / boundary)
                if inliers >= 3 else 0.0)
    if matrix is None or inliers < MIN_HOMOGRAPHY_INLIERS:
        return failed(f"Photo {view}: only {inliers} reliable matches with photo 1 (at least "
                      f"{MIN_HOMOGRAPHY_INLIERS} needed). Select its points manually.", len(pairs), inliers, coverage, matrix)
    if coverage < MIN_INLIER_COVERAGE:
        return failed(f"Photo {view}: the reliable matches cover only {100 * coverage:.0f}% of the object. "
                      "Select its points manually.", len(pairs), inliers, coverage, matrix)
    working = _project(matrix, reference_points_working)
    corners = working[:4]
    if (not np.isfinite(corners).all() or
            not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2))):
        return failed(f"Photo {view}: the matched object outline is distorted. Select its points manually.",
                      len(pairs), inliers, coverage, matrix)
    estimated = to_original(working, image)
    within = inside(estimated, image.original_size)
    status = tuple(PREDICTED if ok else OUTSIDE for ok in within)
    estimated[~within] = np.nan
    outside_count = int((~within).sum())
    message = (f"Photo {view}: {inliers} reliable matches covering {100 * coverage:.0f}% of the object." +
               (f" {outside_count} point(s) fall outside the photo; select them manually."
                if outside_count else ""))
    matched = {int(i): features.points[j].copy() for i, j in pairs[mask]}
    return ViewPrediction(view, estimated, status, True, len(pairs), inliers, coverage, matrix, message, matched)


def _match_views(decoded, reference_points, progress=None):
    """Match photos 2-4 with the features inside the P1-P4 boundary of photo 1.

    reference_points: (N, 2) original pixel coordinates in photo 1, starting with P1-P4.
    Returns (photo-1 features or None, [ViewPrediction for photos 2, 3, 4]).
    """
    import cv2

    reference = decoded[0]
    reference_working = to_working(reference_points, reference)
    mask = np.zeros(reference.rgb.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(reference_working[:4]).astype(np.int32), 255)
    boundary = float(cv2.contourArea(reference_working[:4].astype(np.float32)))
    if progress:
        progress("Finding features in photo 1", 0.1)
    try:
        reference_features = images.detect_features(reference.rgb, mask=mask)
    except images.InsufficientMatchesError:
        message = ("Photo 1 has too few distinctive features inside the object outline, so no points could be "
                   "found automatically. Select the remaining points manually.")
        count = len(reference_points)
        return None, [ViewPrediction(v, np.full((count, 2), np.nan), (UNAVAILABLE,) * count, False,
                                     0, 0, 0.0, None, message) for v in range(2, VIEW_COUNT + 1)]
    predictions = []
    for v in range(1, VIEW_COUNT):
        if progress:
            progress(f"Matching photo {v + 1} with photo 1", 0.1 + 0.9 * v / VIEW_COUNT)
        predictions.append(predict_view(reference, reference_features, boundary, reference_working, decoded[v], v + 1))
    return reference_features, predictions


def select_landmarks(object_points, responses):
    """Deterministic choice of up to four distributed landmarks among candidate features.

    object_points: (N, 2) candidate positions with the P1-P4 boundary mapped to the unit square.
    responses: (N,) feature strengths. First the strongest candidate in each quadrant; a quadrant without one
    is then filled by the remaining candidate farthest from those already chosen. Landmarks keep a margin from
    the boundary and a minimum separation. Returns four candidate indices (None where no candidate qualifies).
    """
    points = np.asarray(object_points, float).reshape(-1, 2)
    usable = ((points >= LANDMARK_MARGIN) & (points <= 1 - LANDMARK_MARGIN)).all(axis=1)
    # strongest first; ties are broken by position so the order never depends on the detector's output order
    order = [int(i) for i in np.lexsort((points[:, 0], points[:, 1], -np.asarray(responses, float))) if usable[i]]
    chosen = [None] * len(LANDMARK_QUADRANTS)

    def clearance(i):
        return min((np.linalg.norm(points[i] - points[j]) for j in chosen if j is not None), default=np.inf)

    for slot, quadrant in enumerate(LANDMARK_QUADRANTS):
        chosen[slot] = next((i for i in order if tuple((points[i] >= 0.5).astype(int)) == quadrant
                             and clearance(i) >= LANDMARK_SEPARATION), None)
    for slot in range(len(chosen)):
        if chosen[slot] is None:
            remaining = [i for i in order if clearance(i) >= LANDMARK_SEPARATION]
            chosen[slot] = max(remaining, key=clearance, default=None)
    return chosen


@dataclass
class Correspondences:
    points: np.ndarray  # (4, 8, 2) original pixel coordinates; NaN where a point must be selected manually
    views: list  # ViewPrediction for photos 2, 3, 4 holding that photo's automatic P1-P8
    landmarks_found: int  # how many of P5-P8 were identified in all four photos


def find_correspondences(decoded, corners, progress=None):
    """The shared correspondence method: four corners in photo 1 -> P1-P8 in all four photos.

    P5-P8 are interior features of photo 1 that are homography-consistent ratio-test matches in all three other
    photos; their positions are the observed feature positions. P1-P4 in photos 2-4 are the homography
    estimates of the selected corners. Nothing is invented: a landmark that cannot be identified in all four
    photos, or a corner that cannot be estimated, stays NaN. Raises PointError for invalid corners.
    """
    import cv2

    if len(decoded) != VIEW_COUNT:
        raise PointError(f"Exactly {VIEW_COUNT} photographs are needed.")
    corners = np.asarray(corners, float)
    check_corners(corners, decoded[0].original_size)
    features, views = _match_views(decoded, corners, progress)
    points = np.full((VIEW_COUNT, POINT_COUNT, 2), np.nan)
    points[0, :4] = corners
    for prediction in views:
        points[prediction.view - 1, :4] = prediction.points

    found = 0
    if features is not None and all(p.reliable for p in views):
        shared = sorted(set.intersection(*(set(p.inlier_matches) for p in views)))
        if shared:
            working = features.points[shared].astype(float)
            unit_square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
            to_object = cv2.getPerspectiveTransform(to_working(corners, decoded[0]).astype(np.float32), unit_square)
            chosen = select_landmarks(_project(to_object, working), features.responses[shared])
            for slot, candidate in enumerate(chosen):
                if candidate is None:
                    continue
                index = shared[candidate]
                points[0, 4 + slot] = to_original(features.points[index], decoded[0])
                for prediction in views:
                    points[prediction.view - 1, 4 + slot] = to_original(prediction.inlier_matches[index],
                                                                        decoded[prediction.view - 1])
                found += 1

    for prediction in views:
        v = prediction.view - 1
        corner_status = prediction.status
        prediction.points = points[v].copy()
        prediction.status = corner_status + tuple(
            PREDICTED if np.isfinite(p).all() else UNAVAILABLE for p in points[v, 4:])
    return Correspondences(points, views, found)


# --------------------------------------------------------------------------- reconstruction

def center_measurements(raw):
    """Subtract each row's mean (the view centroid): returns centroids (F, 2) and the centered matrix."""
    centroids = raw.mean(axis=1, keepdims=True)
    return centroids.reshape(-1, 2), raw - centroids


def measurement_matrix(tracks):
    """tracks (4, 8, 2) -> raw W (8 x 8; rows V1_x, V1_y, ...; columns P1-P8), centroids (4, 2), centered W."""
    tracks = np.asarray(tracks, dtype=np.float64)
    raw = np.concatenate([tracks[v].T for v in range(tracks.shape[0])], axis=0)
    centroids, centered = center_measurements(raw)
    return raw, centroids, centered


def factorize(centered):
    """Rank-3 SVD factorization: W ~ M_hat S_hat with M_hat = U3 Sigma3^(1/2), S_hat = Sigma3^(1/2) V3^T."""
    U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    root = np.diag(np.sqrt(singular_values[:3]))
    return dict(U=U, singular_values=singular_values, Vt=Vt, M_hat=U[:, :3] @ root, S_hat=root @ Vt[:3],
                W3=U[:, :3] @ np.diag(singular_values[:3]) @ Vt[:3])


def symmetric_coefficients(a, b):
    """Coefficients of a L b^T in order L11,L12,L13,L22,L23,L33."""
    return [a[0]*b[0], a[0]*b[1] + a[1]*b[0], a[0]*b[2] + a[2]*b[0],
            a[1]*b[1], a[1]*b[2] + a[2]*b[1], a[2]*b[2]]


def solve_metric(initial_motion):
    """Least-squares L = Q Q^T from unit, orthogonal camera rows in every view (orthographic constraints)."""
    coefficients, target, labels = [], [], []
    for v in range(initial_motion.shape[0] // 2):
        a, b = initial_motion[2*v:2*v+2]
        coefficients.extend([symmetric_coefficients(a, a), symmetric_coefficients(b, b),
                             symmetric_coefficients(a, b)])
        target.extend([1., 1., 0.])
        labels.extend([f"V{v+1}_x_unit", f"V{v+1}_y_unit", f"V{v+1}_xy_orthogonal"])
    A, rhs = np.array(coefficients), np.array(target)
    entries, _, rank, singular_values = np.linalg.lstsq(A, rhs, rcond=None)
    L = np.array([[entries[0], entries[1], entries[2]],
                  [entries[1], entries[3], entries[4]],
                  [entries[2], entries[4], entries[5]]])
    residual = A @ entries - rhs
    eigenvalues = np.linalg.eigvalsh(L)
    # Scale-aware float64 tolerance; never clip eigenvalues or force a solution.
    tolerance = 100 * np.finfo(float).eps * np.linalg.norm(L, ord=2)
    return dict(A=A, rhs=rhs, labels=labels, L=L, residual=residual, rank=int(rank),
                condition=float(np.linalg.cond(A)), constraint_singular_values=singular_values,
                eigenvalues=eigenvalues, tolerance=tolerance,
                positive_definite=bool(np.min(eigenvalues) > tolerance),
                symmetric=bool(np.array_equal(L, L.T)))


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= np.finfo(float).eps:
        raise ValueError("Degenerate frame direction")
    return vector / length


def align(points):
    """Rigid alignment: X along the width (P1->P2, P4->P3), Y along the height, Z the best-fit plane normal.

    points: (8, 3) metric points. Origin = centroid of the four corners. No flattening or rectification.
    """
    centroid = points.mean(axis=0)
    _, spreads, vt = np.linalg.svd(points-centroid, full_matrices=False)
    normal = unit(vt[-1])
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    horizontal = unit((points[1]-points[0]) + (points[2]-points[3]))
    x = unit(horizontal - (horizontal @ normal)*normal)
    z = normal.copy()
    y = unit(np.cross(z, x))
    vertical = (points[3]-points[0]) + (points[2]-points[1])
    if y @ vertical < 0:
        y, z = -y, -z
    basis = np.vstack([x, y, z])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(basis), 1., atol=1e-12):
        raise ArithmeticError("Frame is not orthonormal and right handed")
    origin = points[:4].mean(axis=0)
    book = (points-origin) @ basis.T
    # The corner centroid is generally NOT on the all-point fitted plane.
    plane_z = float((centroid-origin) @ z)
    plane_distances = (points-centroid) @ z
    return dict(centroid=centroid, normal=normal, basis=basis, origin=origin, book=book,
                plane_z=plane_z, distances=plane_distances, rms_spreads=spreads/np.sqrt(len(points)),
                direction_dots=(x@horizontal, y@vertical))


def measure(points):
    """Edge, mean and diagonal lengths of the P1-P4 boundary."""
    pairs = ((0, 1), (3, 2), (0, 3), (1, 2), (0, 2), (1, 3))
    lengths = [float(np.linalg.norm(points[b]-points[a])) for a, b in pairs]
    top, bottom, left, right, diagonal1, diagonal2 = lengths
    return dict(top_width=top, bottom_width=bottom, mean_width=(top+bottom)/2,
                left_height=left, right_height=right, mean_height=(left+right)/2,
                diagonal_P1_P3=diagonal1, diagonal_P2_P4=diagonal2,
                mean_diagonal=(diagonal1+diagonal2)/2)


def scale_factor(width, height, true_width, true_height):
    """One isotropic scale minimizing (s*width - true_width)^2 + (s*height - true_height)^2."""
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    return (width*true_width + height*true_height)/(width**2 + height**2)


def check_dimensions(width_cm, height_cm):
    for name, value in (("width", width_cm), ("height", height_cm)):
        if value is None or not math.isfinite(value) or value <= 0:
            raise PointError(f"The measured {name} must be a positive number of centimeters.")


def check_tracks(tracks):
    tracks = np.asarray(tracks, dtype=np.float64)
    if tracks.shape != (VIEW_COUNT, POINT_COUNT, 2):
        raise PointError(f"Exactly {POINT_COUNT} points are needed in each of the {VIEW_COUNT} photographs.")
    if not np.isfinite(tracks).all():
        raise PointError("Some points are missing. Every point must be verified in all four photographs.")
    for v in range(VIEW_COUNT):
        if len(np.unique(np.rint(tracks[v]), axis=0)) < POINT_COUNT:
            raise PointError(f"Two points in photo {v + 1} are at the same position. Each point must mark a "
                             "different location.")
    return tracks


@dataclass
class Reconstruction:
    tracks: np.ndarray  # (4, 8, 2) verified original pixel coordinates actually used
    width_cm: float
    height_cm: float
    raw: np.ndarray  # 8 x 8 measurement matrix
    centroids: np.ndarray  # (4, 2)
    centered: np.ndarray
    factors: dict  # factorize()
    metric: dict  # solve_metric()
    Q: np.ndarray
    motion: np.ndarray  # M = M_hat Q (8 x 3)
    structure: np.ndarray  # S = Q^-1 S_hat (3 x 8), arbitrary units
    frame: dict  # align()
    unscaled_lengths: dict
    scale: float  # cm per reconstruction unit
    cm: np.ndarray  # (8, 3) aligned, scaled points
    lengths: dict  # cm
    measurements: list  # dicts: key, label, a, b, actual, reconstructed, error (percent)
    view_rms: np.ndarray  # per-view RMS 2D residual of W - M S, pixels

    @property
    def singular_values(self):
        return self.factors["singular_values"]


def reconstruct(tracks, width_cm, height_cm):
    """Full eight-point reconstruction. Raises PointError or ReconstructionError with visitor-safe messages."""
    tracks = check_tracks(tracks)
    check_dimensions(width_cm, height_cm)
    raw, centroids, centered = measurement_matrix(tracks)
    factors = factorize(centered)
    sv = factors["singular_values"]
    if sv[0] <= 0 or sv[2] / sv[0] < DEGENERATE_SIGMA3_RATIO:
        raise images.DegenerateError(
            "The point positions barely change between photographs, so the structure cannot be recovered. "
            "Use four clearly different viewpoints.")
    metric = solve_metric(factors["M_hat"])
    if not metric["positive_definite"]:
        raise images.DegenerateError(
            "The metric upgrade failed: the camera-constraint matrix L is not positive definite, so no valid "
            "reconstruction exists for these points. No correction was applied. Check that every point marks the "
            "same physical location in all four photographs, and use clearly different viewpoints.")
    Q = np.linalg.cholesky(metric["L"])
    motion = factors["M_hat"] @ Q
    structure = np.linalg.solve(Q, factors["S_hat"])
    try:
        frame = align(structure.T)
        unscaled = measure(frame["book"])
        scale = scale_factor(unscaled["mean_width"], unscaled["mean_height"], width_cm, height_cm)
    except (ValueError, ArithmeticError):
        raise images.DegenerateError("The reconstructed corners do not form a usable outline. Check that P1–P4 "
                                     "mark the four corners, in order, in every photograph.") from None
    cm = scale * frame["book"]
    lengths = measure(cm)
    measurements = []
    for key, label, a, b in EDGES:
        actual = width_cm if "width" in key else height_cm if "height" in key else math.hypot(width_cm, height_cm)
        measurements.append(dict(key=key, label=label, a=a, b=b, actual=float(actual), reconstructed=lengths[key],
                                 error=100 * abs(lengths[key] - actual) / actual))
    residual = centered - motion @ structure
    view_rms = np.array([np.sqrt(np.mean(np.sum(residual[2*v:2*v+2]**2, axis=0))) for v in range(VIEW_COUNT)])
    return Reconstruction(tracks=tracks, width_cm=float(width_cm), height_cm=float(height_cm), raw=raw,
                          centroids=centroids, centered=centered, factors=factors, metric=metric, Q=Q,
                          motion=motion, structure=structure, frame=frame, unscaled_lengths=unscaled, scale=scale,
                          cm=cm, lengths=lengths, measurements=measurements, view_rms=view_rms)
