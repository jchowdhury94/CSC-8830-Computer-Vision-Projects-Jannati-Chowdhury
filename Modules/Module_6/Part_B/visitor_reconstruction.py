"""Visitor Structure-from-Motion backend for Module 6 Part B (no Streamlit imports).

Pipeline (the original experiment's Tomasi-Kanade method, with the manual
correspondence step replaced by automatic matching):

    decode images -> SIFT -> ratio-test matching + RANSAC fundamental matrix
    (consecutive views) -> chained tracks seen in every view -> measurement
    matrix W (rows V1_x, V1_y, V2_x, ...; columns = points) -> per-view
    centering -> rank-3 SVD factorization -> explicit orthographic metric
    upgrade (L = Q Q^T, Cholesky; never "fixed" if not positive definite).

The camera model is AFFINE / ORTHOGRAPHIC. The result is an uncalibrated
affine-to-Euclidean reconstruction in arbitrary units, NOT a calibrated
perspective reconstruction and NOT metric unless the caller supplies a known
distance (apply_reference_distance). Nothing here reads or writes the original
experiment's files; the math mirrors factorization_reconstruction.py.
"""

import csv
import io
import math
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # HEIC/HEIF decoding is optional at import time; ImageError explains if missing.
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without pillow-heif
    HEIC_AVAILABLE = False

MIN_VIEWS = 3
MAX_VIEWS = 8
RECOMMENDED_VIEWS = 4
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
ALLOWED_FORMATS = ("JPEG", "PNG", "HEIF", "MPO")  # MPO: multi-picture JPEGs some phones produce
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_PIXELS = 64_000_000  # decoded size cap; a 48 MP phone photo fits
MIN_LONG_SIDE = 320
WORKING_LONG_SIDE = 1600
MAX_FEATURES = 4000
MIN_KEYPOINTS = 50
RATIO = 0.75
RANSAC_THRESHOLD_PX = 1.5
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 5000
MIN_PAIR_INLIERS = 20
MIN_MEDIAN_DISPLACEMENT_PX = 1.0
MIN_TRACKS = 12
TRIM_ROUNDS = 3
TRIM_MAD_FACTOR = 4.0
TRIM_FLOOR_PX = 2.0
DEGENERATE_SIGMA3_RATIO = 1e-4
PLANAR_RATIO = 0.25  # smallest/middle principal spread of the 3D points (book experiment: 0.23)
FEW_POINTS = 30
REPROJ_CAUTION = 0.005  # RMS reprojection error / image diagonal
REPROJ_POOR = 0.015
ENERGY_CAUTION = 0.99
MAX_CONCURRENT_JOBS = 2

_JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
_RANSAC_LOCK = threading.Lock()  # cv2.setRNGSeed is process-global

AFFINE_MODEL = ("Affine/orthographic Tomasi-Kanade factorization (uncalibrated); "
                "not a calibrated perspective reconstruction")
ARBITRARY_UNITS = "arbitrary reconstruction units"


class ReconstructionError(Exception):
    """Base class; str(error) is safe to show to a visitor."""


class ImageError(ReconstructionError):
    pass


class InsufficientMatchesError(ReconstructionError):
    pass


class DegenerateError(ReconstructionError):
    pass


class BusyError(ReconstructionError):
    pass


# --------------------------------------------------------------------------- images

@dataclass
class DecodedImage:
    rgb: np.ndarray  # uint8, H x W x 3, upright, at working resolution
    filename: str
    original_size: tuple  # (width, height) after EXIF orientation, before resizing
    scale: float  # working pixels per original pixel
    orientation: int  # EXIF orientation tag found (1 = none)

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]


def check_file(size, name):
    """Per-file checks (type, emptiness, size). Raises ImageError."""
    extension = Path(name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageError(f"{name}: unsupported file type. Use JPG, PNG or HEIC.")
    if extension in (".heic", ".heif") and not HEIC_AVAILABLE:
        raise ImageError(f"{name}: HEIC/HEIF decoding is not available on this server.")
    if size <= 0:
        raise ImageError(f"{name}: the file is empty.")
    if size > MAX_FILE_BYTES:
        raise ImageError(f"{name}: larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")


def check_uploads(files):
    """files: iterable of (size_bytes, filename). Raises ImageError for unusable sets."""
    files = list(files)
    if not MIN_VIEWS <= len(files) <= MAX_VIEWS:
        raise ImageError(f"Please upload between {MIN_VIEWS} and {MAX_VIEWS} photographs "
                         f"({RECOMMENDED_VIEWS} recommended); you provided {len(files)}.")
    total = 0
    for size, name in files:
        check_file(size, name)
        total += size
    if total > MAX_TOTAL_BYTES:
        raise ImageError(f"Total upload exceeds {MAX_TOTAL_BYTES // (1024 * 1024)} MB.")


def decode_image(data, filename="image", long_side=WORKING_LONG_SIDE):
    """Decode JPEG/PNG/HEIC bytes, apply EXIF orientation, resize so the long side <= long_side."""
    extension = Path(filename).suffix.lower()
    if extension in (".heic", ".heif") and not HEIC_AVAILABLE:
        raise ImageError(f"{filename}: HEIC/HEIF decoding is not available on this server.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImageError(f"{filename}: not a JPEG, PNG or HEIC image.")
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise ImageError(f"{filename}: image has more than {MAX_PIXELS // 1_000_000} megapixels.")
            orientation = int(image.getexif().get(274, 1) or 1)
            if image.format in ("JPEG", "MPO"):
                image.draft("RGB", (long_side, long_side))  # decode at reduced size to save memory
            upright = ImageOps.exif_transpose(image).convert("RGB")
            if orientation in (5, 6, 7, 8):
                width, height = height, width
            original = (width, height)
    except ImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise ImageError(f"{filename}: could not be read as an image.") from None
    if max(original) < MIN_LONG_SIDE:
        raise ImageError(f"{filename}: too small (needs at least {MIN_LONG_SIDE} px on the long side).")
    scale = min(1.0, long_side / max(original))
    if scale < 1.0 or upright.size != original:
        size = (max(1, round(original[0] * scale)), max(1, round(original[1] * scale)))
        if upright.size != size:
            upright = upright.resize(size, Image.LANCZOS)
    return DecodedImage(np.asarray(upright, dtype=np.uint8).copy(), filename, original,
                        upright.size[0] / original[0], orientation)


# --------------------------------------------------------------------------- matching

@dataclass
class ViewFeatures:
    points: np.ndarray  # N x 2 float32 (x, y), working pixels
    descriptors: np.ndarray  # N x 128 float32


def detect_features(rgb, max_features=MAX_FEATURES):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < MIN_KEYPOINTS:
        raise InsufficientMatchesError(
            f"Only {0 if descriptors is None else len(keypoints)} features were found; "
            f"at least {MIN_KEYPOINTS} are needed. Use a textured, well-lit, in-focus object.")
    points = np.array([k.pt for k in keypoints], dtype=np.float32)
    return ViewFeatures(points, descriptors.astype(np.float32))


def match_descriptors(desc_a, desc_b, ratio=RATIO):
    """Lowe ratio test; each descriptor in b is used at most once (best distance wins)."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2), dtype=int)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    best = {}
    for pair in knn:
        if len(pair) < 2 or pair[0].distance >= ratio * pair[1].distance:
            continue
        m = pair[0]
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    return np.array(sorted((m.queryIdx, m.trainIdx) for m in best.values()), dtype=int).reshape(-1, 2)


def ransac_inliers(points_a, points_b, threshold=RANSAC_THRESHOLD_PX):
    """Boolean inlier mask from a RANSAC fundamental matrix (deterministic seed). None if degenerate."""
    if len(points_a) < 8:
        return None
    with _RANSAC_LOCK:
        cv2.setRNGSeed(0)
        matrix, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, threshold,
                                              RANSAC_CONFIDENCE, RANSAC_MAX_ITERS)
    if matrix is None or mask is None or matrix.shape != (3, 3):
        return None
    return mask.ravel().astype(bool)


@dataclass
class PairStats:
    view_a: int
    view_b: int
    keypoints_a: int
    keypoints_b: int
    ratio_matches: int
    inliers: int
    median_displacement_px: float


def match_pair(features_a, features_b, index_a=0, index_b=1):
    """Ratio-test + RANSAC for one pair. Returns (index pairs of inliers, PairStats)."""
    pairs = match_descriptors(features_a.descriptors, features_b.descriptors)
    mask = ransac_inliers(features_a.points[pairs[:, 0]], features_b.points[pairs[:, 1]]) if len(pairs) else None
    inliers = pairs[mask] if mask is not None else np.empty((0, 2), dtype=int)
    displacement = (float(np.median(np.linalg.norm(features_b.points[inliers[:, 1]] -
                                                   features_a.points[inliers[:, 0]], axis=1)))
                    if len(inliers) else 0.0)
    stats = PairStats(index_a + 1, index_b + 1, len(features_a.points), len(features_b.points),
                      len(pairs), len(inliers), displacement)
    if len(inliers) < MIN_PAIR_INLIERS:
        raise InsufficientMatchesError(
            f"Photos {index_a + 1} and {index_b + 1} share too few matching points "
            f"({len(inliers)} reliable of {len(pairs)} candidates; at least {MIN_PAIR_INLIERS} needed). "
            "Consecutive photos must overlap strongly: move the camera in small steps around the object, "
            "and upload the photos in the order you took them.")
    if displacement < MIN_MEDIAN_DISPLACEMENT_PX:
        raise DegenerateError(
            f"Photos {index_a + 1} and {index_b + 1} look almost identical (typical feature shift "
            f"{displacement:.2f} px). Structure from motion needs the camera viewpoint to change.")
    return inliers, stats


def chain_tracks(pair_matches):
    """Chain consecutive-pair matches into tracks present in every view.

    pair_matches[v]: (n, 2) array of (index in view v, index in view v+1).
    Returns a (T, F) int array of keypoint indices, one row per complete track.
    """
    lookups = [dict(map(tuple, m)) for m in pair_matches]
    tracks = []
    for first, second in pair_matches[0]:
        track = [int(first), int(second)]
        for lookup in lookups[1:]:
            nxt = lookup.get(track[-1])
            if nxt is None:
                break
            track.append(int(nxt))
        else:
            tracks.append(track)
    if not tracks:
        return np.empty((0, len(pair_matches) + 1), dtype=int)
    return np.array(tracks, dtype=int)


def track_images(features, progress=None):
    """Match consecutive views and return (tracks (F, P, 2) float64 pixel coords, point indices, [PairStats])."""
    pair_matches, stats = [], []
    for v in range(len(features) - 1):
        if progress:
            progress("Matching photos", (v + 1) / len(features))
        inliers, pair_stats = match_pair(features[v], features[v + 1], v, v + 1)
        pair_matches.append(inliers)
        stats.append(pair_stats)
    chains = chain_tracks(pair_matches)
    if len(chains) < MIN_TRACKS:
        raise InsufficientMatchesError(
            f"Only {len(chains)} points could be followed through all {len(features)} photos "
            f"(at least {MIN_TRACKS} needed). Use photos with more overlap and a more textured object.")
    points = np.stack([features[v].points[chains[:, v]].astype(np.float64) for v in range(len(features))])
    return points, chains[:, 0], stats


# --------------------------------------------------------------------------- reconstruction

def measurement_matrix(tracks):
    """tracks: (F, P, 2) -> raw W (2F x P) with rows V1_x, V1_y, V2_x, ...; centered W; centroids (F, 2)."""
    tracks = np.asarray(tracks, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks must have shape (views, points, 2)")
    if not np.isfinite(tracks).all():
        raise ValueError("tracks must be finite")
    raw = np.concatenate([tracks[v].T for v in range(tracks.shape[0])], axis=0)
    centroids = raw.mean(axis=1, keepdims=True)
    return raw, raw - centroids, centroids.reshape(-1, 2)


def trim_inconsistent(tracks, min_points=MIN_TRACKS, rounds=TRIM_ROUNDS):
    """Drop points whose rank-3 fit residual is a robust outlier; returns (kept index array)."""
    keep = np.arange(tracks.shape[1])
    for _ in range(rounds):
        _, centered, _ = measurement_matrix(tracks[:, keep])
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        residual = np.linalg.norm(centered - (u[:, :3] * s[:3]) @ vt[:3], axis=0)
        median = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - median))
        bad = residual > max(TRIM_FLOOR_PX, median + TRIM_MAD_FACTOR * sigma)
        if not bad.any() or keep.size - bad.sum() < min_points:
            break
        keep = keep[~bad]
    return keep


def _symmetric_coefficients(a, b):
    return [a[0]*b[0], a[0]*b[1] + a[1]*b[0], a[0]*b[2] + a[2]*b[0],
            a[1]*b[1], a[1]*b[2] + a[2]*b[1], a[2]*b[2]]


def solve_metric(initial_motion):
    """Least-squares L = Q Q^T with unit, orthogonal camera rows per view (as the original Step 4)."""
    rows, target = [], []
    for v in range(initial_motion.shape[0] // 2):
        a, b = initial_motion[2*v:2*v+2]
        rows.extend([_symmetric_coefficients(a, a), _symmetric_coefficients(b, b), _symmetric_coefficients(a, b)])
        target.extend([1., 1., 0.])
    A, rhs = np.array(rows), np.array(target)
    entries, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
    L = np.array([[entries[0], entries[1], entries[2]],
                  [entries[1], entries[3], entries[4]],
                  [entries[2], entries[4], entries[5]]])
    residual = A @ entries - rhs
    eigenvalues = np.linalg.eigvalsh(L)
    tolerance = 100 * np.finfo(float).eps * np.linalg.norm(L, ord=2)
    return dict(L=L, residual=residual, rank=int(rank), condition=float(np.linalg.cond(A)),
                eigenvalues=eigenvalues, positive_definite=bool(np.min(eigenvalues) > tolerance))


def principal_axes_frame(points):
    """Rigid (rotation + centering) frame: X = largest spread, Z = smallest. Visualization only."""
    centered = points - points.mean(axis=0)
    _, spreads, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt.copy()
    if np.linalg.det(basis) < 0:
        basis[2] *= -1
    return centered @ basis.T, spreads / math.sqrt(len(points))


@dataclass
class ReconstructionResult:
    status: str  # "ok" or "metric_upgrade_failed"
    message: str
    model: str
    units: str
    n_views: int
    tracks: np.ndarray  # (F, P, 2) pixel coordinates actually used
    point_ids: np.ndarray  # reference-view keypoint index of each track (or input order)
    image_sizes: list  # [(w, h)] working resolution per view
    points_metric: np.ndarray = None  # (P, 3) factorization frame (as the original experiment)
    points: np.ndarray = None  # (P, 3) principal-axes frame, centered; use for display
    colors: np.ndarray = None  # (P, 3) uint8, sampled from the first photo
    point_errors: np.ndarray = None  # (P,) RMS reprojection error per point, pixels
    quality: str = "poor"  # "good" | "caution" | "poor" (heuristic)
    warnings: list = field(default_factory=list)
    pair_stats: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    scale_info: dict = None


def reconstruct_from_tracks(tracks, image_sizes, point_ids=None, colors=None, min_points=MIN_TRACKS,
                            trim_outliers=True, progress=None, pair_stats=()):
    """Tomasi-Kanade rank-3 factorization + metric upgrade of (F, P, 2) pixel tracks."""
    tracks = np.asarray(tracks, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[2] != 2 or not np.isfinite(tracks).all():
        raise ValueError("tracks must be a finite (views, points, 2) array")
    views, initial_points = tracks.shape[:2]
    if views < MIN_VIEWS:
        raise InsufficientMatchesError(f"At least {MIN_VIEWS} views are required.")
    if initial_points < min_points:
        raise InsufficientMatchesError(f"At least {min_points} point tracks are required; got {initial_points}.")
    ids = np.arange(initial_points) if point_ids is None else np.asarray(point_ids)
    keep = trim_inconsistent(tracks, min_points) if trim_outliers else np.arange(initial_points)
    tracks, ids = tracks[:, keep], ids[keep]
    colors = None if colors is None else np.asarray(colors)[keep]
    if progress:
        progress("Factorizing", 0.8)
    _, W, _ = measurement_matrix(tracks)
    U, sv, Vt = np.linalg.svd(W, full_matrices=False)
    if sv[0] <= 0 or sv[2] / sv[0] < DEGENERATE_SIGMA3_RATIO:
        raise DegenerateError("The point motion is almost flat or collinear (rank below 3), so 3D structure "
                              "cannot be recovered. Move the camera around the object, not just sideways.")
    energy = sv ** 2 / np.sum(sv ** 2)
    root = np.diag(np.sqrt(sv[:3]))
    M_hat, S_hat = U[:, :3] @ root, root @ Vt[:3]
    rank3_error = float(np.linalg.norm(sv[3:]) / np.linalg.norm(sv))
    metric = solve_metric(M_hat)
    diagnostics = dict(
        n_views=views, n_points_initial=initial_points, n_points_used=len(keep),
        n_points_trimmed=initial_points - len(keep), matrix_shape=list(W.shape),
        singular_values=sv.tolist(), first_three_energy=float(energy[:3].sum()),
        rank3_relative_error=rank3_error, sigma4_over_sigma3=float(sv[3] / sv[2]) if len(sv) > 3 else 0.0,
        metric_constraint_rank=metric["rank"], metric_constraint_condition=metric["condition"],
        metric_constraint_rms_residual=float(np.linalg.norm(metric["residual"]) / math.sqrt(len(metric["residual"]))),
        L_eigenvalues=metric["eigenvalues"].tolist(), L_positive_definite=metric["positive_definite"],
        coordinate_note=("Reflection (mirror) and global rotation are not determined by the data; "
                         "the display frame aligns principal axes only."))
    common = dict(model=AFFINE_MODEL, n_views=views, tracks=tracks, point_ids=ids,
                  image_sizes=list(image_sizes), pair_stats=list(pair_stats), diagnostics=diagnostics)
    warnings = []
    if not metric["positive_definite"]:
        return ReconstructionResult(
            status="metric_upgrade_failed", units=ARBITRARY_UNITS, quality="poor",
            message=("The metric upgrade failed: the camera-constraint matrix was not positive definite, so no "
                     "valid 3D structure could be computed. This usually means the photos do not fit an "
                     "orthographic camera well (strong perspective, too little viewpoint change, or a nearly flat "
                     "object). No correction was applied. Try photographing from farther away with some zoom, "
                     "with larger viewpoint changes around a non-flat object."),
            warnings=["metric_upgrade_failed"], colors=colors, **common)
    Q = np.linalg.cholesky(metric["L"])
    M, S = M_hat @ Q, np.linalg.solve(Q, S_hat)
    camera_rows = [[float(np.linalg.norm(M[2*v])), float(np.linalg.norm(M[2*v+1])), float(M[2*v] @ M[2*v+1])]
                   for v in range(views)]
    residual = W - M @ S
    per_view = [float(np.sqrt(np.mean(np.sum(residual[2*v:2*v+2] ** 2, axis=0)))) for v in range(views)]
    point_errors = np.sqrt(np.sum(residual.reshape(views, 2, -1) ** 2, axis=1).mean(axis=0))
    points_metric = S.T
    aligned, spreads = principal_axes_frame(points_metric)
    planarity = float(spreads[2] / spreads[1]) if spreads[1] > 0 else 0.0
    overall = float(np.sqrt(np.mean(np.square(per_view))))
    diagonal = float(np.mean([math.hypot(w, h) for w, h in image_sizes]))
    relative = overall / diagonal
    diagnostics.update(camera_rows=camera_rows, reprojection_rms_per_view_px=per_view,
                       reprojection_rms_px=overall, reprojection_rms_relative=relative,
                       principal_spreads=spreads.tolist(), planarity_ratio=planarity)
    quality = "good"
    if len(keep) < FEW_POINTS:
        warnings.append("few_points"); quality = "caution"
    if energy[:3].sum() < ENERGY_CAUTION:
        warnings.append("low_rank3_energy"); quality = "caution"
    if planarity < PLANAR_RATIO:
        warnings.append("near_planar"); quality = "caution"
    if relative > REPROJ_CAUTION:
        warnings.append("high_reprojection_error"); quality = "caution"
    if relative > REPROJ_POOR:
        quality = "poor"
    return ReconstructionResult(
        status="ok", units=ARBITRARY_UNITS, quality=quality, warnings=warnings, colors=colors,
        message="Reconstruction completed (affine/orthographic model, arbitrary units).",
        points_metric=points_metric, points=aligned, point_errors=point_errors, **common)


def sample_colors(rgb, points_xy):
    height, width = rgb.shape[:2]
    x = np.clip(np.rint(points_xy[:, 0]).astype(int), 0, width - 1)
    y = np.clip(np.rint(points_xy[:, 1]).astype(int), 0, height - 1)
    return rgb[y, x]


def _match_views(files, progress):
    """Decode, detect and track. Caller holds a job slot. Returns (images, tracks, point_ids, pair_stats)."""
    images, features = [], []
    for i, (data, name) in enumerate(files):
        if progress:
            progress("Reading photos", i / len(files))
        image = decode_image(data, name)
        images.append(image)
        try:
            features.append(detect_features(image.rgb))
        except InsufficientMatchesError as error:
            raise InsufficientMatchesError(f"{name}: {error}") from None
    tracks, ids, stats = track_images(features, progress)
    return images, tracks, ids, stats


def reconstruct_from_images(files, progress=None):
    """files: list of (bytes, filename) in capture order. Returns a ReconstructionResult.

    Raises ReconstructionError subclasses with visitor-safe messages.
    """
    check_uploads((len(data), name) for data, name in files)
    if not _JOB_SLOTS.acquire(blocking=False):
        raise BusyError("The server is busy with other reconstructions. Please try again in a minute.")
    try:
        images, tracks, ids, stats = _match_views(files, progress)
        colors = sample_colors(images[0].rgb, tracks[0])
        sizes = [image.size for image in images]
        del images
        return reconstruct_from_tracks(tracks, sizes, ids, colors, progress=progress, pair_stats=stats)
    finally:
        _JOB_SLOTS.release()


# --------------------------------------------------------------------------- staged workflow (visitor page)

THUMB_LONG_SIDE = 320
OVERLAY_MAX_WIDTH = 1400
OVERLAY_MAX_LINES = 60
MARKER_MAX_WIDTH = 900
MAX_MARKERS = 40


def _jpeg(rgb, quality=85):
    ok, buffer = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError("Could not encode a preview image.")
    return buffer.tobytes()


def image_summary(data, filename):
    """Validated facts and a small JPEG thumbnail for one upload (raises ImageError)."""
    check_file(len(data), filename)
    image = decode_image(data, filename, long_side=THUMB_LONG_SIDE)
    return dict(name=filename, size_bytes=len(data), width=image.original_size[0],
                height=image.original_size[1], thumbnail=_jpeg(image.rgb, 80))


def _spread_indices(points, count):
    """Deterministic farthest-point sampling over (P, 2) coordinates; sorted original indices."""
    if len(points) <= count:
        return np.arange(len(points))
    chosen = [0]
    distance = np.linalg.norm(points - points[0], axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distance))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(sorted(chosen))


def render_pair_overlay(rgb_a, rgb_b, points_a, points_b, max_lines=OVERLAY_MAX_LINES):
    """Side-by-side photos with lines joining a spread-out subset of the matched points; JPEG bytes."""
    height = max(rgb_a.shape[0], rgb_b.shape[0])
    canvas = np.zeros((height, rgb_a.shape[1] + rgb_b.shape[1], 3), np.uint8)
    canvas[:rgb_a.shape[0], :rgb_a.shape[1]] = rgb_a
    canvas[:rgb_b.shape[0], rgb_a.shape[1]:] = rgb_b
    offset = rgb_a.shape[1]
    scale = min(1.0, OVERLAY_MAX_WIDTH / canvas.shape[1])
    if scale < 1.0:
        canvas = cv2.resize(canvas, (round(canvas.shape[1] * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    shown = np.linspace(0, len(points_a) - 1, min(len(points_a), max_lines)).astype(int)
    for rank, i in enumerate(shown):
        hue = int(179 * rank / max(1, len(shown)))
        color = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])
        a = (round(points_a[i, 0] * scale), round(points_a[i, 1] * scale))
        b = (round((points_b[i, 0] + offset) * scale), round(points_b[i, 1] * scale))
        cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 4, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, b, 4, color, 1, cv2.LINE_AA)
    return _jpeg(canvas)


MARKER_CYAN = (0, 255, 255)
MARKER_OUTLINE = (0, 0, 0)
MARKER_A_COLOR = (255, 230, 0)  # bright yellow
MARKER_B_COLOR = (0, 230, 60)  # bright green
MARKER_LINE_COLOR = (255, 255, 255)
MARKER_RADIUS = 5
MARKER_SELECTED_RADIUS = 9


def marker_base(rgb):
    """First photo shrunk to MARKER_MAX_WIDTH, and the factor applied to its pixel coordinates."""
    scale = min(1.0, MARKER_MAX_WIDTH / rgb.shape[1])
    canvas = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else rgb.copy()
    return canvas, scale


def _draw_marker(canvas, center, label, color, radius, font_scale):
    """Dark-outlined ring and number so the marker reads on both light and dark photo regions."""
    cv2.circle(canvas, center, radius, MARKER_OUTLINE, 4, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, color, 2, cv2.LINE_AA)
    position = (center[0] + radius + 3, center[1] - radius - 1)
    cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, MARKER_OUTLINE, 4, cv2.LINE_AA)
    cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def render_point_markers(rgb, points_xy, indices, selected=None, base=None):
    """First photo with numbered markers (number = index + 1, as in the results table); JPEG bytes.

    selected: optional (a, b) point indices; A is drawn yellow, B green, both larger and joined by a line.
    base: optional (canvas, scale) from marker_base, so a re-render does not need the full-size photo.
    """
    canvas, scale = marker_base(rgb) if base is None else (base[0].copy(), base[1])

    def center_of(i):
        return (round(points_xy[i, 0] * scale), round(points_xy[i, 1] * scale))

    chosen = {} if selected is None else {int(selected[0]): MARKER_A_COLOR, int(selected[1]): MARKER_B_COLOR}
    for i in indices:
        if int(i) not in chosen:
            _draw_marker(canvas, center_of(i), str(int(i) + 1), MARKER_CYAN, MARKER_RADIUS, 0.5)
    if selected is not None and int(selected[0]) != int(selected[1]):
        start, end = center_of(int(selected[0])), center_of(int(selected[1]))
        cv2.line(canvas, start, end, MARKER_OUTLINE, 6, cv2.LINE_AA)
        cv2.line(canvas, start, end, MARKER_LINE_COLOR, 2, cv2.LINE_AA)
    for i, color in chosen.items():
        _draw_marker(canvas, center_of(i), str(i + 1), color, MARKER_SELECTED_RADIUS, 0.7)
    return _jpeg(canvas)


@dataclass
class MatchResult:
    tracks: np.ndarray  # (F, P, 2) pixel coordinates after the consistency check
    point_ids: np.ndarray
    colors: np.ndarray  # (P, 3) uint8 from the first photo
    image_sizes: list
    filenames: list
    pair_stats: list
    n_tracks_found: int  # tracks seen in every photo, before the consistency check
    overlays: list  # JPEG bytes, one per consecutive photo pair
    marker_image: bytes  # JPEG bytes: first photo with numbered candidate points
    marker_candidates: list  # point indices (0-based) shown in marker_image
    marker_base: tuple = None  # (downscaled first photo, coordinate scale) for re-drawing selections

    @property
    def n_tracks(self):
        return int(self.tracks.shape[1])


def find_matches(files, progress=None):
    """Staged workflow step 2: decode, SIFT, ratio test, RANSAC, chain, consistency check, previews.

    The consistency check is the same one reconstruct_from_tracks applies, so the points shown to the
    visitor are exactly the points that get reconstructed.
    """
    check_uploads((len(data), name) for data, name in files)
    if not _JOB_SLOTS.acquire(blocking=False):
        raise BusyError("The server is busy with other reconstructions. Please try again in a minute.")
    try:
        images, tracks, ids, stats = _match_views(files, progress)
        found = tracks.shape[1]
        keep = trim_inconsistent(tracks)
        tracks, ids = tracks[:, keep], ids[keep]
        candidates = _spread_indices(tracks[0], MAX_MARKERS)
        return MatchResult(
            tracks=tracks, point_ids=ids, colors=sample_colors(images[0].rgb, tracks[0]),
            image_sizes=[image.size for image in images], filenames=[image.filename for image in images],
            pair_stats=stats, n_tracks_found=found,
            overlays=[render_pair_overlay(images[v].rgb, images[v + 1].rgb, tracks[v], tracks[v + 1])
                      for v in range(len(images) - 1)],
            marker_image=render_point_markers(images[0].rgb, tracks[0], candidates),
            marker_candidates=[int(i) for i in candidates], marker_base=marker_base(images[0].rgb))
    finally:
        _JOB_SLOTS.release()


def reconstruct_matches(match, scale=None, progress=None):
    """Staged workflow step 4: factorize the tracks from find_matches; optionally apply one known distance.

    scale: None or dict(a=index, b=index, distance=float, unit=str) with 0-based point indices.
    """
    result = reconstruct_from_tracks(match.tracks, match.image_sizes, match.point_ids, match.colors,
                                     trim_outliers=False, progress=progress, pair_stats=match.pair_stats)
    result.diagnostics["n_points_initial"] = match.n_tracks_found
    result.diagnostics["n_points_trimmed"] = match.n_tracks_found - match.n_tracks
    if scale is not None and result.status == "ok":
        try:
            result = apply_reference_distance(result, scale["a"], scale["b"], scale["distance"], scale["unit"])
        except ValueError as error:
            raise ReconstructionError(f"The scale could not be applied: {error}") from None
    return result


# --------------------------------------------------------------------------- scale and plotting

def apply_reference_distance(result, index_a, index_b, distance, unit="cm"):
    """Return a copy scaled so points index_a and index_b are `distance` apart (one isotropic scale)."""
    if result.status != "ok":
        raise ValueError("Cannot scale a failed reconstruction")
    count = len(result.points)
    if not (0 <= index_a < count and 0 <= index_b < count) or index_a == index_b:
        raise ValueError("Choose two different reconstructed points")
    if not (math.isfinite(distance) and distance > 0):
        raise ValueError("The reference distance must be a positive number")
    current = float(np.linalg.norm(result.points[index_a] - result.points[index_b]))
    if current < 1e-9:
        raise ValueError("The two chosen points coincide in the reconstruction")
    factor = distance / current
    info = dict(point_a=index_a, point_b=index_b, distance=float(distance), unit=unit, factor=factor,
                note=("Scaled by one user-supplied distance; absolute accuracy depends on that measurement "
                      "and on the affine model, and is approximate."))
    return replace(result, points=result.points * factor, points_metric=result.points_metric * factor,
                   units=unit, scale_info=info)


def plotly_payload(result):
    """JSON-friendly data for a Plotly 3D scatter (rendering happens in the Streamlit page)."""
    if result.status != "ok":
        raise ValueError("No 3D points are available for a failed reconstruction")
    points = result.points
    colors = (["rgb(%d,%d,%d)" % tuple(c) for c in result.colors] if result.colors is not None
              else ["rgb(201,54,62)"] * len(points))
    unit = result.units
    return dict(
        x=points[:, 0].tolist(), y=points[:, 1].tolist(), z=points[:, 2].tolist(), colors=colors,
        hover=[f"Point {i + 1}<br>reprojection error {e:.2f} px" for i, e in enumerate(result.point_errors)],
        axis_titles=dict(x=f"X ({unit})", y=f"Y ({unit})", z=f"Z ({unit})"), units=unit,
        aspectmode="data", valid_point_count=len(points), model=result.model, quality=result.quality,
        warnings=list(result.warnings),
        match_statistics=[vars(s).copy() for s in result.pair_stats],
        diagnostics=result.diagnostics, scale=result.scale_info)


# --------------------------------------------------------------------------- exports

WARNING_MESSAGES = {
    "few_points": "Few points survived, so the shape is less reliable.",
    "low_rank3_energy": "The measurements do not fit a rank-3 model well; the camera motion or matches may be poor.",
    "near_planar": "The recovered points are nearly flat. Depth from a flat or near-flat object is weak and may be "
                   "model error rather than real relief.",
    "high_reprojection_error": "The reconstruction does not reproduce the photographed points closely "
                               "(high reprojection error), often because of perspective effects.",
    "metric_upgrade_failed": "The metric upgrade failed, so no 3D structure was computed.",
}
UNIT_OPTIONS = ("mm", "cm", "m", "in")


def _unit_label(result):
    return result.units


def points_csv(result):
    """Reconstructed points as CSV text (units in the column names)."""
    if result.status != "ok":
        raise ValueError("No 3D points are available for a failed reconstruction")
    unit = _unit_label(result)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["point", f"X ({unit})", f"Y ({unit})", f"Z ({unit})", "reprojection_error_px",
                     "photo1_x_px", "photo1_y_px"])
    for i, (p, e, xy) in enumerate(zip(result.points, result.point_errors, result.tracks[0]), 1):
        writer.writerow([i, *(f"{v:.6f}" for v in p), f"{e:.4f}", f"{xy[0]:.2f}", f"{xy[1]:.2f}"])
    return out.getvalue()


def points_ply(result):
    """ASCII PLY point cloud with colors sampled from the first photo."""
    if result.status != "ok":
        raise ValueError("No 3D points are available for a failed reconstruction")
    colors = result.colors if result.colors is not None else np.full((len(result.points), 3), 160, np.uint8)
    lines = ["ply", "format ascii 1.0",
             "comment Module 6 Part B structure from motion point cloud",
             f"comment coordinate units: {_unit_label(result)}",
             "comment affine/orthographic Tomasi-Kanade reconstruction; not a calibrated scan",
             f"element vertex {len(result.points)}", "property float x", "property float y", "property float z",
             "property uchar red", "property uchar green", "property uchar blue", "end_header"]
    lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}"
              for p, c in zip(result.points, colors)]
    return "\n".join(lines) + "\n"


def diagnostic_report(result):
    """Plain-text report of the method, statistics, diagnostics and limits."""
    d = result.diagnostics
    lines = ["Module 6 Part B - Structure from Motion reconstruction report", "",
             f"Status: {result.status}", f"Message: {result.message}", f"Model: {result.model}",
             f"Coordinate units: {result.units}"]
    if result.scale_info:
        s = result.scale_info
        lines += [f"Scale: points {s['point_a'] + 1} and {s['point_b'] + 1} set to {s['distance']:g} {s['unit']}",
                  s["note"]]
    else:
        lines += ["Scale: none. Coordinates are in arbitrary reconstruction units, not centimeters or millimeters."]
    lines += ["", "MATCHING", f"Photos: {d['n_views']}",
              f"Tracks followed through every photo: {d['n_points_initial']}",
              f"Tracks discarded as inconsistent: {d['n_points_trimmed']}", f"Points used: {d['n_points_used']}"]
    for s in result.pair_stats:
        lines.append(f"Photos {s.view_a}-{s.view_b}: {s.keypoints_a}/{s.keypoints_b} features, "
                     f"{s.ratio_matches} ratio-test matches, {s.inliers} RANSAC inliers, "
                     f"median shift {s.median_displacement_px:.1f} px")
    lines += ["", "FACTORIZATION", f"Measurement matrix: {d['matrix_shape'][0]} x {d['matrix_shape'][1]} (centered per view)",
              "Singular values: " + ", ".join(f"{v:.6g}" for v in d["singular_values"]),
              f"First-three energy: {100 * d['first_three_energy']:.4f}%",
              f"Rank-3 relative error: {100 * d['rank3_relative_error']:.4f}%",
              f"sigma4 / sigma3: {d['sigma4_over_sigma3']:.4f}", "", "METRIC UPGRADE",
              f"Constraint system rank: {d['metric_constraint_rank']}",
              f"Constraint condition number: {d['metric_constraint_condition']:.6g}",
              f"Constraint RMS residual: {d['metric_constraint_rms_residual']:.6g}",
              "L eigenvalues: " + ", ".join(f"{v:.6g}" for v in d["L_eigenvalues"]),
              f"L positive definite: {d['L_positive_definite']}"]
    if result.status == "ok":
        lines += ["Camera rows (|i|, |j|, i.j): " + "; ".join(f"V{v + 1}: {a:.4f}, {b:.4f}, {c:.4f}"
                                                              for v, (a, b, c) in enumerate(d["camera_rows"])),
                  "", "QUALITY", f"Quality (heuristic): {result.quality}",
                  f"Reprojection RMS: {d['reprojection_rms_px']:.3f} px "
                  f"({100 * d['reprojection_rms_relative']:.3f}% of the image diagonal)",
                  "Per-view reprojection RMS (px): " + ", ".join(f"{v:.3f}" for v in d["reprojection_rms_per_view_px"]),
                  "Principal spreads: " + ", ".join(f"{v:.6g}" for v in d["principal_spreads"]),
                  f"Planarity ratio (smallest/middle spread): {d['planarity_ratio']:.4f}"]
    lines += ["Warnings: " + ("; ".join(WARNING_MESSAGES.get(w, w) for w in result.warnings) or "none"), "",
              "LIMITATIONS",
              "The camera is modeled as affine/orthographic; phone photos are perspective, so errors grow for close-ups.",
              "Reflection (mirror) and global orientation are not determined by the data.",
              "This is an educational reconstruction, not a calibrated metric 3D scan.",
              "A scale, if given, only changes the units; it does not improve the accuracy of the shape."]
    return "\n".join(lines) + "\n"
