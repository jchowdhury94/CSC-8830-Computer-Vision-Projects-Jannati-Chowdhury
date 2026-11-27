"""Part B image handling shared by the experiment and the visitor page (no Streamlit imports).

Upload validation, decoding (EXIF orientation, resizing to a working resolution), SIFT features and
ratio-test matching, numbered point markers, and result export. The correspondence method and the
structure-from-motion mathematics are in sfm_pipeline.py.

This module is imported by the page and the experiment script, not run. Inputs: uploaded image bytes
or decoded images. Outputs (returned in memory): images, features, previews and the results CSV.
Requires NumPy, OpenCV, Pillow and pillow-heif for HEIC photos (repository requirements.txt).
"""

import csv
import io
import threading
from dataclasses import dataclass
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

VIEW_COUNT = 4
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
MAX_CONCURRENT_JOBS = 2

_JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
_RANSAC_LOCK = threading.Lock()  # cv2.setRNGSeed is process-global


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


class job_slot:
    """Context manager limiting concurrent feature-matching jobs on the server."""

    def __enter__(self):
        if not _JOB_SLOTS.acquire(blocking=False):
            raise BusyError("The server is busy with other reconstructions. Please try again in a minute.")
        return self

    def __exit__(self, *exc):
        _JOB_SLOTS.release()
        return False


# --------------------------------------------------------------------------- images

@dataclass
class DecodedImage:
    rgb: np.ndarray  # uint8, H x W x 3, upright, at working resolution
    filename: str
    original_size: tuple  # (width, height) after EXIF orientation, before resizing
    scale: float  # working pixels per original pixel (x axis)
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


# --------------------------------------------------------------------------- features

@dataclass
class ViewFeatures:
    points: np.ndarray  # N x 2 float32 (x, y), working pixels
    descriptors: np.ndarray  # N x 128 float32
    responses: np.ndarray  # N float32, SIFT detector response (contrast strength)


def detect_features(rgb, max_features=MAX_FEATURES, mask=None):
    """SIFT keypoints and descriptors; mask (uint8, nonzero = allowed) restricts detection to a region."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    if descriptors is None or len(keypoints) < MIN_KEYPOINTS:
        raise InsufficientMatchesError(
            f"Only {0 if descriptors is None else len(keypoints)} features were found; "
            f"at least {MIN_KEYPOINTS} are needed. Use a textured, well-lit, in-focus object.")
    points = np.array([k.pt for k in keypoints], dtype=np.float32)
    responses = np.array([k.response for k in keypoints], dtype=np.float32)
    return ViewFeatures(points, descriptors.astype(np.float32), responses)


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


# --------------------------------------------------------------------------- previews and markers

THUMB_LONG_SIDE = 320


def jpeg(rgb, quality=85):
    ok, buffer = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError("Could not encode a preview image.")
    return buffer.tobytes()


def image_summary(data, filename):
    """Validated facts and a small JPEG thumbnail for one upload (raises ImageError)."""
    check_file(len(data), filename)
    image = decode_image(data, filename, long_side=THUMB_LONG_SIDE)
    return dict(name=filename, size_bytes=len(data), width=image.original_size[0],
                height=image.original_size[1], thumbnail=jpeg(image.rgb, 80))


MARKER_OUTLINE = (0, 0, 0)
MARKER_COLORS = {  # RGB, all readable on light and dark photo regions thanks to the dark outline
    "automatic": (0, 230, 255),  # cyan: automatically identified or estimated, not yet changed
    "corrected": (255, 210, 0),  # yellow: moved or selected by hand
    "selected": (255, 210, 0),
}
MARKER_ACTIVE = (255, 60, 60)  # red: the point currently being placed


def draw_marker(canvas, center, label, color, radius=7, font_scale=0.55):
    """Dark-outlined ring with a center dot and number, so the marker reads on light and dark regions."""
    center = (int(round(center[0])), int(round(center[1])))
    cv2.circle(canvas, center, radius, MARKER_OUTLINE, 4, cv2.LINE_AA)
    cv2.circle(canvas, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 1, color, -1, cv2.LINE_AA)
    position = (center[0] + radius + 3, center[1] - radius - 1)
    cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, MARKER_OUTLINE, 4, cv2.LINE_AA)
    cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def render_points(rgb, points, statuses, active=None, outline=True, radius=7, font_scale=0.55):
    """Copy of rgb with numbered markers. points: (8, 2) in rgb's pixel coordinates (NaN = not placed).

    The P1-P4 boundary is drawn when all four corners exist; the active point is drawn red and larger.
    """
    canvas = np.ascontiguousarray(rgb).copy()
    points = np.asarray(points, float)
    if outline and np.isfinite(points[:4]).all():
        corners = np.rint(points[:4]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [corners], True, MARKER_OUTLINE, 4, cv2.LINE_AA)
        cv2.polylines(canvas, [corners], True, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (point, status) in enumerate(zip(points, statuses)):
        if not np.isfinite(point).all():
            continue
        if i == active:
            draw_marker(canvas, point, f"P{i + 1}", MARKER_ACTIVE, radius + 3, font_scale + 0.1)
        else:
            draw_marker(canvas, point, f"P{i + 1}", MARKER_COLORS.get(status, MARKER_COLORS["corrected"]), radius,
                        font_scale)
    return canvas


def zoom_window(center, half, size):
    """(left, top, width, height) square of side 2*half around center, shifted to stay inside an image of size."""
    side_x, side_y = min(2 * half, size[0]), min(2 * half, size[1])
    left = int(np.clip(round(center[0]) - side_x // 2, 0, size[0] - side_x))
    top = int(np.clip(round(center[1]) - side_y // 2, 0, size[1] - side_y))
    return left, top, side_x, side_y


# --------------------------------------------------------------------------- export

def points_csv(result, filenames=None):
    """Verified image coordinates, reconstructed points (cm) and the four edge measurements as CSV text."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    header = ["point", "X_cm", "Y_cm", "Z_cm"]
    header += [f"photo{v}_{a}_px" for v in range(1, VIEW_COUNT + 1) for a in ("x", "y")]
    writer.writerow(header)
    for i, xyz in enumerate(result.cm):
        writer.writerow([f"P{i + 1}", *(f"{value:.4f}" for value in xyz),
                         *(f"{value:.2f}" for v in range(VIEW_COUNT) for value in result.tracks[v, i])])
    writer.writerow([])
    writer.writerow(["edge", "measured_cm", "reconstructed_cm", "percentage_error"])
    for m in result.measurements:
        if m["key"].startswith("diagonal"):  # the export lists the four boundary edges shown on the page
            continue
        writer.writerow([m["label"], f"{m['actual']:g}", f"{m['reconstructed']:.4f}", f"{m['error']:.2f}"])
    writer.writerow([])
    writer.writerow(["scale_cm_per_unit", f"{result.scale:.10f}"])
    if filenames:
        writer.writerow(["photos", *filenames])
    return out.getvalue()
