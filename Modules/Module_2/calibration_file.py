"""Save and reload a visitor monocular calibration as a small, validated NPZ file (no Streamlit import, so it is unit-testable).

    export_calibration(model)        -> bytes   computed calibration -> NPZ file contents (nothing is written to disk here)
    parse_calibration(data)          -> dict    NPZ bytes -> validated calibration, or CalibrationFileError
    with_image_size(model, w, h)     -> dict    legacy file + the image size the visitor confirmed -> validated calibration

The loader never unpickles anything (``allow_pickle=False``), inspects the zip container before reading it, requires every field
to have an expected numeric dtype, shape and range, and rejects unknown fields. A valid file is well-formed and plausible; that
does NOT prove it is authentic or accurate, so loaded models carry ``authenticity_verified=False``.

File layout. Field names match the original Module 2 calibration file (camera_calibration.npz), which is read-only here and has
no image size:
    camera_matrix            f (3, 3)   intrinsic matrix K in pixels                       required
    distortion_coefficients  f (5,) or (1, 5)   k1, k2, p1, p2, k3 (OpenCV's standard model)   required
    image_width, image_height   int     size in pixels of the calibration photographs after EXIF orientation. Required for
                                        activation; a file without them is a "legacy" file whose size the visitor must confirm.
    rms_error                f scalar   OpenCV's RMS reprojection error (pixels)
    checkerboard_columns, checkerboard_rows   int   inner corners
    square_size_mm           f scalar
    successful_images        int        accepted calibration photographs
    format_version           int        1 (written by this app; optional when reading)
"""
import io
import zipfile

import numpy as np

FORMAT_VERSION = 1
MAX_FILE_BYTES = 1_000_000
MAX_MEMBER_BYTES = 64 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024
MAX_MEMBERS = 16
IMAGE_SIDE_LIMITS = (100, 20_000)
CORNER_LIMITS = (3, 40)
SQUARE_LIMITS_MM = (0.5, 1000.0)
MAX_RMS_PX = 1.0e4
MAX_PHOTOS = 1000
FOCAL_RANGE_PER_LONG_SIDE = (0.2, 20.0)      # fx, fy as multiples of the longer image side (phone lenses are roughly 0.4-4)
ASPECT_RANGE = (0.8, 1.25)                   # fx / fy
MAX_RADIAL, MAX_TANGENTIAL = 100.0, 1.0

# field -> (allowed dtype kinds, required shape or None for "any of the listed")
SPEC = {
    "camera_matrix": ("f", [(3, 3)]), "distortion_coefficients": ("f", [(5,), (1, 5)]),
    "image_width": ("iu", [()]), "image_height": ("iu", [()]), "rms_error": ("f", [()]),
    "checkerboard_columns": ("iu", [()]), "checkerboard_rows": ("iu", [()]), "square_size_mm": ("f", [()]),
    "successful_images": ("iu", [()]), "format_version": ("iu", [()]),
}
REQUIRED = ("camera_matrix", "distortion_coefficients")


class CalibrationFileError(ValueError):
    """The calibration file is malformed, unsupported, implausible or unsafe to load."""


def _fail(message):
    raise CalibrationFileError(message)


# ---------------------------------------------------------------- validation

def _int_in(value, limits, label):
    if isinstance(value, (bool, np.bool_)) or not float(value).is_integer() or not limits[0] <= value <= limits[1]:
        _fail(f"{label} must be a whole number from {limits[0]} to {limits[1]} (found {value}).")
    return int(value)


def validate_parameters(camera_matrix, distortion_coefficients, size=None):
    """Check K and D (and, when the image size is known, their plausibility for that size). Returns (K float64 (3,3), D float64 (1,5))."""
    K = np.asarray(camera_matrix, np.float64)
    D = np.asarray(distortion_coefficients, np.float64)
    if K.shape != (3, 3):
        _fail(f"The camera matrix must be 3 x 3 (found {K.shape}).")
    if D.shape not in ((5,), (1, 5)):
        _fail(f"Only OpenCV's five-coefficient distortion model (k1, k2, p1, p2, k3) is supported; found shape {D.shape}.")
    D = D.reshape(1, 5)
    if not (np.isfinite(K).all() and np.isfinite(D).all()):
        _fail("The calibration contains NaN or infinite values.")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if fx <= 0 or fy <= 0:
        _fail(f"The focal lengths must be positive (fx = {fx:g}, fy = {fy:g}).")
    if not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-9) or K[1, 0] != 0 or abs(K[0, 1]) > 1e-6 * fx:
        _fail("The camera matrix must have the form [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] (no skew).")
    if not ASPECT_RANGE[0] <= fx / fy <= ASPECT_RANGE[1]:
        _fail(f"fx / fy = {fx / fy:.2f} is implausible for a phone camera (expected {ASPECT_RANGE[0]} to {ASPECT_RANGE[1]}).")
    if not (0 < cx < IMAGE_SIDE_LIMITS[1] and 0 < cy < IMAGE_SIDE_LIMITS[1]):
        _fail(f"The principal point ({cx:g}, {cy:g}) is outside any plausible image.")
    if size is not None:
        w, h = size
        long_side = max(w, h)
        if not (0 < cx < w and 0 < cy < h):
            _fail(f"The principal point ({cx:.1f}, {cy:.1f}) lies outside a {w} x {h} px image. Check the image size.")
        low, high = (FOCAL_RANGE_PER_LONG_SIDE[0] * long_side, FOCAL_RANGE_PER_LONG_SIDE[1] * long_side)
        if not (low <= fx <= high and low <= fy <= high):
            _fail(f"The focal lengths (fx = {fx:g}, fy = {fy:g} px) are implausible for a {w} x {h} px image "
                  f"(expected {low:g} to {high:g} px). Check the image size.")
    k1, k2, p1, p2, k3 = D[0]
    if max(abs(k1), abs(k2), abs(k3)) > MAX_RADIAL or max(abs(p1), abs(p2)) > MAX_TANGENTIAL:
        _fail("The distortion coefficients are outside any plausible range.")
    return K, D


def _build_model(fields, size):
    K, D = validate_parameters(fields["camera_matrix"], fields["distortion_coefficients"], size)
    model = dict(camera_matrix=K, distortion_coefficients=D,
                 image_width=None if size is None else size[0], image_height=None if size is None else size[1],
                 rms_error=None, checkerboard_columns=None, checkerboard_rows=None, square_size_mm=None, successful_images=None)
    if "rms_error" in fields:
        rms = float(fields["rms_error"])
        if not np.isfinite(rms) or not 0 <= rms <= MAX_RMS_PX:
            _fail(f"rms_error must be a finite number from 0 to {MAX_RMS_PX:g} (found {rms}).")
        model["rms_error"] = rms
    for key in ("checkerboard_columns", "checkerboard_rows"):
        if key in fields:
            model[key] = _int_in(fields[key].item(), CORNER_LIMITS, key)
    if "square_size_mm" in fields:
        square = float(fields["square_size_mm"])
        if not np.isfinite(square) or not SQUARE_LIMITS_MM[0] <= square <= SQUARE_LIMITS_MM[1]:
            _fail(f"square_size_mm must be from {SQUARE_LIMITS_MM[0]:g} to {SQUARE_LIMITS_MM[1]:g} (found {square}).")
        model["square_size_mm"] = square
    if "successful_images" in fields:
        model["successful_images"] = _int_in(fields["successful_images"].item(), (1, MAX_PHOTOS), "successful_images")
    return model


def validate_fields(fields):
    """Validate a dict of arrays. Returns a model; image_width / image_height are None when the file does not state them."""
    unknown = sorted(set(fields) - set(SPEC))
    if unknown:
        _fail(f"Unknown field(s) in the calibration file: {', '.join(unknown[:5])}.")
    missing = [key for key in REQUIRED if key not in fields]
    if missing:
        _fail(f"The calibration file is missing required field(s): {', '.join(missing)}.")
    for key, array in fields.items():
        kinds, shapes = SPEC[key]
        if not isinstance(array, np.ndarray) or array.dtype.kind not in kinds:
            _fail(f"Field '{key}' has the wrong data type ({getattr(array, 'dtype', type(array))}).")
        if array.shape not in shapes:
            _fail(f"Field '{key}' has shape {array.shape}; expected {' or '.join(str(s) for s in shapes)}.")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            _fail(f"Field '{key}' contains NaN or infinite values.")
    if "format_version" in fields and fields["format_version"].item() != FORMAT_VERSION:
        _fail(f"Unsupported calibration format version {fields['format_version'].item()} (this app reads version {FORMAT_VERSION}).")
    has_w, has_h = "image_width" in fields, "image_height" in fields
    if has_w != has_h:
        _fail("The file states only one of image_width and image_height.")
    size = None
    if has_w:
        size = (_int_in(fields["image_width"].item(), IMAGE_SIDE_LIMITS, "image_width"),
                _int_in(fields["image_height"].item(), IMAGE_SIDE_LIMITS, "image_height"))
    return _build_model(fields, size)


# ---------------------------------------------------------------- import

def _read_archive(data):
    if not isinstance(data, (bytes, bytearray)):
        data = data.read() if hasattr(data, "read") else bytes(data)
    if not data:
        _fail("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        _fail(f"The file is {len(data) / 1e6:.1f} MB; calibration files are under {MAX_FILE_BYTES / 1e6:.0f} MB.")
    if not zipfile.is_zipfile(io.BytesIO(data)):
        _fail("This is not an NPZ calibration file (not a valid archive).")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_MEMBERS:
                _fail("The archive has an unexpected number of entries.")
            total = 0
            for info in infos:
                name = info.filename
                if not name.endswith(".npy") or "/" in name or "\\" in name or ".." in name or name.startswith("."):
                    _fail(f"The archive contains an unexpected entry ('{name[:40]}'); only plain .npy arrays are allowed.")
                if info.file_size > MAX_MEMBER_BYTES:
                    _fail(f"Archive entry '{name[:40]}' is too large when unpacked.")
                total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                _fail("The archive is too large when unpacked.")
            if archive.testzip() is not None:
                _fail("The archive is corrupted.")
    except zipfile.BadZipFile as error:
        raise CalibrationFileError("The archive is corrupted.") from error
    fields = {}
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            for key in npz.files:
                fields[key] = npz[key]
    except Exception as error:                      # object arrays, truncated members, bad headers, ...
        raise CalibrationFileError(f"The calibration arrays could not be read safely ({type(error).__name__}).") from error
    return fields


def parse_calibration(data):
    """NPZ bytes (or a file-like object) -> calibration model. Raises CalibrationFileError for anything unsafe or invalid.

    The model has the measurement keys (camera_matrix, distortion_coefficients, image_width, image_height) plus the optional
    metadata, ``source='loaded'``, ``needs_image_size`` (True for a legacy file without image_width / image_height),
    ``size_source`` ('file', or None until the visitor confirms one) and ``authenticity_verified=False``.
    """
    try:
        model = validate_fields(_read_archive(data))
    except CalibrationFileError:
        raise
    except Exception as error:                      # defensive: a malformed file must never surface as a crash
        raise CalibrationFileError(f"The calibration file could not be validated ({type(error).__name__}).") from error
    model.update(source="loaded", needs_image_size=model["image_width"] is None, authenticity_verified=False,
                 size_source="file" if model["image_width"] is not None else None)
    return model


def with_image_size(model, width, height, size_source="confirmed_by_visitor"):
    """A legacy model plus the image size the visitor confirmed. Validates the size and re-checks K for that size.

    size_source: 'confirmed_by_visitor' (typed in) or 'detected_from_photo' (read from an original calibration photograph).
    """
    if size_source not in ("confirmed_by_visitor", "detected_from_photo"):
        _fail("Unknown image size source.")
    if not model.get("needs_image_size"):
        _fail("This calibration already states its image size.")
    size = (_int_in(width, IMAGE_SIDE_LIMITS, "The image width"), _int_in(height, IMAGE_SIDE_LIMITS, "The image height"))
    validate_parameters(model["camera_matrix"], model["distortion_coefficients"], size)
    out = dict(model)
    out.update(image_width=size[0], image_height=size[1], needs_image_size=False, size_source=size_source)
    return out


# ---------------------------------------------------------------- export

def export_calibration(model):
    """Computed (or loaded, sized) calibration -> NPZ bytes. Only plain numeric arrays are written; no photographs, no pickles."""
    try:
        if model.get("image_width") is None or model.get("image_height") is None:
            raise ValueError("the image size is unknown")
        fields = dict(
            format_version=np.array(FORMAT_VERSION),
            camera_matrix=np.asarray(model["camera_matrix"], np.float64),
            distortion_coefficients=np.asarray(model["distortion_coefficients"], np.float64).reshape(1, 5),
            image_width=np.array(int(model["image_width"]), np.int64), image_height=np.array(int(model["image_height"]), np.int64))
        for key, dtype in (("rms_error", np.float64), ("checkerboard_columns", np.int64), ("checkerboard_rows", np.int64),
                           ("square_size_mm", np.float64), ("successful_images", np.int64)):
            if model.get(key) is not None:
                fields[key] = np.array(model[key], dtype)
        validate_fields(fields)                     # never write a file this loader would reject
        buffer = io.BytesIO()
        np.savez(buffer, **fields)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"The calibration cannot be exported ({error}).") from error
    return buffer.getvalue()
