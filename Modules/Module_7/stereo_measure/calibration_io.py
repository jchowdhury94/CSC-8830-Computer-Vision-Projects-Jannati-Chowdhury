"""Save and reload a visitor stereo calibration as a small, validated NPZ file (format version 1).

    export_calibration(cal, ...)  -> bytes   calibrate_stereo() result -> NPZ file contents (nothing is written to disk here)
    load_calibration(data)        -> dict    NPZ bytes -> the same model structure calibrate_stereo() returns, or CalibrationFileError

The loader never unpickles anything (``allow_pickle=False``), inspects the zip container before reading it, requires every field to
have the expected dtype kind, shape and range, and recomputes the derived geometry (baseline, E, F) from K, D, R, T instead of
trusting stored values. A stored baseline that disagrees with |T| is rejected as an inconsistent file. Passing validation proves
the file is well-formed and self-consistent, NOT that it is authentic or that the calibration is accurate: the loaded dict is marked
``authenticity_verified=False``.

File layout (all arrays; scalars are 0-d arrays; "-1" marks a metric that was not available):
    format_version        int      1
    kind                  str      "stereo_calibration"
    K_left, K_right       f8 (3,3) intrinsic matrices (pixels)
    D_left, D_right       f8 (5,)  k1, k2, p1, p2, k3
    R                     f8 (3,3) right-from-left rotation (X_R = R X_L + T)
    T                     f8 (3,)  right-from-left translation (mm)
    image_size_left/right i8 (2,)  (width, height) after EXIF orientation
    pattern               i8 (2,)  inner corners (columns, rows)
    square_mm             f8       checkerboard square size
    pairs_uploaded, pairs_valid   int;  pair_ids i8 (pairs_valid,) the user's pair numbers that were used
    rms_left, rms_right, rms_stereo, baseline_mm      f8 (baseline is cross-checked against |T|)
    epipolar_px, rectification_px, board_square_scale f8 (3,) mean, median, max  (-1 = unavailable)
    board_distance_mm     f8 (2,)  nearest and farthest board distance seen during calibration
    distortion_model_left/right   str  radial2 | radial3 | full  (coefficients must obey the model)
    cv_models str (k,), cv_rms_left/right f8 (k,)   held-out RMS per model (k = 0 if the model was set manually; -1 = failed)
    warnings_json, camera_json    str  JSON text (validated structure and size)
    created_utc, software         str
"""
import io
import json
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np

from .user_calibration import DISTORTION_MODELS, MIN_VALID_PAIRS

FORMAT_VERSION = 1
KIND = "stereo_calibration"
SOFTWARE = "csc8830-module7-stereo-calibration/1"
MAX_FILE_BYTES = 1_000_000
MAX_MEMBER_BYTES = 256 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024
MAX_MEMBERS = 64
MAX_IMAGE_SIDE = 20_000
MAX_JSON_CHARS = 20_000
MAX_WARNINGS = 50
ROT_TOL = 1e-6
UNAVAILABLE = -1.0
# field -> (dtype kinds allowed, required shape)
SPEC = {
    "format_version": ("iu", ()), "kind": ("U", ()), "K_left": ("f", (3, 3)), "K_right": ("f", (3, 3)),
    "D_left": ("f", (5,)), "D_right": ("f", (5,)), "R": ("f", (3, 3)), "T": ("f", (3,)),
    "image_size_left": ("iu", (2,)), "image_size_right": ("iu", (2,)), "pattern": ("iu", (2,)), "square_mm": ("f", ()),
    "pairs_uploaded": ("iu", ()), "pairs_valid": ("iu", ()), "pair_ids": ("iu", None),
    "rms_left": ("f", ()), "rms_right": ("f", ()), "rms_stereo": ("f", ()), "baseline_mm": ("f", ()),
    "epipolar_px": ("f", (3,)), "rectification_px": ("f", (3,)), "board_square_scale": ("f", (3,)),
    "board_distance_mm": ("f", (2,)),
    "distortion_model_left": ("U", ()), "distortion_model_right": ("U", ()),
    "cv_models": ("U", None), "cv_rms_left": ("f", None), "cv_rms_right": ("f", None),
    "warnings_json": ("U", ()), "camera_json": ("U", ()), "created_utc": ("U", ()), "software": ("U", ()),
}


class CalibrationFileError(ValueError):
    """The calibration file is malformed, unsupported, inconsistent or unsafe to load."""


def _fail(message):
    raise CalibrationFileError(message)


# ---------------------------------------------------------------- validation (shared by export and import)

def _scalar(f, key):
    return f[key].item()


def _string(f, key, limit):
    s = _scalar(f, key)
    if len(s) > limit:
        _fail(f"Field '{key}' is too long ({len(s)} characters; limit {limit}).")
    return s


def _finite(f, key):
    a = f[key]
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        _fail(f"Field '{key}' contains NaN or infinite values.")
    return a


def _stats(a):
    a = np.asarray(a, np.float64)
    return None if a[0] == UNAVAILABLE else dict(mean=float(a[0]), median=float(a[1]), max=float(a[2]))


def _parse_json(text, what):
    if len(text) > MAX_JSON_CHARS:
        _fail(f"'{what}' is too large.")
    try:
        return json.loads(text)
    except ValueError as error:
        _fail(f"'{what}' is not valid JSON ({error}).")


def _check_fields(f):
    """Structure: exact field set, dtype kinds, shapes, finiteness. `f` maps field -> ndarray."""
    unknown = sorted(set(f) - set(SPEC))
    if unknown:
        _fail(f"Unknown field(s) in calibration file: {', '.join(unknown[:5])}.")
    missing = sorted(set(SPEC) - set(f))
    if missing:
        _fail(f"Calibration file is missing required field(s): {', '.join(missing)}.")
    for key, (kinds, shape) in SPEC.items():
        a = f[key]
        if not isinstance(a, np.ndarray) or a.dtype.kind not in kinds:
            _fail(f"Field '{key}' has the wrong data type ({getattr(a, 'dtype', type(a))}).")
        if shape is not None and a.shape != shape:
            _fail(f"Field '{key}' has shape {a.shape}; expected {shape}.")
        if shape is None and a.ndim != 1:
            _fail(f"Field '{key}' must be a one-dimensional list.")
        if a.size > 10_000:
            _fail(f"Field '{key}' is unreasonably large.")
        _finite(f, key)


def validate_fields(f):
    """Validate a dict of arrays and build the model. Returns (model, load_warnings). Raises CalibrationFileError."""
    _check_fields(f)
    if _scalar(f, "format_version") != FORMAT_VERSION:
        _fail(f"Unsupported calibration format version {_scalar(f, 'format_version')} (this app reads version {FORMAT_VERSION}).")
    if _scalar(f, "kind") != KIND:
        _fail(f"This is not a stereo calibration file (type '{_string(f, 'kind', 60)}').")
    load_warnings = []

    sizes = {}
    for tag in ("left", "right"):
        w, h = (int(v) for v in f[f"image_size_{tag}"])
        if not (16 <= w <= MAX_IMAGE_SIDE and 16 <= h <= MAX_IMAGE_SIDE):
            _fail(f"The {tag} image size {w} x {h} is outside the supported range (16 to {MAX_IMAGE_SIDE} px per side).")
        sizes[tag] = (w, h)

    Ks, Ds = {}, {}
    for tag in ("left", "right"):
        K, D = f[f"K_{tag}"].astype(np.float64), f[f"D_{tag}"].astype(np.float64)
        if not (K[0, 0] > 0 and K[1, 1] > 0):
            _fail(f"The {tag} camera has a non-positive focal length.")
        if np.abs([K[1, 0], K[2, 0], K[2, 1]]).max() != 0 or abs(K[2, 2] - 1) > 1e-9:
            _fail(f"The {tag} intrinsic matrix is not of the form [[fx, s, cx], [0, fy, cy], [0, 0, 1]].")
        if abs(K[0, 1]) > 0.05 * K[0, 0]:
            _fail(f"The {tag} intrinsic matrix has an implausible skew ({K[0, 1]:.3g}).")
        if np.abs(D).max() > 100:
            _fail(f"The {tag} distortion coefficients are implausibly large.")
        w, h = sizes[tag]
        if not (0 <= K[0, 2] <= w and 0 <= K[1, 2] <= h):
            load_warnings.append(f"The {tag} principal point ({K[0, 2]:.0f}, {K[1, 2]:.0f}) lies outside the {w} x {h} image.")
        Ks[tag], Ds[tag] = K, D

    R, T = f["R"].astype(np.float64), f["T"].astype(np.float64)
    if not np.allclose(R @ R.T, np.eye(3), atol=ROT_TOL):
        _fail("The stereo rotation matrix R is not orthogonal.")
    if abs(np.linalg.det(R) - 1) > ROT_TOL:
        _fail(f"The stereo rotation matrix R has determinant {np.linalg.det(R):.6f}; a rotation needs +1.")
    baseline = float(np.linalg.norm(T))
    if not 1e-6 < baseline <= 1e6:
        _fail("The stereo translation T is zero or implausibly large, so the baseline is invalid.")
    stored = float(f["baseline_mm"])
    if abs(stored - baseline) > 1e-6 * max(1.0, baseline):
        _fail(f"The stored baseline ({stored:.6f} mm) disagrees with the translation vector ({baseline:.6f} mm).")

    cols, rows = (int(v) for v in f["pattern"])
    if not (3 <= cols <= 100 and 3 <= rows <= 100):
        _fail(f"The checkerboard size {cols} x {rows} is outside the supported range (3 to 100 inner corners).")
    square = float(f["square_mm"])
    if not 0 < square <= 10_000:
        _fail("The checkerboard square size must be positive (and at most 10 m).")

    valid, uploaded = int(f["pairs_valid"]), int(f["pairs_uploaded"])
    ids = f["pair_ids"].astype(np.int64)
    if not (1 <= valid <= uploaded <= 1000) or len(ids) != valid or len(set(ids.tolist())) != valid or (ids < 1).any():
        _fail("The pair counts are inconsistent (valid pairs must be 1 or more, no more than the uploaded pairs, and match the pair list).")
    if valid < MIN_VALID_PAIRS:
        load_warnings.append(f"This calibration used only {valid} valid pairs (fewer than the recommended minimum of {MIN_VALID_PAIRS}).")

    for key in ("rms_left", "rms_right", "rms_stereo"):
        if not 0 <= float(f[key]) < 1000:
            _fail(f"Field '{key}' must be a reprojection error between 0 and 1000 pixels.")
    for key in ("epipolar_px", "rectification_px", "board_square_scale"):
        a = f[key].astype(np.float64)
        if not (a[0] == UNAVAILABLE and (a == UNAVAILABLE).all()) and not ((a >= 0).all() and a[2] >= a[1] >= 0):
            _fail(f"Field '{key}' must hold non-negative mean/median/max values (or -1 if unavailable).")
    dist = f["board_distance_mm"].astype(np.float64)
    if not (0 < dist[0] <= dist[1] <= 1e7):
        _fail("The board distance range is invalid.")

    names = {}
    for tag in ("left", "right"):
        name = _string(f, f"distortion_model_{tag}", 20)
        if name not in DISTORTION_MODELS:
            _fail(f"Unsupported {tag} distortion model '{name}' (supported: {', '.join(DISTORTION_MODELS)}).")
        k1, k2, p1, p2, k3 = Ds[tag]
        if (name == "radial2" and (k3 != 0 or p1 != 0 or p2 != 0)) or (name == "radial3" and (p1 != 0 or p2 != 0)):
            _fail(f"The {tag} distortion coefficients do not match the declared '{name}' model.")
        names[tag] = name

    cv_models = [str(m) for m in f["cv_models"]]
    k = len(cv_models)
    if k not in (0, len(DISTORTION_MODELS)) or any(m not in DISTORTION_MODELS for m in cv_models) or len(set(cv_models)) != k:
        _fail("The distortion-model cross-validation table is malformed.")
    for key in ("cv_rms_left", "cv_rms_right"):
        a = f[key]
        if len(a) != k or ((a < 0) & (a != UNAVAILABLE)).any():
            _fail(f"Field '{key}' is malformed.")
    if k and any(names[t] not in cv_models for t in names):
        _fail("The selected distortion model is missing from the cross-validation table.")

    warns = _parse_json(_string(f, "warnings_json", MAX_JSON_CHARS), "warnings_json")
    if not isinstance(warns, list) or len(warns) > MAX_WARNINGS or not all(
            isinstance(w, dict) and set(w) == {"code", "message", "heuristic"} and isinstance(w["code"], str) and len(w["code"]) <= 60
            and isinstance(w["message"], str) and len(w["message"]) <= 1000 and isinstance(w["heuristic"], bool) for w in warns):
        _fail("The stored calibration warnings are malformed.")
    camera = _parse_json(_string(f, "camera_json", MAX_JSON_CHARS), "camera_json")
    ok_value = lambda v: v is None or isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) or isinstance(v, str) and len(v) <= 200
    if not (isinstance(camera, dict) and set(camera) <= {"left", "right"} and all(
            isinstance(c, dict) and len(c) <= 20 and all(isinstance(key, str) and len(key) <= 60 and ok_value(v) for key, v in c.items())
            for c in camera.values())):
        _fail("The stored camera information is malformed.")
    created = _string(f, "created_utc", 40)
    try:
        datetime.fromisoformat(created)
    except ValueError:
        _fail("The creation timestamp is not a valid ISO 8601 date.")
    software = _string(f, "software", 100)

    # derived geometry is recomputed from K, D, R, T (stored values, if any, are never used)
    Tc = T.reshape(3, 1)
    tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    E = tx @ R
    F = np.linalg.inv(Ks["right"]).T @ E @ np.linalg.inv(Ks["left"])
    F = F / np.linalg.norm(F)
    model = dict(
        K1=Ks["left"], D1=Ds["left"].reshape(1, 5), K2=Ks["right"], D2=Ds["right"].reshape(1, 5), R=R, T=Tc, E=E, F=F,
        stereo_rms=float(f["rms_stereo"]), rms_left=float(f["rms_left"]), rms_right=float(f["rms_right"]), baseline_mm=baseline,
        image_size=sizes["left"], flags=int(cv2.CALIB_FIX_INTRINSIC), pairs=[int(v) for v in ids],
        valid=True, errors=[], warnings=[dict(w) for w in warns], load_warnings=load_warnings,
        metrics=dict(rms_left=float(f["rms_left"]), rms_right=float(f["rms_right"]), rms_stereo=float(f["rms_stereo"]),
                     baseline_mm=baseline, baseline_direction=(T / baseline).tolist(), epipolar_px=_stats(f["epipolar_px"]),
                     rectification_vertical_px=_stats(f["rectification_px"]), board_square_scale=_stats(f["board_square_scale"])),
        per_pair=[], distortion_left=names["left"], distortion_right=names["right"],
        distortion_cv=dict(**{tag: [dict(model=m, cv_rms=float("inf") if v == UNAVAILABLE else float(v), chosen=m == names[tag])
                                    for m, v in zip(cv_models, f[f"cv_rms_{tag}"])] for tag in ("left", "right")}),
        board=dict(distance_mm=dict(min=float(dist[0]), max=float(dist[1]))), image_size_left=sizes["left"],
        image_size_right=sizes["right"], pattern=(cols, rows), square_mm=square, pairs_uploaded=uploaded, pairs_valid=valid,
        camera=camera, created_utc=created, software=software, source="loaded_file", authenticity_verified=False)
    return model, load_warnings


# ---------------------------------------------------------------- export

def _metric_array(stats):
    return np.array([UNAVAILABLE] * 3) if stats is None else np.array([stats["mean"], stats["median"], stats["max"]], np.float64)


def _clean_info(info):
    out = {}
    for tag in ("left", "right"):
        d = (info or {}).get(tag)
        if isinstance(d, dict):
            out[tag] = {str(k)[:60]: (v if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) else
                                      str(v)[:200] if isinstance(v, str) else None) for k, v in list(d.items())[:20]}
    return out


def calibration_fields(cal, pairs_uploaded=None, camera_info=None, created=None):
    """The dict of arrays that export_calibration writes (validated by the same rules the loader applies)."""
    if not cal.get("valid", False):
        raise ValueError("Only a valid calibration can be exported.")
    try:
        cv_l, cv_r = cal["distortion_cv"]["left"], cal["distortion_cv"]["right"]
        cv_models = [t["model"] for t in cv_l]
        n_valid = len(cal["pairs"])
        f = dict(
            format_version=np.array(FORMAT_VERSION), kind=np.array(KIND),
            K_left=np.asarray(cal["K1"], np.float64), K_right=np.asarray(cal["K2"], np.float64),
            D_left=np.asarray(cal["D1"], np.float64).ravel()[:5].copy(), D_right=np.asarray(cal["D2"], np.float64).ravel()[:5].copy(),
            R=np.asarray(cal["R"], np.float64), T=np.asarray(cal["T"], np.float64).ravel(),
            image_size_left=np.array(cal["image_size_left"], np.int64), image_size_right=np.array(cal["image_size_right"], np.int64),
            pattern=np.array(cal["pattern"], np.int64), square_mm=np.array(float(cal["square_mm"])),
            pairs_uploaded=np.array(n_valid if pairs_uploaded is None else int(pairs_uploaded)), pairs_valid=np.array(n_valid),
            pair_ids=np.array(cal["pairs"], np.int64), rms_left=np.array(float(cal["rms_left"])),
            rms_right=np.array(float(cal["rms_right"])), rms_stereo=np.array(float(cal["stereo_rms"])),
            baseline_mm=np.array(float(cal["baseline_mm"])), epipolar_px=_metric_array(cal["metrics"]["epipolar_px"]),
            rectification_px=_metric_array(cal["metrics"]["rectification_vertical_px"]),
            board_square_scale=_metric_array(cal["metrics"]["board_square_scale"]),
            board_distance_mm=np.array([cal["board"]["distance_mm"]["min"], cal["board"]["distance_mm"]["max"]], np.float64),
            distortion_model_left=np.array(cal["distortion_left"]), distortion_model_right=np.array(cal["distortion_right"]),
            cv_models=np.array(cv_models, dtype="U20"),
            cv_rms_left=np.array([t["cv_rms"] if np.isfinite(t["cv_rms"]) else UNAVAILABLE for t in cv_l], np.float64),
            cv_rms_right=np.array([t["cv_rms"] if np.isfinite(t["cv_rms"]) else UNAVAILABLE for t in cv_r], np.float64),
            warnings_json=np.array(json.dumps([dict(code=w["code"], message=w["message"], heuristic=bool(w["heuristic"]))
                                               for w in cal["warnings"]])),
            camera_json=np.array(json.dumps(_clean_info(camera_info))),
            created_utc=np.array(created or datetime.now(timezone.utc).isoformat(timespec="seconds")), software=np.array(SOFTWARE))
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"The calibration is missing information required for export ({error!r}).") from error
    validate_fields(f)                                         # never write a file this loader would reject
    return f


def export_calibration(cal, pairs_uploaded=None, camera_info=None, created=None):
    """calibrate_stereo() (or load_calibration()) result -> NPZ file contents as bytes. float64 values are stored exactly.

    pairs_uploaded: how many pairs the user uploaded (defaults to the number of valid pairs). camera_info: optional
    {"left": {...}, "right": {...}} of EXIF-style strings/numbers. No photographs or thumbnails are included.
    """
    fields = calibration_fields(cal, pairs_uploaded, camera_info, created)
    buf = io.BytesIO()
    np.savez(buf, **fields)
    data = buf.getvalue()
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("The calibration file would exceed the size limit.")
    return data


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
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
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
            if z.testzip() is not None:
                _fail("The archive is corrupted.")
    except zipfile.BadZipFile as error:
        raise CalibrationFileError("The archive is corrupted.") from error
    fields = {}
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            for key in npz.files:
                fields[key] = npz[key]
    except CalibrationFileError:
        raise
    except Exception as error:                      # object arrays, truncated members, bad headers, ...
        raise CalibrationFileError(f"The calibration arrays could not be read safely ({type(error).__name__}).") from error
    return fields


def load_calibration(data):
    """NPZ bytes (or a file-like object) -> calibration model dict. Raises CalibrationFileError for anything unsafe or invalid.

    The result has the structure of calibrate_stereo() plus source='loaded_file', authenticity_verified=False and
    load_warnings, and can be passed straight to stereo_measure.live.measure_points().
    """
    try:
        model, _ = validate_fields(_read_archive(data))
    except CalibrationFileError:
        raise
    except Exception as error:                      # defensive: never let a malformed file surface as a crash
        raise CalibrationFileError(f"The calibration file could not be validated ({type(error).__name__}: {error}).") from error
    return model
