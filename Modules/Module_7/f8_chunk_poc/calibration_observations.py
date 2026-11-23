"""F8.29: the checkerboard corner OBSERVATIONS that produced a visitor's stereo calibration, as a documented, versioned, strictly validated record.

Standard library only (numpy is imported lazily by ``corner_arrays``), so ``result_store`` and the worker can both use it.

What is retained
----------------
Exactly the corner sets that ``calibrate_stereo`` received, and nothing else: one record per ACCEPTED calibration pair (pairs whose right-camera ordering was
verified by ``resolve_pair_orderings``). The RIGHT corners are the ORDER-CORRECTED ones (post ``resolve_pair_orderings``), never the raw detections. Corner
coordinates are float64 pixel positions written unrounded (Python's shortest round-trip repr, so a JSON round trip is bit-exact). No image content, no file
paths, no recovery code.

Schema ``f8.calibration_observations/1``::

    schema, pattern{cols, rows}, square_mm, image_size_left [w,h], image_size_right [w,h], n_pairs,
    corner_order, coordinate_space, precision, selection, contains_image_data (always false),
    pairs: [ {pair_id, left_frame, right_frame, left_corners [[x,y] x cols*rows], right_corners [[x,y] x cols*rows]} ],
    observations_sha256   SHA-256 of the canonical JSON of ``pairs``
    calibration_sha256    SHA-256 of the canonical JSON of the calibration parameters these pairs produced (K, D, R, T, sizes, pattern, square)

The two digests tie the observations to the exact parameters stored beside them. They detect accidental mismatch or corruption. They are NOT a signature:
anyone able to edit a file can recompute them.

This module records data. It computes no uncertainty.
"""
from __future__ import annotations

import hashlib
import json
import math

OBS_SCHEMA = "f8.calibration_observations/1"
RESULT_SCHEMA_V1 = "f8.calibration_result/1"
RESULT_SCHEMA_V2 = "f8.calibration_result/2"
SUPPORTED_RESULT_SCHEMAS = (RESULT_SCHEMA_V1, RESULT_SCHEMA_V2)
UNAVAILABLE_TEXT = "calibration uncertainty unavailable"

MAX_PAIRS = 64                       # F7 selects at most 30
MIN_PAIRS = 1
MAX_PATTERN_SIDE = 100               # same range as the NPZ loader
MAX_POINTS = 400
MAX_FRAME_INDEX = 10_000_000
MAX_SQUARE_MM = 10_000.0
MAX_IMAGE_SIDE = 20_000
COORD_MARGIN_PX = 2.0                # a corner may sit at most this far outside the image
MAX_OBS_BYTES = 1_500_000            # canonical (compact) JSON of the whole observation record
MAX_DOC_BYTES = 8 * 1024 * 1024      # same as ResultStore.max_bytes

CORNER_ORDER = "OpenCV row-major board order exactly as passed to calibrate_stereo; RIGHT corners are post resolve_pair_orderings (order-corrected)"
COORD_SPACE = "pixels in the original (distorted) video frame, origin top-left, x right, y down"
PRECISION = "float64, unrounded"
SELECTION = "only the pairs passed to calibrate_stereo (ordering status ok); detections that were not used are not stored"


class ObservationError(ValueError):
    """The observation record is malformed, inconsistent with its calibration, or exceeds a limit."""


def _fail(msg):
    raise ObservationError(msg)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _flat(a, depth=0):
    if depth > 4:
        _fail("calibration parameter nesting is too deep")
    if isinstance(a, (list, tuple)):
        out = []
        for v in a:
            out += _flat(v, depth + 1)
        return out
    if hasattr(a, "tolist"):
        return _flat(a.tolist(), depth + 1)
    if not _is_num(a):
        _fail("calibration parameters must be finite numbers")
    return [float(a)]


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pairs_digest(pairs) -> str:
    return hashlib.sha256(_canon(pairs)).hexdigest()


def calibration_digest(K_left, D_left, K_right, D_right, R, T, image_size_left, image_size_right, pattern, square_mm) -> str:
    """SHA-256 of the calibration parameters (flattened to float lists, so (1,5) and (5,) distortion layouts hash alike)."""
    p = dict(K_left=_flat(K_left), D_left=_flat(D_left), K_right=_flat(K_right), D_right=_flat(D_right), R=_flat(R), T=_flat(T),
             image_size_left=[int(v) for v in _flat(image_size_left)], image_size_right=[int(v) for v in _flat(image_size_right)],
             pattern=[int(v) for v in _flat(pattern)], square_mm=float(square_mm))
    return hashlib.sha256(_canon(p)).hexdigest()


def _points(arr):
    out = []
    for p in (arr.tolist() if hasattr(arr, "tolist") else arr):
        out.append([float(p[0]), float(p[1])])
    return out


def build_observations(*, pattern, square_mm, image_size_left, image_size_right, pair_ids, pair_frames, left_corners, right_corners, params: dict) -> dict:
    """Build the record from the arrays ``calibrate_stereo`` was given. ``params`` holds K1, D1, K2, D2, R, T of the resulting model (for the binding digest).

    Raises ObservationError if the inputs are inconsistent (the caller records that observations are unavailable; it never fails the calibration)."""
    n = len(left_corners)
    if not (n == len(right_corners) == len(pair_frames) == len(pair_ids)):
        _fail("pair lists have different lengths")
    cols, rows = int(pattern[0]), int(pattern[1])
    pairs = [dict(pair_id=int(pair_ids[i]), left_frame=int(pair_frames[i][0]), right_frame=int(pair_frames[i][1]),
                  left_corners=_points(left_corners[i]), right_corners=_points(right_corners[i])) for i in range(n)]
    obs = dict(schema=OBS_SCHEMA, pattern=dict(cols=cols, rows=rows), square_mm=float(square_mm),
               image_size_left=[int(image_size_left[0]), int(image_size_left[1])], image_size_right=[int(image_size_right[0]), int(image_size_right[1])],
               n_pairs=n, corner_order=CORNER_ORDER, coordinate_space=COORD_SPACE, precision=PRECISION, selection=SELECTION, contains_image_data=False,
               pairs=pairs, observations_sha256=pairs_digest(pairs),
               calibration_sha256=calibration_digest(params["K1"], params["D1"], params["K2"], params["D2"], params["R"], params["T"],
                                                     image_size_left, image_size_right, (cols, rows), square_mm))
    validate_observations(obs)
    return obs


_KEYS = {"schema", "pattern", "square_mm", "image_size_left", "image_size_right", "n_pairs", "corner_order", "coordinate_space", "precision", "selection",
         "contains_image_data", "pairs", "observations_sha256", "calibration_sha256"}
_PAIR_KEYS = {"pair_id", "left_frame", "right_frame", "left_corners", "right_corners"}


def _size(v, what):
    if not (isinstance(v, list) and len(v) == 2 and all(_is_int(x) and 16 <= x <= MAX_IMAGE_SIDE for x in v)):
        _fail(f"{what} must be [width, height] integers between 16 and {MAX_IMAGE_SIDE}")
    return v


def _corners(c, n_points, size, what):
    if not (isinstance(c, list) and len(c) == n_points):
        _fail(f"{what} must be a list of exactly {n_points} corners")
    w, h = size
    for p in c:
        if not (isinstance(p, list) and len(p) == 2 and all(_is_num(v) for v in p)):
            _fail(f"{what} contains a corner that is not a finite [x, y] pair")
        if not (-COORD_MARGIN_PX <= p[0] <= w + COORD_MARGIN_PX and -COORD_MARGIN_PX <= p[1] <= h + COORD_MARGIN_PX):
            _fail(f"{what} contains a corner outside the {w} x {h} image")


def validate_observations(obs, *, expected_params: dict | None = None) -> dict:
    """Strict structural validation. With ``expected_params`` (pattern, square_mm, image sizes, digest inputs) also checks the pairing with a calibration.

    Cheap checks run first, so an oversized or malformed record is rejected before any per-coordinate work. Returns ``obs``. Raises ObservationError."""
    if not isinstance(obs, dict):
        _fail("the observation record must be an object")
    if obs.get("schema") != OBS_SCHEMA:
        _fail(f"unsupported observation schema {str(obs.get('schema'))[:60]!r} (this code reads {OBS_SCHEMA})")
    if set(obs) != _KEYS:
        _fail("the observation record has missing or unexpected fields")
    pairs = obs["pairs"]
    if not isinstance(pairs, list):
        _fail("pairs must be a list")
    if not MIN_PAIRS <= len(pairs) <= MAX_PAIRS:
        _fail(f"the number of pairs must be between {MIN_PAIRS} and {MAX_PAIRS} (got {len(pairs)})")
    if not _is_int(obs["n_pairs"]) or obs["n_pairs"] != len(pairs):
        _fail("n_pairs does not match the number of pair records")
    pat = obs["pattern"]
    if not (isinstance(pat, dict) and set(pat) == {"cols", "rows"} and all(_is_int(pat[k]) and 3 <= pat[k] <= MAX_PATTERN_SIDE for k in pat)):
        _fail(f"pattern must be {{cols, rows}} integers between 3 and {MAX_PATTERN_SIDE}")
    n_points = pat["cols"] * pat["rows"]
    if n_points > MAX_POINTS:
        _fail(f"the board has {n_points} corners; the limit is {MAX_POINTS}")
    sq = obs["square_mm"]
    if not (_is_num(sq) and 0 < sq <= MAX_SQUARE_MM):
        _fail("square_mm must be a positive, finite size of at most 10 m")
    size_l, size_r = _size(obs["image_size_left"], "image_size_left"), _size(obs["image_size_right"], "image_size_right")
    if obs["contains_image_data"] is not False:
        _fail("contains_image_data must be false")
    for k in ("corner_order", "coordinate_space", "precision", "selection"):
        if not (isinstance(obs[k], str) and len(obs[k]) <= 400):
            _fail(f"{k} must be a short string")
    for k in ("observations_sha256", "calibration_sha256"):
        if not (isinstance(obs[k], str) and len(obs[k]) == 64 and all(ch in "0123456789abcdef" for ch in obs[k])):
            _fail(f"{k} must be a SHA-256 hex digest")
    try:
        size_bytes = len(_canon(pairs))
    except (TypeError, ValueError, RecursionError):
        _fail("pairs contain values that cannot be serialised (NaN, infinity or an unsupported type)")
    if size_bytes > MAX_OBS_BYTES:
        _fail(f"the observations are {size_bytes} bytes; the limit is {MAX_OBS_BYTES}")
    seen_ids, seen_left = set(), set()
    for i, p in enumerate(pairs):
        if not (isinstance(p, dict) and set(p) == _PAIR_KEYS):
            _fail(f"pair {i} is malformed or incomplete")
        if not (_is_int(p["pair_id"]) and 1 <= p["pair_id"] <= 100_000):
            _fail(f"pair {i} has an invalid pair_id")
        for k in ("left_frame", "right_frame"):
            if not (_is_int(p[k]) and 0 <= p[k] <= MAX_FRAME_INDEX):
                _fail(f"pair {i} has an invalid {k}")
        if p["pair_id"] in seen_ids or p["left_frame"] in seen_left:
            _fail(f"pair {i} repeats a pair_id or a left frame")
        seen_ids.add(p["pair_id"]); seen_left.add(p["left_frame"])
        _corners(p["left_corners"], n_points, size_l, f"pair {i} left_corners")
        _corners(p["right_corners"], n_points, size_r, f"pair {i} right_corners")
    if pairs_digest(pairs) != obs["observations_sha256"]:
        _fail("observations_sha256 does not match the pair records (corrupted or edited)")
    if expected_params is not None:
        e = expected_params
        if [pat["cols"], pat["rows"]] != [int(v) for v in e["pattern"]]:
            _fail("the observations' checkerboard size differs from the calibration's")
        if float(sq) != float(e["square_mm"]):
            _fail("the observations' square size differs from the calibration's")
        if size_l != [int(v) for v in e["image_size_left"]] or size_r != [int(v) for v in e["image_size_right"]]:
            _fail("the observations' image sizes differ from the calibration's")
        try:
            want = calibration_digest(e["K_left"], e["D_left"], e["K_right"], e["D_right"], e["R"], e["T"], e["image_size_left"], e["image_size_right"], e["pattern"], e["square_mm"])
        except (KeyError, TypeError, ObservationError) as ex:
            raise ObservationError("the calibration parameters stored beside the observations are incomplete or malformed") from ex
        if want != obs["calibration_sha256"]:
            _fail("the observations were not produced by the calibration parameters stored beside them (calibration_sha256 differs)")
        if "frame_pairs" in e and e["frame_pairs"] is not None:
            if [[p["left_frame"], p["right_frame"]] for p in pairs] != [list(map(int, fp)) for fp in e["frame_pairs"]]:
                _fail("the observations' frame pairs differ from the selected poses recorded for this calibration")
    return obs


def expected_params_from_document(doc: dict) -> dict:
    """Pull what the observations must agree with out of a result document (``result_store.build_result_document`` layout)."""
    cams, ext, st, sp = doc.get("cameras") or {}, doc.get("extrinsics") or {}, doc.get("settings") or {}, doc.get("selected_poses") or {}
    L, R_ = cams.get("left") or {}, cams.get("right") or {}
    return dict(K_left=L.get("K"), D_left=L.get("distortion"), K_right=R_.get("K"), D_right=R_.get("distortion"), R=ext.get("R"), T=ext.get("T"),
                image_size_left=L.get("image_size_wh"), image_size_right=R_.get("image_size_wh"), pattern=[st.get("cols"), st.get("rows")], square_mm=st.get("square_mm"),
                frame_pairs=sp.get("frame_pairs_left_right"))


def validate_document_observations(doc: dict) -> dict | None:
    """Validate ``doc['calibration_observations']`` against the same document's calibration. Returns the record, or None when the document has none.
    Raises ObservationError when present but invalid, or present in a document that is not schema /2."""
    obs = doc.get("calibration_observations")
    if obs is None:
        return None
    if doc.get("schema") != RESULT_SCHEMA_V2:
        _fail(f"calibration_observations is only allowed in {RESULT_SCHEMA_V2} documents")
    try:
        params = expected_params_from_document(doc)
        return validate_observations(obs, expected_params=params)
    except ObservationError:
        raise
    except Exception as ex:                                       # None parameters, wrong types ...
        raise ObservationError(f"the observations could not be checked against the calibration ({type(ex).__name__})") from ex


def uncertainty_status(doc: dict) -> dict:
    """Whether a result document can support a future calibration-uncertainty estimate. Never invents observations."""
    doc = doc if isinstance(doc, dict) else {}
    obs = doc.get("calibration_observations")
    if obs is not None:
        return dict(available=True, reason=None, n_pairs=obs.get("n_pairs") if isinstance(obs, dict) else None)
    st = doc.get("calibration_observations_status")
    reason = st.get("reason") if isinstance(st, dict) else None
    if doc.get("schema") == RESULT_SCHEMA_V1:
        reason = "saved before observation retention (schema /1)"
    return dict(available=False, reason=reason or "no calibration observations in this result", text=UNAVAILABLE_TEXT)


def parse_result_document(data: bytes | str, *, max_bytes: int = MAX_DOC_BYTES) -> dict:
    """Read a saved/downloaded result JSON (schema /1 or /2). Strict: no NaN/Infinity, size-bounded, observations validated against the calibration.

    Returns dict(document, observations (record or None), uncertainty (see uncertainty_status)). Schema /1 loads with observations None.
    Raises ObservationError for an unsupported schema, bad JSON, oversize input, or invalid observations."""
    raw = data.encode() if isinstance(data, str) else bytes(data)
    if len(raw) > max_bytes:
        _fail(f"the result file is larger than {max_bytes} bytes")

    def _no_const(c):
        raise ValueError(c)
    try:
        doc = json.loads(raw, parse_constant=_no_const)
    except (ValueError, RecursionError) as ex:
        raise ObservationError("the result file is not valid JSON") from ex
    if not isinstance(doc, dict):
        _fail("the result file must contain a JSON object")
    if doc.get("schema") not in SUPPORTED_RESULT_SCHEMAS:
        _fail(f"unsupported result schema {str(doc.get('schema'))[:60]!r} (supported: {', '.join(SUPPORTED_RESULT_SCHEMAS)})")
    obs = validate_document_observations(doc)
    return dict(document=doc, observations=obs, uncertainty=uncertainty_status(doc))


def corner_arrays(obs: dict):
    """Validated record -> (left (n_pairs, N, 2), right (n_pairs, N, 2)) float64 arrays, in stored order. Requires numpy (imported here only)."""
    import numpy as np
    validate_observations(obs)
    return (np.asarray([p["left_corners"] for p in obs["pairs"]], np.float64), np.asarray([p["right_corners"] for p in obs["pairs"]], np.float64))
