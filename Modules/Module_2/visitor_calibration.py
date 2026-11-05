"""State and rules behind the visitor "Calibrate New Camera" workflow of Module 2 (no Streamlit import, so it is unit-testable).

One camera, one set of checkerboard photographs, cv2.calibrateCamera. Same calibration model as Module 2's calibration.py: the
standard intrinsic matrix and five distortion coefficients (k1, k2, p1, p2, k3), corners refined to sub-pixel accuracy. Nothing
here reads or writes my saved experiment calibration, and no photograph is kept: each upload becomes a small record
(corners, thumbnail, size, digest). A wizard ``state`` is a plain dict owned by one visitor session.

    new_state()                                          defaults: 8 x 6 inner corners, 20 mm squares, 10 photo slots
    validate_settings(cols, rows, square, unit)          -> (settings, errors); the square size is converted to millimetres
    apply_settings(state, settings)                      clears every stored photo when the board changed
    set_photo_count / go_to / step                       slot count and navigation; photos in hidden slots are kept
    store_photo / remove_photo / remove_all_photos       one photo -> small record
    evaluate(state)                                      per-photo status, counts, usable photos, blockers, input signature
    run_calibration(state)                               -> calibration model, or CalibrationError
    check_measurement_image(model, size)                 -> error message or None
    detect_photo_resolution(data)                        -> (width, height) of one photograph after EXIF orientation, or CalibrationError

Photograph decoding, EXIF orientation, size limits and checkerboard detection are Module 7's generic single-image helpers
(stereo_measure.user_calibration), imported unchanged. Photos are never resized: corner coordinates are in the original
EXIF-oriented pixels, so the intrinsics describe that resolution.
"""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_MODULE_7 = Path(__file__).resolve().parents[1] / "Module_7"
if str(_MODULE_7) not in sys.path:
    sys.path.insert(0, str(_MODULE_7))

from stereo_measure import user_calibration as uc  # noqa: E402  (single-photo helpers only; no stereo function is used)

try:
    from . import calibration_file as cf
except ImportError:                                  # imported as a top-level module (the Streamlit page adds this folder to sys.path)
    import calibration_file as cf

DEFAULT_SETTINGS = {"cols": 8, "rows": 6, "square_value": 20.0, "unit": "mm", "square_mm": 20.0}
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0}
CORNER_LIMITS = cf.CORNER_LIMITS
SQUARE_LIMITS_MM = cf.SQUARE_LIMITS_MM

# Documented minimum-image rule: calibration needs at least MIN_ACCEPTED accepted photographs. This is a conservative project rule,
# not a mathematical requirement (OpenCV can solve with fewer); RECOMMENDED_ACCEPTED is where results become more robust.
MIN_ACCEPTED = 10
RECOMMENDED_ACCEPTED = (15, 25)
MIN_SLOTS, MAX_SLOTS, DEFAULT_SLOTS = MIN_ACCEPTED, 40, MIN_ACCEPTED

# Heuristic thresholds. They only raise warnings; none is an accuracy guarantee for a later object measurement.
RMS_WARN_PER_4000PX = 1.5          # same scaling Module 7 uses for a single camera
COVERAGE_AREA_MIN = 0.5            # fraction of the frame inside the board hulls of all photos together
COVERAGE_CELLS_MIN = 9             # of the 3 x 3 image regions, how many must contain a corner
TILT_SPREAD_MIN_DEG = 20.0         # largest minus smallest board tilt
DISTANCE_RATIO_MIN = 1.2           # farthest / nearest board distance
OUTLIER_FACTOR, OUTLIER_MIN_PX = 3.0, 1.0
PRINCIPAL_POINT_OFFSET_MAX = 0.2   # fraction of the image side
FOCAL_ASPECT_WARN = 0.05
RADIAL_WARN = (3.0, 3.0, 10.0)     # |k1|, |k2|, |k3|

NOT_UPLOADED, ACCEPTED, REJECTED = "Not uploaded", "Accepted", "Rejected"
SOURCE_COMPUTED = "computed"


class CalibrationError(RuntimeError):
    """The wizard cannot produce a calibration (not enough accepted photographs, or the solve failed)."""


def new_state():
    return {"settings": dict(DEFAULT_SETTINGS), "n_slots": DEFAULT_SLOTS, "current": 0, "photos": {}, "slot_gen": {}}


def pattern_of(state):
    return (state["settings"]["cols"], state["settings"]["rows"])


# ---------------------------------------------------------------- settings

def validate_settings(cols, rows, square_value, unit):
    """Return (settings dict or None, list of error strings). Inner-corner counts are whole numbers; the square size is converted to mm."""
    errors = []
    parsed = []
    for label, value in (("columns", cols), ("rows", rows)):
        try:
            ok = value is not None and not isinstance(value, bool) and float(value) == int(value)
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            errors.append(f"The number of inner-corner {label} must be a whole number.")
            continue
        if not CORNER_LIMITS[0] <= int(value) <= CORNER_LIMITS[1]:
            errors.append(f"The number of inner-corner {label} must be between {CORNER_LIMITS[0]} and {CORNER_LIMITS[1]}.")
        parsed.append(int(value))
    if unit not in UNIT_TO_MM:
        errors.append("The square-size unit must be mm or cm.")
        square_mm = None
    else:
        try:
            square_mm = float(square_value) * UNIT_TO_MM[unit]
        except (TypeError, ValueError):
            square_mm = None
        if square_mm is None or not np.isfinite(square_mm) or not SQUARE_LIMITS_MM[0] <= square_mm <= SQUARE_LIMITS_MM[1]:
            errors.append(f"The square size must be a positive number between {SQUARE_LIMITS_MM[0] / UNIT_TO_MM[unit]:g} and "
                          f"{SQUARE_LIMITS_MM[1] / UNIT_TO_MM[unit]:g} {unit}.")
    if errors:
        return None, errors
    return {"cols": parsed[0], "rows": parsed[1], "square_value": float(square_value), "unit": unit, "square_mm": square_mm}, []


def convert_square(value, from_unit, to_unit):
    return round(float(value) * UNIT_TO_MM[from_unit] / UNIT_TO_MM[to_unit], 6)


def apply_settings(state, settings):
    """Adopt validated settings. Returns True when the board changed; every stored detection is then cleared (it was for another board).

    Only the pattern and the square size in mm matter: switching the unit while keeping the same physical size changes nothing.
    """
    old = state["settings"]
    same_board = (old["cols"], old["rows"], old["square_mm"]) == (settings["cols"], settings["rows"], settings["square_mm"])
    state["settings"] = dict(settings)
    if same_board:
        return False
    remove_all_photos(state)
    return True


# ---------------------------------------------------------------- slots and navigation

def set_photo_count(state, n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not MIN_SLOTS <= n <= MAX_SLOTS:
        raise ValueError(f"The number of photo slots must be a whole number from {MIN_SLOTS} to {MAX_SLOTS}.")
    state["n_slots"] = int(n)
    state["current"] = min(state["current"], int(n) - 1)


def go_to(state, index):
    state["current"] = max(0, min(int(index), state["n_slots"] - 1))


def step(state, delta):
    go_to(state, state["current"] + delta)


def store_photo(state, index, name, data):
    """Analyse one uploaded photograph and keep only its small result record. Returns the record."""
    if not 0 <= index < MAX_SLOTS:
        raise ValueError("Unknown photo slot.")
    rec = uc.process_calibration_photo(data, pattern_of(state))
    rec["name"], rec["bytes"] = str(name), len(data or b"")
    state["photos"][index] = rec
    state["slot_gen"][index] = state["slot_gen"].get(index, 0) + 1
    return rec


def remove_photo(state, index):
    state["photos"].pop(index, None)
    state["slot_gen"][index] = state["slot_gen"].get(index, 0) + 1


def remove_all_photos(state):
    state["photos"].clear()
    for key in list(state["slot_gen"]):
        state["slot_gen"][key] += 1


# ---------------------------------------------------------------- evaluation

def evaluate(state):
    """Status of every active photo slot plus the photographs that can go into the calibration.

    Returns dict(slots=[dict(index, number, status, messages, rec)], usable=[dict(index, number, name, corners, rec)], n_uploaded,
    n_accepted, n_rejected, size ((w, h) shared by the accepted photos or None), can_calibrate, blockers, signature).
    """
    photos = state["photos"]
    slots = []
    for i in range(state["n_slots"]):
        rec = photos.get(i)
        slot = dict(index=i, number=i + 1, rec=rec, messages=[], status=NOT_UPLOADED if rec is None else ACCEPTED)
        if rec is not None and rec["status"] != "ok":
            slot["status"] = REJECTED
            slot["messages"].append(rec["message"])
        slots.append(slot)
    n_uploaded = sum(1 for s in slots if s["rec"] is not None)

    # the same file twice would count as two views of one board position
    seen = {}
    for s in slots:
        if s["status"] == ACCEPTED:
            first = seen.setdefault(s["rec"]["digest"], s["index"])
            if first != s["index"]:
                s["status"] = REJECTED
                s["messages"].append(f"This is the same file as photo {first + 1}. Each photograph must show a different board position.")

    # one resolution and orientation for every photo (the most common size wins); resizing would change the intrinsics
    common = Counter(s["rec"]["size"] for s in slots if s["status"] == ACCEPTED).most_common(1)
    size = common[0][0] if common else None
    for s in slots:
        if s["status"] == ACCEPTED and s["rec"]["size"] != size:
            s["status"] = REJECTED
            s["messages"].append(f"The photo is {s['rec']['size'][0]} x {s['rec']['size'][1]} px but the other photos are "
                                 f"{size[0]} x {size[1]} px. Use one resolution and orientation for all calibration photos.")

    usable = [dict(index=s["index"], number=s["number"], name=s["rec"]["name"], corners=s["rec"]["corners"], rec=s["rec"])
              for s in slots if s["status"] == ACCEPTED]
    blockers = []
    if len(usable) < MIN_ACCEPTED:
        blockers.append(f"{len(usable)} accepted photograph{'s' if len(usable) != 1 else ''}; at least {MIN_ACCEPTED} are required.")
    settings = state["settings"]
    signature = hashlib.sha256(json.dumps([pattern_of(state), settings["square_mm"], [u["rec"]["digest"] for u in usable]]).encode()).hexdigest()
    return dict(slots=slots, usable=usable, n_uploaded=n_uploaded, n_accepted=len(usable), n_rejected=n_uploaded - len(usable), size=size,
                can_calibrate=not blockers, blockers=blockers, signature=signature)


# ---------------------------------------------------------------- calibration

def _warn(code, message):
    return dict(code=code, message=message, heuristic=True)


def _coverage(corner_sets, size):
    """Fraction of the frame inside the board hulls of all photos, and how many of the 3 x 3 image regions hold a corner."""
    w, h = size
    s = 1 / 16
    mask = np.zeros((max(1, int(h * s)), max(1, int(w * s))), np.uint8)
    cells = set()
    for c in corner_sets:
        c = np.asarray(c).reshape(-1, 2)
        cv2.fillConvexPoly(mask, cv2.convexHull((c * s).astype(np.int32)), 1)
        cells.update(zip(np.clip((c[:, 0] / w * 3).astype(int), 0, 2), np.clip((c[:, 1] / h * 3).astype(int), 0, 2)))
    return dict(area_fraction=float(mask.mean()), cells_used=len(cells))


def quality_warnings(model, coverage, tilts, distances):
    """Plain-language warnings, all heuristic. Returns a list of dicts(code, message, heuristic)."""
    warnings = []
    n, (w, h) = model["successful_images"], (model["image_width"], model["image_height"])
    if n < RECOMMENDED_ACCEPTED[0]:
        warnings.append(_warn("few_photos", f"Only {n} photographs were used. {RECOMMENDED_ACCEPTED[0]}-{RECOMMENDED_ACCEPTED[1]} "
                                            "photographs from varied positions usually give a more stable calibration."))
    if coverage["area_fraction"] < COVERAGE_AREA_MIN or coverage["cells_used"] < COVERAGE_CELLS_MIN:
        warnings.append(_warn("poor_coverage", f"The board covered {coverage['area_fraction'] * 100:.0f}% of the frame and reached "
                                               f"{coverage['cells_used']} of 9 image regions. Distortion is poorly constrained where the "
                                               "board never appeared; include photos with the board near the edges and corners."))
    if max(tilts) - min(tilts) < TILT_SPREAD_MIN_DEG:
        warnings.append(_warn("low_tilt_variation", f"The board tilt only varied by {max(tilts) - min(tilts):.0f}° between photos. "
                                                    "Tilt the board in different directions to constrain the focal length."))
    if max(distances) / min(distances) < DISTANCE_RATIO_MIN:
        warnings.append(_warn("low_distance_variation", "The board was photographed at nearly the same distance every time. "
                                                        "Vary the distance as well as the tilt."))
    limit = RMS_WARN_PER_4000PX * max(w, h) / 4000
    if model["rms_error"] > limit:
        warnings.append(_warn("high_rms", f"The RMS reprojection error is {model['rms_error']:.2f} px, above the {limit:.2f} px "
                                          f"heuristic for a {w} x {h} px image. Check for blurred photos, a bent board or a wrong square size."))
    errors = np.array([e["error_px"] for e in model["per_image"]])
    outliers = [e["name"] for e in model["per_image"] if e["error_px"] > max(OUTLIER_FACTOR * np.median(errors), OUTLIER_MIN_PX)]
    if outliers:
        warnings.append(_warn("outlier_photos", "These photographs fit the model much worse than the rest and may be worth replacing: "
                                                + ", ".join(outliers[:5]) + (" ..." if len(outliers) > 5 else "") + "."))
    K, D = model["camera_matrix"], model["distortion_coefficients"][0]
    offset = max(abs(K[0, 2] - w / 2) / w, abs(K[1, 2] - h / 2) / h)
    if offset > PRINCIPAL_POINT_OFFSET_MAX:
        warnings.append(_warn("principal_point", f"The estimated principal point is {offset * 100:.0f}% of the image size away from the "
                                                 "centre, which is unusual and can mean the calibration is unstable."))
    if abs(K[0, 0] / K[1, 1] - 1) > FOCAL_ASPECT_WARN:
        warnings.append(_warn("focal_ratio", f"fx and fy differ by {abs(K[0, 0] / K[1, 1] - 1) * 100:.0f}%; phone sensors have square "
                                             "pixels, so this suggests an unstable result."))
    if any(abs(v) > limit_k for v, limit_k in zip(D[[0, 1, 4]], RADIAL_WARN)):
        warnings.append(_warn("large_distortion", "A radial distortion coefficient is unusually large, which often means the "
                                                  "distortion model is over-fitting the photos."))
    return warnings


def run_calibration(state):
    """Calibrate from the accepted photographs with cv2.calibrateCamera. Raises CalibrationError; never returns an unusable model."""
    ev = evaluate(state)
    if not ev["can_calibrate"]:
        raise CalibrationError("Calibration needs more accepted photographs. " + " ".join(ev["blockers"]))
    settings, size, usable = state["settings"], ev["size"], ev["usable"]
    pattern = pattern_of(state)
    obj = uc.board_object_points(pattern, settings["square_mm"])
    corners = [np.asarray(u["corners"], np.float32).reshape(-1, 1, 2) for u in usable]
    try:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera([obj] * len(corners), corners, size, None, None)
    except cv2.error as error:
        raise CalibrationError(f"OpenCV could not compute a calibration from these photographs: {str(error).strip().splitlines()[-1]}") from error
    try:
        K, D = cf.validate_parameters(K, D, size)
    except cf.CalibrationFileError as error:
        raise CalibrationError(f"The calibration result is not usable: {error} Try more varied photographs.") from error

    per_image, distances = [], []
    for u, c, rv, tv in zip(usable, corners, rvecs, tvecs):
        projected = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
        error = float(np.linalg.norm(projected - c.reshape(-1, 2), axis=1).mean())
        per_image.append(dict(number=u["number"], name=u["name"], error_px=error))
        distances.append(float(np.linalg.norm(tv)))
    tilts = [float(np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(rv)[0][2, 2]))))) for rv in rvecs]
    coverage = _coverage([u["corners"] for u in usable], size)
    info, notes = uc.camera_info_from_photos([u["rec"] for u in usable])

    model = dict(camera_matrix=K, distortion_coefficients=D, image_width=int(size[0]), image_height=int(size[1]), rms_error=float(rms),
                 checkerboard_columns=pattern[0], checkerboard_rows=pattern[1], square_size_mm=float(settings["square_mm"]),
                 successful_images=len(usable), source=SOURCE_COMPUTED, size_source="calibration_photos", needs_image_size=False,
                 per_image=per_image, coverage=coverage, tilts_deg=tilts, distances_mm=distances,
                 photos_uploaded=ev["n_uploaded"], photos_rejected=ev["n_rejected"], camera=info, camera_notes=notes,
                 input_signature=ev["signature"], created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    model["warnings"] = quality_warnings(model, coverage, tilts, distances)
    return model


def is_current(state, model):
    """A computed calibration stays valid only while the photographs and board settings it came from are unchanged."""
    return model is not None and model.get("input_signature") == evaluate(state)["signature"]


# ---------------------------------------------------------------- measurement compatibility

def check_measurement_image(calibration, image_size):
    """None when a measurement photograph of image_size (w, h, EXIF-oriented) may use this calibration, else an error message.

    The intrinsics are never rescaled, so the size must match the calibration photos exactly. A matching size does not prove the
    same camera, lens, zoom or focus were used; the page says so.
    """
    if calibration is None:
        return "No calibration is active."
    try:
        cf.validate_parameters(calibration["camera_matrix"], calibration["distortion_coefficients"])
    except (cf.CalibrationFileError, KeyError, TypeError, ValueError) as error:
        return f"The active calibration is invalid and cannot be used for measurement: {error}"
    width, height = calibration.get("image_width"), calibration.get("image_height")
    if not width or not height:
        return "The active calibration does not state the resolution of its calibration photographs, so the measurement photograph cannot be checked against it."
    if (int(width), int(height)) != (int(image_size[0]), int(image_size[1])):
        return (f"The measurement photograph is {image_size[0]} x {image_size[1]} px but the calibration photographs were {width} x {height} px. "
                "The calibration is not rescaled, so measurement is blocked. Use a photograph with the same resolution and orientation, "
                "or calibrate at this resolution.")
    return None


# ---------------------------------------------------------------- resolution of a legacy calibration

def detect_photo_resolution(data):
    """(width, height) of one original calibration photograph, in the EXIF-oriented frame the calibration pipeline uses.

    Uses Module 7's validation (type, file size, pixel limit) and its explicit EXIF orientation, and decodes the pixels so a
    truncated or corrupted file is rejected. Nothing is resized and nothing is kept. Raises CalibrationError with a visitor-readable message.
    """
    try:
        im, orientation, size, _ = uc.inspect_photo(data, min_side=1)
        uc.decode_gray(im, orientation, size)
    except uc.CalibrationPhotoError as error:
        raise CalibrationError(str(error)) from error
    except Exception as error:                       # truncated data, decompression bombs, odd modes
        raise CalibrationError(f"The image could not be decoded ({type(error).__name__}).") from error
    return int(size[0]), int(size[1])
