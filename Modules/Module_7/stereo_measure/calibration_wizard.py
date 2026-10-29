"""State and rules behind the visitor "Calibrate Your Stereo Cameras" wizard (no Streamlit import, so it is unit-testable).

A wizard ``state`` is a plain dict owned by one visitor session (the page keeps it in st.session_state). Nothing here is module-level
or cached, so sessions cannot see each other, and nothing touches Model B or the disk.

    new_state()                              empty wizard (checkerboard settings unset = 0, 8 pair slots)
    validate_settings(cols, rows, square)    -> (settings, errors)
    apply_settings(state, settings)          clears all detections when the settings differ (they depend on the pattern)
    set_pair_count / go_to / step            slot count and navigation; photos in hidden slots are kept
    store_photo / remove_photo               one photo -> small record (corners, thumbnail, metadata); the image is not kept
    toggle_exclude                           explicit pair exclusion
    evaluate(state)                          per-pair status, counts, usable pairs, blockers
    run_calibration(state)                   -> calibration model (user_calibration.calibrate_stereo) or WizardError
    is_current(state, model)                 whether a newly computed calibration still matches the photos and settings

Pair indices are 0-based internally; ``number`` / pair ids are the 1-based numbers the visitor sees.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np

from . import user_calibration as uc

MIN_PAIRS, MAX_PAIRS, DEFAULT_PAIRS = 8, 50, 8     # the cap keeps stored thumbnails and detections small
UNSET_SETTINGS = {"cols": 0, "rows": 0, "square_mm": 0.0}   # 0 = not configured; the visitor must enter the board
DEFAULT_SETTINGS = UNSET_SETTINGS
CORNER_LIMITS = (3, 40)
SQUARE_LIMITS_MM = (0.5, 1000.0)
SIDES = ("left", "right")
NOT_UPLOADED, INCOMPLETE, VALID, AMBIGUOUS, INVALID, EXCLUDED = (
    "Not uploaded", "Incomplete", "Valid", "Ambiguous", "Invalid", "Excluded")
SOURCE_NEW, SOURCE_FILE = "new_calibration", "loaded_file"


class WizardError(RuntimeError):
    """The wizard cannot produce a calibration (not enough valid pairs, or the solve failed)."""


def new_state():
    return {"settings": dict(DEFAULT_SETTINGS), "n_pairs": DEFAULT_PAIRS, "current": 0, "photos": {}, "excluded": set(),
            "slot_gen": {}, "_resolve": None}


def slot_key(index, side):
    return f"{index}:{side}"


def pattern_of(state):
    return (state["settings"]["cols"], state["settings"]["rows"])


# ---------------------------------------------------------------- settings

def validate_settings(cols, rows, square_mm):
    """Return (settings dict or None, list of error strings). Whole-number corner counts; positive finite square size."""
    errors = []
    parsed = []
    unset = [label for label, value in (("inner corners (columns)", cols), ("inner corners (rows)", rows), ("square size (mm)", square_mm))
             if value is None or (not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0)]
    if unset:
        return None, ["Enter your checkerboard settings before uploading photos. Still missing: " + ", ".join(unset) + "."]
    for label, value in (("columns", cols), ("rows", rows)):
        try:
            ok = value is not None and not isinstance(value, bool) and float(value) == int(value)
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            errors.append(f"The number of inner-corner {label} must be a whole number.")
            continue
        value = int(value)
        if not CORNER_LIMITS[0] <= value <= CORNER_LIMITS[1]:
            errors.append(f"The number of inner-corner {label} must be between {CORNER_LIMITS[0]} and {CORNER_LIMITS[1]}.")
        parsed.append(value)
    try:
        square = float(square_mm)
        square_ok = np.isfinite(square) and SQUARE_LIMITS_MM[0] <= square <= SQUARE_LIMITS_MM[1]
    except (TypeError, ValueError):
        square_ok = False
    if not square_ok:
        errors.append(f"The square size must be a number of millimetres between {SQUARE_LIMITS_MM[0]:g} and {SQUARE_LIMITS_MM[1]:g}.")
    if errors:
        return None, errors
    return {"cols": parsed[0], "rows": parsed[1], "square_mm": square}, []


def apply_settings(state, settings):
    """Adopt validated settings. Returns True when they changed; every stored detection is then cleared (it was for another board)."""
    if settings == state["settings"]:
        return False
    state["settings"] = dict(settings)
    for key, rec in list(state["photos"].items()):
        state["photos"][key] = dict(name=rec.get("name", ""), bytes=rec.get("bytes", 0), status="stale", corners=None, thumbnail=None,
                                    size=None, exif=None, digest=rec.get("digest"), square_px=None, grid_residual=None,
                                    subpix_max_shift=None,
                                    message="The checkerboard settings changed after this photo was analysed. Upload it again.")
    state["_resolve"] = None
    return True


# ---------------------------------------------------------------- slots and navigation

def set_pair_count(state, n):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not MIN_PAIRS <= n <= MAX_PAIRS:
        raise ValueError(f"The number of pairs must be a whole number from {MIN_PAIRS} to {MAX_PAIRS}.")
    n = int(n)
    state["n_pairs"] = n
    state["current"] = min(state["current"], n - 1)


def go_to(state, index):
    state["current"] = max(0, min(int(index), state["n_pairs"] - 1))


def step(state, delta):
    go_to(state, state["current"] + delta)


def store_photo(state, index, side, name, data):
    """Analyse one uploaded photo and keep only its small result record. Returns the record."""
    if side not in SIDES or not 0 <= index < MAX_PAIRS:
        raise ValueError("Unknown pair slot.")
    rec = uc.process_calibration_photo(data, pattern_of(state))
    rec["name"], rec["bytes"] = str(name), len(data or b"")
    state["photos"][slot_key(index, side)] = rec
    state["slot_gen"][slot_key(index, side)] = state["slot_gen"].get(slot_key(index, side), 0) + 1
    return rec


def remove_photo(state, index, side):
    state["photos"].pop(slot_key(index, side), None)
    state["slot_gen"][slot_key(index, side)] = state["slot_gen"].get(slot_key(index, side), 0) + 1


def remove_all_photos(state):
    state["photos"].clear()
    state["excluded"].clear()
    state["_resolve"] = None
    for key in list(state["slot_gen"]):
        state["slot_gen"][key] += 1


def reset_all(state):
    """Return the wizard state to its first-visit defaults, in place. Every uploader generation is bumped so no widget can restore a file."""
    gens = {slot_key(i, side): state["slot_gen"].get(slot_key(i, side), 0) + 1 for i in range(MAX_PAIRS) for side in SIDES}
    state.clear()
    state.update(new_state())
    state["slot_gen"] = gens


def toggle_exclude(state, index):
    state["excluded"].symmetric_difference_update({index})


# ---------------------------------------------------------------- evaluation

def _side_label(side):
    return f"{side} camera"


def _resolve_cached(state, candidates, pattern, max_side):
    key = (pattern, max_side, tuple((i, state["photos"][slot_key(i, "left")]["digest"], state["photos"][slot_key(i, "right")]["digest"])
                                    for i in candidates))
    cached = state.get("_resolve")
    if cached is not None and cached[0] == key:
        return cached[1]
    pairs = [(state["photos"][slot_key(i, "left")]["corners"], state["photos"][slot_key(i, "right")]["corners"]) for i in candidates]
    result = uc.resolve_pair_orderings(pairs, pattern, max_side)
    state["_resolve"] = (key, result)
    return result


def evaluate(state):
    """Status of every active pair slot plus the pairs that can go into the calibration.

    Returns dict(pairs=[dict(index, number, status, messages, left, right)], usable=[dict(index, number, left, right)],
    n_uploaded (slots holding both photos), n_valid, can_calibrate, blockers, signature).
    """
    pattern, n = pattern_of(state), state["n_pairs"]
    photos = state["photos"]
    rows = []
    for i in range(n):
        row = dict(index=i, number=i + 1, status=None, messages=[], left=photos.get(slot_key(i, "left")), right=photos.get(slot_key(i, "right")))
        rows.append(row)
        if row["left"] is None and row["right"] is None:
            row["status"] = NOT_UPLOADED
        elif row["left"] is None or row["right"] is None:
            row["status"] = INCOMPLETE
            row["messages"].append(f"Pair {i + 1} is missing its {'left' if row['left'] is None else 'right'} camera photo.")
        for side in SIDES:
            rec = row[side]
            if rec is not None and rec["status"] != "ok":
                row["messages"].append(f"Pair {i + 1}, {_side_label(side)}: {rec['message']}")
    n_uploaded = sum(1 for r in rows if r["left"] is not None and r["right"] is not None)

    candidates = []
    for r in rows:
        if r["status"] is not None:
            continue
        if all(r[s]["status"] == "ok" for s in SIDES):
            candidates.append(r["index"])
        else:
            r["status"] = INVALID
    # an excluded pair is shown as excluded but never enters any consistency check
    for r in rows:
        if r["index"] in state["excluded"]:
            if r["index"] in candidates:
                candidates.remove(r["index"])
            r["status"] = EXCLUDED

    # same photo twice, or left == right
    seen = {s: {} for s in SIDES}
    for i in list(candidates):
        row = rows[i]
        problem = None
        if row["left"]["digest"] == row["right"]["digest"]:
            problem = f"Pair {i + 1}: the left and right photos are the same file."
        for s in SIDES:
            first = seen[s].setdefault(row[s]["digest"], i)
            if problem is None and first != i:
                problem = f"Pair {i + 1}, {_side_label(s)}: this is the same file as pair {first + 1}. Each pair needs its own photographs."
        if problem:
            row["status"], row["messages"] = INVALID, row["messages"] + [problem]
            candidates.remove(i)

    # every photo of one camera must have that camera's resolution and orientation (the majority wins)
    for s in SIDES:
        common = Counter(rows[i][s]["size"] for i in candidates).most_common(1)
        if not common:
            continue
        for i in list(candidates):
            size = rows[i][s]["size"]
            if size != common[0][0]:
                rows[i]["status"] = INVALID
                rows[i]["messages"].append(
                    f"Pair {i + 1}, {_side_label(s)}: the photo is {size[0]} x {size[1]} px but the other {s} photos are "
                    f"{common[0][0][0]} x {common[0][0][1]} px. Use one resolution and orientation per camera.")
                candidates.remove(i)

    usable = []
    ordering = {}
    if len(candidates) >= uc.MIN_PAIRS_TO_VERIFY:
        max_side = max(max(rows[i][s]["size"]) for i in candidates for s in SIDES)
        res = _resolve_cached(state, candidates, pattern, max_side)
        ordering = dict(zip(candidates, res["pairs"]))
    for i in candidates:
        row, verdict = rows[i], ordering.get(i)
        if verdict is None:
            row["status"] = VALID
            row["messages"].append("Checkerboard found in both photos. The left/right corner correspondence is verified once at least "
                                   f"{uc.MIN_PAIRS_TO_VERIFY} pairs are ready.")
            usable.append(dict(index=i, number=i + 1, left=row["left"]["corners"], right=row["right"]["corners"]))
        elif verdict["status"] == "ok":
            row["status"] = VALID
            row["messages"].append(verdict["message"])
            usable.append(dict(index=i, number=i + 1, left=row["left"]["corners"], right=verdict["right_corners"]))
        else:
            row["status"] = INVALID if verdict["status"] == "inconsistent" else AMBIGUOUS
            row["messages"].append(f"Pair {i + 1}: {verdict['message']}")

    n_valid = len(usable)
    blockers = []
    if n_valid < uc.MIN_VALID_PAIRS:
        blockers.append(f"{n_valid} valid pair{'s' if n_valid != 1 else ''}; at least {uc.MIN_VALID_PAIRS} are required.")
    digest = hashlib.sha256(json.dumps([pattern, state["settings"]["square_mm"],
                                        [(u["index"], rows[u["index"]]["left"]["digest"], rows[u["index"]]["right"]["digest"]) for u in usable]]
                                       ).encode()).hexdigest()
    return dict(pairs=rows, usable=usable, n_uploaded=n_uploaded, n_valid=n_valid, can_calibrate=not blockers, blockers=blockers,
                signature=digest)


def validate_run(settings_ok, ev, max_listed=3):
    """Actionable reasons why a calibration cannot run yet, built from the real evaluation. Empty list = ready."""
    if not settings_ok:
        return ["Enter the checkerboard columns, rows, and square size before running calibration."]
    if not any(r["left"] is not None or r["right"] is not None for r in ev["pairs"]):
        return [f"No valid stereo pairs found. Upload at least {uc.MIN_VALID_PAIRS} valid left/right checkerboard pairs."]
    issues = []
    for r in ev["pairs"]:
        if r["status"] == EXCLUDED or r["status"] == NOT_UPLOADED:
            continue
        n = r["number"]
        if r["status"] == INCOMPLETE:
            issues.append(f"Pair {n} is incomplete. Upload both the left and right checkerboard photographs.")
        elif any(rec is not None and rec["status"] != "ok" for rec in (r["left"], r["right"])):
            issues.append(f"Checkerboard corners could not be detected in Pair {n}. Replace the photograph or exclude this pair.")
        elif r["status"] != VALID:
            issues.append(" ".join(r["messages"]) or f"Pair {n} is not valid. Replace its photographs or exclude it.")
    shown = issues[:max_listed]
    if len(issues) > max_listed:
        shown.append(f"{len(issues) - max_listed} more pair(s) have problems; see the pair list above.")
    missing = uc.MIN_VALID_PAIRS - ev["n_valid"]
    if missing > 0:
        shown.append(f"Only {ev['n_valid']} valid stereo pair{'s are' if ev['n_valid'] != 1 else ' is'} available. "
                     f"At least {uc.MIN_VALID_PAIRS} are required. Add {missing} more valid pair{'s' if missing != 1 else ''}.")
    return shown


# ---------------------------------------------------------------- calibration

def run_calibration(state):
    """Calibrate from the valid pairs with user_calibration.calibrate_stereo. Raises WizardError; never returns an invalid model."""
    ev = evaluate(state)
    if not ev["can_calibrate"]:
        raise WizardError("Calibration needs more valid pairs. " + " ".join(ev["blockers"]))
    usable, photos = ev["usable"], state["photos"]
    left_recs = [photos[slot_key(u["index"], "left")] for u in usable]
    right_recs = [photos[slot_key(u["index"], "right")] for u in usable]
    try:
        cal = uc.calibrate_stereo([u["left"] for u in usable], [u["right"] for u in usable], pattern_of(state),
                                  state["settings"]["square_mm"], left_recs[0]["size"], right_recs[0]["size"],
                                  pair_ids=[u["number"] for u in usable])
    except (ValueError, uc.CalibrationFailed, cv2.error) as error:
        raise WizardError(f"Calibration failed: {error}") from error
    if not cal["valid"]:
        raise WizardError("The calibration is not usable: " + " ".join(cal["errors"]))
    info_l, notes_l = uc.camera_info_from_photos(left_recs)
    info_r, notes_r = uc.camera_info_from_photos(right_recs)
    cal.update(camera={"left": info_l, "right": info_r}, camera_notes=[f"Left: {n}" for n in notes_l] + [f"Right: {n}" for n in notes_r],
               pairs_uploaded=ev["n_uploaded"], pairs_valid=ev["n_valid"], source=SOURCE_NEW, authenticity_verified=None,
               input_signature=ev["signature"], created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return cal


def is_current(state, model):
    """A newly computed calibration stays valid only while the photos, exclusions and settings it came from are unchanged.
    A calibration loaded from a file does not depend on the wizard's photos, so it is always current."""
    if model is None:
        return False
    if model.get("source") != SOURCE_NEW:
        return True
    return model.get("input_signature") == evaluate(state)["signature"]
