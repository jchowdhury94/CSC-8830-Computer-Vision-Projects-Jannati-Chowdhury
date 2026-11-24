"""F8.31C-1C: visitor-facing wording for every reconstruction failure code and every advisory warning (DATA ONLY: no mathematics, no Streamlit, no thresholds).

Kept apart from the reconstruction backend on purpose: wording can change without touching a decision.  Nothing here changes a measured dimension, a Method J
success / failure decision, or a status.  It is looked up AFTER the decision was made and attached to the record; the technical ``code`` and ``message`` are
always preserved beside it for troubleshooting.

Rules the wording follows: say what failed in plain language; do not expose internals (no exception names, paths, parameter values); give a practical next step
where one exists; never call a failure a success; never present an unaccepted number as a result; claim no confidence interval and no guaranteed accuracy.
"""
from __future__ import annotations

# code -> (what happened, what the visitor can do).  The second part may be "" when nothing the visitor does can help.
FAILURE_TEXT = {
    # ---- the visitor's inputs (visitor_input)
    "OWNERSHIP_MISMATCH": ("These inputs belong to another session and were not used.", "Start again from your own session."),
    "CALIBRATION_NOT_ACCEPTED": ("There is no accepted stereo calibration.", "Complete and accept the calibration first."),
    "CALIBRATION_INVALID": ("The accepted calibration is not a valid stereo calibration.", "Calibrate again."),
    "CALIBRATION_CHANGED": ("The calibration changed after the frames were chosen.", "Choose the frames and mark the wall boundaries again."),
    "CALIBRATION_INCONSISTENT": ("The calibration record does not agree with itself, so nothing was measured.", "Calibrate again."),
    "BASELINE_INVALID": ("The measured distance between the two cameras is missing or not a positive number.", "Enter the baseline again and recalibrate."),
    "CALIBRATION_VIDEO_SIZE_MISMATCH": ("The room videos do not have the frame size and orientation of the calibration. Nothing is resized.", "Record the room at the same setting used for calibration."),
    "UPLOAD_EXPIRED": ("An uploaded video is no longer available because it expired.", "Upload the four room videos again."),
    "VIDEO_REPLACED": ("A room video was replaced after the frames were chosen.", "Choose the frames and mark the wall boundaries again."),
    "VIDEO_HASH_MISMATCH": ("A stored room video no longer matches what was uploaded.", "Upload that video again."),
    "FRAME_SET_COUNT": ("The method needs exactly five frame sets.", "Prepare five frame sets."),
    "TOO_FEW_USABLE_SETS": ("Too few frame sets could be used: the wall boundaries could not be found in enough of them.", "Mark the six boundaries again on clearer, well-lit edges, or choose different frames."),
    "FRAME_SET_INCOMPLETE": ("A frame set is incomplete.", "Choose and confirm a left and a right frame for each viewpoint."),
    "FRAMES_NOT_EXTRACTED": ("A confirmed frame has not been extracted yet.", "Go back to the frame step and confirm the frames."),
    "FRAME_STALE": ("A stored frame no longer matches the one you selected.", "Select the frames again."),
    "FRAME_SIZE_MISMATCH": ("A frame does not have the calibrated size. Nothing is resized.", "Record at the calibrated resolution."),
    "ANNOTATION_MISSING": ("A wall boundary is missing or has not been accepted.", "Mark and accept every wall boundary."),
    "ANNOTATION_STALE": ("A wall boundary was marked on a frame that is no longer the selected one.", "Mark it again on the current frame."),
    "ANNOTATION_INVALID": ("A wall boundary cannot be used.", "Mark it again."),
    "REQUEST_INVALID": ("The measurement request is incomplete.", "Go back and check each step."),
    "INPUT_NO_FRAME_SETS": ("No frame sets were provided.", "Choose the frames again."),
    "INPUT_MALFORMED_FRAME_SET": ("The image matches of a frame set could not be used because there were too few or they were malformed.", "Choose frames with more textured floor and wall."),
    "INPUT_LINES_MISSING": ("A wall boundary line is missing for a frame set.", "Mark the wall boundaries again."),
    "INPUT_IMAGE_SIZE_MISMATCH": ("Your calibration and frames are not consistent with each other.", "Calibrate again or choose the frames again."),
    "INPUT_INVALID_CALIBRATION": ("Your calibration and frames are not consistent with each other.", "Calibrate again or choose the frames again."),
    "LINE_RETARGET_FAILED": ("Your calibration and frames are not consistent with each other.", "Calibrate again or choose the frames again."),
    "BASELINE_CHANGED_UNLABELLED": ("Your calibration and frames are not consistent with each other.", "Calibrate again or choose the frames again."),
    # ---- the original Method J rejected the data (method_j): nothing was measured
    "FEATURE_MATCHING_FAILED": ("The cameras did not see enough of the same floor and wall texture.", "Re-record with textured surfaces and overlapping views."),
    "POSE_DEGENERATE_UNRECOVERABLE": ("The relative position of the two cameras could not be determined reliably from these images.", "Choose frames with more shared, textured background."),
    "POSE_JOINT_NOT_CONVERGED": ("The camera position estimate did not settle on these images.", "Choose different frames, or record again with more overlap."),
    "POSE_JOINT_SCALE_ABOVE_LIMIT": ("The images and the calibration disagree more than the method allows.", "Check that the room was recorded at the calibrated setting, or calibrate again."),
    "DATASET_BUILD_FAILED": ("The camera geometry could not be computed from these images. Nothing was concluded about your room.", "Choose different frames."),
    "POSE_SOLVER_EXCEPTION": ("The camera geometry could not be computed from these images. Nothing was concluded about your room.", "Choose different frames."),
    "POSE_NON_FINITE": ("The camera geometry could not be computed from these images. Nothing was concluded about your room.", "Choose different frames."),
    "TRIANGULATION_FAILED": ("The camera geometry could not be computed from these images. Nothing was concluded about your room.", "Choose different frames."),
    "FLOOR_PLANE_FAILED": ("The floor could not be found in one of the frame sets.", "Choose frames where the floor is clearly visible."),
    "REGISTRATION_FAILED": ("The two viewpoints could not be joined.", "Choose frames with more shared background."),
    "WALL_RECONSTRUCTION_FAILED": ("The wall edges could not be reconstructed from these frames.", "Mark the boundaries again or choose different frames."),
    "GEOMETRY_FAILED": ("The wall edges could not be reconstructed from these frames.", "Mark the boundaries again or choose different frames."),
    "NO_USABLE_FRAME_SETS": ("No frame set could be reconstructed, so no dimension was computed.", "Choose different frames or record again."),
    "WIDTH_UNAVAILABLE": ("No width could be computed from these frame sets.", "Choose different frames or mark the boundaries again."),
    "HEIGHT_UNAVAILABLE": ("No height could be computed from these frame sets.", "Choose different frames or mark the boundaries again."),
    "width_not_finite": ("No valid width could be computed from these frame sets.", "Choose different frames or mark the boundaries again."),
    "width_not_positive": ("No valid width could be computed from these frame sets.", "Choose different frames or mark the boundaries again."),
    "height_not_finite": ("No valid height could be computed from these frame sets.", "Choose different frames or mark the boundaries again."),
    "height_not_positive": ("No valid height could be computed from these frame sets.", "Choose different frames or mark the boundaries again."),
    # ---- the machine could not finish (worker_resource): nothing was concluded about the room
    "WORKER_TIMEOUT": ("The measurement took too long and was stopped. Nothing was concluded about your room.", "Try again later."),
    "WORKER_MEMORY_LIMIT": ("The measurement needed more memory than this server allows and was stopped. Nothing was concluded about your room.", "Try again later; if it repeats, the frames may be too large for this server."),
    "WORKER_MEMORY_UNMEASURABLE": ("The measurement was stopped because its memory use could not be monitored. Nothing was concluded about your room.", "Try again later."),
    "INSUFFICIENT_MEMORY": ("This server does not have enough free memory to start the measurement right now. Nothing was started and nothing was concluded about your room.", "Try again in a few minutes."),
    "WORKER_CRASHED": ("The measurement process stopped unexpectedly. Nothing was concluded about your room.", "Try again."),
    "WORKER_OUTPUT_INVALID": ("The measurement process returned an unusable answer. Nothing was concluded about your room.", "Try again."),
    "WORKER_CANCELLED": ("The measurement was cancelled.", ""),
    "WORKER_BUSY": ("Another measurement is running.", "Try again in a moment."),
    "JOB_TOO_LARGE": ("The selected frames or videos are larger than the processing limits allow.", "Use smaller recordings."),
    "WORKSPACE_ERROR": ("A temporary workspace could not be prepared.", "Try again."),
    "INTERNAL_ERROR": ("An internal error stopped the measurement. Nothing was concluded about your room.", "Try again."),
}

GENERIC_FAILURE = ("The measurement could not be completed. Nothing was concluded about your room.", "Try again, or choose different frames.")

# advisory flag -> (what it means, why it may affect reliability, what the visitor can do, can the visitor act?).  ADVISORY ONLY: no number or decision depends on it.
WARNING_TEXT = {
    "POSE_LOW_INLIER_FRACTION": (
        "Only a small share of the image matches agreed on the camera geometry.",
        "A weaker camera fit can shift every dimension that depends on it.",
        "Re-record or choose frames with more textured floor and wall in both views.", True),
    "REGISTRATION_FEW_INLIERS": (
        "Few points were shared between the two viewpoints when they were joined.",
        "With little shared evidence the join of the two views is less certain.",
        "Choose frames where the two viewpoints see more of the same background.", True),
    "REGISTRATION_LOW_INLIER_FRACTION": (
        "Only a small share of the shared points fitted the join of the two viewpoints.",
        "A poor fit between the viewpoints makes the combined geometry less reliable.",
        "Choose frames with more overlap, or frames where nothing moved between recordings.", True),
    "REGISTRATION_HIGH_RMS": (
        "The two viewpoints joined less tightly than usual.",
        "A looser join can distort the distance between the wall ends.",
        "Choose frames with more shared background and check that the scene did not change between recordings.", True),
    "SOME_FRAME_SETS_FAILED": (
        "Some frame sets could not be used; the result comes from the remaining sets.",
        "Fewer sets means less cross-checking between sets.",
        "Choose better frames for the sets that failed, then measure again.", True),
    "SET_DIMENSION_MISSING": (
        "Some frame sets gave a width or a height but not both.",
        "Each dimension is averaged over fewer sets than the others.",
        "Check the wall boundaries in those sets and mark them again if needed.", True),
    "DIMENSION_PARTIAL": (
        "Only one dimension (the width or the height) could be computed.",
        "The other dimension is not available, so the wall is not fully measured.",
        "Mark the missing wall edges again or choose frames that show them clearly.", True),
}
GENERIC_WARNING = (
    "The checks flagged something about this result.",
    "The result may be less reliable than usual.",
    "Treat the numbers as indicative only, or measure again with different frames.", False)

QUESTIONABLE_SUMMARY = "A result was computed, but the checks flagged it. Treat it as indicative only; no accuracy is claimed."
NO_CLAIMS = "Estimates only: no confidence interval and no guaranteed accuracy."


# ------------------------------------------------------------------ three-viewpoint Wall B (additive: the entries above are unchanged)
# The quality checks of ``room_measurement.three_view_quality`` are PROVISIONAL engineering checks, not validated accuracy guarantees; this wording says so where it matters.
_TV_RECORD = "Record the three viewpoints again so that neighbouring views overlap by at least half of the frame and share textured floor and wall areas."
THREE_VIEW_FAILURE_TEXT = {
    "THREE_VIEW_INSUFFICIENT_EVIDENCE": ("Fewer than two frame sets gave a usable reconstruction, so no length is reported.", "Choose different frames, mark the corners again, or record again with more overlap between the viewpoints."),
    "POSE_DEGENERATE": ("The relative position of the two cameras could not be determined reliably in one viewpoint.", "Choose frames with more textured background or record that viewpoint again."),
    "POSE_RESIDUAL": ("The two cameras of one viewpoint do not fit the calibration well enough in this frame set.", "Check that both cameras recorded at the calibrated setting, or choose different frames."),
    "POSE_INLIERS": ("Too few image matches agreed on the camera geometry of one viewpoint.", "Choose frames with more textured floor and wall."),
    "POSE_DEPTH": ("The camera geometry of one viewpoint puts many points behind the cameras.", "Choose different frames or record that viewpoint again."),
    "POSE_FEW_MATCHES": ("Too few image matches between the two cameras of one viewpoint.", "Choose frames with more textured floor and wall."),
    "REG_FEW_INLIERS": ("Too few points were shared between two viewpoints to join them.", _TV_RECORD),
    "REG_INLIER_FRACTION": ("Too small a share of the shared points fitted the join of two viewpoints.", _TV_RECORD),
    "REG_RMS": ("Two viewpoints could not be joined tightly enough.", _TV_RECORD + " Check that nothing moved between the recordings."),
    "REG_COLLINEAR": ("The shared points of two viewpoints lie along a line, so the join is not determined.", _TV_RECORD),
    "REG_PAIRS": ("Two viewpoints share too few usable points to be joined.", _TV_RECORD),
    "FLOOR_FIT": ("The floor could not be fitted reliably in one viewpoint.", "Make sure a large, evenly lit part of the floor is visible in every viewpoint."),
    "FLOOR_FEW_POINTS": ("Too little of the floor was visible in one viewpoint.", "Make sure a large, evenly lit part of the floor is visible in every viewpoint."),
    "FLOOR_ANGLE": ("The floors seen from different viewpoints disagree too much to be the same floor.", "Choose different frames or record again; check that the cameras were not moved between recordings."),
    "CORNER_DIRECTION": ("A wall corner you marked does not come out vertical in 3D.", "Mark the corner edge again with the two points far apart and exactly on the edge in both cameras."),
    "CORNER_OFF_FLOOR": ("A wall corner does not land on the floor of the other viewpoints.", "Mark the corner edges again, or choose different frames."),
    "LENGTH_NOT_HORIZONTAL": ("The two wall ends are at very different heights, so the length is not a floor distance.", "Check that the left and right corners are the two ends of the same wall."),
    "LENGTH_OUT_OF_RANGE": ("The computed length is outside the plausible range for a room wall.", "Check that all three viewpoints show the same wall and that the corners were marked at its ends."),
    "LENGTH_INVALID": ("No valid length could be computed.", "Check the corners and choose different frames."),
    "RECONSTRUCTION_FAILED": ("This frame set could not be reconstructed.", "Choose different frames or mark the corners again."),
}
# the same codes as advisory warnings: (what it means, why it may affect reliability, what the visitor can do, can the visitor act?)
THREE_VIEW_WARNING_TEXT = {
    "POSE_RESIDUAL": ("One viewpoint's two cameras fit the calibration less tightly than usual.", "A looser fit can shift the length.", "Check that both cameras recorded at the calibrated setting.", True),
    "POSE_INLIERS": ("Only some of the image matches agreed on one viewpoint's camera geometry.", "A weaker camera fit can shift the length.", "Choose frames with more textured floor and wall.", True),
    "POSE_DEPTH": ("Some points of one viewpoint fell behind the cameras.", "The camera geometry of that viewpoint is less certain.", "Choose different frames.", True),
    "POSE_FEW_MATCHES": ("One viewpoint had fewer image matches than usual.", "A weaker camera fit can shift the length.", "Choose frames with more textured floor and wall.", True),
    "POSE_FALLBACK_START": ("One viewpoint's camera geometry needed a fallback starting point.", "The fit was reached less directly than usual.", "Check the previews of that viewpoint; choose different frames if unsure.", True),
    "REG_FEW_INLIERS": ("Fewer points than usual were shared between two viewpoints.", "With little shared evidence the join of the views is less certain.", _TV_RECORD, True),
    "REG_INLIER_FRACTION": ("A smaller share of the shared points fitted the join of two viewpoints.", "A poorer join makes the combined geometry less reliable.", _TV_RECORD, True),
    "REG_RMS": ("Two viewpoints joined less tightly than usual.", "A looser join can distort the distance between the wall ends.", _TV_RECORD, True),
    "REG_COLLINEAR": ("The shared points of two viewpoints cover a narrow strip.", "The join is less well determined across the strip.", _TV_RECORD, True),
    "REG_PLANAR": ("The shared points of two viewpoints are almost flat.", "The join is less well determined perpendicular to them.", "Record with some furniture or wall detail visible in both views.", True),
    "REG_COVERAGE": ("The shared points of two viewpoints cover only a small part of the picture.", "Unmatched areas are not checked; the join is less well constrained. This is a caution, not an error.", "Record with more of the scene visible in both viewpoints.", True),
    "FLOOR_FIT": ("The floor fitted less tightly than usual in one viewpoint.", "The floor is used to place the wall corners.", "Make sure a large, evenly lit part of the floor is visible.", True),
    "FLOOR_FEW_POINTS": ("Less of the floor was visible than usual in one viewpoint.", "The floor is used to place the wall corners.", "Make sure a large part of the floor is visible.", True),
    "FLOOR_ANGLE": ("The floors seen from different viewpoints do not line up perfectly.", "Small disagreements between the viewpoints can shift the length.", "Check that the cameras were not moved between recordings.", True),
    "CORNER_DIRECTION": ("A wall corner you marked comes out slightly off vertical in 3D.", "The corner position, and so the length, is less certain.", "Mark the corner edge again with the two points far apart and exactly on the edge.", True),
    "CORNER_OFF_FLOOR": ("A wall corner does not land exactly on the floor of the other viewpoints.", "The corner position, and so the length, is less certain.", "Mark the corner edges again, or choose different frames.", True),
    "LENGTH_NOT_HORIZONTAL": ("The two wall ends are at slightly different heights.", "The length may include a small vertical component.", "Check that the left and right corners are the two ends of the same wall.", True),
    "FRAME_SPREAD": ("The frame sets disagree with each other more than expected.", "Individual frame results differ, so the average is less certain.", "Choose different frames or record again.", True),
    "THREE_VIEW_SETS_EXCLUDED": ("Some frame sets were not used because their corners were not all marked.", "Fewer frame sets means less cross-checking.", "Mark the four wall corners in those frame sets to include them.", True),
}
assert not (set(THREE_VIEW_FAILURE_TEXT) & set(FAILURE_TEXT)) and not (set(THREE_VIEW_WARNING_TEXT) & set(WARNING_TEXT)), "three-viewpoint wording must not shadow an existing entry"
# The original tables above (seven advisory flags, the two-viewpoint failure codes) are NOT modified: ``failure_parts`` / ``warning_parts`` look a code up there first, then here.


def failure_parts(code) -> dict:
    """dict(code, known, what, next_step, text) for any code.  An unknown code gets the generic wording, never the technical message."""
    code = str(code)
    table = FAILURE_TEXT if code in FAILURE_TEXT else THREE_VIEW_FAILURE_TEXT
    known = code in table
    what, nxt = table[code] if known else GENERIC_FAILURE
    return dict(code=code, known=known, what=what, next_step=nxt, text=(what + (" " + nxt if nxt else "")))


def failure_text(code) -> str:
    return failure_parts(code)["text"]


def warning_parts(code) -> dict:
    """dict(code, known, meaning, reliability, action, visitor_can_act, text) for any advisory code (unknown codes get generic, cautious wording)."""
    code = str(code)
    table = WARNING_TEXT if code in WARNING_TEXT else THREE_VIEW_WARNING_TEXT
    known = code in table
    meaning, why, action, can = table[code] if known else GENERIC_WARNING
    return dict(code=code, known=known, meaning=meaning, reliability=why, action=action, visitor_can_act=bool(can), text=f"{meaning} {why} {action}")
