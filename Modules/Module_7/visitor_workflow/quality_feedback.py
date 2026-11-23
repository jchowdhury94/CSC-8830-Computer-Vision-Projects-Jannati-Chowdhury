"""F8.31F: visitor-friendly quality feedback built ONLY from diagnostics the system really produces (no Streamlit import, no new measurement, no new threshold).

Every item answers four questions: what went wrong, which camera / video / stage is affected, what the visitor can do, and whether recording again is needed.
States: ``failed`` | ``needs_improvement`` | ``completed`` | ``not_assessed``.

Where the diagnostics come from (nothing here measures anything itself):
* calibration  - the accepted calibration's quality level and concern/problem texts (``f7_calibration`` / ``f8_calibration_bridge``).  Concerns are recognised by the fixed
                 message templates the calibration code emits; an unrecognised concern is shown verbatim with a general suggestion.
* videos       - ``room_videos.assess_room_set`` errors and warnings (resolution / orientation / frame-rate / length / decode / duplicates).
* boundaries   - Stage D refinement codes of every (frame set, boundary) (``room_frames.line_report``).
* reconstruction - the typed failure and advisory codes of the backend (``visitor_messages``).

What is NOT assessed, because no detector for it exists in the system: image blur of calibration frames, motion of the rig during a recording, focus or exposure changes between videos.  This
module never claims to have checked them.  A "completed" state means the checks that exist did not flag a problem; it is NOT a statement about accuracy (no ground truth is available).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import visitor_messages as VM

FAILED, NEEDS, DONE, NOT_ASSESSED = "failed", "needs_improvement", "completed", "not_assessed"
STATE_LABEL = {FAILED: "Failed", NEEDS: "Needs improvement", DONE: "Completed", NOT_ASSESSED: "Not yet assessed"}
_ORDER = {FAILED: 3, NEEDS: 2, NOT_ASSESSED: 1, DONE: 0}
NO_ACCURACY_CLAIM = "Completed means the available checks found no problem. It does not mean the result is accurate: nothing here can be compared with the true dimensions."

# next-step kinds
RETRY, CORRECT, RECALIBRATE, RERECORD, REUPLOAD, NONE = "retry", "correct_boundary", "recalibrate", "record_again", "upload_again", "none"
ACTION_LABEL = {RETRY: "Try again", CORRECT: "Correct the boundary", RECALIBRATE: "Recalibrate", RERECORD: "Record again", REUPLOAD: "Upload again", NONE: "No action needed"}
RECORD_AGAIN = {RETRY: "No", CORRECT: "No", RECALIBRATE: "No (calibration only)", RERECORD: "Yes", REUPLOAD: "No", NONE: "No"}


@dataclass
class Item:
    state: str
    stage: str                  # calibration | videos | boundaries | reconstruction
    affected: str               # camera / video / viewpoint / processing stage in words
    what: str
    action: str                 # what the visitor can do
    kind: str = NONE            # one of the next-step kinds
    detail: str = ""            # technical code / original message (optional section only)

    @property
    def record_again(self) -> str:
        return RECORD_AGAIN.get(self.kind, "No")

    @property
    def state_label(self) -> str:
        return STATE_LABEL[self.state]

    def to_dict(self):
        return dict(state=self.state, state_label=self.state_label, stage=self.stage, affected=self.affected, what=self.what, action=self.action, kind=self.kind, record_again=self.record_again, detail=self.detail)


def overall(items, assessed: bool = True) -> str:
    """The worst state among the items; ``not_assessed`` when nothing was assessed yet; ``completed`` only when the checks ran and flagged nothing."""
    if not assessed:
        return NOT_ASSESSED
    return max((i.state for i in items), key=lambda s: _ORDER[s], default=DONE)


# ------------------------------------------------------------------ calibration
# (substring of a message template the calibration code emits) -> (what, action, kind).  Only conditions the calibration code really detects.
_CAL_RULES = (
    ("reprojection rms", "The calibration's reprojection error is high: the checkerboard corners do not fit the camera model well.", "Film a new calibration with a flat, rigid board, sharp footage and the board tilted in different positions.", RECALIBRATE),
    ("epipolar distance", "The two cameras' views do not agree well (high epipolar error).", "Film the calibration again with both cameras fixed and the board visible to both at the same time.", RECALIBRATE),
    ("did not appear in enough regions", "The checkerboard covered too little of the frame.", "Film the calibration again and move the board into every region of the frame, including the corners.", RECALIBRATE),
    ("tilted by", "Too few views show the checkerboard tilted.", "Film the calibration again with the board tilted toward and away from the cameras.", RECALIBRATE),
    ("same distance", "The checkerboard was always at about the same distance.", "Film the calibration again with the board at near and far distances.", RECALIBRATE),
    ("valid pairs were used", "Few usable checkerboard views were found.", "Film a longer calibration with more clear, well-lit views of the whole board.", RECALIBRATE),
    ("fits much worse", "At least one checkerboard view fits much worse than the others.", "Film the calibration again; keep the board still and fully visible in each view.", RECALIBRATE),
    ("board poses were discarded", "Many board views were discarded because the left and right corner matches could not be verified.", "Film the calibration again with the board clearly visible to both cameras at once.", RECALIBRATE),
    ("timing could not be verified", "The left and right calibration videos could not be matched in time.", "Film the calibration again, starting both cameras together and holding the board still for a moment.", RECALIBRATE),
    ("baseline you measured", "The distance you entered between the cameras differs from the calibrated one.", "Check the distance between the two lenses and enter it again. If it is right, recalibrate.", RECALIBRATE),
    ("baseline", "The distance between the cameras is small compared with the board distance.", "Keep the cameras the same distance apart for all recordings; a larger gap helps if your rig allows it.", NONE),
    ("principal point", "The calibration's image centre is unusual.", "Recalibrate with the same camera settings you use for the room videos.", RECALIBRATE),
    ("fx/fy", "The calibration's pixel shape is unusual.", "Recalibrate with the same camera settings you use for the room videos.", RECALIBRATE),
    ("capture mode", "The calibration may not match how the cameras were used.", "Record the room and the calibration in the same mode and resolution.", RECALIBRATE),
    ("resolution", "The calibration resolution differs from the recording.", "Record at the calibrated resolution or recalibrate at the new one.", RECALIBRATE),
    ("orientation", "The calibration orientation differs from the recording.", "Record in the calibrated orientation or recalibrate in the new one.", RECALIBRATE),
)


def calibration_feedback(level, concerns=(), problems=(), stereo_rms_px=None) -> list:
    """``level``: 'good' | 'questionable' | 'failed' | None (not assessed), as stored with the accepted calibration."""
    lv = str(level).lower() if level is not None else None
    if lv is None:
        return [Item(NOT_ASSESSED, "calibration", "Camera calibration", "The calibration has not been assessed yet.", "Complete the calibration step.", NONE)]
    items = []
    if lv == "failed":
        for p in (problems or ["The calibration failed its validity checks."]):
            items.append(Item(FAILED, "calibration", "Camera calibration (both cameras)", "The calibration failed its validity checks.", "Film a new calibration pair and calibrate again.", RECALIBRATE, detail=str(p)))
    for c in concerns or ():
        low = str(c).lower()
        for key, what, action, kind in _CAL_RULES:
            if key in low:
                items.append(Item(NEEDS, "calibration", "Camera calibration", what, action, kind, detail=str(c))); break
        else:
            items.append(Item(NEEDS, "calibration", "Camera calibration", "The calibration check flagged something.", "Read the detail below; if unsure, film the calibration again.", RECALIBRATE, detail=str(c)))
    if not items:
        items.append(Item(DONE, "calibration", "Camera calibration", "The calibration checks found no problem" + (f" (stereo reprojection error {float(stereo_rms_px):.2f} px)." if isinstance(stereo_rms_px, (int, float)) else "."), "", NONE))
    return items


# ------------------------------------------------------------------ videos
_VIDEO_RULES = (
    ("different frame sizes", "The left and right videos of one viewpoint have different frame sizes.", "Record both cameras with identical settings.", RERECORD),
    ("do not all have the same frame size", "The four videos do not share one frame size.", "Use the same two cameras and settings for all four recordings.", RERECORD),
    ("orientation", "The video orientation differs from the calibration.", "Record in the calibrated orientation, or recalibrate in the orientation you film in.", RERECORD),
    ("calibration is", "The video resolution differs from the calibration.", "Record at the calibrated resolution, or recalibrate at this resolution.", RERECORD),
    ("same file", "Two slots hold the same file.", "Upload a different recording in each of the four boxes.", REUPLOAD),
    ("could not be decoded", "The video could not be read.", "Re-export it (for example H.264 MP4) and upload it again.", REUPLOAD),
    ("duration", "The video is too short or too long.", "Trim or re-record the video.", RERECORD),
    ("frame count", "The video has too few frames.", "Record a longer clip.", RERECORD),
    ("frame rate", "The frame rates of the left and right videos differ.", "Record both cameras at the same frame rate.", RERECORD),
    ("unvalidated", "This resolution has not been tested with the measurement.", "You can continue, but the result at this size has not been verified.", NONE),
    ("limit for a room video", "The file is larger than the limit.", "Record a shorter clip or lower the video quality.", RERECORD),
    ("made from video", "The calibration was not made from video.", "Recalibrate from video for both cameras.", RECALIBRATE),
)


def _rule_for(msg, rules, default_what, default_action, default_kind):
    low = msg.lower()
    for key, what, action, kind in rules:
        if key in low:
            return what, action, kind
    return default_what, default_action, default_kind


def video_feedback(report) -> list:
    """``report``: ``room_videos.RoomSetReport``."""
    items = []
    for slot, r in report.slots.items():
        who = r.label
        if r.status == "missing":
            items.append(Item(NOT_ASSESSED, "videos", who, "This video has not been uploaded yet.", "Upload it.", NONE)); continue
        if r.status == "pending":
            items.append(Item(NOT_ASSESSED, "videos", who, "This video is still being checked.", "Wait for the check to finish.", NONE)); continue
        for m in r.errors:
            what, act, kind = _rule_for(m, _VIDEO_RULES, "This video cannot be used.", "Read the detail and upload a corrected video.", REUPLOAD)
            items.append(Item(FAILED, "videos", who, what, act, kind, detail=m))
        for m in r.warnings:
            what, act, kind = _rule_for(m, _VIDEO_RULES, "This video was flagged.", "Read the detail.", NONE)
            items.append(Item(NEEDS, "videos", who, what, act, kind, detail=m))
    word = "six" if len(report.slots) == 6 else "four"                  # a three-viewpoint wall has six videos
    for m in report.set_errors:
        what, act, kind = _rule_for(m, _VIDEO_RULES, f"The {word} videos do not fit together.", "Read the detail.", RERECORD)
        items.append(Item(FAILED, "videos", f"All {word} videos", what, act, kind, detail=m))
    for m in report.set_warnings:
        what, act, kind = _rule_for(m, _VIDEO_RULES, f"The {word} videos were flagged.", "Read the detail.", NONE)
        items.append(Item(NEEDS, "videos", f"All {word} videos", what, act, kind, detail=m))
    if not items:
        items.append(Item(DONE, "videos", f"All {word} videos", "The video checks found no problem.", "", NONE))
    return items


# ------------------------------------------------------------------ boundaries (Stage D refinement codes; wording from annotation_ui.core.EXPLANATIONS)
_VIEWPOINT = {"AL": "Viewpoint 1 · left camera", "PL": "Viewpoint 1 · right camera", "AR": "Viewpoint 2 · left camera", "PR": "Viewpoint 2 · right camera"}
_KIND = {"corner": "wall corner", "ceilF": "ceiling edge"}


def boundary_place(line: str) -> str:
    return f"{_VIEWPOINT[line[:2]]} · {_KIND[line.split('_')[1]]}"


def boundary_feedback(wf) -> list:
    """From the refinement of every (frame set, boundary).  ``wf``: a ``RoomFrameWorkflow``.  Not assessed until the boundaries were applied."""
    import room_frames as RF
    from annotation_ui import core as f1core
    if not (wf.seeds_complete() and wf.seeds_applied()):
        return [Item(NOT_ASSESSED, "boundaries", "Wall boundaries", "The boundaries have not been refined yet.", "Confirm the six boundaries.", NONE)]
    items = []
    for n in RF.REQUIRED_LINES:
        failed, warned, issues = [], [], {}
        for i in range(wf.n_sets):
            r = wf.line_report(i, n)
            if r["status"] == "failed":
                failed.append(i + 1)
            elif r["status"] == "ok_warning":
                warned.append(i + 1)
            for m in r["issues"]:
                issues.setdefault(m["code"], m["message"])
        if failed:
            codes = sorted(issues)
            what = " ".join(dict.fromkeys(f1core.explain(c, issues[c]) for c in codes)) or "The edge could not be found."
            items.append(Item(FAILED, "boundaries", boundary_place(n), f"This boundary could not be refined in frame set(s) {', '.join(map(str, failed))}. {what}",
                              "Go back and correct this boundary with two clicks far apart on the visible edge. If the camera moved during this recording, record this viewpoint again with the rig held still.", CORRECT, detail=", ".join(codes)))
        elif warned:
            codes = sorted(issues)
            what = " ".join(dict.fromkeys(f1core.explain(c, issues[c]) for c in codes)) or "The refinement raised a warning."
            items.append(Item(NEEDS, "boundaries", boundary_place(n), f"The edge was found, with warnings in frame set(s) {', '.join(map(str, warned))}. {what}",
                              "Check this boundary and correct it if the line is not on the edge.", CORRECT, detail=", ".join(codes)))
    if not items:
        items.append(Item(DONE, "boundaries", "Wall boundaries", f"All six boundaries were refined in all {wf.n_sets} frame sets.", "", NONE))
    return items


# ------------------------------------------------------------------ reconstruction (typed codes of the backend)
# code -> (stage in words, next-step kind).  The what/next wording is the existing visitor_messages text; nothing is re-worded here.
_STAGE = {
    "FEATURE_MATCHING_FAILED": ("Matching the two cameras", RERECORD), "INPUT_MALFORMED_FRAME_SET": ("Matching the two cameras", RERECORD),
    "POSE_DEGENERATE_UNRECOVERABLE": ("Camera positions", RERECORD), "POSE_JOINT_NOT_CONVERGED": ("Camera positions", RERECORD), "POSE_JOINT_SCALE_ABOVE_LIMIT": ("Camera positions", RECALIBRATE),
    "DATASET_BUILD_FAILED": ("Camera positions", RETRY), "POSE_SOLVER_EXCEPTION": ("Camera positions", RETRY), "POSE_NON_FINITE": ("Camera positions", RETRY), "TRIANGULATION_FAILED": ("Camera positions", RETRY),
    "FLOOR_PLANE_FAILED": ("Floor", RERECORD), "REGISTRATION_FAILED": ("Joining the two viewpoints", RERECORD),
    "WALL_RECONSTRUCTION_FAILED": ("Wall edges", CORRECT), "GEOMETRY_FAILED": ("Wall edges", CORRECT), "NO_USABLE_FRAME_SETS": ("Wall edges", RERECORD),
    "WIDTH_UNAVAILABLE": ("Wall edges", CORRECT), "HEIGHT_UNAVAILABLE": ("Wall edges", CORRECT),
    "ANNOTATION_MISSING": ("Wall boundaries", CORRECT), "ANNOTATION_STALE": ("Wall boundaries", CORRECT), "ANNOTATION_INVALID": ("Wall boundaries", CORRECT), "TOO_FEW_USABLE_SETS": ("Wall boundaries", CORRECT),
    "CALIBRATION_CHANGED": ("Calibration", CORRECT), "CALIBRATION_INVALID": ("Calibration", RECALIBRATE), "CALIBRATION_NOT_ACCEPTED": ("Calibration", RECALIBRATE), "CALIBRATION_VIDEO_SIZE_MISMATCH": ("Videos and calibration", RERECORD),
    "UPLOAD_EXPIRED": ("Uploaded videos", REUPLOAD), "VIDEO_REPLACED": ("Uploaded videos", CORRECT), "VIDEO_HASH_MISMATCH": ("Uploaded videos", REUPLOAD),
    "FRAME_SIZE_MISMATCH": ("Videos and calibration", RERECORD),
}


def failure_item(code) -> Item:
    p = VM.failure_parts(code)
    stage, kind = _STAGE.get(str(code), ("Processing", RETRY))
    if str(code).startswith("WORKER_") or str(code) in ("INSUFFICIENT_MEMORY", "JOB_TOO_LARGE", "WORKSPACE_ERROR", "INTERNAL_ERROR"):
        stage, kind = ("The server", RETRY if str(code) != "JOB_TOO_LARGE" else RERECORD)
    return Item(FAILED, "reconstruction", stage, p["what"], p["next_step"] or "No action is possible from here.", kind, detail=str(code))


def warning_item(code) -> Item:
    p = VM.warning_parts(code)
    return Item(NEEDS, "reconstruction", "Measurement checks", p["meaning"] + " " + p["reliability"], p["action"], RERECORD if p["visitor_can_act"] else NONE, detail=str(code))


def reconstruction_feedback(failures=(), warnings=(), completed: bool = False, assessed: bool = True) -> list:
    """``failures`` / ``warnings``: backend codes.  ``completed``: a measurement ran and produced a result."""
    if not assessed:
        return [Item(NOT_ASSESSED, "reconstruction", "Measurement", "The measurement has not been run.", "", NONE)]
    items = [failure_item(c) for c in failures] + [warning_item(c) for c in warnings]
    if not items and completed:
        items.append(Item(DONE, "reconstruction", "Measurement", "The measurement completed and the checks flagged nothing. " + NO_ACCURACY_CLAIM, "", NONE))
    return items


# ------------------------------------------------------------------ F8.31G: grouped boundary-failure summary (presentation only; every original code and message stays in the workflow)
# cause key -> (visitor guidance by boundary kind).  Only the failure codes Stage D really emits are mapped; anything else falls under "not_located".  No cause is claimed that
# no diagnostic supports: camera motion, blur and lighting are never named as a detected cause.
CAUSE_ORDER = ("direction", "points", "border", "stale", "not_located")
_CODE_CAUSE = {"UNREASONABLE_ORIENTATION": "direction", "LINE_TOO_SHORT": "points", "DUPLICATE_ENDPOINTS": "points", "WINDOW": "border", "SEARCH_WINDOW_OUT_OF_IMAGE": "border",
               "FRAME_FINGERPRINT_MISMATCH": "stale", "FRAME_MISMATCH": "stale", "NOT_RUN": "stale"}
_EDGE_WORDS = {"corner": ("wall corner", "the vertical line where the two walls meet"), "ceilF": ("ceiling boundary", "the horizontal line where the wall meets the ceiling")}
GUIDANCE = {
    "direction": lambda k: f"The selected line does not follow the expected wall edge. For a {_EDGE_WORDS[k][0]}, select two points along {_EDGE_WORDS[k][1]}.",
    "points": lambda k: "The two points are too close together or at the same place. Select two different points, farther apart along the visible boundary.",
    "border": lambda k: "Part of the search area lies outside the image. Place both points on the visible boundary, away from the image border.",
    "stale": lambda k: "This boundary could not be processed with the points it had. Mark it again with two points.",
    "not_located": lambda k: "The algorithm could not reliably locate this edge. Try placing two points directly on the visible boundary, farther apart.",
}
FRAME_SET_GUIDANCE = ("Some frame sets could not be processed. The original Method J workflow requires all five frame sets for boundary fitting and pose estimation.")
RIG_HINT = "If the camera rig may have moved during a recording, keep it stationary when you record that viewpoint again."


def cause_of(code: str) -> str:
    return _CODE_CAUSE.get(str(code), "not_located")


def boundary_failures(wf) -> list:
    """One dict per boundary with at least one failed frame set: ``line``, ``kind`` (corner | ceilF), ``place`` words, ``failed_sets`` (1-based), ``codes`` (sorted, unique),
    ``causes`` (ordered, unique), ``messages`` ({(code, message): [sets]}: the raw diagnostics grouped for the Technical Details section).  Empty until the boundaries were applied."""
    import room_frames as RF
    if not (wf.seeds_complete() and wf.seeds_applied()):
        return []
    out = []
    for n in RF.REQUIRED_LINES:
        failed, msgs = [], {}
        for i in range(wf.n_sets):
            r = wf.line_report(i, n)
            if r["status"] == "failed":
                failed.append(i + 1)
                for m in r["issues"] or [dict(code="REFINEMENT_FAILED", message="The refinement produced no usable line.")]:
                    msgs.setdefault((m["code"], m["message"]), []).append(i + 1)
        if failed:
            codes = sorted({c for c, _ in msgs}); causes = [c for c in CAUSE_ORDER if any(cause_of(x) == c for x in codes)]
            out.append(dict(line=n, kind=n.split("_")[1], place=boundary_place(n), failed_sets=failed, codes=codes, causes=causes, messages=msgs))
    return out


_NUM = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def failure_headline(fails) -> str:
    """'Two ceiling boundaries could not be refined. Please correct the highlighted boundaries before continuing.'  Counts by kind, so nothing is hidden."""
    parts = []
    for kind, one, many in (("corner", "wall corner", "wall corners"), ("ceilF", "ceiling boundary", "ceiling boundaries")):
        k = sum(1 for f in fails if f["kind"] == kind)
        if k:
            parts.append((k, one if k == 1 else many))
    if not parts:
        return ""
    if len(parts) == 1:
        k, w = parts[0]; subject = f"{_NUM[k]} {w}"
    else:
        total = sum(k for k, _ in parts); subject = f"{_NUM[total]} boundaries ({' and '.join(f'{k} {w}' for k, w in parts)})"
    return f"{subject} could not be refined. Please correct the highlighted boundar{'y' if len(fails) == 1 else 'ies'} before continuing."


def failure_guidance(fails) -> list:
    """Distinct guidance lines, one per distinct cause (and kind, for the direction text), in a fixed order.  No line is repeated per frame set or per boundary."""
    seen, out = set(), []
    for cause in CAUSE_ORDER:
        for f in fails:
            if cause in f["causes"]:
                text = GUIDANCE[cause](f["kind"])
                if text not in seen:
                    seen.add(text); out.append(text)
    return out


def boundary_warnings(wf) -> list:
    """Boundaries that were refined WITH warnings (no failed set): ``line``, ``place``, ``warned_sets``, ``codes``, ``messages`` (grouped like ``boundary_failures``)."""
    import room_frames as RF
    if not (wf.seeds_complete() and wf.seeds_applied()):
        return []
    out = []
    for n in RF.REQUIRED_LINES:
        warned, failed, msgs = [], False, {}
        for i in range(wf.n_sets):
            r = wf.line_report(i, n)
            failed = failed or r["status"] == "failed"
            if r["status"] == "ok_warning":
                warned.append(i + 1)
                for m in r["issues"]:
                    msgs.setdefault((m["code"], m["message"]), []).append(i + 1)
        if warned and not failed:
            out.append(dict(line=n, kind=n.split("_")[1], place=boundary_place(n), warned_sets=warned, codes=sorted({c for c, _ in msgs}), messages=msgs))
    return out
