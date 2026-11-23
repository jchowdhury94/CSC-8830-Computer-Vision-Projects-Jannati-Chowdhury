"""F8.31D-2: visitor-facing wording, labels and icons for the integrated visitor app (DATA + pure functions: no Streamlit import, no mathematics).

Everything here is presentation.  Internal identifiers (``AL_corner`` ...), schema fields and backend codes are never changed; this module only maps them to words.
"""
from __future__ import annotations

# ------------------------------------------------------------------ icons (Material symbols; they take the text colour of the theme, unlike colour emoji)
ICON = {"ok": ":green[:material/check_circle:]", "warn": ":orange[:material/warning:]", "bad": ":red[:material/cancel:]", "todo": ":gray[:material/radio_button_unchecked:]",
        "wait": ":blue[:material/hourglass_top:]", "edit": ":orange[:material/edit:]", "check": ":blue[:material/search:]"}


def line(kind: str, text: str) -> str:
    return f"{ICON[kind]} {text}"


# ------------------------------------------------------------------ boundary labels (M2)
_VIEW = {"AL": (1, "left"), "PL": (1, "right"), "AR": (2, "left"), "PR": (2, "right")}      # = room_frames.VIEW_KEY inverted; kept here so this module has no imports
_LINE_KIND = {"corner": "Wall corner", "ceilF": "Ceiling boundary"}
LINE_HELP = {"corner": "Wall corner: the tall vertical edge where two walls meet.", "ceilF": "Ceiling boundary: the edge where the front wall meets the ceiling."}


# F8.31H: the instruction shown immediately above the image, chosen by the ACTIVE boundary (data only; orientation is explained, never enforced -- no snapping)
BOUNDARY_GUIDE = {
    "corner": dict(orientation="vertical", heading="Select the VERTICAL wall corner",
                   body="Click two points along the same wall corner: one near the top and one near the bottom.", note=None),
    "ceilF": dict(orientation="horizontal", heading="Select the HORIZONTAL ceiling boundary",
                  body="Click two points along the edge where the wall meets the ceiling: one toward the left and one toward the right.",
                  note="The ceiling edge may look slightly slanted because of perspective. Follow the actual visible boundary instead of forcing a perfectly horizontal line."),
}


def boundary_guide(name: str) -> dict:
    """'PL_ceilF' (or just 'ceilF') -> the guide of its type.  Raises KeyError for an unknown boundary type."""
    kind = name.split("_", 1)[1] if "_" in name else name
    return BOUNDARY_GUIDE[kind]


def boundary_parts(name: str) -> tuple:
    """'PL_ceilF' -> (viewpoint 1|2, 'left'|'right', 'ceilF')."""
    vk, kind = name.split("_", 1)
    vp, cam = _VIEW[vk]
    return vp, cam, kind


def boundary_label(name: str) -> str:
    """'PL_ceilF' -> 'Viewpoint 1 · Right camera · Ceiling boundary' (the internal identifier is not shown)."""
    vp, cam, kind = boundary_parts(name)
    return f"Viewpoint {vp} · {cam.title()} camera · {_LINE_KIND[kind]}"


def boundary_row_label(name: str) -> str:
    """The same, shorter, for a grid row."""
    vp, cam, kind = boundary_parts(name)
    return f"Viewpoint {vp} · {cam.title()} · {_LINE_KIND[kind]}"


# ------------------------------------------------------------------ calibration file kinds (L6)
KIND_TEXT = {"f8_job": "New calibration from your videos", "f8_recovered": "Saved result found with your recovery code", "f8_file": "Calibration result you saved earlier",
             "contract_file": "Calibration numbers you saved earlier", "phase1_file": "Calibration file from an earlier version"}

# ------------------------------------------------------------------ baseline provenance (M7)
BASELINE_TEXT = {"measured": "measured by you", "calibrated": "estimated from the calibration", "file": "stored in the calibration file", "entered": "entered by you"}


def baseline_phrase(source) -> str:
    return BASELINE_TEXT.get(source, "source not recorded")


def baseline_line(mm, source) -> str:
    """'Baseline 226 mm (estimated from the calibration)'.  Only 'measured' is called measured; an estimate is never labelled as a physical measurement."""
    return f"Baseline {mm:g} mm ({baseline_phrase(source)})"


# ------------------------------------------------------------------ retention (M4): derived from the real limits so the text cannot drift from the backend
def _minutes(seconds) -> str:
    m = max(1, round(float(seconds) / 60.0))
    return f"{m} minute" + ("" if m == 1 else "s")


def _hours_or_minutes(seconds) -> str:
    s = float(seconds)
    if s >= 5400:
        h = round(s / 3600.0)
        return f"{h} hour" + ("" if h == 1 else "s")
    return _minutes(s)


def retention_lines(limits) -> tuple:
    """What really happens to an uploaded video (``chunk_store.Limits``: idle_timeout_s, complete_retention_s, job_max_s).  No permanence and no instant deletion is claimed."""
    return (
        "Your videos are linked to **this browser session only**; nobody else using the app can see or use them.",
        f"They are stored **temporarily** on the server. A finished upload is removed after about **{_minutes(limits.complete_retention_s)}** without activity in this session, "
        f"and in any case after at most about {_hours_or_minutes(limits.job_max_s)}. An unfinished upload is dropped after about {_minutes(limits.idle_timeout_s)} without progress.",
        "If a video has expired you will be asked to upload it again; your other work is kept where it still applies.",
        "**Start over** and **Lock this session** delete your uploads straight away. Closing the page does not delete them at once: they expire as described above.",
    )


def retention_short(limits) -> str:
    return (f"Uploads are temporary and tied to this session: a finished upload expires after about {_minutes(limits.complete_retention_s)} without activity "
            f"(an unfinished one after about {_minutes(limits.idle_timeout_s)} without progress).")


# ------------------------------------------------------------------ upload requirements (Phase D.3)
# The containers ``video_check.sniff_container`` accepts (by file CONTENT, never by extension): ISO MP4/M4V, QuickTime MOV, Matroska/WebM, AVI.  A test pins this wording to that function.
ACCEPTED_FORMATS = "MP4 or M4V, MOV (QuickTime), Matroska/WebM, or AVI"


def upload_requirements(*, n_videos: int, viewpoints, per_file_mb: float, store_total_mb: float, min_duration_s: float, max_duration_s: float, min_fps: float, max_fps: float,
                        fps_tolerance: float, validated_wh) -> tuple:
    """The recording and upload requirements shown beside the room-video uploaders, as markdown lines.  Pure: every number is passed in from the object that ENFORCES it (the room store's
    limits and ``room_videos.F6_LIMITS``), so what is shown can never drift from what the server checks.

    ``viewpoints``: sequence of (number, role or None).  ``store_total_mb`` is the server's TOTAL temporary upload budget: it is shared by all visitors (``ChunkStore.reserved_bytes`` sums
    every upload), it is not a per-visitor allowance.  Nothing here is a new limit: these are the application's existing checks, and the last line says what is NOT yet verified."""
    count = {4: "Four", 6: "Six"}.get(int(n_videos), str(n_videos))
    groups = "; ".join(f"**V{n}**" + (f" ({role})" if role else "") for n, role in viewpoints)
    w, h = validated_wh
    fair = store_total_mb / n_videos
    lines = [
        "**Recording and upload requirements**",
        f"- **Videos:** {count} in total, grouped by viewpoint: {groups}. Each viewpoint needs one video from the **left** camera and one from the **right** camera.",
        f"- **File type:** {ACCEPTED_FORMATS}. The file's content is checked, not its name or extension.",
        f"- **Size:** each video up to **{per_file_mb:,.0f} MB**. The server keeps at most **{store_total_mb:,.0f} MB** of uploaded videos at any time, **shared by all visitors**; there is no separate "
        f"per-visitor limit. It is also **one pool for both walls**: the videos of Wall A and of Wall B are stored together and count against the same {store_total_mb:,.0f} MB; it is not allocated "
        f"separately to each wall. If nobody else has videos stored, {n_videos} videos averaging up to about {fair:,.0f} MB each fit on their own, but while Wall A's videos are still stored, Wall B's videos "
        "must fit in what is left; if the server's storage is full you are asked to try again later. One upload runs at a time.",
        f"- **Length:** {min_duration_s:g}\u2013{max_duration_s:g} seconds per video.",
        f"- **Frame rate:** {min_fps:g}\u2013{max_fps:g} frames per second. Use the same frame rate for the left and right video of a viewpoint (a difference of more than {fps_tolerance:g} fps is flagged).",
        f"- **Resolution:** every video must have **exactly the frame size and orientation of your accepted calibration**. The validated size is {w} \u00d7 {h} px (portrait). "
        "Videos are never resized or rotated: a different size or orientation is rejected.",
        f"- **Same settings:** all {n_videos} videos must have the same frame size, and every video must be a different file.",
        "- **Not yet verified on this server:** decoding of H.265/HEVC videos (they pass the file check, but decoding depends on the server's video library), and whether videos close to these limits can be "
        "processed within the server's time and memory limits. These are the application's upload checks, not a promise of processing time.",
    ]
    return tuple(lines)


# ------------------------------------------------------------------ progress (L5)
def progress_for(step: str, base: dict, *, measurement_complete: bool = False) -> tuple:
    """-> (fraction, label).  100 % is reached only once a measurement was actually accepted; reaching the measurement page is not completion."""
    frac, label = base[step]
    if step == "measure":
        return (1.0, "Measurement complete") if measurement_complete else (min(frac, 0.95), label)
    return min(frac, 0.95), label


# ------------------------------------------------------------------ failure recovery (H3): typed codes -> workflow steps, never the displayed text
STEP_LABEL = {"home": "Go to the calibration", "room": "Go to the room videos", "frames": "Go to frame selection and boundaries"}
# Codes whose advice involves NEW recordings.  The primary action is the room-video step; frame selection is offered when the advice also allows choosing other frames.
_RE_RECORD = {"FEATURE_MATCHING_FAILED": ("room",), "FRAME_SIZE_MISMATCH": ("room",), "CALIBRATION_VIDEO_SIZE_MISMATCH": ("room",), "JOB_TOO_LARGE": ("room",),
              "UPLOAD_EXPIRED": ("room",), "VIDEO_HASH_MISMATCH": ("room",),
              "POSE_JOINT_NOT_CONVERGED": ("frames", "room"), "NO_USABLE_FRAME_SETS": ("frames", "room"), "POSE_JOINT_SCALE_ABOVE_LIMIT": ("room", "home"),
              "INPUT_IMAGE_SIZE_MISMATCH": ("home", "frames"), "INPUT_INVALID_CALIBRATION": ("home", "frames"), "LINE_RETARGET_FAILED": ("home", "frames"),
              "BASELINE_CHANGED_UNLABELLED": ("home", "frames")}

_RE_RECORD.update({c: ("frames", "room") for c in ("THREE_VIEW_INSUFFICIENT_EVIDENCE", "REG_FEW_INLIERS", "REG_INLIER_FRACTION", "REG_RMS", "REG_COLLINEAR", "REG_PAIRS", "FLOOR_ANGLE")})      # three-viewpoint Wall B


def recovery_steps(code, primary) -> list:
    """The ordered workflow steps offered for a failure code.  ``primary`` is the existing typed routing (f8_result_view.step_for_code) used when the code has no explicit entry."""
    code = str(code)
    if code in _RE_RECORD:
        return list(_RE_RECORD[code])
    return [primary] if primary else []
