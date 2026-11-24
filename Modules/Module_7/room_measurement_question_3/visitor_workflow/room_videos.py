"""Stage F8.31A: the four room videos -- slots, limits and metadata validation (NO Streamlit import, no reconstruction).

Four recordings are required, from the SAME two cameras in the SAME physical arrangement used for calibration:

    Viewpoint 1 (left side of the wall):   left camera + right camera
    Viewpoint 2 (right side of the wall):  left camera + right camera

The camera of a recording is decided by the upload SLOT, never by a file name.  This module only compares METADATA that the sandboxed inspector
(``video_check.inspect_video``: signature, MP4/MOV header, decoder in a separate process) already produced.  It never decodes, resizes, crops,
rotates or resamples a video, and it changes no acceptance rule of Method J:

* Duration / frame-count / fps / pixel ranges are the EXISTING F6 Stage B limits (``visitor_video_validation.DEFAULT_VIDEO_LIMITS``), imported, not invented.
* Calibration compatibility follows the existing F6 order (orientation, then resolution, then capture-mode provenance) on the ACCEPTED calibration.
* The per-file size limit is ``room_file_limit_bytes()`` (default 1024 MB), shared by the uploader and, through ``room_video_limits()``, by the F6 validation of the next phase.
* Everything else (unequal left/right length, an unvalidated resolution) is a RECOMMENDATION and never blocks.

Hard errors block ``Continue to Frame Selection``; warnings never do.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(MODULE.parent), str(HERE), str(MODULE / "f8_chunk_poc"), str(MODULE.parents[2])):   # + stereo_measure (Module 7 root) and the repository root
    if _p not in sys.path:
        sys.path.insert(0, _p)

from room_measurement import visitor_video_validation as VV       # noqa: E402  (the existing F6 limits; imported, never copied)
from chunk_store import Limits, MiB, env_limits                   # noqa: E402

# slot id -> (viewpoint, camera, label, F6 video key)
ROOM_UPLOAD_SLOTS = ("p1_left", "p1_right", "p2_left", "p2_right")
SLOT_INFO = {
    "p1_left": (1, "left", "Viewpoint 1 · Left camera video", "placement_1_left"),
    "p1_right": (1, "right", "Viewpoint 1 · Right camera video", "placement_1_right"),
    "p2_left": (2, "left", "Viewpoint 2 · Left camera video", "placement_2_left"),
    "p2_right": (2, "right", "Viewpoint 2 · Right camera video", "placement_2_right"),
}
# Three-viewpoint Wall B (additive: the two-viewpoint tuple above is unchanged and stays the default).  Viewpoint 3 only exists on a wall that uses three viewpoints.
ROOM_UPLOAD_SLOTS_3 = ("p1_left", "p1_right", "p2_left", "p2_right", "p3_left", "p3_right")
SLOT_INFO.update({
    "p3_left": (3, "left", "Viewpoint 3 · Left camera video", "placement_3_left"),
    "p3_right": (3, "right", "Viewpoint 3 · Right camera video", "placement_3_right"),
})
ALL_ROOM_SLOTS = ROOM_UPLOAD_SLOTS_3                       # every slot the room-video store must know (a wall uses the first four, or all six)
COUNT_WORDS = {4: "four", 6: "six"}


def slots_for(n_viewpoints: int) -> tuple:
    """The upload slots of a wall that uses 2 or 3 stereo viewpoints (4 or 6 videos)."""
    if n_viewpoints == 2:
        return ROOM_UPLOAD_SLOTS
    if n_viewpoints == 3:
        return ROOM_UPLOAD_SLOTS_3
    raise ValueError("A wall uses 2 or 3 stereo viewpoints.")


VIEWPOINT_HEADINGS = {1: "Viewpoint 1", 2: "Viewpoint 2", 3: "Viewpoint 3"}          # step-3 headings only; the upload labels carry the left/right camera identity
VIEWPOINT_TITLES = {1: "Viewpoint 1 — LEFT side of the wall", 2: "Viewpoint 2 — RIGHT side of the wall"}
VIEWPOINT_TITLES_3 = {1: "Viewpoint 1 — LEFT end of the wall", 2: "Viewpoint 2 — MIDDLE of the wall", 3: "Viewpoint 3 — RIGHT end of the wall"}
# A LABEL only: this frame size was used by the original experiment.  It never gates anything; every other size is shown as "unvalidated".
VALIDATED_FRAME_SIZES = ((2160, 3840),)
F6_LIMITS = VV.DEFAULT_VIDEO_LIMITS
DURATION_RECOMMENDATION_S = 1.0          # retained for compatibility; no longer used: different left/right durations are not presented as a problem
FPS_MATCH_TOLERANCE = F6_LIMITS.max_fps_difference

OK, WARN, ERROR, MISSING, PENDING = "ok", "warning", "error", "missing", "pending"


def f6_key(slot: str) -> str:
    return SLOT_INFO[slot][3]


DEFAULT_ROOM_MAX_FILE_MB = 1024          # per room video; the F8 uploader's long-standing default
ABSOLUTE_MAX_ROOM_FILE_MB = 2048         # hard ceiling for the F8_ROOM_MAX_FILE_MB override, so a misconfiguration cannot make storage unbounded (4 files = 8 GB at most)


def room_file_limit_bytes(base: Limits | None = None) -> int:
    """THE per-file size limit for a room video: one number used by the uploader (``room_limits``), by this module's validation and by the F6 limits that the
    next phase must hand to the measurement job (``room_video_limits``).  ``F8_ROOM_MAX_FILE_MB`` overrides it; without a (valid) override it is the uploader's own
    per-file cap (``F8_MAX_FILE_MB``, default 1024 MB).  The result is clamped to 1 .. ``ABSOLUTE_MAX_ROOM_FILE_MB``.  ``base`` is only for tests that build a custom ``Limits``."""
    try:
        mb = float(os.environ["F8_ROOM_MAX_FILE_MB"])
        if not mb == mb or mb <= 0:
            raise ValueError
    except (KeyError, ValueError):
        mb = (base or env_limits()).max_file_bytes / MiB
    return int(min(max(mb, 1.0), ABSOLUTE_MAX_ROOM_FILE_MB) * MiB)


def room_video_limits() -> "VV.VideoLimits":
    """The F6 Stage B limits to use for room videos: the existing ``DEFAULT_VIDEO_LIMITS`` with ONLY the file-size ceiling replaced by ``room_file_limit_bytes()``.
    F6's 200 MB figure is an upload-policy value (its source comment: "matches Streamlit's default upload cap"), not a decoder limit: videos are decoded as a stream,
    never loaded or copied whole, and processing cost is bounded by the unchanged duration, frame-size and frame-memory limits.  The F6 default itself is not modified.
    F8.31B must pass this object as ``JobLimits(video=...)`` so that a file accepted here is never rejected for its size by the next stage."""
    return replace(VV.DEFAULT_VIDEO_LIMITS, max_file_bytes=room_file_limit_bytes())


def room_limits(base: Limits | None = None) -> Limits:
    """Bounded storage for the room videos.  Same retention / idle / disk-margin policy as the calibration uploads (``env_limits``); only what must differ is set:
    no sibling reservation (the four sizes are independent) and a disk budget for four files (the real free-disk margin is still enforced when each upload begins)."""
    base = base or env_limits()
    per_file = room_file_limit_bytes(base)
    return replace(base, max_file_bytes=per_file, max_disk_bytes=4 * per_file, reserve_sibling=False)


# ------------------------------------------------------------------ reports
@dataclass
class SlotReport:
    slot: str
    status: str = MISSING                          # ok | warning | error | missing | pending
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    facts: dict = field(default_factory=dict)      # name, size_bytes, width, height, fps, frame_count, duration_s, codec, sha256

    @property
    def label(self) -> str:
        return SLOT_INFO[self.slot][2]


@dataclass
class RoomSetReport:
    slots: dict
    set_errors: list = field(default_factory=list)
    set_warnings: list = field(default_factory=list)
    calibration_size_wh: tuple | None = None
    unvalidated_resolution: bool = False

    @property
    def ready(self) -> bool:
        return not self.set_errors and all(r.status in (OK, WARN) for r in self.slots.values())

    @property
    def missing(self) -> list:
        return [SLOT_INFO[s][2] for s, r in self.slots.items() if r.status == MISSING]

    @property
    def pending(self) -> list:
        return [SLOT_INFO[s][2] for s, r in self.slots.items() if r.status == PENDING]

    @property
    def n_complete(self) -> int:
        return sum(1 for r in self.slots.values() if r.status in (OK, WARN, ERROR))

    @property
    def n_valid(self) -> int:
        """F8.31D-2: videos that passed the checks (a rejected upload is NOT counted).  Display only: readiness is still ``ready``."""
        return sum(1 for r in self.slots.values() if r.status in (OK, WARN))

    @property
    def n_rejected(self) -> int:
        return sum(1 for r in self.slots.values() if r.status == ERROR)

    @property
    def all_errors(self) -> list:
        return [f"{r.label}: {m}" for r in self.slots.values() for m in r.errors] + list(self.set_errors)

    @property
    def all_warnings(self) -> list:
        return [f"{r.label}: {m}" for r in self.slots.values() for m in r.warnings] + list(self.set_warnings)


def orientation_of(w, h) -> str:
    return "portrait" if h > w else "landscape" if w > h else "square"


# ------------------------------------------------------------------ admission for the upload inspector (Phase D.3)
class InspectionBusy(Exception):
    """The inspector could not be started right now because the server is busy.  NOT a verdict on the video: the caller must not cache it and should check again later."""


def inspect_with_admission(path, inspect, wait_s: float = 3.0):
    """Run ``inspect(path)`` (the sandboxed decoder, ``video_check.inspect_video``) under a 'validation' slot of the container-wide resource budget, so it never runs next to a calibration
    or reconstruction and never in unlimited numbers.  Raises ``InspectionBusy`` when refused; the slot is released on success, failure and cancellation alike."""
    import resource_budget as RB
    try:
        with RB.acquire_light("validation", "room-upload-check", wait_s=wait_s):
            return inspect(path)
    except RB.ResourceBusy as busy:
        raise InspectionBusy(busy.message) from None


# ------------------------------------------------------------------ one file
def _check_one(rep: SlotReport, info: dict, size_bytes: int):
    """``info`` is the inspector's result for this upload (never raises for bad input)."""
    f = rep.facts
    f.update(width=info.get("width"), height=info.get("height"), fps=info.get("fps"), frame_count=info.get("frame_count"), duration_s=info.get("duration_s"), codec=info.get("codec"))
    for e in info.get("errors") or []:
        rep.errors.append(e["message"] if isinstance(e, dict) else str(e))
    if rep.errors:
        return
    if not info.get("decodable"):
        rep.errors.append("The video could not be decoded. It may be corrupt, or use a codec this server cannot read. Re-export it (for example H.264 MP4) and upload again.")
        return
    w, h, fps, n, dur = f["width"], f["height"], f["fps"], f["frame_count"], f["duration_s"]
    if not (w and h):
        rep.errors.append("The frame size could not be read."); return
    if min(w, h) < F6_LIMITS.min_side_px or max(w, h) > F6_LIMITS.max_side_px:
        rep.errors.append(f"Frame size {w} × {h} px is outside the supported range ({F6_LIMITS.min_side_px}–{F6_LIMITS.max_side_px} px per side).")
    if not fps or not F6_LIMITS.min_fps <= fps <= F6_LIMITS.max_fps:
        rep.errors.append(f"Frame rate {fps if fps else 'unknown'} is not usable (it must be {F6_LIMITS.min_fps:g}–{F6_LIMITS.max_fps:g} frames per second).")
    if not n or n < F6_LIMITS.min_frames or n > F6_LIMITS.max_total_frames:
        rep.errors.append(f"Frame count {n if n else 'unknown'} is not usable (at least {F6_LIMITS.min_frames} frames are needed).")
    elif fps and dur is not None and not F6_LIMITS.min_duration_s <= dur <= F6_LIMITS.max_duration_s:
        rep.errors.append(f"Duration {dur:.1f} s is outside the supported {F6_LIMITS.min_duration_s:g}–{F6_LIMITS.max_duration_s:g} s. Trim or re-record the video.")
    lim_bytes = room_file_limit_bytes()                # the uploader already refuses larger files; this is defence in depth and keeps the next stage's limit identical
    if size_bytes > lim_bytes:
        rep.errors.append(f"The file is {size_bytes / MiB:.0f} MB; the limit for a room video is {lim_bytes / MiB:.0f} MB. Record a shorter clip or lower the video quality.")


# ------------------------------------------------------------------ the set
def assess_room_set(uploads: dict, calibration: dict | None, slots: tuple = ROOM_UPLOAD_SLOTS) -> RoomSetReport:
    """Validate the wall's slots together (the four two-viewpoint slots by default; ``slots=ROOM_UPLOAD_SLOTS_3`` for the six of a three-viewpoint wall).

    ``uploads``: slot -> dict(name, size_bytes, sha256, state ('complete'|'uploading'), inspection (the ``inspect_video`` result or None)).  Missing slots are absent.
    ``calibration``: the ACCEPTED calibration summary dict(image_size_wh=[w, h], capture_mode={left, right}) or None.  Without it nothing can be ready.
    """
    slot_ids = tuple(slots); count = COUNT_WORDS[len(slot_ids)]; viewpoints = sorted({SLOT_INFO[s][0] for s in slot_ids})
    slots = {s: SlotReport(s) for s in slot_ids}
    rep = RoomSetReport(slots)
    if not calibration or not calibration.get("image_size_wh"):
        rep.set_errors.append("No accepted calibration is available. Go back and accept a calibration before uploading room videos.")
    else:
        rep.calibration_size_wh = (int(calibration["image_size_wh"][0]), int(calibration["image_size_wh"][1]))
    for s, r in slots.items():
        u = uploads.get(s)
        if not u:
            continue
        r.facts.update(name=u.get("name"), size_bytes=u.get("size_bytes"), sha256=u.get("sha256"))
        if u.get("state") != "complete":
            r.status = PENDING; continue
        if u.get("inspection") is None:
            r.status = PENDING; continue
        r.status = OK                                  # provisional; the final verdict is set after the cross-file checks
        _check_one(r, u["inspection"], int(u.get("size_bytes") or 0))
    # ---- duplicates (identical bytes)
    by_hash = {}
    for s, r in slots.items():
        if r.facts.get("sha256") and r.status != MISSING:
            by_hash.setdefault(r.facts["sha256"], []).append(s)
    for group in by_hash.values():
        if len(group) > 1:
            for s in group:
                others = ", ".join(SLOT_INFO[o][2] for o in group if o != s)
                slots[s].errors.append(f"This is the same file as “{others}”. Each of the {count} recordings must be different.")
    # ---- left/right of one viewpoint, and all four together (only files that decoded)
    good = {s: r for s, r in slots.items() if r.status not in (MISSING, PENDING) and not r.errors}
    for vp in viewpoints:
        a, b = good.get(f"p{vp}_left"), good.get(f"p{vp}_right")
        if not (a and b):
            continue
        fa, fb = a.facts, b.facts
        if (fa["width"], fa["height"]) != (fb["width"], fb["height"]):
            msg = (f"The left and right videos of viewpoint {vp} have different frame sizes ({fa['width']} × {fa['height']} and {fb['width']} × {fb['height']}). "
                   "Both cameras must record with the same settings. Nothing is resized.")
            rep.set_errors.append(msg); a.errors.append("Frame size differs from the right camera video."); b.errors.append("Frame size differs from the left camera video.")
        if abs(fa["fps"] - fb["fps"]) > FPS_MATCH_TOLERANCE:
            rep.set_warnings.append(f"Viewpoint {vp}: the left and right videos have different frame rates ({fa['fps']:g} and {fb['fps']:g} fps). "
                                    "Left/right frames are paired by time later; mismatched rates make pairing less reliable. Recommended: record both cameras with the same frame rate.")
        # Different recording lengths are NOT a problem (the original experiment processed such recordings); only real compatibility checks warn.
    sizes = {(r.facts["width"], r.facts["height"]) for r in good.values()}
    if len(sizes) > 1 and not any("different frame sizes" in m for m in rep.set_errors):
        rep.set_errors.append(f"The {count} videos do not all have the same frame size (" + ", ".join(f"{w} × {h}" for w, h in sorted(sizes)) + f"). The same two cameras and settings must be used for all {count}. Nothing is resized.")
    # ---- calibration compatibility (existing F6 order: orientation, resolution, capture-mode provenance)
    if rep.calibration_size_wh:
        cw, ch = rep.calibration_size_wh
        for s, r in good.items():
            w, h = r.facts["width"], r.facts["height"]
            if orientation_of(w, h) != orientation_of(cw, ch):
                r.errors.append(f"The calibration is {orientation_of(cw, ch)} ({cw} × {ch}) but this video is {orientation_of(w, h)} ({w} × {h}). "
                                "Record in the calibrated orientation, or recalibrate in the orientation you film in. Nothing is rotated.")
            elif (w, h) != (cw, ch):
                r.errors.append(f"The calibration is {cw} × {ch} but this video is {w} × {h}. Camera intrinsics are not rescaled: record at the calibrated resolution, "
                                "or recalibrate at this resolution.")
        modes = (calibration.get("capture_mode") or {}) if calibration else {}
        if good and not (modes.get("left") == "video" and modes.get("right") == "video"):
            rep.set_errors.append("The accepted calibration is not marked as made from VIDEO for both cameras (" + f"left={modes.get('left')!r}, right={modes.get('right')!r}" +
                                  "). The measurement backend only accepts video-mode calibrations for room videos, because photo intrinsics can differ from video. "
                                  "Recalibrate from video, or import a calibration that records its capture mode.")
        if any((r.facts["width"], r.facts["height"]) == (cw, ch) for r in good.values()) and (cw, ch) not in VALIDATED_FRAME_SIZES:
            rep.unvalidated_resolution = True
            rep.set_warnings.append(f"Frame size {cw} × {ch} is UNVALIDATED: the measurement has only been tested on {VALIDATED_FRAME_SIZES[0][0]} × {VALIDATED_FRAME_SIZES[0][1]} recordings. "
                                    "It is accepted, but the results at this resolution have not been verified.")
    # ---- final status per slot
    for r in slots.values():
        if r.status in (MISSING, PENDING):
            continue
        r.status = ERROR if r.errors else (WARN if r.warnings else OK)
    return rep
