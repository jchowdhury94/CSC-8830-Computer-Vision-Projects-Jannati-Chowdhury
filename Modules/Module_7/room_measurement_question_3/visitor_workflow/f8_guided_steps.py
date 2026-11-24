"""F8.31D-3B: step state of the guided "Calibrate My Cameras" page (pure logic; no Streamlit, no file bytes, no backend rules).

Three of the six steps are visitor confirmations ("I've prepared my cameras", ...).  The other three are never confirmed by the visitor: they are DERIVED, on every rerun,
from the real backend facts handed in by the page:

    step 4  uploads_valid   both LEFT and RIGHT uploads are complete and verified by the upload store (``VisitorCalibrationService.slots_ready``)
    step 5  result_ready    a calibration job of this session has finished and its saved result can be reviewed (``review.usable``)
    step 6  accepted        the visitor accepted the reviewed calibration (``VisitorSession.accepted``)

so a completed status cannot outlive the input it stands for (an expired or removed upload locks step 5 again; nothing is cached here).
Clearing a confirmation also clears every later one (a later confirmation was made on top of it).  The object lives on ``VisitorSession``, so confirmations belong to
exactly one visitor session; a page refresh creates a new session and therefore starts at step 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

N_STEPS = 6
CONFIRM_STEPS = (1, 2, 3)
TITLES = {1: "Prepare Your Cameras", 2: "Prepare Your Checkerboard", 3: "Record Calibration Videos", 4: "Upload Your Videos", 5: "Review Settings & Start Calibration",
          6: "Review Calibration Results"}
CONFIRM_LABELS = {1: "I've prepared my cameras as instructed.", 2: "I've prepared and measured my checkerboard.", 3: "I've recorded both calibration videos."}
LOCK_REASONS = {2: "Locked. Confirm Step 1 to continue.", 3: "Locked. Confirm Step 2 to continue.", 4: "Locked. Confirm Step 3 to continue.",
                5: "Locked. Upload both videos in Step 4 and wait for them to be verified.", 6: "Locked. Available once your calibration has finished successfully."}
DEFAULT_SETTINGS = {"cols": 8, "rows": 6, "square": 20.0, "baseline": 0.0}


@dataclass
class Facts:
    """What the backend currently says (computed by the page each run)."""
    uploads_valid: bool = False
    result_ready: bool = False
    accepted: bool = False
    calibration_started: bool = False      # a job exists in this session (running or finished)


@dataclass
class GuidedSteps:
    confirmed: dict = field(default_factory=lambda: {1: False, 2: False, 3: False})
    room: "RoomSteps" = field(default_factory=lambda: RoomSteps())             # F8.31D-4: room page confirmations
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))      # step-5 inputs, kept so Back / Home do not lose them (validated by ``parse_setup`` at use)

    # ---- confirmations
    def set_confirmed(self, n: int, value: bool, *, frozen: bool = False) -> bool:
        """Set confirmation ``n``.  Unchecking clears every later confirmation.  ``frozen`` (a calibration job exists) refuses any change.  -> whether anything changed."""
        if n not in CONFIRM_STEPS:
            raise ValueError("only steps 1-3 are confirmed by the visitor")
        value = bool(value)
        if frozen or self.confirmed[n] == value:
            return False
        if value and not all(self.confirmed[k] for k in CONFIRM_STEPS if k < n):
            return False                                     # cannot confirm out of order (the control is not even shown while locked)
        self.confirmed[n] = value
        if not value:
            for k in CONFIRM_STEPS:
                if k > n:
                    self.confirmed[k] = False
        return True

    def set_setting(self, name: str, value):
        if name not in DEFAULT_SETTINGS:
            raise ValueError("unknown setting")
        self.settings[name] = value

    def reset(self):
        self.confirmed = {1: False, 2: False, 3: False}
        self.settings = dict(DEFAULT_SETTINGS)
        self.room.reset()

    # ---- derived state
    def unlocked(self, n: int, f: Facts) -> bool:
        if n == 1:
            return True
        if n in (2, 3):
            return self.confirmed[n - 1]
        if n == 4:
            return self.confirmed[3] or f.calibration_started
        if n == 5:
            return (f.uploads_valid and self.confirmed[3]) or f.result_ready
        if n == 6:
            return f.result_ready
        raise ValueError("steps are 1-6")

    def complete(self, n: int, f: Facts) -> bool:
        if n in CONFIRM_STEPS:
            return self.confirmed[n]
        if n == 4:
            return self.unlocked(4, f) and (f.uploads_valid or f.calibration_started)
        if n == 5:
            return f.result_ready
        if n == 6:
            return f.result_ready and f.accepted
        raise ValueError("steps are 1-6")

    def state(self, n: int, f: Facts) -> str:
        """'locked' | 'active' (unlocked, not done) | 'done'."""
        if not self.unlocked(n, f):
            return "locked"
        return "done" if self.complete(n, f) else "active"

    def current(self, f: Facts) -> int:
        """The first step that is not done (the one the visitor should work on); 6 when everything is done."""
        for n in range(1, N_STEPS + 1):
            if self.state(n, f) != "done":
                return n
        return N_STEPS

    def locked_reason(self, n: int):
        return LOCK_REASONS.get(n)


# ------------------------------------------------------------------ F8.31D-4: the three guided steps of the room-video page
ROOM_N_STEPS = 3
ROOM_CONFIRM_STEPS = (1, 2)
ROOM_TITLES = {1: "Prepare Your Room and Cameras", 2: "Record Two Stereo Viewpoints", 3: "Upload Your Four Videos"}
ROOM_CONFIRM_LABELS = {1: "I've prepared my room and cameras as instructed.", 2: "I've recorded both viewpoints (four videos in total)."}
ROOM_LOCK_REASONS = {2: "Locked. Confirm Step 1 to continue.", 3: "Locked. Confirm Step 2 to continue."}


@dataclass
class RoomSteps:
    """Confirmations of room steps 1-2 (visitor statements).  Step 3 has NO stored flag: it is complete only while ``videos_ready`` (the backend's CURRENT room report) says so,
    so an expired or replaced upload can never leave it 'done'.  Withdrawing a confirmation re-locks the later steps; uploads are not touched."""
    confirmed: dict = field(default_factory=lambda: {1: False, 2: False})

    def set_confirmed(self, n: int, value: bool) -> bool:
        if n not in ROOM_CONFIRM_STEPS:
            raise ValueError("only steps 1-2 are confirmed by the visitor")
        value = bool(value)
        if self.confirmed[n] == value or (value and not all(self.confirmed[k] for k in ROOM_CONFIRM_STEPS if k < n)):
            return False
        self.confirmed[n] = value
        if not value:
            for k in ROOM_CONFIRM_STEPS:
                if k > n:
                    self.confirmed[k] = False
        return True

    def reset(self):
        self.confirmed = {1: False, 2: False}

    def unlocked(self, n: int) -> bool:
        if n == 1:
            return True
        if n in (2, 3):
            return self.confirmed[n - 1]
        raise ValueError("steps are 1-3")

    def state(self, n: int, videos_ready: bool) -> str:
        """'locked' | 'active' | 'done'."""
        if not self.unlocked(n):
            return "locked"
        done = self.confirmed[n] if n in ROOM_CONFIRM_STEPS else bool(videos_ready)
        return "done" if done else "active"

    def can_continue(self, videos_ready: bool) -> bool:
        return bool(videos_ready) and self.unlocked(3)
