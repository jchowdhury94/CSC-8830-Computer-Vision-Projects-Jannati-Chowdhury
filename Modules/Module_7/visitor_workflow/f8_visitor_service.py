"""Stage F8.18: LOCAL visitor calibration integration -- service layer (NO Streamlit import).

Connects, without duplicating any calibration mathematics:
    F8  ChunkStore / CalibrationJobRunner / ResultStore (upload, worker process, recovery code, atomic result)
 -> F8.15 ``parse_f8_result`` / ``Handoff.accept`` (validation + explicit-acceptance rules + baseline semantics)
 -> F6   ``F6Session.import_calibration`` (the existing Stage A validation; never bypassed)

Two paths end in the same review/accept step:
    Option A  new calibration : checkerboard settings -> LEFT/RIGHT chunked upload -> F8 full calibration -> review -> accept
    Option B  saved calibration: F8 result JSON | bare Stage A contract | Phase 1 file -> validation -> review -> accept

What this module never does: open a video, select frames, detect boundaries, triangulate or measure.  ``ROOM_PROCESSING_ENABLED`` is a constant, not a
setting; nothing here calls the F6 room steps (``submit_videos`` / ``select_frames`` / ``measure``), and the tests assert that.

Quality semantics are the F8.15 ones: GOOD may be accepted, QUESTIONABLE only with an acknowledgement, FAILED never.  A calibration FILE that carries no
F8 assessment is shown as NOT_ASSESSED (it is never labelled GOOD) and also needs the acknowledgement.
"""
from __future__ import annotations

import copy
import json
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(HERE), str(MODULE / "f8_chunk_poc"), str(MODULE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import f6_core as F                                                                  # noqa: E402
import f8_calibration_bridge as B                                                    # noqa: E402
from calibration_runner import CalibrationJobRunner, RunnerLimits, active_runner, cancel_by_code  # noqa: E402
import room_videos as RV                                                             # noqa: E402
import room_frames as RF                                                            # noqa: E402  (F8.31B)
import three_view_workflow as TVW                                                   # noqa: E402  (three-viewpoint Wall B frame sets + annotations)
import f8_measurement as MEAS                                                       # noqa: E402  (F8.31C-2A: the trusted measurement boundary; it loads the backend lazily and only behind the gate)
import f8_guided_steps as GS                                                      # noqa: E402  (F8.31D-3B)
JobLimits = F.VM.JobLimits                                                           # the F6 job limits, as f6_core already exposes them (no direct visitor_measurement import)
from room_videos import ROOM_UPLOAD_SLOTS, ALL_ROOM_SLOTS                           # noqa: E402
import f8_viewpoints as VPC                                                     # noqa: E402  (per-wall viewpoint count)
from chunk_store import SLOTS, ChunkStore, UploadError                                # noqa: E402
from result_store import ResultStore, build_result_document, job_id, new_code, valid_code  # noqa: E402

ROOM_PROCESSING_ENABLED = False          # F8.18: a constant on purpose. Room video processing / reconstruction is a LATER phase.
WALL_B_PROCESSING_ENABLED = True        # Cloud test: the three-viewpoint Wall B (room length) calculation, still subject to every admission check (memory budget, timeout, CPU limit). A constant, not a setting.
WALL_A_PROCESSING_ENABLED = True        # Cloud test: the validated two-viewpoint Wall A (width + height) calculation is open. The walls are gated independently of each other (see WALL_B_PROCESSING_ENABLED). A constant, not a setting.
GOOD, QUESTIONABLE, FAILED = B.GOOD, B.QUESTIONABLE, B.FAILED
NOT_ASSESSED, INVALID = "NOT_ASSESSED", "INVALID"
NOT_ASSESSED_CONCERN = "The quality of this calibration was not assessed by F8; only its numbers were checked for validity."
MAX_FILE_BYTES = B.MAX_DOCUMENT_BYTES
STEPS = ("home", "setup", "import", "review", "ready", "room", "frames", "measure")
BASELINE_RANGE_MM = B.BASELINE_RANGE_MM


class ServiceError(Exception):
    """A refusal with a visitor-safe message (never a path, code or traceback)."""


# ------------------------------------------------------------------ the checkerboard / baseline form
@dataclass(frozen=True)
class CalibrationSetup:
    cols: int
    rows: int
    square_mm: float
    baseline_mm: float | None

    def as_settings(self) -> dict:
        return dict(cols=self.cols, rows=self.rows, square_mm=self.square_mm, baseline_mm=self.baseline_mm)


def parse_setup(cols, rows, square_mm, baseline_mm=None) -> CalibrationSetup:
    """Validate the form.  Baseline 0 / None / '' = not entered (the calibrated |T| is then used, as in F8 and F7)."""
    def whole(v, name):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or int(v) != v or not 2 <= v <= 40:
            raise ServiceError(f"{name} must be a whole number from 2 to 40.")
        return int(v)
    c, r = whole(cols, "Inner corners (columns)"), whole(rows, "Inner corners (rows)")
    if c == r:
        raise ServiceError("The checkerboard must have a different number of columns and rows (an equal count cannot tell its orientation).")
    if isinstance(square_mm, bool) or not isinstance(square_mm, (int, float)) or not 1.0 <= square_mm <= 500.0 or square_mm != square_mm:
        raise ServiceError("The square size must be between 1 and 500 mm.")
    b = None
    if baseline_mm not in (None, "", 0, 0.0):
        lo, hi = BASELINE_RANGE_MM
        if isinstance(baseline_mm, bool) or not isinstance(baseline_mm, (int, float)) or baseline_mm != baseline_mm or not lo <= baseline_mm <= hi:
            raise ServiceError(f"The measured baseline must be between {lo:g} and {hi:g} mm (or left empty).")
        b = float(baseline_mm)
    return CalibrationSetup(c, r, float(square_mm), b)


# ------------------------------------------------------------------ review of one calibration candidate
@dataclass(frozen=True)
class Review:
    """A calibration that has been validated and is waiting for the visitor's decision.  Nothing here has touched the F6 session yet."""
    kind: str                                  # f8_job | f8_file | contract_file | phase1_file | f8_recovered
    level: str                                 # GOOD | QUESTIONABLE | FAILED | NOT_ASSESSED | INVALID
    headline: str | None = None
    concerns: tuple = ()
    problems: tuple = ()                       # why it cannot be used (FAILED / INVALID)
    quality_assessed_by_f8: bool = False
    baseline_used_mm: float | None = None
    baseline_source: str | None = None         # measured | calibrated | file | entered
    calibrated_baseline_mm: float | None = None
    stereo_rms_px: float | None = None
    image_size_wh: tuple | None = None
    job_id: str | None = None
    f6_warnings: tuple = ()
    _handoff: object = field(default=None, repr=False)
    _contract: dict | None = field(default=None, repr=False)

    @property
    def requires_acknowledgement(self) -> bool:
        return self.level in (QUESTIONABLE, NOT_ASSESSED)

    @property
    def usable(self) -> bool:
        return self.level in (GOOD, QUESTIONABLE, NOT_ASSESSED)

    def can_accept(self, acknowledged: bool = False) -> bool:
        return self.usable and (not self.requires_acknowledgement or bool(acknowledged))

    def contract_for_accept(self, acknowledged: bool = False) -> dict:
        """The Stage A contract, only when the acceptance rules are met.  F8 results go through ``Handoff.accept`` (the F8.15 rules)."""
        if self.level == FAILED:
            raise ServiceError("A failed calibration cannot be used.")
        if not self.usable:
            raise ServiceError("This calibration file cannot be used.")
        if self.requires_acknowledgement and not acknowledged:
            raise ServiceError("Acknowledge the notes above before using this calibration.")
        if self._handoff is not None:
            try:
                return self._handoff.accept(acknowledged=bool(acknowledged))
            except B.HandoffError as e:
                raise ServiceError(str(e)) from None
        return copy.deepcopy(self._contract)

    def provenance(self) -> dict:
        if self._handoff is not None:
            p = self._handoff.provenance(); p["kind"] = self.kind; p["quality_assessed_by_f8"] = True
            return p
        return dict(source="imported_calibration_file", kind=self.kind, quality_assessed_by_f8=False, level=NOT_ASSESSED, baseline_used_mm=self.baseline_used_mm,
                    baseline_source=self.baseline_source)


def _scratch_validate(contract: dict) -> dict:
    """Run the EXISTING F6 validation on a throw-away session so the review can show F6's verdict before anything is imported into the visitor's session."""
    s = F.F6Session()
    try:
        return s.import_calibration(copy.deepcopy(contract))
    finally:
        s.close()


def _f6_messages(r) -> tuple:
    return tuple(str(f.get("message", f.get("code"))) for f in (r.get("failures") or []))


def review_f8_document(doc, *, measured_baseline_mm=None, kind="f8_file") -> Review:
    """Validate an F8 result document (dict / text / bytes) with the F8.15 adapter, then with F6's own validation."""
    try:
        h = B.parse_f8_result(doc, measured_baseline_mm=measured_baseline_mm)
    except B.HandoffError as e:
        if "FAILED_CALIBRATION" in e.codes:
            return Review(kind=kind, level=FAILED, headline="The calibration failed quality checks.", problems=tuple(p["message"] for p in e.problems), quality_assessed_by_f8=True)
        return Review(kind=kind, level=INVALID, headline="This file cannot be used.", problems=tuple(p["message"] for p in e.problems))
    r = _scratch_validate(h._contract)
    if not r.get("can_proceed"):
        return Review(kind=kind, level=INVALID, headline="The measurement backend rejected this calibration.", problems=_f6_messages(r), quality_assessed_by_f8=True, job_id=h.job_id)
    return Review(kind=kind, level=h.level, headline=h.headline, concerns=h.concerns, quality_assessed_by_f8=True, baseline_used_mm=h.baseline_used_mm, baseline_source=h.baseline_source,
                  calibrated_baseline_mm=h.calibrated_baseline_mm, stereo_rms_px=h.stereo_rms_px, image_size_wh=h.image_size_wh, job_id=h.job_id,
                  f6_warnings=tuple(str(w.get("message")) for w in r.get("warnings", [])), _handoff=h)


def review_saved_file(data, *, capture_mode=None, baseline_mm=None) -> Review:
    """Option B.  ``data``: bytes/str of a calibration JSON.  Recognises (1) a full F8 result, (2) a bare Stage A contract, (3) a Phase 1 file.
    ``baseline_mm`` overrides the measured baseline of an F8 result, and fills a missing one in the other formats.  ``capture_mode`` ('video'|'photo') is only
    used when a non-F8 file does not state it; it is never guessed."""
    if isinstance(data, str):
        data = data.encode()
    if not isinstance(data, (bytes, bytearray)):
        return Review(kind="contract_file", level=INVALID, problems=("No file was provided.",))
    if len(data) > MAX_FILE_BYTES:
        return Review(kind="contract_file", level=INVALID, problems=("The file is too large to be a calibration file.",))
    try:
        d = json.loads(data)
    except (ValueError, RecursionError):
        return Review(kind="contract_file", level=INVALID, problems=("The file is not valid JSON.",))
    if not isinstance(d, dict):
        return Review(kind="contract_file", level=INVALID, problems=("The calibration file must be a JSON object.",))
    if baseline_mm is not None and (isinstance(baseline_mm, bool) or not isinstance(baseline_mm, (int, float)) or baseline_mm != baseline_mm or baseline_mm < 0):
        return Review(kind="contract_file", level=INVALID, problems=("The baseline must be a number of millimetres.",))
    baseline_mm = baseline_mm or None
    if d.get("schema") in B.SUPPORTED_SCHEMAS:                 # /1 (legacy, still valid) and /2 (F8.29); the bridge validates each
        return review_f8_document(d, measured_baseline_mm=baseline_mm, kind="f8_file")
    kind = "contract_file" if all(k in d for k in ("K1", "D1", "K2", "D2")) else "phase1_file"
    cal, probs = F.parse_calibration_json(data, capture_mode=capture_mode, baseline_mm=baseline_mm)
    if probs:
        return Review(kind=kind, level=INVALID, headline="This file cannot be used.", problems=tuple(probs))
    r = _scratch_validate(cal)
    if not r.get("can_proceed"):
        return Review(kind=kind, level=INVALID, headline="This calibration cannot be used.", problems=_f6_messages(r))
    c = r["calibration"]
    src = "entered" if baseline_mm is not None and abs(float(c["baseline_mm"]) - float(baseline_mm)) < 1e-9 else "file"     # report what F6 actually validated
    return Review(kind=kind, level=NOT_ASSESSED, headline="This file contains no calibration quality assessment.", concerns=(
        NOT_ASSESSED_CONCERN,), baseline_used_mm=float(c["baseline_mm"]), baseline_source=src,
        image_size_wh=tuple(c["image_size_wh"]), f6_warnings=tuple(str(w.get("message")) for w in r.get("warnings", [])), _contract=cal)


# ------------------------------------------------------------------ the visitor's session object (plain Python; the page keeps ONE of these)
@dataclass
class AcceptResult:
    ok: bool
    problems: tuple = ()
    baseline_used_mm: float | None = None
    image_size_wh: tuple | None = None
    level: str | None = None
    provenance: dict | None = None
    can_proceed: bool = False


class VisitorSession:
    """All per-visitor state of the integrated experiment.  Holds no file bytes.  ``reset`` is the single cleanup path (Start over / Lock / session end)."""

    def __init__(self):
        self.owner = secrets.token_urlsafe(24)
        self.f6: F.F6Session | None = None
        self.step = "home"
        self.review: Review | None = None
        self.runner: CalibrationJobRunner | None = None
        self.job_code: str | None = None
        self.setup: CalibrationSetup | None = None
        self.accepted: AcceptResult | None = None
        self.contract: dict | None = None            # the accepted Stage A contract (calibration numbers only), for the download
        self.epoch = 0                               # bumped on every cleanup so the upload widgets reset
        self.protos = {s: {} for s in SLOTS}
        self.recovered_code: str | None = None
        self.n_viewpoints = VPC.DEFAULT_COUNT                      # stereo viewpoints of the ACTIVE wall (2 = the original workflow; 3 only for Wall B)
        self.room_protos = {s: {} for s in ALL_ROOM_SLOTS}         # F8.31A: per-slot upload-protocol state of the room videos (holds no file data); a wall uses the first four slots, or all six
        self.room_inspect: dict = {}                               # upload_id -> inspector result (facts only; the verdict is recomputed against the CURRENT calibration)
        self.room_ready = False                                    # set only by ``VisitorCalibrationService.room_report``
        self.room_slot_epoch = {s: 0 for s in ALL_ROOM_SLOTS}      # F8.31E: bumped when ONE room video is removed so that slot's uploader resets without being re-created
        self.frames: RF.RoomFrameWorkflow | None = None            # F8.31B: this visitor's frame selection + annotations (own private workspace)
        self.frame_notes: list = []                                # why earlier work was discarded (shown on the frames page)
        self.measurement = MEAS.MeasurementRecord()                # F8.31C-2A: this visitor's measurement job / result (memory only; bound to the exact inputs)
        self.guide = GS.GuidedSteps()                              # F8.31D-3B: this visitor's step confirmations + step-5 inputs (the other step states are derived from backend facts)
        # Two walls (A = first wall: room width + height, B = second wall: room length).  The attributes above (room_protos, room_inspect, room_ready, room_slot_epoch, frames,
        # frame_notes, measurement, guide.room, room_owner) always hold the ACTIVE wall, so every existing call site and the whole Wall A path work unchanged; the other wall's
        # copy is stashed in ``_walls`` and swapped in by ``switch_wall``.  The calibration (f6, contract, accepted) is shared.  Wall B has its own upload-ownership token,
        # so its videos can never collide with, or be read as, Wall A's.
        self.active_wall = "A"
        self.room_owner = self.owner                              # Wall A's room uploads are owned by the session owner token (unchanged behaviour)
        self._walls = {"A": None, "B": self._fresh_wall(secrets.token_urlsafe(24))}

    # ---- walls
    @staticmethod
    def _fresh_wall(owner) -> dict:
        steps = GS.RoomSteps(); steps.confirmed = {1: True, 2: True}      # the second wall has no preparation / recording steps: only the four uploads
        return dict(room_owner=owner, room_protos={s: {} for s in ALL_ROOM_SLOTS}, room_inspect={}, room_ready=False, room_slot_epoch={s: 500 for s in ALL_ROOM_SLOTS},
                    frames=None, frame_notes=[], measurement=MEAS.MeasurementRecord(), room_steps=steps, ui={}, n_viewpoints=VPC.DEFAULT_COUNT)

    def _snapshot(self) -> dict:
        return dict(room_owner=self.room_owner, room_protos=self.room_protos, room_inspect=self.room_inspect, room_ready=self.room_ready, room_slot_epoch=self.room_slot_epoch,
                    frames=self.frames, frame_notes=self.frame_notes, measurement=self.measurement, room_steps=self.guide.room, ui=getattr(self, "_ui", {}), n_viewpoints=self.n_viewpoints)

    def _restore(self, d: dict):
        self.room_owner = d["room_owner"]; self.room_protos = d["room_protos"]; self.room_inspect = d["room_inspect"]; self.room_ready = d["room_ready"]
        self.room_slot_epoch = d["room_slot_epoch"]; self.frames = d["frames"]; self.frame_notes = d["frame_notes"]; self.measurement = d["measurement"]
        self.guide.room = d["room_steps"]; self._ui = d["ui"]; self.n_viewpoints = d.get("n_viewpoints", VPC.DEFAULT_COUNT)

    def _switch(self, w: str):
        if w == self.active_wall:
            return
        self._walls[self.active_wall] = self._snapshot()
        snap, self._walls[w] = self._walls[w], None
        self._restore(snap); self.active_wall = w

    def switch_wall(self, w: str):
        """Make wall ``w`` the active one (its uploads, frames, boundaries and result are swapped in; the other wall's are kept, untouched)."""
        if w not in ("A", "B"):
            raise ServiceError("Unknown wall.")
        if w != self.active_wall and self.measurement.running:
            raise ServiceError("A measurement is running. Cancel it first, or wait for it to finish.")
        self._switch(w)

    def on_wall(self, w: str):
        """Context manager: run something against wall ``w`` and come back (used to read the other wall; no UI state is touched)."""
        from contextlib import contextmanager

        @contextmanager
        def cm():
            prev = self.active_wall
            self._switch(w)
            try:
                yield self
            finally:
                self._switch(prev)
        return cm()

    def wall_owners(self) -> list:
        """The upload-ownership tokens of both walls (the active one live, the other from its stash)."""
        return [self.room_owner] + [w["room_owner"] for w in self._walls.values() if w]

    # ---- navigation
    @property
    def busy(self) -> bool:
        return self.runner is not None and not self.runner.done

    def go(self, step: str):
        if step not in STEPS:
            raise ServiceError("Unknown page.")
        if step in ("ready", "room", "frames", "measure") and (self.accepted is None or self.contract is None or self.f6 is None or self.f6.closed):
            raise ServiceError("Accept a calibration first.")
        if step in ("frames", "measure") and not self.room_ready:
            raise ServiceError(f"Upload and validate all {VPC.count_word(self.n_viewpoints)} room videos first.")
        if step == "measure" and not (self.frames is not None and not self.frames.closed and (self.frames.readiness()["ready"] if self.n_viewpoints == 3 else self.frames.readiness().ready)):
            raise ServiceError("Select the frames and mark every boundary first.")
        if self.measurement.running and step != "measure":
            raise ServiceError("A measurement is running. Cancel it first, or wait for it to finish.")
        if self.busy and step not in ("setup",):
            raise ServiceError("A calibration is running. Cancel it first, or wait for it to finish.")
        if step in ("home", "import"):
            self.review = None
        self.step = step

    def forget_acceptance(self):
        """F8.26: a NEW calibration attempt (job, import, recovered result) must never silently reuse the previously accepted contract or its F6 session.  Both walls are affected."""
        if self.f6 is not None:
            self.f6.close()
        self.f6 = None; self.accepted = None; self.contract = None
        for w in ("A", "B"):
            with self.on_wall(w):
                self.measurement.discard("Your calibration changed, so the previous measurement result was discarded." if self.measurement.result is not None else None)
                self.room_ready = False                      # room videos were checked against the old calibration; they are re-checked against the next one
                self.guide.room.reset()                      # F8.31D-4: the room-page confirmations were made for the old calibration
                if w == "B":
                    self.guide.room.confirmed = {1: True, 2: True}
                self.close_frames("Your calibration changed, so your frame selections and annotations were discarded." if self.frames is not None else None)
        if self.active_wall == "B":
            self.switch_wall("A")

    def close_frames(self, note: str | None = None):
        """Delete this visitor's frame workspace (previews, extracted frames) and forget the selections and annotations."""
        self.measurement.discard("Your frame selections changed, so the previous measurement result was discarded." if self.measurement.result is not None else None)
        if self.frames is not None:
            self.frames.close()
        self.frames = None
        if note:
            self.frame_notes.append(note)

    def f6_session(self) -> F.F6Session:
        if self.f6 is None or self.f6.closed:
            # F8.31B: the room path must use the ROOM file-size limit, never F6's 200 MB default (otherwise a file accepted at upload fails in the next stage)
            self.f6 = F.F6Session(job_kwargs=dict(limits=JobLimits(video=RV.room_video_limits())))
        return self.f6

    def reset(self, store: ChunkStore | None = None, room_store: ChunkStore | None = None):
        """Start over: cancel a running job, delete this session's uploads, close the F6 session, forget the review / acceptance / code, reset the widgets.
        Saved result files stay on the server until their time-to-live (they are only reachable with the recovery code)."""
        if store is not None:
            try:
                store.release_owner(self.owner)
            except Exception:
                pass
        if room_store is not None:
            for o in {self.owner, self.room_owner, *(w["room_owner"] for w in self._walls.values() if w)}:        # both walls' uploads
                try:
                    room_store.release_owner(o)
                except Exception:
                    pass
        if self.runner is not None and not self.runner.done:
            try:
                self.runner.cancel()
            except Exception:
                pass
        for w in ("A", "B"):
            with self.on_wall(w):
                self.close_frames()
                self.measurement.discard(); self.measurement = MEAS.MeasurementRecord()
        self._switch("A")
        self._walls["B"] = self._fresh_wall(secrets.token_urlsafe(24))
        if self.f6 is not None:
            self.f6.close()
        self.f6 = None; self.review = None; self.runner = None; self.job_code = None; self.setup = None; self.accepted = None; self.contract = None
        self.recovered_code = None; self.step = "home"; self.epoch += 1; self.protos = {s: {} for s in SLOTS}
        self.room_protos = {s: {} for s in ALL_ROOM_SLOTS}; self.room_inspect = {}; self.room_ready = False; self.frame_notes = []; self.n_viewpoints = VPC.DEFAULT_COUNT
        self.owner = secrets.token_urlsafe(24)       # a fresh upload-ownership token
        self.room_owner = self.owner
        self.guide.reset()


# ------------------------------------------------------------------ the service
class VisitorCalibrationService:
    def __init__(self, store: ChunkStore, results: ResultStore, limits: RunnerLimits | None = None, *, runner_cls=CalibrationJobRunner, runner_kwargs: dict | None = None,
                 room_store: ChunkStore | None = None, inspector=None, frame_worker_cmd=None, frame_workspace_root=None, frame_limits: JobLimits | None = None,
                 measurement_runner=None, measurement_gate=None, measurement_limits=None, measurement_coordinator=None):
        self.store, self.results = store, results
        self.room_store = room_store                             # F8.31A: separate bounded store for the four room videos (None = room upload unavailable)
        self._inspector = inspector                              # test hook; default is the sandboxed ``video_check.inspect_video``
        self._frame_worker_cmd, self._frame_root = frame_worker_cmd, frame_workspace_root       # F8.31B test hooks (default: the sandboxed ``frame_worker``)
        self._frame_limits = frame_limits
        self.limits = limits if limits is not None else RunnerLimits.from_env()
        self.runner_cls, self.runner_kwargs = runner_cls, dict(runner_kwargs or {})
        # F8.31C-2A: the measurement_* arguments are server-side configuration and unit-test seams; the application factory passes none of them (a test checks that).
        self.measurement = MEAS.MeasurementService(self, gate=measurement_gate, runner=measurement_runner, limits=measurement_limits, coordinator=measurement_coordinator)

    @property
    def full_calibration_available(self) -> bool:
        return bool(self.limits.allow_full)

    def slots_ready(self, session: VisitorSession) -> bool:
        slots = self.store.job_status(session.owner)["slots"]
        return all(slots.get(s, {}).get("state") == "complete" for s in SLOTS)

    # ---- Option A
    def start_calibration(self, session: VisitorSession, setup: CalibrationSetup) -> str:
        """Only ever called from an explicit button press.  Returns nothing secret-bearing except the recovery code, which the page shows to this session only."""
        if not self.full_calibration_available:
            raise ServiceError("Full calibration is not enabled on this server. Use Option B (import a saved calibration), or ask the administrator to enable it.")
        if session.runner is not None:
            raise ServiceError("A calibration has already been started in this session.")
        if not self.slots_ready(session):
            raise ServiceError("Upload and verify both the LEFT and the RIGHT video first.")
        code = new_code()
        self.store.set_recovery(session.owner, code)             # BEFORE the job starts, so the code can be noted while it runs
        runner = self.runner_cls(self.store, session.owner, setup.as_settings(), self.limits, mode="calibrate", results=self.results, job_code=code, **self.runner_kwargs)
        runner.start()
        session.forget_acceptance()
        session.runner, session.job_code, session.setup, session.review = runner, code, setup, None
        session.step = "setup"
        return code

    def cancel(self, session: VisitorSession) -> bool:
        if session.runner is not None and not session.runner.done:
            session.runner.cancel()
            return True
        return False

    def finished_review(self, session: VisitorSession) -> Review:
        """Review of the FINISHED job of this session (the saved result document; rebuilt from the runner's result only if saving failed)."""
        r = session.runner
        if r is None or not r.done:
            raise ServiceError("The calibration has not finished.")
        doc = self.results.load_result(session.job_code)
        if doc is None:
            doc = build_result_document(r.result, code=session.job_code, settings=session.setup.as_settings() if session.setup else None)
        rev = review_f8_document(doc, kind="f8_job")
        if rev.level == INVALID and (doc.get("outcome") not in (None, "completed")):
            rev = Review(kind="f8_job", level=FAILED, headline="The calibration did not complete.", problems=(f"Outcome: {doc.get('outcome')}. " + str(doc.get("error") or "")[:200],) , quality_assessed_by_f8=True)
        session.review = rev
        return rev

    def result_bytes(self, session: VisitorSession) -> bytes | None:
        code = session.job_code or session.recovered_code
        return self.results.result_bytes(code) if code else None

    def retry_possible(self, session: VisitorSession) -> bool:
        r = session.runner
        return bool(r is not None and r.done and ((r.result or {}).get("cleanup") or {}).get("videos_kept_for_next_step"))

    def clear_result(self, session: VisitorSession):
        """Forget a finished job so another run can start.  The uploaded videos stay only if F8 kept them for a retry."""
        kept = self.retry_possible(session)
        session.runner = None; session.job_code = None; session.review = None
        if not kept:
            session.epoch += 1; session.protos = {s: {} for s in SLOTS}

    # ---- recovery by code (a new browser session / refresh)
    def recover(self, code) -> dict:
        """-> dict(state=invalid|running|result|interrupted|not_found, ...).  The code is a secret: it is never logged or echoed back."""
        code = (code or "").strip() if isinstance(code, str) else ""
        if not valid_code(code):
            return dict(state="invalid", message="That does not look like a recovery code.")
        live = active_runner(code)
        if live is not None:
            s = live.status()
            return dict(state="running", stage=str(s["stage"]).replace("_", " "), elapsed_s=s["elapsed_s"], fraction=s["fraction"])
        doc = self.results.load_result(code)
        if doc is not None:
            if doc.get("mode") == "benchmark":
                return dict(state="not_found", message="That code belongs to a benchmark, not a calibration.")
            return dict(state="result", review=review_f8_document(doc, kind="f8_recovered"))
        st = self.results.read_status(code)
        if st:
            return dict(state="interrupted", message="No saved result. The job is not running now: the server was probably restarted or the job was interrupted.")
        return dict(state="not_found", message="Nothing found for that code: it expired, or the server restarted (its local disk is temporary).")

    def cancel_recovered(self, code) -> bool:
        return bool(valid_code(code or "") and cancel_by_code(code.strip()))

    def use_recovered(self, session: VisitorSession, code) -> Review:
        info = self.recover(code)
        if info["state"] != "result":
            raise ServiceError(info.get("message") or "No saved result is available for that code.")
        session.forget_acceptance()
        session.review = info["review"]; session.recovered_code = code.strip(); session.step = "review"
        return session.review

    def reclaim_videos(self, session: VisitorSession, code) -> int:
        try:
            return int(self.store.adopt(code.strip(), session.owner)["adopted"])
        except UploadError as e:
            raise ServiceError(e.message) from None

    # ---- Option B
    def import_saved(self, session: VisitorSession, data, *, capture_mode=None, baseline_mm=None) -> Review:
        rev = review_saved_file(data, capture_mode=capture_mode, baseline_mm=baseline_mm)
        session.forget_acceptance()
        session.review = rev; session.step = "review"
        return rev

    # ---- the shared acceptance step
    def accept(self, session: VisitorSession, acknowledged: bool = False) -> AcceptResult:
        """Explicit acceptance: acceptance rules -> contract -> ``F6Session.import_calibration`` -> verify ``can_proceed`` -> provenance.  Touches nothing else in F6."""
        rev = session.review
        if rev is None:
            raise ServiceError("There is no calibration to accept.")
        contract = rev.contract_for_accept(acknowledged)         # raises for FAILED / INVALID / missing acknowledgement
        f6 = session.f6_session()
        r = f6.import_calibration(contract)
        if r.get("can_proceed") is not True:
            f6.calibration_provenance = None
            return AcceptResult(ok=False, problems=_f6_messages(r) or ("The measurement backend rejected this calibration.",))
        cal = r["calibration"]
        prov = rev.provenance(); prov["baseline_used_mm"] = float(cal["baseline_mm"])
        f6.calibration_provenance = prov
        res = AcceptResult(ok=True, baseline_used_mm=float(cal["baseline_mm"]), image_size_wh=tuple(cal["image_size_wh"]), level=rev.level, provenance=prov, can_proceed=True)
        session.accepted, session.contract, session.step = res, contract, "ready"
        return res

    def contract_json(self, session: VisitorSession) -> bytes | None:
        """The accepted Stage A contract as JSON (calibration numbers only): re-importable through Option B."""
        return None if session.contract is None else json.dumps(session.contract, indent=1).encode()

    # ---- F8.31A: four room videos (upload + metadata validation ONLY: nothing is decoded beyond the sandboxed inspector, nothing is measured)
    def calibration_context(self, session: VisitorSession) -> dict | None:
        """A read-only SUMMARY of the accepted calibration (never the matrices, never modified here)."""
        c = session.contract
        if session.accepted is None or c is None or session.f6 is None or session.f6.closed:
            return None
        return dict(image_size_wh=list(c["image_size_wh"]), capture_mode=dict(c.get("capture_mode") or {}), baseline_mm=float(session.accepted.baseline_used_mm),
                    quality=session.accepted.level, baseline_source=(session.accepted.provenance or {}).get("baseline_source"))

    def _inspect(self, path):
        if self._inspector is not None:
            return self._inspector(path)
        from video_check import inspect_video
        # Phase D.3: the inspector starts a sandboxed OpenCV decoder: it runs under an admission slot (see room_videos), never next to a calibration or reconstruction.
        # Raises ``RV.InspectionBusy`` when refused; the caller treats that as "not checked yet", never as a verdict on the video.
        return RV.inspect_with_admission(path, inspect_video)

    def room_uploads(self, session: VisitorSession) -> dict:
        """slot -> facts of this session's upload.  A completed file is inspected ONCE (cached by upload id); the cache holds metadata only."""
        if self.room_store is None:
            return {}
        out = {}
        for slot, u in self.room_store.job_status(session.room_owner)["slots"].items():
            if slot not in RV.slots_for(session.n_viewpoints):
                continue
            d = dict(name=u["name"], size_bytes=u["size"], sha256=u["sha256"], state=u["state"], inspection=None, bytes_received=u["bytes_received"], next_index=u["next_index"], n_chunks=u["n_chunks"])
            if u["state"] == "complete":
                uid = u["upload_id"]
                if uid not in session.room_inspect:
                    try:
                        session.room_inspect[uid] = self._inspect(self.room_store.path_of(session.room_owner, uid))
                    except UploadError:
                        continue
                    except RV.InspectionBusy:                    # the server was busy: NOT a verdict on the video and NOT cached.  The slot stays "being checked" and is inspected again on the next page run
                        out[slot] = d
                        continue
                d["inspection"] = session.room_inspect[uid]
            out[slot] = d
        keep = {u["upload_id"] for u in self.room_store.job_status(session.room_owner)["slots"].values()}
        for k in [k for k in session.room_inspect if k not in keep]:
            del session.room_inspect[k]                          # replaced / deleted uploads leave nothing behind
        return out

    def room_report(self, session: VisitorSession) -> RV.RoomSetReport:
        rep = RV.assess_room_set(self.room_uploads(session), self.calibration_context(session), RV.slots_for(session.n_viewpoints))
        session.room_ready = bool(rep.ready and self.calibration_context(session) is not None)
        return rep

    def removal_impact(self, session: VisitorSession, slot: str) -> list:
        """F8.31E: what removing this video would discard besides the video itself (empty = nothing significant), so the page can ask before deleting."""
        if slot not in RV.slots_for(session.n_viewpoints):
            return []
        out, vp = [], RV.SLOT_INFO[slot][0]
        wf = session.frames
        if wf is not None and not wf.closed and session.n_viewpoints == 3:
            out.extend(wf.removal_impact(slot))                                  # three-viewpoint workflow: its own record of what depends on this video
        elif wf is not None and not wf.closed:
            n_pairs = sum(1 for (v, _) in wf.pairs if v == vp)
            n_ann = sum(1 for (i, line), a in wf.ann.items() if line[:2] in (RF.VIEW_KEY[(vp, "left")], RF.VIEW_KEY[(vp, "right")]) and a.accepted)
            if n_pairs:
                out.append(f"your {n_pairs} suggested or chosen frame pairs for Viewpoint {vp}")
            if n_ann:
                out.append(f"{n_ann} accepted wall boundaries for Viewpoint {vp}")
        if session.measurement.result is not None:
            out.append("the measurement result")
        return out

    def remove_room_video(self, session: VisitorSession, slot: str) -> bool:
        """F8.31E: delete ONE of this session's room videos (never another session's: the store checks the owner token) and invalidate whatever depended on it.
        The accepted calibration, the other three videos and every other session are untouched.  Returns True when a file or partial upload was removed."""
        if self.room_store is None or slot not in RV.slots_for(session.n_viewpoints):
            raise ServiceError("Unknown video.")
        info = self.room_store.job_status(session.room_owner)["slots"].get(slot)
        removed = False
        if info is not None:
            try:
                self.room_store.cancel(session.room_owner, info["upload_id"])         # owner-checked deletion of the file / partial upload
                removed = True
            except UploadError:
                pass
        session.room_slot_epoch[slot] = session.room_slot_epoch.get(slot, 0) + 1   # the browser widget of this slot clears itself (same component identity)
        if session.measurement.result is not None:
            session.measurement.discard("A room video was removed, so the previous measurement result was discarded.")
        self.room_report(session)                                                # counter / readiness recomputed from the store NOW
        if session.frames is not None and not session.frames.closed and session.f6 is not None and not session.f6.closed and session.contract is not None:
            try:
                session.frame_notes.extend(session.frames.sync(session.room_owner, session.contract, session.f6.job.cal, self.room_sources(session)))
            except RF.WorkflowError:
                session.close_frames()
        return removed

    def go_frames(self, session: VisitorSession) -> None:
        """Continue to the (placeholder) frame-selection page.  Re-validates now; never trusts a flag left by an earlier render."""
        rep = self.room_report(session)
        if not session.room_ready:
            raise ServiceError(f"The {VPC.count_word(session.n_viewpoints)} room videos are not all valid yet.")
        session.go("frames")

    def set_viewpoint_count(self, session: VisitorSession, wall: str, n: int) -> list:
        """Choose 2 or 3 stereo viewpoints for ONE wall.  Only that wall's incompatible work is discarded: its uploaded videos, frame selections, boundaries and measurement result.
        The other wall (uploads, frames, result), the accepted calibration and the other wall's viewpoint count are untouched.  Wall A cannot use three viewpoints (not validated).
        Returns the notes shown to the visitor (empty when nothing changed).  Raises ServiceError (nothing changed) when refused."""
        if wall not in ("A", "B"):
            raise ServiceError("Unknown wall.")
        why = VPC.refusal(wall, n)
        if why:
            raise ServiceError(why)
        with session.on_wall(wall):
            if session.n_viewpoints == n:
                return []
            if session.measurement.running:
                raise ServiceError("A measurement is running. Cancel it first, or wait for it to finish.")
            if self.room_store is not None:
                try:
                    self.room_store.release_owner(session.room_owner)                 # this wall's uploads only (the other wall has its own ownership token)
                except Exception:
                    pass
            session.measurement.discard("The number of viewpoints changed, so the previous measurement result was discarded." if session.measurement.result is not None else None)
            session.close_frames("The number of viewpoints changed, so your frame selections and boundaries were discarded." if session.frames is not None else None)
            session.n_viewpoints = n
            session.room_inspect = {}; session.room_ready = False
            session.room_protos = {s: {} for s in ALL_ROOM_SLOTS}
            session.room_slot_epoch = {s: session.room_slot_epoch.get(s, 0) + 1 for s in ALL_ROOM_SLOTS}      # every uploader of this wall clears itself
            if wall == "B":
                session.guide.room.confirmed = {1: True, 2: True}
        return [f"Wall {wall} now uses {n} viewpoints ({VPC.n_videos(n)} videos). Its earlier uploads and results were discarded."]

    def room_processing_enabled(self) -> bool:
        return ROOM_PROCESSING_ENABLED


    # ---- F8.31B: frame selection and wall-boundary annotation (previews and frame extraction run ONLY in the sandboxed ``frame_worker``)
    def frame_job_limits(self) -> JobLimits:
        """The limits for every frame inspection / extraction job: the F6 job limits with the ROOM video file-size ceiling (``JobLimits(video=room_video_limits())``),
        never F6's default 200 MB."""
        return self._frame_limits if self._frame_limits is not None else JobLimits(video=RV.room_video_limits())

    def room_sources(self, session: VisitorSession) -> dict:
        """slot -> ``SourceVideo`` for every COMPLETE, inspected, still-existing upload of this session (None for a slot that is missing or expired)."""
        out = {s: None for s in RV.slots_for(session.n_viewpoints)}
        if self.room_store is None:
            return out
        slots = self.room_store.job_status(session.room_owner)["slots"]
        for slot, u in self.room_uploads(session).items():
            info = u.get("inspection")
            if u["state"] != "complete" or not info or not info.get("decodable") or slot not in slots:
                continue
            uid = slots[slot]["upload_id"]
            try:
                path = self.room_store.path_of(session.room_owner, uid)
            except UploadError:
                continue
            if not path.is_file():
                continue
            out[slot] = RF.SourceVideo(slot, str(path), uid, u["sha256"], int(u["size_bytes"]), int(info["frame_count"]), float(info["fps"]), (int(info["width"]), int(info["height"])))
        return out

    def touch(self, session: VisitorSession):
        """A real interaction on the frames page counts as activity (the inactivity rule); an already expired upload is never revived."""
        if self.room_store is not None:
            for o in session.wall_owners():                    # activity on either wall keeps both walls' uploads alive
                self.room_store.touch_owner(o)

    def frame_state(self, session: VisitorSession) -> dict:
        """-> dict(state='ok'|'no_calibration'|'expired'|'not_ready', workflow=..., problems=[...]).  Re-validates NOW and synchronises the workflow with what is on disk:
        a replaced or expired video, or a changed calibration, discards the affected selections and annotations before anything is shown."""
        ctx = self.calibration_context(session)
        if ctx is None or session.f6 is None or session.f6.closed or session.contract is None:
            session.close_frames("Your calibration is no longer accepted, so your frame selections and annotations were discarded." if session.frames is not None else None)
            return dict(state="no_calibration", workflow=None, problems=["There is no accepted calibration."])
        self.touch(session)
        sources = self.room_sources(session)
        cal = session.f6.job.cal
        if session.frames is not None:
            try:
                session.frame_notes.extend(session.frames.sync(session.room_owner, session.contract, cal, sources))
            except RF.WorkflowError:
                session.close_frames()
        missing = [RV.SLOT_INFO[s][2] for s, v in sources.items() if v is None]
        rep = self.room_report(session)
        if missing:
            return dict(state="expired", workflow=session.frames, problems=[f"{m} is missing or has expired." for m in missing])
        if not rep.ready:
            return dict(state="not_ready", workflow=session.frames, problems=rep.all_errors or ["The four room videos are not all valid."])
        if session.frames is None:
            cls = TVW.ThreeViewWorkflow if session.n_viewpoints == 3 else RF.RoomFrameWorkflow           # the two-viewpoint workflow is created exactly as before
            session.frames = cls(session.room_owner, session.contract, cal, sources, self.frame_job_limits(), worker_cmd=self._frame_worker_cmd, workspace_root=self._frame_root)
        return dict(state="ok", workflow=session.frames, problems=[])

    def go_measure(self, session: VisitorSession) -> None:
        """Continue to the (placeholder) Room Measurement page.  Deep re-validation at click time; nothing is executed."""
        st = self.frame_state(session)
        if st["state"] != "ok":
            raise ServiceError("Your uploads or calibration changed. " + " ".join(st["problems"]))
        wf = st["workflow"]
        if not (wf.readiness()["ready"] if session.n_viewpoints == 3 else wf.readiness(deep=True).ready):
            raise ServiceError("Frame selection and annotation are not complete yet.")
        session.go("measure")
