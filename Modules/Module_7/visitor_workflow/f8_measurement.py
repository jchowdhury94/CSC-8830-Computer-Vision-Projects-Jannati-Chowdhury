"""F8.31C-2A: the TRUSTED service boundary between the visitor interface and the validated reconstruction backend (``room_reconstruction``).  No Streamlit import.

This is the only live file (besides the backend itself) allowed to name ``room_reconstruction``; it imports it lazily, and never imports Method J or the adapter.

What it does
    * ``readiness`` -- is every input present and valid?  The verdict IS the backend's own input check (``room_reconstruction.prepare``) plus the frame workflow's checks.  There is
      no UI-only rule: a visible form field being filled never makes a measurement ready.
    * ``start`` -- the only entry to a reconstruction.  FIRST it checks the gate; while ``ROOM_PROCESSING_ENABLED`` is False it raises and nothing else happens.  Its only parameter
      is the visitor session: resource limits, the shared resource coordinator, upload retention, the private source paths, the ownership context and the job identifier are all
      built here, on the server, never from visitor input.
    * binding -- a result belongs to the exact (owner, calibration, four videos, frame sets, annotations) it was computed from.  Any change discards it; it is never shown for
      other inputs or to another visitor, and it is never written to disk.

Gate and test seam
    The gate is the constant ``f8_visitor_service.ROOM_PROCESSING_ENABLED`` (False; no environment variable, query parameter or callback changes it).  The constructor accepts a ``runner``
    and a ``gate`` for unit tests only (dependency injection of synthetic results); the application factory passes neither, and a test checks that.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(HERE), str(MODULE / "f8_chunk_poc"), str(MODULE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import f8_result_view as RVW                                   # noqa: E402
import room_frames as RF                                       # noqa: E402
import room_videos as RV                                       # noqa: E402
import three_view_workflow as TVW                              # noqa: E402  (three-viewpoint Wall B)

READINESS_SCHEMA = "f8.measurement_readiness/1"


class MeasurementError(Exception):
    """A refusal with a visitor-safe message (never a path, code or traceback)."""


def _backend():
    """The reconstruction backend, imported lazily so that merely importing this module (or a page) loads nothing of it."""
    import room_reconstruction as RR
    return RR


# ------------------------------------------------------------------ readiness
@dataclass(frozen=True)
class ReadinessItem:
    key: str                       # calibration | videos | frames | annotations
    label: str
    ok: bool
    detail: str
    step: str | None               # where the visitor can fix it ('home' | 'room' | 'frames')
    problems: tuple = ()           # visitor-readable reasons (from the backend's typed failures)
    codes: tuple = ()


@dataclass(frozen=True)
class MeasurementReadiness:
    ready: bool
    items: tuple
    summary: dict
    warnings: tuple                # annotation warnings the visitor acknowledged
    input_status: str              # current | incomplete | expired | replaced | calibration_changed | not_accepted
    binding: dict | None           # fingerprints of the exact inputs (None when there is nothing to bind)
    gate_open: bool

    @property
    def first_problem(self):
        return next((i for i in self.items if not i.ok), None)

    def to_dict(self) -> dict:
        return dict(schema=READINESS_SCHEMA, ready=self.ready, input_status=self.input_status, gate_open=self.gate_open, summary=self.summary,
                    items=[dict(key=i.key, ok=i.ok, step=i.step, problems=list(i.problems), codes=list(i.codes)) for i in self.items])


_STAGE_ITEM = {"ownership": "calibration", "calibration": "calibration", "videos": "videos", "retention": "videos", "frames": "frames", "annotations": "annotations", "request": "frames"}
ITEM_LABEL = {"calibration": "Accepted calibration", "videos": "Four room videos", "frames": "Five frame sets", "annotations": "Six wall boundaries, marked once and refined in every frame set"}
ITEM_LABEL_3 = {"calibration": "Accepted calibration", "videos": "Six room videos", "frames": "Frame sets (at least two usable)", "annotations": "The four vertical wall corners in each usable frame set"}


def _item_of(failure: dict) -> str:
    code = str(failure.get("code"))
    if code in ("CALIBRATION_CHANGED",):
        return "frames"                                   # the frames were chosen under another calibration
    if code in ("VIDEO_REPLACED",):
        return "videos"
    return _STAGE_ITEM.get(str(failure.get("stage")), "frames")


# ------------------------------------------------------------------ binding (what a result belongs to)
def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def binding_of(owner: str, contract: dict | None, wf, sources: dict) -> dict | None:
    """Fingerprints of the exact inputs a measurement would use (no pixels, no paths).  None when there is no accepted calibration or workflow."""
    if contract is None or wf is None or getattr(wf, "closed", False):
        return None
    if isinstance(wf, TVW.ThreeViewWorkflow):
        videos = sorted((s, v.sha256, v.upload_id) for s, v in (sources or {}).items() if v is not None)
        b = dict(owner=RF.owner_digest(owner), calibration=RF.calibration_id(contract), videos=_digest(videos), method="three_viewpoint_wall_b", **wf.fingerprint())
        b["key"] = _digest(b)
        return b
    videos = sorted((s, v.sha256, v.upload_id) for s, v in (sources or {}).items() if v is not None)
    frames = [dict(frame_set=i, **{vk: (wf.full[(i, vk)]["index"], wf.full[(i, vk)]["fingerprint"]) for vk in RF.VIEW_KEYS if (i, vk) in wf.full}) for i in range(wf.n_sets)]
    anns = []
    for i in range(wf.n_sets):
        for line in RF.REQUIRED_LINES:
            st = wf.ann.get((i, line))
            anns.append((i, line, None if st is None else (bool(st.accepted), int(st.revision), [list(map(float, p)) for p in st.points], st.frame_index, st.frame_fingerprint,
                                                           list(st.acknowledged_codes))))
    b = dict(owner=RF.owner_digest(owner), calibration=RF.calibration_id(contract), videos=_digest(videos), frame_sets=_digest(frames), annotations=_digest(anns))
    b["key"] = _digest(b)
    return b


# ------------------------------------------------------------------ one run
@dataclass
class RunContext:
    """Everything a runner receives.  Built ONLY by the service from server-side state."""
    job_id: str
    workflow: object
    owner: str
    contract: dict
    calibration: object
    sources: dict
    limits: object
    coordinator: object
    retention: object
    cancel: threading.Event


@dataclass
class MeasurementJob:
    job_id: str
    binding: dict
    started: float
    cancel: threading.Event
    stage: str = "validating inputs"
    done: bool = False
    result: dict | None = None
    cancelled: bool = False
    cleanup: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def running(self) -> bool:
        return not self.done

    def elapsed(self) -> float:
        return time.monotonic() - self.started


class MeasurementRecord:
    """The per-visitor measurement state (lives in the visitor's own session object, never on disk, never shared)."""

    def __init__(self):
        self.job: MeasurementJob | None = None
        self.result: dict | None = None
        self.binding: dict | None = None            # what ``result`` was computed from
        self.job_id: str | None = None
        self.notes: list = []                       # why an earlier result was discarded

    @property
    def running(self) -> bool:
        return self.job is not None and self.job.running

    def discard(self, note: str | None = None):
        if self.job is not None and self.job.running:
            self.job.cancel.set(); self.job.cancelled = True
        self.job = None; self.result = None; self.binding = None; self.job_id = None
        if note:
            self.notes.append(note)


# ------------------------------------------------------------------ the service
class MeasurementService:
    def __init__(self, visitor_service, *, gate=None, runner=None, limits=None, coordinator=None):
        """``visitor_service``: the ``VisitorCalibrationService`` (its stores).  ``gate``, ``runner``: TEST SEAMS (see the module docstring).  ``limits`` and ``coordinator``: server-side
        configuration (default: the backend's trusted defaults and the application-wide shared coordinator)."""
        self._svc = visitor_service
        self._gate = gate
        self._runner = runner
        self._limits = limits
        self._coordinator = coordinator
        self._start_lock = threading.Lock()

    # ---- the gate
    def gate_open(self, session=None) -> bool:
        """True only when the code constants allow processing of the ACTIVE wall (or a test injected a gate).  Wall A reads ``WALL_A_PROCESSING_ENABLED`` and Wall B reads
        ``WALL_B_PROCESSING_ENABLED`` (independent of each other); a call without a session reads ``ROOM_PROCESSING_ENABLED``.  All are read at call time."""
        if self._gate is not None:
            return self._gate() is True
        import f8_visitor_service as S
        wall = getattr(session, "active_wall", None) if session is not None else None
        if wall == "A":
            return S.WALL_A_PROCESSING_ENABLED is True
        if wall == "B":
            return S.WALL_B_PROCESSING_ENABLED is True
        return S.ROOM_PROCESSING_ENABLED is True

    # ---- trusted values
    def trusted_limits(self):
        return self._limits if self._limits is not None else _backend().ReconstructionLimits()      # STRICT original behaviour (min_usable_sets=None): every selected set needs all six refined lines before pose estimation

    def trusted_coordinator(self):
        if self._coordinator is not None:
            return self._coordinator
        import resource_budget as RB
        return RB.app_coordinator()

    def resource_policy(self) -> dict:
        """The processing limits that are ACTUALLY enforced, for display (read-only; nothing here sets or changes a limit): the worker wall-clock timeout, the worker CPU-time limit and the
        free memory a measurement needs before it is admitted, for two viewpoints (Wall A's method) and for three (the limits ``three_view_runner.run`` applies)."""
        import three_view_runner as TVRUN
        RR = _backend(); lim = self.trusted_limits()
        return dict(timeout_s=float(lim.timeout_s), cpu_limit_s=float(lim.cpu_limit_s), min_memory_mb_2=float(min(lim.min_tree_rss_mb, lim.max_tree_rss_mb)),
                    min_memory_mb_3=float(TVRUN.three_view_limits(RR, lim).min_tree_rss_mb))

    # ---- readiness
    def readiness(self, session) -> MeasurementReadiness:
        if session.n_viewpoints == 3:
            return self._readiness3(session)
        RR = _backend()
        svc = self._svc
        limits = self.trusted_limits()
        check_limits = RR.ReconstructionLimits(**{**limits.__dict__, "verify_video_hashes": False})        # display only: the full hash check happens at start
        items, problems_by_item, codes_by_item = {}, {k: [] for k in ITEM_LABEL}, {k: [] for k in ITEM_LABEL}
        summary: dict = {}
        ctx = svc.calibration_context(session)
        st = svc.frame_state(session) if ctx is not None else dict(state="no_calibration", workflow=None, problems=["There is no accepted calibration."])
        wf = st["workflow"] if st["state"] == "ok" else None
        sources = svc.room_sources(session) if ctx is not None else {s: None for s in RV.ROOM_UPLOAD_SLOTS}
        status = "current"
        if ctx is None:
            problems_by_item["calibration"].append("There is no accepted calibration."); status = "not_accepted"
        else:
            c = session.contract
            summary.update(calibration_id=RF.calibration_id(c), camera_order=list(RF.CAMERA_ORDER), calibration_size_wh=list(ctx["image_size_wh"]), baseline_mm=float(ctx["baseline_mm"]),
                           quality=ctx["quality"], capture_mode=dict(ctx.get("capture_mode") or {}))
            summary["videos"] = {s: (None if v is None else dict(label=RV.SLOT_INFO[s][2], size_wh=list(v.wh), frames=int(v.frame_count), sha256=v.sha256)) for s, v in sources.items()}
        missing = [RV.SLOT_INFO[s][2] for s, v in sources.items() if v is None]
        if ctx is not None and missing:
            problems_by_item["videos"] += [f"{m} is missing or has expired." for m in missing]; status = "expired"
        elif ctx is not None and st["state"] == "not_ready":
            problems_by_item["videos"] += list(st["problems"])
        warns = []
        if wf is not None:
            summary["n_frame_sets"] = wf.n_sets
            summary["annotations_accepted"] = sum(1 for i in range(wf.n_sets) for n in RF.REQUIRED_LINES if wf.annotation_status(i, n) == "accepted")
            summary["annotations_required"] = wf.n_sets * len(RF.REQUIRED_LINES)
            summary["seeds_marked"] = sum(1 for n in RF.REQUIRED_LINES if len(wf.seed_points(n)) == 2)
            summary["seeds_required"] = len(RF.REQUIRED_LINES)
            summary["frame_sets_usable"] = len(wf.usable_sets())
            summary["frame_sets_skipped"] = [i + 1 for i in range(wf.n_sets) if not wf.set_usable(i)]
            for i in range(wf.n_sets):
                for n in RF.REQUIRED_LINES:
                    s = wf.ann.get((i, n))
                    if s is not None and s.accepted and s.acknowledged_codes:
                        warns.append(f"Frame set {i + 1}, {n}: refined with warnings (" + ", ".join(str(c) for c in s.acknowledged_codes) + ")" + ("; recorded automatically." if s.auto else "; you acknowledged them."))
            # the backend's own input check (no hashing here, no worker, nothing is staged)
            prep = RR.prepare(wf, session.room_owner, session.contract, session.f6.job.cal, sources, check_limits)
            for f in prep.failures:
                k = _item_of(f)
                problems_by_item[k].append(f.get("visitor_message") or RVW.STATE_TEXT[RVW.DONE_FAILED][2]); codes_by_item[k].append(str(f.get("code")))
            codes = {c for v in codes_by_item.values() for c in v}
            if "UPLOAD_EXPIRED" in codes:
                status = "expired"
            elif "VIDEO_REPLACED" in codes:
                status = "replaced"
            elif "CALIBRATION_CHANGED" in codes:
                status = "calibration_changed"
            elif prep.failures:
                status = "incomplete"
            ready_backend = prep.request is not None
        else:
            ready_backend = False
            if ctx is not None and not missing and st["state"] != "not_ready":
                problems_by_item["frames"].append("Select five frame sets.")
                if status == "current":
                    status = "incomplete"
        if wf is None and ctx is not None:
            if not problems_by_item["frames"]:
                problems_by_item["frames"].append("Your uploads are not all valid yet." if missing or st["state"] == "not_ready" else "Select five frame sets.")
            problems_by_item["annotations"].append("Mark the wall boundaries after choosing the frame sets.")
            if status == "current":
                status = "incomplete"
        step_of = {"calibration": "home", "videos": "room", "frames": "frames", "annotations": "frames"}
        for key in ITEM_LABEL:
            probs = tuple(dict.fromkeys(problems_by_item[key]))
            items[key] = ReadinessItem(key, ITEM_LABEL[key], not probs, self._detail(key, summary), None if not probs else step_of[key], probs, tuple(dict.fromkeys(codes_by_item[key])))
        ready = bool(ctx is not None and wf is not None and ready_backend and all(i.ok for i in items.values()))
        b = binding_of(session.room_owner, session.contract, wf, sources) if ctx is not None else None
        return MeasurementReadiness(ready, tuple(items.values()), summary, tuple(warns), status if not ready else "current", b, self.gate_open(session))

    def _readiness3(self, session) -> MeasurementReadiness:
        """Readiness of a three-viewpoint Wall B: the verdict IS the three-viewpoint input check (``three_view_runner.prepare``), nothing is hashed or staged here."""
        import three_view_runner as TVRUN
        RR = _backend(); svc = self._svc; limits = self.trusted_limits()
        check_limits = RR.ReconstructionLimits(**{**limits.__dict__, "verify_video_hashes": False})
        items_p, items_c = {k: [] for k in ITEM_LABEL_3}, {k: [] for k in ITEM_LABEL_3}; summary: dict = {}
        ctx = svc.calibration_context(session)
        st = svc.frame_state(session) if ctx is not None else dict(state="no_calibration", workflow=None, problems=["There is no accepted calibration."])
        wf = st["workflow"] if st["state"] == "ok" else None
        sources = svc.room_sources(session) if ctx is not None else {s: None for s in RV.ROOM_UPLOAD_SLOTS_3}
        status = "current"
        if ctx is None:
            items_p["calibration"].append("There is no accepted calibration."); status = "not_accepted"
        else:
            c = session.contract
            summary.update(calibration_id=RF.calibration_id(c), camera_order=list(RF.CAMERA_ORDER), calibration_size_wh=list(ctx["image_size_wh"]), baseline_mm=float(ctx["baseline_mm"]), quality=ctx["quality"],
                           capture_mode=dict(ctx.get("capture_mode") or {}), method="three_viewpoint_wall_b", n_viewpoints=3)
            summary["videos"] = {s: (None if v is None else dict(label=RV.SLOT_INFO[s][2], size_wh=list(v.wh), frames=int(v.frame_count), sha256=v.sha256)) for s, v in sources.items()}
        missing = [RV.SLOT_INFO[s][2] for s, v in sources.items() if v is None]
        if ctx is not None and missing:
            items_p["videos"] += [f"{m} is missing or has expired." for m in missing]; status = "expired"
        elif ctx is not None and st["state"] == "not_ready":
            items_p["videos"] += list(st["problems"])
        ready_backend = False
        if wf is not None:
            rd = wf.readiness(); summary.update(n_frame_sets=wf.n_sets, frame_sets_usable=len(rd["usable_sets"]), frame_sets_skipped=[i + 1 for i in range(wf.n_sets) if i not in rd["usable_sets"]],
                                                verticals_marked=sum(1 for i in range(wf.n_sets) for n in TVW.REQUIRED_LINES if wf.line_status(i, n) == "marked"), verticals_required=len(TVW.REQUIRED_LINES) * wf.n_sets)
            prep = TVRUN.prepare(RR, wf, session.room_owner, session.contract, session.f6.job.cal, sources, check_limits)
            for f in prep.failures:
                k = _item_of(f); k = k if k in ITEM_LABEL_3 else "frames"
                items_p[k].append(f.get("visitor_message") or "An input is not usable."); items_c[k].append(str(f.get("code")))
            codes = {c for v in items_c.values() for c in v}
            status = "expired" if "UPLOAD_EXPIRED" in codes else "replaced" if "VIDEO_REPLACED" in codes else "calibration_changed" if "CALIBRATION_CHANGED" in codes else "incomplete" if prep.failures else status
            ready_backend = prep.request is not None
        elif ctx is not None and not missing and st["state"] != "not_ready":
            items_p["frames"].append("Choose the frames and mark the wall corners."); status = "incomplete" if status == "current" else status
        if wf is None and ctx is not None and not items_p["frames"]:
            items_p["frames"].append("Your uploads are not all valid yet." if missing or st["state"] == "not_ready" else "Choose the frames and mark the wall corners.")
            status = "incomplete" if status == "current" else status
        step_of = {"calibration": "home", "videos": "room", "frames": "frames", "annotations": "frames"}
        items = []
        for key in ITEM_LABEL_3:
            probs = tuple(dict.fromkeys(items_p[key]))
            detail = {"calibration": (f"{summary['calibration_size_wh'][0]}×{summary['calibration_size_wh'][1]} px · baseline {summary['baseline_mm']:g} mm") if "calibration_size_wh" in summary else "not accepted",
                      "videos": f"{sum(1 for v in (summary.get('videos') or {}).values() if v)} of 6 uploaded", "frames": f"{summary.get('frame_sets_usable', 0)} of {summary.get('n_frame_sets', 0)} frame set(s) usable",
                      "annotations": f"{summary.get('verticals_marked', 0)} vertical corner lines marked"}[key]
            items.append(ReadinessItem(key, ITEM_LABEL_3[key], not probs, detail, None if not probs else step_of[key], probs, tuple(dict.fromkeys(items_c[key]))))
        ready = bool(ctx is not None and wf is not None and ready_backend and all(i.ok for i in items))
        b = binding_of(session.room_owner, session.contract, wf, sources) if ctx is not None else None
        return MeasurementReadiness(ready, tuple(items), summary, (), status if not ready else "current", b, self.gate_open(session))

    @staticmethod
    def _detail(key: str, s: dict) -> str:
        if key == "calibration":
            return (f"{s['calibration_size_wh'][0]}×{s['calibration_size_wh'][1]} px · baseline {s['baseline_mm']:g} mm") if "calibration_size_wh" in s else "not accepted"
        if key == "videos":
            return f"{sum(1 for v in (s.get('videos') or {}).values() if v)} of 4 uploaded"
        if key == "frames":
            return f"{s.get('n_frame_sets', 0)} frame set(s) chosen"
        return f"{s.get('seeds_marked', 0)} of {s.get('seeds_required', 6)} boundaries marked · {s.get('frame_sets_usable', 0)} usable frame set(s)"

    # ---- start / cancel / result
    def start(self, session) -> str:
        """Start ONE measurement for this visitor.  The gate is checked first; while it is closed nothing is validated, staged, started or recorded."""
        if not self.gate_open(session):
            raise MeasurementError(RVW.GATE_MESSAGE)
        rec: MeasurementRecord = session.measurement
        with self._start_lock:
            if rec.running:
                raise MeasurementError("A measurement is already running for this session.")
            rd = self.readiness(session)
            if not rd.ready or rd.binding is None:
                raise MeasurementError("Your inputs are not complete or are no longer valid. " + (" ".join(rd.first_problem.problems) if rd.first_problem else ""))
            if rec.result is not None and rec.binding is not None and rec.binding.get("key") == rd.binding["key"] and RVW.result_state(rec.result) in (RVW.DONE_OK, RVW.DONE_WARN):
                raise MeasurementError("These exact inputs were already measured. Change the calibration, a video, a frame or a boundary to measure again.")
            RR = _backend()
            limits = self.trusted_limits()
            ctx = RunContext(job_id=secrets.token_hex(8), workflow=session.frames, owner=session.room_owner, contract=session.contract, calibration=session.f6.job.cal,
                             sources=self._svc.room_sources(session), limits=limits, coordinator=self.trusted_coordinator(),
                             retention=RR.StoreRetention(self._svc.room_store, session.room_owner, slots=RV.ROOM_UPLOAD_SLOTS_3 if session.n_viewpoints == 3 else None), cancel=threading.Event())
            job = MeasurementJob(ctx.job_id, rd.binding, time.monotonic(), ctx.cancel)
            rec.job, rec.result, rec.binding, rec.job_id = job, None, None, job.job_id
            threading.Thread(target=self._run, args=(rec, job, ctx), daemon=True, name=f"f8m-{job.job_id}").start()
        return job.job_id

    def _run(self, rec: MeasurementRecord, job: MeasurementJob, ctx: RunContext):
        RR = _backend()
        result = None
        try:
            job.stage = "processing"
            runner = self._runner if self._runner is not None else self._default_runner
            result = runner(ctx)
            if not (isinstance(result, dict) and result.get("schema") == RR.SCHEMA_RESULT and result.get("status") in ("SUCCESS", "QUESTIONABLE", "FAILED")):
                result = None
        except Exception as ex:                                                      # nothing may escape as an untyped exception
            result = None
            job.stage = f"internal error ({type(ex).__name__})"
        if result is None:
            result = RR.assemble_result(None, failures=[RR.failure("INTERNAL_ERROR", "the runner returned no valid result", stage="internal")])
        try:
            ctx.retention.release()                                                  # idempotent; the backend already released it on its own path
        except Exception:
            pass
        with job._lock:
            job.cleanup = dict(retention_released=getattr(ctx.retention, "lease_id", None) is None, cancel_requested=ctx.cancel.is_set(), thread_finished=True)
            job.result = result; job.done = True; job.stage = "finished"
        if rec.job is job:                                                           # a discarded job (inputs changed, session reset) can never publish a result
            rec.result, rec.binding = result, job.binding

    @staticmethod
    def _default_runner(ctx: RunContext) -> dict:
        """The one place the real backend is called.  Reached only through ``start`` (after the gate)."""
        if isinstance(ctx.workflow, TVW.ThreeViewWorkflow):                          # three-viewpoint Wall B: its own sandboxed worker under the SAME supervision and limits
            import three_view_runner as TVRUN
            return TVRUN.run(_backend(), ctx.workflow, ctx.owner, ctx.contract, ctx.calibration, ctx.sources, ctx.limits, cancel=ctx.cancel, coordinator=ctx.coordinator, retention=ctx.retention)
        return _backend().run_visitor_reconstruction(ctx.workflow, ctx.owner, ctx.contract, ctx.calibration, ctx.sources, ctx.limits, cancel=ctx.cancel, coordinator=ctx.coordinator,
                                                     retention=ctx.retention)

    def cancel(self, session) -> bool:
        rec: MeasurementRecord = session.measurement
        if rec.running:
            rec.job.cancel.set(); rec.job.cancelled = True
            return True
        return False

    def current(self, session) -> dict | None:
        """The result of THIS visitor for exactly the CURRENT inputs, or None.  A result for any other inputs is discarded (with a note) and never returned."""
        rec: MeasurementRecord = session.measurement
        if rec.result is None:
            return None
        ctx = self._svc.calibration_context(session)
        b = None
        if ctx is not None and session.frames is not None and not session.frames.closed:
            b = binding_of(session.room_owner, session.contract, session.frames, self._svc.room_sources(session))
        if b is None or rec.binding is None or b["key"] != rec.binding.get("key") or b["owner"] != RF.owner_digest(session.room_owner):
            rec.discard("Your calibration, a video, a frame or a boundary changed, so the previous measurement result was discarded.")
            return None
        return rec.result

    def invalidate(self, session, note: str | None = None):
        session.measurement.discard(note)
