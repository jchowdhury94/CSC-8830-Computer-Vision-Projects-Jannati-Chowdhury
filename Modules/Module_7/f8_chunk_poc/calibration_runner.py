"""Stage F8.5: supervisor for a dual-video calibration job (NO Streamlit import; runs in the app process, does the heavy work in a WORKER process).

    acquire lease (both videos protected)  ->  explicit safety validation of the two videos  ->  worker process (sequential LEFT, RIGHT detection, solve,
    assess)  ->  heartbeat / progress / resource sampling / timeout / cancel / memory guard  ->  result  ->  release lease and delete both videos.

Nothing from F7 is re-implemented or changed: the worker imports the existing module and calls its functions. The F7 *visitor-upload* limits
(200 MB, 120 s) are not applied because they are about browser uploads; instead ``RunnerLimits`` states explicit limits for full-length recordings, and
every check is reported in the result (``validation``), never silently skipped.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from chunk_store import ChunkStore, UploadError
import resource_budget as RB                                         # F8.31C-1C: one memory budget / worker slot shared with the room-reconstruction backend
from monitor import ResourceMonitor, probe_environment, read_cgroup
from result_store import ResultStore, build_result_document, new_code
from video_check import VideoSafetyLimits, inspect_video

HERE = Path(__file__).resolve().parent
WORKER = HERE / "calibration_worker.py"
DEFAULT_F7_DIR = HERE.parent / "visitor_workflow"                      # Modules/Module_7/visitor_workflow (holds f7_calibration.py)
MiB = 1024 * 1024
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")

STAGE_ORDER = ("validate", "start", "import", "verify", "detect_left", "detect_right", "verify_after", "solve", "assess", "finish")
MODES = ("calibrate", "benchmark")
STALE_SAMPLE_S = 3.0                          # F8.20c: a memory sample older than this (monotonic clock) is "not measured" -- a frozen monitor can never keep satisfying the guard
CALIBRATION_DESIRED_MB = 2048.0               # F8.31C-1C: ceiling a calibration worker asks the shared budget for when RunnerLimits.max_tree_rss_mb is not set (planning value; the benchmark worker measured about 770 MB locally; not a Cloud measurement)
CALIBRATION_MIN_MB = 1024.0                   # F8.31C-1C: a calibration is not started with a smaller grant (resource-safety planning value)
BLIND_POLLS_MAX = 20                          # consecutive polls (poll_s each) without a usable RSS reading before a configured RSS limit is declared unenforceable


def _env_num(env, name, default, lo, hi):
    """Number from an environment variable, clamped to [lo, hi]; the default when missing, invalid or not finite."""
    try:
        v = float(env[name])
    except (KeyError, ValueError, TypeError):
        return default
    return default if v != v or v in (float("inf"), float("-inf")) else min(max(v, lo), hi)


TREE_RSS_MB_RANGE = (256.0, 65536.0)


def _env_limit_mb(env, name):
    """F8.20a: optional process-tree memory limit in MB from the environment. Unset, invalid, not finite or outside TREE_RSS_MB_RANGE -> None (no limit; never a silent clamp)."""
    try:
        v = float(str(env[name]).strip())
    except (KeyError, ValueError):
        return None
    return v if math.isfinite(v) and TREE_RSS_MB_RANGE[0] <= v <= TREE_RSS_MB_RANGE[1] else None


@dataclass(frozen=True)
class RunnerLimits:
    """Explicit safety limits for FULL-LENGTH calibration recordings (not the F7 browser-upload limits)."""
    timeout_s: float = 3600.0                  # wall clock for the worker; killed beyond this
    heartbeat_s: float = 5.0                   # lease heartbeat period
    poll_s: float = 0.25
    kill_grace_s: float = 3.0                  # SIGTERM first, SIGKILL after this
    min_duration_s: float = 10.0               # same minimum as F7: a pose sweep needs time
    max_duration_s: float = 900.0              # 15 min per video (the real recordings are about 3.6 min)
    max_frames: int = 40_000
    max_fps_difference: float = 1.0            # across the two videos (same as F7's warning threshold)
    max_tree_rss_mb: float | None = None       # kill if parent+worker(+children) RSS exceeds this; from_env: F8_MAX_TREE_RSS_MB (256-65536). The only guard on macOS (no cgroup)
    max_cgroup_fraction: float = 0.90          # kill if container ANONYMOUS memory (not page cache) exceeds this fraction of the cgroup limit
    result_max_bytes: int = 32 * MiB
    inspect_timeout_s: float = 60.0            # per-video sandboxed inspection
    allow_full: bool = False                   # may a FULL calibration be started at all? Safe default OFF; from_env() enables it only with F8_ENABLE_FULL_CALIBRATION=1
    benchmark_timeout_s: float = 300.0         # wall clock for a benchmark worker
    benchmark_frames: int = 600                # frames per video the benchmark reads (every STRIDE-th gets a detection attempt)
    benchmark_frames_cap: int = 3000           # hard ceiling whatever is asked for
    require_memory_guard: bool = False         # F8.10: refuse a FULL run when no container memory limit (or tree-RSS limit) is known, i.e. the guard could not act; from_env: F8_REQUIRE_MEMORY_GUARD=1
    start_headroom_fraction: float = 0.60      # F8.10: a full run (or a retry after a memory stop) does not start while container anonymous memory is already above this fraction of the limit

    @classmethod
    def from_env(cls, env=None):
        """Deployment configuration. F8_CALIB_TIMEOUT_S (60-14400, default 3600), F8_BENCH_TIMEOUT_S (10-900), F8_BENCH_FRAMES (30-cap),
        F8_ENABLE_FULL_CALIBRATION (only the exact value '1' enables full runs). Invalid values fall back to the defaults."""
        env = os.environ if env is None else env
        d = cls()
        return cls(timeout_s=_env_num(env, "F8_CALIB_TIMEOUT_S", d.timeout_s, 60.0, 14400.0), benchmark_timeout_s=_env_num(env, "F8_BENCH_TIMEOUT_S", d.benchmark_timeout_s, 10.0, 900.0),
                   benchmark_frames=int(_env_num(env, "F8_BENCH_FRAMES", d.benchmark_frames, 30, d.benchmark_frames_cap)), allow_full=str(env.get("F8_ENABLE_FULL_CALIBRATION", "")).strip() == "1",
                   require_memory_guard=str(env.get("F8_REQUIRE_MEMORY_GUARD", "")).strip() == "1", max_tree_rss_mb=_env_limit_mb(env, "F8_MAX_TREE_RSS_MB"))


def lease_margin_s(limits: "RunnerLimits") -> float:
    """Lease time the job needs besides the worker's own timeout: the sandboxed inspection of both videos (2 x inspect_timeout_s, worst case) happens after
    the lease starts but before the timeout clock does, plus the SIGTERM->SIGKILL grace and a fixed allowance for start-up and clean-up."""
    return 2.0 * limits.inspect_timeout_s + limits.kill_grace_s + 30.0


class RunnerError(Exception):
    pass


# ---- F8.10: jobs that are still running can be found (status, cancel) by their recovery code from ANOTHER browser session; the supervisor thread itself never
# depends on a session. The registry only holds live runners and forgets a job when it is done.
_ACTIVE: dict = {}
_ACTIVE_LOCK = threading.Lock()


def active_runner(code):
    with _ACTIVE_LOCK:
        return _ACTIVE.get(code)


def cancel_by_code(code) -> bool:
    r = active_runner(code)
    if r is None:
        return False
    r.cancel()
    return True


def _sane(v):
    """A usable measurement: a finite, non-negative real number; anything else (None, NaN, inf, negative, text, bool) is treated as 'not measured'."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 else None


def _fresh_tree_rss(sample, now=None, max_age_s=None):
    """F8.20c: combined parent+worker RSS from a monitor sample, or None if it is missing/invalid OR older than ``max_age_s`` (``mono_s`` is the monitor's monotonic stamp; no stamp = unusable)."""
    if not isinstance(sample, dict):
        return None
    max_age_s = STALE_SAMPLE_S if max_age_s is None else max_age_s
    t = _sane(sample.get("mono_s"))
    if t is None or (time.monotonic() if now is None else now) - t > max_age_s:
        return None
    return _sane(sample.get("combined_rss_mb"))


def guard_memory(cg) -> dict:
    """What the container memory guard judges, from a cgroup reading, using only sane values. Page cache is reclaimable and is NOT counted:
    anonymous memory (+ shmem) when known; else current - file cache; else current (conservative: includes cache). ``limit_mb`` None = no usable limit."""
    cg = cg if isinstance(cg, dict) else {}
    lim = _sane(cg.get("memory_max_mb")); lim = lim if lim and lim > 0 else None
    anon, shm, cur, fil = (_sane(cg.get(k)) for k in ("anon_mb", "shmem_mb", "memory_current_mb", "file_mb"))
    if anon is not None:
        used, basis = anon + (shm or 0.0), "anonymous+shmem"
    elif cur is not None and fil is not None and fil <= cur:
        used, basis = cur - fil + (shm or 0.0), "current minus page cache"
    elif cur is not None:
        used, basis = cur, "current (includes page cache)"
    else:
        used, basis = None, "unknown"
    return dict(used_mb=used, limit_mb=lim, basis=basis)


def _scrubbed_env():
    return {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}


def validate_videos(left_path, right_path, limits: RunnerLimits) -> dict:
    """Explicit checks for the two recordings, via the F8.1 sandboxed inspection. Returns dict(ok, errors, warnings, videos, limits_applied)."""
    vlim = VideoSafetyLimits(timeout_s=limits.inspect_timeout_s)
    rep = dict(ok=True, errors=[], warnings=[], videos={}, limits_applied=dict(asdict(limits), video_safety=asdict(vlim)))
    err = lambda code, msg: rep["errors"].append(dict(code=code, message=msg))
    for side, path in (("left", left_path), ("right", right_path)):
        r = inspect_video(path, max_frames=3, limits=vlim)
        rep["videos"][side] = {k: r.get(k) for k in ("ok", "container", "codec", "width", "height", "fps", "frame_count", "duration_s", "frames_read", "worker_s", "stage")}
        if not r["ok"]:
            for e in r["errors"]:
                err(f"{side}_{e['code']}", f"{side.upper()} video: {e['message']}")
            continue
        if r["duration_s"] is None or r["fps"] is None or r["frame_count"] is None:
            err(f"{side}_no_metadata", f"{side.upper()} video: the file does not report its frame rate and length, so it cannot be checked.")
        else:
            if r["duration_s"] < limits.min_duration_s:
                err(f"{side}_too_short", f"{side.upper()} video: {r['duration_s']:.1f} s is shorter than the {limits.min_duration_s:g} s minimum.")
            if r["duration_s"] > limits.max_duration_s:
                err(f"{side}_too_long", f"{side.upper()} video: {r['duration_s']:.0f} s is longer than the {limits.max_duration_s:g} s limit.")
            if r["frame_count"] > limits.max_frames:
                err(f"{side}_too_many_frames", f"{side.upper()} video: {r['frame_count']} frames exceed the {limits.max_frames} frame limit.")
    L, R = rep["videos"]["left"], rep["videos"]["right"]
    if L.get("width") and R.get("width"):
        if (L["width"], L["height"]) != (R["width"], R["height"]):
            err("size_mismatch", f"The two videos have different resolutions ({L['width']}×{L['height']} vs {R['width']}×{R['height']}). Both cameras must record at the same resolution.")
    if L.get("fps") and R.get("fps") and abs(L["fps"] - R["fps"]) > limits.max_fps_difference:
        err("fps_mismatch", f"The two videos have different frame rates ({L['fps']:.2f} vs {R['fps']:.2f} fps), so they cannot be paired reliably.")
    rep["ok"] = not rep["errors"]
    return rep


class CalibrationJobRunner:
    """One dual-video calibration. ``start()`` returns immediately; poll ``status()``; ``cancel()`` stops it; ``join()`` waits."""

    def __init__(self, store: ChunkStore, owner: str, settings: dict, limits: RunnerLimits = RunnerLimits(), f7_dir=None, worker_cmd=None,
                 monitor_interval: float = 1.0, dump_detections: bool = True, work_dir=None, mode: str = "calibrate",
                 benchmark_frames: int | None = None, results: ResultStore | None = None, job_code: str | None = None, cgroup_root: str = "/sys/fs/cgroup",
                 proc_root: str = "/proc", coordinator: "RB.ResourceCoordinator | None" = None):
        if mode not in MODES:
            raise RunnerError("unknown mode")
        self.mode = mode
        self.store, self.owner, self.settings, self.limits = store, owner, dict(settings), limits
        self.bench_frames = int(min(max(int(benchmark_frames or limits.benchmark_frames), 1), limits.benchmark_frames_cap))
        # the timeout actually applied: the benchmark has its own short one; a full run can never outlive the store's hard lease limit
        self.lease_margin_s = lease_margin_s(limits)
        self.timeout_s = float(limits.benchmark_timeout_s if mode == "benchmark" else max(1.0, min(limits.timeout_s, store.limits.lease_max_s - self.lease_margin_s)))
        self.timeout_info = dict(requested_s=limits.timeout_s if mode == "calibrate" else limits.benchmark_timeout_s, effective_s=self.timeout_s, lease_max_s=store.limits.lease_max_s,
                                 lease_margin_s=self.lease_margin_s, clipped_by_lease=mode == "calibrate" and self.timeout_s < limits.timeout_s)
        self.results = results if results is not None else ResultStore()
        self.job_code = job_code or new_code()
        self.cgroup_root, self.proc_root = cgroup_root, proc_root
        self._coord = coordinator                                       # None: the application-wide coordinator (resolved at start, so a test or operator may replace it)
        self._reservation = None; self._tree_limit_mb = None; self._budget_info = None
        self._guard_info = dict(active=None, basis=None, limit_mb=None, threshold_mb=None, peak_used_mb=None, checks=0)
        self._tree_info = dict(active=False, limit_mb=None, peak_rss_mb=None, checks=0, fresh_checks=0)   # F8.25: process-tree RSS guard reporting
        self._pub_t = 0.0; self._persisted = None; self._delete_rejected = False
        self._last_prog = None; self._blind_polls = 0
        self.f7_dir = Path(f7_dir) if f7_dir else DEFAULT_F7_DIR
        self.worker_cmd, self.monitor_interval, self.dump_detections = worker_cmd, monitor_interval, dump_detections
        self.work_dir = Path(work_dir) if work_dir else None
        self._lock = threading.Lock(); self._cancel = threading.Event(); self._thread = None
        self._status = dict(state="idle", stage="idle", fraction=0.0, message="", started=None, elapsed_s=0.0, outcome=None, monitor=None)
        self.result: dict | None = None
        self._proc = None; self._lease = None; self._job_dir = None; self._monitor = None

    # ---- public
    def start(self):
        if self._thread is not None:
            raise RunnerError("this runner has already been started")
        self._set(state="starting", message="Preparing the calibration…", started=time.time())
        with _ACTIVE_LOCK:
            _ACTIVE[self.job_code] = self
        self._thread = threading.Thread(target=self._run, daemon=True, name="f8-calibration-supervisor"); self._thread.start()

    def cancel(self):
        self._cancel.set()
        lease = self._lease
        if lease:
            try:
                self.store.request_cancel(self.owner, lease)
            except UploadError:
                pass

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
        return self.result

    def status(self) -> dict:
        with self._lock:
            s = dict(self._status)
        if s["started"] and s["state"] in ("starting", "validating", "running", "cleaning"):
            s["elapsed_s"] = round(time.time() - s["started"], 1)
        return s

    @property
    def done(self) -> bool:
        return self.status()["state"] == "done"

    # ---- internals
    def _set(self, **kw):
        with self._lock:
            changed = "state" in kw and kw["state"] != self._status.get("state")
            self._status.update(kw)
        self._publish(force=changed)

    def _publish(self, force=False):
        """Small status record for OTHER sessions (throttled; best effort)."""
        now = time.monotonic()
        if not force and now - self._pub_t < 1.5:
            return
        self._pub_t = now
        s = self.status(); m = s.get("monitor") or {}
        self.results.write_status(self.job_code, dict(mode=self.mode, state=s["state"], stage=s["stage"], fraction=s["fraction"], message=str(s["message"])[:200], elapsed_s=s["elapsed_s"],
                                                      outcome=s["outcome"], timeout_s=self.timeout_s, started_unix=s["started"], result_saved=bool((self._persisted or {}).get("saved")),
                                                      worker_rss_mb=m.get("worker_rss_mb"), parent_rss_mb=m.get("parent_rss_mb")))

    def _disk(self):
        try:
            return self.store.disk_usage_bytes() / MiB, shutil.disk_usage(self.store.root).free / MiB
        except OSError:
            return None, None

    def _finish(self, outcome, result=None, validation=None, error=None):
        res = dict(result or {})
        res.setdefault("ok", False)
        res.update(outcome=outcome, validation=validation, mode=self.mode)
        if error:
            res["error"] = error
        self.result = res
        self._set(state="cleaning", outcome=outcome, message=error or (("Benchmark finished." if self.mode == "benchmark" else "Calibration finished.") if outcome == "completed" else outcome), fraction=1.0 if outcome == "completed" else self._status["fraction"])

    def _spawn(self, spec_path, job_dir):
        if self.worker_cmd:
            argv = list(self.worker_cmd(spec_path, job_dir))
        else:
            argv = [sys.executable, "-I", str(WORKER), str(spec_path)]
        err = open(Path(job_dir) / "worker.stderr", "wb")
        try:
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err, env=_scrubbed_env(), start_new_session=True)
        finally:
            err.close()

    def _terminate(self, proc):
        """Stop the worker and everything it started (own process group): SIGTERM, then SIGKILL after a grace period. Always reaps it."""
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=self.limits.kill_grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()

    def _read_cg(self):
        try:
            return read_cgroup() if (self.cgroup_root, self.proc_root) == ("/sys/fs/cgroup", "/proc") else read_cgroup(self.cgroup_root, self.proc_root)
        except Exception:                                              # an unreadable cgroup must never crash the supervisor: it just means "not measured"
            return {}

    def _guard(self, sample) -> str | None:
        """Memory guard, evaluated every poll. Only sane measurements are used (see ``guard_memory``); a missing or invalid one means 'cannot judge', never a kill."""
        lim = self.limits
        tree = _fresh_tree_rss(sample)
        tree_limit = self._tree_limit_mb if self._tree_limit_mb else lim.max_tree_rss_mb      # F8.31C-1C: the GRANT of the shared budget (never above max_tree_rss_mb when that is set)
        if tree_limit and tree_limit > 0:
            ti = self._tree_info; ti["checks"] += 1; ti["limit_mb"] = tree_limit; ti["active"] = tree is not None   # active only while a fresh reading exists
            if tree is not None:
                ti["fresh_checks"] += 1; ti["peak_rss_mb"] = max(ti["peak_rss_mb"] or 0.0, tree)
            if tree is None:                                           # F8.20a: a configured RSS limit that cannot be measured is not a guard: stop after ~5 s of blindness
                self._blind_polls += 1
                if self._blind_polls >= BLIND_POLLS_MAX:
                    return "memory guard: the process memory could not be measured, so the configured limit cannot be enforced"
            else:
                self._blind_polls = 0
                if tree > tree_limit:
                    return f"memory guard: parent+worker RSS {tree:.0f} MB exceeded {tree_limit:.0f} MB"
        g = guard_memory(self._read_cg())
        frac = lim.max_cgroup_fraction if isinstance(lim.max_cgroup_fraction, (int, float)) and 0 < lim.max_cgroup_fraction <= 1 else 0.90
        info = self._guard_info; info["checks"] += 1; info["basis"] = g["basis"]; info["limit_mb"] = g["limit_mb"]
        info["active"] = bool(g["limit_mb"] and g["used_mb"] is not None); info["threshold_mb"] = g["limit_mb"] * frac if g["limit_mb"] else None
        if g["used_mb"] is not None:
            info["peak_used_mb"] = max(info["peak_used_mb"] or 0.0, g["used_mb"])
        if g["limit_mb"] and g["used_mb"] is not None and g["used_mb"] > frac * g["limit_mb"]:
            return f"memory guard: container memory {g['used_mb']:.0f} MB ({g['basis']}) exceeded {frac:.0%} of its {g['limit_mb']:.0f} MB limit"
        return None

    def _reserve(self) -> str | None:
        """F8.31C-1C: ask the application-wide coordinator for this worker's memory.  Returns a visitor-readable refusal, or None when granted.  Never raises."""
        lim = self.limits
        desired = lim.max_tree_rss_mb if lim.max_tree_rss_mb else CALIBRATION_DESIRED_MB
        try:
            coord = self._coord if self._coord is not None else RB.app_coordinator()
            adm = coord.reserve("calibration", self.owner, desired, min(CALIBRATION_MIN_MB, desired))
        except Exception as ex:                                              # a broken coordinator is a refusal, never an unbudgeted run
            self._budget_info = dict(outcome="UNAVAILABLE", error=type(ex).__name__)
            return "The server's memory budget could not be consulted, so the job was not started. Try again shortly."
        d = adm.detail; b = d.get("budget") or {}
        self._budget_info = dict(outcome=adm.outcome, requested_mb=desired, granted_mb=adm.granted_mb, total_mb=b.get("total_mb"), usable_mb=b.get("usable_mb"), reserve_mb=b.get("reserve_mb"),
                                 source=b.get("source"), limitation=b.get("limitation"), parent_rss_mb=d.get("parent_rss_mb"), reserved_mb_before=d.get("reserved_mb"),
                                 available_mb_before=d.get("available_mb"), reason=d.get("reason"), cross_process=d.get("cross_process"))
        if adm.ok:
            self._reservation = adm.reservation; self._tree_limit_mb = float(adm.granted_mb)
            return None
        if adm.outcome == RB.BUSY:
            if d.get("busy_with") in ("light", "light_limit"):                 # Phase D.3: a frame-extraction / video-check job holds the budget (short); say so, not "another calibration"
                return "The server is busy checking another visitor's videos or frames. Nothing was started. Please try again in a minute."
            return "Another calibration is running on this server. Please try again when it has finished."
        return "This server does not have enough free memory to start the calibration right now. Nothing was started. Try again in a few minutes."

    def _release_reservation(self):
        r, self._reservation = self._reservation, None
        if r is not None:
            r.release()

    def _guard_report(self) -> dict:
        """F8.25: the saved ``memory_guard`` block. Old keys keep their names; ``active`` is true when ANY guard operated, and ``basis`` / ``limit_mb`` describe
        the guard(s) that operated. ``process_tree`` and ``cgroup`` hold each guard's own detail; ``active_guards`` lists the operating ones."""
        cg = dict(self._guard_info); ti = dict(self._tree_info)
        cg_active = bool(cg.get("active")); tree_active = bool(ti.get("active"))
        cg_detail = dict(active=cg_active, basis=cg.get("basis"), limit_mb=cg.get("limit_mb"), threshold_mb=cg.get("threshold_mb"), peak_used_mb=cg.get("peak_used_mb"), checks=cg.get("checks"))
        out = dict(cg)
        out["active"] = cg_active or tree_active
        out["active_guards"] = (["process_tree_rss"] if tree_active else []) + (["cgroup"] if cg_active else [])
        if tree_active and cg_active:
            out["basis"] = "process-tree RSS + cgroup " + str(cg.get("basis"))
        elif tree_active:
            out["basis"] = "process-tree RSS"; out["limit_mb"] = ti["limit_mb"]; out["threshold_mb"] = ti["limit_mb"]; out["peak_used_mb"] = ti["peak_rss_mb"]
            out["checks"] = ti["checks"]
        out["process_tree"] = dict(active=tree_active, limit_mb=ti["limit_mb"], peak_rss_mb=ti["peak_rss_mb"], checks=ti["checks"], fresh_checks=ti["fresh_checks"],
                                   basis="process-tree RSS" if ti["limit_mb"] else None)
        out["cgroup"] = cg_detail
        return out

    def _guard_operative(self) -> bool:
        """Could the guard act at all? True with a MEASURABLE per-process RSS limit or a readable container limit plus a usable usage figure."""
        g = guard_memory(self._read_cg())
        tree_ok = False
        if self.limits.max_tree_rss_mb and self._monitor is not None:           # F8.20a: an RSS limit counts only if a fresh reading of this process works (ps / proc)
            self._monitor._safe_sample()
            tree_ok = _fresh_tree_rss(self._monitor.latest) is not None
        return tree_ok or bool(g["limit_mb"] and g["used_mb"] is not None)

    def _read_json(self, path, limit):
        try:
            if Path(path).stat().st_size > limit:
                return None
            return json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None

    def _run(self):
        lim = self.limits; proc = None; lease_id = None; outcome = "worker_crashed"; validation = None; worker_result = None
        mon = self._monitor = ResourceMonitor(worker_pid=lambda: proc.pid if proc is not None and proc.poll() is None else None, disk=self._disk, interval=self.monitor_interval).start()
        try:
            mon.mark("validate"); self._set(state="validating", stage="validate", message="Checking the two videos…")
            if self.mode == "calibrate" and not lim.allow_full:                  # explicit opt-in: a full run is never started by accident
                outcome = "rejected"
                return self._finish("rejected", error="A full calibration is disabled on this server (set F8_ENABLE_FULL_CALIBRATION=1 to allow it). Run the benchmark instead.")
            if self.mode == "calibrate":                                          # F8.10 start-up safety checks (nothing is deleted when they refuse)
                if lim.require_memory_guard and not self._guard_operative():
                    outcome = "rejected"
                    return self._finish("rejected", error="The memory guard cannot act on this server (no container memory limit could be read), and F8_REQUIRE_MEMORY_GUARD=1 is set, so a full run is refused.")
                g = guard_memory(self._read_cg())
                if g["limit_mb"] and g["used_mb"] is not None and g["used_mb"] > lim.start_headroom_fraction * g["limit_mb"]:
                    outcome = "rejected"
                    return self._finish("rejected", error=f"Container memory is already {g['used_mb']:.0f} MB of {g['limit_mb']:.0f} MB; a full run is not started until it drops below {lim.start_headroom_fraction:.0%}. Try again shortly.")
            denial = self._reserve()                                              # F8.31C-1C: shared memory budget and worker slot (BUSY and INSUFFICIENT_MEMORY are different outcomes)
            if denial:
                outcome = "rejected"
                return self._finish("rejected", result=dict(resource_budget=self._budget_info), error=denial)
            try:
                lease = self.store.acquire_lease(self.owner)
            except UploadError as e:
                return self._finish("rejected", error=e.message)
            lease_id = self._lease = lease["lease_id"]
            validation = validate_videos(lease["left"]["path"], lease["right"]["path"], lim)
            if not validation["ok"]:
                outcome = "rejected"; self._delete_rejected = True        # the videos themselves are unusable: retrying the same files cannot succeed, so free the disk (as before F8.10)
                return self._finish("rejected", validation=validation, error="; ".join(e["message"] for e in validation["errors"][:3]))
            if not (self.f7_dir / "f7_calibration.py").is_file():
                outcome = "rejected"
                return self._finish("rejected", validation=validation, error="The F7 calibration module was not found on this machine.")
            self._job_dir = Path(tempfile.mkdtemp(prefix="f8_job_", dir=self.work_dir)); os.chmod(self._job_dir, 0o700)
            spec = dict(left_path=lease["left"]["path"], right_path=lease["right"]["path"], cols=int(self.settings["cols"]), rows=int(self.settings["rows"]),
                        square_mm=float(self.settings["square_mm"]), baseline_mm=self.settings.get("baseline_mm") or None, f7_dir=str(self.f7_dir),
                        out_dir=str(self._job_dir), parent_pid=os.getpid(), dump_detections=self.dump_detections and self.mode == "calibrate", mode=self.mode,
                        allowed_root=str(self.store.root), expected_sha256=dict(left=lease["left"]["sha256"], right=lease["right"]["sha256"]))
            if self.mode == "benchmark":
                spec.update(max_frames=self.bench_frames, frame_counts={k: validation["videos"][k].get("frame_count") for k in ("left", "right")})
            spec_path = self._job_dir / "spec.json"; spec_path.write_text(json.dumps(spec))
            mon.mark("start"); self._set(state="running", stage="start", message="Starting the calibration worker…")
            proc = self._proc = self._spawn(spec_path, self._job_dir)
            self.store.attach_process(self.owner, lease_id, proc.pid)
            t_start = time.monotonic(); last_hb = 0.0; cur_stage = "start"; reason = None
            while proc.poll() is None:
                time.sleep(lim.poll_s)
                now = time.monotonic()
                if now - last_hb >= lim.heartbeat_s:
                    last_hb = now
                    try:
                        if self.store.heartbeat(self.owner, lease_id).get("cancel_requested"):
                            self._cancel.set()
                    except UploadError:
                        reason = "lease_expired"; break
                prog = self._read_json(self._job_dir / "progress.json", 4096)
                if prog:
                    self._last_prog = prog
                if prog and isinstance(prog.get("stage"), str):
                    stage = prog["stage"][:20]
                    if stage != cur_stage:
                        cur_stage = stage; mon.mark(stage)
                    self._set(stage=stage, fraction=min(max(float(prog.get("fraction") or 0.0), 0.0), 0.999), message=f"{stage.replace('_', ' ')}…", monitor=dict(mon.latest))
                else:
                    self._set(monitor=dict(mon.latest))
                if self._cancel.is_set():
                    reason = "cancelled"; break
                if now - t_start > self.timeout_s:
                    reason = "timeout"; break
                g = self._guard(mon.latest)
                if g:
                    reason = "memory_limit"; self._set(message=g); break
            if reason:
                self._terminate(proc); outcome = reason
            rc = proc.wait()
            self.store.attach_process(self.owner, lease_id, None)
            worker_result = self._read_json(self._job_dir / "result.json", lim.result_max_bytes)
            if reason is None:
                if worker_result is None:
                    outcome = "worker_crashed"
                elif worker_result.get("ok") and (worker_result.get("assessment") or {}).get("level") != "FAILED":
                    outcome = "completed"
                else:
                    outcome = "failed"
            mon.mark("finish")
            err = None
            if outcome == "worker_crashed":
                tail = ""
                try:
                    tail = (self._job_dir / "worker.stderr").read_bytes()[-600:].decode("utf-8", "replace")
                except OSError:
                    pass
                err = f"The calibration worker stopped unexpectedly (exit code {rc})." + (f" Last output: {tail.strip()[-300:]}" if tail.strip() else "")
            elif outcome == "timeout":
                err = f"The {'benchmark' if self.mode == 'benchmark' else 'calibration'} took longer than {self.timeout_s:g} s and was stopped."
            elif outcome == "cancelled":
                err = "The calibration was cancelled."
            elif outcome == "memory_limit":
                err = self.status()["message"]
            elif outcome == "lease_expired":
                err = "The job's lease expired, so the worker was stopped."
            elif outcome == "failed":
                err = (worker_result or {}).get("error") or "The calibration could not be computed."
            self._finish(outcome, result=worker_result, validation=validation, error=err)
        except BaseException as e:                                     # never leave the UI spinning
            if proc is not None:
                self._terminate(proc)
            self._finish("worker_crashed", validation=validation, error=f"Supervisor error: {type(e).__name__}")
        finally:
            try:
                self._cleanup(lease_id, proc)
            finally:
                self._release_reservation()                                # whatever happened above, a reservation never outlives the job

    def _benchmark_report(self, resources) -> dict:
        """One flat, bounded summary of everything the Cloud test must tell us (all numbers measured; unavailable ones are None, never guessed)."""
        res = self.result or {}; b = res.get("benchmark") or {}; r = resources or {}; peak = r.get("peak") or {}; cg = r.get("cgroup") or {}
        env = res.get("environment") or {}; cpu = dict(r.get("cpu") or {}, worker_self_cpu_s=res.get("worker_cpu_s"))
        done = res.get("outcome") == "completed" and bool(b)
        prog = self._last_prog or {}
        est_frames = None if done else prog.get("frames")
        v = (res.get("validation") or {}).get("videos") or {}
        full_frames = sum(int((v.get(k) or {}).get("frame_count") or 0) for k in ("left", "right")) or None
        fps = b.get("frames_per_s")
        projection = None
        if done and fps and full_frames:
            projection = dict(full_length_frames=full_frames, estimated_full_detection_s=round(full_frames / fps, 0),
                              note="Linear extrapolation of the benchmark's frames/s to the whole recordings (detection stage only; solving adds seconds). Frames that show the board "
                                   "cost more than empty ones, so judge it together with detection_rate; treat it as an estimate within about a factor of two.")
        return dict(
            outcome=res.get("outcome"), completed=done, frames_processed=b.get("frames_processed", est_frames), frames_processed_is_estimate=not done and est_frames is not None,
            frames_requested_per_video=self.bench_frames, detection_attempts=b.get("detection_attempts"), checkerboard_detections=b.get("detections"),
            detection_rate=round(b["detections"] / b["detection_attempts"], 4) if b.get("detection_attempts") else None, per_video=b.get("videos"),
            wall_clock_s=res.get("total_wall_s"), detect_wall_s=b.get("detect_wall_s"), supervisor_total_s=r.get("total_wall_s"), frames_per_s=fps,
            cpu=dict(worker_cpu_s=cpu.get("worker_self_cpu_s"), worker_tree_cpu_s=cpu.get("worker_tree_cpu_s"), worker_avg_cores=cpu.get("worker_tree_avg_cores"),
                     container_cpu_s=cpu.get("cgroup_cpu_s"), container_avg_cores=cpu.get("cgroup_avg_cores"), cpu_limit_cores=cpu.get("cpu_limit_cores"), host_cpu_count=cpu.get("host_cpu_count"),
                     cv2_threads=(res.get("env") or {}).get("threads"), periods=cpu.get("periods"), throttled_periods=cpu.get("throttled_periods"), throttled_s=cpu.get("throttled_s"),
                     throttled_fraction=cpu.get("throttled_fraction"), throttling_measurable=cpu.get("throttling_measurable"), container_cpu_measurable=cpu.get("cgroup_cpu_measurable")),
            memory=dict(worker_self_peak_rss_mb=res.get("peak_rss_mb"), worker_tree_peak_rss_mb=peak.get("worker_rss_mb"), parent_peak_rss_mb=peak.get("parent_rss_mb"),
                        combined_peak_rss_mb=peak.get("tree_rss_mb"), combined_peak_pss_mb=peak.get("tree_pss_mb"), container_peak_mb=peak.get("cgroup_peak_mb"),
                        container_sampled_max_mb=peak.get("cgroup_memory_mb"), container_limit_mb=cg.get("memory_max_mb"), oom_kills=cg.get("oom_events"),
                        container_measurable=(r.get("available") or {}).get("cgroup"), pss_measurable=(r.get("available") or {}).get("pss")),
            disk=dict(temp_disk_peak_mb=peak.get("temp_disk_mb"), temp_disk_free_min_mb=min([x["disk_free_min_mb"] for x in r.get("stages", []) if x.get("disk_free_min_mb") is not None] or [None]) if r else None,
                      temp_dir_memory_backed=(env.get("temp_dir_mount") or {}).get("memory_backed"), temp_dir_fstype=(env.get("temp_dir_mount") or {}).get("fstype"), measurable=peak.get("temp_disk_mb") is not None),
            limits=dict(benchmark_timeout_s=self.timeout_s, timed_out=res.get("outcome") == "timeout", cancelled=res.get("outcome") == "cancelled", memory_guard_triggered=res.get("outcome") == "memory_limit",
                        frames_cap=self.limits.benchmark_frames_cap, full_calibration_timeout_s=max(1.0, min(self.limits.timeout_s, self.store.limits.lease_max_s - self.lease_margin_s)), full_calibration_allowed=self.limits.allow_full),
            integrity=dict(sha256_verified_before_use=bool((res.get("stage_timings") or {}).get("verify", {}).get("checked")), videos_unchanged_after=b.get("videos_unchanged")),
            projection=projection)

    def _persist(self) -> dict:
        """Write the result document atomically. Never raises; the outcome is recorded so the caller can decide whether the videos may be deleted."""
        try:
            info = self.results.save_result(self.job_code, build_result_document(self.result, code=self.job_code, settings=self.settings))
            return dict(saved=True, **info)
        except Exception as e:                                          # a full disk, a bad folder, ...: report it, keep the videos
            return dict(saved=False, error=f"{type(e).__name__}: {str(e)[:120]}")

    def _cleanup(self, lease_id, proc):
        """End of a job, in this order:  stop the worker -> measure -> SAVE THE RESULT (atomic) -> only then release the lease and decide about the videos -> done.

        Videos are deleted ONLY after a completed full calibration whose result is safely saved. After any other outcome of a full run (failure, timeout,
        worker crash, memory stop, cancellation, rejection) they are kept, owner-only and unprotected, for ``retry_retention_s`` so a retry needs no re-upload;
        then the sweeper deletes them. A benchmark always keeps them (its own retention). Session lock and abandonment still delete everything."""
        try:
            self._set(state="cleaning")
            if proc is not None:
                self._terminate(proc)
            self._release_reservation()                                   # the worker is gone: its memory is free for the next job (also released in the outer finally)
            resources = self._monitor.stop() if self._monitor else None
            if self.result is not None:
                self.result["resources"] = resources
                self.result["environment"] = probe_environment(self.store.root)
                self.result["timeout"] = self.timeout_info
                self.result["memory_guard"] = dict(self._guard_report(), max_cgroup_fraction=self.limits.max_cgroup_fraction, max_tree_rss_mb=self.limits.max_tree_rss_mb)
                if self._budget_info is not None:
                    self.result["resource_budget"] = self._budget_info
                if self.mode == "benchmark":
                    self.result["benchmark_report"] = self._benchmark_report(resources)
                outcome = self.result.get("outcome", "failed")
                self._persisted = self._persist()                      # BEFORE any video is released
                self.result["persistence"] = dict(self._persisted, scope="server-local, ephemeral; download the result to keep it")
            else:
                outcome = "failed"
            delete = self.mode == "calibrate" and ((outcome == "completed" and bool((self._persisted or {}).get("saved"))) or self._delete_rejected)
            retry_keep = self.mode == "calibrate" and not delete
            released = None
            if lease_id:
                for _ in range(20):
                    try:
                        self.store.release_lease(self.owner, lease_id, outcome, delete_files=delete, retry_keep=retry_keep)
                        released = True
                        break
                    except UploadError as e:
                        if e.code == "in_use":
                            time.sleep(0.25); continue
                        released = False
                        break
                self._lease = None
            if self.result is not None:
                kept = bool(released) and not delete
                self.result["cleanup"] = dict(lease_released=released, videos_deleted=bool(released) and delete, job_dir_removed=True, videos_kept_for_next_step=kept,
                                              kept_for_retry_s=self.store.limits.retry_retention_s if kept and retry_keep else (None if not kept else self.store.limits.complete_retention_s))
                if self._persisted and self._persisted.get("saved"):
                    second = self._persist()                  # second atomic write: the same document with the final cleanup facts (the first copy stands if this fails)
                    if second.get("saved"):
                        self._persisted = second; self.result["persistence"].update({k: v for k, v in second.items() if k in ("saved", "bytes", "sha256")})
            if self._job_dir is not None:
                shutil.rmtree(self._job_dir, ignore_errors=True)
        except BaseException as e:                                     # never leave the UI spinning, whatever went wrong here
            if self.result is not None:
                self.result.setdefault("cleanup", dict(lease_released=None, videos_deleted=False, job_dir_removed=False, error=type(e).__name__))
        finally:
            with self._lock:
                self._status["monitor"] = None
            self._set(state="done")
            self._publish(force=True)
            with _ACTIVE_LOCK:
                if _ACTIVE.get(self.job_code) is self:
                    del _ACTIVE[self.job_code]
