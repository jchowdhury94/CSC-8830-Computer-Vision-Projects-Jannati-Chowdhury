"""Stage F8.5: supervisor for a dual-video calibration job (NO Streamlit import; runs in the app process, does the heavy work in a WORKER process).

    acquire lease (both videos protected)  ->  explicit safety validation of the two videos  ->  worker process (sequential LEFT, RIGHT detection, solve,
    assess)  ->  heartbeat / progress / resource sampling / timeout / cancel / memory guard  ->  result  ->  release lease and delete both videos.

Nothing from F7 is re-implemented or changed: the worker imports the existing module and calls its functions. The F7 *visitor-upload* limits
(200 MB, 120 s) are not applied because they are about browser uploads; instead ``RunnerLimits`` states explicit limits for full-length recordings, and
every check is reported in the result (``validation``), never silently skipped.
"""
from __future__ import annotations

import json
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
from monitor import ResourceMonitor, probe_environment, read_cgroup
from video_check import VideoSafetyLimits, inspect_video

HERE = Path(__file__).resolve().parent
WORKER = HERE / "calibration_worker.py"
DEFAULT_F7_DIR = HERE.parent / "visitor_workflow"                      # Modules/Module_7/visitor_workflow (holds f7_calibration.py)
MiB = 1024 * 1024
_ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH")

STAGE_ORDER = ("validate", "start", "import", "verify", "detect_left", "detect_right", "verify_after", "solve", "assess", "finish")
MODES = ("calibrate", "benchmark")


def _env_num(env, name, default, lo, hi):
    """Number from an environment variable, clamped to [lo, hi]; the default when missing, invalid or not finite."""
    try:
        v = float(env[name])
    except (KeyError, ValueError, TypeError):
        return default
    return default if v != v or v in (float("inf"), float("-inf")) else min(max(v, lo), hi)


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
    max_tree_rss_mb: float | None = None       # kill if parent+worker RSS exceeds this
    max_cgroup_fraction: float = 0.90          # kill if container ANONYMOUS memory (not page cache) exceeds this fraction of the cgroup limit
    result_max_bytes: int = 32 * MiB
    inspect_timeout_s: float = 60.0            # per-video sandboxed inspection
    allow_full: bool = False                   # may a FULL calibration be started at all? Safe default OFF; from_env() enables it only with F8_ENABLE_FULL_CALIBRATION=1
    benchmark_timeout_s: float = 300.0         # wall clock for a benchmark worker
    benchmark_frames: int = 600                # frames per video the benchmark reads (every STRIDE-th gets a detection attempt)
    benchmark_frames_cap: int = 3000           # hard ceiling whatever is asked for

    @classmethod
    def from_env(cls, env=None):
        """Deployment configuration. F8_CALIB_TIMEOUT_S (60-14400, default 3600), F8_BENCH_TIMEOUT_S (10-900), F8_BENCH_FRAMES (30-cap),
        F8_ENABLE_FULL_CALIBRATION (only the exact value '1' enables full runs). Invalid values fall back to the defaults."""
        env = os.environ if env is None else env
        d = cls()
        return cls(timeout_s=_env_num(env, "F8_CALIB_TIMEOUT_S", d.timeout_s, 60.0, 14400.0), benchmark_timeout_s=_env_num(env, "F8_BENCH_TIMEOUT_S", d.benchmark_timeout_s, 10.0, 900.0),
                   benchmark_frames=int(_env_num(env, "F8_BENCH_FRAMES", d.benchmark_frames, 30, d.benchmark_frames_cap)), allow_full=str(env.get("F8_ENABLE_FULL_CALIBRATION", "")).strip() == "1")


class RunnerError(Exception):
    pass


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
                 benchmark_frames: int | None = None):
        if mode not in MODES:
            raise RunnerError("unknown mode")
        self.mode = mode
        self.store, self.owner, self.settings, self.limits = store, owner, dict(settings), limits
        self.bench_frames = int(min(max(int(benchmark_frames or limits.benchmark_frames), 1), limits.benchmark_frames_cap))
        # the timeout actually applied: the benchmark has its own short one; a full run can never outlive the store's hard lease limit
        self.timeout_s = float(limits.benchmark_timeout_s if mode == "benchmark" else max(1.0, min(limits.timeout_s, store.limits.lease_max_s - 120.0)))
        self._last_prog = None
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
            self._status.update(kw)

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
        self._set(state="done", outcome=outcome, message=error or (("Benchmark finished." if self.mode == "benchmark" else "Calibration finished.") if outcome == "completed" else outcome), fraction=1.0 if outcome == "completed" else self._status["fraction"])

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

    def _guard(self, sample) -> str | None:
        lim = self.limits
        tree = sample.get("combined_rss_mb")
        if lim.max_tree_rss_mb and tree is not None and tree > lim.max_tree_rss_mb:
            return f"memory guard: parent+worker RSS {tree:.0f} MB exceeded {lim.max_tree_rss_mb:.0f} MB"
        cg = read_cgroup()
        mx = cg.get("memory_max_mb")
        # page cache is reclaimable, so judge anonymous memory when known -- PLUS shared memory: a memory-backed /tmp (tmpfs) holds the uploaded videos as shmem,
        # which is not reclaimable and counts toward the container's OOM limit but is not "anon"
        used = None if cg.get("anon_mb") is None else cg["anon_mb"] + (cg.get("shmem_mb") or 0.0)
        used = used if used is not None else cg.get("memory_current_mb")
        if mx and used is not None and used > lim.max_cgroup_fraction * mx:
            return f"memory guard: container memory {used:.0f} MB exceeded {lim.max_cgroup_fraction:.0%} of its {mx:.0f} MB limit"
        return None

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
            try:
                lease = self.store.acquire_lease(self.owner)
            except UploadError as e:
                return self._finish("rejected", error=e.message)
            lease_id = self._lease = lease["lease_id"]
            validation = validate_videos(lease["left"]["path"], lease["right"]["path"], lim)
            if not validation["ok"]:
                outcome = "rejected"
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
            self._cleanup(lease_id, proc)

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
                        frames_cap=self.limits.benchmark_frames_cap, full_calibration_timeout_s=min(self.limits.timeout_s, self.store.limits.lease_max_s - 120.0), full_calibration_allowed=self.limits.allow_full),
            integrity=dict(sha256_verified_before_use=bool((res.get("stage_timings") or {}).get("verify", {}).get("checked")), videos_unchanged_after=b.get("videos_unchanged")),
            projection=projection)

    def _cleanup(self, lease_id, proc):
        """Release the lease (deleting both videos) only after the worker is gone; keep trying briefly, then leave it to the lease expiry."""
        self._set(state="cleaning" if self.result is None else self._status["state"])
        if proc is not None:
            self._terminate(proc)
        released = None
        if lease_id:
            for _ in range(20):
                try:
                    self.store.release_lease(self.owner, lease_id, (self.result or {}).get("outcome", "failed"), delete_files=self.mode != "benchmark")
                    released = True
                    break
                except UploadError as e:
                    if e.code == "in_use":
                        time.sleep(0.25); continue
                    released = False
                    break
            self._lease = None
        resources = self._monitor.stop() if self._monitor else None
        if self.result is not None:
            self.result["resources"] = resources
            self.result["environment"] = probe_environment(self.store.root)
            self.result["cleanup"] = dict(lease_released=released, videos_deleted=bool(released) and self.mode != "benchmark", job_dir_removed=True)
            if self.mode == "benchmark":
                self.result["cleanup"]["videos_kept_for_next_step"] = bool(released)
                self.result["benchmark_report"] = self._benchmark_report(resources)
        if self._job_dir is not None:
            shutil.rmtree(self._job_dir, ignore_errors=True)
        with self._lock:
            self._status["monitor"] = None
