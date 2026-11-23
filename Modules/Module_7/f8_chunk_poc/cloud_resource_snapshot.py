"""F8.31C-2C-1: one-shot, read-only resource snapshot for a controlled Streamlit Cloud measurement (NO Streamlit import; standard library plus ``monitor`` and ``resource_budget``).

What it is: a function, ``take_snapshot()``, that returns ONE JSON-safe dict describing the container the app runs in.  Every leaf is tagged with how it is known:

    measured    read from the running system now
    configured  a limit or default written in this repository / the environment (an operator's statement, not a measurement)
    estimated   derived from a LOCAL measurement or a formula; says so in ``basis``
    unknown     could not be determined; ``reason`` says why.  Nothing is ever invented.

What it is NOT: it never runs reconstruction or calibration, never opens a video, never allocates memory or loads the CPU or disk, never loops, never writes anything
(no file, no log), and never reports an environment variable, a path, a file name or a visitor identifier.  The optional process probe starts ONE trivial interpreter
(``python -I -c pass``) to time process creation and test process-group cleanup; it is off unless asked for, injectable, and is not a calibration or reconstruction worker.

One benign side effect, inherited from the shared budget: asking the shared coordinator for its state creates (if absent) its empty per-user 0700 directory and lock file in the
temp directory.  The snapshot says whether the directory existed beforehand.
"""
from __future__ import annotations

import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import monitor as MON
import resource_budget as RB

SCHEMA = "f8.cloud_resource_snapshot/1"
MiB = 1024 * 1024
UPLOAD_PREFIXES = ("f8_chunk_poc_", "f8_room_videos_")        # the calibration store and the room-video store (aggregate size only; no name is reported)
WALK_MAX_ENTRIES = 50_000                                      # bounded directory walk
# Local, NOT Cloud, reference numbers from ARCHITECTURE_PLAN.md 27.4 (one macOS machine).  Used only to label estimates.
LOCAL_REFERENCE = dict(worker_peak_mb=2016.0, parent_peak_mb=318.0, job_dir_peak_mb=72.6, requested_worker_mb=3072.0, runtime_s=17.5, source="ARCHITECTURE_PLAN.md 27.4 (macOS, 32 GB; not Cloud)")
STAGED_BYTES_CONFIGURED_MB = 1024.0                            # ReconstructionLimits.max_job_bytes (a configured ceiling, not an expectation)


# ------------------------------------------------------------------ tagged leaves
def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def measured(v, **kw):
    return dict(kind="measured", value=v, **kw) if v is not None else unknown("not readable here")


def configured(v, **kw):
    return dict(kind="configured", value=v, **kw)


def estimated(v, basis, **kw):
    return dict(kind="estimated", value=v, basis=basis, **kw)


def unknown(reason):
    return dict(kind="unknown", value=None, reason=str(reason))


def _r(v, nd=1):
    return round(float(v), nd) if _fin(v) else None


# ------------------------------------------------------------------ pure parsers (unit-tested without a Linux machine)
def parse_cpu_max(text):
    """cgroup v2 ``cpu.max`` ("<quota|max> <period>") -> cores or None.  'max', zero, negative, garbage -> None (no quota)."""
    try:
        q, p = str(text).split()[:2]
        if q == "max":
            return None
        q, p = float(q), float(p)
        return q / p if q > 0 and p > 0 and math.isfinite(q / p) else None
    except (ValueError, TypeError):
        return None


def parse_cfs(quota, period):
    """cgroup v1 ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` -> cores or None (-1 means unlimited)."""
    try:
        q, p = float(str(quota).strip()), float(str(period).strip())
        return q / p if q > 0 and p > 0 else None
    except (ValueError, TypeError):
        return None


_PATHLIKE = re.compile(r"/tmp/|/var/folders|/private/|/home/|/Users/|/mnt/|/root|[A-Za-z]:\\|~/|f8_chunk_poc_|f8_room_videos_|f8_resource_budget_|\.MOV|\.mp4|\.json")


def scrub(obj):
    """Defence in depth: replaces any string that names a temp/user/visitor location or a private file type.  (Fixed system paths such as the cgroup mount are not private and stay readable.)  The snapshot is not built from paths; this guards against a future regression."""
    if isinstance(obj, dict):
        return {str(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        return "[redacted]" if _PATHLIKE.search(obj) else obj
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# ------------------------------------------------------------------ sections
def memory_section(cgroup_root="/sys/fs/cgroup", proc_root="/proc", coordinator=None, env=None, pid=None) -> dict:
    pid = int(pid or os.getpid())
    out = {}
    container = RB.container_memory_limit(cgroup_root, proc_root)
    cg = MON.read_cgroup(cgroup_root, proc_root)
    out["cgroup_version"] = measured(cg.get("version")) if cg.get("version") else unknown(container.get("reason"))
    if container.get("limit_mb") is not None:
        out["container_limit_mb"] = measured(_r(container["limit_mb"]), source=container.get("source"))
        out["limit_status"] = measured("limited")
    elif cg.get("version"):
        out["container_limit_mb"] = unknown("no finite memory limit is set at this cgroup level (unlimited): " + str(container.get("reason")))
        out["limit_status"] = measured("unlimited")
    else:
        out["container_limit_mb"] = unknown(container.get("reason"))
        out["limit_status"] = unknown("no cgroup visible")
    used = RB.container_used_mb(cg)
    out["container_used_mb"] = measured(_r(used), basis="anonymous + shmem when readable; page cache excluded") if used is not None else unknown("container usage not readable")
    out["container_memory_peak_mb"] = measured(_r(cg.get("memory_peak_mb"))) if cg.get("memory_peak_mb") is not None else unknown("memory.peak not readable")
    out["oom_kill_events"] = measured(cg.get("oom_events")) if cg.get("oom_events") is not None else unknown("memory.events not readable")
    table = {}
    try:
        table = MON.snapshot_processes(proc_root) or {}
    except Exception:
        table = {}
    own = (table.get(pid) or {}).get("rss_mb")
    out["process_rss_mb"] = measured(_r(own)) if _fin(own) else unknown("process table not readable")
    tree = [p for p in MON.tree_pids(table, pid) if p in table] if table else []
    tree_rss = [table[p].get("rss_mb") for p in tree]
    out["process_tree_rss_mb"] = measured(_r(sum(v for v in tree_rss if _fin(v))), processes=len(tree)) if tree and any(_fin(v) for v in tree_rss) else unknown("process tree not readable")
    # shared budget: the SAME functions the runners use
    cfg = RB.BudgetConfig.from_env(env)
    budget = RB.compute_budget(cfg, container)
    out["budget_total_mb"] = dict(kind="configured" if not container.get("reliable") else "measured", value=_r(budget["total_mb"]), source=budget["source"])
    out["safety_reserve_mb"] = configured(_r(budget["reserve_mb"]), floor_mb=cfg.reserve_floor_mb, fraction=cfg.reserve_fraction)
    out["budget_usable_mb"] = estimated(_r(budget["usable_mb"]), "total minus the safety reserve")
    out["budget_limitation"] = budget["limitation"]
    out["memory_ceiling_configured_mb"] = configured(cfg.ceiling_mb) if cfg.ceiling_mb is not None else unknown("F8_APP_MEMORY_CEILING_MB not set")
    out["memory_limit_unknown"] = container.get("limit_mb") is None
    try:
        coord = coordinator or RB.app_coordinator()
        st = coord.state()
        out["available_for_new_worker_mb"] = estimated(st["available_mb"], "usable - parent RSS - other app processes - active reservations (shared budget, resource_budget.state)")
        out["parent_rss_mb_used_by_budget"] = measured(st["parent_rss_mb"]) if st.get("parent_rss_measured") else configured(st["parent_rss_mb"], note="parent RSS unreadable; cautious assumed value")
        out["active_reservations"] = len(st.get("reservations", []))
    except Exception as ex:
        out["available_for_new_worker_mb"] = unknown(f"coordinator state unavailable ({type(ex).__name__})")
    return out


def cpu_section(cgroup_root="/sys/fs/cgroup", proc_root="/proc") -> dict:
    out = dict(logical_cpus=measured(os.cpu_count()))
    try:
        out["affinity_cpus"] = measured(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        out["affinity_cpus"] = unknown("sched_getaffinity not available")
    cg = MON.read_cgroup(cgroup_root, proc_root)
    cores = cg.get("cpu_limit_cores")
    if _fin(cores):
        out["cpu_quota_cores"] = measured(_r(cores, 3), source=f"cgroup_v{cg.get('version')}")
    elif cg.get("version"):
        out["cpu_quota_cores"] = unknown("no CPU quota set (unlimited at this cgroup level)")
    else:
        out["cpu_quota_cores"] = unknown("no cgroup visible")
    cands = [v for v in (cores if _fin(cores) else None, out["affinity_cpus"].get("value"), out["logical_cpus"].get("value")) if _fin(v) and v > 0]
    out["effective_cpus"] = estimated(_r(min(cands), 3), "smallest of CPU quota, affinity and logical CPUs; a shared host may still be contended") if cands else unknown("no CPU figure available")
    out["throttling"] = (measured(dict(periods=cg.get("nr_periods"), throttled=cg.get("nr_throttled"))) if cg.get("nr_periods") is not None else unknown("cpu.stat not readable"))
    return out


def _dir_size(path: Path) -> tuple:
    """(bytes, truncated) of a directory tree without following links; bounded entry count; names are never returned."""
    total, n, trunc, stack = 0, 0, False, [path]
    while stack and not trunc:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    n += 1
                    if n > WALK_MAX_ENTRIES:
                        trunc = True
                        break
                    try:
                        if e.is_symlink():
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total, trunc


def disk_section(tmp_dir=None, mounts="/proc/self/mounts", prefixes=UPLOAD_PREFIXES) -> dict:
    tmp = str(tmp_dir or tempfile.gettempdir())
    out = {}
    try:
        du = shutil.disk_usage(tmp)
        out["temp_free_mb"] = measured(_r(du.free / MiB))
        out["temp_total_mb"] = measured(_r(du.total / MiB))
    except OSError:
        out["temp_free_mb"] = unknown("disk usage not readable")
        out["temp_total_mb"] = unknown("disk usage not readable")
    mi = MON.mount_info(tmp, mounts)
    if mi:
        out["temp_filesystem"] = measured(mi.get("fstype"))
        out["temp_memory_backed"] = measured(bool(mi.get("memory_backed")))
    else:
        out["temp_filesystem"] = unknown("mount table not readable (not Linux, or /proc not visible)")
        out["temp_memory_backed"] = unknown("filesystem type unknown; a memory-backed temp dir would count against the memory budget")
    used, trunc, found = 0, False, 0
    try:
        for p in Path(tmp).iterdir():
            if p.name.startswith(tuple(prefixes)) and p.is_dir() and not p.is_symlink():
                b, t = _dir_size(p)
                used += b
                trunc = trunc or t
                found += 1
    except OSError:
        pass
    out["visitor_upload_storage_mb"] = measured(_r(used / MiB, 2), directories=found, truncated=trunc, basis="aggregate size only; no names or contents read")
    out["scratch_estimate_mb"] = estimated(_r(LOCAL_REFERENCE["job_dir_peak_mb"]), "job directory peak in one LOCAL full-resolution run; not a Cloud measurement",
                                           configured_ceiling_mb=STAGED_BYTES_CONFIGURED_MB)
    out["uploads_plus_scratch_estimate_mb"] = estimated(_r(used / MiB + LOCAL_REFERENCE["job_dir_peak_mb"], 1), "current uploads + local job-dir peak")
    return out


def _spawn_probe(timeout_s=10.0) -> dict:
    """Times process creation and tests process-group cleanup with a trivial interpreter that does nothing (``-I -c pass``).  Not a worker; no project code is loaded."""
    out = {}
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-I", "-c", "pass"], timeout=timeout_s, capture_output=True, stdin=subprocess.DEVNULL)
        out["process_creation"] = measured(p.returncode == 0)
        out["worker_startup_s"] = measured(_r(time.perf_counter() - t0, 3), basis="empty interpreter start (python -I -c pass); a real worker also imports NumPy/OpenCV")
    except Exception as ex:
        out["process_creation"] = measured(False, error=type(ex).__name__)
        out["worker_startup_s"] = unknown("process creation failed")
        out["process_group_cleanup"] = unknown("not tested")
        return out
    try:
        q = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            same_group = os.getpgid(q.pid) == q.pid
            os.killpg(q.pid, signal.SIGKILL)
            q.wait(timeout=5)
            out["process_group_cleanup"] = measured(bool(same_group))
        finally:
            if q.poll() is None:
                q.kill()
                q.wait(timeout=5)
    except Exception as ex:
        out["process_group_cleanup"] = measured(False, error=type(ex).__name__)
    return out


def worker_section(probe_processes=False, spawn_probe=None, coordinator_dir=None) -> dict:
    out = dict(python=measured(sys.version.split()[0]), implementation=measured(platform.python_implementation()))
    try:
        import cv2
        out["opencv"] = measured(str(cv2.__version__))
    except Exception as ex:
        out["opencv"] = unknown(f"OpenCV not importable ({type(ex).__name__})")
    try:
        import numpy
        out["numpy"] = measured(str(numpy.__version__))
    except Exception:
        out["numpy"] = unknown("NumPy not importable")
    if probe_processes:
        out.update((spawn_probe or _spawn_probe)())
    else:
        for k in ("process_creation", "worker_startup_s", "process_group_cleanup"):
            out[k] = unknown("process probe not requested")
    out["fcntl_cross_process_coordination"] = measured(RB.fcntl is not None)
    out["flock_available"] = out["fcntl_cross_process_coordination"]
    root = coordinator_dir
    if root is None:
        root = Path(tempfile.gettempdir()) / f"f8_resource_budget_{os.getuid()}" if hasattr(os, "getuid") else None
    exists = bool(root and Path(root).is_dir())
    out["coordinator_dir_existed"] = measured(exists)
    out["coordinator_dir_usable"] = measured(os.access(root if exists else tempfile.gettempdir(), os.W_OK | os.X_OK))
    return out


def take_snapshot(*, probe_processes=False, cgroup_root="/sys/fs/cgroup", proc_root="/proc", mounts="/proc/self/mounts", tmp_dir=None, coordinator=None, env=None,
                  spawn_probe=None, include_clock=True) -> dict:
    """The whole snapshot.  JSON-safe (``json.dumps(..., allow_nan=False)`` works), path-free, no secrets."""
    snap = dict(
        schema=SCHEMA,
        taken_at_epoch_s=_r(time.time(), 0) if include_clock else None,
        purpose="read-only resource assessment; no reconstruction, calibration or visitor data was touched",
        platform=dict(system=measured(platform.system()), machine=measured(platform.machine())),
        memory=memory_section(cgroup_root, proc_root, coordinator, env),
        cpu=cpu_section(cgroup_root, proc_root),
        disk=disk_section(tmp_dir, mounts),
        worker_environment=worker_section(probe_processes, spawn_probe),
        local_reference=dict(kind="estimated", basis="local macOS measurements, NOT Streamlit Cloud", **LOCAL_REFERENCE),
        planning_values=dict(kind="configured", reconstruction_request_mb=3072.0, reconstruction_minimum_mb=1024.0, calibration_request_mb=2048.0, calibration_minimum_mb=1024.0,
                             fallback_total_mb=RB.FALLBACK_TOTAL_MB, note="shown only; this snapshot changes none of them"),
    )
    return scrub(snap)
