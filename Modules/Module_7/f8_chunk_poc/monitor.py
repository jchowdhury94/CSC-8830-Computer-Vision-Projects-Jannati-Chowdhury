"""Stage F8.5: resource monitoring for the calibration worker (NO Streamlit import; standard library only).

Measures, per processing stage: wall-clock time, parent RSS, worker-tree RSS, worker-tree PSS (Linux ``smaps_rollup``; PSS splits shared pages
between processes so it is not double-counted), container (cgroup v2 / v1) memory and CPU throttling, CPU seconds, and temporary-disk usage.
Every metric is optional: anything the platform cannot provide is reported as None, never guessed. Samples are bounded (a ring that thins
itself) and nothing here reads or logs file contents or environment secrets. Roots (/proc, /sys/fs/cgroup) are parameters so the Linux
code paths are testable on any machine.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MiB = 1024 * 1024
_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") and "SC_CLK_TCK" in getattr(os, "sysconf_names", {}) else 100


def _read(path, limit=65536):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read(limit)
    except (OSError, ValueError):
        return None


def _num(text):
    try:
        return float(str(text).strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return None


# ------------------------------------------------------------------ process table
def snapshot_processes(proc_root: str = "/proc") -> dict:
    """{pid: dict(ppid, rss_mb, cpu_s)} for every visible process. Linux: /proc. Elsewhere: one ``ps`` call. {} if neither works."""
    out = {}
    if Path(proc_root, "self").exists():
        page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
        try:
            names = [n for n in os.listdir(proc_root) if n.isdigit()]
        except OSError:
            names = []
        for n in names:
            stat = _read(f"{proc_root}/{n}/stat", 4096)
            if not stat:
                continue
            try:                                                      # the command name (field 2) may contain spaces/parentheses: split after the last ')'
                rest = stat[stat.rindex(")") + 2:].split()
                ppid, utime, stime = int(rest[1]), int(rest[11]), int(rest[12])
            except (ValueError, IndexError):
                continue
            statm = _read(f"{proc_root}/{n}/statm", 256)
            rss = int(statm.split()[1]) * page / MiB if statm and len(statm.split()) > 1 else None
            out[int(n)] = dict(ppid=ppid, rss_mb=rss, cpu_s=(utime + stime) / _CLK)
        return out
    try:
        txt = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,time="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    for line in txt.splitlines():
        f = line.split()
        if len(f) < 4:
            continue
        try:
            t = f[3].replace("-", ":").split(":"); secs = 0.0
            for part in t:
                secs = secs * 60 + float(part)
            out[int(f[0])] = dict(ppid=int(f[1]), rss_mb=int(f[2]) / 1024.0, cpu_s=secs)
        except ValueError:
            continue
    return out


def tree_pids(table: dict, root_pid: int) -> list:
    """root_pid and all of its descendants present in the table."""
    kids = {}
    for pid, info in table.items():
        kids.setdefault(info["ppid"], []).append(pid)
    out, stack = [], [root_pid]
    while stack:
        p = stack.pop()
        if p in table or p == root_pid:
            out.append(p)
        stack.extend(kids.get(p, []))
    return out


def read_pss_mb(pid: int, proc_root: str = "/proc"):
    """Proportional set size in MB from smaps_rollup (Linux). None if unavailable."""
    txt = _read(f"{proc_root}/{pid}/smaps_rollup", 8192)
    if not txt:
        return None
    for line in txt.splitlines():
        if line.startswith("Pss:"):
            v = _num(line.split(":", 1)[1])
            return None if v is None else v / 1024.0                  # kB -> MB
    return None


# ------------------------------------------------------------------ cgroup (container) memory and CPU
def _cgroup_dir(root: str, proc_root: str = "/proc") -> Path:
    """The directory holding this process's own cgroup files. Usually ``root`` itself (a container with its own cgroup namespace); when ``root`` only holds
    the host's hierarchy, follow the ``0::/path`` line of /proc/self/cgroup (cgroup v2). Falls back to ``root``; never leaves it (no '..')."""
    r = Path(root)
    if (r / "memory.current").exists() or (r / "memory" / "memory.usage_in_bytes").exists():
        return r
    for line in (_read(f"{proc_root}/self/cgroup", 4096) or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
            rel = parts[2].strip().strip("/")
            if rel and ".." not in rel.split("/") and (r / rel / "memory.current").exists():
                return r / rel
    return r


def read_cgroup(root: str = "/sys/fs/cgroup", proc_root: str = "/proc") -> dict:
    """Container-level memory / CPU. Supports cgroup v2 and v1. Missing files give None; ``version`` is None when there is no cgroup."""
    r = _cgroup_dir(root, proc_root); out = dict(version=None, memory_current_mb=None, memory_peak_mb=None, memory_max_mb=None, anon_mb=None, file_mb=None, shmem_mb=None,
                               cpu_usage_s=None, nr_periods=None, nr_throttled=None, throttled_s=None, cpu_limit_cores=None, oom_events=None)
    mb = lambda v: None if v is None else v / MiB
    if (r / "memory.current").exists():                               # cgroup v2
        out["version"] = 2
        out["memory_current_mb"] = mb(_num(_read(r / "memory.current")))
        out["memory_peak_mb"] = mb(_num(_read(r / "memory.peak")))
        mx = _read(r / "memory.max")
        out["memory_max_mb"] = None if mx is None or mx.strip() == "max" else mb(_num(mx))
        for line in (_read(r / "memory.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k == "anon": out["anon_mb"] = mb(_num(v))
            elif k == "file": out["file_mb"] = mb(_num(v))
            elif k == "shmem": out["shmem_mb"] = mb(_num(v))
        for line in (_read(r / "cpu.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k == "usage_usec": out["cpu_usage_s"] = (_num(v) or 0) / 1e6
            elif k == "nr_periods": out["nr_periods"] = _num(v)
            elif k == "nr_throttled": out["nr_throttled"] = _num(v)
            elif k == "throttled_usec": out["throttled_s"] = (_num(v) or 0) / 1e6
        cm = (_read(r / "cpu.max") or "").split()
        if len(cm) == 2 and cm[0] != "max" and _num(cm[1]):
            out["cpu_limit_cores"] = (_num(cm[0]) or 0) / _num(cm[1])
        for line in (_read(r / "memory.events") or "").splitlines():
            if line.startswith("oom_kill "):
                out["oom_events"] = _num(line.split()[1])
    elif (r / "memory" / "memory.usage_in_bytes").exists():           # cgroup v1
        out["version"] = 1
        out["memory_current_mb"] = mb(_num(_read(r / "memory" / "memory.usage_in_bytes")))
        out["memory_peak_mb"] = mb(_num(_read(r / "memory" / "memory.max_usage_in_bytes")))
        lim = _num(_read(r / "memory" / "memory.limit_in_bytes"))
        out["memory_max_mb"] = None if lim is None or lim > 1 << 60 else mb(lim)
        for line in (_read(r / "memory" / "memory.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k in ("rss", "total_rss"): out["anon_mb"] = mb(_num(v))
            elif k in ("cache", "total_cache"): out["file_mb"] = mb(_num(v))
        cpu = r / "cpu"
        for line in (_read(cpu / "cpu.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k == "nr_periods": out["nr_periods"] = _num(v)
            elif k == "nr_throttled": out["nr_throttled"] = _num(v)
            elif k == "throttled_time": out["throttled_s"] = (_num(v) or 0) / 1e9
        q, p = _num(_read(cpu / "cpu.cfs_quota_us")), _num(_read(cpu / "cpu.cfs_period_us"))
        if q and p and q > 0:
            out["cpu_limit_cores"] = q / p
    return out


def mount_info(path: str, mountinfo: str = "/proc/self/mounts") -> dict:
    """File system type and options of the mount holding ``path`` (Linux). Answers the 'is /tmp memory-backed?' question; {} if unknown."""
    txt = _read(mountinfo, 1 << 20)
    if not txt:
        return {}
    try:
        real = os.path.realpath(path)
    except OSError:
        real = str(path)
    best = None
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 4 and (real == f[1] or real.startswith(f[1].rstrip("/") + "/")):
            if best is None or len(f[1]) >= len(best[1]):
                best = f
    return {} if best is None else dict(mount_point=best[1], fstype=best[2], memory_backed=best[2] in ("tmpfs", "ramfs"), options=best[3][:120])


def probe_environment(tmp_dir=None, cgroup_root="/sys/fs/cgroup", mounts="/proc/self/mounts") -> dict:
    """A one-shot description of where we are running. Nothing secret: no environment variables are included."""
    tmp = str(tmp_dir or os.environ.get("TMPDIR") or "/tmp")
    try:
        du = shutil.disk_usage(tmp); disk = dict(total_mb=round(du.total / MiB), free_mb=round(du.free / MiB))
    except OSError:
        disk = {}
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return dict(platform=platform.platform(), machine=platform.machine(), python=sys.version.split()[0], cpu_count=os.cpu_count(), cpu_affinity=affinity,
                cgroup=read_cgroup(cgroup_root), temp_dir_mount=mount_info(tmp, mounts), temp_disk=disk)


# ------------------------------------------------------------------ the monitor
class ResourceMonitor:
    """Samples every ``interval`` s while running. ``mark(stage)`` starts a new stage; ``summary()`` returns per-stage aggregates.

    worker_pid: callable returning the worker's pid or None. disk: callable returning (temp_used_mb, free_mb)."""

    def __init__(self, worker_pid=lambda: None, disk=lambda: (None, None), interval: float = 1.0, parent_pid: int | None = None, proc_root: str = "/proc",
                 cgroup_root: str = "/sys/fs/cgroup", max_samples: int = 600, clock=time.monotonic, snapshot=None, sample_on_mark: bool = True):
        self._worker_pid, self._disk, self.interval = worker_pid, disk, max(0.05, float(interval))
        self._sample_on_mark = sample_on_mark                           # a sample at every stage boundary, so even a stage shorter than the interval has a start and an end
        self._parent = parent_pid or os.getpid()
        self._proc_root, self._cg_root, self._clock = proc_root, cgroup_root, clock
        self._read_cg = lambda: read_cgroup(cgroup_root, proc_root)
        self._snapshot = snapshot or (lambda: snapshot_processes(proc_root))
        self.max_samples = max(10, int(max_samples)); self.samples: list = []; self._thin = 1; self._tick = 0
        self._lock = threading.Lock(); self._stop = threading.Event(); self._thread = None
        self.stages: list = []; self._cur = None; self.latest: dict = {}
        self._t0 = self._clock(); self._cg0 = self._read_cg()

    # ---- stages
    def mark(self, stage: str):
        if self._sample_on_mark:
            self._safe_sample()                                         # closing sample of the previous stage
        now = self._clock()
        with self._lock:
            if self._cur is not None:
                self._close_stage(now)
            self._cur = dict(stage=str(stage)[:40], t0=now, n=0, parent_rss_max=None, parent_rss_sum=0.0, worker_rss_max=None, tree_rss_max=None, tree_pss_max=None,
                             cgroup_current_max=None, cpu_start=None, cpu_last=None, throttled_start=None, throttled_last=None, disk_used_max=None, disk_free_min=None,
                             cgroup_peak_end=None)
        if self._sample_on_mark:
            self._safe_sample()                                         # opening sample of the new stage

    def _safe_sample(self):
        try:
            self.sample_once()
        except Exception:                                               # a monitor must never break a job
            pass

    def _close_stage(self, now):
        c = self._cur; self._cur = None
        cg = self._read_cg()
        c["cgroup_peak_end"] = cg.get("memory_peak_mb")
        self.stages.append(dict(
            stage=c["stage"], wall_s=round(now - c["t0"], 3), samples=c["n"], parent_rss_max_mb=_r(c["parent_rss_max"]),
            parent_rss_mean_mb=_r(c["parent_rss_sum"] / c["n"]) if c["n"] else None, worker_rss_max_mb=_r(c["worker_rss_max"]),
            tree_rss_max_mb=_r(c["tree_rss_max"]), tree_pss_max_mb=_r(c["tree_pss_max"]), cgroup_memory_max_mb=_r(c["cgroup_current_max"]),
            cgroup_peak_at_stage_end_mb=_r(c["cgroup_peak_end"]),
            cpu_s=_r(None if c["cpu_start"] is None or c["cpu_last"] is None else max(c["cpu_last"] - c["cpu_start"], 0.0)),
            throttled_periods=None if c["throttled_start"] is None or c["throttled_last"] is None else int(c["throttled_last"] - c["throttled_start"]),
            temp_disk_max_mb=_r(c["disk_used_max"]), disk_free_min_mb=_r(c["disk_free_min"])))

    # ---- sampling
    def sample_once(self) -> dict:
        table = self._snapshot() or {}
        wp = self._worker_pid()
        parent = table.get(self._parent, {}).get("rss_mb")
        tree = tree_pids(table, wp) if wp and wp in table else []
        worker_rss = sum((table[p]["rss_mb"] or 0.0) for p in tree) if tree else None
        pss_vals = [read_pss_mb(p, self._proc_root) for p in tree]
        pss = sum(v for v in pss_vals if v is not None) if tree and all(v is not None for v in pss_vals) else None
        # combined tree = parent + worker tree (parent has no PSS here unless Linux; use RSS for the parent and flag the mix)
        parent_pss = read_pss_mb(self._parent, self._proc_root)
        combined_pss = None if pss is None or parent_pss is None else pss + parent_pss
        cg = self._read_cg()
        used, free = self._disk()
        cpu = sum(table[p]["cpu_s"] for p in tree) if tree else None
        s = dict(t=round(self._clock() - self._t0, 2), parent_rss_mb=_r(parent), worker_rss_mb=_r(worker_rss), tree_pss_mb=_r(pss), combined_pss_mb=_r(combined_pss),
                 combined_rss_mb=_r((parent or 0) + (worker_rss or 0)) if parent is not None else None, cgroup_current_mb=_r(cg.get("memory_current_mb")),
                 cgroup_anon_mb=_r(cg.get("anon_mb")), cgroup_file_mb=_r(cg.get("file_mb")), nr_throttled=cg.get("nr_throttled"), temp_disk_mb=_r(used), disk_free_mb=_r(free))
        with self._lock:
            self.latest = s
            c = self._cur
            if c is not None:
                c["n"] += 1
                _max(c, "parent_rss_max", parent); c["parent_rss_sum"] += parent or 0.0
                _max(c, "worker_rss_max", worker_rss)
                _max(c, "tree_rss_max", s["combined_rss_mb"]); _max(c, "tree_pss_max", combined_pss if combined_pss is not None else pss)
                _max(c, "cgroup_current_max", cg.get("memory_current_mb")); _max(c, "disk_used_max", used)
                if free is not None and (c["disk_free_min"] is None or free < c["disk_free_min"]): c["disk_free_min"] = free
                if cpu is not None:
                    if c["cpu_start"] is None: c["cpu_start"] = cpu
                    c["cpu_last"] = cpu
                if cg.get("nr_throttled") is not None:
                    if c["throttled_start"] is None: c["throttled_start"] = cg["nr_throttled"]
                    c["throttled_last"] = cg["nr_throttled"]
            self._tick += 1
            if self._tick % self._thin == 0:
                self.samples.append(s)
                if len(self.samples) > self.max_samples:             # bounded: drop every second sample and halve the future rate
                    self.samples = self.samples[::2]; self._thin *= 2
        return s

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:                                         # a monitor must never break a job
                pass
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="f8-monitor"); self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            self.sample_once()
        except Exception:
            pass
        with self._lock:
            if self._cur is not None:
                self._close_stage(self._clock())
        return self.summary()

    def summary(self) -> dict:
        with self._lock:
            st = list(self.stages)
        cg1 = self._read_cg()

        def mx(k):
            v = [x[k] for x in st if x.get(k) is not None]
            return max(v) if v else None
        return dict(stages=st, total_wall_s=round(self._clock() - self._t0, 2), peak=dict(
            parent_rss_mb=mx("parent_rss_max_mb"), worker_rss_mb=mx("worker_rss_max_mb"), tree_rss_mb=mx("tree_rss_max_mb"), tree_pss_mb=mx("tree_pss_max_mb"),
            cgroup_memory_mb=mx("cgroup_memory_max_mb"), cgroup_peak_mb=cg1.get("memory_peak_mb"), temp_disk_mb=mx("temp_disk_max_mb")),
            cgroup=dict(version=cg1.get("version"), memory_max_mb=cg1.get("memory_max_mb"), cpu_limit_cores=cg1.get("cpu_limit_cores"),
                        oom_events=cg1.get("oom_events"), nr_throttled_total=None if cg1.get("nr_throttled") is None or self._cg0.get("nr_throttled") is None
                        else int(cg1["nr_throttled"] - self._cg0["nr_throttled"])),
            available=dict(pss=any(x.get("tree_pss_max_mb") is not None for x in st), cgroup=cg1.get("version") is not None),
            cpu=self._cpu_summary(st, cg1), samples=len(self.samples))

    def _cpu_summary(self, st, cg1) -> dict:
        """CPU use and throttling over the monitored period. Every field is None when the platform cannot say (no cgroup, no /proc)."""
        wall = max(self._clock() - self._t0, 1e-9)
        tree = [x["cpu_s"] for x in st if x.get("cpu_s") is not None]
        d = lambda k: None if cg1.get(k) is None or self._cg0.get(k) is None else cg1[k] - self._cg0[k]
        cg_cpu, periods, thr, thr_s = d("cpu_usage_s"), d("nr_periods"), d("nr_throttled"), d("throttled_s")
        return dict(cgroup_cpu_measurable=cg_cpu is not None, throttling_measurable=thr is not None, worker_tree_cpu_s=_r(sum(tree), 2) if tree else None, worker_tree_avg_cores=_r(sum(tree) / wall, 2) if tree else None,
                    cgroup_cpu_s=_r(cg_cpu, 2), cgroup_avg_cores=None if cg_cpu is None else _r(cg_cpu / wall, 2), cpu_limit_cores=cg1.get("cpu_limit_cores"),
                    host_cpu_count=os.cpu_count(), periods=None if periods is None else int(periods), throttled_periods=None if thr is None else int(thr),
                    throttled_s=_r(thr_s, 3), throttled_fraction=None if not periods or thr is None else _r(thr / periods, 4))

    def timeline(self, n: int = 120) -> list:
        with self._lock:
            s = list(self.samples)
        if len(s) <= n:
            return s
        step = len(s) / n
        return [s[int(i * step)] for i in range(n)]


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def _max(d, key, v):
    if v is not None and (d[key] is None or v > d[key]):
        d[key] = v
