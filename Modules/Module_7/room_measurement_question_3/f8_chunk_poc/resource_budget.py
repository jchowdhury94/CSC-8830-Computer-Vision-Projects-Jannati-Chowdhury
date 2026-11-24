"""F8.31C-1C: container-aware memory budget and application-wide worker coordination (NO Streamlit import; standard library plus ``monitor``).

Why this exists.  The calibration runner and the room-reconstruction backend each started sandboxed workers with their own, independent memory limit
(the reconstruction default, 6144 MB, was never matched to the real container) and the reconstruction limit covered the WORKER tree only.  This module
gives both ONE budget:

    container limit (cgroup v2 / v1, nearest ancestor limit)  ->  total   (or an explicit, documented conservative fallback when it cannot be read)
    total - safety reserve                                      ->  usable
    usable - Streamlit parent RSS - every active reservation    ->  available for the next worker

When it is CERTAIN that no container limit applies (macOS or Windows, which have no Linux cgroups, or a Linux cgroup that is explicitly unlimited) the computer's own physical
memory is the total (``host_memory_mb``), and the memory that is free right now is checked again at every admission, because other programs may be using part of it.
When the limit merely could not be read (an error, an unknown platform) nothing is assumed: the conservative fallback or the operator's ceiling applies, as before.

``reserve()`` answers immediately (never queues): GRANTED with a ceiling (<= what was asked, <= available), BUSY (another job of the same kind holds the
single slot) or INSUFFICIENT_MEMORY (not enough budget left).  A reservation is a small file in a per-user 0700 directory on which the holder keeps an
exclusive ``flock``: the kernel drops the lock when the holder dies, so a crashed, killed or timed-out process can never leave a permanent reservation
(the next admission removes the orphan file).  A Python module-level lock would only coordinate threads of ONE process; this coordinates processes.

What this is NOT: it is not hard enforcement.  The ceiling is enforced by sampling (supervisor poll, adaptive) plus an in-worker peak-RSS watchdog, and the
container's own OOM killer is the only real hard limit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from monitor import _cgroup_dir, _num, _read, read_cgroup, snapshot_processes

try:
    import fcntl
except ImportError:                                                    # pragma: no cover (Windows): in-process coordination only, reported as such
    fcntl = None

MiB = 1024 * 1024
# ---- explicit, documented defaults (planning values, NOT acceptance thresholds on any measurement)
FALLBACK_TOTAL_MB = 2048.0          # used ONLY when no container limit can be read and nothing is configured.  Conservative on purpose: not a claim about Streamlit Cloud.
RESERVE_FLOOR_MB = 256.0            # kept free for the Python runtime, page cache pressure and everything not accounted for
RESERVE_FRACTION = 0.10             # ... or this fraction of the total, whichever is larger
ASSUMED_PARENT_MB = 512.0           # used only when the parent's own RSS cannot be read (fail towards less memory for workers)
CEILING_ENV, RESERVE_ENV = "F8_APP_MEMORY_CEILING_MB", "F8_MEMORY_RESERVE_MB"
CEILING_RANGE_MB, RESERVE_RANGE_MB = (512.0, 262144.0), (64.0, 16384.0)
HEAVY_KINDS = ("calibration", "reconstruction")   # a worker that runs for minutes and needs 1-3 GB
LIGHT_KINDS = ("frames", "validation")            # short decode workers: frame extraction / previews and upload inspection.  They used to be outside this budget altogether.
KINDS = HEAVY_KINDS + LIGHT_KINDS
MAX_ACTIVE_PER_KIND = 1             # one calibration and one reconstruction at a time; no queue
MAX_LIGHT_ENV, MAX_LIGHT_RANGE = "F8_MAX_LIGHT_JOBS", (1, 8)
MAX_LIGHT_DEFAULT = 2               # frame / validation jobs that may run together (not per kind: they are the same kind of load)
# (desired, minimum) MB a light job claims.  PLANNING values, not measurements: the RSS of a decode worker has not been measured on Cloud (an estimate is 0.3-0.8 GB; see the Phase D report).
# A claim is an accounting entry: it keeps heavy jobs from being admitted on top of it.  It is not an enforced ceiling (those workers keep their own timeout and address-space cap).
LIGHT_CLAIMS_MB = {"frames": (768.0, 512.0), "validation": (512.0, 256.0)}
GRANTED, BUSY, INSUFFICIENT_MEMORY = "GRANTED", "BUSY", "INSUFFICIENT_MEMORY"


def _env_mb(env, name, rng):
    try:
        v = float(str(env[name]).strip())
    except (KeyError, ValueError):
        return None
    return v if math.isfinite(v) and rng[0] <= v <= rng[1] else None


@dataclass(frozen=True)
class BudgetConfig:
    fallback_total_mb: float = FALLBACK_TOTAL_MB
    reserve_floor_mb: float = RESERVE_FLOOR_MB
    reserve_fraction: float = RESERVE_FRACTION
    ceiling_mb: float | None = None          # the configured application safety ceiling (an operator's statement; never exceeded)
    assumed_parent_mb: float = ASSUMED_PARENT_MB
    max_light_jobs: int = MAX_LIGHT_DEFAULT  # frame / validation workers allowed at once

    @classmethod
    def from_env(cls, env=None):
        """``F8_APP_MEMORY_CEILING_MB`` (512-262144), ``F8_MEMORY_RESERVE_MB`` (64-16384), ``F8_MAX_LIGHT_JOBS`` (1-8).  Invalid values are ignored (never silently clamped)."""
        env = os.environ if env is None else env
        light = _env_mb(env, MAX_LIGHT_ENV, MAX_LIGHT_RANGE)
        return cls(ceiling_mb=_env_mb(env, CEILING_ENV, CEILING_RANGE_MB), reserve_floor_mb=_env_mb(env, RESERVE_ENV, RESERVE_RANGE_MB) or RESERVE_FLOOR_MB,
                   max_light_jobs=int(light) if light is not None and float(light).is_integer() else MAX_LIGHT_DEFAULT)


# ------------------------------------------------------------------ container limit discovery
def _ancestor_limits_v2(start: Path, root: Path) -> list:
    """memory.max of ``start`` and every parent directory up to (and including) ``root``; 'max' entries are skipped.  Never leaves ``root`` (no '..')."""
    out, p = [], start
    for _ in range(64):
        v = _read(p / "memory.max")
        if v is not None and v.strip() not in ("max", ""):
            n = _num(v)
            if n is not None and n > 0:
                out.append(n / MiB)
        if p == root or p.parent == p:
            break
        p = p.parent
        try:
            p.relative_to(root)
        except ValueError:
            break
    return out


def _explicitly_unlimited(version, root: Path, cgroup_root: str, proc_root: str) -> bool:
    """True only when the cgroup itself SAYS it is unlimited: ``memory.max`` reads exactly "max" (v2) or ``memory.limit_in_bytes`` (and the hierarchical limit) carry the kernel's
    'unlimited' sentinel (v1).  A missing, empty or garbage value is "unknown", never "unlimited"."""
    try:
        if version == 2:
            v = _read(_cgroup_dir(cgroup_root, proc_root) / "memory.max")
            return v is not None and v.strip() == "max"
        lim = _num(_read(root / "memory" / "memory.limit_in_bytes"))
        hier = None
        for line in (_read(root / "memory" / "memory.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k == "hierarchical_memory_limit":
                hier = _num(v)
        return lim is not None and lim >= 1 << 60 and (hier is None or hier >= 1 << 60)
    except Exception:
        return False


def container_memory_limit(cgroup_root: str = "/sys/fs/cgroup", proc_root: str = "/proc") -> dict:
    """What the container allows.  ``limit_mb`` is None unless a real, finite limit was read; ``reason`` always says why.

    cgroup v2: the smallest ``memory.max`` from this process's cgroup up to the mount root (a limit on a parent applies to its children).
    cgroup v1: ``memory.limit_in_bytes``, or ``hierarchical_memory_limit`` from ``memory.stat`` when that is smaller; the kernel's 'unlimited' sentinel is ignored.
    Missing files, 'max', a zero or non-numeric value, or no cgroup at all give ``limit_mb = None`` (never a guess).

    ``host_bound`` is True only when it is CERTAIN that no container limit applies, so the computer's own memory is the real bound: a platform that has no Linux cgroups at all
    (macOS, Windows), or a Linux cgroup that explicitly says it is unlimited.  An unreadable or garbage limit is NOT host-bound."""
    root = Path(cgroup_root)
    try:
        cg = read_cgroup(cgroup_root, proc_root)
    except Exception as ex:                                            # an unreadable cgroup is "unknown", never an error
        return dict(limit_mb=None, version=None, source=None, reliable=False, reason=f"cgroup not readable ({type(ex).__name__})", current_mb=None, host_bound=False)
    version = cg.get("version")
    if version is None:
        return dict(limit_mb=None, version=None, source=None, reliable=False, reason="no cgroup memory controller visible (not a Linux container, or /sys/fs/cgroup is not mounted)", current_mb=None,
                    host_bound=not sys.platform.startswith("linux"))      # macOS / Windows cannot be a Linux container; on Linux a missing controller is "unknown"
    cands = []
    if version == 2:
        d = _cgroup_dir(cgroup_root, proc_root)
        cands = _ancestor_limits_v2(d, root)
    else:
        if cg.get("memory_max_mb"):
            cands.append(float(cg["memory_max_mb"]))
        for line in (_read(root / "memory" / "memory.stat") or "").splitlines():
            k, _, v = line.partition(" ")
            if k == "hierarchical_memory_limit":
                n = _num(v)
                if n is not None and 0 < n <= 1 << 60:
                    cands.append(n / MiB)
    cands = [c for c in cands if math.isfinite(c) and c > 0]
    cur = cg.get("memory_current_mb")
    if not cands:
        return dict(limit_mb=None, version=version, source=f"cgroup_v{version}", reliable=False, reason=f"cgroup v{version} present but no memory limit is set (unlimited)", current_mb=cur,
                    host_bound=_explicitly_unlimited(version, root, cgroup_root, proc_root))
    return dict(limit_mb=min(cands), version=version, source=f"cgroup_v{version}", reliable=True, reason=f"cgroup v{version} memory limit", current_mb=cur, host_bound=False)


def container_used_mb(cg: dict):
    """Memory the container is really using now, from a ``read_cgroup`` dict: anonymous + shmem (page cache is reclaimable and not counted); else current minus file cache; else None."""
    if not isinstance(cg, dict):
        return None
    g = lambda k: cg.get(k) if isinstance(cg.get(k), (int, float)) and math.isfinite(cg.get(k)) and cg.get(k) >= 0 else None
    anon, shm, cur, fil = g("anon_mb"), g("shmem_mb"), g("memory_current_mb"), g("file_mb")
    if anon is not None:
        return anon + (shm or 0.0)
    if cur is not None and fil is not None and fil <= cur:
        return cur - fil + (shm or 0.0)
    return cur


# ------------------------------------------------------------------ the computer's own memory (used ONLY when it is certain that no container limit applies)
HOST_TTL_S = 1.0                                   # how long a reading of the free memory is reused (reading it can start a short child process)
_host_cache = [0.0, None]
_host_lock = threading.Lock()


def parse_vm_stat(text: str):
    """macOS ``vm_stat`` output -> megabytes the system can hand out without swapping: free + inactive + speculative pages.  Purgeable pages are deliberately NOT added (they may
    already be counted in the others), so the figure errs towards less.  None if the text is not recognisable."""
    m = re.search(r"page size of (\d+) bytes", text or "")
    if not m:
        return None
    page, total = int(m.group(1)), 0
    found = False
    for key in ("Pages free", "Pages inactive", "Pages speculative"):
        mm = re.search(rf"{key}:\s+(\d+)", text)
        if mm:
            total += int(mm.group(1)); found = True
    return total * page / MiB if found and page > 0 else None


def parse_meminfo(text: str):
    """Linux ``/proc/meminfo`` -> (MemTotal MB, MemAvailable MB); either is None when absent."""
    kv = {}
    for line in (text or "").splitlines():
        k, _, v = line.partition(":")
        n = _num(v)
        if n is not None:
            kv[k.strip()] = n / 1024.0                                   # kB -> MB
    return kv.get("MemTotal"), kv.get("MemAvailable")


def _host_total_mb():
    try:
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return pages * size / MiB
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "win32":                                          # pragma: no cover (Windows)
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullTotalPhys / MiB
        except Exception:
            pass
    return None


def _host_available_mb():
    try:
        if sys.platform.startswith("linux"):
            return parse_meminfo(_read("/proc/meminfo") or "")[1]
        if sys.platform == "darwin":
            out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=3, check=False)    # fixed absolute path, no shell
            return parse_vm_stat(out.stdout) if out.returncode == 0 else None
        if sys.platform == "win32":                                      # pragma: no cover (Windows)
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
            return ms.ullAvailPhys / MiB if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None
    except Exception:
        return None
    return None


def host_memory_mb() -> dict:
    """The computer's physical memory and the memory that is free now: ``dict(total_mb, available_mb, source)``; a figure that cannot be read is None.
    Standard library only.  The free-memory reading is reused for ``HOST_TTL_S`` seconds."""
    with _host_lock:
        now = time.monotonic()
        if _host_cache[1] is not None and now - _host_cache[0] < HOST_TTL_S:
            return dict(_host_cache[1])
        tot, av = _pos(_host_total_mb()), _pos(_host_available_mb())
        if tot is not None and av is not None and av > tot:
            av = tot
        d = dict(total_mb=tot, available_mb=av, source=sys.platform)
        _host_cache[0], _host_cache[1] = now, d
        return dict(d)


def compute_budget(cfg: BudgetConfig, container: dict, host: dict | None = None) -> dict:
    """total / reserve / usable and WHERE the total came from.  ``total_mb`` never exceeds the configured ceiling.

    ``host`` (``host_memory_mb()``) is consulted only when the container report says it is CERTAIN that no container limit applies (``host_bound``); in every other case where the
    limit is unknown the behaviour is what it always was: the operator's ceiling, else the conservative fallback."""
    lim = container.get("limit_mb") if container.get("reliable") else None
    host_total = _pos(host.get("total_mb")) if isinstance(host, dict) and container.get("host_bound") else None
    if lim is not None:
        total, source = lim, container.get("source") or "cgroup"
        if cfg.ceiling_mb is not None and cfg.ceiling_mb < total:
            total, source = cfg.ceiling_mb, "configured_ceiling (below the container limit)"
        limitation = None
    elif host_total:
        total, source = host_total, "host_physical_memory"
        if cfg.ceiling_mb is not None and cfg.ceiling_mb < total:
            total, source = cfg.ceiling_mb, "configured_ceiling (below this computer's physical memory)"
        limitation = ("no container memory limit applies, so this computer's physical memory is the budget; the memory that is free is checked again at every start, because other programs "
                      "may be using part of it")
    elif cfg.ceiling_mb is not None:
        total, source = cfg.ceiling_mb, "configured_ceiling (operator-asserted; the container limit could not be read: " + str(container.get("reason")) + ")"
        limitation = "the container memory limit could not be determined; the configured ceiling is the operator's statement and is not verified"
    else:
        total, source = cfg.fallback_total_mb, "conservative_fallback"
        limitation = (f"the container memory limit could not be determined ({container.get('reason')}); a conservative fallback of {cfg.fallback_total_mb:.0f} MB is used. "
                      "It is not a measurement of any deployment: set F8_APP_MEMORY_CEILING_MB, or measure the container")
    reserve = min(max(cfg.reserve_floor_mb, cfg.reserve_fraction * total), total)
    return dict(total_mb=float(total), reserve_mb=float(reserve), usable_mb=float(max(total - reserve, 0.0)), source=source, limitation=limitation,
                container=dict(container), ceiling_mb=cfg.ceiling_mb, fallback_total_mb=cfg.fallback_total_mb)


# ------------------------------------------------------------------ admission result and reservation
@dataclass
class Admission:
    outcome: str                                  # GRANTED | BUSY | INSUFFICIENT_MEMORY
    granted_mb: float | None = None
    reservation: "Reservation | None" = None
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome == GRANTED


class Reservation:
    """A granted claim.  ``release()`` is idempotent; a ``with`` block, a garbage-collected object or the death of the process all release it."""

    def __init__(self, coordinator, path, fh, kind, granted_mb):
        self._coord, self.path, self._fh, self.kind, self.granted_mb = coordinator, path, fh, kind, float(granted_mb)
        self.released = False
        self._lock = threading.Lock()

    def release(self):
        with self._lock:
            if self.released:
                return
            self.released = True
        self._coord._drop(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def __del__(self):                                                  # a leaked object must not hold memory forever
        try:
            self.release()
        except Exception:
            pass


class ResourceCoordinator:
    """Admission control for workers, shared by every process of the application that uses the same ``root``.

    ``snapshot`` / ``cgroup_root`` / ``proc_root`` / ``container`` are injectable so the Linux paths and every outcome are testable on any machine."""

    def __init__(self, root=None, cfg: BudgetConfig | None = None, *, cgroup_root="/sys/fs/cgroup", proc_root="/proc", snapshot=None, container=None, pid=None, host_memory=None):
        self.cfg = cfg or BudgetConfig.from_env()
        self._cg_root, self._proc_root = cgroup_root, proc_root
        self._snapshot = snapshot or (lambda: snapshot_processes(proc_root))
        self._container_override = container
        self._host_memory = host_memory or host_memory_mb                # injectable: the computer's total and free memory (used only when no container limit applies)
        self._pid = int(pid or os.getpid())
        self._tlock = threading.RLock()
        self._mem = weakref.WeakValueDictionary()                       # in-process registry (the only registry without fcntl).  Weak: a leaked Reservation releases itself when collected
        self.cross_process = fcntl is not None
        self.root = self._init_root(root)

    # ---- directory
    def _init_root(self, root):
        if fcntl is None:
            return None
        try:
            p = Path(root) if root is not None else Path(tempfile.gettempdir()) / f"f8_resource_budget_{os.getuid()}"
            p.mkdir(mode=0o700, parents=True, exist_ok=True)
            st = p.lstat()
            if p.is_symlink() or st.st_uid != os.getuid():             # somebody else's directory or a planted link: do not use it
                raise OSError("unsafe coordination directory")
            os.chmod(p, 0o700)
            return p
        except OSError:
            self.cross_process = False
            return None

    # ---- state
    def container(self) -> dict:
        return dict(self._container_override) if self._container_override is not None else container_memory_limit(self._cg_root, self._proc_root)

    def budget(self) -> dict:
        c = self.container()
        host = None
        if c.get("host_bound") and not c.get("reliable"):                # only when it is certain that no container limit applies
            try:
                host = self._host_memory()
            except Exception:
                host = None                                              # unreadable: the conservative fallback / operator ceiling applies, exactly as before
        return compute_budget(self.cfg, c, host)

    def _active_records(self) -> list:
        """Active reservations of ALL processes (caller holds the admission lock).  Orphans (holder dead: their flock was released by the kernel) are deleted."""
        out = []
        with self._tlock:
            mine = {r.path: r for r in list(self._mem.values())}
        if self.root is not None:
            try:
                names = sorted(n for n in os.listdir(self.root) if n.endswith(".res"))
            except OSError:
                names = []
            for n in names:
                p = self.root / n
                try:
                    fh = open(p, "r+b")
                except OSError:
                    continue
                try:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError:                                    # held by a live holder: active
                        fh.seek(0)
                        try:
                            d = json.loads(fh.read(4096).decode() or "{}")
                        except ValueError:
                            d = {}
                        out.append(dict(path=str(p), kind=str(d.get("kind", "?")), granted_mb=_pos(d.get("granted_mb")), pid=_int(d.get("pid")), owner=str(d.get("owner", ""))[:16], unreadable=not d))
                        continue
                    try:                                               # we got the lock: nobody holds it -> orphan
                        if str(p) not in {str(k) for k in mine}:
                            p.unlink()
                    except OSError:
                        pass
                    finally:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()
        else:
            with self._tlock:
                out = [dict(path=str(r.path), kind=r.kind, granted_mb=r.granted_mb, pid=self._pid, owner="", unreadable=False) for r in list(self._mem.values()) if not r.released]
        return out

    def state(self) -> dict:
        """A read-only picture (no owner identifiers): budget, parent RSS, active reservations and what a new worker could get now."""
        with self._admission():
            s = self._state_locked()
        s.pop("_recs", None)
        return s

    def _state_locked(self, kind=None) -> dict:
        b = self.budget(); recs = self._active_records()
        table = {}
        try:
            table = self._snapshot() or {}
        except Exception:
            table = {}
        prss = (table.get(self._pid) or {}).get("rss_mb")
        parent_measured = isinstance(prss, (int, float)) and prss > 0
        parent = float(prss) if parent_measured else self.cfg.assumed_parent_mb
        others = 0.0                                                    # other processes of the app that hold reservations: their own parent RSS is not ours to free
        for pid in {r["pid"] for r in recs if r["pid"] and r["pid"] != self._pid}:
            v = (table.get(pid) or {}).get("rss_mb")
            others += float(v) if isinstance(v, (int, float)) and v > 0 else self.cfg.assumed_parent_mb
        reserved = sum((r["granted_mb"] if r["granted_mb"] is not None else b["usable_mb"]) for r in recs)       # unreadable record: assume it holds everything (fail closed)
        accounting = b["usable_mb"] - parent - others - reserved
        live_used = container_used_mb(read_cgroup(self._cg_root, self._proc_root)) if b["container"].get("reliable") and self._container_override is None else None
        live_avail = None if live_used is None else b["usable_mb"] - live_used
        live_host = None                                                 # no container limit: the memory the computer has free RIGHT NOW (other programs may be using part of it)
        if b["container"].get("host_bound") and str(b.get("source", "")).startswith(("host_physical_memory", "configured_ceiling (below this computer")):
            try:
                av = (self._host_memory() or {}).get("available_mb")
            except Exception:
                av = None
            if isinstance(av, (int, float)) and math.isfinite(av) and av >= 0:
                live_host = av - b["reserve_mb"]                         # the same safety reserve that is kept back from the total
        candidates = [x for x in (accounting, live_avail, live_host) if x is not None]
        available = min(candidates)
        return dict(budget=b, parent_rss_mb=round(parent, 1), parent_rss_measured=parent_measured, other_process_parents_mb=round(others, 1), reserved_mb=round(reserved, 1),
                    reservations=[dict(kind=r["kind"], granted_mb=r["granted_mb"], pid=r["pid"]) for r in recs], available_mb=round(max(available, 0.0), 1), accounting_available_mb=round(accounting, 1),
                    live_container_used_mb=None if live_used is None else round(live_used, 1), live_host_available_mb=None if live_host is None else round(live_host, 1),
                    cross_process=self.cross_process, _recs=recs)

    # ---- admission lock: a short flock on one file (processes) plus an RLock (threads)
    def _admission(self):
        coord = self

        class _Lock:
            def __enter__(self_):
                coord._tlock.acquire()
                self_.fh = None
                if coord.root is not None:
                    try:
                        self_.fh = open(coord.root / ".admission.lock", "a+b"); fcntl.flock(self_.fh.fileno(), fcntl.LOCK_EX)
                    except OSError:
                        self_.fh = None
                return self_

            def __exit__(self_, *exc):
                if self_.fh is not None:
                    try:
                        fcntl.flock(self_.fh.fileno(), fcntl.LOCK_UN)
                    finally:
                        self_.fh.close()
                coord._tlock.release()
                return False
        return _Lock()

    # ---- reserve / release
    def reserve(self, kind: str, owner: str, desired_mb: float, min_mb: float) -> Admission:
        """Admit a worker now or refuse now.  ``desired_mb`` is the ceiling the job would like; the grant is ``min(desired, available)`` and is refused when it would be
        below ``min_mb``.  Never blocks beyond the short admission lock; never queues.  ``owner`` is only hashed (it is not exposed by ``state``)."""
        if kind not in KINDS:
            raise ValueError("unknown reservation kind")
        desired = float(desired_mb) if isinstance(desired_mb, (int, float)) and math.isfinite(desired_mb) and desired_mb > 0 else 0.0
        floor = float(min_mb) if isinstance(min_mb, (int, float)) and math.isfinite(min_mb) and min_mb > 0 else 0.0
        with self._admission():
            st = self._state_locked(kind); recs = st.pop("_recs")
            info = {k: v for k, v in st.items()}
            light = [r for r in recs if r["kind"] in LIGHT_KINDS]
            heavy = [r for r in recs if r["kind"] not in LIGHT_KINDS]                  # an unreadable / unknown holder counts as heavy (fail closed)
            if kind in LIGHT_KINDS:
                if heavy:                                                               # a calibration / reconstruction holds (nearly) all the memory: never start a decoder next to it
                    return Admission(BUSY, None, None, dict(info, kind=kind, busy_with="heavy", reason="a calibration or measurement job is running"))
                if len(light) >= self.cfg.max_light_jobs:
                    return Admission(BUSY, None, None, dict(info, kind=kind, busy_with="light_limit", reason=f"{len(light)} frame or video-check jobs are already running (limit {self.cfg.max_light_jobs})"))
            else:
                if light:                                                               # and never admit a big job on top of running decoders: its grant would be cut below what it needs
                    return Admission(BUSY, None, None, dict(info, kind=kind, busy_with="light", reason="a frame-extraction or video-check job is running"))
                if sum(1 for r in recs if r["kind"] == kind) >= MAX_ACTIVE_PER_KIND:
                    return Admission(BUSY, None, None, dict(info, kind=kind, busy_with="same_kind", reason=f"another {kind} job already holds the only slot"))
            avail = st["available_mb"]; granted = min(desired, avail)
            if granted <= 0 or granted < floor:
                lim = (st.get("budget") or {}).get("limitation")                        # e.g. "the container memory limit could not be determined ... a conservative fallback is used"
                return Admission(INSUFFICIENT_MEMORY, None, None, dict(info, kind=kind, requested_mb=desired, minimum_mb=floor,
                                                                        reason=f"{avail:.0f} MB are available under the memory budget; this job needs at least {floor:.0f} MB" + (f" ({lim})" if lim else "")))
            res = self._create(kind, owner, granted)
            return Admission(GRANTED, granted, res, dict(info, kind=kind, requested_mb=desired, minimum_mb=floor, granted_mb=round(granted, 1)))

    def _create(self, kind, owner, granted) -> Reservation:
        fh = path = None
        if self.root is not None:
            name = f"{kind}-{secrets.token_hex(8)}.res"; path = self.root / name
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600); fh = os.fdopen(fd, "r+b")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)      # held until release or process death
            fh.write(json.dumps(dict(kind=kind, granted_mb=round(granted, 3), pid=self._pid, owner=hashlib.sha256(str(owner).encode()).hexdigest()[:16], t=time.time())).encode()); fh.flush()
        else:
            path = Path(f"mem:{kind}-{secrets.token_hex(8)}")
        r = Reservation(self, path, fh, kind, granted)
        with self._tlock:
            self._mem[str(path)] = r
        return r

    def _drop(self, r: Reservation):
        with self._tlock:
            self._mem.pop(str(r.path), None)
        try:
            if r._fh is not None:
                try:
                    if self.root is not None:
                        os.unlink(r.path)
                except OSError:
                    pass
        finally:
            if r._fh is not None:
                try:
                    r._fh.close()                                       # closing the descriptor releases the flock
                except OSError:
                    pass
                r._fh = None


def _pos(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 else None


def _int(v):
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


# ------------------------------------------------------------------ light jobs (frame extraction, upload inspection)
class ResourceBusy(Exception):
    """A light job was refused: the server is busy or short of budget.  ``message`` is safe to show to a visitor; nothing was started and nothing is lost."""

    def __init__(self, kind: str, reason: str, busy_with: str | None = None, outcome: str = BUSY):
        super().__init__(reason)
        self.kind, self.reason, self.busy_with, self.outcome = kind, reason, busy_with, outcome
        self.message = "The server is busy with another job right now (a measurement, or another visitor's videos). Nothing was lost: please try again in a minute."


def acquire_light(kind: str, owner: str = "", coordinator: ResourceCoordinator | None = None, wait_s: float = 0.0, poll_s: float = 0.2) -> Reservation:
    """Admit a frame-extraction or upload-inspection worker, or raise ``ResourceBusy``.  Use as ``with acquire_light("frames", owner): ...``: the reservation is released on success,
    failure, timeout and cancellation alike, and by the kernel if the process dies (see the module docstring).  Waits up to ``wait_s`` for a short job to finish (so two quick previews do
    not fail each other) and never longer: a calibration or reconstruction can run for minutes, and a visitor must not be left hanging.  Fails closed if the coordinator is broken."""
    if kind not in LIGHT_KINDS:
        raise ValueError("not a light job kind")
    desired, floor = LIGHT_CLAIMS_MB[kind]
    deadline = time.monotonic() + max(float(wait_s), 0.0)
    while True:
        try:
            coord = coordinator if coordinator is not None else app_coordinator()
            adm = coord.reserve(kind, owner, desired, floor)
        except Exception as ex:                                        # a broken coordinator is a refusal, never an unbudgeted run
            raise ResourceBusy(kind, f"the memory budget could not be consulted ({type(ex).__name__})", None, "UNAVAILABLE") from None
        if adm.ok:
            return adm.reservation
        if time.monotonic() >= deadline:
            raise ResourceBusy(kind, str(adm.detail.get("reason")), adm.detail.get("busy_with"), adm.outcome)
        time.sleep(min(poll_s, max(deadline - time.monotonic(), 0.0)))


# ------------------------------------------------------------------ application-wide default
_APP: ResourceCoordinator | None = None
_APP_LOCK = threading.Lock()


def app_coordinator() -> ResourceCoordinator:
    """The coordinator every runner uses unless it is given another.  Same per-user directory for every process of the application."""
    global _APP
    with _APP_LOCK:
        if _APP is None:
            _APP = ResourceCoordinator()
        return _APP


def set_app_coordinator(c: ResourceCoordinator | None):
    """Replace (or, with None, reset) the application-wide coordinator.  For tests and for an operator that wants a specific configuration."""
    global _APP
    with _APP_LOCK:
        _APP = c
