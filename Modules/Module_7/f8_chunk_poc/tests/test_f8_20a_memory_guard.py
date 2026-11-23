"""F8.20a: process-tree memory guard (the only guard on macOS) -- activation, refusal, enforcement incl. child processes, cancellation and cleanup.

SYNTHETIC data only: generated stereo checkerboard clips and fake worker commands (no original, derived or regression data). Fake workers take the same shared file locks as the real one."""
import os
import subprocess
import sys

import pytest

import calibration_runner as CR
from calibration_runner import CalibrationJobRunner, RunnerLimits, _env_limit_mb
from chunk_store import ChunkStore, Limits, MiB
from result_store import ResultStore
from synthetic_stereo import make_stereo_pair
from test_f8_10 import FULL, SLEEP, run
from test_runner import A, SETTINGS, alive, assert_retained, fake, files, put_pair, wait_for


@pytest.fixture
def results(tmp_path): return ResultStore(tmp_path / "res")


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60), root=tmp_path / "store"); yield s; s.close()


@pytest.fixture(scope="module")
def syn(tmp_path_factory): return make_stereo_pair(tmp_path_factory.mktemp("syn820a"))


def own_rss_mb():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.split()[0]) / 1024.0


def lim(**kw):
    return RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0, **kw)


# ---------------------------------------------------------------- configuration
@pytest.mark.parametrize("val,expect", [("3000", 3000.0), (" 512 ", 512.0), ("256", 256.0), ("65536", 65536.0), ("255", None), ("65537", None), ("0", None), ("-5", None),
                                         ("nan", None), ("inf", None), ("abc", None), ("", None)])
def test_env_limit_parsing_never_clamps_silently(val, expect):
    assert _env_limit_mb({"F8_MAX_TREE_RSS_MB": val}, "F8_MAX_TREE_RSS_MB") == expect


def test_from_env_is_off_by_default_and_reads_the_limit():
    d = RunnerLimits.from_env({})
    assert d.allow_full is False and d.max_tree_rss_mb is None and d.require_memory_guard is False
    e = RunnerLimits.from_env({"F8_ENABLE_FULL_CALIBRATION": "1", "F8_MAX_TREE_RSS_MB": "3000", "F8_REQUIRE_MEMORY_GUARD": "1"})
    assert e.allow_full and e.max_tree_rss_mb == 3000.0 and e.require_memory_guard
    assert RunnerLimits.from_env({"F8_MAX_TREE_RSS_MB": "oops"}).max_tree_rss_mb is None


# ---------------------------------------------------------------- activation / refusal
def test_guard_is_operative_with_a_measurable_tree_limit(store, results):
    r = CalibrationJobRunner(store, A, SETTINGS, lim(max_tree_rss_mb=3000.0, require_memory_guard=True), results=results, cgroup_root="/nonexistent-cgroup")
    assert r._guard_operative() is False                                  # no monitor yet
    from monitor import ResourceMonitor
    r._monitor = ResourceMonitor(parent_pid=os.getpid(), interval=0.25)
    assert r._guard_operative() is True


def test_guard_is_not_operative_when_rss_cannot_be_measured(store, results):
    from monitor import ResourceMonitor
    r = CalibrationJobRunner(store, A, SETTINGS, lim(max_tree_rss_mb=3000.0), results=results, cgroup_root="/nonexistent-cgroup")
    r._monitor = ResourceMonitor(parent_pid=os.getpid(), interval=0.25, snapshot=lambda: {})
    assert r._guard_operative() is False


def test_full_run_refused_when_required_guard_cannot_operate(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r = CalibrationJobRunner(store, A, SETTINGS, lim(require_memory_guard=True), worker_cmd=fake(SLEEP), results=results, cgroup_root="/nonexistent-cgroup", monitor_interval=0.25)
    r.start(); r.join(30)
    assert r.result["outcome"] == "rejected" and "memory guard cannot act" in r.result["error"]
    assert len(files(store)) == 2                                          # refusal deletes nothing; no worker was started
    assert r._proc is None


def test_full_run_starts_when_the_tree_limit_is_measurable(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, results, FULL, limits=lim(max_tree_rss_mb=own_rss_mb() + 3000, require_memory_guard=True), cgroup_root="/nonexistent-cgroup")
    assert r.result["outcome"] == "completed" and r.result["memory_guard"]["max_tree_rss_mb"] > 0


def test_blind_guard_stops_a_running_job(store, results, syn, monkeypatch):
    put_pair(store, syn["left"], syn["right"])
    import monitor as M
    monkeypatch.setattr(CR, "BLIND_POLLS_MAX", 3)
    calls = {"n": 0}
    orig = M.ResourceMonitor.sample_once
    def sample(self):                                                      # readable at start-up, then blind
        calls["n"] += 1
        s = orig(self)
        if calls["n"] > 4:
            s["combined_rss_mb"] = None; self.latest = s
        return s
    monkeypatch.setattr(M.ResourceMonitor, "sample_once", sample)
    r = run(store, results, SLEEP, limits=lim(max_tree_rss_mb=3000.0), cgroup_root="/nonexistent-cgroup")
    assert r.result["outcome"] == "memory_limit" and "could not be measured" in r.result["error"]
    assert_retained(store)


# ---------------------------------------------------------------- enforcement
def test_limit_exceeded_stops_worker_and_keeps_videos(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, results, SLEEP, limits=lim(max_tree_rss_mb=1.0), cgroup_root="/nonexistent-cgroup")      # far below the parent's own RSS: trips on the first reading
    assert r.result["outcome"] == "memory_limit" and "parent+worker RSS" in r.result["error"] and "exceeded" in r.result["error"]
    assert r._proc.poll() is not None
    assert_retained(store)


def test_child_process_memory_counts_and_whole_group_is_killed(store, results, syn, tmp_path):
    put_pair(store, syn["left"], syn["right"])
    pidfile = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(f"import os, time\nopen({str(pidfile)!r}, 'w').write(str(os.getpid()))\nb = bytearray(400 * 1024 * 1024)\nfor i in range(0, len(b), 4096): b[i] = 1\ntime.sleep(120)\n")
    body = f"c = subprocess.Popen([sys.executable, '-I', {str(child)!r}])\ntime.sleep(120)"
    r = run(store, results, body, limits=lim(max_tree_rss_mb=own_rss_mb() + 200.0), cgroup_root="/nonexistent-cgroup")      # the worker itself is tiny: only its CHILD exceeds the limit
    assert r.result["outcome"] == "memory_limit", r.result.get("error")
    cpid = int(pidfile.read_text())
    assert wait_for(lambda: not alive(cpid), 10), "the child of the worker survived the memory stop"
    assert_retained(store)


def test_normal_run_below_the_limit_is_not_stopped(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, results, FULL, limits=lim(max_tree_rss_mb=own_rss_mb() + 2000), cgroup_root="/nonexistent-cgroup")
    assert r.result["outcome"] == "completed" and files(store) == []


# ---------------------------------------------------------------- cancellation and cleanup
def test_cancel_kills_worker_group_and_retains_videos(store, results, syn, tmp_path):
    put_pair(store, syn["left"], syn["right"])
    pidfile = tmp_path / "w.pid"
    body = f"open({str(pidfile)!r},'w').write(str(os.getpid())); c = subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); time.sleep(120)"
    r = CalibrationJobRunner(store, A, SETTINGS, lim(max_tree_rss_mb=own_rss_mb() + 2000), worker_cmd=fake(body), monitor_interval=0.25, results=results, cgroup_root="/nonexistent-cgroup")
    r.start()
    assert wait_for(pidfile.exists, 30)
    pid = int(pidfile.read_text()); r.cancel(); r.join(30)
    assert r.done and r.result["outcome"] == "cancelled"
    assert wait_for(lambda: not alive(pid), 10)
    assert_retained(store)


def test_session_lock_deletes_everything(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    assert len(files(store)) == 2
    store.release_owner(A)
    assert files(store) == []
