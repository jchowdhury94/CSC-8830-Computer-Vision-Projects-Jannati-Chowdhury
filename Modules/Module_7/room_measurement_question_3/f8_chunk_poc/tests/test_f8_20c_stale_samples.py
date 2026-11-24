"""F8.20c: a stale memory sample must never keep satisfying the process-tree guard. SYNTHETIC only (fake workers, generated clips); no original or derived data."""
import time

import pytest

import calibration_runner as CR
from calibration_runner import CalibrationJobRunner, RunnerLimits, _fresh_tree_rss
from chunk_store import ChunkStore, Limits, MiB
from monitor import ResourceMonitor
from result_store import ResultStore
from synthetic_stereo import make_stereo_pair
from test_f8_10 import SLEEP, run
from test_runner import A, SETTINGS, assert_retained, put_pair


@pytest.fixture
def results(tmp_path): return ResultStore(tmp_path / "res")


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB, lease_heartbeat_timeout_s=60), root=tmp_path / "store"); yield s; s.close()


@pytest.fixture(scope="module")
def syn(tmp_path_factory): return make_stereo_pair(tmp_path_factory.mktemp("syn820c"))


def lim(**kw): return RunnerLimits(allow_full=True, heartbeat_s=0.5, kill_grace_s=1.0, **kw)


def runner(store, results, **kw):
    return CalibrationJobRunner(store, A, SETTINGS, lim(**kw), results=results, cgroup_root="/nonexistent-cgroup")


def sample(rss=500.0, age=0.0): return dict(combined_rss_mb=rss, mono_s=time.monotonic() - age)


# ---------------------------------------------------------------- 1 / 2: freshness
def test_fresh_reading_is_accepted():
    assert _fresh_tree_rss(sample(500.0, age=0.0)) == 500.0
    assert _fresh_tree_rss(sample(500.0, age=2.5)) == 500.0


@pytest.mark.parametrize("s", [sample(500.0, age=3.5), sample(500.0, age=3600), dict(combined_rss_mb=500.0), dict(combined_rss_mb=500.0, mono_s=None),
                               dict(combined_rss_mb=500.0, mono_s=float("nan")), dict(combined_rss_mb=500.0, mono_s=True), None, {}, "x"])
def test_stale_or_unstamped_reading_is_rejected(s):
    assert _fresh_tree_rss(s) is None


def test_monitor_stamps_every_successful_sample_monotonically():
    m = ResourceMonitor(parent_pid=__import__("os").getpid(), interval=0.25)
    a = m.sample_once(); time.sleep(0.02); b = m.sample_once()
    assert isinstance(a["mono_s"], float) and b["mono_s"] > a["mono_s"] and abs(time.monotonic() - b["mono_s"]) < 2


def test_stale_reading_over_the_limit_does_not_kill_and_counts_as_blind(store, results):
    r = runner(store, results, max_tree_rss_mb=1000.0)
    assert r._guard(sample(5000.0, age=10)) is None and r._blind_polls == 1      # old data is not judged either way
    assert "exceeded" in r._guard(sample(5000.0, age=0))                          # a fresh over-limit reading still kills


def test_a_frozen_sample_cannot_satisfy_the_guard_indefinitely(store, results, monkeypatch):
    monkeypatch.setattr(CR, "BLIND_POLLS_MAX", 3)
    r = runner(store, results, max_tree_rss_mb=3000.0)
    frozen = sample(100.0)
    time.sleep(0.01); monkeypatch.setattr(CR, "STALE_SAMPLE_S", 0.0)
    out = [r._guard(frozen) for _ in range(3)]
    assert out[:2] == [None, None] and "could not be measured" in out[2]


# ---------------------------------------------------------------- 4: recovery
def test_a_new_valid_reading_restores_monitoring_before_the_timeout(store, results, monkeypatch):
    monkeypatch.setattr(CR, "BLIND_POLLS_MAX", 4)
    r = runner(store, results, max_tree_rss_mb=3000.0)
    for _ in range(3): assert r._guard(sample(100.0, age=10)) is None
    assert r._blind_polls == 3
    assert r._guard(sample(100.0)) is None and r._blind_polls == 0               # recovered: counter reset
    for _ in range(3): assert r._guard(sample(100.0, age=10)) is None            # a fresh blind period starts from zero, not from 3
    assert "could not be measured" in r._guard(sample(100.0, age=10))


# ---------------------------------------------------------------- 3 / 5: a frozen monitor stops a real (fake) worker; the old paths still work
def test_repeated_monitor_failures_terminate_the_worker(store, results, syn, monkeypatch):
    put_pair(store, syn["left"], syn["right"])
    monkeypatch.setattr(CR, "BLIND_POLLS_MAX", 3); monkeypatch.setattr(CR, "STALE_SAMPLE_S", 0.5)
    orig = ResourceMonitor.sample_once; n = {"c": 0}
    def flaky(self):                                                              # works for a moment, then raises forever: `latest` freezes (the F8.20b defect)
        n["c"] += 1
        if n["c"] > 3: raise RuntimeError("ps failed")
        return orig(self)
    monkeypatch.setattr(ResourceMonitor, "sample_once", flaky)
    r = run(store, results, SLEEP, limits=lim(max_tree_rss_mb=3000.0), cgroup_root="/nonexistent-cgroup")
    assert r.result["outcome"] == "memory_limit" and "could not be measured" in r.result["error"]
    assert r._proc.poll() is not None
    assert_retained(store)


def test_limit_exceeded_still_terminates_with_fresh_readings(store, results, syn):
    put_pair(store, syn["left"], syn["right"])
    r = run(store, results, SLEEP, limits=lim(max_tree_rss_mb=1.0), cgroup_root="/nonexistent-cgroup")
    assert r.result["outcome"] == "memory_limit" and "parent+worker RSS" in r.result["error"]


# ---------------------------------------------------------------- 6: cgroup path unchanged
def test_cgroup_guard_still_acts_when_rss_is_stale(store, results, monkeypatch):
    monkeypatch.setattr(CR, "read_cgroup", lambda *a, **k: dict(version=2, memory_max_mb=1000.0, anon_mb=950.0, shmem_mb=0.0, memory_current_mb=950.0, file_mb=0.0))
    r = runner(store, results, max_tree_rss_mb=3000.0)
    assert "container memory" in r._guard(sample(100.0, age=10))
    r2 = runner(store, results)                                                   # no RSS limit: staleness is irrelevant, cgroup alone decides
    assert "container memory" in r2._guard({})
