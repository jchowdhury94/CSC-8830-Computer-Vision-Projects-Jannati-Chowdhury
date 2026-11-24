"""F8.25: the saved ``memory_guard`` block reports which guards actually operated. SYNTHETIC samples only; no videos, no calibration."""
import time

import pytest

from calibration_runner import CalibrationJobRunner, RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from test_runner import A, SETTINGS

CG = dict(version=2, memory_max_mb=4096.0, anon_mb=500.0, shmem_mb=0.0, memory_current_mb=600.0, file_mb=100.0)


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "store"); yield s; s.close()


def runner(store, tree_limit, cg):
    r = CalibrationJobRunner(store, A, SETTINGS, RunnerLimits(allow_full=True, max_tree_rss_mb=tree_limit), cgroup_root="/nonexistent-cgroup")
    r._read_cg = lambda: cg
    return r


def fresh(rss): return dict(mono_s=time.monotonic(), combined_rss_mb=rss)


def test_macos_style_tree_rss_only(store):
    r = runner(store, 3000.0, {})
    for v in (500.0, 918.7, 700.0): assert r._guard(fresh(v)) is None
    g = r._guard_report()
    assert g["active"] is True and g["basis"] == "process-tree RSS" and g["limit_mb"] == 3000.0 and g["peak_used_mb"] == 918.7 and g["checks"] == 3
    assert g["active_guards"] == ["process_tree_rss"] and g["cgroup"]["active"] is False and g["process_tree"]["active"] is True
    assert g["process_tree"]["peak_rss_mb"] == 918.7


def test_linux_cgroup_only(store):
    r = runner(store, None, CG)
    assert r._guard(fresh(500.0)) is None
    g = r._guard_report()
    assert g["active"] is True and g["basis"] == "anonymous+shmem" and g["limit_mb"] == 4096.0 and g["active_guards"] == ["cgroup"]
    assert g["process_tree"]["active"] is False and g["cgroup"]["active"] is True


def test_both_guards_together(store):
    r = runner(store, 3000.0, CG)
    assert r._guard(fresh(500.0)) is None
    g = r._guard_report()
    assert g["active"] is True and g["active_guards"] == ["process_tree_rss", "cgroup"] and "process-tree RSS" in g["basis"] and "anonymous+shmem" in g["basis"]
    assert g["process_tree"]["limit_mb"] == 3000.0 and g["cgroup"]["limit_mb"] == 4096.0 and g["limit_mb"] == 4096.0


def test_unavailable_monitoring_reports_inactive(store):
    r = runner(store, 3000.0, {})
    stale = dict(mono_s=time.monotonic() - 3600, combined_rss_mb=500.0)
    for s in (None, stale, fresh(float("nan"))): r._guard(s)
    g = r._guard_report()
    assert g["active"] is False and g["active_guards"] == [] and g["process_tree"]["active"] is False and g["cgroup"]["active"] is False and g["basis"] == "unknown"


def test_no_limit_configured_and_no_cgroup_is_inactive(store):
    r = runner(store, None, {})
    r._guard(fresh(500.0)); g = r._guard_report()
    assert g["active"] is False and g["process_tree"]["limit_mb"] is None


def test_enforcement_unchanged(store):
    r = runner(store, 800.0, CG)
    assert "exceeded 800 MB" in r._guard(fresh(900.0))
    r2 = runner(store, None, dict(CG, anon_mb=4000.0))
    assert "container memory" in r2._guard(fresh(1.0))
