import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2]
for p in (str(MODULE), str(MODULE.parent), str(MODULE / "visitor_workflow"), str(MODULE / "f8_chunk_poc")):   # MODULE.parent: the shared stereo_measure package
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def _results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("F8_RESULTS_DIR", str(tmp_path / "f8_results"))
    monkeypatch.delenv("F8_ENABLE_FULL_CALIBRATION", raising=False)


@pytest.fixture(autouse=True)
def _isolated_resource_budget(tmp_path):
    """F8.31C-1C: every test gets its own, deterministic memory budget and reservation directory (a generous configured ceiling and no container limit), so no test
    depends on the machine's memory, on another test's reservation or on the real per-user coordination directory."""
    import resource_budget as RB
    c = RB.ResourceCoordinator(root=tmp_path / "rb", cfg=RB.BudgetConfig(ceiling_mb=65536.0), container=dict(limit_mb=None, reliable=False, reason="test isolation"))
    RB.set_app_coordinator(c)
    yield c
    RB.set_app_coordinator(None)


@pytest.fixture(autouse=True)
def _visitor_job_slots_are_released(request):
    """VisitorMeasurementJob counts its active jobs in a class attribute shared by the whole pytest process, and a slot is released only by ``close()``.
    A test must therefore close every job it opens.  If a test still leaves the count changed, this fixture FAILS that test at teardown, so a leak is never hidden, and then
    resets the count so that one leak cannot make unrelated later tests fail with JobLimitExceeded.  A session that merely awaits garbage collection is not a leak
    (``F6Session`` closes its job from a finalizer), so a collection is run before the count is judged.  Same policy as tests/conftest.py."""
    import gc
    from room_measurement.visitor_measurement import VisitorMeasurementJob as Job
    yield
    if Job._slots > 0:
        gc.collect()
    leaked = Job._slots
    Job._slots = 0
    if leaked != 0:
        pytest.fail(f"{request.node.nodeid} left the VisitorMeasurementJob count at {leaked:+d} after it finished: every job a test opens must be closed (close() was not called, or was called twice).", pytrace=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_video: expensive test on the protected Wall B recordings (opt in with M7_REAL_VIDEO_TESTS=1; skipped when the data are absent)")
