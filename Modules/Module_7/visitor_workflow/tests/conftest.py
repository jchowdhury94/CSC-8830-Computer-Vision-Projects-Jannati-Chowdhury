import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2]
for p in (str(MODULE), str(MODULE / "visitor_workflow"), str(MODULE / "f8_chunk_poc")):
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


def pytest_configure(config):
    config.addinivalue_line("markers", "real_video: expensive test on the protected Wall B recordings (opt in with M7_REAL_VIDEO_TESTS=1; skipped when the data are absent)")
