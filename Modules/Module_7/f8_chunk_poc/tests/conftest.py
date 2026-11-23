import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _results_dir(tmp_path, monkeypatch):
    """F8.10: every test writes its saved results under its own temporary folder (never the machine's temp dir)."""
    monkeypatch.setenv("F8_RESULTS_DIR", str(tmp_path / "f8_results"))
