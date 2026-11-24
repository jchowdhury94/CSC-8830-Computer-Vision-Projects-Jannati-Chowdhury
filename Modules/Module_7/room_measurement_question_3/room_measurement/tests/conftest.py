import sys
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[2]
for _p in (str(MODULE), str(MODULE.parent)):      # MODULE.parent: the shared stereo_measure package
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_video: expensive test on the protected Wall B recordings (opt in with M7_REAL_VIDEO_TESTS=1; skipped when the data are absent)")
