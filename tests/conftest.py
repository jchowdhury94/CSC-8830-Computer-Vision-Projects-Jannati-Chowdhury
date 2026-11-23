"""Repository-level pytest fixtures (test infrastructure only; no production code is touched)."""
import gc
import sys

import pytest


def _job_classes():
    """VisitorMeasurementJob classes of every already-imported ``visitor_measurement`` module (nothing is imported here)."""
    found = []
    for name, module in list(sys.modules.items()):
        if name.rsplit(".", 1)[-1] == "visitor_measurement":
            cls = getattr(module, "VisitorMeasurementJob", None)
            if cls is not None and hasattr(cls, "_slots") and cls not in found:
                found.append(cls)
    return found


@pytest.fixture(autouse=True)
def _visitor_job_slots_are_released(request):
    """VisitorMeasurementJob counts its active jobs in a class attribute shared by the whole pytest process, and a slot is released only by ``close()``.
    A test must therefore close every job it opens (for AppTest-driven visitor pages see ``_close_app_sessions`` in test_module7_f8_visitor_integration.py).
    If a test still leaves the count changed, this fixture FAILS that test at teardown, so a leak is never hidden, and then resets the count so that one leak cannot
    make unrelated later tests fail with JobLimitExceeded.  A session that merely awaits garbage collection is not a leak: ``F6Session`` closes its job from a finalizer,
    so a collection is run before the count is judged."""
    yield
    leaks = []
    for cls in _job_classes():
        if cls._slots > 0:
            gc.collect()
        if cls._slots != 0:
            leaks.append(cls._slots)
            cls._slots = 0
    if leaks:
        pytest.fail(f"{request.node.nodeid} left the VisitorMeasurementJob count at {leaks[0]:+d} after it finished: every job a test opens must be closed (close() was not called, or was called twice).", pytrace=False)
