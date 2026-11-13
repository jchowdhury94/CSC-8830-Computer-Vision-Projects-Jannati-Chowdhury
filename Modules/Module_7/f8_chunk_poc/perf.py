"""Stage F8 PoC: small process-memory helpers (no third-party dependency)."""
from __future__ import annotations

import os
import subprocess
import sys


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process (monotonic)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except Exception:
        return None


def current_rss_mb() -> float | None:
    """Current resident set size: /proc on Linux (Streamlit Cloud), ps on macOS."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        pass
    try:
        return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=2).stdout.strip()) / 1024
    except Exception:
        return None
