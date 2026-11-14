"""Stage F8.5: cross-process 'file in use' protection (NO Streamlit import).

A reader (the calibration worker) takes a SHARED flock on every video file it reads and keeps it until it exits. The store takes a
NON-BLOCKING EXCLUSIVE flock before it deletes a file: if any reader holds the file, the deletion is refused. The kernel drops a dead
process's locks, so a killed worker can never leave a file locked, and a live reader can never lose its file to a sweeper, a cancel,
a replacement upload or a lease expiry. Locks are advisory: every deleter in this package goes through ``try_exclusive``.
On a platform without ``fcntl`` the helpers degrade to no-ops (the lease still protects the files inside one process).
"""
from __future__ import annotations

try:
    import fcntl
except ImportError:                                                    # pragma: no cover (Windows)
    fcntl = None


class _Held:
    """An open file object whose lock is released on close()."""

    def __init__(self, fh=None):
        self._fh = fh

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()                                       # closing the descriptor releases the flock
            finally:
                self._fh = None


def try_exclusive(path):
    """Exclusive, non-blocking. Returns a held lock (call .close()), or None if any reader holds the file. A missing file counts as free."""
    if fcntl is None:
        return _Held()
    try:
        fh = open(path, "rb")
    except FileNotFoundError:
        return _Held()
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return _Held(fh)


def acquire_shared(paths):
    """Shared (reader) locks on all ``paths``; returns a list of held locks (keep them referenced until the reads are finished).
    Raises FileNotFoundError if a file is already gone."""
    held = []
    try:
        for p in paths:
            fh = open(p, "rb")
            held.append(_Held(fh))
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
    except Exception:
        for h in held:
            h.close()
        raise
    return held
