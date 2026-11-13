"""Stage F8.1: minimal passcode gate for the TEMPORARY Cloud test app (NO Streamlit import, so it is unit-testable).

* The passcode comes from Streamlit secrets (``F8_PASSCODE``) or the environment variable ``F8_PASSCODE``. It is never hard-coded or written anywhere.
* FAIL CLOSED: if it is missing, not text, or shorter than MIN_LENGTH, nobody gets in.
* Comparison is constant-time (HMAC compare of SHA-256 digests, so length does not leak).
* Wrong guesses are throttled per session and globally (simple lock-out); the passcode and guesses are never logged.
"""
from __future__ import annotations

import hashlib
import hmac
import time

SECRET_NAME = "F8_PASSCODE"
MIN_LENGTH = 8


def load_passcode(secrets_get, environ) -> tuple:
    """Returns (passcode, None) or (None, reason). ``secrets_get(name)`` may raise or return None (no secrets file); the environment is the fallback.
    Surrounding whitespace/newlines are removed (environment variables often carry a trailing newline)."""
    value = None
    try:
        value = secrets_get(SECRET_NAME)
    except Exception:
        value = None
    if value is None:
        value = environ.get(SECRET_NAME)
    if value is None:
        return None, "missing"
    if not isinstance(value, str):
        return None, "invalid"
    value = value.strip()
    if not value:
        return None, "missing"
    if len(value) < MIN_LENGTH:
        return None, "too_short"
    return value, None


def passcode_matches(supplied, expected: str) -> bool:
    """Constant-time comparison. Anything that is not text never matches."""
    if not isinstance(supplied, str) or not isinstance(expected, str) or not expected:
        return False
    a = hashlib.sha256(supplied.strip().encode("utf-8", "replace")).digest()
    b = hashlib.sha256(expected.encode("utf-8", "replace")).digest()
    return hmac.compare_digest(a, b)


class AttemptLimiter:
    """Lock-out after repeated wrong guesses: ``max_failures`` within ``window_s`` locks that key for ``lock_s``.
    One key per session plus one global key (so many sessions cannot be used to guess faster)."""

    def __init__(self, clock=time.monotonic, max_failures=5, window_s=300.0, lock_s=60.0, global_max_failures=30, global_window_s=600.0):
        self._clock = clock
        self._cfg = {"session": (max_failures, window_s, lock_s), "global": (global_max_failures, global_window_s, lock_s)}
        self._fails: dict = {}
        self._locked_until: dict = {}

    def _keys(self, session):
        return [("session", str(session)), ("global", "*")]

    def wait_seconds(self, session) -> float:
        now = self._clock()
        return max([self._locked_until.get(k, 0.0) - now for k in self._keys(session)] + [0.0])

    def record_failure(self, session):
        now = self._clock()
        for k in self._keys(session):
            mx, win, lock = self._cfg[k[0]]
            recent = [t for t in self._fails.get(k, []) if now - t < win] + [now]
            self._fails[k] = recent
            if len(recent) >= mx:
                self._locked_until[k] = now + lock; self._fails[k] = []
        if len(self._fails) > 5000:                                   # bound memory against many sessions
            for k in [k for k in self._fails if k[0] == "session"][:2500]:
                self._fails.pop(k, None); self._locked_until.pop(k, None)

    def record_success(self, session):
        self._fails.pop(("session", str(session)), None)

    def attempt(self, session, supplied, expected) -> tuple:
        """Returns (granted, message). A locked-out key is refused WITHOUT comparing, so lock-out cannot be probed."""
        wait = self.wait_seconds(session)
        if wait > 0:
            return False, f"Too many incorrect attempts. Please wait {int(wait) + 1} s and try again."
        if passcode_matches(supplied, expected):
            self.record_success(session)
            return True, ""
        self.record_failure(session)
        return False, "Incorrect passcode."
