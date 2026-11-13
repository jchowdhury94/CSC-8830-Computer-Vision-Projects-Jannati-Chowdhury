"""F8.1: passcode gate (pure logic + the real app via Streamlit's AppTest). No real passcode is used or stored anywhere."""
import hmac
import logging
import re
from pathlib import Path

import pytest
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto

import auth
from auth import AttemptLimiter, load_passcode, passcode_matches

GOOD = "test-only-passcode-1234"            # a throw-away value that exists only in this test file
APP = str(Path(__file__).resolve().parents[1] / "app.py")


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def raises(*a, **k): raise KeyError("no such secret")


# ---------------------------------------------------------------- loading (fail closed)
def test_passcode_from_secrets_then_environment():
    assert load_passcode(lambda n: GOOD, {}) == (GOOD, None)
    assert load_passcode(raises, {"F8_PASSCODE": GOOD}) == (GOOD, None)                  # no secrets file / key: fall back to the environment
    assert load_passcode(lambda n: None, {"F8_PASSCODE": GOOD}) == (GOOD, None)
    assert load_passcode(lambda n: GOOD, {"F8_PASSCODE": "environment-value-xyz"})[0] == GOOD   # secrets win


def test_missing_or_unusable_passcode_fails_closed():
    assert load_passcode(raises, {}) == (None, "missing")
    assert load_passcode(lambda n: None, {}) == (None, "missing")
    for blank in ("", "   ", "\n"): assert load_passcode(lambda n, b=blank: b, {}) == (None, "missing")
    assert load_passcode(lambda n: "short", {}) == (None, "too_short")
    assert load_passcode(lambda n: "1234567", {}) == (None, "too_short") and load_passcode(lambda n: "12345678", {})[0] == "12345678"
    for bad in (12345678, ["x" * 20], b"x" * 20, {"a": 1}): assert load_passcode(lambda n, b=bad: b, {}) == (None, "invalid")
    assert load_passcode(lambda n: (_ for _ in ()).throw(RuntimeError("boom")), {}) == (None, "missing")   # any secrets error is treated as "not set"


def test_trailing_newline_from_environment_is_ignored():
    assert load_passcode(raises, {"F8_PASSCODE": GOOD + "\n"}) == (GOOD, None)


# ---------------------------------------------------------------- comparison
def test_correct_and_incorrect_passcodes():
    assert passcode_matches(GOOD, GOOD) and passcode_matches("  " + GOOD + " \n", GOOD)
    for bad in ("", " ", GOOD[:-1], GOOD + "x", GOOD.upper(), "x" * 5000, "pässcode", None, 5, b"bytes", [GOOD]):
        assert not passcode_matches(bad, GOOD), bad
    assert not passcode_matches(GOOD, "") and not passcode_matches(GOOD, None) and not passcode_matches("", "")


def test_comparison_is_constant_time_over_equal_length_digests(monkeypatch):
    calls = []
    real = hmac.compare_digest
    monkeypatch.setattr(auth.hmac, "compare_digest", lambda a, b: (calls.append((len(a), len(b))), real(a, b))[1])
    passcode_matches("a", GOOD); passcode_matches("a" * 1000, GOOD)
    assert calls == [(32, 32), (32, 32)]                                                  # same-length digests whatever the input length


# ---------------------------------------------------------------- throttling
def test_lockout_after_repeated_failures_and_recovery():
    clk = Clock(); lim = AttemptLimiter(clock=clk, max_failures=3, window_s=100, lock_s=30)
    for _ in range(3): assert lim.attempt("s1", "wrong", GOOD)[0] is False
    ok, msg = lim.attempt("s1", GOOD, GOOD)                                               # even the right passcode is refused while locked
    assert not ok and "wait" in msg
    clk.t += 31
    assert lim.attempt("s1", GOOD, GOOD) == (True, "")


def test_lockout_does_not_compare_while_locked(monkeypatch):
    clk = Clock(); lim = AttemptLimiter(clock=clk, max_failures=2, lock_s=30)
    lim.attempt("s", "x", GOOD); lim.attempt("s", "x", GOOD)
    seen = []; monkeypatch.setattr(auth, "passcode_matches", lambda *a: seen.append(a) or True)
    assert lim.attempt("s", GOOD, GOOD)[0] is False and seen == []                        # locked: no comparison, so lock-out cannot be probed


def test_success_resets_the_session_counter_and_other_sessions_are_independent():
    clk = Clock(); lim = AttemptLimiter(clock=clk, max_failures=3, lock_s=30)
    lim.attempt("a", "x", GOOD); lim.attempt("a", "x", GOOD); assert lim.attempt("a", GOOD, GOOD)[0]
    lim.attempt("a", "x", GOOD); lim.attempt("a", "x", GOOD); assert lim.attempt("a", GOOD, GOOD)[0]     # counter was reset by the success
    for _ in range(3): lim.attempt("b", "x", GOOD)
    assert lim.attempt("c", GOOD, GOOD)[0]                                                # another session is not locked by b's failures


def test_global_limit_stops_guessing_across_many_sessions():
    clk = Clock(); lim = AttemptLimiter(clock=clk, max_failures=5, global_max_failures=10, lock_s=30)
    for i in range(10): lim.attempt(f"bot{i}", "x", GOOD)                                 # one guess each: no session limit is reached
    assert lim.attempt("bot-new", GOOD, GOOD)[0] is False and lim.wait_seconds("anyone") > 0
    clk.t += 31; assert lim.attempt("bot-new", GOOD, GOOD)[0] is True


def test_old_failures_expire():
    clk = Clock(); lim = AttemptLimiter(clock=clk, max_failures=3, window_s=50, lock_s=30)
    lim.attempt("s", "x", GOOD); lim.attempt("s", "x", GOOD); clk.t += 60
    lim.attempt("s", "x", GOOD)
    assert lim.attempt("s", GOOD, GOOD)[0]                                                # only one failure inside the window


def test_limiter_memory_is_bounded():
    lim = AttemptLimiter(max_failures=100)
    for i in range(8000): lim.record_failure(f"session-{i}")
    assert len(lim._fails) <= 5100


# ---------------------------------------------------------------- never logged / never stored
def test_passcode_and_guesses_are_never_logged_or_printed(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    lim = AttemptLimiter(); lim.attempt("s", "my-wrong-guess-777", GOOD); lim.attempt("s", GOOD, GOOD)
    load_passcode(raises, {"F8_PASSCODE": GOOD})
    out = capsys.readouterr()
    for text in (caplog.text, out.out, out.err): assert GOOD not in text and "my-wrong-guess-777" not in text


def test_no_credentials_are_hardcoded_in_the_project_files():
    root = Path(__file__).resolve().parents[1]; bad = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".toml", ".txt", ".html", ".example", ".json"} and "outputs" not in p.parts and "__pycache__" not in p.parts and p.name != "test_auth.py":
            for m in re.finditer(r"F8_PASSCODE\s*[:=]\s*[\"']([^\"']*)[\"']", p.read_text(errors="ignore")):
                if not m.group(1).startswith("<"): bad.append((p.name, m.group(1)))     # only a <placeholder> may follow the name
    assert not bad, bad
    assert not (root / ".streamlit" / "secrets.toml").exists()


# ---------------------------------------------------------------- the real app: gate order (Streamlit AppTest)
@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Run app.py with spies that record whether the store or the upload component was ever created."""
    import streamlit as st
    import chunk_store, component
    st.cache_resource.clear()
    made, drawn = [], []

    class SpyStore(chunk_store.ChunkStore):
        def __init__(self, *a, **k): made.append(1); super().__init__(*a, **{**k, "root": tmp_path / "root"})

    monkeypatch.setattr(chunk_store, "ChunkStore", SpyStore)
    monkeypatch.setattr(component, "chunk_uploader", lambda *a, **k: drawn.append(k.get("key")) or None)
    monkeypatch.delenv("F8_PASSCODE", raising=False)
    yield dict(made=made, drawn=drawn)
    st.cache_resource.clear()


def run_app(secrets=None):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=30)
    if secrets is not None: at.secrets["F8_PASSCODE"] = secrets
    return at.run()


def test_app_fails_closed_when_the_passcode_is_missing(app_env):
    at = run_app()
    assert not at.exception and at.error and "locked" in at.error[0].value
    assert not at.text_input and not at.button                                            # not even a login form: nothing to attack
    assert app_env["made"] == [] and app_env["drawn"] == []                               # no store, no upload component


def test_app_fails_closed_when_the_passcode_is_too_short(app_env):
    at = run_app("short")
    assert at.error and "locked" in at.error[0].value and app_env["made"] == [] and app_env["drawn"] == []


def test_unauthenticated_visitor_sees_only_the_form(app_env):
    at = run_app(GOOD)
    assert not at.exception and len(at.text_input) == 1 and at.text_input[0].proto.type == TextInputProto.PASSWORD
    assert app_env["made"] == [] and app_env["drawn"] == []                               # upload component and store do not exist yet
    assert "Lock this session" not in " ".join(b.label for b in at.button)


def test_wrong_passcode_stays_locked(app_env):
    at = run_app(GOOD)
    at.text_input[0].set_value("definitely-wrong"); at.button[0].click(); at.run()
    assert at.error and "Incorrect passcode" in at.error[0].value
    assert "authed" not in at.session_state or not at.session_state["authed"]
    assert app_env["made"] == [] and app_env["drawn"] == []


def test_correct_passcode_unlocks_and_only_then_creates_the_uploader(app_env):
    at = run_app(GOOD)
    at.text_input[0].set_value(GOOD); at.button[0].click(); at.run()
    assert not at.exception and at.session_state["authed"] is True
    assert app_env["made"] == [1] and app_env["drawn"] == ["f8_uploader"]
    assert any("Lock this session" in b.label for b in at.button)


def test_environment_variable_also_works_and_lockout_applies_in_the_app(app_env, monkeypatch):
    monkeypatch.setenv("F8_PASSCODE", GOOD)
    at = run_app()
    for _ in range(5):
        at.text_input[0].set_value("nope-nope-nope"); at.button[0].click(); at.run()
    at.text_input[0].set_value(GOOD); at.button[0].click(); at.run()                      # right passcode, but this session is now locked out
    assert at.error and "wait" in at.error[0].value and app_env["drawn"] == [] and app_env["made"] == []
