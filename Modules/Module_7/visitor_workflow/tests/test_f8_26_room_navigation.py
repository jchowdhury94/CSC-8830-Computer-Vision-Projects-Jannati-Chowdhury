"""F8.26 Phase A: navigation to the Room Measurement placeholder page. SYNTHETIC only: hand-built calibration documents; no video is opened, nothing is processed."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import f8_visitor_service as S
from calibration_runner import RunnerLimits
from chunk_store import ChunkStore, Limits, MiB
from result_store import ResultStore
from test_f8_visitor_service import make_doc

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path):
    s = ChunkStore(Limits(max_file_bytes=64 * MiB), root=tmp_path / "chunks"); yield s; s.close()


@pytest.fixture
def svc(store, tmp_path): return S.VisitorCalibrationService(store, ResultStore(tmp_path / "results"), RunnerLimits())


def accepted(svc):
    s = S.VisitorSession(); s.review = S.review_f8_document(make_doc("GOOD")[1]); svc.accept(s); return s


# ---------------------------------------------------------------- service rules
def test_navigation_after_acceptance_and_back_preserve_calibration(svc):
    s = accepted(svc); acc, con, f6 = s.accepted, dict(s.contract), s.f6
    s.go("room"); assert s.step == "room"
    s.go("ready")
    assert s.step == "ready" and s.accepted is acc and s.contract == con and s.f6 is f6 and not f6.closed and f6.calibration_provenance["baseline_used_mm"] == 226.0


def test_navigation_blocked_without_acceptance(svc):
    s = S.VisitorSession()
    with pytest.raises(S.ServiceError, match="Accept a calibration"):
        s.go("room")
    assert s.step == "home"
    s.review = S.review_f8_document(make_doc("GOOD")[1])          # reviewed but not accepted
    with pytest.raises(S.ServiceError):
        s.go("room")


def test_navigation_blocked_if_f6_session_is_gone(svc):
    s = accepted(svc); s.f6.close()
    with pytest.raises(S.ServiceError):
        s.go("room")


def test_new_calibration_attempt_discards_old_acceptance(svc):
    for attempt in (lambda s: svc.import_saved(s, b"not json"), lambda s: svc.use_recovered(s, "bad") if False else svc.import_saved(s, b"[1]")):
        s = accepted(svc); old_f6 = s.f6
        attempt(s)
        assert s.accepted is None and s.contract is None and s.f6 is None and old_f6.closed
        with pytest.raises(S.ServiceError):
            s.go("room")


def test_start_over_clears_everything(svc, store):
    s = accepted(svc); s.go("room"); f6 = s.f6
    s.reset(store)
    assert s.step == "home" and s.accepted is None and s.contract is None and s.f6 is None and f6.closed
    with pytest.raises(S.ServiceError):
        s.go("room")


def test_room_processing_stays_disabled_everywhere(svc, monkeypatch):
    assert S.ROOM_PROCESSING_ENABLED is False and svc.room_processing_enabled() is False
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ENABLE_ROOM_PROCESSING", "1")
    assert S.ROOM_PROCESSING_ENABLED is False and svc.room_processing_enabled() is False
    src = (HERE / "f8_visitor_service.py").read_text()
    assert "ROOM_PROCESSING_ENABLED = False" in src and "environ" not in src.split("ROOM_PROCESSING_ENABLED = False")[0].splitlines()[-1]


def test_visiting_room_page_creates_no_worker_or_storage(svc, store, monkeypatch, tmp_path):
    import f6_core as F
    for name in ("submit_videos", "select_frames", "measure", "triangulate", "detect_boundaries"):
        if hasattr(F.F6Session, name):
            monkeypatch.setattr(F.F6Session, name, lambda *a, **k: pytest.fail("F6 processing called"))
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("worker started"))
    s = accepted(svc); s.go("room"); s.go("ready"); s.go("room")
    assert s.runner is None and store.job_status(s.owner) == dict(slots={}, lease=None) and list(Path(store.root).rglob("*.part")) == []


# ---------------------------------------------------------------- page rendering (Streamlit AppTest, no browser)
def _app(accept):
    import sys
    from pathlib import Path
    for p in (HERE_S, HERE_S + "/..", HERE_S + "/../f8_chunk_poc"):
        sys.path.insert(0, p)
    import streamlit as st
    import f8_visitor_pages as P
    import f8_visitor_service as S
    from calibration_runner import RunnerLimits
    from chunk_store import ChunkStore, Limits
    from result_store import ResultStore
    import tempfile
    if "svc" not in st.session_state:
        d = tempfile.mkdtemp(); st.session_state["svc"] = S.VisitorCalibrationService(ChunkStore(Limits(), root=d + "/c"), ResultStore(d + "/r"), RunnerLimits(),
                                                                         room_store=ChunkStore(Limits(reserve_sibling=False), root=d + "/v", slots=("p1_left", "p1_right", "p2_left", "p2_right")))
    svc = st.session_state["svc"]; sess = P.session_of(st.session_state)
    if accept and sess.accepted is None and "tested" not in st.session_state:
        from test_f8_visitor_service import make_doc
        st.session_state["tested"] = True
        sess.review = S.review_f8_document(make_doc("GOOD")[1]); svc.accept(sess)
    P.render(svc, sess, st.session_state)


def _at(accept):
    at = AppTest.from_function(_app, args=(accept,), default_timeout=30)
    return at


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setitem(globals(), "HERE_S", str(HERE))
    import builtins; monkeypatch.setattr(builtins, "HERE_S", str(HERE), raising=False)
    monkeypatch.syspath_prepend(str(HERE / "tests"))


def test_page_flow_shows_four_slots_and_back_and_lock():
    at = _at(True).run()
    assert not at.exception
    next(b for b in at.button if b.key == "f8i_to_room").click(); at.run()
    assert not at.exception and at.session_state["f8i_session"].step == "room"
    s = at.session_state["f8i_session"]; assert not at.get("progress") and any(b.key == "f8i_to_frames" and b.disabled for b in at.button)   # F8.31D-4: only step 1 is open at first
    s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); at.run()
    text = " ".join(m.value for m in at.markdown)
    for label in ("#### Viewpoint 1", "#### Viewpoint 2", "Viewpoint 1 · Left camera video", "Viewpoint 1 · Right camera video",
                  "Viewpoint 2 · Left camera video", "Viewpoint 2 · Right camera video"):
        assert label in text
    assert "LEFT side of the wall" not in text and "RIGHT side of the wall" not in text
    assert "Limitation: Accurate wall measurement depends on clear wall boundaries" in text and "follows the original experiment" not in text
    assert "two different viewpoints" in text and "226" in " ".join(s.value for s in at.success)
    cont = [b for b in at.button if b.key == "f8i_to_frames"][0]
    assert cont.disabled and cont.label == "Continue to Frame Selection"                      # F8.31A: nothing uploaded yet
    assert not any("measure room" in b.label.lower() for b in at.button)
    assert not any(b.key in ("f8i_lock", "f8i_reset") for b in at.button)                        # F8.31D-4: no Start over / Lock on the room-upload page
    s.accepted.provenance["kind"] = "f8_job"                                                      # (a calibration made in this app: its accepted page keeps Start over / Lock)
    next(b for b in at.button if b.key == "f8i_nav_back").click(); at.run()
    sess = at.session_state["f8i_session"]
    assert sess.step == "ready" and sess.accepted is not None
    next(b for b in at.button if b.key == "f8i_lock").click(); at.run()
    fresh = at.session_state["f8i_session"]                      # re-created empty by the next render: nothing of the accepted calibration survives
    assert fresh is not sess and fresh.accepted is None and fresh.f6 is None and sess.f6 is None and fresh.step == "home"
