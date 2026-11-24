"""Step 3: what the visitor sees -- readiness, the closed gate, the result with its warnings, a failure, and the final room page (Streamlit AppTest, INJECTED results)."""
import json

import pytest

import f8_measure_page as MP
import f8_visitor_service as S
import three_view_result as TVR
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31c2a_measurement_interface import ok_result, run_to_completion, use_measurement
from test_f8_35_three_view_annotations import b3, mark_set       # noqa: F401  (fixtures + helper)
from test_f8_35_three_view_results import BAD, CLEAN, META, worker_doc
from test_f8_35_viewpoint_state import clips6, svc6               # noqa: F401  (fixtures)

WARN3 = [(3790.889, dict(ang_v1_v3_deg=2.5)), (3811.971, None), (3803.973, dict(ang_v1_v3_deg=2.6))]
A_RES = ok_result(width=2850.0, height=2358.0)


def app(svc, sess, body):
    from streamlit.testing.v1 import AppTest
    from test_f8_31b_pages import F8, HERE, MODULE, ROOT
    src = ("def run():\n    import sys as _s\n    import streamlit as st\n    for p in %r:\n        if p not in _s.path: _s.path.insert(0, p)\n    import f8_visitor_pages as P\n    import f8_measure_page as MP\n"
           "    svc = st.session_state['_svc']; sess = P.session_of(st.session_state)\n%s\nrun()\n") % ((str(MODULE), str(HERE), str(F8), str(ROOM := ROOT)), body)
    at = AppTest.from_string(src, default_timeout=90); at.session_state["_svc"] = svc; at.session_state["f8i_session"] = sess
    return at


@pytest.fixture
def ready_b(svc6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0, horizontals=False); mark_set(wf, 1, horizontals=False); s.step = "measure"
    return s, wf


def test_measure_page_shows_three_viewpoint_readiness_and_the_closed_gate(svc6, ready_b, monkeypatch):
    monkeypatch.setattr(S, "WALL_B_PROCESSING_ENABLED", False)                                          # closed Wall B gate
    s, wf = ready_b; at = page_app(svc6, s); at.run(); assert not at.exception, at.exception
    t = texts(at); assert "Ready to calculate" in t and "6 videos (3 viewpoints)" in t and "2 usable frame sets" in t and "not yet available" in t
    assert button(at, "Calculate Room Length").disabled and "Calculate Room Length" in t                      # the processing gate is a code constant: no handler, no start
    assert S.ROOM_PROCESSING_ENABLED is False and s.measurement.job is None


def test_measure_page_lists_what_is_missing_and_offers_the_fix(svc6, b3):
    s, wf = b3; wf.extract_frames(); mark_set(wf, 0, horizontals=False); s.step = "measure"            # only ONE usable frame set
    at = page_app(svc6, s); at.run(); assert not at.exception, at.exception
    t = texts(at); assert "Ready to calculate" not in t and "two frame sets" in t and "six boundaries" not in t
    assert has_button(at, "Fix this: go to the") and not has_button(at, "Calculate Room Length") or button(at, "Calculate Room Length").disabled


def test_the_result_page_shows_the_length_the_warnings_and_the_limitations_statement(svc6, ready_b):
    s, wf = ready_b; result = TVR.assemble(META, worker_out=worker_doc(WARN3)); use_measurement(svc6, runner=lambda ctx: result); run_to_completion(svc6.measurement, s)
    at = app(svc6, s, "    MP.render_result(svc, sess, sess.measurement.result)"); at.run(); assert not at.exception, at.exception
    t = texts(at)
    assert "Estimated Room Length" in t and "380 cm" in t and "3.80 m" in t and "Approximate length — 1 quality warning" in t and "3 viewpoints" in t
    assert "usable, with warnings" in t and "Frame set 1" in t and "Frame set 3" in t and "do not line up perfectly" in t           # the warning is in the open, per frame set, with its explanation
    assert TVR.LIMITATIONS in t and "393.7" not in t and "3937" not in t and "3.43" not in t
    assert "ground truth" not in t.lower() and "tape" not in t.lower()


def test_a_clean_result_is_still_labelled_approximate(svc6, ready_b):
    s, wf = ready_b; result = TVR.assemble(META, worker_out=worker_doc(CLEAN)); use_measurement(svc6, runner=lambda ctx: result); run_to_completion(svc6.measurement, s)
    at = app(svc6, s, "    MP.render_result(svc, sess, sess.measurement.result)"); at.run(); assert not at.exception
    t = texts(at); assert "Approximate length" in t and "Approximate length —" not in t and "no quality warning" in t


def test_a_failure_shows_the_frame_level_reasons_and_no_length(svc6, ready_b):
    s, wf = ready_b; result = TVR.assemble(META, worker_out=worker_doc([(3790.0, None), (5000.0, BAD), (5100.0, BAD)])); use_measurement(svc6, runner=lambda ctx: result); run_to_completion(svc6.measurement, s)
    at = app(svc6, s, "    MP.render_result(svc, sess, sess.measurement.result)"); at.run(); assert not at.exception, at.exception
    t = texts(at)
    assert "Fewer than two frame sets gave a usable reconstruction" in t and "No dimension is reported" in t and "not usable" in t
    assert "disagree too much to be the same floor" in t                                                              # the frame-level hard failure, with its reason
    assert "cm" not in " ".join(m.value for m in at.markdown if "f8m-value" in m.value)                          # no metric card with a number
    assert has_button(at, "Go to frame selection and boundaries")                                                   # a useful corrective action


def test_the_room_page_combines_wall_a_width_and_height_with_wall_b_length(svc6, ready_b):
    s, wf = ready_b; b = TVR.assemble(META, worker_out=worker_doc(WARN3)); at = app(svc6, s, "    MP.render_room(svc, sess, st.session_state['ra'], st.session_state['rb'])")
    at.session_state["ra"] = A_RES; at.session_state["rb"] = b; at.run(); assert not at.exception, at.exception
    t = texts(at)
    assert "Room Width" in t and "285 cm" in t and "Room Length" in t and "380 cm" in t and "Room Height" in t and "236 cm" in t and "From Wall A" in t and "From Wall B" in t
    assert t.count("Measurement Limitations") == 1 and TVR.LIMITATIONS in t and "Wall B (room length)" in t and "Approximate length — 1 quality warning" in t
    assert "Room dimensions are approximate estimates obtained from stereo videos of two walls" not in t              # the owner's statement replaces the two-wall one when Wall B used three viewpoints


def test_the_room_page_for_two_viewpoint_walls_is_unchanged(svc6, ready_b):
    s, wf = ready_b; at = app(svc6, s, "    MP.render_room(svc, sess, st.session_state['ra'], st.session_state['rb'])")
    at.session_state["ra"] = A_RES; at.session_state["rb"] = ok_result(width=3400.0, height=2400.0); at.run(); assert not at.exception
    t = texts(at); assert "340 cm" in t and "Room dimensions are approximate estimates obtained from stereo videos of two walls" in t and "Approximate length" not in t and "quality warning" not in t
