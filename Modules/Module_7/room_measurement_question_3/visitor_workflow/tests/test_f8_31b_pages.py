"""F8.31B page-level tests (Streamlit AppTest) on SYNTHETIC clips.  The pages draw; every rule is tested in test_f8_31b_frames.py."""
import time
from pathlib import Path

import pytest

import f8_frames_page as FP
import f8_visitor_service as S
import room_frames as RF
from test_f8_31b_frames import (LINE_POINTS, accepted, annotate_all, choose_and_confirm, clips, cstore, complete, ready, rstore, svc, upload_all)   # noqa: F401  (fixtures)

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parent
F8 = MODULE / "f8_chunk_poc"
ROOT = MODULE.parents[2]


def page_app(svc, sess):
    from streamlit.testing.v1 import AppTest
    src = ("def run():\n    import sys as _s\n    import streamlit as st\n    for p in %r:\n        if p not in _s.path: _s.path.insert(0, p)\n"
           "    import f8_visitor_pages as P\n    P.render(st.session_state['_svc'], P.session_of(st.session_state), st.session_state)\nrun()\n") % ((str(MODULE), str(HERE), str(F8), str(ROOT)),)
    at = AppTest.from_string(src, default_timeout=90)
    at.session_state["_svc"] = svc; at.session_state["f8i_session"] = sess
    return at


def texts(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [e.value for e in at.warning] + [e.value for e in at.success] + [e.value for e in at.info] + [c.value for c in at.caption])


def button(at, part):
    for b in at.button:
        if part in b.label:
            return b
    raise AssertionError(f"no button {part!r}: {[b.label for b in at.button]}")


def has_button(at, part):
    return any(part in b.label for b in at.button)


def finish_boundaries(at, wf, points=None):
    """F8.31F: confirm the six boundaries through the real guided page (the seed points are stored as the visitor's confirmed lines, the sixth screen's Finish is pressed)."""
    import room_frames as RF
    from test_f8_31b_frames import LINE_POINTS
    for line in RF.REQUIRED_LINES:
        wf.seed_set(line, (points or LINE_POINTS)[line.split("_")[1]])
    at.session_state["f8i_bd_idx"] = len(RF.REQUIRED_LINES) - 1; at.run()
    button(at, "Finish").click().run()


def test_frames_page_replaces_the_placeholder_and_selects_automatically_once(svc, ready):
    s, wf = ready; s.step = "frames"; at = page_app(svc, s); at.run()
    assert not at.exception, at.exception
    t = texts(at)
    assert "Frames and boundaries" in t and "Coming in the next phase" not in t and "not implemented yet" not in t
    assert not has_button(at, "Suggest Frames") and not has_button(at, "Use these frames") and not has_button(at, "Overview of both videos")      # technical controls stay hidden
    assert not has_button(at, "Calculate Wall Dimensions") and wf.frames_ready()                              # F8.31D-6: frames were selected on arrival
    log = list(wf.log)
    for _ in range(2):
        at.run()
    assert wf.log == log                                                                                   # reruns read no video again


def test_manual_confirm_is_offered_only_after_the_previews_were_shown(svc, ready):
    s, wf = ready; s.step = "frames"; wf.auto_select()
    wf.set_pair(1, 0, 11, 15)                                                  # the visitor changed one pair: it needs confirming again
    assert not wf.previews_viewed(1, 0) and not wf.all_pairs_confirmed()
    at = page_app(svc, s); at.run(); assert not at.exception
    assert not has_button(at, "Confirm this pair")                             # manual controls stay hidden until Change Frames is switched on
    at.toggle(key="f8i_fs_manual").set_value(True).run(); assert not at.exception and button(at, "Confirm this pair").disabled
    button(at, "Preview both frames").click().run(); assert wf.previews_viewed(1, 0) and not button(at, "Confirm this pair").disabled
    button(at, "Confirm this pair").click().run()
    assert wf.pairs[(1, 0)]["confirmed"] and wf.frames_ready()                 # confirmed pairs are read at full quality automatically


def test_annotation_step_renders_the_frame_and_the_controls(svc, ready):
    s, wf = ready; s.step = "frames"; wf.auto_select(); at = page_app(svc, s); at.run()
    assert not at.exception, at.exception
    assert has_button(at, "Next") and button(at, "Next").disabled and has_button(at, "Back") and button(at, "Back").disabled and not has_button(at, "Check this boundary")
    t = texts(at).replace("*", ""); assert "1 of 6" in t and not at.radio                                         # F8.31F: one boundary per screen, no radio list
    wf.seed_click("AL_corner", 894.0, 350.0); wf.seed_click("AL_corner", 912.0, 950.0); at.run()
    assert not at.exception and not button(at, "Next").disabled and not has_button(at, "Accept this boundary")      # two points are enough to go on; no per-boundary Accept


def test_continue_is_gated_and_leads_to_the_placeholder(svc, ready, monkeypatch):
    monkeypatch.setattr(S, "WALL_A_PROCESSING_ENABLED", False)                                          # closed-gate wording is tested with the Wall A gate closed
    s, wf = ready; s.step = "frames"; wf.auto_select()
    at = page_app(svc, s); at.run()
    assert not at.exception and "Locked. Confirm all six wall boundaries first." in texts(at) and not has_button(at, "Continue")      # step 3 is locked until the six marks were applied
    finish_boundaries(at, wf)
    assert not button(at, "Continue").disabled and "Boundary refinement completed" in texts(at)
    button(at, "Continue").click().run()
    assert s.step == "measure" and not at.exception
    t = texts(at); assert "not yet available" in t and "Your calibration, frame selections, and annotations can still be reviewed" in t
    assert button(at, "Calculate Wall Dimensions").disabled and S.ROOM_PROCESSING_ENABLED is False
    button(at, "Go back").click().run(); assert s.step == "frames"


def test_expired_uploads_guide_the_visitor_to_upload_again(svc, complete):
    s, wf = complete; s.step = "frames"
    base = svc.room_store._clock(); svc.room_store._clock = lambda: base + 3600; svc.room_store.sweep()
    at = page_app(svc, s); at.run()
    assert not at.exception, at.exception
    t = texts(at); assert "no longer usable" in t and "15 minutes" in t and has_button(at, "Upload the videos again")
    assert not has_button(at, "Confirm this pair") and not has_button(at, "Refine boundary")
    button(at, "Upload the videos again").click().run(); assert s.step == "room"


def test_replaced_video_message_is_shown_on_the_frames_page(svc, complete, clips):
    s, wf = complete; s.step = "frames"
    s.frame_notes.append("The Viewpoint 1 · Left camera video was replaced, so Viewpoint 1 frame selections and annotations were discarded.")
    at = page_app(svc, s); at.run(); assert "was replaced" in texts(at) and s.frame_notes == []


def test_namespace_is_isolated_on_the_new_pages(svc, ready):
    s, wf = ready; s.step = "frames"; at = page_app(svc, s); at.run(); at.session_state["f8i_fs_mode"] = FP.MODES[1]; at.run()
    own = {k for k in list(at.session_state) if not str(k).startswith("_") and str(k) != "f8i_session"}
    assert own and all(str(k).startswith("f8i_") for k in own), own


def test_navigation_with_valid_uploads_room_to_frames_and_back_keeps_reconstruction_disabled(svc, ready):
    s, wf = ready; svc.room_report(s); assert s.room_ready; s.step = "room"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True)
    at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    assert not button(at, "Continue to Frame Selection").disabled
    button(at, "Continue to Frame Selection").click().run()
    assert not at.exception and s.step == "frames" and "Frames and boundaries" in texts(at)
    assert not has_button(at, "Calculate Wall Dimensions") and S.ROOM_PROCESSING_ENABLED is False and svc.room_processing_enabled() is False
    button(at, "Go back").click().run()
    assert s.step == "room" and not at.exception
