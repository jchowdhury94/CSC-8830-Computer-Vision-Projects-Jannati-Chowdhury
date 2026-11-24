"""F8.31D-2: regression tests for the local website audit corrections (C1, H1-H3, M1-M7, L1-L7).  SYNTHETIC fixtures only: no original recording, no reconstruction,
no Method J.  Page-level tests use Streamlit's AppTest; browser behaviour (theme, scroll position) is verified separately in a real browser."""
import re
from pathlib import Path

import pytest

import f8_frames_page as FP
import f8_nav as NAV
import f8_result_view as RVW
import f8_ui_text as UT
import f8_visitor_pages as P
import f8_visitor_service as S
import room_frames as RF
import room_videos as RV
import visitor_messages as VM
from chunk_store import Limits
from test_f8_31a_room_videos import put
from test_f8_31b_frames import (annotate_all, choose_and_confirm, clips, cstore, complete, ready, rstore, svc, upload_all, accepted, PAIRS)    # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31c2a_measurement_interface import failed_result, finished, use_measurement          # noqa: F401
from test_f8_visitor_service import make_doc

HERE = Path(__file__).resolve().parents[1]
MODULE = HERE.parent
FORBIDDEN = ("F6", "can_proceed", "Method J", "local integration experiment", "SHA-256", "F8 ")


def _walk(node, out):
    for el in node.children.values():
        if getattr(el, "type", "") == "expander" and ("(for developers)" in str(getattr(el, "label", "")) or str(getattr(el, "label", "")) == "Technical Details"):
            continue                                                                                # developer diagnostics may name internal terms
        for attr in ("value", "label", "body"):
            if attr == "value" and getattr(el, "type", "") in ("selectbox", "radio", "select_slider", "number_input", "checkbox", "text_input"):
                continue                                                                            # a widget's internal value is not text the visitor reads
            v = getattr(el, attr, None)
            if v is not None and not callable(v):
                out.append(str(v))
        if getattr(el, "children", None):
            _walk(el, out)


def all_text(at):
    """Everything a visitor can read on the page: markdown, captions, alerts, labels, buttons, table cells, expander contents (except the developer-diagnostics expander)."""
    out = []
    _walk(at.main, out)
    return " ".join(out)


def confirmed_not_extracted(wf):
    wf.set_n_sets(2)
    for vp in RF.VIEWPOINTS:
        for i in range(2):
            l, r = PAIRS[vp][i]; wf.set_pair(vp, i, l, r); wf.preview(f"p{vp}_left", l); wf.preview(f"p{vp}_right", r); wf.confirm_pair(vp, i)


# ================================================================ C1
def test_c1_assigning_a_widget_key_after_the_widget_exists_is_the_original_crash():
    """Documents the Streamlit rule behind C1 (this is exactly what the old extract button did)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("import streamlit as st\nst.radio('Step', ['a', 'b'], key='m')\nif st.button('go'):\n    st.session_state['m'] = 'b'\n", default_timeout=30).run()
    at.button[0].click().run()
    assert at.exception and "cannot be modified after the widget" in str(at.exception[0].value)


def test_c1_a_stray_goto_value_is_ignored(svc, ready):
    s, wf = ready; s.step = "frames"; at = page_app(svc, s); at.session_state["f8i_fs_goto"] = "not a mode"; at.run()
    assert not at.exception and at.session_state["f8i_fs_mode"] == FP.MODES[1]          # frames are automatic: the visitor is at step 2


# ================================================================ H1
def test_h1_uploader_can_be_pinned_dark_and_the_default_is_unchanged():
    html = (MODULE / "f8_chunk_poc" / "frontend" / "index.html").read_text()
    assert ':root[data-theme="dark"]' in html and 'args.theme === "dark"' in html
    assert ":root:not([data-theme])" in html and "prefers-color-scheme: dark" in html          # no theme argument = the old behaviour (the F8 proof of concept)
    import inspect, component
    assert inspect.signature(component.chunk_uploader).parameters["theme"].default is None


def test_h1_every_uploader_of_the_visitor_app_is_dark():
    src = (HERE / "f8_visitor_pages.py").read_text()
    calls = re.findall(r"component\.chunk_uploader\(.*", src)
    assert len(calls) == 2 and all('theme="dark"' in c for c in calls)
    assert ":root {{ color-scheme: dark; }}" in src                                              # native widgets (the JSON file picker) stay dark on a light OS


# ================================================================ H2
STEP_SETUPS = ("home", "ready", "room", "frames", "measure")


def test_h2_back_targets_are_the_previous_step():
    s = S.VisitorSession()
    for step, back in (("home", None), ("setup", "home"), ("import", "home"), ("ready", "home")):
        s.step = step; assert NAV.back_target(s) == back
    s.review = S.review_f8_document(make_doc("GOOD")[1]); s.step = "review"; assert NAV.back_target(s) == "import"
    s.step = "room"; assert NAV.back_target(s) == "home"                                           # nothing accepted: the only place to go is the start


def test_h2_nav_row_is_on_every_page_and_home_and_back_keep_progress(svc, ready):
    s, wf = ready; s.step = "room"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); upload_state = svc.room_store.job_status(s.owner)["slots"].copy(); acc, con = s.accepted, dict(s.contract)
    at = page_app(svc, s); at.run(); assert not at.exception
    assert [b.label for b in at.button][:2] == ["Go back", "Camera Calibration"]
    assert not button(at, "Camera Calibration").disabled and not button(at, "Go back").disabled
    button(at, "Go back").click().run(); assert s.step == "ready"
    button(at, "Camera Calibration").click().run(); assert s.step == "home"
    assert s.accepted is acc and s.contract == con and s.f6 is not None and not s.f6.closed        # navigation never discards the calibration ...
    assert svc.room_store.job_status(s.owner)["slots"].keys() == upload_state.keys()               # ... or the uploads
    assert not has_button(at, "Camera Calibration") and not has_button(at, "Go back")              # first page: the row holds only links (Back leaves for the website's Room Measurement page, Home for its homepage)
    links = " ".join(h.proto.body for h in at.get("html"))
    assert 'aria-label="Go back"' in links and 'href="http://127.0.0.1:8501/module-7?view=3_visitor"' in links and 'href="http://127.0.0.1:8501/"' in links and "target" not in links
    s.step = "frames"; at.run()
    assert not has_button(at, "Start over") and not has_button(at, "Lock this session")            # F8.31F: not on the boundary page either
    s.step = "room"; at.run(); assert not has_button(at, "Start over") and not has_button(at, "Lock this session")     # F8.31D-4: not on the room-upload page
    assert button(at, "Camera Calibration").label == "Camera Calibration" and not any(b.label in ("Camera Calibration", "Go back") for b in at.button if "Start over" in b.label or "Lock" in b.label)


def test_h2_navigation_cannot_skip_readiness(svc, ready):
    s, wf = ready
    s.step = "ready"; s.room_ready = False
    with pytest.raises(S.ServiceError):
        s.go("frames")                                                                           # the nav row calls go(): same refusal
    with pytest.raises(S.ServiceError):
        s.go("measure")
    s2 = S.VisitorSession(); at = page_app(svc, s2); at.run()
    assert s2.step == "home" and not has_button(at, "Go back")


def test_h2_frames_back_returns_to_the_room_videos_and_forward_is_still_gated(svc, ready):
    s, wf = ready; s.step = "frames"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); at = page_app(svc, s); at.run()
    button(at, "Go back").click().run(); assert s.step == "room" and wf is s.frames                # selections are kept
    assert button(at, "Continue to Frame Selection").disabled is False


def test_h2_navigation_is_disabled_while_a_measurement_runs(svc, complete):
    import threading
    s, wf = complete; ev = threading.Event()
    ms = use_measurement(svc, runner=lambda ctx: (ev.wait(10), None)[1]); s.step = "measure"; ms.start(s)
    at = page_app(svc, s); at.session_state["f8i_m_autorefresh"] = False; at.run()
    assert button(at, "Go back").disabled and button(at, "Camera Calibration").disabled
    ev.set()
    ms.cancel(s)


# ================================================================ H3
@pytest.mark.parametrize("code,first,others", [("FEATURE_MATCHING_FAILED", "room", []), ("FRAME_SIZE_MISMATCH", "room", []), ("CALIBRATION_VIDEO_SIZE_MISMATCH", "room", []),
                                               ("UPLOAD_EXPIRED", "room", []), ("JOB_TOO_LARGE", "room", []), ("POSE_JOINT_NOT_CONVERGED", "frames", ["room"]),
                                               ("NO_USABLE_FRAME_SETS", "frames", ["room"]), ("INPUT_IMAGE_SIZE_MISMATCH", "home", ["frames"]),
                                               ("FLOOR_PLANE_FAILED", "frames", []), ("ANNOTATION_MISSING", "frames", []), ("CALIBRATION_NOT_ACCEPTED", "home", [])])
def test_h3_typed_codes_route_to_the_step_that_fixes_them(code, first, others):
    card = RVW.failure_card(failed_result(code))
    assert card["retry"]["step"] == first and [a["step"] for a in card["actions"]] == [first] + others


def test_h3_every_code_whose_advice_says_record_again_offers_the_room_videos():
    for code, (what, nxt) in VM.FAILURE_TEXT.items():
        if re.search(r"re-?record|record again|record at the|record the room|smaller recordings|upload .*again", nxt, re.I):
            steps = [a["step"] for a in RVW.failure_card(failed_result(code))["actions"]]
            assert "room" in steps, (code, nxt, steps)


def test_h3_routing_ignores_the_displayed_text(monkeypatch):
    before = RVW.failure_card(failed_result("FEATURE_MATCHING_FAILED"))["actions"]
    monkeypatch.setitem(VM.FAILURE_TEXT, "FEATURE_MATCHING_FAILED", ("Something else entirely.", "Go to the frames."))
    after = RVW.failure_card(failed_result("FEATURE_MATCHING_FAILED"))["actions"]
    assert before == after and after[0]["step"] == "room"


def test_h3_failed_result_page_buttons_lead_to_each_step(svc, complete):
    s, wf = complete
    finished(svc, s, failed_result("POSE_JOINT_NOT_CONVERGED", cls="method_j"))
    at = page_app(svc, s); at.run(); assert not at.exception
    labels = [b.label for b in at.button]
    assert "Go to frame selection and boundaries" in labels and "Go to the room videos" in labels and labels.index("Go to frame selection and boundaries") < labels.index("Go to the room videos")
    button(at, "Go to the room videos").click().run(); assert s.step == "room" and not at.exception
    s.step = "measure"; at = page_app(svc, s); at.run()
    button(at, "Go to frame selection and boundaries").click().run(); assert s.step == "frames"


def test_h3_unavailable_measurement_does_not_offer_a_dead_retry(svc, complete):
    s, wf = complete
    finished(svc, s, failed_result("WORKER_TIMEOUT", cls="worker_resource")); svc.measurement.gate_open = lambda *a: False
    at = page_app(svc, s); at.run(); assert not has_button(at, "Try again") and "nothing to retry" in texts(at)


# ================================================================ M1
def test_m1_scroll_to_top_is_emitted_only_when_the_view_changes(svc, ready):
    s, wf = ready; s.step = "room"; at = page_app(svc, s); at.run()
    first = at.session_state["f8i_scroll_n"]
    at.run(); assert at.session_state["f8i_scroll_n"] == first                                      # an ordinary rerun never scrolls
    button(at, "Go back").click().run(); assert at.session_state["f8i_scroll_n"] == first + 1       # a new page does
    at.run(); assert at.session_state["f8i_scroll_n"] == first + 1


def test_m1_the_frames_substep_counts_as_a_new_view(svc, ready):
    s, wf = ready; s.step = "frames"; at = page_app(svc, s); at.run()
    at.run(); n = at.session_state["f8i_scroll_n"]; assert at.session_state["f8i_fs_mode"] == FP.MODES[1]
    at.run(); assert at.session_state["f8i_scroll_n"] == n                                           # nothing changed: no scroll
    from test_f8_31b_pages import finish_boundaries
    finish_boundaries(at, wf, {"corner": [(894.0, 350.0), (912.0, 950.0)], "ceilF": [(1050.0, 270.0), (1650.0, 282.0)]})
    assert at.session_state["f8i_fs_mode"] == FP.MODES[2] and at.session_state["f8i_scroll_n"] == n + 1     # exactly one scroll, for the new step


# ================================================================ M2
def test_m2_boundary_labels_are_words_and_unique():
    labels = [UT.boundary_label(n) for n in RF.REQUIRED_LINES]
    assert len(set(labels)) == 6 == len(set(UT.boundary_row_label(n) for n in RF.REQUIRED_LINES))
    for n, lab in zip(RF.REQUIRED_LINES, labels):
        vp, cam, kind = UT.boundary_parts(n)
        assert (vp, cam) == (RF.VIEWPOINT_OF_VIEW[n[:2]], {"AL": "left", "AR": "left", "PL": "right", "PR": "right"}[n[:2]])
        assert re.search(r"Viewpoint [12] · (Left|Right) camera · (Wall corner|Ceiling boundary)", lab) and not re.search(r"\b(AL|AR|PL|PR)\b", lab)


def test_m2_the_annotation_step_shows_no_abbreviations(svc, ready):
    s, wf = ready; s.step = "frames"; wf.auto_select(); at = page_app(svc, s); at.run()
    assert not at.exception
    t = all_text(at)
    assert "Viewpoint 1 · Left camera · Wall corner" in t and not at.radio                                  # F8.31F: one screen per boundary, named in words
    tech = [e for e in at.expander if e.label == "Technical Details"]; assert tech and any("AL_corner" in m.value for m in tech[0].markdown)
    visible = t
    for m in tech[0].markdown:
        visible = visible.replace(m.value, "")
    assert not re.search(r"\b(AL|AR|PL|PR)(_corner|_ceilF)\b", visible)                                    # identifiers live only inside Technical Details


# ================================================================ M3
@pytest.mark.parametrize("step", ["room", "frames", "ready", "measure"])
def test_m3_no_internal_terms_on_visitor_pages(svc, complete, step):
    s, wf = complete; s.step = step
    if step == "measure":
        use_measurement(svc, gate=False)
    at = page_app(svc, s); at.run(); assert not at.exception
    visible = all_text(at)
    for term in FORBIDDEN:
        assert term not in visible, (step, term)
    if step == "ready":
        assert not any("Technical details" in str(e.label) for e in at.expander) and "developers" not in visible and "can_proceed" not in visible


def test_m3_annotation_and_review_steps_have_no_internal_terms(svc, complete):
    s, wf = complete; s.step = "frames"
    at = page_app(svc, s); at.run(); assert not at.exception                                         # step 2 (and step 3 once applied, as `complete` has)
    for term in FORBIDDEN:
        assert term not in all_text(at), term


def test_m3_hashes_move_to_developer_details(svc, complete):
    s, wf = complete; use_measurement(svc, gate=False); s.step = "measure"; at = page_app(svc, s); at.run()
    assert "SHA-256" not in " ".join(m.value for m in at.markdown) and any(e.label == "Technical Details" for e in at.expander)


# ================================================================ M4
def test_m4_retention_text_matches_the_backend_limits():
    lim = Limits(complete_retention_s=900.0, idle_timeout_s=300.0, job_max_s=7200.0)
    t = " ".join(UT.retention_lines(lim))
    assert "this browser session" in t and "temporarily" in t and "15 minutes" in t and "5 minutes" in t and "2 hours" in t and "upload it again" in t
    assert "permanent" not in t.lower().replace("permanently", "") and "immediately after" not in t.lower()
    t2 = " ".join(UT.retention_lines(Limits(complete_retention_s=1200.0, idle_timeout_s=60.0, job_max_s=3600.0)))
    assert "20 minutes" in t2 and "1 minute " in t2 + " " and "60 minutes" in t2                                          # derived from the limits, not hard-coded


def test_m4_default_backend_limits_are_what_the_text_says():
    lim = Limits(); assert (lim.complete_retention_s, lim.idle_timeout_s, lim.job_max_s) == (900.0, 300.0, 7200.0)


def test_m4_room_page_shows_retention_before_the_uploaders(svc, ready):
    s, wf = ready; s.step = "room"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); at = page_app(svc, s); at.run(); assert not at.exception
    t = texts(at); assert "How long are my videos kept?" not in " ".join(str(e.label) for e in at.expander) and "15 minutes" in t       # F8.31E: a concise line, no dropdown
    assert "Start over" not in t and "Lock this session" not in t


def test_m4_expired_upload_message_uses_the_same_numbers():
    assert "15 minutes" in UT.retention_short(Limits()) and "5 minutes" in UT.retention_short(Limits())


# ================================================================ M5
def test_m5_ready_page_back_keeps_everything_and_replacing_the_calibration_invalidates(svc, ready):
    s, wf = ready; s.step = "ready"; frames = s.frames; at = page_app(svc, s); at.run(); assert not at.exception
    assert "replaces" in texts(at) and not has_button(at, "Start calibration")
    button(at, "Go back").click().run(); assert s.step == "home" and s.accepted is not None and s.frames is frames
    s.go("ready"); assert s.step == "ready" and s.accepted is not None
    svc.import_saved(s, b"not json")                                                                # an actual replacement attempt
    assert s.accepted is None and s.frames is None and not s.room_ready                              # ... invalidates every dependent input


# ================================================================ M6
def _slot(slot, status):
    return RV.SlotReport(slot, status)


def test_m6_only_accepted_videos_count():
    rep = RV.RoomSetReport(slots={"p1_left": _slot("p1_left", RV.OK), "p1_right": _slot("p1_right", RV.WARN), "p2_left": _slot("p2_left", RV.ERROR), "p2_right": _slot("p2_right", RV.MISSING)})
    assert rep.n_valid == 2 and rep.n_rejected == 1 and rep.n_complete == 3 and not rep.ready            # n_complete (uploads received) is unchanged; the visitor sees n_valid


def test_m6_the_page_shows_accepted_not_uploaded(svc, ready):
    s, wf = ready; s.step = "room"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True); at = page_app(svc, s); at.run()
    assert "4 of 4 videos accepted" in " ".join(str(p.proto.text) for p in at.get("progress"))


def test_m6_a_rejected_upload_does_not_reach_four(svc, clips):
    s = accepted(svc); upload_all(svc, s, clips, only=["p1_left", "p1_right", "p2_left"])
    rep = svc.room_report(s); assert rep.n_valid <= 3 and not rep.ready


# ================================================================ M7
def test_m7_baseline_phrase_follows_the_provenance():
    assert UT.baseline_line(226, "calibrated") == "Baseline 226 mm (estimated from the calibration)"
    assert UT.baseline_line(226, "measured") == "Baseline 226 mm (measured by you)"
    assert "measured" not in UT.baseline_line(226, "calibrated") and "measured" not in UT.baseline_line(226, "file") and "measured" not in UT.baseline_line(226, "entered")
    assert "not recorded" in UT.baseline_line(226, None)


def test_m7_context_carries_the_real_source_and_pages_show_it(svc):
    s = accepted(svc); ctx = svc.calibration_context(s); assert ctx["baseline_source"] == "calibrated" and ctx["baseline_mm"] == 226.0
    s.step = "ready"; at = page_app(svc, s); at.run(); assert "estimated from the calibration" in texts(at) and "measured by you" not in texts(at)
    code, doc = make_doc("GOOD"); doc["settings"]["measured_baseline_mm"] = 230.0
    s2 = S.VisitorSession(); svc._test_sessions.append(s2)
    s2.review = S.review_f8_document(doc, measured_baseline_mm=230.0); svc.accept(s2)
    ctx2 = svc.calibration_context(s2); assert ctx2["baseline_source"] == "measured" and ctx2["baseline_mm"] == 230.0
    s2.step = "ready"; at = page_app(svc, s2); at.run(); assert "measured by you" in texts(at)


# ================================================================ L1-L7
def test_l1_l2_no_duplicate_page_heading_and_back_is_only_in_the_nav_row(svc, ready):
    s, wf = ready
    for step in ("room", "frames"):
        s.step = step; at = page_app(svc, s); at.run()
        heads = [m.value for m in at.markdown if m.value.startswith("### ")]
        assert not heads, (step, heads)
        if step == "room":
            assert not any(b.label.startswith("Back") for b in at.button)                             # one standard Back control, in the nav row (the frames page's Back/Next move between the six boundary screens)
        assert [b.label for b in at.button][0] == "Go back"


def test_l3_pages_use_the_module_style_navigation_containers(svc, ready):
    s, wf = ready; s.step = "room"; at = page_app(svc, s); at.run()
    keys = {b.key for b in at.button}; assert {"f8i_nav_back", "f8i_nav_calibration"} <= keys
    import home_navigation as HN
    assert "st-key-nav_row" in HN._CSS


def test_l4_no_colour_emoji_indicators_on_the_pages(svc, complete):
    bad = re.compile("[⬜✅⛔⚠\U0001f50e✏⏳⬛]")                       # white/black square, check, no-entry, warning, magnifier, pencil, hourglass
    for step in ("room", "frames", "ready"):
        s, wf = complete; s.step = step; at = page_app(svc, s); at.run(); assert not at.exception
        assert not bad.search(all_text(at)), step
    s, wf = complete; s.step = "frames"
    for mode in FP.MODES:
        at = page_app(svc, s); at.session_state["f8i_fs_mode"] = mode; at.run(); assert not bad.search(all_text(at)), mode
    for src in ("f8_visitor_pages.py", "f8_frames_page.py", "f8_measure_page.py"):
        assert not bad.search((HERE / src).read_text()), src


def test_l5_progress_is_not_complete_before_a_measurement():
    base = P.PROGRESS
    for step in base:
        assert UT.progress_for(step, base)[0] < 1.0
    assert UT.progress_for("measure", base, measurement_complete=True) == (1.0, "Measurement complete")
    assert UT.progress_for("measure", base)[0] <= 0.95


def test_l5_page_bar_is_not_full_on_the_measure_page_without_a_result(svc, complete):
    s, wf = complete; use_measurement(svc, gate=False); s.step = "measure"; at = page_app(svc, s); at.run()
    assert "width:100%" not in " ".join(m.value for m in at.markdown)
    finished(svc, s, __import__("test_f8_31c2a_measurement_interface").ok_result()); at = page_app(svc, s); at.run()
    assert "width:100%" not in " ".join(m.value for m in at.markdown)          # one finished wall is Step 1 of 2: the bar is never full (see test_f8_33_second_wall for the finished room)


def test_l6_saved_calibration_labels_are_plain():
    assert set(UT.KIND_TEXT) == {"f8_job", "f8_recovered", "f8_file", "contract_file", "phase1_file"}                       # the internal kinds are unchanged
    for v in UT.KIND_TEXT.values():
        assert not re.search(r"\bF8\b|contract|Phase 1", v), v


def test_l7_a_rejected_room_video_is_explained_once(svc, clips):
    s = accepted(svc); upload_all(svc, s, clips); s.step = "room"; s.guide.room.set_confirmed(1, True); s.guide.room.set_confirmed(2, True)
    at = page_app(svc, s); at.run()
    md = [m.value for m in at.markdown]
    assert sum("videos accepted" in str(p.proto.text) for p in at.get("progress")) == 1 and sum("How long are my videos kept" in str(e.label) for e in at.expander) == 0
    assert not any("Select a video file to upload in chunks." in m for m in md)                     # that text lives inside the component iframe, once per box
    assert not any(m.strip() in ("**Viewpoint 1 · Left camera video**",) for m in md)                # the box label is drawn by the uploader itself, not repeated above it


def test_camera_calibration_selection_page_uses_the_shared_card_design_and_website_breadcrumb(svc):
    s = S.VisitorSession(); at = page_app(svc, s); at.run(); assert not at.exception and s.step == "home"
    assert [h.value for h in at.header] == ["Camera Calibration"] and not at.title                  # the old oversized page title is gone
    text = " ".join(m.value for m in at.markdown)
    assert "MODULE 7" in text and "Choose how you'd like to get started." in text
    assert [b.key for b in at.button] == ["f8i_pick_new", "f8i_pick_import"]                       # two stacked cards, nothing else (no bottom Back)
    assert [b.label for b in at.button] == ["**Calibrate My Cameras**  \nPerform a new stereo camera calibration", "**Upload Saved Calibration**  \nUse your existing camera calibration file"]
    links = " ".join(h.proto.body for h in at.get("html"))
    for label in ("Home", "Module 7", "Run Your Own Experiment", "Room Measurement"):
        assert f">{label}</a>" in links, label
    assert "view=run_own" in links and "view=3_visitor" in links
    button(at, "**Calibrate My Cameras**  \nPerform a new stereo camera calibration").click().run(); assert s.step == "setup"
    s.step = "home"; at.run()
    button(at, "**Upload Saved Calibration**  \nUse your existing camera calibration file").click().run(); assert s.step == "import"
