"""F8.31G: red line between the two points, zoom toolbar (no expander), Step 2 / Step 3 wording, grouped failure summary, direct correction of one boundary.
SYNTHETIC fixtures only; no original recording is opened; Method J is never run; ROOM_PROCESSING_ENABLED stays False.  UI, drawing and message tests: no algorithm is changed."""
import re
from pathlib import Path

import numpy as np
import pytest

import f8_frames_page as FP
import f8_visitor_service as S
import quality_feedback as QF
import room_frames as RF
from annotation_ui import core as C
from room_measurement import visitor_annotations as V
from test_f8_31b_frames import LINE_POINTS, accepted, clips, cstore, ready, rstore, svc, upload_all, H, W        # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31d6_simplified_frames import auto, applied, seed_all                                                  # noqa: F401  (fixtures / helpers)
from test_f8_31f_boundary_review import patch_suggestions, _app                                                     # noqa: F401  (fixtures / helpers)

MODULE = Path(__file__).resolve().parents[2]
RED = np.array(C.COLORS["manual"])
PAGE_SRC = (MODULE / "visitor_workflow" / "f8_frames_page.py").read_text()


# ------------------------------------------------------------------ helpers
def red_at(img, uv, r=2):
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = img[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1].reshape(-1, 3).astype(int)
    return bool((np.abs(patch - RED).sum(axis=1) < 60).any())


def count_red(img):
    return int((np.abs(img.astype(int) - RED).sum(axis=2) < 40).sum())


@pytest.fixture
def spy(monkeypatch):
    """Record every render_view call of the boundary page (the arguments ARE what the visitor sees)."""
    calls = []
    real = FP.f1.render_view
    def wrapper(gray, view, points=(), clicked_poly=None, refined_poly=None, edit_index=None, manual_pts=None):
        calls.append(dict(points=[list(p) for p in points], amber=clicked_poly is not None, green=refined_poly is not None, red=None if manual_pts is None else [list(p) for p in manual_pts]))
        return real(gray, view, points, clicked_poly, refined_poly, edit_index, manual_pts)
    monkeypatch.setattr(FP.f1, "render_view", wrapper)
    return calls


@pytest.fixture
def fail_lines(monkeypatch):
    """Make the real Stage D function reject the chosen boundaries (wrong direction) while ``active`` is true."""
    cfg = dict(lines=set(), active=True)
    real = V.process_annotation
    def wrapper(raw, ctx, cal, *a, **k):
        if cfg["active"] and any(f"_{n}_" in raw.get("annotation_id", "") for n in cfg["lines"]):
            x = raw["points"][0][0]; raw = dict(raw, points=[[x, 100.0], [x + 5.0, 500.0]])      # a near-vertical "ceiling" line: the real Stage D orientation check rejects it
        return real(raw, ctx, cal, *a, **k)
    monkeypatch.setattr(V, "process_annotation", wrapper)
    return cfg


def confirm_all(at, wf):
    for k in range(6):
        button(at, "Looks Correct").click().run(); assert not at.exception, at.exception
        button(at, "Finish" if k == 5 else "Next").click().run(); assert not at.exception, at.exception


def outside_expanders(at):
    """Visible text that is NOT inside an expander (Technical Details)."""
    inner = set()
    for e in at.expander:
        inner |= {m.value for m in e.markdown} | {c.value for c in e.caption}
    return " ".join(m.value for m in at.markdown if m.value not in inner) + " " + " ".join(m.value for m in list(at.error) + list(at.warning) + list(at.success) if m.value not in inner) \
        + " " + " ".join(c.value for c in at.caption if c.value not in inner)


# ================================================================== 1. red line: drawing
GRAY_SIZES = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]


@pytest.mark.parametrize("wh", GRAY_SIZES)
def test_one_point_shows_only_its_marker_two_points_draw_a_solid_red_line(wh):
    w, h = wh; gray = np.full((h, w), 90, np.uint8); view = C.make_view((w, h), C.ViewState(), max_wh=(720, 900))
    p1, p2 = [0.30 * w, 0.20 * h], [0.34 * w, 0.80 * h]
    one = C.render_view(gray, view, [p1])
    assert count_red(one) == 0                                                  # one click: marker only (white), no line
    two = C.render_view(gray, view, [p1, p2], manual_pts=[p1, p2])
    assert count_red(two) > 50
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):                                       # the red pixels follow the straight segment between the two SAVED points
        x, y = (1 - t) * p1[0] + t * p2[0], (1 - t) * p1[1] + t * p2[1]
        assert red_at(two, view.to_view(x, y)), t


@pytest.mark.parametrize("wh", GRAY_SIZES)
@pytest.mark.parametrize("zoom", [1.0, 2.0, 4.0, 8.0])
def test_red_line_stays_aligned_under_zoom_and_pan(wh, zoom):
    w, h = wh; gray = np.full((h, w), 90, np.uint8)
    p1, p2 = [0.45 * w, 0.40 * h], [0.50 * w, 0.60 * h]
    vs = C.ViewState()
    for _ in range(int(np.log2(zoom))):
        vs = C.zoom_by((w, h), vs, 2.0)
    vs = C.center_on((w, h), vs, ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)) if zoom > 1 else vs
    view = C.make_view((w, h), vs, max_wh=(720, 900))
    img = C.render_view(gray, view, [p1, p2], manual_pts=[p1, p2])
    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    assert view.contains(*mid) and red_at(img, view.to_view(*mid))
    # the click mapping inverts the drawing mapping: clicking where an endpoint is drawn returns that endpoint (within one display pixel)
    u, v = (float(c) for c in view.to_view(*p1))
    back = view.click_to_full(dict(x=int(u), y=int(v), width=view.shown_wh[0], height=view.shown_wh[1]))
    assert abs(back[0] - p1[0]) <= 1.0 / view.sx + 1 and abs(back[1] - p1[1]) <= 1.0 / view.sy + 1
    panned = C.pan_by((w, h), vs, 0.25, 0) if zoom > 1 else vs
    v2 = C.make_view((w, h), panned, max_wh=(720, 900)); img2 = C.render_view(gray, v2, [p1, p2], manual_pts=[p1, p2])
    if v2.contains(*mid):
        assert red_at(img2, v2.to_view(*mid))


def test_render_view_without_manual_pts_is_unchanged():
    gray = np.full((480, 640), 90, np.uint8); view = C.make_view((640, 480))
    a = C.render_view(gray, view, [[100, 100], [120, 300]]); b = C.render_view(gray, view, [[100, 100], [120, 300]], None, None, None, None)
    assert np.array_equal(a, b) and count_red(a) == 0


# ================================================================== 2. red line in the page: persistence, Correct Line, colour meanings
def test_page_draws_amber_for_a_suggestion_and_red_only_after_confirmation(svc, ready, patch_suggestions, spy):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    assert spy[-1]["amber"] and spy[-1]["red"] is None and not spy[-1]["green"]            # unconfirmed suggestion: amber dashed only
    button(at, "Looks Correct").click().run(); assert not at.exception, at.exception
    saved = wf.seed_points("AL_corner")
    assert spy[-1]["red"] == saved and not spy[-1]["amber"] and not spy[-1]["green"]       # confirmed: the SAVED points joined in red; not green; amber gone


def test_one_click_shows_the_point_and_no_line_two_clicks_the_line(svc, ready, patch_suggestions, spy):
    patch_suggestions(available=False); s, wf = ready; at = _app(svc, s)
    wf.seed_click("AL_corner", *LINE_POINTS["corner"][0]); at.run()
    assert len(spy[-1]["points"]) == 1 and spy[-1]["red"] is None and not spy[-1]["green"] and button(at, "Next").disabled           # incomplete: not confirmed
    wf.seed_click("AL_corner", *LINE_POINTS["corner"][1]); at.run()
    assert spy[-1]["red"] == wf.seed_points("AL_corner") == [list(map(float, p)) for p in LINE_POINTS["corner"]]                   # exactly the saved coordinates
    assert not spy[-1]["green"] and not button(at, "Next").disabled


def test_red_line_survives_reruns_and_back_next(svc, ready, patch_suggestions, spy):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    button(at, "Looks Correct").click().run(); saved = wf.seed_points("AL_corner")
    at.run(); at.run(); assert spy[-1]["red"] == saved                                       # plain reruns
    button(at, "Next").click().run(); assert "2 of 6" in texts(at) and spy[-1]["red"] is None and spy[-1]["amber"]          # another boundary: its own state
    button(at, "Back").click().run(); assert "1 of 6" in texts(at) and spy[-1]["red"] == saved and wf.seed_points("AL_corner") == saved


def test_correct_line_clears_the_old_line_and_allows_two_new_clicks(svc, ready, patch_suggestions, spy):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    button(at, "Looks Correct").click().run()
    button(at, "Correct Line").click().run()
    assert wf.seed_points("AL_corner") == [] and spy[-1]["red"] is None and spy[-1]["points"] == [] and button(at, "Next").disabled
    wf.seed_click("AL_corner", 200.0, 100.0); at.run()
    assert spy[-1]["red"] is None and button(at, "Next").disabled and not wf.seeds_complete()                  # an incomplete new line is never confirmed
    wf.seed_click("AL_corner", 210.0, 400.0); at.run()
    assert spy[-1]["red"] == [[200.0, 100.0], [210.0, 400.0]] and not button(at, "Next").disabled


def test_green_only_for_a_real_accepted_refinement(svc, ready, patch_suggestions, spy):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    confirm_all(at, wf)
    assert wf.seeds_applied() and not any(c["green"] for c in spy)                           # two clicks / confirmations alone never produce green
    wf.reopen_boundary("PL_ceilF"); st = __import__("streamlit").session_state
    at.session_state["f8i_bd_idx"] = 0; at.run(); assert not at.exception, at.exception
    assert "1 of 6" in texts(at) and spy[-1]["green"] and spy[-1]["red"] is not None        # an accepted refinement of THIS boundary exists: green appears
    at.session_state["f8i_bd_idx"] = 2; at.run()
    assert "3 of 6" in texts(at) and not spy[-1]["green"] and spy[-1]["red"] == wf.seed_points("PL_ceilF")   # the reopened boundary has no result: red only


# ================================================================== 3. zoom toolbar
def test_no_zoom_expander_and_the_toolbar_buttons_are_visible(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    assert "Zoom" not in [e.label for e in at.expander] and "Technical Details" in [e.label for e in at.expander]
    for label in ("−", "+", "Fit"):
        assert has_button(at, label), label
    assert 'st.expander("Zoom")' not in PAGE_SRC and "f8i_bd_toolbar" in PAGE_SRC
    assert not has_button(at, "←") and not has_button(at, "→")                                # panning only when zoomed in


def test_zoom_in_out_fit_and_pan_never_touch_saved_points(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    button(at, "Looks Correct").click().run(); saved = wf.seed_points("AL_corner"); rev = wf.state(0, "AL_corner").revision
    button(at, "+").click().run(); assert not at.exception, at.exception
    views = at.session_state["f8i_an_view"]; (key, vs), = [(k, v) for k, v in views.items() if k[1] == "AL_corner"]; assert vs.zoom == 2.0
    assert has_button(at, "←") and has_button(at, "↓")                                         # pan controls appear when zoomed
    button(at, "→").click().run(); vs2 = [v for k, v in at.session_state["f8i_an_view"].items() if k[1] == "AL_corner"][0]; assert vs2.cx != vs.cx or vs2.zoom == 2.0
    button(at, "−").click().run(); button(at, "+").click().run(); button(at, "+").click().run()
    assert [v for k, v in at.session_state["f8i_an_view"].items() if k[1] == "AL_corner"][0].zoom == 4.0
    button(at, "Fit").click().run()
    assert not [k for k in at.session_state["f8i_an_view"] if k[1] == "AL_corner"]
    assert wf.seed_points("AL_corner") == saved and wf.state(0, "AL_corner").revision == rev   # coordinates and revision untouched: no point was added or moved


def test_zoom_state_is_kept_while_annotating_and_keyed_to_the_frame(svc, ready, patch_suggestions):
    patch_suggestions(available=False); s, wf = ready; at = _app(svc, s)
    button(at, "+").click().run()
    wf.seed_click("AL_corner", 200.0, 100.0); at.run()
    assert [v for k, v in at.session_state["f8i_an_view"].items() if k[1] == "AL_corner"][0].zoom == 2.0          # zoom kept between the two clicks
    assert 'vkey = (i, line, wf.full[(i, vk)]["fingerprint"]' in PAGE_SRC                       # a different camera frame has a different view key, so it starts at Fit


# ================================================================== 4. wording: Step 2 versus Step 3
def test_step2_wording_reports_only_what_happened():
    manual = {n: "visitor" for n in FP.BOUNDARY_ORDER}
    assert FP.completion_text(manual) == "All six wall boundaries have been marked. Review the refinement results below."
    assert FP.completion_text({**manual, "AL_corner": "suggestion"}) == "All six wall boundaries have been confirmed. Review the refinement results below."
    assert FP.completion_text(None) == "All six wall boundaries have been confirmed. Review the refinement results below."
    assert "refined in all five" not in PAGE_SRC and "confirmed and refined" not in PAGE_SRC


def test_step2_and_step3_agree_on_success(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s); confirm_all(at, wf)
    t = texts(at)
    assert "All six wall boundaries have been confirmed. Review the refinement results below." in t
    assert "All five frame sets were processed" in t and "Boundary refinement failed" not in t and not has_button(at, "Correct Boundary")
    assert not button(at, "Continue").disabled


def test_step2_never_claims_refinement_when_step3_reports_failure(svc, ready, patch_suggestions, fail_lines):
    patch_suggestions(); fail_lines["lines"] = {"PL_ceilF", "PR_ceilF"}
    s, wf = ready; at = _app(svc, s); confirm_all(at, wf); t = texts(at)
    assert "All six wall boundaries have been confirmed. Review the refinement results below." in t
    assert "Boundary refinement failed" in t and "refined successfully" not in t and "refined in all" not in t and "All five frame sets were processed" not in t
    assert button(at, "Continue").disabled


# ================================================================== 5. grouped failure summary and guidance
def test_failure_summary_is_short_grouped_and_actionable(svc, ready, patch_suggestions, fail_lines):
    patch_suggestions(); fail_lines["lines"] = {"PL_ceilF", "PR_ceilF"}
    s, wf = ready; at = _app(svc, s); confirm_all(at, wf)
    assert wf.usable_sets() == []                                                           # every set failed for those two boundaries (real Stage D on a flat frame)
    main = outside_expanders(at)
    assert "Boundary refinement failed" in main
    assert "Two ceiling boundaries could not be refined. Please correct the highlighted boundaries before continuing." in main
    assert "Viewpoint 1 · Right camera · Ceiling boundary" in main and "Viewpoint 2 · Right camera · Ceiling boundary" in main
    assert sum(1 for b in at.button if b.label == "Correct Boundary") == 2
    assert len(re.findall(r"select two points along the horizontal line where the wall meets the ceiling", main)) == 1, main[-1400:]                # no per-frame-set repetition of the advice
    assert main.count("Frame set") == 0 and "PL_ceilF" not in main and "REFINEMENT_FAILED" not in main and "frame set(s)" not in main
    assert QF.FRAME_SET_GUIDANCE in main and QF.RIG_HINT in main
    assert "motion was detected" not in main and "blur" not in main.lower() and "lighting" not in main.lower()    # no cause is invented


def test_technical_details_keep_every_original_diagnostic(svc, ready, patch_suggestions, fail_lines):
    patch_suggestions(); fail_lines["lines"] = {"PL_ceilF"}
    s, wf = ready; at = _app(svc, s); confirm_all(at, wf)
    tech = [e for e in at.expander if e.label == "Technical Details"][-1]
    body = " ".join(m.value for m in tech.markdown) + " " + " ".join(c.value for c in tech.caption)
    assert "PL_ceilF" in body and "failed in frame sets 1, 2, 3, 4, 5" in body
    raw = {(m["code"], m["message"]) for i in range(5) for m in wf.line_report(i, "PL_ceilF")["issues"]}
    assert raw and all(code in body and msg in body for code, msg in raw)                      # the backend diagnostics are listed (grouped, not dropped)


def test_failure_helpers_group_by_cause_without_inventing_one():
    fails = [dict(line="AL_corner", kind="corner", causes=["direction"]), dict(line="PL_ceilF", kind="ceilF", causes=["direction", "not_located"]), dict(line="PR_ceilF", kind="ceilF", causes=["not_located"])]
    g = QF.failure_guidance(fails)
    assert len(g) == len(set(g)) == 3                                                          # corner direction, ceiling direction, not located: each ONCE
    assert any("select two points along the horizontal line where the wall meets the ceiling" in x for x in g)
    assert any("vertical line where the two walls meet" in x for x in g)
    assert QF.failure_headline(fails).startswith("Three boundaries (1 wall corner and 2 ceiling boundaries) could not be refined.")
    assert QF.failure_headline(fails[:1]).startswith("One wall corner could not be refined. Please correct the highlighted boundary before")
    assert QF.cause_of("UNREASONABLE_ORIENTATION") == "direction" and QF.cause_of("SOMETHING_ELSE") == "not_located"


def test_grouping_is_presentation_only_nothing_is_suppressed(applied, fail_lines):
    s, wf = applied
    before = {(i, n): wf.line_report(i, n) for i in range(5) for n in RF.REQUIRED_LINES}
    QF.boundary_failures(wf); QF.boundary_warnings(wf)
    assert before == {(i, n): wf.line_report(i, n) for i in range(5) for n in RF.REQUIRED_LINES}


# ================================================================== 6. direct correction, recovery
def test_correct_boundary_opens_that_boundary_keeps_the_others_and_recovers(svc, ready, patch_suggestions, fail_lines, spy):
    patch_suggestions(); fail_lines["lines"] = {"PL_ceilF"}
    s, wf = ready; at = _app(svc, s); confirm_all(at, wf)
    seeds = {n: wf.seed_points(n) for n in RF.REQUIRED_LINES}
    others = {n: [wf.line_report(i, n)["status"] for i in range(5)] for n in RF.REQUIRED_LINES if n != "PL_ceilF"}
    assert sum(1 for b in at.button if b.label == "Correct Boundary") == 1
    button(at, "Correct Boundary").click().run(); assert not at.exception, at.exception
    t = texts(at)
    assert "3 of 6" in t and "Viewpoint 1 · Right camera · Ceiling boundary" in t                # straight to the failed boundary
    assert spy[-1]["red"] == seeds["PL_ceilF"] and not spy[-1]["green"]                         # its existing points and red line are shown
    assert {n: wf.seed_points(n) for n in RF.REQUIRED_LINES} == seeds                           # nothing was cleared: every seed is kept
    assert all([wf.line_report(i, n)["status"] for i in range(5)] == st for n, st in others.items())      # the other five boundaries keep their results
    assert all(wf.line_report(i, "PL_ceilF")["status"] == "not_run" for i in range(5))           # only this boundary's results were dropped
    assert FP.step_states(wf)[3] == "locked" and not has_button(at, "Continue")
    # replace the line (Correct Line, two clicks), then refine again without visiting the other boundaries
    fail_lines["active"] = False
    button(at, "Correct Line").click().run(); assert wf.seed_points("PL_ceilF") == [] and button(at, "Next").disabled
    wf.seed_click("PL_ceilF", *LINE_POINTS["ceilF"][0]); at.run(); assert spy[-1]["red"] is None
    wf.seed_click("PL_ceilF", *LINE_POINTS["ceilF"][1]); at.run(); assert spy[-1]["red"] is not None
    assert has_button(at, "Refine All Boundaries Now")
    button(at, "Refine All Boundaries Now").click().run(); assert not at.exception, at.exception
    t = texts(at)
    assert wf.seeds_applied() and wf.usable_sets() == [0, 1, 2, 3, 4] and "All five frame sets were processed" in t and "Boundary refinement failed" not in t
    assert {n: wf.seed_points(n) for n in RF.REQUIRED_LINES if n != "PL_ceilF"} == {n: v for n, v in seeds.items() if n != "PL_ceilF"}      # unrelated boundaries were not re-marked
    assert not button(at, "Continue").disabled


def test_reopen_boundary_is_per_boundary_and_validated(applied):
    s, wf = applied
    wf.reopen_boundary("AR_corner")
    assert all(wf.line_report(i, "AR_corner")["status"] == "not_run" for i in range(5)) and not wf.seeds_applied()
    assert all(wf.line_report(i, n)["status"] == "ok" for i in range(5) for n in RF.REQUIRED_LINES if n != "AR_corner") and len(wf.seed_points("AR_corner")) == 2
    with pytest.raises(RF.WorkflowError):
        wf.reopen_boundary("PL_ceilS")


# ================================================================== 7. layout (CSS / structure; the real-browser check covers 1300 px and 390 px)
def test_toolbar_and_failure_rows_are_laid_out_for_desktop_and_390px():
    css = (MODULE / "visitor_workflow" / "f8_visitor_pages.py").read_text()
    block = css[css.index("F8.31G: zoom toolbar"):css.index("F8.31D-5: the keyed step BOX")]
    assert "st-key-f8i_bd_toolbar" in block and "min-width: 40px" in block and "min-height: 40px" in block                  # touch-sized targets
    assert 'horizontal=True, horizontal_alignment="right"' in PAGE_SRC and "wrap=False" in PAGE_SRC[PAGE_SRC.index("def _toolbar"):PAGE_SRC.index("def _step2")]        # one row, right-aligned
    assert "flex: 1 1 14rem" in block                                                                                       # a failed-boundary row wraps its button below the label on a narrow screen
    assert "width: 34px" in css and "border: 2px solid {ACCENT}" in css                                                       # the remove-video circles are untouched


# ================================================================== 8. safety
def test_gate_closed_reconstruction_untouched_and_no_algorithm_edited():
    assert S.ROOM_PROCESSING_ENABLED is False
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow" / "f8_visitor_service.py").read_text(), re.M)
    for forbidden in ("estimate_pose", "match_set", "import room_reconstruction", "edge_points", "fit_line"):
        assert forbidden not in PAGE_SRC, forbidden
    qf = (MODULE / "visitor_workflow" / "quality_feedback.py").read_text()
    assert "edge_points" not in qf and "fit_line" not in qf
    core = (MODULE / "annotation_ui" / "core.py").read_text()
    assert not re.search(r"lines\.(edge_points|fit_line)|import lines|fit_lines", core)
