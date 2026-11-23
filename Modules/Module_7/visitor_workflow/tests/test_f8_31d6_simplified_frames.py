"""F8.31D-6: automatic frame selection, six boundary seeds marked once, propagation to five frame sets, partial failure.  SYNTHETIC fixtures only (generated clips of a drawn wall
corner / ceiling edge); no original recording is opened, Method J is never run, ROOM_PROCESSING_ENABLED stays False."""
import hashlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import f8_measurement as FM
import f8_visitor_service as S
import room_frames as RF
import room_reconstruction as RR
from room_measurement import lines as L, visitor_annotations as V
from test_f8_31b_frames import (LINE_POINTS, PEAK, accepted, clips, cstore, ready, rstore, svc, upload_all, H, W)       # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts

MODULE = Path(__file__).resolve().parents[2]
LIM = RR.ReconstructionLimits(min_usable_sets=RF.MIN_USABLE_SETS, verify_video_hashes=False)


def seed_all(wf):
    for line in RF.REQUIRED_LINES:
        wf.set_points(RF.SEED_SET, line, LINE_POINTS[line.split("_")[1]])
        wf._clear_derived(line)


def prepare(svc, s, wf, limits=LIM):
    return RR.prepare(wf, s.owner, s.contract, s.f6.job.cal, svc.room_sources(s), limits)


@pytest.fixture
def auto(ready):
    s, wf = ready
    wf.auto_select()
    return s, wf


@pytest.fixture
def applied(auto):
    s, wf = auto
    seed_all(wf); wf.apply_seeds()
    return s, wf


# ------------------------------------------------------------ 1-2 automatic five-set selection, independent indices
def test_auto_select_makes_five_confirmed_sets_without_any_review(ready):
    s, wf = ready
    assert wf.n_sets == RF.DEFAULT_SETS and not wf.pairs
    n = wf.auto_select()
    assert n == 20 and wf.frames_ready() and wf.all_pairs_confirmed()                              # 5 sets x 4 views extracted at full resolution
    assert all(wf.pairs[(vp, i)]["auto"] for vp in RF.VIEWPOINTS for i in range(5))
    assert wf.readiness().missing == [] or all("boundary" in m for m in wf.readiness().missing)    # frames need nothing more from the visitor


def test_auto_select_uses_the_original_rule_on_every_video_independently(ready):
    s, wf = ready; wf.auto_select()
    for vp in RF.VIEWPOINTS:
        for side in RF.SIDES:
            slot = f"p{vp}_{side}"; expected = RF.original_rule_indices(wf.scan_sharpness(slot), 5)
            got = [wf.pairs[(vp, i)][side] for i in range(5)]
            assert got == expected and got[0] == PEAK[slot]                                         # set 0 = the sharpest frame of THAT video
            assert len(set(got)) == 5                                                              # a frame is never chosen twice
    firsts = {slot: wf.pairs[(int(slot[1]), 0)][slot[3:]] for slot in PEAK}
    assert firsts == PEAK and len(set(firsts.values())) == 4                                       # four videos, four independent indices
    assert wf.pair_info(1, 0)["offset_frames"] != wf.pair_info(2, 0)["offset_frames"]              # not synchronised, not forced to a common offset


def test_motion_penalty_is_not_invented():
    src = (MODULE / "visitor_workflow" / "room_frames.py").read_text()
    body = src[src.index("def original_rule_indices"):src.index("# ------------------------------------------------------------------ identity")]
    assert "motion" in body.lower() and "not applied" in body                                       # documented as absent
    assert not re.search(r"penalt\w*\s*[*\-=]|\*\s*penalt", body.replace("small penalty", ""))


def test_auto_select_failure_invents_nothing(ready, monkeypatch):
    s, wf = ready
    monkeypatch.setattr(wf, "scan_sharpness", lambda slot: [1.0, 2.0])                               # a video with only two decodable frames
    with pytest.raises(RF.WorkflowError, match="decodable frames"):
        wf.auto_select()
    assert not wf.full and not wf.all_pairs_confirmed()


# ------------------------------------------------------------ 3 exactly six seeds
def test_exactly_six_seed_lines_on_set_zero(auto):
    s, wf = auto
    assert RF.REQUIRED_LINES == ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF") and len(RF.REQUIRED_LINES) == 6
    assert not wf.seeds_complete()
    for line in RF.REQUIRED_LINES[:-1]:
        wf.seed_click(line, *LINE_POINTS[line.split("_")[1]][0]); wf.seed_click(line, *LINE_POINTS[line.split("_")[1]][1])
    assert not wf.seeds_complete()                                                                 # five are not six
    wf.seed_click("PR_ceilF", *LINE_POINTS["ceilF"][0]); wf.seed_click("PR_ceilF", *LINE_POINTS["ceilF"][1])
    assert wf.seeds_complete() and all(k[0] == RF.SEED_SET for k in wf.ann)                         # only set 0 carries anything yet
    with pytest.raises(RF.WorkflowError):
        wf.state(0, "PL_ceilS")                                                                    # the two optional ceiling/side lines are never requested


def test_apply_requires_all_six_seeds(auto):
    s, wf = auto
    wf.set_points(0, "AL_corner", LINE_POINTS["corner"])
    with pytest.raises(RF.WorkflowError, match="all six"):
        wf.apply_seeds()


# ------------------------------------------------------------ 4 propagation across all five sets
def test_seeds_propagate_to_all_five_sets_and_are_accepted_automatically(applied):
    s, wf = applied
    assert wf.usable_sets() == [0, 1, 2, 3, 4] and wf.seeds_applied()
    for line in RF.REQUIRED_LINES:
        seed = wf.seed_points(line)
        for i in range(5):
            st = wf.ann[(i, line)]
            assert st.accepted and st.auto and st.points == seed and wf.line_report(i, line)["status"] in ("ok", "ok_warning")
            assert st.frame_index == wf.full[(i, line[:2])]["index"]                                # refined on ITS OWN set's frame, not set 0's


# ------------------------------------------------------------ 5 original refinement functions are reused
def test_refinement_calls_the_original_line_functions(auto, monkeypatch):
    s, wf = auto; calls = {"edge_points": 0, "fit_line": 0}
    real_e, real_f = L.edge_points, L.fit_line
    monkeypatch.setattr(L, "edge_points", lambda *a, **k: (calls.__setitem__("edge_points", calls["edge_points"] + 1), real_e(*a, **k))[1])
    monkeypatch.setattr(L, "fit_line", lambda *a, **k: (calls.__setitem__("fit_line", calls["fit_line"] + 1), real_f(*a, **k))[1])
    seed_all(wf); wf.apply_seeds()
    assert calls["edge_points"] >= 30 * 2 and calls["fit_line"] >= 30                                # six lines x five sets, two passes each


def test_original_parameter_map_is_explicit_and_applies_only_the_generalisable_setting():
    assert RF.RoomFrameWorkflow.limits_for("AL_corner") == V.DEFAULT_LIMITS == RF.RoomFrameWorkflow.limits_for("PR_corner")       # corner windows 45 vs 10 do not generalise
    assert RF.RoomFrameWorkflow.limits_for("PL_ceilF").search_half_window_px == 30 and RF.RoomFrameWorkflow.limits_for("PR_ceilF").search_half_window_px == 30
    for line in ("PL_ceilF", "AL_corner"):                                                           # nothing else differs from the validated Stage D defaults
        a, b = RF.RoomFrameWorkflow.limits_for(line), V.DEFAULT_LIMITS
        assert {k: v for k, v in a.__dict__.items() if k != "search_half_window_px"} == {k: v for k, v in b.__dict__.items() if k != "search_half_window_px"}
    assert all("why" in v for v in RF.ORIGINAL_REFINEMENT_MAP.values()) and len(RF.NOT_GENERALISED_SETTINGS) == 3
    src = (MODULE / "visitor_workflow" / "room_frames.py").read_text()
    assert not re.search(r"\bc0\b|\"slope\"|\"span\"", src)                                           # none of the owner's per-seed fields (position, slope, span) exist as data here


# ------------------------------------------------------------ 6 partial refinement failure
@pytest.fixture
def fail_sets(monkeypatch):
    """Make the edge search find nothing in the chosen sets (a flat frame) through the real Stage D function."""
    def install(bad):
        real = V.process_annotation
        def wrapper(raw, ctx, cal, *a, **k):
            if k.get("expected_set_index") in bad:
                ctx = V.CameraContext(np.full((H, W), 128, np.uint8), ctx.K, ctx.D)
            return real(raw, ctx, cal, *a, **k)
        monkeypatch.setattr(V, "process_annotation", wrapper)
    return install


def test_failed_set_is_reported_and_skipped_not_hidden(auto, fail_sets, svc):
    s, wf = auto; fail_sets({2}); seed_all(wf); wf.apply_seeds()
    assert wf.usable_sets() == [0, 1, 3, 4] and not wf.set_usable(2)
    assert all(wf.line_report(2, n)["status"] == "failed" for n in RF.REQUIRED_LINES) and wf.line_report(2, "AL_corner")["issues"]
    rd = wf.readiness(min_usable_sets=2)
    assert rd.ready and any("Frame set(s) 3 will be skipped" in n for n in rd.notes)
    assert not wf.readiness().ready                                                                  # strict mode (the earlier behaviour) still refuses an incomplete set
    p = prepare(svc, s, wf)
    assert p.request is not None and p.request.source_sets == (0, 1, 3, 4) and p.request.skipped_sets == (2,) and p.request.n_sets == 4 and len(p.request.line_sets) == 4
    assert [t["AL"] for t in p.request.frame_table] == [wf.full[(i, "AL")]["index"] for i in (0, 1, 3, 4)]
    assert prepare(svc, s, wf, RR.ReconstructionLimits(verify_video_hashes=False)).request is None   # default (strict) backend limits are unchanged


def test_too_few_usable_sets_is_a_clear_failure(auto, fail_sets, svc):
    s, wf = auto; fail_sets({1, 2, 3, 4}); seed_all(wf); wf.apply_seeds()
    assert wf.usable_sets() == [0]
    rd = wf.readiness(min_usable_sets=2)
    assert not rd.ready and any("at least 2" in m for m in rd.missing)
    p = prepare(svc, s, wf)
    assert p.request is None and [f["code"] for f in p.failures] == ["TOO_FEW_USABLE_SETS"]
    from visitor_messages import FAILURE_TEXT
    assert "TOO_FEW_USABLE_SETS" in FAILURE_TEXT and RR.failure_class_of("TOO_FEW_USABLE_SETS") == "visitor_input"


def test_a_stale_accepted_line_is_still_an_integrity_failure_not_a_skip(applied, svc):
    s, wf = applied
    wf.ann[(1, "AL_corner")].frame_fingerprint = "0" * 64                                            # accepted but made on another frame
    assert [f["code"] for f in prepare(svc, s, wf).failures] == ["ANNOTATION_STALE"]


# ------------------------------------------------------------ 7 no repetitive per-set confirmation
def test_warnings_are_recorded_automatically_not_asked_about(auto, monkeypatch, svc):
    s, wf = auto; real = V.process_annotation
    def warn(raw, ctx, cal, *a, **k):
        res = real(raw, ctx, cal, *a, **k)
        return replace(res, status=V.Status.WARNING, issues=list(res.issues) + [V.Issue(V.Status.WARNING, "WEAK_EDGE", "weak edge", raw.get("boundary_type", ""))]) if res.downstream_permitted else res
    monkeypatch.setattr(V, "process_annotation", warn); seed_all(wf); wf.apply_seeds()
    assert wf.usable_sets() == [0, 1, 2, 3, 4] and all(wf.ann[(i, n)].acknowledged_codes == ("WEAK_EDGE",) and wf.ann[(i, n)].auto for i in range(5) for n in RF.REQUIRED_LINES)
    p = prepare(svc, s, wf)
    assert p.request is not None and all(a["warnings_acknowledged_automatically"] and a["accepted_warning_codes"] == ["WEAK_EDGE"] for a in p.request.annotations)
    assert wf.line_report(0, "AL_corner")["status"] == "ok_warning"


def test_page_has_no_per_set_accept_refine_reject_controls():
    src = (MODULE / "visitor_workflow" / "f8_frames_page.py").read_text()
    for gone in ("Accept this boundary", "Reject and correct", "Refine boundary", "I looked at the green line", "Suggest Frames", "Accept these frames", "Use these points as the starting seed",
                 "f8i_an_set", 'st.radio("Step"'):
        assert gone not in src, gone


# ------------------------------------------------------------ 8 invalidation
def test_changing_a_seed_invalidates_every_derived_set_and_relocks_step_3(applied):
    s, wf = applied
    import f8_frames_page as FP
    assert FP.step_states(wf) == {1: "done", 2: "done", 3: "active"}
    wf.seed_click("PL_ceilF", 1000.0, 300.0)                                                        # moves a point of one seed
    assert all((i, "PL_ceilF") not in wf.ann for i in range(1, 5)) and not wf.set_usable(0)
    assert wf.usable_sets() == [] and not wf.seeds_applied() and FP.step_states(wf) == {1: "done", 2: "active", 3: "locked"}
    assert all((i, "AL_corner") in wf.ann for i in range(5))                                       # other boundaries are untouched


def test_changing_the_seed_frame_or_any_set_frame_drops_what_depended_on_it(applied):
    s, wf = applied
    before = {i: wf.ann[(i, "PL_corner")].revision for i in range(5)}
    p = wf.pairs[(1, 3)]; wf.set_pair(1, 3, p["left"] + 1 if p["left"] < 30 else p["left"] - 1, p["right"])      # a later set's LEFT frame changes
    assert not wf.set_usable(3) and wf.set_usable(0) and wf.ann[(3, "PL_corner")].revision == before[3]            # only that set's AL result is gone
    assert (3, "AL_corner") not in wf.ann and (3, "PL_corner") in wf.ann
    wf.set_pair(1, 0, wf.pairs[(1, 0)]["left"] + 1, wf.pairs[(1, 0)]["right"])                                     # the SEED frame of AL changes
    assert all((i, "AL_corner") not in wf.ann for i in range(5)) and wf.seed_points("AL_corner") == []             # the seed and everything derived are gone
    assert not wf.seeds_complete() and (0, "PL_corner") in wf.ann                                                   # the other views keep theirs


def test_new_calibration_discards_seeds_and_results(applied, svc):
    import copy
    s, wf = applied
    c2 = copy.deepcopy(s.contract); c2["baseline_mm"] = 230.0
    cal2 = type("C", (), dict(image_size=(W, H), K_left=wf.cal.K_left, D_left=wf.cal.D_left, K_right=wf.cal.K_right, D_right=wf.cal.D_right, baseline_mm=230.0))()
    notes = wf.sync(s.owner, c2, cal2, svc.room_sources(s))
    assert notes and not wf.ann and not wf.pairs and not wf.full and wf.seed_points("AL_corner") == []


def test_replaced_video_discards_that_viewpoints_seeds_and_results(applied, svc):
    s, wf = applied
    import dataclasses
    src = dict(svc.room_sources(s)); src["p1_left"] = dataclasses.replace(src["p1_left"], sha256="f" * 64, upload_id="replaced")
    notes = wf.sync(s.owner, s.contract, wf.cal, src)
    assert notes and all(k[1][:2] in ("AR", "PR") for k in wf.ann) and wf.seed_points("AL_corner") == [] and wf.seed_points("AR_corner") != []   # viewpoint 2 keeps its work


def test_measurement_binding_changes_when_a_seed_changes(applied):
    s, wf = applied
    b1 = FM.binding_of(s.owner, s.contract, wf, dict(wf.sources))
    wf.seed_click("AR_corner", 900.0, 400.0)
    b2 = FM.binding_of(s.owner, s.contract, wf, dict(wf.sources))
    assert b1["annotations"] != b2["annotations"] and b1["key"] != b2["key"]


# ------------------------------------------------------------ 9 locked steps in the real page (AppTest)
def _app(svc, s):
    s.step = "frames"; at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    return at


def test_page_runs_frame_selection_once_and_locks_later_steps(svc, ready):
    s, wf = ready; at = _app(svc, s)
    assert wf.frames_ready() and wf.all_pairs_confirmed()                                            # selected on arrival, no Suggest / Accept press
    assert [x.value for x in at.subheader] == ["Part 1 · Select Frames Automatically", "Part 2 · Confirm Wall Boundaries", "Part 3 · Review and Continue"]
    t = texts(at)
    assert "Locked. Confirm all six wall boundaries first." in t and "Locked. The frames are selected first." not in t
    assert not has_button(at, "Suggest Frames") and not has_button(at, "Accept these frames") and not has_button(at, "Refine boundary")
    assert "1 of 6" in t and button(at, "Next").disabled                                              # six confirmations first (F8.31F guided screens)
    log = list(wf.log)
    at.run(); assert wf.log == log                                                                   # a rerun reads no video again
    assert "keep each camera pair still" in t.lower()


def test_page_full_path_to_review_and_continue(svc, ready):
    from test_f8_31b_pages import finish_boundaries
    s, wf = ready; at = _app(svc, s)
    finish_boundaries(at, wf); assert not at.exception, at.exception
    assert wf.usable_sets() == [0, 1, 2, 3, 4]
    t = texts(at)
    assert "All five frame sets were processed." in t and "Locked." not in t
    assert not button(at, "Continue").disabled
    sub = [x.value for x in at.subheader]
    assert sub == ["Part 1 · Select Frames Automatically", "Part 2 · Confirm Wall Boundaries", "Part 3 · Review and Continue"]


def test_page_blocks_continue_when_sets_fail_the_strict_original_rule(svc, ready, fail_sets):
    from test_f8_31b_pages import finish_boundaries
    s, wf = ready; at = _app(svc, s); fail_sets({1, 2, 3, 4})
    finish_boundaries(at, wf); assert not at.exception, at.exception
    t = texts(at)
    assert "Boundary refinement failed" in t and "requires all five frame sets" in t
    assert button(at, "Continue").disabled


def test_page_one_failed_set_also_blocks_it_no_silent_skipping(svc, ready, fail_sets):
    from test_f8_31b_pages import finish_boundaries
    s, wf = ready; at = _app(svc, s); fail_sets({3})
    finish_boundaries(at, wf); assert not at.exception, at.exception
    t = texts(at)
    assert "Boundary refinement failed" in t and "requires all five frame sets" in t and button(at, "Continue").disabled


def test_automatic_selection_failure_is_explained_and_not_looped(svc, ready, monkeypatch):
    s, wf = ready
    monkeypatch.setattr(wf, "scan_sharpness", lambda slot: [1.0])
    at = _app(svc, s); t = texts(at)
    assert "could not be selected automatically" in t and "What to do" in t and has_button(at, "Try again") and not wf.pairs
    at.run(); assert not wf.pairs and wf.log == []                                                    # no retry loop, nothing invented


# ------------------------------------------------------------ 11 gate stays closed, nothing reconstructs
def test_gate_is_closed_and_no_reconstruction_happens(applied, svc):
    s, wf = applied
    assert S.ROOM_PROCESSING_ENABLED is False
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow" / "f8_visitor_service.py").read_text(), re.M)
    for name in ("room_frames.py", "f8_frames_page.py"):
        t = (MODULE / "visitor_workflow" / name).read_text()
        assert "room_reconstruction" not in t.replace("never", "") and "estimate_pose" not in t and "match_set" not in t
    with pytest.raises(FM.MeasurementError):
        svc.measurement.start(s)


# ------------------------------------------------------------ 12 original Method J / calibration mathematics are untouched
GUARDED = {
    "room_measurement/lines.py": "68cb593a02167361c3e091c661bbfe3b355e014b6b673497dce3424c7ec67613",
    "room_measurement/features.py": "fdbe36914b07824b7b7012ce06ea545797944169ecd0c339ad82f0d308ddd687",
    "room_measurement/pose.py": "ab7fed947e59db6341a4627ecf7d1054c733f5941a4929842768fa50a545e822",
    "room_measurement/geometry.py": "0150412b9c15bfe0f13632e6ac1f08c727f61b442ab59419043f4f853e3dc698",
    "room_measurement/measure.py": "35b9672ef5cfa9ba4bc7b7c6ffde49c97942eb56629b86da86d69b45d51ebae8",
    "room_measurement/calibration.py": "b2b6487b67a4358deff6bf7ec064a25075bba44b9c44a6236df381fb59fa6bbc",
    "room_measurement/visitor_method_j_adapter.py": "24204dcb59ffed66ef155f913855b4abfadc16a6967408b3b6556e5064749676",
    "room_measurement/visitor_annotations.py": "7243ea035b0083236cfb4705e97e6e94c50aed1ab8209c9cd13a4cc21bf4e759",
}


@pytest.mark.parametrize("rel", sorted(GUARDED))
def test_method_j_and_calibration_sources_are_byte_identical(rel):
    assert hashlib.sha256((MODULE / rel).read_bytes()).hexdigest() == GUARDED[rel], f"{rel} changed: F8.31D-6 must not touch Method J, line fitting or calibration mathematics"


def test_f8_31d6_adds_no_original_room_data():
    for name in ("room_frames.py", "f8_frames_page.py"):
        t = (MODULE / "visitor_workflow" / name).read_text()
        assert not re.search(r"\.MOV\b|DEFAULT_VIDEO_DIR|data/room_measurement|room_video_recipe|frame_manifest\.json['\"]|load_annotations|ground_truth", t), name


# ------------------------------------------------------------ 10 layout guards (the real-browser check is in the manual guide / browser run)
def test_layout_has_no_fixed_wide_elements_and_the_diagram_is_narrow():
    import f8_frames_page as FP
    svg = FP.boundary_svg("corner", "left") + FP.boundary_svg("ceilF", "right")
    assert all(int(w) <= 190 for w in re.findall(r"width='(\d+)'", svg)) and "viewBox" in svg and "aria-label" in svg
    src = (MODULE / "visitor_workflow" / "f8_frames_page.py").read_text()
    assert "st.columns(2)" in src and not re.search(r"width=\d{4}", src)
    assert "max_wh=(836, 900)" in src and 'width="stretch", cursor="crosshair"' in src                 # F8.31G: drawn at most column-wide (836 px) and stretched to the column, so a 390 px screen scales it down instead of clipping it
