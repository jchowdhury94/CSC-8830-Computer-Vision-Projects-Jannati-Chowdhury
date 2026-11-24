"""F8.31F: optional F2 boundary suggestions, the guided six-boundary review, STRICT original Method J set requirements, quality feedback, remove-video styling.
SYNTHETIC fixtures only; no original recording is opened; Method J is never run; ROOM_PROCESSING_ENABLED stays False.  These tests do NOT establish real-world detection accuracy."""
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import boundary_suggestions as BS
import f8_frames_page as FP
import f8_measurement as FM
import f8_visitor_service as S
import quality_feedback as QF
import room_frames as RF
import room_reconstruction as RR
import room_videos as RV
from room_measurement import lines as L, measure as M, stage_f2_boundary_detection as F2, visitor_annotations as V
from test_f8_31b_frames import LINE_POINTS, accepted, clips, cstore, ready, rstore, svc, upload_all, H, W        # noqa: F401  (fixtures)
from test_f8_31b_pages import button, has_button, page_app, texts
from test_f8_31d6_simplified_frames import auto, applied, fail_sets, prepare, seed_all                              # noqa: F401  (fixtures / helpers)

MODULE = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ helpers
def fake_suggestion(line, offset=0.0, status="UNCERTAIN", available=True):
    kind = line.split("_")[1]; p = [(x + offset, y) for x, y in LINE_POINTS[kind]]
    (x1, y1), (x2, y2) = p
    if kind == "corner":
        a = (x2 - x1) / (y2 - y1); ln = dict(kind="v", a=a, b=x1 - a * y1, span=(y1, y2))
    else:
        a = (y2 - y1) / (x2 - x1); ln = dict(kind="h", a=a, b=y1 - a * x1, span=(x1, x2))
    if not available:
        return BS._none(line, "NOT_DETECTED", ["no candidate line was found"])
    return dict(line=line, available=True, status=status, reasons=["synthetic"], points_distorted_full=[list(q) for q in p], line_undistorted=ln,
                f2=dict(visible_length_px=600.0, confuser_class="ISOLATED", support_fraction=0.9, rank_score=1.0, n_candidates=3))


@pytest.fixture
def patch_suggestions(monkeypatch):
    def install(offset=0.0, available=True, bad=()):
        def fake(wf):
            return dict(key=BS.fingerprints(wf), by_line={n: fake_suggestion(n, 150.0 if n in bad else offset, available=available) for n in RF.REQUIRED_LINES})
        monkeypatch.setattr(BS, "suggest_for_workflow", fake)
    return install


def _app(svc, s):
    s.step = "frames"; at = page_app(svc, s); at.run(); assert not at.exception, at.exception
    return at


# ================================================================== 1. F2 suggestions: generation, reuse, fallback
def test_suggestions_run_the_existing_f2_detector_on_the_visitors_own_frames(auto, monkeypatch):
    s, wf = auto; seen = {}
    real = F2.detect_set
    def spy(frames, cal, **k):
        seen["views"] = sorted(frames); seen["shapes"] = {k_: v.shape for k_, v in frames.items()}
        return real(frames, cal, **k)
    monkeypatch.setattr(F2, "detect_set", spy)
    out = BS.suggest_for_workflow(wf)
    assert seen["views"] == ["AL", "AR", "PL", "PR"] and all(sh == (H, W) for sh in seen["shapes"].values())
    assert set(out["by_line"]) == set(RF.REQUIRED_LINES) and len(out["by_line"]) == 6
    assert out["key"] == BS.fingerprints(wf) and len(out["key"]) == 4
    for n, sg in out["by_line"].items():
        assert sg["status"] in ("DETECTED", "UNCERTAIN", "NOT_DETECTED") and isinstance(sg["available"], bool)
        if sg["available"]:                                                    # points are in the (distorted) frame, two of them, finite
            pts = np.asarray(sg["points_distorted_full"]); assert pts.shape == (2, 2) and np.all(pts >= 0) and np.all(pts[:, 0] <= W - 1) and np.all(pts[:, 1] <= H - 1)


def test_suggestions_are_never_accepted_by_themselves(auto):
    s, wf = auto
    BS.suggest_for_workflow(wf)
    assert not any(wf.seed_points(n) for n in RF.REQUIRED_LINES) and not wf.seeds_complete()      # computing a suggestion stores no seed point


def test_f2_failure_gives_no_suggestion_and_manual_path_still_works(auto, monkeypatch):
    s, wf = auto
    monkeypatch.setattr(F2, "detect_set", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = BS.suggest_for_workflow(wf)["by_line"]
    assert all(not v["available"] and v["status"] == "NOT_DETECTED" and v["points_distorted_full"] is None for v in out.values())
    seed_all(wf); res = wf.apply_seeds()                                                              # manual two-point seeds still reach the unchanged refinement
    assert wf.usable_sets() == [0, 1, 2, 3, 4] and len(res) == 30


def test_a_candidate_outside_the_frame_is_not_offered(auto, monkeypatch):
    s, wf = auto; real = F2.detect_set
    def shifted(frames, cal, **k):
        r = real(frames, cal, **k)
        for b in r["boundaries"].values():
            for c in b.candidates:
                c["endpoints_undistorted_full"] = [[-500.0, 10.0], [-500.0, 900.0]]
        return r
    monkeypatch.setattr(F2, "detect_set", shifted)
    out = BS.suggest_for_workflow(wf)["by_line"]
    assert all(not v["available"] for v in out.values() if v["status"] != "NOT_DETECTED" or v["reasons"])


def test_new_modules_use_no_ml_no_original_room_data_and_no_reconstruction():
    for name in ("boundary_suggestions.py", "quality_feedback.py"):
        t = (MODULE / "visitor_workflow" / name).read_text()
        assert not re.search(r"import (torch|tensorflow|transformers|onnx|keras|sklearn)|from (torch|transformers)", t), name
        assert not re.search(r"\.MOV\b|DEFAULT_VIDEO_DIR|data/room_measurement|frame_manifest|wall_boundary_annotations|load_annotations|ground_truth|estimate_pose|match_set|room_reconstruction", t), name
    assert "no ML/DL" in (MODULE / "visitor_workflow" / "boundary_suggestions.py").read_text()


# ================================================================== 2. guided six-step review (page)
def test_first_screen_shows_one_boundary_and_no_radio_list(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s); t = texts(at)
    assert "1 of 6" in t and "Viewpoint 1 · Left camera · Wall corner" in t
    assert not any(r.label == "Boundary to mark" for r in at.radio) and not at.radio                  # no six-option radio list
    assert "Technical Details" in [e.label for e in at.expander]
    assert has_button(at, "Looks Correct") and has_button(at, "Correct Line") and has_button(at, "Back") and button(at, "Back").disabled
    assert button(at, "Next").disabled                                                                # nothing confirmed yet
    assert not wf.seed_points("AL_corner")                                                           # a shown suggestion is NOT accepted
    for gone in ("PL_ceilF", "PR_corner", "Check this boundary", "Apply to all five frame sets", "boundary_svg", "Place both points"):
        assert gone not in t.replace("`", ""), gone                                                  # identifiers live only inside Technical Details
    assert not has_button(at, "Start over") and not has_button(at, "Lock this session")


def test_looks_correct_confirms_the_suggestion_then_next_advances(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    button(at, "Looks Correct").click().run(); assert not at.exception, at.exception
    assert wf.seed_points("AL_corner") == [list(map(float, p)) for p in LINE_POINTS["corner"]]
    assert "You confirmed the suggested line." in texts(at) and not button(at, "Next").disabled
    button(at, "Next").click().run(); assert not at.exception, at.exception
    t = texts(at); assert "2 of 6" in t and "Viewpoint 1 · Right camera · Wall corner" in t
    button(at, "Back").click().run()
    assert "1 of 6" in texts(at) and wf.seed_points("AL_corner")                                       # Back keeps the confirmed line


def test_full_path_six_confirmations_then_automatic_refinement_of_five_sets(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    for k in range(6):
        assert f"{k + 1} of 6" in texts(at)
        button(at, "Looks Correct").click().run(); assert not at.exception, at.exception
        button(at, "Finish" if k == 5 else "Next").click().run(); assert not at.exception, at.exception
    assert wf.seeds_complete() and wf.seeds_applied() and wf.usable_sets() == [0, 1, 2, 3, 4]         # propagated and refined in all five sets without per-set approval
    assert len({tuple(map(tuple, wf.seed_points(n))) for n in RF.REQUIRED_LINES}) == 2                 # exactly six seeds (two shapes in the synthetic frames)
    assert sum(1 for k in wf.ann if k[0] == RF.SEED_SET) == 6
    t = texts(at)
    assert "All five frame sets were processed" in t and not button(at, "Continue").disabled
    assert not has_button(at, "Looks Correct") and not has_button(at, "Next")


def test_wrong_suggestion_is_corrected_with_two_clicks(svc, ready, patch_suggestions):
    patch_suggestions(bad=("AL_corner",)); s, wf = ready; at = _app(svc, s)
    button(at, "Correct Line").click().run(); assert not at.exception, at.exception                    # the visitor sees the amber line is off and chooses to correct it
    assert not wf.seed_points("AL_corner") and not has_button(at, "Looks Correct") and button(at, "Next").disabled
    wf.seed_click("AL_corner", *LINE_POINTS["corner"][0]); at.run()                                    # first click
    assert "Now click the second point." in texts(at) and button(at, "Next").disabled
    wf.seed_click("AL_corner", *LINE_POINTS["corner"][1]); at.run()                                    # second click
    assert not button(at, "Next").disabled and wf.seed_points("AL_corner") == [list(map(float, p)) for p in LINE_POINTS["corner"]]
    assert "Line marked." in texts(at)                                                                # NOT recorded as a confirmed suggestion
    assert wf.seed_points("AL_corner") != [[x + 150.0, y] for x, y in LINE_POINTS["corner"]]


def test_correct_line_after_confirming_removes_the_old_points(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    button(at, "Looks Correct").click().run()
    button(at, "Correct Line").click().run()
    assert wf.seed_points("AL_corner") == [] and button(at, "Next").disabled


def test_no_suggestion_means_manual_marking_only(svc, ready, patch_suggestions):
    patch_suggestions(available=False); s, wf = ready; at = _app(svc, s); t = texts(at)
    assert "No suggestion was found for this line. Mark it yourself." in t and not has_button(at, "Looks Correct")
    wf.seed_click("AL_corner", *LINE_POINTS["corner"][0]); wf.seed_click("AL_corner", *LINE_POINTS["corner"][1]); at.run()
    assert not button(at, "Next").disabled


def test_suggestion_failure_in_the_real_page_falls_back_to_manual(svc, ready, monkeypatch):
    monkeypatch.setattr(F2, "detect_set", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    s, wf = ready; at = _app(svc, s)
    assert not has_button(at, "Looks Correct") and "Mark it yourself" in texts(at) and not at.exception


def test_technical_details_hold_the_identifiers_and_the_f2_status(svc, ready, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    tech = [e for e in at.expander if e.label == "Technical Details"][0]
    body = " ".join(m.value for m in tech.markdown)
    assert "AL_corner" in body and "UNCERTAIN" in body and "not verified" in " ".join(c.value for c in tech.caption)


def test_changed_seed_frame_makes_old_suggestions_stale(auto, patch_suggestions):
    s, wf = auto; k1 = BS.fingerprints(wf)
    p = wf.pairs[(1, 0)]; wf.set_pair(1, 0, p["left"] + 1, p["right"])
    assert BS.fingerprints(wf) != k1 or len(BS.fingerprints(wf)) < 4                                    # the suggestion key follows the set-0 frames


def test_reopen_keeps_seeds_and_reapplies(applied):
    s, wf = applied; seeds = {n: wf.seed_points(n) for n in RF.REQUIRED_LINES}
    wf.seeds_reopen()
    assert not wf.seeds_applied() and {n: wf.seed_points(n) for n in RF.REQUIRED_LINES} == seeds and FP.step_states(wf)[3] == "locked"
    wf.apply_seeds(); assert wf.usable_sets() == [0, 1, 2, 3, 4]


# ================================================================== 3. STRICT original Method J set requirements
def test_visitor_default_limits_are_strict_everywhere(svc):
    assert RR.ReconstructionLimits().min_usable_sets is None
    assert svc.measurement.trusted_limits().min_usable_sets is None                                    # the normal visitor path
    src = (MODULE / "visitor_workflow" / "f8_measurement.py").read_text()
    assert "ReconstructionLimits(min_usable_sets" not in src.replace("(min_usable_sets=None)", "")


def test_one_failed_set_blocks_the_visitor_path_before_pose(auto, fail_sets, svc):
    s, wf = auto; fail_sets({2}); seed_all(wf); wf.apply_seeds()
    assert wf.usable_sets() == [0, 1, 3, 4] and not wf.readiness().ready                              # strict readiness: all five sets needed
    p = prepare(svc, s, wf, svc.measurement.trusted_limits().__class__(verify_video_hashes=False))
    assert p.request is None and "ANNOTATION_MISSING" in [f["code"] for f in p.failures]               # nothing is silently dropped; no request, so no pose estimate can start
    p2 = prepare(svc, s, wf, replace(svc.measurement.trusted_limits(), verify_video_hashes=False))
    assert p2.request is None


def test_all_five_sets_reach_the_request_unchanged_in_the_strict_path(applied, svc):
    s, wf = applied
    p = prepare(svc, s, wf, RR.ReconstructionLimits(verify_video_hashes=False))
    assert p.request is not None and p.request.n_sets == 5 and p.request.source_sets == (0, 1, 2, 3, 4) and p.request.skipped_sets == () and len(p.request.line_sets) == 5


def test_page_blocks_continue_when_any_set_fails_and_names_what_to_do(svc, ready, fail_sets, patch_suggestions):
    patch_suggestions(); s, wf = ready; at = _app(svc, s)
    fail_sets({3})
    for k in range(6):
        button(at, "Looks Correct").click().run(); button(at, "Finish" if k == 5 else "Next").click().run(); assert not at.exception, at.exception
    t = texts(at)
    assert wf.usable_sets() == [0, 1, 2, 4] and button(at, "Continue").disabled
    assert "requires all five frame sets" in t
    assert "Boundary refinement failed" in t and "Correct Boundary" in [b.label for b in at.button]


def test_original_pipeline_semantics_are_what_the_tests_assume():
    """The original (read-only inspection): line fitting for ALL sets precedes pose, which pools ALL sets; a failed final measurement of one set is recorded; none OK raises."""
    src = (MODULE / "room_measurement" / "pipeline.py").read_text()
    assert src.index("line_sets = [L.fit_lines(s, cal, ann) for s in sets]") < src.index("P.estimate_pose(D)")
    assert "D = P.build_dataset(cal, match_sets)" in src and 'raise RuntimeError("no frame set produced a measurement")' in src
    cal = SimpleNamespace(baseline_mm=225.0)
    rows = M.measure_all_sets(cal, np.zeros(5), [{}], [{}])                                           # a set that cannot be measured is RECORDED, not raised
    assert len(rows) == 1 and rows[0]["ok"] is False and "error" in rows[0]


def test_adapter_keeps_the_original_final_measurement_rule():
    src = (MODULE / "room_measurement" / "visitor_method_j_adapter.py").read_text()
    assert "NO_USABLE_FRAME_SETS" in src and "SOME_FRAME_SETS_FAILED" in src and "if not ok:" in src      # >= 1 OK set needed; some failing sets are flagged, not hidden


# ================================================================== 4. quality feedback (real triggers only)
def test_calibration_feedback_from_real_message_templates():
    c = ["The left single-camera reprojection RMS is 2.10 px, above the heuristic guide of 1.00 px.", "The board did not appear in enough regions of the frame (fewer than 6 of 9); place it elsewhere.",
         "Only 7 valid pairs were used; 12-20 are recommended."]
    items = QF.calibration_feedback("questionable", c, stereo_rms_px=1.4)
    assert [i.state for i in items] == [QF.NEEDS] * 3 and all(i.kind == QF.RECALIBRATE and i.record_again.startswith("No") and i.affected for i in items)
    assert "reprojection error" in items[0].what and "every region" in items[1].action and items[0].detail == c[0]
    assert QF.overall(items) == QF.NEEDS


def test_calibration_states_failed_completed_and_not_assessed():
    assert QF.calibration_feedback(None)[0].state == QF.NOT_ASSESSED
    f = QF.calibration_feedback("failed", problems=["bad matrix"])[0]; assert f.state == QF.FAILED and f.kind == QF.RECALIBRATE
    d = QF.calibration_feedback("good", stereo_rms_px=0.8); assert d[0].state == QF.DONE and "0.80 px" in d[0].what
    assert QF.overall([], assessed=False) == QF.NOT_ASSESSED
    u = QF.calibration_feedback("questionable", ["something new the checks said"])[0]; assert u.state == QF.NEEDS and u.detail == "something new the checks said"


def test_blur_and_camera_settings_are_not_claimed_because_nothing_detects_them():
    src = (MODULE / "visitor_workflow" / "quality_feedback.py").read_text()
    body = src[src.index("_CAL_RULES"):src.index("def calibration_feedback")]
    assert "blur" not in body.lower() and "exposure" not in body.lower() and "focus" not in body.lower()
    assert "NOT assessed" in src and "blur" in src                                                       # the limitation is stated


def _upload(name, wh, fps=30.0, n=120, sha=None):
    return dict(name=name, size_bytes=5_000_000, sha256=sha or name * 4, state="complete",
                inspection=dict(decodable=True, width=wh[0], height=wh[1], fps=fps, frame_count=n, duration_s=n / fps, codec="h264", errors=[]))


def test_video_feedback_names_the_video_and_says_whether_to_record_again():
    cal = dict(image_size_wh=[1080, 1920], capture_mode=dict(left="video", right="video"))
    ups = {"p1_left": _upload("a", (1080, 1920)), "p1_right": _upload("b", (1920, 1080)), "p2_left": _upload("c", (1080, 1920))}
    rep = RV.assess_room_set(ups, cal); items = QF.video_feedback(rep)
    bad = [i for i in items if i.state == QF.FAILED]
    assert bad and any("orientation" in i.what.lower() or "frame sizes" in i.what.lower() for i in bad)
    assert any(i.affected == rep.slots["p1_right"].label and i.record_again == "Yes" for i in bad)
    assert [i for i in items if i.state == QF.NOT_ASSESSED and i.affected == rep.slots["p2_right"].label]       # not uploaded yet = not assessed, never "ok"
    assert QF.overall(items) == QF.FAILED


def test_video_feedback_completed_when_checks_pass():
    cal = dict(image_size_wh=[1080, 1920], capture_mode=dict(left="video", right="video"))
    ups = {s: _upload(s, (1080, 1920), sha=s * 8) for s in RV.ROOM_UPLOAD_SLOTS}
    items = QF.video_feedback(RV.assess_room_set(ups, cal))
    assert QF.overall(items) in (QF.DONE, QF.NEEDS) and not [i for i in items if i.state == QF.FAILED]


def test_boundary_feedback_not_assessed_before_apply(auto):
    s, wf = auto
    assert QF.boundary_feedback(wf)[0].state == QF.NOT_ASSESSED                                          # nothing applied yet


def test_boundary_feedback_states(applied):
    s, wf = applied
    items = QF.boundary_feedback(wf)
    assert QF.overall(items) in (QF.DONE, QF.NEEDS) and not [i for i in items if i.state == QF.FAILED]
    if QF.overall(items) == QF.DONE:
        assert "all 5 frame sets" in items[0].what


def test_boundary_feedback_failed_names_the_camera_and_the_action(auto, fail_sets):
    s, wf = auto; fail_sets({2}); seed_all(wf); wf.apply_seeds()
    items = QF.boundary_feedback(wf); failed = [i for i in items if i.state == QF.FAILED]
    assert len(failed) == 6 and all(i.kind == QF.CORRECT and "frame set(s) 3" in i.what and i.record_again == "No" for i in failed)
    assert {i.affected for i in failed} >= {"Viewpoint 1 · left camera · wall corner", "Viewpoint 2 · right camera · ceiling edge"}
    assert all("record this viewpoint again" in i.action for i in failed)                                # only suggested for a possible rig move, not demanded


def test_reconstruction_feedback_uses_backend_codes_and_stays_honest():
    f = QF.failure_item("FEATURE_MATCHING_FAILED")
    assert f.state == QF.FAILED and f.affected == "Matching the two cameras" and f.kind == QF.RERECORD and f.record_again == "Yes" and f.detail == "FEATURE_MATCHING_FAILED"
    assert QF.failure_item("REGISTRATION_FAILED").affected == "Joining the two viewpoints"
    assert QF.failure_item("WORKER_TIMEOUT").kind == QF.RETRY and QF.failure_item("WORKER_TIMEOUT").record_again == "No"
    g = QF.failure_item("SOMETHING_UNKNOWN"); assert g.state == QF.FAILED and g.what and "Nothing was concluded" in g.what
    w = QF.warning_item("REGISTRATION_HIGH_RMS"); assert w.state == QF.NEEDS and w.stage == "reconstruction"
    done = QF.reconstruction_feedback(completed=True)
    assert done[0].state == QF.DONE and "does not mean the result is accurate" in done[0].what and "accurate" not in done[0].action
    assert QF.reconstruction_feedback(assessed=False)[0].state == QF.NOT_ASSESSED
    assert QF.reconstruction_feedback(completed=False) == []                                              # nothing ran and nothing failed: no invented "completed"


def test_every_known_failure_code_has_visitor_feedback():
    import visitor_messages as VM
    for code in VM.FAILURE_TEXT:
        i = QF.failure_item(code); assert i.what and i.kind in (QF.RETRY, QF.CORRECT, QF.RECALIBRATE, QF.RERECORD, QF.REUPLOAD, QF.NONE) and i.state == QF.FAILED


def test_failure_card_carries_the_feedback_without_changing_the_decision():
    import f8_result_view as RVW
    card = RVW.failure_card(dict(status="failed", failures=[dict(code="FLOOR_PLANE_FAILED", failure_class="method_j", stage="floor", message="m")]))
    assert card["code"] == "FLOOR_PLANE_FAILED" and card["feedback"]["affected"] == "Floor" and card["feedback"]["record_again"] == "Yes" and card["retry"]["kind"] == "fix"


# ================================================================== 5. remove-video control
def test_remove_button_styling_and_tooltip():
    src = (MODULE / "visitor_workflow" / "f8_visitor_pages.py").read_text()
    css = src[src.index("F8.31F: remove-video control"):src.index("F8.31D-5: the keyed step BOX")]
    assert "width: 34px" in css and "height: 34px" in css and "border-radius: 50%" in css and "border: 2px solid {ACCENT}" in css and "color: #FFFFFF" in css
    assert 'help="Remove video"' in src and "Remove this video" not in src
    assert "flex: 0 0 34px" in css                                                                        # never squeezed on a 390 px screen
    row = src[src.index("with st.container(horizontal=True, wrap=False"):src.index("if ss.get(\"f8i_rm_pending\") == slot")]
    assert row.count("r.facts['name']") == 1                                                              # the filename is written once in the card row


# ================================================================== 6. safety
def test_gate_stays_closed_and_nothing_reconstructs():
    assert S.ROOM_PROCESSING_ENABLED is False
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow" / "f8_visitor_service.py").read_text(), re.M)
    t = (MODULE / "visitor_workflow" / "f8_frames_page.py").read_text()
    assert "estimate_pose" not in t and "match_set" not in t and "import room_reconstruction" not in t
