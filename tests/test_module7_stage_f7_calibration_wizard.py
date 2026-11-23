"""Phase 2B Stage F7: guided visitor calibration wizard (Modules/Module_7/visitor_workflow/f7_calibration.py + f7_pages.py).

The pipeline is exercised on SYNTHETIC stereo videos rendered from a known rig (known baseline, known left/right time offset), so the recovered values can be
checked against ground truth.  Wizard rules (one step at a time, validation, no automatic acceptance, FAILED blocked, QUESTIONABLE needs acknowledgement) are
tested on the pure state machine and, for what the page shows, with Streamlit's AppTest (no browser)."""
import ast
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
WF = MODULE / "visitor_workflow"
for p in (str(MODULE), str(WF), str(ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path: sys.path.insert(0, p)

import f7_calibration as K  # noqa: E402
import f6_core as F  # noqa: E402
from room_measurement import visitor_calibration as VC  # noqa: E402
from stereo_measure import user_calibration as uc  # noqa: E402
from test_module7_user_calibration import render_board, rot, sample_views  # noqa: E402

PATTERN, SQ_MM = (8, 6), 30.0
SIZE = (960, 1280)                                          # (w, h) portrait, both cameras
TRUE_T = np.array([[-120.0], [2.0], [3.0]])


def rig():
    K1 = np.array([[1100, 0, 480], [0, 1100, 640], [0, 0, 1]], float); K2 = np.array([[1050, 0, 470], [0, 1050, 650], [0, 0, 1]], float)
    return dict(K1=K1, D1=np.zeros(5), K2=K2, D2=np.zeros(5), R=rot(0.01, np.radians(3), -0.015), T=TRUE_T)


def write_videos(tmp, n_poses=26, hold=12, offset=4, seed=1, size=SIZE, fps=30, name="v"):
    """Left / right videos of a board held still for ``hold`` frames per pose; the right video starts ``offset`` frames LATER than the left one."""
    r = rig(); _, _, poses = sample_views(r, n_poses, seed=seed, noise=0.0, size_l=size, size_r=size, roll=0.3)
    paths = {}
    for side in ("left", "right"):
        p = Path(tmp) / f"{name}_{side}.mp4"; vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, size); rng = np.random.default_rng(5 if side == "left" else 6)
        hrng = np.random.default_rng(11); holds = hrng.integers(max(hold - 4, 8), hold + 14, len(poses)); seq = [(i, k) for i, _ in enumerate(poses) for k in range(int(holds[i]))]
        if side == "right": seq = [seq[0]] * offset + seq                                  # right video starts `offset` frames after the left one: its frame i+offset shows left frame i
        for pi, _ in seq:
            Rb, tb = poses[pi]
            img, _ = render_board(size, r["K1"], Rb, tb, PATTERN, SQ_MM) if side == "left" else render_board(size, r["K2"], r["R"] @ Rb, r["R"] @ tb + r["T"].ravel(), PATTERN, SQ_MM)
            img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, img.shape), 0, 255).astype(np.uint8); vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        vw.release(); paths[side] = p
    return paths


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("f7synth"); return write_videos(d)


def configured(paths=None, baseline=None):
    w = K.CalibrationWizard(); w.set_info(left_camera="Test phone L", right_camera="Test phone R", cols=PATTERN[0], rows=PATTERN[1], square_mm=SQ_MM, baseline_mm=baseline)
    if paths:
        for side in ("left", "right"): w.set_video(side, Path(paths[side]).read_bytes(), Path(paths[side]).name)
    return w


@pytest.fixture(scope="module")
def processed(synth):
    w = configured(synth); w.process_blocking(); yield w; w.close()


# ===================================================================== the calibration pipeline on videos with known truth
class TestPipelineOnSyntheticVideos:
    def test_calibration_succeeds_and_recovers_the_true_rig(self, processed):
        r = processed.result; assert r["ok"], r.get("error")
        m = r["model"]; assert r["pairs_used"] >= K.MIN_PAIRS and m["valid"]
        assert m["baseline_mm"] == pytest.approx(np.linalg.norm(TRUE_T), abs=2.0)                                   # metric scale comes from the board square size
        assert m["K1"][0, 0] == pytest.approx(1100, rel=0.03) and m["K2"][0, 0] == pytest.approx(1050, rel=0.03)
        assert m["stereo_rms"] < 1.0

    def test_time_offset_is_recovered(self, processed):
        s = processed.result["sync"]; assert s["confident"] and s["offset_frames"] == 4, s

    def test_only_board_still_poses_are_paired_and_poses_are_diverse(self, processed):
        p = processed.result["pairing"]; assert p["n_selected"] >= K.MIN_PAIRS and p["n_selected"] <= K.MAX_PAIRS and p["n_still_left"] >= p["n_selected"]
        frames = processed.result["pair_frames"]; assert len({f for f, _ in frames}) == len(frames)                    # no frame used twice

    def test_assessment_is_not_failed_and_has_plain_metrics(self, processed):
        a = processed.assessment(); assert a["level"] in (K.GOOD, K.QUESTIONABLE) and a["metrics"]["rms_stereo_px"] < 1.0 and a["metrics"]["baseline_used_mm"] == pytest.approx(120.1, abs=2.0)
        assert a["metrics"]["image_size_wh"] == list(SIZE)

    def test_calibration_is_never_accepted_automatically(self, processed):
        assert processed.decision is None and processed.contract is None and processed.acknowledged is False
        assert processed.decision is None and processed.contract is None

    def test_contract_is_accepted_by_stage_a_and_marked_video(self, synth):
        w = configured(synth); w.process_blocking(); a = w.assessment()
        if a["level"] == K.QUESTIONABLE: w.acknowledge(True)
        c = w.accept(); assert c["capture_mode"] == dict(left="video", right="video") and c["image_size_wh"] == list(SIZE) and len(c["D1"]) == 5 and len(c["D2"]) == 5
        s = F.F6Session()
        try: r = s.import_calibration(c); assert r["can_proceed"], r
        finally: s.close(); w.close()

    def test_measured_baseline_overrides_the_calibrated_one_and_disagreement_is_flagged(self, synth):
        w = configured(synth, baseline=150.0); w.process_blocking(); a = w.assessment()
        assert a["metrics"]["baseline_used_mm"] == 150.0 and any("differs" in c for c in a["concerns"]) and a["level"] != K.FAILED
        w.acknowledge(True); assert w.accept()["baseline_mm"] == 150.0; w.close()

    def test_wrong_square_size_is_not_hidden(self, synth):
        w = configured(synth); w.set_info(square_mm=45.0); w.process_blocking(); a = w.assessment()
        assert a["metrics"]["calibrated_baseline_mm"] == pytest.approx(120.1 * 1.5, rel=0.05); w.close()

    def test_board_not_in_the_videos_fails_safely(self, tmp_path):
        paths = {}
        for side in ("left", "right"):
            p = tmp_path / f"{side}.mp4"; vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, SIZE); rng = np.random.default_rng(1)
            for _ in range(330): vw.write(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8))
            vw.release(); paths[side] = p
        w = configured(paths); r = w.process_blocking(); a = w.assessment()
        assert not r["ok"] and a["level"] == K.FAILED and a["problems"] and a["metrics"] is None and not w.can_accept()
        with pytest.raises(K.WizardError): w.accept()
        w.close()

    def test_too_few_poses_fails_with_a_clear_reason(self, tmp_path):
        paths = write_videos(tmp_path, n_poses=5, hold=70, name="few"); w = configured(paths); r = w.process_blocking()
        assert not r["ok"] and "at least" in r["error"] and w.assessment()["level"] == K.FAILED; w.close()

    def test_mismatched_resolution_between_videos_is_blocked_before_processing(self, tmp_path, synth):
        small = write_videos(tmp_path, n_poses=3, hold=110, size=(720, 960), name="small"); w = configured()
        w.set_video("left", Path(synth["left"]).read_bytes(), "l.mp4"); w.set_video("right", Path(small["right"]).read_bytes(), "r.mp4")
        assert any("different resolutions" in m for m in w.video_problems()) and any("different resolutions" in m for m in w.blockers(2))
        with pytest.raises(K.WizardError): w.start_processing()
        w.close()


# ===================================================================== wizard rules (state machine)
class TestWizardRules:
    def test_there_are_exactly_six_steps_in_the_required_order(self):
        assert K.STEPS == ("Instructions", "Cameras & Checkerboard", "Upload Calibration Videos", "Process Calibration", "Quality Assessment", "Accept or Discard")

    def test_starts_at_step_one_and_instructions_need_nothing(self):
        w = K.CalibrationWizard()
        try: assert w.step == 0 and w.can_continue() and w.next() and w.step == 1
        finally: w.close()

    def test_cannot_continue_from_info_until_required_fields_are_valid(self):
        w = K.CalibrationWizard(); w.next()
        try:
            assert not w.can_continue() and not w.next() and w.step == 1
            w.set_info(left_camera="A"); assert not w.can_continue()
            w.set_info(right_camera="B", cols=8, rows=6); assert not w.can_continue()
            w.set_info(square_mm=20.0); assert w.can_continue() and w.next() and w.step == 2
        finally: w.close()

    @pytest.mark.parametrize("field,value", [("cols", 2), ("cols", 3.5), ("rows", 0), ("rows", None), ("square_mm", 0), ("square_mm", -3.0), ("square_mm", float("nan")), ("square_mm", None), ("left_camera", "   "), ("right_camera", ""), ("baseline_mm", -5.0), ("baseline_mm", float("inf")), ("baseline_mm", 5.0)])
    def test_invalid_information_blocks_continue(self, field, value):
        w = configured(); w.step = 1; w.set_info(**{field: value})
        try: assert w.info_problems() and not w.can_continue() and not w.next()
        finally: w.close()

    def test_optional_baseline_may_be_empty(self):
        w = configured(); w.set_info(baseline_mm=None); assert not w.info_problems(); w.set_info(baseline_mm=225.0); assert not w.info_problems(); w.close()

    def test_values_are_preserved_when_going_back(self):
        w = configured(baseline=200.0); w.step = 3; before = copy.deepcopy(w.info)
        for _ in range(3): assert w.back()
        assert w.step == 0 and not w.back() and w.info == before
        assert not w.next() and w.step == 0 or w.step == 1                                                  # forward again is allowed only where requirements hold
        assert w.info == before; w.close()

    def test_future_steps_are_not_reachable_by_jumping(self):
        w = K.CalibrationWizard()
        try:
            with pytest.raises(K.WizardError): w.go_to_step(4)
            assert w.step == 0
            with pytest.raises(ValueError): w.go_to_step(9)
        finally: w.close()

    def test_upload_step_requires_both_valid_videos(self, tmp_path):
        w = configured(); w.step = 2
        try:
            assert not w.can_continue() and any("left" in m for m in w.blockers()) and any("right" in m for m in w.blockers())
            w.set_video("left", b"not a video", "a.mp4"); w.set_video("right", b"nope", "b.mp4")
            assert not w.can_continue() and not w.next() and w.step == 2 and w.video_problems()
        finally: w.close()

    def test_processing_step_blocks_until_a_result_exists(self):
        w = configured(); w.step = 3
        try: assert not w.can_continue() and "Start the calibration first." in w.blockers()
        finally: w.close()

    def test_starting_processing_before_earlier_steps_are_complete_is_refused(self):
        w = K.CalibrationWizard()
        try:
            with pytest.raises(K.WizardError): w.start_processing()
        finally: w.close()

    def test_changing_checkerboard_settings_invalidates_a_computed_result(self, processed, synth):
        w = configured(synth); w.result = copy.deepcopy(processed.result); w._used_settings = w._settings_tuple(); w._assessment = None; assert w.assessment()
        w.set_info(square_mm=SQ_MM + 5); assert w.result is None and w.assessment() is None and w.decision is None; w.close()

    def test_replacing_a_video_invalidates_the_result_and_decision(self, processed, synth):
        w = configured(synth); w.result = copy.deepcopy(processed.result); w._used_settings = w._settings_tuple(); w.acknowledge(True)
        w.set_video("left", Path(synth["left"]).read_bytes(), "again.mp4"); assert w.result is None and w.acknowledged is False and w.contract is None; w.close()


# ===================================================================== assessment gating: GOOD / QUESTIONABLE / FAILED
def _wizard_with(result_mutator, synth_result, level_ok=True):
    w = configured(); w.result = copy.deepcopy(synth_result); w.result["sync"]["confident"] = True; w._used_settings = w._settings_tuple()
    result_mutator(w.result); w._assessment = None; w.step = 4; return w


class TestAssessmentGating:
    def test_good_calibration_can_be_accepted_only_by_an_explicit_call(self, processed):
        w = _wizard_with(lambda r: (r["model"].__setitem__("warnings", []), r.__setitem__("pairs_discarded", 0)), processed.result)
        try:
            a = w.assessment()
            if a["level"] != K.GOOD: pytest.skip(f"synthetic result carries stage-A concerns: {a['concerns']}")
            assert w.decision is None and w.can_accept() and w.can_continue(); c = w.accept(); assert w.decision == "accepted" and c["baseline_mm"] > 0 and not w.can_accept()
        finally: w.close()

    def test_questionable_needs_acknowledgement(self, processed):
        w = _wizard_with(lambda r: r["model"].__setitem__("warnings", [dict(code="x", message="the coverage is poor", heuristic=True)]), processed.result)
        try:
            a = w.assessment(); assert a["level"] == K.QUESTIONABLE and "the coverage is poor" in a["concerns"] and w.can_continue()
            assert not w.can_accept()
            with pytest.raises(K.WizardError): w.accept()
            w.acknowledge(True); assert w.can_accept(); assert w.accept() and w.decision == "accepted"
            assert w.provenance()["level"] == K.QUESTIONABLE and "the coverage is poor" in w.provenance()["acknowledged_concerns"]
        finally: w.close()

    def test_acknowledgement_does_not_carry_to_a_new_result(self, processed):
        w = _wizard_with(lambda r: r["model"].__setitem__("warnings", [dict(code="x", message="m", heuristic=True)]), processed.result)
        try: w.acknowledge(True); w._invalidate_result(); assert w.acknowledged is False
        finally: w.close()

    def test_failed_validity_checks_block_everything(self, processed):
        w = _wizard_with(lambda r: r["model"].__setitem__("errors", ["R is not a proper rotation"]), processed.result)
        try:
            a = w.assessment(); assert a["level"] == K.FAILED and not w.can_continue() and "failed calibration" in " ".join(w.blockers()).lower()
            w.acknowledge(True); assert not w.can_accept()
            with pytest.raises(K.WizardError): w.accept()
            assert w.decision is None and w.contract is None and not w.next() and w.step == 4
        finally: w.close()

    def test_failed_result_without_a_model_is_failed(self):
        w = configured(); w.result = dict(ok=False, error="no board found", model=None); w.step = 4
        try:
            a = w.assessment(); assert a["level"] == K.FAILED and a["problems"] == ["no board found"] and not w.can_continue()
            with pytest.raises(K.WizardError): w.accept()
        finally: w.close()

    def test_accept_requires_a_result(self):
        w = configured()
        try:
            with pytest.raises(K.WizardError): w.accept()
        finally: w.close()

    def test_settings_changed_after_the_result_blocks_acceptance(self, processed):
        w = _wizard_with(lambda r: r["model"].__setitem__("warnings", []), processed.result)
        try:
            w.info["square_mm"] = 99.0                                                                         # bypass set_info on purpose: accept() must still notice
            w.acknowledge(True)
            with pytest.raises(K.WizardError): w.accept()
        finally: w.close()

    def test_discard_and_recalibrate_keeps_inputs_and_returns_to_upload(self, processed, synth):
        w = configured(synth); w.result = copy.deepcopy(processed.result); w._used_settings = w._settings_tuple(); w.acknowledge(True); w.step = 5; info = copy.deepcopy(w.info)
        w.discard(); assert w.result is None and w.assessment() is None and w.decision is None and w.acknowledged is False and w.step == 2 and w.info == info and all(w.videos.values())
        assert w.blockers(3) == ["Start the calibration first."]; w.close()

    def test_a_failed_attempt_can_be_discarded_and_retried(self):
        w = configured(); w.result = dict(ok=False, error="x", model=None); w.step = 4; w.discard(); assert w.step == 2 and w.result is None; w.close()


# ===================================================================== page: one step at a time (AppTest) + transition to the upload workflow
def _app(wiz=None, step=0):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(WF / "app_f6.py"), default_timeout=90)
    if wiz is not None: at.session_state["f7"] = wiz
    at.session_state["f6_step"] = 0
    at.run(); return at


def _kinds(at):
    return dict(text=len(at.text_input), number=len(at.number_input), uploader=len(at.get("file_uploader")), checkbox=len(at.checkbox))


def _labels(at): return [b.label for b in at.button]


class TestPageShowsOneStepAtATime:
    def test_step1_shows_only_instructions(self):
        at = _app()
        try:
            assert not at.exception and _kinds(at) == dict(text=0, number=0, uploader=0, checkbox=0)
            text = " ".join(m.value for m in at.markdown); assert "Calibrate Your Cameras" in text and "Hold the board still" in text
            assert "Continue" in _labels(at) and "Back" in _labels(at) and [b for b in at.button if b.label == "Back"][0].disabled
            assert "Calibration part 1 of 6" in text and "Step 1 of 4 — Camera Calibration" in text
            assert "f7-pill" not in text and "f6-step" not in text and "Stereo videos in" not in text                       # no pill rows, no subtitle
            assert all(t not in text for t in ("Upload Calibration Videos", "Accept Calibration", "Review Calibration Results"))   # only the current step is named
            assert [e.label for e in at.expander] == ["Detailed Recording Instructions", "Already Have a Calibration File?"]
            assert "Set Up Your Cameras" in text and "Prepare a Checkerboard" in text and "Record Calibration Videos" in text
            for need in ("inner corners", "square", "baseline", "simultaneously", "Same setup", "lens, zoom and recording mode", "200 MB", "Visible in both cameras"): assert need in text, need
        finally: at.session_state["f7"].close() if "f7" in at.session_state else None

    def test_step2_shows_only_camera_and_checkerboard_fields_and_continue_is_disabled_until_valid(self):
        w = K.CalibrationWizard(); w.step = 1; at = _app(w)
        try:
            assert not at.exception and _kinds(at)["text"] == 2 and _kinds(at)["number"] == 4 and _kinds(at)["uploader"] == 0
            assert [b for b in at.button if b.label == "Continue"][0].disabled
            assert "Start calibration" not in _labels(at) and "Accept & Continue" not in _labels(at)
        finally: w.close()

    def test_step2_values_entered_before_are_shown_again(self):
        w = configured(baseline=225.0); w.step = 1; at = _app(w)
        try:
            vals = {t.label: t.value for t in at.text_input}; assert vals["Left camera"] == "Test phone L" and vals["Right camera"] == "Test phone R"
            nums = {n.label: n.value for n in at.number_input}; assert nums["Inner corners — columns"] == 8 and nums["Inner corners — rows"] == 6 and nums["Square size (mm)"] == SQ_MM
            assert not [b for b in at.button if b.label == "Continue"][0].disabled
        finally: w.close()

    def test_step3_shows_only_the_two_uploaders(self):
        w = configured(); w.step = 2; at = _app(w)
        try:
            assert not at.exception and _kinds(at) == dict(text=0, number=0, uploader=2, checkbox=0) and [b for b in at.button if b.label == "Continue"][0].disabled
        finally: w.close()

    def test_step4_shows_only_the_processing_controls(self):
        w = configured(); w.step = 3; at = _app(w)
        try:
            assert not at.exception and _kinds(at) == dict(text=0, number=0, uploader=0, checkbox=0) and "Start calibration" in _labels(at) and [b for b in at.button if b.label == "Continue"][0].disabled
        finally: w.close()

    def test_step5_failed_blocks_continue_and_offers_discard(self):
        w = configured(); w.result = dict(ok=False, error="no board found", model=None); w.step = 4; at = _app(w)
        try:
            assert not at.exception and [b for b in at.button if b.label == "Continue"][0].disabled and "Discard & Recalibrate" in _labels(at) and "no board found" in " ".join(m.value for m in at.markdown)
        finally: w.close()

    def test_step5_shows_the_assessment_before_any_decision_is_possible(self, processed):
        w = _wizard_with(lambda r: None, processed.result); at = _app(w)
        try:
            assert not at.exception and "Accept & Continue" not in _labels(at) and "Review Calibration Results" in " ".join(m.value for m in at.markdown) and len(at.table) >= 1
            assert w.decision is None
        finally: w.close()

    def test_step6_failed_cannot_be_accepted(self):
        w = configured(); w.result = dict(ok=False, error="x", model=None); w.step = 5; at = _app(w)
        try:
            acc = [b for b in at.button if b.label == "Accept & Continue"]; assert acc and acc[0].disabled and "Discard & Recalibrate" in _labels(at)
        finally: w.close()

    def test_step6_questionable_needs_the_checkbox_before_accept_is_enabled(self, processed):
        w = _wizard_with(lambda r: r["model"].__setitem__("warnings", [dict(code="x", message="coverage is poor", heuristic=True)]), processed.result); w.step = 5; at = _app(w)
        try:
            assert len(at.checkbox) == 1 and [b for b in at.button if b.label == "Accept & Continue"][0].disabled
            at.checkbox[0].check().run(); assert w.acknowledged is True and not [b for b in at.button if b.label == "Accept & Continue"][0].disabled
            assert w.decision is None                                                                           # still not accepted: only the explicit button accepts
        finally: w.close()

    def test_accepting_transitions_to_the_room_video_upload(self, processed):
        w = _wizard_with(lambda r: r["model"].__setitem__("warnings", []), processed.result); w.step = 5; w.acknowledge(True); at = _app(w)
        try:
            [b for b in at.button if b.label == "Accept & Continue"][0].click().run()
            assert at.session_state["f6_step"] == 1 and w.decision == "accepted" and not at.exception
            assert at.session_state["f6"].job.stage == "calibration_validated" and at.session_state["f6"].calibration_provenance["source"] == "visitor_video_calibration"
            assert "Upload your four videos" in " ".join(m.value for m in at.markdown)
        finally: at.session_state["f6"].close(); w.close()

    def test_discard_button_returns_to_the_upload_step(self, processed):
        w = _wizard_with(lambda r: None, processed.result); w.step = 5; at = _app(w)
        try:
            [b for b in at.button if b.label == "Discard & Recalibrate"][0].click().run(); assert w.step == 2 and w.result is None and at.session_state["f6_step"] == 0
        finally: w.close()

    def test_import_alternative_is_not_shown_alongside_the_wizard_steps(self):
        w = K.CalibrationWizard(); w.step = 1; at = _app(w)
        try: assert "Import a calibration file" not in _labels(at) and "Calibration file (JSON)" not in " ".join(str(x.label) for x in at.get("file_uploader"))
        finally: w.close()

    def test_import_lives_in_a_collapsed_expander_on_step_one_and_fixture_is_hidden(self):
        at = _app()
        try:
            assert "Import a calibration file" in _labels(at) and [e for e in at.expander if e.label == "Already Have a Calibration File?"]
            [b for b in at.button if b.label == "Import a calibration file"][0].click().run()
            assert at.session_state["f7_mode"] == "import" and not at.exception and "Use original-room calibration" not in _labels(at)       # fixture hidden from normal visitors
            assert "Continue" in _labels(at) and [b for b in at.button if b.label == "Continue"][0].disabled
        finally: at.session_state["f7"].close() if "f7" in at.session_state else None

    def test_every_step_has_only_back_left_and_continue_right_and_a_single_heading(self):
        for st_i in range(5):
            w = configured(); w.result = dict(ok=False, error="x", model=None) if st_i >= 4 else None; w.step = st_i; at = _app(w)
            try:
                labs = _labels(at); assert labs.index("Back") < labs.index("Continue") and labs.count("Back") == 1 and labs.count("Continue") == 1
                text = " ".join(m.value for m in at.markdown); assert text.count(f"Calibration part {st_i + 1} of 6") == 1
            finally: w.close()

    def test_progress_text_and_headings_follow_the_current_step(self):
        for st_i, head in enumerate(("Calibrate Your Cameras", "Enter Camera Details", "Upload Calibration Videos", "Calibrating Your Cameras", "Review Calibration Results", "Accept Calibration")):
            w = configured(); w.result = dict(ok=False, error="x", model=None) if st_i >= 4 else None; w.step = st_i; at = _app(w)
            try: assert f"## {head}" in [m.value for m in at.markdown]
            finally: w.close()

    def test_failed_calibration_keeps_the_error_and_stays_blocked(self):
        w = configured(); w.result = dict(ok=False, error="no board found", model=None); w.step = 4; at = _app(w)
        try: assert "no board found" in " ".join(m.value for m in at.markdown) and [b for b in at.button if b.label == "Continue"][0].disabled
        finally: w.close()


# ===================================================================== safety / classical-only / isolation
class TestSafety:
    FORBIDDEN = {"torch", "torchvision", "tensorflow", "keras", "transformers", "onnx", "onnxruntime", "sklearn", "timm", "sam2", "mediapipe", "huggingface_hub"}

    def test_no_ml_imports_in_f7_files(self):
        for f in ("f7_calibration.py", "f7_pages.py"):
            names = set()
            for n in ast.walk(ast.parse((WF / f).read_text())):
                if isinstance(n, ast.Import): names |= {a.name.split(".")[0].lower() for a in n.names}
                if isinstance(n, ast.ImportFrom) and n.module: names.add(n.module.split(".")[0].lower())
            assert not names & self.FORBIDDEN, (f, names & self.FORBIDDEN)

    def test_core_is_streamlit_free(self):
        names = {a.name.split(".")[0] for n in ast.walk(ast.parse((WF / "f7_calibration.py").read_text())) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(ast.parse((WF / "f7_calibration.py").read_text())) if isinstance(n, ast.ImportFrom) and n.module}
        assert "streamlit" not in names

    def test_wizards_have_separate_workspaces_and_close_removes_only_their_own(self, tmp_path):
        keep = tmp_path / "precious.txt"; keep.write_text("x"); a, b = K.CalibrationWizard(), K.CalibrationWizard()
        try:
            assert a.ws.path != b.ws.path and a.ws.path.exists() and b.ws.path.exists()
            a.set_info(left_camera="only in A"); assert b.info["left_camera"] == ""
            wa = a.ws.path; a.close(); a.close(); assert not wa.exists() and b.ws.path.exists() and keep.exists()
        finally: a.close(); b.close()

    def test_uploaded_videos_are_staged_with_generated_names(self, tmp_path):
        w = K.CalibrationWizard()
        try:
            v = w.set_video("left", b"abc", "../../evil name.mp4"); assert Path(v["path"]).parent == w.ws.path and "evil" not in Path(v["path"]).name and v["name"] == "evil name.mp4"
        finally: w.close()

    def test_original_phase1_files_untouched_by_name(self):
        for f in ("f7_calibration.py", "f7_pages.py"):
            t = (WF / f).read_text(); assert "estimation_result" not in t and "wall_boundary_annotations" not in t and "frame_manifest" not in t
        assert "visitor_workflow" not in (ROOT / "Pages" / "app_module7.py").read_text()

    def test_stage_limits_are_not_loosened(self):
        assert K.LIMITS.max_file_bytes == 200 * 1024 * 1024 and K.LIMITS.max_duration_s == 120.0 and K.LIMITS.min_duration_s == 10.0


# ---- F7 page-level Home / Back navigation (standalone prototype) ----
def test_f7_page_nav_urls_target_real_main_app_routes(monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modules" / "Module_7" / "visitor_workflow"))
    import page_nav
    monkeypatch.setenv(page_nav.MAIN_APP_URL_ENV, "http://localhost:8517/")
    assert page_nav.home_url() == "http://localhost:8517/" and page_nav.back_url() == "http://localhost:8517/module-7"
    monkeypatch.delenv(page_nav.MAIN_APP_URL_ENV)
    assert page_nav.home_url() == "http://localhost:8501/"
    # the destination must be Module 7's real url_path in home.py
    assert 'url_path' in (Path(__file__).resolve().parents[1] / "home.py").read_text() and '"module-7"' in (Path(__file__).resolve().parents[1] / "home.py").read_text()
