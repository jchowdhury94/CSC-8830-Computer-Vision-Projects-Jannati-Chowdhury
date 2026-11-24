"""Stage F8.15: F8 calibration result -> F6 hand-off (visitor_workflow/f8_calibration_bridge.py).

Everything here is SYNTHETIC: the F8 document is produced by the real ``result_store.build_result_document`` from a made-up supervisor result, and the contract by the
unchanged F7 ``contract_from_model``.  No video is opened, no calibration or reconstruction is run; ``F6Session.import_calibration`` only validates numbers."""
import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7" / "room_measurement_question_3"      # Question 3 packages, data and outputs
WF = MODULE / "visitor_workflow"
F8 = MODULE / "f8_chunk_poc"
for p in (str(MODULE), str(MODULE.parent), str(WF), str(F8), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)

import f6_core as F  # noqa: E402
import f7_calibration as K7  # noqa: E402
import f8_calibration_bridge as B  # noqa: E402
from result_store import build_result_document  # noqa: E402

SIZE = [2160, 3840]
K1 = [[3050.0, 0.0, 1075.0], [0.0, 3052.0, 1925.0], [0.0, 0.0, 1.0]]
K2 = [[3061.0, 0.0, 1081.0], [0.0, 3060.0, 1918.0], [0.0, 0.0, 1.0]]
D1 = [0.012, -0.045, 0.0004, -0.0002, 0.03]
D2 = [0.010, -0.040, -0.0003, 0.0001, 0.02]
CAL_BASELINE = 226.826


def synthetic_doc(level="GOOD", measured=226.0, concerns=(), outcome="completed", mode="calibrate", ok=True):
    """An F8 result document built exactly the way the app builds it, from a made-up worker/supervisor result."""
    model = dict(K1=K1, D1=D1, K2=K2, D2=D2, image_size_left=SIZE, image_size_right=SIZE, baseline_mm=CAL_BASELINE, stereo_rms=0.7357, rms_left=0.6, rms_right=0.65,
                 R=np.eye(3).tolist(), T=[-CAL_BASELINE, 0.0, 0.0], distortion_left=None, distortion_right=None)
    contract = K7.contract_from_model(model, measured) if level != "FAILED" else None
    metrics = dict(pairs_used=15, pairs_discarded=1, rms_stereo_px=0.7357, calibrated_baseline_mm=CAL_BASELINE, measured_baseline_mm=measured, baseline_used_mm=measured or CAL_BASELINE)
    res = dict(mode=mode, outcome=outcome, ok=ok, model=model, contract=contract, calibration=dict(pairs_used=15, pairs_discarded=1),
               assessment=dict(level=level, headline="synthetic", problems=[], concerns=list(concerns), advice="", metrics=metrics),
               validation=dict(videos={s: dict(width=2160, height=3840, fps=30.0) for s in ("left", "right")}))
    return build_result_document(res, code="synthetic-recovery-code-0123456789", settings=dict(cols=8, rows=6, square_mm=20.0, baseline_mm=measured))


# ===================================================================== the adapter
class TestAdapter:
    def test_good_result_is_extracted_exactly(self):
        h = B.parse_f8_result(synthetic_doc())
        c = h.accept()
        assert (h.level, h.baseline_source, h.baseline_used_mm) == ("GOOD", "measured", 226.0)
        assert c["K1"] == K1 and c["K2"] == K2 and c["D1"] == D1 and c["D2"] == D2
        assert c["image_size_wh"] == SIZE and c["capture_mode"] == dict(left="video", right="video") and c["baseline_mm"] == 226.0
        assert set(c) == set(B.CONTRACT_FIELDS)

    def test_accepts_json_text_and_bytes(self):
        d = synthetic_doc()
        assert B.parse_f8_result(json.dumps(d)).accept() == B.parse_f8_result(json.dumps(d).encode()).accept() == B.parse_f8_result(d).accept()

    def test_calibrated_baseline_is_used_when_nothing_was_measured(self):
        h = B.parse_f8_result(synthetic_doc(measured=None))
        assert (h.baseline_source, h.baseline_used_mm) == ("calibrated", pytest.approx(CAL_BASELINE))
        assert h.accept()["baseline_mm"] == pytest.approx(CAL_BASELINE)

    def test_supplied_measured_baseline_overrides_the_stored_one(self):
        h = B.parse_f8_result(synthetic_doc(measured=None), measured_baseline_mm=226)
        assert (h.baseline_source, h.baseline_used_mm, h.calibrated_baseline_mm) == ("measured", 226.0, pytest.approx(CAL_BASELINE))
        assert h.accept()["baseline_mm"] == 226.0 and h.level == "GOOD"            # 0.37 % difference: no concern

    def test_large_baseline_disagreement_becomes_a_concern_that_needs_acknowledgement(self):
        h = B.parse_f8_result(synthetic_doc(), measured_baseline_mm=300)
        assert h.level == "QUESTIONABLE" and any("differs from the calibrated" in x for x in h.concerns)
        with pytest.raises(B.HandoffError) as e: h.accept()
        assert e.value.codes == ["ACKNOWLEDGEMENT_REQUIRED"]
        assert h.accept(acknowledged=True)["baseline_mm"] == 300.0

    @pytest.mark.parametrize("bad", [0, -5, 9.9, 2000.1, float("nan"), float("inf"), True, "226", [226]])
    def test_invalid_measured_baseline_rejected(self, bad):
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(synthetic_doc(), measured_baseline_mm=bad)
        assert "INVALID_MEASURED_BASELINE" in e.value.codes

    def test_questionable_needs_acknowledgement_and_good_does_not(self):
        q = B.parse_f8_result(synthetic_doc(level="QUESTIONABLE", concerns=["Left/right timing could not be verified."]))
        assert q.requires_acknowledgement and not q.can_accept() and q.can_accept(True)
        with pytest.raises(B.HandoffError) as e: q.accept()
        assert e.value.codes == ["ACKNOWLEDGEMENT_REQUIRED"]
        assert q.accept(acknowledged=True)["baseline_mm"] == 226.0
        assert B.parse_f8_result(synthetic_doc()).can_accept()

    def test_a_good_result_with_concerns_is_treated_as_questionable(self):
        h = B.parse_f8_result(synthetic_doc(level="GOOD", concerns=["something"]))
        assert h.level == "QUESTIONABLE" and not h.can_accept()

    def test_failed_calibration_is_never_accepted(self):
        d = synthetic_doc(level="FAILED", outcome="failed", ok=False)
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert {"FAILED_CALIBRATION", "JOB_NOT_COMPLETED", "CONTRACT_MISSING"} <= set(e.value.codes)

    def test_failed_level_with_an_otherwise_complete_document_is_still_refused(self):
        d = synthetic_doc(); d["quality"]["level"] = "FAILED"
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert "FAILED_CALIBRATION" in e.value.codes

    @pytest.mark.parametrize("level", [None, "", "BEST", "good", 3])
    def test_unknown_quality_is_refused(self, level):
        d = synthetic_doc(); d["quality"]["level"] = level
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert "UNKNOWN_QUALITY" in e.value.codes

    def test_benchmark_and_unfinished_jobs_are_refused(self):
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(synthetic_doc(mode="benchmark"))
        assert "NOT_A_CALIBRATION" in e.value.codes
        for outcome in ("timeout", "memory_limit", "cancelled", "worker_crashed"):
            with pytest.raises(B.HandoffError) as e: B.parse_f8_result(synthetic_doc(outcome=outcome, ok=False))
            assert "JOB_NOT_COMPLETED" in e.value.codes

    def test_assessment_problems_block_even_a_good_label(self):
        d = synthetic_doc(); d["quality"]["problems"] = ["bad reprojection"]
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert "ASSESSMENT_PROBLEMS" in e.value.codes

    @pytest.mark.parametrize("payload,code", [("not json {", "NOT_JSON"), ("[1,2]", "NOT_AN_OBJECT"), (b"\xff\xfe", "NOT_JSON"), (None, "NOT_AN_OBJECT"), (42, "NOT_AN_OBJECT")])
    def test_garbage_input(self, payload, code):
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(payload)
        assert e.value.codes == [code]

    def test_oversized_text_is_refused_before_parsing(self):
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result("x" * (B.MAX_DOCUMENT_BYTES + 1))
        assert e.value.codes == ["DOCUMENT_TOO_LARGE"]

    def test_wrong_schema_points_to_the_existing_importer(self):
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(dict(K1=K1, D1=D1, K2=K2, D2=D2))
        assert e.value.codes == ["UNRECOGNISED_SCHEMA"] and "import" in str(e.value)

    @pytest.mark.parametrize("field", list(B.CONTRACT_FIELDS))
    def test_every_missing_contract_field_is_reported(self, field):
        d = synthetic_doc(); del d["contract"][field]
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert e.value.codes == ["CONTRACT_MISSING_FIELD"] and field in str(e.value)

    def test_malformed_contract_values(self):
        for edit, code in ((lambda c: c.update(K1=[[1, 2], [3, 4]]), "INVALID_MATRIX"), (lambda c: c["K2"][0].__setitem__(0, None), "CONTRACT_MISSING_FIELD" if False else "INVALID_MATRIX"),
                           (lambda c: c.update(D1=[0.1] * 4), "INVALID_DISTORTION"), (lambda c: c["D2"].__setitem__(1, float("nan")), "INVALID_DISTORTION"),
                           (lambda c: c.update(image_size_wh=[2160]), "INVALID_IMAGE_SIZE"), (lambda c: c.update(image_size_wh=[2160.5, 3840]), "INVALID_IMAGE_SIZE"),
                           (lambda c: c.update(image_size_wh=[0, 3840]), "INVALID_IMAGE_SIZE"), (lambda c: c.update(capture_mode=None), "CONTRACT_MISSING_FIELD"),
                           (lambda c: c.update(capture_mode=dict(left="video")), "CAPTURE_MODE_PROVENANCE_REQUIRED"), (lambda c: c.update(capture_mode=dict(left="video", right=" ")), "CAPTURE_MODE_PROVENANCE_REQUIRED"),
                           (lambda c: c.update(baseline_mm=-1), "INVALID_BASELINE"), (lambda c: c.update(baseline_mm=True), "INVALID_BASELINE"), (lambda c: c.update(baseline_mm="226"), "INVALID_BASELINE")):
            d = synthetic_doc(); edit(d["contract"])
            with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
            assert code in e.value.codes, (code, e.value.codes)

    def test_contract_that_disagrees_with_the_results_own_cameras_is_refused(self):
        d = synthetic_doc(); d["contract"]["K1"][0][0] += 5.0
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert "INCONSISTENT_RESULT" in e.value.codes
        d = synthetic_doc(); d["cameras"]["right"]["image_size_wh"] = [1080, 1920]
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert "IMAGE_SIZE_MISMATCH" in e.value.codes

    def test_incompatible_room_video_size_gives_a_clear_error(self):
        assert B.parse_f8_result(synthetic_doc(), expected_image_size_wh=[2160, 3840]).accept()
        for want in ([3840, 2160], [1080, 1920]):                                 # landscape / other resolution
            with pytest.raises(B.HandoffError) as e: B.parse_f8_result(synthetic_doc(), expected_image_size_wh=want)
            assert e.value.codes == ["CALIBRATION_VIDEO_INCOMPATIBLE"] and "2160x3840" in str(e.value)
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(synthetic_doc(), expected_image_size_wh="big")
        assert e.value.codes == ["INVALID_EXPECTED_SIZE"]

    def test_all_problems_are_listed_together(self):
        d = synthetic_doc(); d["contract"]["K1"] = "x"; d["contract"]["baseline_mm"] = 0
        with pytest.raises(B.HandoffError) as e: B.parse_f8_result(d)
        assert {"INVALID_MATRIX", "INVALID_BASELINE"} <= set(e.value.codes)

    def test_accept_returns_an_independent_copy_and_the_input_is_not_mutated(self):
        d = synthetic_doc(); before = copy.deepcopy(d)
        h = B.parse_f8_result(d); c = h.accept(); c["K1"][0][0] = -1; c["capture_mode"]["left"] = "photo"
        assert d == before and h.accept()["K1"][0][0] == K1[0][0] and h.accept()["capture_mode"]["left"] == "video"

    def test_provenance_carries_numbers_only(self):
        p = B.parse_f8_result(synthetic_doc()).provenance()
        assert p["source"] == "f8_cloud_calibration" and p["baseline_used_mm"] == 226.0 and p["baseline_source"] == "measured" and p["level"] == "GOOD"
        json.dumps(p)

    def test_module_is_classical_and_streamlit_free(self):
        tree = ast.parse((WF / "f8_calibration_bridge.py").read_text())
        mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not mods & {"streamlit", "torch", "tensorflow", "sklearn", "onnx", "onnxruntime", "cv2", "numpy"}, mods


# ===================================================================== F6 compatibility (validation only; no video, no reconstruction)
class TestF6Compatibility:
    @pytest.fixture
    def sess(self):
        s = F.F6Session(); yield s; s.close()

    def test_the_extracted_contract_is_accepted_by_f6_import_calibration(self, sess):
        c = B.parse_f8_result(synthetic_doc()).accept()
        r = sess.import_calibration(c)
        assert r["can_proceed"] is True and r["step"] == "validate_calibration", r
        cal = r["calibration"]
        assert tuple(sess.job.cal.image_size) == (2160, 3840) and sess.job.baseline_mm == 226.0
        assert np.allclose(sess.job.cal.K_left, K1) and np.allclose(sess.job.cal.K_right, K2)
        assert sess.job.stage == "calibration_validated" and cal is not None

    def test_the_calibrated_baseline_is_carried_when_none_was_measured(self, sess):
        r = sess.import_calibration(B.parse_f8_result(synthetic_doc(measured=None)).accept())
        assert r["can_proceed"] and sess.job.baseline_mm == pytest.approx(CAL_BASELINE)

    def test_questionable_contract_after_acknowledgement_is_accepted_by_f6(self, sess):
        h = B.parse_f8_result(synthetic_doc(level="QUESTIONABLE", concerns=["timing"]))
        r = sess.import_calibration(h.accept(acknowledged=True)); assert r["can_proceed"]

    def test_provenance_can_be_stored_on_the_session(self, sess):
        h = B.parse_f8_result(synthetic_doc()); sess.import_calibration(h.accept()); sess.calibration_provenance = h.provenance()
        assert json.dumps(sess.summary_for_export())

    def test_f6_still_rejects_what_the_adapter_would_not_catch(self, sess):
        c = B.parse_f8_result(synthetic_doc()).accept(); c["K1"][0][0] = -10.0               # structurally fine, physically impossible: F6's own checks must refuse it
        r = sess.import_calibration(c); assert r["can_proceed"] is False

    def test_the_f6_file_importer_does_not_understand_the_raw_f8_document_but_does_understand_the_contract(self):
        d = synthetic_doc()
        cal, probs = F.parse_calibration_json(json.dumps(d)); assert probs                  # documented incompatibility: the contract is nested
        cal, probs = F.parse_calibration_json(json.dumps(B.parse_f8_result(d).accept())); assert not probs and cal["baseline_mm"] == 226.0

    def test_no_f6_f7_or_phase1_source_was_changed_by_the_adapter(self):
        import subprocess
        out = subprocess.run(["git", "status", "--porcelain", "--", "Modules/Module_7/room_measurement_question_3/room_measurement", "Modules/Module_7/stereo_measure", "Modules/Module_7/room_measurement_question_3/f8_chunk_poc/calibration_worker.py",
                              "Modules/Module_7/room_measurement_question_3/visitor_workflow/f7_calibration.py", "Pages", "home.py"], cwd=ROOT, capture_output=True, text=True).stdout
        changed = [l for l in out.splitlines() if not l.startswith("??")]
        assert changed == [], changed
