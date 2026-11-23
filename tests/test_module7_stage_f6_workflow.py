"""Phase 2B Stage F6: simplified classical visitor workflow (Modules/Module_7/visitor_workflow).

Fast tests exercise the pure core (units, optional ground truth, percentage error, calibration import, upload contract, ML-free imports, page smoke test).
The integration class runs the REAL pipeline (Stage A-D, F2, E2) on the original-room recordings, which are only READ; it is skipped when they are absent.
The scripted 'visitor decisions' in the integration class are fixed choices (made by looking at the candidate overlays); they test mechanics, not accuracy."""
import ast
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
WF = MODULE / "visitor_workflow"
for p in (str(MODULE), str(WF), str(ROOT)):
    if p not in sys.path: sys.path.insert(0, p)

import f6_core as F  # noqa: E402
from room_measurement import calibration as C, visitor_annotations as V, visitor_measurement as VM  # noqa: E402

VIDEO_DIR = MODULE / "data" / "room_measurement" / "calibration_videos"
FIXTURE_VIDEOS = dict(placement_1_left="left_air_wall_left.MOV", placement_1_right="right_pro_wall_left.MOV", placement_2_left="left_air_wall_right.MOV", placement_2_right="right_pro_wall_right.MOV")
HAVE_VIDEOS = all((VIDEO_DIR / f).exists() for f in FIXTURE_VIDEOS.values()) and Path(C.DEFAULT_CALIBRATION_PATH).exists()
PHASE1_CAL = Path(C.DEFAULT_CALIBRATION_PATH)


def fixture_calibration(capture_mode="video", baseline=225.0):
    cal, probs = F.parse_calibration_json(PHASE1_CAL.read_bytes(), capture_mode=capture_mode, baseline_mm=baseline); assert not probs; return cal


# ===================================================================== units, optional actual dimensions, percentage error
class TestUnitsAndError:
    @pytest.mark.parametrize("value,unit,mm", [(2900, "mm", 2900.0), (290, "cm", 2900.0), (2.9, "m", 2900.0), (100, "in", 2540.0), (10, "ft", 3048.0), ("2,9", "m", 2900.0), (" 2900 ", "mm", 2900.0)])
    def test_conversion_to_mm(self, value, unit, mm): assert F.to_mm(value, unit) == pytest.approx(mm)

    @pytest.mark.parametrize("bad", [0, -1, -0.001, float("nan"), float("inf"), float("-inf"), True, False, "abc", "", None, [], "1e999"])
    def test_invalid_lengths_rejected(self, bad):
        with pytest.raises(ValueError): F.to_mm(bad, "mm")

    def test_unknown_unit_rejected(self):
        with pytest.raises(ValueError): F.to_mm(1, "yard")

    def test_round_trip_display_units(self):
        for u in F.UNIT_TO_MM: assert F.from_mm(F.to_mm(3.5, u), u) == pytest.approx(3.5)

    def test_blank_means_not_provided(self):
        assert F.parse_actual_dimensions("", "  ", "mm") == F.ActualDimensions(None, None, None) and not F.parse_actual_dimensions(None, None).any

    def test_each_dimension_is_optional(self):
        a = F.parse_actual_dimensions("2.9", "", "m"); assert a.width_mm == pytest.approx(2900) and a.height_mm is None and a.any
        b = F.parse_actual_dimensions(None, "8", "ft"); assert b.width_mm is None and b.height_mm == pytest.approx(2438.4)

    def test_invalid_actual_dimension_raises_even_if_other_is_valid(self):
        with pytest.raises(ValueError): F.parse_actual_dimensions("2900", "-5", "mm")
        with pytest.raises(ValueError): F.parse_actual_dimensions("nan", "", "mm")

    def test_absolute_percentage_error_formula(self):
        r = F.compare_dimension(2879.5660614084068, 2900.0)
        assert r["absolute_percent_error"] == pytest.approx(abs(2879.5660614084068 - 2900) / 2900 * 100) and r["absolute_percent_error"] == pytest.approx(0.7046, abs=1e-3)
        assert r["absolute_error_mm"] == pytest.approx(20.4339, abs=1e-3)

    def test_signed_and_absolute_are_distinct(self):
        under, over = F.compare_dimension(2700, 3000), F.compare_dimension(3300, 3000)
        assert under["signed_percent_error"] == pytest.approx(-10) and under["absolute_percent_error"] == pytest.approx(10)
        assert over["signed_percent_error"] == pytest.approx(10) and over["absolute_percent_error"] == pytest.approx(10)
        assert under["signed_error_mm"] < 0 < over["signed_error_mm"]

    def test_zero_error(self): assert F.compare_dimension(2900, 2900)["absolute_percent_error"] == 0

    @pytest.mark.parametrize("est,act", [(0, 1), (1, 0), (-1, 1), (1, -1), (float("nan"), 1), (1, float("inf")), (True, 1), ("1", 1), (None, 1)])
    def test_comparison_rejects_invalid_values(self, est, act):
        with pytest.raises(ValueError): F.compare_dimension(est, act)

    DIMS = dict(width_mm=2879.566, height_mm=2380.209)

    def test_missing_dimensions_show_estimate_only(self):
        r = F.validation_report(self.DIMS, F.ActualDimensions())
        assert r["available"] and r["width"]["comparison"] is None and r["height"]["comparison"] is None and r["width"]["estimated_mm"] == 2879.566
        assert "percent" not in json.dumps(r["width"]).lower()          # no percentage may appear without ground truth

    def test_partial_actual_gives_only_that_comparison(self):
        r = F.validation_report(self.DIMS, F.parse_actual_dimensions("2900", "", "mm")); assert r["width"]["comparison"] and r["height"]["comparison"] is None

    def test_no_estimate_no_report(self):
        for dims in (None, {}, dict(width_mm=None, height_mm=None)): assert F.validation_report(dims, F.parse_actual_dimensions(1, 1))["available"] is False

    def test_ground_truth_never_changes_the_estimate(self):
        before = copy.deepcopy(self.DIMS); F.validation_report(self.DIMS, F.parse_actual_dimensions("1", "1", "m")); F.validation_report(self.DIMS, F.parse_actual_dimensions("9999", "9999", "mm"))
        assert self.DIMS == before

    def test_ground_truth_is_not_a_parameter_of_the_estimation_path(self):
        import inspect
        for fn in (F.F6Session.measure, F.F6Session.select_frames, F.F6Session.detect_boundaries, F.F6Session.evaluate, F.F6Session.confirm, F.F6Session.import_calibration, F.F6Session.submit_videos):
            names = " ".join(inspect.signature(fn).parameters).lower(); assert "actual" not in names and "truth" not in names
        assert "actual" not in " ".join(inspect.signature(VM.VisitorMeasurementJob.measure).parameters).lower()
        src = (WF / "f6_core.py").read_text(); body = src[src.index("def measure(self)"):src.index("def dimensions(self)")]
        assert "self.actual" not in body and "ActualDimensions" not in body


# ===================================================================== calibration import (truthful) / Stage A
class TestCalibration:
    def test_invalid_json(self): assert F.parse_calibration_json(b"{not json")[0] is None
    def test_not_an_object(self): assert F.parse_calibration_json(b"[1,2]")[0] is None
    def test_unrecognised_layout(self): cal, probs = F.parse_calibration_json(b'{"foo": 1}', capture_mode="video"); assert cal is None and probs

    def test_phase1_layout_maps_without_inventing_values(self):
        cal = fixture_calibration(); d = json.loads(PHASE1_CAL.read_text())
        assert cal["K1"] == d["left"]["K"] and cal["D2"] == d["right"]["distortion_k1_k2_p1_p2_k3"] and cal["image_size_wh"] == d["image_size_wh"] and cal["baseline_mm"] == 225.0
        assert cal["capture_mode"] == dict(left="video", right="video")

    def test_capture_mode_is_never_inferred(self):
        cal, _ = F.parse_calibration_json(PHASE1_CAL.read_bytes(), capture_mode=None, baseline_mm=225.0); assert "capture_mode" not in cal

    def test_missing_capture_mode_is_rejected_by_stage_a(self):
        cal, _ = F.parse_calibration_json(PHASE1_CAL.read_bytes(), capture_mode=None, baseline_mm=225.0); s = F.F6Session()
        try:
            r = s.import_calibration(cal); assert not r["can_proceed"] and r["failures"][0]["code"] in ("INPUT_CONTRACT_MISSING_FIELD", "CAPTURE_MODE_PROVENANCE_REQUIRED") and "capture_mode" in json.dumps(r["failures"])
        finally: s.close()

    def test_missing_baseline_is_not_defaulted(self):
        d = json.loads(PHASE1_CAL.read_text()); d.pop("physical_baseline_mm", None); cal, _ = F.parse_calibration_json(json.dumps(d), capture_mode="video"); s = F.F6Session()
        try:
            assert cal["baseline_mm"] is None; r = s.import_calibration(cal); assert not r["can_proceed"] and "baseline_mm" in json.dumps(r["failures"])
        finally: s.close()

    def test_valid_calibration_passes_and_is_marked_unverified(self):
        s = F.F6Session()
        try:
            r = s.import_calibration(fixture_calibration()); assert r["can_proceed"] and r["overall_status"] in ("PASS", "WARNING") and r["calibration"]["accuracy_verified"] is False
            assert s.job.stage == "calibration_validated"
        finally: s.close()

    def test_bad_matrix_rejected(self):
        cal = fixture_calibration(); cal["K1"] = [[1, 2], [3, 4]]; s = F.F6Session()
        try: assert not s.import_calibration(cal)["can_proceed"]
        finally: s.close()

    def test_nonfinite_values_rejected(self):
        cal = fixture_calibration(); cal["D1"] = [float("nan")] * 5; s = F.F6Session()
        try: assert not s.import_calibration(cal)["can_proceed"]
        finally: s.close()

    def test_negative_baseline_rejected(self):
        s = F.F6Session()
        try: assert not s.import_calibration(fixture_calibration(baseline=-5.0))["can_proceed"]
        finally: s.close()

    def test_stage_a_contract_is_accepted_directly(self):
        cal = fixture_calibration(); cal2, probs = F.parse_calibration_json(json.dumps(cal)); assert not probs and cal2["K2"] == cal["K2"] and cal2["capture_mode"] == cal["capture_mode"]

    def test_f6_files_contain_no_calibration_algorithm_of_their_own(self):
        src = (WF / "f6_core.py").read_text() + (WF / "app_f6.py").read_text()
        for forbidden in ("calibrateCamera", "stereoCalibrate", "findChessboardCorners", "cornerSubPix"): assert forbidden not in src
        assert "does **not** calibrate" in (WF / "app_f6.py").read_text() and "F7" in (WF / "README_F6.md").read_text()      # calibration lives in the separate F7 wizard, which reuses the existing solver

    def test_stage_order_is_enforced(self):
        s = F.F6Session()
        try:
            r = s.submit_videos({k: "x" for k in F.VM.VIDEO_FIELDS}); assert not r["can_proceed"] and r["failures"][0]["code"] == "STAGE_OUT_OF_ORDER"
            with pytest.raises(F.F6Error): s.detect_boundaries()
        finally: s.close()


# ===================================================================== four-video upload mapping / Stage B
def _tiny_video(path, w=64, h=64, n=45, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)); rng = np.random.default_rng(0)
    for _ in range(n): vw.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    vw.release(); return path


class TestVideoUpload:
    def test_slot_to_camera_mapping_is_preserved(self):
        assert VM.VIDEO_VIEW == dict(placement_1_left="AL", placement_1_right="PL", placement_2_left="AR", placement_2_right="PR")
        for key, vk in VM.VIDEO_VIEW.items():
            pl, cam = V.VIEW_PLACEMENT[vk]; assert key == f"{pl}_{cam}"
        assert F.f1core.association("AL_corner")["camera"] == "left" and F.f1core.association("PL_ceilF")["camera"] == "right" and F.f1core.association("AR_corner")["placement"] == "placement_2"

    def test_required_boundaries_and_order(self):
        assert F.BOUNDARY_ORDER == ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF") and VM.TOP_VARIANT == 0
        assert set(F.LABELS) == set(F.BOUNDARY_ORDER) and all(" — placement " in v for v in F.LABELS.values())
        assert not any("_corner" in v or "_ceil" in v for v in F.LABELS.values())          # technical ids never in the visitor labels

    def test_missing_slot_rejected(self):
        s = F.F6Session()
        try:
            s.import_calibration(fixture_calibration()); r = s.submit_videos(dict(placement_1_left="a", placement_1_right="b", placement_2_left="c")); assert not r["can_proceed"] and r["failures"][0]["code"] == "INPUT_CONTRACT_MISSING_FIELD"
        finally: s.close()

    def test_nonexistent_paths_rejected(self, tmp_path):
        s = F.F6Session()
        try:
            s.import_calibration(fixture_calibration()); r = s.submit_videos({k: str(tmp_path / f"{k}.mov") for k in VM.VIDEO_FIELDS}); assert not r["can_proceed"] and s.job.stage == "calibration_validated"
        finally: s.close()

    def test_garbage_bytes_rejected_clearly(self):
        s = F.F6Session()
        try:
            s.import_calibration(fixture_calibration()); r = s.submit_videos({k: b"not a video" for k in VM.VIDEO_FIELDS}); assert not r["can_proceed"] and r["failures"] and s.job.stage == "calibration_validated"
        finally: s.close()

    def test_wrong_resolution_is_not_silently_accepted(self, tmp_path):
        vids = {k: str(_tiny_video(tmp_path / f"{k}.mp4")) for k in VM.VIDEO_FIELDS}; s = F.F6Session()
        try:
            s.import_calibration(fixture_calibration()); r = s.submit_videos(vids); assert not r["can_proceed"]
            assert any("INCOMPATIBLE" in f["code"] or "VALIDATION_BLOCKING" in f["code"] for f in r["failures"]) and s.job.stage == "calibration_validated"
        finally: s.close()

    def test_stage_b_limits_are_not_loosened(self):
        from room_measurement import visitor_video_validation as VV
        assert VV.DEFAULT_VIDEO_LIMITS.max_file_bytes == 200 * 1024 * 1024 and VV.DEFAULT_VIDEO_LIMITS.max_duration_s == 120.0 and VV.DEFAULT_VIDEO_LIMITS.min_duration_s == 1.0
        assert "--server.maxUploadSize\", \"200\"" in (MODULE / "run_stage_f6_visitor_app.py").read_text()

    def test_videos_given_by_path_are_never_copied_or_deleted(self, tmp_path):
        vids = {k: str(_tiny_video(tmp_path / f"{k}.mp4")) for k in VM.VIDEO_FIELDS}; s = F.F6Session(); s.import_calibration(fixture_calibration()); s.submit_videos(vids); ws = s.workspace_path(); s.close()
        assert all(Path(v).exists() for v in vids.values()) and not ws.exists()


# ===================================================================== no ML / no deployment edits
class TestClassicalOnly:
    FORBIDDEN = ("torch", "torchvision", "tensorflow", "keras", "transformers", "onnx", "onnxruntime", "sklearn", "segformer", "depth_anything", "timm", "sam2", "huggingface_hub", "mediapipe")
    FILES = ("f6_core.py", "app_f6.py", "browser_walkthrough_f6.py")

    def test_no_ml_imports_in_f6_files(self):
        for f in self.FILES:
            tree = ast.parse((WF / f).read_text()); names = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import): names |= {a.name.split(".")[0].lower() for a in n.names}
                if isinstance(n, ast.ImportFrom) and n.module: names.add(n.module.split(".")[0].lower())
            assert not names & set(self.FORBIDDEN), (f, names & set(self.FORBIDDEN))

    def test_no_f4_f5_model_code_referenced(self):
        for f in self.FILES:
            txt = (WF / f).read_text().lower(); assert "stage_f4" not in txt and "model_boundaries" not in txt and "segformer" not in txt, f

    def test_importing_the_core_does_not_load_any_ml_library(self):
        code = ("import sys; sys.path[:0]=[%r, %r]; import f6_core; bad=[m for m in sys.modules if m.split('.')[0] in %r]; print('LOADED', bad)" % (str(MODULE), str(WF), self.FORBIDDEN))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT)).stdout
        assert "LOADED []" in out, out

    def test_requirements_and_navigation_untouched_by_name(self):
        assert "streamlit-image-coordinates" in (ROOT / "requirements.txt").read_text()
        page = (ROOT / "Pages" / "app_module7.py").read_text(); assert "visitor_workflow" not in page and "f6_core" not in page and "app_f6" not in page


# ===================================================================== page smoke test (AppTest; no browser)
class TestPageSmoke:
    def test_first_page_renders_the_wizard_instructions_only(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(WF / "app_f6.py"), default_timeout=90).run()
        try:
            assert not at.exception, at.exception
            text = " ".join(m.value for m in at.markdown); assert "Measure a Room" in text and "Calibration" in text and "Calibrate Your Cameras" in text          # F7 wizard, step 1
            assert not at.text_input and not at.number_input and not at.get("file_uploader")                                                                      # only the current step is drawn
            assert at.session_state["f6_step"] == 0
        finally:
            try: at.session_state["f6"].close()
            except Exception: pass


# ===================================================================== integration on the original-room recordings (read-only)
DECISIONS = {"AL_corner": ("manual", [[509, 450], [510, 1450]]), "PL_corner": ("pick", 1), "PL_ceilF": ("pick", 1), "AR_corner": ("pick", 3), "PR_corner": ("pick", 3), "PR_ceilF": ("pick", 2)}


def _video_fingerprint():
    return {k: (VIDEO_DIR / f).stat().st_size for k, f in FIXTURE_VIDEOS.items()}, {k: (VIDEO_DIR / f).stat().st_mtime_ns for k, f in FIXTURE_VIDEOS.items()}


def _protected_hashes():
    ref = json.loads((MODULE / "outputs/room_measurement/stage_e2_visitor_backend/results.json").read_text())["protected_sha256"]; out = {}
    for rel, h in ref.items():
        if rel.startswith("video:"): continue
        out[rel] = (hashlib.sha256((MODULE / rel).read_bytes()).hexdigest(), h)
    return out


@pytest.fixture(scope="module")
def ready():
    if not HAVE_VIDEOS: pytest.skip("original-room recordings / calibration not present")
    before = _video_fingerprint(); s = F.F6Session()
    r1 = s.import_calibration(fixture_calibration()); assert r1["can_proceed"], r1
    r2 = s.submit_videos({k: str(VIDEO_DIR / f) for k, f in FIXTURE_VIDEOS.items()}); assert r2["can_proceed"], r2
    r3 = s.select_frames(); assert r3["can_proceed"], r3
    yield s, before
    s.close()


@pytest.mark.skipif(not HAVE_VIDEOS, reason="original-room recordings not present")
class TestIntegration:
    def test_01_frame_selection_integration(self, ready):
        s, _ = ready; assert s.job.stage == "frames_selected" and s.job.n_sets == 2
        for si in range(2):
            for vk in VM.VIEW_KEYS:
                rec = s.frame_record(si, vk); assert isinstance(rec["frame_index"], int) and len(rec["frame_fingerprint"]) == 64 and Path(rec["preview_png"]).exists()
                assert (rec["placement"], rec["camera"]) == V.VIEW_PLACEMENT[vk]

    def test_02_candidates_generated_for_every_required_boundary_in_every_set(self, ready):
        s, _ = ready; assert len(s.detections) == 12 and set(ln for _, ln in s.detections) == set(F.BOUNDARY_ORDER)
        for (si, ln), br in s.detections.items():
            assert br.candidates and br.boundary_id == ln and br.set_index == si and br.frame_fingerprint == s.frame_record(si, br.view_key)["frame_fingerprint"] and br.frame_index == s.frame_record(si, br.view_key)["frame_index"]
            assert [c["rank"] for c in br.candidates] == list(range(1, len(br.candidates) + 1)) and all("rank_score" in c and "confuser_class" in c for c in br.candidates)
            assert br.status in ("DETECTED", "UNCERTAIN", "NOT_DETECTED") and br.diagnostics is not None          # rankings + diagnostics retained

    def test_03_top_candidate_is_only_a_proposal(self, ready):
        s, _ = ready
        assert s.progress()["confirmed"] == 0 and not s.progress()["all_confirmed"]
        with pytest.raises(F.F6Error): s.export_for_measurement()
        for (si, ln) in s.reviews: assert s.review(si, ln).acceptance is None and not s.is_confirmed(si, ln)

    def test_04_stage_d_integration_and_provenance(self, ready):
        s, _ = ready; res = s.evaluate(0, "PL_corner"); rv = s.review(0, "PL_corner")
        assert res["status"] in ("PASS", "WARNING", "FAIL") and res["refined_line"] and res["identity_ok"]
        assert rv.raw["coordinate_space"] == "undistorted_full" and rv.raw["source"] == "stage_f2_automatic_detection" and rv.raw["frame_fingerprint"] == s.frame_record(0, "PL").get("frame_fingerprint")
        assert rv.raw["set_index"] == 0 and rv.raw["frame_index"] == s.frame_record(0, "PL")["frame_index"] and rv.raw["placement"] == "placement_1" and rv.raw["camera"] == "right" and rv.raw["boundary_type"] == "corner"
        assert "cand1" in rv.raw["annotation_id"]

    def test_05_pass_boundary_needs_explicit_confirmation_and_records_identity(self, ready):
        s, _ = ready; s.select_candidate(0, "PL_corner", 1); assert s.evaluate(0, "PL_corner")["decision"] == "ALLOWED" and s.status(0, "PL_corner") == "PROPOSED"
        assert s.confirm(0, "PL_corner") and s.status(0, "PL_corner") == "CONFIRMED"
        a = s.review(0, "PL_corner").acceptance
        for k in ("set_index", "line_name", "camera", "placement", "frame_index", "frame_fingerprint", "coordinate_space", "annotation_id", "annotation_digest", "refined_line_digest", "source", "rank", "candidate_provenance"): assert a[k] is not None, k
        assert (a["camera"], a["placement"], a["source"], a["rank"]) == ("right", "placement_1", "candidate", 1) and not s.acceptance_problems(0, "PL_corner")

    def test_06_candidate_switching_invalidates_confirmation(self, ready):
        s, _ = ready; assert s.is_confirmed(0, "PL_corner"); old = s.review(0, "PL_corner").raw["annotation_id"]
        s.select_candidate(0, "PL_corner", 2)
        assert not s.is_confirmed(0, "PL_corner") and s.review(0, "PL_corner").acceptance is None and s.review(0, "PL_corner").confirmation is None
        assert s.evaluate(0, "PL_corner")["annotation_id"] != old and s.status(0, "PL_corner") == "PROPOSED"
        s.select_candidate(0, "PL_corner", 1); assert not s.is_confirmed(0, "PL_corner")                    # going back does NOT resurrect the old confirmation
        with pytest.raises(ValueError): s.select_candidate(0, "PL_corner", 99)
        with pytest.raises(ValueError): s.select_candidate(0, "PL_corner", 0)

    def test_07_fail_can_never_be_confirmed(self, ready):
        s, _ = ready; s.select_candidate(0, "PR_ceilF", 1); res = s.evaluate(0, "PR_ceilF")
        assert res["status"] == "FAIL" and res["decision"] == "BLOCKED_FAIL" and s.status(0, "PR_ceilF") == "BLOCKED"
        assert s.confirm(0, "PR_ceilF") is False and s.review(0, "PR_ceilF").acceptance is None and s.review(0, "PR_ceilF").confirmation is None
        with pytest.raises(F.F6Error): s.export_for_measurement()

    def test_08_warning_needs_the_exact_stage_e2_confirmation(self, ready):
        s, _ = ready; s.select_candidate(0, "AL_corner", 1); res = s.evaluate(0, "AL_corner")
        assert res["status"] == "WARNING" and res["decision"] == "BLOCKED_UNCONFIRMED_WARNING" and res["confirmation_request"] and s.status(0, "AL_corner") == "NEEDS_CONFIRMATION"
        assert s.confirm(0, "AL_corner") and s.status(0, "AL_corner") == "CONFIRMED"
        c = s.review(0, "AL_corner").confirmation; assert isinstance(c, VM.Confirmation) and c.confirmed is True and c.confirmed_by == "f6_visitor"
        assert c.frame_fingerprint == s.frame_record(0, "AL")["frame_fingerprint"] and set(c.confirmed_codes) == set(res["warning_codes"]) and c.line_name == "AL_corner" and c.set_index == 0
        s.reject(0, "AL_corner"); assert s.review(0, "AL_corner").confirmation is None and not s.is_confirmed(0, "AL_corner")

    def test_09_stale_confirmations_are_rejected(self, ready):
        s, _ = ready; s.select_candidate(0, "PL_corner", 1); assert s.confirm(0, "PL_corner"); good = copy.deepcopy(s.review(0, "PL_corner").acceptance)
        for field, value in (("frame_fingerprint", "0" * 64), ("frame_index", good["frame_index"] + 1), ("annotation_digest", "x"), ("refined_line_digest", "x"), ("annotation_id", "other"), ("set_index", 1), ("line_name", "PR_corner"), ("source", "manual"), ("rank", 2)):
            s.review(0, "PL_corner").acceptance = dict(good, **{field: value}); assert s.acceptance_problems(0, "PL_corner"), field; assert not s.is_confirmed(0, "PL_corner")
        s.review(0, "PL_corner").acceptance = dict(good, revision=good["revision"] + 1); assert not s.is_confirmed(0, "PL_corner")
        s.review(0, "PL_corner").acceptance = copy.deepcopy(good); assert s.is_confirmed(0, "PL_corner")
        tampered = copy.deepcopy(s.review(0, "PL_corner").raw); tampered["points"] = [[p[0] + 5, p[1]] for p in tampered["points"]]; keep = s.review(0, "PL_corner").raw; s.review(0, "PL_corner").raw = tampered
        assert not s.is_confirmed(0, "PL_corner"); s.review(0, "PL_corner").raw = keep; assert s.is_confirmed(0, "PL_corner")

    def test_10_confirmation_does_not_carry_to_another_set_camera_or_boundary(self, ready):
        s, _ = ready; assert s.is_confirmed(0, "PL_corner"); acc = s.review(0, "PL_corner").acceptance
        s.evaluate(1, "PL_corner"); s.review(1, "PL_corner").acceptance = copy.deepcopy(acc); assert not s.is_confirmed(1, "PL_corner"); s.review(1, "PL_corner").acceptance = None            # another frame
        s.evaluate(0, "AL_corner"); s.review(0, "AL_corner").acceptance = copy.deepcopy(acc); assert not s.is_confirmed(0, "AL_corner"); s.review(0, "AL_corner").acceptance = None            # another camera / boundary
        assert not s.is_confirmed(1, "PL_corner") and not s.is_confirmed(0, "AL_corner")

    def test_11_frame_fingerprint_validation_never_substitutes(self, ready):
        s, _ = ready; rec = s.frame_record(1, "PR"); p = Path(rec["preview_png"]); good = p.read_bytes()
        try:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE); img[100:140, 100:140] = 255 - img[100:140, 100:140]; cv2.imwrite(str(p), img)
            with pytest.raises(F.f1core.FrameIntegrityError): s.load_gray(1, "PR")
            s.select_candidate(1, "PR_corner", 3)
            with pytest.raises(F.f1core.FrameIntegrityError): s.evaluate(1, "PR_corner")
        finally: p.write_bytes(good)
        assert V.frame_fingerprint(s.load_gray(1, "PR")) == rec["frame_fingerprint"]
        wrong = F2_annotation_with_wrong_fp(s); ev = VM.evaluate_set([wrong], {"PR": s.load_gray(1, "PR")}, {"PR": rec["frame_fingerprint"]}, {"PR": rec["frame_index"]}, s.job.cal, 1, s.job.annotation_limits)
        assert not ev["entries"][0]["identity"]["ok"] and "fingerprint" in " ".join(ev["entries"][0]["identity"]["reasons"])

    def test_12_manual_correction_fallback(self, ready):
        s, _ = ready; s.use_manual(0, "AL_corner"); rv = s.review(0, "AL_corner"); assert rv.source == "manual" and s.status(0, "AL_corner") == "NO_SUGGESTION" and not s.is_confirmed(0, "AL_corner")
        ms = s.manual_focus(0, "AL_corner"); assert ms.click(509, 450) == "point 1 placed" and ms.click(510, 1450) == "point 2 placed"
        assert s.confirm(0, "AL_corner") is False                                                              # not refined yet -> nothing to confirm
        res = ms.submit(); assert res["status"] in ("PASS", "WARNING", "FAIL") and res["refined_line"]
        if res["decision"] in ("ALLOWED", "BLOCKED_UNCONFIRMED_WARNING"):
            assert s.confirm(0, "AL_corner") and s.is_confirmed(0, "AL_corner"); a = s.review(0, "AL_corner").acceptance; assert a["source"] == "manual" and a["coordinate_space"] == "distorted_full" and a["rank"] is None
            assert s.review(0, "AL_corner").raw["frame_fingerprint"] == s.frame_record(0, "AL")["frame_fingerprint"] and s.review(0, "AL_corner").raw["source"] == F.f1core.SOURCE
            ms.clear(); assert not s.is_confirmed(0, "AL_corner")                                              # editing the points invalidates it
            ms.click(509, 450); ms.click(510, 1450); assert not s.is_confirmed(0, "AL_corner")                  # re-entering the same points does not restore it
            ms.submit(); assert s.confirm(0, "AL_corner") and s.is_confirmed(0, "AL_corner")
        s.select_candidate(0, "AL_corner", 2); assert s.review(0, "AL_corner").source == "candidate" and not s.is_confirmed(0, "AL_corner")

    def test_13_overlay_image_renders_proposal_and_refinement(self, ready):
        s, _ = ready; s.select_candidate(0, "PL_ceilF", 1); img = F.overlay_image(s, 0, "PL_ceilF"); assert img.ndim == 3 and img.shape[2] == 3 and img.shape[0] <= 900 and img.shape[1] <= 560
        assert (img[..., 1].astype(int) - img[..., 0].astype(int) > 80).any()                                   # green refined line drawn

    def test_14_measurement_refuses_until_everything_is_confirmed_and_returns_no_fabricated_dimensions(self, ready):
        s, _ = ready
        with pytest.raises(F.F6Error): s.measure()
        assert s.dimensions() is None and s.validation()["available"] is False

    def test_15_full_measurement_stage_e2_optional_ground_truth_and_cleanup(self, ready):
        s, before = ready; ms = None
        for si in range(2):
            for ln in F.BOUNDARY_ORDER:
                kind, arg = DECISIONS[ln]
                if kind == "pick": s.select_candidate(si, ln, arg)
                else:
                    s.use_manual(si, ln); ms = s.manual_focus(si, ln); ms.clear(); [ms.click(*pt) for pt in arg]; ms.submit()
                assert s.evaluate(si, ln)["decision"] in ("ALLOWED", "BLOCKED_UNCONFIRMED_WARNING"), (si, ln, s.evaluate(si, ln)["decision"])
                assert s.confirm(si, ln), (si, ln)
        assert s.progress() == dict(confirmed=12, total=12, all_confirmed=True)
        anns, confs = s.export_for_measurement(); assert len(anns) == 12 and all(a["frame_fingerprint"] for a in anns) and {a["annotation_id"] for a in anns} == {s.review(si, ln).raw["annotation_id"] for si in range(2) for ln in F.BOUNDARY_ORDER}
        m = s.measure()
        pend = s.pending_set_confirmations()
        for q in pend: s.confirm_set_issue(q)
        if pend: m = s.measure()
        assert m["step"] == "measure" and m["ground_truth_used"] is False and m["top_variant"] == 0
        if m["can_proceed"]:                                                                                    # geometrically valid -> dimensions exist; accuracy is NOT asserted
            d = s.dimensions(); assert d and all(math.isfinite(d[k]) and d[k] > 0 for k in ("width_mm", "height_mm")) and 500 < d["width_mm"] < 20000 and 500 < d["height_mm"] < 20000
            snap = copy.deepcopy(d); assert s.validation()["width"]["comparison"] is None                       # no actual -> estimate only
            s.set_actual_dimensions("2900", "2400", "mm"); v = s.validation(); assert v["width"]["comparison"]["absolute_percent_error"] == pytest.approx(abs(d["width_mm"] - 2900) / 2900 * 100)
            assert v["height"]["comparison"]["absolute_percent_error"] == pytest.approx(abs(d["height_mm"] - 2400) / 2400 * 100) and v["ground_truth_used_in_estimation"] is False
            s.set_actual_dimensions("9.5", "7.9", "ft"); assert s.dimensions() == snap                         # changing the actual values never moves the estimate
        else:
            assert s.dimensions() is None and s.validation()["available"] is False                              # failed reconstruction -> NO numbers
        exported = s.summary_for_export(); assert exported["ground_truth_used_in_estimation"] is False and exported["no_machine_learning"] is True
        assert _video_fingerprint() == before                                                                   # original recordings untouched
        assert all(a == b for a, b in _protected_hashes().values())                                             # Phase 1 + earlier-stage protected files unchanged

    def test_16_unconfirming_after_measurement_blocks_a_second_measurement(self, ready):
        s, _ = ready; s.select_candidate(0, "PL_corner", 2); assert not s.is_confirmed(0, "PL_corner")
        with pytest.raises(F.F6Error): s.measure()


def F2_annotation_with_wrong_fp(s):
    raw = copy.deepcopy(s.review(1, "PR_corner").raw) if s.review(1, "PR_corner").raw else F.F2.candidate_to_annotation(s.detections[(1, "PR_corner")], 3, set_index=1); raw["frame_fingerprint"] = "f" * 64; return raw


# ===================================================================== session isolation / cleanup / safe failure
@pytest.mark.skipif(not PHASE1_CAL.exists(), reason="calibration fixture missing")
class TestIsolationAndSafety:
    def test_sessions_have_separate_workspaces_and_state(self):
        a, b = F.F6Session(), F.F6Session()
        try:
            assert a.session_id != b.session_id and a.workspace_path() != b.workspace_path() and a.workspace_path().exists() and b.workspace_path().exists()
            a.import_calibration(fixture_calibration()); assert a.job.stage == "calibration_validated" and b.job.stage == "created"
            a.set_actual_dimensions("1", "1", "m"); assert not b.actual.any and a.actual.any
            a.reviews[(0, "AL_corner")] = F.BoundaryReview(0, "AL_corner"); assert not b.reviews
            ws_a = a.workspace_path(); a.close(); assert not ws_a.exists() and b.workspace_path().exists() and b.job.stage == "created"
        finally: a.close(); b.close()

    def test_close_is_idempotent_and_releases_the_job_slot(self):
        n0 = VM.VisitorMeasurementJob.active_jobs(); s = F.F6Session(); assert VM.VisitorMeasurementJob.active_jobs() == n0 + 1
        s.close(); s.close(); assert VM.VisitorMeasurementJob.active_jobs() == n0

    def test_close_only_removes_the_sessions_own_workspace(self, tmp_path):
        keep = tmp_path / "precious.txt"; keep.write_text("x"); s = F.F6Session(); ws = s.workspace_path(); assert tmp_path not in ws.parents; s.close(); assert keep.exists() and not ws.exists()

    def test_set_count_is_validated(self):
        for bad in (0, 1, 4, True, "2", None, 2.0):
            with pytest.raises(ValueError): F.F6Session(n_sets=bad)
        s = F.F6Session(); s.set_frame_set_count(3); assert s.n_sets == 3
        with pytest.raises(ValueError): s.set_frame_set_count(5)
        s.close()

    def test_failed_measure_gives_no_dimensions(self):
        s = F.F6Session()
        try:
            s.measurement = dict(step="measure", can_proceed=False, failures=[dict(code="X", message="failed")], dimensions=dict(width_mm=1.0, height_mm=1.0))      # even a stray dict must not be shown
            assert s.dimensions() is None and s.validation()["available"] is False
            s.measurement = dict(step="submit_confirmations", can_proceed=True, dimensions=dict(width_mm=1.0, height_mm=1.0)); assert s.dimensions() is None
        finally: s.close()

    def test_review_calls_before_detection_fail_safely(self):
        s = F.F6Session()
        try:
            with pytest.raises(F.F6Error): s.review(0, "AL_corner")
            assert s.progress() == dict(confirmed=0, total=0, all_confirmed=False)
            with pytest.raises(F.F6Error): s.measure()
        finally: s.close()
