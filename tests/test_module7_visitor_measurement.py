"""Phase 2B Stage E2: general visitor measurement backend (room_measurement/visitor_measurement.py).
Synthetic videos / frames; the numeric measurement stages are mocked where noted (the real-video regression is the runner script)."""
import ast
import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7" / "room_measurement_question_3"      # Question 3 packages, data and outputs
sys.path.insert(0, str(MODULE)); sys.path.insert(0, str(MODULE.parent)); sys.path.insert(0, str(ROOT / "tests"))

from room_measurement import calibration as C, frames as Fr, lines as L, visitor_annotations as V, visitor_frame_selection as FS, visitor_measurement as VM  # noqa: E402
from room_measurement.pipeline import to_jsonable  # noqa: E402
import test_module7_visitor_annotations as T  # noqa: E402  (analytic scene renderer reused, not modified)

W, H = T.W, T.H
N_FRAMES = 15


def scene(): return T.render()


def write_video(path, frame=None, n=N_FRAMES, size=(W, H)):
    f = scene() if frame is None else frame; w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for _ in range(n): w.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f)
    w.release()


def calibration(size=(W, H), baseline=225.0, mode="video"):
    d = dict(K1=T.K.tolist(), D1=[0.0] * 5, K2=T.K.tolist(), D2=[0.0] * 5, image_size_wh=list(size), baseline_mm=baseline)
    if mode is not None: d["capture_mode"] = dict(left=mode, right=mode)
    return d


def fake_selection(n_sets=2, status="SUCCESS"):
    sets = []
    for i in range(n_sets):
        pf = lambda pl, base: FS.PlacementFrame(pl, i, base + i, base + i + 1, 0.1 * i, 0.1 * i, 100.0, 100.0, 500, 500, 300, 250, 0.8, 0.9)
        sets.append(FS.FrameSet(i, pf("placement_1", 1), pf("placement_2", 5), FS.CrossPlacementDiagnostics(i, 200, 150, 120, True, 0.9)))
    iss = [FS.Issue(FS.Severity.WARNING, "SETS_ARE_CORRELATED", "correlated", "selection")]
    return FS.FrameSelectionResult(status, n_sets, sets if status == "SUCCESS" else [], iss if status == "SUCCESS" else [FS.Issue(FS.Severity.ERROR, "STEREO_INSUFFICIENT", "no stereo", "placement_1")], {}, {}, {}, {})


def geom(**kw):
    v = dict(n_stereo_matches=1000, n_gated_points=950, gated_fraction=0.95, reprojection_rms_px=0.2, epipolar_rms_px=0.3, median_depth_mm=4000.0, n_floor_candidates=200, floor_plane_rms_mm=2.0, floor_tilt_deg=5.0, floor_offset_mm=1500.0)
    v.update(kw); return dict(view1=dict(v), view2=dict(v))


def rows(n, w=2850.0, h=2350.0, **over):
    out = []
    for i in range(n):
        r = dict(ok=True, width_mm=w + i, height_mm=h + i, H1=h, H2=h, vertical_error_deg=(0.5, 0.5), skew_mm=(2.0, 2.0), registration=dict(rms_mm=3.0, n_inliers=140, n_pairs=145), geometry=geom(),
                 coverage=dict(n_pairs=145, bbox_fraction=0.4, cells_occupied=6, cells_total=9), bottom_corners=dict(V1=[0, 0, 0], V2=[0, 0, 0])); r.update(over); out.append(r)
    return out


def est_factory(yaws=None, degenerate=False, inlier=0.95):
    seq = list(yaws or []); calls = []
    def est(D):
        y = seq[len(calls)] if len(calls) < len(seq) else 6.0; calls.append(y)
        flags = dict(converged=True, scale_px=0.26, inlier_fraction=inlier, degenerate=degenerate)
        return np.array([0.0, np.radians(y), 0.0, np.pi / 2, 0.0]), None, dict(method_J=dict(flags=flags, n_inlier=9000, n_residuals=9500, n_tracks=300)), {}
    return est


class Mocks:
    """Patch the heavy numeric stages; the visitor workflow, identity checks, gate, workspace and checks run for real."""
    def __init__(self, n_sets=2, rows_fn=None, selection=None, yaws=None, degenerate=False, stereo=V.Status.PASS, measure_side=None):
        self.n_sets, self.rows_fn, self.selection, self.yaws, self.degenerate, self.stereo, self.measure_side = n_sets, rows_fn, selection, yaws, degenerate, stereo, measure_side; self.p = []

    def __enter__(self):
        sel = self.selection or fake_selection(self.n_sets)
        def measure_sets(cal, p5, ms, ls, wh, seed=3):
            if self.measure_side: return self.measure_side(p5, ms, ls)
            r = rows(len(ms)); k = 300.0 * (np.degrees(p5[1]) - 6.0)
            for x in r: x["width_mm"] += k
            return self.rows_fn(len(ms)) if self.rows_fn else r
        self.p = [mock.patch.object(FS, "select_frame_sets", return_value=sel), mock.patch.object(VM.features, "match_set", return_value={"fake": 1}), mock.patch.object(VM.P, "build_dataset", return_value=dict(n1=np.zeros((10, 2)))),
                  mock.patch.object(VM.P, "estimate_pose", side_effect=est_factory(self.yaws, self.degenerate)), mock.patch.object(VM, "_measure_sets", side_effect=measure_sets),
                  mock.patch.object(VM.V, "stereo_consistency", return_value=(self.stereo, [], {}))]
        [p.start() for p in self.p]; return self

    def __exit__(self, *a): [p.stop() for p in reversed(self.p)]; return False


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name); cls.paths = {}
        for f in VM.VIDEO_FIELDS: write_video(d / f"{f}.mp4"); cls.paths[f] = str(d / f"{f}.mp4")

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def job(self, **kw):
        j = VM.VisitorMeasurementJob(**kw); self.addCleanup(j.close); return j

    def to_frames(self, job, n=2, mocks=None):
        self.assertTrue(job.validate_calibration(calibration())["can_proceed"]); self.assertTrue(job.validate_videos(self.paths)["can_proceed"])
        r = job.select_frames(n); self.assertTrue(r["can_proceed"], r["failures"]); return r

    def annotations(self, sel, shift=None, skip=(), ceils=False, **override):
        shift = shift or {}; out = []; ex, cy = T.EDGE_X, T.CEIL_Y
        for f in sel["selected_frames"]:
            for vk, v in f["views"].items():
                names = [n for n in VM.REQUIRED_LINES if n[:2] == vk] + (["PL_ceilS"] if ceils and vk == "PL" else [])
                for nm in names:
                    if (f["set_index"], nm) in skip or nm in skip: continue
                    s = shift.get(nm, 0.0); pts = ((ex + s, 150), (ex + s, 270)) if nm.endswith("corner") else ((60, cy + s), (340, cy + s)) if nm.endswith("ceilF") else ((60, 150), (340, 250))
                    out.append(V.build_annotation(set_index=f["set_index"], placement=v["placement"], camera=v["camera"], frame_index=v["frame_index"], boundary_type=nm.split("_")[1], image_wh=(v["image"]["width"], v["image"]["height"]),
                                                  coordinate_space="undistorted_full", points=pts, annotation_id=f"t:{f['set_index']}:{nm}", frame_fingerprint=v["frame_fingerprint"]))
        for a in out:
            for k, v in override.items(): a[k] = v
        return out

    def ready_job(self, n=2, **annkw):
        m = Mocks(n); m.__enter__(); self.addCleanup(m.__exit__); j = self.job(); sel = self.to_frames(j, n); r = j.submit_annotations(self.annotations(sel, **annkw)); return j, sel, r


def fcodes(r): return [f["code"] for f in r["failures"]]


class CompleteWorkflow(Base):
    def test_complete_visitor_workflow(self):
        with Mocks(2):
            j = self.job(); r1 = j.validate_calibration(calibration()); self.assertEqual((r1["overall_status"], r1["stage"]), ("PASS", "calibration_validated")); self.assertFalse(r1["calibration"]["accuracy_verified"])
            r2 = j.validate_videos(self.paths); self.assertEqual(r2["stage"], "videos_validated"); self.assertEqual(r2["video_diagnostics"]["calibration_compatibility"]["status"], "COMPATIBLE_METADATA")
            r3 = j.select_frames(2); self.assertEqual(r3["stage"], "frames_selected"); self.assertEqual(len(r3["selected_frames"]), 2); ins = r3["annotation_instructions"]
            self.assertEqual(len(ins["per_frame"]), 8); self.assertEqual(ins["required_lines"], list(VM.REQUIRED_LINES)); self.assertIn("need not be visible", ins["hidden_corner_note"])
            for f in r3["selected_frames"]:
                for v in f["views"].values(): self.assertTrue(Path(v["preview_png"]).exists()); self.assertEqual(len(v["frame_fingerprint"]), 64)
            r5 = j.submit_annotations(self.annotations(r3)); self.assertEqual(r5["stage"], "annotations_reviewed"); self.assertTrue(r5["can_proceed"]); self.assertTrue(r5["annotation_review"]["measurement_ready"])
            self.assertTrue(all(v == "PASS" for s in r5["annotation_review"]["per_set"] for n, v in s["annotation_statuses"].items() if n in VM.REQUIRED_LINES))
            self.assertTrue(all(c == "NOT_EVALUATED" for s in r5["annotation_review"]["per_set"] for c in s["stereo_consistency"].values()))
            r = j.measure(); self.assertEqual(r["stage"], "measured", r["failures"]); self.assertTrue(r["can_proceed"]); self.assertIsNotNone(r["dimensions"]); self.assertEqual(r["top_variant"], 0); self.assertFalse(r["ground_truth_used"])
            for k in ("pose_diagnostics", "quality_checks", "registration", "geometry", "pose_stability", "stereo_consistency_post_pose", "runtime_s", "peak_rss_mb", "timings_s", "frame_indices"): self.assertIn(k, r, k)
            json.dumps(r)

    def test_run_workflow_helper(self):
        with Mocks(2):
            out = VM.run_workflow(calibration(), self.paths, 2, lambda sel, job: self.annotations(sel)); self.assertEqual(out["final"]["stage"], "measured"); self.assertEqual(len(out["steps"]), 5)

    def test_stages_enforce_order(self):
        j = self.job(); r = j.validate_videos(self.paths); self.assertFalse(r["can_proceed"]); self.assertEqual(fcodes(r), ["STAGE_OUT_OF_ORDER"]); self.assertIn("calibration", r["required_next_action"])
        self.assertEqual(fcodes(j.measure()), ["STAGE_OUT_OF_ORDER"])

    def test_changing_an_early_input_invalidates_later_work(self):
        j, sel, r = self.ready_job(); self.assertTrue(j.annotations)
        j.validate_calibration(calibration()); self.assertEqual(j.stage, "calibration_validated"); self.assertEqual(j.annotations, []); self.assertEqual(j.frames, [])

    def test_stop_and_resume_between_steps(self):
        with Mocks(2):
            j = self.job(); sel = self.to_frames(j, 2); state = json.loads(json.dumps(j.export_state())); j2 = VM.VisitorMeasurementJob.resume(state, j.ws); self.addCleanup(j2.close)
            self.assertEqual(j2.stage, "frames_selected"); r = j2.submit_annotations(self.annotations(sel)); self.assertTrue(r["can_proceed"])
            state2 = json.loads(json.dumps(j2.export_state())); j3 = VM.VisitorMeasurementJob.resume(state2, j.ws); self.addCleanup(j3.close); self.assertEqual(j3.stage, "annotations_reviewed")
            self.assertEqual(j3.measure()["stage"], "measured")
            with self.assertRaises(ValueError): VM.VisitorMeasurementJob.resume(state2, VM.VV.TemporaryVideoWorkspace())          # different workspace refused
            self.assertTrue(os.path.exists(j.ws.path))                                                                          # resumed jobs never delete a workspace they do not own


class InputContract(Base):
    def test_invalid_calibration(self):
        j = self.job()
        for bad, code in ((None, "INPUT_NOT_A_MAPPING"), ({k: v for k, v in calibration().items() if k != "K2"}, "INPUT_CONTRACT_MISSING_FIELD"), (calibration(mode=None), "INPUT_CONTRACT_MISSING_FIELD")):
            r = j.validate_calibration(bad); self.assertFalse(r["can_proceed"]); self.assertEqual(fcodes(r), [code])
        c = calibration(); c["capture_mode"] = dict(left="video"); self.assertEqual(fcodes(j.validate_calibration(c)), ["CAPTURE_MODE_PROVENANCE_REQUIRED"])
        c = calibration(); c["K1"] = [[0, 0, 0]] * 3; r = j.validate_calibration(c); self.assertEqual(fcodes(r), ["CALIBRATION_INVALID"]); self.assertEqual(j.stage, "created")

    def test_invalid_baseline(self):
        j = self.job()
        for b in (-5, 0, float("nan"), True, "225", 1e9, 1.0):
            r = j.validate_calibration(calibration(baseline=b)); self.assertFalse(r["can_proceed"], b); self.assertEqual(fcodes(r), ["CALIBRATION_INVALID"], b)

    def test_incompatible_video_dimensions(self):
        j = self.job(); self.assertTrue(j.validate_calibration(calibration(size=(W + 40, H + 30)))["can_proceed"])
        r = j.validate_videos(self.paths); self.assertFalse(r["can_proceed"]); self.assertIn("CALIBRATION_VIDEO_INCOMPATIBLE", fcodes(r)); self.assertEqual(j.stage, "calibration_validated")

    def test_photo_capture_mode_is_not_accepted_for_video(self):
        j = self.job(); j.validate_calibration(calibration(mode="photo")); r = j.validate_videos(self.paths); self.assertIn("CALIBRATION_VIDEO_INCOMPATIBLE", fcodes(r))

    def test_video_decode_failure_and_missing_videos(self):
        j = self.job(); j.validate_calibration(calibration()); bad = dict(self.paths, placement_2_right=b"this is not a video")
        r = j.validate_videos(bad); self.assertFalse(r["can_proceed"]); self.assertIn("VIDEO_VALIDATION_BLOCKING", fcodes(r))
        r = j.validate_videos({k: v for k, v in self.paths.items() if k != "placement_1_left"}); self.assertEqual(fcodes(r), ["INPUT_CONTRACT_MISSING_FIELD"])
        r = j.validate_videos(dict(self.paths, placement_2_left=Path(self.tmp.name) / "missing.mp4")); self.assertIn("VIDEO_VALIDATION_BLOCKING", fcodes(r))

    def test_frame_pixel_limit(self):
        j = self.job(limits=VM.JobLimits(max_frame_pixels=1000)); j.validate_calibration(calibration()); self.assertIn("FRAME_TOO_LARGE", fcodes(j.validate_videos(self.paths)))

    def test_frame_selection_failure(self):
        with Mocks(2, selection=fake_selection(2, "FAILED")):
            j = self.job(); j.validate_calibration(calibration()); j.validate_videos(self.paths); r = j.select_frames(2)
            self.assertFalse(r["can_proceed"]); self.assertEqual(fcodes(r), ["STEREO_INSUFFICIENT"]); self.assertEqual(j.stage, "videos_validated"); self.assertEqual(j.frames, [])
        j = self.job(); j.validate_calibration(calibration()); j.validate_videos(self.paths)
        self.assertEqual(fcodes(j.select_frames(0)), ["INVALID_FRAME_SET_COUNT"]); self.assertEqual(fcodes(j.select_frames(99)), ["INVALID_FRAME_SET_COUNT"])

    def test_real_frame_selection_failure_on_blank_videos(self):
        d = Path(self.tmp.name) / "blank"; d.mkdir(exist_ok=True); p = {}
        for f in VM.VIDEO_FIELDS: write_video(d / f"{f}.mp4", frame=np.full((H, W), 128, np.uint8)); p[f] = str(d / f"{f}.mp4")
        j = self.job(); j.validate_calibration(calibration()); j.validate_videos(p); r = j.select_frames(2); self.assertFalse(r["can_proceed"]); self.assertEqual(j.stage, "videos_validated")


class FrameIdentity(Base):
    def test_missing_annotations_block(self):
        j, sel, r = self.ready_job(skip=("PR_ceilF",))
        self.assertFalse(r["can_proceed"]); self.assertIn("REQUIRED_ANNOTATION_MISSING", fcodes(r)); self.assertFalse(r["annotation_review"]["measurement_ready"])
        m = j.measure(); self.assertFalse(m["can_proceed"]); self.assertIsNone(m["dimensions"]); self.assertIn("REQUIRED_ANNOTATION_MISSING", fcodes(m))
        j2 = self.job(); self.to_frames(j2); r = j2.submit_annotations([]); self.assertFalse(r["can_proceed"])

    def test_wrong_client_fingerprint_is_not_trusted(self):
        j, sel, r = self.ready_job(); anns = self.annotations(sel); anns[0]["frame_fingerprint"] = "0" * 64
        r = j.submit_annotations(anns); self.assertFalse(r["can_proceed"]); d = r["annotation_review"]["per_set"][0]["decisions"]["AL_corner"]
        self.assertEqual(d["decision"], "BLOCKED_IDENTITY"); self.assertFalse(d["identity"]["checks"]["fingerprint_matches"])
        anns = self.annotations(sel); del anns[0]["frame_fingerprint"]; r = j.submit_annotations(anns); self.assertIn("INVALID_ANNOTATION_RECORD", fcodes(r)); self.assertFalse(r["can_proceed"])

    def test_fingerprint_of_another_frame_in_the_same_job(self):
        j, sel, r = self.ready_job(); anns = self.annotations(sel); other = sel["selected_frames"][1]["views"]["AL"]["frame_fingerprint"]
        for a in anns:                                                     # set 0 annotation carrying set 1's fingerprint (identical pixels would collide; scene frames differ only by index here)
            if a["set_index"] == 0 and a["line_name"] == "AL_corner": a["frame_fingerprint"] = "a" * 64
        r = j.submit_annotations(anns); self.assertFalse(r["can_proceed"])

    def test_wrong_camera_placement_frame_index_and_set(self):
        j, sel, r = self.ready_job(); base = self.annotations(sel)
        def variant(**kw):
            a = copy.deepcopy(base); t = next(x for x in a if x["line_name"] == "AL_corner" and x["set_index"] == 0); t.update(kw); return a
        for kw in (dict(camera="right"), dict(placement="placement_2"), dict(frame_index=14), dict(set_index=1), dict(line_name="PL_corner")):
            r = j.submit_annotations(variant(**kw)); self.assertFalse(r["can_proceed"], kw)
        r = j.submit_annotations(variant(set_index=7)); self.assertIn("INVALID_ANNOTATION_RECORD", fcodes(r))
        r = j.submit_annotations(variant(image=dict(width=W // 2, height=H // 2))); self.assertFalse(r["can_proceed"])

    def test_tampered_stored_frame_is_detected(self):
        j, sel, r = self.ready_job(); p = Path(sel["selected_frames"][0]["views"]["AL"]["preview_png"]); cv2.imwrite(str(p), np.full((H, W), 7, np.uint8))
        j._evals = None; m = j.measure(); self.assertFalse(m["can_proceed"]); self.assertEqual(fcodes(m), ["FRAME_INTEGRITY_FAILED"]); self.assertIsNone(m["dimensions"])

    def test_calibration_size_vs_frames(self):
        with Mocks(2):
            j = self.job(); j.validate_calibration(calibration()); j.validate_videos(self.paths)
            j.cal = dataclasses.replace(j.cal, image_size=(W + 8, H + 8)); r = j.select_frames(2); self.assertFalse(r["can_proceed"]); self.assertEqual(fcodes(r), ["FRAME_SIZE_MISMATCH"])


class ConfirmationPolicy(Base):
    SHIFT = {"AL_corner": 12.0}

    def test_unconfirmed_warning_requires_confirmation_and_blocks(self):
        j, sel, r = self.ready_job(shift=self.SHIFT)
        self.assertFalse(r["can_proceed"]); self.assertEqual(r["overall_status"], "WARNING"); reqs = r["annotation_review"]["confirmation_requests"]; self.assertEqual(len(reqs), 2)       # one per set
        q = reqs[0]; self.assertEqual(q["warning_codes"], ["LARGE_REFINEMENT_DISPLACEMENT"]); self.assertTrue(q["required"])
        for k in ("annotation_digest", "refined_line_digest", "frame_fingerprint", "frame_index", "annotation_id"): self.assertTrue(q[k] is not None and q[k] != "", k)
        m = j.measure(); self.assertFalse(m["can_proceed"]); self.assertIsNone(m["dimensions"]); self.assertIn("REQUIRED_ANNOTATION_BLOCKED", fcodes(m))
        self.assertIn("confirm", r["required_next_action"])

    def confirmed_job(self):
        j, sel, r = self.ready_job(shift=self.SHIFT); confs = [VM.Confirmation.from_request(q, "tester", "looks right") for q in r["annotation_review"]["confirmation_requests"]]
        r2 = j.submit_confirmations(confs); return j, sel, r2

    def test_explicit_confirmation_allows_measurement(self):
        j, sel, r = self.confirmed_job(); self.assertTrue(r["can_proceed"], r["failures"]); self.assertEqual(r["annotation_review"]["confirmation_state"]["pending"], 0); self.assertEqual(r["annotation_review"]["confirmation_state"]["applied"], 2)
        m = j.measure(); self.assertEqual(m["stage"], "measured"); self.assertEqual(len(m["annotation_review"]["confirmations_applied"]), 2); self.assertTrue(all(c["confirmed"] for c in m["annotation_review"]["confirmations_applied"]))

    def test_confirmations_accepted_as_plain_dicts(self):
        j, sel, r = self.ready_job(shift=self.SHIFT); recs = [VM.Confirmation.from_request(q, "t").to_dict() for q in r["annotation_review"]["confirmation_requests"]]
        self.assertTrue(j.submit_confirmations(json.loads(json.dumps(recs)))["can_proceed"])

    def test_confirmation_contract_rejects_missing_identity_and_declines(self):
        j, sel, r = self.ready_job(shift=self.SHIFT); good = VM.Confirmation.from_request(r["annotation_review"]["confirmation_requests"][0], "t").to_dict()
        for key in ("annotation_digest", "refined_line_digest", "frame_fingerprint", "confirmed", "confirmed_codes", "annotation_id"):
            bad = dict(good); del bad[key]; rr = j.submit_confirmations([bad]); self.assertEqual(len(rr["annotation_review"]["confirmation_state"]["rejected_records"]), 1, key); self.assertFalse(rr["can_proceed"])
        rr = j.submit_confirmations([dict(good, confirmed="yes")]); self.assertEqual(len(rr["annotation_review"]["confirmation_state"]["rejected_records"]), 1)
        rr = j.submit_confirmations([dict(good, confirmed=False)]); self.assertFalse(rr["can_proceed"])                      # declined = not confirmed
        self.assertEqual(rr["annotation_review"]["per_set"][0]["decisions"]["AL_corner"]["decision"], "BLOCKED_UNCONFIRMED_WARNING")

    def test_confirmation_is_invalidated_by_any_binding_change(self):
        j, sel, r = self.ready_job(shift=self.SHIFT); reqs = r["annotation_review"]["confirmation_requests"]; c0 = VM.Confirmation.from_request(reqs[0], "t"); c1 = VM.Confirmation.from_request(reqs[1], "t")
        for bad in (dataclasses.replace(c0, frame_fingerprint="f" * 64), dataclasses.replace(c0, annotation_digest="0" * 64), dataclasses.replace(c0, refined_line_digest="0" * 64), dataclasses.replace(c0, confirmed_codes=("WEAK_EDGE",)),
                    dataclasses.replace(c0, confirmed_codes=("LARGE_REFINEMENT_DISPLACEMENT", "WEAK_EDGE")), dataclasses.replace(c0, annotation_id="other"), dataclasses.replace(c0, frame_index=14)):
            rr = j.submit_confirmations([bad, c1]); self.assertFalse(rr["can_proceed"], bad)
        self.assertTrue(j.submit_confirmations([c0, c1])["can_proceed"])

    def test_resubmitted_changed_annotation_invalidates_prior_confirmation(self):
        j, sel, r = self.confirmed_job(); self.assertTrue(r["can_proceed"])
        anns = self.annotations(sel, shift=self.SHIFT); t = next(a for a in anns if a["line_name"] == "AL_corner" and a["set_index"] == 0); t["points"][0][1] += 1.0
        r2 = j.submit_annotations(anns); self.assertFalse(r2["can_proceed"]); self.assertEqual(j.confirmations, [])               # a new submission clears confirmations ...
        old = [VM.Confirmation.from_request(q, "t") for q in r["annotation_review"]["confirmation_requests"]] if r["annotation_review"]["confirmation_requests"] else []
        r3 = j.submit_confirmations([dataclasses.replace(VM.Confirmation.from_request(q, "t"), annotation_digest=VM.annotation_digest(self.annotations(sel, shift=self.SHIFT)[0])) for q in r2["annotation_review"]["confirmation_requests"]])
        self.assertFalse(r3["can_proceed"])                                                                                         # ... and a stale digest does not match the changed annotation

    def test_confirmation_cannot_override_blocking_failure(self):
        j, sel, r = self.ready_job(shift={"AL_corner": 25.0}); self.assertEqual(r["annotation_review"]["per_set"][0]["decisions"]["AL_corner"]["decision"], "BLOCKED_FAIL"); self.assertEqual(r["annotation_review"]["confirmation_requests"], [])
        forged = VM.Confirmation("annotation", 0, "AL_corner", "t:0:AL_corner", sel["selected_frames"][0]["views"]["AL"]["frame_index"], sel["selected_frames"][0]["views"]["AL"]["frame_fingerprint"], "x", "y", (), True, "attacker")
        rr = j.submit_confirmations([forged]); self.assertFalse(rr["can_proceed"]); d = rr["annotation_review"]["per_set"][0]["decisions"]["AL_corner"]; self.assertEqual(d["decision"], "BLOCKED_FAIL"); self.assertIn("confirmation_ignored", d)
        self.assertIsNone(j.measure()["dimensions"])

    def test_backend_never_auto_confirms(self):
        j, sel, r = self.ready_job(shift=self.SHIFT); self.assertEqual(j.confirmations, []); self.assertFalse(j.measure()["can_proceed"]); self.assertEqual(j.confirmations, [])

    def test_failed_optional_ceilS_does_not_block(self):
        j, sel, r = self.ready_job(ceils=True)
        self.assertTrue(r["can_proceed"], r["failures"]); self.assertIn("OPTIONAL_LINE_NOT_USED", [w["code"] for w in r["warnings"]]); self.assertEqual(r["annotation_review"]["per_set"][0]["decisions"]["PL_ceilS"]["decision"], "BLOCKED_FAIL")
        m = j.measure(); self.assertEqual(m["stage"], "measured"); self.assertEqual(j.confirmations, [])

    def test_optional_warning_does_not_demand_confirmation(self):
        gate_in = mock.Mock(); j, sel, r = self.ready_job(ceils=True); self.assertEqual(r["annotation_review"]["confirmation_state"]["pending"], 0)

    def test_not_evaluated_is_preserved_and_blocks_as_annotation_status(self):
        j, sel, r = self.ready_job(); ev = j._evaluate()[0]; ev["entries"][0]["status"] = V.Status.NOT_EVALUATED
        g = VM.gate_set(ev, []); name = ev["entries"][0]["line_name"]; self.assertEqual(g["decisions"][name]["decision"], "BLOCKED_NOT_EVALUATED"); self.assertFalse(g["ok"])
        ok = VM.gate_set(j._evaluate()[1], [])["decisions"]["AL_corner"]; self.assertEqual(ok["checks_not_evaluated"], ["stereo_consistency"]); self.assertEqual(ok["stage_d_status"], "PASS")     # a not-evaluated CHECK is preserved, not a pass


class MeasurementFailuresAndChecks(Base):
    def run_with(self, **mk):
        n = mk.pop("n_sets", 2)
        with Mocks(n, **mk):
            with VM.VisitorMeasurementJob() as j:
                sel = self.to_frames(j, n); self.assertTrue(j.submit_annotations(self.annotations(sel))["can_proceed"]); return j.measure()

    def test_pose_degenerate(self):
        r = self.run_with(degenerate=True); self.assertFalse(r["can_proceed"]); self.assertEqual(fcodes(r), ["POSE_DEGENERATE"]); self.assertIsNone(r["dimensions"]); self.assertEqual(r["overall_status"], "FAIL")

    def test_pose_instability_is_a_warning_with_three_sets(self):
        r = self.run_with(n_sets=3, yaws=[6.0, 6.0, 6.1, 5.9]); st = r["pose_stability"]; self.assertEqual(st["status"], "WARNING"); self.assertGreater(st["yaw_range_deg"], 0.03); self.assertIn("confidence interval", st["note"])
        self.assertTrue(r["can_proceed"]); self.assertIsNotNone(r["dimensions"]); self.assertEqual(r["overall_status"], "WARNING"); self.assertTrue(any(w["code"] == "CHECK_POSE_STABILITY" for w in r["warnings"]))
        self.assertAlmostEqual(r["pose_diagnostics"]["yaw_deg"], 6.0)                                                                # the reported pose is the all-sets pose, never replaced

    def test_stable_pose_passes_and_two_sets_not_evaluated(self):
        r = self.run_with(n_sets=3, yaws=[6.0, 6.0, 6.0, 6.0]); self.assertEqual(r["pose_stability"]["status"], "PASS")
        r = self.run_with(n_sets=2); self.assertEqual(r["pose_stability"]["status"], "NOT_EVALUATED"); self.assertTrue(r["can_proceed"])
        c = {x["name"]: x for x in r["quality_checks"]}; self.assertEqual(c["pose_stability"]["status"], "NOT_EVALUATED"); self.assertFalse(c["pose_stability"]["required"]); self.assertEqual(r["overall_status"], "WARNING")

    def test_triangulation_floor_registration_wall_failures(self):
        for code in ("TRIANGULATION_FAILED", "FLOOR_PLANE_FAILED", "REGISTRATION_FAILED", "WALL_RECONSTRUCTION_FAILED", "GEOMETRY_FAILED"):
            r = self.run_with(measure_side=lambda p5, ms, ls, code=code: [dict(ok=False, code=code, error="synthetic", width_mm=float("nan"), height_mm=float("nan"))] * len(ms))
            self.assertFalse(r["can_proceed"], code); self.assertIn(code, fcodes(r)); self.assertIsNone(r["dimensions"]); self.assertEqual(r["overall_status"], "FAIL")

    def test_quality_check_failures_withhold_dimensions(self):
        cases = {"registration_inlier_count": dict(registration=dict(rms_mm=3.0, n_inliers=5, n_pairs=6)), "registration_inlier_fraction": dict(registration=dict(rms_mm=3.0, n_inliers=60, n_pairs=145)),
                 "registration_rms": dict(registration=dict(rms_mm=60.0, n_inliers=140, n_pairs=145)), "triangulation_depth_validity": dict(geometry=geom(gated_fraction=0.1, n_gated_points=900)),
                 "floor_plane_plausibility": dict(geometry=geom(floor_plane_rms_mm=80.0)), "boundary_conditioning": dict(vertical_error_deg=(30.0, 1.0)), "registration_spatial_coverage": dict(coverage=dict(n_pairs=10, bbox_fraction=0.001, cells_occupied=1, cells_total=9))}
        for name, over in cases.items():
            r = self.run_with(rows_fn=lambda n, over=over: rows(n, **over)); self.assertFalse(r["can_proceed"], name); self.assertIsNone(r["dimensions"], name); self.assertIn(f"CHECK_{name.upper()}", fcodes(r), name)
            self.assertIn("unaccepted_estimate_for_diagnostics_only", r)

    def test_required_check_not_evaluated_blocks_with_reason(self):
        r = self.run_with(rows_fn=lambda n: rows(n, geometry=geom(floor_plane_rms_mm=None))); c = {x["name"]: x for x in r["quality_checks"]}
        self.assertEqual(c["floor_plane_plausibility"]["status"], "NOT_EVALUATED"); self.assertFalse(r["can_proceed"]); self.assertIsNone(r["dimensions"]); self.assertIn("floor", r["failures"][0]["message"])

    def test_warning_checks_do_not_block_but_are_reported(self):
        r = self.run_with(rows_fn=lambda n: rows(n, registration=dict(rms_mm=20.0, n_inliers=30, n_pairs=36), coverage=dict(n_pairs=30, bbox_fraction=0.1, cells_occupied=2, cells_total=9), geometry=geom(floor_tilt_deg=60.0)))
        self.assertTrue(r["can_proceed"]); self.assertIsNotNone(r["dimensions"]); self.assertEqual(r["overall_status"], "WARNING"); names = {w["code"] for w in r["warnings"]}
        self.assertTrue({"CHECK_REGISTRATION_RMS", "CHECK_REGISTRATION_INLIER_COUNT", "CHECK_REGISTRATION_SPATIAL_COVERAGE", "CHECK_FLOOR_PLANE_PLAUSIBILITY"} <= names)

    def test_invalid_dimensions(self):
        for w, h in ((float("nan"), 2350.0), (-5.0, 2350.0), (2850.0, 0.0), (1e9, 2350.0)):
            r = self.run_with(rows_fn=lambda n, w=w, h=h: rows(n, w=w, h=h)); self.assertFalse(r["can_proceed"], (w, h)); self.assertIsNone(r["dimensions"]); self.assertIn("CHECK_DIMENSION_VALIDITY", fcodes(r))

    def test_stereo_consistency_failure_blocks_and_not_evaluated_is_preserved(self):
        r = self.run_with(stereo=V.Status.FAIL); self.assertFalse(r["can_proceed"]); self.assertIn("CHECK_STEREO_CONSISTENCY_POST_POSE", fcodes(r))
        r = self.run_with(stereo=V.Status.NOT_EVALUATED); c = {x["name"]: x for x in r["quality_checks"]}; self.assertEqual(c["stereo_consistency_post_pose"]["status"], "NOT_EVALUATED"); self.assertTrue(r["can_proceed"])

    def test_floor_row_threshold_scales_with_height_and_is_phase1_at_3840(self):
        self.assertEqual(VM.floor_row_threshold(3840), 2600.0); self.assertAlmostEqual(VM.floor_row_threshold(1920), 1300.0)
        r = self.run_with(); self.assertEqual({c["name"]: c for c in r["quality_checks"]}["floor_region_assumption"]["status"], "WARNING")        # 300 px high frames: scaled, unvalidated

    def test_every_failure_response_is_structured_and_serialisable(self):
        j = self.job(); outs = [j.validate_videos(self.paths), j.validate_calibration({}), j.validate_calibration(calibration(baseline=-1)), j.measure(), j.select_frames(1), j.submit_annotations([])]
        for r in outs:
            json.dumps(r)
            for k in ("schema", "step", "stage", "overall_status", "can_proceed", "required_next_action", "failures", "warnings", "accuracy_note", "runtime_s"): self.assertIn(k, r, (r["step"], k))
            self.assertEqual(r["overall_status"], "FAIL"); self.assertFalse(r["can_proceed"]); self.assertTrue(r["failures"])
            for f in r["failures"]: self.assertTrue({"stage", "code", "message"} <= set(f))
            self.assertIsNone(r.get("dimensions"))


class LimitsAndResources(Base):
    def test_time_and_memory_limits_are_checked(self):
        j = self.job(limits=VM.JobLimits(max_processing_s=0.0)); j.validate_calibration(calibration()); r = j.validate_videos(self.paths); self.assertEqual(fcodes(r), ["TIME_LIMIT_EXCEEDED"]); self.assertIn("cooperative", r["limits_note"])
        j = self.job(limits=VM.JobLimits(max_working_memory_mb=1.0)); j.validate_calibration(calibration()); self.assertEqual(fcodes(j.validate_videos(self.paths)), ["MEMORY_LIMIT_EXCEEDED"])
        j = self.job(limits=VM.JobLimits(max_working_memory_mb=0.001)); j.validate_calibration(calibration()); j.stage = "videos_validated"; j._vr = mock.Mock()
        self.assertEqual(fcodes(j.select_frames(2)), ["MEMORY_ESTIMATE_EXCEEDED"])

    def test_max_frame_sets_and_concurrent_job_limit(self):
        j = self.job(limits=VM.JobLimits(max_frame_sets=2)); j.validate_calibration(calibration()); j.validate_videos(self.paths); self.assertEqual(fcodes(j.select_frames(3)), ["INVALID_FRAME_SET_COUNT"])
        base = VM.VisitorMeasurementJob.active_jobs(); lim = VM.JobLimits(max_concurrent_jobs=base + 1); a = VM.VisitorMeasurementJob(limits=lim)
        try:
            with self.assertRaises(VM.JobLimitExceeded): VM.VisitorMeasurementJob(limits=lim)
        finally: a.close()
        self.assertEqual(VM.VisitorMeasurementJob.active_jobs(), base)
        b = VM.VisitorMeasurementJob(limits=lim); b.close(); b.close(); self.assertEqual(VM.VisitorMeasurementJob.active_jobs(), base)       # close is idempotent

    def test_uploaded_bytes_are_staged_privately_and_paths_never_modified(self):
        d = Path(self.tmp.name); before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.paths.values()}; data = {k: Path(v).read_bytes() for k, v in self.paths.items()}
        j = self.job(); j.validate_calibration(calibration()); r = j.validate_videos(data); self.assertTrue(r["can_proceed"], r["failures"])
        staged = sorted(Path(j.ws.path).glob("video_*.mp4")); self.assertEqual(len(staged), 4); self.assertTrue(all(str(p.parent) == str(j.ws.path) for p in staged)); self.assertTrue(all(Path(p).name.startswith("video_") for p in j.video_paths.values()))
        j2 = self.job(); j2.validate_calibration(calibration()); j2.validate_videos(self.paths)
        self.assertEqual({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in self.paths.values()}, before)

    def test_workspace_cleanup_after_success_failure_and_exception(self):
        with Mocks(2):
            with VM.VisitorMeasurementJob() as j: self.to_frames(j, 2); ws = Path(j.ws.path); self.assertTrue(ws.exists()); self.assertTrue(any((ws / "frames").glob("*.png")))
            self.assertFalse(ws.exists())
        with VM.VisitorMeasurementJob() as j: j.validate_calibration({}); ws = Path(j.ws.path)
        self.assertFalse(ws.exists())
        with self.assertRaises(RuntimeError):
            with VM.VisitorMeasurementJob() as j: ws = Path(j.ws.path); raise RuntimeError("boom")
        self.assertFalse(ws.exists())
        foreign = Path(self.tmp.name) / "foreign.txt"; foreign.write_text("x")
        with VM.VisitorMeasurementJob() as j: j.validate_calibration(calibration()); j.validate_videos(self.paths)
        self.assertTrue(foreign.exists()); self.assertTrue(all(Path(p).exists() for p in self.paths.values()))                                    # only the job's own directory is removed

    def test_concurrent_jobs_have_isolated_workspaces(self):
        with Mocks(2):
            a, b = self.job(), self.job(); sa, sb = self.to_frames(a, 2), self.to_frames(b, 2)
            self.assertNotEqual(a.ws.path, b.ws.path); fa = {v["preview_png"] for f in sa["selected_frames"] for v in f["views"].values()}; fb = {v["preview_png"] for f in sb["selected_frames"] for v in f["views"].values()}
            self.assertFalse(fa & fb); self.assertTrue(all(Path(p).parent.parent == Path(a.ws.path) for p in fa))
            ws_a = Path(a.ws.path); a.close(); self.assertFalse(ws_a.exists()); self.assertTrue(all(Path(p).exists() for p in fb))                  # closing one never touches the other
            self.assertTrue(b.submit_annotations(self.annotations(sb))["can_proceed"])

    def test_threaded_jobs_do_not_collide(self):
        out = []; lock = threading.Lock()
        def work():
            with VM.VisitorMeasurementJob() as j:
                j.validate_calibration(calibration()); j.validate_videos(self.paths); st = Path(j.ws.path)
                with lock: out.append(str(st))
        ts = [threading.Thread(target=work) for _ in range(3)]; [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(set(out)), 3); self.assertTrue(all(not Path(p).exists() for p in out))


class IndependenceFromTheOriginalExperiment(unittest.TestCase):
    SRC = (MODULE / "room_measurement" / "visitor_measurement.py").read_text()

    def test_import_loads_no_experiment_or_ground_truth_modules(self):
        code = ("import sys; sys.path.insert(0, %r); import room_measurement.visitor_measurement; "
                "bad=[m for m in sys.modules if any(x in m for x in ('stage_c_integration','stage_e1','room_measurement.validation','single_view','stage_c_shift'))]; print(bad)") % str(MODULE)
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(ROOT)); self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_static_scan_for_original_room_and_ground_truth_references(self):
        tree = ast.parse(self.SRC); names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for bad in ("GROUND_TRUTH", "DEFAULT_CALIBRATION_PATH", "DEFAULT_MANIFEST", "DEFAULT_ANNOTATIONS", "DEFAULT_VIDEO_DIR", "load_calibration", "load_manifest", "load_annotations", "reference_consensus", "index_table_from_selection"): self.assertNotIn(bad, names, bad)
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertFalse({i for i in imported if i and any(x in i for x in ("stage_c_integration", "stage_e1", "validation"))} - {"visitor_video_validation"})
        for lit in ("2879", "2380", "2900", "2400", "estimation_result", "wall_boundary_annotations", "frame_manifest", "room_video_recipe"): self.assertNotIn(lit, self.SRC, lit)

    def test_workflow_never_reads_the_original_room_files(self):
        boom = mock.Mock(side_effect=AssertionError("original-room data was read"))
        with mock.patch.object(C, "load_calibration", boom), mock.patch.object(L, "load_annotations", boom), mock.patch.object(Fr, "load_manifest", boom), mock.patch.object(Fr, "load_frame_sets", boom), \
                mock.patch.object(Fr, "verify_videos", boom), Mocks(2):
            with tempfile.TemporaryDirectory() as td:
                p = {}
                for f in VM.VIDEO_FIELDS: write_video(Path(td) / f"{f}.mp4"); p[f] = str(Path(td) / f"{f}.mp4")
                helper = Base(); out = VM.run_workflow(calibration(), p, 2, lambda sel, job: helper.annotations(sel)); self.assertEqual(out["final"]["stage"], "measured", out["final"]["failures"])
        boom.assert_not_called()

    def test_no_hard_coded_room_geometry_in_function_defaults(self):
        self.assertEqual(VM.floor_row_threshold(VM.PHASE1_FRAME_HEIGHT), VM.PHASE1_FLOOR_ROW)
        self.assertFalse(hasattr(VM, "GROUND_TRUTH"))


class SavedRegressionResults(unittest.TestCase):
    OUT = MODULE / "outputs" / "room_measurement" / "stage_e2_visitor_backend"

    def test_saved_results_if_present(self):
        p = self.OUT / "results.json"
        if not p.exists(): self.skipTest("regression not run")
        d = json.loads(p.read_text()); self.assertTrue(d["protected_inputs_unchanged"]); self.assertFalse(d["ground_truth_used_by_backend"]); self.assertFalse(d["validates_arbitrary_visitor_rooms"])
        for n, r in d["configurations"].items():
            f = r["final"]; self.assertEqual(f["can_proceed"], f["dimensions"] is not None); self.assertFalse(f["ground_truth_used"]); self.assertTrue(r["workspace_removed_after_close"])
            if f["dimensions"] is None: self.assertEqual(f["overall_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
