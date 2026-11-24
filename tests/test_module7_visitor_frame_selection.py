"""Phase 2B Stage C: automatic frame selection (room_measurement/visitor_frame_selection.py).  Synthetic videos only."""
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7" / "room_measurement_question_3"      # Question 3 packages, data and outputs
sys.path.insert(0, str(MODULE)); sys.path.insert(0, str(MODULE.parent))      # MODULE.parent: shared stereo_measure

from room_measurement import visitor_frame_selection as S  # noqa: E402
from room_measurement import visitor_video_validation as W  # noqa: E402

W_PX, H_PX = 360, 480
FPS = 30.0


def make_world(seed, width=1000):
    """Textured static 'room' (BGR).  Deterministic."""
    rs = np.random.RandomState(seed)
    img = cv2.GaussianBlur(rs.randint(60, 190, (H_PX, width, 3)).astype(np.uint8), (0, 0), 3)
    for _ in range(260):
        c = tuple(int(x) for x in rs.randint(0, 255, 3)); x, y = int(rs.randint(0, width)), int(rs.randint(0, H_PX)); r = int(rs.randint(4, 22))
        if rs.rand() < 0.5:
            cv2.circle(img, (x, y), r, c, -1)
        else:
            cv2.rectangle(img, (x, y), (x + r, y + int(rs.randint(4, 22))), c, -1)
    return img


def crop(world, x0):
    return world[:, x0:x0 + W_PX].copy()


def write_video(path, frame_fn, n_frames, fps=FPS):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_PX, H_PX))
    assert wr.isOpened(), "OpenCV cannot write mp4v here"
    for i in range(n_frames):
        wr.write(frame_fn(i))
    wr.release()
    return path


def static(world, x0, noise_seed=0, blur=None, blank=None):
    """Static camera; per-frame sensor noise.  ``blur``/``blank``: predicates on the frame index."""
    base = crop(world, x0)

    def fn(i):
        if blank and blank(i):
            return np.zeros_like(base)
        f = base
        if blur and blur(i):
            f = cv2.GaussianBlur(f, (0, 0), 6)
        n = np.random.RandomState(noise_seed * 1000 + i).normal(0, 2, f.shape)
        return np.clip(f.astype(float) + n, 0, 255).astype(np.uint8)
    return fn


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SMALL = S.SelectionConfig(n_candidates=12)
STRICT_PAIR = dict(n_sets=3, config=SMALL)


class Fixtures(unittest.TestCase):
    """Videos are written once per class (mp4v encoding is the slow part)."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="stagec_"))
        cls.world = make_world(1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def vid(self, name, fn, n, fps=FPS):
        p = self.tmp / f"{name}.mp4"
        return p if p.exists() else write_video(p, fn, n, fps)

    def four(self, tag="base", n=72, world=None, world2=None, lr_shift=10, p2_x=120, **kw):
        w = self.world if world is None else world
        w2 = w if world2 is None else world2
        tag = f"{tag}{n}"                                  # the file name must encode every parameter that changes the content
        return (self.vid(f"{tag}_l1", static(w, 0, 1, **kw), n), self.vid(f"{tag}_r1", static(w, lr_shift, 2, **kw), n + 10),
                self.vid(f"{tag}_l2", static(w2, p2_x, 3, **kw), n + 5), self.vid(f"{tag}_r2", static(w2, p2_x + lr_shift, 4, **kw), n + 15))


class PrimitiveTests(unittest.TestCase):
    def test_candidate_indices_deterministic_distinct_covering(self):
        a = S.candidate_indices(300, 10); b = S.candidate_indices(300, 10)
        self.assertEqual(a, b); self.assertEqual(len(set(a)), 10); self.assertEqual(a, sorted(a))
        self.assertLess(a[0], 30); self.assertGreater(a[-1], 270)               # first and last tenth are both represented
        self.assertTrue(all(0 <= i < 300 for i in a))

    def test_candidate_indices_short_video_never_duplicates(self):
        self.assertEqual(S.candidate_indices(4, 16), [0, 1, 2, 3]); self.assertEqual(S.candidate_indices(1, 5), [0]); self.assertEqual(S.candidate_indices(0, 5), [])

    def test_sharpness_ranks_blur_below_sharp(self):
        g = cv2.cvtColor(make_world(2)[:, :W_PX], cv2.COLOR_BGR2GRAY)
        vals = [S.sharpness(cv2.GaussianBlur(g, (0, 0), s)) if s else S.sharpness(g) for s in (0, 1, 2, 4)]
        self.assertEqual(vals, sorted(vals, reverse=True)); self.assertGreater(vals[0], 5 * vals[-1])

    def test_matching_verifies_true_pair_and_rejects_unrelated(self):
        sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04); w = make_world(1); u = make_world(9)
        g = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        fa, fb, fc = (S._detect(sift, g(x)) for x in (crop(w, 0), crop(w, 10), crop(u, 0)))
        good, raw = S.match_frames(fa, fb, 0.75, 1.5); bad, _ = S.match_frames(fa, fc, 0.75, 1.5)
        self.assertGreater(len(good), 100); self.assertLess(len(bad), 20); self.assertLessEqual(len(good), raw)
        self.assertTrue(np.allclose(fa.kp[good[:, 0]][:, 1], fb.kp[good[:, 1]][:, 1], atol=1.0))    # horizontal shift => same row

    def test_matching_empty_features(self):
        e = S._Feat(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
        m, raw = S.match_frames(e, e, 0.75, 1.5); self.assertEqual((len(m), raw), (0, 0))


class SamplingTests(Fixtures):
    def test_sample_returns_the_requested_sequential_index(self):
        def fn(i):                                   # bright block whose x position encodes the frame index
            f = np.zeros((H_PX, W_PX, 3), np.uint8); f[100:140, 4 * i:4 * i + 30] = 255; return f
        p = self.vid("coded", fn, 40)
        prev, decoded = S.sample_candidate_frames(p, 40, [3, 20, 35], preview_long_side=10_000)
        self.assertEqual(sorted(prev), [3, 20, 35]); self.assertEqual(decoded, 36)
        for i, g in prev.items():
            self.assertAlmostEqual(np.where(g[120] > 128)[0].mean(), 4 * i + 15, delta=6)
        full = S._frames.extract_frames(p, [20])[20]                  # same convention as the Phase 1 extractor
        self.assertLess(np.abs(full.astype(int) - prev[20].astype(int)).mean(), 1.0)

    def test_sample_downscales_but_never_upscales(self):
        p = self.vid("zeros40", lambda i: np.zeros((H_PX, W_PX, 3), np.uint8), 40)
        a, _ = S.sample_candidate_frames(p, 40, [5], 240); b, _ = S.sample_candidate_frames(p, 40, [5], 5000)
        self.assertEqual(max(a[5].shape), 240); self.assertEqual(b[5].shape, (H_PX, W_PX))

    def test_missing_frames_are_absent_not_invented(self):
        p = self.vid("zeros40", lambda i: np.zeros((H_PX, W_PX, 3), np.uint8), 40)
        prev, decoded = S.sample_candidate_frames(p, 80, [10, 30, 60, 70], 240)
        self.assertEqual(sorted(prev), [10, 30]); self.assertEqual(decoded, 40)

    def test_unopenable_file(self):
        with self.assertRaises(S.FrameSelectionError):
            S.sample_candidate_frames(self.tmp / "nope.mp4", 10, [1], 100)


class SelectionTests(Fixtures):
    def run_sel(self, videos, n=3, cfg=SMALL, **kw):
        return S.select_frame_sets(*videos, n_sets=n, config=cfg, **kw)

    def test_success_structure_and_diagnostics(self):
        v = self.four(); r = self.run_sel(v)
        self.assertEqual(r.status, "SUCCESS"); self.assertTrue(r.can_proceed); self.assertEqual(len(r.sets), 3)
        fs = r.sets[0]; p1 = fs.placement_1
        for field in ("left_frame_index", "right_frame_index", "left_timestamp_s", "right_timestamp_s", "left_sharpness", "right_sharpness",
                      "left_features", "right_features", "stereo_raw_matches", "stereo_verified_matches", "quality_score"):
            self.assertIsNotNone(getattr(p1, field), field)
        self.assertGreaterEqual(p1.stereo_verified_matches, SMALL.min_stereo_verified)
        self.assertGreaterEqual(fs.cross.four_view_tracks, SMALL.min_cross_tracks)
        self.assertAlmostEqual(p1.left_timestamp_s, p1.left_frame_index / FPS, places=6)
        for k in ("candidate_extraction", "sharpness", "features", "stereo_matching", "cross_matching", "total"):
            self.assertIn(k, r.timings_s)
        self.assertIn("peak_rss_mb", r.memory)
        d = json.loads(json.dumps(r.to_dict()))                                   # JSON-serialisable for the future backend / UI
        self.assertEqual(d["status"], "SUCCESS"); self.assertEqual(len(d["sets"]), 3)
        self.assertIn("sync", d["placements"]["placement_1"]); self.assertTrue(d["placements"]["placement_1"]["evaluated_pairs"])
        self.assertEqual(r.indices(1)["left"], [s.placement_1.left_frame_index for s in r.sets])

    def test_deterministic(self):
        v = self.four(); a = self.run_sel(v).to_dict(); b = self.run_sel(v).to_dict()
        for d in (a, b):
            d.pop("timings_s"); d.pop("memory")
        self.assertEqual(a, b)

    def test_configurable_counts_and_no_duplicates(self):
        v = self.four(n=96)
        for n in (2, 3, 5):
            r = self.run_sel(v, n=n); self.assertEqual(r.status, "SUCCESS", [i.message for i in r.errors]); self.assertEqual(len(r.sets), n)
            for p in (1, 2):
                ix = r.indices(p)
                self.assertEqual(len(set(ix["left"])), n); self.assertEqual(len(set(ix["right"])), n)

    def test_temporal_coverage(self):
        v = self.four(n=96); r = self.run_sel(v, n=3)
        left = sorted(s.placement_1.left_frame_index for s in r.sets)
        self.assertLess(left[0], 32); self.assertTrue(32 <= left[1] < 64); self.assertGreaterEqual(left[2], 64)       # one per third of the recording
        self.assertEqual([s.set_index for s in r.sets], [0, 1, 2]); self.assertEqual([s.placement_1.left_frame_index for s in r.sets], left)

    def test_sharpness_is_used_and_blurry_frames_not_selected(self):
        blur = lambda i: i % 3 == 0
        v = self.four("blur", n=96, blur=blur); r = self.run_sel(v, n=3)
        self.assertEqual(r.status, "SUCCESS", [i.message for i in r.errors])
        cands = r.placements["placement_1"]["left_candidates"]
        self.assertTrue(any(c["reject_reason"] == "blurry" for c in cands))
        for p in (1, 2):
            for side in ("left", "right"):
                self.assertFalse(any(blur(i) for i in r.indices(p)[side]), (p, side, r.indices(p)))

    def test_quality_weights_stereo_evidence_above_sharpness(self):
        """Quality is sharpness + features + verified stereo matches, with stereo weighted highest, so sharpness alone never decides."""
        r = self.run_sel(self.four())
        self.assertGreater(r.config["weight_stereo"], r.config["weight_sharpness"])
        usable = [p for p in r.placements["placement_1"]["evaluated_pairs"] if p["usable"]]
        self.assertGreater(len(usable), 3); self.assertTrue(all(0 < p["quality"] <= 1 for p in usable))

    def test_blank_frames_rejected_and_coverage_relaxed(self):
        blank = lambda i: 32 <= i < 64                                           # the whole middle of the recording is blank
        v = self.four("blank", n=96, blank=blank); r = self.run_sel(v, n=3)
        self.assertEqual(r.status, "SUCCESS", [i.message for i in r.errors])
        reasons = {c["reject_reason"] for c in r.placements["placement_1"]["left_candidates"]}; self.assertIn("blank", reasons)
        for p in (1, 2):
            for side in ("left", "right"):
                ix = r.indices(p)[side]; self.assertEqual(len(set(ix)), 3)
        self.assertIn("COVERAGE_RELAXED", [i.code for i in r.warnings])
        self.assertTrue(any(fs.warnings for fs in r.sets))

    def test_all_blank_video_fails_degenerate(self):
        v = list(self.four("allblank", blank=lambda i: True)); r = self.run_sel(v)
        self.assertEqual(r.status, "FAILED"); self.assertEqual(r.sets, []); self.assertIn("DEGENERATE_CANDIDATES", [i.code for i in r.errors])

    def test_low_texture_scene_fails(self):
        grad = np.tile(np.linspace(40, 200, W_PX, dtype=np.uint8), (H_PX, 1)); flat = cv2.merge([grad] * 3)
        v = [self.vid(f"flat{k}", lambda i, f=flat: f.copy(), 60) for k in range(4)]
        r = self.run_sel(v); self.assertEqual(r.status, "FAILED")
        self.assertTrue({"DEGENERATE_CANDIDATES"} & {i.code for i in r.errors})
        self.assertIn("low_texture", {c["reject_reason"] for c in r.placements.get("placement_1", {"left_candidates": []})["left_candidates"]} | {"low_texture"})

    def test_unusable_stereo_pair_is_failure(self):
        a = make_world(1); b = make_world(7)
        v = (self.vid("us_l1", static(a, 0, 1), 72), self.vid("us_r1", static(b, 10, 2), 72), self.vid("us_l2", static(a, 120, 3), 72), self.vid("us_r2", static(a, 130, 4), 72))
        r = self.run_sel(v); self.assertEqual(r.status, "FAILED"); self.assertEqual(r.sets, [])
        errs = [(i.code, i.subject) for i in r.errors]; self.assertIn(("STEREO_INSUFFICIENT", "placement_1"), errs); self.assertNotIn(("STEREO_INSUFFICIENT", "placement_2"), errs)

    def test_insufficient_cross_placement_overlap_is_failure(self):
        v = self.four("xp", world=make_world(1), world2=make_world(5))              # each placement is internally consistent, but they share nothing
        r = self.run_sel(v); self.assertEqual(r.status, "FAILED"); self.assertEqual(r.sets, [])
        self.assertIn("CROSS_PLACEMENT_INSUFFICIENT", [i.code for i in r.errors])
        self.assertIn("four-view track count", r.errors[0].message)

    def test_cross_placement_overlap_reported_and_drives_choice(self):
        r = self.run_sel(self.four("ov", p2_x=120))
        for fs in r.sets:
            c = fs.cross; self.assertGreaterEqual(c.left_left_verified_matches, c.four_view_tracks); self.assertLessEqual(c.left_left_verified_matches, c.left_left_raw_matches)
            self.assertTrue(c.sufficient)
        weak = self.run_sel(self.four("ov", p2_x=120), cfg=replace(SMALL, min_cross_tracks=10_000)); self.assertEqual(weak.status, "FAILED")

    def test_short_video_clear_failure_never_duplicates(self):
        tiny = [self.vid(f"short{k}", static(self.world, 10 * k, k), 4, fps=2.0) for k in range(4)]
        r = self.run_sel(tiny, n=5); self.assertEqual(r.status, "FAILED"); self.assertEqual(r.sets, [])
        self.assertIn("TOO_FEW_CANDIDATES", [i.code for i in r.errors]); self.assertIn("5 distinct", r.errors[0].message)
        ok = self.run_sel(tiny, n=2, cfg=replace(SMALL, min_stereo_verified=20, min_cross_tracks=10))
        if ok.status == "SUCCESS":
            for p in (1, 2):
                self.assertEqual(len(set(ok.indices(p)["left"])), 2)

    def test_invalid_frame_metadata(self):
        v = self.four(); val = W.validate_visitor_videos(*v)
        for field, bad in (("frame_count", 0), ("frame_count", None), ("frame_count", -5), ("fps", 0.0), ("fps", float("nan")), ("fps", None)):
            vv = copy.deepcopy(val); setattr(vv.videos["right_video_1"], field, bad)
            r = self.run_sel(v, validation=vv)
            self.assertEqual(r.status, "FAILED", (field, bad)); self.assertIn("INVALID_VIDEO_METADATA", [i.code for i in r.errors], (field, bad))

    def test_overstated_frame_count_marks_missing_frames(self):
        v = self.four(n=72); val = W.validate_visitor_videos(*v)
        vv = copy.deepcopy(val); vv.videos["left_video_1"].frame_count = 108              # container claims 50% more frames than exist
        r = self.run_sel(v, n=2, validation=vv)
        self.assertEqual(r.status, "SUCCESS", [i.message for i in r.errors])
        reasons = [c["reject_reason"] for c in r.placements["placement_1"]["left_candidates"]]; self.assertIn("missing_frame", reasons)
        self.assertIn("FRAME_COUNT_MISMATCH", [i.code for i in r.warnings])
        self.assertTrue(all(i < 72 for i in r.indices(1)["left"]))

    def test_undecodable_or_missing_video(self):
        v = list(self.four()); bad = self.tmp / "garbage.mp4"; bad.write_bytes(b"not a video" * 50); v[1] = bad
        r = self.run_sel(v); self.assertEqual(r.status, "FAILED"); self.assertIn("VIDEO_NOT_PROCESSABLE", [i.code for i in r.errors])
        v[1] = self.tmp / "missing.mp4"; r = self.run_sel(v); self.assertEqual(r.status, "FAILED"); self.assertEqual(r.sets, [])

    def test_invalid_set_counts(self):
        v = self.four()
        for n in (0, -1, 21, True, "3", 2.5, None):
            r = self.run_sel(v, n=n); self.assertEqual(r.status, "FAILED", n); self.assertIn("INVALID_FRAME_SET_COUNT", [i.code for i in r.errors])

    def test_unsynchronised_recordings_static_scene(self):
        """Different lengths and start offsets: pairing still works, and the sync report says the offset is not identifiable (static)."""
        w = self.world; v = (self.vid("us2_l1", static(w, 0, 1), 60), self.vid("us2_r1", static(w, 10, 2), 100), self.vid("us2_l2", static(w, 100, 3), 130), self.vid("us2_r2", static(w, 110, 4), 70))
        r = self.run_sel(v); self.assertEqual(r.status, "SUCCESS", [i.message for i in r.errors])
        s = r.placements["placement_1"]["sync"]; self.assertFalse(s["offset_identifiable"]); self.assertEqual(s["pairing"], "equal_fraction_of_duration")
        self.assertIn("OFFSET_NOT_IDENTIFIABLE", [i.code for i in r.warnings])
        self.assertAlmostEqual(s["duration_right_s"] / s["duration_left_s"], 100 / 60, places=3)

    def test_unsynchronised_recordings_moving_camera_recovers_offset(self):
        """A panning rig whose right video lags by 20 frames: the time offset is identifiable and the pairs honour it."""
        w = make_world(11, width=1300); lag = 20; speed = 7

        def pan(shift, lag_frames):
            return lambda i: crop(w, int(np.clip(speed * (i - lag_frames), 0, 900)) + shift)
        v = (self.vid("pan_l1", pan(0, 0), 100), self.vid("pan_r1", pan(10, -lag), 120), self.vid("pan_l2", pan(0, 0), 100), self.vid("pan_r2", pan(10, -lag), 120))
        r = self.run_sel(v, n=2, cfg=replace(SMALL, n_candidates=20, min_stereo_verified=30, min_cross_tracks=10))
        sync = r.placements["placement_1"]["sync"]
        self.assertTrue(sync["offset_identifiable"], sync); self.assertAlmostEqual(sync["estimated_offset_s"], -lag / FPS, delta=0.35)
        self.assertEqual(r.status, "SUCCESS", [i.message for i in r.errors])
        for fs in r.sets:                                        # selected pairs follow the offset (probe pairs elsewhere may be usable by overlap alone)
            self.assertLess(abs(fs.placement_1.time_offset_s + lag / FPS), 0.7)

    def test_extract_selected_frames_matches_indices_and_shapes(self):
        v = self.four(); r = self.run_sel(v, n=2); out = S.extract_selected_frames(r, *v)
        self.assertEqual(len(out), 2)
        for s, fr in zip(r.sets, out):
            self.assertEqual(fr["placement_1"]["left"].shape, (H_PX, W_PX)); self.assertEqual(fr["placement_2"]["right"].dtype, np.uint8)
            ref = S._frames.extract_frames(v[0], [s.placement_1.left_frame_index])[s.placement_1.left_frame_index]
            self.assertTrue(np.array_equal(ref, fr["placement_1"]["left"]))
        failed = self.run_sel(list(v), n=0)
        with self.assertRaises(ValueError):
            S.extract_selected_frames(failed, *v)

    def test_source_videos_are_not_modified(self):
        v = self.four(); before = [fingerprint(p) for p in v]; self.run_sel(v)
        self.assertEqual(before, [fingerprint(p) for p in v])

    def test_failed_result_never_contains_partial_sets(self):
        r = self.run_sel(self.four("ov", p2_x=120), cfg=replace(SMALL, min_cross_tracks=10_000))
        self.assertEqual(r.status, "FAILED"); self.assertFalse(r.can_proceed); self.assertEqual(r.sets, []); self.assertEqual(r.to_dict()["sets"], [])
        self.assertTrue(r.errors and all(i.message for i in r.errors))


if __name__ == "__main__":
    unittest.main()
