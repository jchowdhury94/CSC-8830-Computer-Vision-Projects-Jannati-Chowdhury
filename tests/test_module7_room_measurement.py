"""Module 7 Part 3 Phase 1: stereo-assisted room measurement (Method J).  Synthetic rigs with known truth + a real-video regression."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))

from room_measurement import calibration as C, features as Ft, frames as Fr, geometry as G, measure as M, pose as P  # noqa: E402

BASE = 225.0
HAVE_VIDEOS = Fr.DEFAULT_VIDEO_DIR.exists() and all((Fr.DEFAULT_VIDEO_DIR / v["file"]).exists() for v in Fr.load_manifest()["videos"].values())

def synth_cal():
    K1 = np.array([[3178.0, 0, 1080.0], [0, 3178.0, 1909.0], [0, 0, 1.0]]); K2 = np.array([[2937.0, 0, 1086.0], [0, 2937.0, 1910.0], [0, 0, 1.0]])
    return C.StereoCalibration("synthetic", K1, np.zeros(5), K2, np.zeros(5), (2160, 3840), BASE)

TRUE_P = np.r_[np.radians([-2.7, 6.2, -1.3]), np.arccos(0.14), np.arctan2(0.03, -0.99)]

def project(K, X): return (X @ K.T)[:, :2] / (X @ K.T)[:, 2:3]

def synth_sets(cal, nsets=3, npts=500, noise=0.3, seed=0, R0_deg=30.0):
    """nsets frame sets of an ideal rig seeing a random 3D scene from two placements (view 2 = view 1 rotated about y)."""
    rng = np.random.default_rng(seed); R, T = G.pose_from_params(TRUE_P, BASE); R0 = cv2.Rodrigues(np.array([0, np.radians(R0_deg), 0]))[0]; t0 = np.array([300.0, 0, -200.0]); sets = []
    for s in range(nsets):
        X = np.c_[rng.uniform(-1500, 1500, npts), rng.uniform(-900, 640, npts), rng.uniform(2500, 6500, npts)]; X2 = X @ R0.T + t0
        def view(Xl): return project(cal.K_left, Xl) + rng.normal(0, noise, (npts, 2)), project(cal.K_right, Xl @ R.T + T) + rng.normal(0, noise, (npts, 2))
        a1, b1 = view(X); a2, b2 = view(X2); idx = np.arange(npts)[:, None].repeat(2, 1)
        sets.append(dict(A1=a1, B1=b1, M1=idx, A2=a2, B2=b2, M2=idx, MX=idx.copy()))
    return sets

class TestCalibration(unittest.TestCase):
    def test_default_artifact_loads_with_full_provenance(self):
        c = C.load_calibration()
        self.assertEqual(c.image_size, (2160, 3840)); self.assertEqual(c.baseline_mm, 225.0); self.assertEqual(c.K_left.shape, (3, 3)); self.assertEqual(c.D_right.shape, (5,))
        self.assertAlmostEqual(c.K_left[0, 0], 3178.846526387882, places=9); self.assertAlmostEqual(c.D_left[4], 3.029804166682109, places=12)
        self.assertEqual(len(c.provenance["source_pickle_sha256"]), 64); self.assertIn("Model B", c.provenance["not_the_old_photo_calibration"])

    def test_roundtrip_is_bit_exact_and_never_overwrites(self):
        c = C.load_calibration()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"; C.save_calibration(c, p); c2 = C.load_calibration(p)
            self.assertTrue(np.array_equal(c.K_left, c2.K_left) and np.array_equal(c.D_right, c2.D_right) and np.array_equal(c.K_right, c2.K_right))
            with self.assertRaises(FileExistsError): C.save_calibration(c, p)

    def test_rejects_malformed(self):
        c = synth_cal()
        with self.assertRaises(ValueError): C.validate(C.StereoCalibration("x", np.eye(2), np.zeros(5), c.K_right, np.zeros(5), (2160, 3840), 225.0))
        with self.assertRaises(ValueError): C.validate(C.StereoCalibration("x", c.K_left, np.zeros(4), c.K_right, np.zeros(5), (2160, 3840), 225.0))
        with self.assertRaises(ValueError): C.validate(C.StereoCalibration("x", c.K_left, np.zeros(5), c.K_right, np.zeros(5), (2160, 3840), -1.0))

    def test_legacy_extrinsics_are_never_used_and_old_calibration_untouched(self):
        for f in ("pose.py", "measure.py", "pipeline.py", "geometry.py", "features.py", "uncertainty.py"): self.assertNotIn("legacy_extrinsics", (MODULE / "room_measurement" / f).read_text(), f)
        self.assertTrue((MODULE / "calibration" / "model_b" / "stereo_calibration_30pairs.npz").exists())
        self.assertNotEqual(C.DEFAULT_CALIBRATION_PATH.parent.name, "model_b")

class TestFrameIndexing(unittest.TestCase):
    def test_manifest_table(self):
        m = Fr.load_manifest(); t = Fr.frame_index_table(m)
        self.assertEqual(len(t), 5); self.assertEqual(Fr.n_sets(m), 5); self.assertEqual(t[0], {"AL": 247, "PL": 292, "AR": 308, "PR": 488}); self.assertEqual(t[4]["AL"], 247)
        for k in Fr.VIEW_KEYS:
            v = m["videos"][k]; self.assertEqual(len(v["sha256"]), 64); self.assertTrue(all(0 <= i < v["frame_count"] for i in v["frame_indices_per_set"]))
        self.assertEqual(m["resolution_wh"], [2160, 3840])

    def test_manifest_hashes_match_wall_videos_when_present(self):
        if not HAVE_VIDEOS: self.skipTest("protected videos not on this machine")
        self.assertTrue(all(r["ok"] for r in Fr.verify_videos(Fr.load_manifest())))

    def test_missing_video_reported_not_raised(self):
        rows = Fr.verify_videos(Fr.load_manifest(), "/nonexistent"); self.assertTrue(all(not r["ok"] and r["actual_sha256"] is None for r in rows))

    def test_extract_frames_by_index_on_synthetic_video(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "v.avi"); w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            for i in range(10): w.write(np.full((48, 64, 3), 20 * i + 10, np.uint8))
            w.release(); out = Fr.extract_frames(p, [7, 2, 7])
            self.assertEqual(sorted(out), [2, 7]); self.assertAlmostEqual(float(out[2].mean()), 50, delta=6); self.assertAlmostEqual(float(out[7].mean()), 150, delta=6)
            with self.assertRaises(IndexError): Fr.extract_frames(p, [99])

class TestCoordinates(unittest.TestCase):
    def test_pose_param_roundtrip_and_fixed_baseline(self):
        R, T = G.pose_from_params(TRUE_P, BASE); self.assertAlmostEqual(float(np.linalg.norm(T)), BASE, places=9); self.assertTrue(np.allclose(R @ R.T, np.eye(3)))
        self.assertTrue(np.allclose(G.params_from_pose(R, T), TRUE_P, atol=1e-12))
    def test_yaw_offset(self):
        q = G.rotate_yaw(TRUE_P, 0.1, BASE); R, T = G.pose_from_params(q, BASE); self.assertAlmostEqual(float(np.linalg.norm(T)), BASE, places=9)
        self.assertAlmostEqual(G.yaw_deg(q) - G.yaw_deg(TRUE_P), 0.1, delta=0.02)
    def test_undistort_consistency(self):
        cal = C.load_calibration(); px = np.array([[500.0, 900.0], [1500.0, 3000.0]]); n = G.undistort_normalized(px, cal.K_left, cal.D_left); u = G.undistort_pixels(px, cal.K_left, cal.D_left)
        self.assertTrue(np.allclose((n @ cal.K_left[:2, :2].T) + cal.K_left[:2, 2], u, atol=1e-6))

class TestStereoGeometry(unittest.TestCase):
    def test_triangulation_epipolar_reprojection_noise_free(self):
        cal = synth_cal(); R, T = G.pose_from_params(TRUE_P, BASE); rng = np.random.default_rng(1); X = np.c_[rng.uniform(-1500, 1500, 50), rng.uniform(-900, 600, 50), rng.uniform(2500, 6500, 50)]
        a = project(cal.K_left, X); b = project(cal.K_right, X @ R.T + T); Xr = G.triangulate(cal, R, T, a, b)
        self.assertLess(np.abs(Xr - X).max(), 1e-3); self.assertLess(G.epipolar_distances(cal, R, T, a, b).max(), 1e-6)
        e1, e2 = G.reprojection_errors(cal, R, T, Xr, a, b); self.assertLess(max(e1.max(), e2.max()), 1e-6)
        self.assertLess(np.abs(G.sampson_residual(TRUE_P, G.undistort_normalized(a, cal.K_left, cal.D_left), G.undistort_normalized(b, cal.K_right, cal.D_right), BASE)).max(), 1e-6)
    def test_wrong_pose_gives_large_sampson(self):
        cal = synth_cal(); s = synth_sets(cal, 1, noise=0.0)[0]; n1 = G.undistort_normalized(s["A1"], cal.K_left, cal.D_left); n2 = G.undistort_normalized(s["B1"], cal.K_right, cal.D_right)
        self.assertGreater(np.median(np.abs(G.sampson_residual(TRUE_P + np.r_[np.radians([0, 0, 1.0]), 0, 0], n1, n2, BASE))), 1.0)
    def test_pose_recovery_from_synthetic_sets(self):
        cal = synth_cal(); sets = synth_sets(cal); D = P.build_dataset(cal, sets); pJ, pA, rep, _ = P.estimate_pose(D)
        RJ, TJ = G.pose_from_params(pJ, BASE); R0, T0 = G.pose_from_params(TRUE_P, BASE)
        ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ RJ) - 1) / 2, -1, 1))); tdir = np.degrees(np.arccos(np.clip(TJ @ T0 / BASE ** 2, -1, 1)))
        self.assertLess(ang, 0.2); self.assertLess(tdir, 0.6); self.assertFalse(rep["method_J"]["flags"]["degenerate"]); self.assertAlmostEqual(float(np.linalg.norm(TJ)), BASE, places=6)

class TestDegenerateAndFallback(unittest.TestCase):
    def test_good_data_not_degenerate_garbage_is_degenerate(self):
        cal = synth_cal(); D = P.build_dataset(cal, synth_sets(cal, 2)); p, info = P.fit_A(D); self.assertFalse(P.diagnose(D, p, info)["degenerate"])
        rng = np.random.default_rng(3); Dg = dict(D); Dg["n2"] = D["n2"][rng.permutation(len(D["n2"]))]            # destroy correspondences
        p, info = P.fit_A(Dg); self.assertTrue(P.diagnose(Dg, p, info)["degenerate"])
    def test_fallback_start_converges_to_truth(self):
        cal = synth_cal(); D = P.build_dataset(cal, synth_sets(cal, 3)); p, info = P.fit_A(D, p0=P.FALLBACK_START); self.assertFalse(P.diagnose(D, p, info)["degenerate"])
        R, T = G.pose_from_params(p, BASE); R0, _ = G.pose_from_params(TRUE_P, BASE); self.assertLess(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1))), 0.3)
    def test_guard_switches_to_fallback_when_first_start_is_degenerate(self):
        cal = synth_cal(); D = P.build_dataset(cal, synth_sets(cal, 3)); orig = P.init_pose
        try:
            P.init_pose = lambda *a, **k: G.params_from_pose(cv2.Rodrigues(np.array([0.0, 0.0, 2.6]))[0], np.array([0.3, 0.9, -0.3]))      # a deliberately wrong-basin start
            p, info, rec = P.fit_A_guarded(D)
        finally: P.init_pose = orig
        self.assertTrue(rec["first_attempt"]["degenerate"], rec); self.assertTrue(rec["fallback_used"]); self.assertFalse(rec["degenerate"])
        R, _ = G.pose_from_params(p, BASE); R0, _ = G.pose_from_params(TRUE_P, BASE); self.assertLess(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1))), 0.3)

class TestPlanesLinesRegistration(unittest.TestCase):
    def test_floor_plane_irls_with_outliers_and_row_selection(self):
        rng = np.random.default_rng(5); n_true = np.array([0.02, -0.999, 0.03]); n_true /= np.linalg.norm(n_true); d_true = 640.0
        u = rng.uniform(-2000, 2000, (400, 2)); basis = np.linalg.svd(n_true[None])[2][1:]
        X = u[:, :1] * basis[0] + u[:, 1:] * basis[1] - n_true * d_true; X += rng.normal(0, 2, X.shape) * n_true      # points on n.x + d = 0 (+ 2 mm noise)
        out = rng.uniform(-3000, 3000, (80, 3)); px = np.c_[np.zeros(480), np.full(480, 3000.0)]; Xall = np.vstack([X, out]); n, d = G.fit_floor_plane(Xall, px)
        self.assertGreater(abs(float(n @ n_true)), 0.9995); self.assertAlmostEqual(d, 640.0, delta=5.0); self.assertGreaterEqual(d, 0)
        n2, d2 = G.fit_floor_plane(Xall, px); self.assertTrue(np.array_equal(n, n2) and d == d2)                       # deterministic
        px_hi = px.copy(); px_hi[:, 1] = 100.0
        with self.assertRaises(Exception): G.fit_floor_plane(Xall, px_hi)                                                # rows above y_min are excluded -> nothing to fit
    def test_ray_plane_and_homogeneous_lines(self):
        n = np.array([0.0, -1.0, 0.0]); p = G.ray_plane_intersection(np.zeros(3), np.array([0.0, 0.2, 1.0]) / np.linalg.norm([0, 0.2, 1]), n, 640.0)
        self.assertTrue(abs(n @ p + 640.0) < 1e-9 and p[2] > 0)
        v = dict(kind="v", a=0.0, b=100.0); h = dict(kind="h", a=0.0, b=50.0); self.assertTrue(np.allclose(G.line_intersection(v, h), [100.0, 50.0]))
    def test_corner_line_floor_and_top_point(self):
        cal = synth_cal(); R, T = G.pose_from_params(TRUE_P, BASE); floor = (np.array([0.0, -1.0, 0.0]), 640.0); B = np.array([-800.0, 640.0, 3500.0]); Hh = 2400.0; top = B + Hh * floor[0]
        def hline(K, Xs): p = project(K, Xs); return np.cross(np.r_[p[0], 1], np.r_[p[1], 1])
        lL = hline(cal.K_left, np.array([B, top])); lR = hline(cal.K_right, np.array([B, top]) @ R.T + T)
        Bh, u = G.corner_line_on_floor(cal, R, T, lL, lR, floor); self.assertLess(np.linalg.norm(Bh - B), 1e-4); self.assertAlmostEqual(abs(float(u @ floor[0])), 1.0, places=6)
        px = project(cal.K_right, (top @ R.T + T)[None])[0]; t, skew = G.top_point_on_vertical(cal, R, T, px, Bh, floor[0]); self.assertAlmostEqual(t, Hh, delta=1e-3); self.assertLess(skew, 1e-3)
    def test_rigid_registration_robust_to_outliers(self):
        rng = np.random.default_rng(7); P0 = rng.uniform(-2000, 2000, (200, 3)); Rt = cv2.Rodrigues(np.array([0.1, 0.5, -0.05]))[0]; tt = np.array([300.0, -20, 150]); Q = P0 @ Rt.T + tt + rng.normal(0, 2, P0.shape)
        Q[:40] += rng.uniform(300, 900, (40, 3)); R, t, rms, ni, n = G.rigid_ransac(P0, Q, np.random.default_rng(0)); self.assertLess(np.abs(R - Rt).max(), 2e-3); self.assertLess(np.linalg.norm(t - tt), 5.0); self.assertLess(rms, 6.0)
        R2, t2 = G.umeyama_irls(P0, Q); self.assertLess(np.abs(R2 - Rt).max(), 2e-3)
    def test_register_views_uses_index_pairs(self):
        rng = np.random.default_rng(8); P0 = rng.uniform(-2000, 2000, (60, 3)); Rt = cv2.Rodrigues(np.array([0.0, 0.4, 0.0]))[0]; tt = np.array([100.0, 0, 50]); Q = P0 @ Rt.T + tt
        i1 = np.arange(60) + 1000; i2 = np.arange(60) + 5000; MX = np.c_[i1, i2]; R, t, rms, ni, n = M.register_views(P0, i1, Q, i2, MX, np.random.default_rng(0)); self.assertLess(np.linalg.norm(t - tt), 1e-6); self.assertEqual(n, 60)

class TestMeasurementCalculation(unittest.TestCase):
    def test_wall_width_height_recovered_from_synthetic_wall(self):
        cal = synth_cal(); R, T = G.pose_from_params(TRUE_P, BASE); floor = (np.array([0.0, -1.0, 0.0]), 640.0); up = floor[0]; W_true, H_true = 3000.0, 2500.0
        R0 = cv2.Rodrigues(np.array([0, np.radians(30.0), 0]))[0]; t0 = np.array([400.0, 0.0, -300.0])             # view-1 frame -> view-2 frame (no y translation: same floor plane)
        Ca = np.array([-1200.0, 640.0, 4200.0]); Cb = Ca + np.array([W_true * np.cos(0.4), 0.0, W_true * np.sin(0.4)])  # two bottom corners in the view-1 frame; wall along one direction
        def lines_for(Bl):                                      # Bl: bottom corner in this view's left frame
            top = Bl + H_true * up; out = {}
            for X, K, nm in ((np.array([Bl, top]), cal.K_left, "A"), (np.array([Bl, top]) @ R.T + T, cal.K_right, "P")):
                out[nm] = project(K, X)
            return out
        lines = {}
        for v, Bl, names in (("V1", Ca, ("AL_corner", "PL_corner", "PL_ceilF", "PL_ceilS")), ("V2", R0 @ Cb + t0, ("AR_corner", "PR_corner", "PR_ceilF", "PR_ceilS"))):
            d = lines_for(Bl)
            for key, nm in (("A", names[0]), ("P", names[1])):
                p = d[key]; a = (p[1, 0] - p[0, 0]) / (p[1, 1] - p[0, 1]); lines[nm] = dict(kind="v", a=a, b=p[0, 0] - a * p[0, 1])
            xtop, ytop = d["P"][1]; lines[names[2]] = dict(kind="h", a=0.0, b=ytop); lines[names[3]] = dict(kind="h", a=0.3, b=ytop - 0.3 * xtop)   # both ceiling lines pass through the top corner
        geom = dict(f1=floor, f2=floor, R0=R0, t0=t0)
        for variant in (0, 1):
            o = M.measure_wall(cal, R, T, geom, lines, variant); self.assertAlmostEqual(o["width_mm"], W_true, delta=0.05); self.assertAlmostEqual(o["height_mm"], H_true, delta=0.5); self.assertLess(o["V1"]["vertical_error_deg"], 1e-4)
    def test_measurement_perturbation_changes_result_smoothly(self):
        # a 0-pixel perturbation must be a no-op
        cal = synth_cal(); R, T = G.pose_from_params(TRUE_P, BASE); floor = (np.array([0.0, -1.0, 0.0]), 640.0); B = np.array([0.0, 640.0, 4000.0]); top = B + 2400 * floor[0]
        pl = project(cal.K_left, np.array([B, top])); pr = project(cal.K_right, np.array([B, top]) @ R.T + T); mk = lambda p: dict(kind="v", a=(p[1, 0] - p[0, 0]) / (p[1, 1] - p[0, 1]), b=p[0, 0] - (p[1, 0] - p[0, 0]) / (p[1, 1] - p[0, 1]) * p[0, 1])
        ln = {"AL_corner": mk(pl), "PL_corner": mk(pr), "AR_corner": mk(pl), "PR_corner": mk(pr), "PL_ceilF": dict(kind="h", a=0.0, b=pr[1, 1]), "PR_ceilF": dict(kind="h", a=0.0, b=pr[1, 1]), "PL_ceilS": dict(kind="h", a=0.0, b=pr[1, 1]), "PR_ceilS": dict(kind="h", a=0.0, b=pr[1, 1])}
        g = dict(f1=floor, f2=floor, R0=np.eye(3), t0=np.array([1000.0, 0, 0])); a = M.measure_wall(cal, R, T, g, ln); b = M.measure_wall(cal, R, T, g, ln, 0, {"V1": (0, 0), "V2": (0, 0), "dy": {"V1": 0, "V2": 0}})
        self.assertEqual(a["width_mm"], b["width_mm"]); self.assertAlmostEqual(a["width_mm"], 1000.0, delta=0.05)
        c = M.measure_wall(cal, R, T, g, ln, 0, {"V1": (2, 0), "V2": (0, 0), "dy": {"V1": 0, "V2": 0}}); self.assertNotEqual(a["width_mm"], c["width_mm"])

class TestNoGroundTruthLeakage(unittest.TestCase):
    ESTIMATION = ("calibration", "features", "frames", "geometry", "lines", "measure", "pose", "uncertainty", "pipeline")
    def test_estimation_sources_contain_no_ground_truth_or_lens_height(self):
        for m in self.ESTIMATION:
            src = (MODULE / "room_measurement" / f"{m}.py").read_text()
            self.assertNotIn("validation", src.replace("validation.py", ""), m + " must not reference the validation module")
            self.assertIsNone(re.search(r"(?<![\d.])(2900|2400|640)(\.0)?(?![\d.])", re.sub(r"#.*", "", src)), m + " contains a ground-truth / lens-height literal")
    def test_importing_pipeline_does_not_import_validation(self):
        code = f"import sys; sys.path.insert(0, {str(MODULE)!r}); import room_measurement.pipeline; print('room_measurement.validation' in sys.modules)"
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, check=True).stdout.strip(); self.assertEqual(out, "False")
    def test_estimate_independent_of_ground_truth_values(self):
        from room_measurement import validation as V
        cal = synth_cal(); sets = synth_sets(cal, 2); D = P.build_dataset(cal, sets); p1 = P.estimate_pose(D)[0]; old = dict(V.GROUND_TRUTH)
        try: V.GROUND_TRUTH.update(width_mm=1.0, height_mm=1.0); V.LENS_HEIGHT_MM = 5.0; p2 = P.estimate_pose(D)[0]
        finally: V.GROUND_TRUTH.update(old); V.LENS_HEIGHT_MM = 640.0
        self.assertTrue(np.array_equal(p1, p2))
    def test_comparison_is_post_hoc_and_pure(self):
        from room_measurement import validation as V
        res = dict(summary=dict(width_mm=2880.0, height_mm=2380.0), per_set=[dict(ok=True, floor_height_mm=[642.0, 640.0])]); before = json.dumps(res, sort_keys=True); v = V.compare_to_ground_truth(res)
        self.assertEqual(json.dumps(res, sort_keys=True), before); self.assertAlmostEqual(v["width_mm"]["error_percent"], 100 * (2880 - 2900) / 2900)

class TestDeterminism(unittest.TestCase):
    def test_pose_and_measurement_repeat_bit_exactly(self):
        cal = synth_cal(); sets = synth_sets(cal, 2); D1 = P.build_dataset(cal, sets); D2 = P.build_dataset(cal, sets)
        a = P.estimate_pose(D1)[0]; b = P.estimate_pose(D2)[0]; self.assertEqual(hashlib.sha256(a.tobytes()).hexdigest(), hashlib.sha256(b.tobytes()).hexdigest())
    def test_bootstrap_resample_seeded(self):
        cal = synth_cal(); D = P.build_dataset(cal, synth_sets(cal, 2)); r1 = P.resample_dataset(D, np.random.default_rng(1)); r2 = P.resample_dataset(D, np.random.default_rng(1)); self.assertTrue(np.array_equal(r1["n1"], r2["n1"]))

@unittest.skipUnless(HAVE_VIDEOS and os.environ.get("ROOM_MEASUREMENT_SKIP_VIDEO_TEST") != "1", "protected wall videos not available (or skipped)")
class TestRealVideoRegression(unittest.TestCase):
    def test_matches_validated_method_j_experiment_and_is_repeatable(self):
        from room_measurement import pipeline
        ref = json.loads((MODULE / "room_measurement" / "reference" / "method_j_experiment_reference.json").read_text()); tol = ref["tolerance"]
        r1 = pipeline.run_estimation(with_uncertainty=False, log=lambda *a: None); r2 = pipeline.run_estimation(with_uncertainty=False, log=lambda *a: None)
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))                                         # deterministic end to end
        self.assertFalse(r1["ground_truth_used"]); self.assertNotIn("ground_truth", json.dumps({k: v for k, v in r1.items() if k != "ground_truth_used"}).lower())   # only the False flag may mention it
        self.assertTrue(all(v["ok"] for v in r1["inputs"]["videos"]))
        self.assertAlmostEqual(r1["summary"]["width_mm"], ref["mean_width_mm"], delta=tol["dimension_mm"]); self.assertAlmostEqual(r1["summary"]["height_mm"], ref["mean_height_mm"], delta=tol["dimension_mm"])
        for got, want in zip(r1["per_set"], zip(ref["per_set_width_mm"], ref["per_set_height_mm"])): self.assertAlmostEqual(got["width_mm"], want[0], delta=tol["dimension_mm"]); self.assertAlmostEqual(got["height_mm"], want[1], delta=tol["dimension_mm"])
        self.assertTrue(np.allclose(r1["pose"]["rodrigues_deg"], ref["rodrigues_deg"], atol=tol["rodrigues_deg"])); self.assertAlmostEqual(r1["pose"]["baseline_mm"], 225.0, places=6)

if __name__ == "__main__":
    unittest.main()
