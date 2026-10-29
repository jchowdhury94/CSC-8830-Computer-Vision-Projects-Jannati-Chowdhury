"""Phase 1 tests for the visitor stereo calibration core (stereo_measure/user_calibration.py).

Synthetic rigs with known parameters check parameter recovery, distortion-model selection, corner-ordering resolution and warning
logic. Rendered checkerboard images check detection. The 30 real calibration pairs (skipped if absent) check the whole pipeline
against the stored Model B.
"""
import io
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))

from stereo_measure import load_model_b, measure_pair  # noqa: E402
from stereo_measure import user_calibration as uc  # noqa: E402
from stereo_measure.config import CALIB_PHOTO_DIR, NOTEBOOK_PAIR_IDS, PATTERN, SQUARE_MM  # noqa: E402
from stereo_measure.live import measure_points  # noqa: E402

PATTERN_T = (8, 6)
SQ = 30.0
LEFT_SIZE, RIGHT_SIZE = (4000, 3000), (3264, 2448)     # deliberately different resolutions


def rot(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], np.float64))[0]


def make_rig(seed=0, k_left=(0.02, -0.05, 0, 0, 0), k_right=(0.03, -0.06, 0, 0, 0), principal_right=None):
    """A known stereo rig: intrinsics, distortion, stereo R, T."""
    K1 = np.array([[3500, 0, 2010], [0, 3510, 1490], [0, 0, 1]], np.float64)
    cx, cy = principal_right or (1630, 1225)
    K2 = np.array([[2900, 0, cx], [0, 2895, cy], [0, 0, 1]], np.float64)
    R = rot(0.01, np.radians(3), -0.015)
    T = np.array([[-120.0], [2.0], [3.0]])
    return dict(K1=K1, D1=np.array(k_left, np.float64), K2=K2, D2=np.array(k_right, np.float64), R=R, T=T)


def sample_views(rig, n, seed=0, noise=0.25, pattern=PATTERN_T, square=SQ, size_l=LEFT_SIZE, size_r=RIGHT_SIZE, roll=0.3):
    """Board views visible in both cameras -> (left corners list, right corners list, true board poses)."""
    rng = np.random.default_rng(seed)
    cols, rows = pattern
    obj = uc.board_object_points(pattern, square).astype(np.float64)
    centre = obj.mean(0)
    left, right, poses = [], [], []
    while len(left) < n:
        Rb = rot(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-roll, roll))
        c = np.array([rng.uniform(-200, 200), rng.uniform(-150, 150), rng.uniform(450, 1100)])
        tb = c - Rb @ centre
        pl = cv2.projectPoints(obj, cv2.Rodrigues(Rb)[0], tb, rig["K1"], rig["D1"])[0].reshape(-1, 2)
        Rr, tr = rig["R"] @ Rb, rig["R"] @ tb + rig["T"].ravel()
        pr = cv2.projectPoints(obj, cv2.Rodrigues(Rr)[0], tr, rig["K2"], rig["D2"])[0].reshape(-1, 2)
        ok = all((p[:, 0] > 20).all() and (p[:, 0] < s[0] - 20).all() and (p[:, 1] > 20).all() and (p[:, 1] < s[1] - 20).all()
                 for p, s in ((pl, size_l), (pr, size_r)))
        if ok:
            left.append(pl + rng.normal(0, noise, pl.shape))
            right.append(pr + rng.normal(0, noise, pr.shape))
            poses.append((Rb, tb))
    return left, right, poses


def angle_deg(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1)))


class SyntheticRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = make_rig()
        cls.left, cls.right, _ = sample_views(cls.rig, 14, seed=1)
        cls.cal = uc.calibrate_stereo(cls.left, cls.right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)

    def test_recovers_known_parameters(self):
        cal, rig = self.cal, self.rig
        self.assertTrue(cal["valid"], cal["errors"])
        self.assertAlmostEqual(cal["baseline_mm"], np.linalg.norm(rig["T"]), delta=0.01 * np.linalg.norm(rig["T"]))
        self.assertLess(angle_deg(cal["R"], rig["R"]), 0.5)
        for key in ("K1", "K2"):
            self.assertLess(abs(cal[key][0, 0] / rig[key][0, 0] - 1), 0.02, key)
            self.assertLess(abs(cal[key][1, 1] / rig[key][1, 1] - 1), 0.02, key)
        self.assertLess(cal["rms_left"], 0.6)
        self.assertLess(cal["rms_right"], 0.6)
        self.assertLess(cal["stereo_rms"], 0.8)
        self.assertEqual(cal["image_size_left"], LEFT_SIZE)
        self.assertEqual(cal["image_size_right"], RIGHT_SIZE)           # different resolutions are accepted
        self.assertLess(abs(cal["metrics"]["board_square_scale"]["median"] - 1), 0.02)
        self.assertLess(cal["metrics"]["epipolar_px"]["median"], 1.0)
        self.assertLess(cal["metrics"]["rectification_vertical_px"]["median"], 1.0)

    def test_result_plugs_into_measurement(self):
        """The calibration dict is a drop-in model for the verified triangulation/measurement code."""
        obj = uc.board_object_points(PATTERN_T, SQ)
        # triangulate one board: reconstructed adjacent-corner spacing equals the true square size
        from stereo_measure.triangulation import triangulate
        X = triangulate(self.cal, self.left[0], self.right[0]).reshape(6, 8, 3)
        spacing = np.linalg.norm(np.diff(X, axis=1), axis=2).mean()
        self.assertAlmostEqual(spacing, SQ, delta=0.03 * SQ)
        self.assertTrue(set(("K1", "D1", "K2", "D2", "R", "T", "F")) <= set(self.cal))
        del obj

    def test_minimum_pairs_enforced(self):
        with self.assertRaises(ValueError):
            uc.calibrate_stereo(self.left[:7], self.right[:7], PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        uc.calibrate_stereo(self.left[:8], self.right[:8], PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)   # exactly 8 is allowed

    def test_bad_arguments(self):
        for kwargs in (dict(square_mm=0), dict(square_mm=float("nan")), dict(square_mm=-5)):
            with self.assertRaises(ValueError):
                uc.calibrate_stereo(self.left, self.right, PATTERN_T, kwargs["square_mm"], LEFT_SIZE, RIGHT_SIZE)
        with self.assertRaises(ValueError):
            uc.calibrate_stereo(self.left, self.right, (2, 6), SQ, LEFT_SIZE, RIGHT_SIZE)
        with self.assertRaises(ValueError):
            uc.calibrate_stereo(self.left, self.right[:-1], PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        nan = [c.copy() for c in self.left]
        nan[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            uc.calibrate_stereo(nan, self.right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        with self.assertRaises(ValueError):
            uc.calibrate_stereo(self.left, self.right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE, distortion="bogus")

    def test_wrong_square_size_scales_baseline_not_validity(self):
        """The entered square size sets the metric scale: doubling it doubles the baseline (and nothing else can detect this)."""
        double = uc.calibrate_stereo(self.left, self.right, PATTERN_T, 2 * SQ, LEFT_SIZE, RIGHT_SIZE)
        self.assertAlmostEqual(double["baseline_mm"] / self.cal["baseline_mm"], 2.0, delta=0.01)
        self.assertTrue(double["valid"])

    def test_unusual_principal_point_is_a_warning_not_an_error(self):
        rig = make_rig(principal_right=(3500, 600))                   # true optical axis outside the 3264 x 2448 frame
        left, right, _ = sample_views(rig, 12, seed=4)
        cal = uc.calibrate_stereo(left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        self.assertTrue(cal["valid"], cal["errors"])
        codes = [w["code"] for w in cal["warnings"]]
        self.assertTrue({"principal_point_outside", "principal_point_offset"} & set(codes), codes)

    def test_few_pairs_warning_and_heuristic_flag(self):
        cal = uc.calibrate_stereo(self.left[:10], self.right[:10], PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        few = [w for w in cal["warnings"] if w["code"] == "few_pairs"]
        self.assertEqual(len(few), 1)
        self.assertFalse(few[0]["heuristic"])
        self.assertTrue(all("heuristic" in w for w in cal["warnings"]))

    def test_short_baseline_is_a_warning(self):
        rig = make_rig()
        rig["T"] = np.array([[-4.0], [0.1], [0.2]])
        left, right, _ = sample_views(rig, 12, seed=6)
        cal = uc.calibrate_stereo(left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        self.assertTrue(cal["valid"], cal["errors"])
        self.assertIn("short_baseline", [w["code"] for w in cal["warnings"]])

    def test_outlier_pair_flagged_by_number(self):
        left = [c.copy() for c in self.left]
        left[3] = left[3] + np.random.default_rng(0).normal(0, 6.0, left[3].shape)      # one badly detected photo
        cal = uc.calibrate_stereo(left, self.right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE, pair_ids=[10 + i for i in range(14)])
        flagged = [p["pair"] for p in cal["per_pair"] if p["outlier"]]
        self.assertIn(13, flagged)
        self.assertTrue(any("Pair 13" in w["message"] for w in cal["warnings"]))

    def test_scrambled_correspondences_are_invalid_or_flagged(self):
        """Mis-ordered corners must never produce a clean-looking calibration."""
        right = [c[::-1].copy() if i % 2 else c for i, c in enumerate(self.right)]
        cal = uc.calibrate_stereo(self.left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        bad = (not cal["valid"]) or any(w["code"] in ("rms_stereo", "epipolar", "board_scale", "pair_outlier") for w in cal["warnings"])
        self.assertTrue(bad)
        self.assertGreater(cal["stereo_rms"], 10 * self.cal["stereo_rms"])


class DistortionModelTests(unittest.TestCase):
    def test_simple_lens_keeps_model_b_distortion_model(self):
        rig = make_rig()
        left, right, _ = sample_views(rig, 16, seed=2)
        cal = uc.calibrate_stereo(left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        self.assertEqual((cal["distortion_left"], cal["distortion_right"]), ("radial2", "radial2"))
        self.assertTrue(all(len(cal["distortion_cv"][s]) == 3 for s in ("left", "right")))

    def test_complex_lens_selects_richer_model_and_fits_better(self):
        rig = make_rig(k_left=(0.02, -0.05, 0, 0, 0), k_right=(-0.28, 0.45, 0.002, -0.0015, -0.30))
        left, right, _ = sample_views(rig, 16, seed=3, noise=0.15)
        auto = uc.calibrate_stereo(left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        forced = uc.calibrate_stereo(left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE, distortion="radial2")
        self.assertEqual(auto["distortion_left"], "radial2")
        self.assertIn(auto["distortion_right"], ("radial3", "full"))
        self.assertLess(auto["rms_right"], 0.9 * forced["rms_right"])                  # a clearly better fit...
        self.assertLess(auto["rms_right"], 0.3)                                        # ...down to the 0.15 px/axis noise floor
        self.assertLess(auto["stereo_rms"], forced["stereo_rms"])
        table = {t["model"]: t["cv_rms"] for t in auto["distortion_cv"]["right"]}
        self.assertLess(min(table["radial3"], table["full"]), table["radial2"])

    def test_explicit_model_skips_selection(self):
        rig = make_rig()
        left, right, _ = sample_views(rig, 10, seed=5)
        cal = uc.calibrate_stereo(left, right, PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE, distortion="full")
        self.assertEqual((cal["distortion_left"], cal["distortion_right"]), ("full", "full"))
        self.assertEqual(cal["distortion_cv"], {"left": [], "right": []})


class OrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = make_rig()
        cls.left, cls.right, _ = sample_views(cls.rig, 14, seed=7, roll=0.15)

    def _resolve(self, right, pattern=PATTERN_T, max_side=4000):
        return uc.resolve_pair_orderings(list(zip(self.left, right)), pattern, max_side)

    def test_correct_pairs_are_accepted_unchanged(self):
        res = self._resolve(self.right)
        self.assertEqual([p["status"] for p in res["pairs"]], ["ok"] * 14)
        self.assertEqual({p["action"] for p in res["pairs"]}, {"as_detected"})
        for p, r in zip(res["pairs"], self.right):
            np.testing.assert_array_equal(p["right_corners"], r)

    def test_reversed_detections_are_corrected_and_reported(self):
        flipped = {1, 4, 5, 9}
        right = [c[::-1].copy() if i in flipped else c for i, c in enumerate(self.right)]
        res = self._resolve(right)
        for i, p in enumerate(res["pairs"]):
            self.assertEqual(p["status"], "ok", p["message"])
            self.assertEqual(p["action"], "rotated_180" if i in flipped else "as_detected")      # reported, never silent
            np.testing.assert_allclose(p["right_corners"], self.right[i])
            self.assertGreaterEqual(p["ratio"], uc.ORDER_RATIO_MIN)
        # and the corrected set calibrates properly
        cal = uc.calibrate_stereo(self.left, [p["right_corners"] for p in res["pairs"]], PATTERN_T, SQ, LEFT_SIZE, RIGHT_SIZE)
        self.assertLess(cal["stereo_rms"], 1.0)

    def test_square_board_rotations_are_resolved(self):
        rig = make_rig()
        left, right, _ = sample_views(rig, 14, seed=8, pattern=(6, 6), roll=0.15)
        idx = np.arange(36).reshape(6, 6)
        rotated = []
        for i, c in enumerate(right):
            perm = {0: idx.ravel(), 1: np.rot90(idx, 1).ravel(), 2: idx[::-1, ::-1].ravel(), 3: np.rot90(idx, 3).ravel()}[i % 4]
            rotated.append(c[perm].copy())
        res = uc.resolve_pair_orderings(list(zip(left, rotated)), (6, 6), 4000)
        for p, c in zip(res["pairs"], right):
            self.assertEqual(p["status"], "ok", p["message"])
            np.testing.assert_allclose(p["right_corners"], c)

    def test_garbage_pair_is_flagged_not_guessed(self):
        right = [c.copy() for c in self.right]
        right[6] = np.random.default_rng(1).uniform([200, 200], [3000, 2200], right[6].shape)
        res = self._resolve(right)
        self.assertNotEqual(res["pairs"][6]["status"], "ok")
        self.assertEqual(res["pairs"][6]["action"], "as_detected")                      # left exactly as detected
        np.testing.assert_array_equal(res["pairs"][6]["right_corners"], right[6])
        self.assertEqual([p["status"] for i, p in enumerate(res["pairs"]) if i != 6], ["ok"] * 13)

    def test_too_few_pairs_cannot_be_verified(self):
        for k in (1, 2):                                    # one or two planar boards leave the epipolar geometry undetermined
            res = uc.resolve_pair_orderings(list(zip(self.left[:k], self.right[:k])), PATTERN_T, 4000)
            self.assertEqual([p["status"] for p in res["pairs"]], ["unverifiable"] * k)
            self.assertEqual({p["action"] for p in res["pairs"]}, {"as_detected"})
        res = uc.resolve_pair_orderings(list(zip(self.left[:3], self.right[:3])), PATTERN_T, 4000)
        self.assertEqual([p["status"] for p in res["pairs"]], ["ok"] * 3)


def render_board(size, K, R, t, pattern=PATTERN_T, square=SQ):
    """A synthetic photograph (grayscale) of a printed checkerboard, plus its true inner-corner pixels (no lens distortion)."""
    cols, rows = pattern
    ppm = 4.0                                                # texture pixels per mm
    x0, y0 = -2 * square, -2 * square                        # texture origin in board mm (one square of white margin)
    tw, th = (cols + 1 + 2) * square * ppm, (rows + 1 + 2) * square * ppm
    tex = np.full((int(th), int(tw)), 255, np.uint8)
    for j in range(rows + 1):
        for i in range(cols + 1):
            if (i + j) % 2 == 0:
                xa, ya = int((-square + i * square - x0) * ppm), int((-square + j * square - y0) * ppm)
                tex[ya:ya + int(square * ppm), xa:xa + int(square * ppm)] = 20
    Hb = K @ np.column_stack([R[:, 0], R[:, 1], t])          # board mm -> pixels
    S = np.array([[1 / ppm, 0, x0], [0, 1 / ppm, y0], [0, 0, 1]])   # texture px -> board mm
    img = cv2.warpPerspective(tex, Hb @ S, size, flags=cv2.INTER_LINEAR, borderValue=170)
    obj = uc.board_object_points(pattern, square).astype(np.float64)
    pts = cv2.perspectiveTransform(obj[:, :2].reshape(-1, 1, 2), Hb).reshape(-1, 2)
    return img, pts


def encode(img, fmt="JPEG", orientation=None):
    buf = io.BytesIO()
    exif = Image.Exif()
    if orientation:
        exif[274] = orientation
    pil = Image.fromarray(img)
    pil.save(buf, fmt, **({"quality": 92, "exif": exif} if fmt == "JPEG" else {}))
    return buf.getvalue()


class DetectionTests(unittest.TestCase):
    K = np.array([[1700.0, 0, 1000], [0, 1700.0, 750], [0, 0, 1]])

    def view(self, **kw):
        R = rot(0.25, -0.3, 0.1)
        t = np.array([-100.0, -60.0, 650.0])
        return render_board((2000, 1500), self.K, R, t, **kw)

    def test_detects_rendered_board_subpixel(self):
        img, truth = self.view()
        r = uc.process_calibration_photo(encode(img), PATTERN_T)
        self.assertEqual(r["status"], "ok", r["message"])
        self.assertEqual(r["size"], (2000, 1500))
        c = r["corners"]
        d = np.linalg.norm(c - truth, axis=1)
        d2 = np.linalg.norm(c[::-1] - truth, axis=1)             # detection numbering may be reversed
        self.assertLess(min(d.mean(), d2.mean()), 0.6)
        self.assertIsNotNone(r["thumbnail"])
        self.assertLessEqual(max(Image.open(io.BytesIO(r["thumbnail"])).size), uc.THUMBNAIL_SIDE)

    def test_png_and_exif_rotated_photos(self):
        img, truth = self.view()
        self.assertEqual(uc.process_calibration_photo(encode(img, "PNG"), PATTERN_T)["status"], "ok")
        # stored sideways with EXIF orientation 6: must come out upright with the board where it was
        sideways = np.ascontiguousarray(np.rot90(img, 1))        # stored image is the upright photo rotated counter-clockwise
        r = uc.process_calibration_photo(encode(sideways, orientation=6), PATTERN_T)
        self.assertEqual(r["size"], (2000, 1500))
        self.assertEqual(r["status"], "ok", r["message"])
        d = min(np.linalg.norm(r["corners"] - truth, axis=1).mean(), np.linalg.norm(r["corners"][::-1] - truth, axis=1).mean())
        self.assertLess(d, 0.8)

    def test_orientation_handling_matches_pillow(self):
        base = np.arange(60 * 40, dtype=np.uint16).reshape(40, 60) % 251
        base = base.astype(np.uint8)
        for orientation in range(1, 9):
            data = encode(np.repeat(base[:, :, None], 3, 2), "PNG") if False else None
            buf = io.BytesIO()
            exif = Image.Exif()
            exif[274] = orientation
            Image.fromarray(base).save(buf, "PNG", exif=exif)
            data = buf.getvalue()
            im = Image.open(io.BytesIO(data))
            expect = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L"))
            got = uc.decode_gray(im, orientation, (expect.shape[1], expect.shape[0]))
            np.testing.assert_array_equal(got, expect, f"orientation {orientation}")

    def test_blank_photo_not_found(self):
        r = uc.process_calibration_photo(encode(np.full((1500, 2000), 128, np.uint8)), PATTERN_T)
        self.assertEqual(r["status"], "not_found")
        self.assertIsNone(r["corners"])
        self.assertIn("8 x 6", r["message"])

    def test_wrong_pattern_reports_not_found_or_swap(self):
        img, _ = self.view()
        r = uc.process_calibration_photo(encode(img), (9, 7))
        self.assertEqual(r["status"], "not_found")
        swapped = uc.process_calibration_photo(encode(img), (6, 8))
        self.assertIn(swapped["status"], ("swapped_pattern", "ok", "not_found"))
        if swapped["status"] == "swapped_pattern":
            self.assertIn("Swap", swapped["message"])

    def test_rejections_are_explained(self):
        too_big = uc.process_calibration_photo(b"\0" * (uc.MAX_FILE_BYTES + 1), PATTERN_T)
        self.assertEqual(too_big["status"], "rejected")
        self.assertIn("30 MB", too_big["message"])
        buf = io.BytesIO()
        Image.new("L", (8000, 7000)).save(buf, "PNG")                  # 56 MP, tiny file
        mp = uc.process_calibration_photo(buf.getvalue(), PATTERN_T)
        self.assertEqual(mp["status"], "rejected")
        self.assertIn("megapixels", mp["message"])
        for data, word in ((b"", "empty"), (b"not an image", "could not be read"),
                           (encode(np.zeros((200, 200), np.uint8)), "too small")):
            r = uc.process_calibration_photo(data, PATTERN_T)
            self.assertEqual(r["status"], "rejected")
            self.assertIn(word, r["message"])
        gif = io.BytesIO()
        Image.new("P", (800, 600)).save(gif, "GIF")
        r = uc.process_calibration_photo(gif.getvalue(), PATTERN_T)
        self.assertEqual(r["status"], "rejected")
        self.assertIn("Unsupported format", r["message"])

    def test_warped_non_board_is_not_trusted(self):
        """Noise that happens to contain a checker-like grid must not pass the grid-regularity check unnoticed."""
        rng = np.random.default_rng(0)
        r = uc.process_calibration_photo(encode(rng.integers(0, 255, (1500, 2000), np.uint8)), PATTERN_T)
        self.assertNotEqual(r["status"], "ok")

    def test_full_pipeline_on_rendered_stereo_photos(self):
        """Render photos from a known rig, detect, order, calibrate: end to end, no ground-truth corners involved."""
        rig = dict(K1=self.K, D1=np.zeros(5), K2=np.array([[1500.0, 0, 900], [0, 1500.0, 700], [0, 0, 1]]), D2=np.zeros(5),
                   R=rot(0.0, np.radians(2), 0.0), T=np.array([[-100.0], [1.0], [2.0]]))
        rng = np.random.default_rng(11)
        lefts, rights = [], []
        obj = uc.board_object_points(PATTERN_T, SQ).astype(np.float64)
        centre = obj.mean(0)
        n = 0
        while n < 10:
            Rb = rot(rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.15, 0.15))
            c = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(520, 900)])
            tb = c - Rb @ centre
            il, pl = render_board((2000, 1500), rig["K1"], Rb, tb)
            ir, pr = render_board((1800, 1400), rig["K2"], rig["R"] @ Rb, rig["R"] @ tb + rig["T"].ravel())
            if min(pl.min(), pr.min()) < 15 or (pl[:, 0] > 1985).any() or (pl[:, 1] > 1485).any() or (pr[:, 0] > 1785).any() or (pr[:, 1] > 1385).any():
                continue
            lefts.append(uc.process_calibration_photo(encode(il), PATTERN_T))
            rights.append(uc.process_calibration_photo(encode(ir), PATTERN_T))
            n += 1
        self.assertTrue(all(p["status"] == "ok" for p in lefts + rights), [p["message"] for p in lefts + rights if p["status"] != "ok"])
        res = uc.resolve_pair_orderings([(l["corners"], r["corners"]) for l, r in zip(lefts, rights)], PATTERN_T, 2000)
        self.assertEqual([p["status"] for p in res["pairs"]], ["ok"] * 10, [p["message"] for p in res["pairs"]])
        cal = uc.calibrate_stereo([l["corners"] for l in lefts], [p["right_corners"] for p in res["pairs"]], PATTERN_T, SQ,
                                  lefts[0]["size"], rights[0]["size"])
        self.assertTrue(cal["valid"], cal["errors"])
        self.assertAlmostEqual(cal["baseline_mm"], np.linalg.norm(rig["T"]), delta=0.02 * np.linalg.norm(rig["T"]))
        self.assertLess(cal["stereo_rms"], 1.5)


class MemoryTests(unittest.TestCase):
    def test_fifty_megapixel_photo_is_processed(self):
        """A photo at the pixel limit decodes and is searched without exhausting memory (peak RSS reported, not asserted tightly)."""
        import resource
        rng = np.random.default_rng(0)
        small = rng.integers(0, 255, (250, 200), np.uint8)
        big = cv2.resize(small, (8000, 6200), interpolation=cv2.INTER_NEAREST)       # 49.6 MP
        buf = io.BytesIO()
        Image.fromarray(big).save(buf, "JPEG", quality=70)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        r = uc.process_calibration_photo(buf.getvalue(), PATTERN_T)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.assertIn(r["status"], ("not_found", "irregular", "ok"))
        self.assertEqual(r["size"], (8000, 6200))
        grown_mb = max(0, after - before) / (1e6 if sys.platform == "darwin" else 1e3)
        self.assertLess(grown_mb, 1500)


@unittest.skipUnless((CALIB_PHOTO_DIR / "left" / "air_1.jpeg").is_file(), "original calibration photographs are not present")
class RealCalibrationPairsTests(unittest.TestCase):
    """The visitor pipeline on the author's 30 pairs, compared with the stored Model B (which is never modified)."""

    @classmethod
    def setUpClass(cls):
        photos = {s: [uc.process_calibration_photo((CALIB_PHOTO_DIR / s / f"{p}_{i}.jpeg").read_bytes(), PATTERN)
                      for i in range(1, 31)] for s, p in (("left", "air"), ("right", "pro"))}
        cls.photos = photos
        cls.order = uc.resolve_pair_orderings([(l["corners"], r["corners"]) for l, r in zip(photos["left"], photos["right"])],
                                              PATTERN, max(photos["left"][0]["size"]))
        cls.cal = uc.calibrate_stereo([l["corners"] for l in photos["left"]], [p["right_corners"] for p in cls.order["pairs"]],
                                      PATTERN, SQUARE_MM, photos["left"][0]["size"], photos["right"][0]["size"])
        cls.model_b = load_model_b()

    def test_all_photos_detected_and_pairs_verified(self):
        self.assertEqual([p["status"] for p in self.photos["left"] + self.photos["right"]], ["ok"] * 60)
        self.assertEqual([p["status"] for p in self.order["pairs"]], ["ok"] * 30)
        self.assertEqual({p["size"] for p in self.photos["left"] + self.photos["right"]}, {(4284, 5712)})

    def test_close_to_model_b(self):
        cal, b = self.cal, self.model_b
        self.assertTrue(cal["valid"], cal["errors"])
        self.assertAlmostEqual(cal["baseline_mm"], b["baseline_mm"], delta=0.01 * b["baseline_mm"])
        self.assertLess(angle_deg(cal["R"], b["R"]), 1.5)
        self.assertLess(abs(cal["K1"][0, 0] / b["K1"][0, 0] - 1), 0.02)
        self.assertLess(abs(cal["K2"][0, 0] / b["K2"][0, 0] - 1), 0.03)
        self.assertLess(abs(cal["rms_left"] - b["rms_left"]), 0.3)
        self.assertLess(abs(cal["stereo_rms"] - b["stereo_rms"]), 1.0)

    def test_reproduces_notebook_measurements(self):
        """Measuring the three saved notebook corner sets with the visitor calibration lands near the Model B results."""
        for pair in NOTEBOOK_PAIR_IDS:
            ref = measure_pair(self.model_b, pair)
            res = measure_points(self.cal, ref["left_px"], ref["right_px"], (305.0, 215.0))
            self.assertEqual(res["errors"], [])
            s, sb = res["summary"], ref["summary"]
            self.assertLess(abs(s["width_mm"] - sb["width_mm"]), 2.0)
            self.assertLess(abs(s["height_mm"] - sb["height_mm"]), 2.0)
            self.assertLess(abs(s["mean_Z_depth_mm"] - sb["mean_Z_depth_mm"]), 10.0)

    def test_model_b_file_untouched(self):
        import hashlib
        import json
        ref = json.loads((MODULE / "reference" / "expected_model_b_results.json").read_text())
        sha = hashlib.sha256((MODULE / "calibration" / "model_b" / "stereo_calibration_30pairs.npz").read_bytes()).hexdigest()
        self.assertEqual(sha, ref["model_b_npz_sha256"])


if __name__ == "__main__":
    unittest.main()
