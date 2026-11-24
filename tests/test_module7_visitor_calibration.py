"""Phase 2B Stage A: visitor calibration adapter (room_measurement/visitor_calibration.py)."""
import ast
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7" / "room_measurement_question_3"      # Question 3 packages, data and outputs
sys.path.insert(0, str(MODULE)); sys.path.insert(0, str(MODULE.parent))      # MODULE.parent: shared stereo_measure

from room_measurement import calibration as C  # noqa: E402
from room_measurement import visitor_calibration as V  # noqa: E402
from room_measurement.visitor_calibration import CompatibilityStatus as S, Severity  # noqa: E402

SRC = MODULE / "room_measurement"


def model(size=(2160, 3840), **over):
    w, h = size
    m = dict(K1=np.array([[3178.0, 0, w / 2], [0, 3178.0, h / 2], [0, 0, 1]]), D1=np.array([[0.1, -0.2, 0, 0, 0.0]]),
             K2=np.array([[2937.0, 0, w / 2 + 5], [0, 2937.0, h / 2 - 5], [0, 0, 1]]), D2=np.array([[0.05, -0.1, 0, 0, 0.0]]),
             R=np.eye(3), T=np.array([[-225.0], [0], [0]]), E=np.eye(3), F=np.eye(3), baseline_mm=225.0,
             image_size_left=(w, h), image_size_right=(w, h))
    m.update(over)
    return m


def codes(a, sev=Severity.ERROR):
    return {i.code for i in a.issues if i.severity is sev}


def with_modes(left, right, **kw):
    cam = {}
    if left is not None: cam["left"] = {"capture_mode": left}
    if right is not None: cam["right"] = {"capture_mode": right}
    return model(camera=cam, **kw)


class Conversion(unittest.TestCase):
    def test_valid_conversion_and_ordering(self):
        a = V.adapt_visitor_calibration(model(), 230.0)
        self.assertTrue(a.ok, a.issues)
        c = a.calibration
        self.assertIsInstance(c, C.StereoCalibration)
        np.testing.assert_array_equal(c.K_left, model()["K1"]); np.testing.assert_array_equal(c.K_right, model()["K2"])
        self.assertEqual(c.D_left.shape, (5,)); self.assertAlmostEqual(c.D_left[0], 0.1); self.assertAlmostEqual(c.D_right[0], 0.05)
        self.assertEqual(c.image_size, (2160, 3840)); self.assertEqual(c.baseline_mm, 230.0)
        C.validate(c)                                   # accepted by the existing validator
        self.assertFalse(c.provenance["accuracy_verified"])

    def test_left_right_not_swapped(self):
        m = model(); m["K1"][0, 0], m["K2"][0, 0] = 3000.0, 2500.0
        c = V.adapt_visitor_calibration(m, 100).calibration
        self.assertEqual(c.K_left[0, 0], 3000.0); self.assertEqual(c.K_right[0, 0], 2500.0)

    def test_invalid_shapes(self):
        for key, bad in (("K1", np.eye(2)), ("K2", np.ones(9)), ("D1", np.zeros(4)), ("D2", np.zeros((2, 5)))):
            with self.subTest(key=key):
                a = V.adapt_visitor_calibration(model(**{key: bad}), 100)
                self.assertFalse(a.ok); self.assertIn("INVALID_SHAPE", codes(a))
        self.assertIn("MISSING_FIELD", codes(V.adapt_visitor_calibration({k: v for k, v in model().items() if k != "K1"}, 100)))
        self.assertIn("INVALID_MODEL", codes(V.adapt_visitor_calibration([1, 2], 100)))

    def test_nonfinite(self):
        for key in ("K1", "D2"):
            bad = model()[key].astype(float); bad.flat[0] = np.nan if key == "K1" else np.inf
            a = V.adapt_visitor_calibration(model(**{key: bad}), 100)
            self.assertIn("NONFINITE_VALUE", codes(a))

    def test_intrinsics_plausibility(self):
        m = model(); m["K1"][0, 0] = -5
        self.assertIn("INVALID_FOCAL_LENGTH", codes(V.adapt_visitor_calibration(m, 100)))
        m = model(); m["K1"][0, 0] = m["K1"][1, 1] = 50
        self.assertIn("IMPLAUSIBLE_FOCAL_LENGTH", codes(V.adapt_visitor_calibration(m, 100)))
        m = model(); m["K2"][0, 2] = 99999
        self.assertIn("PRINCIPAL_POINT_OUTSIDE_IMAGE", codes(V.adapt_visitor_calibration(m, 100)))
        m = model(); m["K2"][0, 1] = 3.0
        self.assertIn("NONZERO_SKEW", codes(V.adapt_visitor_calibration(m, 100)))
        m = model(); m["K1"][0, 2] = 2160 * 0.85
        a = V.adapt_visitor_calibration(m, 100); self.assertTrue(a.ok); self.assertIn("PRINCIPAL_POINT_OFF_CENTRE", codes(a, Severity.WARNING))

    def test_baseline(self):
        for bad in (0, -5, float("nan"), float("inf"), "225", None, True):
            with self.subTest(bad=bad):
                self.assertIn("INVALID_BASELINE", codes(V.adapt_visitor_calibration(model(), bad)))
        for bad in (5.0, 5000.0):
            self.assertIn("BASELINE_OUT_OF_RANGE", codes(V.adapt_visitor_calibration(model(), bad)))
        lim = V.CalibrationLimits(min_baseline_mm=1, max_baseline_mm=10)
        self.assertTrue(V.adapt_visitor_calibration(model(), 5.0, lim).ok)
        self.assertTrue(V.adapt_visitor_calibration(model(), np.float64(225.0)).ok)

    def test_missing_or_bad_image_size(self):
        m = model(); del m["image_size_left"]
        a = V.adapt_visitor_calibration(m, 100); self.assertIn("MISSING_IMAGE_SIZE", codes(a)); self.assertIsNone(a.calibration)
        self.assertIn("MISSING_IMAGE_SIZE", codes(V.adapt_visitor_calibration(model(image_size_right=None), 100)))
        self.assertIn("INVALID_IMAGE_SIZE", codes(V.adapt_visitor_calibration(model(image_size_left=(0, 100)), 100)))
        self.assertIn("INVALID_IMAGE_SIZE", codes(V.adapt_visitor_calibration(model(image_size_left=(100.5, 100)), 100)))
        self.assertIn("CALIBRATION_SIZE_MISMATCH", codes(V.adapt_visitor_calibration(model(image_size_right=(3840, 2160)), 100)))
        with self.assertRaises(V.VisitorCalibrationError):
            V.adapt_visitor_calibration(m, 100).require()

    def test_errors_are_ui_ready(self):
        a = V.adapt_visitor_calibration(model(), -1)
        d = a.to_dict()
        self.assertEqual(d["status"], "ERROR"); self.assertTrue(all({"severity", "code", "message", "subject"} <= set(i) for i in d["issues"]))
        self.assertTrue(all(i.message.endswith(".") or i.message for i in a.errors))


class ReferenceExtrinsics(unittest.TestCase):
    def test_rt_never_affects_result(self):
        a = V.adapt_visitor_calibration(model(), 225.0).calibration
        b = V.adapt_visitor_calibration(model(R=np.diag([1, -1, -1.0]), T=np.array([[1e4], [7], [9]]), baseline_mm=9999.0), 225.0).calibration
        np.testing.assert_array_equal(a.K_left, b.K_left); np.testing.assert_array_equal(a.D_right, b.D_right)
        self.assertEqual(a.baseline_mm, b.baseline_mm)
        self.assertGreater(b.legacy_extrinsics["baseline_mm"], 1e4)       # kept as reference only
        self.assertEqual(b.baseline_mm, 225.0)                            # scale is the visitor-entered baseline

    def test_missing_or_garbage_rt_is_not_an_error(self):
        m = model(); del m["R"], m["T"]
        a = V.adapt_visitor_calibration(m, 225.0); self.assertTrue(a.ok); self.assertIsNone(a.calibration.legacy_extrinsics)
        a = V.adapt_visitor_calibration(model(T=np.array([np.nan, 0, 0])), 225.0)
        self.assertTrue(a.ok); self.assertIsNone(a.calibration.legacy_extrinsics); self.assertIn("REFERENCE_EXTRINSICS_DROPPED", codes(a, Severity.WARNING))

    def test_measurement_code_does_not_read_legacy_extrinsics(self):
        for f in SRC.glob("*.py"):
            if f.name in ("calibration.py", "visitor_calibration.py", "validation.py"):
                continue
            self.assertNotIn("legacy_extrinsics", f.read_text(), f.name)


class Compatibility(unittest.TestCase):
    def assess(self, m, left, right=None, baseline=225.0):
        a = V.adapt_visitor_calibration(m, baseline); self.assertTrue(a.ok, a.issues)
        return V.assess_capture_compatibility(a, left, left if right is None else right)

    P, L = [(2160, 3840)] * 2, [(3840, 2160)] * 2

    def test_missing_provenance(self):
        r = self.assess(model(), self.P)
        self.assertEqual(r.status, S.MISSING_PROVENANCE); self.assertEqual(r.severity, Severity.WARNING); self.assertFalse(r.accuracy_verified)
        self.assertEqual(self.assess(with_modes("video", None), self.P).status, S.MISSING_PROVENANCE)

    def test_unverified_capture_mode(self):
        for modes in (("photo", "photo"), ("video", "photo"), ("weird", "video")):
            self.assertEqual(self.assess(with_modes(*modes), self.P).status, S.UNVERIFIED_CAPTURE_MODE, modes)

    def test_compatible_metadata_is_not_accuracy(self):
        r = self.assess(with_modes(" Video", "VIDEO"), self.P)
        self.assertEqual(r.status, S.COMPATIBLE_METADATA); self.assertFalse(r.accuracy_verified)
        self.assertIn("not a verification", r.message); self.assertIn("does not establish", r.note)

    def test_resolution_mismatch(self):
        r = self.assess(with_modes("video", "video"), [(1080, 1920)] * 2)
        self.assertEqual(r.status, S.INCOMPATIBLE_RESOLUTION); self.assertEqual(r.severity, Severity.ERROR)
        self.assertEqual(self.assess(model(), self.P, [(2160, 3840), (2161, 3840)]).status, S.INCOMPATIBLE_RESOLUTION)

    def test_orientation_mismatch_takes_precedence(self):
        r = self.assess(with_modes("video", "video"), self.L)
        self.assertEqual(r.status, S.INCOMPATIBLE_ORIENTATION)
        self.assertEqual(self.assess(model(), self.P, self.L).status, S.INCOMPATIBLE_ORIENTATION)   # right side only

    def test_side_is_named(self):
        self.assertIn("right video", self.assess(model(), self.P, [(2160, 3840), (1, 1)][:1] + [(1080, 1920)]).message)

    def test_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):
            V.assess_capture_compatibility(V.adapt_visitor_calibration(model(), -1), self.P, self.P)


class NoGroundTruth(unittest.TestCase):
    def test_no_ground_truth_imports(self):
        for name in ("visitor_calibration.py", "visitor_video_validation.py"):
            tree = ast.parse((SRC / name).read_text())
            mods = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import): mods |= {a.name for a in n.names}
                elif isinstance(n, ast.ImportFrom): mods |= {(n.module or "") + "." + a.name for a in n.names} | {n.module or ""}
            self.assertFalse([m for m in mods if {"validation", "reference"} & set(m.split("."))], (name, mods))

    def test_ground_truth_not_loaded_at_runtime(self):
        code = ("import sys; sys.path.insert(0, %r); import room_measurement.visitor_calibration, room_measurement.visitor_video_validation;"
                "sys.exit(1 if 'room_measurement.validation' in sys.modules else 0)" % str(MODULE))
        self.assertEqual(subprocess.run([sys.executable, "-I", "-c", code]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
