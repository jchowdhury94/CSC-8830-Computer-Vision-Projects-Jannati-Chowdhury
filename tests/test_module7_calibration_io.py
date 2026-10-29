"""Phase 2 tests: exporting a visitor calibration to NPZ and loading it back safely (stereo_measure/calibration_io.py)."""
import hashlib
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(ROOT / "tests"))

import test_module7_user_calibration as base  # noqa: E402  (synthetic rigs; only the helpers are used)
from stereo_measure import calibration_io as cio  # noqa: E402
from stereo_measure import load_model_b, measure_pair  # noqa: E402
from stereo_measure import user_calibration as uc  # noqa: E402
from stereo_measure.config import CALIB_PHOTO_DIR, NOTEBOOK_PAIR_IDS, PATTERN, SQUARE_MM  # noqa: E402
from stereo_measure.live import measure_points  # noqa: E402

FIXED_TIME = "2026-10-28T12:00:00+00:00"
CORNERS = [0, 7, 47, 40]          # TL, TR, BR, BL of an 8 x 6 inner-corner grid (row-major)
EXECUTED = {"flag": False}


def set_flag():
    EXECUTED["flag"] = True


class Evil:
    """Unpickling this would run set_flag(); the loader must never unpickle."""

    def __reduce__(self):
        return (set_flag, ())


def fields_of(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def pack(fields, compressed=False):
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, **fields)
    return buf.getvalue()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = base.make_rig()
        cls.left, cls.right, _ = base.sample_views(cls.rig, 14, seed=21)
        cls.cal = uc.calibrate_stereo(cls.left, cls.right, base.PATTERN_T, base.SQ, base.LEFT_SIZE, base.RIGHT_SIZE,
                                      pair_ids=[2 * i + 1 for i in range(14)])
        cls.data = cio.export_calibration(cls.cal, pairs_uploaded=16, created=FIXED_TIME,
                                          camera_info={"left": {"model": "Left cam", "focal_length_mm": 5.96},
                                                       "right": {"model": "Right cam", "focal_length_mm": None}})
        cls.fields = fields_of(cls.data)

    def mutated(self, **changes):
        f = {k: v.copy() for k, v in self.fields.items()}
        for k, v in changes.items():
            if v is None:
                f.pop(k, None)
            else:
                f[k] = np.asarray(v)
        return pack(f)

    def assertRejected(self, data, fragment=None):
        with self.assertRaises(cio.CalibrationFileError) as ctx:
            cio.load_calibration(data)
        if fragment:
            self.assertIn(fragment.lower(), str(ctx.exception).lower())


class RoundTripTests(Base):
    def test_values_are_preserved_exactly(self):
        m = cio.load_calibration(self.data)
        for key in ("K1", "K2", "R"):
            np.testing.assert_array_equal(m[key], self.cal[key])                      # bit-for-bit float64
        np.testing.assert_array_equal(m["D1"].ravel(), self.cal["D1"].ravel())
        np.testing.assert_array_equal(m["D2"].ravel(), self.cal["D2"].ravel())
        np.testing.assert_array_equal(m["T"].ravel(), self.cal["T"].ravel())
        for key in ("stereo_rms", "rms_left", "rms_right", "baseline_mm"):
            self.assertEqual(m[key], self.cal[key])
        self.assertEqual((m["image_size_left"], m["image_size_right"]), (base.LEFT_SIZE, base.RIGHT_SIZE))
        self.assertEqual(m["image_size"], base.LEFT_SIZE)
        self.assertEqual((m["pattern"], m["square_mm"]), (base.PATTERN_T, base.SQ))
        self.assertEqual((m["pairs_uploaded"], m["pairs_valid"], m["pairs"]), (16, 14, self.cal["pairs"]))
        self.assertEqual((m["distortion_left"], m["distortion_right"]), (self.cal["distortion_left"], self.cal["distortion_right"]))
        for tag in ("left", "right"):
            self.assertEqual([(t["model"], t["cv_rms"], t["chosen"]) for t in m["distortion_cv"][tag]],
                             [(t["model"], t["cv_rms"], t["chosen"]) for t in self.cal["distortion_cv"][tag]])
        self.assertEqual(m["warnings"], [dict(code=w["code"], message=w["message"], heuristic=w["heuristic"]) for w in self.cal["warnings"]])
        for key in ("epipolar_px", "rectification_vertical_px", "board_square_scale"):
            self.assertEqual(m["metrics"][key], self.cal["metrics"][key])
        self.assertEqual(m["board"]["distance_mm"], self.cal["board"]["distance_mm"])
        self.assertEqual(m["created_utc"], FIXED_TIME)
        self.assertEqual(m["camera"], {"left": {"model": "Left cam", "focal_length_mm": 5.96}, "right": {"model": "Right cam", "focal_length_mm": None}})
        self.assertEqual((m["source"], m["authenticity_verified"], m["valid"], m["errors"]), ("loaded_file", False, True, []))

    def test_full_float64_precision_survives(self):
        """Awkward last-digit values (not 'nice' numbers) come back identical."""
        cal = dict(self.cal)
        cal["K1"] = self.cal["K1"] * (1 + 1e-13)
        cal["K1"][2] = [0, 0, 1]
        cal["T"] = self.cal["T"] * (1 + 1e-12)
        cal["baseline_mm"] = float(np.linalg.norm(cal["T"]))
        m = cio.load_calibration(cio.export_calibration(cal, created=FIXED_TIME))
        np.testing.assert_array_equal(m["K1"], cal["K1"])
        np.testing.assert_array_equal(m["T"], cal["T"])
        self.assertEqual(m["baseline_mm"], cal["baseline_mm"])

    def test_equal_and_different_resolutions(self):
        self.assertNotEqual(base.LEFT_SIZE, base.RIGHT_SIZE)
        m = cio.load_calibration(self.data)
        self.assertNotEqual(m["image_size_left"], m["image_size_right"])
        rig = base.make_rig()
        left, right, _ = base.sample_views(rig, 10, seed=22, size_l=(4000, 3000), size_r=(4000, 3000))
        cal = uc.calibrate_stereo(left, right, base.PATTERN_T, base.SQ, (4000, 3000), (4000, 3000))
        same = cio.load_calibration(cio.export_calibration(cal))
        self.assertEqual((same["image_size_left"], same["image_size_right"]), ((4000, 3000), (4000, 3000)))

    def test_each_distortion_model_round_trips(self):
        rig = base.make_rig(k_right=(-0.28, 0.45, 0.002, -0.0015, -0.30))
        left, right, _ = base.sample_views(rig, 14, seed=23)
        seen = set()
        for requested in ("radial2", "radial3", "full", "auto"):
            cal = uc.calibrate_stereo(left, right, base.PATTERN_T, base.SQ, base.LEFT_SIZE, base.RIGHT_SIZE, distortion=requested)
            m = cio.load_calibration(cio.export_calibration(cal))
            self.assertEqual((m["distortion_left"], m["distortion_right"]), (cal["distortion_left"], cal["distortion_right"]))
            np.testing.assert_array_equal(m["D2"].ravel(), cal["D2"].ravel())
            self.assertEqual(len(m["distortion_cv"]["right"]), 0 if requested != "auto" else 3)
            seen.add(m["distortion_right"])
            if m["distortion_right"] == "radial2":
                self.assertTrue((m["D2"].ravel()[2:] == 0).all())
        self.assertEqual(seen & {"radial2", "radial3", "full"}, {"radial2", "radial3", "full"})

    def test_failed_cv_entry_round_trips_as_infinity(self):
        cal = dict(self.cal)
        cal["distortion_cv"] = {"left": [dict(t) for t in self.cal["distortion_cv"]["left"]],
                                "right": [dict(t) for t in self.cal["distortion_cv"]["right"]]}
        cal["distortion_cv"]["left"][2]["cv_rms"] = float("inf")
        m = cio.load_calibration(cio.export_calibration(cal))
        self.assertEqual(m["distortion_cv"]["left"][2]["cv_rms"], float("inf"))

    def test_reexporting_a_loaded_calibration_is_stable(self):
        once = cio.load_calibration(self.data)
        twice = cio.load_calibration(cio.export_calibration(once, pairs_uploaded=once["pairs_uploaded"], created=once["created_utc"],
                                                           camera_info=once["camera"]))
        for key in ("K1", "K2", "D1", "D2", "R", "T", "E", "F"):
            np.testing.assert_array_equal(once[key], twice[key])
        self.assertEqual((once["pairs"], once["warnings"], once["camera"], once["created_utc"]), (twice["pairs"], twice["warnings"], twice["camera"], twice["created_utc"]))

    def test_file_contents(self):
        self.assertLess(len(self.data), 20_000)
        self.assertEqual(set(self.fields), set(cio.SPEC))
        with zipfile.ZipFile(io.BytesIO(self.data)) as z:
            self.assertTrue(all(n.endswith(".npy") for n in z.namelist()))
        self.assertFalse({"E", "F", "thumbnail", "image", "photo"} & set(self.fields))        # derived data and photos are never stored
        self.assertTrue(all(v.dtype.kind in "fiuU" for v in self.fields.values()))               # no object arrays
        self.assertEqual(int(self.fields["format_version"]), cio.FORMAT_VERSION)
        self.assertEqual(str(self.fields["kind"]), cio.KIND)
        self.assertEqual(self.data, cio.export_calibration(self.cal, pairs_uploaded=16, created=FIXED_TIME,
                                                           camera_info={"left": {"model": "Left cam", "focal_length_mm": 5.96},
                                                                        "right": {"model": "Right cam", "focal_length_mm": None}}))

    def test_export_requirements(self):
        bad = dict(self.cal, valid=False, errors=["x"])
        with self.assertRaises(ValueError):
            cio.export_calibration(bad)
        incomplete = {k: v for k, v in self.cal.items() if k != "metrics"}
        with self.assertRaises(ValueError):
            cio.export_calibration(incomplete)
        broken = dict(self.cal, R=np.eye(3) * 1.5)                                 # exporter refuses what the loader would refuse
        with self.assertRaises(cio.CalibrationFileError):
            cio.export_calibration(broken)

    def test_inputs_other_than_bytes(self):
        self.assertEqual(cio.load_calibration(bytearray(self.data))["baseline_mm"], self.cal["baseline_mm"])
        self.assertEqual(cio.load_calibration(io.BytesIO(self.data))["baseline_mm"], self.cal["baseline_mm"])


class DerivedGeometryTests(Base):
    def test_essential_and_fundamental_are_recomputed_and_consistent(self):
        m = cio.load_calibration(self.data)
        unit = lambda a: a / np.linalg.norm(a)
        for key in ("E", "F"):
            a, b = unit(m[key]), unit(self.cal[key])
            self.assertLess(min(np.abs(a - b).max(), np.abs(a + b).max()), 5e-5, key)      # same matrix up to scale and sign
        # the recomputed F satisfies the epipolar constraint on the calibration corners themselves
        epi = np.concatenate([uc._epipolar_px(m["F"], m["K1"], m["D1"], m["K2"], m["D2"], l, r) for l, r in zip(self.left, self.right)])
        self.assertLess(np.median(epi), 1.0)
        # E on normalised coordinates
        n1 = cv2.undistortPoints(self.left[0].reshape(-1, 1, 2), m["K1"], m["D1"]).reshape(-1, 2)
        n2 = cv2.undistortPoints(self.right[0].reshape(-1, 1, 2), m["K2"], m["D2"]).reshape(-1, 2)
        resid = np.abs(np.einsum("ij,jk,ik->i", np.c_[n2, np.ones(len(n2))], m["E"], np.c_[n1, np.ones(len(n1))]))
        self.assertLess(np.median(resid) / np.linalg.norm(m["E"]), 1e-3)

    def test_stored_baseline_cannot_override_translation(self):
        self.assertRejected(self.mutated(baseline_mm=self.fields["baseline_mm"] * 1.5), "disagrees")
        m = cio.load_calibration(self.data)
        self.assertEqual(m["baseline_mm"], float(np.linalg.norm(self.fields["T"])))        # derived from T, not trusted from the file


class MeasurementCompatibilityTests(Base):
    def _measure(self, model, i):
        return measure_points(model, self.left[i][CORNERS], self.right[i][CORNERS], (210.0, 150.0))

    def test_loaded_model_measures_identically(self):
        loaded = cio.load_calibration(self.data)
        for i in range(len(self.left)):
            a, b = self._measure(self.cal, i), self._measure(loaded, i)
            self.assertEqual(a["errors"], b["errors"])
            for key in ("mean_Z_depth_mm", "top_width", "bottom_width", "left_height", "right_height", "width_mm", "height_mm",
                        "width_err_mm", "height_err_pct"):
                self.assertAlmostEqual(a["summary"][key], b["summary"][key], delta=1e-9, msg=key)
            np.testing.assert_allclose(a["X"], b["X"], rtol=0, atol=1e-9)
            np.testing.assert_allclose(a["epipolar_px"], b["epipolar_px"], rtol=0, atol=1e-9)
        s = self._measure(loaded, 0)["summary"]
        self.assertLess(abs(s["width_mm"] / 210.0 - 1), 0.03)                                # and it measures a real size
        self.assertLess(abs(s["height_mm"] / 150.0 - 1), 0.03)

    def test_loaded_model_has_what_live_module_expects(self):
        loaded = cio.load_calibration(self.data)
        for key in ("K1", "D1", "K2", "D2", "R", "T", "F"):
            self.assertIn(key, loaded)
        self.assertEqual(loaded["T"].shape, (3, 1))
        self.assertEqual(loaded["D1"].shape, (1, 5))
        self.assertEqual(loaded["F"].shape, (3, 3))
        self.assertIsNot(loaded, load_model_b())                                           # nothing is shared with Model B
        self.assertNotEqual(loaded["baseline_mm"], load_model_b()["baseline_mm"])


class RejectionTests(Base):
    def test_every_required_field(self):
        for key in cio.SPEC:
            with self.subTest(key):
                self.assertRejected(self.mutated(**{key: None}), key)

    def test_unknown_field(self):
        self.assertRejected(self.mutated(extra=np.zeros(2)), "unknown field")

    def test_wrong_shapes(self):
        cases = dict(K_left=np.eye(3)[:2], K_right=np.eye(2), D_left=np.zeros(4), D_right=np.zeros((1, 5)), R=np.eye(4)[:3],
                     T=np.ones(2), image_size_left=np.array([4000, 3000, 1]), image_size_right=np.array([[4000, 3000]]),
                     pattern=np.array([8, 6, 1]), board_distance_mm=np.array([1.0, 2.0, 3.0]), epipolar_px=np.zeros(2),
                     pair_ids=np.ones((2, 7), np.int64), square_mm=np.array([20.0]), format_version=np.array([1]))
        for key, value in cases.items():
            with self.subTest(key):
                self.assertRejected(self.mutated(**{key: value}), key)

    def test_wrong_dtypes(self):
        for key, value in (("K_left", self.fields["K_left"].astype(np.float32).astype(str)), ("R", self.fields["R"].astype(np.int64)),
                           ("image_size_left", np.array([4000.0, 3000.0])), ("pattern", np.array([8.0, 6.0])),
                           ("pairs_valid", np.array(14.0)), ("kind", np.array(5)), ("rms_left", np.array(1))):
            with self.subTest(key):
                self.assertRejected(self.mutated(**{key: value}), key)

    def test_nan_and_infinity(self):
        for key in ("K_left", "K_right", "D_left", "D_right", "R", "T", "rms_left", "rms_right", "rms_stereo", "baseline_mm",
                    "square_mm", "epipolar_px", "board_distance_mm", "cv_rms_left"):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(key=key, bad=bad):
                    a = self.fields[key].copy().astype(np.float64)
                    a.flat[0] = bad
                    self.assertRejected(self.mutated(**{key: a}), "nan or infinite")

    def test_invalid_rotation_matrices(self):
        R = self.fields["R"]
        for name, bad in (("scaled", R * 1.01), ("shear", R + np.array([[0, 0.01, 0], [0, 0, 0], [0, 0, 0]])),
                          ("reflection", np.diag([1.0, 1.0, -1.0]) @ R), ("zeros", np.zeros((3, 3))), ("identity x2", np.eye(3) * 2)):
            with self.subTest(name):
                self.assertRejected(self.mutated(R=bad), "rotation")

    def test_zero_and_degenerate_baseline(self):
        self.assertRejected(self.mutated(T=np.zeros(3), baseline_mm=0.0), "baseline")
        self.assertRejected(self.mutated(T=np.array([1e-9, 0, 0]), baseline_mm=1e-9), "baseline")
        self.assertRejected(self.mutated(T=np.array([1e9, 0, 0]), baseline_mm=1e9), "baseline")

    def test_intrinsics(self):
        K = self.fields["K_left"].copy()
        for name, edit in (("zero fx", lambda k: k.__setitem__((0, 0), 0.0)), ("negative fy", lambda k: k.__setitem__((1, 1), -3000.0)),
                           ("lower triangle", lambda k: k.__setitem__((2, 0), 0.5)), ("K22", lambda k: k.__setitem__((2, 2), 2.0)),
                           ("skew", lambda k: k.__setitem__((0, 1), 0.5 * k[0, 0]))):
            with self.subTest(name):
                k = K.copy()
                edit(k)
                self.assertRejected(self.mutated(K_left=k))
        self.assertRejected(self.mutated(D_left=np.array([1e3, 0, 0, 0, 0])), "implausibly large")

    def test_principal_point_outside_image_is_a_warning(self):
        k = self.fields["K_left"].copy()
        k[0, 2] = 9000.0
        m = cio.load_calibration(self.mutated(K_left=k))
        self.assertTrue(any("principal point" in w for w in m["load_warnings"]))

    def test_invalid_image_sizes(self):
        for value in ([0, 3000], [-4000, 3000], [4000, 0], [8, 3000], [25000, 3000], [4000, 25000]):
            for key in ("image_size_left", "image_size_right"):
                with self.subTest(key=key, value=value):
                    self.assertRejected(self.mutated(**{key: np.array(value, np.int64)}), "image size")

    def test_unsupported_formats(self):
        for value in (np.array(2), np.array(0), np.array(-1), np.array(99)):
            self.assertRejected(self.mutated(format_version=value), "version")
        self.assertRejected(self.mutated(format_version=np.array(1.0)), "format_version")
        self.assertRejected(self.mutated(kind=np.array("camera_calibration")), "not a stereo calibration")

    def test_checkerboard_and_pairs(self):
        for value in ([2, 6], [8, 2], [500, 6], [0, 0], [-8, 6]):
            self.assertRejected(self.mutated(pattern=np.array(value, np.int64)), "checkerboard")
        for value in (0.0, -1.0, 1e6):
            self.assertRejected(self.mutated(square_mm=np.array(value)), "square size")
        self.assertRejected(self.mutated(pairs_valid=np.array(0)), "pair counts")
        self.assertRejected(self.mutated(pairs_valid=np.array(20)), "pair counts")              # more valid than uploaded
        self.assertRejected(self.mutated(pair_ids=np.array([1, 1] + list(range(3, 15)), np.int64)), "pair counts")
        self.assertRejected(self.mutated(pair_ids=np.arange(1, 14, dtype=np.int64)), "pair counts")
        self.assertRejected(self.mutated(pair_ids=np.arange(0, 14, dtype=np.int64)), "pair counts")

    def test_few_valid_pairs_is_a_warning_not_an_error(self):
        f = {k: v.copy() for k, v in self.fields.items()}
        f.update(pairs_valid=np.array(5), pair_ids=np.arange(1, 6, dtype=np.int64))
        m = cio.load_calibration(pack(f))
        self.assertTrue(any("fewer than the recommended" in w for w in m["load_warnings"]))

    def test_distortion_metadata(self):
        self.assertRejected(self.mutated(distortion_model_left=np.array("fisheye")), "unsupported")
        d = self.fields["D_left"].copy()
        d[4] = 0.01
        self.assertRejected(self.mutated(distortion_model_left=np.array("radial2"), D_left=d), "do not match")
        d = np.array([0.1, -0.1, 0.001, 0, 0])
        self.assertRejected(self.mutated(distortion_model_right=np.array("radial3"), D_right=d), "do not match")
        self.assertRejected(self.mutated(cv_models=np.array(["radial2", "full"]), cv_rms_left=np.ones(2), cv_rms_right=np.ones(2)), "cross-validation")
        self.assertRejected(self.mutated(cv_models=np.array(["radial2", "radial3", "bogus"])), "cross-validation")
        self.assertRejected(self.mutated(cv_rms_left=np.array([-2.0, 1.0, 1.0])), "cv_rms_left")
        self.assertRejected(self.mutated(cv_rms_right=np.ones(2)), "cv_rms_right")
        empty = dict(cv_models=np.array([], dtype="U20"), cv_rms_left=np.zeros(0), cv_rms_right=np.zeros(0))
        self.assertEqual(cio.load_calibration(self.mutated(**empty))["distortion_cv"], {"left": [], "right": []})

    def test_metrics_and_text_fields(self):
        self.assertRejected(self.mutated(rms_left=np.array(-0.1)), "rms_left")
        self.assertRejected(self.mutated(rms_stereo=np.array(5000.0)), "rms_stereo")
        self.assertRejected(self.mutated(epipolar_px=np.array([1.0, 5.0, 2.0])), "epipolar_px")      # median above max
        self.assertRejected(self.mutated(rectification_px=np.array([-1.0, 2.0, 3.0])), "rectification_px")
        self.assertRejected(self.mutated(board_distance_mm=np.array([900.0, 500.0])), "board distance")
        self.assertRejected(self.mutated(warnings_json=np.array("{not json")), "json")
        self.assertRejected(self.mutated(warnings_json=np.array('{"a": 1}')), "warnings are malformed")
        self.assertRejected(self.mutated(warnings_json=np.array(json.dumps([dict(code="x", message="m", heuristic=True)] * 60))), "malformed")
        self.assertRejected(self.mutated(warnings_json=np.array(json.dumps([dict(code="x", message="m" * 5000, heuristic=True)]))), "malformed")
        self.assertRejected(self.mutated(warnings_json=np.array(json.dumps([dict(code="x", message="m")]))), "malformed")
        self.assertRejected(self.mutated(camera_json=np.array(json.dumps({"middle": {}}))), "camera")
        self.assertRejected(self.mutated(camera_json=np.array(json.dumps({"left": {"model": ["x"]}}))), "camera")
        self.assertRejected(self.mutated(camera_json=np.array("x" * 30000)), "too long")
        self.assertRejected(self.mutated(created_utc=np.array("yesterday")), "timestamp")
        self.assertRejected(self.mutated(software=np.array("s" * 500)), "too long")


class ArchiveSafetyTests(Base):
    def test_oversized_file(self):
        self.assertRejected(self.data + b"\0" * (cio.MAX_FILE_BYTES + 1), "mb")

    def test_malformed_files(self):
        for name, data in (("empty", b""), ("random", np.random.default_rng(0).bytes(5000)), ("text", b"hello world"),
                           ("truncated", self.data[: len(self.data) // 2]), ("header only", self.data[:60]),
                           ("png header", b"\x89PNG\r\n\x1a\n" + b"\0" * 100), ("json", b'{"kind": "stereo_calibration"}')):
            with self.subTest(name):
                self.assertRejected(data)

    def test_bit_flips_never_crash(self):
        rng = np.random.default_rng(3)
        for _ in range(60):
            b = bytearray(self.data)
            for pos in rng.integers(0, len(b), 5):
                b[pos] ^= int(rng.integers(1, 255))
            try:
                cio.load_calibration(bytes(b))
            except cio.CalibrationFileError:
                pass                                       # rejected cleanly; any other exception type fails the test

    def test_non_array_and_hostile_zip_members(self):
        for name in ("evil.py", "../escape.npy", "/abs.npy", "sub/dir.npy", ".hidden.npy", "notes.txt"):
            with self.subTest(name):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    for k, v in self.fields.items():
                        b = io.BytesIO()
                        np.save(b, v)
                        z.writestr(k + ".npy", b.getvalue())
                    z.writestr(name, b"payload")
                self.assertRejected(buf.getvalue(), "unexpected entry")

    def test_zip_bomb(self):
        f = {k: v.copy() for k, v in self.fields.items()}
        f["R"] = np.zeros(1_000_000)                        # 8 MB unpacked, a few KB compressed
        data = pack(f, compressed=True)
        self.assertLess(len(data), cio.MAX_FILE_BYTES)
        self.assertRejected(data, "too large when unpacked")

    def test_many_members(self):
        f = {k: v.copy() for k, v in self.fields.items()}
        f.update({f"x{i}": np.zeros(1) for i in range(100)})
        self.assertRejected(pack(f), "number of entries")

    def test_pickled_object_arrays_are_refused_and_never_run(self):
        EXECUTED["flag"] = False
        for payload in (np.array([Evil()], dtype=object), np.array([{"a": 1}], dtype=object), np.array(Evil(), dtype=object)):
            self.assertRejected(self.mutated(K_left=payload), "read safely")
            self.assertRejected(self.mutated(kind=payload), "read safely")
        self.assertRejected(pack({"anything": np.array([Evil()], dtype=object)}))
        self.assertFalse(EXECUTED["flag"], "the loader must never unpickle")

    def test_unrelated_npz_and_npy(self):
        self.assertRejected(pack({"a": np.zeros(3)}), "unknown field")
        buf = io.BytesIO()
        np.save(buf, np.zeros(3))
        self.assertRejected(buf.getvalue(), "not an npz")


@unittest.skipUnless((CALIB_PHOTO_DIR / "left" / "air_1.jpeg").is_file(), "original calibration photographs are not present")
class RealCalibrationRoundTripTests(unittest.TestCase):
    """Calibrate from the author's 30 pairs, export, reload, and measure the notebook with both: results must agree to float precision."""

    @classmethod
    def setUpClass(cls):
        photos = {s: [uc.process_calibration_photo((CALIB_PHOTO_DIR / s / f"{p}_{i}.jpeg").read_bytes(), PATTERN) for i in range(1, 31)]
                  for s, p in (("left", "air"), ("right", "pro"))}
        order = uc.resolve_pair_orderings([(l["corners"], r["corners"]) for l, r in zip(photos["left"], photos["right"])], PATTERN,
                                          max(photos["left"][0]["size"]))
        cls.cal = uc.calibrate_stereo([l["corners"] for l in photos["left"]], [p["right_corners"] for p in order["pairs"]], PATTERN,
                                      SQUARE_MM, photos["left"][0]["size"], photos["right"][0]["size"])
        cls.data = cio.export_calibration(cls.cal, camera_info={"left": photos["left"][0]["exif"], "right": photos["right"][0]["exif"]})
        cls.loaded = cio.load_calibration(cls.data)
        cls.model_b = load_model_b()

    def test_round_trip_differences_are_zero(self):
        worst = max(float(np.abs(self.loaded[k] - self.cal[k]).max()) for k in ("K1", "K2", "R"))
        worst = max(worst, float(np.abs(self.loaded["D1"].ravel() - self.cal["D1"].ravel()).max()),
                    float(np.abs(self.loaded["D2"].ravel() - self.cal["D2"].ravel()).max()),
                    float(np.abs(self.loaded["T"].ravel() - self.cal["T"].ravel()).max()))
        self.assertEqual(worst, 0.0)
        self.assertEqual(self.loaded["camera"]["left"]["model"], "iPhone Air")
        self.assertEqual(self.loaded["camera"]["right"]["model"], "iPhone 17 Pro Max")

    def test_notebook_measurements_reproduce(self):
        for pair in NOTEBOOK_PAIR_IDS:
            ref = measure_pair(self.model_b, pair)
            a = measure_points(self.cal, ref["left_px"], ref["right_px"], (305.0, 215.0))
            b = measure_points(self.loaded, ref["left_px"], ref["right_px"], (305.0, 215.0))
            for key in ("mean_Z_depth_mm", "width_mm", "height_mm", "top_width", "bottom_width", "left_height", "right_height"):
                self.assertAlmostEqual(a["summary"][key], b["summary"][key], delta=1e-9, msg=f"pair {pair} {key}")
            np.testing.assert_allclose(a["X"], b["X"], rtol=0, atol=1e-9)

    def test_model_b_untouched_and_not_used(self):
        ref = json.loads((MODULE / "reference" / "expected_model_b_results.json").read_text())
        sha = hashlib.sha256((MODULE / "calibration" / "model_b" / "stereo_calibration_30pairs.npz").read_bytes()).hexdigest()
        self.assertEqual(sha, ref["model_b_npz_sha256"])
        self.assertNotEqual(self.loaded["baseline_mm"], self.model_b["baseline_mm"])


if __name__ == "__main__":
    unittest.main()
