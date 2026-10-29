"""Phase 3 tests: object-measurement validation and thresholds driven by the active calibration (selection.py, live.py)."""
import copy
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(ROOT / "tests"))

import test_module7_user_calibration as base  # noqa: E402  (synthetic rigs; only the helpers are used)
from stereo_measure import calibration_io as cio  # noqa: E402
from stereo_measure import live  # noqa: E402
from stereo_measure import load_model_b, measure_pair  # noqa: E402
from stereo_measure import selection as sel  # noqa: E402
from stereo_measure import user_calibration as uc  # noqa: E402
from stereo_measure.config import CALIB_PHOTO_DIR, IMAGE_SIZE, NOTEBOOK_PAIR_IDS, NOTEBOOK_PHOTO_DIR, PATTERN, SQUARE_MM  # noqa: E402

CORNERS = [0, 7, 47, 40]                 # TL, TR, BR, BL of an 8 x 6 inner-corner grid
EXPECTED = dict(make="Apple", model="iPhone Air", lens_model="iPhone Air back camera 5.96mm f/1.6", focal_length_mm=5.96,
                focal_length_35mm=26.0, orientation=6)
OTHER = dict(make="Apple", model="iPhone 17 Pro Max", lens_model="iPhone 17 Pro Max back triple camera 6.765mm f/1.78",
             focal_length_mm=6.765, focal_length_35mm=24.0, orientation=6)


def photo(stored_size, orientation=None, fmt="JPEG", **exif_fields):
    """A small test photograph with chosen EXIF (make, model, lens_model, focal_length_mm, focal_length_35mm)."""
    exif = Image.Exif()
    if orientation:
        exif[274] = orientation
    for tag, key in ((271, "make"), (272, "model")):
        if key in exif_fields:
            exif[tag] = exif_fields[key]
    sub = exif.get_ifd(0x8769)
    for tag, key in ((0xA434, "lens_model"), (0x920A, "focal_length_mm"), (0xA405, "focal_length_35mm")):
        if key in exif_fields:
            sub[tag] = exif_fields[key]
    rng = np.random.default_rng(sum(stored_size))
    arr = rng.integers(0, 255, (stored_size[1], stored_size[0]), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **({"exif": exif} if fmt == "JPEG" else {}))
    return buf.getvalue()


def codes(report):
    return [w["code"] for w in report["warnings"]]


class SizeValidationTests(unittest.TestCase):
    LEFT, RIGHT = (640, 480), (480, 360)              # different resolutions for the two cameras
    CAL = dict(image_size_left=LEFT, image_size_right=RIGHT)

    def test_each_image_is_checked_against_its_own_camera(self):
        left = sel.check_upload_for_calibration(photo(self.LEFT), "left", self.CAL)
        right = sel.check_upload_for_calibration(photo(self.RIGHT), "right", self.CAL)
        self.assertEqual((left["size"], right["size"]), (self.LEFT, self.RIGHT))
        self.assertFalse(left["metadata_compared"])                         # calibration recorded no camera metadata

    def test_swapped_sides_are_rejected_with_a_hint(self):
        with self.assertRaises(sel.UploadError) as ctx:
            sel.check_upload_for_calibration(photo(self.RIGHT), "left", self.CAL)
        self.assertIn("640 x 480", str(ctx.exception))
        self.assertIn("may be swapped", str(ctx.exception))
        with self.assertRaises(sel.UploadError):
            sel.check_upload_for_calibration(photo(self.LEFT), "right", self.CAL)

    def test_wrong_sizes_explain_themselves(self):
        cases = {"rotated": (photo((480, 640)), "rotated by 90"), "half resolution": (photo((320, 240)), "resolution differs"),
                 "other shape": (photo((600, 400)), "uses 640 x 480")}
        for name, (data, fragment) in cases.items():
            with self.subTest(name), self.assertRaises(sel.UploadError) as ctx:
                sel.check_upload(data, "left", self.LEFT, None)
            self.assertIn(fragment, str(ctx.exception))
            self.assertIn("does not resize or rotate", str(ctx.exception))
        with self.assertRaises(sel.UploadError) as ctx:
            sel.check_upload(photo((480, 640), orientation=3), "left", self.LEFT, None)
        self.assertIn("orientation tag 3", str(ctx.exception))               # the message reports what the file says
        self.assertIn("stored 480 x 640", str(ctx.exception))

    def test_exif_orientation_is_applied_before_comparing(self):
        ew, eh = self.LEFT
        for orientation in range(1, 9):
            stored = (eh, ew) if orientation in (5, 6, 7, 8) else (ew, eh)
            with self.subTest(orientation=orientation):
                report = sel.check_upload(photo(stored, orientation=orientation), "left", self.LEFT, None)
                self.assertEqual((report["size"], report["orientation"]), (self.LEFT, orientation))
            wrong = (ew, eh) if orientation in (5, 6, 7, 8) else (eh, ew)           # stored the other way round: oriented size differs
            with self.subTest(wrong=orientation), self.assertRaises(sel.UploadError):
                sel.check_upload(photo(wrong, orientation=orientation), "left", self.LEFT, None)
        self.assertEqual(sel.check_upload(photo((ew, eh)), "left", self.LEFT, None)["orientation"], 1)         # no tag = upright

    def test_decode_uses_the_calibration_frame_and_never_resizes(self):
        for size, orientation in (((640, 480), None), ((480, 640), 6), ((480, 360), None)):
            expected = (size[1], size[0]) if orientation == 6 else size
            arr = sel.decode_upload(photo(size, orientation=orientation), expected)
            self.assertEqual((arr.shape[1], arr.shape[0]), expected)
        with self.assertRaises(sel.UploadError):
            sel.decode_upload(photo((640, 480)), (480, 360))
        with self.assertRaises(sel.UploadError):
            sel.decode_upload(photo((640, 480)), IMAGE_SIZE)                       # a frame that does not match the stated size

    def test_format_and_size_limits(self):
        with self.assertRaises(sel.UploadError) as ctx:
            sel.check_upload(b"\0" * (uc.MAX_FILE_BYTES + 1), "left", self.LEFT, None)
        self.assertIn("30 MB", str(ctx.exception))
        big = io.BytesIO()
        Image.new("L", (8000, 7000)).save(big, "PNG")
        with self.assertRaises(sel.UploadError) as ctx:
            sel.check_upload(big.getvalue(), "left", (8000, 7000), None)
        self.assertIn("megapixels", str(ctx.exception))
        gif = io.BytesIO()
        Image.new("P", (640, 480)).save(gif, "GIF")
        for data, word in ((gif.getvalue(), "Unsupported format"), (b"", "empty"), (b"junk", "could not be read")):
            with self.subTest(word), self.assertRaises(sel.UploadError) as ctx:
                sel.check_upload(data, "left", self.LEFT, None)
            self.assertIn(word, str(ctx.exception))
        self.assertEqual(sel.check_upload(photo((640, 480), fmt="PNG"), "left", self.LEFT, None)["size"], self.LEFT)
        with self.assertRaises(ValueError):
            sel.check_upload(photo(self.LEFT), "middle", self.LEFT, None)

    def test_calibration_arguments_are_required(self):
        # No implicit Model B: omitting the expected size / camera / frame is a TypeError, never a silent fallback.
        for call in (lambda: sel.inspect_upload(photo(IMAGE_SIZE), "left"),
                     lambda: sel.inspect_upload(photo(IMAGE_SIZE), "left", IMAGE_SIZE),
                     lambda: sel.check_upload(photo(IMAGE_SIZE), "left"),
                     lambda: sel.check_upload(photo(IMAGE_SIZE), "left", IMAGE_SIZE),
                     lambda: sel.decode_upload(photo(IMAGE_SIZE))):
            with self.assertRaises(TypeError):
                call()

    def test_explicit_expectation_is_used_completely(self):
        size, warnings = sel.inspect_upload(photo(self.LEFT), "left", self.LEFT, None)
        self.assertEqual((size, warnings), (self.LEFT, []))
        with self.assertRaises(sel.UploadError):                                       # Model B's size is not assumed
            sel.inspect_upload(photo(self.LEFT), "left", IMAGE_SIZE, None)

    def test_expected_for_side(self):
        self.assertEqual(sel.expected_for_side(self.CAL, "right"), dict(size=self.RIGHT, camera=None))
        model_b = load_model_b()
        for side in ("left", "right"):
            self.assertEqual(sel.expected_for_side(model_b, side), dict(size=IMAGE_SIZE, camera=None))
        cal = dict(self.CAL, camera=dict(left=dict(EXPECTED)))
        self.assertEqual(sel.expected_for_side(cal, "left")["camera"], EXPECTED)
        self.assertIsNone(sel.expected_for_side(cal, "right")["camera"])
        with self.assertRaises(ValueError):                                           # never silently assumes a size
            sel.expected_for_side({}, "left")
        with self.assertRaises(ValueError):
            sel.check_upload_for_calibration(photo(IMAGE_SIZE), "left", {})


class MetadataTests(unittest.TestCase):
    SIZE = (640, 480)

    def check(self, data, expected=EXPECTED, other=OTHER, side="left"):
        return sel.check_upload(data, side, self.SIZE, expected, self.SIZE, other)

    def upright(self, **kw):
        fields = {**EXPECTED, **kw}
        orientation = fields.pop("orientation")
        return photo((self.SIZE[1], self.SIZE[0]), orientation=orientation, **fields)       # stored portrait + tag 6 -> 640 x 480

    def test_matching_metadata_gives_no_warning(self):
        report = self.check(self.upright())
        self.assertEqual(report["warnings"], [])
        self.assertTrue(report["metadata_compared"])
        self.assertEqual(report["exif"]["model"], "iPhone Air")

    def test_mismatches_are_warnings_never_errors(self):
        cases = {"model_mismatch": dict(model="Pixel 8"), "make_mismatch": dict(make="Google"), "lens_model_mismatch": dict(lens_model="ultra wide"),
                 "focal_length_mm_mismatch": dict(focal_length_mm=2.2), "focal_length_35mm_mismatch": dict(focal_length_35mm=13.0)}
        for code, change in cases.items():
            with self.subTest(code):
                report = self.check(self.upright(**change))            # does not raise
                self.assertIn(code, codes(report))
                self.assertEqual(len([c for c in codes(report) if c.endswith("_mismatch")]), 1)
                self.assertTrue(all("calibration" in w["message"] for w in report["warnings"]))

    def test_small_focal_difference_is_tolerated(self):
        self.assertEqual(self.check(self.upright(focal_length_mm=5.97))["warnings"], [])
        self.assertEqual(self.check(self.upright(focal_length_mm=5.96 + 0.2))["warnings"][0]["code"], "focal_length_mm_mismatch")

    def test_text_comparison_ignores_case_and_spacing(self):
        self.assertEqual(self.check(self.upright(model="  IPHONE air "))["warnings"], [])

    def test_swapped_cameras_hint(self):
        report = self.check(self.upright(model="iPhone 17 Pro Max"))
        msg = [w["message"] for w in report["warnings"] if w["code"] == "model_mismatch"][0]
        self.assertIn("may be swapped", msg)
        self.assertIn("right-camera", msg)

    def test_orientation_tag_difference_with_same_size(self):
        report = self.check(photo((self.SIZE[1], self.SIZE[0]), orientation=8, **{k: v for k, v in EXPECTED.items() if k != "orientation"}))
        self.assertEqual(codes(report), ["orientation_differs"])                    # 8 and 6 both give 640 x 480
        self.assertEqual(report["size"], self.SIZE)

    def test_missing_exif(self):
        report = self.check(photo(self.SIZE))
        self.assertEqual(codes(report), ["no_exif"])
        self.assertIn("could not be compared", report["warnings"][0]["message"])
        partial = self.check(photo((self.SIZE[1], self.SIZE[0]), orientation=6, make="Apple", model="iPhone Air"))
        self.assertEqual(codes(partial), ["metadata_incomplete"])
        self.assertIn("lens", partial["warnings"][0]["message"])
        self.assertIn("focal length", partial["warnings"][0]["message"])

    def test_no_reference_metadata_means_no_comparison(self):
        for expected in (None, {}, {"n_photos": 30}, {"model": ""}):
            with self.subTest(expected=expected):
                report = self.check(photo(self.SIZE), expected=expected)
                self.assertEqual(report["warnings"], [])
                self.assertFalse(report["metadata_compared"])

    def test_caveat_does_not_overclaim(self):
        text = sel.METADATA_CAVEAT.lower()
        self.assertIn("does not prove", text)
        self.assertIn("physical stereo rig", text)

    def test_camera_info_from_photos(self):
        good = [dict(exif=dict(EXPECTED)) for _ in range(5)]
        info, notes = uc.camera_info_from_photos(good)
        self.assertEqual({k: v for k, v in info.items() if k != "n_photos"}, EXPECTED)
        self.assertEqual((info["n_photos"], notes), (5, []))
        mixed = [dict(exif=dict(EXPECTED)) for _ in range(4)] + [dict(exif=dict(EXPECTED, model="Other"))]
        info, notes = uc.camera_info_from_photos(mixed)
        self.assertEqual(info["model"], "iPhone Air")
        self.assertTrue(any("camera model" in n for n in notes))
        info, notes = uc.camera_info_from_photos([dict(exif=dict(EXPECTED)), dict(exif=None)])
        self.assertEqual(info["model"], "iPhone Air")
        self.assertTrue(any("no readable metadata" in n for n in notes))
        info, notes = uc.camera_info_from_photos([dict(exif=None)])
        self.assertEqual(info, {"n_photos": 1})

    def test_calibration_camera_info_survives_save_and_load(self):
        rig = base.make_rig()
        left, right, _ = base.sample_views(rig, 10, seed=31)
        cal = uc.calibrate_stereo(left, right, base.PATTERN_T, base.SQ, base.LEFT_SIZE, base.RIGHT_SIZE)
        loaded = cio.load_calibration(cio.export_calibration(cal, camera_info=dict(left=EXPECTED, right=OTHER)))
        self.assertEqual(loaded["camera"], dict(left=EXPECTED, right=OTHER))
        ok = sel.check_upload_for_calibration(photo((480, 640), orientation=6, **{k: v for k, v in EXPECTED.items() if k != "orientation"}),
                                              "left", dict(loaded, image_size_left=(640, 480), image_size_right=(640, 480)))
        self.assertEqual(ok["warnings"], [])


class ThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig = base.make_rig()
        cls.left, cls.right, _ = base.sample_views(cls.rig, 14, seed=32)
        cls.cal = uc.calibrate_stereo(cls.left, cls.right, base.PATTERN_T, base.SQ, base.LEFT_SIZE, base.RIGHT_SIZE)
        cls.loaded = cio.load_calibration(cio.export_calibration(cls.cal))
        cls.model_b = load_model_b()

    def corners(self, i=0, shift=0.0):
        r = self.right[i][CORNERS].copy()
        r[:, 1] += shift
        return self.left[i][CORNERS], r

    def test_calibration_derived_thresholds(self):
        th = live.thresholds_from_calibration(self.cal)
        cal_max = self.cal["metrics"]["epipolar_px"]["max"]
        self.assertEqual(th["source"], "calibration")
        self.assertEqual(th["epipolar_basis"], "calibration_residual")
        self.assertAlmostEqual(th["epipolar_warn_px"], max(live.EPIPOLAR_FLOOR_FRACTION * 4000, live.EPIPOLAR_FACTOR * cal_max))
        d = self.cal["board"]["distance_mm"]
        self.assertEqual(th["depth_range_mm"], (0.5 * d["min"], 2.0 * d["max"]))
        self.assertEqual(th["depth_basis"], "calibration_board_distance")
        self.assertEqual(live.thresholds_from_calibration(self.loaded), th)               # a reloaded calibration gives the same thresholds

    def test_epipolar_threshold_scales_with_the_calibrations_own_residual(self):
        noisy = copy.deepcopy(self.cal)
        noisy["metrics"]["epipolar_px"] = dict(mean=10.0, median=10.0, max=40.0)
        self.assertEqual(live.thresholds_from_calibration(noisy)["epipolar_warn_px"], 120.0)       # 3 x worst residual beats the floor
        sharp = copy.deepcopy(self.cal)
        sharp["metrics"]["epipolar_px"] = dict(mean=0.1, median=0.1, max=0.2)
        self.assertEqual(live.thresholds_from_calibration(sharp)["epipolar_warn_px"], 0.004 * 4000)   # floor
        small = copy.deepcopy(self.cal)
        small.update(image_size_left=(1000, 800), image_size_right=(900, 700))
        small["metrics"]["epipolar_px"] = dict(mean=0.1, median=0.1, max=0.2)
        self.assertEqual(live.thresholds_from_calibration(small)["epipolar_warn_px"], 4.0)          # floor follows the image size

    def test_model_b_keeps_the_fixed_defaults(self):
        th = live.thresholds_from_calibration(self.model_b)
        self.assertEqual((th["source"], th["epipolar_warn_px"], th["epipolar_basis"], th["depth_range_mm"], th["depth_basis"]),
                         ("default", 150.0, "raw_right_image", (200.0, 3000.0), "default"))
        bare = {k: v for k, v in self.cal.items() if k not in ("metrics", "board")}
        self.assertEqual(live.thresholds_from_calibration(bare)["source"], "default")        # nothing to derive from -> defaults
        only_board = {k: v for k, v in self.cal.items() if k != "metrics"}
        th = live.thresholds_from_calibration(only_board)
        self.assertEqual((th["epipolar_basis"], th["depth_basis"]), ("raw_right_image", "calibration_board_distance"))

    def test_clean_correspondences_warn_nothing(self):
        for model in (self.cal, self.loaded):
            res = live.measure_points(model, *self.corners(0))
            self.assertEqual((res["errors"], res["warnings"]), ([], []))

    def test_epipolar_warning_uses_calibration_threshold_and_does_not_reject(self):
        res = live.measure_points(self.cal, *self.corners(0, shift=40.0), truth=(210.0, 150.0))
        self.assertTrue(any("epipolar" in w for w in res["warnings"]), res["warnings"])
        self.assertIn("still calculated", " ".join(res["warnings"]))
        self.assertEqual(res["errors"], [])                                    # a heuristic never blocks
        self.assertTrue(res["summary"]["valid"])
        self.assertIn("width_mm", res["summary"])
        self.assertIn("width_err_pct", res["summary"])
        self.assertGreater(res["epipolar_residual_px"].max(), res["thresholds"]["epipolar_warn_px"])
        # the same shift is within Model B's fixed 150 px default -> that model stays silent
        refm = measure_pair(self.model_b, 1)
        shifted = refm["right_px"].copy()
        shifted[:, 1] += 40.0
        self.assertEqual(live.measure_points(self.model_b, refm["left_px"], shifted)["warnings"], [])

    def test_overriding_thresholds(self):
        loose = live.measure_points(self.cal, *self.corners(0, 40.0), thresholds={"epipolar_warn_px": 500.0})
        self.assertEqual(loose["warnings"], [])
        self.assertEqual(loose["thresholds"]["source"], "custom")
        off = live.measure_points(self.cal, *self.corners(0, 40.0), thresholds={"epipolar_warn_px": None, "depth_range_mm": None})
        self.assertEqual(off["warnings"], [])
        tight = live.measure_points(self.cal, *self.corners(0, 0.0), thresholds={"epipolar_warn_px": 1e-6})
        self.assertTrue(any("epipolar" in w for w in tight["warnings"]))
        for bad in ({"nonsense": 1}, {"epipolar_warn_px": -1}, {"epipolar_warn_px": float("nan")}, {"epipolar_basis": "other"},
                    {"depth_range_mm": (500, 100)}, {"depth_range_mm": (0, 100)}, {"depth_range_mm": (1, 2, 3)}, "text"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                live.measure_points(self.cal, *self.corners(0), thresholds=bad)

    def test_depth_warning_uses_calibration_board_distances(self):
        mean_z = live.measure_points(self.cal, *self.corners(0))["summary"]["mean_Z_depth_mm"]
        near = copy.deepcopy(self.cal)
        near["board"]["distance_mm"] = dict(min=100.0, max=120.0)                 # calibrated only very close: range (50, 240)
        res = live.measure_points(near, *self.corners(0), truth=(210.0, 150.0))
        self.assertTrue(any("board distances used in calibration (100-120 mm)" in w for w in res["warnings"]), res["warnings"])
        self.assertEqual(res["errors"], [])                                       # still measured
        self.assertAlmostEqual(res["summary"]["mean_Z_depth_mm"], mean_z, delta=1e-9)
        wide = copy.deepcopy(self.cal)
        wide["board"]["distance_mm"] = dict(min=300.0, max=2000.0)
        self.assertEqual(live.measure_points(wide, *self.corners(0))["warnings"], [])

    def test_default_depth_message_is_unchanged_for_models_without_calibration_info(self):
        bare = {k: v for k, v in self.cal.items() if k not in ("metrics", "board")}
        with mock.patch.object(live, "DEPTH_WARN_MM", (800.0, 1000.0)):
            res = live.measure_points(bare, *self.corners(0))
        self.assertTrue(any("~700-850 mm range this calibration was tested at" in w for w in res["warnings"]))
        res = live.measure_points(self.cal, *self.corners(0), thresholds={"depth_range_mm": (800.0, 1000.0)})
        self.assertTrue(any("outside the allowed range 800-1000 mm" in w for w in res["warnings"]))

    def test_thresholds_do_not_change_the_mathematics(self):
        base_res = live.measure_points(self.cal, *self.corners(2), truth=(210.0, 150.0))
        for th in ({"epipolar_warn_px": 1e-6, "depth_range_mm": (1.0, 2.0)}, {"epipolar_warn_px": None, "depth_range_mm": None}):
            other = live.measure_points(self.cal, *self.corners(2), truth=(210.0, 150.0), thresholds=th)
            np.testing.assert_array_equal(base_res["X"], other["X"])
            self.assertEqual(base_res["summary"], other["summary"])
            self.assertEqual(base_res["edges"], other["edges"])


class PipelineCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rig = base.make_rig()
        cls.left, cls.right, _ = base.sample_views(rig, 14, seed=33)
        cls.cal = uc.calibrate_stereo(cls.left, cls.right, base.PATTERN_T, base.SQ, base.LEFT_SIZE, base.RIGHT_SIZE)
        cls.loaded = cio.load_calibration(cio.export_calibration(cls.cal))
        cls.model_b = load_model_b()

    def test_computed_and_reloaded_calibrations_measure_identically(self):
        for i in range(len(self.left)):
            a = live.measure_points(self.cal, self.left[i][CORNERS], self.right[i][CORNERS], (210.0, 150.0))
            b = live.measure_points(self.loaded, self.left[i][CORNERS], self.right[i][CORNERS], (210.0, 150.0))
            self.assertEqual((a["errors"], a["warnings"], a["summary"]), (b["errors"], b["warnings"], b["summary"]))
            np.testing.assert_array_equal(a["X"], b["X"])
            np.testing.assert_allclose(a["epipolar_residual_px"], b["epipolar_residual_px"], rtol=0, atol=1e-9)   # F is recomputed on load
            self.assertEqual(a["thresholds"], b["thresholds"])
            self.assertLess(abs(a["summary"]["width_mm"] / 210 - 1), 0.03)

    def test_model_b_notebook_measurements_are_exactly_as_before(self):
        saved = pd.read_csv(MODULE / "outputs" / "pair_summary.csv").set_index("pair")
        for pair in NOTEBOOK_PAIR_IDS:
            ref = measure_pair(self.model_b, pair)
            res = live.measure_points(self.model_b, ref["left_px"], ref["right_px"], (305.0, 215.0))
            np.testing.assert_array_equal(res["X"], ref["X"])
            np.testing.assert_array_equal(res["Xr"], ref["Xr"])
            for key in ("mean_Z_depth_mm", "top_width", "bottom_width", "left_height", "right_height", "width_mm", "height_mm",
                        "width_err_mm", "width_err_pct", "height_err_mm", "height_err_pct"):
                self.assertEqual(res["summary"][key], ref["summary"][key], key)             # bit-for-bit
            for key in ("mean_Z_depth_mm", "width_mm", "height_mm"):
                self.assertAlmostEqual(res["summary"][key], saved.loc[pair, key], places=3)   # and the saved CSV (4 decimals)
            self.assertEqual((res["errors"], res["warnings"]), ([], []))
            self.assertEqual(res["thresholds"]["source"], "default")

    def test_model_b_epipolar_diagnostic_is_unchanged(self):
        ref = measure_pair(self.model_b, 1)
        res = live.measure_points(self.model_b, ref["left_px"], ref["right_px"])
        np.testing.assert_allclose(res["epipolar_px"], live.epipolar_distance_px(self.model_b, ref["left_px"], ref["right_px"]), rtol=0, atol=0)
        # the legacy raw-pixel diagnostic and the undistorted residual are different quantities
        self.assertFalse(np.allclose(res["epipolar_px"], res["epipolar_residual_px"]))

    def test_no_model_b_fallback_anywhere(self):
        for forbidden in ("load_model_b", "get_model"):
            for module in (live, sel):
                self.assertNotIn(forbidden, vars(module), f"{module.__name__} must not reference {forbidden}")
        with self.assertRaises(ValueError):
            sel.check_upload_for_calibration(photo((640, 480)), "left", {})       # a calibration with no size is an error, not Model B


@unittest.skipUnless((CALIB_PHOTO_DIR / "left" / "air_1.jpeg").is_file(), "original photographs are not present")
class RealDataTests(unittest.TestCase):
    """Calibrate from the author's real calibration photos as if they were a visitor's, then validate and measure the notebook."""

    @classmethod
    def setUpClass(cls):
        photos = {s: [uc.process_calibration_photo((CALIB_PHOTO_DIR / s / f"{p}_{i}.jpeg").read_bytes(), PATTERN) for i in range(1, 31)]
                  for s, p in (("left", "air"), ("right", "pro"))}
        order = uc.resolve_pair_orderings([(l["corners"], r["corners"]) for l, r in zip(photos["left"], photos["right"])], PATTERN,
                                          max(photos["left"][0]["size"]))
        cal = uc.calibrate_stereo([l["corners"] for l in photos["left"]], [p["right_corners"] for p in order["pairs"]], PATTERN, SQUARE_MM,
                                  photos["left"][0]["size"], photos["right"][0]["size"])
        cal["camera"] = dict(left=uc.camera_info_from_photos(photos["left"])[0], right=uc.camera_info_from_photos(photos["right"])[0])
        cls.cal, cls.model_b = cal, load_model_b()
        cls.loaded = cio.load_calibration(cio.export_calibration(cal, camera_info=cal["camera"]))

    def photo_bytes(self, side, pair):
        name = ("air_image" if side == "left" else "pro_image") + f"{pair}.jpeg"
        return (NOTEBOOK_PHOTO_DIR / side / name).read_bytes()

    def test_notebook_photos_validate_against_the_derived_calibration(self):
        for cal in (self.cal, self.loaded):
            for pair in NOTEBOOK_PAIR_IDS:
                for side in ("left", "right"):
                    report = sel.check_upload_for_calibration(self.photo_bytes(side, pair), side, cal)
                    self.assertEqual(report["size"], IMAGE_SIZE)
                    self.assertEqual(report["warnings"], [], (pair, side))
                    self.assertTrue(report["metadata_compared"])

    def test_swapped_notebook_photos_are_warned_about(self):
        report = sel.check_upload_for_calibration(self.photo_bytes("right", 1), "left", self.loaded)       # same size, other phone
        self.assertIn("model_mismatch", codes(report))
        self.assertTrue(any("swapped" in w["message"] for w in report["warnings"]))

    def test_notebook_measured_with_visitor_calibration(self):
        for pair in NOTEBOOK_PAIR_IDS:
            ref = measure_pair(self.model_b, pair)
            res = live.measure_points(self.loaded, ref["left_px"], ref["right_px"], (305.0, 215.0))
            self.assertEqual(res["errors"], [])
            self.assertEqual(res["thresholds"]["source"], "calibration")
            self.assertEqual(res["warnings"], [], res["warnings"])
            self.assertLess(abs(res["summary"]["width_mm"] - ref["summary"]["width_mm"]), 2.0)
            self.assertLess(abs(res["summary"]["height_mm"] - ref["summary"]["height_mm"]), 2.0)


if __name__ == "__main__":
    unittest.main()
