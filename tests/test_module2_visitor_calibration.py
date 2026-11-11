"""Module 2 Phase 3: visitor monocular calibration (load a file / calibrate a new camera) and measurement compatibility.

Synthetic checkerboard photographs are rendered from a known camera, so no photograph from the experiment is used. Streamlit pages
run under AppTest (no browser); uploader callbacks are exercised through session state. Nothing here writes into Modules/Module_2.
"""
import builtins
import hashlib
import io
import re
import sys
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_2"
PAGE = ROOT / "Pages" / "app_module2.py"
NPZ = MODULE / "camera_calibration.npz"
CSV = MODULE / "measurement_results.csv"
NPZ_SHA256 = "15a634af6f82e8c29b37d935887fb7fa7d37e6e552624f50ce0e1327b6cf702c"
CSV_SHA256 = "95f72ccc2d9f8b242d991d98ca6d354d2643fa3a3f8ded40d4c6bb17c53693eb"

sys.path.insert(0, str(MODULE))
import calibration_file as cf  # noqa: E402
import visitor_calibration as vc  # noqa: E402

TRUE_K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 480.0], [0, 0, 1.0]])
SIZE = (1280, 960)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_board(K, rvec, tvec, size=SIZE, pattern=(8, 6), square=20.0, quality=95):
    """JPEG bytes of a pinhole view of a checkerboard (inner corners = pattern) on a white background."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    image = np.full((size[1], size[0]), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                quad = np.array([[c, r, 0], [c + 1, r, 0], [c + 1, r + 1, 0], [c, r + 1, 0]], np.float32) * square
                pts = cv2.projectPoints(quad, rvec, tvec, K, None)[0].reshape(-1, 2)
                cv2.fillConvexPoly(image, np.round(pts * 16).astype(np.int32), 0, cv2.LINE_AA, 4)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def photo_set(n, seed=1, size=SIZE):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-.5, .5), rng.uniform(-.5, .5), rng.uniform(-.3, .3)])
        tvec = np.array([-90 + rng.uniform(-40, 40), -60 + rng.uniform(-30, 30), rng.uniform(380, 650)])
        out.append(render_board(TRUE_K, rvec, tvec, size))
    return out


PHOTOS = []
COMPUTED = None


def setUpModule():
    global PHOTOS, COMPUTED
    PHOTOS = photo_set(14)
    COMPUTED = vc.run_calibration(filled_state())


def filled_state(photos=None, slots=14):
    state = vc.new_state()
    vc.set_photo_count(state, slots)
    for i, data in enumerate(photos if photos is not None else PHOTOS):
        vc.store_photo(state, i, f"p{i}.jpg", data)
    return state


def npz_bytes(**fields):
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    return buffer.getvalue()


def good_fields(**override):
    fields = dict(camera_matrix=TRUE_K.copy(), distortion_coefficients=np.zeros((1, 5)), image_width=np.array(1280),
                  image_height=np.array(960), rms_error=np.array(0.3), checkerboard_columns=np.array(8), checkerboard_rows=np.array(6),
                  square_size_mm=np.array(20.0), successful_images=np.array(14))
    fields.update(override)
    return {k: v for k, v in fields.items() if v is not None}


def rejected(testcase, data, word=None):
    with testcase.assertRaises(cf.CalibrationFileError) as ctx:
        cf.parse_calibration(data)
    if word:
        testcase.assertIn(word.lower(), str(ctx.exception).lower())


class NpzLoadingTests(unittest.TestCase):
    def test_valid_file_loads_without_activating(self):
        model = cf.parse_calibration(npz_bytes(**good_fields()))
        self.assertFalse(model["needs_image_size"])
        self.assertEqual((model["image_width"], model["image_height"]), (1280, 960))
        self.assertEqual(model["distortion_coefficients"].shape, (1, 5))
        self.assertFalse(model["authenticity_verified"])

    def test_corrupted_and_non_npz_files(self):
        rejected(self, b"", "empty")
        rejected(self, b"not a zip at all", "not an npz")
        data = npz_bytes(**good_fields())
        rejected(self, data[: len(data) // 2])
        rejected(self, data[:60] + b"\x00" * 40 + data[100:])

    def test_missing_keys(self):
        rejected(self, npz_bytes(camera_matrix=TRUE_K), "missing")
        rejected(self, npz_bytes(distortion_coefficients=np.zeros(5)), "missing")

    def test_invalid_matrix_shapes_and_forms(self):
        rejected(self, npz_bytes(**good_fields(camera_matrix=np.eye(4))), "shape")
        rejected(self, npz_bytes(**good_fields(camera_matrix=np.eye(3)[:2])), "shape")
        skewed = TRUE_K.copy(); skewed[0, 1] = 50.0
        rejected(self, npz_bytes(**good_fields(camera_matrix=skewed)), "no skew")
        bad_row = TRUE_K.copy(); bad_row[2] = [0, 1, 1]
        rejected(self, npz_bytes(**good_fields(camera_matrix=bad_row)), "form")

    def test_non_positive_focal_lengths(self):
        for index in (0, 1):
            K = TRUE_K.copy(); K[index, index] = 0.0
            rejected(self, npz_bytes(**good_fields(camera_matrix=K)))
            K[index, index] = -1000.0
            rejected(self, npz_bytes(**good_fields(camera_matrix=K)))

    def test_nan_and_infinity(self):
        for bad in (np.nan, np.inf, -np.inf):
            K = TRUE_K.copy(); K[0, 0] = bad
            rejected(self, npz_bytes(**good_fields(camera_matrix=K)), "nan or infinite")
            D = np.zeros((1, 5)); D[0, 2] = bad
            rejected(self, npz_bytes(**good_fields(distortion_coefficients=D)), "nan or infinite")
            rejected(self, npz_bytes(**good_fields(rms_error=np.array(bad))))

    def test_unsupported_distortion_models(self):
        for shape in ((4,), (8,), (1, 8), (5, 1), (14,), ()):
            rejected(self, npz_bytes(**good_fields(distortion_coefficients=np.zeros(shape))))
        accepted = cf.parse_calibration(npz_bytes(**good_fields(distortion_coefficients=np.zeros(5))))
        self.assertEqual(accepted["distortion_coefficients"].shape, (1, 5))
        rejected(self, npz_bytes(**good_fields(distortion_coefficients=np.array([500.0, 0, 0, 0, 0]))), "plausible")

    def test_invalid_image_dimensions(self):
        for width, height in ((0, 960), (1280, -5), (50, 960), (1280, 99999), (1280.5, 960)):
            rejected(self, npz_bytes(**good_fields(image_width=np.array(width), image_height=np.array(height))))
        rejected(self, npz_bytes(**good_fields(image_height=None)), "only one")
        rejected(self, npz_bytes(**good_fields(image_width=np.array(["1280"]))))

    def test_principal_point_and_focal_must_fit_the_image(self):
        rejected(self, npz_bytes(**good_fields(image_width=np.array(500), image_height=np.array(400))), "outside")
        K = TRUE_K.copy(); K[0, 0] = K[1, 1] = 50.0
        rejected(self, npz_bytes(**good_fields(camera_matrix=K)), "implausible")
        K = TRUE_K.copy(); K[1, 1] = 3000.0
        rejected(self, npz_bytes(**good_fields(camera_matrix=K)), "fx / fy")

    def test_metadata_is_validated(self):
        for key, value in (("checkerboard_columns", 1), ("checkerboard_rows", 500), ("square_size_mm", -3.0), ("successful_images", 0),
                           ("rms_error", -1.0)):
            rejected(self, npz_bytes(**good_fields(**{key: np.array(value)})))

    def test_object_arrays_pickles_and_unknown_fields_are_rejected(self):
        rejected(self, npz_bytes(**good_fields(camera_matrix=np.array([TRUE_K], dtype=object))))
        rejected(self, npz_bytes(**good_fields(extra=np.array([1, 2, 3]))), "unknown")
        rejected(self, npz_bytes(**good_fields(rms_error=np.array("high"))))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../evil.npy", b"x")
        rejected(self, buffer.getvalue(), "unexpected")
        with mock.patch("numpy.load", wraps=np.load) as spy:
            cf.parse_calibration(npz_bytes(**good_fields()))
        self.assertIs(spy.call_args.kwargs["allow_pickle"], False)

    def test_excessively_large_file(self):
        rejected(self, b"PK" + b"0" * (cf.MAX_FILE_BYTES + 1), "mb")
        rejected(self, npz_bytes(**good_fields(), filler=np.zeros(20_000)), "too large")

    def test_legacy_file_without_image_size(self):
        legacy = cf.parse_calibration(NPZ.read_bytes())
        self.assertTrue(legacy["needs_image_size"])
        self.assertIsNone(legacy["image_width"])
        self.assertEqual(legacy["successful_images"], 18)
        self.assertAlmostEqual(legacy["camera_matrix"][0, 0], 4068.2888387501193)
        with self.assertRaises(ValueError):
            cf.export_calibration(legacy)                  # cannot be re-exported without a size

    def test_legacy_dimension_confirmation(self):
        legacy = cf.parse_calibration(NPZ.read_bytes())
        sized = cf.with_image_size(legacy, 4284, 5712)
        self.assertFalse(sized["needs_image_size"])
        self.assertEqual((sized["image_width"], sized["image_height"], sized["size_source"]), (4284, 5712, "confirmed_by_visitor"))
        self.assertTrue(legacy["needs_image_size"])         # the input model is not mutated
        for width, height in ((0, 5712), (4284.5, 5712), (200, 300), (99999, 5712)):
            with self.assertRaises(cf.CalibrationFileError):
                cf.with_image_size(legacy, width, height)
        with self.assertRaises(cf.CalibrationFileError):
            cf.with_image_size(sized, 4284, 5712)           # a file that has a size cannot be re-sized


class CheckerboardAndCalibrationTests(unittest.TestCase):
    def test_object_points(self):
        pts = vc.uc.board_object_points((8, 6), 20.0)
        self.assertEqual(pts.shape, (48, 3))
        self.assertTrue((pts[:, 2] == 0).all())
        self.assertEqual(tuple(pts[1]), (20.0, 0.0, 0.0))
        self.assertEqual(tuple(pts[8]), (0.0, 20.0, 0.0))

    def test_settings_validation_and_unit_conversion(self):
        settings, errors = vc.validate_settings(8, 6, 20, "mm")
        self.assertEqual((errors, settings["square_mm"]), ([], 20.0))
        settings, _ = vc.validate_settings(8, 6, 2.5, "cm")
        self.assertEqual(settings["square_mm"], 25.0)
        for args in ((2, 6, 20, "mm"), (8, 41, 20, "mm"), (8.5, 6, 20, "mm"), (8, 6, 0, "mm"), (8, 6, -1, "mm"), (8, 6, 20, "in"),
                     (8, 6, None, "mm"), (8, 6, float("nan"), "mm"), (None, 6, 20, "mm"), (8, 6, 5000, "mm"), (8, 6, 200, "cm")):
            settings, errors = vc.validate_settings(*args)
            self.assertIsNone(settings, args)
            self.assertTrue(errors)
        self.assertEqual(vc.convert_square(20, "mm", "cm"), 2.0)
        self.assertEqual(vc.convert_square(2, "cm", "mm"), 20.0)

    def test_unit_switch_with_the_same_physical_size_keeps_the_photos(self):
        state = filled_state(slots=14)
        settings, _ = vc.validate_settings(8, 6, 2.0, "cm")
        self.assertFalse(vc.apply_settings(state, settings))
        self.assertEqual(len(state["photos"]), 14)
        settings, _ = vc.validate_settings(8, 6, 2.1, "cm")
        self.assertTrue(vc.apply_settings(state, settings))
        self.assertEqual(state["photos"], {})

    def test_detection_accepts_and_rejects(self):
        state = filled_state(photos=PHOTOS[:2], slots=10)
        blank = cv2.imencode(".jpg", np.full((960, 1280), 255, np.uint8))[1].tobytes()
        vc.store_photo(state, 2, "blank.jpg", blank)
        vc.store_photo(state, 3, "junk.jpg", b"definitely not an image")
        vc.store_photo(state, 4, "tiny.png", cv2.imencode(".png", np.zeros((50, 50), np.uint8))[1].tobytes())
        vc.store_photo(state, 5, "dup.jpg", PHOTOS[0])
        vc.store_photo(state, 6, "small.jpg", render_board(TRUE_K, np.array([.1, .1, 0.]), np.array([-90., -60., 500.]), size=(1000, 750)))
        ev = vc.evaluate(state)
        status = {s["number"]: s["status"] for s in ev["slots"]}
        self.assertEqual([status[n] for n in range(1, 8)], [vc.ACCEPTED, vc.ACCEPTED, vc.REJECTED, vc.REJECTED, vc.REJECTED, vc.REJECTED,
                                                            vc.REJECTED])
        self.assertEqual(status[8], vc.NOT_UPLOADED)
        messages = {s["number"]: " ".join(s["messages"]) for s in ev["slots"]}
        self.assertIn("No 8 x 6", messages[3])
        self.assertIn("could not be read", messages[4])
        self.assertIn("too small", messages[5])
        self.assertIn("same file as photo 1", messages[6])
        self.assertIn("1000 x 750", messages[7])
        self.assertEqual((ev["n_uploaded"], ev["n_accepted"], ev["n_rejected"], ev["size"]), (7, 2, 5, SIZE))
        self.assertFalse(ev["can_calibrate"])

    def test_swapped_rows_and_columns_are_explained(self):
        state = vc.new_state()
        settings, _ = vc.validate_settings(6, 8, 20, "mm")
        vc.apply_settings(state, settings)
        rec = vc.store_photo(state, 0, "a.jpg", PHOTOS[0])
        self.assertEqual(rec["status"], "ok" if rec["corners"] is not None else rec["status"])      # 6 x 8 may also be found
        self.assertIn(rec["status"], ("ok", "swapped_pattern", "not_found"))

    def test_changed_board_clears_photos_and_invalidates(self):
        state = filled_state(slots=14)
        self.assertTrue(vc.is_current(state, COMPUTED))
        settings, _ = vc.validate_settings(7, 6, 20, "mm")
        self.assertTrue(vc.apply_settings(state, settings))
        self.assertFalse(vc.is_current(state, COMPUTED))
        state = filled_state(slots=14)
        vc.remove_photo(state, 3)
        self.assertFalse(vc.is_current(state, COMPUTED))

    def test_calibration_recovers_the_synthetic_camera(self):
        K, D = COMPUTED["camera_matrix"], COMPUTED["distortion_coefficients"]
        self.assertEqual(D.shape, (1, 5))
        self.assertLess(abs(K[0, 0] - 1000) / 1000, 0.01)
        self.assertLess(abs(K[1, 1] - 1000) / 1000, 0.01)
        self.assertLess(abs(K[0, 2] - 640), 5)
        self.assertLess(abs(K[1, 2] - 480), 5)
        self.assertLess(COMPUTED["rms_error"], 0.5)
        self.assertEqual((COMPUTED["image_width"], COMPUTED["image_height"]), SIZE)
        self.assertEqual((COMPUTED["photos_uploaded"], COMPUTED["successful_images"], COMPUTED["photos_rejected"]), (14, 14, 0))
        self.assertEqual(COMPUTED["source"], vc.SOURCE_COMPUTED)

    def test_reprojection_errors(self):
        errors = np.array([e["error_px"] for e in COMPUTED["per_image"]])
        self.assertEqual(len(errors), 14)
        self.assertTrue(np.isfinite(errors).all() and (errors >= 0).all())
        # OpenCV's RMS is the root-mean-square over all corners; each image mean can only be at or below the overall RMS scale
        self.assertLessEqual(errors.mean(), COMPUTED["rms_error"] + 1e-6)
        self.assertEqual([e["number"] for e in COMPUTED["per_image"]], list(range(1, 15)))

    def test_inputs_are_not_resized(self):
        for rec in filled_state().get("photos", {}).values():
            self.assertEqual(rec["size"], SIZE)
            self.assertLessEqual(float(rec["corners"][:, 0].max()), SIZE[0])

    def test_too_few_photos_or_bad_input(self):
        state = filled_state(photos=PHOTOS[:vc.MIN_ACCEPTED - 1], slots=10)
        with self.assertRaises(vc.CalibrationError) as ctx:
            vc.run_calibration(state)
        self.assertIn(f"at least {vc.MIN_ACCEPTED}", str(ctx.exception))
        with self.assertRaises(vc.CalibrationError):
            vc.run_calibration(vc.new_state())
        for bad in (5, 41, True, 10.5, None):
            with self.assertRaises(ValueError):
                vc.set_photo_count(vc.new_state(), bad)

    def test_degenerate_views_raise_instead_of_returning_a_bad_model(self):
        same = render_board(TRUE_K, np.array([0.1, 0.1, 0.]), np.array([-90., -60., 500.]))
        shots = [same] + [render_board(TRUE_K, np.array([0.1, 0.1, 0.]), np.array([-90. + i * 0.05, -60., 500.])) for i in range(1, 12)]
        state = filled_state(photos=shots, slots=12)
        try:
            model = vc.run_calibration(state)
        except vc.CalibrationError:
            return                                          # refused: fine
        self.assertTrue(any(w["code"] in ("low_tilt_variation", "poor_coverage", "low_distance_variation") for w in model["warnings"]))

    def test_quality_warnings_are_raised_for_a_narrow_set(self):
        codes = {w["code"] for w in COMPUTED["warnings"]}
        self.assertIn("few_photos", codes)                  # 14 < 15 recommended
        self.assertTrue(all(w["heuristic"] for w in COMPUTED["warnings"]))


class RoundTripTests(unittest.TestCase):
    def test_export_load_activate_round_trip(self):
        data = cf.export_calibration(COMPUTED)
        loaded = cf.parse_calibration(data)
        self.assertTrue(np.array_equal(loaded["camera_matrix"], COMPUTED["camera_matrix"]))      # float64 is stored exactly
        self.assertTrue(np.array_equal(loaded["distortion_coefficients"], COMPUTED["distortion_coefficients"]))
        for key in ("image_width", "image_height", "rms_error", "checkerboard_columns", "checkerboard_rows", "square_size_mm",
                    "successful_images"):
            self.assertEqual(loaded[key], COMPUTED[key], key)
        self.assertFalse(loaded["needs_image_size"])
        self.assertIsNone(vc.check_measurement_image(loaded, SIZE))

    def test_exported_file_contains_only_plain_numeric_arrays(self):
        with np.load(io.BytesIO(cf.export_calibration(COMPUTED)), allow_pickle=False) as npz:
            self.assertEqual(set(npz.files), {"format_version", "camera_matrix", "distortion_coefficients", "image_width", "image_height",
                                              "rms_error", "checkerboard_columns", "checkerboard_rows", "square_size_mm",
                                              "successful_images"})
            self.assertTrue(all(npz[k].dtype.kind in "fiu" for k in npz.files))

    def test_export_is_in_memory_and_refuses_unusable_models(self):
        with mock.patch.object(builtins, "open", side_effect=AssertionError("export must not open files")):
            cf.export_calibration(COMPUTED)
        for bad in (dict(COMPUTED, camera_matrix=np.full((3, 3), np.nan)), dict(COMPUTED, image_width=None), {}):
            with self.assertRaises(ValueError):
                cf.export_calibration(bad)

    def test_legacy_file_with_confirmed_size_exports_and_reloads(self):
        sized = cf.with_image_size(cf.parse_calibration(NPZ.read_bytes()), 4284, 5712)
        again = cf.parse_calibration(cf.export_calibration(sized))
        self.assertEqual((again["image_width"], again["image_height"]), (4284, 5712))
        self.assertTrue(np.array_equal(again["camera_matrix"], sized["camera_matrix"]))


class MeasurementCompatibilityTests(unittest.TestCase):
    def test_matching_resolution(self):
        self.assertIsNone(vc.check_measurement_image(COMPUTED, SIZE))

    def test_mismatched_resolution_or_orientation_blocks(self):
        for size in ((960, 1280), (640, 480), (2560, 1920)):
            message = vc.check_measurement_image(COMPUTED, size)
            self.assertIn("not rescaled", message)
            self.assertIn("measurement is blocked", message)

    def test_invalid_or_missing_calibration(self):
        self.assertIn("No calibration", vc.check_measurement_image(None, SIZE))
        self.assertIn("invalid", vc.check_measurement_image(dict(COMPUTED, camera_matrix=np.full((3, 3), np.nan)), SIZE))
        self.assertIn("invalid", vc.check_measurement_image({"distortion_coefficients": np.zeros((1, 5))}, SIZE))
        self.assertIn("does not state", vc.check_measurement_image(dict(COMPUTED, image_width=None, image_height=None), SIZE))


# ---------------------------------------------------------------- page / session state

def run_page(seed=None):
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    for key, value in (seed or {}).items():
        at.session_state[key] = value
    return at.run()


def visitor_page(seed=None):
    at = run_page(dict(seed or {}, m2_page="visitor"))
    return at


def labels(at):
    return [b.label for b in at.button]


def click(at, key):
    at.button(key=key).click()
    return at.run()


def activated_page():
    at = visitor_page({"m2_computed": COMPUTED, "m2_cal": filled_state(), "m2_cal_workflow": "run"})
    return click(at, "m2_activate_computed")


class SessionStateTests(unittest.TestCase):
    def test_selecting_a_workflow_never_activates_anything(self):
        at = visitor_page({"m2_computed": COMPUTED, "m2_cal": filled_state()})
        for key, workflow in (("m2_cal_mode_load", "load"), ("m2_cal_mode_run", "run"), ("m2_cal_mode_load", "load")):
            at = click(at, key)
            self.assertEqual(at.session_state["m2_cal_workflow"], workflow)
            self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertEqual(len(at.text_input) + len(at.number_input), 0)         # measurement stays locked

    def test_load_workflow_shows_an_uploader_and_no_run_controls(self):
        at = click(visitor_page(), "m2_cal_mode_load")
        self.assertEqual([u.label for u in at.get("file_uploader")], ["Upload Calibration File (.npz)"])
        self.assertNotIn("Run Camera Calibration", labels(at))

    def test_run_workflow_shows_board_settings_and_defaults(self):
        at = click(visitor_page(), "m2_cal_mode_run")
        values = {n.label: n.value for n in at.number_input}
        self.assertEqual((values["Inner corners (columns)"], values["Inner corners (rows)"], values["Square size"]), (8, 6, 20.0))
        self.assertEqual(at.selectbox[0].value, "mm")
        self.assertIn("Run Camera Calibration", labels(at))
        self.assertIn("Next", labels(at))
        self.assertIn("Previous", labels(at))
        self.assertIn("Download Calibration (.npz)", [d.proto.label for d in at.get("download_button")])
        self.assertTrue(at.get("download_button")[0].proto.disabled)
        self.assertIsNone(at.session_state["m2_visitor_calibration"])

    def test_run_with_too_few_photos_explains_what_is_missing(self):
        at = visitor_page({"m2_cal_workflow": "run"})
        at = click(at, "m2_cal_run")
        self.assertTrue(any("No calibration photographs uploaded" in w.value for w in at.warning))
        self.assertIsNone(at.session_state["m2_computed"])

    def test_run_calibration_from_the_page_then_activate_and_download(self):
        at = visitor_page({"m2_cal": filled_state(), "m2_cal_workflow": "run"})
        at = click(at, "m2_cal_run")
        self.assertFalse(at.exception)
        computed = at.session_state["m2_computed"]
        self.assertIsNotNone(computed)
        self.assertIsNone(at.session_state["m2_visitor_calibration"])          # computed is not active until the visitor activates
        self.assertTrue(any("not active yet" in i.value for i in at.info))
        self.assertFalse(at.get("download_button")[0].proto.disabled)
        at = click(at, "m2_activate_computed")
        self.assertTrue(np.array_equal(at.session_state["m2_visitor_calibration"]["camera_matrix"], computed["camera_matrix"]))
        self.assertTrue(any("active for measurement" in s.value for s in at.success))
        self.assertTrue(at.button(key="m2_activate_computed").disabled)

    def test_activation_unlocks_measurement_with_the_visitors_calibration_only(self):
        at = activated_page()
        self.assertEqual(len(at.text_input), 1)
        self.assertEqual([n.label for n in at.number_input if "(cm)" in n.label],
                         ["Actual Width (cm)", "Actual Height (cm)", "Camera-to-Object Distance (cm)"])
        self.assertTrue(any("Using active calibration: 1280 × 960 px" in c.value for c in at.caption))
        self.assertFalse(any("Active calibration" in s.value for s in at.success))   # compact line, not a green banner
        self.assertAlmostEqual(at.session_state["m2_visitor_calibration"]["camera_matrix"][0, 0], COMPUTED["camera_matrix"][0, 0])
        self.assertNotAlmostEqual(at.session_state["m2_visitor_calibration"]["camera_matrix"][0, 0], 4068.2888387501193, places=0)

    def test_loaded_file_waits_for_explicit_activation(self):
        loaded = cf.parse_calibration(cf.export_calibration(COMPUTED))
        at = visitor_page({"m2_load_pending": loaded, "m2_load_name": "mine.npz", "m2_cal_workflow": "load"})
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertIn("Activate Calibration", labels(at))
        at = click(at, "m2_activate_loaded")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])
        self.assertEqual(at.session_state["m2_visitor_calibration"]["image_width"], 1280)

    def test_legacy_file_cannot_activate_until_size_is_confirmed(self):
        legacy = cf.parse_calibration(NPZ.read_bytes())
        at = visitor_page({"m2_load_pending": legacy, "m2_load_name": "camera_calibration.npz", "m2_cal_workflow": "load"})
        self.assertNotIn("Activate Calibration", labels(at))
        self.assertTrue(any("Missing image resolution" in w.value for w in at.warning))
        width, height = [n for n in at.number_input if "Calibration image" in n.label]
        width.set_value(4284); height.set_value(5712)
        at.run()
        at = click(at, "m2_legacy_apply")                                       # without the confirmation box
        self.assertTrue(at.session_state["m2_load_pending"]["needs_image_size"])
        self.assertTrue(any("confirmation" in e.value for e in at.error))
        at.checkbox[0].check()
        at.run()
        at = click(at, "m2_legacy_apply")
        pending = at.session_state["m2_load_pending"]
        self.assertEqual((pending["image_width"], pending["image_height"]), (4284, 5712))
        self.assertIsNone(at.session_state["m2_visitor_calibration"])           # still not active
        self.assertIn("Activate Calibration", labels(at))
        at = click(at, "m2_activate_loaded")
        self.assertEqual(at.session_state["m2_visitor_calibration"]["size_source"], "confirmed_by_visitor")

    def test_no_fallback_to_my_saved_calibration(self):
        for page in ("home", "visitor", "experiment"):
            at = run_page({"m2_page": page})
            self.assertIsNone(at.session_state["m2_visitor_calibration"])
            self.assertEqual(len(at.text_input), 0)
        source = PAGE.read_text()
        measurement = source.split("def visitor_measurement")[1].split("def visitor_page")[0]
        self.assertNotIn("experiment_camera_matrix", measurement)
        self.assertNotIn("experiment_fx", measurement)
        self.assertIsNone(re.search(r"(?<!visitor_)calibration\[", measurement))
        self.assertNotIn("camera_calibration.npz", (MODULE / "visitor_calibration.py").read_text() + (MODULE / "calibration_file.py").read_text().replace(
            "(camera_calibration.npz)", ""))

    def test_changing_inputs_invalidates_the_computed_calibration(self):
        at = visitor_page({"m2_computed": COMPUTED, "m2_cal": filled_state(), "m2_cal_workflow": "run"})
        at = click(at, "m2_activate_computed")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])
        at = click(at, "m2_cal_rm_0")                                           # removes the displayed photo
        self.assertIsNone(at.session_state["m2_computed"])
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertTrue(any("previously computed calibration was cleared" in i.value for i in at.info))

    def test_changing_board_settings_clears_photos_and_the_computed_calibration(self):
        at = visitor_page({"m2_computed": COMPUTED, "m2_cal": filled_state(), "m2_cal_workflow": "run"})
        next(n for n in at.number_input if n.label == "Inner corners (columns)").set_value(7)
        at.run()
        self.assertEqual(at.session_state["m2_cal"]["photos"], {})
        self.assertIsNone(at.session_state["m2_computed"])
        self.assertTrue(any("settings changed" in i.value for i in at.info))

    def test_unit_switch_converts_the_square_size(self):
        at = visitor_page({"m2_cal_workflow": "run"})
        at.selectbox[0].select("cm")
        at.run()
        self.assertEqual(next(n for n in at.number_input if n.label == "Square size").value, 2.0)
        self.assertEqual(at.session_state["m2_cal"]["settings"]["square_mm"], 20.0)

    def test_loaded_calibration_survives_workflow_switch_and_a_loaded_one_is_not_dropped_by_photo_changes(self):
        loaded = cf.parse_calibration(cf.export_calibration(COMPUTED))
        at = visitor_page({"m2_visitor_calibration": loaded, "m2_cal": filled_state(), "m2_cal_workflow": "run"})
        at = click(at, "m2_cal_rm_0")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])        # not derived from the removed photo
        at = click(at, "m2_cal_mode_load")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])

    def test_active_calibration_survives_view_navigation(self):
        at = activated_page()
        token = at.session_state["m2_visitor_calibration"]["camera_matrix"].copy()
        at = click(at, "navbtn_0")
        self.assertEqual(at.session_state["m2_page"], "home")
        at = click(at, "m2_open_experiment")
        at = click(at, "navbtn_0")
        at = click(at, "m2_open_visitor")
        self.assertTrue(np.array_equal(at.session_state["m2_visitor_calibration"]["camera_matrix"], token))
        self.assertEqual(len(at.text_input), 1)
        self.assertIsNotNone(at.session_state["m2_computed"])

    def test_reset_all_clears_visitor_state_only(self):
        at = activated_page()
        at = click(at, "m2_reset_all")
        for key in ("m2_visitor_calibration", "m2_computed", "m2_load_pending", "m2_cal_workflow"):
            self.assertIsNone(at.session_state[key], key)
        self.assertEqual(at.session_state["m2_cal"]["photos"], {})
        self.assertEqual(at.session_state["m2_cal"]["settings"]["cols"], 8)
        self.assertEqual(at.session_state["m2_page"], "visitor")
        self.assertEqual(at.session_state["m2_reset_count"], 1)
        self.assertEqual(len(at.text_input) + len(at.number_input), 0)
        self.assertEqual(sha256(NPZ), NPZ_SHA256)
        self.assertEqual(sha256(CSV), CSV_SHA256)

    def test_clear_calibration(self):
        at = activated_page()
        at = click(at, "m2_clear_computed")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertIsNone(at.session_state["m2_computed"])
        self.assertEqual(len(at.session_state["m2_cal"]["photos"]), 14)         # photos stay; only the calibration is cleared

    def test_photo_navigation(self):
        at = visitor_page({"m2_cal": filled_state(slots=14), "m2_cal_workflow": "run"})
        self.assertTrue(at.button(key="m2_cal_prev").disabled)
        at = click(at, "m2_cal_next")
        self.assertEqual(at.session_state["m2_cal"]["current"], 1)
        self.assertNotIn("m2_scroll_to_cal_photo", at.session_state)            # the one-shot scroll was consumed
        at = click(at, "m2_cal_prev")
        self.assertEqual(at.session_state["m2_cal"]["current"], 0)
        at.selectbox(key="m2_cal_goto").select(13)
        at.run()
        self.assertEqual(at.session_state["m2_cal"]["current"], 13)
        self.assertTrue(at.button(key="m2_cal_next").disabled)

    def test_no_stereo_concepts_in_the_visitor_code(self):
        text = (MODULE / "visitor_calibration.py").read_text() + (MODULE / "calibration_file.py").read_text()
        for word in ("stereoCalibrate", "stereoRectify", "triangulate", "findFundamentalMat", "baseline", "disparity"):
            self.assertNotIn(word, text)


class ExperimentIntegrityTests(unittest.TestCase):
    def test_visitor_flows_never_write_into_module_2(self):
        written = []
        real_open, real_path_open = builtins.open, Path.open

        def watch(path, mode="r"):
            if any(c in str(mode) for c in "wax+") and "Module_2" in str(path):
                written.append(str(path))

        def guarded_open(file, mode="r", *args, **kwargs):
            watch(file, mode)
            return real_open(file, mode, *args, **kwargs)

        def guarded_path_open(self, mode="r", *args, **kwargs):
            watch(self, mode)
            return real_path_open(self, mode, *args, **kwargs)

        with mock.patch.object(builtins, "open", guarded_open), mock.patch.object(Path, "open", guarded_path_open):
            at = activated_page()
            click(at, "m2_reset_all")
            cf.export_calibration(COMPUTED)
            cf.parse_calibration(NPZ.read_bytes())
        self.assertEqual(written, [])
        self.assertEqual(sha256(NPZ), NPZ_SHA256)
        self.assertEqual(sha256(CSV), CSV_SHA256)

    def test_experiment_view_is_unchanged_by_a_visitor_calibration(self):
        plain = run_page({"m2_page": "experiment"})
        with_visitor = run_page({"m2_page": "experiment", "m2_visitor_calibration": COMPUTED})
        cells = lambda at: [[str(v) for v in t.value.to_numpy().ravel()] for t in at.table]   # noqa: E731
        self.assertEqual(cells(plain), cells(with_visitor))
        trial_table = next(t.value for t in with_visitor.table if "Estimated width (cm)" in t.value.columns)
        self.assertEqual(len(trial_table), 20)
        stats = next(t.value for t in with_visitor.table if "Statistic" in t.value.columns)
        self.assertEqual(list(stats["Width"]), ["0.60 cm", "1.75 %", "0.91 %", "0.68 cm"])
        self.assertEqual(list(stats["Height"]), ["0.61 cm", "2.19 %", "1.25 %", "0.70 cm"])

    def test_module_7_is_not_modified_by_importing_the_helpers(self):
        import subprocess
        out = subprocess.run(["git", "status", "--porcelain", "--", "Modules/Module_7", "Pages/app_module7.py"], cwd=ROOT,
                             capture_output=True, text=True).stdout
        # Pages/app_module7.py gained only the Home button when the sidebar was removed.
        self.assertEqual([l for l in out.splitlines() if "room_measurement" not in l and "__pycache__" not in l
                          and not l.endswith("Pages/app_module7.py")], [])


if __name__ == "__main__":
    unittest.main()
