"""Module 7 tests: live measurement vs saved results, coordinate mapping, zoom refinement, upload validation, undo/reset."""
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))

from stereo_measure import load_model_b, measure_pair  # noqa: E402
from stereo_measure import selection as sel  # noqa: E402
from stereo_measure.config import IMAGE_SIZE, NOTEBOOK_PAIR_IDS, PATTERN, SQUARE_MM  # noqa: E402
from stereo_measure.live import measure_points  # noqa: E402

PAGE = ROOT / "Pages" / "app_module7.py"
W, H = IMAGE_SIZE


def inspect_model_b(data, side):
    """The author's Model B expectation, passed explicitly (the library has no implicit default)."""
    other = "right" if side == "left" else "left"
    return sel.inspect_upload(data, side, IMAGE_SIZE, sel.MODEL_B_CAMERAS[side], None, sel.MODEL_B_CAMERAS[other])


def jpeg_bytes(size, orientation=None, fmt="JPEG", seed=0):
    rng = np.random.default_rng(seed)
    arr = np.repeat(np.repeat(rng.integers(0, 255, (size[1] // 64 + 1, size[0] // 64 + 1, 3), np.uint8), 64, 0), 64, 1)
    im = Image.fromarray(arr[:size[1], :size[0]])
    exif = Image.Exif()
    if orientation:
        exif[274] = orientation
    buf = io.BytesIO()
    im.save(buf, fmt, **({"exif": exif} if fmt == "JPEG" else {}), **({"quality": 60} if fmt == "JPEG" else {}))
    return buf.getvalue()


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model_b()
        cls.saved = pd.read_csv(MODULE / "outputs" / "pair_summary.csv").set_index("pair")

    def test_live_matches_measure_pair_and_saved_outputs(self):
        for pair in NOTEBOOK_PAIR_IDS:
            ref = measure_pair(self.model, pair)
            live = measure_points(self.model, ref["left_px"], ref["right_px"], (305.0, 215.0))
            self.assertEqual(live["errors"], [])
            for key in ("mean_Z_depth_mm", "top_width", "bottom_width", "left_height", "right_height", "width_mm", "height_mm"):
                self.assertAlmostEqual(live["summary"][key], ref["summary"][key], places=9)
            for key, col in (("mean_Z_depth_mm", "mean_Z_depth_mm"), ("width_mm", "width_mm"), ("height_mm", "height_mm"),
                             ("width_err_mm", "width_err_mm"), ("height_err_pct", "height_err_pct")):
                self.assertAlmostEqual(live["summary"][key], self.saved.loc[pair, col], places=3)   # CSVs hold 4 decimals
            np.testing.assert_allclose(live["X"], ref["X"], atol=1e-9)

    def test_no_truth_means_no_error_fields(self):
        ref = measure_pair(self.model, 1)
        s = measure_points(self.model, ref["left_px"], ref["right_px"])["summary"]
        self.assertNotIn("width_err_mm", s)
        s = measure_points(self.model, ref["left_px"], ref["right_px"], (305.0, None))["summary"]
        self.assertIn("width_err_pct", s)
        self.assertNotIn("height_err_pct", s)

    def test_wrong_corner_order_is_rejected(self):
        ref = measure_pair(self.model, 1)
        bad = measure_points(self.model, ref["left_px"][[1, 0, 2, 3]], ref["right_px"][[1, 0, 2, 3]])
        self.assertTrue(bad["errors"])
        self.assertNotIn("width_mm", bad["summary"])
        self.assertFalse(bad["summary"]["valid"])

    def test_mismatched_correspondences_flagged(self):
        ref = measure_pair(self.model, 1)
        swapped = measure_points(self.model, ref["left_px"], ref["right_px"][::-1].copy())
        self.assertTrue(swapped["errors"] or swapped["warnings"])

    def test_point_behind_camera_is_rejected(self):
        ref = measure_pair(self.model, 1)
        right = ref["right_px"].copy()
        right[:, 0] += 3000          # huge negative disparity -> negative depth
        self.assertTrue(measure_points(self.model, ref["left_px"], right)["errors"])


class MappingTests(unittest.TestCase):
    def test_click_roundtrip_with_different_axis_scales(self):
        # rendered image deliberately NOT proportional to the source: scales differ per axis
        click = {"x": 123, "y": 456, "width": 700, "height": 1100}
        pt = sel.click_to_source(click, (0, 0), (W, H))
        self.assertAlmostEqual(pt[0], (123.5) * W / 700 - 0.5)
        self.assertAlmostEqual(pt[1], (456.5) * H / 1100 - 0.5)
        u, v = sel.source_to_shown(pt, (0, 0), (W, H), (700, 1100))
        self.assertAlmostEqual(u, 123, places=9)
        self.assertAlmostEqual(v, 456, places=9)

    def test_preview_scale_uses_actual_size(self):
        image = np.zeros((H, W, 3), np.uint8)
        preview = sel.make_preview(image, 700)
        self.assertEqual(preview.shape[1], 700)
        self.assertEqual(preview.shape[0], round(H * 700 / W))
        pt = (2000.25, 3000.75)
        shown = sel.source_to_shown(pt, (0, 0), (W, H), (preview.shape[1], preview.shape[0]))
        click = {"x": shown[0], "y": shown[1], "width": preview.shape[1], "height": preview.shape[0]}
        back = sel.click_to_source(click, (0, 0), (W, H))
        self.assertAlmostEqual(back[0], pt[0], places=6)
        self.assertAlmostEqual(back[1], pt[1], places=6)

    def test_zoom_window_stays_inside_image(self):
        for pt in ((0, 0), (W - 1, H - 1), (5.5, H - 3), (2000, 3000)):
            x0, y0, w, h = sel.zoom_window(pt, 80)
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x0 + w, W)
            self.assertLessEqual(y0 + h, H)
            self.assertEqual((w, h), (160, 160))

    def test_zoom_refinement_recovers_known_point(self):
        rng = np.random.default_rng(1)
        full = rng.integers(0, 255, (H, W, 3), np.uint8)
        for coarse, truth in (((1520.0, 3920.0), (1531.3, 3926.6)), ((10.0, 12.0), (3.25, 20.5)), ((W - 5.0, H - 5.0), (W - 20.0, H - 9.5))):
            window = sel.zoom_window(coarse, 80)
            zoom, crop = sel.make_zoom(full, window, 640)
            self.assertEqual(zoom.shape[:2], (640, 640))
            shown = sel.source_to_shown(truth, (window[0], window[1]), crop, (640, 640))
            click = {"x": int(round(shown[0])), "y": int(round(shown[1])), "width": 640, "height": 640}
            recovered = sel.click_to_source(click, (window[0], window[1]), crop)
            # a whole-pixel click on the 4x zoom resolves the true point to within 1/8 original pixel (plus rounding)
            self.assertLess(abs(recovered[0] - truth[0]), 0.126)
            self.assertLess(abs(recovered[1] - truth[1]), 0.126)

    def test_zoom_with_browser_scaled_rendering(self):
        window = (1000, 2000, 160, 160)
        click = {"x": 300, "y": 150, "width": 500, "height": 500}      # browser shrank 640 px image to 500 px
        x, y = sel.click_to_source(click, (1000, 2000), (160, 160))
        self.assertAlmostEqual(x, 1000 + 300.5 * 160 / 500 - 0.5)
        self.assertAlmostEqual(y, 2000 + 150.5 * 160 / 500 - 0.5)

    def test_bad_clicks_raise(self):
        with self.assertRaises(ValueError):
            sel.click_to_source({"x": 1, "y": 1, "width": 0, "height": 0}, (0, 0), (W, H))
        with self.assertRaises(ValueError):
            sel.click_to_source({"x": 800, "y": 1, "width": 700, "height": 900}, (0, 0), (W, H))
        with self.assertRaises(KeyError):
            sel.click_to_source({"x": 1}, (0, 0), (W, H))


class UploadTests(unittest.TestCase):
    def test_valid_portrait_and_exif_rotated_landscape(self):
        _, warns = inspect_model_b(jpeg_bytes((W, H)), "left")
        self.assertTrue(any("camera model" in w for w in warns))           # synthetic file has no EXIF model
        data = jpeg_bytes((H, W), orientation=6)                           # stored landscape, EXIF says rotate -> 4284 x 5712
        size, _ = inspect_model_b(data, "left")
        self.assertEqual(size, IMAGE_SIZE)
        self.assertEqual(sel.decode_upload(data, IMAGE_SIZE).shape[:2], (H, W))

    def test_png_portrait_accepted(self):
        data = jpeg_bytes((W, H), fmt="PNG")
        self.assertEqual(inspect_model_b(data, "right")[0], IMAGE_SIZE)

    def test_rejections(self):
        cases = {
            "wrong size": jpeg_bytes((4000, 3000)),
            "landscape without EXIF": jpeg_bytes((H, W)),
            "half resolution": jpeg_bytes((W // 2, H // 2)),
            "corrupt": b"not an image at all",
            "empty": b"",
            "unsupported format": self._gif(),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(sel.UploadError):
                inspect_model_b(data, "left")

    @staticmethod
    def _gif():
        buf = io.BytesIO()
        Image.new("RGB", (W, H)).convert("P").save(buf, "GIF")
        return buf.getvalue()

    def test_real_notebook_photos_orientation(self):
        for side, name, expected in (("left", "air_image1.jpeg", "iPhone Air"), ("right", "pro_image1.jpeg", "iPhone 17 Pro Max")):
            path = MODULE / "data" / "object_measurement" / side / name
            if not path.is_file():
                self.skipTest("original photographs are not present")
            data = path.read_bytes()
            size, warns = inspect_model_b(data, side)
            self.assertEqual(size, IMAGE_SIZE)
            self.assertEqual(warns, [], expected)
            self.assertEqual(sel.decode_upload(data, IMAGE_SIZE).shape[:2], (H, W))
        # swapped cameras produce a hint
        left = (MODULE / "data" / "object_measurement" / "right" / "pro_image1.jpeg")
        if left.is_file():
            _, warns = inspect_model_b(left.read_bytes(), "left")
            self.assertTrue(any("right-camera" in w for w in warns))


class PageStateTests(unittest.TestCase):
    def _app(self):
        at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.session_state["module7_part"] = "1"
        return at

    def _app_with_visitor_model(self):
        """Stand-in for a finished visitor calibration (Phase 6 will set this key); exercises the preserved measurement flow."""
        at = self._app()
        model = load_model_b()
        model.update(source="loaded_file", metrics=dict(epipolar_px=None, rectification_vertical_px=None), distortion_left="radial2",
                     distortion_right="radial2", image_size_left=model["image_size"], image_size_right=model["image_size"],
                     pattern=PATTERN, square_mm=SQUARE_MM, warnings=[], per_pair=[], distortion_cv={}, camera={})
        at.session_state["m7_visitor_model"] = model
        at.session_state["m7_measurement_enabled"] = True       # the Phase 5 gate is off by default
        return at

    def test_home_has_only_parts_1_and_3(self):
        at = AppTest.from_file(str(PAGE), default_timeout=60).run()
        self.assertFalse(at.exception)
        labels = [b.label for b in at.button]
        self.assertEqual(labels, ["Part 1 – Calibrated Stereo Object Measurement", "Part 3 – Room Measurement from Video"])

    def test_part3_placeholder(self):
        at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.session_state["module7_part"] = "3"
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("not implemented" in i.value.lower() for i in at.info))

    def test_part1_without_uploads_shows_reference_experiment_button_and_no_exception(self):
        at = self._app().run()
        self.assertFalse(at.exception)
        # with an icon, AppTest reports the expander as a Status-type block rather than in at.expander
        blocks = [b for b in at.main if getattr(b, "type", "") in ("expander", "status")]
        self.assertEqual(len(blocks), 1)                 # the reference now lives inside the experiment
        self.assertIn("Calibrate Your Stereo Cameras", blocks[0].label)
        self.assertFalse(blocks[0].proto.expanded)       # collapsed by default
        self.assertIn("View My Experiment & Results", [b.label for b in at.button])
        self.assertTrue(any("First calibrate your stereo cameras" in i.value for i in at.info))
        self.assertFalse([u for u in at.get("file_uploader") if "Camera Image" in u.label])   # visitor measurement disabled
        self.assertNotIn("Calculate Dimensions", [b.label for b in at.button])

    def test_reference_is_inside_collapsed_experiment_section(self):
        at = self._app().run()
        [b for b in at.button if b.label == "View My Experiment & Results"][0].click().run()
        self.assertFalse(at.exception)
        blocks = [b for b in at.main if getattr(b, "type", "") in ("expander", "status")]
        ref = [b for b in blocks if "My Camera Calibration Reference" in b.label]
        self.assertEqual(len(ref), 1)
        self.assertIn("Click to View My Stereo Calibration Parameters", ref[0].label)
        self.assertFalse(ref[0].proto.expanded)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("iPhone Air", text)
        self.assertIn("iPhone 17 Pro Max", text)

    def test_experiment_without_uploads_shows_all_pairs_from_saved_files(self):
        at = self._app().run()
        [b for b in at.button if b.label == "View My Experiment & Results"][0].click().run()
        self.assertFalse(at.exception)
        saved = pd.read_csv(MODULE / "outputs" / "pair_summary.csv").set_index("pair")
        for pair in NOTEBOOK_PAIR_IDS:
            at.radio(key="m7_exp_pair").set_value(pair).run()
            self.assertFalse(at.exception)
            metrics = {m.label: m.value for m in at.metric}
            self.assertEqual(metrics["Width"], f"{saved.loc[pair, 'width_mm']:.2f} mm")
            self.assertEqual(metrics["Mean Z-depth"], f"{saved.loc[pair, 'mean_Z_depth_mm']:.2f} mm")

    def test_close_experiment_button_is_last_and_closes_section(self):
        at = self._app().run()
        [b for b in at.button if b.label == "View My Experiment & Results"][0].click().run()
        labels = [b.label for b in at.button]
        self.assertEqual(labels.count("Close experiment"), 1)
        self.assertEqual(labels[-1], "Close experiment")             # after every pair table, the summary and the discussion
        for pair in NOTEBOOK_PAIR_IDS:
            at.radio(key="m7_exp_pair").set_value(pair).run()
            self.assertEqual([b.label for b in at.button][-1], "Close experiment")
        [b for b in at.button if b.label == "Close experiment"][0].click().run()
        self.assertFalse(at.exception)
        self.assertNotIn("Close experiment", [b.label for b in at.button])
        self.assertFalse(at.session_state["m7_show_experiment"])
        self.assertIn("View My Experiment & Results", [b.label for b in at.button])

    def test_invalid_upload_is_rejected_in_page(self):
        at = self._app_with_visitor_model()
        at.session_state["m7_file_left"] = ("bad.jpg", jpeg_bytes((4000, 3000)))
        at.session_state["m7_file_right"] = ("ok.jpg", jpeg_bytes((W, H), seed=2))
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("rejected" in e.value for e in at.error))
        self.assertNotIn("Calculate Dimensions", [b.label for b in at.button])

    def test_undo_and_reset_buttons(self):
        left = ("l.jpg", jpeg_bytes((W, H), seed=1))
        right = ("r.jpg", jpeg_bytes((W, H), seed=2))
        pair_id = hashlib.sha256((hashlib.sha256(left[1]).hexdigest() + hashlib.sha256(right[1]).hexdigest()).encode()).hexdigest()[:16]
        at = self._app_with_visitor_model()
        at.session_state["m7_file_left"], at.session_state["m7_file_right"] = left, right
        at.session_state["m7_pair_id"] = pair_id
        at.session_state["m7_ack"] = pair_id
        at.session_state["m7_dialog_opened"] = pair_id
        pts = [(100.5, 200.5), (900.25, 210.0), (880.0, 700.0), (120.0, 690.0)]
        at.session_state["m7_points"] = {"left": list(pts), "right": list(pts[:2])}
        at.run()
        self.assertFalse(at.exception)

        def click(label):
            [b for b in at.button if b.label == label][0].click().run()
            self.assertFalse(at.exception)

        click("Undo last point")                                  # left stage: undo acts on the left image, right corners are kept
        self.assertEqual(at.session_state["m7_points"]["left"], pts[:3])      # floats preserved, order intact
        self.assertEqual(len(at.session_state["m7_points"]["right"]), 2)
        click("Reset left")
        self.assertEqual(at.session_state["m7_points"]["left"], [])
        at.session_state["m7_points"] = {"left": list(pts), "right": list(pts)}
        at.run()
        self.assertIn("Confirm Left Image Selections", [b.label for b in at.button])
        click("Confirm Left Image Selections")
        at.session_state["m7_ack_right"] = pair_id
        at.run()
        click("Confirm Right Image Selections")
        self.assertIn("Calculate Dimensions", [b.label for b in at.button])
        click("Undo last point")                                   # changing a point withdraws the confirmation
        self.assertFalse(at.session_state["m7_right_confirmed"])
        self.assertNotIn("Calculate Dimensions", [b.label for b in at.button])
        click("Reset both")
        self.assertEqual(at.session_state["m7_points"], {"left": [], "right": []})

    def test_calculate_with_saved_corners_reproduces_saved_results(self):
        left = ("l.jpg", jpeg_bytes((W, H), seed=1))
        right = ("r.jpg", jpeg_bytes((W, H), seed=2))
        pair_id = hashlib.sha256((hashlib.sha256(left[1]).hexdigest() + hashlib.sha256(right[1]).hexdigest()).encode()).hexdigest()[:16]
        sel_json = json.loads((MODULE / "selections" / "pair_1_corners.json").read_text())
        at = self._app_with_visitor_model()
        at.session_state["m7_file_left"], at.session_state["m7_file_right"] = left, right
        at.session_state["m7_pair_id"] = at.session_state["m7_ack"] = at.session_state["m7_dialog_opened"] = pair_id
        at.session_state["m7_points"] = {"left": [tuple(p) for p in sel_json["left_px"]], "right": [tuple(p) for p in sel_json["right_px"]]}
        at.run()
        [b for b in at.button if b.label == "Confirm Left Image Selections"][0].click().run()
        at.session_state["m7_ack_right"] = pair_id
        at.run()
        [b for b in at.button if b.label == "Confirm Right Image Selections"][0].click().run()
        [b for b in at.button if b.label == "Calculate Dimensions"][0].click().run()
        self.assertFalse(at.exception)
        saved = pd.read_csv(MODULE / "outputs" / "pair_summary.csv").set_index("pair")
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Mean Z-depth"], f"{saved.loc[1, 'mean_Z_depth_mm']:.1f} mm")
        self.assertEqual(metrics["Estimated width"], f"{saved.loc[1, 'width_mm']:.1f} mm")
        self.assertEqual(metrics["Estimated height"], f"{saved.loc[1, 'height_mm']:.1f} mm")


if __name__ == "__main__":
    unittest.main()
