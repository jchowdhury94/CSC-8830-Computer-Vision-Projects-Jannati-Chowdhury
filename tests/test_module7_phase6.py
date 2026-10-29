"""Phase 6 tests: download-calibration button states, loaded/computed calibration activation and the visitor measurement flow."""
import io
import re
import sys
import unittest
from pathlib import Path

import cv2
import hashlib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import (AppTest, PAGE, RIG, button, filled_state, page, texts, wiz,  # noqa: E402
                                             export_calibration, load_calibration)
from test_module7_user_calibration import LEFT_SIZE, RIGHT_SIZE  # noqa: E402

SOURCE = Path(PAGE).read_text(encoding="utf-8")
DOWNLOAD_KEY = "m7_cal_download"


def computed_model():
    at = page(filled_state())
    button(at, "Run Stereo Calibration").click().run()
    assert at.session_state["m7_visitor_model"] is not None
    return at


def jpeg(size):
    out = io.BytesIO()
    Image.fromarray(np.random.default_rng(size[0]).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(out, "JPEG")
    return out.getvalue()


def download(at):
    return [d for d in at.get("download_button") if d.proto.label == "Download Calibration (.npz)"]


def download_proto(at):
    found = download(at)
    assert len(found) == 1, "exactly one download button expected"
    return found[0].proto


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class DownloadButtonTests(unittest.TestCase):
    def test_initial_state_visible_and_disabled_with_instruction(self):
        at = page()
        proto = download_proto(at)
        self.assertTrue(proto.disabled)
        self.assertIn("Run the stereo calibration to enable downloading.", texts(at))
        self.assertIsNone(at.session_state["m7_visitor_model"])

    def test_enabled_after_computed_calibration_with_valid_npz(self):
        at = computed_model()
        self.assertFalse(download_proto(at).disabled)
        model = at.session_state["m7_visitor_model"]
        data = export_calibration(model, model.get("pairs_uploaded"), model.get("camera"), model.get("created_utc"))
        self.assertTrue(data.startswith(b"PK"))                    # an npz is a zip archive
        back = load_calibration(data)
        np.testing.assert_allclose(back["K1"], model["K1"])
        np.testing.assert_allclose(back["T"], model["T"])
        self.assertEqual(tuple(back["image_size_left"]), tuple(model["image_size_left"]))

    def test_placement_between_results_and_details_in_section_3(self):
        body = SOURCE.split("def calibration_download")[1].split("# ----")[0]
        run = body.split("def run_workflow")[1].split("def calibrate_workflow")[0]
        order = [run.index(t) for t in ('for problem in ss["m7_cal_validation"]', "calibration_results(model)", "calibration_download(model)",
                                        "calibration_actions(model)")]
        self.assertEqual(order, sorted(order))
        self.assertIn('"View Calibration Details"', SOURCE)
        self.assertEqual(body.count("calibration_download("), 1)
        self.assertEqual(SOURCE.count("st.download_button("), 1)    # never a duplicate
        at = computed_model()
        self.assertEqual(len(download(at)), 1)

    def test_loaded_npz_does_not_enable_download_and_is_absent_in_load_workflow(self):
        source = computed_model().session_state["m7_visitor_model"]
        loaded = load_calibration(export_calibration(source, source.get("pairs_uploaded"), source.get("camera"), source.get("created_utc")))
        at = page(None, m7_visitor_model=loaded, m7_cal_loaded_name="mine.npz")
        self.assertEqual(download(at), [])                         # Load workflow: no download button anywhere
        self.assertIn("Loaded from NPZ", " ".join(m.value for m in at.metric))
        self.assertIn("validated structurally", texts(at))
        self.assertIs(at.session_state["m7_visitor_model"], loaded)
        at.session_state["m7_cal_workflow"] = "run"                # Run workflow with a loaded calibration: visible but disabled
        at.run()
        self.assertTrue(download_proto(at).disabled)
        self.assertNotIn("Stereo Calibration Loaded", texts(at))
        self.assertIn("mine.npz", texts(at))

    def test_clear_calibration_and_reset_all_disable_it_again(self):
        for label in ("Clear Calibration", "Reset All"):
            at = computed_model()
            button(at, label).click().run()
            self.assertIsNone(at.session_state["m7_visitor_model"])
            if label == "Clear Calibration":                       # workflow stays selected: button visible but disabled
                self.assertTrue(download_proto(at).disabled)
            else:                                                  # Reset All restores the initial state: no workflow selected
                self.assertIsNone(at.session_state["m7_cal_workflow"])
                self.assertEqual(download(at), [])
                at.session_state["m7_cal_workflow"] = "run"
                at.run()
                self.assertTrue(download_proto(at).disabled)
            self.assertEqual(at.session_state["m7_cal_source"], "none")
            self.assertFalse(at.session_state["m7_measurement_enabled"])

    def test_download_exports_the_computed_model(self):
        from unittest import mock
        import stereo_measure.calibration_io as cio
        seen = []
        real = cio.export_calibration

        def spy(model, *args, **kwargs):
            seen.append(model)
            return real(model, *args, **kwargs)

        with mock.patch.object(cio, "export_calibration", spy):
            at = computed_model()
        self.assertTrue(seen)
        self.assertIs(seen[-1], at.session_state["m7_visitor_model"])

    def test_model_b_is_never_the_download_source(self):
        self.assertNotIn("load_model_b", SOURCE.split("def calibration_download")[1].split("def active_calibration_message")[0])
        self.assertNotIn("get_model", SOURCE.split("def calibration_download")[1].split("def active_calibration_message")[0])
        self.assertNotIn("get_model", SOURCE.split("def upload_section")[1].split("def corner_instructions")[0])
        at = page()
        self.assertTrue(download_proto(at).disabled)               # saved Model B exists, yet nothing is downloadable

    def test_blue_css_is_scoped_to_the_enabled_download_button(self):
        css = SOURCE.split("DOWNLOAD_CSS = ")[1].split("PART_ONE_CSS")[0].replace("{DOWNLOAD_BLUE}", "#2563EB").replace("{{", "{").replace("}}", "}")
        selectors = [m.group(1) for m in re.finditer(r"([^{}]+)\{[^{}]*\}", css) if m.group(1).strip()]
        self.assertTrue(selectors)
        for selector in selectors:
            for part in selector.split(","):
                self.assertIn(".st-key-m7_cal_download", part, part)
        blue = [m.group(1) for m in re.finditer(r"([^{}]+)\{[^{}]*(?:#2563EB|\{DOWNLOAD_BLUE\})[^{}]*\}", css)]
        self.assertTrue(blue)
        for selector in blue:
            self.assertIn(":not(:disabled)", selector)             # disabled stays grey
        for key in ("m7_cal_reset_top", "m7_cal_reset_bottom", "m7_cal_run", "m7_cal_clear", "m7_experiment_btn"):
            self.assertNotIn(f".st-key-{key}", css)
        self.assertEqual(SOURCE.count('DOWNLOAD_BLUE = "#2563EB"'), 1)  # one constant backs the download button
        self.assertIn('"Download Calibration (.npz)"', SOURCE)
        self.assertIn('mime="application/octet-stream"', SOURCE)

    def test_blue_css_is_scoped_to_the_measure_another_image_button(self):
        block = SOURCE.split('/* "Measure Another Image"')[1].split('"""')[0]
        rules = [m.group(1) for m in re.finditer(r"([^{}]+)\{[^{}]*\}", block.replace("{{", "{").replace("}}", "}"))]
        rules = [r for r in rules if "*/" not in r or r.split("*/")[-1].strip()]
        selectors = [r.split("*/")[-1].strip() for r in rules]
        self.assertTrue(selectors)
        for selector in selectors:
            for part in selector.split(","):
                self.assertTrue(part.strip().startswith(".st-key-m7_another_image button"), part)
        self.assertIn("background-color: #2563EB", block)
        # neither blue rule set reaches Reset All (any of its three keys) or the other buttons
        css = SOURCE.split("DOWNLOAD_CSS = ")[1].split("PART_ONE_CSS")[0] + block
        for key in ("m7_cal_reset_top", "m7_cal_reset_bottom", "m7_results_reset_all", "m7_cal_run", "m7_cal_clear", "m7_experiment_btn"):
            self.assertNotIn(f"st-key-{key}", css)
        # and the download rules never mention the Measure Another Image key
        download = SOURCE.split("DOWNLOAD_CSS = ")[1].split("PART_ONE_CSS")[0]
        self.assertNotIn("m7_another_image", download)

    def test_export_failure_keeps_button_disabled_with_error(self):
        source = computed_model().session_state["m7_visitor_model"]
        broken = load_calibration(export_calibration(source, source.get("pairs_uploaded"), source.get("camera"), source.get("created_utc")))
        broken["T"] = np.array([np.nan, 0.0, 0.0])           # no longer exportable
        at = page(None, m7_visitor_model=broken)
        self.assertTrue(download_proto(at).disabled)
        self.assertTrue(any("cannot be exported" in e.value for e in at.error) or at.exception)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class MeasurementFlowTests(unittest.TestCase):
    def test_measurement_disabled_without_calibration(self):
        at = page()
        self.assertFalse([u for u in at.get("file_uploader") if str(u.proto.id).find("m7_up_") >= 0 or "m7_up_" in str(getattr(u, "key", ""))])
        self.assertIn("First calibrate your stereo cameras", texts(at))
        self.assertFalse(at.session_state["m7_measurement_enabled"])

    def test_measurement_enabled_by_computed_calibration_with_summary(self):
        at = computed_model()
        self.assertTrue(at.session_state["m7_measurement_enabled"])
        labels = [u.label for u in at.get("file_uploader")]
        self.assertIn("Left camera image", labels)
        self.assertIn("Right camera image", labels)
        metrics = {m.label: m.value for m in at.metric}
        self.assertEqual(metrics["Calibration source"], "Newly computed")
        self.assertEqual(metrics["Left image resolution"], f"{LEFT_SIZE[0]} × {LEFT_SIZE[1]} px")
        self.assertEqual(metrics["Right image resolution"], f"{RIGHT_SIZE[0]} × {RIGHT_SIZE[1]} px")

    def test_calibration_summary_two_by_two_layout_and_untruncated(self):
        at = computed_model()
        labels = ["Calibration source", "Baseline", "Left image resolution", "Right image resolution"]
        summary = [m for m in at.metric if m.label in labels]
        self.assertEqual({m.label for m in summary}, set(labels))
        for m in summary:
            self.assertNotIn("...", m.value)
            self.assertNotIn("…", m.value)
        import Pages.app_module7 as page_module
        css = page_module.CALIBRATION_SUMMARY_CSS
        self.assertIn("white-space: normal", css)
        self.assertIn("overflow: visible", css)
        self.assertNotIn("ellipsis", css)

    def test_different_resolutions_accepted_and_mismatch_blocks(self):
        at = computed_model()
        at.session_state["m7_file_left"] = ("l.jpg", jpeg(LEFT_SIZE))
        at.session_state["m7_file_right"] = ("r.jpg", jpeg(RIGHT_SIZE))
        at.run()
        self.assertFalse(at.exception)
        self.assertFalse([e for e in at.error if "rejected" in e.value])
        at.session_state["m7_file_right"] = ("r.jpg", jpeg(LEFT_SIZE))      # right image at the left camera's resolution
        at.run()
        self.assertTrue([e for e in at.error if "Right image" in e.value and "rejected" in e.value])
        self.assertNotIn("Calculate Dimensions", [b.label for b in at.button])

    def test_end_to_end_measurement_with_different_resolutions_and_ground_truth(self):
        at = computed_model()
        left_bytes, right_bytes = jpeg(LEFT_SIZE), jpeg(RIGHT_SIZE)
        at.session_state["m7_file_left"] = ("l.jpg", left_bytes)
        at.session_state["m7_file_right"] = ("r.jpg", right_bytes)
        at.run()
        pair_id = hashlib.sha256((hashlib.sha256(left_bytes).hexdigest() + hashlib.sha256(right_bytes).hexdigest()).encode()).hexdigest()[:16]
        self.assertEqual(at.session_state["m7_pair_id"], pair_id)
        obj = np.array([[-150, -100, 0], [150, -100, 0], [150, 100, 0], [-150, 100, 0]], np.float64) + [0, 0, 800]   # TL TR BR BL, mm
        left = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), RIG["K1"], RIG["D1"])[0].reshape(-1, 2)
        Rr, tr = RIG["R"], RIG["T"].ravel()
        right = cv2.projectPoints(obj, cv2.Rodrigues(Rr)[0], tr, RIG["K2"], RIG["D2"])[0].reshape(-1, 2)
        at.session_state["m7_points"] = {"left": [tuple(p) for p in left], "right": [tuple(p) for p in right]}
        at.session_state["m7_ack"] = pair_id
        at.session_state["m7_left_confirmed"] = at.session_state["m7_right_confirmed"] = True
        at.session_state["m7_calculated"] = True
        at.run()
        self.assertFalse(at.exception)
        metrics = {m.label: m.value for m in at.metric}
        self.assertAlmostEqual(float(metrics["Estimated width"].split()[0]), 300.0, delta=9)
        self.assertAlmostEqual(float(metrics["Estimated height"].split()[0]), 200.0, delta=6)
        self.assertAlmostEqual(float(metrics["Mean Z-depth"].split()[0]), 800.0, delta=24)
        self.assertIn("No actual dimensions were entered", texts(at))
        gen = at.session_state["m7_upload_gen"]
        at.number_input(key=f"m7_gt_w_{gen}").set_value(300.0).run()
        body = texts(at)
        self.assertNotIn("No actual dimensions were entered", body)
        self.assertIn("Comparison with the actual size", " ".join(m.value for m in at.markdown))
        button(at, "Measure Another Image").click().run()
        self.assertFalse([m for m in at.metric if m.label == "Estimated width"])
        self.assertEqual(at.session_state["m7_upload_gen"], gen + 1)
        self.assertFalse(download_proto(at).disabled)

    def test_reset_all_clears_measurement_but_not_the_saved_experiment(self):
        at = computed_model()
        at.session_state["m7_file_left"] = ("l.jpg", jpeg(LEFT_SIZE))
        at.run()
        button(at, "Reset All").click().run()
        self.assertIsNone(at.session_state["m7_file_left"])
        self.assertEqual(download(at), [])
        button(at, "View My Experiment & Results").click().run()
        self.assertFalse(at.exception)
        self.assertIn("Close experiment", [b.label for b in at.button])

    def test_experiment_available_without_visitor_calibration(self):
        at = page()
        button(at, "View My Experiment & Results").click().run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["m7_visitor_model"])


if __name__ == "__main__":
    unittest.main()
