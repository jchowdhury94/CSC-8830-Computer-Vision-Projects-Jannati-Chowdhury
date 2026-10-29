"""Calibration expander: workflow selector buttons at the top, Load / Calibrate New workflows, download rules, switching and reset."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import (AppTest, PAGE, button, filled_state, page, texts, wiz,  # noqa: E402
                                             export_calibration, load_calibration)

SOURCE = Path(PAGE).read_text(encoding="utf-8")
LABEL = "Download Calibration (.npz)"


def downloads(at):
    return [d for d in at.get("download_button") if d.proto.label == LABEL]


def npz_uploaders(at):
    return [u for u in at.get("file_uploader") if u.label == "Load Saved Calibration (.npz)"]


def start():
    return page(None, m7_cal_workflow=None)           # nothing selected: the initial state


def computed():
    at = page(filled_state())
    button(at, "Run Stereo Calibration").click().run()
    assert at.session_state["m7_cal_source"] == "computed"
    return at


def loaded_at():
    model = computed().session_state["m7_visitor_model"]
    loaded = load_calibration(export_calibration(model, model.get("pairs_uploaded"), model.get("camera"), model.get("created_utc")))
    return page(None, m7_visitor_model=loaded, m7_cal_loaded_name="mine.npz"), loaded


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class LayoutTests(unittest.TestCase):
    def test_initial_view_has_only_the_two_selector_buttons(self):
        at = start()
        self.assertEqual([b.label for b in at.button if b.key in ("m7_cal_mode_load", "m7_cal_mode_run")],
                         ["Load Saved Calibration", "Calibrate New Camera Setup"])
        self.assertEqual(button(at, "Load Saved Calibration").proto.type, "secondary")
        self.assertEqual(button(at, "Calibrate New Camera Setup").proto.type, "secondary")
        self.assertIn("Choose how to set up your stereo cameras.", texts(at))
        self.assertEqual([h.value for h in at.subheader if h.value[0] in "1234"], [])
        self.assertIsNone(button(at, "Run Stereo Calibration"))
        self.assertEqual(npz_uploaders(at), [])
        self.assertEqual([u.label for u in at.get("file_uploader") if "alibration" in u.label or "pair" in u.label.lower()], [])
        self.assertEqual(at.number_input, [])
        self.assertEqual(downloads(at), [])

    def test_selector_buttons_precede_everything_and_are_not_duplicated(self):
        body = SOURCE.split("def workflow_selector")[1].split("def new_calibration_steps")[0]
        self.assertEqual(SOURCE.count('"m7_cal_mode_load"'), 1)
        self.assertEqual(SOURCE.count('"m7_cal_mode_run"'), 1)
        flow = SOURCE.split("def calibrate_workflow")[1].split("# ----")[0]
        self.assertLess(flow.index("workflow_selector()"), flow.index("new_calibration_steps"))
        self.assertLess(flow.index("workflow_selector()"), flow.index("load_workflow"))
        self.assertIn("on_click=select_workflow", body)
        for at in (start(), page(filled_state())):
            self.assertEqual(len([b for b in at.button if b.label == "Calibrate New Camera Setup"]), 1)
            self.assertEqual(len([b for b in at.button if b.label == "Load Saved Calibration"]), 1)

    def test_initial_state_shows_neither_workflow(self):
        at = start()
        self.assertEqual(npz_uploaders(at), [])
        self.assertEqual(downloads(at), [])
        self.assertNotIn("Not ready yet", texts(at))

    def test_selector_css_is_scoped(self):
        css = SOURCE.split("MODE_CSS = ")[1].split("DOWNLOAD_BLUE")[0]
        for key in ("m7_cal_mode_run", "m7_cal_mode_load"):
            self.assertIn(f".st-key-{key}", css)
        self.assertNotIn("stButton", css)
        self.assertNotIn("\nbutton", css)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class WorkflowTests(unittest.TestCase):
    def test_new_setup_selector_shows_three_steps_and_explains_missing_inputs(self):
        at = start()
        button(at, "Calibrate New Camera Setup").click().run()
        self.assertEqual(at.session_state["m7_cal_workflow"], "run")
        self.assertEqual(button(at, "Calibrate New Camera Setup").proto.type, "primary")
        self.assertEqual(button(at, "Load Saved Calibration").proto.type, "secondary")
        self.assertEqual([h.value for h in at.subheader][:3], ["1. Checkerboard settings", "2. Upload stereo pairs", "3. Run calibration"])
        self.assertEqual(downloads(at)[0].proto.disabled, True)
        self.assertNotIn("Choose how to set up your stereo cameras.", texts(at))
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertEqual(at.session_state["m7_cal_validation"], [])
        button(at, "Run Stereo Calibration").click().run()
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        self.assertIn("Enter the checkerboard columns, rows, and square size before running calibration.", texts(at))
        self.assertEqual(npz_uploaders(at), [])
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertFalse(at.exception)

    def test_single_run_button_in_step_three(self):
        self.assertEqual(SOURCE.count('"Run Stereo Calibration"'), 1)
        at = page(filled_state())
        self.assertEqual([b.label for b in at.button if "Run" in b.label], ["Run Stereo Calibration"])
        self.assertEqual([b.key for b in at.button if b.label == "Run Stereo Calibration"], ["m7_cal_run"])

    def test_run_executes_calibration_exactly_once(self):
        at = page(filled_state(), m7_cal_workflow=None)
        button(at, "Calibrate New Camera Setup").click().run()
        calls = []
        real = wiz.run_calibration
        wiz.run_calibration = lambda state: (calls.append(1), real(state))[1]
        try:
            button(at, "Run Stereo Calibration").click().run()
            at.run()                                  # a plain rerun must not recalibrate
            at.toggle(key="m7_cal_details").set_value(True).run()
        finally:
            wiz.run_calibration = real
        self.assertEqual(len(calls), 1)
        self.assertEqual(at.session_state["m7_cal_source"], "computed")
        self.assertTrue(at.session_state["m7_measurement_enabled"] or at.session_state["m7_visitor_model"] is not None)
        self.assertIn("Stereo Calibration Completed", texts(at))
        self.assertFalse(downloads(at)[0].proto.disabled)

    def test_switching_to_load_does_not_run_calibration(self):
        at = page(filled_state(), m7_cal_workflow=None)
        button(at, "Load Saved Calibration").click().run()
        self.assertIsNone(at.session_state["m7_visitor_model"])

    def test_too_few_pairs_message_has_real_counts(self):
        at = page(filled_state(count=5))
        button(at, "Run Stereo Calibration").click().run()
        body = texts(at)
        self.assertIn("Only 5 valid stereo pairs are available. At least 8 are required. Add 3 more valid pairs.", body)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)

    def test_no_photos_message(self):
        state = filled_state()
        wiz.remove_all_photos(state)
        at = page(state)
        button(at, "Run Stereo Calibration").click().run()
        self.assertIn("No valid stereo pairs found. Upload at least 8 valid left/right checkerboard pairs.", texts(at))

    def test_incomplete_and_undetected_pairs_named(self):
        state = filled_state()
        state["photos"].pop(wiz.slot_key(3, "right"))
        state["photos"][wiz.slot_key(5, "left")]["status"] = "not_found"
        state["photos"][wiz.slot_key(5, "left")]["message"] = "no checkerboard"
        at = page(state)
        button(at, "Run Stereo Calibration").click().run()
        body = texts(at)
        self.assertIn("Pair 4 is incomplete. Upload both the left and right checkerboard photographs.", body)
        self.assertIn("Checkerboard corners could not be detected in Pair 6. Replace the photograph or exclude this pair.", body)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertFalse(at.exception)

    def test_failed_run_keeps_existing_models(self):
        at, loaded = loaded_at()
        button(at, "Calibrate New Camera Setup").click().run()
        button(at, "Run Stereo Calibration").click().run()
        self.assertIs(at.session_state["m7_visitor_model"], loaded)
        self.assertEqual(at.session_state["m7_cal_source"], "loaded")
        self.assertNotIn("Stereo Calibration Completed", texts(at))

    def test_calibration_error_keeps_computed_model(self):
        at = computed()
        model = at.session_state["m7_visitor_model"]
        real = wiz.run_calibration

        def fail(state):
            raise wiz.WizardError("Calibration failed: mocked")
        wiz.run_calibration = fail
        try:
            button(at, "Run Stereo Calibration").click().run()
        finally:
            wiz.run_calibration = real
        self.assertIs(at.session_state["m7_visitor_model"], model)
        self.assertEqual(at.session_state["m7_cal_source"], "computed")
        self.assertIn("Calibration failed: mocked", [e.value for e in at.error])
        self.assertFalse(at.exception)

    def test_load_selector_reveals_uploader_only(self):
        at = start()
        button(at, "Load Saved Calibration").click().run()
        self.assertEqual(at.session_state["m7_cal_workflow"], "load")
        self.assertEqual(len(npz_uploaders(at)), 1)
        self.assertIn("Load Saved Calibration (.npz)", " ".join(m.value for m in at.markdown))
        self.assertNotIn("Not ready yet", texts(at))
        self.assertEqual(downloads(at), [])
        self.assertIsNone(button(at, "Run Stereo Calibration"))
        self.assertEqual(at.number_input, [])
        self.assertEqual([h.value for h in at.subheader if h.value[0] in "123"], [])
        self.assertIsNone(at.session_state["m7_visitor_model"])

    def test_failed_npz_upload_keeps_active_calibration(self):
        at, loaded = loaded_at()
        at.session_state["m7_cal_load_error"] = "'bad.npz' was not loaded: invalid"
        at.run()
        self.assertIn("'bad.npz' was not loaded: invalid", [e.value for e in at.error])
        self.assertIs(at.session_state["m7_visitor_model"], loaded)

    def test_active_selector_is_primary(self):
        at = start()
        button(at, "Load Saved Calibration").click().run()
        self.assertEqual(button(at, "Load Saved Calibration").proto.type, "primary")
        self.assertEqual(button(at, "Calibrate New Camera Setup").proto.type, "secondary")


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class DownloadRuleTests(unittest.TestCase):
    def test_disabled_before_and_enabled_after_new_calibration(self):
        at = page(filled_state())
        self.assertEqual(len(downloads(at)), 1)
        self.assertTrue(downloads(at)[0].proto.disabled)
        button(at, "Run Stereo Calibration").click().run()
        self.assertFalse(downloads(at)[0].proto.disabled)
        self.assertEqual(len(downloads(at)), 1)
        body = texts(at)
        for text in ("Stereo Calibration Completed",):
            self.assertIn(text, body)
        self.assertIn("#2563EB", SOURCE)

    def test_loaded_npz_never_enables_download(self):
        at, _ = loaded_at()
        self.assertEqual(downloads(at), [])
        at.session_state["m7_cal_workflow"] = "run"
        at.run()
        self.assertTrue(downloads(at)[0].proto.disabled)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class LoadedResultsTests(unittest.TestCase):
    def test_loaded_results_appear_in_load_workflow(self):
        at, loaded = loaded_at()
        body = texts(at)
        self.assertEqual(at.session_state["m7_cal_workflow"], "load")
        for text in ("Stereo Calibration Loaded", "not authenticated"):
            self.assertIn(text, body)
        labels = {m.label for m in at.metric}
        for label in ("Uploaded pairs", "Valid pairs used", "Baseline", "Left camera RMS", "Right camera RMS", "Stereo RMS"):
            self.assertIn(label, labels)
        self.assertIsNotNone(button(at, "Clear Calibration"))
        self.assertIsNotNone(button(at, "Reset All"))
        self.assertNotIn("Not ready yet", body)

    def test_clear_in_load_workflow(self):
        at, _ = loaded_at()
        button(at, "Clear Calibration").click().run()
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertEqual(at.session_state["m7_cal_source"], "none")
        self.assertEqual(at.session_state["m7_cal_workflow"], "load")


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class SwitchingTests(unittest.TestCase):
    def test_switching_preserves_computed_calibration_and_hides_results(self):
        at = computed()
        model = at.session_state["m7_visitor_model"]
        button(at, "Load Saved Calibration").click().run()
        self.assertIs(at.session_state["m7_visitor_model"], model)
        self.assertEqual(at.session_state["m7_cal_source"], "computed")
        body = texts(at)
        self.assertNotIn("Stereo Calibration Completed", body)
        self.assertNotIn("Stereo Calibration Loaded", body)
        self.assertIn("computed from your photographs is currently active", body)
        self.assertEqual(downloads(at), [])
        self.assertTrue(at.session_state["m7_measurement_enabled"])
        at.session_state["m7_cal_workflow"] = "run"
        at.run()
        self.assertIs(at.session_state["m7_visitor_model"], model)
        self.assertIn("Stereo Calibration Completed", texts(at))
        self.assertFalse(downloads(at)[0].proto.disabled)

    def test_switching_preserves_loaded_calibration_and_is_not_shown_as_computed(self):
        at, loaded = loaded_at()
        button(at, "Calibrate New Camera Setup").click().run()
        button(at, "Run Stereo Calibration").click().run()
        self.assertIs(at.session_state["m7_visitor_model"], loaded)
        self.assertEqual(at.session_state["m7_cal_source"], "loaded")
        body = texts(at)
        self.assertNotIn("Stereo Calibration Completed", body)
        self.assertNotIn("Stereo Calibration Loaded", body)
        self.assertIn("mine.npz", body)
        self.assertTrue(downloads(at)[0].proto.disabled)
        self.assertTrue(at.session_state["m7_measurement_enabled"])


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class MeasurementAndResetTests(unittest.TestCase):
    def test_measurement_activates_for_both_sources_in_either_workflow(self):
        for at in (computed(), loaded_at()[0]):
            for name in ("Calibrate New Camera Setup", "Load Saved Calibration"):
                button(at, name).click().run()
                self.assertTrue(at.session_state["m7_measurement_enabled"])
                self.assertIn("Left camera image", [u.label for u in at.get("file_uploader")])
        self.assertIn('ss["m7_visitor_model"]', SOURCE)
        self.assertNotIn("load_model_b", SOURCE.split("def upload_section")[1].split("def corner_instructions")[0])

    def test_reset_all_restores_initial_state(self):
        at = computed()
        button(at, "Reset All").click().run()
        ss = at.session_state
        self.assertIsNone(ss["m7_visitor_model"])
        self.assertIsNone(ss["m7_cal_workflow"])
        self.assertEqual(ss["m7_cal_source"], "none")
        self.assertFalse(ss["m7_measurement_enabled"])
        self.assertEqual(downloads(at), [])
        self.assertEqual(npz_uploaders(at), [])
        self.assertEqual(ss["m7_cal"]["n_pairs"], 8)

    def test_model_b_experiment_unaffected(self):
        at = computed()
        button(at, "Reset All").click().run()
        button(at, "View My Experiment & Results").click().run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIn("Close experiment", [b.label for b in at.button])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class ClearAndResetSelectionTests(unittest.TestCase):
    def test_clear_calibration_keeps_new_setup_workflow(self):
        at = computed()
        button(at, "Clear Calibration").click().run()
        self.assertEqual(at.session_state["m7_cal_workflow"], "run")
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIsNotNone(button(at, "Run Stereo Calibration"))
        self.assertTrue(downloads(at)[0].proto.disabled)

    def test_reset_all_from_each_workflow_returns_to_choice_screen(self):
        for at in (computed(), loaded_at()[0]):
            button(at, "Reset All").click().run()
            self.assertIsNone(at.session_state["m7_cal_workflow"])
            self.assertIn("Choose how to set up your stereo cameras.", texts(at))
            self.assertEqual(npz_uploaders(at), [])
            self.assertEqual(downloads(at), [])
            self.assertIsNone(button(at, "Run Stereo Calibration"))
            self.assertEqual(button(at, "Load Saved Calibration").proto.type, "secondary")
            self.assertEqual(button(at, "Calibrate New Camera Setup").proto.type, "secondary")

    def test_reset_all_then_new_setup_starts_blank(self):
        at = computed()
        button(at, "Reset All").click().run()
        button(at, "Calibrate New Camera Setup").click().run()
        self.assertNotIn("Stereo Calibration Completed", texts(at))
        self.assertTrue(downloads(at)[0].proto.disabled)
        self.assertIsNone(at.session_state["m7_visitor_model"])
