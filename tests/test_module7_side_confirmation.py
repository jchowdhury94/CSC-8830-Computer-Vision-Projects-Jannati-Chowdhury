"""Per-image corner instructions and the Confirm Left / Right Image Selections workflow."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import PAGE, button  # noqa: E402
from test_module7_phase6 import computed_model, jpeg  # noqa: E402
from test_module7_results_actions import keys, measured  # noqa: E402
from test_module7_user_calibration import LEFT_SIZE, RIGHT_SIZE  # noqa: E402

PTS = [(100.5, 200.5), (900.25, 210.0), (880.0, 700.0), (120.0, 690.0)]


def labels(at):
    return [b.label for b in at.button]


def dialog_text(at):
    """All text in the page including any open dialog."""
    out = []
    def walk(node):
        for child in getattr(node, "children", {}).values():
            if hasattr(child, "value") and isinstance(child.value, str):
                out.append(child.value)
            walk(child)
    walk(at.main)
    try:
        walk(at.dialog)
    except Exception:
        pass
    return " ".join(out)


def fresh():
    at = computed_model()
    at.session_state["m7_file_left"] = ("l.jpg", jpeg(LEFT_SIZE))
    at.session_state["m7_file_right"] = ("r.jpg", jpeg(RIGHT_SIZE))
    at.run()
    return at


def left_started():
    at = fresh()
    at.session_state["m7_ack"] = at.session_state["m7_pair_id"]
    at.run()
    return at


def left_done(n=4):
    at = left_started()
    at.session_state["m7_points"]["left"] = list(PTS[:n])
    at.run()
    return at


def right_stage():
    at = left_done()
    button(at, "Confirm Left Image Selections").click().run()
    return at


def right_started(n=0):
    at = right_stage()
    at.session_state["m7_ack_right"] = at.session_state["m7_pair_id"]
    at.session_state["m7_points"]["right"] = list(PTS[:n])
    at.run()
    return at


class LeftStageTests(unittest.TestCase):
    def test_left_popup_first_and_right_not_shown(self):
        at = fresh()
        ss = at.session_state
        self.assertEqual(ss["m7_dialog_opened"], ss["m7_pair_id"])
        self.assertIsNone(ss["m7_dialog_right"])
        self.assertEqual(ss["m7_stage"], "left")
        self.assertIn("Step 1: Select Left Image Corners", dialog_text(at))
        self.assertNotIn("Step 2", dialog_text(at))
        self.assertNotIn("Confirm Left Image Selections", labels(at))

    def test_popup_text_identifies_left_and_right(self):
        src = Path(PAGE).read_text(encoding="utf-8")
        self.assertIn('@st.dialog("Select Corners — Left Image")', src)
        self.assertIn('@st.dialog("Select Corners — Right Image")', src)
        self.assertIn("You are selecting corners on the {side.upper()} camera image.", src)
        for text in ("Top-left (TL)", "Top-right (TR)", "Bottom-right (BR)", "Bottom-left (BL)", "Accept Corner"):
            self.assertIn(text, src)

    def test_four_left_corners_enable_confirmation_fewer_do_not(self):
        self.assertNotIn("Confirm Left Image Selections", labels(left_done(3)))
        self.assertNotIn("Confirm Left Image Selections", labels(left_done(0)))
        self.assertIn("Confirm Left Image Selections", labels(left_done(4)))

    def test_no_measurement_before_confirmation(self):
        at = left_done()
        self.assertNotIn("Calculate Dimensions", labels(at))
        self.assertNotIn("Confirm Right Image Selections", labels(at))
        self.assertFalse(at.session_state["m7_calculated"])


class RightStageTests(unittest.TestCase):
    def test_left_confirmation_advances_and_opens_right_popup_once(self):
        at = right_stage()
        ss = at.session_state
        self.assertTrue(ss["m7_left_confirmed"])
        self.assertEqual(ss["m7_points"]["left"], PTS)
        self.assertEqual(ss["m7_dialog_right"], ss["m7_pair_id"])
        self.assertIn("Step 2: Select Right Image Corners", dialog_text(at))
        self.assertEqual(ss["m7_stage"], "right")
        self.assertIsNone(ss["m7_ack_right"])
        self.assertNotIn("Confirm Left Image Selections", labels(at))
        at.run()                                                # ordinary rerun: not reopened
        self.assertEqual(ss["m7_dialog_right"], ss["m7_pair_id"])
        self.assertIn("Show corner-selection instructions", labels(at))
        self.assertNotIn("m7_zoom_accept", keys(at))

    def test_right_popup_reopens_only_after_a_new_transition(self):
        at = right_stage()
        pair = at.session_state["m7_pair_id"]
        at.session_state["m7_points"]["left"].pop()             # editing left returns to the left stage
        at.session_state["m7_left_confirmed"] = False
        at.session_state["m7_dialog_right"] = None
        at.run()
        self.assertEqual(at.session_state["m7_stage"], "left")
        at.session_state["m7_points"]["left"] = list(PTS)
        at.run()
        button(at, "Confirm Left Image Selections").click().run()
        self.assertEqual(at.session_state["m7_dialog_right"], pair)

    def test_right_confirmation_needs_four_right_corners(self):
        self.assertNotIn("Confirm Right Image Selections", labels(right_started(3)))
        at = right_started(4)
        self.assertIn("Confirm Right Image Selections", labels(at))
        self.assertNotIn("Calculate Dimensions", labels(at))

    def test_both_confirmations_required_for_measurement(self):
        at = right_started(4)
        self.assertNotIn("Calculate Dimensions", labels(at))
        button(at, "Confirm Right Image Selections").click().run()
        self.assertTrue(at.session_state["m7_right_confirmed"])
        self.assertEqual(at.session_state["m7_points"]["right"], PTS)
        self.assertEqual(at.session_state["m7_stage"], "ready")
        self.assertIn("Calculate Dimensions", labels(at))
        # right confirmation alone (left withdrawn) must not allow measurement
        at.session_state["m7_left_confirmed"] = False
        at.run()
        self.assertNotIn("Calculate Dimensions", labels(at))

    def test_right_confirm_ignored_before_left_confirmation(self):
        at = left_done()
        at.session_state["m7_points"]["right"] = list(PTS)
        at.run()
        self.assertNotIn("Confirm Right Image Selections", labels(at))
        self.assertFalse(at.session_state["m7_right_confirmed"])

    def test_right_selection_inactive_until_left_confirmed(self):
        at = left_done(2)
        self.assertEqual(at.session_state["m7_stage"], "left")
        at.session_state["m7_pending"] = ("left", (300.0, 400.0))
        at.run()
        self.assertEqual(at.session_state["m7_stage"], "left")


class InvalidationTests(unittest.TestCase):
    def confirmed(self):
        at = right_started(4)
        button(at, "Confirm Right Image Selections").click().run()
        button(at, "Calculate Dimensions").click().run()
        self.assertTrue(at.session_state["m7_calculated"])
        return at

    def test_undo_right_invalidates_right_only(self):
        at = self.confirmed()
        button(at, "Undo last point").click().run()
        ss = at.session_state
        self.assertFalse(ss["m7_right_confirmed"])
        self.assertTrue(ss["m7_left_confirmed"])
        self.assertFalse(ss["m7_calculated"])
        self.assertEqual(len(ss["m7_points"]["right"]), 3)
        self.assertNotIn("m7_another_image", keys(at))

    def test_reset_left_invalidates_both_and_keeps_right_corners(self):
        at = self.confirmed()
        button(at, "Reset left").click().run()
        ss = at.session_state
        self.assertFalse(ss["m7_left_confirmed"] or ss["m7_right_confirmed"] or ss["m7_calculated"])
        self.assertEqual(ss["m7_points"]["left"], [])
        self.assertEqual(ss["m7_points"]["right"], PTS)
        self.assertEqual(ss["m7_stage"], "left")
        self.assertIsNone(ss["m7_dialog_right"])

    def test_reset_both_and_new_images(self):
        at = self.confirmed()
        button(at, "Reset both").click().run()
        ss = at.session_state
        self.assertFalse(ss["m7_left_confirmed"] or ss["m7_right_confirmed"])
        self.assertEqual(ss["m7_points"], {"left": [], "right": []})
        at = self.confirmed()
        at.session_state["m7_file_left"] = ("l2.jpg", jpeg(LEFT_SIZE) + b"\0")     # a replaced photograph is a new pair
        at.run()
        ss = at.session_state
        self.assertFalse(ss["m7_left_confirmed"] or ss["m7_right_confirmed"] or ss["m7_calculated"])
        self.assertEqual(ss["m7_points"], {"left": [], "right": []})
        self.assertEqual(ss["m7_stage"], "left")
        self.assertIsNone(ss["m7_ack_right"])

    def test_measure_another_image_restarts_workflow(self):
        at = measured()
        button(at, "Measure Another Image").click().run()
        ss = at.session_state
        self.assertFalse(ss["m7_left_confirmed"] or ss["m7_right_confirmed"])
        self.assertEqual(ss["m7_stage"], "left")
        self.assertIsNone(ss["m7_ack"]); self.assertIsNone(ss["m7_ack_right"])
        self.assertIsNone(ss["m7_dialog_opened"]); self.assertIsNone(ss["m7_dialog_right"])
        self.assertIsNotNone(ss["m7_visitor_model"])

    def test_reset_all_clears_both_confirmations(self):
        at = self.confirmed()
        next(b for b in at.button if b.key == "m7_cal_reset_top").click().run()
        ss = at.session_state
        self.assertFalse(ss["m7_left_confirmed"] or ss["m7_right_confirmed"])
        self.assertIsNone(ss["m7_ack_right"]); self.assertIsNone(ss["m7_dialog_right"])
        self.assertEqual(ss["m7_stage"], "left")


class RefinementStillWorksTests(unittest.TestCase):
    def test_accept_cancel_layout_and_acceptance_on_right_stage(self):
        at = right_started(0)
        at.session_state["m7_pending"] = ("right", (300.25, 400.5))
        at.run()
        self.assertEqual(keys(at).count("m7_zoom_accept"), 1)
        self.assertEqual(keys(at).count("m7_zoom_cancel"), 1)
        at.session_state["m7_refined"] = (301.5, 401.5)
        button(at, "Accept Corner").click().run()
        self.assertEqual(at.session_state["m7_points"]["right"], [(301.5, 401.5)])
        self.assertTrue(at.session_state["m7_left_confirmed"])

    def test_accepting_a_left_corner_withdraws_confirmations(self):
        at = right_stage()
        at.session_state["m7_points"]["left"].pop()
        at.session_state["m7_left_confirmed"] = False
        at.run()
        at.session_state["m7_pending"] = ("left", (300.25, 400.5))
        at.run()
        button(at, "Accept Corner").click().run()
        self.assertFalse(at.session_state["m7_left_confirmed"])
        self.assertEqual(len(at.session_state["m7_points"]["left"]), 4)

    def test_saved_experiment_unaffected(self):
        at = right_started(4)
        before = at.session_state["m7_show_experiment"]
        button(at, "Confirm Right Image Selections").click().run()
        self.assertEqual(at.session_state["m7_show_experiment"], before)
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
