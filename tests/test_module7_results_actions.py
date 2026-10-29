"""Measure Another Image / Reset All buttons below the measurement results."""
import hashlib
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import PAGE, RIG, button  # noqa: E402
from test_module7_phase6 import computed_model, jpeg  # noqa: E402
from test_module7_user_calibration import LEFT_SIZE, RIGHT_SIZE  # noqa: E402

SOURCE = Path(PAGE).read_text(encoding="utf-8")
SECTIONS = ("Measurement summary", "Detailed measurements", "Reconstructed 3D coordinates", "Comparison with the actual size")


def keys(at):
    return [getattr(b, "key", None) for b in at.button]


def uploaded(at):
    left, right = jpeg(LEFT_SIZE), jpeg(RIGHT_SIZE)
    at.session_state["m7_file_left"] = ("l.jpg", left)
    at.session_state["m7_file_right"] = ("r.jpg", right)
    at.run()
    return at


def measured(actual_width=300.0):
    at = uploaded(computed_model())
    pair_id = at.session_state["m7_pair_id"]
    obj = np.array([[-150, -100, 0], [150, -100, 0], [150, 100, 0], [-150, 100, 0]], np.float64) + [0, 0, 800]
    left = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), RIG["K1"], RIG["D1"])[0].reshape(-1, 2)
    right = cv2.projectPoints(obj, cv2.Rodrigues(RIG["R"])[0], RIG["T"].ravel(), RIG["K2"], RIG["D2"])[0].reshape(-1, 2)
    at.session_state["m7_points"] = {"left": [tuple(p) for p in left], "right": [tuple(p) for p in right]}
    at.session_state["m7_ack"] = pair_id
    at.session_state["m7_left_confirmed"] = at.session_state["m7_right_confirmed"] = at.session_state["m7_calculated"] = True
    at.run()
    at.number_input(key=f"m7_gt_w_{at.session_state['m7_upload_gen']}").set_value(actual_width).run()
    assert not at.exception and [m for m in at.metric if m.label == "Estimated width"]
    return at


class VisibilityTests(unittest.TestCase):
    def test_old_button_removed_everywhere(self):
        self.assertNotIn("Measure Another Object", SOURCE)
        at = uploaded(computed_model())
        self.assertNotIn("Measure Another Object", [b.label for b in at.button])

    def test_absent_before_results(self):
        for at in (computed_model(), uploaded(computed_model())):
            self.assertNotIn("m7_another_image", keys(at))
            self.assertNotIn("m7_results_reset_all", keys(at))
        at = uploaded(computed_model())
        at.session_state["m7_ack"] = at.session_state["m7_pair_id"]
        at.run()
        self.assertNotIn("m7_another_image", keys(at))
        self.assertNotIn("m7_results_reset_all", keys(at))

    def test_present_after_results_and_hidden_once_cleared(self):
        at = measured()
        self.assertIn("m7_another_image", keys(at))
        self.assertIn("m7_results_reset_all", keys(at))
        self.assertIn("m7_cal_reset_top", keys(at))             # calibration workflow buttons are kept
        self.assertIn("m7_cal_reset_bottom", keys(at))
        at.session_state["m7_calculated"] = False
        at.run()
        self.assertNotIn("m7_another_image", keys(at))
        self.assertNotIn("m7_results_reset_all", keys(at))

    def test_layout_left_right_and_below_all_results(self):
        at = measured()
        labels = [b.label for b in at.button]
        self.assertEqual(labels.count("Measure Another Image"), 1)
        cols = [c for c in at.columns if any(getattr(e, "key", None) in ("m7_another_image", "m7_results_reset_all") for e in c.children.values())]
        self.assertEqual([[e.key for e in c.children.values()] for c in cols], [["m7_another_image"], ["m7_results_reset_all"]])
        self.assertEqual(next(e for e in cols[1].children.values()).label, "Reset All")
        # nothing but the buttons follows the last result section in source order
        body = SOURCE[SOURCE.index("def show_results"):SOURCE.index("def part_one")]
        for section in SECTIONS + ("How stereo triangulation estimates depth and size",):
            self.assertLess(body.index(section), body.index('"Measure Another Image"'), section)
        self.assertLess(body.index('"Measure Another Image"'), body.index('"Reset All"'))
        self.assertTrue([m for m in at.markdown if "Comparison with the actual size" in m.value])

    def test_buttons_follow_every_result_element_in_tree(self):
        at = measured()
        order = []
        def walk(node):
            for child in getattr(node, "children", {}).values():
                order.append(child)
                walk(child)
        walk(at.main)
        pos = {id(e): i for i, e in enumerate(order)}
        another = next(pos[id(e)] for e in order if getattr(e, "key", None) == "m7_another_image")
        for table in at.get("table"):
            self.assertLess(pos[id(table)], another)
        for m in at.metric:
            self.assertLess(pos[id(m)], another)
        expander = [e for e in order if getattr(e, "label", None) == "How stereo triangulation estimates depth and size"]
        self.assertTrue(expander)
        self.assertLess(pos[id(expander[0])], another)


class BehaviourTests(unittest.TestCase):
    def test_measure_another_image_clears_object_state_keeps_calibration(self):
        at = measured()
        model, source, gen = at.session_state["m7_visitor_model"], at.session_state["m7_cal_source"], at.session_state["m7_upload_gen"]
        at.session_state["m7_pending"] = ("left", (1.0, 2.0))
        at.session_state["m7_refined"] = (1.5, 2.5)
        button(at, "Measure Another Image").click().run()
        ss = at.session_state
        self.assertFalse(at.exception)
        self.assertIsNone(ss["m7_file_left"]); self.assertIsNone(ss["m7_file_right"])
        self.assertEqual(ss["m7_points"], {"left": [], "right": []})
        self.assertIsNone(ss["m7_pending"]); self.assertIsNone(ss["m7_refined"])
        self.assertFalse(ss["m7_left_confirmed"] or ss["m7_right_confirmed"]); self.assertFalse(ss["m7_calculated"])
        self.assertIsNone(ss["m7_pair_id"]); self.assertIsNone(ss["m7_ack"])
        self.assertEqual(ss["m7_upload_gen"], gen + 1)
        self.assertFalse([m for m in at.metric if m.label == "Estimated width"])
        self.assertNotIn("m7_another_image", keys(at))
        self.assertIs(ss["m7_visitor_model"], model)
        self.assertEqual(ss["m7_cal_source"], source)
        self.assertTrue(ss["m7_measurement_enabled"])
        # fresh uploaders and actual-size inputs: no stale files or values come back
        self.assertFalse([n for n in at.number_input if n.key == f"m7_gt_w_{gen}"])
        uploaded(at)                                            # a new pair starts with empty actual-size inputs
        self.assertIsNone(at.number_input(key=f"m7_gt_w_{gen + 1}").value) if at.session_state["m7_ack"] else None
        self.assertIn(f"m7_up_left_{gen + 1}", [u.key for u in at.get("file_uploader")])
        self.assertNotIn(f"m7_up_left_{gen}", [u.key for u in at.get("file_uploader")])

    def test_reset_all_below_results_clears_everything(self):
        at = measured()
        at.session_state["m7_pending"] = ("left", (1.0, 2.0))
        next(b for b in at.button if b.key == "m7_results_reset_all").click().run()
        ss = at.session_state
        self.assertFalse(at.exception)
        self.assertIsNone(ss["m7_visitor_model"])
        self.assertEqual(ss["m7_cal_source"], "none")
        self.assertFalse(ss["m7_measurement_enabled"])
        self.assertIsNone(ss["m7_file_left"]); self.assertIsNone(ss["m7_file_right"])
        self.assertEqual(ss["m7_points"], {"left": [], "right": []})
        self.assertIsNone(ss["m7_pending"])
        self.assertIsNone(ss["m7_cal_workflow"])
        self.assertFalse([m for m in at.metric if m.label == "Estimated width"])
        self.assertNotIn("m7_results_reset_all", keys(at))

    def test_buttons_use_the_existing_reset_all_callback(self):
        self.assertIn('key="m7_results_reset_all", on_click=reset_all', SOURCE)

    def test_saved_experiment_unchanged(self):
        for action in ("m7_another_image", "m7_results_reset_all"):
            at = measured()
            before = at.session_state["m7_show_experiment"]
            next(b for b in at.button if b.key == action).click().run()
            self.assertEqual(at.session_state["m7_show_experiment"], before)
            self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
