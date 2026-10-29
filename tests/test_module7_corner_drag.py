"""Draggable zoom-refinement tests: coordinate conversion, pending/accepted state transitions and the Accept/Cancel layout."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modules" / "Module_7"))

from stereo_measure import corner_drag as cd  # noqa: E402
from stereo_measure import selection as sel  # noqa: E402
from test_module7_calibration_wizard import AppTest, PAGE, button  # noqa: E402
from test_module7_phase6 import computed_model, jpeg  # noqa: E402
from test_module7_user_calibration import LEFT_SIZE, RIGHT_SIZE  # noqa: E402

SOURCE = Path(PAGE).read_text(encoding="utf-8")
FRONTEND = (Path(cd.__file__).parent / "corner_drag_frontend" / "index.html").read_text(encoding="utf-8")
ORIGIN, CROP, IMAGE = (100, 200), (160, 160), (1000, 800)


class ConversionTests(unittest.TestCase):
    def test_centre_of_crop_maps_to_crop_centre(self):
        self.assertEqual(cd.fraction_to_source((0.5, 0.5), ORIGIN, CROP, IMAGE), (100 + 80 - 0.5, 200 + 80 - 0.5))

    def test_round_trip_preserves_subpixel_precision(self):
        pt = (123.4567, 251.8901)
        frac = cd.source_to_fraction(pt, ORIGIN, CROP)
        back = cd.fraction_to_source(frac, ORIGIN, CROP, IMAGE)
        self.assertAlmostEqual(back[0], pt[0], places=9)
        self.assertAlmostEqual(back[1], pt[1], places=9)

    def test_display_scale_independent_and_matches_click_mapping(self):
        # a pixel-centre click at rendered offset u of a rw-wide image is the same point as fraction (u + 0.5) / rw
        for rw in (320, 640, 733):
            click = dict(x=200, y=40, width=rw, height=rw)
            expected = sel.click_to_source(click, ORIGIN, CROP)
            got = cd.fraction_to_source(((200 + 0.5) / rw, (40 + 0.5) / rw), ORIGIN, CROP, IMAGE)
            self.assertAlmostEqual(got[0], expected[0], places=9)
            self.assertAlmostEqual(got[1], expected[1], places=9)

    def test_marker_clamped_inside_image(self):
        self.assertEqual(cd.fraction_to_source((-3, -3), (0, 0), CROP, IMAGE), (0.0, 0.0))
        self.assertEqual(cd.fraction_to_source((9, 9), (840, 640), CROP, IMAGE), (999.0, 799.0))

    def test_limits_keep_marker_on_image_for_edge_crops(self):
        fx0, fx1, fy0, fy1 = cd.fraction_limits((0, 0), CROP, IMAGE)
        self.assertAlmostEqual(cd.fraction_to_source((fx0, fy0), (0, 0), CROP, IMAGE)[0], 0.0)
        fx0, fx1, fy0, fy1 = cd.fraction_limits((840, 640), CROP, IMAGE)
        self.assertAlmostEqual(cd.fraction_to_source((fx1, fy1), (840, 640), CROP, IMAGE)[0], 999.0)
        self.assertEqual(cd.fraction_limits(ORIGIN, CROP, IMAGE), (0.0, 1.0, 0.0, 1.0))

    def test_refined_point_ignores_other_tokens_and_bad_values(self):
        value = dict(fx=0.25, fy=0.75, seq=2, token="left-3-0")
        self.assertEqual(cd.refined_point(value, "left-3-0", ORIGIN, CROP, IMAGE), (100 + 40 - 0.5, 200 + 120 - 0.5))
        self.assertIsNone(cd.refined_point(value, "left-4-0", ORIGIN, CROP, IMAGE))
        self.assertIsNone(cd.refined_point(None, "x", ORIGIN, CROP, IMAGE))
        self.assertIsNone(cd.refined_point(dict(fx="a", fy=0, token="t"), "t", ORIGIN, CROP, IMAGE))
        self.assertIsNone(cd.refined_point(dict(fx=float("nan"), fy=0, token="t"), "t", ORIGIN, CROP, IMAGE))

    def test_multiple_drags_keep_only_the_latest(self):
        latest = None
        for fx in (0.2, 0.9, 0.55):
            latest = cd.refined_point(dict(fx=fx, fy=0.5, seq=1, token="t"), "t", ORIGIN, CROP, IMAGE)
        self.assertAlmostEqual(latest[0], 100 + 0.55 * 160 - 0.5)


class FrontendTests(unittest.TestCase):
    def test_dragging_is_real_pointer_dragging_and_commits_only_on_release(self):
        for needle in ("pointerdown", "pointermove", "pointerup", "setPointerCapture", "limits[0]"):
            self.assertIn(needle, FRONTEND)
        move = FRONTEND.split('"pointermove"')[1].split("function release")[0]
        self.assertNotIn("setComponentValue", move)
        self.assertEqual(FRONTEND.count("setComponentValue"), 1)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class PageTests(unittest.TestCase):
    def ready(self):
        at = computed_model()
        at.session_state["m7_file_left"] = ("l.jpg", jpeg(LEFT_SIZE))
        at.session_state["m7_file_right"] = ("r.jpg", jpeg(RIGHT_SIZE))
        at.run()
        at.session_state["m7_ack"] = at.session_state["m7_pair_id"]
        at.run()
        return at

    def pend(self, at, side="left", point=(300.25, 400.5)):
        at.session_state["m7_pending"] = (side, point)
        at.session_state["m7_refined"] = None
        at.run()

    def labels(self, at):
        return [b.label for b in at.button]

    def test_pending_corner_opens_zoom_with_accept_left_and_cancel_right(self):
        at = self.ready()
        self.assertNotIn("Accept Corner", self.labels(at))
        self.pend(at)
        self.assertFalse(at.exception)
        self.assertIn("Accept Corner", self.labels(at))
        self.assertLess(self.labels(at).index("Accept Corner"), self.labels(at).index("Cancel"))
        self.assertEqual(self.labels(at).index("Cancel"), self.labels(at).index("Accept Corner") + 1)
        accept, cancel = [b for b in at.button if b.label in ("Accept Corner", "Cancel")]
        self.assertEqual((accept.key, cancel.key), ("m7_zoom_accept", "m7_zoom_cancel"))

    def test_buttons_sit_in_two_equal_columns_directly_after_the_zoom_component(self):
        at = self.ready()
        self.pend(at)
        container = [e for e in at.main if getattr(e, "key", None) == "m7_zoom_actions"] or None
        body = SOURCE.split("def corner_selection")[1]
        zoom = body.index("cd.corner_drag(")
        self.assertLess(zoom, body.index('st.container(key="m7_zoom_actions")'))
        self.assertIn("st.columns(2)", body.split('st.container(key="m7_zoom_actions")')[1][:200])
        between = body[zoom:body.index('st.container(key="m7_zoom_actions")')]
        self.assertNotIn("st.image", between)
        self.assertNotIn("st.button", between)

    def test_marker_starts_at_the_clicked_point_and_cancel_discards_everything(self):
        at = self.ready()
        at.session_state["m7_points"]["left"].append((10.5, 20.5))
        self.pend(at)
        self.assertEqual(at.session_state["m7_pending"], ("left", (300.25, 400.5)))
        at.session_state["m7_refined"] = (305.0, 402.0)         # an unaccepted drag
        button(at, "Cancel").click().run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["m7_pending"])
        self.assertIsNone(at.session_state["m7_refined"])
        self.assertEqual(at.session_state["m7_points"]["left"], [(10.5, 20.5)])
        self.assertEqual(at.session_state["m7_points"]["right"], [])
        self.assertNotIn("Accept Corner", self.labels(at))

    def test_accept_commits_dragged_point_to_the_active_side_only(self):
        at = self.ready()
        at.session_state["m7_points"]["left"].append((10.5, 20.5))
        self.pend(at)
        at.session_state["m7_refined"] = (305.125, 402.75)
        button(at, "Accept Corner").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["m7_points"]["left"], [(10.5, 20.5), (305.125, 402.75)])
        self.assertEqual(at.session_state["m7_points"]["right"], [])
        self.assertIsNone(at.session_state["m7_pending"])
        self.assertIsNone(at.session_state["m7_refined"])
        self.assertNotIn("Accept Corner", self.labels(at))

    def test_accept_without_a_drag_keeps_the_clicked_point(self):
        at = self.ready()
        self.pend(at, point=(300.25, 400.5))
        button(at, "Accept Corner").click().run()
        self.assertEqual(at.session_state["m7_points"]["left"], [(300.25, 400.5)])

    def test_right_side_selection_is_independent(self):
        at = self.ready()
        at.session_state["m7_points"]["left"] = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)]
        at.session_state["m7_left_confirmed"] = True
        at.session_state["m7_ack_right"] = at.session_state["m7_pair_id"]
        self.pend(at, "right", (50.5, 60.5))
        at.session_state["m7_refined"] = (51.5, 61.5)
        button(at, "Accept Corner").click().run()
        self.assertEqual(at.session_state["m7_points"]["right"], [(51.5, 61.5)])
        self.assertEqual(len(at.session_state["m7_points"]["left"]), 4)

    def test_undo_reset_and_reset_all_clear_pending(self):
        for action in ("Undo last point", "Reset both", "Reset All"):
            at = self.ready()
            self.pend(at)
            at.session_state["m7_refined"] = (301.0, 401.0)
            button(at, action).click().run()
            self.assertFalse(at.exception, action)
            self.assertIsNone(at.session_state["m7_pending"], action)
            self.assertIsNone(at.session_state["m7_refined"], action)

    def test_new_pair_invalidates_pending_refinement(self):
        at = self.ready()
        self.pend(at)
        at.session_state["m7_refined"] = (301.0, 401.0)
        at.session_state["m7_file_left"] = ("l2.jpg", jpeg(LEFT_SIZE) + b"\0")
        at.run()
        self.assertIsNone(at.session_state["m7_pending"])
        self.assertIsNone(at.session_state["m7_refined"])


class StyleScopeTests(unittest.TestCase):
    def test_css_targets_only_the_two_keyed_buttons(self):
        css = SOURCE.split("ZOOM_BUTTON_CSS = ")[1].split("EXPERIMENT_OPEN_CSS")[0]
        selectors = re.findall(r"^(\.[^{]+)\{", css, re.M)
        self.assertTrue(selectors)
        self.assertTrue(all(s.startswith((".st-key-m7_zoom_accept", ".st-key-m7_zoom_cancel")) for s in selectors))
        self.assertIn("#1b8a3a", css.split("m7_zoom_cancel")[0])
        self.assertIn("font-weight: 700", css)


class MeasurementUnchangedTests(unittest.TestCase):
    def test_drag_component_does_not_touch_measurement_code(self):
        module = Path(cd.__file__).read_text(encoding="utf-8")
        for name in ("triangulation", "measurement", "live", "calibration"):
            self.assertNotIn(name, module)
        # accepted points reach measure_points exactly as stored, in original pixel coordinates
        self.assertIn("measure_points(", SOURCE)


if __name__ == "__main__":
    unittest.main()
