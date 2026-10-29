"""Compact side-by-side images during corner selection (display only; coordinates stay in original pixels)."""
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Modules" / "Module_7"))

from test_module7_calibration_wizard import button  # noqa: E402
from test_module7_side_confirmation import PTS, left_started, right_started  # noqa: E402
from stereo_measure import selection as sel  # noqa: E402


PAGE_COMPACT_WIDTH = int(re.search(r"^COMPACT_WIDTH = (\d+)", (Path(__file__).resolve().parent.parent / "Pages" / "app_module7.py")
                                   .read_text(), re.M).group(1))


APP_SOURCE = (Path(__file__).resolve().parent.parent / "Pages" / "app_module7.py").read_text()


def app_compact_height(preview):
    ns = {"COMPACT_WIDTH": PAGE_COMPACT_WIDTH}
    start = APP_SOURCE.index("def compact_height")
    exec(APP_SOURCE[start:APP_SOURCE.index("def compact_pair")], ns)
    return ns["compact_height"](preview)


def run_with_component(builder, click=None):
    """Build the page state with the click component mocked; returns (at, mock)."""
    with patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=click) as comp:
        at = builder()
        comp.reset_mock()
        at.run()                                  # one clean rerun: counts below are for a single render
    return at, comp


class CompactLayoutTests(unittest.TestCase):
    def test_active_image_has_explicit_bounded_pixel_width_and_inactive_matches(self):
        for builder in (left_started, lambda: right_started(2)):
            with patch("streamlit.image") as image:
                at, comp = run_with_component(builder)
            self.assertEqual(comp.call_count, 1)
            width = comp.call_args.kwargs["width"]
            self.assertIsInstance(width, int)                              # not "stretch"/None: bounded in pixels
            self.assertTrue(350 <= width <= 400)
            self.assertEqual(width, PAGE_COMPACT_WIDTH)
            self.assertEqual(len([c for c in at.caption if "image:" in c.value]), 2)   # both images captioned
            pair = [c for c in image.call_args_list if c.kwargs.get("caption") is None]    # excludes the captioned guide image
            self.assertTrue(pair)                                          # inactive image is a plain, non-clickable st.image
            self.assertEqual({c.kwargs.get("width") for c in pair}, {width})   # ...at the same width

    def test_active_image_is_rendered_at_its_complete_aspect_ratio(self):
        for builder, side in ((left_started, "left"), (lambda: right_started(2), "right")):
            at, comp = run_with_component(builder)
            drawn = comp.call_args.args[0]
            width, height = comp.call_args.kwargs["width"], comp.call_args.kwargs["height"]
            self.assertIsInstance(height, int)                             # explicit: never left to the component's default
            self.assertLessEqual(abs(height - width * drawn.shape[0] / drawn.shape[1]), 0.5)   # full image, not a crop

    def test_compact_height_keeps_aspect_ratio_for_portrait_and_landscape(self):
        import numpy as np
        for shape in ((1333, 1000), (1000, 1333), (900, 900)):
            h = app_compact_height(np.zeros((*shape, 3), np.uint8))
            self.assertAlmostEqual(h, PAGE_COMPACT_WIDTH * shape[0] / shape[1], delta=0.5)

    def test_iframe_is_never_given_height_auto(self):
        css = APP_SOURCE.split("COMPACT_CSS = ")[1].split("</style>")[0]
        for rule in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            if "iframe" in rule[0]:
                self.assertNotIn("height", rule[1])                        # height:auto collapsed it to 150 px and cropped the image
        self.assertNotIn("object-fit", APP_SOURCE.split("def compact_pair")[0][-1200:])
        self.assertNotIn("overflow", css)

    def test_pair_sits_in_two_equal_columns(self):
        at, _ = run_with_component(left_started)
        pair = [c for c in at.columns if any("image:" in x.value for x in c.caption)]
        self.assertEqual(len(pair), 2)
        self.assertEqual(pair[0].proto.weight, pair[1].proto.weight)

    def test_no_full_width_selection_image_remains(self):
        src = (Path(__file__).resolve().parent.parent / "Pages" / "app_module7.py").read_text()
        body = src[src.index("def compact_pair"):src.index("def zoom_token")]
        self.assertNotIn("stretch", body)
        self.assertEqual(body.count("width=COMPACT_WIDTH"), 2)

    def test_only_active_side_takes_clicks_in_every_stage(self):
        for builder, side in ((left_started, "left"), (lambda: right_started(2), "right")):
            _, comp = run_with_component(builder)
            self.assertEqual(comp.call_count, 1)
            self.assertIn(f"m7_coarse_{side}_", comp.call_args.kwargs["key"])

    def test_pair_shown_without_component_when_four_corners_accepted(self):
        _, comp = run_with_component(left_started_done)
        self.assertEqual(comp.call_count, 0)


def left_started_done():
    from test_module7_side_confirmation import left_done
    return left_done()


class CoordinateMappingTests(unittest.TestCase):
    def test_compact_click_maps_to_original_resolution_per_side(self):
        for size in ((3024, 4032), (4284, 5712)):
            click = {"x": 100.0, "y": 150.0, "width": 400.0, "height": 400.0 * size[1] / size[0]}
            x, y = sel.click_to_source(click, (0, 0), size)
            self.assertAlmostEqual(x, (100.5) * size[0] / 400.0 - 0.5)
            self.assertAlmostEqual(y, (150.5) * size[1] / click["height"] - 0.5)
            self.assertIsInstance(x, float)

    def test_mapping_is_independent_of_rendered_size(self):
        size = (3024, 4032)
        a = sel.click_to_source({"x": 50, "y": 75, "width": 200, "height": 266.6667}, (0, 0), size)
        b = sel.click_to_source({"x": 100, "y": 150, "width": 400, "height": 533.3333}, (0, 0), size)
        self.assertLess(abs(a[0] - b[0]), size[0] / 200)       # same location to within one display pixel

    def test_click_moves_to_zoom_with_original_coordinate(self):
        click = {"x": 200.0, "y": 300.0, "width": 400.0, "height": 533.0}
        at, _ = run_with_component(left_started, click=click)
        pending = at.session_state["m7_pending"]
        self.assertEqual(pending[0], "left")
        size = at.session_state["m7_images_size"]["left"]
        self.assertEqual(pending[1], sel.click_to_source(click, (0, 0), size))


class ZoomUnchangedTests(unittest.TestCase):
    def test_accept_and_cancel_keep_coordinates_and_return_to_pair(self):
        click = {"x": 200.0, "y": 300.0, "width": 400.0, "height": 533.0}
        at, _ = run_with_component(left_started, click=click)
        coarse = at.session_state["m7_pending"][1]
        with patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=None):
            button(at, "Cancel").click().run()
            self.assertIsNone(at.session_state["m7_pending"])
            self.assertEqual(at.session_state["m7_points"]["left"], [])
        at, _ = run_with_component(left_started, click=click)
        with patch("streamlit_image_coordinates.streamlit_image_coordinates", return_value=None):
            button(at, "Accept Corner").click().run()
        self.assertEqual(at.session_state["m7_points"]["left"], [coarse])


if __name__ == "__main__":
    unittest.main()
