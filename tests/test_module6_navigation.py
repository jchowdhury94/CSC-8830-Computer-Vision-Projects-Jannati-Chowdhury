"""Module 6 navigation tests: home -> Part A selection -> visitor placeholder / saved results, and Part B (AppTest, no browser)."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module6.py"
OUTPUT = ROOT / "Modules" / "Module_6" / "Part_A" / "output"

HOME_BUTTONS = ["Home", "View Optical Flow and Motion Tracking", "View Structure from Motion Reconstruction"]
PART_B_SELECT_BUTTONS = ["**Reconstruct an Object from Your Own Images**  \nUpload images to reconstruct an object",
                         "**View My Experiment & Results**  \nExplore the completed reconstruction"]
SELECT_BUTTONS = ["**Track Motion in Your Own Video**  \nUse your own recording",
                  "**View My Experiment & Results**  \nExplore the completed experiment"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def back_keys(app):
    return [b.key for b in app.button if b.label == "Back"]


def open_select(app):
    click(app, "m6_open_A")


class Module6NavigationTests(unittest.TestCase):
    def test_home_has_original_buttons(self):
        app = new_app()
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Structure from Motion"])
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_part_a_button_opens_selection(self):
        app = new_app()
        open_select(app)
        self.assertEqual(app.session_state["module6_section"], "a_select")

    def test_selection_heading_subtitle_and_buttons(self):
        app = new_app()
        open_select(app)
        self.assertEqual(len(app.title), 0)
        self.assertIn("MODULE 6", [m.value for m in app.markdown])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Motion Tracking"])
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], SELECT_BUTTONS)
        self.assertEqual([b.key for b in app.button if not (b.key or "").startswith("nav")], ["m6_a_visitor", "m6_a_results"])

    def test_selection_has_exactly_one_back_at_top(self):
        app = new_app()
        open_select(app)
        labels = [b.label for b in app.button if not (b.key or "").startswith("nav")]
        self.assertEqual(labels.count("Back"), 0)  # navigation is the breadcrumb
        self.assertEqual(back_keys(app), [])
        click(app, "navbtn_0")
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_visitor_page_opens_with_only_top_back(self):
        app = new_app()
        open_select(app)
        click(app, "m6_a_visitor")
        self.assertEqual(app.session_state["module6_section"], "a_visitor")
        self.assertEqual([h.value for h in app.header],
                         ["Track Motion in Your Own Video", "Step 1 — Upload Your Video"])
        self.assertEqual([b.label for b in app.button if b.label == "Back"], [])
        self.assertEqual(back_keys(app), [])
        self.assertNotIn("Video upload and optical-flow tracking will be available here soon.",
                         [m.value for m in app.markdown])
        click(app, "navbtn_1")
        self.assertEqual(app.session_state["module6_section"], "a_select")
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], SELECT_BUTTONS)


    def test_results_page_content_preserved(self):
        app = new_app()
        open_select(app)
        click(app, "m6_a_results")
        self.assertEqual(app.session_state["module6_section"], "a_results")
        self.assertEqual([h.value for h in app.header], ["Part A — Optical Flow and Motion Tracking"])
        self.assertEqual([s.value for s in app.subheader], [
            "Video 1 — Horizontal Motion", "Video 2 — Diagonal Motion",
            "What Can Be Inferred from Optical Flow?", "Tracking Validation with Consecutive Frames",
            "Lucas-Kanade Validation", "Bilinear Interpolation"])
        self.assertGreaterEqual(len(app.table), 6)
        self.assertEqual(len(app.latex), 2)
        tables = [t.value.to_string() for t in app.table]
        for expected in ("0.818", "0.939", "+1.3060", "87.087321520467"):
            self.assertTrue(any(expected in t for t in tables), expected)
        for name in ("video_1_optical_flow_safari.mp4", "video_2_optical_flow_safari.mp4",
                     "video_1_validation_comparison.png", "video_2_validation_comparison.png",
                     "video_2_bilinear_interpolation.png"):
            self.assertTrue((OUTPUT / name).is_file(), name)
        self.assertEqual(len(app.warning), 0)

    def test_results_back_buttons_return_to_selection(self):
        for key in ("navbtn_1", "m6_a_results_back_bottom"):
            app = new_app()
            open_select(app)
            click(app, "m6_a_results")
            self.assertEqual(back_keys(app), ["m6_a_results_back_bottom"])
            self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], ["Back"])
            click(app, key)
            self.assertEqual(app.session_state["module6_section"], "a_select")

    def test_part_b_button_opens_selection(self):
        app = new_app()
        click(app, "m6_open_B")
        self.assertEqual(app.session_state["module6_section"], "b_select")
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], PART_B_SELECT_BUTTONS)
        click(app, "navbtn_0")
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_selection_panels_styled_as_option_7_and_home_keeps_original_css(self):
        for open_key, keys in (("m6_open_A", ("m6_a_visitor", "m6_a_results")), ("m6_open_B", ("m6_b_visitor", "m6_b_results"))):
            app = new_app()
            click(app, open_key)
            css = "".join(m.value for m in app.markdown if "<style>" in m.value)
            for key in keys:
                self.assertIn(f".st-key-{key} button:hover", css)
                self.assertIn(f".st-key-{key} button:focus-visible", css)
                self.assertIn(f".st-key-{key} button::before", css)
                self.assertIn(f".st-key-{key} button::after", css)
            self.assertIn("max-width: 500px", css)
            self.assertIn("@media (max-width: 640px)", css)
            self.assertNotIn("m6_home_layout", css)
        home = new_app()
        home_css = "".join(m.value for m in home.markdown if "<style>" in m.value)
        self.assertIn("m6_home_layout", home_css)
        self.assertNotIn("m6_sel_layout", home_css)


if __name__ == "__main__":
    unittest.main()
