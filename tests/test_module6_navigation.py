"""Module 6 navigation tests: home -> Part A selection -> visitor placeholder / saved results, and Part B (AppTest, no browser)."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module6.py"
OUTPUT = ROOT / "Modules" / "Module_6" / "Part_A" / "output"

HOME_BUTTONS = ["View Optical Flow and Motion Tracking", "View Structure from Motion Reconstruction"]
SELECT_BUTTONS = ["Track Motion in Your Own Video", "View My Experiment & Results"]


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
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Motion Tracking"])
        self.assertEqual([b.label for b in app.button if b.label != "Back"], SELECT_BUTTONS)

    def test_selection_has_exactly_one_back_at_top(self):
        app = new_app()
        open_select(app)
        labels = [b.label for b in app.button]
        self.assertEqual(labels[0], "Back")
        self.assertEqual(labels.count("Back"), 1)
        self.assertEqual(back_keys(app), ["m6_a_select_back_top"])
        click(app, "m6_a_select_back_top")
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_visitor_page_opens_with_only_top_back(self):
        app = new_app()
        open_select(app)
        click(app, "m6_a_visitor")
        self.assertEqual(app.session_state["module6_section"], "a_visitor")
        self.assertEqual([h.value for h in app.header],
                         ["Track Motion in Your Own Video", "Step 1 — Upload Your Video"])
        self.assertEqual([b.label for b in app.button if b.label == "Back"], ["Back"])
        self.assertEqual(back_keys(app), ["m6_a_visitor_back_top"])
        self.assertNotIn("Video upload and optical-flow tracking will be available here soon.",
                         [m.value for m in app.markdown])
        click(app, "m6_a_visitor_back_top")
        self.assertEqual(app.session_state["module6_section"], "a_select")
        self.assertEqual([b.label for b in app.button if b.label != "Back"], SELECT_BUTTONS)


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
        for key in ("m6_a_results_back_top", "m6_a_results_back_bottom"):
            app = new_app()
            open_select(app)
            click(app, "m6_a_results")
            self.assertEqual(back_keys(app), ["m6_a_results_back_top", "m6_a_results_back_bottom"])
            self.assertEqual([b.label for b in app.button], ["Back", "Back"])
            click(app, key)
            self.assertEqual(app.session_state["module6_section"], "a_select")

    def test_part_b_button_opens_selection(self):
        app = new_app()
        click(app, "m6_open_B")
        self.assertEqual(app.session_state["module6_section"], "b_select")
        self.assertEqual([b.label for b in app.button], ["Back", "Reconstruct an Object from Your Own Images",
                                                         "View My Experiment & Results"])
        click(app, "m6_b_select_back_top")
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)


if __name__ == "__main__":
    unittest.main()
