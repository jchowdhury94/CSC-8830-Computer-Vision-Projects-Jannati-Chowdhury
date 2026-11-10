"""Module 6 Part B navigation tests: home -> selection -> visitor tool / saved experiment results (AppTest, no browser)."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module6.py"
PART_B = ROOT / "Modules" / "Module_6" / "Part_B"

HOME_BUTTONS = ["View Optical Flow and Motion Tracking", "View Structure from Motion Reconstruction"]
SELECT_BUTTONS = ["Reconstruct an Object from Your Own Images", "View My Experiment & Results"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def open_select():
    app = new_app()
    click(app, "m6_open_B")
    return app


class PartBNavigationTests(unittest.TestCase):
    def test_home_unchanged(self):
        app = new_app()
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Structure from Motion"])
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_part_a_navigation_unchanged(self):
        app = new_app()
        click(app, "m6_open_A")
        self.assertEqual(app.session_state["module6_section"], "a_select")

    def test_selection_page(self):
        app = open_select()
        self.assertEqual(app.session_state["module6_section"], "b_select")
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Structure from Motion Reconstruction"])
        labels = [b.label for b in app.button]
        self.assertEqual(labels[0], "Back")
        self.assertEqual(labels[1:], SELECT_BUTTONS)
        self.assertEqual([b.key for b in app.button if b.label == "Back"], ["m6_b_select_back_top"])

    def test_selection_back_returns_home(self):
        app = open_select()
        click(app, "m6_b_select_back_top")
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_visitor_page_opens_with_top_back_only(self):
        app = open_select()
        click(app, "m6_b_visitor")
        self.assertEqual(app.session_state["module6_section"], "b_visitor")
        self.assertEqual([b.label for b in app.button], ["Back"])
        self.assertEqual([h.value for h in app.header],
                         ["Reconstruct an Object from Your Own Images", "Step 1 — Upload Your Images"])
        click(app, "m6_b_visitor_back_top")
        self.assertEqual(app.session_state["module6_section"], "b_select")

    def test_results_preserved_with_top_and_bottom_back(self):
        for key in ("m6_b_results_back_top", "m6_b_results_back_bottom"):
            app = open_select()
            click(app, "m6_b_results")
            self.assertEqual(app.session_state["module6_section"], "b_results")
            self.assertEqual([b.key for b in app.button], ["m6_b_results_back_top", "m6_b_results_back_bottom"])
            self.assertEqual([h.value for h in app.header], ["Part B — Structure from Motion"])
            subheaders = [s.value for s in app.subheader]
            for name in ("Four Camera Viewpoints", "Measurement Matrix and Rank Analysis", "Scaled 3D Reconstruction",
                         "Physical Validation", "Recovered Depth Error", "Interpretation and Limitations"):
                self.assertIn(name, subheaders)
            self.assertEqual(len(app.metric), 6)
            click(app, key)
            self.assertEqual(app.session_state["module6_section"], "b_select")

    def test_experiment_assets_present(self):
        for name in ("part_b_four_views.png", "part_b_reconstruction_scaled_cm.png", "part_b_physical_validation.csv"):
            self.assertTrue((PART_B / "output" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
