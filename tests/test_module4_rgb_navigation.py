"""Module 4 RGB navigation tests: home -> RGB selection -> visitor / results, with a top Back button on the selection page, and two on the visitor and results pages.

Streamlit pages are exercised with AppTest (no browser). Nothing here writes into the repository.
"""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module4.py"
DISPLAY = ROOT / "Modules" / "Module_4" / "display"

HOME_BUTTONS = ["Home", "Detect Human Boundaries in RGB Images", "Detect Human Boundaries in Thermal Images"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def open_select(app):
    click(app, "m4_open_rgb")


def back_keys(app):
    return [b.key for b in app.button if b.label == "Back"]


class Module4RgbNavigationTests(unittest.TestCase):
    def test_home_unchanged_with_two_buttons(self):
        app = new_app()
        self.assertEqual([t.value for t in app.title], ["Module 4"])
        self.assertEqual([h.value for h in app.header], ["Human Boundary Detection"])
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], HOME_BUTTONS)

    def test_rgb_button_opens_selection_page(self):
        app = new_app()
        open_select(app)
        self.assertEqual(len(app.title), 0)
        self.assertIn("MODULE 4", [m.value for m in app.markdown])
        self.assertEqual([h.value for h in app.header], ["RGB Image Boundary Detection"])
        labels = [b.label for b in app.button if not (b.key or "").startswith("nav")]
        self.assertEqual([l for l in labels if l != "Back"],
                         ["**Detect Human Boundaries in Your Own RGB Image**  \nUpload and analyze your own RGB image",
                          "**View My Experiment & Results**  \nExplore my RGB boundary detection results"])
        self.assertEqual(labels.count("Back"), 0)
        self.assertEqual(back_keys(app), [])

    def test_selection_back_returns_home(self):
        app = new_app()
        open_select(app)
        click(app, "navbtn_0")
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], HOME_BUTTONS)

    def test_visitor_page_keeps_existing_experiment_and_two_backs(self):
        app = new_app()
        open_select(app)
        click(app, "m4_rgb_visitor")
        labels = [b.label for b in app.button if not (b.key or "").startswith("nav")]
        self.assertEqual(labels[-1], "Back")
        self.assertEqual(len(back_keys(app)), 1)
        self.assertEqual(app.subheader[0].value, "Question 1 – Human Boundary Detection in an RGB Image")
        self.assertIn("Reset RGB Experiment", labels)
        # Back is above the heading: the first element rendered is the button, not the title.
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")

    def test_visitor_back_returns_to_selection_not_home(self):
        for key in ("navbtn_1", "m4_rgb_visitor_back_bottom"):
            app = new_app()
            open_select(app)
            click(app, "m4_rgb_visitor")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "rgb_select")
            self.assertIn("m4_rgb_results", [b.key for b in app.button])

    def test_results_page_steps_values_and_two_backs(self):
        app = new_app()
        open_select(app)
        click(app, "m4_rgb_results")
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertEqual([h.value for h in app.header], ["My RGB Experiment & Results"])
        self.assertEqual([s.value for s in app.subheader], [
            "Step 1 — Original RGB Image & Parameters", "Step 2 — Gaussian Smoothing",
            "Step 3 — Canny Edge Detection", "Step 4 — GrabCut Segmentation",
            "Step 5 — Classical Human Boundary", "Step 6 — SAM2 Segmentation",
            "Step 7 — Classical vs. SAM2 Comparison"])
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(metrics["Gaussian kernel size"], "3")
        self.assertEqual(metrics["Sigma"], "0.80")
        self.assertEqual(metrics["Canny thresholds (lower / upper)"], "50 / 150")
        self.assertEqual(metrics["GrabCut iterations"], "5")
        self.assertEqual(metrics["Morphological cleanup"], "Disabled")
        self.assertEqual(metrics["GrabCut rectangle (width × height)"], "1766 × 5620 pixels")
        self.assertEqual(metrics["Classical contour area"], "6,574,847.0 pixels²")
        self.assertEqual(metrics["Predicted mask-quality score"], "0.989319")
        self.assertEqual(metrics["Device"], "MPS")
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("cast shadow", text)
        self.assertIn("SAM2 follows", text)
        self.assertEqual(len(app.get("image")), 9)
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], ["Back"])

    def test_results_back_returns_to_selection(self):
        for key in ("navbtn_1", "m4_rgb_results_back_bottom"):
            app = new_app()
            open_select(app)
            click(app, "m4_rgb_results")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "rgb_select")

    def test_navigation_state_survives_reruns(self):
        app = new_app()
        open_select(app)
        click(app, "m4_rgb_results")
        app.run()
        self.assertEqual(app.session_state["module4_section"], "rgb_results")
        click(app, "navbtn_1")
        click(app, "m4_rgb_visitor")
        app.run()
        self.assertEqual(app.session_state["module4_section"], "rgb")

    def test_thermal_button_opens_thermal_selection(self):
        app = new_app()
        click(app, "m4_open_thermal")
        self.assertEqual(app.session_state["module4_section"], "thermal_select")
        self.assertEqual([h.value for h in app.header], ["Thermal Image Boundary Detection"])

    def test_display_assets_exist(self):
        for name in ("step1_original.jpg", "step2_gaussian_reproduced.jpg", "step3_canny.jpg",
                     "step4_grabcut_mask.png", "step5_classical_boundary.png", "step6_sam2_mask.png",
                     "step6_sam2_boundary.png", "step7_mask_comparison.png", "step7_boundary_comparison.png"):
            self.assertTrue((DISPLAY / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
