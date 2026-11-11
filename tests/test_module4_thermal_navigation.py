"""Module 4 thermal navigation tests: home -> thermal selection -> visitor / results, with one top Back button on the selection page and two (top and bottom) on the visitor and results pages.

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
    click(app, "m4_open_thermal")


def back_keys(app):
    return [b.key for b in app.button if b.label == "Back"]


class Module4ThermalNavigationTests(unittest.TestCase):
    def test_home_unchanged_and_opens_thermal_selection(self):
        app = new_app()
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)
        open_select(app)
        self.assertEqual(app.session_state["module4_section"], "thermal_select")

    def test_selection_page_layout(self):
        app = new_app()
        open_select(app)
        self.assertEqual(len(app.title), 0)
        self.assertIn("MODULE 4", [m.value for m in app.markdown])
        self.assertEqual([h.value for h in app.header], ["Thermal Image Boundary Detection"])
        labels = [b.label for b in app.button if not (b.key or "").startswith("nav")]
        self.assertEqual(labels, ["**Detect Human Boundaries in Your Own Thermal Image**  \nUpload and analyze your own thermal image",
                                  "**View My Experiment & Results**  \nExplore my thermal boundary detection results"])
        self.assertEqual(back_keys(app), [])
        # The breadcrumb row sits above the heading; selection pages have no Back button.
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertEqual([b.key for b in app.main[0].button], ["nav_arrow_button", "nav_home_button", "navbtn_0"])  # Back arrow, Home > Module 4 (ancestor)

    def test_selection_matches_rgb_selection_structure(self):
        rgb = new_app()
        click(rgb, "m4_open_rgb")
        thermal = new_app()
        open_select(thermal)
        self.assertEqual(len(rgb.button), len(thermal.button))
        self.assertEqual([b.label for b in rgb.button][:3], [b.label for b in thermal.button][:3])
        self.assertTrue(all("**  \n" in b.label for page in (rgb, thermal) for b in list(page.button)[3:]))
        self.assertEqual([t.value for t in rgb.title], [t.value for t in thermal.title])
        self.assertEqual(len(rgb.markdown), len(thermal.markdown))

    def test_selection_back_returns_home(self):
        app = new_app()
        open_select(app)
        click(app, "navbtn_0")
        self.assertEqual(app.session_state["module4_section"], "home")
        self.assertEqual([b.label for b in app.button], HOME_BUTTONS)

    def test_visitor_page_keeps_existing_thermal_experiment_and_two_backs(self):
        app = new_app()
        open_select(app)
        click(app, "m4_thermal_visitor")
        labels = [b.label for b in app.button if not (b.key or "").startswith("nav")]
        self.assertEqual(labels[-1], "Back")
        self.assertEqual(len(back_keys(app)), 1)
        self.assertEqual(app.subheader[0].value, "Question 2 – Thermal Image Human Boundary Detection")
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertNotIn("Back to Module 4", labels)

    def test_visitor_back_returns_to_selection(self):
        for key in ("navbtn_1", "m4_thermal_visitor_back_bottom"):
            app = new_app()
            open_select(app)
            click(app, "m4_thermal_visitor")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "thermal_select")

    def test_results_page_steps_values_and_two_backs(self):
        app = new_app()
        open_select(app)
        click(app, "m4_thermal_results")
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertEqual([h.value for h in app.header], ["My Thermal Experiment & Results"])
        self.assertEqual([s.value for s in app.subheader], [
            "Step 1 — Original Thermal Image & Settings", "Step 2 — Classical Thermal Segmentation",
            "Step 3 — SAM2 Thermal Segmentation", "Step 4 — Mask Comparison",
            "Step 5 — Boundary Comparison", "Step 6 — Quantitative Comparison"])
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(metrics, {
            "Image size (width × height)": "1080 × 1440 pixels",
            "Morphological closing": "3 × 3 elliptical",
            "Connected components": "8-connectivity",
            "Foreground components found": "20",
            "Selected foreground size": "392,206 pixels",
            "Image occupancy": "25.22%",
            "SAM2 box (x, y, width, height)": "(263, 57, 574, 1380) pixels",
            "Predicted mask quality": "0.977383",
            "Device": "MPS",
            "IoU with SAM2": "0.9813",
        })
        self.assertEqual(len(app.get("image")), 3)
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], ["Back"])

    def test_results_page_has_no_rgb_values(self):
        app = new_app()
        open_select(app)
        click(app, "m4_thermal_results")
        shown = " ".join([*(m.value for m in app.metric), *(m.value for m in app.markdown)])
        for rgb_only in ("1766", "6,574,847", "0.989319", "GrabCut", "Canny", "0.80"):
            self.assertNotIn(rgb_only, shown)

    def test_results_back_returns_to_selection(self):
        for key in ("navbtn_1", "m4_thermal_results_back_bottom"):
            app = new_app()
            open_select(app)
            click(app, "m4_thermal_results")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "thermal_select")

    def test_navigation_state_survives_reruns(self):
        app = new_app()
        open_select(app)
        click(app, "m4_thermal_results")
        app.run()
        self.assertEqual(app.session_state["module4_section"], "thermal_results")
        click(app, "navbtn_1")
        click(app, "m4_thermal_visitor")
        app.run()
        self.assertEqual(app.session_state["module4_section"], "thermal")

    def test_rgb_section_regression(self):
        app = new_app()
        click(app, "m4_open_rgb")
        self.assertEqual(app.session_state["module4_section"], "rgb_select")
        self.assertEqual(back_keys(app), [])
        click(app, "m4_rgb_visitor")
        self.assertEqual(app.subheader[0].value, "Question 1 – Human Boundary Detection in an RGB Image")
        self.assertEqual(len(back_keys(app)), 1)
        click(app, "navbtn_1")
        click(app, "m4_rgb_results")
        self.assertEqual([h.value for h in app.header], ["My RGB Experiment & Results"])
        self.assertEqual(len(app.get("image")), 9)

    def test_thermal_display_assets_exist(self):
        for name in ("thermal_step1_original.jpg", "thermal_step4_mask_comparison.png",
                     "thermal_step5_boundary_comparison.png"):
            self.assertTrue((DISPLAY / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
