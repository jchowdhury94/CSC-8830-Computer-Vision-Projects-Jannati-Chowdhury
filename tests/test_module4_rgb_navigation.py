"""Module 4 RGB navigation tests: homepage -> RGB visitor directly, and homepage -> My Experiments & Results -> RGB results, with top navigation only.

Streamlit pages are exercised with AppTest (no browser). Nothing here writes into the repository.
"""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module4.py"
DISPLAY = ROOT / "Modules" / "Module_4" / "display"

HOME_BUTTONS = ["Home", "View My Experiments & Results", "Detect Human Boundaries in Your Own RGB Image",
                "Detect Human Boundaries in Your Own Thermal Image"]
RESULT_LABELS = ["**Human Boundary Detection in RGB Images**  \n(Question 1)",
                 "**Human Boundary Detection in Thermal Images**  \n(Question 2)"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def page_buttons(app):
    return [b.label for b in app.button if not (b.key or "").startswith("nav")]


def back_keys(app):
    return [b.key for b in app.button if b.label == "Back"]


class Module4RgbNavigationTests(unittest.TestCase):
    def test_home_has_three_stacked_buttons_in_order(self):
        app = new_app()
        self.assertEqual([t.value for t in app.title], ["Module 4"])
        self.assertEqual([h.value for h in app.header], ["Human Boundary Detection"])
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], HOME_BUTTONS)
        self.assertEqual([b.key for b in app.button if (b.key or "").startswith("m4_")],
                         ["m4_open_results", "m4_open_rgb", "m4_open_thermal"])

    def test_results_selection_page(self):
        app = new_app()
        click(app, "m4_open_results")
        self.assertEqual(app.session_state["module4_section"], "my_results")
        self.assertEqual([t.value for t in app.title], ["My Experiments & Results"])
        self.assertEqual([h.value for h in app.header], ["Human Boundary Detection"])
        self.assertEqual(page_buttons(app), RESULT_LABELS)
        self.assertEqual(back_keys(app), [])

    def test_selection_back_arrow_and_breadcrumb_return_home(self):
        for key in ("nav_arrow_button", "navbtn_0"):
            app = new_app()
            click(app, "m4_open_results")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "home")

    def test_rgb_visitor_opens_directly_with_no_bottom_back(self):
        app = new_app()
        click(app, "m4_open_rgb")
        self.assertEqual(app.session_state["module4_section"], "rgb")
        self.assertEqual([t.value for t in app.title], ["Human Boundary Detection in RGB Images"])
        self.assertIn("Reset Experiment", page_buttons(app))
        self.assertEqual(back_keys(app), [])
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")

    def test_rgb_visitor_back_returns_home(self):
        for key in ("nav_arrow_button", "navbtn_0"):
            app = new_app()
            click(app, "m4_open_rgb")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "home")

    def test_results_page_steps_values(self):
        app = new_app()
        click(app, "m4_open_results")
        click(app, "m4_res_rgb")
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertEqual([t.value for t in app.title], ["My Experiment & Results"])
        self.assertEqual([h.value for h in app.header], [
            "Human Boundary Detection in RGB Images",
            "Step 1 — Original RGB Image & Experimental Parameters", "Step 2 — Gaussian Smoothing",
            "Step 3 — Canny Edge Detection", "Step 4 — GrabCut Segmentation",
            "Step 5 — Classical Human Boundary Detection", "Step 6 — SAM2 Segmentation & Boundary Detection",
            "Step 7 — Classical vs. SAM2 Comparison", "Conclusion"])
        self.assertEqual([s.value for s in app.subheader], ["Mask Comparison", "Boundary Comparison"])
        tables = [{k: v["Value"] for k, v in t.value.iterrows()} for t in app.table]
        self.assertEqual(tables, [
            {"Gaussian kernel": "3 × 3", "Gaussian sigma": "0.80", "Canny thresholds": "50 / 150", "GrabCut iterations": "5",
             "Morphological cleanup": "Disabled", "GrabCut rectangle": "1766 × 5620 pixels"},
            {"Classical contour area": "6,574,847.0 pixels²"},
            {"Model": "SAM 2.1 Hiera Tiny", "Prompt": "Bounding box", "Device": "MPS",
             "Predicted mask-quality score": "0.989319"}])
        self.assertEqual(len(app.metric), 0)
        text = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
        self.assertIn("Detect the human boundary in an RGB image", text)
        self.assertIn("cast shadow", text)
        self.assertIn("not a measured segmentation accuracy", text)
        self.assertNotIn("report", text.lower())
        self.assertEqual(len(app.get("image")), 12)  # Step 7 shows each comparison as two separate pictures
        self.assertEqual(page_buttons(app), [])  # no bottom Back

    def test_results_back_returns_to_selection(self):
        for key in ("nav_arrow_button", "navbtn_1"):
            app = new_app()
            click(app, "m4_open_results")
            click(app, "m4_res_rgb")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "my_results")
        click(app, "navbtn_0")
        self.assertEqual(app.session_state["module4_section"], "home")

    def test_navigation_state_survives_reruns(self):
        app = new_app()
        click(app, "m4_open_results")
        click(app, "m4_res_rgb")
        app.run()
        self.assertEqual(app.session_state["module4_section"], "rgb_results")

    def test_stale_selection_state_falls_back_to_home(self):
        for stale in ("rgb_select", "thermal_select"):
            app = AppTest.from_file(str(PAGE), default_timeout=60)
            app.session_state["module4_section"] = stale
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["module4_section"], "home")
            self.assertEqual([t.value for t in app.title], ["Module 4"])

    def test_display_assets_exist(self):
        for name in ("step1_original.jpg", "step2_gaussian_reproduced.jpg", "step3_canny.jpg",
                     "step4_grabcut_mask.png", "step5_classical_boundary.png", "step6_sam2_mask.png",
                     "step6_sam2_boundary.png", "step7_mask_comparison.png", "step7_boundary_comparison.png"):
            self.assertTrue((DISPLAY / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
