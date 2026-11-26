"""Module 4 thermal navigation tests: homepage -> thermal visitor directly, and homepage -> My Experiments & Results -> thermal results, with top navigation only.

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


class Module4ThermalNavigationTests(unittest.TestCase):
    def test_thermal_visitor_opens_directly_with_no_bottom_back(self):
        app = new_app()
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], HOME_BUTTONS)
        click(app, "m4_open_thermal")
        self.assertEqual(app.session_state["module4_section"], "thermal")
        self.assertEqual([t.value for t in app.title], ["Human Boundary Detection in Thermal Images"])
        self.assertEqual(back_keys(app), [])
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertEqual([b.key for b in app.main[0].button], ["nav_arrow_button", "nav_home_button", "navbtn_0"])

    def test_thermal_visitor_back_returns_home(self):
        for key in ("nav_arrow_button", "navbtn_0"):
            app = new_app()
            click(app, "m4_open_thermal")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "home")

    def test_results_selection_opens_thermal_results(self):
        app = new_app()
        click(app, "m4_open_results")
        self.assertEqual(page_buttons(app), RESULT_LABELS)
        click(app, "m4_res_thermal")
        self.assertEqual(app.session_state["module4_section"], "thermal_results")
        self.assertEqual(getattr(app.main[0], "key", None), "nav_row")
        self.assertEqual([b.key for b in app.main[0].button],
                         ["nav_arrow_button", "nav_home_button", "navbtn_0", "navbtn_1"])

    def test_results_page_steps_values(self):
        app = new_app()
        click(app, "m4_open_results")
        click(app, "m4_res_thermal")
        self.assertEqual([t.value for t in app.title], ["My Experiment & Results"])
        self.assertEqual([h.value for h in app.header], [
            "Human Boundary Detection in Thermal Images",
            "Step 1 — Original Thermal Image & Experimental Parameters", "Step 2 — Classical Thermal Segmentation",
            "Step 3 — Classical Human Boundary Detection", "Step 4 — SAM2 Segmentation & Boundary Detection",
            "Step 5 — Classical vs. SAM2 Comparison", "Step 6 — Quantitative Comparison", "Conclusion"])
        self.assertEqual([s.value for s in app.subheader], ["Mask Comparison", "Boundary Comparison"])
        tables = [{k: v.iloc[0] for k, v in t.value.iterrows()} for t in app.table]
        self.assertEqual(tables, [
            {"Image resolution": "1080 × 1440 pixels", "Color space": "HSV", "Segmentation method": "Color thresholding",
             "Morphological closing": "3 × 3 elliptical kernel", "Connected components": "8-connectivity",
             "Region selection": "Largest foreground component"},
            {"Foreground components found": "20", "Selected foreground size": "392,206 pixels", "Image occupancy": "25.22%"},
            {"Prompt": "Bounding box", "Bounding box (x, y)": "(263, 57) pixels",
             "Bounding box (width × height)": "574 × 1380 pixels", "Experimental device": "MPS",
             "Predicted mask-quality score": "0.977383"},
            {"Intersection over Union (IoU)": "0.9813"}])
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.latex), 1)
        text = " ".join(m.value for m in app.markdown) + " ".join(c.value for c in app.caption)
        self.assertIn("Detect the human boundary in a thermal image", text)
        self.assertIn("estimated segmentation quality", text)
        self.assertIn("very high agreement between the two methods", text)
        self.assertNotIn("accuracy", text)
        self.assertNotIn("98.13", text)
        self.assertNotIn("report", text.lower())
        self.assertEqual(len(app.get("image")), 9)  # original, classical mask + boundary, SAM2 mask + boundary, 2 + 2 comparisons
        self.assertEqual(page_buttons(app), [])  # no bottom Back

    def test_results_page_has_no_rgb_values(self):
        app = new_app()
        click(app, "m4_open_results")
        click(app, "m4_res_thermal")
        shown = " ".join([*(m.value for m in app.metric), *(m.value for m in app.markdown)])
        for rgb_only in ("1766", "6,574,847", "0.989319", "GrabCut", "Canny", "0.80"):
            self.assertNotIn(rgb_only, shown)

    def test_results_back_returns_to_selection_then_home(self):
        for key in ("nav_arrow_button", "navbtn_1"):
            app = new_app()
            click(app, "m4_open_results")
            click(app, "m4_res_thermal")
            click(app, key)
            self.assertEqual(app.session_state["module4_section"], "my_results")
        click(app, "navbtn_0")
        self.assertEqual(app.session_state["module4_section"], "home")

    def test_navigation_state_survives_reruns(self):
        app = new_app()
        click(app, "m4_open_thermal")
        app.run()
        self.assertEqual(app.session_state["module4_section"], "thermal")

    def test_rgb_section_regression(self):
        app = new_app()
        click(app, "m4_open_rgb")
        self.assertEqual(app.session_state["module4_section"], "rgb")
        self.assertEqual([t.value for t in app.title], ["Human Boundary Detection in RGB Images"])
        self.assertEqual(back_keys(app), [])

    def test_thermal_display_assets_exist(self):
        for name in ("thermal_step1_original.jpg", "thermal_step4_mask_comparison.png",
                     "thermal_step5_boundary_comparison.png"):
            self.assertTrue((DISPLAY / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
