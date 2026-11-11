"""Module 6 homepage tests: navigation between the homepage and the saved Part A / Part B views (AppTest, no browser)."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "Pages" / "app_module6.py"
LABELS = ["View Optical Flow and Motion Tracking", "View Structure from Motion Reconstruction"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


class Module6HomepageTests(unittest.TestCase):
    def test_home_content(self):
        app = new_app()
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Structure from Motion"])
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], ["Home", *LABELS])
        self.assertEqual(len(app.caption) + len(app.subheader) + len(app.table), 0)

    def test_part_a_opens_selection_and_back(self):
        app = new_app()
        click(app, "m6_open_A")
        self.assertEqual(app.session_state["module6_section"], "a_select")
        self.assertEqual(len(app.title), 0)
        self.assertIn("MODULE 6", [m.value for m in app.markdown])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Motion Tracking"])
        click(app, "navbtn_0")
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], ["Home", *LABELS])

    def test_part_b_and_back(self):
        app = new_app()
        click(app, "m6_open_B")
        self.assertEqual(app.session_state["module6_section"], "b_select")
        self.assertEqual([h.value for h in app.header], ["Structure from Motion Reconstruction"])
        click(app, "m6_b_results")
        self.assertEqual(app.session_state["module6_section"], "b_results")
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], ["Back"])
        self.assertEqual(len(app.title), 0)
        self.assertEqual([h.value for h in app.header], ["Part B — Structure from Motion"])
        self.assertIn("Recovered Depth Error", [s.value for s in app.subheader])
        self.assertEqual(len(app.metric), 6)
        click(app, "navbtn_1")
        self.assertEqual(app.session_state["module6_section"], "b_select")
        click(app, "navbtn_0")
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"], ["Home", *LABELS])

    def test_a_back_b_back(self):
        app = new_app()
        for key in ("m6_open_A", "navbtn_0", "m6_open_B", "navbtn_0"):
            click(app, key)
        self.assertEqual(app.session_state["module6_section"], "home")


if __name__ == "__main__":
    unittest.main()
