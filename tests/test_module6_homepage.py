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
        self.assertEqual(app.session_state["module6_part"], "home")
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Structure from Motion"])
        self.assertEqual([b.label for b in app.button], LABELS)
        self.assertEqual(len(app.caption) + len(app.subheader) + len(app.table), 0)

    def test_part_a_and_back(self):
        app = new_app()
        click(app, "m6_open_A")
        self.assertEqual(app.session_state["module6_part"], "A")
        self.assertEqual([b.label for b in app.button], ["Back to Module 6"])
        self.assertEqual(len(app.title), 0)
        self.assertEqual([h.value for h in app.header], ["Part A — Optical Flow and Motion Tracking"])
        self.assertIn("Bilinear Interpolation", [s.value for s in app.subheader])
        self.assertGreaterEqual(len(app.table), 6)
        click(app, "m6_back")
        self.assertEqual([t.value for t in app.title], ["Module 6"])

    def test_part_b_and_back(self):
        app = new_app()
        click(app, "m6_open_B")
        self.assertEqual(app.session_state["module6_part"], "B")
        self.assertEqual([b.label for b in app.button], ["Back to Module 6"])
        self.assertEqual(len(app.title), 0)
        self.assertEqual([h.value for h in app.header], ["Part B — Structure from Motion"])
        self.assertIn("Recovered Depth Error", [s.value for s in app.subheader])
        self.assertEqual(len(app.metric), 6)
        click(app, "m6_back")
        self.assertEqual([b.label for b in app.button], LABELS)

    def test_a_back_b_back(self):
        app = new_app()
        for key in ("m6_open_A", "m6_back", "m6_open_B", "m6_back"):
            click(app, key)
        self.assertEqual(app.session_state["module6_part"], "home")


if __name__ == "__main__":
    unittest.main()
