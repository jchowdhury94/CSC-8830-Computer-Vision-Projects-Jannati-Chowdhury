"""Module 6 homepage tests: homepage, results selection page and navigation (AppTest, no browser)."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = Path(__file__).resolve().parents[1] / "Pages" / "app_module6.py"
LABELS = ["View My Experiments & Results", "Track Motion in Your Own Video", "Reconstruct an Object from Your Own Images"]
HOME_KEYS = ["m6_open_results", "m6_open_a_visitor", "m6_open_b_visitor"]
RESULT_LABELS = ["**Optical Flow and Motion Tracking**  \n(Part A)", "**Structure from Motion Reconstruction**  \n(Part B)"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def content_buttons(app):
    return [b for b in app.button if not (b.key or "").startswith("nav")]


class Module6HomepageTests(unittest.TestCase):
    def test_home_content_and_button_styles(self):
        app = new_app()
        self.assertEqual(app.session_state["module6_section"], "home")
        self.assertEqual([t.value for t in app.title], ["Module 6"])
        self.assertEqual([h.value for h in app.header], ["Optical Flow and Structure from Motion"])
        self.assertEqual([b.label for b in content_buttons(app)], LABELS)
        self.assertEqual([b.key for b in content_buttons(app)], HOME_KEYS)
        self.assertEqual(len(app.caption) + len(app.subheader) + len(app.table), 0)
        css = "".join(m.value for m in app.markdown if "<style>" in m.value)
        self.assertIn("flex-direction: column", css)
        self.assertIn("height: 110px", css)
        self.assertIn(".st-key-m6_open_results button { background-color: #1D5FD1", css)
        self.assertNotIn(".st-key-m6_open_a_visitor button { background-color: #1D5FD1", css)

    def test_results_selection_page(self):
        app = new_app()
        click(app, "m6_open_results")
        self.assertEqual(app.session_state["module6_section"], "my_results")
        self.assertIn("MODULE 6", [m.value for m in app.markdown])
        self.assertEqual([t.value for t in app.title], ["My Experiments & Results"])
        self.assertEqual(len(app.header), 0)
        self.assertEqual([b.label for b in content_buttons(app)], RESULT_LABELS)

    def test_visitor_buttons_open_the_workflows_directly_and_return_home(self):
        for key, section, title in (("m6_open_a_visitor", "a_visitor", LABELS[1]),
                                    ("m6_open_b_visitor", "b_visitor", LABELS[2])):
            for back in ("nav_arrow_button", "navbtn_0"):
                app = new_app()
                click(app, key)
                self.assertEqual(app.session_state["module6_section"], section)
                self.assertEqual(app.title[0].value, title)  # both visitor pages share the Part A title style
                self.assertEqual([b.label for b in app.button if b.label == "Back"], [])
                click(app, back)
                self.assertEqual(app.session_state["module6_section"], "home")

    def test_results_pages_open_and_return_through_the_hierarchy(self):
        for key, section, headers in (("m6_res_a", "a_results", ["My Experiment & Results"]),
                                      ("m6_res_b", "b_results", ["My Experiment & Results"])):
            for back in ("nav_arrow_button", "navbtn_1"):
                app = new_app()
                click(app, "m6_open_results")
                click(app, key)
                self.assertEqual(app.session_state["module6_section"], section)
                self.assertEqual([h.value for h in app.header][:1], headers)
                self.assertEqual(content_buttons(app), [])
                click(app, back)
                self.assertEqual(app.session_state["module6_section"], "my_results")
                click(app, "nav_arrow_button")
                self.assertEqual(app.session_state["module6_section"], "home")

    def test_removed_selection_sections_fall_back_to_home(self):
        for stale in ("a_select", "b_select"):
            app = AppTest.from_file(str(PAGE), default_timeout=60)
            app.session_state["module6_section"] = stale
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["module6_section"], "home")


if __name__ == "__main__":
    unittest.main()
