"""Persistent Home button: exactly one on every reachable view of Modules 2, 3, 4, 6 and 7, always routed to the dashboard."""
import re
import unittest
from pathlib import Path
from unittest import mock

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
HELPER = (ROOT / "home_navigation.py").read_text(encoding="utf-8")
HOME_SOURCE = (ROOT / "home.py").read_text(encoding="utf-8")

# module -> (session-state key, every reachable view)
VIEWS = {
    2: ("m2_page", ("home", "visitor", "experiment")),
    3: ("m3_page", ("home", "visitor", "experiment")),
    4: ("module4_section", ("home", "rgb_select", "rgb_results", "rgb", "thermal_select", "thermal_results", "thermal")),
    6: ("module6_section", ("home", "a_select", "a_visitor", "a_results", "b_select", "b_visitor", "b_results")),
    7: ("module7_part", ("home", "q1_home", "1", "experiment", "3")),  # "3" is the room-measurement placeholder
}


def source(n):
    return (ROOT / "Pages" / f"app_module{n}.py").read_text(encoding="utf-8")


def run(n, view):
    key, _ = VIEWS[n]
    at = AppTest.from_file(str(ROOT / "Pages" / f"app_module{n}.py"), default_timeout=120)
    at.session_state[key] = view
    return at.run()


def home_buttons(at):
    return [b for b in at.button if b.key == "nav_home_button"]


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class HomeButtonOnEveryViewTests(unittest.TestCase):
    def test_exactly_one_home_button_on_every_reachable_view(self):
        for n, (_, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    self.assertFalse(at.exception)
                    self.assertEqual(len(home_buttons(at)), 1)
                    self.assertEqual(home_buttons(at)[0].label, "Home")
                    self.assertEqual([b.label for b in at.button].count("Home"), 1)
                    self.assertEqual([e.key for e in at.main if getattr(e, "key", None) == "nav_home"], ["nav_home"])
                    self.assertEqual(sum("Back to Home" in b.label for b in at.button), 0)

    def test_home_button_has_home_icon_and_is_a_separate_control_from_back(self):
        for n, (_, views) in VIEWS.items():
            for view in views[1:]:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    home = home_buttons(at)[0]
                    self.assertEqual(home.proto.icon, ":material/home:")
                    backs = [b for b in at.button if "Back" in b.label]
                    self.assertTrue(backs or view in ("home",), "nested view keeps its own Back")
                    self.assertTrue(all(b.key != home.key for b in backs))

    def test_home_always_switches_to_the_registered_dashboard_page(self):
        for n, (_, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = AppTest.from_file(str(ROOT / "Pages" / f"app_module{n}.py"), default_timeout=120)
                    at.session_state[VIEWS[n][0]] = view
                    at.session_state["_home_page"] = "HOME_PAGE_SENTINEL"
                    at.run()
                    with mock.patch("home_navigation.st.switch_page") as switch:
                        home_buttons(at)[0].click()
                        at.run()
                    switch.assert_called_once_with("HOME_PAGE_SENTINEL")

    def test_existing_back_navigation_is_unchanged(self):
        at = run(4, "rgb_select")
        next(b for b in at.button if b.key == "m4_rgb_select_back_top").click().run()
        self.assertEqual(at.session_state["module4_section"], "home")
        at = run(6, "b_results")
        next(b for b in at.button if b.key == "m6_b_results_back_top").click().run()
        self.assertEqual(at.session_state["module6_section"], "b_select")
        at = run(7, "experiment")
        next(b for b in at.button if b.key == "m7_back").click().run()
        self.assertEqual(at.session_state["module7_part"], "q1_home")
        at = run(3, "experiment")
        next(b for b in at.button if b.key == "m3_back_experiment_top").click().run()
        self.assertEqual(at.session_state["m3_page"], "home")


class HomeNavigationSourceTests(unittest.TestCase):
    def test_every_module_renders_the_shared_button_once_at_its_dispatcher(self):
        for n in VIEWS:
            with self.subTest(module=n):
                text = source(n)
                self.assertEqual(text.count("nav_row("), 1)
                self.assertIn("from home_navigation import back_button, nav_row", text)
                self.assertNotIn("back_to_home", text)
                self.assertNotIn("nav_home", text)  # no per-module copies of the key or CSS

    def test_helper_is_shared_and_uses_existing_page_navigation(self):
        self.assertIn("st.switch_page(st.session_state[HOME_PAGE_KEY])", HELPER)
        self.assertIn('key="nav_home_button"', HELPER)
        self.assertIn('icon=":material/home:"', HELPER)
        self.assertIn("st.Page(show_home", HOME_SOURCE)
        self.assertEqual(re.findall(r'MODULE_PAGES\["(\d)"\],', HOME_SOURCE), ["2", "3", "4", "6", "7"])

    def test_css_contract_red_solid_white_text_top_right_no_overlay(self):
        for fragment in ("background: {ACCENT}", "color: #FFFFFF", "margin-left: auto", 'BUTTON_WIDTH = "110px"',
                         'BUTTON_HEIGHT = "40px"', "button:focus-visible", "outline: 3px solid", "horizontal=True",
                         'vertical_alignment="center"'):
            self.assertIn(fragment, HELPER)
        self.assertNotIn("position: fixed", HELPER)
        self.assertNotIn("position: absolute", HELPER)
        self.assertNotIn("sticky", HELPER)
        self.assertNotIn("stSidebar", HELPER)
        self.assertIn('ACCENT = "#C9363E"', HELPER)

    def test_every_page_navigation_back_is_labelled_exactly_back(self):
        # Workflow steps (scroll-to-step buttons in Module 6) are deliberately not page navigation and keep their labels.
        workflow = {"Back to Scale", "Back to Selection", "Back to Settings"}
        for n, (_, views) in VIEWS.items():
            text = source(n)
            self.assertNotIn("← Back", text)
            for label in re.findall(r'"(Back[^"]*)"', text):
                self.assertTrue(label == "Back" or label in workflow, (n, label))
            for view in views:
                with self.subTest(module=n, view=view):
                    labels = [b.label for b in run(n, view).button]
                    self.assertFalse([l for l in labels if l.startswith("Back") and l != "Back"])

    def test_back_and_home_share_one_row_back_first(self):
        for n, (_, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    row = next(e for e in at.main if getattr(e, "key", None) == "nav_row")
                    content = [e for e in at.main if e.type in ("title", "header", "subheader")]
                    self.assertLess(list(at.main).index(row), list(at.main).index(content[0]) if content else 99)
                    keys = [b.key for b in row.button]
                    self.assertEqual(keys[-1], "nav_home_button")
                    self.assertEqual(len(keys), 1 if view == "home" else 2)

    def test_sidebar_stays_hidden_and_home_has_no_home_button(self):
        self.assertEqual(HOME_SOURCE.count('position="hidden"'), 1)
        self.assertNotIn("home_button", HOME_SOURCE)
        self.assertNotIn("stSidebar", HOME_SOURCE)
        for n in VIEWS:
            self.assertNotIn("st.sidebar", source(n))


def scroll_resets(at):
    """Number of one-shot scroll-to-top scripts the last run emitted."""
    return sum("scrollTo" in el.proto.body for el in at.get("html"))


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class OpenAtTopTests(unittest.TestCase):
    def test_no_page_scrolls_a_heading_into_view_on_navigation(self):
        # Module 2's one-shot "scroll to heading" put the heading under the sticky header and hid Back / Home.
        self.assertNotIn("SCROLL_TO_HEADING_JS", source(2))
        self.assertNotIn("scroll_to_heading", source(2))
        self.assertNotIn("m2_scroll_to\"", source(2))
        self.assertNotIn("scrollIntoView", HELPER)

    def test_scroll_reset_is_emitted_once_per_view_change_not_per_rerun(self):
        for n, (key, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    self.assertEqual(scroll_resets(at), 1)       # first render of the view: open at the top
                    at.run()                                     # ordinary rerun of the same view
                    self.assertEqual(scroll_resets(at), 0)
                    at.run()
                    self.assertEqual(scroll_resets(at), 0)

    def test_back_and_home_navigation_resets_the_destination(self):
        for n, (key, views) in VIEWS.items():
            at = run(n, views[1])
            self.assertEqual(scroll_resets(at), 1)
            back = [b for b in at.button if b.key and "back" in b.key and b.key.startswith("m")][0]
            back.click()
            at.run()
            self.assertEqual(scroll_resets(at), 1, msg=f"module {n}")
            self.assertNotEqual(at.session_state[key], views[1])

    def test_nav_row_keeps_the_navigation_design(self):
        self.assertIn('BUTTON_WIDTH = "110px"', HELPER)
        self.assertIn('BUTTON_HEIGHT = "40px"', HELPER)
        self.assertEqual(HELPER.count("window.scrollTo(0, 0)"), 1)


if __name__ == "__main__":
    unittest.main()
