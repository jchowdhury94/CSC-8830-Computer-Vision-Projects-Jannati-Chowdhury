"""Breadcrumb navigation (Home > ... > current page): exactly one on every reachable view of Modules 2, 3, 4, 6 and 7, always routed to the dashboard."""
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


# number of breadcrumb levels below Home per view (the last one is the current page)
DEPTH = {2: {"home": 1, "visitor": 2, "experiment": 2}, 3: {"home": 1, "visitor": 2, "experiment": 2},
         4: {"home": 1, "rgb_select": 2, "thermal_select": 2, "rgb": 3, "rgb_results": 3, "thermal": 3, "thermal_results": 3},
         6: {"home": 1, "a_select": 2, "b_select": 2, "a_visitor": 3, "a_results": 3, "b_visitor": 3, "b_results": 3},
         7: {"home": 1, "q1_home": 2, "1": 3, "experiment": 3, "3": 2}}


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

    def test_home_crumb_has_home_icon_and_no_top_back_button_remains(self):
        for n, (_, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    self.assertEqual(home_buttons(at)[0].proto.icon, ":material/home:")
                    row = next(e for e in at.main if getattr(e, "key", None) == "nav_row")
                    self.assertEqual([b for b in row.button if "Back" in b.label], [])
                    self.assertFalse([b.key for b in at.button if b.key and b.key.endswith("_top") and "back" in b.key])

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

    def test_breadcrumb_trails_and_clickability(self):
        for n, (_, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    crumbs = [b for b in at.button if b.key and b.key.startswith("navbtn_")]
                    depth = DEPTH[n][view]
                    self.assertEqual(len(crumbs), depth - 1)       # every ancestor clickable, current page is not a button
                    row = next(e for e in at.main if getattr(e, "key", None) == "nav_row")
                    self.assertEqual(len(row.button), depth + 1)  # Home + ancestors + the Back arrow (on every module view)

    def test_back_arrow_on_every_module_view_and_returns_to_the_immediate_parent(self):
        parents = {(2, "visitor"): "home", (2, "experiment"): "home", (3, "visitor"): "home", (3, "experiment"): "home",
                   (4, "rgb_select"): "home", (4, "rgb"): "rgb_select", (4, "rgb_results"): "rgb_select",
                   (4, "thermal"): "thermal_select", (4, "thermal_results"): "thermal_select",
                   (6, "a_select"): "home", (6, "a_visitor"): "a_select", (6, "b_results"): "b_select",
                   (7, "q1_home"): "home", (7, "1"): "q1_home", (7, "experiment"): "q1_home", (7, "3"): "home"}
        for n, (key, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    arrows = [b for b in at.button if b.key == "nav_arrow_button"]
                    self.assertEqual(len(arrows), 1)
                    if view == "home":  # module homepage: the arrow leads to the main dashboard
                        with mock.patch("home_navigation.st.switch_page") as switch:
                            at.session_state["_home_page"] = "HOME_PAGE_SENTINEL"
                            arrows[0].click().run()
                        self.assertFalse(at.exception)
                        switch.assert_called_once_with("HOME_PAGE_SENTINEL")
                        continue
                    if (n, view) in parents:
                        arrows[0].click().run()
                        self.assertFalse(at.exception)
                        self.assertEqual(at.session_state[key], parents[(n, view)])

    def test_breadcrumb_navigation_goes_to_the_parent_view(self):
        expectations = [(2, "experiment", 0, "m2_page", "home"), (3, "visitor", 0, "m3_page", "home"),
                        (4, "rgb", 0, "module4_section", "home"), (4, "rgb", 1, "module4_section", "rgb_select"),
                        (4, "thermal_results", 1, "module4_section", "thermal_select"),
                        (6, "b_results", 0, "module6_section", "home"), (6, "b_results", 1, "module6_section", "b_select"),
                        (6, "a_visitor", 1, "module6_section", "a_select"),
                        (7, "experiment", 0, "module7_part", "home"), (7, "experiment", 1, "module7_part", "q1_home"),
                        (7, "1", 1, "module7_part", "q1_home"), (7, "3", 0, "module7_part", "home")]
        for n, view, index, key, target in expectations:
            with self.subTest(module=n, view=view, crumb=index):
                at = run(n, view)
                next(b for b in at.button if b.key == f"navbtn_{index}").click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state[key], target)


class HomeNavigationSourceTests(unittest.TestCase):
    def test_every_module_renders_the_shared_button_once_at_its_dispatcher(self):
        for n in VIEWS:
            with self.subTest(module=n):
                text = source(n)
                self.assertEqual(text.count("nav_row("), 1)
                self.assertIn("from home_navigation import back_button, nav_row", text)
                self.assertIn("CRUMBS = {", text)
                self.assertNotIn("back_to_home", text)
                self.assertNotIn("nav_home", text)  # no per-module copies of the key or CSS

    def test_helper_is_shared_and_uses_existing_page_navigation(self):
        self.assertIn("st.switch_page(st.session_state[HOME_PAGE_KEY])", HELPER)
        self.assertIn('key="nav_home_button"', HELPER)
        self.assertIn('icon=":material/home:"', HELPER)
        self.assertIn("st.Page(show_home", HOME_SOURCE)
        self.assertEqual(re.findall(r'MODULE_PAGES\["(\d)"\],', HOME_SOURCE), ["2", "3", "4", "6", "7"])

    def test_css_contract_breadcrumb_red_home_muted_ancestors_white_current_wrapping(self):
        for fragment in ("color: {ACCENT}", "var(--nav-muted)", "var(--nav-text)", "font-weight: 700", "flex-wrap: wrap",
                         'BUTTON_WIDTH = "110px"', 'BUTTON_HEIGHT = "40px"', "button:focus-visible", "horizontal=True",
                         'vertical_alignment="center"', "203A", 'key="nav_arrow_button"', ":material/arrow_back:", "2rem"):
            self.assertIn(fragment, HELPER)
        self.assertNotIn("position: fixed", HELPER)
        self.assertEqual(HELPER.count("position: absolute"), 1)  # only the Back arrow tooltip; the row itself is in normal flow
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

    def test_breadcrumb_row_is_above_the_content_and_ends_with_the_current_page(self):
        for n, (_, views) in VIEWS.items():
            for view in views:
                with self.subTest(module=n, view=view):
                    at = run(n, view)
                    row = next(e for e in at.main if getattr(e, "key", None) == "nav_row")
                    content = [e for e in at.main if e.type in ("title", "header", "subheader")]
                    self.assertLess(list(at.main).index(row), list(at.main).index(content[0]) if content else 99)
                    keys = [b.key for b in row.button if b.key != "nav_arrow_button"]
                    self.assertEqual(keys[0], "nav_home_button")
                    self.assertEqual(len(keys), DEPTH[n][view])

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
            at.button(key="navbtn_0").click()
            at.run()
            self.assertEqual(scroll_resets(at), 1, msg=f"module {n}")
            self.assertNotEqual(at.session_state[key], views[1])

    def test_nav_row_keeps_the_navigation_design(self):
        self.assertIn('BUTTON_WIDTH = "110px"', HELPER)  # bottom Back buttons keep the shared size
        self.assertEqual(HELPER.count("window.scrollTo(0, 0)"), 1)


if __name__ == "__main__":
    unittest.main()
