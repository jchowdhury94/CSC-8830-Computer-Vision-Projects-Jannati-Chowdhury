"""Home resets every module's navigation: a module card always opens that module's homepage."""
import unittest
from pathlib import Path
from unittest import mock

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

import home_navigation as hn

ROOT = Path(__file__).resolve().parents[1]
KEYS = hn.MODULE_VIEW_KEYS
# module -> nested view reached from the module homepage
NESTED = {"2": "experiment", "3": "experiment", "4": "thermal_results", "6": "b_select", "7": "experiment"}
VISITOR = {"2": "visitor", "3": "visitor", "4": "thermal", "6": "b_visitor", "7": "1"}


def module(n, **state):
    at = AppTest.from_file(str(ROOT / "Pages" / f"app_module{n}.py"), default_timeout=120)
    at.session_state["_home_page"] = "HOME_SENTINEL"
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def click_home(at):
    with mock.patch("home_navigation.st.switch_page") as switch:
        next(b for b in at.button if b.key == "nav_home_button").click().run()
    switch.assert_called_once_with("HOME_SENTINEL")


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class HomeResetTests(unittest.TestCase):
    def test_home_from_nested_view_resets_all_five_modules(self):
        for n, view in {**NESTED, "2": "visitor"}.items():
            with self.subTest(module=n):
                others = {KEYS[m]: NESTED[m] for m in KEYS if m != n}
                at = module(n, **{KEYS[n]: view}, **others)
                click_home(at)
                for m in KEYS:
                    self.assertEqual(at.session_state[KEYS[m]], "home", m)

    def test_each_scenario_reopens_on_the_module_homepage(self):
        for n in NESTED:
            for view in (NESTED[n], VISITOR[n]):
                with self.subTest(module=n, view=view):
                    at = module(n, **{KEYS[n]: view})
                    click_home(at)
                    # the card click runs the same reset, then the module page runs again
                    at.run()
                    self.assertEqual(at.session_state[KEYS[n]], "home")
                    self.assertFalse(at.exception)

    def test_module2_visitor_workflow_selection_is_cleared_but_data_is_kept(self):
        at = module("2", m2_page="visitor", m2_cal_workflow="run", m2_visitor_calibration={"keep": 1},
                    m2_saved_inputs={"m2_distance": 5})
        click_home(at)
        self.assertIsNone(at.session_state["m2_cal_workflow"])
        self.assertEqual(at.session_state["m2_visitor_calibration"], {"keep": 1})
        self.assertEqual(at.session_state["m2_saved_inputs"], {"m2_distance": 5})

    def test_reset_touches_only_navigation_keys(self):
        fake = {"m4_data": object(), "m6b_files": [1], "m7_visitor_model": {"x": 1}, "module3_frequency_result": 3}
        with mock.patch.object(hn.st, "session_state", dict(fake, _nav_last_view="m4:rgb", module4_section="rgb")) as ss:
            hn.reset_module_navigation()
            for k, v in fake.items():
                self.assertIs(ss[k], v)
            self.assertEqual(ss["module4_section"], "home")
            self.assertNotIn("_nav_last_view", ss)

    def test_card_resets_only_its_module_then_a_different_module_also_opens_home(self):
        src = (ROOT / "home.py").read_text(encoding="utf-8")
        self.assertIn("reset_module_navigation([key])", src)
        self.assertIn("reset_module_navigation()  # also covers", src)
        ss = {KEYS[m]: NESTED[m] for m in KEYS}
        with mock.patch.object(hn.st, "session_state", ss):
            hn.reset_module_navigation(["4"])
            self.assertEqual(ss["module4_section"], "home")
            self.assertEqual(ss["m2_page"], "experiment")
            hn.reset_module_navigation()
            self.assertTrue(all(ss[KEYS[m]] == "home" for m in KEYS))

    def test_dashboard_run_resets_every_module(self):
        at = AppTest.from_file(str(ROOT / "home.py"), default_timeout=60)
        for m in KEYS:
            at.session_state[KEYS[m]] = NESTED[m]
        at.run()
        for m in KEYS:
            self.assertEqual(at.session_state[KEYS[m]], "home")

    def test_back_still_returns_to_parent_and_experiment_data_persists(self):
        at = module("4", module4_section="thermal_results")
        next(b for b in at.button if b.key == "m4_thermal_results_back_top").click().run()
        self.assertEqual(at.session_state["module4_section"], "thermal_select")
        at = module("6", module6_section="b_results")
        next(b for b in at.button if b.key == "m6_b_results_back_top").click().run()
        self.assertEqual(at.session_state["module6_section"], "b_select")

    def test_ordinary_reruns_do_not_reset_navigation(self):
        for n, view in {"4": "thermal_results", "7": "experiment", "3": "experiment", "2": "experiment",
                        "6": "b_select"}.items():
            with self.subTest(module=n):
                at = module(n, **{KEYS[n]: view})
                at.run()
                at.run()
                self.assertEqual(at.session_state[KEYS[n]], view)
        at = module("3", m3_page="visitor")
        for w in list(at.slider)[:1]:
            w.set_value(w.value).run()
        self.assertEqual(at.session_state["m3_page"], "visitor")


if __name__ == "__main__":
    unittest.main()
