"""Module 7 navigation: home -> q1_home -> "1" (measurement) / "experiment", and "3";  and state preservation across navigation."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import AppTest, PAGE, button, filled_state, page, wiz  # noqa: E402

SOURCE = Path(PAGE).read_text(encoding="utf-8")
Q1 = "Calibrated Stereo Object Measurement"
Q3 = "Room Measurement Using Stereo Videos"
MEASURE = "**Measure Objects with Your Own Stereo Cameras**  \nCalibrate your cameras and measure objects"
EXPERIMENT = "**View My Experiment & Results**  \nExplore my stereo measurement results"
BACK = "Back"
BACK_Q1 = "Back"
BACK_CAL = "Back"


def crumb(at, index):
    """Breadcrumb ancestor button (0 = Module 7, 1 = the Stereo Object Measurement level)."""
    return at.button(key=f"navbtn_{index}")


def fresh(part=None, **state):
    at = AppTest.from_file(PAGE, default_timeout=120)
    if part is not None:
        at.session_state["module7_part"] = part
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def titles(at):
    return [t.value for t in at.title]


def labels(at):
    return [b.label for b in at.button if not (b.key or "").startswith("nav")]   # Home is covered by the Home navigation tests


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class HomepageTests(unittest.TestCase):
    def test_home_title_subtitle_and_exact_buttons(self):
        at = fresh()
        self.assertFalse(at.exception)
        self.assertEqual(titles(at), ["Module 7"])
        self.assertEqual([h.value for h in at.header], ["Stereo Vision"])
        self.assertEqual([b.label for b in at.button if b.key != "nav_arrow_button"], ["Home", Q1, Q3])
        self.assertEqual(len(at.caption), 0)
        self.assertEqual(len(at.subheader), 0)

    def test_question_two_is_not_in_navigation(self):
        for part in (None, "q1_home"):
            at = fresh(part)
            self.assertFalse(any("Question 2" in b or "Uncalibrated" in b for b in labels(at)))
            self.assertFalse(any("Question 2" in m.value or "Uncalibrated" in m.value for m in at.markdown))
        self.assertNotIn("Question 2", SOURCE)

    def test_css_matches_approved_specification(self):
        for part in (None,):
            css = next(m.value for m in fresh(part).markdown if "st-key-m7_home_layout" in m.value)
            for fragment in ("clamp(2.5rem, 13vw, 80px)", "font-weight: 800", "clamp(1.1rem, 4.6vw, 32px)", "font-weight: 600",
                             "padding-top: 44px", "padding-top: 64px", "height: 110px", "gap: 24px", "#C9363E", "#A62A31", "#8E2229",
                             "border-radius: 12px", "font-size: 21px", "@media (max-width: 640px)", "gap: 16px",
                             "@media (min-width: 801px)", ".st-key-m7_open_q1 button", ".st-key-m7_open_q3 button"):
                self.assertIn(fragment, css)

    def test_q1_selection_page_uses_horizontal_panel_css(self):
        at = fresh("q1_home")
        css = " ".join(m.value for m in at.markdown if "<style>" in m.value)
        self.assertNotIn("m7_home_layout", css)
        for fragment in ("max-width: 500px", "padding-bottom: 270px", "margin-bottom: 24px", "margin-bottom: 44px",
                         "clamp(1.5rem, 6.5vw, 2.25rem)", "white-space: nowrap", "@media (max-width: 640px)",
                         ".st-key-m7_open_measure button:hover", ".st-key-m7_open_measure button:focus-visible",
                         ".st-key-m7_open_experiment button::before", ".st-key-m7_open_experiment button::after"):
            self.assertIn(fragment, css)
        self.assertIn("MODULE 7", [m.value for m in at.markdown])
        self.assertEqual(len(at.title), 0)
        self.assertEqual([h.value for h in at.header], [Q1])

    def test_heading_size_scoped_to_calibrated_stereo_landing(self):
        def css(part):
            return " ".join(m.value for m in fresh(part).markdown if "<style>" in m.value)
        self.assertIn("clamp(2.5rem, 13vw, 80px)", css(None))
        self.assertNotIn("60px", css(None))
        self.assertNotIn("60px", css("q1_home"))   # selection page has its own heading CSS
        for part in ("1", "experiment"):
            self.assertNotIn("60px", css(part), part)
        self.assertIn("clamp(2.5rem, 8vw, 60px)", css("3"))

    def test_home_css_only_on_landing_pages(self):
        self.assertTrue(any("st-key-m7_home_layout" in m.value for m in fresh(None).markdown))
        self.assertFalse(any("st-key-m7_home_layout" in m.value for m in fresh("q1_home").markdown))
        for part in ("1", "experiment"):
            self.assertFalse(any("st-key-m7_home_layout" in m.value for m in fresh(part).markdown), part)

    def test_q1_button_opens_question_one_landing(self):
        at = fresh()
        button(at, Q1).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "q1_home")
        self.assertEqual(titles(at), [])
        self.assertEqual([h.value for h in at.header], [Q1])
        self.assertFalse(any("Question 1" in m.value for m in at.markdown))
        self.assertEqual(labels(at), [MEASURE, EXPERIMENT])
        self.assertEqual(len(at.radio), 0)
        self.assertEqual(len(at.metric), 0)

    def test_measure_button_opens_part_one(self):
        at = fresh("q1_home")
        button(at, MEASURE).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "1")
        self.assertIn("Object Size Estimation Using Calibrated Stereo", [h.value for h in at.header])

    def test_experiment_button_opens_saved_experiment_directly(self):
        at = fresh("q1_home")
        button(at, EXPERIMENT).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "experiment")
        self.assertIn("My Experiment & Results", [h.value for h in at.header])
        self.assertFalse(any("Part 1" in h.value for h in at.header))

    def test_question_three_button_opens_room_landing_with_two_buttons(self):
        at = fresh()
        button(at, Q3).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "3")
        self.assertEqual(titles(at), [Q3])
        self.assertFalse(any("Part 3" in h.value for h in list(at.header) + list(at.title)))
        self.assertEqual(labels(at), ["Measure a Room Using Your Own Stereo Videos", "View My Experiment & Results"])
        self.assertEqual(len(at.info), 0)

    def test_room_visitor_button_opens_the_placeholder(self):
        at = fresh("3")
        button(at, "Measure a Room Using Your Own Stereo Videos").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "3_visitor")
        self.assertEqual(len(at.info), 1)
        self.assertTrue(any("not implemented" in i.value.lower() for i in at.info))

    def test_each_view_has_a_breadcrumb_and_no_module_title(self):
        expected = {"q1_home": BACK, "1": BACK_Q1, "3": BACK}
        for part, back in expected.items():
            at = fresh(part)
            self.assertFalse(at.exception, part)
            self.assertEqual(labels(at).count(back), 0, part)       # no top Back: the breadcrumb is the navigation
            self.assertEqual(sum(l.startswith("Back") for l in labels(at)), 0, part)
            self.assertIsNotNone(crumb(at, 0), part)
            self.assertFalse(any(h.value == "Module 7" for h in list(at.header) + list(at.subheader) + list(at.title)), part)
            if part == "3":
                self.assertEqual(titles(at), [Q3], part)
            elif part != "q1_home":
                self.assertEqual(titles(at), [], part)

    def test_back_navigation(self):
        for part, back, dest in (("1", BACK_Q1, "q1_home"), ("experiment", BACK_CAL, "q1_home"),
                                 ("q1_home", BACK, "home"), ("3", BACK, "home")):
            at = fresh(part)
            crumb(at, 0 if part in ("q1_home", "3") else 1).click().run()
            self.assertFalse(at.exception, part)
            self.assertEqual(at.session_state["module7_part"], dest, part)
        self.assertEqual(titles(at), ["Module 7"])

    def test_full_navigation_loop(self):
        at = fresh()
        # (control, resulting view): a label is a panel button, an int is the breadcrumb ancestor at that index
        for control, part in ((Q1, "q1_home"), (MEASURE, "1"), (1, "q1_home"), (EXPERIMENT, "experiment"),
                              (1, "q1_home"), (0, "home"), (Q3, "3"), (0, "home")):
            (crumb(at, control) if isinstance(control, int) else button(at, control)).click().run()
            self.assertFalse(at.exception, control)
            self.assertEqual(at.session_state["module7_part"], part, control)

    def test_bottom_back_returns_to_calibrated_stereo_landing(self):
        at = fresh("experiment")
        [b for b in at.button if b.key == "m7_back_bottom"][0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "q1_home")
        self.assertEqual(labels(at).count(BACK), 0)      # the landing page has no Back, only the breadcrumb

    def test_bottom_experiment_back_works(self):
        at = fresh("experiment")
        keys = [b.key for b in at.button if not (b.key or "").startswith("nav")]
        self.assertEqual(keys, ["m7_back_bottom"])
        self.assertEqual(len(set(keys)), 1)
        for key in keys:
            at = fresh("experiment")
            [b for b in at.button if b.key == key][0].click().run()
            self.assertFalse(at.exception, key)
            self.assertEqual(at.session_state["module7_part"], "q1_home", key)

    def test_inline_experiment_has_no_extra_back_buttons(self):
        at = fresh("1", m7_show_experiment=True)
        self.assertEqual(labels(at).count(BACK), 0)
        self.assertEqual(sum(l.startswith("Back") for l in labels(at)), 0)

    def test_breadcrumb_precedes_experiment_heading(self):
        source = SOURCE[SOURCE.index('view = st.session_state["module7_part"]'):]
        self.assertLess(source.index("nav_row("), source.index("show_experiment(standalone=True)"))

    def test_standalone_experiment_has_content_and_no_close_button(self):
        at = fresh("experiment")
        self.assertFalse(at.exception)
        self.assertEqual(labels(at), [BACK_CAL])
        self.assertEqual([b.key for b in at.button if not (b.key or "").startswith("nav")], ["m7_back_bottom"])
        self.assertEqual(len(at.radio), 1)
        self.assertGreaterEqual(len(at.metric), 3)
        self.assertGreaterEqual(len(at.table), 4)
        self.assertEqual([h.value for h in at.header].count("My Experiment & Results"), 1)
        self.assertIn("Summary of all three pairs", [s.value for s in at.subheader])
        # never started calibration or measurement
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertFalse(at.session_state["m7_cal_run_requested"])

    def test_standalone_experiment_works_in_every_pair(self):
        at = fresh("experiment")
        for pair in (1, 2, 3):
            at.radio[0].set_value(pair).run()
            self.assertEqual(at.session_state["m7_exp_pair"], pair)
            self.assertFalse(at.exception)

    def test_selected_pair_results_are_grouped_in_one_themed_container(self):
        for view, state in (("experiment", {}), ("1", {"m7_show_experiment": True})):
            at = fresh(view, **state)
            for pair in (1, 2, 3):
                at.radio(key="m7_exp_pair").set_value(pair).run()
                self.assertFalse(at.exception)
                self.assertIn(f"Pair {pair}: photographs with the final approved corners", [s.value for s in at.subheader])
        # structure (source order): selector, then the container, which ends before the summary heading
        body = SOURCE[SOURCE.index("def show_experiment"):]
        selector, box = body.index('"Notebook stereo pair"'), body.index('st.container(key="m7_selected_pair_results")')
        heading = body.index('st.subheader(f"Pair {pair_id}: photographs')
        summary = body.index('st.subheader("Summary of all three pairs")')
        self.assertLess(selector, box)
        self.assertLess(box, heading)
        self.assertLess(heading, summary)
        inside = body[box:summary]
        self.assertIn("st.metric", inside.replace("c1.metric", "st.metric"))
        self.assertIn("final annotated stereo pair", inside)
        self.assertNotIn("depth_topdown.png", inside)
        self.assertNotIn("m7_exp_pair", inside)
        for colour in ("#F3F4F6", "#D9DDE3", "#253240", "#405466"):
            self.assertEqual(SOURCE.count(colour), 1)
        css = SOURCE[SOURCE.index("SELECTED_PAIR_CSS = "):]
        css = css[:css.index("</style>")]
        self.assertIn("prefers-color-scheme: dark", css)
        self.assertEqual(css.count("border-radius: 12px"), 1)
        self.assertNotIn("stVerticalBlock", css)

    def test_inline_experiment_in_part_one_still_works_with_close(self):
        at = fresh("1", m7_show_experiment=True)
        self.assertFalse(at.exception)
        self.assertEqual(labels(at).count("Close experiment"), 1)
        self.assertEqual(labels(at).count(BACK_Q1), 0)
        self.assertEqual([h.value for h in at.header].count("My Experiment & Results"), 1)

    def test_legacy_routes_work(self):
        for part in ("1", "experiment", "3"):
            self.assertFalse(fresh(part).exception, part)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class NavigationStateTests(unittest.TestCase):
    def setUp(self):
        self.cal = wiz.run_calibration(filled_state())

    def test_part_one_state_survives_home_and_experiment_round_trip(self):
        points = {"left": [(1.0, 2.0), (3.0, 4.0)], "right": [(5.0, 6.0)]}
        at = page(filled_state(), m7_visitor_model=self.cal, m7_file_left=("l.jpg", b"L"), m7_file_right=("r.jpg", b"R"),
                  m7_points=points, m7_pending=("left", (9.0, 9.0)), m7_refined=(9.5, 9.5), m7_left_confirmed=True, m7_stage="right")
        before = {k: at.session_state[k] for k in ("m7_file_left", "m7_file_right", "m7_points", "m7_pending", "m7_refined",
                                                   "m7_left_confirmed", "m7_stage", "m7_cal_source", "m7_measurement_enabled")}
        cal_before = at.session_state["m7_visitor_model"]
        slots_before = dict(at.session_state["m7_cal"]["photos"])
        crumb(at, 1).click().run()
        button(at, EXPERIMENT).click().run()
        self.assertEqual(at.session_state["module7_part"], "experiment")
        crumb(at, 1).click().run()
        crumb(at, 0).click().run()
        button(at, Q1).click().run()
        button(at, MEASURE).click().run()
        self.assertFalse(at.exception)
        for key, value in before.items():
            self.assertEqual(at.session_state[key], value, key)
        self.assertIs(at.session_state["m7_visitor_model"], cal_before)
        self.assertEqual(dict(at.session_state["m7_cal"]["photos"]), slots_before)

    def test_experiment_pair_selection_is_preserved(self):
        at = fresh("experiment")
        at.radio[0].set_value(2).run()
        crumb(at, 1).click().run()
        self.assertNotIn("m7_exp_pair", at.session_state)
        button(at, EXPERIMENT).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, 2)
        self.assertEqual(at.session_state["m7_exp_pair"], 2)

    def test_actual_size_inputs_are_saved_and_restored_for_current_generation(self):
        at = page(filled_state(), m7_visitor_model=self.cal)
        at.session_state["m7_gt_w_0"], at.session_state["m7_gt_h_0"] = 123.5, 45.0
        at.session_state["m7_gt_w_7"] = 999.0                 # another generation: never saved
        crumb(at, 1).click().run()
        saved = at.session_state["m7_saved_inputs"]
        self.assertEqual((saved["m7_gt_w_0"], saved["m7_gt_h_0"]), (123.5, 45.0))
        self.assertNotIn("m7_gt_w_7", saved)
        button(at, MEASURE).click().run()
        self.assertEqual((at.session_state["m7_gt_w_0"], at.session_state["m7_gt_h_0"]), (123.5, 45.0))

    def test_restore_never_overwrites_newer_values(self):
        at = fresh("experiment", m7_saved_inputs={"m7_exp_pair": 1})
        at.session_state["m7_exp_pair"] = 3
        at.run()
        self.assertEqual(at.radio[0].value, 3)

    def test_calibration_details_toggle_is_preserved(self):
        at = page(filled_state(), m7_visitor_model=self.cal)
        toggle = next((t for t in at.toggle if t.key == "m7_cal_details"), None)
        if toggle is None:
            self.skipTest("details toggle is not shown in this state")
        toggle.set_value(True).run()
        crumb(at, 1).click().run()
        button(at, MEASURE).click().run()
        self.assertTrue(next(t for t in at.toggle if t.key == "m7_cal_details").value)

    def test_reset_all_still_clears_everything_and_saved_inputs(self):
        at = page(filled_state(), m7_visitor_model=self.cal, m7_file_left=("l.jpg", b"L"), m7_file_right=("r.jpg", b"R"),
                  m7_saved_inputs={"m7_exp_pair": 2, "m7_gt_w_0": 5.0})
        button(at, "Reset All").click().run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIsNone(at.session_state["m7_file_left"])
        self.assertEqual(at.session_state["m7_points"], {"left": [], "right": []})
        self.assertNotIn("m7_gt_w_0", at.session_state["m7_saved_inputs"])
        self.assertNotIn("m7_exp_pair", at.session_state["m7_saved_inputs"])

    def test_navigation_has_no_rerun_calls_and_one_state_variable(self):
        navigation = SOURCE[SOURCE.index("def select_part"):SOURCE.index("def store_upload")]
        self.assertNotIn("st.rerun", navigation)
        self.assertEqual(len(re.findall(r'module7_part', SOURCE[SOURCE.index("def select_part"):])), 2)


if __name__ == "__main__":
    unittest.main()
