"""Module 7 navigation: home -> my_results -> experiment / 3_results, home -> run_own -> 1 / 3_visitor (Setup Instructions is an inline expander on the 3_visitor page);  and state preservation."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import AppTest, PAGE, button, filled_state, page, wiz  # noqa: E402

SOURCE = Path(PAGE).read_text(encoding="utf-8")
Q1 = "Calibrated Stereo Object Measurement"
Q3 = "Room Measurement Using Stereo Videos"
RESULTS = "View My Experiments & Results"
RUN = "Run Your Own Experiment"
Q1_BTN = f"**{Q1}**  \n(Question 1)"
Q3_BTN = f"**{Q3}**  \n(Question 3)"
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
        self.assertEqual([b.label for b in at.button if b.key != "nav_arrow_button"], ["Home", RESULTS, RUN])
        self.assertEqual(len(at.caption), 0)
        self.assertEqual(len(at.subheader), 0)

    def test_question_two_is_not_in_navigation(self):
        for part in (None, "my_results", "run_own"):
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
                             "@media (min-width: 801px)", ".st-key-m7_open_results button", ".st-key-m7_open_run button",
                             "#1D5FD1", "flex-direction: column", "max-width: 560px"):
                self.assertIn(fragment, css)

    def test_selection_pages_share_layout_and_show_question_numbers_inside_buttons(self):
        for part, heading, keys, q1, q3 in (("my_results", "My Experiments & Results", ("m7_res_q1", "m7_res_q3"), "experiment", "3_results"),
                                            ("run_own", "Run Your Own Experiment", ("m7_run_q1", "m7_run_q3"), "1", "3_visitor")):
            at = fresh(part)
            self.assertFalse(at.exception, part)
            self.assertEqual(titles(at), [heading])
            self.assertEqual(len(at.header), 0)
            self.assertEqual(labels(at), [Q1_BTN, Q3_BTN])
            self.assertEqual([b.key for b in at.button if not (b.key or "").startswith("nav")], list(keys))
            css = " ".join(m.value for m in at.markdown if "<style>" in m.value)
            for fragment in ("height: 110px", "font-size: 21px", "p strong", "flex-direction: column", "max-width: 560px"):
                self.assertIn(fragment, css, part)
            button(at, Q1_BTN).click().run()
            self.assertFalse(at.exception, part)
            self.assertEqual(at.session_state["module7_part"], q1, part)
            at = fresh(part)
            button(at, Q3_BTN).click().run()
            self.assertFalse(at.exception, part)
            self.assertEqual(at.session_state["module7_part"], q3, part)

    def test_only_the_results_button_is_blue(self):
        css = " ".join(m.value for m in fresh().markdown if "<style>" in m.value)
        self.assertEqual(len(re.findall(r"\.st-key-m7_open_results button \{ background-color: \{?", css)), 1)
        self.assertNotIn(".st-key-m7_open_run button { background-color: #1D5FD1", css)

    def test_heading_size_scoped_to_calibrated_stereo_landing(self):
        def css(part):
            return " ".join(m.value for m in fresh(part).markdown if "<style>" in m.value)
        self.assertIn("clamp(2.5rem, 13vw, 80px)", css(None))
        self.assertNotIn("clamp(2.5rem, 8vw, 60px)", css(None))
        self.assertIn("clamp(2.5rem, 8vw, 60px)", css("my_results"))   # selection pages use the smaller heading
        for part in ("1", "experiment"):
            self.assertNotIn("clamp(2.5rem, 8vw, 60px)", css(part), part)
        self.assertIn("clamp(2.5rem, 8vw, 60px)", css("run_own"))

    def test_home_css_only_on_landing_pages(self):
        self.assertTrue(any("st-key-m7_home_layout" in m.value for m in fresh(None).markdown))
        for part in ("1", "experiment"):
            self.assertFalse(any("st-key-m7_home_layout" in m.value for m in fresh(part).markdown), part)

    def test_homepage_buttons_open_the_selection_pages(self):
        for label, part, heading in ((RESULTS, "my_results", "My Experiments & Results"), (RUN, "run_own", "Run Your Own Experiment")):
            at = fresh()
            button(at, label).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["module7_part"], part)
            self.assertEqual(titles(at), [heading])

    def test_results_selection_opens_results_directly(self):
        at = fresh("my_results")
        button(at, Q1_BTN).click().run()
        self.assertEqual(at.session_state["module7_part"], "experiment")
        self.assertIn("My Experiment & Results", [h.value for h in at.header])
        self.assertFalse(any("Part 1" in h.value for h in at.header))
        at = fresh("my_results")
        button(at, Q3_BTN).click().run()
        self.assertEqual(at.session_state["module7_part"], "3_results")
        self.assertEqual(titles(at), ["My Experiment & Results"])

    def test_run_selection_opens_visitor_workflows_directly(self):
        at = fresh("run_own")
        button(at, Q1_BTN).click().run()
        self.assertIn("Object Size Estimation Using Calibrated Stereo", [h.value for h in at.header])
        at = fresh("run_own")
        button(at, Q3_BTN).click().run()
        self.assertEqual(at.session_state["module7_part"], "3_visitor")
        self.assertEqual([b.key for b in at.button if (b.key or "").startswith("m7_open_room_")], ["m7_open_room_run"])   # Setup Instructions is an inline expander, not a navigation button
        self.assertEqual(len(at.get("expander")), 1)
        self.assertEqual(len(at.get("file_uploader")), 0)

    def test_breadcrumb_trails(self):
        trails = {"home": ["Module 7"], "my_results": ["Module 7", "My Experiments & Results"],
                  "run_own": ["Module 7", "Run Your Own Experiment"],
                  "experiment": ["Module 7", "My Experiments & Results", Q1],
                  "3_results": ["Module 7", "My Experiments & Results", "Room Measurement"],
                  "1": ["Module 7", "Run Your Own Experiment", Q1],
                  "3_visitor": ["Module 7", "Run Your Own Experiment", "Room Measurement"]}
        for part, trail in trails.items():
            at = fresh(part)
            row = next(e for e in at.main if getattr(e, "key", None) == "nav_row")
            ancestors = [b.label for b in row.button if (b.key or "").startswith("navbtn_")]
            self.assertEqual(ancestors, trail[:-1], part)
            self.assertEqual(next(e for e in row if getattr(e, "key", None) == "nav_current").markdown[0].value, trail[-1], part)

    def test_back_arrow_destinations_and_no_bottom_back_buttons(self):
        dest = {"home": None, "my_results": "home", "run_own": "home", "experiment": "my_results", "3_results": "my_results",
                "1": "run_own", "3_visitor": "run_own"}
        for part, target in dest.items():
            at = fresh(part)
            self.assertFalse(at.exception, part)
            self.assertFalse([b.key for b in at.button if "back_bottom" in (b.key or "") or b.label == "Back"], part)
            if target is not None:
                next(b for b in at.button if b.key == "nav_arrow_button").click().run()
                self.assertFalse(at.exception, part)
                self.assertEqual(at.session_state["module7_part"], target, part)

    def test_full_navigation_loop(self):
        at = fresh()
        # (control, resulting view): a label is a button, an int is the breadcrumb ancestor at that index
        for control, part in ((RESULTS, "my_results"), (Q1_BTN, "experiment"), (1, "my_results"), (Q3_BTN, "3_results"),
                              (1, "my_results"), (0, "home"), (RUN, "run_own"), (Q1_BTN, "1"), (1, "run_own"),
                              (Q3_BTN, "3_visitor"), (1, "run_own"), (0, "home")):
            (crumb(at, control) if isinstance(control, int) else button(at, control)).click().run()
            self.assertFalse(at.exception, control)
            self.assertEqual(at.session_state["module7_part"], part, control)

    def test_unknown_view_falls_back_to_the_homepage(self):
        at = fresh("q1_home")
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "home")

    def test_inline_experiment_has_no_extra_back_buttons(self):
        at = fresh("1")
        self.assertEqual(labels(at).count(BACK), 0)
        self.assertEqual(sum(l.startswith("Back") for l in labels(at)), 0)

    def test_breadcrumb_precedes_experiment_heading(self):
        source = SOURCE[SOURCE.index('view = st.session_state["module7_part"]'):]
        self.assertLess(source.index("nav_row("), source.index("show_experiment()"))

    def test_standalone_experiment_has_content_and_no_close_button(self):
        at = fresh("experiment")
        self.assertFalse(at.exception)
        self.assertEqual(labels(at), [])
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
        at = fresh("experiment")
        self.assertIn("Notebook Stereo Pair", [s.value for s in at.subheader])
        for pair in (1, 2, 3):
            at.radio(key="m7_exp_pair").set_value(pair).run()
            self.assertFalse(at.exception)
            self.assertIn(f"#### Pair {pair}: Photographs with the Final Approved Corners", [m.value for m in at.markdown])
        # structure (source order): the shaded container starts with the Notebook Stereo Pair heading and the radio, then the photograph
        # heading, and ends before the summary heading; there is one container, not a nested one
        body = SOURCE[SOURCE.index("def show_experiment"):]
        box = body.index('st.container(key="m7_selected_pair_results")')
        title, selector = body.index('st.subheader("Notebook Stereo Pair"'), body.index('"Notebook stereo pair"')
        heading = body.index('Pair {pair_id}: Photographs with the Final Approved Corners')
        summary = body.index('st.subheader("Summary of all three pairs")')
        self.assertLess(box, title)
        self.assertLess(title, selector)
        self.assertLess(selector, heading)
        self.assertLess(heading, summary)
        self.assertEqual(body.count('st.container(key="m7_selected_pair_results")'), 1)
        inside = body[box:summary]
        self.assertIn("m7_exp_pair", inside)
        self.assertIn("st.metric", inside.replace("c1.metric", "st.metric"))
        self.assertIn("final annotated stereo pair", inside)
        self.assertNotIn("depth_topdown.png", inside)
        for colour in ("#F3F4F6", "#D9DDE3", "#253240", "#405466"):
            self.assertEqual(SOURCE.count(colour), 1)
        css = SOURCE[SOURCE.index("SELECTED_PAIR_CSS = "):]
        css = css[:css.index("</style>")]
        self.assertIn("prefers-color-scheme: dark", css)
        self.assertEqual(css.count("border-radius: 12px"), 1)
        self.assertNotIn("stVerticalBlock", css)
        self.assertIn("h3 { font-size: 1.6rem; font-weight: 700", css)
        self.assertIn("h4 { font-size: 1.2rem; font-weight: 700", css)

    def test_objective_and_experimental_setup_precede_the_calibration_reference(self):
        at = fresh("experiment")
        self.assertFalse(at.exception)
        md = [m.value for m in at.markdown]
        self.assertEqual(md.count("##### Objective"), 1)
        self.assertEqual(md.count("##### Experimental setup"), 1)
        self.assertLess(md.index("##### Objective"), md.index("##### Experimental setup"))
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("Estimate the real-world width and height of a notebook using calibrated stereo images and classical computer vision techniques.", text)
        setup = next(m.value for m in at.markdown if m.value.startswith("Two phones were used as a stereo pair"))
        for fact in ("iPhone Air as the left camera", "iPhone 17 Pro Max as the right camera", "225 mm", "30 matched checkerboard image pairs",
                     "8 \u00d7 6 inner corners", "20 mm squares", "305 \u00d7 215 mm", "three stereo image pairs"):
            self.assertIn(fact, setup)
        body = SOURCE[SOURCE.index("def show_experiment"):]
        self.assertLess(body.index('"##### Experimental setup"'), body.index("show_calibration_reference(get_model())"))

    def test_homepage_button_text_is_18px_only_on_the_two_homepage_buttons(self):
        css = " ".join(m.value for m in fresh().markdown if "<style>" in m.value)
        self.assertIn(".st-key-m7_open_results button p, .st-key-m7_open_run button p { font-size: 18px; }", css)
        self.assertEqual(css.count("font-size: 18px"), 1)

    def test_visitor_page_no_longer_embeds_the_experiment(self):
        at = fresh("1")
        self.assertFalse(at.exception)
        self.assertEqual(labels(at).count("Close experiment"), 0)
        self.assertNotIn("View My Experiment & Results", labels(at))
        self.assertEqual([h.value for h in at.header].count("My Experiment & Results"), 0)
        for fragment in ("m7_experiment_btn", "toggle_experiment", "m7_show_experiment", "Close experiment"):
            self.assertNotIn(fragment, SOURCE)

    def test_legacy_routes_work(self):
        for part in ("1", "experiment", "3_visitor", "3_results", "3_setup"):
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
        crumb(at, 0).click().run()
        button(at, RESULTS).click().run()
        button(at, Q1_BTN).click().run()
        self.assertEqual(at.session_state["module7_part"], "experiment")
        crumb(at, 1).click().run()
        crumb(at, 0).click().run()
        button(at, RUN).click().run()
        button(at, Q1_BTN).click().run()
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
        button(at, Q1_BTN).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio[0].value, 2)
        self.assertEqual(at.session_state["m7_exp_pair"], 2)

    def test_actual_size_inputs_are_saved_and_restored_for_current_generation(self):
        at = page(filled_state(), m7_visitor_model=self.cal)
        at.session_state["m7_gt_w_0"], at.session_state["m7_gt_h_0"] = 123.5, 45.0
        at.session_state["m7_gt_w_7"] = 999.0                 # another generation: never saved
        crumb(at, 1).click().run()
        self.assertEqual(at.session_state["module7_part"], "run_own")
        saved = at.session_state["m7_saved_inputs"]
        self.assertEqual((saved["m7_gt_w_0"], saved["m7_gt_h_0"]), (123.5, 45.0))
        self.assertNotIn("m7_gt_w_7", saved)
        button(at, Q1_BTN).click().run()
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
        button(at, Q1_BTN).click().run()
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
