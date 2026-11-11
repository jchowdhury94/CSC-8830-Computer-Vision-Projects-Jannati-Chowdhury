"""Module 6 visitor page workflow tests (AppTest; session state seeded instead of a browser upload)."""
import hashlib
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_module6_visitor_backend import make_video  # noqa: E402

PAGE = ROOT / "Pages" / "app_module6.py"


def visitor_app(data=None, name="clip.mp4"):
    app = AppTest.from_file(str(PAGE), default_timeout=90)
    app.session_state["module6_section"] = "a_visitor"
    if data is not None:
        app.session_state["m6v_bytes"] = data
        app.session_state["m6v_filename"] = name
        app.session_state["m6v_digest"] = hashlib.sha256(data).hexdigest()
    app.run()
    assert not app.exception, app.exception
    return app


def uploaded_app(data, name="clip.mp4"):
    """Open the empty page and upload through the native uploader (runs on_change, analysis, auto-advance)."""
    app = visitor_app()
    app.get("file_uploader")[0].set_value((name, data, "video/mp4"))
    run(app)
    return app


def at_step(data, step, points=None, upload=True):
    """Open the page with a video, then set the inputs a later step needs and rerun."""
    app = uploaded_app(data) if upload else visitor_app(data)
    if step >= 3:
        app.session_state["m6v_points"] = points
        app.session_state["m6v_settings_open"] = True
        run(app)
    return app


def run(app):
    app.run()
    assert not app.exception, app.exception


def button(app, key):
    return next(b for b in app.button if b.key == key)


def scroll_targets(app):
    """Steps the page asked the browser to scroll to during the last run (the one-shot script elements)."""
    import re
    found = []
    for el in app.get("html"):
        found += [int(n) for n in re.findall(r"\.st-key-m6v_step_(\d)\"", el.proto.body)]
    return found


def keys(app):
    return [getattr(el, "key", None) for el in app.main]


class VisitorPageTests(unittest.TestCase):
    CORNERS = [(30, 90), (89, 149)]

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            cls.data = Path(make_video(d, frames=12, speed=3)).read_bytes()
            cls.other = Path(make_video(d, frames=10, speed=2)).read_bytes()

    def headers(self, app):
        return [h.value for h in app.header]

    def run_tracking(self, app):
        button(app, "m6v_run").click()
        run(app)

    def test_empty_page_shows_only_step_one(self):
        app = visitor_app()
        self.assertEqual(self.headers(app), ["Track Motion in Your Own Video", "Step 1 — Upload Your Video"])
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertEqual(app.metric.values, [])
        self.assertIn("m6v_step_1", keys(app))
        for step in (2, 3, 4):
            self.assertNotIn(f"m6v_step_{step}", keys(app))
        labels = [b.label for b in app.button]
        self.assertNotIn("Clear Video", labels)
        self.assertNotIn("Change Video", labels)
        self.assertEqual(scroll_targets(app), [])
        self.assertFalse(app.get("dialog") or app.toast)

    def test_uploader_limit_single_file_and_scoped_css(self):
        app = visitor_app()
        proto = app.get("file_uploader")[0].proto
        self.assertEqual(proto.max_upload_size_mb, 50)
        self.assertFalse(proto.multiple_files)
        source = PAGE.read_text()
        css = source[source.index("VISITOR_UPLOADER_CSS = "):source.index("def visitor_upload_key")]
        self.assertIn('[class*="st-key-m6v_upload_"] button[aria-label="Add files"]', css)
        self.assertEqual(css.count("{"), 1)
        info = source[source.index("VISITOR_INFO_CSS = "):source.index("def visitor_video_info")]
        for rule in [line for line in info.splitlines() if "{" in line]:
            self.assertIn("m6v_video_info", rule)
        for other in ("app_module2.py", "app_module3.py", "app_module4.py", "app_module7.py"):
            self.assertNotIn("Add files", (ROOT / "Pages" / other).read_text())

    def test_upload_reveals_step_two_and_requests_scroll_once(self):
        app = uploaded_app(self.data)
        self.assertEqual(self.headers(app), ["Track Motion in Your Own Video", "Step 1 — Upload Your Video",
                                             "Step 2 — Select the Object"])
        self.assertEqual(scroll_targets(app), [2])
        self.assertEqual(len(app.get("component_instance")), 1)
        self.assertIsNone(app.session_state["m6v_results"])  # tracking never starts automatically
        self.assertEqual([m.label for m in app.metric], ["Resolution", "Duration", "Frame rate", "Frames"])
        self.assertEqual(app.metric[0].value, "320 × 240")
        for _ in range(2):  # unrelated reruns do not scroll
            run(app)
            self.assertEqual(scroll_targets(app), [])

    def test_render_order_is_vertical_and_complete(self):
        app = at_step(self.data, 3, self.CORNERS)
        order = keys(app)
        marks = ["navbtn_1", "m6v_step_1", "m6v_step_2", "m6v_step_3", "m6_a_visitor_back_bottom"]
        positions = [order.index(m) for m in marks]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.headers(app), ["Track Motion in Your Own Video", "Step 1 — Upload Your Video",
                                             "Step 2 — Select the Object", "Step 3 — Tracking Settings"])

    def test_step_two_content_order(self):
        app = uploaded_app(self.data)
        step2 = next(el for el in app.main if getattr(el, "key", None) == "m6v_step_2")
        inner = [getattr(el, "key", None) or el.type for el in step2.children.values()]
        actions = next(el for el in step2.children.values() if getattr(el, "key", None) == "m6v_select_actions")
        row = [getattr(el, "key", None) for el in actions.children.values()]
        self.assertLess(inner.index("m6v_click"), inner.index("m6v_select_actions"))
        self.assertEqual(row, ["m6v_reset_selection", "m6v_continue"])

    def test_rectangle_text_sits_between_preview_and_buttons(self):
        app = uploaded_app(self.data)
        app.session_state["m6v_points"] = self.CORNERS
        run(app)
        step2 = next(el for el in app.main if getattr(el, "key", None) == "m6v_step_2")
        inner = [getattr(el, "key", None) or el.type for el in step2.children.values()]
        caption = next(i for i, el in enumerate(step2.children.values())
                       if el.type == "caption" and "Selected rectangle" in el.value)
        self.assertGreater(caption, inner.index("m6v_click"))
        self.assertLess(caption, inner.index("m6v_select_actions"))

    def test_later_steps_require_prerequisites(self):
        app = uploaded_app(self.data)
        self.assertTrue(button(app, "m6v_continue").disabled)
        app.session_state["m6v_settings_open"] = True  # open flag alone, no ROI
        run(app)
        self.assertNotIn("m6v_step_3", keys(app))
        app.session_state["m6v_results"] = object()  # results alone, step 3 closed
        app.session_state["m6v_settings_open"] = False
        app.session_state["m6v_points"] = self.CORNERS
        run(app)
        self.assertNotIn("m6v_step_3", keys(app))
        self.assertNotIn("m6v_step_4", keys(app))

    def test_continue_requires_valid_roi_and_scrolls_to_step_three(self):
        app = uploaded_app(self.data)
        app.session_state["m6v_points"] = [(10, 10), (12, 40)]
        run(app)
        self.assertTrue(button(app, "m6v_continue").disabled)
        self.assertTrue(any("too small" in w.value for w in app.warning))
        app.session_state["m6v_points"] = self.CORNERS
        run(app)
        self.assertFalse(button(app, "m6v_continue").disabled)
        button(app, "m6v_continue").click()
        run(app)
        self.assertIn("m6v_step_3", keys(app))
        self.assertEqual(scroll_targets(app), [3])
        self.assertEqual(len(app.get("component_instance")), 1)  # image stays mounted
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertIn("m6v_step_2", keys(app))
        run(app)
        self.assertEqual(scroll_targets(app), [])

    def test_back_to_selection_scrolls_to_step_two_and_keeps_everything(self):
        app = at_step(self.data, 3, self.CORNERS)
        button(app, "m6v_back_selection").click()
        run(app)
        self.assertEqual(scroll_targets(app), [2])
        self.assertEqual(app.session_state["m6v_points"], self.CORNERS)
        for key in ("m6v_step_1", "m6v_step_2", "m6v_step_3"):
            self.assertIn(key, keys(app))

    def test_reset_selection_keeps_video_and_preview_and_closes_later_steps(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        analysis = app.session_state["m6v_analysis"]
        for _ in range(2):
            button(app, "m6v_reset_selection").click()
            run(app)
            self.assertEqual(scroll_targets(app), [2])
            self.assertEqual(app.session_state["m6v_points"], [])
            self.assertIsNone(app.session_state["m6v_results"])
            self.assertIs(app.session_state["m6v_analysis"], analysis)
            self.assertEqual(len(app.get("component_instance")), 1)
            self.assertEqual(len(app.get("file_uploader")), 1)
            self.assertIn("m6v_bytes", app.session_state)
            self.assertNotIn("m6v_step_3", keys(app))
            self.assertNotIn("m6v_step_4", keys(app))
            self.assertFalse(app.get("dialog") or app.toast)
        self.assertTrue(button(app, "m6v_continue").disabled)

    def test_tracking_only_on_click_then_shows_results_below_and_scrolls(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.assertFalse(button(app, "m6v_run").disabled)
        run(app)
        self.assertIsNone(app.session_state["m6v_results"])
        self.run_tracking(app)
        self.assertIsNotNone(app.session_state["m6v_results"])
        self.assertEqual(scroll_targets(app), [4])
        self.assertEqual(self.headers(app)[-1], "Step 4 — Your Tracking Results")
        for key in ("m6v_step_1", "m6v_step_2", "m6v_step_3", "m6v_step_4"):
            self.assertIn(key, keys(app))
        self.assertEqual(len(app.get("component_instance")), 1)
        self.assertEqual(len(app.get("video")), 1)
        self.assertEqual(len(app.get("vega_lite_chart")), 3)
        self.assertEqual(app.get("video")[0].proto.width_config.pixel_width, 320)  # same as preview width
        for key in ("m6v_dl_video", "m6v_dl_csv", "m6v_dl_json"):
            self.assertTrue(any(d.proto.id and key in d.proto.id for d in app.get("download_button")))
        result = app.session_state["m6v_results"]
        for _ in range(2):
            run(app)
            self.assertEqual(scroll_targets(app), [])
            self.assertIs(app.session_state["m6v_results"], result)

    def test_tracking_graphs_in_collapsed_expander_with_labeled_axes(self):
        import json
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        expanders = [e for e in app.expander if e.label == "View Tracking Graphs"]
        self.assertEqual(len(expanders), 1)
        self.assertFalse(expanders[0].proto.expanded)
        charts = [json.loads(c.proto.spec) for c in app.get("vega_lite_chart")]
        self.assertEqual(len(charts), 3)
        expected_y = [["Number of", "Tracked Features"], ["Mean Displacement", "per Step (px)"],
                      ["Mean Cumulative", "Displacement (px)"]]
        rows = {r["time_s"] for r in app.session_state["m6v_results"].per_frame}
        for spec, y_title in zip(charts, expected_y):
            layer = spec["layer"][-1]
            self.assertEqual(layer["encoding"]["x"]["axis"]["title"], "Time (seconds)")
            # the Y label is a chart-level title (axis titles are merged across layers and lost the line break):
            # a list of two short strings, rotated on the left, regular weight; the Y axis itself has no title
            self.assertIsNone(layer["encoding"]["y"]["axis"]["title"])
            title = spec["title"]
            self.assertEqual(title["text"], y_title)
            self.assertEqual((title["orient"], title["angle"], title["fontWeight"]), ("left", 270, "normal"))
            self.assertLess(max(len(line) for line in y_title) * 9, spec["height"])
            for axis_name in ("x", "y"):
                axis = layer["encoding"][axis_name]["axis"]
                self.assertEqual(axis["titleFontWeight"], "normal")
                self.assertEqual(axis["labelFontWeight"], "normal")
            self.assertEqual(spec["config"]["axis"]["titleFontWeight"], "normal")
            self.assertEqual(spec["config"]["title"]["fontWeight"], "normal")
            titles = [t["title"] for t in layer["encoding"]["tooltip"]]
            self.assertNotIn("time_s", titles)
            self.assertEqual(titles[0], "Time (seconds)")
            self.assertEqual(layer["encoding"]["tooltip"][0]["format"], ".2f")
        tracked = charts[0]["layer"][-1]
        self.assertEqual(tracked["encoding"]["y"]["axis"]["format"], "d")
        counts = [r["tracked"] for r in app.session_state["m6v_results"].per_frame]
        self.assertGreater(tracked["encoding"]["y"]["scale"]["domain"][1], max(counts))
        self.assertEqual(tracked["encoding"]["y"]["scale"]["domain"][0], 0)
        self.assertTrue(rows)

    def test_padded_domain_covers_data_and_handles_flat_series(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("m6page_domain", PAGE)
        source = PAGE.read_text()
        ns = {"pd": __import__("pandas")}
        start = source.index("def padded_domain")
        exec(source[start:source.index("def tracking_charts")], ns)
        low, high = ns["padded_domain"]([-2.0, 4.0, None])
        self.assertLess(low, -2.0)
        self.assertGreater(high, 4.0)
        low, high = ns["padded_domain"]([5.0, 5.0])
        self.assertLess(low, 5.0 + 1e-9)
        self.assertGreater(high, 5.0)
        self.assertEqual(ns["padded_domain"]([3, 9], floor=0)[0], 0)

    def test_parameter_change_invalidates_results_without_scrolling(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        app.number_input(key="m6v_p_corners").set_value(30)
        run(app)
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertNotIn("m6v_step_4", keys(app))
        self.assertEqual(scroll_targets(app), [])
        self.assertIn("m6v_step_3", keys(app))

    def test_results_navigation_is_spaced_and_styled_in_scope(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        step4 = next(el for el in app.main if getattr(el, "key", None) == "m6v_step_4")
        inner = [getattr(el, "key", None) or el.type for el in step4.children.values()]
        nav = next(el for el in step4.children.values() if getattr(el, "key", None) == "m6v_results_nav")
        self.assertEqual(inner[-1], "m6v_results_nav")  # after the downloads, last in Step 4
        self.assertFalse(button(app, "m6v_back_settings").disabled)
        self.assertFalse(button(app, "m6v_start_over").disabled)
        source = PAGE.read_text()
        css = source[source.index("VISITOR_RESULTS_NAV_CSS = "):source.index("def visitor_results")]
        self.assertIn("margin-top: 28px", css)
        self.assertIn("border: 2px solid #C9363E", css)
        self.assertNotIn("button {", css.replace(".st-key-m6v_results_nav button", ""))

    def test_back_to_settings_from_results_keeps_results(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        button(app, "m6v_back_settings").click()
        run(app)
        self.assertEqual(scroll_targets(app), [3])
        self.assertIsNotNone(app.session_state["m6v_results"])
        self.assertIn("m6v_step_4", keys(app))

    def test_settings_defaults_and_params_survive_navigation(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.assertEqual(len(app.expander), 2)  # upload instructions + advanced settings
        self.assertEqual(app.session_state["m6v_p_corners"], 50)
        self.assertEqual(app.session_state["m6v_p_quality"], 0.3)
        self.assertEqual(app.session_state["m6v_p_window"], 21)
        self.assertEqual(app.session_state["m6v_p_iter"], 30)
        app.number_input(key="m6v_p_corners").set_value(30)
        run(app)
        button(app, "m6v_back_selection").click()
        run(app)
        self.assertEqual(app.session_state["m6v_p_corners"], 30)

    def test_invalid_parameter_disables_run(self):
        app = at_step(self.data, 3, self.CORNERS)
        app.number_input(key="m6v_p_window").set_value(20)
        run(app)
        self.assertTrue(any("odd" in w.value for w in app.warning))
        self.assertTrue(button(app, "m6v_run").disabled)

    def test_unrelated_reruns_do_not_clear_or_reprocess_upload(self):
        app = at_step(self.data, 2)
        analysis = app.session_state["m6v_analysis"]
        with unittest.mock.patch("Modules.Module_6.Part_A.visitor_tracking.analyze_upload") as analyze:
            for _ in range(3):
                run(app)
            button(app, "m6v_reset_selection").click()
            run(app)
            analyze.assert_not_called()
        self.assertIs(app.session_state["m6v_analysis"], analysis)
        self.assertEqual(app.session_state["m6v_bytes"], self.data)
        self.assertEqual(app.get("file_uploader")[0].value.name, "clip.mp4")

    def test_removing_uploaded_file_clears_everything(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        app.get("file_uploader")[0].clear()
        run(app)
        for key in ("m6v_bytes", "m6v_analysis", "m6v_filename", "m6v_digest", "m6v_error", "m6v_params_saved"):
            self.assertNotIn(key, app.session_state)
        self.assertEqual(app.session_state["m6v_points"], [])
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertFalse(app.session_state["m6v_settings_open"])
        self.assertIsNone(app.get("file_uploader")[0].value)
        self.assertEqual(app.metric.values, [])
        self.assertEqual(len(app.get("video")), 0)
        self.assertEqual(len(app.get("component_instance")), 0)
        self.assertEqual(self.headers(app), ["Track Motion in Your Own Video", "Step 1 — Upload Your Video"])
        self.assertEqual(scroll_targets(app), [])

    def test_replacing_video_invalidates_roi_and_results(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        old_digest = app.session_state["m6v_digest"]
        app.get("file_uploader")[0].set_value(("other.mp4", self.other, "video/mp4"))
        run(app)
        self.assertEqual(app.session_state["m6v_filename"], "other.mp4")
        self.assertNotEqual(app.session_state["m6v_digest"], old_digest)
        self.assertEqual(app.session_state["m6v_points"], [])
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertNotIn("m6v_step_3", keys(app))
        self.assertNotIn("m6v_step_4", keys(app))
        self.assertIn("m6v_step_2", keys(app))
        self.assertEqual(scroll_targets(app), [2])
        self.assertTrue(button(app, "m6v_continue").disabled)

    def test_start_over_clears_experiment_and_scrolls_to_top(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        button(app, "m6v_start_over").click()
        run(app)
        self.assertEqual(scroll_targets(app), [1])
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertEqual(app.session_state["m6v_points"], [])
        for key in ("m6v_bytes", "m6v_analysis", "m6v_filename", "m6v_digest"):
            self.assertNotIn(key, app.session_state)
        self.assertIsNone(app.get("file_uploader")[0].value)
        self.assertNotIn("m6v_step_2", keys(app))

    def test_page_reopened_with_empty_uploader_drops_stale_video(self):
        app = at_step(self.data, 2)
        button(app, "navbtn_1").click()
        run(app)
        self.assertEqual(app.session_state["module6_section"], "a_select")
        app.session_state["module6_section"] = "a_visitor"
        run(app)
        self.assertNotIn("m6v_bytes", app.session_state)
        self.assertEqual(app.metric.values, [])
        self.assertNotIn("m6v_step_2", keys(app))

    def test_scroll_script_is_scoped_to_step_anchors(self):
        source = PAGE.read_text()
        js = source[source.index("VISITOR_SCROLL_JS = "):source.index("VISITOR_SCROLL_HOST_CSS")]
        self.assertEqual(js.count("querySelector"), 1)
        self.assertIn('".st-key-m6v_step_%d"', js)
        self.assertNotIn("window.scrollTo", js)
        # Part B has its own one-shot script, so count only Part A's emitter (visitor_scroll_script).
        emitter = source[source.index("def visitor_scroll_script"):source.index("def visitor_preview_scale")]
        self.assertEqual(emitter.count("unsafe_allow_javascript=True"), 1)

    def test_corrupt_video_shows_error_not_crash(self):
        app = visitor_app(b"not a video" * 200)
        self.assertTrue(any("could not be decoded" in e.value for e in app.error))
        self.assertNotIn("m6v_step_2", keys(app))
        self.assertEqual(scroll_targets(app), [])

    def test_top_and_bottom_back_return_to_part_a_selection(self):
        for key in ("navbtn_1", "m6_a_visitor_back_bottom"):
            app = uploaded_app(self.data)
            button(app, key).click()
            run(app)
            self.assertEqual(app.session_state["module6_section"], "a_select")

    def test_bottom_back_hidden_without_video_and_after_removal(self):
        app = visitor_app()
        self.assertNotIn("m6_a_visitor_back_bottom", keys(app))
        self.assertIn("navbtn_1", keys(app))
        app = uploaded_app(self.data)
        self.assertIn("m6_a_visitor_back_bottom", keys(app))
        app.get("file_uploader")[0].set_value(None)  # the × on the file chip
        run(app)
        self.assertNotIn("m6_a_visitor_back_bottom", keys(app))
        self.assertIn("navbtn_1", keys(app))


if __name__ == "__main__":
    unittest.main()
