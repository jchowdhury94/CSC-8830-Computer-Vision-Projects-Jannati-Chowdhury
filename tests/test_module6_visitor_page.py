"""Module 6 visitor page workflow tests (AppTest; session state seeded instead of a browser upload)."""
import dataclasses
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
        self.assertEqual([t.value for t in app.title], ["Optical Flow and Motion Tracking"])
        self.assertIn("MODULE 6 — PART A", [m.value for m in app.markdown])
        self.assertEqual(self.headers(app), ["Step 1 — Upload Your Video"])
        self.assertEqual([e.label for e in app.expander], ["Video Recording Instructions"])
        self.assertFalse(app.expander[0].proto.expanded)
        self.assertEqual(len(app.info), 0)
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
        self.assertEqual(self.headers(app), ["Step 1 — Upload Your Video", "Step 2 — Select the Object"])
        self.assertEqual(scroll_targets(app), [2])
        self.assertEqual(len(app.get("component_instance")), 1)
        self.assertIsNone(app.session_state["m6v_results"])  # tracking never starts automatically
        self.assertEqual([m.label for m in app.metric], ["Resolution", "Duration", "Frame Rate", "Frames"])
        self.assertEqual(app.metric[0].value, "320 × 240")
        for _ in range(2):  # unrelated reruns do not scroll
            run(app)
            self.assertEqual(scroll_targets(app), [])

    def test_render_order_is_vertical_and_complete(self):
        app = at_step(self.data, 3, self.CORNERS)
        order = keys(app)
        marks = ["navbtn_0", "m6v_step_1", "m6v_step_2", "m6v_step_3"]
        positions = [order.index(m) for m in marks]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.headers(app), ["Step 1 — Upload Your Video", "Step 2 — Select the Object",
                                             "Step 3 — Run Tracking"])

    def test_step_two_content_order(self):
        app = uploaded_app(self.data)
        step2 = next(el for el in app.main if getattr(el, "key", None) == "m6v_step_2")
        def walk(node):
            yield node
            for child in getattr(node, "children", {}).values():
                yield from walk(child)

        nodes = list(walk(step2))
        inner = [getattr(el, "key", None) for el in nodes]
        actions = next(el for el in nodes if getattr(el, "key", None) == "m6v_select_actions")
        row = [getattr(el, "key", None) for el in actions.children.values()]
        self.assertLess(inner.index("m6v_click"), inner.index("m6v_select_actions"))
        guide = next(e for e in app.expander if e.label == "How to Select the Object")
        self.assertFalse(guide.proto.expanded)
        self.assertEqual([m.value for m in guide.markdown], ["Keep the rectangle close to the object."])
        # preview box and buttons share the preview's width
        box = next(el for el in nodes if getattr(el, "key", None) == "m6v_preview_box")
        self.assertEqual(box.proto.width_config.pixel_width, 320)
        self.assertEqual(actions.proto.width_config.pixel_width, 320)
        self.assertEqual(row, ["m6v_reset_selection", "m6v_continue"])

    def test_one_instruction_per_selection_state_and_no_raw_coordinates(self):
        app = uploaded_app(self.data)
        states = (([], "Click the first corner."), ([(30, 90)], "Click the opposite corner."),
                  (self.CORNERS, "Object selected. Click Continue."))
        for points, expected in states:
            app.session_state["m6v_points"] = points
            run(app)
            shown = " ".join(m.value for m in app.markdown)
            self.assertIn(expected, shown)
            for _, other in states:
                if other != expected:
                    self.assertNotIn(other, shown)
        text = " ".join([*(m.value for m in app.markdown), *(c.value for c in app.caption)])
        self.assertNotIn("(30, 90, 60, 60)", text)
        self.assertNotIn("Selected rectangle", text)

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
        self.assertEqual(self.headers(app)[-1], "Step 4 — Tracking Results")
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
        expected_y = [["Features"], ["Displacement (px)"], ["Displacement (px)"]]
        rows = {r["time_s"] for r in app.session_state["m6v_results"].per_frame}
        for spec, y_title in zip(charts, expected_y):
            layer = spec["layer"][-1]
            self.assertEqual(layer["encoding"]["x"]["axis"]["title"], "Time (s)")
            self.assertEqual(layer["encoding"]["x"]["scale"]["domain"][0], 0)
            self.assertGreaterEqual(layer["encoding"]["x"]["scale"]["domain"][1], max(rows))
            self.assertNotIn("format", layer["encoding"]["x"]["axis"])
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

    def test_time_axis_starts_at_zero_with_clean_ticks(self):
        import math
        source = PAGE.read_text()
        ns = {"pd": __import__("pandas"), "math": math}
        exec(source[source.index("def time_axis_range"):source.index("def tracking_charts")], ns)
        for longest, expected in ((19.97, (20, 10)), (20.0, (20, 10)), (7.3, (8, 8)), (0.37, (0.4, 8)),
                                  (3.0, (3, 6)), (0.0, (1, 10))):
            with self.subTest(longest=longest):
                self.assertEqual(ns["time_axis_range"]([0.0, longest]), expected)

    def test_parameter_change_invalidates_results_without_scrolling(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        app.number_input(key="m6v_p_corners").set_value(30)
        run(app)
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertNotIn("m6v_step_4", keys(app))
        self.assertEqual(scroll_targets(app), [])
        self.assertIn("m6v_step_3", keys(app))

    def test_results_end_with_reset_all_and_have_no_back_buttons(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        step4 = next(el for el in app.main if getattr(el, "key", None) == "m6v_step_4")
        inner = [getattr(el, "key", None) or el.type for el in step4.children.values()]
        self.assertEqual(inner[-1], "m6v_results_nav")
        self.assertEqual(button(app, "m6v_reset_all").label, "Reset All")
        self.assertEqual([b.label for b in app.button if b.label.startswith("Back") or b.label == "Start Over"], [])

    def test_results_sections_cards_and_downloads(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        self.assertEqual([h.value for h in app.subheader],
                         ["Optical Flow Visualization", "Tracking Summary", "Mean Motion per Step",
                          "Mean Cumulative Displacement", "Download Results"])
        summary = app.session_state["m6v_results"].summary
        cards = [(m.label, m.value) for m in app.metric][4:]  # after the four video metadata metrics
        self.assertEqual(cards, [
            ("Initial Features", str(summary["initial_features"])), ("Final Features", str(summary["final_features"])),
            ("Retention", f"{summary['feature_retention_percent']:.1f}%"),
            ("Horizontal (u)", f"{summary['mean_u_per_step']:+.3f} px"),
            ("Vertical (v)", f"{summary['mean_v_per_step']:+.3f} px"),
            ("Magnitude", f"{summary['mean_magnitude_per_step']:.3f} px"), ("Direction", summary["motion_direction"]),
            ("Horizontal (u)", f"{summary['mean_cumulative_u_final']:+.1f} px"),
            ("Vertical (v)", f"{summary['mean_cumulative_v_final']:+.1f} px")])
        helps = {m.label: m.proto.help for m in app.metric}
        self.assertIn("individual displacement lengths", helps["Magnitude"])
        self.assertIn("mean (u, v) displacement vector", helps["Direction"])
        self.assertIn("starting positions", helps["Horizontal (u)"])
        self.assertIn("camera motion or both", app.subheader[2].proto.help)
        captions = [c.value for c in app.caption]
        self.assertIn("Yellow dots: tracked features. Arrows: motion direction and displacement magnified 30× "
                      "(maximum 90 px).", captions)
        step = app.session_state["m6v_analysis"].plan.step_seconds
        self.assertIn("Original-frame pixels, without magnification. Positive u: rightward; positive v: downward. "
                      f"Step interval: {step:.3f} s.", captions)
        self.assertEqual([d.proto.label for d in app.get("download_button")],
                         ["Video (MP4)", "Tracking (CSV)", "Summary (JSON)"])
        source = PAGE.read_text()
        for name in ('"tracked_video.mp4"', '"tracking.csv"', '"tracking_summary.json"'):
            self.assertIn(name, source)

    def test_settings_expander_collapsed_with_help_and_defaults(self):
        app = at_step(self.data, 3, self.CORNERS)
        expander = next(e for e in app.expander if e.label == "Advanced Settings (Optional)")
        self.assertFalse(expander.proto.expanded)
        self.assertEqual([(n.label, n.value) for n in expander.number_input], [
            ("Maximum corners", 50), ("Quality level", 0.3), ("Minimum distance (px)", 7), ("Block size", 7),
            ("Lucas–Kanade window", 21), ("Pyramid maximum level", 3), ("Termination iterations", 30),
            ("Termination epsilon", 0.01)])
        self.assertTrue(all(n.proto.help for n in expander.number_input))
        self.assertEqual([(n.min, n.max, n.step) for n in expander.number_input], [
            (1, 200, 1), (0.01, 0.95, 0.05), (1, 50, 1), (3, 31, 2), (5, 51, 2), (0, 5, 1), (1, 100, 1),
            (0.001, 1.0, 0.005)])
        self.assertIn("not replaced", " ".join(c.value for c in expander.caption))
        self.assertEqual([b.key for b in expander.button], ["m6v_reset_params"])
        self.assertNotIn("Settings changed. Run tracking again.", [i.value for i in app.info])

    def test_parameter_change_shows_notice_until_next_run(self):
        app = at_step(self.data, 3, self.CORNERS)
        app.number_input(key="m6v_p_corners").set_value(40)  # before the first run: nothing is outdated
        run(app)
        self.assertEqual(len(app.info), 0)
        self.run_tracking(app)
        app.number_input(key="m6v_p_corners").set_value(30)
        run(app)
        run(app)
        self.assertEqual([i.value for i in app.info], ["Settings changed. Run tracking again."])
        self.run_tracking(app)
        self.assertEqual(len(app.info), 0)
        self.assertIn("m6v_step_4", keys(app))

    def test_reset_to_defaults_restores_all_and_invalidates_only_on_change(self):
        app = at_step(self.data, 3, self.CORNERS)
        self.run_tracking(app)
        result = app.session_state["m6v_results"]
        button(app, "m6v_reset_params").click()
        run(app)
        self.assertIs(app.session_state["m6v_results"], result)  # already defaults
        self.assertEqual(len(app.info), 0)
        changed = {"m6v_p_corners": 30, "m6v_p_quality": 0.2, "m6v_p_distance": 5, "m6v_p_block": 9,
                   "m6v_p_window": 15, "m6v_p_level": 2, "m6v_p_iter": 20, "m6v_p_eps": 0.02}
        for key, value in changed.items():
            app.number_input(key=key).set_value(value)
        run(app)
        self.run_tracking(app)
        button(app, "m6v_reset_params").click()
        run(app)
        self.assertEqual([app.session_state[k] for k in changed], [50, 0.3, 7, 7, 21, 3, 30, 0.01])
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertEqual([i.value for i in app.info], ["Settings changed. Run tracking again."])
        self.assertEqual(app.session_state["m6v_points"], self.CORNERS)

    def assert_error_persists_then_retry(self, side_effect, expected):
        app = at_step(self.data, 3, self.CORNERS)
        target = "Modules.Module_6.Part_A.visitor_tracking.run_tracking_job"
        with unittest.mock.patch(target, side_effect=side_effect):
            self.run_tracking(app)
        for _ in range(3):  # the error survives the rerun after the failure and later unrelated reruns
            self.assertEqual(len(app.error), 1)
            self.assertIn(expected, app.error[0].value)
            self.assertIsNone(app.session_state["m6v_results"])
            self.assertEqual(len(app.info), 0)
            run(app)
        self.assertFalse(button(app, "m6v_run").disabled)
        self.run_tracking(app)
        self.assertEqual(len(app.error), 0)
        self.assertIsNotNone(app.session_state["m6v_results"])

    def test_tracking_errors_persist_and_allow_retry(self):
        from Modules.Module_6.Part_A import visitor_tracking as vt
        cases = ((vt.NoFeaturesError("x"), "No trackable features"), (vt.BusyError("x"), "server is busy"),
                 (vt.EncodingError("x"), "could not be created"),
                 (vt.VideoError("The selected rectangle is outside the video frame."), "outside the video frame"),
                 (RuntimeError("internal detail"), "Tracking could not complete"))
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                self.assert_error_persists_then_retry(error, expected)

    def test_real_no_features_error_is_visible(self):
        app = at_step(self.data, 3, [(0, 0), (12, 12)])  # a featureless corner of the synthetic scene
        app.number_input(key="m6v_p_quality").set_value(0.95)
        run(app)
        with unittest.mock.patch("cv2.goodFeaturesToTrack", return_value=None):
            self.run_tracking(app)
        run(app)
        self.assertIn("No trackable features", app.error[0].value)

    def test_relevant_changes_clear_a_run_error(self):
        target = "Modules.Module_6.Part_A.visitor_tracking.run_tracking_job"
        actions = (lambda a: a.number_input(key="m6v_p_corners").set_value(30),
                   lambda a: button(a, "m6v_reset_params").click(),
                   lambda a: button(a, "m6v_reset_selection").click(),
                   lambda a: a.get("file_uploader")[0].set_value(("other.mp4", self.other, "video/mp4")))
        for number, action in enumerate(actions):
            with self.subTest(action=number):
                app = at_step(self.data, 3, self.CORNERS)
                with unittest.mock.patch(target, side_effect=RuntimeError("x")):
                    self.run_tracking(app)
                self.assertEqual(len(app.error), 1)
                action(app)
                run(app)
                self.assertEqual(len(app.error), 0)
                self.assertNotIn("m6v_run_error", app.session_state)

    def test_unexpected_analysis_failure_shows_friendly_message(self):
        target = "Modules.Module_6.Part_A.visitor_tracking.analyze_upload"
        with unittest.mock.patch(target, side_effect=RuntimeError("internal detail")):
            app = uploaded_app(self.data)
        run(app)
        self.assertEqual(len(app.error), 1)
        self.assertIn("could not be analyzed", app.error[0].value)
        self.assertNotIn("internal detail", app.error[0].value)
        self.assertNotIn("m6v_step_2", keys(app))

    def test_feature_loss_warning_uses_the_overlay_frame_number(self):
        from Modules.Module_6.Part_A import visitor_tracking as vt
        for stride, expected in ((1, "at frame 3."), (3, "at source frame 7.")):
            with self.subTest(stride=stride):
                app = at_step(self.data, 3, self.CORNERS)
                analysis = app.session_state["m6v_analysis"]
                app.session_state["m6v_analysis"] = dataclasses.replace(
                    analysis, plan=dataclasses.replace(analysis.plan, stride=stride, frames_to_process=12 // stride))
                real, calls, labels = vt.cv2.calcOpticalFlowPyrLK, [], []

                def lose_on_second_step(*args, **kwargs):
                    points, status, error = real(*args, **kwargs)
                    calls.append(1)
                    return points, status * 0 if len(calls) == 2 else status, error

                real_annotate = vt.annotate

                def spy(frame, number, points, label="Frame", **style):
                    labels.append((label, number))
                    real_annotate(frame, number, points, label, **style)

                with unittest.mock.patch.object(vt.cv2, "calcOpticalFlowPyrLK", lose_on_second_step), \
                        unittest.mock.patch.object(vt, "annotate", spy):
                    self.run_tracking(app)
                result = app.session_state["m6v_results"]
                self.assertEqual(result.summary["all_features_lost_at_frame"], 3)  # data keeps processed numbering
                self.assertIn(f"All features were lost {expected}", app.warning[0].value)
                name = "Frame" if stride == 1 else "Source frame"
                self.assertEqual(labels[2], (name, 2 * stride + 1))

    def test_reset_selection_preserves_settings(self):
        app = at_step(self.data, 3, self.CORNERS)
        app.number_input(key="m6v_p_corners").set_value(30)
        run(app)
        button(app, "m6v_reset_selection").click()
        run(app)
        self.assertNotIn("m6v_step_3", keys(app))
        run(app)
        app.session_state["m6v_points"] = self.CORNERS
        run(app)
        button(app, "m6v_continue").click()
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
        self.assertEqual(self.headers(app), ["Step 1 — Upload Your Video"])
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

    def test_reset_all_restores_the_initial_state(self):
        app = at_step(self.data, 3, self.CORNERS)
        app.number_input(key="m6v_p_corners").set_value(30)
        run(app)
        self.run_tracking(app)
        button(app, "m6v_reset_all").click()
        run(app)
        self.assertEqual(scroll_targets(app), [1])
        self.assertIsNone(app.session_state["m6v_results"])
        self.assertEqual(app.session_state["m6v_points"], [])
        for key in ("m6v_bytes", "m6v_analysis", "m6v_filename", "m6v_digest", "m6v_params_saved", "m6v_p_corners",
                    "m6v_stale", "m6v_run_error", "m6v_error"):
            self.assertNotIn(key, app.session_state)
        self.assertIsNone(app.get("file_uploader")[0].value)
        self.assertEqual(self.headers(app), ["Step 1 — Upload Your Video"])
        app.get("file_uploader")[0].set_value(("clip.mp4", self.data, "video/mp4"))
        run(app)
        app.session_state["m6v_points"] = self.CORNERS
        run(app)
        button(app, "m6v_continue").click()
        run(app)
        self.assertEqual(app.session_state["m6v_p_corners"], 50)

    def test_page_reopened_with_empty_uploader_drops_stale_video(self):
        app = at_step(self.data, 2)
        button(app, "navbtn_0").click()
        run(app)
        self.assertEqual(app.session_state["module6_section"], "home")
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

    def test_top_back_returns_home_and_there_is_no_bottom_back(self):
        app = uploaded_app(self.data)
        self.assertEqual([b.label for b in app.button if b.label == "Back"], [])
        button(app, "nav_arrow_button").click()
        run(app)
        self.assertEqual(app.session_state["module6_section"], "home")


if __name__ == "__main__":
    unittest.main()
