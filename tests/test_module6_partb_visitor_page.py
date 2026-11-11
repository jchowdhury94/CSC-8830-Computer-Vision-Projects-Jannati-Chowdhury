"""Module 6 Part B visitor page workflow tests (AppTest with synthetic photos; no browser)."""
import re
import sys
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_module6_partb_backend import cube_files, png_bytes, texture  # noqa: E402
from Modules.Module_6.Part_B import visitor_reconstruction as vr  # noqa: E402

PAGE = ROOT / "Pages" / "app_module6.py"
TITLE = "Reconstruct an Object from Your Own Images"
STEP = {1: "Step 1 — Upload Your Images", 2: "Step 2 — Find Matching Points", 3: "Step 3 — Set Scale (Optional)",
        4: "Step 4 — Reconstruct Your Object", 5: "Step 5 — View Results"}


def run(app):
    app.run()
    assert not app.exception, app.exception


def button(app, key):
    return next(b for b in app.button if b.key == key)


def click(app, key):
    button(app, key).click()
    run(app)


def headers(app):
    return [h.value for h in app.header]


def scroll_targets(app):
    found = []
    for el in app.get("html"):
        found += [int(n) for n in re.findall(r"\.st-key-m6b_step_(\d)\"", el.proto.body)]
    return found


def keys(app):
    return [getattr(el, "key", None) for el in app.main]


def triples(files):
    return [(name, data, "image/png") for data, name in files]


def empty_app():
    app = AppTest.from_file(str(PAGE), default_timeout=120)
    app.session_state["module6_section"] = "b_visitor"
    run(app)
    return app


def uploaded_app(files):
    app = empty_app()
    app.get("file_uploader")[0].set_value(triples(files))
    run(app)
    return app


def matched_app(files):
    app = uploaded_app(files)
    click(app, "m6b_find")
    return app


def reconstructed_app(files, scale=False):
    app = matched_app(files)
    if scale:
        app.number_input[0].set_value(5.0)
        run(app)
        click(app, "m6b_use_scale")
    else:
        click(app, "m6b_skip_scale")
    click(app, "m6b_run")
    return app


class VisitorPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = cube_files()

    # ------------------------------------------------------------------ step 1
    def test_empty_page_shows_only_step_one(self):
        app = empty_app()
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], [])
        self.assertEqual(len(app.get("file_uploader")), 1)
        for step in (2, 3, 4, 5):
            self.assertNotIn(f"m6b_step_{step}", keys(app))
        self.assertEqual(scroll_targets(app), [])
        self.assertFalse(app.get("dialog") or app.toast or app.info or app.warning or app.error)
        self.assertEqual(app.metric.values, [])

    def test_uploader_configuration(self):
        proto = empty_app().get("file_uploader")[0].proto
        self.assertTrue(proto.multiple_files)
        self.assertEqual(proto.max_upload_size_mb, 25)
        self.assertEqual(sorted(proto.type), sorted(vr.ALLOWED_EXTENSIONS))
        self.assertIn(".heic", proto.type)
        self.assertEqual((vr.MIN_VIEWS, vr.MAX_VIEWS, vr.RECOMMENDED_VIEWS), (3, 8, 4))
        self.assertEqual((vr.MAX_FILE_BYTES, vr.MAX_TOTAL_BYTES, vr.MAX_PIXELS), (25 << 20, 100 << 20, 64_000_000))

    def test_instructions_are_in_an_expander(self):
        app = empty_app()
        expander = next(e for e in app.expander if e.label == "Capture and upload instructions")
        text = " ".join(m.value for m in expander.markdown)
        for phrase in ("4 is recommended", "order", "overlap", "must stay still",
                       "texture", "not flat", "25 MB", "100 MB", "64 megapixels", "HEIC"):
            self.assertIn(phrase, text)
        self.assertFalse(any("must stay still" in m.value for m in app.markdown if m not in expander.markdown))

    def test_upload_shows_thumbnails_and_details_and_advances(self):
        app = uploaded_app(self.files)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2]])
        self.assertEqual(len(app.get("image")), 4)
        captions = " ".join(c.value for c in app.caption)
        for number, (data, name) in enumerate(self.files, 1):
            self.assertIn(f"{number}. {name}", captions)
        self.assertIn("900 × 700 px", captions)
        self.assertRegex(captions, r"\d+ KB")
        self.assertIn("4 photographs", captions)
        self.assertEqual(scroll_targets(app), [2])
        self.assertEqual([b.key for b in app.button if b.label == "Back"],
                         ["m6_b_visitor_back_bottom"])
        self.assertEqual(app.session_state["module6_section"], "b_visitor")

    def test_too_few_photos_wait_without_error(self):
        app = uploaded_app(self.files[:2])
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertEqual(len(app.get("image")), 2)
        self.assertIn("add at least 3", " ".join(c.value for c in app.caption))
        self.assertEqual([b.key for b in app.button if not (b.key or "").startswith("nav")], [])  # no bottom Back yet
        self.assertEqual(len(app.error), 0)

    def test_three_photos_are_accepted(self):
        self.assertIn(STEP[2], headers(uploaded_app(self.files[:3])))

    def test_too_many_photos(self):
        extra = [(png_bytes(texture(400, i)), f"extra{i}.png") for i in range(6)]
        app = uploaded_app(self.files + extra[:5])
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertIn("at most 8", app.error[0].value)

    def test_invalid_and_unsupported_files(self):
        app = uploaded_app(self.files[:3] + [(b"not an image", "broken.jpg")])
        self.assertIn("broken.jpg", app.error[0].value)
        self.assertIn(STEP[1], headers(app))
        self.assertNotIn(STEP[2], headers(app))
        import io
        from PIL import Image
        gif = io.BytesIO()
        Image.new("RGB", (400, 400)).save(gif, "GIF")
        app = uploaded_app(self.files[:3] + [(gif.getvalue(), "disguised.jpg")])  # the native uploader filters extensions
        self.assertIn("disguised.jpg", app.error[0].value)
        self.assertIn("not a JPEG, PNG or HEIC", app.error[0].value)

    def test_error_clears_when_bad_file_is_removed(self):
        app = uploaded_app(self.files[:3] + [(b"broken", "broken.jpg")])
        self.assertEqual(len(app.error), 1)
        app.get("file_uploader")[0].set_value(triples(self.files[:3]))
        run(app)
        self.assertEqual(len(app.error), 0)
        self.assertIn(STEP[2], headers(app))

    def test_oversized_file_is_rejected_without_decoding(self):
        big = (b"\0" * (vr.MAX_FILE_BYTES + 1), "huge.png")
        app = uploaded_app(self.files[:3] + [big])
        self.assertIn("larger than 25 MB", app.error[0].value)

    # ------------------------------------------------------------------ invalidation
    def test_removing_or_replacing_images_invalidates_everything(self):
        app = reconstructed_app(self.files)
        self.assertIn(STEP[5], headers(app))
        app.get("file_uploader")[0].set_value(triples(self.files[:3]))  # one image removed
        run(app)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2]])
        self.assertIsNone(app.session_state["m6b_match"] if "m6b_match" in app.session_state else None)
        self.assertEqual(app.metric.values, [])
        self.assertFalse(app.session_state["m6b_recon_open"])
        self.assertIsNone(app.session_state["m6b_scale"])
        app = reconstructed_app(self.files)
        app.get("file_uploader")[0].set_value(triples(self.files[1:] + [self.files[0]]))  # replaced / reordered
        run(app)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2]])
        for key in ("m6b_match", "m6b_result", "m6b_result_sig"):
            self.assertNotIn(key, app.session_state)

    def test_removing_all_images_returns_to_empty_page(self):
        app = matched_app(self.files)
        app.get("file_uploader")[0].set_value(None)
        run(app)
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertEqual([b.key for b in app.button if not (b.key or "").startswith("nav")], [])
        self.assertEqual(len(app.get("image")), 0)

    def test_leaving_and_returning_drops_stale_upload(self):
        app = matched_app(self.files)
        click(app, "navbtn_1")
        self.assertEqual(app.session_state["module6_section"], "b_select")
        click(app, "m6_b_visitor")
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertNotIn("m6b_match", app.session_state)

    # ------------------------------------------------------------------ step 2
    def test_matching_runs_only_on_click(self):
        with unittest.mock.patch.object(vr, "find_matches", wraps=vr.find_matches) as spy:
            app = uploaded_app(self.files)
            run(app)
            self.assertEqual(spy.call_count, 0)
            self.assertEqual(app.metric.values, [])
            click(app, "m6b_find")
            self.assertEqual(spy.call_count, 1)
            run(app)  # an unrelated rerun
            self.assertEqual(spy.call_count, 1)

    def test_matching_results_are_displayed(self):
        app = matched_app(self.files)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2], STEP[3]])
        self.assertEqual([m.label for m in app.metric], ["Photos", "Points in every photo", "Consistent points kept"])
        match = app.session_state["m6b_match"]
        self.assertEqual(app.metric[0].value, "4")
        self.assertEqual(app.metric[1].value, str(match.n_tracks_found))
        self.assertEqual(app.metric[2].value, str(match.n_tracks))
        images = app.get("image")
        self.assertEqual(len(images), 4 + 1 + 2 + 1)  # thumbnails, first overlay, two overlays in the expander, markers
        self.assertEqual(scroll_targets(app), [3])
        self.assertTrue(any(e.label == "Match statistics per photo pair" for e in app.expander))
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.dataframe[0].value), 3)

    def test_matching_failure_is_explained_inline(self):
        files = self.files[:3] + [(png_bytes(texture(700, 77)), "other.png")]
        app = matched_app(files)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2]])
        self.assertIn("Photos 3 and 4", app.error[0].value)
        self.assertIn("overlap strongly", " ".join(c.value for c in app.caption))
        self.assertEqual(app.metric.values, [])
        self.assertNotIn("m6b_match", app.session_state)
        self.assertEqual(scroll_targets(app), [])

    def test_unexpected_matching_error_is_generic(self):
        with unittest.mock.patch.object(vr, "find_matches", side_effect=ValueError("boom")):
            app = matched_app(self.files)
        self.assertEqual(app.error[0].value, "Matching could not complete. Please try different photos.")

    def test_busy_server_message(self):
        with unittest.mock.patch.object(vr, "find_matches", side_effect=vr.BusyError("The server is busy")):
            app = matched_app(self.files)
        self.assertIn("busy", app.error[0].value)

    def test_find_button_visible_before_matching(self):
        app = uploaded_app(self.files)
        self.assertEqual([b.label for b in app.button if b.key in ("m6b_find", "m6b_start_over_match")],
                         ["Find Matching Points", "Start Over"])
        self.assertEqual(button(app, "m6b_find").proto.type, "primary")
        self.assertFalse(button(app, "m6b_find").disabled)
        self.assertEqual(button(app, "m6b_start_over_match").proto.type, "secondary")

    def test_click_starts_matching_once_and_removes_button(self):
        app = uploaded_app(self.files)
        button(app, "m6b_find").click()
        with unittest.mock.patch.object(vr, "find_matches", wraps=vr.find_matches) as matcher:
            app.run()
        self.assertEqual(matcher.call_count, 1)
        self.assertFalse(app.session_state["m6b_matching"])
        self.assertNotIn("m6b_find", [b.key for b in app.button])

    def test_matching_flag_is_cleared_and_never_auto_retries(self):
        # While the flag is set the button is rendered with disabled=True (see b_stage_matching).
        app = uploaded_app(self.files)
        app.session_state["m6b_matching"] = True
        with unittest.mock.patch.object(vr, "find_matches", side_effect=vr.ReconstructionError("x")) as matcher:
            app.run()
        self.assertEqual(matcher.call_count, 1)
        self.assertFalse(app.session_state["m6b_matching"])  # never auto-retries

    def test_find_button_hidden_after_success_start_over_stays(self):
        app = matched_app(self.files)
        self.assertNotIn("m6b_find", [b.key for b in app.button])
        self.assertIn("m6b_start_over_match", [b.key for b in app.button])
        self.assertEqual(len(app.metric), 3)
        self.assertEqual(len(app.dataframe), 1)
        run(app)  # an unrelated rerun neither rematches nor brings the button back
        self.assertNotIn("m6b_find", [b.key for b in app.button])

    def test_find_button_stays_after_failure_and_retries(self):
        with unittest.mock.patch.object(vr, "find_matches", side_effect=vr.ReconstructionError("nope")):
            app = matched_app(self.files)
        self.assertFalse(button(app, "m6b_find").disabled)
        with unittest.mock.patch.object(vr, "find_matches", wraps=vr.find_matches) as spy:
            click(app, "m6b_find")
            self.assertEqual(spy.call_count, 1)
        self.assertNotIn("m6b_find", [b.key for b in app.button])

    def test_new_images_allow_matching_again(self):
        app = matched_app(self.files)
        app.get("file_uploader")[0].set_value(triples(self.files[:3]))
        run(app)
        self.assertNotIn("m6b_match", app.session_state)
        self.assertFalse(button(app, "m6b_find").disabled)

    def test_start_over_from_step_two_resets_everything(self):
        app = matched_app(self.files)
        generation = app.session_state["m6b_upload_generation"]
        click(app, "m6b_start_over_match")
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertEqual(app.session_state["m6b_upload_generation"], generation + 1)
        self.assertEqual(scroll_targets(app), [1])
        self.assertEqual(app.get("file_uploader")[0].value, [])
        self.assertEqual(len(app.get("image")), 0)
        for key in ("m6b_match", "m6b_files", "m6b_infos", "m6b_matching", "m6b_error", "m6b_match_error"):
            self.assertNotIn(key, app.session_state)
        self.assertIsNone(app.session_state["m6b_scale"])
        self.assertFalse(app.session_state["m6b_recon_open"])
        run(app)
        self.assertEqual(scroll_targets(app), [])  # scrolls only once
        app.get("file_uploader")[0].set_value(triples(self.files))
        run(app)
        self.assertFalse(button(app, "m6b_find").disabled)

    def test_rematching_is_not_offered_after_success(self):
        app = reconstructed_app(self.files)
        self.assertNotIn("m6b_find", [b.key for b in app.button])
        self.assertIn(STEP[5], headers(app))

    # ------------------------------------------------------------------ step 3
    def test_scale_step_controls(self):
        app = matched_app(self.files)
        self.assertEqual([s.label for s in app.selectbox], ["Point A", "Point B", "Unit"])
        self.assertEqual(app.selectbox[2].options, list(vr.UNIT_OPTIONS))
        candidates = [i + 1 for i in app.session_state["m6b_match"].marker_candidates]
        self.assertEqual(app.selectbox[0].options, [str(c) for c in candidates])
        self.assertTrue(button(app, "m6b_use_scale").proto.disabled)  # no distance yet
        self.assertFalse(button(app, "m6b_skip_scale").proto.disabled)
        text = " ".join(m.value for m in app.markdown)
        for phrase in ("only changes", "does not make the reconstruction more accurate"):
            self.assertIn(phrase, text)

    def test_selection_redraws_marker_image_from_match_data(self):
        app = matched_app(self.files)
        match = app.session_state["m6b_match"]
        before = list(match.tracks[0].flatten())
        app.selectbox[1].select(app.selectbox[1].options[1])
        run(app)
        a = int(app.selectbox[0].value) - 1
        b = int(app.selectbox[1].value) - 1
        expected = vr.render_point_markers(None, match.tracks[0], match.marker_candidates, selected=(a, b),
                                           base=match.marker_base)
        self.assertNotEqual(expected, vr.render_point_markers(None, match.tracks[0], match.marker_candidates,
                                                              selected=(a, match.marker_candidates[-1]),
                                                              base=match.marker_base))
        self.assertEqual(list(app.session_state["m6b_match"].tracks[0].flatten()), before)

    def stale_copy(self, match, drop_attribute=True):
        import dataclasses
        import types
        fields = {f.name: getattr(match, f.name) for f in dataclasses.fields(match) if f.name != "marker_base"}
        old = types.SimpleNamespace(**fields)  # an instance of an older class without marker_base
        if not drop_attribute:
            old.marker_base = None
        return old

    def test_stale_match_without_marker_base_is_invalidated_not_a_crash(self):
        for drop_attribute in (True, False):
            app = reconstructed_app(self.files)
            app.session_state["m6b_match"] = self.stale_copy(app.session_state["m6b_match"], drop_attribute)
            run(app)  # no exception
            self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2]])
            self.assertEqual(app.error[0].value, "Your earlier matching results are out of date. "
                                                 "Please run Find Matching Points again.")
            self.assertNotIn("m6b_match", app.session_state)
            self.assertNotIn("m6b_result", app.session_state)
            self.assertIsNone(app.session_state["m6b_scale"])
            self.assertFalse(app.session_state["m6b_recon_open"])
            self.assertEqual(len(app.session_state["m6b_infos"]), 4)  # uploads are kept
            self.assertEqual(len(app.get("image")), 4)  # thumbnails only
            with unittest.mock.patch.object(vr, "find_matches", wraps=vr.find_matches) as spy:
                run(app)
                run(app)
                self.assertEqual(spy.call_count, 0)  # no automatic rematching or rerun loop
                click(app, "m6b_find")
                self.assertEqual(spy.call_count, 1)
            self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2], STEP[3]])  # fresh result renders step 3
            self.assertEqual(len(app.error), 0)

    def test_stale_module_and_stale_match_are_rejected_before_step_three(self):
        """The server still holds the old module (class without marker_base) and an old match in session state."""
        import dataclasses
        app = reconstructed_app(self.files)
        app.session_state["m6b_match"] = self.stale_copy(app.session_state["m6b_match"])
        old_class = dataclasses.make_dataclass(
            "MatchResult", [(f.name, object) for f in dataclasses.fields(vr.MatchResult) if f.name != "marker_base"])
        # reload is stubbed so the old class stays in place: a class-based check alone would accept the old match
        with unittest.mock.patch.object(vr, "MatchResult", old_class), \
                unittest.mock.patch("importlib.reload", side_effect=lambda module: module) as reload:
            run(app)  # must not raise AttributeError
        self.assertGreaterEqual(reload.call_count, 1)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2]])
        self.assertIn("out of date", app.error[0].value)
        self.assertNotIn("m6b_match", app.session_state)
        self.assertNotIn("m6b_result", app.session_state)
        self.assertEqual(len(app.session_state["m6b_infos"]), 4)
        self.assertFalse(button(app, "m6b_find").disabled)
        click(app, "m6b_find")
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2], STEP[3]])
        self.assertIsNotNone(app.session_state["m6b_match"].marker_base)

    def test_stale_module_is_reloaded_so_fresh_matches_are_complete(self):
        import dataclasses
        app = matched_app(self.files)
        old_class = dataclasses.make_dataclass(
            "MatchResult", [(f.name, object) for f in dataclasses.fields(vr.MatchResult) if f.name != "marker_base"])
        current = vr.MatchResult
        with unittest.mock.patch.object(vr, "MatchResult", old_class):
            run(app)  # real reload restores the module's real class
        self.assertTrue(hasattr(vr.MatchResult, "marker_base"))
        self.assertEqual(len(app.error), 0)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2], STEP[3]])
        self.assertIsNot(current, None)

    def test_fresh_match_has_marker_base(self):
        app = matched_app(self.files)
        self.assertIsNotNone(app.session_state["m6b_match"].marker_base)
        self.assertEqual(len(app.error), 0)

    def test_same_point_twice_is_rejected(self):
        app = matched_app(self.files)
        app.number_input[0].set_value(5.0)
        app.selectbox[1].set_value(app.selectbox[0].value)
        run(app)
        self.assertTrue(button(app, "m6b_use_scale").proto.disabled)
        self.assertIn("Choose two different points.", [c.value for c in app.caption])

    def test_skip_scaling_continues_in_arbitrary_units(self):
        app = matched_app(self.files)
        click(app, "m6b_skip_scale")
        self.assertIsNone(app.session_state["m6b_scale"])
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2], STEP[3], STEP[4]])
        self.assertEqual(scroll_targets(app), [4])
        self.assertIn("arbitrary reconstruction units", " ".join(m.value for m in app.markdown))
        self.assertNotIn("m6b_result", app.session_state)  # nothing runs automatically

    def test_use_scale_stores_choice(self):
        app = matched_app(self.files)
        app.number_input[0].set_value(5.0)
        app.selectbox[2].set_value("mm")
        run(app)
        click(app, "m6b_use_scale")
        scale = app.session_state["m6b_scale"]
        candidates = app.session_state["m6b_match"].marker_candidates
        self.assertEqual(scale, dict(a=candidates[0], b=candidates[-1], distance=5.0, unit="mm"))
        self.assertIn(f"Points {candidates[0] + 1} and {candidates[-1] + 1} are 5 mm apart",
                      " ".join(m.value for m in app.markdown))

    # ------------------------------------------------------------------ step 4 / 5
    def test_reconstruction_runs_only_on_click(self):
        with unittest.mock.patch.object(vr, "reconstruct_matches", wraps=vr.reconstruct_matches) as spy:
            app = matched_app(self.files)
            click(app, "m6b_skip_scale")
            self.assertEqual(spy.call_count, 0)
            run(app)
            self.assertEqual(spy.call_count, 0)
            click(app, "m6b_run")
            self.assertEqual(spy.call_count, 1)
            run(app)
            app.selectbox[0].set_value(app.selectbox[0].options[1])  # unrelated widget rerun
            run(app)
            self.assertEqual(spy.call_count, 1)
            self.assertIn(STEP[5], headers(app))

    def test_results_arbitrary_units(self):
        app = reconstructed_app(self.files)
        self.assertEqual(headers(app), [TITLE, STEP[1], STEP[2], STEP[3], STEP[4], STEP[5]])
        self.assertEqual(scroll_targets(app), [5])
        self.assertEqual([m.label for m in app.metric][-4:], ["Points reconstructed", "Reprojection error (RMS)",
                                                              "Rank-3 energy", "Planarity ratio"])
        result = app.session_state["m6b_result"]
        self.assertEqual(app.metric[-4].value, str(len(result.points)))
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("Coordinate units: arbitrary reconstruction units", captions)
        self.assertIn("not centimeters or millimeters", captions)
        self.assertIn("Coordinate units: arbitrary reconstruction units.", [c.value for c in app.caption])
        self.assertEqual(len(app.get("plotly_chart")), 4)  # 3D cloud, top view, side view, singular values
        self.assertEqual(len(app.get("download_button")), 3)
        self.assertEqual([e.label for e in app.expander][-2:], ["Reconstruction diagnostics", "Reconstructed 3D points"])
        points_table = app.dataframe[-1].value
        self.assertEqual(list(points_table.columns)[:4], ["Point", "X (arbitrary reconstruction units)",
                                                          "Y (arbitrary reconstruction units)",
                                                          "Z (arbitrary reconstruction units)"])
        self.assertEqual(len(points_table), len(result.points))
        self.assertFalse(app.get("dialog") or app.toast or app.warning)

    def test_results_plot_data_and_units(self):
        app = reconstructed_app(self.files, scale=True)
        result = app.session_state["m6b_result"]
        self.assertEqual(result.units, "cm")
        self.assertEqual(vr.plotly_payload(result)["axis_titles"]["x"], "X (cm)")
        self.assertIn("Coordinate units: cm", " ".join(c.value for c in app.caption))
        self.assertNotIn("arbitrary", " ".join(c.value for c in app.caption if c.value.startswith("Coordinate units")))
        self.assertIn("X (cm)", list(app.dataframe[-1].value.columns))
        spec = app.get("plotly_chart")[0].proto.spec
        self.assertIn("scatter3d", spec)
        self.assertNotIn("X (cm)", spec)  # the 3D chart titles are plain X/Y/Z; units are in the caption
        self.assertIn("Coordinate units: cm. Scale is approximate.", [c.value for c in app.caption])

    def test_downloads_match_backend_exports(self):
        app = reconstructed_app(self.files)
        result = app.session_state["m6b_result"]
        self.assertEqual(vr.points_csv(result).splitlines()[0].split(",")[0], "point")
        self.assertEqual(sorted(b.proto.label for b in app.get("download_button")),
                         ["Diagnostic report", "PLY point cloud", "Points CSV"])

    def test_changing_scale_invalidates_results(self):
        app = reconstructed_app(self.files)
        self.assertIn(STEP[5], headers(app))
        app.number_input[0].set_value(2.0)
        run(app)
        click(app, "m6b_use_scale")  # a different scale choice than the one the results used
        self.assertEqual(headers(app)[-1], STEP[4])
        self.assertNotIn("m6b_result", app.session_state)
        self.assertEqual(scroll_targets(app), [4])
        click(app, "m6b_run")
        self.assertEqual(app.session_state["m6b_result"].units, "cm")

    def test_reconstruction_errors_are_inline(self):
        app = matched_app(self.files)
        click(app, "m6b_skip_scale")
        with unittest.mock.patch.object(vr, "reconstruct_matches", side_effect=vr.DegenerateError("Too flat.")):
            click(app, "m6b_run")
        self.assertEqual(app.error[0].value, "Too flat.")
        self.assertEqual(headers(app)[-1], STEP[4])
        with unittest.mock.patch.object(vr, "reconstruct_matches", side_effect=np.linalg.LinAlgError("x")):
            click(app, "m6b_run")
        self.assertEqual(app.error[0].value, "Reconstruction could not complete. Please try different photos.")
        self.assertNotIn("m6b_result", app.session_state)

    def test_metric_upgrade_failure_is_reported_without_a_3d_view(self):
        bad = dict(vr.solve_metric(np.random.default_rng(2).normal(size=(8, 3))))
        app = matched_app(self.files)
        click(app, "m6b_skip_scale")
        with unittest.mock.patch.object(vr, "solve_metric", return_value=bad):
            click(app, "m6b_run")
        self.assertEqual(headers(app)[-1], STEP[4])
        self.assertIn("metric upgrade failed", app.error[0].value.lower())
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any(e.label == "Diagnostics" for e in app.expander))
        self.assertEqual([b.proto.label for b in app.get("download_button")], ["Diagnostic report"])
        self.assertNotIn("m6b_result", app.session_state)
        click(app, "m6b_run")  # a normal run afterwards clears the failure
        self.assertEqual(headers(app)[-1], STEP[5])
        self.assertEqual(len(app.error), 0)

    def test_warnings_are_plain_text_not_banners(self):
        app = reconstructed_app(self.files)
        result = app.session_state["m6b_result"]
        flat = vr.reconstruct_from_tracks(
            np.stack([result.tracks[v] * [1, 1] for v in range(4)]), result.image_sizes, trim_outliers=False)
        flat.warnings.append("near_planar")
        app.session_state["m6b_result"] = flat
        run(app)
        notes = [m.value for m in app.markdown if m.value.startswith("**Notes**")]
        self.assertEqual(len(notes), 1)
        self.assertIn(vr.WARNING_MESSAGES["near_planar"], notes[0])
        self.assertEqual(len(app.warning), 0)

    # ------------------------------------------------------------------ navigation and scrolling
    def test_results_navigation_buttons(self):
        app = reconstructed_app(self.files)
        labels = [b.label for b in app.button]
        self.assertEqual(labels[-3:], ["Back to Scale", "Start Over", "Back"])
        click(app, "m6b_results_back_scale")
        self.assertEqual(scroll_targets(app), [3])
        self.assertIn(STEP[5], headers(app))  # earlier and later steps stay rendered

    def test_step4_has_only_centered_run_button(self):
        app = matched_app(self.files)
        click(app, "m6b_skip_scale")
        self.assertEqual([b.key for b in app.button if b.key and b.key.startswith("m6b_")
                          and b.key not in ("m6b_skip_scale", "m6b_use_scale", "m6b_start_over_match")],
                         ["m6b_run"])
        self.assertNotIn("Back to Scale", [b.label for b in app.button])
        self.assertEqual(button(app, "m6b_run").proto.type, "primary")
        source = (Path(__file__).resolve().parents[1] / "Pages" / "app_module6.py").read_text()
        self.assertIn("_, run_column, _ = st.columns([1, 2, 1])", source)
        click(app, "m6b_run")  # still runs
        self.assertIn("m6b_result", app.session_state)

    def test_step4_scaled_path_still_works(self):
        app = matched_app(self.files)
        app.number_input[0].set_value(5.0)
        run(app)
        click(app, "m6b_use_scale")
        self.assertIsNotNone(app.session_state["m6b_scale"])
        self.assertNotIn("Back to Scale", [b.label for b in app.button])
        click(app, "m6b_run")
        self.assertIn("m6b_result", app.session_state)

    def test_start_over_resets_everything(self):
        app = reconstructed_app(self.files)
        generation = app.session_state["m6b_upload_generation"]
        click(app, "m6b_start_over")
        self.assertEqual(headers(app), [TITLE, STEP[1]])
        self.assertEqual(app.session_state["m6b_upload_generation"], generation + 1)
        self.assertEqual(scroll_targets(app), [1])
        self.assertEqual([b.key for b in app.button if not (b.key or "").startswith("nav")], [])
        self.assertEqual(len(app.get("image")), 0)
        for key in ("m6b_match", "m6b_result", "m6b_files", "m6b_infos"):
            self.assertNotIn(key, app.session_state)

    def test_scroll_requests_fire_once(self):
        app = matched_app(self.files)
        self.assertEqual(scroll_targets(app), [3])
        run(app)
        self.assertEqual(scroll_targets(app), [])
        click(app, "m6b_skip_scale")
        self.assertEqual(scroll_targets(app), [4])
        click(app, "m6b_run")
        self.assertEqual(scroll_targets(app), [5])
        run(app)
        self.assertEqual(scroll_targets(app), [])

    def test_scroll_script_is_scoped_and_one_shot(self):
        source = PAGE.read_text()
        script = source[source.index("B_SCROLL_JS = "):source.index("B_SCROLL_HOST_CSS")]
        self.assertIn(".st-key-m6b_step_%d", script)
        self.assertIn("scrollIntoView", script)
        self.assertIn("prefers-reduced-motion", script)
        self.assertNotIn("m6v_", script)

    def test_top_and_bottom_back_return_to_selection(self):
        for key in ("navbtn_1", "m6_b_visitor_back_bottom"):
            app = uploaded_app(self.files)
            click(app, key)
            self.assertEqual(app.session_state["module6_section"], "b_select")

    def test_state_uses_m6b_prefix_and_leaves_part_a_alone(self):
        app = reconstructed_app(self.files)
        mine = [k for k in app.session_state.keys() if k.startswith("m6b_")]
        self.assertTrue(mine)
        self.assertFalse([k for k in app.session_state.keys() if k.startswith("m6v_") and
                          k not in ("m6v_upload_generation",)])


if __name__ == "__main__":
    unittest.main()
