"""Module 6 Part B visitor page workflow tests (AppTest with synthetic photos; no browser).

The click component cannot be clicked in AppTest, so point placement is simulated through session state, as in
the Part A visitor tests; the click-to-pixel mapping itself is tested in test_module6_partb_backend.py.
"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_module6_partb_backend import orthographic_tracks, planar_views, png_bytes, texture  # noqa: E402
from Modules.Module_6.Part_B import sfm_pipeline as sfm  # noqa: E402
from Modules.Module_6.Part_B import visitor_reconstruction as vr  # noqa: E402

PAGE = ROOT / "Pages" / "app_module6.py"
TITLE = "Structure from Motion Reconstruction"
STEP = {1: "Step 1 — Upload Your Images", 2: "Step 2 — Select Corresponding Points",
        3: "Step 3 — Enter Object Dimensions", 4: "Step 4 — Reconstruct Your Object", 5: "Step 5 — View Results"}
FILES, TRUTH = planar_views()
# A non-planar eight-point configuration that fits inside the 1000 x 900 synthetic photos.
VERIFIED = orthographic_tracks(scale=28.0)


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


def active_step(app):
    styles = " ".join(el.proto.body for el in app.get("html"))
    return [int(n) for n in re.findall(r"\.st-key-m6b_step_(\d) \{ border-color", styles)]


def empty_app():
    app = AppTest.from_file(str(PAGE), default_timeout=120)
    app.session_state["module6_section"] = "b_visitor"
    run(app)
    return app


def uploaded_app(files=FILES):
    app = empty_app()
    app.get("file_uploader")[0].set_value([(name, data, "image/png") for data, name in files])
    run(app)
    return app


def set_points(app, view, points, status):
    stored = app.session_state["m6b_points"]
    stored[view] = points
    app.session_state["m6b_points"] = stored
    statuses = app.session_state["m6b_status"]
    statuses[view] = [status if np.isfinite(p).all() else None for p in np.asarray(points)]
    app.session_state["m6b_status"] = statuses


def corners_only(points):
    return np.where(np.arange(8)[:, None] < 4, points, np.nan)


def selected_app(files=FILES):
    app = uploaded_app(files)
    set_points(app, 0, corners_only(TRUTH[0]), "selected")
    run(app)
    return app


def found_app(files=FILES):
    app = selected_app(files)
    click(app, "m6b_find")
    return app


def add_photos(app, files):
    app.get("file_uploader")[0].set_value([(name, data, "image/png") for data, name in files])
    run(app)


def pills(app):
    return next(w for w in app.get("button_group") if "P1" in w.proto.options[0].content)


def photo_choice(app):
    return next(w for w in app.get("button_group") if "Photo 1" in w.proto.options[0].content)


def confirmed_app(tracks=VERIFIED):
    app = found_app()
    for view in range(4):
        set_points(app, view, tracks[view], "corrected" if view else "selected")
    app.session_state["m6b_confirmed"] = [False] * 4  # the corrections replace every photo's points
    run(app)
    for _ in range(4):
        click(app, "m6b_confirm")
    return app


def dimensions_app(width=10.0, height=20.0, tracks=VERIFIED):
    app = confirmed_app(tracks)
    app.number_input(key="m6b_width").set_value(width)
    app.number_input(key="m6b_height").set_value(height)
    run(app)
    click(app, "m6b_dims_continue")
    return app


def reconstructed_app(tracks=VERIFIED):
    app = dimensions_app(tracks=tracks)
    click(app, "m6b_run")
    return app


class UploadTests(unittest.TestCase):
    def test_empty_page_shows_only_step_one_in_the_part_a_style(self):
        app = empty_app()
        self.assertEqual([t.value for t in app.title], [TITLE])
        self.assertIn("MODULE 6 — PART B", [m.value for m in app.markdown])
        self.assertEqual(headers(app), [STEP[1]])
        self.assertEqual(active_step(app), [1])
        self.assertEqual([b.key for b in app.button if not (b.key or "").startswith("nav")], [])
        self.assertIn("**0 of 4 photos uploaded**", [m.value for m in app.markdown])
        self.assertEqual(scroll_targets(app), [])
        self.assertFalse(app.error or app.warning)

    def test_uploader_configuration_and_limits(self):
        app = empty_app()
        proto = app.get("file_uploader")[0].proto
        self.assertTrue(proto.multiple_files)
        self.assertEqual(proto.max_upload_size_mb, 25)
        self.assertEqual(sorted(proto.type), sorted(vr.ALLOWED_EXTENSIONS))
        captions = " ".join(c.value for c in app.caption)
        for phrase in ("Exactly 4 photographs", "JPG, PNG, HEIC/HEIF", "25 MB per photo", "100 MB in total",
                       "64 megapixels"):
            self.assertIn(phrase, captions)

    def test_capture_instructions_are_collapsed(self):
        app = empty_app()
        expander = next(e for e in app.expander if e.label == "Image Capture Instructions")
        self.assertFalse(expander.proto.expanded)
        text = " ".join(m.value for m in expander.markdown)
        for phrase in ("stationary", "approximately flat", "rectangular", "four corners", "distinctive",
                       "different angles", "centimeters"):
            self.assertIn(phrase, text)

    def test_four_photos_open_step_two(self):
        app = uploaded_app()
        self.assertEqual(headers(app), [STEP[1], STEP[2]])
        self.assertEqual(len(app.get("file_uploader")), 0)  # no add control once four photos are uploaded
        self.assertIn("**4 of 4 photos uploaded**", [m.value for m in app.markdown])
        self.assertEqual(len([i for i in app.get("image")]), 5)  # four previews and the selection example
        captions = " ".join(c.value for c in app.caption)
        for number in range(1, 5):
            self.assertIn(f"Photo {number}", captions)
        self.assertIn("1000 × 900 px", captions)
        self.assertEqual(scroll_targets(app), [2])
        self.assertEqual(active_step(app), [2])
        self.assertEqual(len(app.session_state["m6b_images"]), 4)

    def test_photos_can_be_added_individually_in_order_and_removed(self):
        app = empty_app()
        for number in range(4):
            self.assertIn(f"**{number} of 4 photos uploaded**", [m.value for m in app.markdown])
            self.assertEqual(app.get("file_uploader")[0].proto.label,
                             "+ Add More Photos" if number else "+ Add Photos")
            add_photos(app, [FILES[number]])
        self.assertEqual([name for _, name in app.session_state["m6b_files"]], [name for _, name in FILES])
        self.assertEqual(headers(app), [STEP[1], STEP[2]])
        click(app, "m6b_remove_1")
        self.assertEqual([name for _, name in app.session_state["m6b_files"]], ["view1.png", "view3.png", "view4.png"])
        self.assertIn("**3 of 4 photos uploaded**", [m.value for m in app.markdown])
        self.assertEqual(headers(app), [STEP[1]])
        self.assertEqual(len(app.get("file_uploader")), 1)
        add_photos(app, [FILES[1]])
        self.assertEqual(app.session_state["m6b_files"][-1][1], "view2.png")

    def test_more_than_four_photos_are_rejected(self):
        app = uploaded_app(FILES[:3])
        add_photos(app, FILES[:2])
        self.assertIn("only 1 more", app.error[0].value)
        self.assertEqual(len(app.session_state["m6b_files"]), 3)

    def test_unsupported_file_is_rejected_and_valid_ones_kept(self):
        app = uploaded_app(FILES[:3] + [(b"not an image", "notes.png")])
        self.assertEqual(headers(app), [STEP[1]])
        self.assertIn("could not be read", app.error[0].value)
        self.assertEqual(len(app.session_state["m6b_files"]), 3)

    def test_total_size_limit(self):
        original = vr.MAX_TOTAL_BYTES
        vr.MAX_TOTAL_BYTES = len(FILES[0][0]) + 10
        try:
            app = uploaded_app(FILES[:1])
            add_photos(app, FILES[1:2])
        finally:
            vr.MAX_TOTAL_BYTES = original
        self.assertIn("Total upload exceeds", app.error[0].value)
        self.assertEqual(len(app.session_state["m6b_files"]), 1)

    def test_leaving_the_page_discards_the_photos(self):
        app = uploaded_app()
        click(app, "nav_arrow_button")
        self.assertNotIn("m6b_files", app.session_state)
        self.assertNotIn("m6b_images", app.session_state)


class PointSelectionTests(unittest.TestCase):
    def test_only_four_corners_are_requested_first(self):
        app = uploaded_app()
        self.assertTrue(button(app, "m6b_find").disabled)
        self.assertEqual(len(app.get("button_group")), 1)  # no photo chooser yet
        self.assertEqual([o.content for o in pills(app).proto.options], ["P1 ?", "P2 ?", "P3 ?", "P4 ?"])
        self.assertIn("click **P1** — top-left corner", " ".join(m.value for m in app.markdown))
        expander = next(e for e in app.expander if e.label == "How to Select the Points")
        text = " ".join(m.value for m in expander.markdown)
        for phrase in ("clockwise", "top-left", "P5–P8", "same physical point", "zoom", "?"):
            self.assertIn(phrase, text)

    def test_four_corners_enable_finding(self):
        app = selected_app()
        self.assertEqual(button(app, "m6b_find").label, "Find Corresponding Points")
        self.assertFalse(button(app, "m6b_find").disabled)
        self.assertFalse(app.warning)

    def test_corners_out_of_order_are_explained(self):
        app = uploaded_app()
        set_points(app, 0, corners_only(TRUTH[0][[0, 2, 1, 3, 4, 5, 6, 7]]), "selected")
        run(app)
        self.assertTrue(button(app, "m6b_find").disabled)
        self.assertIn("P1–P4 must be the top-left", app.warning[0].value)

    def test_finding_fills_all_photos_with_the_shared_function(self):
        app = found_app()
        expected = sfm.find_correspondences(app.session_state["m6b_images"], TRUTH[0][:4])
        points, statuses = app.session_state["m6b_points"], app.session_state["m6b_status"]
        np.testing.assert_array_equal(points, expected.points)
        self.assertEqual(statuses[0], ["selected"] * 4 + ["automatic"] * 4)
        self.assertEqual(statuses[1], ["automatic"] * 8)
        self.assertEqual(statuses[3][1], None)  # estimated outside the photo: must be selected manually
        self.assertEqual(app.session_state["m6b_view"], 0)
        self.assertEqual(app.session_state["m6b_confirmed"], [False] * 4)
        self.assertEqual(len(pills(app).proto.options), 8)

    def test_no_prediction_wording_is_visible(self):
        for app in (selected_app(), found_app()):
            for view in range(4 if "m6b_matches" in app.session_state else 1):
                if view:
                    photo_choice(app).set_value(view)
                    run(app)
                visible = " ".join([m.value for m in app.markdown] + [c.value for c in app.caption]
                                   + [w.value for w in app.warning] + [b.label for b in app.button])
                self.assertNotIn("predict", visible.lower())

    def test_missing_point_blocks_confirmation_until_selected(self):
        app = found_app()
        for _ in range(3):
            click(app, "m6b_confirm")
        self.assertEqual(app.session_state["m6b_view"], 3)
        self.assertTrue(button(app, "m6b_confirm").disabled)
        self.assertEqual(pills(app).proto.options[1].content, "P2 ?")
        self.assertEqual(app.session_state["m6b_active"], 1)  # the missing point is highlighted
        points = app.session_state["m6b_points"][3].copy()
        points[1] = [5, 300]  # manual selection of the missing point
        set_points(app, 3, points, "corrected")
        run(app)
        click(app, "m6b_confirm")
        self.assertEqual(app.session_state["m6b_confirmed"], [True] * 4)
        self.assertEqual(headers(app), [STEP[1], STEP[2], STEP[3]])
        self.assertEqual(scroll_targets(app), [3])

    def test_unidentified_landmarks_fall_back_to_manual_selection(self):
        unrelated = (png_bytes(texture(900, 42, 1000)), "other.png")
        app = found_app([*FILES[:3], unrelated])
        self.assertIn("0 of the 4 interior points", app.warning[0].value)
        self.assertTrue(np.isnan(app.session_state["m6b_points"][:, 4:]).all())
        self.assertEqual(app.session_state["m6b_active"], 4)
        self.assertTrue(button(app, "m6b_confirm").disabled)
        set_points(app, 0, TRUTH[0], "corrected")
        run(app)
        self.assertFalse(button(app, "m6b_confirm").disabled)

    def test_restore_automatic_points_and_reset_all_points(self):
        app = found_app()
        automatic = app.session_state["m6b_matches"].points
        for view in (0, 1):
            photo_choice(app).set_value(view)
            run(app)
            set_points(app, view, VERIFIED[view], "corrected")
            run(app)
            self.assertEqual(button(app, "m6b_restore").label, "Restore Automatic Points")
            click(app, "m6b_restore")
            np.testing.assert_array_equal(app.session_state["m6b_points"][view], automatic[view])
        photo_choice(app).set_value(0)
        run(app)
        click(app, "m6b_reset_points")
        self.assertTrue(np.isnan(app.session_state["m6b_points"]).all())
        self.assertIsNone(app.session_state.get("m6b_matches") if "m6b_matches" in app.session_state else None)
        self.assertEqual(len(pills(app).proto.options), 4)

    def test_changing_photo_one_requires_confirming_every_photo_again(self):
        app = confirmed_app()
        photo_choice(app).set_value(0)
        run(app)
        click(app, "m6b_restore")
        self.assertEqual(app.session_state["m6b_confirmed"], [False] * 4)
        self.assertEqual(headers(app), [STEP[1], STEP[2]])


class DimensionAndResultTests(unittest.TestCase):
    def test_dimensions_must_be_positive(self):
        app = confirmed_app()
        self.assertEqual(active_step(app), [3])
        self.assertTrue(button(app, "m6b_dims_continue").disabled)
        app.number_input(key="m6b_width").set_value(0.0)
        app.number_input(key="m6b_height").set_value(20.0)
        run(app)
        self.assertTrue(button(app, "m6b_dims_continue").disabled)
        self.assertIn("greater than zero", app.warning[0].value)
        labels = [n.label for n in app.number_input]
        self.assertEqual(labels, ["Measured Width (cm)", "Measured Height (cm)"])

    def test_reconstruction_uses_the_shared_pipeline(self):
        app = reconstructed_app()
        self.assertEqual(headers(app), list(STEP.values()))
        self.assertEqual(active_step(app), [5])
        self.assertEqual(scroll_targets(app), [5])
        result = app.session_state["m6b_result"]
        expected = sfm.reconstruct(VERIFIED, 10.0, 20.0)
        np.testing.assert_array_equal(result.tracks, VERIFIED)
        np.testing.assert_array_equal(result.cm, expected.cm)
        self.assertEqual(result.measurements, expected.measurements)
        table = next(t.value for t in app.table if t.value.index.name == "Measurement")
        self.assertEqual(list(table.index), ["Top edge", "Bottom edge", "Left edge", "Right edge"])
        self.assertEqual(list(table["Error (%)"]), ["0.00"] * 4)
        points = next(t.value for t in app.table if t.value.index.name == "Point")
        self.assertEqual(list(points.index), [f"P{i}" for i in range(1, 9)])
        self.assertEqual(len(app.get("plotly_chart")), 2)
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("not an independent validation", captions)
        self.assertEqual([d.label for d in app.get("download_button")], ["Points and Edges (CSV)"])
        # the top breadcrumb is the only Back control; there is no bottom Back button
        self.assertEqual([b.key for b in app.button if b.label in ("Back", "Go back")], ["nav_arrow_button"])
        self.assertEqual(button(app, "m6b_reset_all").label, "Reset All")

    def test_reconstruction_failure_is_explained(self):
        bad = VERIFIED.copy()
        bad[2, 6] = bad[2, 5]  # two points at one position
        app = reconstructed_app(bad)
        self.assertNotIn(STEP[5], headers(app))
        self.assertIn("same position", app.error[0].value)
        self.assertIsNone(app.session_state.get("m6b_result"))

    def test_changing_dimensions_invalidates_the_results(self):
        app = reconstructed_app()
        app.number_input(key="m6b_width").set_value(12.0)
        run(app)
        self.assertEqual(headers(app), [STEP[1], STEP[2], STEP[3]])
        self.assertIsNone(app.session_state.get("m6b_result"))
        self.assertIsNone(app.session_state["m6b_dims"])

    def test_changing_points_invalidates_the_results(self):
        app = reconstructed_app()
        photo_choice(app).set_value(2)
        run(app)
        click(app, "m6b_restore")  # restoring photo 3 changes its points
        self.assertIsNone(app.session_state.get("m6b_result"))
        self.assertEqual(headers(app), [STEP[1], STEP[2]])
        self.assertFalse(app.session_state["m6b_confirmed"][2])
        self.assertEqual(app.session_state["m6b_dims"], (10.0, 20.0))  # the measured size is kept

    def test_new_images_invalidate_everything(self):
        app = reconstructed_app()
        other = [(png_bytes(texture(900, 50 + i, 1000)), f"other{i}.png") for i in range(4)]
        for _ in range(4):
            click(app, "m6b_remove_0")
        self.assertIsNone(app.session_state["m6b_result"] if "m6b_result" in app.session_state else None)
        add_photos(app, other)
        self.assertEqual(headers(app), [STEP[1], STEP[2]])
        self.assertIsNone(app.session_state.get("m6b_result"))
        self.assertNotIn("m6b_matches", app.session_state)
        self.assertTrue(np.isnan(app.session_state["m6b_points"]).all())

    def test_reset_all_returns_to_step_one(self):
        app = reconstructed_app()
        click(app, "m6b_reset_all")
        self.assertEqual(headers(app), [STEP[1]])
        self.assertEqual(scroll_targets(app), [1])
        self.assertIsNone(app.session_state.get("m6b_images"))
        self.assertIsNone(app.session_state.get("m6b_result"))


if __name__ == "__main__":
    unittest.main()
