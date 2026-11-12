"""Module 7 Part 3, Phase 2A: the "View My Experiment & Results" page for Room Measurement Using Stereo Videos.

The page must read the committed JSON files, never need the original recordings, never run the estimator, and leave the visitor
button an explicit placeholder.  Pure data tests need no Streamlit; page tests use AppTest.
"""
import ast
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from room_measurement import display_data as rd  # noqa: E402

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

PAGE = ROOT / "Pages" / "app_module7.py"
OUT = MODULE / "outputs" / "room_measurement"
VISITOR = "Measure a Room Using Your Own Stereo Videos"
RESULTS = "View My Experiment & Results"


def raw(name):
    return json.loads((OUT / name).read_text())


def page(view):
    at = AppTest.from_file(str(PAGE), default_timeout=120)
    at.session_state["module7_part"] = view
    return at.run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def text(at):
    parts = [m.value for m in at.markdown] + [m.value for m in at.info] + [m.value for m in at.error] + [m.value for m in at.warning]
    parts += [c.value for c in at.caption] + [m.value for m in at.metric] + [e.label for e in at.expander]
    parts += [t.value for t in at.title] + [h.value for h in at.header] + [h.value for h in at.subheader] + [m.value for m in at.markdown]
    return "\n".join(parts)


def imports_of(path):
    tree = ast.parse(Path(path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {f"{node.module}.{a.name}" if node.module else a.name for a in node.names}
    return names


class DataLoaderTests(unittest.TestCase):
    def test_values_come_from_the_committed_json(self):
        r = rd.load_results()
        est, val = raw("estimation_result.json"), raw("validation.json")
        for key in ("width", "height"):
            self.assertEqual(r[key]["estimate_mm"], est["summary"][f"{key}_mm"])
            self.assertEqual(r[key]["actual_mm"], val[f"{key}_mm"]["ground_truth_mm"])
            self.assertEqual(r[key]["abs_error_mm"], val[f"{key}_mm"]["abs_error_mm"])
            self.assertEqual(r[key]["error_percent"], val[f"{key}_mm"]["error_percent"])
            self.assertEqual(list(r[key]["interval_95_mm"]), est["uncertainty"]["total"][key]["interval_95_mm"])
        self.assertEqual([rd.cm(r["width"]["estimate_mm"]), rd.cm(r["width"]["actual_mm"])], ["288", "290"])
        self.assertEqual([rd.cm(r["height"]["estimate_mm"]), rd.cm(r["height"]["actual_mm"])], ["238", "240"])

    def test_the_loader_follows_edited_json_not_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("estimation_result.json", "validation.json"):
                shutil.copy(OUT / name, tmp)
            est = json.loads((Path(tmp) / "estimation_result.json").read_text())
            val = json.loads((Path(tmp) / "validation.json").read_text())
            est["summary"]["width_mm"] = val["width_mm"]["estimate_mm"] = est["uncertainty"]["total"]["width"]["estimate_mm"] = 1234.0
            (Path(tmp) / "estimation_result.json").write_text(json.dumps(est))
            (Path(tmp) / "validation.json").write_text(json.dumps(val))
            self.assertEqual(rd.load_results(tmp)["width"]["estimate_mm"], 1234.0)

    def test_missing_files_give_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rd.ResultsUnavailable) as cm:
                rd.load_results(tmp)
            self.assertIn("estimation_result.json", str(cm.exception))
            shutil.copy(OUT / "estimation_result.json", tmp)
            with self.assertRaises(rd.ResultsUnavailable) as cm:
                rd.load_results(tmp)
            self.assertIn("validation.json", str(cm.exception))

    def test_corrupt_incomplete_or_inconsistent_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("estimation_result.json", "validation.json"):
                shutil.copy(OUT / name, tmp)
            est_path, val_path = Path(tmp) / "estimation_result.json", Path(tmp) / "validation.json"
            good_est, good_val = est_path.read_text(), val_path.read_text()
            est_path.write_text("{not json")
            with self.assertRaises(rd.ResultsUnavailable):
                rd.load_results(tmp)
            est = json.loads(good_est)
            est["ground_truth_used"] = True
            est_path.write_text(json.dumps(est))
            with self.assertRaises(rd.ResultsUnavailable) as cm:
                rd.load_results(tmp)
            self.assertIn("without ground truth", str(cm.exception))
            est_path.write_text(good_est)
            val = json.loads(good_val)
            val["width_mm"]["estimate_mm"] += 5
            val_path.write_text(json.dumps(val))
            with self.assertRaises(rd.ResultsUnavailable) as cm:
                rd.load_results(tmp)
            self.assertIn("disagree", str(cm.exception))
            val_path.write_text(json.dumps({}))
            with self.assertRaises(rd.ResultsUnavailable):
                rd.load_results(tmp)

    def test_committed_estimate_certifies_no_ground_truth(self):
        self.assertIs(raw("estimation_result.json")["ground_truth_used"], False)
        self.assertIs(rd.load_results()["ground_truth_used_in_estimate"], False)


class NoGroundTruthNoRecalculationTests(unittest.TestCase):
    def test_loader_imports_neither_estimator_nor_validation_nor_opencv(self):
        names = imports_of(MODULE / "room_measurement" / "display_data.py")
        self.assertEqual({n.split(".")[0] for n in names}, {"json", "pathlib"})

    def test_page_imports_only_the_display_reader_from_room_measurement(self):
        room = {n for n in imports_of(PAGE) if n.startswith("room_measurement")}
        self.assertEqual(room, {"room_measurement.display_data"})

    def test_page_source_never_mentions_ground_truth_constants_or_recordings(self):
        source = PAGE.read_text()
        section = source[source.index("# ---------------------------------------------------------------- Part 3"):
                         source.index("# ---------------------------------------------------------------- page\n")]
        for forbidden in ("2900", "2400", "ground_truth_mm", ".MOV", "VideoCapture", "run_estimation", "compare_to_ground_truth",
                          "calibration_videos", "data/room_measurement"):
            self.assertNotIn(forbidden, section, forbidden)

    def test_validation_is_the_only_module_that_holds_the_ground_truth(self):
        for path in (MODULE / "room_measurement").glob("*.py"):
            if path.name in ("validation.py", "display_data.py"):
                continue
            self.assertNotIn("validation", {n.split(".")[-1] for n in imports_of(path)}, path.name)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class NavigationTests(unittest.TestCase):
    def test_landing_has_two_buttons_and_a_breadcrumb_back(self):
        at = page("3")
        self.assertFalse(at.exception)
        self.assertEqual([t.value for t in at.title], ["Room Measurement Using Stereo Videos"])
        self.assertEqual([b.label for b in at.button if not (b.key or "").startswith("nav")], [VISITOR, RESULTS])
        self.assertIsNotNone(at.button(key="navbtn_0"))

    def test_results_button_opens_results_and_back_returns_to_the_landing(self):
        at = page("3")
        button(at, RESULTS).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "3_results")
        self.assertIn("My Experiment & Results", [t.value for t in at.title])
        [b for b in at.button if b.label == "Back"][0].click().run()
        self.assertEqual(at.session_state["module7_part"], "3")

    def test_breadcrumb_ancestors_navigate(self):
        for view, index, dest in (("3_results", 1, "3"), ("3_visitor", 1, "3"), ("3_results", 0, "home")):
            at = page(view)
            at.button(key=f"navbtn_{index}").click().run()
            self.assertFalse(at.exception, view)
            self.assertEqual(at.session_state["module7_part"], dest, view)

    def test_existing_module7_pages_still_render(self):
        for view in ("home", "q1_home", "1", "experiment"):
            self.assertFalse(page(view).exception, view)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class ResultsPageTests(unittest.TestCase):
    def test_estimates_actuals_errors_and_intervals_are_shown(self):
        at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual([m.value for m in at.metric], ["288 cm", "290 cm", "238 cm", "240 cm"])
        body = text(at)
        val, est = raw("validation.json"), raw("estimation_result.json")
        for key in ("width", "height"):
            v = val[f"{key}_mm"]
            self.assertIn(f"{v['abs_error_mm'] / 10:.1f} cm absolute", body)
            self.assertIn(f"{abs(v['error_percent']):.2f}%", body)
            low, high = est["uncertainty"]["total"][key]["interval_95_mm"]
            self.assertIn(f"{low / 10:.0f} to {high / 10:.0f} cm", body)

    def test_caveats_are_stated(self):
        body = text(page("3_results")).lower()
        for phrase in ("**not** a guaranteed accuracy", "assumed", "camera-intrinsics", "not** direct triangulation of all four wall corners"):
            self.assertIn(phrase, body)

    def test_method_sequence_has_ten_steps_in_order(self):
        at = page("3_results")
        steps = next(m.value for m in at.markdown if m.value.startswith("1. **Stereo video collection**"))
        lines = steps.splitlines()
        self.assertEqual(len(lines), 10)
        for number, expected in enumerate(("Stereo video collection", "Camera calibration", "Stereo feature matching",
                                           "Stereo camera pose estimation", "Floor-plane reconstruction",
                                           "Wall-boundary detection", "Geometric corner inference", "Two-view registration",
                                           "Width and height estimation", "Validation against physical measurements"), 1):
            self.assertTrue(lines[number - 1].startswith(f"{number}. **{expected}**"), lines[number - 1])
        self.assertGreaterEqual(len(at.expander), 3)       # technical details are collapsed
        how = [e for e in at.expander if e.label == "How it works"]
        self.assertEqual(len(how), 1)
        self.assertFalse(how[0].proto.expanded)            # collapsed by default
        self.assertIn(steps, [m.value for m in how[0].markdown])
        self.assertNotIn("How it works", [s.value for s in at.subheader])
        self.assertNotIn("small derived copies", text(at))

    def test_images_display_and_page_works_without_any_video_access(self):
        with mock.patch("cv2.VideoCapture", side_effect=AssertionError("the results page must not open a video")):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("image")), len(rd.IMAGES))
        self.assertEqual(len(at.warning), 0)
        self.assertEqual(len(at.tabs), 3)

    def test_missing_results_show_a_clear_message_and_a_back_button(self):
        with mock.patch.object(rd, "ESTIMATION_FILE", "does_not_exist.json"):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.error), 1)
        self.assertIn("unavailable", at.error[0].value)
        self.assertIn("does_not_exist.json", at.error[0].value)
        self.assertEqual(len(at.metric), 0)
        self.assertIn("Back", [b.label for b in at.button])

    def test_a_missing_image_is_reported_not_fatal(self):
        images = (("missing_image.jpg", "stereo", "x"),) + rd.IMAGES[1:]
        with mock.patch.object(rd, "IMAGES", images):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.warning), 1)
        self.assertIn("missing_image.jpg", at.warning[0].value)
        self.assertEqual(len(at.get("image")), len(images) - 1)


VIDEO_FILES = ["left_air_wall_left.mp4", "right_pro_wall_left.mp4", "left_air_wall_right.mp4", "right_pro_wall_right.mp4"]


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class RevisionTests(unittest.TestCase):
    def test_heading_and_subtitle(self):
        at = page("3_results")
        self.assertEqual([t.value for t in at.title], ["My Experiment & Results"])
        self.assertEqual([h.value for h in at.header], ["Room Measurement Using Stereo Videos"])
        css = next(m.value for m in at.markdown if "m7_room_head h2" in m.value)
        self.assertIn("m7_room_head h2 {\n    font-size: clamp(1.4rem, 4.6vw, 2.4rem)", css)   # 2.4rem = 38px on desktop, responsive below
        self.assertIn("padding-bottom: 28px", css)                                              # heading -> subtitle spacing
        self.assertIn("m7_room_intro", css)

    def test_intro_is_the_short_sentence_and_details_are_collapsed(self):
        at = page("3_results")
        intro = "This experiment estimates wall dimensions using stereo videos captured from two different viewpoints."
        self.assertIn(intro, [m.value for m in at.markdown])
        about = next(e for e in at.expander if e.label == "About this measurement")
        body = " ".join(m.value for m in about.markdown)
        for phrase in ("not** direct triangulation of all four wall corners", "inferred", "approximate", "not a guarantee"):
            self.assertIn(phrase, body)
        self.assertNotIn("direct triangulation", intro)

    def test_no_user_facing_method_j(self):
        for view in ("3", "3_visitor", "3_results"):
            self.assertNotIn("method j", text(page(view)).lower(), view)
        at = page("3_results")
        self.assertNotIn("method j", " ".join(e.label for e in at.expander).lower())
        section = PAGE.read_text()
        section = section[section.index("# ---------------------------------------------------------------- Part 3"):
                          section.index("# ---------------------------------------------------------------- page\n")]
        self.assertNotIn("Method J", section)
        # the underlying identifiers and saved metadata are untouched
        self.assertIn("method_J", raw("estimation_result.json")["pose"]["report"])
        self.assertTrue((MODULE / "room_measurement" / "pose.py").read_text().count("Method J") > 0)

    def test_camera_calibration_expander_is_collapsed_and_precedes_results(self):
        at = page("3_results")
        labels = [e.label for e in at.expander]
        self.assertIn("Camera Calibration", labels)
        cal = next(e for e in at.expander if e.label == "Camera Calibration")
        self.assertFalse(cal.proto.expanded)
        self.assertLess(labels.index("Camera Calibration"), labels.index("Technical details: joint stereo pose refinement"))
        order = []

        def walk(block):
            for child in getattr(block, "children", {}).values():
                value = getattr(child, "label", None) if child.type == "expander" else getattr(child, "value", None)
                order.append(value if isinstance(value, str) else None)
                walk(child)
        walk(at.main)
        self.assertLess(next(i for i, v in enumerate(order) if v == "Camera Calibration"),
                        next(i for i, v in enumerate(order) if v == "Results"))

    def test_every_room_data_attribute_used_by_the_page_exists(self):
        """Regression: the page crashed with AttributeError (no attribute 'load_calibration') when display_data lacked a name
        the page calls.  Every ``room_data.<name>`` in the page must resolve on the display_data module."""
        used = {node.attr for node in ast.walk(ast.parse(PAGE.read_text()))
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "room_data"}
        self.assertIn("load_calibration", used)
        missing = sorted(name for name in used if not hasattr(rd, name))
        self.assertEqual(missing, [], f"app_module7.py uses room_data attributes missing from display_data: {missing}")
        self.assertTrue(callable(rd.load_calibration))

    def test_calibration_values_are_read_from_the_json(self):
        data = json.loads(rd.CALIBRATION_FILE.read_text())
        cal = rd.load_calibration()
        self.assertEqual(cal["left"]["K"], data["left"]["K"])
        self.assertEqual(cal["right"]["K"], data["right"]["K"])
        self.assertEqual(list(cal["left"]["distortion"].values()), data["left"]["distortion_k1_k2_p1_p2_k3"])
        self.assertEqual((cal["width"], cal["height"], cal["baseline_mm"]), (2160, 3840, 225.0))
        self.assertEqual((cal["pattern"], cal["square_mm"]), ((8, 6), 20.0))
        self.assertNotIn("legacy", json.dumps(cal).lower())
        self.assertNotIn("T_mm", json.dumps(cal))
        at = page("3_results")
        cell = " ".join(df.value.to_string() for df in at.table)
        for side in ("left", "right"):
            self.assertIn(f"{data[side]['K'][0][0]:.4f}", cell)
            self.assertIn(f"{data[side]['distortion_k1_k2_p1_p2_k3'][0]:.5f}", cell)
        body = text(at)
        for phrase in ("iPhone Air", "iPhone 17 Pro Max", "2160 \u00d7 3840", "225 mm", "8 \u00d7 6", "20 mm"):
            self.assertIn(phrase, cell + body)
        self.assertNotIn("227.8", cell + body)           # the legacy calibrated baseline is not presented

    def test_calibration_file_missing_is_reported_inside_the_expander(self):
        with mock.patch.object(rd, "CALIBRATION_FILE", Path("/nonexistent/cal.json")), \
                mock.patch.object(rd, "load_calibration", wraps=lambda: rd._read(Path("/nonexistent/cal.json"))):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertTrue(any("calibration details are unavailable" in e.value for e in at.error))
        self.assertEqual([m.value for m in at.metric], ["288 cm", "290 cm", "238 cm", "240 cm"])   # results unaffected

    def test_videos_render_in_two_groups_with_players(self):
        at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("video")), 4)
        body = text(at)
        self.assertIn("View 1 \u2014 Left Side of Wall", body)
        self.assertIn("View 2 \u2014 Right Side of Wall", body)
        self.assertEqual(body.count("iPhone Air (Left Camera)"), 2)
        self.assertEqual(body.count("iPhone 17 Pro Max (Right Camera)"), 2)
        self.assertLess(body.index("View 1"), body.index("View 2"))

    def test_missing_videos_fall_back_without_breaking_the_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rd, "video_status", side_effect=lambda: [(Path(tmp) / n, g, l, False) for g, n, l in rd.VIDEOS]):
                at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("video")), 0)
        self.assertEqual(len([i for i in at.info if "Video not available" in i.value]), 4)
        self.assertEqual([m.value for m in at.metric], ["288 cm", "290 cm", "238 cm", "240 cm"])
        self.assertEqual(len(at.get("image")), len(rd.IMAGES))

    def test_existing_results_are_unchanged(self):
        at = page("3_results")
        body = text(at)
        for phrase in ("−2.0 cm (2.0 cm absolute, −0.70%)", "−2.0 cm (2.0 cm absolute, −0.82%)", "266 to 310 cm", "219 to 257 cm"):
            self.assertIn(phrase, body)
        self.assertEqual(len(at.tabs), 3)
        self.assertEqual(len(at.get("image")), 5)
        est, val = raw("estimation_result.json"), raw("validation.json")
        self.assertAlmostEqual(est["summary"]["width_mm"], 2879.5660614084068)
        self.assertAlmostEqual(val["height_mm"]["ground_truth_mm"], 2400.0)


class DisplayVideoTests(unittest.TestCase):
    def test_committed_videos_are_small_h264_and_match_the_sources(self):
        manifest = json.loads((rd.VIDEO_DIR / "videos_manifest.json").read_text())
        frames = json.loads((MODULE / "room_measurement" / "config" / "frame_manifest.json").read_text())["videos"]
        self.assertLess(manifest["total_bytes"], 8_000_000)
        total = 0
        for key, row in manifest["videos"].items():
            path = rd.VIDEO_DIR / row["file"]
            self.assertTrue(path.is_file(), path.name)
            total += path.stat().st_size
            self.assertEqual(path.stat().st_size, row["bytes"])
            self.assertLess(path.stat().st_size, 3_000_000)
            self.assertEqual(row["source_sha256"], frames[key]["sha256"])
            self.assertEqual(row["frames"], frames[key]["frame_count"])
            head = path.read_bytes()[:4096]
            self.assertIn(b"ftyp", head)
            self.assertTrue(b"avc1" in path.read_bytes()[:100000], "H.264 (avc1) expected")
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), row["frames"])
            self.assertEqual((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (720, 1280))
            cap.release()
        self.assertEqual(sorted(r["file"] for r in manifest["videos"].values()), sorted(VIDEO_FILES))
        self.assertLess(total, 8_000_000)

    def test_video_list_matches_the_four_wall_recordings(self):
        self.assertEqual([n for _, n, _ in rd.VIDEOS], VIDEO_FILES)


class DisplayAssetTests(unittest.TestCase):
    def test_committed_images_are_small_valid_jpegs(self):
        total = 0
        for path, _, _, exists in rd.image_status():
            self.assertTrue(exists, path.name)
            size = path.stat().st_size
            total += size
            self.assertLess(size, 300_000, path.name)
            img = cv2.imread(str(path))
            self.assertIsNotNone(img, path.name)
            self.assertLessEqual(img.shape[1], 1000, path.name)
        self.assertLess(total, 1_500_000)

    def test_display_manifest_records_a_faithful_derivation(self):
        manifest = json.loads((rd.DISPLAY_DIR / "display_manifest.json").read_text())
        self.assertIs(manifest["ground_truth_used"], False)
        self.assertIs(manifest["reproduces_committed_estimate"], True)
        self.assertEqual(manifest["frame_indices"], raw("estimation_result.json")["inputs"]["frame_indices"][manifest["frame_set"]])

    def test_no_raw_videos_in_the_deployable_outputs(self):
        self.assertEqual([p for p in OUT.rglob("*") if p.suffix.lower() in (".mov", ".avi", ".m4v")], [])
        self.assertLess(sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()), 8_000_000)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class VisitorPlaceholderTests(unittest.TestCase):
    def test_placeholder_is_labelled_and_offers_no_upload(self):
        at = page("3_visitor")
        self.assertFalse(at.exception)
        self.assertEqual([t.value for t in at.title], [VISITOR])
        self.assertTrue(any("not implemented" in i.value.lower() for i in at.info))
        self.assertEqual(len(at.get("file_uploader")), 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.get("image")), 0)

    def test_placeholder_never_runs_the_estimator_or_loads_the_saved_experiment(self):
        from room_measurement import pipeline
        with mock.patch.object(pipeline, "run_estimation", side_effect=AssertionError("estimator must not run")) as run, \
                mock.patch.object(rd, "load_results", side_effect=AssertionError("saved results must not load")) as load, \
                mock.patch("cv2.VideoCapture", side_effect=AssertionError("no video access")):
            at = page("3_visitor")
        self.assertFalse(at.exception)
        run.assert_not_called()
        load.assert_not_called()

    def test_landing_never_runs_anything_either(self):
        with mock.patch.object(rd, "load_results", side_effect=AssertionError("landing must not load results")):
            self.assertFalse(page("3").exception)


if __name__ == "__main__":
    unittest.main()
