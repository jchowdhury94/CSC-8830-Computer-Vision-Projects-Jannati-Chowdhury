"""Module 7 Part 3, Phase 2A: the "View My Experiments & Results" page for Room Measurement Using Stereo Videos.

The page must read the committed JSON files, never need the original recordings, never run the estimator, and leave the visitor
button an explicit placeholder.  Pure data tests need no Streamlit; page tests use AppTest.
"""
import ast
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7" / "room_measurement_question_3"      # Question 3 packages, data and outputs
sys.path.insert(0, str(MODULE)); sys.path.insert(0, str(MODULE.parent))      # MODULE.parent: shared stereo_measure
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


def in_page_order(at):
    """[(element type, text)] of the rendered page in top-to-bottom order (videos have no text and are listed as ('video', ''))."""
    out = []

    def walk(block):
        for child in getattr(block, "children", {}).values():
            kind = child.type
            if kind in ("markdown", "caption", "header", "subheader", "title", "info", "warning", "error", "metric"):
                out.append((kind, str(getattr(child, "value", "") or "")))
            elif kind == "expander":
                out.append((kind, child.label))
            elif kind == "video":
                out.append((kind, ""))
            walk(child)
    walk(at.main)
    return out


def keyed_blocks(at):
    """{key: Block} for every keyed container on the page."""
    out = {}

    def walk(block):
        for child in getattr(block, "children", {}).values():
            if getattr(child, "key", None):
                out[child.key] = child
            walk(child)
    walk(at.main)
    return out


def items_in(block):
    """[(element type, text)] inside one container, in page order (videos listed as ('video', ''))."""
    out = []

    def walk(b):
        for child in getattr(b, "children", {}).values():
            kind = child.type
            if kind in ("markdown", "caption", "header", "subheader", "info", "warning", "error", "metric", "latex"):
                out.append((kind, str(getattr(child, "value", "") or "")))
            elif kind == "expander":
                out.append((kind, child.label))
            elif kind in ("video", "table", "vega_lite_chart", "imgs", "image"):
                out.append((kind, ""))
            walk(child)
    walk(block)
    return out


def section(at, start, end=None):
    """Text elements between the header ``start`` and the header ``end`` (page order)."""
    items = in_page_order(at)
    a = next(i for i, (k, v) in enumerate(items) if k == "header" and v == start)
    b = next((i for i, (k, v) in enumerate(items) if i > a and k == "header" and v == end), len(items)) if end else len(items)
    return items[a:b]


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
    def test_results_selection_opens_results_and_the_top_arrow_returns_to_the_selection(self):
        at = page("my_results")
        button(at, "**Room Measurement Using Stereo Videos**  \n(Question 3)").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["module7_part"], "3_results")
        self.assertIn("My Experiment & Results", [t.value for t in at.title])
        self.assertFalse([b for b in at.button if b.label == "Back"])
        at.button(key="nav_arrow_button").click().run()
        self.assertEqual(at.session_state["module7_part"], "my_results")

    def test_breadcrumb_ancestors_navigate(self):
        for view, index, dest in (("3_results", 1, "my_results"), ("3_visitor", 1, "run_own"), ("3_results", 0, "home")):
            at = page(view)
            at.button(key=f"navbtn_{index}").click().run()
            self.assertFalse(at.exception, view)
            self.assertEqual(at.session_state["module7_part"], dest, view)

    def test_existing_module7_pages_still_render(self):
        for view in ("home", "my_results", "run_own", "1", "experiment"):
            self.assertFalse(page(view).exception, view)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class ResultsPageTests(unittest.TestCase):
    def test_estimates_actuals_errors_and_intervals_are_shown(self):
        at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual([m.value for m in at.metric][:4], ["288 cm", "290 cm", "238 cm", "240 cm"])   # Wall A cards, unchanged
        body = text(at)
        val, est = raw("validation.json"), raw("estimation_result.json")
        for key in ("width", "height"):
            v = val[f"{key}_mm"]
            self.assertIn(f"**Absolute error:** {v['abs_error_mm'] / 10:.1f} cm", body)
            self.assertIn(f"{abs(v['error_percent']):.2f}%", body)
            low, high = est["uncertainty"]["total"][key]["interval_95_mm"]
            self.assertIn(f"{low / 10:.0f} to {high / 10:.0f} cm", body)

    def test_caveats_are_stated(self):
        body = text(page("3_results")).lower()
        for phrase in ("assumed rather than measured", "camera-intrinsics", "dominates"):
            self.assertIn(phrase, body)

    def test_one_unified_methodology_with_eleven_numbered_steps(self):
        at = page("3_results")
        how = [e for e in at.expander if e.label == "How It Works \u2014 Methodology"]
        self.assertEqual(len(how), 1)
        self.assertFalse(how[0].proto.expanded)
        values = [m.value for m in how[0].markdown]
        titles = [re.match(r"#### \d+\. (.+)", v).group(1) for v in values if re.match(r"#### \d+\. ", v)]
        self.assertEqual(titles, ["Stereo Video Acquisition and Frame Selection", "Camera Model and Calibration", "SIFT Feature Detection and Matching",
                                  "Epipolar Geometry and Stereo Pose Estimation", "3D Triangulation", "Floor-Plane Estimation",
                                  "Wall-Boundary Annotation and Line Fitting", "Geometric Corner Estimation", "Multi-View Registration Using RANSAC",
                                  "Room Dimension Estimation", "Measurement Uncertainty and Consistency"])
        self.assertEqual(len(how[0].table), 2)                         # pose refinement metrics and the Wall A uncertainty sources

    def test_images_display_and_page_works_without_any_video_access(self):
        with mock.patch("cv2.VideoCapture", side_effect=AssertionError("the results page must not open a video")):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("image")), len(rd.IMAGES))
        self.assertEqual(len(at.warning), 0)
        self.assertEqual(len(at.tabs), 3)

    def test_missing_results_show_a_clear_message_and_no_bottom_back_button(self):
        with mock.patch.object(rd, "ESTIMATION_FILE", "does_not_exist.json"):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.error), 1)
        self.assertIn("unavailable", at.error[0].value)
        self.assertIn("does_not_exist.json", at.error[0].value)
        self.assertEqual(len(at.metric), 0)
        self.assertNotIn("Back", [b.label for b in at.button])   # the top arrow is the only Back

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
        self.assertEqual([h.value for h in at.header], ["Room Measurement Using Stereo Videos", "Wall A \u2014 Width and Height Measurement",
                                                         "Wall B \u2014 Length Measurement", "Final Room Dimensions"])
        css = next(m.value for m in at.markdown if "m7_room_head h2" in m.value)
        self.assertIn("m7_room_head h2 {\n    font-size: clamp(1.4rem, 4.6vw, 2.4rem)", css)   # 2.4rem = 38px on desktop, responsive below
        self.assertIn("padding-bottom: 28px", css)                                              # heading -> subtitle spacing
        self.assertIn("m7_room_intro", css)

    def test_intro_is_the_short_sentence_and_the_about_expander_is_gone(self):
        at = page("3_results")
        intro = ("This page reports one room measurement experiment, carried out in two measurement stages on the same room: Wall A, from two "
                 "viewpoints, gives the width and height, and Wall B, from three viewpoints, gives the length.")
        self.assertIn(intro, [m.value for m in at.markdown])
        self.assertNotIn("About this measurement", [e.label for e in at.expander])
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
        self.assertLess(labels.index("Camera Calibration"), labels.index("How It Works \u2014 Methodology"))
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
        self.assertEqual([m.value for m in at.metric][:4], ["288 cm", "290 cm", "238 cm", "240 cm"])   # results unaffected

    def test_videos_render_in_two_groups_with_players(self):
        at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("video")), 4 + len(rd.WALL_B_VIDEOS))      # Wall A's four, then Wall B's six
        body = text(at)
        self.assertIn("View 1 \u2014 Left Side of Wall", body)
        self.assertIn("View 2 \u2014 Right Side of Wall", body)
        wall_a = section(at, "Wall A \u2014 Width and Height Measurement", "Wall B \u2014 Length Measurement")   # Wall B has its own recordings
        self.assertEqual([v for k, v in wall_a if k == "caption"].count("iPhone Air (Left Camera)"), 2)
        self.assertEqual([v for k, v in wall_a if k == "caption"].count("iPhone 17 Pro Max (Right Camera)"), 2)
        self.assertEqual(len([1 for k, _ in wall_a if k == "video"]), 4)
        self.assertLess(body.index("View 1"), body.index("View 2"))

    def test_missing_videos_fall_back_without_breaking_the_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rd, "video_status", side_effect=lambda: [(Path(tmp) / n, g, l, False) for g, n, l in rd.VIDEOS]), \
                    mock.patch.object(rd, "wall_b_video_status", side_effect=lambda: [(Path(tmp) / n, g, l, False) for g, n, l in rd.WALL_B_VIDEOS]):
                at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("video")), 0)
        self.assertEqual(len([i for i in at.info if "Video not available" in i.value]), 4)
        self.assertEqual([m.value for m in at.metric][:4], ["288 cm", "290 cm", "238 cm", "240 cm"])
        self.assertEqual(len(at.get("image")), len(rd.IMAGES))

    def test_existing_results_are_unchanged(self):
        at = page("3_results")
        body = text(at)
        for phrase in ("**Signed error:** −2.0 cm", "**Absolute error:** 2.0 cm", "**Absolute percentage error:** 0.70%", "**Absolute percentage error:** 0.82%", "266 to 310 cm", "219 to 257 cm"):
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
        # display images, the Wall A and Wall B playback copies (about 10 MB together) and the JSON results; far below GitHub's limits
        self.assertLess(sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()), 16_000_000)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class VisitorLocalGuideTests(unittest.TestCase):
    def test_guide_is_labelled_and_offers_no_upload(self):
        at = page("3_visitor")                                   # the guide is the inline Setup Instructions expander of the landing page
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("expander")), 1)
        self.assertIn("Setup Instructions", at.get("expander")[0].label)
        self.assertTrue(any("runs on your own computer" in m.value.lower() for m in at.markdown))
        self.assertEqual(len(at.get("file_uploader")), 0)
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.get("image")), 0)

    def test_guide_never_runs_the_estimator_or_loads_the_saved_experiment(self):
        from room_measurement import pipeline
        with mock.patch.object(pipeline, "run_estimation", side_effect=AssertionError("estimator must not run")) as run, \
                mock.patch.object(rd, "load_results", side_effect=AssertionError("saved results must not load")) as load, \
                mock.patch("cv2.VideoCapture", side_effect=AssertionError("no video access")):
            at = page("3_visitor")
        self.assertFalse(at.exception)
        run.assert_not_called()
        load.assert_not_called()

    def test_selection_pages_never_run_anything_either(self):
        with mock.patch.object(rd, "load_results", side_effect=AssertionError("selection must not load results")):
            for view in ("my_results", "run_own"):
                self.assertFalse(page(view).exception)


if __name__ == "__main__":
    unittest.main()


WALL_B = OUT / "wall_b_accepted_results.json"


class WallBDataTests(unittest.TestCase):
    def test_loader_reads_the_accepted_values_from_the_tracked_json(self):
        data, wb = json.loads(WALL_B.read_text()), rd.load_wall_b()
        self.assertEqual([(r["frame"], r["length_mm"]) for r in wb["per_frame"]], [(60, 3790.889), (150, 3811.971), (240, 3803.973)])
        self.assertEqual([rd.cm4(r["length_mm"]) for r in wb["per_frame"]], ["379.0889", "381.1971", "380.3973"])
        self.assertEqual(rd.cm4(wb["mean_mm"]), "380.2278")
        self.assertEqual(rd.cm(wb["mean_mm"]), "380")
        self.assertEqual((wb["tape_mm"], wb["difference_percent"], wb["tolerance_mm"]), (3937.0, -3.42, 5.0))
        self.assertEqual(wb["mean_mm"], data["accepted_results"]["mean_length_mm"])

    def test_the_tracked_json_agrees_with_the_regression_tests_reference_values(self):
        src = (MODULE / "visitor_workflow" / "tests" / "test_f8_35_real_three_view.py").read_text()
        for value in ("3790.889", "3811.971", "3803.973"):
            self.assertIn(value, src)
        self.assertIn("TOLERANCE_MM = 5.0", src)

    def test_inconsistent_or_missing_files_are_rejected_with_a_clear_message(self):
        data = json.loads(WALL_B.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(rd.ResultsUnavailable) as ctx:
                rd.load_wall_b(tmp)
            self.assertIn("not found", str(ctx.exception))
            for edit in (lambda d: d["accepted_results"].update(mean_length_mm=3900.0),
                         lambda d: d["comparison_with_tape_measurement"].update(mean_difference_percent=-1.0),
                         lambda d: d.update(schema="other")):
                bad = json.loads(json.dumps(data)); edit(bad)
                (Path(tmp) / rd.WALL_B_FILE).write_text(json.dumps(bad))
                with self.assertRaises(rd.ResultsUnavailable):
                    rd.load_wall_b(tmp)

    def test_six_wall_b_videos_are_three_labelled_viewpoint_pairs(self):
        self.assertEqual([(n, label) for n, _, label in rd.WALL_B_VIDEOS],
                         [(n, label) for n in (1, 2, 3) for label in ("iPhone Air (Left Camera)", "iPhone 17 Pro Max (Right Camera)")])
        self.assertEqual([name for _, name, _ in rd.WALL_B_VIDEOS],
                         [f"wall_b_viewpoint{n}_{side}.mp4" for n in (1, 2, 3) for side in ("left_air", "right_pro")])


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class WallBAndFinalDimensionsPageTests(unittest.TestCase):
    def test_wall_b_section_shows_the_accepted_values(self):
        at = page("3_results")
        self.assertFalse(at.exception)
        metrics = [m.value for m in at.metric]
        self.assertEqual(metrics[4:9], ["379.0889 cm", "381.1971 cm", "380.3973 cm", "380 cm", "393.7 cm"])
        body = text(at)
        for phrase in ("Frame 60", "Frame 150", "Frame 240", "380.2278 cm", "**Signed error:** −13.5 cm", "**Absolute error:** 13.5 cm", "**Absolute percentage error:** 3.42%"):
            self.assertIn(phrase, body)
        self.assertEqual([e.label for e in at.expander].count("How It Works \u2014 Methodology"), 1)

    def test_final_room_dimensions_cards(self):
        at = page("3_results")
        self.assertEqual([m.value for m in at.metric][9:12], ["\u2248 288 cm", "\u2248 380 cm", "\u2248 238 cm"])
        body = text(at)
        for phrase in ("From Wall A", "From Wall B", "Actual dimension (tape measurement): 290 cm", "Actual dimension (tape measurement): 393.7 cm", "Actual dimension (tape measurement): 240 cm"):
            self.assertIn(phrase, body)
        headers = [v for k, v in in_page_order(at) if k == "header"]
        self.assertLess(headers.index("Wall B \u2014 Length Measurement"), headers.index("Final Room Dimensions"))
        self.assertNotIn("volume:", body.lower())
        self.assertNotIn("These three values are", body)                                 # the former blue paragraph is gone

    def test_section_order_and_wall_a_content_is_unchanged(self):
        at = page("3_results")
        order = [h.value for h in at.header]
        self.assertEqual(order[1:], ["Wall A \u2014 Width and Height Measurement", "Wall B \u2014 Length Measurement", "Final Room Dimensions"])
        self.assertEqual(len(at.get("image")), 5)
        self.assertEqual(len(at.tabs), 3)
        body = text(at)
        for phrase in ("**Signed error:** −2.0 cm", "**Absolute error:** 2.0 cm", "**Absolute percentage error:** 0.70%", "**Absolute percentage error:** 0.82%", "266 to 310 cm", "219 to 257 cm"):
            self.assertIn(phrase, body)

    def test_wall_b_without_playback_copies_says_so_and_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(rd, "wall_b_video_status", side_effect=lambda: [(Path(tmp) / n, g, l, False) for g, n, l in rd.WALL_B_VIDEOS]):
                at = page("3_results")
        self.assertEqual(len(at.get("video")), 4)                                   # only Wall A's four
        notes = [i.value for i in at.info if "recordings are not included in this version" in i.value]
        self.assertEqual(len(notes), 3)
        body = text(at)
        for title in ("Viewpoint 1", "Viewpoint 2", "Viewpoint 3"):
            self.assertIn(title, body)

    def test_wall_b_playback_copies_render_in_three_viewpoint_groups(self):
        sample = rd.VIDEO_DIR / "left_air_wall_left.mp4"                           # any small committed H.264 file stands in for a playback copy
        with tempfile.TemporaryDirectory() as tmp:
            for _, name, _ in rd.WALL_B_VIDEOS:
                shutil.copy(sample, Path(tmp) / name)
            present = [(Path(tmp) / name, viewpoint, label, True) for viewpoint, name, label in rd.WALL_B_VIDEOS]
            with mock.patch.object(rd, "wall_b_video_status", return_value=present):
                at = page("3_results")
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("video")), 10)                                  # 4 Wall A + 6 Wall B
        body = text(at)
        order = [v for _, v in in_page_order(at)]
        start = order.index("Wall B \u2014 Length Measurement")
        positions = [order.index(f"##### Viewpoint {n}", start) for n in (1, 2, 3)]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len([i for i in at.info if "recordings are not included" in i.value]), 0)

    def test_missing_wall_b_file_does_not_break_wall_a_or_the_final_summary(self):
        with mock.patch.object(rd, "load_wall_b", side_effect=rd.ResultsUnavailable("Required results file not found: wall_b_accepted_results.json")):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertTrue(any("Wall B results are unavailable" in e.value for e in at.error))
        metrics = [m.value for m in at.metric]
        self.assertEqual(metrics[:4], ["288 cm", "290 cm", "238 cm", "240 cm"])
        self.assertEqual(metrics[-3:], ["\u2248 288 cm", "\u2014", "\u2248 238 cm"])

    def test_no_machine_learning_and_no_private_data_in_the_new_sections(self):
        source = PAGE.read_text()
        section = source[source.index("def _methodology"):source.index("# ---------------------------------------------------------------- page\n")]
        for forbidden in (".MOV", "VideoCapture", "annotations_backup", "data/room_measurement", "torch", "tensorflow", "sam2", "Method J", "2900", "2400"):
            self.assertNotIn(forbidden, section, forbidden)
        self.assertNotIn("method j", text(page("3_results")).lower())


WALL_B_FILES = [f"wall_b_viewpoint{n}_{side}.mp4" for n in (1, 2, 3) for side in ("left_air", "right_pro")]


class WallBPlaybackCopiesTests(unittest.TestCase):
    def test_the_six_committed_copies_are_small_h264_and_match_their_manifest(self):
        manifest = json.loads((rd.VIDEO_DIR / "wall_b_videos_manifest.json").read_text())
        self.assertEqual(sorted(manifest["videos"]), sorted(WALL_B_FILES))
        self.assertEqual(manifest["audio"], "removed")
        self.assertEqual(sorted(name for _, name, _ in rd.WALL_B_VIDEOS), sorted(WALL_B_FILES))
        total = 0
        for name, row in manifest["videos"].items():
            path = rd.VIDEO_DIR / name
            self.assertTrue(path.is_file(), name)
            size = path.stat().st_size
            total += size
            self.assertEqual(size, row["bytes"])
            self.assertLess(size, 3_000_000)
            n = row["viewpoint"]
            side = "left_air" if "left" in row["camera"] else "right_pro"
            self.assertEqual(name, f"wall_b_viewpoint{n}_{side}.mp4")
            self.assertEqual(row["source"], f"{side}_2_{n}.MOV")                        # viewpoint n, correct camera
            self.assertEqual(row["camera"], "left (iPhone Air)" if side == "left_air" else "right (iPhone 17 Pro Max)")
            data = path.read_bytes()
            self.assertIn(b"ftyp", data[:4096])
            self.assertIn(b"avc1", data[:100000])                                       # H.264
            self.assertNotIn(b"mp4a", data)                                             # no audio track
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), row["frames"])
            self.assertEqual((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (720, 1280))   # portrait
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), row["fps"], places=1)
            cap.release()
        self.assertEqual(total, manifest["total_bytes"])
        self.assertLess(total, 8_000_000)

    def test_the_manifest_publishes_no_private_hashes_or_paths(self):
        raw = (rd.VIDEO_DIR / "wall_b_videos_manifest.json").read_text()
        self.assertNotIn("sha256", raw)
        self.assertNotIn("/Users/", raw)
        self.assertNotIn("data/room_measurement", raw)

    def test_each_video_sits_in_its_viewpoint_group_left_before_right(self):
        shown = []
        real_video = __import__("streamlit").video
        with mock.patch("streamlit.video", side_effect=lambda data, *a, **k: (shown.append(Path(str(data)).name), real_video(data, *a, **k))[1]):
            at = page("3_results")
        self.assertFalse(at.exception)
        wall_a = [name for _, name, _ in rd.VIDEOS]
        self.assertEqual(shown, wall_a + WALL_B_FILES)                                   # Wall A first, then Wall B in viewpoint order
        wall_b = section(at, "Wall B \u2014 Length Measurement", "Final Room Dimensions")
        kinds = [(k, v) for k, v in wall_b if k == "video" or v.startswith("##### Viewpoint") or v in ("iPhone Air (Left Camera)", "iPhone 17 Pro Max (Right Camera)")]
        self.assertEqual(kinds, [item for n in (1, 2, 3) for item in (("markdown", f"##### Viewpoint {n}"),
                                                                       ("caption", "iPhone Air (Left Camera)"), ("video", ""),
                                                                       ("caption", "iPhone 17 Pro Max (Right Camera)"), ("video", ""))])
        self.assertNotIn("recordings are not included in this version", text(at))


def raw_error_inputs():
    """Estimates and references read straight from the tracked JSON files (cm), independent of display_data's own calculation."""
    est, val, wb = raw("estimation_result.json"), raw("validation.json"), raw("wall_b_accepted_results.json")
    estimates = [est["summary"]["width_mm"] / 10, wb["accepted_results"]["mean_length_mm"] / 10, est["summary"]["height_mm"] / 10]
    references = [val["width_mm"]["ground_truth_mm"] / 10, wb["comparison_with_tape_measurement"]["tape_measured_length_mm"] / 10,
                  val["height_mm"]["ground_truth_mm"] / 10]
    return estimates, references


def reference_statistics(estimates, references):
    errors = [e - r for e, r in zip(estimates, references)]
    n = len(errors)
    return dict(errors=errors, mae=sum(abs(x) for x in errors) / n, mape=sum(abs(x) / r * 100 for x, r in zip(errors, references)) / n,
                rmse=(sum(x * x for x in errors) / n) ** 0.5)


class ErrorAnalysisDataTests(unittest.TestCase):
    def test_a_hand_worked_example(self):
        results = dict(width=dict(estimate_mm=100.0, actual_mm=110.0), height=dict(estimate_mm=200.0, actual_mm=180.0))
        wb = dict(mean_mm=330.0, tape_mm=300.0)
        out = rd.error_analysis(results, wb)
        self.assertEqual([r["dimension"] for r in out["rows"]], ["Width", "Length", "Height"])
        self.assertEqual([r["wall"] for r in out["rows"]], ["Wall A", "Wall B", "Wall A"])
        self.assertEqual([round(r["error_cm"], 9) for r in out["rows"]], [-1.0, 3.0, 2.0])                  # estimate - reference, in cm
        self.assertEqual([round(r["abs_error_cm"], 9) for r in out["rows"]], [1.0, 3.0, 2.0])
        self.assertEqual([round(r["abs_percent"], 6) for r in out["rows"]], [round(100 / 11, 6), 10.0, round(2000 / 180, 6)])
        self.assertAlmostEqual(out["stats"]["mae_cm"], 2.0)
        self.assertAlmostEqual(out["stats"]["mape_percent"], (100 / 11 + 10.0 + 2000 / 180) / 3)
        self.assertAlmostEqual(out["stats"]["rmse_cm"], ((1 + 9 + 4) / 3) ** 0.5)
        self.assertEqual(out["stats"]["n"], 3)

    def test_values_match_an_independent_calculation_from_the_tracked_json(self):
        estimates, references = raw_error_inputs()
        want = reference_statistics(estimates, references)
        got = rd.error_analysis(rd.load_results(), rd.load_wall_b())
        for row, e, r, err in zip(got["rows"], estimates, references, want["errors"]):
            self.assertAlmostEqual(row["estimate_cm"], e, places=9)
            self.assertAlmostEqual(row["reference_cm"], r, places=9)
            self.assertAlmostEqual(row["error_cm"], err, places=9)
        self.assertAlmostEqual(got["stats"]["mae_cm"], want["mae"], places=9)
        self.assertAlmostEqual(got["stats"]["mape_percent"], want["mape"], places=9)
        self.assertAlmostEqual(got["stats"]["rmse_cm"], want["rmse"], places=9)
        # the accepted values, to the precision they are reported at
        self.assertAlmostEqual(got["stats"]["mae_cm"], 5.8316, places=3)
        self.assertAlmostEqual(got["stats"]["mape_percent"], 1.6504, places=3)
        self.assertAlmostEqual(got["stats"]["rmse_cm"], 7.9497, places=3)

    def test_full_precision_is_used_not_the_rounded_card_values(self):
        got = rd.error_analysis(rd.load_results(), rd.load_wall_b())
        self.assertNotEqual(got["rows"][0]["estimate_cm"], 288.0)                      # the card shows 288; the analysis uses 287.9566...
        self.assertAlmostEqual(got["rows"][0]["estimate_cm"], raw("estimation_result.json")["summary"]["width_mm"] / 10, places=9)
        self.assertNotEqual(got["rows"][1]["estimate_cm"], 380.0)

    def test_width_and_height_errors_agree_with_the_saved_validation(self):
        val = raw("validation.json")
        got = rd.error_analysis(rd.load_results(), rd.load_wall_b())["rows"]
        for row, key in ((got[0], "width_mm"), (got[2], "height_mm")):
            self.assertAlmostEqual(row["error_cm"] * 10, val[key]["error_mm"], places=6)
            self.assertAlmostEqual(row["abs_error_cm"] * 10, val[key]["abs_error_mm"], places=6)
            self.assertAlmostEqual(row["abs_percent"], abs(val[key]["error_percent"]), places=6)
        wb = rd.load_wall_b()
        self.assertAlmostEqual(got[1]["error_cm"] * 10, wb["difference_mm"], places=2)
        self.assertAlmostEqual(got[1]["abs_percent"], abs(wb["difference_percent"]), places=2)

    def test_the_statistics_follow_the_inputs_nothing_is_hard_coded(self):
        base = rd.error_analysis(rd.load_results(), rd.load_wall_b())
        wb = dict(rd.load_wall_b()); wb["mean_mm"] = wb["tape_mm"]                      # a perfect length
        changed = rd.error_analysis(rd.load_results(), wb)
        self.assertAlmostEqual(changed["rows"][1]["error_cm"], 0.0)
        self.assertLess(changed["stats"]["mae_cm"], base["stats"]["mae_cm"])
        self.assertLess(changed["stats"]["rmse_cm"], base["stats"]["rmse_cm"])
        self.assertGreaterEqual(base["stats"]["rmse_cm"], base["stats"]["mae_cm"])      # RMSE >= MAE always


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class ErrorAnalysisPageTests(unittest.TestCase):
    def test_table_columns_and_values(self):
        at = page("3_results")
        self.assertFalse(at.exception)
        self.assertIn("Validation & Error Analysis", [h.value for h in at.subheader])
        table = next(t for t in at.table if "Absolute percentage error" in list(t.value.columns))
        self.assertEqual(list(table.value.columns), ["Estimated dimension (cm)", "Actual dimension, tape measurement (cm)", "Signed error (cm)", "Absolute error (cm)", "Absolute percentage error"])
        self.assertEqual(list(table.value.index), ["Width (Wall A)", "Length (Wall B)", "Height (Wall A)"])
        estimates, references = raw_error_inputs()
        stats = reference_statistics(estimates, references)
        for i, (e, r, err) in enumerate(zip(estimates, references, stats["errors"])):
            row = table.value.iloc[i]
            self.assertEqual(row["Estimated dimension (cm)"], f"{e:.2f}")
            self.assertEqual(row["Actual dimension, tape measurement (cm)"], f"{r:.2f}")
            self.assertEqual(row["Signed error (cm)"], f"{err:+.2f}".replace("-", "\u2212"))
            self.assertEqual(row["Absolute error (cm)"], f"{abs(err):.2f}")
            self.assertEqual(row["Absolute percentage error"], f"{abs(err) / r * 100:.2f}%")
        self.assertEqual(table.value.iloc[0]["Estimated dimension (cm)"], "287.96")
        self.assertEqual(table.value.iloc[1]["Signed error (cm)"], "\u221213.47")

    def test_three_summary_statistic_cards(self):
        at = page("3_results")
        stats = reference_statistics(*raw_error_inputs())
        metrics = [(m.label, m.value) for m in at.metric][12:15]
        self.assertEqual(metrics, [("Mean Absolute Error (MAE)", f"{stats['mae']:.2f} cm"),
                                   ("Mean Absolute Percentage Error (MAPE)", f"{stats['mape']:.2f} %"),
                                   ("Root Mean Square Error (RMSE)", f"{stats['rmse']:.2f} cm")])
        self.assertEqual([m.value for m in at.metric][12:15], ["5.83 cm", "1.65 %", "7.95 cm"])

    def test_the_three_final_dimension_cards_are_unchanged_and_come_first(self):
        at = page("3_results")
        self.assertEqual([m.value for m in at.metric][9:12], ["\u2248 288 cm", "\u2248 380 cm", "\u2248 238 cm"])
        order = in_page_order(at)
        kinds = [(k, v) for k, v in order]
        pos = {name: next(i for i, (k, v) in enumerate(kinds) if v == name or (name == "cards" and v == "\u2248 380 cm"))
               for name in ("cards", "Validation & Error Analysis", "##### Estimated vs. actual dimensions", "Error calculation formulas", "##### Interpretation")}
        self.assertLess(pos["cards"], pos["Validation & Error Analysis"])
        self.assertLess(pos["Validation & Error Analysis"], pos["##### Estimated vs. actual dimensions"])
        self.assertLess(pos["##### Estimated vs. actual dimensions"], pos["Error calculation formulas"])
        self.assertLess(pos["Error calculation formulas"], pos["##### Interpretation"])

    def test_grouped_bar_chart_compares_estimated_and_measured_in_cm(self):
        at = page("3_results")
        charts = at.get("vega_lite_chart")
        self.assertEqual(len(charts), 1)
        spec = json.loads(charts[0].proto.spec)
        enc = spec["layer"][0]["encoding"]
        self.assertEqual(spec["layer"][0]["mark"]["type"], "bar")
        self.assertIn("xOffset", enc)                                                     # grouped, not stacked
        self.assertEqual(enc["x"]["sort"], ["Width", "Length", "Height"])
        self.assertEqual(enc["y"]["title"], "Dimension (cm)")
        self.assertEqual(enc["color"]["scale"]["domain"], ["Estimated dimension", "Actual dimension (tape measurement)"])
        self.assertEqual(enc["color"]["scale"]["range"][0], "#C9363E")                    # the project's red accent
        import pyarrow as pa                                                              # Streamlit ships chart data as an Arrow table
        dataset = charts[0].proto.datasets[0]
        self.assertEqual(spec["data"]["name"], dataset.name)
        values = pa.ipc.open_stream(dataset.data.data).read_all().to_pylist()
        estimates, references = raw_error_inputs()
        got = {(v["Dimension"], v["Series"]): v["Centimetres"] for v in values}
        self.assertEqual(len(got), 6)
        for name, e, r in zip(("Width", "Length", "Height"), estimates, references):
            self.assertAlmostEqual(got[(name, "Estimated dimension")], e, places=9)
            self.assertAlmostEqual(got[(name, "Actual dimension (tape measurement)")], r, places=9)

    def test_formulas_expander_is_collapsed_and_defines_all_six_quantities(self):
        at = page("3_results")
        exp = next(e for e in at.expander if e.label == "Error calculation formulas")
        self.assertFalse(exp.proto.expanded)
        latex = [l.value for l in exp.get("latex")]
        self.assertEqual(len(latex), 6)
        joined = " ".join(latex)
        for token in ("e_i", "|e_i|", "\\mathrm{APE}", "\\mathrm{MAE}", "\\mathrm{MAPE}", "\\mathrm{RMSE}", "\\sqrt"):
            self.assertIn(token, joined)
        body = " ".join(m.value for m in exp.markdown)
        for phrase in ("Signed error", "Absolute error", "Absolute percentage error", "Mean absolute error", "Mean absolute percentage error", "Root mean square error"):
            self.assertIn(phrase, body)

    def test_interpretation_and_conclusion_use_the_computed_values(self):
        at = page("3_results")
        estimates, references = raw_error_inputs()
        st_ = reference_statistics(estimates, references)
        rows = list(zip(("width", "length", "height"), estimates, references, st_["errors"]))
        length_err = abs(st_["errors"][1])
        wall_a_err = max(abs(st_["errors"][0]), abs(st_["errors"][2]))
        body = text(at)
        for phrase in ("All three estimated dimensions are slightly smaller than their tape measurements.",
                       f"differ from the measured values by approximately {wall_a_err:.0f} cm, with percentage errors below 1%.",
                       "Both tape measurements fall within their respective approximate 95% uncertainty intervals.",
                       f"The length has the largest observed error of approximately {length_err:.1f} cm ({length_err / references[1] * 100:.2f}%).",
                       f"The overall mean absolute error is {st_['mae']:.2f} cm, with a mean absolute percentage error of {st_['mape']:.2f}%.",
                       f"The RMSE of {st_['rmse']:.2f} cm reflects the greater influence of the length measurement error."):
            self.assertIn(phrase, body)
        self.assertIn("The length has the largest observed error of approximately 13.5 cm (3.42%).", body)
        self.assertIn("The overall mean absolute error is 5.83 cm, with a mean absolute percentage error of 1.65%.", body)
        self.assertIn("Conclusion", body)
        self.assertIn("Using calibrated stereo video recordings captured from five viewpoints, the room dimensions were estimated to be "
                      "approximately 288 cm wide, 380 cm long and 238 cm high, with a mean absolute percentage error of 1.65% across the three "
                      "measured dimensions.", body)
        self.assertIn("These results demonstrate how classical stereo vision and geometric reconstruction can be applied to video recordings to estimate real-world room dimensions.", body)
        for gone in ("only these three comparisons", "no combined uncertainty interval is claimed", "guaranteed-accuracy"):
            self.assertNotIn(gone, body)

    def test_without_wall_b_the_analysis_is_replaced_by_a_clear_note(self):
        with mock.patch.object(rd, "load_wall_b", side_effect=rd.ResultsUnavailable("missing")):
            at = page("3_results")
        self.assertFalse(at.exception)
        self.assertNotIn("Validation & Error Analysis", [h.value for h in at.subheader])
        self.assertEqual(len(at.get("vega_lite_chart")), 0)
        self.assertTrue(any("error analysis needs both" in i.value for i in at.info))
        self.assertEqual([m.value for m in at.metric][-3:], ["\u2248 288 cm", "\u2014", "\u2248 238 cm"])

    def test_the_page_follows_changed_results(self):
        wb = dict(rd.load_wall_b()); wb["mean_mm"] = wb["tape_mm"] - 100.0                 # length 10 cm short
        with mock.patch.object(rd, "load_wall_b", return_value=wb):
            at = page("3_results")
        table = next(t for t in at.table if "Absolute percentage error" in list(t.value.columns))
        self.assertEqual(table.value.iloc[1]["Signed error (cm)"], "\u221210.00")
        self.assertNotEqual([m.value for m in at.metric][12:15], ["5.83 cm", "1.65 %", "7.95 cm"])


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class PresentationRevisionTests(unittest.TestCase):
    """The presentation refinements: shaded experiment containers, a red-bordered Camera Calibration expander, no blue boxes under Wall B or at the end."""

    @classmethod
    def setUpClass(cls):
        cls.at = page("3_results")
        cls.blocks = keyed_blocks(cls.at)

    def test_page_is_laid_out_calibration_then_wall_a_wall_b_final(self):
        keys = [k for k in self.blocks if k in ("m7_room_calibration", "m7_room_wall_a", "m7_room_wall_b", "m7_room_final")]
        self.assertEqual(keys, ["m7_room_calibration", "m7_room_wall_a", "m7_room_wall_b", "m7_room_final"])

    def test_camera_calibration_is_a_collapsed_expander_in_its_own_container(self):
        items = items_in(self.blocks["m7_room_calibration"])
        self.assertEqual([v for k, v in items if k == "expander"], ["Camera Calibration"])
        cal = next(e for e in self.at.expander if e.label == "Camera Calibration")
        self.assertFalse(cal.proto.expanded)
        self.assertNotIn("About this measurement", [e.label for e in self.at.expander])

    def test_wall_a_container_holds_the_whole_wall_a_experiment(self):
        block = self.blocks["m7_room_wall_a"]
        items = items_in(block)
        self.assertEqual(items[0], ("header", "Wall A \u2014 Width and Height Measurement"))
        self.assertIn(("markdown", "Two viewpoints, four video recordings. This stage estimates the room's width and height."), items)
        self.assertEqual(len([1 for k, _ in items if k == "video"]), 4)
        self.assertEqual([v for k, v in items if k == "metric"], ["288 cm", "290 cm", "238 cm", "240 cm"])
        self.assertIn(("subheader", "What the cameras saw"), items)
        self.assertEqual(len(block.get("tab")) if hasattr(block, "get") else 3, 3)
        expanders = [v for k, v in items if k == "expander"]
        self.assertEqual(expanders, [])                                                # the methodology is one unified expander before Wall A
        self.assertFalse(any(k == "info" for k, _ in items))                            # the blue "How to read these numbers" box is gone
        self.assertFalse(any("Wall B" in v for k, v in items if k == "header"))

    def test_wall_b_container_holds_the_whole_wall_b_experiment(self):
        items = items_in(self.blocks["m7_room_wall_b"])
        self.assertEqual(items[0], ("header", "Wall B \u2014 Length Measurement"))
        self.assertIn(("markdown", "Three viewpoints, six video recordings. This stage estimates the room's length."), items)
        self.assertEqual(len([1 for k, _ in items if k == "video"]), 6)
        self.assertEqual([v for k, v in items if k == "metric"], ["379.0889 cm", "381.1971 cm", "380.3973 cm", "380 cm", "393.7 cm"])
        self.assertEqual([v for k, v in items if k == "expander"], [])

    def test_final_container_holds_the_cards_the_analysis_and_the_conclusion(self):
        items = items_in(self.blocks["m7_room_final"])
        self.assertEqual(items[0], ("header", "Final Room Dimensions"))
        self.assertEqual([v for k, v in items if k == "metric"],
                         ["\u2248 288 cm", "\u2248 380 cm", "\u2248 238 cm", "5.83 cm", "1.65 %", "7.95 cm"])
        kinds = [k for k, _ in items]
        self.assertEqual(kinds.count("table"), 1)
        self.assertEqual(kinds.count("vega_lite_chart"), 1)
        self.assertIn(("expander", "Error calculation formulas"), items)
        texts = [v for _, v in items]
        self.assertLess(texts.index("##### Interpretation"), texts.index("##### Conclusion"))
        self.assertEqual(items[-1][0], "markdown")                                       # the conclusion is the last thing on the page body

    def test_the_removed_blue_boxes_are_gone(self):
        infos = [i.value for i in self.at.info]
        for gone in ("How to read the tape comparison", "tolerance means", "These three values are"):
            self.assertFalse(any(gone in v for v in infos), gone)
        for key in ("m7_room_wall_b", "m7_room_final"):
            self.assertEqual([k for k, _ in items_in(self.blocks[key]) if k in ("info", "warning", "error")], [], key)

    def test_standalone_technical_and_limitation_expanders_are_gone(self):
        labels = [e.label for e in self.at.expander]
        for gone in ("Wall B limitations", "Limitations", "Technical details: reproducibility tolerance", "Technical details: joint stereo pose refinement",
                     "Technical details: how the uncertainty interval is built", "Technical details: what was manual and what was automatic"):
            self.assertNotIn(gone, labels, gone)
        body = text(self.at)
        for gone in ("reproducibility", "How to read these numbers", "not documented", "measurement conditions", "H.264", "without audio"):
            self.assertNotIn(gone, body, gone)

    def test_video_captions_and_tape_terminology(self):
        captions = [c.value for c in self.at.caption]
        self.assertEqual(captions.count("Resized copies of my original stereo video recordings for web playback."), 2)   # Wall A and Wall B
        body = text(self.at) + " ".join(m.label for m in self.at.metric)
        for term in ("Estimated dimension", "Actual dimension (tape measurement)", "Signed error", "Absolute error", "Absolute percentage error"):
            self.assertIn(term, body, term)
        self.assertNotIn("Tape reference", body)
        self.assertNotIn("Measured reference", body)

    def test_methodology_content_numbers_and_equations(self):
        how = next(e for e in self.at.expander if e.label == "How It Works \u2014 Methodology")
        body = " ".join(m.value for m in how.markdown)
        latex = [l.value for l in how.get("latex")]
        self.assertEqual(len(latex), len(set(latex)))                                       # no equation is repeated
        self.assertEqual(len(latex), 13)
        joined = " ".join(latex)
        for gone in ("Sampson", "operatorname{MAD}", "1.4826", "4.685", "(1 - ", "d_S"):    # the detailed robust-fitting equations are not shown
            self.assertNotIn(gone, joined, gone)
        for token in ("[T]_{\\times}", "\\arg\\min", "\\sqrt{\\sum_k", "\\pm 1.96", "\\begin{bmatrix}"):
            self.assertIn(token, joined, token)
        for phrase in ("Sampson distance and Tukey biweight", "eight-point algorithm", "Direct Linear Transform (DLT)", "Principal Component Analysis (PCA)",
                       "Iteratively Reweighted Least Squares (IRLS)", "1,500 iterations, three-point samples and a 40 mm inlier threshold",
                       "Frames 60, 150 and 240", "A total of 805 feature tracks", "baseline, is 225 mm", "approximately 2.1 cm",
                       "No confidence interval is calculated for Wall B", "Camera calibration contributes the largest uncertainty"):
            self.assertIn(phrase, body, phrase)
        tables = {tuple(t.value.columns): t.value for t in how.table}
        pose = tables[("Result",)]
        self.assertEqual(list(pose["Result"]), ["18,600", "805", "0.27 px", "95.4%"])
        unc = next(v for c, v in tables.items() if "Width (cm)" in c)
        self.assertEqual(list(unc["Width (cm)"]), ["1.9", "1.9", "3.0", "10.7"])
        self.assertEqual(list(unc["Height (cm)"]), ["1.7", "1.6", "1.7", "9.4"])
        for claimed in ("neural", "deep learning", "yolo", "bundle adjustment"):
            self.assertNotIn(claimed, body.lower(), claimed)

    def test_objective_setup_and_validation_appear_once_each(self):
        sub = [v for k, v in in_page_order(self.at) if k == "markdown"]
        for heading in ("##### Objective", "##### Experimental setup"):
            self.assertEqual(sub.count(heading), 1, heading)
        self.assertEqual(sub.count("##### Validation"), 0)
        self.assertEqual([h.value for h in self.at.subheader].count("Validation & Error Analysis"), 1)
        body = text(self.at)
        self.assertIn("Estimate the width, length and height of a room using calibrated stereo video recordings and classical computer vision techniques.", body)
        for fact in ("iPhone Air as the left camera", "iPhone 17 Pro Max as the right camera", "baseline of 225 mm", "five viewpoints in total",
                     "ten video recordings that form five stereo pairs"):
            self.assertIn(fact, body, fact)
        order = [v for _, v in in_page_order(self.at)]
        self.assertLess(order.index("##### Objective"), order.index("##### Experimental setup"))
        self.assertLess(order.index("##### Experimental setup"), order.index("Camera Calibration"))
        self.assertLess(order.index("Camera Calibration"), order.index("How It Works \u2014 Methodology"))
        self.assertLess(order.index("How It Works \u2014 Methodology"), order.index("Wall A \u2014 Width and Height Measurement"))
        for gone in ("two experiments", "second experiment", "both experiments", "two stereo-video experiments"):
            self.assertNotIn(gone, body, gone)

    def test_page_order_objective_setup_calibration_methodology_then_the_three_containers(self):
        keys = [k for k in self.blocks if k in ("m7_room_objective", "m7_room_calibration", "m7_room_methodology", "m7_room_wall_a", "m7_room_wall_b", "m7_room_final")]
        self.assertEqual(keys, ["m7_room_objective", "m7_room_calibration", "m7_room_methodology", "m7_room_wall_a", "m7_room_wall_b", "m7_room_final"])

    def test_no_developer_terminology_on_the_page(self):
        body = text(self.at).lower()
        for term in ("measured_with_warnings", "regression", "usable but not clean", "repository", "method j", "about this measurement"):
            self.assertNotIn(term, body, term)

    def test_styles_are_scoped_and_shared_by_the_three_containers(self):
        css = next(m.value for m in self.at.markdown if "m7_room_head h2" in m.value)
        self.assertIn(".st-key-m7_room_wall_a, .st-key-m7_room_wall_b, .st-key-m7_room_final {", css)
        self.assertIn("border-radius: 16px", css)
        self.assertIn('.st-key-m7_room_calibration [data-testid="stExpander"] { border: 1px solid #C9363E;', css)
        self.assertIn('.st-key-m7_room_calibration [data-testid="stExpander"] summary p { font-weight: 700; }', css)
        for rule in css.split("}"):
            if "m7_room_calibration" in rule:
                self.assertNotIn("stMetric", rule)                                       # the calibration styling touches nothing else
        self.assertNotIn("m7_room_recordings_b {\n    background", css)                  # no redundant nested backgrounds
