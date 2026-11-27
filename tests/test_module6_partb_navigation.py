"""Module 6 Part B saved experiment results content (AppTest, no browser)."""
import base64
import csv
import json
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from Modules.Module_6.Part_B import results_data

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module6.py"
PART_B = ROOT / "Modules" / "Module_6" / "Part_B"

def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


SECTIONS = ["1. Experimental Setup", "2. Corresponding Feature Points", "3. Structure from Motion Reconstruction",
            "4. Estimated Object Boundary", "5. Experimental Validation", "6. Conclusion"]
DROPDOWNS = ["View Coordinate Centering Calculations", "View Orthographic Constraints",
             "View Coordinate Alignment and Scaling Calculations", "View Validation Calculations"]
SUBSECTIONS = ["Camera Viewpoints", "Camera Parameters", "Corresponding Landmark Close-ups",
               "Orthographic Projection Model", "Measurement Matrix and Coordinate Centering",
               "Tomasi–Kanade Factorization", "Singular Value Decomposition (SVD)", "Metric Upgrade",
               "Coordinate Alignment", "Physical Scaling", "Reconstructed Points", "Estimated Boundary",
               "Euclidean Distance", "Percentage Error"]


def open_results():
    app = new_app()
    click(app, "m6_open_results")
    click(app, "m6_res_b")
    return app


class PartBNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = open_results()

    def test_results_header_and_six_sections(self):
        app = self.app
        self.assertEqual(app.session_state["module6_section"], "b_results")
        self.assertEqual([t.value for t in app.title], ["Structure from Motion Reconstruction"])
        self.assertEqual([h.value for h in app.header], ["My Experiment & Results", *SECTIONS])
        self.assertIn("MODULE 6 — PART B", [m.value for m in app.markdown])
        self.assertFalse(app.warning)
        self.assertFalse(app.error)

    def test_metric_cards_use_the_saved_results(self):
        self.assertEqual([(m.label, m.value) for m in self.app.metric], [
            ("Camera Viewpoints", "4"), ("Corresponding Points", "8"), ("Actual Cover Size", "14.5 × 21.7 cm")])

    def test_dropdowns_are_present_and_collapsed(self):
        self.assertEqual([e.label for e in self.app.expander], DROPDOWNS)
        self.assertFalse(any(e.proto.expanded for e in self.app.expander))

    def test_subsections_are_chronological(self):
        self.assertEqual([s.value for s in self.app.subheader], SUBSECTIONS)

    def test_main_equations_are_outside_the_dropdowns(self):
        inside = [e.value for expander in self.app.expander for e in expander.latex]
        self.assertEqual((len(self.app.latex) - len(inside), len(inside)), (10, 7))

    def test_validation_table_and_conclusion(self):
        tables = [t.value for t in self.app.table]
        validation = next(t for t in tables if t.index.name == "Measurement")
        self.assertEqual(list(validation.index), ["Top edge", "Bottom edge", "Left edge", "Right edge"])
        self.assertEqual(list(validation["Error (%)"]), ["9.60", "9.98", "1.63", "1.46"])
        conclusion = self.app.markdown[-1].value
        self.assertIn("individual edge errors ranged from 1.46% to 9.98%.", conclusion)

    def test_only_the_retained_tables_graphs_and_equations_are_shown(self):
        app = self.app
        self.assertEqual([t.value.index.name for t in app.table],
                         ["Parameter", "Point", "Measurement"])
        self.assertEqual(len(app.get("plotly_chart")), 2)
        # four photographs, the four-view correspondence figure and landmark close-ups
        self.assertEqual(len(app.get("image")), 6)
        latex = " ".join(e.value for e in app.latex)
        for retained in (r"x_{fp} &= \mathbf{i}_f^{T}\mathbf{s}_p", r"\bar{x}_f = \frac{1}{8}", r"\tilde{W} \approx MS",
                         r"\tilde{W} = U\Sigma V^{T} \approx U_3\Sigma_3V_3^{T}", r"M = \hat{M}Q", r"L = QQ^{T}",
                         r"\mathbf{b}_p = B(", r"\mathbf{p}_p = s\,\mathbf{b}_p", r"d_{ab} = \sqrt", r"\text{Error}"):
            self.assertIn(retained, latex)
        for removed in (r"\arg\min", r"\delta", r"\varepsilon", r"\mathbf{r}_{fp}", r"B \approx", "L = 10^",
                        r"d_{\mathrm{ref}} = ", r"(\hat{M}Q)(Q^{-1}", r"10^{-2}", r"\sigma_", r"\Sigma_3^{1/2}",
                        r"\mathbf{s}_1", r"\bar{w} &=", r"\mathbb{R}"):
            self.assertNotIn(removed, latex)
        self.assertEqual(latex.count(r"\begin{bmatrix}"), 2)  # aligned P1 and its coordinates in cm
        text = " ".join([m.value for m in app.markdown] + [c.value for c in app.caption]).lower()
        for removed in ("aperture", "reprojection", "diagonal p", "plane deviation", "approximation error"):
            self.assertNotIn(removed, text)
        self.assertNotIn("f/1.78", " ".join(t.value.to_string() for t in app.table))
        for equation in app.latex:
            self.assertIn(r"\boxed", equation.value)

    def test_each_graph_is_grouped_with_its_text_and_observation(self):
        def walk(node):
            yield node
            for child in getattr(node, "children", {}).values():
                yield from walk(child)

        for key, title in (("top", "Estimated Object Boundary — Top View"), ("3d", "Scaled 3D Reconstruction")):
            group = next(n for n in walk(self.app.main) if getattr(getattr(n, "proto", None), "id", "")
                         .endswith(f"m6bres_graph_{key}"))
            items = [n for n in walk(group) if n is not group]
            self.assertEqual([n.type for n in items],
                             ["markdown", "markdown", "plotly_chart", "caption", "divider", "markdown"])
            self.assertEqual(items[0].value, f"##### {title}")
            self.assertTrue(items[-1].value.startswith("**Observation**"))

    def test_landmark_closeups_follow_the_comparison_in_section_2(self):
        def walk(node):
            yield node
            for child in getattr(node, "children", {}).values():
                yield from walk(child)

        nodes = list(walk(self.app.main))
        order = [n for n in nodes if n.type in ("header", "subheader", "image", "imgs", "markdown", "expander", "expandable")]
        heading = next(i for i, n in enumerate(order) if n.type == "subheader"
                       and n.value == "Corresponding Landmark Close-ups")
        self.assertIn(order[heading - 1].type, ("image", "imgs", "markdown"))
        self.assertTrue(order[heading + 1].value.startswith("The close-ups show the same eight physical landmarks"))
        self.assertIn("origin at the upper-left", order[heading + 2].value)
        self.assertIn(order[heading + 3].type, ("image", "imgs"))
        headers = [(i, n.value) for i, n in enumerate(order) if n.type == "header"]
        start = next(i for i, v in headers if v == SECTIONS[1])
        end = next(i for i, v in headers if v == SECTIONS[2])
        self.assertTrue(start < heading < end)

    def test_corresponding_points_text_and_figure_in_section_2(self):
        text = " ".join(m.value for m in self.app.markdown)
        self.assertIn("Eight corresponding points were used: four book cover corners (P1–P4) and four interior "
                      "feature points (P5–P8). The four corners were selected in View 1, and the remaining "
                      "corresponding points were identified automatically. All corresponding points were verified, and "
                      "missing or inaccurate positions were manually selected or corrected before reconstruction.", text)
        captions = " ".join([c.value for c in self.app.caption]
                            + [c for i in self.app.get("image") for c in (img.caption for img in i.proto.imgs)])
        self.assertIn("Corresponding points P1–P8 across four viewpoints.", captions)
        self.assertNotIn("Cyan", captions)
        visible = " ".join([text, captions] + [t.value.to_string() for t in self.app.table]
                           + [e.label for e in self.app.expander]).lower()
        for word in ("predict", "sift", "ransac", "homography", "15 pixels"):
            self.assertNotIn(word, visible)
        for word in ("assisted visual identification", "AI", "assistant"):
            self.assertNotIn(word, text)

    def test_no_values_from_the_previous_experiment(self):
        text = " ".join([m.value for m in self.app.markdown] + [c.value for c in self.app.caption]
                        + [e.value for e in self.app.latex] + [t.value.to_string() for t in self.app.table])
        for old in ("99.918", "2.8605", "216.946", "393.504", "0.0061004657", "1.3235", "2.4006", "8234.41",
                    "0.0061112", "13.266", "16.071", "10.84"):
            self.assertNotIn(old, text)
        self.assertNotIn("manually identified", text.lower())

    def test_back_navigation_and_no_bottom_back_button(self):
        app = open_results()
        self.assertEqual([b.key for b in app.button if b.label in ("Back", "Go back")], ["nav_arrow_button"])
        click(app, "nav_arrow_button")
        self.assertEqual(app.session_state["module6_section"], "my_results")

    def test_experiment_assets_present(self):
        for name in ("part_b_correspondences.csv", "part_b_correspondence_views.png", "part_b_landmark_closeups.png",
                     "part_b_reconstructed_points_cm.csv", "part_b_measurements.csv"):
            self.assertTrue((results_data.REVISED / name).is_file(), name)
        # the results page's dataset is the only saved experiment
        self.assertEqual(sorted(p.name for p in (PART_B / "output").iterdir() if not p.name.startswith(".")),
                         ["revised"])


def saved_rows(name):
    with (results_data.REVISED / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class PartBSavedResultTests(unittest.TestCase):
    """The page's values come from the approved coordinates and agree with the saved results."""

    @classmethod
    def setUpClass(cls):
        cls.data = results_data.load()

    def test_measurement_matrix_follows_the_approved_correspondences(self):
        result = self.data["result"]
        rows = saved_rows("part_b_correspondences.csv")
        raw = np.array([[float(r[f"view{v}_{a}"]) for r in rows] for v in range(1, 5) for a in "xy"])
        np.testing.assert_array_equal(result.raw, raw)
        np.testing.assert_allclose(result.centered, raw - raw.mean(axis=1, keepdims=True), atol=1e-12)
        np.testing.assert_allclose(result.singular_values, np.linalg.svd(result.centered, compute_uv=False),
                                   atol=1e-9)

    def test_points_scale_and_measurements_match_the_saved_results(self):
        d = self.data
        cm = [[float(r[f"{a}_cm"]) for a in "XYZ"] for r in saved_rows("part_b_reconstructed_points_cm.csv")]
        np.testing.assert_allclose(d["cm"], cm, atol=1e-9)
        w, h = d["mean_width"], d["mean_height"]
        self.assertAlmostEqual(d["scale"], (14.5 * w + 21.7 * h) / (w * w + h * h), places=15)
        np.testing.assert_allclose(d["cm"], d["scale"] * d["book"], atol=1e-12)
        for saved, m in zip(saved_rows("part_b_measurements.csv"), d["measurements"]):
            self.assertEqual(saved["edge"], m["label"])
            self.assertAlmostEqual(float(saved["reconstructed_cm"]), m["reconstructed"], places=9)
            self.assertAlmostEqual(float(saved["error_percent"]), m["error"], places=9)
            length = np.linalg.norm(d["cm"][m["b"]] - d["cm"][m["a"]])
            self.assertAlmostEqual(m["reconstructed"], length, places=10)

    def test_top_view_and_3d_graphs_use_the_saved_points(self):
        app = open_results()
        for chart in app.get("plotly_chart"):
            traces = {t["name"]: t for t in json.loads(chart.proto.spec)["data"]}
            corners = next(v for k, v in traces.items() if "P1–P4" in k)
            landmarks = traces["Landmarks (P5–P8)"]
            self.assertEqual(len(decode(corners["x"])), 5)
            np.testing.assert_allclose(decode(landmarks["x"]), self.data["cm"][4:, 0])
            if "Ideal rectangle (reference)" in traces:
                reference = traces["Ideal rectangle (reference)"]
                self.assertEqual(sorted(set(np.abs(decode(reference["x"])))), [7.25])
                self.assertEqual(sorted(set(np.abs(decode(reference["y"])))), [10.85])


def decode(values):
    """Plotly serializes numeric arrays either as lists or as base64 typed arrays."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
    return np.array(values, float)


class PartBWorkedExampleTests(unittest.TestCase):
    """The displayed scale factor and P1 conversion reproduce the saved results."""

    def test_displayed_scaling_and_p1_conversion(self):
        d = results_data.load()
        w, h = round(d["mean_width"], 4), round(d["mean_height"], 4)
        self.assertAlmostEqual((14.5 * w + 21.7 * h) / (w * w + h * h), d["scale"], places=6)
        p1 = round(d["scale"], 10) * np.round(d["book"][0], 4)
        np.testing.assert_array_equal(np.round(p1, 3), np.round(d["cm"][0], 3))


class PartBSubmissionPhotoTests(unittest.TestCase):
    def test_photographs_are_full_size_without_location_metadata(self):
        for name in results_data.PHOTO_NAMES:
            path = results_data.PHOTOS / name
            with Image.open(path) as image:
                self.assertEqual(image.size, (5712, 4284), name)
                exif = image.getexif()
                self.assertEqual(exif.get(274), 6, name)
                self.assertNotIn(0x8825, exif, name)
                self.assertEqual(exif.get(272), "iPhone 17 Pro Max", name)
                self.assertNotIn("xmp", image.info, name)


if __name__ == "__main__":
    unittest.main()
