"""Module 6 Part B final visual refinements: 3D cloud styling, camera, axis titles, projections, Step 3 wording."""
import json
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_module6_partb_backend import cube_files  # noqa: E402
from test_module6_partb_visitor_page import matched_app, reconstructed_app  # noqa: E402
from Modules.Module_6.Part_B import visitor_reconstruction as vr  # noqa: E402

PAGE = ROOT / "Pages" / "app_module6.py"
NEW_SENTENCE = ("Without a scale, the object's shape and proportions are estimated in arbitrary units rather than "
                "centimeters or millimeters. Reconstruction accuracy depends on the images and camera geometry.")


def specs(app):
    return [json.loads(c.proto.spec) for c in app.get("plotly_chart")]


class VisualRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = cube_files()
        cls.app = reconstructed_app(cls.files)
        cls.result = cls.app.session_state["m6b_result"]
        cls.scaled = reconstructed_app(cls.files, scale=True)

    def test_3d_marker_and_contrast(self):
        trace = specs(self.app)[0]["data"][0]
        self.assertEqual(trace["type"], "scatter3d")
        self.assertEqual(trace["marker"]["size"], 5)
        self.assertEqual(trace["marker"]["opacity"], 1.0)
        self.assertGreaterEqual(trace["marker"]["line"]["width"], 1)
        scene = specs(self.app)[0]["layout"]["scene"]
        for axis in ("xaxis", "yaxis", "zaxis"):
            self.assertTrue(scene[axis]["showline"])
            self.assertEqual(scene[axis]["backgroundcolor"], scene["xaxis"]["backgroundcolor"])
            self.assertIn("color", scene[axis]["tickfont"])
            self.assertIn("gridcolor", scene[axis])

    def test_3d_camera_and_scaling(self):
        scene = specs(self.app)[0]["layout"]["scene"]
        self.assertEqual(scene["camera"]["eye"], {"x": 1.5, "y": -1.5, "z": 1.0})
        self.assertEqual(scene["camera"]["up"], {"x": 0, "y": 0, "z": 1})
        self.assertEqual(scene["aspectmode"], "data")
        self.assertGreater(scene["camera"]["eye"]["z"], 0.5)  # not flattened

    def test_axis_titles_and_unit_captions(self):
        for app in (self.app, self.scaled):
            scene = specs(app)[0]["layout"]["scene"]
            self.assertEqual([scene[a]["title"]["text"] for a in ("xaxis", "yaxis", "zaxis")], ["X", "Y", "Z"])
        self.assertIn("Coordinate units: arbitrary reconstruction units.", [c.value for c in self.app.caption])
        self.assertIn("Coordinate units: cm. Scale is approximate.", [c.value for c in self.scaled.caption])
        self.assertTrue(any(c.value.startswith("Drag to rotate and scroll to zoom") for c in self.app.caption))

    def test_projection_styling(self):
        top, side = specs(self.app)[1], specs(self.app)[2]
        for spec, h, v in ((top, "X", "Y"), (side, "X", "Z")):
            marker = spec["data"][0]["marker"]
            self.assertEqual(marker["size"], 8)
            self.assertGreaterEqual(marker["line"]["width"], 1)
            self.assertIn(f"{h} (arbitrary reconstruction units)", spec["layout"]["xaxis"]["title"]["text"])
            self.assertIn(f"{v} (arbitrary reconstruction units)", spec["layout"]["yaxis"]["title"]["text"])
            self.assertEqual(spec["layout"]["yaxis"]["scaleanchor"], "x")
        self.assertEqual(specs(self.app)[0]["layout"]["scene"]["xaxis"]["backgroundcolor"],
                         top["layout"]["plot_bgcolor"])

    def test_data_unchanged(self):
        payload = vr.plotly_payload(self.result)
        for spec in specs(self.app)[:1]:
            trace = spec["data"][0]
            self.assertEqual(trace["x"], payload["x"])
            self.assertEqual(trace["y"], payload["y"])
            self.assertEqual(trace["z"], payload["z"])
            self.assertEqual(list(trace["marker"]["color"]), payload["colors"])
        self.assertEqual(len(payload["x"]), len(self.result.points))
        self.assertEqual(payload["axis_titles"]["x"], f"X ({vr.ARBITRARY_UNITS})")  # backend/export labels intact
        self.assertEqual(len(self.app.get("download_button")), 3)

    def test_step3_sentence(self):
        app = matched_app(self.files)
        text = " ".join(m.value for m in app.markdown)
        self.assertIn(NEW_SENTENCE, text)
        self.assertNotIn("correct for shape and proportions", text)


if __name__ == "__main__":
    unittest.main()


class ThemeAndScaleChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = cube_files()

    def test_3d_scene_is_light_neutral_in_any_theme(self):
        layout = specs(reconstructed_app(self.files))[0]["layout"]
        self.assertEqual(layout["scene"]["bgcolor"], "rgb(238,240,244)")
        self.assertEqual(layout["paper_bgcolor"], "rgb(238,240,244)")
        for axis in ("xaxis", "yaxis", "zaxis"):
            self.assertEqual(layout["scene"][axis]["backgroundcolor"], "rgb(238,240,244)")
            self.assertEqual(layout["scene"][axis]["tickfont"]["color"], "rgb(40,44,52)")
            self.assertEqual(layout["scene"][axis]["title"]["font"]["color"], "rgb(40,44,52)")
        self.assertEqual(layout["scene"]["camera"]["eye"], dict(x=1.5, y=-1.5, z=1.0))

    def test_3d_chart_opts_out_of_streamlit_theme(self):
        source = PAGE.read_text()
        call = source[source.index('key="m6b_plot_3d"') - 80:source.index('key="m6b_plot_3d"') + 80]
        self.assertIn("theme=None", call)

    def test_step3_labels_and_sentence(self):
        app = matched_app(self.files)
        labels = {b.key: b.label for b in app.button}
        self.assertEqual(labels["m6b_skip_scale"], "Continue Without Scaling")
        self.assertEqual(labels["m6b_use_scale"], "Apply Scale & Continue")
        self.assertIn("Continue without scaling to use arbitrary units, or enter a known distance between two "
                      "points to estimate real-world measurements.", [m.value for m in app.markdown])
        self.assertTrue(next(b for b in app.button if b.key == "m6b_use_scale").proto.disabled)
        self.assertFalse(next(b for b in app.button if b.key == "m6b_skip_scale").proto.disabled)
