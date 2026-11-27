"""Module 6 Part A saved results content (AppTest, no browser)."""
import importlib.util
import re
import unittest
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module6.py"
OUTPUT = ROOT / "Modules" / "Module_6" / "Part_A" / "output"
SECTIONS = ["1. Experimental Setup", "2. Optical Flow Visualization and Motion Analysis",
            "3. Theory: Motion Tracking Equations", "4. Theory: Bilinear Interpolation",
            "5. Experimental Validation of Motion Tracking", "6. Conclusion"]


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def page_module():
    """The page imported as a module (it renders its homepage on import, which is harmless outside a server)."""
    spec = importlib.util.spec_from_file_location("module6_page", PAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_numbers(name, label):
    """All numbers on the saved report line(s) that follow a label."""
    text = (OUTPUT / name).read_text(encoding="utf-8")
    match = re.search(re.escape(label) + r"(.*)", text)
    return [float(n) for n in re.findall(r"-?\d+\.\d+", match.group(1))]


class Module6PartAResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = new_app()
        click(app, "m6_open_results")
        click(app, "m6_res_a")
        cls.app = app
        cls.tables = "\n".join(t.value.to_string() for t in app.table)
        cls.latex = "\n".join(e.value for e in app.latex)

    def test_header_and_six_sections(self):
        self.assertEqual(self.app.session_state["module6_section"], "a_results")
        self.assertEqual([t.value for t in self.app.title], ["Optical Flow and Motion Tracking"])
        self.assertIn("MODULE 6 — PART A", [m.value for m in self.app.markdown])
        headers = [h.value for h in self.app.header]
        self.assertEqual(headers, ["My Experiment & Results", *SECTIONS])
        self.assertEqual(len(self.app.warning), 0)

    def test_exactly_five_collapsed_dropdowns_without_nesting(self):
        expanders = list(self.app.expander)
        self.assertEqual([e.label for e in expanders], ["View Mathematical Derivation", "4.1 Mathematical Derivation",
                                                        *["View Numerical Calculations"] * 3])
        self.assertFalse(any(e.proto.expanded for e in expanders))
        for expander in expanders:
            self.assertEqual(len(expander.expander), 1)  # the query includes the dropdown itself only
            for kind in ("table", "metric", "image", "video"):
                self.assertEqual(len(expander.get(kind)), 0, kind)
        hidden = [e.value for expander in expanders for e in expander.latex]
        visible = [e.value for e in self.app.latex]
        for value in hidden:
            visible.remove(value)
        for essential in (r"I_x u + I_y v + I_t = 0", r"(A^{T}A)^{-1}A^{T}\mathbf{b}", r"x_2 = x_1 + u",
                          r"d_x(1-d_y)\,I_{21}", r"e &= \sqrt{(\Delta x)^{2}"):
            self.assertTrue(any(essential in v for v in visible), essential)
        self.assertEqual([s.value[:3] for s in expanders[0].subheader], ["3.1", "3.2", "3.3", "3.4"])

    def test_numerical_calculations_are_inside_their_dropdowns(self):
        def content(blocks):
            return " ".join(e.value for block in blocks for e in [*block.latex, *block.markdown])

        bilinear, first, second = list(self.app.expander)[2:]
        for dropdown, values in ((first, ("P_1 = (508.8892", "0.150348 & -0.037867", r"\frac{1}{0.078916}",
                                          r"\approx 510.1953", r"\approx 1131.6883")),
                                 (second, ("P_1 = (401.2964", "0.813748 & -0.308143", r"\frac{1}{1.081259}",
                                           r"\approx 402.3764", r"\approx 1644.7607"))):
            text = content([dropdown])
            for expected in (*values, "Normal equations", "Solving with the 2 × 2 inverse",
                             "Applying the tracking equations"):
                self.assertIn(expected, text)
        text = content([bilinear])
        for expected in (*(f"w_{{{n}}} = " for n in (11, 21, 12, 22)), r"w_{11} + w_{21} + w_{12} + w_{22} = 1",
                         "I(P) = 0.634241(83)", r"I(P) \approx 87.08732152"):
            self.assertIn(expected, text)
        self.assertEqual(len(bilinear.caption), 1)
        self.assertIn("Weights are shown rounded to six decimals", bilinear.caption[0].value)
        outside = [e.value for e in [*self.app.latex, *self.app.markdown]]
        for value in [e.value for block in (bilinear, first, second) for e in [*block.latex, *block.markdown]]:
            outside.remove(value)
        outside = " ".join(outside)
        for hidden in ("0.150348", "0.813748", "510.1953", "402.3764", "87.08732152", "w_{11} ="):
            self.assertNotIn(hidden, outside)
        for visible in ("Feature 14 — Frames 484 and 485", "Feature 12 — Frames 482 and 483", "401.29635620",
                        "closely matches OpenCV's interpolated intensity"):
            self.assertIn(visible, outside)

    def test_removed_content_is_not_displayed(self):
        self.assertNotIn("View Calculation Methodology", [e.label for e in self.app.expander])
        self.assertNotIn("Maximum pyramid level", self.tables)
        text = " ".join(m.value for m in self.app.markdown)
        for removed in ("Lost features", "**Relative motion:**", "OpenCV is the reference implementation"):
            self.assertNotIn(removed, text)
        self.assertIn("using a 21 × 21 pixel window.", text)
        self.assertIn("feature trajectories in image coordinates.", text)

    def test_every_displayed_equation_is_boxed_once(self):
        flow = [m.value for m in self.app.markdown if m.value.startswith(r"$\displaystyle")]
        self.assertEqual(len(flow), 13)
        for value in [e.value for e in self.app.latex] + flow:
            self.assertEqual(value.count(r"\boxed"), 1, value)

    def test_experimental_calculations_are_displayed(self):
        text = " ".join(m.value for m in self.app.markdown)
        for expected in (r"= (0.70364380)(0.90136719) \approx 0.634241", r"w_{11} + w_{21} + w_{12} + w_{22} = 1",
                         r"\approx 87.08732152", r"0.150348 & -0.037867", r"0.813748 & -0.308143",
                         r"\frac{1}{0.078916}", r"\frac{1}{1.081259}",
                         r"x_2 = x_1 + u = 508.889221 + 1.306049 \approx 510.1953",
                         r"y_2 = y_1 + v = 1131.701172 - 0.012911 \approx 1131.6883",
                         r"x_2 = x_1 + u = 401.296356 + 1.080078 \approx 402.3764",
                         r"y_2 = y_1 + v = 1644.098633 + 0.662031 \approx 1644.7607"):
            self.assertIn(expected, text + " " + self.latex)

    def test_derivations_are_displayed(self):
        for expected in (r"I(x, y, t) = I(x+u", r"\boxed{\,I_x u + I_y v + I_t = 0", r"\nabla E(\mathbf{d})",
                         r"A^{T}A\,\mathbf{d} &= A^{T}\mathbf{b}", r"\boxed{\,\mathbf{d} = (A^{T}A)^{-1}A^{T}\mathbf{b}",
                         r"-\sum I_x I_t", r"\boxed{\,x_2 = x_1 + u", r"I_{\mathrm{top}}", r"I_{\mathrm{bottom}}",
                         r"\boxed{\begin{aligned}"):
            self.assertIn(expected, self.latex)

    def test_displayed_values_and_evidence(self):
        for expected in ("+0.818", "+0.013", "0.821", "+0.939", "+0.260", "0.991", "+1.3060", "−0.0129", "510.1953",
                         "1131.6883", "509.8865", "+1.0801", "+0.6620", "402.3764", "1644.7607", "31.5061", "32.1485",
                         "0.634241", "0.267126", "0.069402", "0.029230"):
            self.assertIn(expected, self.tables)
        self.assertEqual([m.value for m in self.app.metric][2:],
                         ["87.08732152", "87.08732605", "4.529 × 10⁻⁶", "0.3088 px", "0.3150 px", "0.6424°"])
        for name in ("video_1_optical_flow_arrows.mp4", "video_2_optical_flow_arrows.mp4",
                     "video_1_validation_comparison_arrows.png",
                     "video_2_validation_comparison_arrows.png",
                     "video_2_bilinear_interpolation.png"):
            self.assertTrue((OUTPUT / name).is_file(), name)
        self.assertEqual(len(self.app.get("video")), 2)
        self.assertEqual(len(self.app.get("image")), 3)


class Module6PartANumbersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = page_module()

    def test_tracking_values_match_saved_reports(self):
        for data in self.page.VALIDATIONS:
            name = f"video_{data['video']}_manual_lk_validation.txt"
            result = self.page.tracking_comparison(data)
            ata = np.array(data["ata"])
            np.testing.assert_allclose(np.linalg.solve(ata, data["atb"]), data["manual"], atol=1e-9)
            np.testing.assert_allclose(data["manual"], [report_numbers(name, "Manual u:")[0],
                                                        report_numbers(name, "Manual v:")[0]], atol=1e-12)
            np.testing.assert_allclose(result["new"], report_numbers(name, "Predicted new coordinate:"), atol=1e-8)
            self.assertAlmostEqual(result["position_difference"], report_numbers(name, "Position error:")[0], places=8)
            if data["video"] == 2:
                self.assertAlmostEqual(result["angle_difference"],
                                       report_numbers(name, "Absolute angle difference:")[0], places=8)

    def test_normal_equations_match_original_frames(self):
        for data in self.page.VALIDATIONS:
            old, new = (cv2.cvtColor(cv2.imread(str(OUTPUT / f"video_{data['video']}_validation_frame_{k}.png")),
                                     cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0 for k in ("old", "new"))
            average, (x, y) = (old + new) / 2.0, data["center"]
            window = (slice(y - 10, y + 11), slice(x - 10, x + 11))
            a = np.column_stack([cv2.Sobel(average, cv2.CV_64F, dx, 1 - dx, ksize=3, scale=1 / 8)[window].ravel()
                                 for dx in (1, 0)])
            b = -(new - old)[window].ravel()
            self.assertEqual(a.shape, (441, 2))
            np.testing.assert_allclose(a.T @ a, data["ata"], atol=1e-11)
            np.testing.assert_allclose(a.T @ b, data["atb"], atol=1e-11)

    def test_bilinear_values_match_saved_report(self):
        example = self.page.bilinear_example()
        name = "video_2_bilinear_interpolation.txt"
        self.assertAlmostEqual(sum(example["weights"]), 1.0, places=12)
        self.assertAlmostEqual(example["intensity"], report_numbers(name, "Manual interpolated intensity =")[0], places=10)
        self.assertEqual(self.page.BILINEAR_OPENCV, report_numbers(name, "OpenCV interpolated intensity =")[0])
        self.assertAlmostEqual(example["difference"], report_numbers(name, "Absolute difference =")[0], places=11)


if __name__ == "__main__":
    unittest.main()
