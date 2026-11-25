"""Module 3 homepage tests: navigation between home / visitor / experiment views, preserved controls, and unchanged algorithms.

Streamlit pages are exercised with AppTest (no browser). Nothing here writes into the repository.
"""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Pages" / "app_module3.py"


def new_app():
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.run()
    assert not app.exception
    return app


def click(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def make_png(width=24, height=16):
    """A small deterministic PNG for upload tests."""
    import cv2
    import numpy as np
    pixels = np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8)
    return cv2.imencode(".png", pixels)[1].tobytes()


def open_visitor():
    app = new_app()
    click(app, "m3_open_visitor")
    return app


def upload(app, content=None, name="flower.png"):
    key = f"module3_upload_{app.session_state['module3_uploader_version']}"
    app.file_uploader(key=key).upload(name, content if content is not None else make_png())
    app.run()
    assert not app.exception
    return app


def press(app, key):
    next(b for b in app.button if b.key == key).click()
    app.run()
    assert not app.exception


def decode_png(data):
    import cv2
    import numpy as np
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64)


def identity_notes(app):
    return [m for m in app.markdown if m.value.startswith("A 1 \u00d7 1 Gaussian kernel acts as an identity filter")]


def step_titles(app):
    return [s.value for s in app.subheader]


class Module3HomepageTests(unittest.TestCase):
    def test_home_has_only_heading_subtitle_and_two_buttons(self):
        app = new_app()
        self.assertEqual([t.value for t in app.title], ["Module 3"])
        self.assertEqual([h.value for h in app.header], ["Gaussian Blur — Spatial & Frequency Domain"])
        self.assertEqual([b.label for b in app.button if b.key != "nav_arrow_button"],
                         ["Home", "View My Experiments & Results", "Apply Gaussian Blur to Your Own Image"])
        self.assertEqual(len(app.slider) + len(app.selectbox) + len(app.dataframe), 0)

    def test_homepage_button_order_and_breadcrumb(self):
        app = new_app()
        keys = [b.key for b in app.button if not (b.key or "").startswith("nav")]
        self.assertEqual(keys, ["m3_open_experiment", "m3_open_visitor"])
        self.assertEqual([m.value for m in app.markdown if m.value == "Module 3"], ["Module 3"])  # current crumb
        click(app, "m3_open_experiment")
        self.assertEqual([t.value for t in app.title], ["My Experiment & Results"])

    def test_red_button_opens_visitor_application(self):
        app = new_app()
        click(app, "m3_open_visitor")
        self.assertEqual([t.value for t in app.title], ["Apply Gaussian Blur to Your Own Image"])
        self.assertEqual([h.value for h in app.header], ["Gaussian Blur \u2014 Spatial & Frequency Domain"])
        # Progressive disclosure: only Step 1 is visible before an image is uploaded.
        self.assertEqual([s.value for s in app.subheader], ["Step 1: Upload Your Image"])
        self.assertEqual(len(app.selectbox) + len(app.slider) + len(app.dataframe) + len(app.image), 0)
        self.assertEqual([b.label for b in app.button if not (b.key or "").startswith("nav")], [])
        self.assertEqual(len(app.file_uploader), 1)
        self.assertIn("JPG, JPEG and PNG", " ".join(c.value for c in app.caption))
        self.assertIn("Maximum upload size: 200 MB", " ".join(c.value for c in app.caption))

    def test_settings_survive_round_trip_home(self):
        app = new_app()
        click(app, "m3_open_visitor")
        upload(app)
        app.selectbox(key="gaussian_kernel_size").set_value(9)
        app.slider(key="gaussian_sigma").set_value(2.5)
        app.run()
        click(app, "navbtn_0")
        self.assertEqual([t.value for t in app.title], ["Module 3"])
        click(app, "m3_open_visitor")
        self.assertEqual(app.session_state["gaussian_kernel_size"], 9)
        self.assertEqual(app.session_state["gaussian_sigma"], 2.5)
        upload(app)
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").value, 9)
        self.assertEqual(app.slider(key="gaussian_sigma").value, 2.5)

    def test_top_navigation_returns_home_from_visitor(self):
        for key in ("navbtn_0", "nav_arrow_button"):
            app = new_app()
            click(app, "m3_open_visitor")
            click(app, key)
            self.assertEqual([t.value for t in app.title], ["Module 3"])

    def test_experiment_report_structure_values_and_images(self):
        app = new_app()
        click(app, "m3_open_experiment")
        self.assertEqual([t.value for t in app.title], ["My Experiment & Results"])
        self.assertEqual([h.value for h in app.header],
                         ["Gaussian Blur \u2014 Spatial & Frequency Domain",
                          "1. Experimental Setup", "2. Gaussian Kernel Generation",
                          "3. Spatial-Domain Gaussian Filtering (Convolution)",
                          "4. Frequency-Domain Gaussian Filtering (Fourier Transform)",
                          "5. Convolution Theorem (Theory & Experimental Validation)", "6. Conclusion"])
        self.assertEqual([s.value for s in app.subheader], ["Mathematical Demonstration", "Experimental Evidence & Validation"])
        self.assertEqual(len(app.image), 5)   # original once; spatial and frequency results twice each
        self.assertEqual(len(app.expander), 0)
        self.assertEqual(len(app.file_uploader) + len(app.slider) + len(app.selectbox), 0)
        text = " ".join(m.value for m in app.markdown)
        self.assertNotIn("Discussion", text)
        self.assertNotIn("original CSC 8830", text)
        sci_max, sci_mean = "5.115908 \u00d7 10\u207b\u00b9\u00b3", "8.668103 \u00d7 10\u207b\u00b9\u2074"
        metrics = [(m.label, m.value) for m in app.metric]
        self.assertEqual(metrics[:4], [("Kernel Size", "9 \u00d7 9"), ("Sigma", "2.0"),
                                       ("Maximum Absolute Difference", sci_max),
                                       ("Mean Absolute Difference (MAE)", sci_mean)])
        self.assertEqual(metrics[4:], [("Maximum Absolute Difference", sci_max), ("Mean Absolute Difference (MAE)", sci_mean)])
        self.assertIn("Kernel sum = 1.000000000000", " ".join(c.value for c in app.caption))
        self.assertIn("documented in my original experiment", " ".join(c.value for c in app.caption))
        kernel_table = next(t.value for t in app.table if t.value.shape == (9, 9))
        self.assertEqual(kernel_table.iloc[4, 4], "0.041683")
        self.assertAlmostEqual(kernel_table.astype(float).to_numpy().sum(), 1.0, places=4)
        setup = next(t.value for t in app.table if "Boundary handling" in list(t.value.index))
        self.assertEqual(list(setup["Value"]), ["9 \u00d7 9", "2.0", "Reflection", "Each channel independently"])

    def test_theory_section_equations_and_conclusion(self):
        app = new_app()
        click(app, "m3_open_experiment")
        latex = [e.value for e in app.get("latex")]
        self.assertEqual(len(latex), 14)
        joined = " ".join(latex)
        for fragment in (r"\mathcal{F}\{g_{\mathrm{s}}\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{h\}",
                         r"\boxed{", r"g_{\mathrm{s}} = g_{\mathrm{f}}", r"D_i", r"\max_i D_i", r"\dfrac{1}{N}"):
            self.assertIn(fragment, joined)
        for removed in ("e^{-", "P \\times Q", r"k - 1"):
            self.assertNotIn(removed, joined)
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("The convolution theorem states that convolution in the spatial domain is equivalent to multiplication in "
                      "the Fourier (frequency) domain.", text)
        self.assertIn("experimentally supporting the convolution theorem.", text)
        self.assertIn("The right-hand side represents Fourier-domain filtering.", text)
        self.assertIn("equivalence of Gaussian filtering using spatial-domain convolution and Fourier-domain multiplication", text)
        self.assertIn("spatial-domain filtering (convolution) and frequency-domain filtering (Fourier transform)", text)
        self.assertIn("using spatial-domain convolution and Fourier-domain multiplication", text)
        self.assertIn("Fast Fourier Transform (FFT)", text)
        self.assertIn("Inverse Fast Fourier Transform (IFFT)", text)

    def test_experiment_page_has_no_bottom_back_and_top_navigation_returns_home(self):
        app = new_app()
        click(app, "m3_open_experiment")
        self.assertEqual([b.key for b in app.button if not (b.key or "").startswith("nav")], [])
        for key in ("navbtn_0", "nav_arrow_button"):
            app = new_app()
            click(app, "m3_open_experiment")
            click(app, key)
            self.assertEqual([t.value for t in app.title], ["Module 3"])

    def test_experiment_figures_exist(self):
        for name in ("step_original", "step_spatial_blur", "step_fft_blur"):
            self.assertTrue((ROOT / "Modules" / "Module_3" / "display" / f"{name}.png").is_file())


class Module3VisitorWorkflowTests(unittest.TestCase):
    def test_steps_appear_one_at_a_time(self):
        app = upload(open_visitor())
        self.assertEqual(step_titles(app), ["Step 1: Upload Your Image", "Step 2: Configure Gaussian Filter"])
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").options, [f"{v} \u00d7 {v}" for v in range(1, 16, 2)])
        self.assertIn("flower.png", " ".join(c.value for c in app.caption))
        self.assertIn("24 \u00d7 16 pixels", " ".join(c.value for c in app.caption))
        self.assertEqual(len(app.dataframe), 0)
        press(app, "m3_generate_kernel")
        self.assertEqual(step_titles(app)[-1], "Step 3: Gaussian Kernel")
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value.shape, (1, 1))
        self.assertAlmostEqual(float(app.metric[0].value), 1.0, places=9)
        self.assertEqual(len(app.image), 1)               # only the original image so far
        press(app, "m3_apply_filters")
        self.assertEqual(step_titles(app)[-2:], ["Step 4: Filtering Results", "Step 5: Compare & Validate Results"])
        self.assertEqual(len(app.image), 3)               # original + the two results, not repeated in Step 5
        labels = [m.label for m in app.metric]
        self.assertIn("Maximum Absolute Difference", labels)
        self.assertIn("Mean Absolute Difference (MAE)", labels)
        maximum_text = next(m.value for m in app.metric if m.label == "Maximum Absolute Difference")
        mantissa, exponent = maximum_text.split(" \u00d7 10")
        self.assertLess(float(mantissa), 10)
        self.assertTrue(exponent.startswith("\u207b"))   # far below one intensity level

    def test_fresh_session_starts_with_identity_defaults_and_explanation(self):
        app = upload(open_visitor())
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").value, 1)
        self.assertEqual(app.slider(key="gaussian_sigma").value, 0.1)
        self.assertEqual(len(identity_notes(app)), 1)
        app.selectbox(key="gaussian_kernel_size").set_value(3)
        app.run()
        self.assertEqual(len(identity_notes(app)), 0)
        app.selectbox(key="gaussian_kernel_size").set_value(1)
        app.run()
        self.assertEqual(len(identity_notes(app)), 1)

    def test_one_by_one_kernel_leaves_the_image_unchanged(self):
        import numpy as np
        app = upload(open_visitor())
        press(app, "m3_generate_kernel")
        press(app, "m3_apply_filters")
        for key in ("module3_spatial_result", "module3_frequency_result"):
            result = app.session_state[key]
            self.assertEqual(result.shape, (16, 24, 3))
            self.assertLess(np.abs(result - decode_png(make_png())).max(), 1e-9)

    def test_validation_metrics_use_scientific_notation(self):
        app = upload(open_visitor())
        app.selectbox(key="gaussian_kernel_size").set_value(9)
        app.slider(key="gaussian_sigma").set_value(2.0)
        app.run()
        press(app, "m3_generate_kernel")
        press(app, "m3_apply_filters")
        values = [m.value for m in app.metric if "Absolute Difference" in m.label]
        self.assertEqual(len(values), 2)
        for value in values:
            self.assertRegex(value, r"^\d\.\d{6} \u00d7 10[\u207b\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+$")

    def test_scientific_text_formatting(self):
        namespace = {}
        source = PAGE.read_text()
        start = source.index("_SUPERSCRIPT = ")
        end = source.index("def to_display")
        exec(source[start:end], namespace)
        fmt = namespace["scientific_text"]
        self.assertEqual(fmt(5.115908e-13), "5.115908 \u00d7 10\u207b\u00b9\u00b3")
        self.assertEqual(fmt(8.668103e-14), "8.668103 \u00d7 10\u207b\u00b9\u2074")
        self.assertEqual(fmt(0.0), "0.000000 \u00d7 10\u2070")
        self.assertEqual(fmt(123.456), "1.234560 \u00d7 10\u00b2")

    def test_fourier_terminology_in_visitor_steps(self):
        app = upload(open_visitor())
        press(app, "m3_generate_kernel")
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("Spatial-domain filtering uses convolution with the Gaussian kernel.", text)
        self.assertIn("Fast Fourier Transform (FFT)", text)
        self.assertIn("Inverse Fast Fourier Transform (IFFT)", text)
        press(app, "m3_apply_filters")
        captions = [img.caption for element in app.image for img in element.proto.imgs]
        self.assertIn("Spatial-Domain Result (Convolution)", captions)
        self.assertIn("Frequency-Domain Result (Fourier Transform)", captions)
        self.assertIn("spatial-domain convolution and Fourier-domain multiplication", " ".join(m.value for m in app.markdown))

    def test_invalid_file_keeps_later_steps_hidden(self):
        app = upload(open_visitor(), content=b"not an image", name="bad.png")
        self.assertEqual(step_titles(app), ["Step 1: Upload Your Image"])
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.selectbox) + len(app.slider), 0)

    def test_changing_parameters_hides_dependent_steps(self):
        app = upload(open_visitor())
        press(app, "m3_generate_kernel")
        press(app, "m3_apply_filters")
        self.assertEqual(len(step_titles(app)), 5)
        app.slider(key="gaussian_sigma").set_value(2.0)
        app.run()
        self.assertEqual(step_titles(app), ["Step 1: Upload Your Image", "Step 2: Configure Gaussian Filter"])
        self.assertNotIn("module3_frequency_result", app.session_state)
        press(app, "m3_generate_kernel")
        self.assertEqual(step_titles(app)[-1], "Step 3: Gaussian Kernel")   # kernel returns, filtered results do not
        app.selectbox(key="gaussian_kernel_size").set_value(9)
        app.run()
        self.assertEqual(len(step_titles(app)), 2)

    def test_unrelated_rerun_keeps_completed_steps(self):
        app = upload(open_visitor())
        press(app, "m3_generate_kernel")
        press(app, "m3_apply_filters")
        first = app.session_state["module3_spatial_result"]
        app.run()
        self.assertEqual(len(step_titles(app)), 5)
        self.assertIs(app.session_state["module3_spatial_result"], first)   # not recomputed

    def test_replacing_the_image_invalidates_results(self):
        app = upload(open_visitor())
        press(app, "m3_generate_kernel")
        press(app, "m3_apply_filters")
        upload(app, content=make_png(20, 20), name="other.png")
        self.assertEqual(len(step_titles(app)), 2)
        self.assertNotIn("module3_spatial_result", app.session_state)

    def test_reset_all_returns_to_step_one_with_defaults(self):
        app = upload(open_visitor())
        app.selectbox(key="gaussian_kernel_size").set_value(9)
        app.slider(key="gaussian_sigma").set_value(3.0)
        app.run()
        press(app, "m3_generate_kernel")
        press(app, "m3_apply_filters")
        press(app, "m3_reset_all")
        self.assertEqual(step_titles(app), ["Step 1: Upload Your Image"])
        self.assertEqual(app.session_state["gaussian_kernel_size"], 1)
        self.assertEqual(app.session_state["gaussian_sigma"], 0.1)
        self.assertNotIn("module3_frequency_result", app.session_state)
        upload(app)
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").value, 1)

    def test_parameter_reset_restores_defaults_and_hides_results(self):
        app = upload(open_visitor())
        app.selectbox(key="gaussian_kernel_size").set_value(7)
        app.slider(key="gaussian_sigma").set_value(3.0)
        app.run()
        press(app, "m3_generate_kernel")
        press(app, next(b.key for b in app.button if b.label == "Reset"))
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").value, 1)
        self.assertEqual(app.slider(key="gaussian_sigma").value, 0.1)
        self.assertEqual(step_titles(app), ["Step 1: Upload Your Image", "Step 2: Configure Gaussian Filter"])


class Module3AlgorithmTests(unittest.TestCase):
    def test_spatial_and_frequency_blur_agree(self):
        import sys
        import numpy as np
        sys.path.insert(0, str(ROOT))
        from Modules.Module_3.image_filtering import (create_gaussian_kernel, gaussian_blur,
                                                      gaussian_blur_frequency)
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, (24, 31, 3), dtype=np.uint8)
        kernel = create_gaussian_kernel(9, 2.0)
        self.assertAlmostEqual(kernel.sum(), 1.0, places=12)
        difference = np.abs(gaussian_blur(image, 9, 2.0) - gaussian_blur_frequency(image, 9, 2.0))
        self.assertLess(difference.max(), 1e-8)


if __name__ == "__main__":
    unittest.main()
