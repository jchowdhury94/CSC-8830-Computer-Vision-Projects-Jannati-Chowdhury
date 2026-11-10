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


class Module3HomepageTests(unittest.TestCase):
    def test_home_has_only_heading_subtitle_and_two_buttons(self):
        app = new_app()
        self.assertEqual([t.value for t in app.title], ["Module 3"])
        self.assertEqual([h.value for h in app.header], ["Gaussian Blur — Spatial & Frequency Domain"])
        self.assertEqual([b.label for b in app.button],
                         ["Home", "Apply Gaussian Blur to Your Own Image", "View My Experiment & Results"])
        self.assertEqual(len(app.slider) + len(app.selectbox) + len(app.dataframe), 0)

    def test_first_button_opens_existing_interface(self):
        app = new_app()
        click(app, "m3_open_visitor")
        self.assertEqual([t.value for t in app.title], ["Module 3 - Image Filtering"])
        self.assertEqual([s.value for s in app.subheader[:2]],
                         ["Spatial Gaussian Blurring", "Fourier / Frequency Domain Gaussian Blurring"])
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").value, 5)
        self.assertEqual(app.slider(key="gaussian_sigma").value, 1.5)
        labels = [b.label for b in app.button if b.key != "nav_home_button"]
        for label in ("Reset", "Apply Fourier Filter", "Reset All", "Back"):
            self.assertIn(label, labels)
        self.assertTrue(next(b for b in app.button if b.label == "Apply Fourier Filter").disabled)
        self.assertEqual(len(app.dataframe), 1)
        self.assertAlmostEqual(float(app.metric[0].value), 1.0, places=9)   # kernel sum

    def test_controls_change_kernel(self):
        app = new_app()
        click(app, "m3_open_visitor")
        app.selectbox(key="gaussian_kernel_size").set_value(9)
        app.slider(key="gaussian_sigma").set_value(2.0)
        app.run()
        self.assertEqual(app.dataframe[0].value.shape, (9, 9))
        self.assertAlmostEqual(float(app.metric[0].value), 1.0, places=9)

    def test_settings_survive_round_trip_home(self):
        app = new_app()
        click(app, "m3_open_visitor")
        app.selectbox(key="gaussian_kernel_size").set_value(9)
        app.slider(key="gaussian_sigma").set_value(2.5)
        app.run()
        click(app, "m3_back_top")
        self.assertEqual([t.value for t in app.title], ["Module 3"])
        click(app, "m3_open_visitor")
        self.assertEqual(app.selectbox(key="gaussian_kernel_size").value, 9)
        self.assertEqual(app.slider(key="gaussian_sigma").value, 2.5)

    def test_back_buttons_return_home(self):
        for key in ("m3_back_top", "m3_back_bottom"):
            app = new_app()
            click(app, "m3_open_visitor")
            click(app, key)
            self.assertEqual([t.value for t in app.title], ["Module 3"])

    def test_experiment_page_steps_images_and_values(self):
        app = new_app()
        click(app, "m3_open_experiment")
        self.assertEqual([h.value for h in app.header], ["My Experiment & Results"])
        self.assertEqual([s.value for s in app.subheader],
                         ["Step 1 \u2014 Original Image & Parameters", "Step 2 \u2014 Gaussian Kernel",
                          "Step 3 \u2014 Spatial Gaussian Blur", "Step 4 \u2014 Frequency-Domain Blur",
                          "Step 5 \u2014 Compare & Validate Results", "Conclusion"])
        self.assertEqual(len(app.image), 5)
        self.assertEqual(len(app.expander), 0)
        self.assertEqual(len(app.file_uploader) + len(app.slider) + len(app.selectbox), 0)
        text = " ".join(m.value for m in app.markdown)
        self.assertNotIn("were not re-run", text)
        self.assertNotIn("original CSC 8830", text)
        metrics = {m.label: m.value for m in app.metric}
        self.assertEqual(metrics["Kernel size"], "9 \u00d7 9")
        self.assertEqual(metrics["Sigma"], "2.0")
        self.assertEqual(metrics["Kernel sum"], "1.000000000000")
        self.assertEqual(metrics["Maximum Absolute Difference"], "5.115908 \u00d7 10\u207b\u00b9\u00b3")
        self.assertEqual(metrics["Mean Absolute Difference"], "8.668103 \u00d7 10\u207b\u00b9\u2074")
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(app.dataframe[0].value.shape, (9, 9))
        self.assertAlmostEqual(app.dataframe[0].value.to_numpy().sum(), 1.0, places=12)

    def test_experiment_page_has_exactly_two_back_buttons(self):
        app = new_app()
        click(app, "m3_open_experiment")
        self.assertEqual([(b.key, b.label) for b in app.button if b.key != "nav_home_button"],
                         [("m3_back_experiment_top", "Back"), ("m3_back_experiment_bottom", "Back")])
        for key in ("m3_back_experiment_top", "m3_back_experiment_bottom"):
            app = new_app()
            click(app, "m3_open_experiment")
            click(app, key)
            self.assertEqual([t.value for t in app.title], ["Module 3"])

    def test_experiment_figures_exist(self):
        for name in ("step_original", "step_spatial_blur", "step_fft_blur"):
            self.assertTrue((ROOT / "Modules" / "Module_3" / "display" / f"{name}.png").is_file())


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
