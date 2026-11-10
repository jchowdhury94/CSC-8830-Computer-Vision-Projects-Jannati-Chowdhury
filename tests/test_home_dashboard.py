"""Main homepage dashboard: intro text, five module cards in order, navigation targets, and CSS contract."""
import re
import unittest
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

HOME = str(Path(__file__).resolve().parents[1] / "home.py")
SOURCE = Path(HOME).read_text(encoding="utf-8")

CARDS = [
    ("2", "MODULE 02", "Camera Calibration", "2D Object Measurement", "module-2"),
    ("3", "MODULE 03", "Image Filtering", "Spatial & FFT Gaussian Blur", "module-3"),
    ("4", "MODULE 04", "Human Boundary Detection", "RGB & Thermal Images", "module-4"),
    ("6", "MODULE 06", "Optical Flow & SfM", "Motion Tracking & 3D Reconstruction", "module-6"),
    ("7", "MODULE 07", "Stereo Vision", "Stereo Calibration & Measurement", "module-7"),
]


def home():
    return AppTest.from_file(HOME, default_timeout=60).run()


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class HomeDashboardTests(unittest.TestCase):
    def test_intro_content_renders(self):
        at = home()
        self.assertFalse(at.exception)
        text = " ".join(m.value for m in at.markdown)
        for expected in ("Georgia State University · CSC 8830", "Computer Vision Projects",
                         "Interactive experiments in image processing, motion analysis, and 3D reconstruction.",
                         "Jannati Chowdhury"):
            self.assertIn(expected, text)
        self.assertNotIn("Explore the Modules", text)
        self.assertNotIn("5 modules", text)
        self.assertNotIn("home-section", SOURCE)

    def test_cards_in_order_with_exact_content(self):
        at = home()
        self.assertEqual([b.key for b in at.button], [f"home_card_{c[0]}" for c in CARDS])
        for button, (_, number, title, desc, _) in zip(at.button, CARDS):
            self.assertEqual(button.label, f"{number}  \n**{title}**  \n{desc}")

    def test_each_card_targets_its_module_page(self):
        for index, (key, *_rest, path) in enumerate(CARDS):
            with self.subTest(module=key):
                at = home()
                self.assertTrue(at.button[index].key.endswith(key))
                at.button[index].click()
                try:
                    at.run()
                except Exception as exc:  # switch_page may be unsupported in AppTest
                    self.assertNotIsInstance(exc, AssertionError)
                self.assertFalse(at.exception)
        self.assertEqual(
            re.findall(r'"Pages/app_module\d\.py",\s*"(module-\d)"', SOURCE), [c[4] for c in CARDS])

    def test_sidebar_navigation_unchanged(self):
        self.assertEqual(
            re.findall(r'MODULE_PAGES\["(\d)"\],', SOURCE), ["2", "3", "4", "6", "7"])
        for title in ("Module 2 - 2D Object Measurement", "Module 3 - Image Filtering",
                      "Module 4 - Edge Detection, Boundary Detection, and Features",
                      "Module 6 - Optical Flow and Structure from Motion", "Module 7 - Stereo Vision"):
            self.assertIn(f'"{title}"', SOURCE)
        self.assertIn('st.Page(show_home, title="Home", default=True)', SOURCE)

    def test_css_layout_hover_focus_and_responsive(self):
        css = next(m.value for m in home().markdown if "st-key-home_cards" in m.value)
        for fragment in ("calc((100% - 48px) / 3)", "justify-content: center", "gap: 24px", "flex-wrap: wrap",
                         "button:hover", "button:focus-visible", "@media (max-width: 900px)",
                         "@media (max-width: 600px)", "width: 100%", "#C9363E"):
            self.assertIn(fragment, css)
        self.assertNotIn('stSidebar', css)
        self.assertNotIn("transform: scale", css)
        self.assertNotIn("gradient", css)

    def test_sidebar_hidden_by_navigation_for_every_page(self):
        self.assertEqual(SOURCE.count('position="hidden"'), 1)
        self.assertNotIn("stSidebar", SOURCE)
        for n in (2, 3, 4, 6, 7):
            module = (Path(HOME).parent / "Pages" / f"app_module{n}.py").read_text(encoding="utf-8")
            self.assertNotIn("stSidebar", module)
            self.assertNotIn("st.sidebar", module)

    def test_home_button_on_every_module_homepage_only_once(self):
        for n in (2, 3, 4, 6, 7):
            module = (Path(HOME).parent / "Pages" / f"app_module{n}.py").read_text(encoding="utf-8")
            with self.subTest(module=n):
                self.assertEqual(module.count("nav_row("), 1)
                at = AppTest.from_file(str(Path(HOME).parent / "Pages" / f"app_module{n}.py"), default_timeout=60).run()
                self.assertFalse(at.exception)
                self.assertEqual([b.label for b in at.button].count("Home"), 1)
                self.assertEqual(at.button[0].label, "Home")
                self.assertEqual(at.button[0].key, "nav_home_button")

    def test_home_page_object_is_registered_for_back_navigation(self):
        at = home()
        self.assertIn("_home_page", at.session_state)
        self.assertEqual(at.session_state["_home_page"].title, "Home")


if __name__ == "__main__":
    unittest.main()
