"""Module 2 safety tests: the completed experiment stays read-only and visitor measurements never use my saved calibration.

Streamlit pages are exercised with AppTest (no browser). Nothing here writes into Modules/Module_2.
"""
import ast
import builtins
import csv
import hashlib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_2"
PAGE = ROOT / "Pages" / "app_module2.py"
NPZ = MODULE / "camera_calibration.npz"
CSV = MODULE / "measurement_results.csv"

# Recorded before the Phase 1 refactor.
NPZ_SHA256 = "15a634af6f82e8c29b37d935887fb7fa7d37e6e552624f50ce0e1327b6cf702c"
CSV_SHA256 = "95f72ccc2d9f8b242d991d98ca6d354d2643fa3a3f8ded40d4c6bb17c53693eb"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_function(path, name, namespace=None):
    """Compile one top-level function out of a script, without running the script around it."""
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"np": np, **(namespace or {})}
    exec(compile(ast.Module([node], []), str(path), "exec"), scope)
    return scope[name]


def run_page(seed=None):
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    if seed:
        for key, value in seed.items():
            at.session_state[key] = value
    return at.run()


class SavedExperimentTests(unittest.TestCase):
    def test_saved_calibration_is_readable(self):
        with np.load(NPZ) as data:
            k = data["camera_matrix"]
            self.assertEqual(k.shape, (3, 3))
            self.assertAlmostEqual(k[0, 0], 4068.2888387501193, places=6)
            self.assertAlmostEqual(k[1, 1], 4065.1583596080636, places=6)
            self.assertEqual(data["distortion_coefficients"].shape, (1, 5))
            self.assertAlmostEqual(float(data["rms_error"]), 3.087451468378783, places=9)
            self.assertEqual(int(data["successful_images"]), 18)
            self.assertEqual((int(data["checkerboard_columns"]), int(data["checkerboard_rows"])), (8, 6))
            self.assertEqual(float(data["square_size_mm"]), 20.0)

    def test_csv_has_the_original_20_trials_and_statistics(self):
        with CSV.open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([int(r["Trial"]) for r in rows], list(range(1, 21)))
        expected = {"Width": (0.603, 1.746, 0.906, 0.678), "Height": (0.614, 2.191, 1.251, 0.701)}
        for dimension, (mae, mean_pct, std_pct, rmse) in expected.items():
            err = np.array([float(r[f"{dimension}_Error_cm"]) for r in rows])
            pct = np.array([float(r[f"{dimension}_Error_Percent"]) for r in rows])
            self.assertAlmostEqual(err.mean(), mae, places=3)
            self.assertAlmostEqual(pct.mean(), mean_pct, places=3)
            self.assertAlmostEqual(pct.std(), std_pct, places=3)
            self.assertAlmostEqual(float(np.sqrt((err ** 2).mean())), rmse, places=3)

    def test_experiment_files_are_unchanged(self):
        self.assertEqual(sha256(NPZ), NPZ_SHA256)
        self.assertEqual(sha256(CSV), CSV_SHA256)


class MeasurementFormulaTests(unittest.TestCase):
    CASES = [((0, 0), (1000, 0), 210.0), ((100, 200), (100, 2200), 250.0), ((10.5, 20.5), (3000, 4000.25), 230.0)]
    FX, FY = 4068.2888387501193, 4065.1583596080636

    def test_page_formula_matches_original_formula(self):
        page = load_function(PAGE, "calculate_length")
        for p1, p2, z in self.CASES:
            expected = z * np.sqrt(((p2[0] - p1[0]) / self.FX) ** 2 + ((p2[1] - p1[1]) / self.FY) ** 2)
            self.assertEqual(page(p1, p2, z, self.FX, self.FY), expected)

    def test_page_formula_matches_cli_script(self):
        page = load_function(PAGE, "calculate_length")
        cli = load_function(MODULE / "measurement.py", "calculate_length", {"fx": self.FX, "fy": self.FY})
        for p1, p2, z in self.CASES:
            self.assertEqual(page(p1, p2, z, self.FX, self.FY), cli(p1, p2, z))

    def test_known_value(self):
        page = load_function(PAGE, "calculate_length")
        self.assertAlmostEqual(page((0, 0), (self.FX, 0), 100.0, self.FX, self.FY), 100.0)


VISITOR = {"camera_matrix": np.diag([1000.0, 1000.0, 1.0]), "distortion_coefficients": np.zeros((1, 5))}


def labels(at):
    return [b.label for b in at.button]


def table_cells(at):
    cells = []
    for table in at.table:
        frame = table.value
        cells += [str(v) for v in frame.to_numpy().ravel()] + [str(v) for v in frame.columns] + [str(v) for v in frame.index]
    return cells


def open_experiment(at):
    at.button(key="m2_experiment_btn").click()
    return at.run()


def header_texts(at):
    return [h.value for h in at.header]


def open_experiment(at):
    at.button(key="m2_open_experiment").click()
    return at.run()


def open_visitor(at):
    at.button(key="m2_open_visitor").click()
    return at.run()


def header_texts(at):
    return [h.value for h in at.header]


def visitor_run(seed=None):
    return open_visitor(run_page(seed))


class HomePageTests(unittest.TestCase):
    def test_initial_page_shows_exactly_two_navigation_choices(self):
        at = run_page()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["m2_page"], "home")
        self.assertEqual(labels(at), ["Home", "Calibrate Your Camera & Measure an Object", "View My Experiment & Results"])
        self.assertEqual([t.value for t in at.title], ["Module 2"])
        self.assertEqual(header_texts(at), ["Camera Calibration & 2D Object Measurement"])

    def test_visitor_controls_are_hidden_initially(self):
        at = run_page()
        for label in ("Load Saved Calibration", "Calibrate New Camera", "Reset All"):
            self.assertNotIn(label, labels(at))
        self.assertEqual(len(at.text_input) + len(at.number_input) + len(at.get("file_uploader")) + len(at.expander), 0)
        self.assertEqual(len(at.info), 0)
        self.assertEqual(header_texts(at), ["Camera Calibration & 2D Object Measurement"])
        self.assertNotIn("Back", labels(at))

    def test_experiment_content_is_hidden_initially(self):
        at = run_page()
        self.assertEqual(len(at.table) + len(at.get("imgs")) + len(at.get("arrow_bar_chart")), 0)
        self.assertNotIn("My Experiment & Results", header_texts(at))


class VisitorWorkflowTests(unittest.TestCase):
    def test_button_one_opens_the_visitor_workflow(self):
        at = visitor_run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["m2_page"], "visitor")
        self.assertEqual([t.value for t in at.title], ["Calibrate Your Camera & Measure an Object"])
        self.assertNotIn("My Experiment & Results", header_texts(at))
        for label in ("Load Saved Calibration", "Calibrate New Camera", "Reset All", "Module 2"):
            self.assertIn(label, labels(at))
        self.assertNotIn("View My Experiment & Results", labels(at))
        self.assertEqual(len(at.table), 0)
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertNotIn("m2_scroll_to_experiment", at.session_state)
        self.assertNotIn("m2_scroll_to", at.session_state)                 # heading scroll was removed (it hid Back / Home)

    def test_valid_measurement_requirements_are_a_compact_expander(self):
        at = visitor_run()
        expander = next(e for e in at.expander if e.label == "What a valid measurement requires")
        text = " ".join(m.value for m in expander.markdown)
        for phrase in ("same camera, lens and settings", "fronto-parallel", "camera-to-object distance Z", "top-left"):
            self.assertIn(phrase, text)

    def test_measurement_controls_are_disabled_without_a_visitor_calibration(self):
        at = visitor_run()
        self.assertEqual(len(at.text_input) + len(at.number_input) + len(at.get("file_uploader")), 0)
        self.assertTrue(any("First load a saved calibration" in i.value for i in at.info))

    def test_workflow_buttons_only_select_a_workflow(self):
        at = visitor_run()
        at.button(key="m2_cal_mode_load").click()
        at.run()
        self.assertEqual(at.session_state["m2_cal_workflow"], "load")
        self.assertEqual(len(at.get("file_uploader")), 1)                      # the .npz uploader (Phase 3)
        at.button(key="m2_cal_mode_run").click()
        at.run()
        self.assertEqual(at.session_state["m2_cal_workflow"], "run")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])         # choosing a workflow never activates a calibration

    def test_measurement_controls_appear_only_with_a_visitor_calibration(self):
        at = visitor_run({"m2_visitor_calibration": VISITOR})
        self.assertFalse(at.exception)
        self.assertEqual(len(at.text_input), 1)
        self.assertEqual(len(at.number_input), 3)
        self.assertFalse(any("First load a saved calibration" in i.value for i in at.info))

    def test_no_visitor_calibration_fallback_exists(self):
        for page in ("home", "visitor", "experiment"):
            at = run_page({"m2_page": page})
            self.assertFalse(at.exception)
            self.assertIsNone(at.session_state["m2_visitor_calibration"])
            self.assertEqual(len(at.text_input) + len(at.number_input), 0)
        source = PAGE.read_text()
        self.assertNotIn('ss["m2_visitor_calibration"] = experiment', source)
        self.assertNotIn("m2_visitor_calibration or", source)

    def test_reset_all_clears_only_visitor_state(self):
        at = visitor_run({"m2_visitor_calibration": VISITOR, "m2_object_name": "Box"})
        at.button(key="m2_reset_all").click()
        at.run()
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertEqual(at.session_state["m2_page"], "visitor")
        self.assertEqual(len(at.text_input), 0)
        self.assertEqual(sha256(CSV), CSV_SHA256)
        self.assertEqual(sha256(NPZ), NPZ_SHA256)

    def test_single_back_button_and_headings_at_top(self):
        at = visitor_run()
        self.assertEqual(labels(at).count("Back"), 0)  # the top Back is replaced by the breadcrumb
        self.assertEqual(labels(at)[:3], ["Go back", "Home", "Module 2"])  # Back arrow, then the breadcrumb
        self.assertNotIn("← Back", labels(at))
        self.assertNotIn("Back to Module 2", labels(at))
        self.assertEqual([t.value for t in at.title], ["Calibrate Your Camera & Measure an Object"])
        self.assertNotIn("Calibrate Your Camera & Measure an Object", header_texts(at))
        self.assertIn("Step 1: Calibrate Your Camera", [s.value for s in at.subheader])
        self.assertIn("Step 2: Measure an Object", [s.value for s in at.subheader])
        for at_ in (visitor_run(), visitor_run({"m2_cal_workflow": "load"}), visitor_run({"m2_cal_workflow": None})):
            self.assertNotIn("Choose how to set up your camera.", [c.value for c in at_.caption])

    def test_back_buttons_return_home(self):
        for key in ("navbtn_0",):
            at = visitor_run()
            at.button(key=key).click()
            at.run()
            self.assertEqual(at.session_state["m2_page"], "home")
            self.assertEqual(len(labels(at)), 3)


class ExperimentViewTests(unittest.TestCase):
    def test_button_two_opens_experiment_and_hides_visitor_workflow(self):
        at = open_experiment(run_page())
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["m2_page"], "experiment")
        self.assertIn("My Experiment & Results", header_texts(at))
        self.assertNotIn("Calibrate Your Camera & Measure an Object", header_texts(at))
        self.assertNotIn("Load Saved Calibration", labels(at))
        self.assertNotIn("Reset All", labels(at))
        self.assertEqual(len(at.text_input) + len(at.number_input) + len(at.get("file_uploader")), 0)   # no duplicate controls
        self.assertNotIn("m2_scroll_to", at.session_state)                                             # heading scroll was removed

    def test_back_buttons_return_home(self):
        for key in ("navbtn_0", "m2_back_bottom"):
            at = open_experiment(run_page())
            at.button(key=key).click()
            at.run()
            self.assertEqual(at.session_state["m2_page"], "home")
            self.assertEqual(labels(at), ["Home", "Calibrate Your Camera & Measure an Object", "View My Experiment & Results"])
            self.assertNotIn("My Experiment & Results", header_texts(at))

    def test_visitor_state_survives_navigation(self):
        at = visitor_run({"m2_visitor_calibration": VISITOR, "m2_object_name": "Box", "m2_actual_width": 12.5, "m2_cal_workflow": "load"})
        at.button(key="navbtn_0").click()
        at.run()
        at = open_experiment(at)
        self.assertIs(at.session_state["m2_visitor_calibration"], VISITOR)
        at.button(key="m2_back_bottom").click()
        at.run()
        at = open_visitor(at)
        self.assertEqual(at.session_state["m2_visitor_calibration"]["camera_matrix"][0, 0], 1000.0)
        self.assertEqual(at.text_input[0].value, "Box")
        self.assertEqual(at.number_input[0].value, 12.5)
        self.assertEqual(at.session_state["m2_cal_workflow"], "load")

    def test_saved_calibration_values_are_displayed_from_the_npz(self):
        cells = table_cells(open_experiment(run_page()))
        for expected in ("4068.29", "4065.16", "2117.68", "2873.31", "0.22384", "-0.84856", "0.00151", "-0.00256", "0.95771",
                         "3.0875", "2.358"):
            self.assertIn(expected, cells, expected)

    def test_all_20_trials_and_original_statistics_are_displayed(self):
        at = open_experiment(run_page())
        with CSV.open(newline="") as file:
            rows = list(csv.DictReader(file))
        trial_table = next(t.value for t in at.table if "Estimated width (cm)" in t.value.columns)
        self.assertEqual(len(trial_table), 20)
        self.assertEqual(list(trial_table.index), [r["Trial"] for r in rows])
        for (_, shown), row in zip(trial_table.iterrows(), rows):
            self.assertEqual(shown["Estimated width (cm)"], f"{float(row['Estimated_Width_cm']):.2f}")
            self.assertEqual(shown["Height error (%)"], f"{float(row['Height_Error_Percent']):.2f}")
            self.assertEqual(shown["Distance (cm)"], f"{float(row['Distance_cm']):g}")
        stats = next(t.value for t in at.table if "Statistic" in t.value.columns)
        self.assertEqual(list(stats["Width"]), ["0.60 cm", "1.75 %", "0.91 %", "0.68 cm"])
        self.assertEqual(list(stats["Height"]), ["0.61 cm", "2.19 %", "1.25 %", "0.70 cm"])

    def test_experiment_view_has_no_write_controls(self):
        at = open_experiment(run_page({"m2_visitor_calibration": VISITOR}))
        self.assertFalse(any("Update" in l or "Save" in l or "Replace" in l for l in labels(at)))
        self.assertEqual(len(at.selectbox) + len(at.checkbox) + len(at.text_input) + len(at.number_input), 0)

    def test_display_images_exist_and_are_small(self):
        files = sorted((MODULE / "display").glob("*.jpg"))
        self.assertEqual(len(files), 11)
        self.assertLess(sum(f.stat().st_size for f in files), 2_000_000)
        at = open_experiment(run_page())
        self.assertEqual(len(at.warning), 0)     # no "display image unavailable" warning


class ReadOnlyTests(unittest.TestCase):
    def test_page_never_writes_experiment_files(self):
        written = []
        real_open = builtins.open
        real_path_open = Path.open

        def watch(path, mode="r"):
            if any(c in str(mode) for c in "wax+"):
                written.append(str(path))

        def guarded_open(file, mode="r", *args, **kwargs):
            watch(file, mode)
            return real_open(file, mode, *args, **kwargs)

        def guarded_path_open(self, mode="r", *args, **kwargs):
            watch(self, mode)
            return real_path_open(self, mode, *args, **kwargs)

        with mock.patch.object(builtins, "open", guarded_open), mock.patch.object(Path, "open", guarded_path_open), \
                mock.patch("os.replace") as replace, mock.patch("os.rename") as rename:
            for seed in (None, {"m2_visitor_calibration": VISITOR}):
                at = run_page(seed)
                self.assertFalse(at.exception)
                at = open_visitor(at)
                self.assertFalse(at.exception)
                at.button(key="navbtn_0").click()
                at.run()
                at = open_experiment(at)
                self.assertFalse(at.exception)
                at.button(key="navbtn_0").click()
                at.run()
                self.assertFalse(at.exception)
        self.assertEqual([w for w in written if "Module_2" in w], [])
        replace.assert_not_called()
        rename.assert_not_called()
        self.assertEqual(sha256(CSV), CSV_SHA256)
        self.assertEqual(sha256(NPZ), NPZ_SHA256)


if __name__ == "__main__":
    unittest.main()
