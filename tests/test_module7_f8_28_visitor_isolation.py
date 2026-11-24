"""F8.28 V0: static visitor-isolation checks for the visitor measurement path.

The visitor path must take intrinsics / distortion / baseline from the visitor's calibration only, and must never touch the
historical experiment (Recipe A file, 225/226 mm, frame manifest, boundary annotations, ground truth).  No videos are opened."""
import ast
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7" / "room_measurement_question_3"      # Question 3 packages, data and outputs
RM = MODULE / "room_measurement"
sys.path.insert(0, str(MODULE)); sys.path.insert(0, str(MODULE.parent)); sys.path.insert(0, str(ROOT / "tests"))

from room_measurement import visitor_annotations as VA, visitor_calibration as VC, visitor_measurement as VM  # noqa: E402
from test_module7_visitor_calibration import model  # noqa: E402

VISITOR_FILES = sorted(RM.glob("visitor_*.py"))
FORBIDDEN_MODULES = {"validation", "stage_c_integration", "stage_c_shift_diagnostic", "stage_e1_integration", "stage_e3_readiness",
                     "stage_f2_boundary_detection", "stage_f3_stereo_boundary", "stage_f4_model_boundaries", "single_view"}
# Phase 1 loaders / default paths that read the historical experiment's files
FORBIDDEN_NAMES = {"DEFAULT_MANIFEST", "DEFAULT_ANNOTATIONS", "DEFAULT_CALIBRATION_PATH", "DEFAULT_VIDEO_DIR", "load_manifest",
                   "load_annotations", "load_calibration", "run_estimation", "compare_to_ground_truth", "verify_videos", "load_frame_sets"}


def code_only(path):
    """Source with comments/docstrings removed (so prose may mention the experiment, code may not)."""
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body[0].value.value = ""
    return tree


class StaticIsolation(unittest.TestCase):
    def test_files_found(self):
        self.assertGreaterEqual(len(VISITOR_FILES), 5)

    def test_no_experiment_imports(self):
        for f in VISITOR_FILES:
            for n in ast.walk(code_only(f)):
                mods = [n.module or ""] + [a.name for a in n.names] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names] if isinstance(n, ast.Import) else []
                for m in mods:
                    self.assertFalse(set(m.split(".")) & FORBIDDEN_MODULES, f"{f.name} imports {m}")

    def test_no_experiment_loaders_or_default_paths(self):
        for f in VISITOR_FILES:
            for n in ast.walk(code_only(f)):
                name = n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else None
                self.assertNotIn(name, FORBIDDEN_NAMES, f"{f.name} uses {name}")

    def test_no_hardcoded_baseline_or_experiment_literals(self):
        for f in VISITOR_FILES:
            for n in ast.walk(code_only(f)):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                    self.assertNotIn(float(n.value), (225.0, 226.0, 2879.6, 2380.2), f"{f.name}:{n.lineno} numeric literal {n.value}")
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    self.assertNotRegex(n.value, r"frame_manifest|wall_boundary_annotations|method_j_experiment_reference|recipe_a_calibration|ground_truth\.json",
                                        f"{f.name}:{n.lineno}")

    def test_phase1_modules_used_by_visitor_path_hardcode_no_baseline(self):
        for name in ("geometry", "pose", "measure", "features"):
            for n in ast.walk(code_only(RM / f"{name}.py")):
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                    self.assertNotIn(float(n.value), (225.0, 226.0), f"{name}.py:{n.lineno}")

    def test_recipe_a_string_is_label_only(self):
        """The 'Recipe-A K, D' text is only a provenance label; the K/D actually used come from the caller's calibration."""
        src = (RM / "visitor_annotations.py").read_text()
        self.assertEqual(len(re.findall(r"Recipe-A", src)), 0)       # the stale label has since been removed; no Recipe-A text may remain (the K/D check below is unchanged)
        ctxs = VA.build_contexts({"AL": np.zeros((4, 4), np.uint8), "PL": np.zeros((4, 4), np.uint8)},
                                 VC.adapt_visitor_calibration(model(), 187.0).calibration)
        cal = VC.adapt_visitor_calibration(model(), 187.0).calibration
        for k, side in (("AL", "left"), ("PL", "right")):
            K, D = getattr(ctxs[k], "K", None), getattr(ctxs[k], "D", None)
            if K is None:                                               # attribute names differ: fall back to positional dataclass fields
                vals = list(vars(ctxs[k]).values()); K, D = vals[1], vals[2]
            np.testing.assert_array_equal(K, getattr(cal, f"K_{side}")); np.testing.assert_array_equal(D, getattr(cal, f"D_{side}"))


class BehaviouralIsolation(unittest.TestCase):
    def test_visitor_intrinsics_and_baseline_used(self):
        m = model(); m["K1"][0, 0] = 2999.0; m["K2"][0, 0] = 2501.0; m["D1"] = np.array([[0.11, -0.22, 0, 0, 0.0]])
        a = VC.adapt_visitor_calibration(m, 187.5)          # model's own baseline is 225 and must be ignored
        self.assertTrue(a.ok, a.issues); c = a.calibration
        self.assertEqual(c.baseline_mm, 187.5); self.assertEqual(c.K_left[0, 0], 2999.0); self.assertEqual(c.K_right[0, 0], 2501.0)
        self.assertAlmostEqual(c.D_left[0], 0.11); self.assertAlmostEqual(c.D_left[1], -0.22)

    def test_baseline_not_taken_from_model(self):
        for b in (150.0, 310.0):
            self.assertEqual(VC.adapt_visitor_calibration(model(baseline_mm=225.0), b).calibration.baseline_mm, b)

    def test_floor_row_scales_with_frame_height(self):
        self.assertEqual(VM.floor_row_threshold(3840), 2600)            # Phase 1 value reproduced exactly at its native height
        self.assertAlmostEqual(VM.floor_row_threshold(1920), 1300.0)
        self.assertAlmostEqual(VM.floor_row_threshold(2160), 2600 * 2160 / 3840)
        self.assertAlmostEqual(VM.floor_row_threshold(1080) / 1080, 2600 / 3840)   # same fraction of the frame at any height


if __name__ == "__main__":
    unittest.main()
