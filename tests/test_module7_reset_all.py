"""Reset All tests: both buttons share one reset that returns the visitor calibration/measurement workflow to its first-visit state."""
import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_module7_calibration_wizard import (AppTest, PAGE, ROOT, button, filled_state, page, texts, wiz,  # noqa: E402
                                             export_calibration, load_calibration, fk)

SELECTIONS = ROOT / "Modules" / "Module_7" / "selections"
OUTPUTS = ROOT / "Modules" / "Module_7" / "outputs"


def snapshot(*dirs):
    return {str(p): p.stat().st_mtime_ns for d in dirs if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


class WizardResetTests(unittest.TestCase):
    def test_reset_restores_defaults_and_bumps_every_uploader_generation(self):
        state = filled_state(n_pairs=12, count=12)
        wiz.toggle_exclude(state, 3)
        wiz.go_to(state, 7)
        before = dict(state["slot_gen"])
        wiz.reset_all(state)
        fresh = wiz.new_state()
        for key in ("settings", "n_pairs", "current", "photos", "excluded", "_resolve"):
            self.assertEqual(state[key], fresh[key], key)
        self.assertEqual((state["settings"], state["n_pairs"], state["current"]), ({"cols": 0, "rows": 0, "square_mm": 0.0}, 8, 0))
        self.assertEqual(set(state["slot_gen"]), {wiz.slot_key(i, s) for i in range(wiz.MAX_PAIRS) for s in wiz.SIDES})
        self.assertTrue(all(g >= before.get(k, 0) + 1 for k, g in state["slot_gen"].items()))


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class PageResetTests(unittest.TestCase):
    def assert_initial(self, at):
        self.assertFalse(at.exception)
        if at.session_state["m7_cal_workflow"] is None:                            # fresh from Reset All: neither workflow selected
            self.assertEqual(at.number_input, [])
            self.assertIn("Choose how to set up your stereo cameras.", texts(at))
            button(at, "Calibrate New Camera Setup").click().run()                 # reveal the (reset) form to check its defaults
        self.assertEqual([n.value for n in at.number_input][:4], [0, 0, 0.0, 8])
        self.assertEqual(at.session_state["m7_cal"]["photos"], {})
        self.assertEqual(at.session_state["m7_cal"]["excluded"], set())
        self.assertEqual(at.session_state["m7_cal"]["current"], 0)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertFalse(at.session_state["m7_measurement_enabled"])
        self.assertIsNone(at.session_state["m7_cal_loaded_name"])
        self.assertIsNone(at.session_state["m7_cal_notice"])
        self.assertEqual(at.session_state["m7_points"], {"left": [], "right": []})
        self.assertIsNone(at.session_state["m7_file_left"])
        self.assertIsNone(at.session_state["m7_file_right"])
        body = texts(at)
        self.assertIn("First calibrate your stereo cameras", body)
        self.assertIn("0 of 8 pairs uploaded", body)
        self.assertIn("Pair 1 of 8", body)
        self.assertNotIn("Stereo Calibration Completed", body)
        self.assertNotIn("Stereo Calibration Loaded", body)
        self.assertIsNone(button(at, "Clear Calibration"))
        self.assertEqual(len([b for b in at.button if b.label == "Reset All"]), 1)      # bottom button only with an active calibration

    def test_button_before_calibration_resets_wizard(self):
        state = filled_state(n_pairs=12, count=5)
        wiz.toggle_exclude(state, 2)
        wiz.go_to(state, 4)
        at = page(state)
        at.number_input(key=fk(at, "npairs")).set_value(12).run()
        button(at, "Reset All").click().run()
        self.assert_initial(at)
        self.assertNotIn("uploaded", " ".join(str(k) for k in at.session_state["m7_cal"]["photos"]))

    def test_both_buttons_exist_after_calibration_and_give_identical_results(self):
        results = []
        for which in (0, 1):
            at = page(filled_state())
            button(at, "Run Stereo Calibration").click().run()
            self.assertIsNotNone(at.session_state["m7_visitor_model"])
            at.toggle(key="m7_cal_details").set_value(True).run()
            resets = [b for b in at.button if b.label == "Reset All"]
            self.assertEqual([b.key for b in resets], ["m7_cal_reset_top", "m7_cal_reset_bottom"])
            at.session_state["m7_measurement_enabled"] = True
            at.run()
            resets[which].click().run()
            self.assert_initial(at)
            self.assertFalse(at.toggle(key="m7_cal_details").value if at.toggle else False)
            results.append({k: at.session_state[k] for k in at.session_state.keys()})
        a, b = (r.copy() for r in results)
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            if k not in ("m7_visitor_model",):
                self.assertEqual(repr(a[k]), repr(b[k]), k)

    def test_loaded_calibration_removed_and_uploader_generation_rotated(self):
        cal = wiz.run_calibration(filled_state())
        loaded = load_calibration(export_calibration(cal, 10, cal["camera"]))
        at = page(None, m7_visitor_model=loaded, m7_cal_loaded_name="mine.npz", m7_measurement_enabled=True,
                  m7_file_left=("l.jpg", b"x"), m7_file_right=("r.jpg", b"y"), m7_pair_id="abc", m7_ack="abc",
                  m7_points={"left": [(1, 2)], "right": []}, m7_left_confirmed=True, m7_right_confirmed=True, m7_calculated=True)
        gens = (at.session_state["m7_upload_gen"], at.session_state["m7_cal_npz_gen"])
        button(at, "Reset All").click().run()
        self.assert_initial(at)
        self.assertGreater(at.session_state["m7_upload_gen"], gens[0])
        self.assertGreater(at.session_state["m7_cal_npz_gen"], gens[1])
        for key in ("m7_pair_id", "m7_ack", "m7_dialog_opened"):
            self.assertIsNone(at.session_state[key])
        self.assertFalse(at.session_state["m7_left_confirmed"] or at.session_state["m7_right_confirmed"] or at.session_state["m7_calculated"])

    def test_checkerboard_widgets_return_to_zero_after_edit(self):
        at = page()
        at.number_input(key=fk(at, "cols")).set_value(8).run()
        at.number_input(key=fk(at, "rows")).set_value(6).run()
        at.number_input(key=fk(at, "square")).set_value(25.0).run()
        at.number_input(key=fk(at, "npairs")).set_value(15).run()
        button(at, "Reset All").click().run()
        self.assert_initial(at)

    def test_widgets_render_defaults_after_reset_from_filled_state(self):
        for which in ("m7_cal_reset_top", "m7_cal_reset_bottom"):
            state = filled_state(n_pairs=10, count=10 if which == "m7_cal_reset_bottom" else 5)
            square = 30.0 if which == "m7_cal_reset_bottom" else 20.0       # bottom button needs a calibratable (fixture) board
            state["settings"]["square_mm"] = square
            at = page(state)
            if which == "m7_cal_reset_bottom":
                button(at, "Run Stereo Calibration").click().run()
                self.assertIsNotNone(at.session_state["m7_visitor_model"])
            self.assertEqual([n.value for n in at.number_input][:4], [8, 6, square, 10])
            self.assertIn(f"{10 if which == 'm7_cal_reset_bottom' else 5} of 10 pairs uploaded", texts(at))
            old_keys = [n.key for n in at.number_input][:4]
            next(b for b in at.button if b.key == which).click().run()
            self.assert_initial(at)
            self.assertEqual([n.value for n in at.number_input][:4], [0, 0, 0.0, 8])
            self.assertEqual(at.number_input(key=fk(at, "npairs")).value, at.session_state["m7_cal"]["n_pairs"])
            self.assertNotEqual(old_keys, [n.key for n in at.number_input][:4])     # fresh widgets, so a stale browser value cannot return
            for _ in range(2):                                                       # a later rerun must not restore old values
                at.run()
                self.assert_initial(at)
            self.assertEqual(at.session_state["m7_cal"]["settings"], {"cols": 0, "rows": 0, "square_mm": 0.0})

    def test_repeated_reset_all_clicks(self):
        at = page(filled_state(n_pairs=10, count=5))
        for _ in range(3):
            button(at, "Reset All").click().run()
            self.assert_initial(at)
            at.number_input(key=fk(at, "cols")).set_value(8).run()
            at.number_input(key=fk(at, "npairs")).set_value(11).run()
        button(at, "Reset All").click().run()
        self.assert_initial(at)

    def test_page_only_calls_wizard_functions_that_exist(self):
        tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "wiz"}
        self.assertIn("reset_all", used)
        self.assertTrue(callable(wiz.reset_all))
        self.assertEqual([a for a in sorted(used) if not hasattr(wiz, a)], [])

    def test_both_buttons_use_the_same_callback(self):
        tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
        callbacks = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "button" and node.args \
                    and getattr(node.args[0], "value", None) == "Reset All":
                callbacks[next(k.value.id for k in node.keywords if k.arg == "on_click")] = True
        self.assertEqual(list(callbacks), ["reset_all"])
        self.assertEqual(sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "Reset All"), 3)   # calibration top + bottom, results bottom

    def test_reset_never_touches_saved_experiment(self):
        before = snapshot(SELECTIONS, OUTPUTS, ROOT / "Modules" / "Module_7" / "calibration", ROOT / "Modules" / "Module_7" / "data")
        cal = wiz.run_calibration(filled_state())
        at = page(filled_state(), m7_visitor_model=cal, m7_show_experiment=True)
        experiment = at.session_state["m7_show_experiment"]
        button(at, "Reset All").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["m7_show_experiment"], experiment)
        self.assertIn("My Experiment & Results", [h.value for h in at.header])
        self.assertEqual(at.session_state["module7_part"], "1")
        self.assertEqual(snapshot(SELECTIONS, OUTPUTS, ROOT / "Modules" / "Module_7" / "calibration",
                                  ROOT / "Modules" / "Module_7" / "data"), before)

    def test_reset_functions_do_not_reference_files_or_model_b(self):
        tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "reset_all")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        self.assertFalse(names & {"get_model", "load_model_b", "load_selection", "open", "unlink", "remove", "rmtree", "write_text"})


if __name__ == "__main__":
    unittest.main()
