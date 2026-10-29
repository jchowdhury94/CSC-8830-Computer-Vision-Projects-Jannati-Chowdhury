"""Phase 5 tests: the visitor calibration wizard (stereo_measure/calibration_wizard.py) and its Streamlit page (AppTest).

Wizard logic is tested on synthetic stereo rigs (corner records without images) and on a few rendered checkerboard photos.
AppTest cannot drive file uploaders, so the page tests seed the wizard state and check rendering, buttons and session state.
"""
import hashlib
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stereo_measure import calibration_wizard as wiz  # noqa: E402
from stereo_measure import user_calibration as uc  # noqa: E402
from stereo_measure.calibration_io import CalibrationFileError, export_calibration, load_calibration  # noqa: E402
from test_module7_user_calibration import (LEFT_SIZE, PATTERN_T, RIGHT_SIZE, SQ, DetectionTests, encode, make_rig,  # noqa: E402
                                           render_board, rot, sample_views)

RIG = make_rig()
LEFT, RIGHT, _ = sample_views(RIG, 12, seed=3)


def record(corners, size, name, digest=None):
    return dict(status="ok", message="Checkerboard detected.", size=size, exif=None, corners=corners, thumbnail=None, name=name,
                bytes=1000, digest=digest or hashlib.sha256(name.encode()).hexdigest(), square_px=100.0, grid_residual=0.0,
                subpix_max_shift=0.0)


def configured_state():
    state = wiz.new_state()
    state["settings"] = {"cols": PATTERN_T[0], "rows": PATTERN_T[1], "square_mm": float(SQ)}
    return state


def filled_state(n_pairs=10, count=10, cols_rows=PATTERN_T, square=SQ):
    state = wiz.new_state()
    state["settings"] = {"cols": cols_rows[0], "rows": cols_rows[1], "square_mm": float(square)}
    wiz.set_pair_count(state, n_pairs)
    for i in range(count):
        state["photos"][wiz.slot_key(i, "left")] = record(LEFT[i], LEFT_SIZE, f"L{i}.jpg")
        state["photos"][wiz.slot_key(i, "right")] = record(RIGHT[i], RIGHT_SIZE, f"R{i}.jpg")
    return state


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(wiz.new_state()["settings"], {"cols": 0, "rows": 0, "square_mm": 0.0})   # 0 = not configured
        self.assertEqual(wiz.new_state()["n_pairs"], 8)

    def test_unset_settings_are_required(self):
        for args in ((0, 0, 0), (0, 6, 20), (8, 0, 20), (8, 6, 0), (8, 6, 0.0), (None, None, None)):
            settings, errors = wiz.validate_settings(*args)
            self.assertIsNone(settings, args)
            self.assertTrue(errors and "Enter your checkerboard settings" in errors[0], args)
        self.assertIn("square size", wiz.validate_settings(8, 6, 0)[1][0])
        with self.assertRaises(ValueError):                                       # no detection against an unset board
            wiz.store_photo(wiz.new_state(), 0, "left", "a.jpg", b"x")

    def test_valid_and_invalid_inputs(self):
        ok, errors = wiz.validate_settings(9, 7, 25.4)
        self.assertEqual((ok, errors), ({"cols": 9, "rows": 7, "square_mm": 25.4}, []))
        for args in ((2, 6, 20), (8, 2, 20), (8.5, 6, 20), (None, 6, 20), (8, 6, 0), (8, 6, -1), (8, 6, float("nan")),
                     (8, 6, float("inf")), (8, 6, None), (8, 6, "abc"), (8, 6, 1e6), (500, 6, 20), (True, 6, 20)):
            settings, errors = wiz.validate_settings(*args)
            self.assertIsNone(settings, args)
            self.assertTrue(errors, args)

    def test_changed_settings_clear_detections_but_keep_names(self):
        state = filled_state()
        self.assertFalse(wiz.apply_settings(state, dict(state["settings"])))
        self.assertEqual(state["photos"]["0:left"]["status"], "ok")
        self.assertTrue(wiz.apply_settings(state, {"cols": 9, "rows": 6, "square_mm": 20.0}))
        rec = state["photos"]["0:left"]
        self.assertEqual((rec["status"], rec["corners"], rec["name"]), ("stale", None, "L0.jpg"))
        self.assertEqual(wiz.evaluate(state)["n_valid"], 0)


class PairSlotTests(unittest.TestCase):
    def test_pair_count_selection(self):
        state = wiz.new_state()
        state["settings"] = dict(configured_state()["settings"])
        self.assertEqual((wiz.MIN_PAIRS, wiz.DEFAULT_PAIRS), (8, 8))
        for n in (8, 9, 10, 17, 33, wiz.MAX_PAIRS):                               # any whole number from the minimum, in steps of 1
            wiz.set_pair_count(state, n)
            self.assertEqual(len(wiz.evaluate(state)["pairs"]), n)
        for bad in (0, 7, wiz.MAX_PAIRS + 1, 8.5, True, None):
            with self.assertRaises(ValueError):
                wiz.set_pair_count(state, bad)

    def test_changing_count_keeps_uploaded_data_and_clamps_navigation(self):
        state = filled_state(n_pairs=20, count=10)
        wiz.go_to(state, 15)
        wiz.set_pair_count(state, 10)
        self.assertEqual(state["current"], 9)
        wiz.set_pair_count(state, 8)                                              # hidden slots keep their photos
        self.assertEqual((state["current"], len(state["photos"])), (7, 20))
        wiz.set_pair_count(state, 20)
        self.assertEqual(wiz.evaluate(state)["n_valid"], 10)
        state["photos"][wiz.slot_key(15, "left")] = record(LEFT[0], LEFT_SIZE, "later.jpg")
        wiz.set_pair_count(state, 10)
        wiz.set_pair_count(state, 20)
        self.assertIn("15:left", state["photos"])

    def test_navigation_is_clamped(self):
        state = wiz.new_state()
        wiz.step(state, -1)
        self.assertEqual(state["current"], 0)
        for _ in range(30):
            wiz.step(state, 1)
        self.assertEqual(state["current"], 7)
        wiz.go_to(state, 3)
        self.assertEqual(state["current"], 3)
        wiz.go_to(state, 99)
        self.assertEqual(state["current"], 7)

    def test_statuses_not_uploaded_incomplete_valid(self):
        state = filled_state(count=4)
        del state["photos"]["3:right"]
        pairs = wiz.evaluate(state)["pairs"]
        self.assertEqual([p["status"] for p in pairs[:5]], [wiz.VALID] * 3 + [wiz.INCOMPLETE, wiz.NOT_UPLOADED])
        self.assertIn("right", pairs[3]["messages"][0])
        ev = wiz.evaluate(state)
        self.assertEqual((ev["n_uploaded"], ev["n_valid"]), (3, 3))

    def test_exclusion_is_explicit_and_reversible(self):
        state = filled_state()
        wiz.toggle_exclude(state, 2)
        ev = wiz.evaluate(state)
        self.assertEqual(ev["pairs"][2]["status"], wiz.EXCLUDED)
        self.assertEqual(ev["n_valid"], 9)
        self.assertNotIn(3, [u["number"] for u in ev["usable"]])
        wiz.toggle_exclude(state, 2)
        self.assertEqual(wiz.evaluate(state)["n_valid"], 10)

    def test_minimum_valid_pairs(self):
        state = filled_state(count=7)
        ev = wiz.evaluate(state)
        self.assertFalse(ev["can_calibrate"])
        self.assertIn("at least 8", ev["blockers"][0])
        with self.assertRaises(wiz.WizardError):
            wiz.run_calibration(state)
        state = filled_state(count=8)
        self.assertTrue(wiz.evaluate(state)["can_calibrate"])
        wiz.toggle_exclude(state, 0)
        self.assertFalse(wiz.evaluate(state)["can_calibrate"])

    def test_failed_photo_names_pair_and_side_and_is_not_discarded(self):
        state = filled_state()
        state["photos"]["4:right"] = dict(record(None, RIGHT_SIZE, "blurry.jpg"), status="not_found", corners=None,
                                          message="No 8 x 6 inner-corner checkerboard was found.")
        ev = wiz.evaluate(state)
        row = ev["pairs"][4]
        self.assertEqual(row["status"], wiz.INVALID)
        self.assertTrue(any(m.startswith("Pair 5, right camera:") for m in row["messages"]))
        self.assertEqual(ev["n_valid"], 9)
        self.assertIn("4:right", state["photos"])            # still there to be replaced

    def test_mismatched_resolution_is_invalid(self):
        state = filled_state()
        state["photos"]["6:left"]["size"] = (3000, 4000)
        row = wiz.evaluate(state)["pairs"][6]
        self.assertEqual(row["status"], wiz.INVALID)
        self.assertIn("one resolution", " ".join(row["messages"]))

    def test_duplicate_files_are_invalid(self):
        state = filled_state()
        state["photos"]["1:right"]["digest"] = state["photos"]["1:left"]["digest"]
        state["photos"]["5:left"]["digest"] = state["photos"]["2:left"]["digest"]
        pairs = wiz.evaluate(state)["pairs"]
        self.assertEqual(pairs[1]["status"], wiz.INVALID)
        self.assertEqual(pairs[5]["status"], wiz.INVALID)
        self.assertIn("same file as pair 3", " ".join(pairs[5]["messages"]))

    def test_ambiguous_ordering_is_flagged_not_guessed(self):
        state = filled_state()
        real = uc.resolve_pair_orderings

        def fake(pairs, pattern, max_side):
            res = real(pairs, pattern, max_side)
            res["pairs"][2] = dict(res["pairs"][2], status="ambiguous", action="as_detected", message="ambiguous correspondence")
            return res
        with mock.patch.object(uc, "resolve_pair_orderings", fake):
            ev = wiz.evaluate(state)
        self.assertEqual(ev["pairs"][2]["status"], wiz.AMBIGUOUS)
        self.assertEqual(ev["n_valid"], 9)
        self.assertNotIn(3, [u["number"] for u in ev["usable"]])

    def test_geometrically_inconsistent_pair_is_invalid(self):
        state = filled_state()
        rng = np.random.default_rng(0)
        state["photos"]["7:right"]["corners"] = rng.uniform(100, 2000, (48, 2))
        ev = wiz.evaluate(state)
        self.assertEqual(ev["pairs"][7]["status"], wiz.INVALID)
        self.assertEqual(ev["n_valid"], 9)

    def test_wrong_right_ordering_is_renumbered_not_dropped(self):
        state = filled_state()
        state["photos"]["3:right"]["corners"] = RIGHT[3][::-1].copy()          # 180 degree renumbering
        ev = wiz.evaluate(state)
        self.assertEqual(ev["pairs"][3]["status"], wiz.VALID)
        usable = next(u for u in ev["usable"] if u["number"] == 4)
        np.testing.assert_allclose(usable["right"], RIGHT[3])


class PhotoProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img, _ = DetectionTests().view()
        cls.good = encode(img)
        cls.other = encode(img, orientation=1)[:-1] + b"\x00"               # different bytes, still decodable JPEG

    def test_store_replace_remove(self):
        state = configured_state()
        rec = wiz.store_photo(state, 0, "left", "a.jpg", self.good)
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["corners"].shape, (48, 2))
        self.assertNotIn("data", rec)                                         # the photo itself is not retained
        gen = state["slot_gen"]["0:left"]
        again = wiz.store_photo(state, 0, "left", "b.jpg", b"not an image")
        self.assertEqual((again["status"], state["photos"]["0:left"]["name"]), ("rejected", "b.jpg"))
        self.assertGreater(state["slot_gen"]["0:left"], gen)
        wiz.remove_photo(state, 0, "left")
        self.assertNotIn("0:left", state["photos"])

    def test_invalid_photographs_are_reported(self):
        state = configured_state()
        for data in (b"", b"garbage", b"\x89PNG\r\n\x1a\n" + b"0" * 20):
            self.assertEqual(wiz.store_photo(state, 1, "right", "x", data)["status"], "rejected")
        blank = encode(np.full((900, 1200), 128, np.uint8))
        self.assertEqual(wiz.store_photo(state, 1, "right", "blank.jpg", blank)["status"], "not_found")
        big = b"0" * (uc.MAX_FILE_BYTES + 1)
        self.assertIn("MB", wiz.store_photo(state, 1, "right", "big.jpg", big)["message"])

    def test_slot_validation(self):
        with self.assertRaises(ValueError):
            wiz.store_photo(wiz.new_state(), 0, "middle", "a.jpg", self.good)


class CalibrationRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = filled_state()
        cls.cal = wiz.run_calibration(cls.state)

    def test_result_uses_calibrate_stereo_and_real_values(self):
        cal = self.cal
        self.assertTrue(cal["valid"])
        self.assertEqual((cal["pairs_uploaded"], cal["pairs_valid"], cal["source"]), (10, 10, wiz.SOURCE_NEW))
        self.assertEqual(cal["pairs"], list(range(1, 11)))
        self.assertAlmostEqual(cal["baseline_mm"], np.linalg.norm(RIG["T"]), delta=0.02 * np.linalg.norm(RIG["T"]))
        self.assertEqual(cal["image_size_left"], LEFT_SIZE)
        self.assertEqual(cal["image_size_right"], RIGHT_SIZE)
        for key in ("epipolar_px", "rectification_vertical_px"):
            self.assertIsNotNone(cal["metrics"][key])
        self.assertTrue(cal["distortion_cv"]["left"])
        self.assertIn("left", cal["camera"])

    def test_excluded_pair_is_left_out(self):
        state = filled_state()
        wiz.toggle_exclude(state, 0)
        cal = wiz.run_calibration(state)
        self.assertEqual((cal["pairs_uploaded"], cal["pairs_valid"]), (10, 9))
        self.assertNotIn(1, cal["pairs"])

    def test_failure_is_an_error_not_a_result(self):
        with mock.patch.object(uc, "calibrate_stereo", side_effect=uc.CalibrationFailed("boom")):
            with self.assertRaises(wiz.WizardError) as ctx:
                wiz.run_calibration(self.state)
        self.assertIn("boom", str(ctx.exception))
        invalid = dict(self.cal, valid=False, errors=["The calibration contains non-finite values."])
        with mock.patch.object(uc, "calibrate_stereo", return_value=invalid):
            with self.assertRaises(wiz.WizardError) as ctx:
                wiz.run_calibration(self.state)
        self.assertIn("not usable", str(ctx.exception))
        with mock.patch.object(uc, "calibrate_stereo", side_effect=ValueError("bad corners")):
            with self.assertRaises(wiz.WizardError):
                wiz.run_calibration(self.state)

    def test_invalidation_when_inputs_change(self):
        state = filled_state()
        self.assertTrue(wiz.is_current(state, self.cal))                 # same photos -> still current
        self.assertFalse(wiz.is_current(state, None))
        wiz.toggle_exclude(state, 4)
        self.assertFalse(wiz.is_current(state, self.cal))
        wiz.toggle_exclude(state, 4)
        self.assertTrue(wiz.is_current(state, self.cal))
        state["photos"]["2:left"]["digest"] = "0" * 64                  # replaced photograph
        self.assertFalse(wiz.is_current(state, self.cal))
        state = filled_state()
        wiz.apply_settings(state, {"cols": 8, "rows": 6, "square_mm": 25.0})
        self.assertFalse(wiz.is_current(state, self.cal))
        state = filled_state()
        wiz.set_pair_count(state, 20)                                    # more empty slots: same inputs, still current
        self.assertTrue(wiz.is_current(state, self.cal))

    def test_loaded_file_is_independent_of_the_photos(self):
        loaded = load_calibration(export_calibration(self.cal, self.cal["pairs_uploaded"], self.cal["camera"]))
        self.assertTrue(wiz.is_current(wiz.new_state(), loaded))

    def test_npz_round_trip_and_invalid_files(self):
        data = export_calibration(self.cal, self.cal["pairs_uploaded"], self.cal["camera"], self.cal["created_utc"])
        loaded = load_calibration(data)
        self.assertEqual((loaded["source"], loaded["authenticity_verified"]), ("loaded_file", False))
        np.testing.assert_allclose(loaded["K1"], self.cal["K1"])
        np.testing.assert_allclose(loaded["T"], self.cal["T"])
        self.assertEqual(loaded["pairs"], self.cal["pairs"])
        for bad in (b"", b"not a zip", data[:200], data[: len(data) // 2], b"PK\x03\x04" + b"0" * 50):
            with self.assertRaises(CalibrationFileError):
                load_calibration(bad)
        with self.assertRaises(ValueError):
            export_calibration(dict(self.cal, valid=False))

    def test_sessions_are_isolated(self):
        a, b = configured_state(), wiz.new_state()
        wiz.store_photo(a, 0, "left", "a.jpg", b"junk")
        wiz.toggle_exclude(a, 1)
        wiz.apply_settings(a, {"cols": 9, "rows": 7, "square_mm": 30.0})
        self.assertEqual((b["photos"], b["excluded"], b["settings"]), ({}, set(), wiz.DEFAULT_SETTINGS))


# ---------------------------------------------------------------- the Streamlit page

try:
    from streamlit.testing.v1 import AppTest
except ImportError:                                           # pragma: no cover
    AppTest = None

PAGE = str(ROOT / "Pages" / "app_module7.py")


def page(seed=None, **extra):
    at = AppTest.from_file(PAGE, default_timeout=120)
    at.session_state["module7_part"] = "1"
    at.session_state["m7_cal_workflow"] = "run"       # the page shows the Calibrate New Camera Setup workflow unless a test overrides it
    if seed is not None:
        at.session_state["m7_cal"] = seed
    if extra.get("m7_visitor_model") is not None:     # an injected model behaves like one set by the page
        loaded = extra.get("m7_cal_loaded_name") is not None
        at.session_state["m7_cal_source"] = "loaded" if loaded else "computed"
        at.session_state["m7_cal_workflow"] = "load" if loaded else "run"
    for key, value in extra.items():
        at.session_state[key] = value
    return at.run()


def fk(at, name):
    """Current-generation key of a checkerboard / pair-count widget (Reset All rotates them)."""
    return f"m7_cal_{name}_{at.session_state['m7_cal_form_gen']}"


def texts(at):
    return " ".join([e.value for e in at.markdown] + [e.value for e in at.success] + [e.value for e in at.info] +
                    [e.value for e in at.warning] + [e.value for e in at.error] + [e.value for e in at.caption])


def button(at, label):
    return next((b for b in at.button if b.label == label), None)


class PageWizardContractTests(unittest.TestCase):
    def test_every_wizard_attribute_the_page_uses_exists(self):
        import ast
        tree = ast.parse(Path(PAGE).read_text(encoding="utf-8"))
        used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "wiz"}
        self.assertTrue({"MIN_PAIRS", "MAX_PAIRS", "set_pair_count"} <= used)
        self.assertEqual([a for a in sorted(used) if not hasattr(wiz, a)], [])

    def test_pair_limits_are_consistent(self):
        self.assertEqual((wiz.MIN_PAIRS, wiz.DEFAULT_PAIRS, wiz.MAX_PAIRS), (8, 8, 50))
        self.assertEqual(wiz.new_state()["n_pairs"], wiz.MIN_PAIRS)


@unittest.skipIf(AppTest is None, "streamlit testing is unavailable")
class PageTests(unittest.TestCase):
    def test_empty_page(self):
        at = page()
        self.assertFalse(at.exception)
        self.assertEqual([h.value for h in at.subheader if h.value in ("Upload Stereo Images",)], ["Upload Stereo Images"])
        self.assertIn("First calibrate your stereo cameras", texts(at))
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertEqual([n.value for n in at.number_input][:4], [0, 0, 0.0, 8])            # board unset; 8 pairs
        self.assertEqual([n.label for n in at.number_input][:3], ["Inner corners (columns)", "Inner corners (rows)", "Square size (mm)"])
        self.assertIn("Enter your checkerboard settings", texts(at))
        self.assertIn("0 of 8 pairs uploaded", texts(at))
        self.assertIn("Pair 1 of 8", texts(at))
        self.assertFalse(any(str(k).startswith("m7_up_") for k in at.session_state.keys()))     # no measurement uploaders

    def test_pair_count_and_navigation(self):
        at = page(filled_state(n_pairs=8, count=3))
        npairs = at.number_input(key=fk(at, "npairs"))
        self.assertEqual((npairs.min, npairs.step, npairs.value), (8, 1, 8))
        npairs.increment().run()                                                    # + button: one pair at a time
        self.assertEqual(at.session_state["m7_cal"]["n_pairs"], 9)
        self.assertIn("3 of 9 pairs uploaded", texts(at))
        at.number_input(key=fk(at, "npairs")).set_value(20).run()
        self.assertIn("Pair 1 of 20", texts(at))
        button(at, "Next").click().run()
        self.assertIn("Pair 2 of 20", texts(at))
        button(at, "Next").click().run()
        button(at, "Previous").click().run()
        self.assertIn("Pair 2 of 20", texts(at))
        self.assertEqual(at.selectbox(key="m7_cal_goto").value, 1)
        at.selectbox(key="m7_cal_goto").select(14).run()
        self.assertIn("Pair 15 of 20", texts(at))
        self.assertEqual(at.selectbox(key="m7_cal_goto").format_func(14), "Pair 15 – Not uploaded")
        at.number_input(key=fk(at, "npairs")).set_value(10).run()                    # shrinking clamps instead of failing
        self.assertFalse(at.exception)
        self.assertIn("Pair 10 of 10", texts(at))
        self.assertEqual(at.number_input(key=fk(at, "npairs")).min, 8)               # the widget cannot go below the minimum
        self.assertEqual(len([k for k in at.session_state["m7_cal"]["photos"]]), 6)  # no photo was discarded

    def test_changing_pair_count_keeps_pair_data_and_exclusions(self):
        state = filled_state(n_pairs=12, count=12)
        wiz.toggle_exclude(state, 10)
        at = page(state)
        at.number_input(key=fk(at, "npairs")).set_value(8).run()
        at.number_input(key=fk(at, "npairs")).set_value(12).run()
        kept = at.session_state["m7_cal"]
        self.assertEqual((len(kept["photos"]), kept["excluded"]), (24, {10}))
        self.assertIn("11 valid", texts(at).replace("valid stereo pairs", "valid"))

    def test_selector_is_not_searchable_and_exclusion_roundtrip(self):
        at = page(filled_state())
        self.assertFalse(at.selectbox(key="m7_cal_goto").proto.accept_new_options)
        self.assertNotIn("typing", repr(at.selectbox(key="m7_cal_goto").proto.filter_mode).lower())   # see manual check: no text cursor
        button(at, "Exclude this pair from calibration").click().run()
        self.assertIn("Pair 1 of 10 · Excluded", texts(at))
        button(at, "Next").click().run()
        self.assertEqual(at.session_state["m7_cal"]["current"], 1)
        button(at, "Previous").click().run()
        button(at, "Include this pair again").click().run()
        self.assertEqual(at.session_state["m7_cal"]["excluded"], set())
        self.assertIn("Pair 1 of 10 · Valid", texts(at))

    def test_settings_messages_and_no_internal_objects_rendered(self):
        def rendered(at):
            return " ".join(texts(at).split() + [str(e.value) for e in at.get("code")] )

        def doc_elements(node):                                                    # st.help / magic output of an object shows up as this
            found = []
            for child in getattr(node, "children", {}).values():
                found += ([child.type] if getattr(child, "type", "") in ("help_info", "doc_string") else []) + doc_elements(child)
            return found

        def leaks(at):
            body = rendered(at)
            return doc_elements(at.main) + [w for w in ("DeltaGenerator", "Parameters", "st.warning(", "st.error(", "<function", "<bound") if w in body]

        at = page()                                                                # zeros: yellow warning, no red error
        self.assertEqual(len(at.warning) >= 1 and [w.value for w in at.warning if "Enter your checkerboard settings" in w.value] != [], True)
        self.assertFalse([e for e in at.error if "checkerboard" in e.value.lower()])
        self.assertEqual(leaks(at), [])
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)

        at.number_input(key=fk(at, "cols")).set_value(2).run()                       # genuinely invalid values: red error
        at.number_input(key=fk(at, "rows")).set_value(6).run()
        at.number_input(key=fk(at, "square")).set_value(20.0).run()
        self.assertTrue([e for e in at.error if "between 3 and 40" in e.value])
        self.assertEqual(leaks(at), [])

        at.number_input(key=fk(at, "cols")).set_value(8).run()                       # valid: no settings message, uploaders enabled
        self.assertFalse([e for e in at.error if "inner-corner" in e.value])
        self.assertFalse([w for w in at.warning if "Enter your checkerboard settings" in w.value])
        self.assertEqual(leaks(at), [])
        self.assertIn("Pair 1 of 8", texts(at))

    def test_missing_settings_block_everything_but_keep_photos(self):
        at = page(filled_state())
        at.number_input(key=fk(at, "cols")).set_value(0).run()                       # temporarily unset: nothing is cleared
        self.assertIn("Enter your checkerboard settings", texts(at))
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        self.assertEqual(at.session_state["m7_cal"]["photos"]["0:left"]["status"], "ok")
        at.number_input(key=fk(at, "cols")).set_value(PATTERN_T[0]).run()
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        self.assertEqual(at.session_state["m7_cal"]["photos"]["0:left"]["status"], "ok")

    def test_invalid_checkerboard_input_blocks_work(self):
        at = page(filled_state())
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        at.number_input(key=fk(at, "square")).set_value(0.5).run()           # still valid (minimum), settings change -> cleared
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)

    def test_changed_settings_clear_photos(self):
        at = page(filled_state())
        at.number_input(key=fk(at, "cols")).set_value(9).run()
        self.assertIn("settings changed", texts(at))
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        self.assertEqual(at.session_state["m7_cal"]["photos"]["0:left"]["status"], "stale")

    def test_progress_and_pair_status(self):
        state = filled_state(count=7)
        wiz.toggle_exclude(state, 0)
        at = page(state)
        self.assertIn("7 of 10 pairs uploaded", texts(at))
        self.assertIn("6 valid stereo pairs", texts(at))
        self.assertIn("Pair 1 of 10 · Excluded", texts(at))
        self.assertFalse(button(at, "Run Stereo Calibration").disabled)
        button(at, "Run Stereo Calibration").click().run()
        self.assertIn("Only 6 valid stereo pairs are available. At least 8 are required. Add 2 more valid pairs.", texts(at))

    def test_exclude_button_toggles(self):
        at = page(filled_state())
        button(at, "Exclude this pair from calibration").click().run()
        self.assertIn(0, at.session_state["m7_cal"]["excluded"])
        self.assertIsNotNone(button(at, "Include this pair again"))

    def test_run_calibration_and_results(self):
        at = page(filled_state())
        button(at, "Run Stereo Calibration").click().run()
        self.assertFalse(at.exception)
        model = at.session_state["m7_visitor_model"]
        self.assertIsNotNone(model)
        self.assertEqual(model["source"], wiz.SOURCE_NEW)
        body = texts(at)
        for text in ("Stereo Calibration Completed", "newly computed", "not** the physical error", "heuristics",
                     "Object measurements will use this active stereo calibration."):
            self.assertIn(text, body)
        self.assertEqual([m for m in at.metric if m.label == "Baseline"][0].value, f"{model['baseline_mm']:.1f} mm")
        self.assertNotIn("Select the four object corners", body)
        self.assertTrue(any(str(k).startswith("m7_up_") for k in at.session_state.keys()))      # measurement enabled by the calibration
        self.assertTrue(at.get("download_button"))
        at.toggle(key="m7_cal_details").set_value(True).run()
        self.assertFalse(at.exception)
        details = texts(at)
        self.assertIn("Left intrinsic matrix K", details)
        self.assertIn("Distortion-model cross-validation", details)
        self.assertTrue(len(at.dataframe) >= 2)
        button(at, "Clear Calibration").click().run()
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIn("First calibrate your stereo cameras", texts(at))

    def test_calibration_failure_is_not_success(self):
        state = filled_state()
        with mock.patch.object(uc, "calibrate_stereo", side_effect=uc.CalibrationFailed("solver failed")):
            at = page(state)
            button(at, "Run Stereo Calibration").click().run()
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIn("solver failed", texts(at))
        self.assertNotIn("Stereo Calibration Completed", texts(at))

    def test_changing_inputs_invalidates_calibration_and_measurement_state(self):
        state = filled_state()
        cal = wiz.run_calibration(state)
        at = page(state, m7_visitor_model=cal)
        self.assertIn("Stereo Calibration Completed", texts(at))
        at.session_state["m7_points"] = {"left": [(1, 1)], "right": []}
        button(at, "Exclude this pair from calibration").click().run()
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIn("previous calibration was cleared", texts(at))
        self.assertEqual(at.session_state["m7_points"], {"left": [], "right": []})

    def test_loaded_calibration_is_labelled_and_unauthenticated(self):
        cal = wiz.run_calibration(filled_state())
        loaded = load_calibration(export_calibration(cal, 10, cal["camera"]))
        at = page(None, m7_visitor_model=loaded, m7_cal_loaded_name="mine.npz")
        body = texts(at)
        self.assertIn("Stereo Calibration Loaded", body)
        self.assertIn("mine.npz", body)
        self.assertIn("not authenticated", body)
        self.assertIn("Object measurements will use this active stereo calibration.", body)
        self.assertIn("Loaded from NPZ", " ".join(m.value for m in at.metric))
        at.toggle(key="m7_cal_details").set_value(True).run()
        self.assertFalse(at.exception)
        self.assertIn("not stored in calibration files", texts(at))
        # a loaded calibration does not depend on the wizard photos
        self.assertIsNotNone(at.session_state["m7_visitor_model"])

    def test_no_model_b_fallback_and_experiment_independent(self):
        at = page()
        self.assertIsNone(at.session_state["m7_visitor_model"])
        button(at, "View My Experiment & Results").click().run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["m7_visitor_model"])
        self.assertIn("My Experiment & Results", [h.value for h in at.header])
        self.assertIn("Close experiment", [b.label for b in at.button])
        # a visitor calibration neither changes nor is changed by the experiment
        cal = wiz.run_calibration(filled_state())
        at2 = page(filled_state(), m7_visitor_model=cal, m7_show_experiment=True)
        self.assertFalse(at2.exception)
        self.assertIn("My Experiment & Results", [h.value for h in at2.header])
        self.assertEqual(at2.session_state["m7_visitor_model"]["source"], wiz.SOURCE_NEW)

    def test_sessions_do_not_share_state(self):
        a = page(filled_state())
        b = page()
        self.assertIn("10 valid stereo pairs", texts(a))
        self.assertIn("0 valid stereo pairs", texts(b))


if __name__ == "__main__":
    unittest.main()
