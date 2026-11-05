"""Module 2 Phase 3: resolution of a legacy calibration file detected from an original calibration photograph.

Synthetic images only. Streamlit pages run under AppTest, including simulated file uploads (no browser).
Nothing here writes into Modules/Module_2.
"""
import hashlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_2"
PAGE = ROOT / "Pages" / "app_module2.py"
NPZ = MODULE / "camera_calibration.npz"
CSV = MODULE / "measurement_results.csv"
NPZ_SHA256 = "15a634af6f82e8c29b37d935887fb7fa7d37e6e552624f50ce0e1327b6cf702c"
CSV_SHA256 = "95f72ccc2d9f8b242d991d98ca6d354d2643fa3a3f8ded40d4c6bb17c53693eb"

sys.path.insert(0, str(MODULE))
import calibration_file as cf  # noqa: E402
import visitor_calibration as vc  # noqa: E402

PHOTO_LABEL = "Upload Calibration Image"


def image_bytes(width, height, fmt="JPEG", orientation=None):
    """Stored pixel size is (width, height); `orientation` is written to the EXIF header without touching the pixels."""
    im = Image.new("RGB", (width, height), (120, 130, 140))
    out = io.BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        kwargs["exif"] = exif
    im.save(out, fmt, **kwargs)
    return out.getvalue()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def photo_uploader(at):
    found = [u for u in at.get("file_uploader") if u.label == PHOTO_LABEL]
    return found[0] if found else None


def legacy_page(extra=None):
    legacy = cf.parse_calibration(NPZ.read_bytes())
    seed = {"m2_page": "visitor", "m2_load_pending": legacy, "m2_load_base": legacy,
            "m2_load_name": "camera_calibration.npz", "m2_cal_workflow": "load", **(extra or {})}
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    for k, v in seed.items():
        at.session_state[k] = v
    return at.run()


def upload(at, data, name="IMG_0001.jpg"):
    photo_uploader(at).set_value((name, data, "image/jpeg"))
    return at.run()


def click(at, key):
    at.button(key=key).click()
    return at.run()


def tick_photo_box(at):
    box = [c for c in at.checkbox if "used for the original calibration" in c.label][0]
    box.check()
    return at.run()


class DetectResolutionTests(unittest.TestCase):
    def test_portrait_landscape_and_png(self):
        self.assertEqual(vc.detect_photo_resolution(image_bytes(600, 800)), (600, 800))
        self.assertEqual(vc.detect_photo_resolution(image_bytes(800, 600)), (800, 600))
        self.assertEqual(vc.detect_photo_resolution(image_bytes(700, 500, "PNG")), (700, 500))

    def test_exif_rotation_swaps_the_reported_size(self):
        for orientation in (5, 6, 7, 8):
            self.assertEqual(vc.detect_photo_resolution(image_bytes(800, 600, orientation=orientation)), (600, 800))
        for orientation in (1, 2, 3, 4):
            self.assertEqual(vc.detect_photo_resolution(image_bytes(800, 600, orientation=orientation)), (800, 600))

    def test_matches_the_calibration_pipeline(self):
        data = image_bytes(800, 600, orientation=6)
        _, _, size, _ = vc.uc.inspect_photo(data, min_side=1)
        self.assertEqual(vc.detect_photo_resolution(data), tuple(size))

    def test_invalid_images_are_rejected(self):
        good = image_bytes(600, 800)
        for bad in (b"", b"not an image", good[: len(good) // 2], image_bytes(600, 800, "GIF")):
            with self.assertRaises(vc.CalibrationError):
                vc.detect_photo_resolution(bad)

    def test_oversized_file_and_image_are_rejected(self):
        with self.assertRaises(vc.CalibrationError):
            vc.detect_photo_resolution(b"\xff\xd8" + b"0" * (vc.uc.MAX_FILE_BYTES + 1))
        huge = image_bytes(8000, 7000, "PNG")                      # 56 MP, compresses to almost nothing
        self.assertLess(len(huge), vc.uc.MAX_FILE_BYTES)
        with self.assertRaises(vc.CalibrationError):
            vc.detect_photo_resolution(huge)

    def test_with_image_size_source(self):
        legacy = cf.parse_calibration(NPZ.read_bytes())
        self.assertEqual(cf.with_image_size(legacy, 4284, 5712, "detected_from_photo")["size_source"], "detected_from_photo")
        with self.assertRaises(cf.CalibrationFileError):
            cf.with_image_size(legacy, 4284, 5712, "guess")


class LegacyPhotoPageTests(unittest.TestCase):
    def test_legacy_file_shows_photo_workflow_and_collapsed_manual_expander(self):
        at = legacy_page()
        self.assertIsNotNone(photo_uploader(at))
        warnings = [w.value for w in at.warning if w.value.startswith("Missing image resolution.")]
        self.assertEqual(warnings, ["Missing image resolution. Choose one option below."])
        self.assertFalse(any("Original Calibration Photograph" in m.value or "resolution required" in m.value for m in at.markdown))
        self.assertFalse(any("detect its image resolution automatically" in c.value for c in at.caption))
        manual = [e for e in at.expander if e.label == "Show manual entry"]
        self.assertEqual(len(manual), 1)
        self.assertIn("Show manual entry", [e.label for e in at.expander])
        self.assertNotIn("Activate Calibration", [b.label for b in at.button])
        self.assertEqual([t for t in ("m2_photo_apply",) if t in [b.key for b in at.button]], [])   # nothing detected yet

    def test_detection_confirmation_and_separate_activation(self):
        at = upload(legacy_page(), image_bytes(4284, 5712, orientation=None), "IMG_0001.jpg")
        self.assertEqual(at.session_state["m2_legacy_photo"], {"name": "IMG_0001.jpg", "width": 4284, "height": 5712})
        self.assertTrue(any("Detected: **4284 × 5712 px**" in m.value for m in at.markdown))
        self.assertTrue(any("IMG_0001.jpg" in m.value for m in at.markdown))
        # without the confirmation box nothing is applied
        at = click(at, "m2_photo_apply")
        self.assertTrue(at.session_state["m2_load_pending"]["needs_image_size"])
        self.assertTrue(any("confirmation" in e.value for e in at.error))
        at = click(tick_photo_box(at), "m2_photo_apply")
        pending = at.session_state["m2_load_pending"]
        self.assertEqual((pending["image_width"], pending["image_height"], pending["size_source"]), (4284, 5712, "detected_from_photo"))
        self.assertIsNone(at.session_state["m2_visitor_calibration"])             # not activated automatically
        self.assertIn("Activate Calibration", [b.label for b in at.button])
        at = click(at, "m2_activate_loaded")
        active = at.session_state["m2_visitor_calibration"]
        self.assertEqual((active["image_width"], active["image_height"]), (4284, 5712))

    def test_exif_rotated_upload_reports_oriented_size(self):
        at = upload(legacy_page(), image_bytes(800, 600, orientation=6))
        self.assertEqual((at.session_state["m2_legacy_photo"]["width"], at.session_state["m2_legacy_photo"]["height"]), (600, 800))

    def test_invalid_photo_shows_error_without_crashing(self):
        at = upload(legacy_page(), b"definitely not an image", "bad.jpg")
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        self.assertTrue(any("bad.jpg" in e.value for e in at.error))
        self.assertNotIn("m2_photo_apply", [b.key for b in at.button])

    def test_detection_persists_across_reruns(self):
        at = upload(legacy_page(), image_bytes(640, 480))
        at.run(); at.run()
        self.assertEqual(at.session_state["m2_legacy_photo"]["width"], 640)
        self.assertTrue(any("Detected: **640 × 480 px**" in m.value for m in at.markdown))
        self.assertEqual(len([u for u in at.get("file_uploader") if u.label == PHOTO_LABEL]), 1)

    def test_replacing_or_removing_photo_invalidates_everything(self):
        at = tick_photo_box(upload(legacy_page(), image_bytes(4284, 5712)))
        at = upload(at, image_bytes(4032, 3024), "other.jpg")                         # replace
        self.assertTrue(at.session_state["m2_load_pending"]["needs_image_size"])
        self.assertEqual(at.session_state["m2_legacy_photo"]["width"], 4032)
        self.assertFalse([c for c in at.checkbox if "used for the original calibration" in c.label and c.value])   # confirmation reset
        at = tick_photo_box(at)
        photo_uploader(at).clear()                                                  # remove
        at.run()
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        self.assertTrue(at.session_state["m2_load_pending"]["needs_image_size"])
        self.assertNotIn("Activate Calibration", [b.label for b in at.button])

    def test_photo_workflow_is_hidden_once_the_resolution_is_confirmed(self):
        at = click(tick_photo_box(upload(legacy_page(), image_bytes(4284, 5712))), "m2_photo_apply")
        self.assertIsNone(photo_uploader(at))
        self.assertFalse(any("Original Calibration Photograph" in m.value for m in at.markdown))
        self.assertNotIn("Show manual entry", [e.label for e in at.expander])   # whole section is gone
        self.assertFalse([w for w in at.warning if "Missing image resolution" in w.value])
        self.assertEqual(len([n for n in at.number_input if "Calibration image" in n.label]), 0)
        self.assertEqual(len([m for m in at.markdown if "**4284 × 5712 px** |" in m.value]), 0)   # validated is not activated
        at = click(at, "m2_activate_loaded")
        self.assertEqual(len([m for m in at.markdown if "**4284 × 5712 px** |" in m.value]), 1)   # one summary line, not repeated

    def test_manual_entry_fallback_still_works(self):
        at = legacy_page()
        width, height = [n for n in at.number_input if "Calibration image" in n.label]
        width.set_value(4284); height.set_value(5712)
        at.run()
        at = click(at, "m2_legacy_apply")
        self.assertTrue(at.session_state["m2_load_pending"]["needs_image_size"])    # needs its confirmation
        at.checkbox[0].check(); at.run()
        at = click(at, "m2_legacy_apply")
        pending = at.session_state["m2_load_pending"]
        self.assertEqual((pending["image_width"], pending["image_height"], pending["size_source"]), (4284, 5712, "confirmed_by_visitor"))
        self.assertIsNone(at.session_state["m2_visitor_calibration"])

    def test_confirmed_resolution_cannot_be_changed_without_removing_the_file(self):
        at = click(tick_photo_box(upload(legacy_page(), image_bytes(4284, 5712))), "m2_photo_apply")
        at = click(at, "m2_activate_loaded")
        self.assertIsNone(photo_uploader(at))
        self.assertEqual(len([n for n in at.number_input if "Calibration image" in n.label]), 0)
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])

    def test_new_format_npz_does_not_ask_for_a_photo(self):
        legacy = cf.parse_calibration(NPZ.read_bytes())
        sized = cf.parse_calibration(cf.export_calibration(cf.with_image_size(legacy, 4284, 5712)))
        at = AppTest.from_file(str(PAGE), default_timeout=90)
        for k, v in {"m2_page": "visitor", "m2_load_pending": sized, "m2_load_name": "mine.npz", "m2_cal_workflow": "load"}.items():
            at.session_state[k] = v
        at.run()
        self.assertIsNone(photo_uploader(at))
        self.assertNotIn("Show manual entry", [e.label for e in at.expander])
        self.assertFalse([w for w in at.warning if "Missing image resolution" in w.value])
        self.assertIn("Activate Calibration", [b.label for b in at.button])

    def test_removing_loaded_file_clears_photo_state(self):
        at = upload(legacy_page(), image_bytes(600, 800))
        self.assertIn('m2_clear_loaded', [b.key for b in at.button])
        at = click(at, "m2_clear_loaded")
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        self.assertIsNone(at.session_state["m2_load_base"])
        self.assertIsNone(photo_uploader(at))

    def test_never_falls_back_to_saved_experiment_calibration(self):
        at = upload(legacy_page(), image_bytes(600, 800))
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        source = PAGE.read_text()
        self.assertNotIn("experiment_camera_matrix", source.split("def legacy_resolution")[1].split("def load_workflow")[0])

    def test_original_experiment_files_unchanged(self):
        legacy_page()
        self.assertEqual(sha(NPZ), NPZ_SHA256)
        self.assertEqual(sha(CSV), CSV_SHA256)



class LoadedCalibrationCompactUiTests(unittest.TestCase):
    """Load Saved Calibration: one compact status, key numbers visible, technical detail in one collapsed expander."""

    def activated(self):
        at = click(tick_photo_box(upload(legacy_page(), image_bytes(4284, 5712))), "m2_photo_apply")
        return click(at, "m2_activate_loaded")

    def test_ready_state_hides_summary_and_details_until_activation(self):
        at = click(tick_photo_box(upload(legacy_page(), image_bytes(4284, 5712))), "m2_photo_apply")
        self.assertIn("Activate Calibration", [b.label for b in at.button])
        self.assertFalse(any("ready for activation" in m.value or "Active for Measurement" in m.value for m in at.markdown))
        self.assertFalse(any("RMS" in m.value and "px" in m.value for m in at.markdown))
        self.assertEqual([e for e in at.expander if e.label == "Calibration Details & Important Notes"], [])
        self.assertEqual(len(at.table), 0)
        self.assertEqual(len(at.metric), 0)
        at = click(at, "m2_activate_loaded")
        self.assertTrue(any("**4284 × 5712 px** | RMS " in m.value for m in at.markdown))
        details = [e for e in at.expander if e.label == "Calibration Details & Important Notes"]
        self.assertEqual(len(details), 1)
        self.assertEqual([t.label for t in at.toggle], [])
        self.assertTrue(any("not accuracy or origin" in m.value for m in details[0].markdown))

    def test_active_state_is_compact_and_has_no_disabled_activate_button(self):
        at = self.activated()
        self.assertTrue(any("Active for Measurement" in m.value for m in at.markdown))
        self.assertFalse(any("ready for activation" in m.value for m in at.markdown))
        self.assertNotIn("Activate Calibration", [b.label for b in at.button])
        self.assertIn("Clear Calibration", [b.label for b in at.button])
        self.assertTrue(any("Using active calibration: 4284 × 5712 px" in c.value for c in at.caption))   # Step 2 line
        at = click(at, "m2_clear_loaded")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertIsNone(at.session_state["m2_load_pending"])


def npz_uploader(at):
    found = [u for u in at.get("file_uploader") if u.label == "Upload Calibration File (.npz)"]
    return found[0] if found else None


def load_page(extra=None):
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    for k, v in {"m2_page": "visitor", "m2_cal_workflow": "load", **(extra or {})}.items():
        at.session_state[k] = v
    return at.run()


def upload_npz(at, data, name="camera_calibration.npz"):
    npz_uploader(at).set_value((name, data, "application/octet-stream"))
    return at.run()


def legacy_bytes():
    return NPZ.read_bytes()


def sized_bytes(width=4284, height=5712):
    return cf.export_calibration(cf.with_image_size(cf.parse_calibration(NPZ.read_bytes()), width, height))


def texts(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


class NpzUploadCardTests(unittest.TestCase):
    """The NPZ uploader is replaced by a file summary after a successful upload; the name survives reruns."""

    def test_empty_state_shows_only_the_uploader(self):
        at = load_page()
        self.assertIsNotNone(npz_uploader(at))
        buttons = [b.key for b in at.button]
        self.assertNotIn("m2_load_replace", buttons)
        self.assertNotIn("m2_clear_loaded", buttons)
        self.assertEqual([e.label for e in at.expander if "Calibration Details" in e.label], [])

    def test_successful_upload_hides_uploader_and_shows_file_summary(self):
        at = upload_npz(load_page(), sized_bytes())
        self.assertIsNone(npz_uploader(at))
        self.assertEqual(len(at.get("file_uploader")), 0)                           # no duplicate upload widget
        self.assertIn("camera_calibration.npz", texts(at))
        self.assertIn("File uploaded and validated", texts(at))
        self.assertEqual([b.label for b in at.button if b.key in ("m2_clear_loaded", "m2_load_replace", "m2_load_replace_cancel")],
                         ["Clear Calibration"])
        self.assertFalse({"Replace File", "Cancel"} & {b.label for b in at.button})
        self.assertIsNone(at.session_state["m2_visitor_calibration"])               # loaded, not activated

    def test_filename_persists_across_reruns(self):
        at = upload_npz(load_page(), sized_bytes(), "mine.npz")
        for _ in range(3):
            at.run()
            self.assertIn("mine.npz", texts(at))
            self.assertIsNone(npz_uploader(at))

    def test_remove_file_clears_file_and_resolution_confirmation(self):
        at = upload_npz(load_page(), legacy_bytes())
        at = tick_photo_box(upload(at, image_bytes(4284, 5712)))
        at = click(at, "m2_clear_loaded")
        s = at.session_state
        self.assertIsNone(s["m2_load_pending"]); self.assertIsNone(s["m2_load_name"])
        self.assertIsNone(s["m2_load_base"]); self.assertIsNone(s["m2_legacy_photo"])
        self.assertIsNotNone(npz_uploader(at))
        self.assertIsNone(photo_uploader(at))
        self.assertIsNone(s["m2_cal_notice"])                                       # nothing was active: nothing to announce

    def test_removing_the_active_file_deactivates_it_and_says_so(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])
        at = click(at, "m2_clear_loaded")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertIsNone(at.session_state["m2_cal_notice"])

    def test_removing_a_file_restores_the_empty_uploader_and_a_different_file_can_follow(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_clear_loaded")
        self.assertIsNotNone(npz_uploader(at))
        self.assertNotIn("Clear Calibration", [b.label for b in at.button])         # nothing loaded: no button
        at = upload_npz(at, sized_bytes(4300, 5800), "second.npz")
        self.assertEqual(at.session_state["m2_load_pending"]["image_width"], 4300)

    def test_removing_an_inactive_file_does_not_touch_a_computed_active_calibration(self):
        other = cf.parse_calibration(sized_bytes(4200, 5700))
        at = load_page({"m2_visitor_calibration": other})
        at = click(upload_npz(at, sized_bytes()), "m2_clear_loaded")
        self.assertEqual(at.session_state["m2_visitor_calibration"]["image_width"], 4200)

    def test_replacement_state_key_is_gone(self):
        at = upload_npz(load_page(), sized_bytes())
        self.assertNotIn("m2_load_replacing", at.session_state)
        self.assertNotIn("Replace File", PAGE.read_text())
        self.assertNotIn("m2_load_replacing", PAGE.read_text())

    def test_legacy_upload_then_detection_confirmation_and_activation(self):
        at = upload_npz(load_page(), legacy_bytes())
        self.assertIsNotNone(photo_uploader(at))
        self.assertIn("camera_calibration.npz", texts(at))
        at = click(tick_photo_box(upload(at, image_bytes(4284, 5712), "IMG_7373.jpeg")), "m2_photo_apply")
        at = click(at, "m2_activate_loaded")
        self.assertEqual(at.session_state["m2_visitor_calibration"]["image_width"], 4284)
        self.assertIn("camera_calibration.npz", texts(at))                           # still visible after all those reruns


class CompactSummaryTests(unittest.TestCase):
    def active_page(self):
        return click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded")

    def test_summary_uses_dynamic_values_and_hides_intrinsics_on_main_page(self):
        at = click(upload_npz(load_page(), sized_bytes(4300, 5800)), "m2_activate_loaded")
        model = at.session_state["m2_load_pending"]
        self.assertTrue(any(f"**4300 × 5800 px** | RMS {model['rms_error']:.3f} px" in m.value for m in at.markdown))
        details = [e for e in at.expander if e.label == "Calibration Details & Important Notes"][0]
        inside = {m.value for m in details.markdown}
        main = " ".join([m.value for m in at.markdown if m.value not in inside] + [c.value for c in at.caption])
        for label in ("fx", "fy", "cx", "cy"):
            self.assertNotIn(label + " =", main)
        self.assertIn("fx =", " ".join(inside))                                       # still available on request
        self.assertEqual(len(at.metric), 0)
        self.assertEqual(len(at.table), len(details.table))                          # matrix tables only inside the expander

    def test_active_state_has_one_status_no_disabled_activate_and_a_step2_line(self):
        at = self.active_page()
        self.assertEqual(len([m for m in at.markdown if "Active for Measurement" in m.value]), 1)
        self.assertNotIn("Activate Calibration", [b.label for b in at.button])
        self.assertEqual(len(at.success), 0)
        self.assertTrue(any(c.value == "Using active calibration: 4284 × 5712 px" for c in at.caption))
        step2 = " ".join(c.value for c in at.caption if "Using active" in c.value)
        self.assertNotIn("fx", step2)

    def test_technical_details_are_in_one_collapsed_expander(self):
        at = self.active_page()
        details = [e for e in at.expander if e.label == "Calibration Details & Important Notes"]
        self.assertEqual(len(details), 1)
        inner = " ".join(m.value for m in details[0].markdown) + " ".join(w.value for w in details[0].warning)
        for needle in ("fx =", "fy =", "cx =", "cy =", "Checkerboard", "Resolution:", "same camera, lens, zoom", "older NPZ files",
                       "not accuracy or origin"):
            self.assertIn(needle, inner)
        for gone in ("reprojection error", "Compatibility", "EXIF", "heuristics"):
            self.assertNotIn(gone, inner)
        self.assertLessEqual(len([line for line in inner.split("\n") if line.startswith("- ")]), 3)
        self.assertGreaterEqual(len(details[0].table), 3)                            # K, distortion, metadata

    def test_personal_experiment_sentence_is_absent(self):
        for at in (load_page(), upload_npz(load_page(), sized_bytes()), upload_npz(load_page(), legacy_bytes())):
            everything = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info])
            self.assertNotIn("My own calibration", everything)
            self.assertNotIn("never used for your measurements", everything)

    def test_legacy_upload_shows_one_warning_then_uploader_and_collapsed_manual_expander(self):
        at = upload_npz(load_page(), legacy_bytes())
        self.assertEqual(len([w for w in at.warning if "Missing image resolution" in w.value]), 1)
        self.assertIsNotNone(photo_uploader(at))
        manual = [e for e in at.expander if e.label == "Show manual entry"]
        self.assertEqual(len(manual), 1)
        self.assertFalse(manual[0].proto.expanded)

    def test_reset_all_clears_file_state_and_shows_the_uploader_again(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded")
        reset = [b for b in at.button if b.label == "Reset All"][0]
        reset.click(); at.run()
        s = at.session_state
        self.assertIsNone(s["m2_load_pending"]); self.assertIsNone(s["m2_visitor_calibration"])
        self.assertNotIn("m2_load_replacing", s)

    def test_original_experiment_files_unchanged_by_upload_and_remove(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded")
        click(at, "m2_clear_loaded")
        self.assertEqual(sha(NPZ), NPZ_SHA256)
        self.assertEqual(sha(CSV), CSV_SHA256)


if __name__ == "__main__":
    unittest.main()


class ClearLoadedCalibrationTests(unittest.TestCase):
    """Clear Calibration is the only button on a loaded file and resets the whole Load Saved Calibration workflow."""

    def assert_reset(self, at):
        s = at.session_state
        self.assertIsNotNone(npz_uploader(at))
        self.assertIsNone(photo_uploader(at))
        self.assertNotIn("camera_calibration.npz", texts(at))
        self.assertNotIn("File uploaded and validated", texts(at))
        self.assertEqual([e.label for e in at.expander if "Calibration Details" in e.label], [])
        self.assertNotIn("Clear Calibration", [b.label for b in at.button])
        for key in ("m2_load_pending", "m2_load_name", "m2_load_base", "m2_legacy_photo", "m2_load_error", "m2_legacy_photo_error"):
            self.assertIsNone(s[key], key)
        self.assertFalse(any("Using active calibration" in c.value for c in at.caption))

    def test_single_button_and_never_remove_file(self):
        at = upload_npz(load_page(), legacy_bytes())
        labels = [b.label for b in at.button]
        self.assertEqual(labels.count("Clear Calibration"), 1)
        self.assertNotIn("Remove File", labels)

    def test_missing_resolution(self):
        at = upload_npz(load_page(), legacy_bytes())
        self.assertIn("Clear Calibration", [b.label for b in at.button])
        self.assert_reset(click(at, "m2_clear_loaded"))

    def test_resolution_confirmed_photo_and_checkbox_cleared(self):
        at = tick_photo_box(upload(upload_npz(load_page(), legacy_bytes()), image_bytes(4284, 5712)))
        self.assertIsNotNone(at.session_state["m2_legacy_photo"])
        at = click(at, "m2_clear_loaded")
        self.assert_reset(at)
        at = upload_npz(at, legacy_bytes())                                          # fresh file: no stale photo or tick
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        self.assertFalse([c for c in at.checkbox if c.value])

    def test_manual_resolution_cleared(self):
        at = upload_npz(load_page(), legacy_bytes())
        for n in at.number_input:
            n.set_value(4284 if "width" in n.label else 5712)
        at.run()
        [c for c in at.checkbox if "exact dimensions" in c.label][0].check()
        at.run()
        at = upload_npz(click(at, "m2_clear_loaded"), legacy_bytes())
        self.assertTrue(all(n.value is None for n in at.number_input if "Calibration image" in n.label))
        self.assertFalse([c for c in at.checkbox if c.value])

    def test_ready_for_activation(self):
        at = upload_npz(load_page(), sized_bytes())
        self.assertIn("Activate Calibration", [b.label for b in at.button])
        self.assert_reset(click(at, "m2_clear_loaded"))

    def test_active_is_deactivated_and_step_two_unavailable(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])
        self.assertIn("Clear Calibration", [b.label for b in at.button])
        at = click(at, "m2_clear_loaded")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assert_reset(at)

    def test_new_npz_can_be_uploaded_immediately_after_clearing(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_clear_loaded")
        at = upload_npz(at, sized_bytes(4300, 5800), "second.npz")
        self.assertEqual(at.session_state["m2_load_pending"]["image_width"], 4300)
        self.assertIn("second.npz", texts(at))

    def test_separate_computed_calibration_is_not_cleared(self):
        other = cf.parse_calibration(sized_bytes(4200, 5700))
        at = load_page({"m2_visitor_calibration": other})
        for loaded in (sized_bytes(), legacy_bytes()):
            at = click(upload_npz(at, loaded), "m2_clear_loaded")
            self.assertEqual(at.session_state["m2_visitor_calibration"]["image_width"], 4200)

    def test_reset_all_still_works_after_clearing(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_clear_loaded")
        at = upload_npz(at, sized_bytes())
        at = click(at, "m2_activate_loaded")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])
        at = click(at, "m2_reset_all")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertIsNone(at.session_state["m2_load_pending"])
        self.assertIsNone(at.session_state["m2_cal_workflow"])

    def test_original_files_unchanged(self):
        before = (sha(NPZ), sha(CSV))
        click(upload_npz(load_page(), sized_bytes()), "m2_clear_loaded")
        self.assertEqual(before, (sha(NPZ), sha(CSV)))
        self.assertEqual(before, (NPZ_SHA256, CSV_SHA256))


def flatten(node):
    """Elements of a page in display order (containers expanded)."""
    out = []
    for child in getattr(node, "children", {}).values():
        out.append(child)
        out.extend(flatten(child))
    return out


def step1_controls(at):
    """Widgets between the Step 1 heading and the Step 2 heading, in display order."""
    elements = flatten(at.main)
    start = next(i for i, e in enumerate(elements) if e.type == "subheader" and e.value.startswith("Step 1"))
    stop = next(i for i, e in enumerate(elements) if e.type == "subheader" and e.value.startswith("Step 2"))
    return [e for e in elements[start:stop] if e.type in ("button", "file_uploader", "checkbox", "number_input", "toggle")]


def confirmed_page():
    return click(tick_photo_box(upload(upload_npz(load_page(), legacy_bytes()), image_bytes(4284, 5712), "IMG_7373.jpeg")),
                 "m2_photo_apply")


def summary_shown(at):
    return any("IMG_7373.jpeg" in m.value and "Resolution: 4284 × 5712 px" in m.value for m in at.markdown)


class ClearPlacementAndPhotoSummaryTests(unittest.TestCase):
    def assert_clear_last(self, at):
        controls = step1_controls(at)
        self.assertEqual([c.key for c in controls if c.type == "button" and c.label == "Clear Calibration"], ["m2_clear_loaded"])
        self.assertEqual(controls[-1].key, "m2_clear_loaded")

    def test_clear_is_last_control_in_every_loaded_state(self):
        missing = upload_npz(load_page(), legacy_bytes())
        self.assert_clear_last(missing)
        confirmed = confirmed_page()
        self.assert_clear_last(confirmed)
        self.assert_clear_last(click(confirmed, "m2_activate_loaded"))
        self.assert_clear_last(upload_npz(load_page(), sized_bytes()))
        self.assert_clear_last(click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded"))

    def test_no_clear_button_without_a_file(self):
        self.assertEqual([b for b in step1_controls(load_page()) if b.type == "button" and b.label == "Clear Calibration"], [])

    def test_photo_summary_and_hidden_confirmation_controls(self):
        at = confirmed_page()
        self.assertTrue(summary_shown(at))
        self.assertIn("m2_photo_remove", [b.key for b in at.button])
        self.assertNotIn("m2_photo_apply", [b.key for b in at.button])
        self.assertIsNone(photo_uploader(at))
        self.assertFalse([c for c in at.checkbox if "used for the original calibration" in c.label])
        self.assertFalse([w for w in at.warning if "Missing image resolution" in w.value])
        self.assertEqual([e for e in at.expander if e.label == "Show manual entry"], [])
        self.assertIsNotNone(at.session_state["m2_legacy_photo"])

    def test_summary_uses_actual_file_values(self):
        at = click(tick_photo_box(upload(upload_npz(load_page(), legacy_bytes()), image_bytes(4032, 3024), "other.png")), "m2_photo_apply")
        self.assertTrue(any("other.png" in m.value and "Resolution: 4032 × 3024 px" in m.value for m in at.markdown))

    def test_photo_summary_remains_after_activation_and_reruns(self):
        at = click(confirmed_page(), "m2_activate_loaded")
        at.run(); at.run()
        self.assertTrue(summary_shown(at))
        self.assertIn("m2_photo_remove", [b.key for b in at.button])

    def test_remove_image_restores_missing_resolution_workflow(self):
        at = click(confirmed_page(), "m2_photo_remove")
        s = at.session_state
        self.assertIsNone(s["m2_legacy_photo"])
        self.assertTrue(s["m2_load_pending"]["needs_image_size"])
        self.assertEqual(s["m2_load_name"], "camera_calibration.npz")                  # NPZ stays loaded
        self.assertFalse(summary_shown(at))
        self.assertNotIn("m2_photo_remove", [b.key for b in at.button])
        self.assertEqual(len([w for w in at.warning if "Missing image resolution" in w.value]), 1)
        self.assertIsNotNone(photo_uploader(at))
        self.assertEqual(len([e for e in at.expander if e.label == "Show manual entry"]), 1)
        self.assertNotIn("m2_photo_apply", [b.key for b in at.button])
        self.assertFalse([c for c in at.checkbox if c.value])                          # confirmation cleared
        self.assert_clear_last(at)

    def test_remove_image_deactivates_an_active_calibration(self):
        at = click(confirmed_page(), "m2_activate_loaded")
        self.assertIsNotNone(at.session_state["m2_visitor_calibration"])
        at = click(at, "m2_photo_remove")
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        self.assertNotIn("Activate Calibration", [b.label for b in at.button])
        self.assertFalse(any("Using active calibration" in c.value for c in at.caption))
        self.assertTrue(any("First load a saved calibration" in i.value for i in at.info))   # measurement disabled again

    def test_remove_image_leaves_a_separate_active_calibration_alone(self):
        other = cf.parse_calibration(sized_bytes(4200, 5700))
        at = load_page({"m2_visitor_calibration": other})
        at = click(tick_photo_box(upload(upload_npz(at, legacy_bytes()), image_bytes(4284, 5712))), "m2_photo_apply")
        at = click(at, "m2_photo_remove")
        self.assertEqual(at.session_state["m2_visitor_calibration"]["image_width"], 4200)

    def test_the_workflow_can_be_completed_again_after_remove_image(self):
        at = click(confirmed_page(), "m2_photo_remove")
        at = click(tick_photo_box(upload(at, image_bytes(4032, 3024), "second.jpg")), "m2_photo_apply")
        self.assertTrue(any("second.jpg" in m.value and "Resolution: 4032 × 3024 px" in m.value for m in at.markdown))
        at = click(at, "m2_activate_loaded")
        self.assertEqual(at.session_state["m2_visitor_calibration"]["image_width"], 4032)

    def test_no_stale_state_after_reruns_following_remove_image(self):
        at = click(confirmed_page(), "m2_photo_remove")
        at.run(); at.run()
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        self.assertTrue(at.session_state["m2_load_pending"]["needs_image_size"])
        self.assertFalse(summary_shown(at))

    def test_embedded_resolution_file_shows_no_photo_controls(self):
        for at in (upload_npz(load_page(), sized_bytes()), click(upload_npz(load_page(), sized_bytes()), "m2_activate_loaded")):
            self.assertIsNone(photo_uploader(at))
            self.assertNotIn("m2_photo_remove", [b.key for b in at.button])
            self.assertFalse(any("Upload Calibration Image" in m.value for m in at.markdown))

    def test_manual_resolution_shows_no_photo_summary(self):
        at = upload_npz(load_page(), legacy_bytes())
        for n in at.number_input:
            n.set_value(4284 if "width" in n.label else 5712)
        at.run()
        [c for c in at.checkbox if "exact dimensions" in c.label][0].check()
        at.run()
        at = click(at, "m2_legacy_apply")
        self.assertNotIn("m2_photo_remove", [b.key for b in at.button])
        self.assertEqual(at.session_state["m2_load_pending"]["size_source"], "confirmed_by_visitor")
        self.assert_clear_last(at)

    def test_clear_calibration_and_reset_all_clear_the_summary(self):
        at = click(confirmed_page(), "m2_clear_loaded")
        self.assertFalse(summary_shown(at))
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        at = click(click(confirmed_page(), "m2_activate_loaded"), "m2_reset_all")
        self.assertIsNone(at.session_state["m2_legacy_photo"])
        self.assertIsNone(at.session_state["m2_load_pending"])
        self.assertFalse(summary_shown(at))

    def test_original_files_unchanged(self):
        click(confirmed_page(), "m2_photo_remove")
        self.assertEqual((sha(NPZ), sha(CSV)), (NPZ_SHA256, CSV_SHA256))


class VisibilityRulesTests(unittest.TestCase):
    """NO FILE / MISSING RESOLUTION / READY / ACTIVE / CLEARED: what the Load Saved Calibration workflow shows."""
    DETAILS = "Calibration Details & Important Notes"

    def shown(self, at):
        return (any("RMS" in m.value for m in at.markdown), any(e.label == self.DETAILS for e in at.expander),
                any("Active for Measurement" in m.value for m in at.markdown))

    def test_no_file_shows_only_the_uploader(self):
        at = load_page()
        self.assertEqual(len(at.get("file_uploader")), 1)
        self.assertEqual(self.shown(at), (False, False, False))

    def test_missing_resolution_shows_workflow_without_results(self):
        at = upload_npz(load_page(), legacy_bytes())
        self.assertEqual(self.shown(at), (False, False, False))
        self.assertIn("Clear Calibration", [b.label for b in at.button])
        self.assertIn("Upload Calibration Image", [m.value.strip("*") for m in at.markdown])
        self.assertIn("Enter Resolution Manually", [m.value.strip("*") for m in at.markdown])
        self.assertIn("Detect resolution automatically", [c.value for c in at.caption])
        self.assertIn("Specify width and height", [c.value for c in at.caption])
        self.assertEqual(len([u for u in at.get("file_uploader") if u.label == PHOTO_LABEL]), 1)   # no duplicate widget
        self.assertEqual(len(at.get("file_uploader")), 1)
        self.assertFalse([e for e in at.expander if e.label == "Show manual entry"][0].proto.expanded)

    def test_new_format_file_skips_missing_resolution_controls(self):
        at = upload_npz(load_page(), sized_bytes())
        self.assertEqual(len(at.get("file_uploader")), 0)
        self.assertFalse([w for w in at.warning if "Missing image resolution" in w.value])
        self.assertEqual([e for e in at.expander if e.label == "Show manual entry"], [])

    def test_ready_then_active_then_cleared(self):
        at = upload_npz(load_page(), sized_bytes())
        self.assertEqual(self.shown(at), (False, False, False))
        self.assertIn("Activate Calibration", [b.label for b in at.button])
        at = click(at, "m2_activate_loaded")
        self.assertEqual(self.shown(at), (True, True, True))
        at = click(at, "m2_clear_loaded")
        self.assertEqual(self.shown(at), (False, False, False))
        self.assertEqual(len(at.get("file_uploader")), 1)
        self.assertNotIn("Clear Calibration", [b.label for b in at.button])
        self.assertIsNone(at.session_state["m2_load_pending"]); self.assertIsNone(at.session_state["m2_load_name"])
        self.assertIsNone(at.session_state["m2_visitor_calibration"])
        at.run()                                                                       # nothing stale after another rerun
        self.assertEqual(self.shown(at), (False, False, False))
        self.assertFalse(any("camera" in m.value and ".npz" in m.value for m in at.markdown))

    def test_clear_while_validated_not_active_also_resets(self):
        at = click(upload_npz(load_page(), sized_bytes()), "m2_clear_loaded")
        self.assertEqual(len(at.get("file_uploader")), 1)
        self.assertEqual(self.shown(at), (False, False, False))
