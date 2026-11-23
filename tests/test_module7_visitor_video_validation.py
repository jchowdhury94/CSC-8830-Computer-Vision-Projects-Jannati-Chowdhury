"""Phase 2B Stage B: four-video inspection/validation (room_measurement/visitor_video_validation.py). Small synthetic videos only."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "Modules" / "Module_7"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(ROOT / "tests"))

from room_measurement import visitor_calibration as V  # noqa: E402
from room_measurement import visitor_video_validation as W  # noqa: E402
from room_measurement.visitor_calibration import Severity  # noqa: E402
from test_module7_visitor_calibration import model, with_modes  # noqa: E402

SZ = (96, 160)           # portrait (w, h)


def make_video(path, size=SZ, frames=24, fps=12.0):
    w, h = size
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened(), "OpenCV cannot write mp4v here"
    for i in range(frames):
        f = np.full((h, w, 3), (i * 9) % 255, np.uint8); cv2.circle(f, (w // 2, h // 2), 10 + i, (255, 255, 255), 2); wr.write(f)
    wr.release()
    return path


def codes(r, sev=Severity.ERROR):
    return {i.code for i in r.issues if i.severity is sev}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vv_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def four(self, sizes=(SZ,) * 4, **kw):
        return [str(make_video(self.tmp / f"v{i}.mp4", s, **kw)) for i, s in enumerate(sizes)]

    def cal(self, size=SZ, modes=("video", "video")):
        m = with_modes(*modes, size=size)
        m["K1"][:2, :] *= 0 + 1; return V.adapt_visitor_calibration(self.scaled(m, size), 100.0)

    @staticmethod
    def scaled(m, size):
        f = size[1] / 3840
        for k in ("K1", "K2"):
            m[k] = np.array([[3178 * f, 0, size[0] / 2], [0, 3178 * f, size[1] / 2], [0, 0, 1]])
        return m


class Valid(Base):
    def test_four_valid_videos(self):
        r = W.validate_visitor_videos(*self.four(), calibration=self.cal())
        self.assertEqual(r.status, Severity.PASS, [i.to_dict() for i in r.issues if i.severity != Severity.PASS]); self.assertTrue(r.can_proceed)
        self.assertEqual(list(r.videos), list(W.VIDEO_LABELS))
        v = r.videos["left_video_1"]
        self.assertEqual((v.width, v.height, v.orientation), (96, 160, "portrait")); self.assertEqual(v.frame_count, 24)
        self.assertAlmostEqual(v.fps, 12.0); self.assertAlmostEqual(v.duration_s, 2.0); self.assertTrue(v.frame_decoded); self.assertGreater(v.size_bytes, 0)
        self.assertEqual(v.decoder, "opencv"); self.assertIsNotNone(v.codec)
        self.assertEqual(r.calibration_compatibility.status.value, "COMPATIBLE_METADATA")
        self.assertIn("does not establish", r.note)

    def test_no_filename_assumptions_and_sources_untouched(self):
        paths = self.four()
        before = [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths]
        W.validate_visitor_videos(*paths)
        self.assertEqual(before, [(Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in paths])
        self.assertEqual(len(list(self.tmp.iterdir())), 4)

    def test_wizard_style_calibration_without_provenance_warns(self):
        r = W.validate_visitor_videos(*self.four(), calibration=self.cal(modes=(None, None)))
        self.assertEqual(r.status, Severity.WARNING); self.assertTrue(r.can_proceed)
        self.assertIn("MISSING_PROVENANCE", codes(r, Severity.WARNING))

    def test_no_calibration_warns(self):
        r = W.validate_visitor_videos(*self.four())
        self.assertIn("NO_CALIBRATION", codes(r, Severity.WARNING)); self.assertTrue(r.can_proceed)

    def test_to_dict_structure(self):
        d = W.validate_visitor_videos(*self.four(), calibration=self.cal()).to_dict()
        for k in ("status", "can_proceed", "videos", "set_checks", "calibration_compatibility", "errors", "warnings", "note"):
            self.assertIn(k, d)
        self.assertEqual(set(d["videos"]), set(W.VIDEO_LABELS))
        self.assertTrue(all({"code", "message", "severity"} <= set(c) for c in d["videos"]["right_video_2"]["checks"]))


class PerFileFailures(Base):
    def run_with(self, label_index, path):
        p = self.four(); p[label_index] = str(path)
        return W.validate_visitor_videos(*p)

    def test_missing_video(self):
        r = self.run_with(2, self.tmp / "nope.mp4")
        self.assertIn("FILE_NOT_FOUND", codes(r)); self.assertFalse(r.can_proceed); self.assertEqual(r.status, Severity.ERROR)
        self.assertEqual(r.videos["left_video_2"].status, Severity.ERROR); self.assertEqual(r.videos["left_video_1"].status, Severity.PASS)
        self.assertIn("left video 2", r.videos["left_video_2"].errors[0].message)
        self.assertIn("CONSISTENCY_NOT_CHECKED", codes(r, Severity.WARNING))

    def test_not_provided_and_directory(self):
        p = self.four(); p[0] = None; p[1] = str(self.tmp)
        r = W.validate_visitor_videos(*p)
        self.assertIn("FILE_NOT_PROVIDED", codes(r)); self.assertIn("NOT_A_FILE", codes(r))

    def test_empty_video(self):
        e = self.tmp / "empty.mp4"; e.write_bytes(b"")
        r = self.run_with(1, e); self.assertIn("EMPTY_FILE", codes(r)); self.assertFalse(r.can_proceed)

    def test_corrupt_video(self):
        c = self.tmp / "corrupt.mp4"; c.write_bytes(np.random.default_rng(0).bytes(5000))
        r = self.run_with(3, c)
        self.assertTrue({"OPEN_FAILED", "DECODE_FAILED"} & codes(r)); self.assertFalse(r.can_proceed)
        self.assertTrue(c.exists())          # never deleted

    def test_truncated_video(self):
        good = make_video(self.tmp / "g.mp4"); t = self.tmp / "trunc.mp4"; t.write_bytes(good.read_bytes()[:400])
        self.assertFalse(self.run_with(0, t).can_proceed)

    def test_opens_but_no_frame_decodes(self):
        class Cap:
            def __init__(self, *_): pass
            def isOpened(self): return True
            def get(self, prop): return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: 100.0, cv2.CAP_PROP_FRAME_WIDTH: 96.0, cv2.CAP_PROP_FRAME_HEIGHT: 160.0}.get(prop, 0.0)
            def read(self): return False, None
            def release(self): pass
        rep = self.four()
        with mock.patch.object(W.cv2, "VideoCapture", Cap):
            rep0 = W.inspect_video(rep[0], "left_video_1")
        self.assertTrue(rep0.opened); self.assertFalse(rep0.frame_decoded)
        self.assertEqual([c.code for c in rep0.errors], ["DECODE_FAILED"]); self.assertIn("H.264", rep0.errors[0].message)

    def test_ffmpeg_fallback_is_diagnostic_only(self):
        class Dead:
            def __init__(self, *_): pass
            def isOpened(self): return False
            def get(self, prop): return 0.0
            def release(self): pass
        p = self.four()[0]
        with mock.patch.object(W.cv2, "VideoCapture", Dead), mock.patch.object(W, "_ffmpeg_diagnostic", return_value=dict(meta=dict(codec="hevc"), decoded=True)):
            rep = W.inspect_video(p, "left_video_1")
        self.assertEqual(rep.status, Severity.ERROR); self.assertEqual(rep.codec, "hevc"); self.assertIn("diagnostic", rep.decoder)
        self.assertIn("ffmpeg can read it", rep.errors[0].message)

    def test_invalid_frame_metadata(self):
        class Cap:
            fps, n = 0.0, -1.0
            def __init__(self, *_): pass
            def isOpened(self): return True
            def get(self, prop): return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.n}.get(prop, 0.0)
            def read(self): return True, np.zeros((160, 96, 3), np.uint8)
            def release(self): pass
        p = self.four()[0]
        with mock.patch.object(W.cv2, "VideoCapture", Cap):
            rep = W.inspect_video(p, "left_video_1")
            self.assertEqual({c.code for c in rep.errors}, {"INVALID_FRAME_RATE", "INVALID_FRAME_COUNT"})
            Cap.fps, Cap.n = float("nan"), float("inf")
            self.assertEqual({c.code for c in W.inspect_video(p, "x").errors}, {"INVALID_FRAME_RATE", "INVALID_FRAME_COUNT"})

    def test_duration_limits(self):
        p = self.four()                                                   # 2 s clips
        self.assertIn("VIDEO_TOO_SHORT", codes(W.validate_visitor_videos(*p, limits=W.VideoLimits(min_duration_s=5))))
        self.assertIn("VIDEO_TOO_LONG", codes(W.validate_visitor_videos(*p, limits=W.VideoLimits(max_duration_s=1.5))))
        self.assertIn("TOO_FEW_FRAMES", codes(W.validate_visitor_videos(*p, limits=W.VideoLimits(min_frames=100))))

    def test_file_size_limit(self):
        r = W.validate_visitor_videos(*self.four(), limits=W.VideoLimits(max_file_bytes=100))
        self.assertIn("FILE_TOO_LARGE", codes(r)); self.assertFalse(r.can_proceed)
        self.assertIn("MB", r.videos["left_video_1"].errors[0].message)

    def test_dimension_limits(self):
        self.assertIn("INVALID_DIMENSIONS", codes(W.validate_visitor_videos(*self.four(), limits=W.VideoLimits(min_side_px=500))))

    def test_duplicate_file(self):
        p = self.four(); p[1] = p[0]
        self.assertIn("DUPLICATE_FILE", codes(W.validate_visitor_videos(*p)))


class CrossVideo(Base):
    def test_mismatched_resolution(self):
        r = W.validate_visitor_videos(*self.four((SZ, SZ, (120, 200), SZ)))
        self.assertIn("VIDEO_RESOLUTION_MISMATCH", codes(r)); self.assertFalse(r.can_proceed)
        self.assertIn("120 x 200", r.set_checks[0].message)

    def test_mismatched_orientation(self):
        r = W.validate_visitor_videos(*self.four((SZ, SZ, SZ, (160, 96))))
        self.assertIn("VIDEO_ORIENTATION_MISMATCH", codes(r)); self.assertNotIn("VIDEO_RESOLUTION_MISMATCH", codes(r))

    def test_frame_rate_difference_warns(self):
        p = [str(make_video(self.tmp / f"f{i}.mp4", fps=f, frames=40)) for i, f in enumerate((12, 12, 12, 30))]
        r = W.validate_visitor_videos(*p); self.assertIn("FRAME_RATE_DIFFERS", codes(r, Severity.WARNING)); self.assertTrue(r.can_proceed)


class CalibrationVsVideo(Base):
    def test_resolution_mismatch(self):
        r = W.validate_visitor_videos(*self.four(), calibration=self.cal(size=(144, 240)))
        self.assertIn("INCOMPATIBLE_RESOLUTION", codes(r)); self.assertFalse(r.can_proceed)
        self.assertEqual(r.calibration_compatibility.status.value, "INCOMPATIBLE_RESOLUTION")

    def test_orientation_mismatch(self):
        r = W.validate_visitor_videos(*self.four(), calibration=self.cal(size=(160, 96)))
        self.assertIn("INCOMPATIBLE_ORIENTATION", codes(r)); self.assertFalse(r.can_proceed)

    def test_photo_calibration_is_warning_not_pass(self):
        r = W.validate_visitor_videos(*self.four(), calibration=self.cal(modes=("photo", "photo")))
        self.assertEqual(r.status, Severity.WARNING); self.assertIn("UNVERIFIED_CAPTURE_MODE", codes(r, Severity.WARNING))

    def test_invalid_calibration_blocks(self):
        bad = V.adapt_visitor_calibration(model(), -1)
        r = W.validate_visitor_videos(*self.four(), calibration=bad)
        self.assertIn("CALIBRATION_INVALID", codes(r)); self.assertFalse(r.can_proceed)


class TempWorkspace(unittest.TestCase):
    def test_cleans_only_own_dir(self):
        outside = Path(tempfile.mkdtemp(prefix="outside_")); (outside / "keep.txt").write_text("x")
        try:
            with W.TemporaryVideoWorkspace() as ws:
                a = ws.write(b"abc", ".MP4"); b = ws.write(b"def", "../../evil")
                self.assertEqual(a.parent, ws.path); self.assertEqual(b.parent, ws.path); self.assertNotEqual(a, b)
                self.assertTrue(a.name.endswith(".mp4")); d = ws.path
            self.assertFalse(d.exists()); self.assertTrue((outside / "keep.txt").exists())
            with self.assertRaises(RuntimeError):
                ws.write(b"x")
        finally:
            shutil.rmtree(outside, True)

    def test_cleanup_after_exception(self):
        try:
            with W.TemporaryVideoWorkspace() as ws:
                d = ws.path; ws.write(b"abc"); raise KeyError("boom")
        except KeyError:
            pass
        self.assertFalse(d.exists())


if __name__ == "__main__":
    unittest.main()
