"""Module 6 visitor tracking backend tests (synthetic videos; no Streamlit)."""
import csv
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Modules.Module_6.Part_A import visitor_tracking as vt  # noqa: E402

PART_A = ROOT / "Modules" / "Module_6" / "Part_A"


def make_video(directory, name="clip.mp4", width=320, height=240, fps=30.0, frames=20, speed=3, vanish_at=None):
    """Textured square moving right over a noisy static background."""
    rng = np.random.default_rng(1)
    background = rng.integers(60, 120, (height, width), dtype=np.uint8)
    patch = rng.integers(0, 255, (60, 60), dtype=np.uint8)
    path = str(Path(directory) / name)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(frames):
        gray = background.copy()
        if vanish_at is None or i < vanish_at:
            x = 30 + speed * i
            gray[90:150, x:x + 60] = patch
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def fake_info(width=640, height=480, fps=30.0, frames=300):
    return vt.VideoInfo(width, height, fps, False, frames, frames / fps)


class ProbeAndLimitTests(unittest.TestCase):
    def test_probe_reports_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            info = vt.probe_video(make_video(d, frames=15))
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.fps, 30.0, places=1)
        self.assertEqual(info.frame_count, 15)
        self.assertAlmostEqual(info.duration, 0.5, places=1)

    def test_supported_formats_and_filename_never_a_path(self):
        for name in ("a.mp4", "b.MOV", "c.m4v", "d.avi", "e.mkv", "f.webm"):
            self.assertIsNotNone(vt.safe_extension(name))
        self.assertIsNone(vt.safe_extension("movie.exe"))
        self.assertEqual(vt.safe_extension("../../etc/x.mp4"), ".mp4")
        with self.assertRaises(vt.VideoError):
            vt.check_upload(1000, "clip.gif")

    def test_file_size_limit(self):
        vt.check_upload(vt.MAX_FILE_BYTES, "a.mp4")
        with self.assertRaisesRegex(vt.VideoError, "50 MB"):
            vt.check_upload(vt.MAX_FILE_BYTES + 1, "a.mp4")
        with self.assertRaises(vt.VideoError):
            vt.check_upload(0, "a.mp4")

    def test_duration_limit_truncates_with_disclosure(self):
        plan = vt.plan_processing(fake_info(frames=900, fps=30.0))  # 30 s
        self.assertTrue(plan.truncated)
        self.assertEqual(plan.source_frames_used, 600)
        self.assertEqual(plan.stride, 1)
        self.assertTrue(any("only the first segment" in m for m in plan.messages()))

    def test_within_limits_no_messages(self):
        plan = vt.plan_processing(fake_info(frames=300))
        self.assertFalse(plan.truncated or plan.resized or plan.stride > 1)
        self.assertEqual(plan.messages(), [])

    def test_frame_limit_uses_stride_and_keeps_timing(self):
        plan = vt.plan_processing(fake_info(frames=1200, fps=60.0))  # exactly 20 s at 60 fps
        self.assertFalse(plan.truncated)
        self.assertEqual(plan.stride, 2)
        self.assertLessEqual(plan.frames_to_process, vt.MAX_FRAMES)
        self.assertAlmostEqual(plan.effective_fps, 30.0)
        self.assertAlmostEqual(plan.step_seconds, 2 / 60.0)
        self.assertTrue(any("every 2nd frame" in m for m in plan.messages()))

    def test_unknown_length_is_capped_and_disclosed(self):
        info = vt.VideoInfo(640, 480, 30.0, False, 0, 0.0)
        plan = vt.plan_processing(info)
        self.assertLessEqual(plan.frames_to_process, vt.MAX_FRAMES)
        self.assertTrue(any("could not be determined" in m for m in plan.messages()))

    def test_resize_preserves_aspect_and_message(self):
        plan = vt.plan_processing(fake_info(3840, 2160, frames=100))
        self.assertEqual((plan.width, plan.height), (960, 540))
        self.assertIn("automatically resized from 3840 × 2160 to 960 × 540", plan.messages()[0])
        portrait = vt.plan_processing(fake_info(1080, 1920, frames=100))
        self.assertEqual((portrait.width, portrait.height), (540, 960))
        small = vt.plan_processing(fake_info(641, 481, frames=10))
        self.assertFalse(small.resized)
        self.assertEqual((small.width % 2, small.height % 2), (0, 0))

    def test_iter_frames_skips_with_correct_source_indices_and_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_video(d, width=640, height=480, frames=10)
            plan = vt.ProcessingPlan(640, 480, 320, 240, 3, 30.0, 10, 4, False, True, True)
            got = list(vt.iter_frames(path, plan))
        self.assertEqual([g[1] for g in got], [0, 3, 6, 9])
        self.assertEqual(got[0][2].shape, (240, 320, 3))

    def test_corrupt_video_gives_friendly_error(self):
        with self.assertRaisesRegex(vt.VideoError, "could not be decoded"):
            vt.analyze_upload(b"this is not a video" * 100, "bad.mp4")


class SelectionTests(unittest.TestCase):
    def test_click_mapping_scales_and_clamps(self):
        self.assertEqual(vt.map_click_to_frame(350, 175, 700, 350, 960, 480), (480, 240))
        self.assertEqual(vt.map_click_to_frame(-5, 9999, 700, 350, 960, 480), (0, 479))
        with self.assertRaises(ValueError):
            vt.map_click_to_frame(1, 1, 0, 0, 960, 480)

    def test_roi_corners_any_order_and_validation(self):
        self.assertEqual(vt.roi_from_corners((50, 60), (10, 20), 100, 100), (10, 20, 41, 41))
        with self.assertRaisesRegex(ValueError, "too small"):
            vt.roi_from_corners((10, 10), (12, 40), 100, 100)
        with self.assertRaisesRegex(ValueError, "inside"):
            vt.roi_from_corners((10, 10), (120, 40), 100, 100)

    def test_parameter_validation(self):
        vt.TrackingParams().validate()
        for bad in (dict(window=20), dict(block_size=4), dict(quality_level=0), dict(max_corners=0),
                    dict(max_level=9), dict(epsilon=float("nan")), dict(max_iterations=2.5)):
            with self.assertRaises(ValueError, msg=bad):
                vt.TrackingParams(**bad).validate()

    def test_defaults_match_original_experiment(self):
        from Modules.Module_6.Part_A import optical_flow as original
        p = vt.TrackingParams()
        self.assertEqual(p.feature_params(), original.FEATURE_PARAMS)
        self.assertEqual(p.lk_params(), original.LK_PARAMS)
        self.assertEqual((p.max_corners, p.quality_level, p.min_distance, p.block_size), (50, 0.3, 7, 7))


class TrackingTests(unittest.TestCase):
    def run_job(self, d, name="clip.mp4", roi=(30, 90, 60, 60), params=None, **video_args):
        path = make_video(d, name, **video_args)
        data = Path(path).read_bytes()
        analysis = vt.analyze_upload(data, name)
        return analysis, vt.run_tracking_job(data, name, analysis.plan, roi, params or vt.TrackingParams())

    def test_tracking_stats_csv_and_video(self):
        with tempfile.TemporaryDirectory() as d:
            analysis, result = self.run_job(d, frames=20, speed=3)
        s = result.summary
        self.assertGreater(s["initial_features"], 4)
        self.assertEqual(s["processed_frames"], 20)
        self.assertAlmostEqual(s["mean_u_per_step"], 3.0, delta=0.3)
        self.assertAlmostEqual(s["mean_v_per_step"], 0.0, delta=0.3)
        self.assertAlmostEqual(s["mean_magnitude_per_step"], 3.0, delta=0.3)
        self.assertEqual(s["motion_direction"], "Right")
        self.assertGreater(s["feature_retention_percent"], 50)
        self.assertAlmostEqual(s["mean_cumulative_u_final"], 3.0 * 19, delta=3)
        rows = list(csv.DictReader(io.StringIO(result.csv_bytes.decode())))
        self.assertEqual(list(rows[0].keys()), vt.VISITOR_CSV_COLUMNS)
        self.assertEqual(vt.VISITOR_CSV_COLUMNS[:9], ["frame", "feature_id", "old_x", "old_y", "new_x", "new_y", "u", "v", "magnitude"])
        self.assertEqual(min(int(r["frame"]) for r in rows), 2)
        for r in rows[:50]:
            self.assertAlmostEqual(float(r["magnitude"]), np.hypot(float(r["u"]), float(r["v"])), places=5)
            self.assertAlmostEqual(float(r["u"]), float(r["new_x"]) - float(r["old_x"]), places=5)
        self.assertEqual(len(result.per_frame), 19)
        self.assertEqual(json.loads(result.summary_json())["initial_features"], s["initial_features"])

    def test_output_is_browser_compatible_h264(self):
        with tempfile.TemporaryDirectory() as d:
            _, result = self.run_job(d, frames=12)
            out = Path(d) / "out.mp4"
            out.write_bytes(result.video_bytes)
            import imageio_ffmpeg
            probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(out)],
                                   capture_output=True, text=True).stderr
            capture = cv2.VideoCapture(str(out))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = capture.read()
            capture.release()
        self.assertIn("Video: h264", probe)
        self.assertIn("yuv420p", probe)
        self.assertLess(result.video_bytes.find(b"moov"), result.video_bytes.find(b"mdat"))  # fast start
        self.assertTrue(ok)
        self.assertEqual(frame.shape[:2], (240, 320))
        self.assertEqual(frames, 12)

    def test_feature_loss_is_handled_and_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            _, result = self.run_job(d, frames=16, vanish_at=6)
        counts = [f["tracked"] for f in result.per_frame]
        self.assertGreater(result.summary["initial_features"], 0)
        self.assertLess(result.summary["final_features"], result.summary["initial_features"])
        self.assertTrue(all(b <= a for a, b in zip([result.summary["initial_features"], *counts], counts)))
        self.assertEqual(result.summary["processed_frames"], 16)  # video still fully written
        if result.summary["final_features"] == 0:
            self.assertIsNotNone(result.summary["all_features_lost_at_frame"])
            self.assertIsNone(result.per_frame[-1]["mean_u"])

    def test_no_corners_raises_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "flat.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
            for _ in range(5):
                writer.write(np.full((240, 320, 3), 128, np.uint8))
            writer.release()
            data = path.read_bytes()
            analysis = vt.analyze_upload(data, "flat.mp4")
            with self.assertRaises(vt.NoFeaturesError):
                vt.run_tracking_job(data, "flat.mp4", analysis.plan, (20, 20, 100, 100), vt.TrackingParams())

    def test_stride_changes_time_interval(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_video(d, frames=30, speed=2)
            data = Path(path).read_bytes()
            base = vt.analyze_upload(data, "c.mp4").plan
            plan = vt.ProcessingPlan(base.source_width, base.source_height, base.width, base.height, 2, 30.0, 30, 15, False, True, False)
            result = vt.run_tracking_job(data, "c.mp4", plan, (30, 90, 60, 60), vt.TrackingParams())
        self.assertAlmostEqual(result.summary["mean_u_per_step"], 4.0, delta=0.5)
        self.assertEqual(result.summary["frame_stride"], 2)
        self.assertEqual(result.per_frame[0]["source_frame"], 3)
        self.assertAlmostEqual(result.per_frame[0]["time_s"], 2 / 30, places=5)

    def test_resized_displacements_reported_in_original_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            # Full-size scene moving 6 px/frame, tracked at half resolution.
            big = make_video(d, "big.mp4", width=640, height=480, frames=15, speed=6)
            data = Path(big).read_bytes()
            plan = vt.ProcessingPlan(640, 480, 320, 240, 1, 30.0, 15, 15, False, True, True)
            result = vt.run_tracking_job(data, "big.mp4", plan, (10, 40, 100, 100), vt.TrackingParams())
        self.assertAlmostEqual(result.summary["mean_u_per_step"], 6.0, delta=1.0)

    def test_progress_callback_and_temp_cleanup(self):
        seen, dirs = [], []
        real = tempfile.TemporaryDirectory

        class Spy(real):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                dirs.append(self.name)

        with tempfile.TemporaryDirectory() as d:
            path = make_video(d, frames=8)
            data = Path(path).read_bytes()
            analysis = vt.analyze_upload(data, "c.mp4")
            with mock.patch.object(vt.tempfile, "TemporaryDirectory", Spy):
                vt.run_tracking_job(data, "c.mp4", analysis.plan, (30, 90, 60, 60), vt.TrackingParams(),
                                    lambda f, t: seen.append(f))
                with self.assertRaises(vt.VideoError):
                    vt.run_tracking_job(data, "c.mp4", analysis.plan, (0, 0, 9999, 9999), vt.TrackingParams())
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(len(dirs), 2)
        self.assertFalse(any(Path(x).exists() for x in dirs))

    def test_concurrency_limit(self):
        taken = [vt._JOB_SLOTS.acquire(blocking=False) for _ in range(vt.MAX_CONCURRENT_JOBS)]
        try:
            with self.assertRaises(vt.BusyError):
                vt.run_tracking_job(b"x", "a.mp4", None, (0, 0, 10, 10), vt.TrackingParams())
        finally:
            for ok in taken:
                if ok:
                    vt._JOB_SLOTS.release()

    def test_encoder_unavailable_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_video(d, frames=5)
            data = Path(path).read_bytes()
            analysis = vt.analyze_upload(data, "c.mp4")
            with mock.patch.object(vt, "encoder_status", return_value=(False, "no libx264")):
                with self.assertRaisesRegex(vt.VideoError, "encoding is unavailable"):
                    vt.run_tracking_job(data, "c.mp4", analysis.plan, (30, 90, 60, 60), vt.TrackingParams())
        self.assertTrue(vt.encoder_status()[0])

    def test_direction_labels(self):
        self.assertEqual(vt.direction_label(1, 0), "Right")
        self.assertEqual(vt.direction_label(-1, 0), "Left")
        self.assertEqual(vt.direction_label(0, 1), "Down")
        self.assertEqual(vt.direction_label(0, -1), "Up")
        self.assertEqual(vt.direction_label(1, -1), "Up-right")
        self.assertEqual(vt.direction_label(0.001, 0), "Nearly stationary")


class OriginalExperimentComparisonTests(unittest.TestCase):
    @unittest.skipUnless((PART_A / "input" / "video_1_horizontal.MOV").is_file(), "original video not available")
    def test_matches_original_video_1_within_resize_tolerance(self):
        data = (PART_A / "input" / "video_1_horizontal.MOV").read_bytes()
        analysis = vt.analyze_upload(data, "video_1_horizontal.MOV")
        s = analysis.plan.scale_x
        # ROI around the soda can, taken from the original experiment's first tracked features.
        roi = (int(55 * s), int(1120 * s), int(135 * s), int(235 * s))
        result = vt.run_tracking_job(data, "video_1_horizontal.MOV", analysis.plan, roi, vt.TrackingParams())
        with (PART_A / "output" / "video_1_tracking.csv").open() as handle:
            rows = [r for r in csv.DictReader(handle) if int(r["frame"]) <= analysis.plan.frames_to_process]
        original_u = np.mean([float(r["u"]) for r in rows])
        self.assertAlmostEqual(result.summary["mean_u_per_step"], original_u, delta=0.05)
        self.assertEqual(result.summary["motion_direction"], "Right")
        self.assertGreater(result.summary["feature_retention_percent"], 90)


if __name__ == "__main__":
    unittest.main()
