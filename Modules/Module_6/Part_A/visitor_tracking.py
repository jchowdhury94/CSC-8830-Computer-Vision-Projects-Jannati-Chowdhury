"""Visitor video tracking backend for Module 6 Part A (no Streamlit imports).

Reuses the original experiment's Shi-Tomasi / pyramidal Lucas-Kanade settings,
annotation style and CSV columns from optical_flow.py. Frames are decoded
sequentially, resized to the processing resolution, tracked, and streamed into
an H.264 MP4 encoder; nothing but the previous grayscale frame is retained.

Positions and displacements in the CSV and summary are in ORIGINAL-frame pixels
(processing-resolution values divided by the resize scale). One displacement is
the motion between two consecutive *processed* frames, which are `stride`
source frames (stride / source_fps seconds) apart.
"""

import csv
import functools
import io
import json
import math
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from Modules.Module_6.Part_A.optical_flow import (
    CSV_COLUMNS, FEATURE_PARAMS, LK_PARAMS, annotate, TRACK_COLOR,
)

MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_DURATION_S = 20.0
MAX_FRAMES = 600
MAX_SIDE = 960
MIN_ROI_SIDE = 8
MAX_CONCURRENT_JOBS = 2
ALLOWED_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
FALLBACK_FPS = 30.0
VISITOR_CSV_COLUMNS = CSV_COLUMNS + ["source_frame", "time_s", "cumulative_u", "cumulative_v"]

_JOB_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


class VideoError(Exception):
    """A problem with the visitor's video that can be shown as-is."""


class NoFeaturesError(VideoError):
    pass


class BusyError(VideoError):
    pass


# ---------------------------------------------------------------- parameters
@dataclass(frozen=True)
class TrackingParams:
    max_corners: int = FEATURE_PARAMS["maxCorners"]
    quality_level: float = FEATURE_PARAMS["qualityLevel"]
    min_distance: int = FEATURE_PARAMS["minDistance"]
    block_size: int = FEATURE_PARAMS["blockSize"]
    window: int = LK_PARAMS["winSize"][0]
    max_level: int = LK_PARAMS["maxLevel"]
    max_iterations: int = LK_PARAMS["criteria"][1]
    epsilon: float = LK_PARAMS["criteria"][2]

    def validate(self):
        """Raise ValueError naming the first out-of-range setting."""
        checks = (
            ("Maximum corners", self.max_corners, 1, 200, int),
            ("Quality level", self.quality_level, 0.01, 0.95, float),
            ("Minimum distance", self.min_distance, 1, 50, int),
            ("Block size", self.block_size, 3, 31, int),
            ("Lucas-Kanade window", self.window, 5, 51, int),
            ("Pyramid maximum level", self.max_level, 0, 5, int),
            ("Iterations", self.max_iterations, 1, 100, int),
            ("Epsilon", self.epsilon, 0.001, 1.0, float),
        )
        for name, value, low, high, kind in checks:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or (kind is int and int(value) != value):
                raise ValueError(f"{name} must be {'a whole number' if kind is int else 'a number'}.")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}.")
        if self.window % 2 == 0:
            raise ValueError("Lucas-Kanade window must be an odd number.")
        if self.block_size % 2 == 0:
            raise ValueError("Block size must be an odd number.")

    def feature_params(self):
        return dict(maxCorners=int(self.max_corners), qualityLevel=float(self.quality_level),
                    minDistance=int(self.min_distance), blockSize=int(self.block_size))

    def lk_params(self):
        return dict(winSize=(int(self.window), int(self.window)), maxLevel=int(self.max_level),
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                              int(self.max_iterations), float(self.epsilon)))


# ------------------------------------------------------------- probing/plan
@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_assumed: bool
    frame_count: int  # 0 when the container does not report it
    duration: float  # seconds; 0.0 when unknown


@dataclass(frozen=True)
class ProcessingPlan:
    source_width: int
    source_height: int
    width: int  # processing resolution (even numbers for H.264)
    height: int
    stride: int  # process every stride-th source frame
    source_fps: float
    source_frames_used: int  # first N source frames considered
    frames_to_process: int
    truncated: bool
    length_known: bool
    resized: bool

    @property
    def scale_x(self):
        return self.width / self.source_width

    @property
    def scale_y(self):
        return self.height / self.source_height

    @property
    def effective_fps(self):
        return self.source_fps / self.stride

    @property
    def step_seconds(self):
        return self.stride / self.source_fps

    def messages(self):
        """Disclosures to show whenever optimization changes the video."""
        notes = []
        if self.resized:
            notes.append(f"Your video was automatically resized from {self.source_width} × {self.source_height} "
                         f"to {self.width} × {self.height} for faster processing.")
        if self.truncated:
            notes.append(f"Your video is longer than the limit, so only the first segment "
                         f"({self.source_frames_used / self.source_fps:.1f} s, "
                         f"{self.source_frames_used} frames) will be tracked.")
        elif not self.length_known:
            notes.append(f"The video length could not be determined, so tracking stops at "
                         f"{MAX_DURATION_S:.0f} s or {MAX_FRAMES} frames, whichever comes first.")
        if self.stride > 1:
            notes.append(f"To stay within {MAX_FRAMES} processed frames, every {self.stride}"
                         f"{_ordinal_suffix(self.stride)} frame is tracked ({self.effective_fps:.1f} frames per second "
                         f"instead of {self.source_fps:.1f}). Each tracking step now spans "
                         f"{self.step_seconds:.3f} s, so per-step displacements are larger than in the full-rate video.")
        return notes


def _ordinal_suffix(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def check_upload(size_bytes, filename):
    """Reject files outside the size/type limits. The filename is only inspected, never used as a path."""
    if size_bytes > MAX_FILE_BYTES:
        raise VideoError(f"This file is {size_bytes / 1024 / 1024:.1f} MB; the limit is "
                         f"{MAX_FILE_BYTES // 1024 // 1024} MB. Please trim or compress the video and upload it again.")
    if size_bytes <= 0:
        raise VideoError("The uploaded file is empty.")
    if safe_extension(filename) is None:
        raise VideoError("Unsupported file type. Please upload " + ", ".join(e.upper().lstrip(".") for e in ALLOWED_EXTENSIONS) + ".")


def safe_extension(filename):
    suffix = Path(str(filename or "")).suffix.lower()
    return suffix if suffix in ALLOWED_EXTENSIONS else None


def probe_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError(_decode_hint())
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoError(_decode_hint())
        height, width = frame.shape[:2]  # authoritative, includes rotation metadata
    except cv2.error as error:
        raise VideoError(_decode_hint()) from error
    finally:
        capture.release()
    fps_assumed = not (np.isfinite(fps) and 1.0 <= fps <= 240.0)
    if fps_assumed:
        fps = FALLBACK_FPS
    count = int(count) if np.isfinite(count) and count > 0 else 0
    return VideoInfo(width, height, float(fps), fps_assumed, count, count / fps if count else 0.0)


def _decode_hint():
    return ("This video could not be decoded. It may be corrupted or use a codec this server cannot read "
            "(some HEVC/H.265 MOV files are unsupported). Try re-exporting it as H.264 MP4.")


def plan_processing(info):
    longest = max(info.width, info.height)
    scale = min(1.0, MAX_SIDE / longest)
    resized = scale < 1.0
    width, height = info.width, info.height
    if resized:
        width, height = round(info.width * scale), round(info.height * scale)
    width, height = max(2, width - width % 2), max(2, height - height % 2)  # H.264 needs even sizes

    duration_frames = int(math.floor(MAX_DURATION_S * info.fps + 0.5))
    length_known = info.frame_count > 0
    if length_known:
        used = min(info.frame_count, duration_frames)
        truncated = info.frame_count > duration_frames
    else:
        used, truncated = duration_frames, False
    stride = max(1, math.ceil(used / MAX_FRAMES))
    to_process = min(MAX_FRAMES, math.ceil(used / stride))
    return ProcessingPlan(info.width, info.height, width, height, stride, info.fps, used, to_process,
                          truncated, length_known, resized)


def iter_frames(path, plan):
    """Yield (processed_index, source_index, resized BGR frame), decoding sequentially."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError(_decode_hint())
        source_index = emitted = 0
        while source_index < plan.source_frames_used and emitted < plan.frames_to_process:
            if source_index % plan.stride == 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if frame.shape[1] != plan.width or frame.shape[0] != plan.height:
                    interpolation = cv2.INTER_AREA if plan.resized else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (plan.width, plan.height), interpolation=interpolation)
                yield emitted, source_index, frame
                emitted += 1
            elif not capture.grab():
                break
            source_index += 1
    except cv2.error as error:
        raise VideoError(_decode_hint()) from error
    finally:
        capture.release()


@dataclass(frozen=True)
class UploadAnalysis:
    info: VideoInfo
    plan: ProcessingPlan
    first_frame: np.ndarray  # BGR at processing resolution


def analyze_upload(data, filename):
    """Validate an upload and return its metadata, processing plan and first processed frame."""
    check_upload(len(data), filename)
    with tempfile.TemporaryDirectory(prefix="m6_probe_") as directory:
        path = Path(directory) / f"input{safe_extension(filename)}"
        path.write_bytes(data)
        info = probe_video(path)
        plan = plan_processing(info)
        first = next(iter_frames(path, plan), None)
    if first is None:
        raise VideoError(_decode_hint())
    return UploadAnalysis(info, plan, first[2])


# ----------------------------------------------------------------- selection
def map_click_to_frame(x, y, display_width, display_height, frame_width, frame_height):
    """Map a click on the displayed preview to processing-frame pixel coordinates."""
    if display_width <= 0 or display_height <= 0:
        raise ValueError("The image preview is not ready. Please click again.")
    px = int(np.clip(x * frame_width / display_width, 0, frame_width - 1))
    py = int(np.clip(y * frame_height / display_height, 0, frame_height - 1))
    return px, py


def roi_from_corners(first, second, frame_width, frame_height):
    """Return a validated (x, y, w, h) processing-frame rectangle from two opposite corners."""
    left, right = sorted((first[0], second[0]))
    top, bottom = sorted((first[1], second[1]))
    x, y, width, height = left, top, right - left + 1, bottom - top + 1
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("The rectangle must stay inside the video frame.")
    if width < MIN_ROI_SIDE or height < MIN_ROI_SIDE:
        raise ValueError(f"The rectangle is too small. Make it at least {MIN_ROI_SIDE} × {MIN_ROI_SIDE} pixels.")
    return x, y, width, height


# ------------------------------------------------------------------ encoding
@functools.lru_cache(maxsize=1)
def encoder_status():
    """Return (available, detail) for the bundled FFmpeg's libx264 encoder."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        output = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    except Exception as error:  # missing package, missing binary, or blocked execution
        return False, f"FFmpeg is not available on this server ({type(error).__name__})."
    if "libx264" not in output:
        return False, "The server's FFmpeg build does not include the H.264 (libx264) encoder."
    return True, exe


class H264Writer:
    """Stream BGR frames to a browser-compatible H.264 MP4 (yuv420p, fast-start)."""

    def __init__(self, path, width, height, fps):
        available, detail = encoder_status()
        if not available:
            raise VideoError(f"Video encoding is unavailable: {detail}")
        import imageio_ffmpeg
        self._writer = imageio_ffmpeg.write_frames(
            str(path), (width, height), fps=fps, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
            macro_block_size=2, quality=6, ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart", "-preset", "veryfast"])
        self._writer.send(None)
        self._closed = False

    def write(self, frame):
        try:
            self._writer.send(np.ascontiguousarray(frame).tobytes())
        except Exception as error:
            self.abort()
            raise VideoError("Video encoding failed while writing frames.") from error

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._writer.close()
            except Exception as error:
                raise VideoError("Video encoding failed while finishing the file.") from error

    def abort(self):
        if not self._closed:
            self._closed = True
            try:
                self._writer.close()
            except Exception:
                pass


# ------------------------------------------------------------------ tracking
@dataclass
class TrackingResult:
    video_bytes: bytes
    csv_bytes: bytes
    summary: dict
    per_frame: list = field(default_factory=list)  # dicts, one per processed frame after the first

    def summary_json(self):
        return json.dumps(self.summary, indent=2).encode("utf-8")


def direction_label(mean_u, mean_v, threshold=0.05):
    """Describe image-plane motion; image y grows downward so positive v is 'down'."""
    if math.hypot(mean_u, mean_v) < threshold:
        return "Nearly stationary"
    angle = math.degrees(math.atan2(-mean_v, mean_u)) % 360
    names = ("right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right")
    return names[int((angle + 22.5) // 45) % 8].capitalize()


def _draw_tracks(frame, old_points, new_points):
    for old, new in zip(old_points, new_points):
        cv2.line(frame, tuple(np.rint(old).astype(int)), tuple(np.rint(new).astype(int)),
                 TRACK_COLOR, 1, cv2.LINE_AA)


def track_video(video_path, plan, roi, params, output_path, progress=None):
    """Track features inside `roi` (processing-frame x, y, w, h) and write an annotated H.264 MP4.

    Mirrors optical_flow.py: Shi-Tomasi inside the ROI on frame 1, pyramidal LK between
    consecutive processed frames, lost features dropped permanently and never replaced.
    """
    params.validate()
    x, y, roi_w, roi_h = roi
    if x < 0 or y < 0 or roi_w < 1 or roi_h < 1 or x + roi_w > plan.width or y + roi_h > plan.height:
        raise VideoError("The selected rectangle is outside the video frame.")
    available, detail = encoder_status()
    if not available:
        raise VideoError(f"Video encoding is unavailable: {detail}")

    lk_params = params.lk_params()
    csv_buffer = io.StringIO(newline="")
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(VISITOR_CSV_COLUMNS)
    sx, sy = plan.scale_x, plan.scale_y
    writer = None
    per_frame, points, feature_ids, previous_gray = [], None, None, None
    initial_points_orig = None
    initial_count = 0
    first_lost_frame = None
    processed = 0
    try:
        for index, source_index, frame in iter_frames(video_path, plan):
            processed = index + 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            source_frame = source_index + 1
            time_s = source_index / plan.source_fps
            if index == 0:
                mask = np.zeros_like(gray)
                mask[y:y + roi_h, x:x + roi_w] = 255
                points = cv2.goodFeaturesToTrack(gray, mask=mask, **params.feature_params())
                if points is None or len(points) == 0:
                    raise NoFeaturesError("No trackable corners were found inside the selected rectangle. "
                                          "Choose a region with more texture or edges, or lower the quality level.")
                initial_count = len(points)
                feature_ids = np.arange(initial_count)
                initial_points_orig = points.reshape(-1, 2) / np.array([sx, sy])
                writer = H264Writer(output_path, plan.width, plan.height, plan.effective_fps)
            elif len(points):
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None, **lk_params)
                if next_points is None or status is None:
                    valid = np.zeros(len(points), dtype=bool)
                    next_points = np.empty_like(points)
                else:
                    valid = (status.reshape(-1) == 1) & np.isfinite(next_points.reshape(-1, 2)).all(axis=1)
                old_points = points.reshape(-1, 2)[valid]
                new_points = next_points.reshape(-1, 2)[valid]
                feature_ids = feature_ids[valid]
                _draw_tracks(frame, old_points, new_points)
                scale = np.array([sx, sy])
                old_orig, new_orig = old_points / scale, new_points / scale
                delta = new_orig.astype(np.float64) - old_orig.astype(np.float64)
                magnitude = np.hypot(delta[:, 0], delta[:, 1])
                cumulative = new_orig - initial_points_orig[feature_ids]
                for fid, o, n, d, m, c in zip(feature_ids, old_orig, new_orig, delta, magnitude, cumulative):
                    values = [*o, *n, d[0], d[1], m]
                    csv_writer.writerow([index + 1, int(fid), *[f"{v:.8f}" for v in values],
                                         source_frame, f"{time_s:.6f}", f"{c[0]:.8f}", f"{c[1]:.8f}"])
                points = new_points.reshape(-1, 1, 2)
                count = len(points)
                per_frame.append(dict(
                    frame=index + 1, source_frame=source_frame, time_s=time_s, tracked=count,
                    mean_u=float(delta[:, 0].mean()) if count else None,
                    mean_v=float(delta[:, 1].mean()) if count else None,
                    mean_magnitude=float(magnitude.mean()) if count else None,
                    mean_cumulative_u=float(cumulative[:, 0].mean()) if count else None,
                    mean_cumulative_v=float(cumulative[:, 1].mean()) if count else None))
                if count == 0 and first_lost_frame is None:
                    first_lost_frame = index + 1
            else:
                per_frame.append(dict(frame=index + 1, source_frame=source_frame, time_s=time_s, tracked=0,
                                      mean_u=None, mean_v=None, mean_magnitude=None,
                                      mean_cumulative_u=None, mean_cumulative_v=None))
            annotate(frame, source_frame, points)
            writer.write(frame)
            previous_gray = gray
            if progress is not None:
                progress(min(1.0, processed / plan.frames_to_process),
                         f"Tracking frame {processed} of {plan.frames_to_process}")
        if processed == 0:
            raise VideoError(_decode_hint())
        writer.close()
    except BaseException:
        if writer is not None:
            writer.abort()
        raise

    video_bytes = Path(output_path).read_bytes()
    if not video_bytes:
        raise VideoError("Video encoding produced an empty file.")
    summary = summarize(plan, initial_count, len(points), processed, csv_buffer.getvalue(), per_frame, first_lost_frame)
    return TrackingResult(video_bytes, csv_buffer.getvalue().encode("utf-8"), summary, per_frame)


def summarize(plan, initial_count, final_count, processed, csv_text, per_frame, first_lost_frame=None):
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    u = np.array([float(r["u"]) for r in rows])
    v = np.array([float(r["v"]) for r in rows])
    magnitude = np.array([float(r["magnitude"]) for r in rows])
    mean_u = float(u.mean()) if len(u) else 0.0
    mean_v = float(v.mean()) if len(v) else 0.0
    last = next((f for f in reversed(per_frame) if f["tracked"]), None)
    return {
        "initial_features": int(initial_count),
        "final_features": int(final_count),
        "feature_retention_percent": round(100.0 * final_count / initial_count, 2) if initial_count else 0.0,
        "all_features_lost_at_frame": first_lost_frame,
        "mean_u_per_step": mean_u,
        "mean_v_per_step": mean_v,
        "mean_magnitude_per_step": float(magnitude.mean()) if len(magnitude) else 0.0,
        "mean_cumulative_u_final": last["mean_cumulative_u"] if last else None,
        "mean_cumulative_v_final": last["mean_cumulative_v"] if last else None,
        "motion_direction": direction_label(mean_u, mean_v),
        "displacement_units": "pixels in the original video frame",
        "step_definition": f"one step = {plan.stride} source frame(s) = {plan.step_seconds:.6f} s",
        "processed_frames": int(processed),
        "source_fps": plan.source_fps,
        "frame_stride": plan.stride,
        "source_resolution": [plan.source_width, plan.source_height],
        "processing_resolution": [plan.width, plan.height],
    }


def run_tracking_job(data, filename, plan, roi, params, progress=None):
    """Run one isolated job: private temp dir, bounded concurrency, guaranteed cleanup."""
    extension = safe_extension(filename)
    if extension is None:
        raise VideoError("Unsupported file type.")
    if not _JOB_SLOTS.acquire(blocking=False):
        raise BusyError("The server is busy tracking other videos. Please try again in a minute.")
    try:
        with tempfile.TemporaryDirectory(prefix="m6_track_") as directory:
            source = Path(directory) / f"input{extension}"
            source.write_bytes(data)
            return track_video(source, plan, roi, params, Path(directory) / "annotated.mp4", progress)
    finally:
        _JOB_SLOTS.release()
