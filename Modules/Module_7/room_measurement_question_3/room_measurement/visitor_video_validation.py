"""Visitor four-video inspection and validation (Phase 2B, Stage B).

Accepts four local video paths (left/right for each of two stereo placements), inspects each one and reports structured results.
Read-only: source files are never written, moved or deleted.  No feature matching, pose estimation or measurement happens here.

Decoding policy
---------------
OpenCV (``cv2.VideoCapture``) is the decoder the measurement pipeline uses, so it is the one that decides whether a file is
processable.  Opening a file is not trusted: one real frame must decode.  If OpenCV fails, ``imageio-ffmpeg`` (already a project
dependency) is tried as a *diagnostic only*: it fills in codec/size information and lets the error message say "ffmpeg can read this
but OpenCV cannot -- re-encode to H.264", but the file is still an ERROR because the pipeline could not process it.
No transcoding is performed.  Codec support depends on the local OpenCV/ffmpeg build; nothing here claims universal HEVC support,
or that a deployment host decodes what this machine does.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .visitor_calibration import (ACCURACY_NOTE, AdaptedCalibration, CompatibilityResult, Issue, Severity,
                                  assess_capture_compatibility, orientation_of, worst)

VIDEO_LABELS = ("left_video_1", "right_video_1", "left_video_2", "right_video_2")


@dataclass(frozen=True)
class VideoLimits:
    min_file_bytes: int = 1
    max_file_bytes: int = 200 * 1024 * 1024    # matches Streamlit's default upload cap; raise it for local use
    min_duration_s: float = 1.0
    max_duration_s: float = 120.0
    min_frames: int = 2
    min_side_px: int = 16
    max_side_px: int = 16_384
    min_fps: float = 1.0
    max_fps: float = 480.0
    max_fps_difference: float = 1.0            # across the four videos; larger gives a WARNING
    max_total_frames: int = 10_000_000         # sanity bound on a corrupt container's frame-count field


DEFAULT_VIDEO_LIMITS = VideoLimits()


@dataclass
class VideoReport:
    label: str
    path: str
    exists: bool = False
    size_bytes: int | None = None
    opened: bool = False
    frame_decoded: bool = False
    width: int | None = None              # decoded frame width (after OpenCV applies rotation metadata)
    height: int | None = None
    orientation: str | None = None
    fps: float | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    codec: str | None = None
    decoder: str | None = None            # "opencv", or "imageio-ffmpeg (diagnostic only)"
    checks: list = field(default_factory=list)

    @property
    def status(self) -> Severity:
        return worst(self.checks)

    @property
    def errors(self):
        return [c for c in self.checks if c.severity is Severity.ERROR]

    @property
    def warnings(self):
        return [c for c in self.checks if c.severity is Severity.WARNING]

    def to_dict(self):
        return dict(label=self.label, path=self.path, status=self.status.value, exists=self.exists, size_bytes=self.size_bytes,
                    opened=self.opened, frame_decoded=self.frame_decoded, width=self.width, height=self.height,
                    orientation=self.orientation, fps=self.fps, frame_count=self.frame_count, duration_s=self.duration_s,
                    codec=self.codec, decoder=self.decoder, checks=[c.to_dict() for c in self.checks])


@dataclass
class ValidationResult:
    status: Severity                      # worst severity over everything
    can_proceed: bool                     # True only when there is no ERROR
    videos: dict                          # label -> VideoReport, in VIDEO_LABELS order
    set_checks: list                      # cross-video / calibration checks (Issue)
    calibration_compatibility: CompatibilityResult | None
    note: str = ACCURACY_NOTE

    @property
    def issues(self):
        out = [c for v in self.videos.values() for c in v.checks] + list(self.set_checks)
        return out

    @property
    def errors(self):
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def to_dict(self):
        return dict(status=self.status.value, can_proceed=self.can_proceed, note=self.note,
                    videos={k: v.to_dict() for k, v in self.videos.items()}, set_checks=[c.to_dict() for c in self.set_checks],
                    calibration_compatibility=None if self.calibration_compatibility is None else self.calibration_compatibility.to_dict(),
                    errors=[i.to_dict() for i in self.errors], warnings=[i.to_dict() for i in self.warnings])


# ---------------------------------------------------------------- single-video inspection

def _fourcc(code) -> str | None:
    try:
        c = int(code)
        if c <= 0:
            return None
        s = c.to_bytes(4, "little").decode("latin-1")
        return s if s.isprintable() and s.strip() else None
    except (TypeError, ValueError, OverflowError):
        return None


def _ffmpeg_diagnostic(path):
    """imageio-ffmpeg probe + one-frame decode.  Returns dict(meta=..., decoded=bool) or None if unavailable/failed."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    gen = None
    try:
        gen = imageio_ffmpeg.read_frames(str(path))
        meta = next(gen)
        decoded = len(next(gen)) > 0
        return dict(meta=meta, decoded=decoded)
    except Exception:                       # ffmpeg missing, unreadable file, ...
        return None
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:
                pass


def inspect_video(path, label: str, limits: VideoLimits = DEFAULT_VIDEO_LIMITS) -> VideoReport:
    """Inspect one video (read-only) and run the per-file checks.  Never raises for bad input."""
    rep = VideoReport(label=label, path=str(path))
    add = lambda sev, code, msg: rep.checks.append(Issue(sev, code, msg, label))
    p = Path(path)

    # file-level
    try:
        if not p.exists():
            add(Severity.ERROR, "FILE_NOT_FOUND", f"The {label.replace('_', ' ')} file was not found. Choose the file again."); return rep
        if not p.is_file():
            add(Severity.ERROR, "NOT_A_FILE", f"The {label.replace('_', ' ')} path is not a file."); return rep
        rep.exists = True
        rep.size_bytes = p.stat().st_size
    except OSError as e:
        add(Severity.ERROR, "FILE_UNREADABLE", f"The {label.replace('_', ' ')} file cannot be accessed ({e.strerror or type(e).__name__})."); return rep
    if rep.size_bytes < limits.min_file_bytes:
        add(Severity.ERROR, "EMPTY_FILE", f"The {label.replace('_', ' ')} file is empty (0 bytes). The upload or copy probably failed; choose the file again."); return rep
    mb = rep.size_bytes / 1048576
    if rep.size_bytes > limits.max_file_bytes:
        add(Severity.ERROR, "FILE_TOO_LARGE", f"The {label.replace('_', ' ')} file is {mb:.0f} MB; the limit is {limits.max_file_bytes / 1048576:.0f} MB. Record a shorter clip or lower the video quality.")
    else:
        add(Severity.PASS, "FILE_SIZE_OK", f"File size {mb:.1f} MB is within the {limits.max_file_bytes / 1048576:.0f} MB limit.")

    # decoding with OpenCV
    cap = None
    frame = None
    try:
        cap = cv2.VideoCapture(str(p))
        rep.opened = bool(cap.isOpened())
        if rep.opened:
            rep.decoder = "opencv"
            rep.codec = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
            fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            rep.fps = float(fps) if fps is not None and math.isfinite(fps) else None
            rep.frame_count = int(n) if n is not None and math.isfinite(n) and abs(n) < 1e15 else None
            ok, frame = cap.read()
            rep.frame_decoded = bool(ok) and frame is not None and getattr(frame, "size", 0) > 0
    except Exception as e:                  # OpenCV can raise on exotic input
        add(Severity.ERROR, "DECODER_EXCEPTION", f"OpenCV raised {type(e).__name__} while reading the {label.replace('_', ' ')} file.")
    finally:
        if cap is not None:
            cap.release()

    if not rep.opened or not rep.frame_decoded:
        what = "could not be opened" if not rep.opened else "opened, but no frame could be decoded"
        diag = _ffmpeg_diagnostic(p)
        if diag and diag["decoded"]:
            m = diag["meta"]
            rep.decoder = "imageio-ffmpeg (diagnostic only)"
            rep.codec = rep.codec or m.get("codec")
            hint = (f"ffmpeg can read it (codec {m.get('codec')}), but OpenCV, which the measurement pipeline uses, cannot. "
                    "Re-export the video as H.264 MP4 and try again.")
        else:
            hint = "The file may be corrupt, truncated, not a video, or use a codec this system cannot decode. Re-export or re-record it as H.264 MP4."
        add(Severity.ERROR, "OPEN_FAILED" if not rep.opened else "DECODE_FAILED", f"The {label.replace('_', ' ')} file {what}. {hint}")
        return rep
    add(Severity.PASS, "DECODE_OK", "Opened, and a frame decoded successfully.")

    # frame geometry from the DECODED frame (authoritative; honours rotation metadata)
    h, w = frame.shape[:2]
    rep.width, rep.height, rep.orientation = int(w), int(h), orientation_of(w, h)
    if not (limits.min_side_px <= w <= limits.max_side_px and limits.min_side_px <= h <= limits.max_side_px):
        add(Severity.ERROR, "INVALID_DIMENSIONS", f"Frame size {w} x {h} is outside the supported range ({limits.min_side_px}-{limits.max_side_px} px per side).")
    else:
        add(Severity.PASS, "DIMENSIONS_OK", f"Frame size {w} x {h} ({rep.orientation}).")

    # frame rate / count / duration
    fps_ok = rep.fps is not None and limits.min_fps <= rep.fps <= limits.max_fps
    n_ok = rep.frame_count is not None and 1 <= rep.frame_count <= limits.max_total_frames
    if not fps_ok:
        add(Severity.ERROR, "INVALID_FRAME_RATE", f"The reported frame rate ({rep.fps}) is not valid (expected {limits.min_fps:g}-{limits.max_fps:g} fps). The file's metadata may be damaged.")
    else:
        add(Severity.PASS, "FRAME_RATE_OK", f"Frame rate {rep.fps:.3g} fps.")
    if not n_ok:
        add(Severity.ERROR, "INVALID_FRAME_COUNT", f"The reported frame count ({rep.frame_count}) is not valid. The file's metadata may be damaged.")
    elif rep.frame_count < limits.min_frames:
        add(Severity.ERROR, "TOO_FEW_FRAMES", f"The video has only {rep.frame_count} frame(s); at least {limits.min_frames} are needed.")
    else:
        add(Severity.PASS, "FRAME_COUNT_OK", f"{rep.frame_count} frames.")
    if fps_ok and n_ok:
        rep.duration_s = rep.frame_count / rep.fps
        if rep.duration_s < limits.min_duration_s:
            add(Severity.ERROR, "VIDEO_TOO_SHORT", f"Duration {rep.duration_s:.1f} s is shorter than the {limits.min_duration_s:g} s minimum. Record a longer clip.")
        elif rep.duration_s > limits.max_duration_s:
            add(Severity.ERROR, "VIDEO_TOO_LONG", f"Duration {rep.duration_s:.1f} s exceeds the {limits.max_duration_s:g} s limit. Trim the clip.")
        else:
            add(Severity.PASS, "DURATION_OK", f"Approximate duration {rep.duration_s:.1f} s (frame count / frame rate).")
    return rep


# ---------------------------------------------------------------- four-video validation

def validate_visitor_videos(left_video_1, right_video_1, left_video_2, right_video_2, calibration: AdaptedCalibration | None = None,
                            limits: VideoLimits = DEFAULT_VIDEO_LIMITS) -> ValidationResult:
    """Inspect and validate the four videos (and, if given, their compatibility with an adapted visitor calibration).

    PASS statuses refer to input checks only and never imply measurement accuracy.  can_proceed is False when any ERROR exists.
    """
    paths = dict(zip(VIDEO_LABELS, (left_video_1, right_video_1, left_video_2, right_video_2)))
    videos = {}
    for label, path in paths.items():
        videos[label] = inspect_video(path, label, limits) if path is not None and str(path) != "" else _missing(label)
    set_checks: list = []
    add = lambda sev, code, msg, subj="videos": set_checks.append(Issue(sev, code, msg, subj))

    # the same file must not be used twice (a common upload mistake)
    seen = {}
    for label, rep in videos.items():
        if rep.exists:
            try:
                key = os.path.realpath(rep.path)
            except OSError:
                continue
            if key in seen:
                add(Severity.ERROR, "DUPLICATE_FILE", f"{seen[key].replace('_', ' ')} and {label.replace('_', ' ')} are the same file. Each of the four videos must be a different recording.")
            seen[key] = label

    decoded = {k: v for k, v in videos.items() if v.frame_decoded and v.width}
    if len(decoded) == 4:
        sizes = {(v.width, v.height) for v in decoded.values()}
        orients = {v.orientation for v in decoded.values()}
        desc = ", ".join(f"{k.replace('_', ' ')}: {v.width} x {v.height}" for k, v in decoded.items())
        if len(orients) > 1:
            add(Severity.ERROR, "VIDEO_ORIENTATION_MISMATCH", f"The videos mix portrait and landscape ({desc}). Film all four in the same orientation.")
        elif len(sizes) > 1:
            add(Severity.ERROR, "VIDEO_RESOLUTION_MISMATCH", f"The four videos do not share one resolution ({desc}). Film all four with the same camera settings.")
        else:
            w, h = next(iter(sizes))
            add(Severity.PASS, "VIDEOS_CONSISTENT", f"All four videos are {w} x {h} ({next(iter(orients))}).")
        fps = [v.fps for v in decoded.values() if v.fps]
        if fps and max(fps) - min(fps) > limits.max_fps_difference:
            add(Severity.WARNING, "FRAME_RATE_DIFFERS", f"Frame rates differ across the videos ({min(fps):.3g}-{max(fps):.3g} fps); synchronisation-sensitive steps may be less reliable.")
    elif any(v.exists for v in videos.values()):
        add(Severity.WARNING, "CONSISTENCY_NOT_CHECKED", "Cross-video resolution/orientation checks were skipped because not every video could be decoded.")

    compat = None
    if calibration is None:
        add(Severity.WARNING, "NO_CALIBRATION", "No calibration was supplied, so calibration/video compatibility was not checked.", "calibration")
    elif not calibration.ok:
        add(Severity.ERROR, "CALIBRATION_INVALID", "The calibration is invalid, so compatibility with the videos cannot be assessed: "
            + "; ".join(i.message for i in calibration.errors), "calibration")
    elif not decoded:
        add(Severity.WARNING, "COMPATIBILITY_NOT_CHECKED", "Calibration compatibility was skipped because no video could be decoded.", "calibration")
    else:
        left = [(v.width, v.height) for k, v in decoded.items() if k.startswith("left")]
        right = [(v.width, v.height) for k, v in decoded.items() if k.startswith("right")]
        compat = assess_capture_compatibility(calibration, left, right)
        add(compat.severity, compat.status.value, compat.message, "calibration")
        if len(decoded) < 4:
            add(Severity.WARNING, "COMPATIBILITY_PARTIAL", "Calibration compatibility was assessed on the decodable videos only.", "calibration")

    everything = [c for v in videos.values() for c in v.checks] + set_checks
    status = worst(everything)
    return ValidationResult(status=status, can_proceed=status is not Severity.ERROR, videos=videos, set_checks=set_checks,
                            calibration_compatibility=compat)


def _missing(label):
    rep = VideoReport(label=label, path="")
    rep.checks.append(Issue(Severity.ERROR, "FILE_NOT_PROVIDED", f"No file was provided for the {label.replace('_', ' ')}.", label))
    return rep


# ---------------------------------------------------------------- temp-file safety for future uploads

class TemporaryVideoWorkspace:
    """A dedicated temporary directory for staging uploaded bytes before inspection.

    * Files get generated names (never the visitor's filename), created exclusively, so nothing existing can be overwritten.
    * ``close()`` / leaving the ``with`` block removes only the directory this object created, even after an exception.
    * It never touches paths it did not create.
    """

    def __init__(self, prefix="visitor_videos_"):
        self._dir = Path(tempfile.mkdtemp(prefix=prefix))
        self._n = 0

    @property
    def path(self) -> Path:
        return self._dir

    def write(self, data: bytes, suffix: str = ".mp4") -> Path:
        if self._dir is None:
            raise RuntimeError("workspace is closed")
        suffix = "." + "".join(c for c in suffix.lstrip(".").lower() if c.isalnum())[:8] if suffix.strip(".") else ""
        self._n += 1
        target = self._dir / f"video_{self._n:02d}{suffix}"
        with open(target, "xb") as fh:         # 'x': fail rather than overwrite
            fh.write(data)
        return target

    def close(self):
        d, self._dir = self._dir, None
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
