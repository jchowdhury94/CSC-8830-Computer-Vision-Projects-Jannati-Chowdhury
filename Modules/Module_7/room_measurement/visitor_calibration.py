"""Visitor calibration adapter (Phase 2B, Stage A).

Converts a visitor stereo-calibration model (the dict returned by ``stereo_measure.user_calibration.calibrate_stereo`` or
``stereo_measure.calibration_io.load_calibration``: K1, D1, K2, D2, R, T, E, F, baseline_mm, image_size_left/right) into the
existing ``room_measurement.calibration.StereoCalibration`` without touching that format.

Rules
-----
* K1/D1 -> left camera, K2/D2 -> right camera.  Ordering is never swapped.
* The metric scale comes ONLY from the visitor-entered physical baseline.  R, T (and E, F, the calibrated baseline) are kept as
  optional reference extrinsics in ``StereoCalibration.legacy_extrinsics`` (provenance only; the measurement code never reads them).
* Nothing here imports experiment ground truth (``room_measurement.validation``).
* Capture-mode compatibility is metadata-only.  A resolution match does not prove that intrinsics estimated from photographs hold
  for video frames (field of view, crop, stabilisation and resolution can differ), so the best possible outcome,
  ``COMPATIBLE_METADATA``, still is NOT verified calibration accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np

from . import calibration as _cal

ACCURACY_NOTE = ("A PASS or a compatible-metadata result only means the listed input checks passed. It does not establish that "
                 "the calibration or the final wall measurement is accurate.")


class Severity(str, Enum):
    """ERROR: cannot be processed safely.  WARNING: processable, but accuracy/compatibility is not established.  PASS: that one check passed."""
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str
    subject: str = ""          # which input it concerns, e.g. "calibration", "left_video_1"

    def to_dict(self):
        return dict(severity=self.severity.value, code=self.code, message=self.message, subject=self.subject)


def worst(issues) -> Severity:
    sev = {i.severity for i in issues}
    return Severity.ERROR if Severity.ERROR in sev else Severity.WARNING if Severity.WARNING in sev else Severity.PASS


@dataclass(frozen=True)
class CalibrationLimits:
    """Configurable plausibility limits (defaults are deliberately wide; they reject typos and unit mistakes, not unusual rigs)."""
    min_baseline_mm: float = 30.0
    max_baseline_mm: float = 1000.0
    min_image_side_px: int = 16
    max_image_side_px: int = 20_000
    min_focal_over_long_side: float = 0.25     # fx / max(w, h); 0.5x ultra-wide is ~0.4-0.6, 1x ~0.8-0.9, 2x ~1.7
    max_focal_over_long_side: float = 8.0
    max_focal_aspect_error: float = 0.10       # warn when |fx/fy - 1| exceeds this (error above 5x this)
    max_principal_offset_frac: float = 0.20    # warn when the principal point is further than this fraction of the size from the centre
    max_distortion_abs: float = 100.0


DEFAULT_LIMITS = CalibrationLimits()


class CompatibilityStatus(str, Enum):
    """How far the calibration can be trusted for the supplied VIDEOS, judged from metadata only.

    COMPATIBLE_METADATA     Calibration declares capture_mode == "video" for both cameras, and the calibration size equals the video
                            frame size with the same orientation.  Metadata agrees; accuracy is NOT verified.
    UNVERIFIED_CAPTURE_MODE Sizes/orientation match, but the declared capture mode is not "video" (e.g. "photo"), is unrecognised,
                            or differs between cameras.  Photo intrinsics are not assumed valid for video.
    MISSING_PROVENANCE      Sizes/orientation match, but the calibration carries no capture-mode metadata (for one or both
                            cameras), so compatibility cannot be established.  This is what current wizard output produces.
    INCOMPATIBLE_RESOLUTION Same orientation, different width/height.  Intrinsics are tied to the pixel grid and are not rescaled.
    INCOMPATIBLE_ORIENTATION Calibration and video differ in orientation (portrait vs landscape).
    """
    COMPATIBLE_METADATA = "COMPATIBLE_METADATA"
    UNVERIFIED_CAPTURE_MODE = "UNVERIFIED_CAPTURE_MODE"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    INCOMPATIBLE_RESOLUTION = "INCOMPATIBLE_RESOLUTION"
    INCOMPATIBLE_ORIENTATION = "INCOMPATIBLE_ORIENTATION"


_STATUS_SEVERITY = {
    CompatibilityStatus.COMPATIBLE_METADATA: Severity.PASS,
    CompatibilityStatus.UNVERIFIED_CAPTURE_MODE: Severity.WARNING,
    CompatibilityStatus.MISSING_PROVENANCE: Severity.WARNING,
    CompatibilityStatus.INCOMPATIBLE_RESOLUTION: Severity.ERROR,
    CompatibilityStatus.INCOMPATIBLE_ORIENTATION: Severity.ERROR,
}


@dataclass(frozen=True)
class CompatibilityResult:
    status: CompatibilityStatus
    severity: Severity
    message: str
    calibration_size_wh: tuple | None
    capture_mode_left: str | None
    capture_mode_right: str | None
    accuracy_verified: bool = False        # always False: this module never verifies accuracy
    note: str = ACCURACY_NOTE

    def to_dict(self):
        return dict(status=self.status.value, severity=self.severity.value, message=self.message,
                    calibration_size_wh=self.calibration_size_wh, capture_mode_left=self.capture_mode_left,
                    capture_mode_right=self.capture_mode_right, accuracy_verified=self.accuracy_verified, note=self.note)


@dataclass(frozen=True)
class AdaptedCalibration:
    """calibration is None when any ERROR issue exists."""
    calibration: _cal.StereoCalibration | None
    issues: tuple
    capture_mode_left: str | None = None
    capture_mode_right: str | None = None

    @property
    def ok(self) -> bool:
        return self.calibration is not None and not self.errors

    @property
    def errors(self):
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def status(self) -> Severity:
        return worst(self.issues)

    def require(self) -> _cal.StereoCalibration:
        if not self.ok:
            raise VisitorCalibrationError(self.errors)
        return self.calibration

    def to_dict(self):
        return dict(status=self.status.value, ok=self.ok, issues=[i.to_dict() for i in self.issues],
                    image_size_wh=None if self.calibration is None else list(self.calibration.image_size),
                    baseline_mm=None if self.calibration is None else self.calibration.baseline_mm,
                    capture_mode_left=self.capture_mode_left, capture_mode_right=self.capture_mode_right)


class VisitorCalibrationError(ValueError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"[{i.code}] {i.message}" for i in self.issues) or "invalid visitor calibration")


# ---------------------------------------------------------------- helpers

def _err(code, msg):
    return Issue(Severity.ERROR, code, msg, "calibration")


def _warn(code, msg):
    return Issue(Severity.WARNING, code, msg, "calibration")


def _ok(code, msg):
    return Issue(Severity.PASS, code, msg, "calibration")


def _matrix(model, key, shape, issues, label):
    if key not in model or model[key] is None:
        issues.append(_err("MISSING_FIELD", f"The calibration is missing '{key}' ({label}).")); return None
    try:
        a = np.asarray(model[key], np.float64)
    except (TypeError, ValueError):
        issues.append(_err("INVALID_VALUE", f"'{key}' ({label}) is not numeric.")); return None
    if a.size != int(np.prod(shape)) or (key[0] == "K" and a.shape != shape):
        issues.append(_err("INVALID_SHAPE", f"'{key}' ({label}) must have shape {shape}, got {a.shape}.")); return None
    a = a.reshape(shape)
    if not np.isfinite(a).all():
        issues.append(_err("NONFINITE_VALUE", f"'{key}' ({label}) contains NaN or infinite values.")); return None
    return a


def _size(model, key, issues, limits):
    v = model.get(key)
    if v is None:
        issues.append(_err("MISSING_IMAGE_SIZE", f"The calibration does not record '{key}' (width, height in pixels); it is required to check video compatibility.")); return None
    try:
        seq = list(v)
        ok = len(seq) == 2 and all(isinstance(x, Integral) and not isinstance(x, bool) or isinstance(x, Real) and float(x).is_integer() for x in seq)
    except TypeError:
        ok = False
    if not ok:
        issues.append(_err("INVALID_IMAGE_SIZE", f"'{key}' must be two whole numbers (width, height), got {v!r}.")); return None
    w, h = int(seq[0]), int(seq[1])
    if not (limits.min_image_side_px <= w <= limits.max_image_side_px and limits.min_image_side_px <= h <= limits.max_image_side_px):
        issues.append(_err("INVALID_IMAGE_SIZE", f"'{key}' = {w} x {h} is outside the supported range ({limits.min_image_side_px} to {limits.max_image_side_px} px per side).")); return None
    return (w, h)


def orientation_of(width, height) -> str:
    return "portrait" if height > width else "landscape" if width > height else "square"


def _camera_checks(K, D, size, side, limits, issues):
    n0 = len(issues)
    w, h = size
    if abs(K[1, 0]) + abs(K[2, 0]) + abs(K[2, 1]) != 0 or abs(K[2, 2] - 1) > 1e-9:
        issues.append(_err("MALFORMED_INTRINSICS", f"The {side} intrinsic matrix is not of the form [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."))
    elif K[0, 1] != 0:
        issues.append(_err("NONZERO_SKEW", f"The {side} intrinsic matrix has non-zero skew ({K[0, 1]:.3g}); the room-measurement pipeline requires zero skew."))
    fx, fy = K[0, 0], K[1, 1]
    if fx <= 0 or fy <= 0:
        issues.append(_err("INVALID_FOCAL_LENGTH", f"The {side} focal lengths must be positive (fx={fx:.4g}, fy={fy:.4g})."))
    else:
        lo, hi = limits.min_focal_over_long_side, limits.max_focal_over_long_side
        for name, f in (("fx", fx), ("fy", fy)):
            r = f / max(w, h)
            if not lo <= r <= hi:
                issues.append(_err("IMPLAUSIBLE_FOCAL_LENGTH", f"The {side} {name} = {f:.1f} px is implausible for a {w} x {h} image (expected {lo:g}-{hi:g} x the longer side)."))
        asp = abs(fx / fy - 1)
        if asp > 5 * limits.max_focal_aspect_error:
            issues.append(_err("IMPLAUSIBLE_FOCAL_LENGTH", f"The {side} fx/fy ratio {fx / fy:.3f} is far from square pixels."))
        elif asp > limits.max_focal_aspect_error:
            issues.append(_warn("UNUSUAL_FOCAL_ASPECT", f"The {side} fx/fy ratio is {fx / fy:.3f}; square pixels are expected."))
    cx, cy = K[0, 2], K[1, 2]
    if not (0 < cx < w and 0 < cy < h):
        issues.append(_err("PRINCIPAL_POINT_OUTSIDE_IMAGE", f"The {side} principal point ({cx:.0f}, {cy:.0f}) lies outside the {w} x {h} image."))
    elif abs(cx - w / 2) > limits.max_principal_offset_frac * w or abs(cy - h / 2) > limits.max_principal_offset_frac * h:
        issues.append(_warn("PRINCIPAL_POINT_OFF_CENTRE", f"The {side} principal point ({cx:.0f}, {cy:.0f}) is far from the image centre of a {w} x {h} image; check the calibration."))
    if np.abs(D).max() > limits.max_distortion_abs:
        issues.append(_err("IMPLAUSIBLE_DISTORTION", f"The {side} distortion coefficients are implausibly large."))
    if not any(i.severity is Severity.ERROR for i in issues[n0:]):
        issues.append(_ok(f"{side.upper()}_INTRINSICS_OK", f"The {side} intrinsics have valid shape, finite values, plausible focal length and a principal point inside the image."))


def _capture_mode(model, side, issues):
    """Declared capture mode from model['camera'][side]['capture_mode'], or None.  Never inferred."""
    cam = model.get("camera")
    info = cam.get(side) if isinstance(cam, Mapping) else None
    raw = info.get("capture_mode") if isinstance(info, Mapping) else None
    if raw is None:
        return None
    if not isinstance(raw, str):
        issues.append(_warn("UNRECOGNISED_CAPTURE_MODE", f"The {side} capture_mode metadata is not text and was ignored.")); return None
    return raw.strip().lower() or None


def _reference_extrinsics(model, issues):
    """R, T kept for provenance only.  Problems are warnings: these values are never used."""
    R, T = model.get("R"), model.get("T")
    if R is None or T is None:
        return None
    try:
        R = np.asarray(R, np.float64).reshape(3, 3); T = np.asarray(T, np.float64).reshape(3, 1)
    except (TypeError, ValueError):
        issues.append(_warn("REFERENCE_EXTRINSICS_DROPPED", "R/T have an unexpected shape and were not kept (they are not used for measurement)."))
        return None
    if not (np.isfinite(R).all() and np.isfinite(T).all()):
        issues.append(_warn("REFERENCE_EXTRINSICS_DROPPED", "R/T contain non-finite values and were not kept (they are not used for measurement)."))
        return None
    ref = dict(R=R, T=T, baseline_mm=float(np.linalg.norm(T)))
    for k in ("E", "F"):
        try:
            v = np.asarray(model[k], np.float64)
            if v.shape == (3, 3) and np.isfinite(v).all():
                ref[k] = v
        except (KeyError, TypeError, ValueError):
            pass
    return ref


# ---------------------------------------------------------------- public API

def adapt_visitor_calibration(model: Mapping, baseline_mm, limits: CalibrationLimits = DEFAULT_LIMITS,
                              name: str = "visitor_stereo") -> AdaptedCalibration:
    """Visitor calibration model + visitor-entered physical baseline (mm) -> AdaptedCalibration.

    Never raises for bad input; inspect ``.issues`` / ``.ok`` (or call ``.require()`` to raise VisitorCalibrationError).
    The model's own baseline_mm / T are NOT used for scale.
    """
    issues: list = []
    if not isinstance(model, Mapping):
        issues.append(_err("INVALID_MODEL", "The calibration must be a mapping with K1, D1, K2, D2 and image sizes."))
        return AdaptedCalibration(None, tuple(issues))

    K1 = _matrix(model, "K1", (3, 3), issues, "left intrinsics"); K2 = _matrix(model, "K2", (3, 3), issues, "right intrinsics")
    D1 = _matrix(model, "D1", (5,), issues, "left distortion"); D2 = _matrix(model, "D2", (5,), issues, "right distortion")
    sl = _size(model, "image_size_left", issues, limits); sr = _size(model, "image_size_right", issues, limits)
    if sl and sr:
        if sl != sr:
            issues.append(_err("CALIBRATION_SIZE_MISMATCH", f"The left ({sl[0]} x {sl[1]}) and right ({sr[0]} x {sr[1]}) calibration image sizes differ; "
                                                          "room measurement needs both cameras calibrated at the same frame size."))
        else:
            issues.append(_ok("IMAGE_SIZE_OK", f"Left and right calibration image sizes agree ({sl[0]} x {sl[1]})."))

    if K1 is not None and D1 is not None and sl:
        _camera_checks(K1, D1, sl, "left", limits, issues)
    if K2 is not None and D2 is not None and sr:
        _camera_checks(K2, D2, sr, "right", limits, issues)

    # baseline: positive, finite, within the configurable range; accepts real numbers only (not bool / text)
    b = baseline_mm
    if isinstance(b, bool) or not isinstance(b, Real):
        issues.append(_err("INVALID_BASELINE", "Enter the camera baseline as a number in millimetres."))
        b = None
    elif not math.isfinite(b) or b <= 0:
        issues.append(_err("INVALID_BASELINE", f"The baseline must be a positive, finite number of millimetres (got {b})."))
        b = None
    elif not limits.min_baseline_mm <= b <= limits.max_baseline_mm:
        issues.append(_err("BASELINE_OUT_OF_RANGE", f"The baseline {b:g} mm is outside the accepted range of {limits.min_baseline_mm:g}-{limits.max_baseline_mm:g} mm. "
                                                    "Measure the distance between the two camera lenses in millimetres."))
        b = None
    else:
        issues.append(_ok("BASELINE_OK", f"Baseline {float(b):g} mm is within {limits.min_baseline_mm:g}-{limits.max_baseline_mm:g} mm."))

    mode_l, mode_r = _capture_mode(model, "left", issues), _capture_mode(model, "right", issues)
    ref = _reference_extrinsics(model, issues)

    if any(i.severity is Severity.ERROR for i in issues):
        return AdaptedCalibration(None, tuple(issues), mode_l, mode_r)

    prov = dict(source="visitor_calibration_model", adapter="room_measurement.visitor_calibration",
                capture_mode_left=mode_l, capture_mode_right=mode_r, accuracy_verified=False,
                baseline_source="visitor_entered_physical_baseline",
                note="Intrinsics come from the visitor calibration; capture-mode compatibility with the videos is metadata-only.")
    cal = _cal.StereoCalibration(name=name, K_left=K1, D_left=D1.reshape(-1), K_right=K2, D_right=D2.reshape(-1), image_size=sl,
                                 baseline_mm=float(b), provenance=prov, legacy_extrinsics=ref)
    try:
        cal = _cal.validate(cal)               # the existing validator stays the final authority
    except ValueError as e:
        issues.append(_err("EXISTING_VALIDATION_FAILED", str(e)))
        return AdaptedCalibration(None, tuple(issues), mode_l, mode_r)
    return AdaptedCalibration(cal, tuple(issues), mode_l, mode_r)


def assess_capture_compatibility(adapted: AdaptedCalibration, left_video_sizes: Sequence, right_video_sizes: Sequence) -> CompatibilityResult:
    """Metadata-only compatibility of the calibration with the decoded frame sizes (width, height) of the left / right videos.

    Precedence: orientation > resolution > capture-mode provenance.  See CompatibilityStatus for exact meanings.
    """
    if not adapted.ok:
        raise ValueError("assess_capture_compatibility needs a valid AdaptedCalibration")
    cw, ch = adapted.calibration.image_size
    cal_orient = orientation_of(cw, ch)
    pairs = [("left", s) for s in left_video_sizes] + [("right", s) for s in right_video_sizes]
    mode_l, mode_r = adapted.capture_mode_left, adapted.capture_mode_right

    def result(status, msg):
        return CompatibilityResult(status, _STATUS_SEVERITY[status], msg, (cw, ch), mode_l, mode_r)

    bad_orient = [(side, s) for side, s in pairs if orientation_of(*s) != cal_orient]
    if bad_orient:
        side, s = bad_orient[0]
        return result(CompatibilityStatus.INCOMPATIBLE_ORIENTATION,
                      f"The calibration is {cal_orient} ({cw} x {ch}) but a {side} video is {orientation_of(*s)} ({s[0]} x {s[1]}). "
                      "Recalibrate in the orientation you film in, or re-film in the calibrated orientation.")
    bad_res = [(side, s) for side, s in pairs if tuple(s) != (cw, ch)]
    if bad_res:
        side, s = bad_res[0]
        return result(CompatibilityStatus.INCOMPATIBLE_RESOLUTION,
                      f"The calibration is {cw} x {ch} but a {side} video is {s[0]} x {s[1]}. Intrinsics are not rescaled; "
                      "film at the calibrated resolution or recalibrate at the video resolution.")
    if mode_l is None or mode_r is None:
        return result(CompatibilityStatus.MISSING_PROVENANCE,
                      f"Frame sizes match ({cw} x {ch}, {cal_orient}), but the calibration records no capture-mode metadata for "
                      f"{'either camera' if mode_l is None and mode_r is None else 'the left camera' if mode_l is None else 'the right camera'}. "
                      "Whether it was made from photos or video is unknown, so its validity for video measurement is unverified.")
    if mode_l != "video" or mode_r != "video":
        return result(CompatibilityStatus.UNVERIFIED_CAPTURE_MODE,
                      f"Frame sizes match, but the calibration capture mode is left='{mode_l}', right='{mode_r}', not 'video'. "
                      "Photo intrinsics can differ from video (field of view, crop, stabilisation), so they are not assumed valid.")
    return result(CompatibilityStatus.COMPATIBLE_METADATA,
                  f"Metadata is consistent: video-mode calibration at {cw} x {cal_orient} matches the video frame size. "
                  "This is not a verification of calibration accuracy.")
