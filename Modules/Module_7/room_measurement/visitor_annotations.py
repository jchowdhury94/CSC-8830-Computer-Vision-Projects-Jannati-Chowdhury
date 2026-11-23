"""Visitor wall-boundary annotation backend (Phase 2B, Stage D).

Turns visitor-provided two-point wall-boundary annotations (image coordinates) into validated line information in the exact
form the unchanged Phase 1 measurement code consumes: ``{line name: fitted-line dict}`` in UNDISTORTED full-resolution pixels
(the dicts produced by ``lines.fit_line``).  There is no user interface here.

What Phase 1 needs (``measure.measure_wall``, ``top_variant=0``)
----------------------------------------------------------------
Per placement: the left-camera wall-corner line, the right-camera wall-corner line and the right-camera ceiling/front-wall line
(``AL_corner PL_corner PL_ceilF`` / ``AR_corner PR_corner PR_ceilF``).  The ceiling/side-wall lines (``PL_ceilS``, ``PR_ceilS``) are only
used by ``top_variant=1`` and are OPTIONAL here.  (Stage C's ``REQUIRED_LINES`` still lists all eight; ``complete_all_phase1_lines``
reports that stricter condition.)  No new boundary types are introduced; ``boundary_type`` is the Phase 1 suffix (corner/ceilF/ceilS).
The Air (left) camera cannot see the ceiling, so ceilF/ceilS exist for the right camera only.

Coordinate spaces (never silently interchanged)
-----------------------------------------------
``preview_display``  : what the visitor clicked on -- a canvas showing the full frame uniformly resized, possibly letter-boxed.
                       Continuous, edge-aligned browser coordinates (pixel i covers [i, i+1)), origin = canvas top-left.
``distorted_full``   : original lens-distorted frame pixels, OpenCV convention (pixel centres at integers), full resolution.
``undistorted_full`` : ``cv2.undistort(frame, K, D, None, K)`` pixels at full resolution -- the space ``lines.fit_lines`` works in.
Preview -> distorted_full uses the explicit ``DisplayMapping`` (per-axis scale, offset/padding, half-pixel convention).  distorted ->
undistorted uses the existing calibration (``geometry.undistort_pixels``).  Intrinsics are never rescaled: the frame size must equal
the calibration size, otherwise the annotation FAILS.

Refinement (reuses Phase 1)
---------------------------
Two clicks -> initial line.  Gradient polarity is taken from the visitor (optional) or inferred from the image.  Pass 1 searches
``lines.edge_points`` within +-``search_half_window_px`` of the click line over the annotated (visible) extent only; pass 2 repeats the
search in a narrow window around the pass-1 fit (the Phase 1 "track2" idea); ``lines.fit_line`` (Tukey IRLS) fits.  A search window
that leaves the image is clipped explicitly (``lines.edge_points`` itself does not check bounds).  A failed refinement is a FAIL; the
unrefined click line is NEVER passed downstream (a click is accurate to several px, Phase 1 geometry works at sub-pixel level).

Status model
------------
PASS / WARNING / FAIL per check; NOT_EVALUATED where a check could not be run (e.g. stereo consistency without a pose).  Downstream
use is permitted only if no required annotation FAILS and all required boundaries are present.  A good fit RMS does NOT show the line
is the intended physical boundary; competing-edge, polarity, displacement and orientation checks reduce, but cannot remove, that risk.

Nothing here imports ground truth, opens a video or writes a file.  Thresholds are fixed a priori (``AnnotationLimits``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping, Sequence

import cv2
import numpy as np

from . import geometry as G, lines as L, measure as M
from .pipeline import to_jsonable

SCHEMA = "csc8830.module7.visitor_wall_annotations.v1"
COORDINATE_SPACES = ("preview_display", "distorted_full", "undistorted_full")
BOUNDARY_TYPES = ("corner", "ceilF", "ceilS")
PLACEMENTS = ("placement_1", "placement_2")
CAMERAS = ("left", "right")
VIEW_KEY = {("placement_1", "left"): "AL", ("placement_1", "right"): "PL", ("placement_2", "left"): "AR", ("placement_2", "right"): "PR"}
VIEW_PLACEMENT = {v: k for k, v in VIEW_KEY.items()}
CAMERA_BOUNDARIES = {"left": ("corner",), "right": ("corner", "ceilF", "ceilS")}      # Air cannot see the ceiling
REQUIRED_FIELDS = ("set_index", "placement", "camera", "frame_index", "boundary_type", "image", "coordinate_space", "points")
# Phase 1 line names, taken from measure.VIEW_LINES (corner Air, corner Pro, ceilF Pro, ceilS Pro per view)
REQUIRED_LINES = tuple(n for v in M.VIEW_LINES.values() for n in v[:3])
OPTIONAL_LINES = tuple(v[3] for v in M.VIEW_LINES.values())
ALL_PHASE1_LINES = tuple(n for v in M.VIEW_LINES.values() for n in v)
STEREO_NOTE = ("Stereo consistency only tests compatibility with one common 3D line under the supplied pose; it does not show that "
               "both annotations mark the same physical wall edge.")


class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


def worst(statuses) -> Status:
    s = set(statuses)
    return Status.FAIL if Status.FAIL in s else Status.WARNING if Status.WARNING in s else Status.PASS if Status.PASS in s else Status.NOT_EVALUATED


class Code:
    """Explicit failure / warning codes."""
    SCHEMA = "UNSUPPORTED_SCHEMA"; NOT_A_MAPPING = "ANNOTATION_NOT_A_MAPPING"; MISSING_FIELD = "MISSING_FIELD"
    BAD_VALUE = "BAD_FIELD_VALUE"; BAD_POINTS = "MALFORMED_POINTS"; NON_FINITE = "NON_FINITE_COORDINATE"
    BAD_SPACE = "UNSUPPORTED_COORDINATE_SPACE"; BAD_BOUNDARY = "UNSUPPORTED_BOUNDARY_TYPE"; BAD_CAMERA_BOUNDARY = "BOUNDARY_NOT_VISIBLE_TO_CAMERA"
    LINE_NAME = "LINE_NAME_MISMATCH"; WRONG_SET = "SET_INDEX_MISMATCH"; WRONG_FRAME = "FRAME_MISMATCH"; FINGERPRINT = "FRAME_FINGERPRINT_MISMATCH"
    NO_FRAME = "FRAME_NOT_PROVIDED"; SIZE = "IMAGE_SIZE_MISMATCH"; CAL_SIZE = "IMAGE_SIZE_CALIBRATION_MISMATCH"
    DISPLAY = "DISPLAY_MAPPING_INVALID"; ASPECT = "DISPLAY_ASPECT_MISMATCH"; PADDING = "POINT_IN_DISPLAY_PADDING"
    OUT_OF_BOUNDS = "POINT_OUT_OF_BOUNDS"; UNDIST_OUT = "UNDISTORTED_POINT_OUT_OF_IMAGE"
    DUPLICATE_ENDPOINTS = "DUPLICATE_ENDPOINTS"; TOO_SHORT = "LINE_TOO_SHORT"; ORIENTATION = "UNREASONABLE_ORIENTATION"
    WINDOW = "SEARCH_WINDOW_OUT_OF_IMAGE"; SPAN_CLIPPED = "SEARCH_SPAN_CLIPPED"; REFINE_FAILED = "REFINEMENT_FAILED"
    FEW_POINTS = "TOO_FEW_EDGE_POINTS"; LOW_COVERAGE = "LOW_EDGE_COVERAGE"; LOW_INLIERS = "LOW_INLIER_FRACTION"; HIGH_RMS = "HIGH_FIT_RMS"
    WEAK_EDGE = "WEAK_EDGE"; AT_LIMIT = "REFINEMENT_AT_WINDOW_LIMIT"; BIG_SHIFT = "LARGE_REFINEMENT_DISPLACEMENT"
    ANGLE_CHANGE = "LARGE_ANGLE_CHANGE"; RIM = "EDGE_POINTS_AT_WINDOW_RIM"; COMPETING = "COMPETING_EDGE_NEARBY"
    POLARITY = "AMBIGUOUS_EDGE_POLARITY"; PASS2 = "SECOND_PASS_FAILED"
    DUPLICATE_ANNOTATION = "DUPLICATE_ANNOTATION"; MISSING_REQUIRED = "MISSING_REQUIRED_BOUNDARY"; REQUIRED_INVALID = "REQUIRED_BOUNDARY_INVALID"
    OPTIONAL_REJECTED = "OPTIONAL_BOUNDARY_REJECTED"; UNKNOWN_LINE = "UNEXPECTED_ANNOTATION"
    STEREO_PARALLEL = "STEREO_PLANES_NEAR_PARALLEL"; STEREO_DEPTH = "STEREO_INCONSISTENT_DEPTH"; STEREO_OVERLAP = "STEREO_SEGMENTS_DO_NOT_OVERLAP"
    STEREO_ONE_CAM = "STEREO_ONLY_ONE_CAMERA"; STEREO_NO_POSE = "STEREO_POSE_NOT_AVAILABLE"
    CORNER_PARALLEL = "CORNER_CEILING_NEAR_PARALLEL"; EXTRAPOLATION = "LONG_EXTRAPOLATION_TO_HIDDEN_CORNER"; NO_CORNER = "CORNER_NOT_RECONSTRUCTIBLE"


@dataclass(frozen=True)
class Issue:
    status: Status
    code: str
    message: str
    subject: str = ""

    def to_dict(self):
        return dict(status=self.status.value, code=self.code, message=self.message, subject=self.subject)


def _fail(code, msg, subject=""): return Issue(Status.FAIL, code, msg, subject)
def _warn(code, msg, subject=""): return Issue(Status.WARNING, code, msg, subject)


@dataclass(frozen=True)
class AnnotationLimits:
    """Configurable thresholds.  Fixed a priori from generic image-processing considerations; none was tuned on the wall dimensions
    (and none on any measurement result)."""
    min_length_px: float = 100.0                 # full-resolution pixels between the two clicks
    min_endpoint_separation_px: float = 1.0
    bounds_tolerance_px: float = 0.5             # preview clicks may land half a pixel outside after the half-pixel convention
    aspect_tolerance: float = 0.01               # relative difference of the two display scales
    search_half_window_px: int = 20
    second_pass_half_window_px: int = 6
    search_step_px: int = 6                      # Phase 1 edge_points default
    min_edge_points_fail: int = 12
    min_edge_points_warn: int = 18
    min_coverage_fail: float = 0.50              # edge candidates / search positions
    min_coverage_warn: float = 0.75
    min_inlier_fraction_fail: float = 0.60
    min_inlier_fraction_warn: float = 0.80
    max_rms_px_warn: float = 1.5
    max_rms_px_fail: float = 3.0
    min_edge_strength_fail: float = 1.0          # gray levels / px in the 2-px-blurred image (Phase 1 gradient floor)
    min_edge_strength_warn: float = 2.0
    shift_warn_frac_of_window: float = 0.50      # max |refined - click line| relative to the search half-window
    shift_fail_frac_of_window: float = 0.90      # edge is probably outside the window / a different edge was grabbed
    angle_change_warn_deg: float = 4.0
    angle_change_fail_deg: float = 10.0
    rim_px: float = 1.5
    rim_fraction_warn: float = 0.25
    competing_window_px: int = 60                # look for another same-polarity edge this far from the refined line
    competing_exclusion_px: int = 6
    competing_ratio: float = 0.60                # competitor gradient >= ratio * main gradient -> ambiguous
    polarity_ratio: float = 0.70
    max_off_vertical_deg_corner: float = 20.0    # wall corners are near-vertical in a roughly upright camera
    max_off_horizontal_deg_ceilF: float = 40.0
    tilt_range_deg_ceilS: tuple = (10.0, 80.0)   # oblique in perspective
    min_corner_ceiling_angle_deg: float = 20.0   # corner/ceilF must intersect at a well-conditioned angle
    max_extrapolation_ratio: float = 4.0         # distance annotated edge -> hidden corner / annotated length
    min_stereo_plane_angle_deg: float = 3.0
    require_ceiling_side: bool = False           # True = demand all eight Phase 1 lines (Stage C's REQUIRED_LINES)
    allow_warnings_downstream: bool = True


DEFAULT_LIMITS = AnnotationLimits()


class AnnotationError(RuntimeError):
    pass


# ------------------------------------------------------------------ coordinate transformations
@dataclass(frozen=True)
class DisplayMapping:
    """Where the full frame is drawn inside the clicked canvas.  The image occupies [image_x, image_x+image_w] x [image_y, image_y+image_h]
    in canvas coordinates; everything else is padding.  Scales are independent per axis and must agree (no silent stretching)."""
    full_width: int
    full_height: int
    canvas_width: float
    canvas_height: float
    image_x: float
    image_y: float
    image_w: float
    image_h: float

    @property
    def scale_x(self): return self.image_w / self.full_width
    @property
    def scale_y(self): return self.image_h / self.full_height

    @classmethod
    def letterbox(cls, full_wh, canvas_wh):
        """Uniform 'contain' fit, centred (what a browser does for object-fit: contain)."""
        fw, fh = full_wh; cw, ch = canvas_wh; s = min(cw / fw, ch / fh); w, h = fw * s, fh * s
        return cls(int(fw), int(fh), float(cw), float(ch), (cw - w) / 2, (ch - h) / 2, w, h)

    @classmethod
    def from_dict(cls, d: Mapping, full_wh):
        need = ("canvas_width", "canvas_height")
        if not isinstance(d, Mapping) or any(k not in d for k in need): raise ValueError(f"display needs {need}")
        vals = [d.get(k) for k in ("canvas_width", "canvas_height", "image_x", "image_y", "image_w", "image_h")]
        cw, ch = float(d["canvas_width"]), float(d["canvas_height"])
        if all(v is None for v in vals[2:]): return cls.letterbox(full_wh, (cw, ch))
        if any(v is None for v in vals[2:]): raise ValueError("display must give all of image_x, image_y, image_w, image_h, or none (letter-box assumed)")
        return cls(int(full_wh[0]), int(full_wh[1]), cw, ch, *(float(v) for v in vals[2:]))

    def to_dict(self):
        return dict(full_width=self.full_width, full_height=self.full_height, canvas_width=self.canvas_width, canvas_height=self.canvas_height,
                    image_x=self.image_x, image_y=self.image_y, image_w=self.image_w, image_h=self.image_h, scale_x=self.scale_x, scale_y=self.scale_y,
                    convention="edge-aligned continuous display coordinates; full = (display - image_xy)/scale - 0.5 (OpenCV pixel centres)")

    def validate(self, limits=DEFAULT_LIMITS) -> list:
        vals = (self.canvas_width, self.canvas_height, self.image_x, self.image_y, self.image_w, self.image_h)
        if not all(math.isfinite(v) for v in vals) or self.image_w <= 0 or self.image_h <= 0 or self.full_width <= 0 or self.full_height <= 0:
            return [_fail(Code.DISPLAY, "display mapping has non-finite or non-positive sizes")]
        out = []
        if self.image_x < -1e-6 or self.image_y < -1e-6 or self.image_x + self.image_w > self.canvas_width + 1e-6 or self.image_y + self.image_h > self.canvas_height + 1e-6:
            out.append(_fail(Code.DISPLAY, "rendered image rectangle lies outside the canvas"))
        if abs(self.scale_x - self.scale_y) / max(self.scale_x, self.scale_y) > limits.aspect_tolerance:
            out.append(_fail(Code.ASPECT, f"display scales differ (x {self.scale_x:.5f}, y {self.scale_y:.5f}): the preview is stretched; refusing to guess"))
        return out

    def inside_image_rect(self, pts):
        p = np.asarray(pts, float); return ((p[:, 0] >= self.image_x) & (p[:, 0] <= self.image_x + self.image_w) & (p[:, 1] >= self.image_y) & (p[:, 1] <= self.image_y + self.image_h))

    def to_full(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return np.c_[(p[:, 0] - self.image_x) / self.scale_x - 0.5, (p[:, 1] - self.image_y) / self.scale_y - 0.5]

    def from_full(self, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return np.c_[(p[:, 0] + 0.5) * self.scale_x + self.image_x, (p[:, 1] + 0.5) * self.scale_y + self.image_y]


def distort_pixels(px_undist, K, D):
    """Inverse of ``geometry.undistort_pixels`` (same model as ``lines.retarget_lines``)."""
    P = np.asarray(px_undist, np.float64).reshape(-1, 2); n = (np.linalg.inv(K) @ np.c_[P, np.ones(len(P))].T).T
    return cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, np.asarray(D, np.float64)) [0].reshape(-1, 2)


# ------------------------------------------------------------------ small helpers
def _finite_number(x):
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x))


def line_name(placement, camera, boundary_type): return f"{VIEW_KEY[(placement, camera)]}_{boundary_type}"


def line_from_points(p1, p2):
    """Phase 1 line parameterisation through two points: steeper-than-45-degree -> 'v' (x = a*y + b), else 'h' (y = a*x + b)."""
    (x1, y1), (x2, y2) = p1, p2
    if abs(y2 - y1) >= abs(x2 - x1):
        a = (x2 - x1) / (y2 - y1); return dict(kind="v", a=float(a), b=float(x1 - a * y1), span=(float(min(y1, y2)), float(max(y1, y2))))
    a = (y2 - y1) / (x2 - x1); return dict(kind="h", a=float(a), b=float(y1 - a * x1), span=(float(min(x1, x2)), float(max(x1, x2))))


def _w(line, u): return line["a"] * np.asarray(u, float) + line["b"]


def _points_on(line, u):
    u = np.asarray(u, float); w = _w(line, u); return np.c_[w, u] if line["kind"] == "v" else np.c_[u, w]


def segment_endpoints(line):
    return _points_on(line, line["span"])


def line_angle_deg(line):
    """Direction angle in (-90, 90], degrees, image axes (x right, y down)."""
    dx, dy = (line["a"], 1.0) if line["kind"] == "v" else (1.0, line["a"]); ang = math.degrees(math.atan2(dy, dx))
    return ang - 180 if ang > 90 else ang + 180 if ang <= -90 else ang


def angle_between_deg(l1, l2):
    d = abs(line_angle_deg(l1) - line_angle_deg(l2)) % 180; return min(d, 180 - d)


def _perp_distance(line, pts):
    h = G.line_to_h(line); p = np.asarray(pts, float); return np.abs(p @ h[:2] + h[2]) / math.hypot(h[0], h[1])


def line_difference(a, b, span=None, n=25):
    """Position (perpendicular px) and angular difference of two Phase 1 lines over a common along-axis span.  Used to compare a refined
    visitor line with an existing Phase 1 line.  kind mismatch -> compared in the kind of ``a``'s span via sampled points."""
    lo, hi = span if span is not None else (max(a["span"][0], b["span"][0]), min(a["span"][1], b["span"][1]))
    if a["kind"] != b["kind"] or not hi > lo: return dict(comparable=False, reason="lines of different kind or no common span")
    d = _perp_distance(b, _points_on(a, np.linspace(lo, hi, n)))
    return dict(comparable=True, max_px=float(d.max()), mean_px=float(d.mean()), angle_deg=float(angle_between_deg(a, b)), span=(float(lo), float(hi)))


def _gray(frame):
    f = np.asarray(frame)
    if f.ndim == 3: f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return f


class CameraContext:
    """One camera frame of a set: the grayscale frame plus lazily computed undistorted-blurred image (shared by all lines of that camera)."""
    def __init__(self, gray, K, D):
        self.gray, self.K, self.D = _gray(gray), np.asarray(K, float), np.asarray(D, float); self._g = None

    @property
    def g(self):
        if self._g is None: self._g = L.undistorted_blur(self.gray, self.K, self.D)
        return self._g

    @property
    def size_wh(self): return (int(self.gray.shape[1]), int(self.gray.shape[0]))


def build_contexts(frames: Mapping, cal) -> dict:
    """frames: {AL,PL,AR,PR: grayscale full-res frame} (any subset)."""
    return {k: CameraContext(f, *((cal.K_left, cal.D_left) if k[0] == "A" else (cal.K_right, cal.D_right))) for k, f in frames.items()}


# ------------------------------------------------------------------ parsing
@dataclass
class ParsedAnnotation:
    raw: dict
    annotation_id: str
    set_index: int
    placement: str
    camera: str
    frame_index: int
    boundary_type: str
    image_wh: tuple
    coordinate_space: str
    points: np.ndarray            # 2x2, in coordinate_space
    display: DisplayMapping | None
    edge_polarity: int | None
    line_name_claimed: str | None
    fingerprint: str | None
    source: str

    @property
    def view_key(self): return VIEW_KEY[(self.placement, self.camera)]
    @property
    def line_name(self): return line_name(self.placement, self.camera, self.boundary_type)


def parse_annotation(raw) -> tuple:
    """-> (ParsedAnnotation | None, issues).  Pure validation of the structure; no image access."""
    if not isinstance(raw, Mapping): return None, [_fail(Code.NOT_A_MAPPING, "an annotation must be a mapping")]
    issues = []; missing = [k for k in REQUIRED_FIELDS if raw.get(k) is None]
    if missing: return None, [_fail(Code.MISSING_FIELD, f"missing required field(s): {', '.join(missing)}")]
    def bad(msg, code=Code.BAD_VALUE): issues.append(_fail(code, msg))
    si, fi = raw["set_index"], raw["frame_index"]
    for nm, v in (("set_index", si), ("frame_index", fi)):
        if not (isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v >= 0): bad(f"{nm} must be a non-negative integer, got {v!r}")
    if raw["placement"] not in PLACEMENTS: bad(f"placement must be one of {PLACEMENTS}, got {raw['placement']!r}")
    if raw["camera"] not in CAMERAS: bad(f"camera must be 'left' or 'right', got {raw['camera']!r}")
    if raw["boundary_type"] not in BOUNDARY_TYPES: bad(f"boundary_type must be one of {BOUNDARY_TYPES}, got {raw['boundary_type']!r}", Code.BAD_BOUNDARY)
    if raw["coordinate_space"] not in COORDINATE_SPACES: bad(f"coordinate_space must be one of {COORDINATE_SPACES}, got {raw['coordinate_space']!r}", Code.BAD_SPACE)
    im = raw["image"]
    if not (isinstance(im, Mapping) and all(_finite_number(im.get(k)) and im.get(k) > 0 and float(im.get(k)).is_integer() for k in ("width", "height"))): bad("image must give positive integer width and height")
    try:
        pts = np.array(raw["points"], dtype=float)
        if pts.shape != (2, 2): raise ValueError
    except (ValueError, TypeError): pts = None; bad("points must be [[x1, y1], [x2, y2]]", Code.BAD_POINTS)
    if pts is not None and not np.isfinite(pts).all(): bad("point coordinates must be finite numbers", Code.NON_FINITE)
    pol = raw.get("edge_polarity")
    if pol is not None and pol not in (-1, 1): bad("edge_polarity must be +1, -1 or omitted")
    if issues: return None, issues
    wh = (int(im["width"]), int(im["height"])); display = None
    if raw["coordinate_space"] == "preview_display":
        if raw.get("display") is None: return None, [_fail(Code.MISSING_FIELD, "preview_display coordinates require a 'display' mapping (canvas size, image placement)")]
        try: display = DisplayMapping.from_dict(raw["display"], wh)
        except (ValueError, TypeError) as e: return None, [_fail(Code.DISPLAY, str(e))]
    return ParsedAnnotation(raw=dict(raw), annotation_id=str(raw.get("annotation_id", "")), set_index=int(si), placement=raw["placement"], camera=raw["camera"], frame_index=int(fi),
                            boundary_type=raw["boundary_type"], image_wh=wh, coordinate_space=raw["coordinate_space"], points=pts, display=display, edge_polarity=pol,
                            line_name_claimed=raw.get("line_name"), fingerprint=raw.get("frame_fingerprint"), source=str(raw.get("source", "visitor"))), []


def build_annotation(*, set_index, placement, camera, frame_index, boundary_type, image_wh, coordinate_space, points, display=None, edge_polarity=None,
                     annotation_id="", source="visitor", frame_fingerprint=None) -> dict:
    """Construct a schema-conformant annotation dict (what the future UI will send)."""
    d = dict(annotation_id=annotation_id, set_index=set_index, placement=placement, camera=camera, frame_index=frame_index, boundary_type=boundary_type,
             line_name=line_name(placement, camera, boundary_type), image=dict(width=int(image_wh[0]), height=int(image_wh[1])), coordinate_space=coordinate_space,
             points=[[float(x), float(y)] for x, y in points], source=source)
    if display is not None: d["display"] = display.to_dict() if isinstance(display, DisplayMapping) else dict(display)
    if edge_polarity is not None: d["edge_polarity"] = edge_polarity
    if frame_fingerprint is not None: d["frame_fingerprint"] = frame_fingerprint
    return d


def frame_fingerprint(gray) -> str:
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(_gray(gray)).tobytes()).hexdigest()


# ------------------------------------------------------------------ edge refinement (reuses lines.edge_points / lines.fit_line)
def _search(g, kind, span, line, half, sign, step):
    """lines.edge_points around ``line`` with explicit image-bounds clipping.  Returns (points, n_positions, clipped)."""
    n_along, n_across = (g.shape[0], g.shape[1]) if kind == "v" else (g.shape[1], g.shape[0]); avg = 9
    pos = np.arange(int(math.ceil(span[0])), int(math.floor(span[1])) + 1, step); c = _w(line, pos)
    ok = (pos - avg // 2 >= 0) & (pos + avg // 2 + 1 <= n_along) & (c - half - 2 >= 0) & (c + half + 3 <= n_across)
    if not ok.any(): return np.empty((0, 2)), 0, True
    p = pos[ok]; P = L.edge_points(g, kind, (int(p[0]), int(p[-1]) + 1), lambda s: line["a"] * s + line["b"], half, sign, avg=avg, step=step)
    return (P if len(P) else np.empty((0, 2))), int(len(p)), bool(not ok.all())


def _profile(g, kind, line, span, offsets, step=6):
    """Mean intensity profile across the line (sampled along its annotated extent)."""
    u = np.arange(span[0], span[1] + 1e-6, step, dtype=np.float32); offs = np.asarray(offsets, np.float32); w = (line["a"] * u + line["b"]).astype(np.float32)[:, None] + offs[None, :]
    uu = np.repeat(u[:, None], len(offs), 1); mx, my = (w, uu) if kind == "v" else (uu, w)
    return cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).mean(0)


def infer_polarity(g, line, half, limits):
    """Gradient sign across the edge from the strongest gradient near the click line.  -> (sign, ambiguous, info)"""
    offs = np.arange(-half, half + 1); d = np.gradient(_profile(g, line["kind"], line, line["span"], offs, limits.search_step_px))
    i = int(np.argmax(np.abs(d))); sign = 1 if d[i] >= 0 else -1; opp = float(np.max(-sign * d)); main = float(abs(d[i]))
    return sign, bool(main > 0 and opp >= limits.polarity_ratio * main), dict(peak_gradient=main, opposite_peak_ratio=opp / main if main > 0 else float("nan"))


def _edge_context(g, line, sign, limits):
    """Strength of the edge under the refined line and the strongest same-polarity competitor within +-competing_window_px."""
    ctx = limits.competing_window_px; offs = np.arange(-ctx, ctx + 1); d = sign * np.gradient(_profile(g, line["kind"], line, line["span"], offs, limits.search_step_px))
    core = np.abs(offs) <= 3; strength = float(d[core].max())
    cand = [(float(d[i]), int(offs[i])) for i in range(1, len(d) - 1) if abs(offs[i]) > limits.competing_exclusion_px and d[i] > d[i - 1] and d[i] >= d[i + 1] and d[i] > 0]
    best = max(cand) if cand else (0.0, 0)
    return dict(edge_strength=strength, competitor_gradient=best[0], competitor_offset_px=best[1], competitor_ratio=float(best[0] / strength) if strength > 0 else float("inf"))


def refine_line(g, initial, limits=DEFAULT_LIMITS, polarity=None):
    """Two-pass Phase 1 edge search + robust fit starting from ``initial`` (a ``line_from_points`` dict).
    Returns (refined fit dict | None, diagnostics dict, issues).  Never substitutes the initial line for a failed fit."""
    issues = []; kind = initial["kind"]; half = int(limits.search_half_window_px)
    diag = dict(search_half_window_px=half, polarity=None, polarity_source=None)
    if polarity is None:
        sign, amb, info = infer_polarity(g, initial, half, limits); diag.update(polarity=sign, polarity_source="inferred", **{f"polarity_{k}": v for k, v in info.items()})
        if amb: issues.append(_warn(Code.POLARITY, f"edges of both polarities of similar strength near the clicked line (ratio {info['opposite_peak_ratio']:.2f}); the chosen edge may not be the intended one"))
    else: sign = int(polarity); diag.update(polarity=sign, polarity_source="visitor")
    P, npos, clipped = _search(g, kind, initial["span"], initial, half, sign, limits.search_step_px); diag["n_search_positions"] = npos
    if clipped: issues.append(_warn(Code.SPAN_CLIPPED, "part of the annotated extent was dropped because its search window would leave the image"))
    if npos == 0: issues.append(_fail(Code.WINDOW, "the whole search window lies outside the image")); return None, diag, issues
    try: f1 = L.fit_line(P, kind)
    except ValueError as e: issues.append(_fail(Code.REFINE_FAILED, f"pass 1: {e}")); diag["n_edge_candidates"] = int(len(P)); return None, diag, issues
    fit, used, center = f1, "pass1", initial
    P2, npos2, _ = _search(g, kind, initial["span"], f1, int(limits.second_pass_half_window_px), sign, limits.search_step_px)
    try: f2 = L.fit_line(P2, kind); fit, used, npos, center = f2, "pass2", npos2, f1
    except ValueError as e: issues.append(_warn(Code.PASS2, f"narrow second pass failed ({e}); pass-1 fit kept"))
    center = initial if used == "pass1" else f1
    diag.update(search_center=dict(kind=kind, a=center["a"], b=center["b"]), pass_used=used, n_search_positions=int(npos), pass1_n=int(f1["n"]), pass1_rms_px=f1["rms"])
    return fit, diag, issues


# ------------------------------------------------------------------ per-annotation result
@dataclass
class AnnotationResult:
    annotation_id: str
    line_name: str | None
    status: Status
    issues: list
    metadata: dict
    points_original: list | None
    transforms: dict
    initial_line: dict | None
    refined_line: dict | None
    diagnostics: dict
    downstream_line: dict | None       # Phase 1 fitted-line dict (with 'pts'); None unless downstream use is permitted
    downstream_permitted: bool
    runtime_s: float
    stereo_consistency: Status = Status.NOT_EVALUATED

    @property
    def failure_codes(self): return [i.code for i in self.issues if i.status is Status.FAIL]
    @property
    def warning_codes(self): return [i.code for i in self.issues if i.status is Status.WARNING]

    def to_dict(self):
        return to_jsonable(dict(annotation_id=self.annotation_id, line_name=self.line_name, status=self.status.value, downstream_permitted=self.downstream_permitted,
                                failure_codes=self.failure_codes, warning_codes=self.warning_codes, issues=[i.to_dict() for i in self.issues], metadata=self.metadata,
                                points_original=self.points_original, transforms=self.transforms, initial_line=self.initial_line, refined_line=self.refined_line,
                                diagnostics=self.diagnostics, stereo_consistency=self.stereo_consistency.value, runtime_s=round(self.runtime_s, 4)))


def _public_line(l):
    if l is None: return None
    return dict(kind=l["kind"], a=l["a"], b=l["b"], span=tuple(l["span"]), angle_deg=line_angle_deg(l), **{k: l[k] for k in ("n", "N", "rms") if k in l})


def _orientation_issues(init, boundary_type, limits):
    ang = abs(line_angle_deg(init)); from_h = ang; from_v = 90 - ang
    if boundary_type == "corner" and from_v > limits.max_off_vertical_deg_corner: return [_fail(Code.ORIENTATION, f"a wall corner should be near-vertical; clicked line is {from_v:.1f} deg off vertical (limit {limits.max_off_vertical_deg_corner})")]
    if boundary_type == "ceilF" and from_h > limits.max_off_horizontal_deg_ceilF: return [_fail(Code.ORIENTATION, f"a ceiling/front-wall edge should be near-horizontal; clicked line is {from_h:.1f} deg off horizontal (limit {limits.max_off_horizontal_deg_ceilF})")]
    lo, hi = limits.tilt_range_deg_ceilS
    if boundary_type == "ceilS" and not lo <= from_h <= hi: return [_fail(Code.ORIENTATION, f"a ceiling/side-wall edge should be oblique; clicked line is {from_h:.1f} deg from horizontal (allowed {lo}-{hi})")]
    return []


def _finish(parsed, raw, status_issues, t0, **kw):
    status = worst([i.status for i in status_issues] or [Status.PASS]); meta = dict(set_index=None, placement=None, camera=None, frame_index=None, boundary_type=None, view_key=None, image_wh=None,
                                                                              coordinate_space=None, source=None)
    if parsed is not None:
        meta.update(set_index=parsed.set_index, placement=parsed.placement, camera=parsed.camera, frame_index=parsed.frame_index, boundary_type=parsed.boundary_type, view_key=parsed.view_key,
                    image_wh=list(parsed.image_wh), coordinate_space=parsed.coordinate_space, source=parsed.source)
    return AnnotationResult(annotation_id=parsed.annotation_id if parsed else str(raw.get("annotation_id", "")) if isinstance(raw, Mapping) else "", line_name=parsed.line_name if parsed else None,
                            status=status, issues=list(status_issues), metadata=meta, points_original=parsed.points.tolist() if parsed else None, runtime_s=time.perf_counter() - t0,
                            downstream_permitted=False, **kw)


def process_annotation(raw, ctx: CameraContext | None, cal, limits: AnnotationLimits = DEFAULT_LIMITS, *, expected_set_index=None, expected_frame_index=None) -> AnnotationResult:
    """Validate, transform and refine ONE annotation.  ``ctx`` is the camera frame the annotation refers to."""
    t0 = time.perf_counter(); empty = dict(transforms={}, initial_line=None, refined_line=None, diagnostics={}, downstream_line=None)
    parsed, issues = parse_annotation(raw)
    if parsed is None: return _finish(None, raw, issues, t0, **empty)
    p = parsed
    # -- association checks: line name, camera/boundary, set, frame, image
    if p.boundary_type not in CAMERA_BOUNDARIES[p.camera]: issues.append(_fail(Code.BAD_CAMERA_BOUNDARY, f"the {p.camera} camera has no {p.boundary_type} boundary in the Phase 1 geometry"))
    if p.line_name_claimed is not None and p.line_name_claimed != p.line_name: issues.append(_fail(Code.LINE_NAME, f"line_name {p.line_name_claimed!r} contradicts placement/camera/boundary ({p.line_name!r})"))
    if expected_set_index is not None and p.set_index != expected_set_index: issues.append(_fail(Code.WRONG_SET, f"annotation is for set {p.set_index}, expected {expected_set_index}"))
    if expected_frame_index is not None and p.frame_index != expected_frame_index: issues.append(_fail(Code.WRONG_FRAME, f"annotation was made on frame {p.frame_index}, but the selected {p.view_key} frame is {expected_frame_index}"))
    if ctx is None: issues.append(_fail(Code.NO_FRAME, f"no frame was provided for {p.view_key}"))
    else:
        if p.image_wh != ctx.size_wh: issues.append(_fail(Code.SIZE, f"annotation image size {p.image_wh} != frame size {ctx.size_wh}"))
        if tuple(cal.image_size) != ctx.size_wh: issues.append(_fail(Code.CAL_SIZE, f"frame size {ctx.size_wh} != calibration size {tuple(cal.image_size)}; intrinsics are not rescaled"))
        if p.fingerprint is not None and p.fingerprint != frame_fingerprint(ctx.gray): issues.append(_fail(Code.FINGERPRINT, "annotation was made on a different image than the frame supplied"))
    if any(i.status is Status.FAIL for i in issues): return _finish(p, raw, issues, t0, **empty)
    W, H = ctx.size_wh; tf = dict(coordinate_space=p.coordinate_space, points_in_space=p.points.tolist(), distortion_model="calibration K, D supplied for the " + p.camera + " camera (cv2.undistortPoints, P=K)", intrinsics_rescaled=False)
    # -- coordinate space -> distorted_full
    tol = limits.bounds_tolerance_px
    if p.coordinate_space == "preview_display":
        issues += p.display.validate(limits)
        if any(i.status is Status.FAIL for i in issues): return _finish(p, raw, issues, t0, **empty)
        if not p.display.inside_image_rect(p.points).all(): return _finish(p, raw, issues + [_fail(Code.PADDING, "a click landed in the letter-box padding outside the displayed image")], t0, **empty)
        dist = p.display.to_full(p.points); tf["display"] = p.display.to_dict()
        if ((dist < -tol) | (dist[:, 0] > W - 1 + tol) | (dist[:, 1] > H - 1 + tol)).any(): return _finish(p, raw, issues + [_fail(Code.OUT_OF_BOUNDS, "mapped click lies outside the full-resolution image")], t0, **empty)
        dist = np.clip(dist, 0, [W - 1, H - 1])
    else:
        dist = p.points.copy()
        if ((dist < 0) | (dist[:, 0] > W - 1) | (dist[:, 1] > H - 1)).any(): return _finish(p, raw, issues + [_fail(Code.OUT_OF_BOUNDS, f"point outside the {W}x{H} image")], t0, **empty)
    # -- distorted_full -> undistorted_full
    if p.coordinate_space == "undistorted_full": und = dist.copy(); tf["distorted_to_undistorted"] = "not applied (points already undistorted)"
    else: und = G.undistort_pixels(dist, ctx.K, ctx.D); tf["distorted_to_undistorted"] = "applied"
    tf.update(points_distorted_full=dist.tolist() if p.coordinate_space != "undistorted_full" else None, points_undistorted_full=und.tolist())
    if np.linalg.norm(und[0] - und[1]) < limits.min_endpoint_separation_px or np.linalg.norm(dist[0] - dist[1]) < limits.min_endpoint_separation_px:
        return _finish(p, raw, issues + [_fail(Code.DUPLICATE_ENDPOINTS, "the two points coincide")], t0, **dict(empty, transforms=tf))
    length = float(np.linalg.norm(und[0] - und[1]))
    if length < limits.min_length_px: return _finish(p, raw, issues + [_fail(Code.TOO_SHORT, f"clicked segment is {length:.1f} px long (< {limits.min_length_px})")], t0, **dict(empty, transforms=tf))
    if ((und < 0) | (und[:, 0] > W - 1) | (und[:, 1] > H - 1)).any(): issues.append(_fail(Code.UNDIST_OUT, "undistorted click lies outside the image (the undistorted frame has no valid data there)"))
    init = line_from_points(*und); issues += _orientation_issues(init, p.boundary_type, limits); tf["clicked_length_px"] = length
    if any(i.status is Status.FAIL for i in issues): return _finish(p, raw, issues, t0, **dict(empty, transforms=tf, initial_line=_public_line(init)))
    # -- refinement
    fit, diag, ri = refine_line(ctx.g, init, limits, p.edge_polarity); issues += ri; init_pub = _public_line(init)
    if fit is None: return _finish(p, raw, issues, t0, **dict(empty, transforms=tf, initial_line=init_pub, diagnostics=diag))
    half_used = limits.second_pass_half_window_px if diag.get("pass_used") == "pass2" else limits.search_half_window_px
    d = _perp_distance(init, _points_on(fit, np.linspace(init["span"][0], init["span"][1], 25))); ang = angle_between_deg(init, fit)
    pts = np.asarray(fit["pts"], float); uu, ww = (pts[:, 1], pts[:, 0]) if fit["kind"] == "v" else (pts[:, 0], pts[:, 1])
    cen = diag["search_center"]; rim = float(np.mean(np.abs(ww - _w(cen, uu)) > half_used - limits.rim_px))
    ctxd = _edge_context(ctx.g, fit, diag["polarity"], limits); N, n = int(fit["N"]), int(fit["n"])
    diag.update(n_edge_points=n, n_edge_candidates=N, inlier_fraction=n / max(N, 1), coverage_fraction=N / max(diag["n_search_positions"], 1), fit_rms_px=float(fit["rms"]),
                displacement_max_px=float(d.max()), displacement_mean_px=float(d.mean()), displacement_frac_of_window=float(d.max() / limits.search_half_window_px), angle_change_deg=float(ang),
                rim_fraction=rim, line_length_px=length, **ctxd)
    sh = diag["displacement_frac_of_window"]
    def chk(cond_fail, cond_warn, code, msg):
        if cond_fail: issues.append(_fail(code, msg))
        elif cond_warn: issues.append(_warn(code, msg))
    chk(n < limits.min_edge_points_fail, n < limits.min_edge_points_warn, Code.FEW_POINTS, f"{n} inlier edge points (fail < {limits.min_edge_points_fail}, warn < {limits.min_edge_points_warn})")
    cov = diag["coverage_fraction"]; chk(cov < limits.min_coverage_fail, cov < limits.min_coverage_warn, Code.LOW_COVERAGE, f"edge found at {cov:.0%} of the search positions")
    inl = diag["inlier_fraction"]; chk(inl < limits.min_inlier_fraction_fail, inl < limits.min_inlier_fraction_warn, Code.LOW_INLIERS, f"inlier fraction {inl:.2f}")
    chk(not fit["rms"] <= limits.max_rms_px_fail, not fit["rms"] <= limits.max_rms_px_warn, Code.HIGH_RMS, f"fit rms {fit['rms']:.2f} px")
    st = ctxd["edge_strength"]; chk(st < limits.min_edge_strength_fail, st < limits.min_edge_strength_warn, Code.WEAK_EDGE, f"edge gradient {st:.2f} gray levels/px")
    chk(sh > limits.shift_fail_frac_of_window, sh > limits.shift_warn_frac_of_window, Code.AT_LIMIT if sh > limits.shift_fail_frac_of_window else Code.BIG_SHIFT,
        f"refined line moved up to {d.max():.1f} px from the clicked line ({sh:.0%} of the search half-window): the edge may lie outside the window or be a different edge")
    chk(ang > limits.angle_change_fail_deg, ang > limits.angle_change_warn_deg, Code.ANGLE_CHANGE, f"refined direction differs from the clicked direction by {ang:.1f} deg")
    if rim > limits.rim_fraction_warn: issues.append(_warn(Code.RIM, f"{rim:.0%} of edge points sit at the rim of the search window"))
    if ctxd["competitor_ratio"] >= limits.competing_ratio: issues.append(_warn(Code.COMPETING, f"another same-polarity edge {ctxd['competitor_offset_px']:+d} px away has {ctxd['competitor_ratio']:.2f}x the gradient; the fit may be on the wrong edge"))
    refined = dict(fit); status = worst([i.status for i in issues] or [Status.PASS]); ok = status is Status.PASS or (status is Status.WARNING and limits.allow_warnings_downstream)
    r = _finish(p, raw, issues, t0, **dict(transforms=tf, initial_line=init_pub, refined_line=_public_line(refined), diagnostics=diag, downstream_line=refined if status is not Status.FAIL else None))
    r.downstream_permitted = bool(ok); r.downstream_line = refined if ok else None
    return r


# ------------------------------------------------------------------ stereo consistency (needs a pose)
def stereo_consistency(cal, R, T, left_line, right_line, limits=DEFAULT_LIMITS):
    """Compatibility of a left/right line pair (same boundary, same placement) with one common 3D line under the pose (R, T).
    With a known pose ANY two non-parallel back-projected planes meet in a line, so this is a weak test: it checks plane conditioning,
    that the 3D line lies in front of both cameras within the Phase 1 depth range, and that the visible portions are not disjoint."""
    nL = cal.K_left.T @ G.line_to_h(left_line); nR = R.T @ (cal.K_right.T @ G.line_to_h(right_line)); T = np.asarray(T, float).reshape(3); C2 = -R.T @ T
    sin = np.linalg.norm(np.cross(nL, nR)) / (np.linalg.norm(nL) * np.linalg.norm(nR)); ang = math.degrees(math.asin(min(1.0, sin))); info = dict(plane_angle_deg=ang); issues = []
    if ang < limits.min_stereo_plane_angle_deg: issues.append(_warn(Code.STEREO_PARALLEL, f"back-projected planes meet at only {ang:.2f} deg; the 3D line is poorly conditioned"))
    m = segment_endpoints(left_line).mean(0); d = np.linalg.inv(cal.K_left) @ np.array([m[0], m[1], 1.0]); den = nR @ d
    if abs(den) < 1e-12: issues.append(_fail(Code.STEREO_DEPTH, "left sight line is parallel to the right back-projected plane")); return Status.FAIL, issues, info
    lam = (nR @ C2) / den; X = lam * d; XR = R @ X + T; info.update(depth_left_mm=float(X[2]), depth_right_mm=float(XR[2]))
    lo, hi = M.DEPTH_RANGE_MM
    if not (lam > 0 and lo < X[2] < hi and XR[2] > 0): issues.append(_fail(Code.STEREO_DEPTH, f"the common 3D line is behind a camera or outside the {lo:.0f}-{hi:.0f} mm range (depth left {X[2]:.0f}, right {XR[2]:.0f} mm)")); return Status.FAIL, issues, info
    pr = cal.K_right @ XR; pr = pr[:2] / pr[2]; u = pr[1] if right_line["kind"] == "v" else pr[0]; lo_u, hi_u = right_line["span"]; gap = max(lo_u - u, u - hi_u, 0.0)
    info.update(along_line_gap_px=float(gap), right_visible_length_px=float(hi_u - lo_u))
    if gap > (hi_u - lo_u): issues.append(_warn(Code.STEREO_OVERLAP, f"the left segment's midpoint projects {gap:.0f} px beyond the visible right segment; the two may show different parts of the edge (or different edges)"))
    return (Status.WARNING if issues else Status.PASS), issues, info


# ------------------------------------------------------------------ annotation set
@dataclass
class SetResult:
    status: Status
    downstream_use_permitted: bool
    failure_codes: list
    issues: list
    annotations: list
    completeness: dict
    reconstruction: dict
    stereo_consistency: dict
    thresholds: dict
    runtime_s: float
    line_set: dict | None = None

    def lines_for_measurement(self) -> dict:
        """Phase 1 line dicts ({name: fitted line in undistorted px}) -- refuses unless downstream use is permitted."""
        if not self.downstream_use_permitted or self.line_set is None: raise AnnotationError(f"annotations are not usable downstream (status {self.status.value}; {self.failure_codes})")
        return dict(self.line_set)

    def to_dict(self):
        return to_jsonable(dict(schema="csc8830.module7.visitor_wall_annotation_result.v1", status=self.status.value, downstream_use_permitted=self.downstream_use_permitted,
                                failure_codes=self.failure_codes, issues=[i.to_dict() for i in self.issues], completeness=self.completeness, reconstruction=self.reconstruction,
                                stereo_consistency=self.stereo_consistency, annotations=[a.to_dict() for a in self.annotations], thresholds=self.thresholds, runtime_s=round(self.runtime_s, 3),
                                status_meaning=dict(PASS="all checks passed", WARNING="usable but flagged; accuracy not established", FAIL="must not be used", NOT_EVALUATED="check could not be run"),
                                note="Stage D produces validated image lines only; it does not measure the wall and says nothing about physical accuracy."))


def _reconstruction_viability(lines, cal, limits):
    """Can the UNCHANGED Phase 1 corner reconstruction run?  (Uses only geometry.line_intersection; no new algorithm.)  Corner/ceilF per placement."""
    out = {}; issues = []
    for v, (_, cp, ce, _) in M.VIEW_LINES.items():
        if cp not in lines or ce not in lines: out[v] = dict(status=Status.FAIL.value, reason="corner or ceilF line unavailable"); continue
        ang = angle_between_deg(lines[cp], lines[ce]); row = dict(corner_ceilF_angle_deg=ang)
        try:
            pt = G.line_intersection(lines[cp], lines[ce]); finite = bool(np.isfinite(pt).all())
        except Exception: finite = False
        if not finite or ang < limits.min_corner_ceiling_angle_deg:
            row.update(status=Status.FAIL.value, reason="corner and ceiling lines are (nearly) parallel"); issues.append(_fail(Code.CORNER_PARALLEL, f"{v}: corner/ceilF angle {ang:.1f} deg < {limits.min_corner_ceiling_angle_deg}", v))
        else:
            cseg = segment_endpoints(lines[cp]); dist = float(np.min(np.linalg.norm(cseg - pt, axis=1))); inside = bool(min(cseg[:, 1]) <= pt[1] <= max(cseg[:, 1])); length = float(np.linalg.norm(cseg[0] - cseg[1]))
            row.update(status=Status.PASS.value, intersection_px=pt.tolist(), intersection_on_annotated_segment=inside, extrapolation_px=0.0 if inside else dist, extrapolation_ratio=0.0 if inside else dist / length)
            if not inside and dist / length > limits.max_extrapolation_ratio:
                row["status"] = Status.WARNING.value; issues.append(_warn(Code.EXTRAPOLATION, f"{v}: hidden corner is {dist:.0f} px beyond the annotated corner edge ({dist / length:.1f}x its length); its position is extrapolated", v))
            row["note"] = "corner intersection need not be visible; it is inferred from the lines (Phase 1 method)"
        out[v] = row
    return out, issues


def process_annotation_set(raw, frames: Mapping, cal, limits: AnnotationLimits = DEFAULT_LIMITS, *, set_index=None, expected_frame_indices: Mapping | None = None,
                           pose=None) -> SetResult:
    """Process all annotations of one frame set.  ``frames``: {AL,PL,AR,PR: grayscale full-res frame}; ``expected_frame_indices``: the Stage C
    selection for this set ({AL: idx, ...}, see ``stage_c_integration.index_table_from_selection``); ``pose``: optional (R, T)."""
    t0 = time.perf_counter(); issues = []
    if isinstance(raw, Mapping):
        if raw.get("schema") not in (None, SCHEMA): issues.append(_fail(Code.SCHEMA, f"unsupported schema {raw.get('schema')!r}"))
        items = list(raw.get("annotations") or [])
    else: items = list(raw)
    ctxs = build_contexts(frames, cal); results = []
    for a in items:
        vk = None
        if isinstance(a, Mapping) and a.get("placement") in PLACEMENTS and a.get("camera") in CAMERAS: vk = VIEW_KEY[(a["placement"], a["camera"])]
        results.append(process_annotation(a, ctxs.get(vk), cal, limits, expected_set_index=set_index, expected_frame_index=None if expected_frame_indices is None or vk is None else expected_frame_indices.get(vk)))
    by_name = {}
    for r in results:
        if r.line_name: by_name.setdefault(r.line_name, []).append(r)
    usable = {}
    for nm, rs in by_name.items():
        if nm not in ALL_PHASE1_LINES: issues.append(_warn(Code.UNKNOWN_LINE, f"{nm} is not a Phase 1 boundary; ignored", nm)); continue
        if len(rs) > 1:
            issues.append(_fail(Code.DUPLICATE_ANNOTATION, f"{len(rs)} annotations for {nm}; none is used (no silent choice)", nm))
            for r in rs: r.issues.append(_fail(Code.DUPLICATE_ANNOTATION, "another annotation exists for the same boundary")); r.status = Status.FAIL; r.downstream_permitted = False; r.downstream_line = None
            continue
        if rs[0].downstream_permitted: usable[nm] = rs[0].downstream_line
    required = ALL_PHASE1_LINES if limits.require_ceiling_side else REQUIRED_LINES; present = set(by_name)
    for nm in required:
        if nm not in present: issues.append(_fail(Code.MISSING_REQUIRED, f"required boundary {nm} was not annotated", nm))
        elif nm not in usable: issues.append(_fail(Code.REQUIRED_INVALID, f"required boundary {nm} was annotated but rejected: {by_name[nm][0].failure_codes or 'duplicate'}", nm))
    for nm in ALL_PHASE1_LINES:
        if nm not in required and nm in present and nm not in usable: issues.append(_warn(Code.OPTIONAL_REJECTED, f"optional boundary {nm} was rejected and is not used", nm))
    complete = dict(required_lines=list(required), complete_for_variant0=all(n in usable for n in REQUIRED_LINES), complete_all_phase1_lines=all(n in usable for n in ALL_PHASE1_LINES),
                    usable_lines=sorted(usable), missing=[n for n in required if n not in usable], hidden_corners_note="a wall corner need not be visible; only the visible portions of the edges are annotated")
    recon, ri = _reconstruction_viability(usable, cal, limits) if complete["complete_for_variant0"] else ({}, []); issues += ri
    stereo = {}
    for v in ("V1", "V2"):
        cl, cp = M.VIEW_LINES[v][:2]
        if pose is None: stereo[v] = dict(status=Status.NOT_EVALUATED.value, reason=Code.STEREO_NO_POSE, note=STEREO_NOTE)
        elif cl not in usable or cp not in usable: stereo[v] = dict(status=Status.NOT_EVALUATED.value, reason=Code.STEREO_ONE_CAM, note=STEREO_NOTE)
        else:
            st, si, info = stereo_consistency(cal, pose[0], pose[1], usable[cl], usable[cp], limits); stereo[v] = dict(status=st.value, issues=[i.to_dict() for i in si], note=STEREO_NOTE, **info)
            issues += [replace(i, subject=v) for i in si]
            for nm in (cl, cp): by_name[nm][0].stereo_consistency = st
    # ceilF/ceilS have only one camera: stereo consistency is not defined for them (stay NOT_EVALUATED)
    status = worst([i.status for i in issues] + [r.status for r in results if r.line_name in required]) if (issues or results) else Status.FAIL
    if not results: issues.append(_fail(Code.MISSING_REQUIRED, "no annotations supplied")); status = Status.FAIL
    permitted = status is not Status.FAIL and complete["complete_for_variant0"] and (status is Status.PASS or limits.allow_warnings_downstream)
    codes = sorted({i.code for i in issues if i.status is Status.FAIL} | {c for r in results if r.line_name in required for c in r.failure_codes})
    sr = SetResult(status=status, downstream_use_permitted=bool(permitted), failure_codes=codes, issues=issues, annotations=results, completeness=complete, reconstruction=recon,
                   stereo_consistency=stereo, thresholds={k: v for k, v in limits.__dict__.items()}, runtime_s=time.perf_counter() - t0)
    sr.line_set = {n: usable[n] for n in ALL_PHASE1_LINES if n in usable} if permitted else None
    return sr
