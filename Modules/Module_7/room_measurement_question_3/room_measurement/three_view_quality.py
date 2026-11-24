"""Quality classification for the three-viewpoint Wall B reconstruction (no Streamlit, no I/O).

PROVISIONAL ENGINEERING CHECKS.  The numbers in ``QualityThresholds`` were derived from a stereo noise-floor estimate and from a sensitivity study on ONE room, ONE camera rig and one set of
recordings.  They are not statistically validated accuracy guarantees, and a measurement that passes every check is still an approximate estimate: a calibration error common to both cameras
(for example a focal-length bias) leaves every one of these checks quiet while shifting the length by tens to hundreds of millimetres.

Three outcomes are kept apart:

    HARD FAILURE   the geometry is invalid or cannot support a measurement (this frame is not used)
    WARNING        the geometry is valid but uncertain (the number is reported, WITH the warning)
    (neither)      no check fired -- still an approximate estimate, never an accuracy claim

Comparison rule (uniform): an upper limit fires when ``value > limit``; a lower limit fires when ``value < limit``.  A value exactly at a limit does not fire.  Non-finite values always fire the
hard check of their metric.

Frame statuses: ``OK`` (no finding), ``WARNINGS`` (usable, at least one warning), ``FAILED`` (at least one hard failure; excluded).
Aggregate statuses: ``MEASURED`` (>= ``min_usable_frames`` usable frames, no warning anywhere), ``MEASURED_WITH_WARNINGS``, ``INSUFFICIENT_EVIDENCE`` (fewer usable frames; NO length is returned).

Spatial coverage (occupied grid cells) is WARNING-ONLY: it is never a reason to reject a reconstruction.  Degeneracy failures are limited to mathematically justified ones (a diverged or degenerate
pose, collinear or too few registration inliers).  The plausible-length range is a configurable sanity check, not a physical limit (``length_range_mm=None`` turns it off).  Frame-to-frame spread is a
warning, never a failure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

HARD, WARNING = "hard", "warning"
OK, WARNINGS, FAILED = "OK", "WARNINGS", "FAILED"
MEASURED, MEASURED_WITH_WARNINGS, INSUFFICIENT_EVIDENCE = "MEASURED", "MEASURED_WITH_WARNINGS", "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class QualityThresholds:
    """(hard, warn) pairs.  ``None`` disables a limit.  Pixel / angle / fraction limits do not depend on the rig; the millimetre limits assume about 225 mm baseline and 5 m depth
    (stereo depth noise is about z^2 * sigma_px / (f * b)) and should be scaled in proportion to z^2 / (f * b) for other rigs."""
    pose_scale_px: tuple = (1.0, 0.6)                 # robust residual scale of the stereo pose, per viewpoint (upper)
    pose_inlier_fraction: tuple = (0.5, 0.75)         # lower
    positive_depth_fraction: tuple = (0.95, 0.99)     # lower
    pose_matches: tuple = (300, 1000)                 # stereo matches per viewpoint (lower)
    reg_inliers: tuple = (20, 80)                     # registration inliers (lower)
    reg_inlier_fraction: tuple = (0.5, 0.8)           # lower
    reg_rms_mm: tuple = (30.0, 20.0)                  # upper
    reg_collinearity: tuple = (0.05, 0.15)            # SD2/SD1 of the inlier cloud (lower): ~0 = collinear = rotation about the line is unconstrained
    reg_planarity: tuple = (None, 0.05)               # SD3/SD1 of the inlier cloud (lower), warning only
    reg_coverage_cells: tuple = (None, 3)             # occupied cells of the 4 x 4 grid (lower), WARNING ONLY
    floor_inliers_20mm: tuple = (0.5, 0.9)            # fraction of floor points within 20 mm of the plane (lower)
    floor_points: tuple = (200, 500)                  # points below the floor row (lower)
    floor_angle_to_v2_deg: tuple = (3.0, 1.0)         # registered V1 / V3 floor vs V2 floor (upper)
    floor_angle_v1_v3_deg: tuple = (4.0, 2.0)         # registered V1 vs V3 floor (upper; two registrations accumulate)
    corner_direction_deg: tuple = (10.0, 3.0)         # reconstructed vertical edge vs floor normal (upper)
    corner_off_floor_mm: tuple = (150.0, 40.0)        # a corner's distance from another view's floor plane (upper)
    length_vertical_fraction: tuple = (0.10, 0.03)    # |corner-to-corner vector . floor normal| / length (upper)
    length_range_mm: tuple | None = (500.0, 30000.0)  # configurable sanity range (hard when outside); None disables
    frame_spread_fraction: float = 0.03               # (max - min) / mean over usable frames: WARNING only
    min_usable_frames: int = 2                        # geometrically usable frames needed for an aggregate length


DEFAULT_THRESHOLDS = QualityThresholds()

ACTIONS = dict(
    pose="Re-record both cameras with a steadier hold and a wall with more texture, and check that the same calibration is used for both cameras.",
    registration="Record the three viewpoints so that neighbouring views overlap by at least half of the frame and share textured floor and wall areas.",
    floor="Make sure a large, evenly lit part of the floor is visible in every viewpoint and that the floor is not covered by rugs or furniture.",
    corner="Mark the vertical corner edges again with the two points far apart and exactly on the edge in both the Air and the Pro frames.",
    length="Check that all three viewpoints show the same wall and that the corners were marked at the wall ends.",
    spread="Select different frames or record the viewpoints again; individual frames disagree more than expected.",
)


def _finding(severity, code, metric, scope, value, limit, message, action):
    return dict(severity=severity, code=code, metric=metric, scope=scope, value=None if value is None or not math.isfinite(value) else float(value), limit=limit, message=message, action=ACTIONS[action])


def _check(out, value, pair, *, upper, code, metric, scope, text, action):
    """Append findings for one metric.  ``pair`` = (hard, warn)."""
    hard, warn = pair
    v = None if value is None else float(value)
    if v is None or not math.isfinite(v):
        if hard is not None or warn is not None:
            out.append(_finding(HARD, code, metric, scope, v, None, f"{text} could not be computed.", action))
        return
    def fires(limit): return limit is not None and (v > limit if upper else v < limit)
    if fires(hard): out.append(_finding(HARD, code, metric, scope, v, hard, f"{text} is {v:.3g} ({'above' if upper else 'below'} the {hard:g} limit).", action))
    elif fires(warn): out.append(_finding(WARNING, code, metric, scope, v, warn, f"{text} is {v:.3g} ({'above' if upper else 'below'} the {warn:g} warning level).", action))


def classify_frame(result: dict, thr: QualityThresholds = DEFAULT_THRESHOLDS) -> dict:
    """Findings for one reconstructed frame (``reconstruct_frame``'s dict).  Returns ``dict(status, hard_failures, warnings)``."""
    out: list = []
    m = result["metrics"]; L = result.get("length_mm")
    for v, d in (m.get("pose") or {}).items():
        if not d: continue
        s = f"V{v}"
        if d.get("degenerate") or d.get("converged") is False:
            out.append(_finding(HARD, "POSE_DEGENERATE", "pose.degenerate", s, None, None, f"The stereo pose of viewpoint {v} did not converge to a valid solution.", "pose"))
        _check(out, d.get("scale_px"), thr.pose_scale_px, upper=True, code="POSE_RESIDUAL", metric="pose.scale_px", scope=s, text=f"Viewpoint {v}'s stereo residual (px)", action="pose")
        _check(out, d.get("inlier_fraction"), thr.pose_inlier_fraction, upper=False, code="POSE_INLIERS", metric="pose.inlier_fraction", scope=s, text=f"Viewpoint {v}'s stereo inlier fraction", action="pose")
        _check(out, d.get("positive_depth_fraction"), thr.positive_depth_fraction, upper=False, code="POSE_DEPTH", metric="pose.positive_depth_fraction", scope=s, text=f"Viewpoint {v}'s fraction of points in front of both cameras", action="pose")
        _check(out, d.get("n_matches"), thr.pose_matches, upper=False, code="POSE_FEW_MATCHES", metric="pose.n_matches", scope=s, text=f"Viewpoint {v}'s stereo matches", action="pose")
        if d.get("fallback_used"):
            out.append(_finding(WARNING, "POSE_FALLBACK_START", "pose.fallback_used", s, None, None, f"Viewpoint {v}'s pose needed the fallback starting point.", "pose"))
    for key, scope in (("reg_v1_v2", "V1->V2"), ("reg_v3_v2", "V3->V2")):
        g = m[key]; sd = g["spread"]["sd_mm"]
        _check(out, g["n_inliers"], thr.reg_inliers, upper=False, code="REG_FEW_INLIERS", metric=f"{key}.n_inliers", scope=scope, text=f"Registration {scope} inliers", action="registration")
        _check(out, g["inlier_fraction"], thr.reg_inlier_fraction, upper=False, code="REG_INLIER_FRACTION", metric=f"{key}.inlier_fraction", scope=scope, text=f"Registration {scope} inlier fraction", action="registration")
        _check(out, g["rms_mm"], thr.reg_rms_mm, upper=True, code="REG_RMS", metric=f"{key}.rms_mm", scope=scope, text=f"Registration {scope} RMS (mm)", action="registration")
        _check(out, sd[1] / sd[0] if sd[0] > 0 else float("nan"), thr.reg_collinearity, upper=False, code="REG_COLLINEAR", metric=f"{key}.collinearity", scope=scope, text=f"Registration {scope} inlier spread (second/first axis)", action="registration")
        _check(out, sd[2] / sd[0] if sd[0] > 0 else float("nan"), thr.reg_planarity, upper=False, code="REG_PLANAR", metric=f"{key}.planarity", scope=scope, text=f"Registration {scope} inlier depth spread (third/first axis)", action="registration")
        cells = min(g["coverage_a"]["cells"], g["coverage_b"]["cells"])
        _check(out, cells, thr.reg_coverage_cells, upper=False, code="REG_COVERAGE", metric=f"{key}.coverage_cells", scope=scope, text=f"Registration {scope}: grid cells (of 16) occupied by matched points", action="registration")
    for v, f in (m.get("floor") or {}).items():
        s = f"V{v}"
        _check(out, f["floor_inl20"], thr.floor_inliers_20mm, upper=False, code="FLOOR_FIT", metric="floor.floor_inl20", scope=s, text=f"Viewpoint {v}'s floor points within 20 mm of the plane (fraction)", action="floor")
        _check(out, f["n_floor"], thr.floor_points, upper=False, code="FLOOR_FEW_POINTS", metric="floor.n_floor", scope=s, text=f"Viewpoint {v}'s floor points", action="floor")
    fc = m["floor_consistency"]
    _check(out, fc["ang_v1_v2_deg"], thr.floor_angle_to_v2_deg, upper=True, code="FLOOR_ANGLE", metric="floor_consistency.ang_v1_v2_deg", scope="V1-V2", text="Floor-plane disagreement V1/V2 (deg)", action="floor")
    _check(out, fc["ang_v3_v2_deg"], thr.floor_angle_to_v2_deg, upper=True, code="FLOOR_ANGLE", metric="floor_consistency.ang_v3_v2_deg", scope="V3-V2", text="Floor-plane disagreement V3/V2 (deg)", action="floor")
    _check(out, fc["ang_v1_v3_deg"], thr.floor_angle_v1_v3_deg, upper=True, code="FLOOR_ANGLE", metric="floor_consistency.ang_v1_v3_deg", scope="V1-V3", text="Floor-plane disagreement V1/V3 (deg)", action="floor")
    c = m["corner"]
    for v, key in ((1, "v1"), (3, "v3")):
        _check(out, c["vertical_edge_vs_floor_normal_deg"][key], thr.corner_direction_deg, upper=True, code="CORNER_DIRECTION", metric="corner.vertical_edge_vs_floor_normal_deg", scope=f"V{v}", text=f"Viewpoint {v}'s corner-edge direction error (deg)", action="corner")
    for key, scope in (("b1_off_v2_floor_mm", "V1 corner/V2 floor"), ("b3_off_v2_floor_mm", "V3 corner/V2 floor"), ("b1_off_v3_floor_mm", "V1 corner/V3 floor"), ("b3_off_v1_floor_mm", "V3 corner/V1 floor")):
        _check(out, abs(c[key]), thr.corner_off_floor_mm, upper=True, code="CORNER_OFF_FLOOR", metric=f"corner.{key}", scope=scope, text=f"Distance of the {scope} (mm)", action="corner")
    if L is not None and math.isfinite(L) and L > 0:
        _check(out, c["length_vertical_component_mm"] / L, thr.length_vertical_fraction, upper=True, code="LENGTH_NOT_HORIZONTAL", metric="corner.length_vertical_fraction", scope="length", text="Vertical share of the corner-to-corner vector", action="length")
    if L is None or not math.isfinite(L) or L <= 0:
        out.append(_finding(HARD, "LENGTH_INVALID", "length_mm", "length", None, None, "No valid length was produced.", "length"))
    elif thr.length_range_mm is not None and not (thr.length_range_mm[0] <= L <= thr.length_range_mm[1]):
        out.append(_finding(HARD, "LENGTH_OUT_OF_RANGE", "length_mm", "length", L, list(thr.length_range_mm), f"The length {L:.0f} mm is outside the configured plausible range {thr.length_range_mm[0]:g}-{thr.length_range_mm[1]:g} mm.", "length"))
    hard = [f for f in out if f["severity"] == HARD]; warns = [f for f in out if f["severity"] == WARNING]
    return dict(status=FAILED if hard else (WARNINGS if warns else OK), hard_failures=hard, warnings=warns)


def failed_frame(frame: int, code: str, message: str) -> dict:
    """A frame whose reconstruction could not even be completed (structural or numerical failure): reported, never hidden."""
    f = _finding(HARD, code, "reconstruction", "frame", None, None, message, "registration" if "REG" in code else "pose")
    return dict(frame=int(frame), length_mm=None, metrics=None, quality=dict(status=FAILED, hard_failures=[f], warnings=[]))


def aggregate(per_frame: list, thr: QualityThresholds = DEFAULT_THRESHOLDS) -> dict:
    """Combine classified frames.  The length is the mean of the USABLE frames (no hard failure), and only when at least ``min_usable_frames`` are usable."""
    usable = [r for r in per_frame if r["quality"]["status"] != FAILED]
    findings = [dict(w, frame=r["frame"]) for r in per_frame for w in r["quality"]["warnings"]]
    failures = [dict(h, frame=r["frame"]) for r in per_frame for h in r["quality"]["hard_failures"]]
    out = dict(frames_total=len(per_frame), frames_usable=len(usable), frames_failed=[r["frame"] for r in per_frame if r["quality"]["status"] == FAILED], frame_warnings=findings, frame_failures=failures, aggregate_warnings=[])
    if len(usable) < thr.min_usable_frames:
        out.update(status=INSUFFICIENT_EVIDENCE, length_mm=None, length_spread_mm=None,
                   message=f"Only {len(usable)} of {len(per_frame)} frames gave a usable reconstruction; at least {thr.min_usable_frames} are needed, so no length is reported.")
        return out
    L = np.array([r["length_mm"] for r in usable]); mean = float(L.mean())
    spread = dict(min=float(L.min()), max=float(L.max()), range=float(np.ptp(L)), sd=float(L.std(ddof=1)) if len(L) > 1 else 0.0)
    if mean > 0 and spread["range"] / mean > thr.frame_spread_fraction:
        out["aggregate_warnings"].append(_finding(WARNING, "FRAME_SPREAD", "frame_spread_fraction", "frames", spread["range"] / mean, thr.frame_spread_fraction,
                                                  f"The frames disagree by {spread['range']:.0f} mm ({100 * spread['range'] / mean:.1f}% of the mean).", "spread"))
    any_warn = bool(findings or out["aggregate_warnings"] or failures)       # a failed frame that was excluded is also shown, and makes the result "with warnings"
    out.update(status=MEASURED_WITH_WARNINGS if any_warn else MEASURED, length_mm=mean, length_spread_mm=spread)
    return out
