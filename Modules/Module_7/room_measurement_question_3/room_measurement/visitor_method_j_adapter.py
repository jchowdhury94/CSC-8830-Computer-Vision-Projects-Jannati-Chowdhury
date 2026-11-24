"""F8.30D-A: thin adapter between Method J and the F8.30B visitor uncertainty framework.  BACKEND ONLY, wired to nothing live.

Purpose
-------
Run the ORIGINAL Method J reconstruction with the visitor's own calibration (K, D of both cameras), the visitor's measured baseline and already
prepared reconstruction inputs (frame-set matches and fitted wall-boundary lines), and return width / height with typed failure reasons,
quality warnings and diagnostics.  The same entry point is what the uncertainty engine calls once per sampled calibration, so Method J is
never implemented a second time.

Reused UNCHANGED (called, not copied)
-------------------------------------
``pose.build_dataset`` and ``pose.estimate_pose`` (guarded epipolar start -> joint refinement, degenerate guards),
``visitor_measurement._measure_sets`` (the visitor-path mirror of ``measure.measure_all_sets``: ``geometry.fit_floor_plane`` with the floor row
scaled to the frame height, ``measure.register_views``, ``measure.measure_wall``), ``lines.retarget_lines`` (re-express fitted lines under other
intrinsics), ``calibration.StereoCalibration`` and ``visitor_uncertainty.classify_measurement_outcome``.  No mathematics lives here: only input
validation, classification of what Method J returned, and bookkeeping.

What this module does NOT do
----------------------------
It reads no video, extracts no frame, matches no feature, fits no line from pixels, and runs no calibration.  It contains no original-experiment calibration values, no
hardcoded baseline, no annotation seeds and no ground truth.  Estimates are never rounded, clipped, corrected or re-scaled.  It does not weaken
or bypass any Method J guard: a guard that fires is reported, and what it means for the outcome is stated in the table below.

Outcome policy (SUCCESS / QUESTIONABLE / FAILED)
-------------------------------------------------
FAILED         the ORIGINAL pipeline would reject or raise (F8.30D-B Stage 1B, behavioural parity with ``pipeline.run_estimation``): any Method J joint
               degenerate flag (``pipeline.py:48``: unresolved epipolar start, joint refinement not converged, joint robust scale above 1 px), no usable
               frame set, plus the typed equivalents of exceptions the original would raise (invalid inputs, solver / dataset / geometry exceptions,
               non-finite pose, no valid dimension).  No dimensions are reported; the intermediate pose diagnostics are kept under ``diagnostics``.
QUESTIONABLE   the original accepts the result but advisory (visitor-backend) concerns exist: low pose inlier fraction, poor registration, some
               frame sets failed, only one dimension available.  Not Method J criteria; estimates unmodified.
SUCCESS        the original accepts the result and no advisory concern.  A fallback start that resolved the initial degeneracy is SUCCESS with a
               typed diagnostic note (the original is silent about it).
Vertical-edge error and skew are RECORDED only: the original pipeline never uses them as acceptance criteria.
Finite intermediate values are never treated as proof of validity: each dimension must be finite, positive and below a sanity bound.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import calibration as C, geometry as G, lines as L, measure as M, pose as P, visitor_measurement as VM, visitor_uncertainty as VU

SCHEMA = "f8.method_j_adapter/1"
SUCCESS, QUESTIONABLE, FAILED = VU.SUCCESS, VU.QUESTIONABLE, VU.FAILED
REQUIRED_MATCH_KEYS = ("A1", "B1", "M1", "A2", "B2", "M2", "MX")
REQUIRED_LINES = ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF")      # what measure_wall reads for the visitor top variant 0
MIN_STEREO_MATCHES = 8                                                                         # POOLED over all frame sets and both views: the 8-point initialisation is undetermined on fewer (not a quality threshold)
BASELINE_RTOL = 1e-9
NOTE_FALLBACK_START_USED = "POSE_FALLBACK_START_USED"                                         # typed diagnostic note (not a warning): SUCCESS is kept


class FailureReason(str, Enum):
    """Typed reasons for a FAILED outcome (``measure_all_sets`` only provides free text; the visitor path supplies the per-set geometry codes)."""
    INPUT_NO_FRAME_SETS = "INPUT_NO_FRAME_SETS"
    INPUT_MALFORMED_FRAME_SET = "INPUT_MALFORMED_FRAME_SET"
    INPUT_LINES_MISSING = "INPUT_LINES_MISSING"
    INPUT_IMAGE_SIZE_MISMATCH = "INPUT_IMAGE_SIZE_MISMATCH"
    INPUT_INVALID_CALIBRATION = "INPUT_INVALID_CALIBRATION"
    LINE_RETARGET_FAILED = "LINE_RETARGET_FAILED"
    BASELINE_CHANGED_UNLABELLED = "BASELINE_CHANGED_UNLABELLED"
    DATASET_BUILD_FAILED = "DATASET_BUILD_FAILED"
    POSE_SOLVER_EXCEPTION = "POSE_SOLVER_EXCEPTION"
    POSE_NON_FINITE = "POSE_NON_FINITE"
    POSE_DEGENERATE_UNRECOVERABLE = "POSE_DEGENERATE_UNRECOVERABLE"
    POSE_JOINT_NOT_CONVERGED = "POSE_JOINT_NOT_CONVERGED"            # original pipeline.py:48 raises on the joint degenerate flag; these two are its other causes
    POSE_JOINT_SCALE_ABOVE_LIMIT = "POSE_JOINT_SCALE_ABOVE_LIMIT"
    TRIANGULATION_FAILED = "TRIANGULATION_FAILED"            # per-set codes from visitor_measurement._measure_sets
    FLOOR_PLANE_FAILED = "FLOOR_PLANE_FAILED"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    WALL_RECONSTRUCTION_FAILED = "WALL_RECONSTRUCTION_FAILED"
    GEOMETRY_FAILED = "GEOMETRY_FAILED"
    NO_USABLE_FRAME_SETS = "NO_USABLE_FRAME_SETS"
    WIDTH_UNAVAILABLE = "WIDTH_UNAVAILABLE"
    HEIGHT_UNAVAILABLE = "HEIGHT_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class QualityFlag(str, Enum):
    """Typed reasons that make an otherwise valid result QUESTIONABLE."""
    POSE_LOW_INLIER_FRACTION = "POSE_LOW_INLIER_FRACTION"
    REGISTRATION_FEW_INLIERS = "REGISTRATION_FEW_INLIERS"
    REGISTRATION_LOW_INLIER_FRACTION = "REGISTRATION_LOW_INLIER_FRACTION"
    REGISTRATION_HIGH_RMS = "REGISTRATION_HIGH_RMS"
    SOME_FRAME_SETS_FAILED = "SOME_FRAME_SETS_FAILED"
    SET_DIMENSION_MISSING = "SET_DIMENSION_MISSING"
    DIMENSION_PARTIAL = "DIMENSION_PARTIAL"


# ------------------------------------------------------------------ calibration context and inputs
@dataclass(frozen=True, eq=False)
class VisitorCalibrationContext:
    """The visitor's calibration and physical baseline, supplied explicitly (never defaulted, never read from a file)."""
    K_left: np.ndarray
    D_left: np.ndarray
    K_right: np.ndarray
    D_right: np.ndarray
    image_size_wh: tuple
    baseline_mm: float
    label: str = "visitor"

    @classmethod
    def from_params(cls, params: dict, image_size_wh, label: str = "visitor") -> "VisitorCalibrationContext":
        """``params``: {K_left, D_left, K_right, D_right, baseline_mm}, the contract used by ``visitor_uncertainty`` estimators."""
        return cls(np.array(params["K_left"], np.float64), np.array(params["D_left"], np.float64).reshape(-1), np.array(params["K_right"], np.float64),
                   np.array(params["D_right"], np.float64).reshape(-1), tuple(int(v) for v in image_size_wh), float(params["baseline_mm"]), label)

    def to_params(self) -> dict:
        return dict(K_left=self.K_left.copy(), D_left=self.D_left.copy(), K_right=self.K_right.copy(), D_right=self.D_right.copy(), baseline_mm=self.baseline_mm)

    def to_stereo_calibration(self) -> C.StereoCalibration:
        """Validated Method J calibration object (raises ValueError if malformed).  Calibrated R, T are not part of Method J's inputs."""
        return C.validate(C.StereoCalibration(name=self.label, K_left=self.K_left, D_left=self.D_left, K_right=self.K_right, D_right=self.D_right,
                                              image_size=tuple(self.image_size_wh), baseline_mm=self.baseline_mm, provenance=dict(source="visitor calibration context")))

    def same_intrinsics(self, other: "VisitorCalibrationContext") -> bool:
        return all(np.array_equal(getattr(self, k), getattr(other, k)) for k in ("K_left", "D_left", "K_right", "D_right"))

    def digest(self) -> str:
        blob = json.dumps(dict(K_left=self.K_left.tolist(), D_left=self.D_left.tolist(), K_right=self.K_right.tolist(), D_right=self.D_right.tolist(),
                               size=list(self.image_size_wh), baseline_mm=self.baseline_mm), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass(frozen=True, eq=False)
class ReconstructionInputs:
    """Already prepared inputs: per frame set the matches (distorted pixels, as ``features.match_set`` returns) and the fitted wall-boundary
    lines (as ``lines.fit_lines`` returns, in undistorted pixels of ``lines_calibration``).  Both are produced upstream; this module never makes them."""
    match_sets: tuple
    line_sets: tuple
    lines_calibration: VisitorCalibrationContext        # the intrinsics the line fits were expressed under

    def problems(self, ctx: VisitorCalibrationContext) -> list:
        out = []
        if not self.match_sets:
            return [_rec(FailureReason.INPUT_NO_FRAME_SETS, "no frame sets were supplied", "input")]
        if len(self.match_sets) != len(self.line_sets):
            out.append(_rec(FailureReason.INPUT_LINES_MISSING, f"{len(self.match_sets)} match sets but {len(self.line_sets)} line sets", "input"))
        for i, F in enumerate(self.match_sets):
            try:
                miss = [k for k in REQUIRED_MATCH_KEYS if k not in F]
                n1, n2 = len(F["A1"]), len(F["A2"])
                bad = (len(F["B1"]) != n1 or len(F["B2"]) != n2 or np.shape(F["A1"])[-1:] != (2,) or np.shape(F["A2"])[-1:] != (2,) or np.shape(F["B1"])[-1:] != (2,) or np.shape(F["B2"])[-1:] != (2,)
                       or len(F["M1"]) != n1 or len(F["M2"]) != n2)       # structure only: a sparse set is NOT rejected here, the original rejects it per set in the geometry stage
                bad = bad or any(not np.isfinite(np.asarray(F[k], float)).all() for k in ("A1", "B1", "A2", "B2"))
            except Exception as ex:
                miss, bad = [], True
                out.append(_rec(FailureReason.INPUT_MALFORMED_FRAME_SET, f"frame set {i}: {type(ex).__name__}: {ex}", "input", frame_set=i))
                continue
            if miss or bad:
                out.append(_rec(FailureReason.INPUT_MALFORMED_FRAME_SET, f"frame set {i}: " + (f"missing {miss}" if miss else "unusable matches (need finite, paired, (n, 2) stereo matches with one index pair per match)"), "input", frame_set=i))
        if not out:
            try: pooled = sum(len(F["A1"]) + len(F["A2"]) for F in self.match_sets)
            except Exception: pooled = 0
            if pooled < MIN_STEREO_MATCHES:
                out.append(_rec(FailureReason.INPUT_MALFORMED_FRAME_SET, f"only {pooled} stereo matches in total; the 8-point initialisation needs at least {MIN_STEREO_MATCHES}", "input"))
        for i, Ls in enumerate(self.line_sets):
            absent = [n for n in REQUIRED_LINES if not (isinstance(Ls, dict) and n in Ls)]
            if absent:
                out.append(_rec(FailureReason.INPUT_LINES_MISSING, f"frame set {i}: missing line(s) {absent}", "input", frame_set=i))
        if tuple(self.lines_calibration.image_size_wh) != tuple(ctx.image_size_wh):
            out.append(_rec(FailureReason.INPUT_IMAGE_SIZE_MISMATCH, f"lines were fitted at {tuple(self.lines_calibration.image_size_wh)} but the calibration image size is {tuple(ctx.image_size_wh)}", "input"))
        return out


@dataclass(frozen=True)
class MethodJStages:
    """The Method J functions the adapter calls.  Defaults are the production functions; tests may substitute fakes to reach classification paths."""
    build_dataset: object = P.build_dataset
    estimate_pose: object = P.estimate_pose
    measure_sets: object = None                          # default: ``measure_sets_original`` (set in __post_init__); tests may substitute fakes
    retarget_lines: object = L.retarget_lines

    def __post_init__(self):
        if self.measure_sets is None: object.__setattr__(self, "measure_sets", measure_sets_original)


_STAGE_CODE = {"triangulation": FailureReason.TRIANGULATION_FAILED, "floor": FailureReason.FLOOR_PLANE_FAILED, "registration": FailureReason.REGISTRATION_FAILED, "wall": FailureReason.WALL_RECONSTRUCTION_FAILED}


def measure_sets_original(cal, p5, match_sets, line_sets, image_wh, seed=3):
    """Original-parity geometry: the SAME calls in the SAME order as ``measure.measure_all_sets`` -> ``reconstruct_geometry`` -> ``measure_wall``
    (``M.view_points``, ``G.fit_floor_plane``, ``M.register_views`` with ``default_rng(seed + i)``, ``M.measure_wall``, top variant 0), with one input-scale
    adaptation: the floor row threshold is the Phase 1 value scaled to the frame height (identical to 2600 at the original 3840 px).  Unlike the visitor mirror
    ``visitor_measurement._measure_sets`` it adds NO minimum-point guards: a set fails only where the original raises (plane fit < 3 points, < 6 registration
    pairs, singular geometry), and the failure is typed instead of free text.  No mathematics lives here."""
    R, T = G.pose_from_params(p5, cal.baseline_mm); y_min = VM.floor_row_threshold(image_wh[1]); rows = []
    for i, (F, Ls) in enumerate(zip(match_sets, line_sets)):
        stage = "triangulation"
        try:
            X1, a1, i1 = M.view_points(cal, R, T, F["A1"], F["B1"], F["M1"]); X2, a2, i2 = M.view_points(cal, R, T, F["A2"], F["B2"], F["M2"])
            stage = "floor"; f1, f2 = G.fit_floor_plane(X1, a1, y_min=y_min), G.fit_floor_plane(X2, a2, y_min=y_min)
            stage = "registration"; R0, t0, rms, ni, nt = M.register_views(X1, i1[:, 0], X2, i2[:, 0], F["MX"], np.random.default_rng(seed + i))
            stage = "wall"; g = dict(f1=f1, f2=f2, R0=R0, t0=t0, registration=dict(rms_mm=rms, n_inliers=ni, n_pairs=nt), n_points=(len(X1), len(X2)))
            o = M.measure_wall(cal, R, T, g, Ls, 0)
            rows.append(dict(ok=True, width_mm=o["width_mm"], height_mm=o["height_mm"], H1=o["V1"]["H"], H2=o["V2"]["H"], vertical_error_deg=(o["V1"]["vertical_error_deg"], o["V2"]["vertical_error_deg"]),
                             skew_mm=(o["V1"]["skew"], o["V2"]["skew"]), registration=g["registration"], n_points=g["n_points"]))
        except Exception as ex:
            rows.append(dict(ok=False, code=_STAGE_CODE[stage].value, error=f"{type(ex).__name__}: {ex}", width_mm=float("nan"), height_mm=float("nan")))
    return rows


def _rec(code, message: str, stage: str, **extra) -> dict:
    return dict(code=code.value if isinstance(code, Enum) else str(code), message=message, stage=stage, **extra)


def _finite_pos(v, bound) -> bool:
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(float(v)) and 0 < float(v) < bound


def _jsonable(o):
    if isinstance(o, dict): return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)): return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o


# ------------------------------------------------------------------ one evaluation
def evaluate(inputs: ReconstructionInputs, ctx: VisitorCalibrationContext, *, kind: str = "original", label: str = "", stages: MethodJStages | None = None,
             limits: VM.QualityLimits | None = None) -> dict:
    """Run Method J once with ``ctx`` and classify the result.  Never raises: every problem becomes a typed record in the returned dict."""
    stages = stages or MethodJStages(); limits = limits or VM.QualityLimits(); t0 = time.perf_counter()
    failures, flags, notes = [], [], []
    ev = dict(schema=SCHEMA, ground_truth_used=False, evaluation=dict(kind=kind, label=label), outcome=FAILED, width_mm=None, height_mm=None, partial=False,
              dimensions=dict(width=dict(status="unavailable", value=None, n_sets=0), height=dict(status="unavailable", value=None, n_sets=0)),
              failures=failures, warnings=flags, notes=notes, diagnostic_notes=[], diagnostics={},
              calibration=dict(digest=None, baseline_mm=None, image_size_wh=None, label=None, lines_retargeted=False))
    try:
        ev["calibration"].update(digest=ctx.digest(), baseline_mm=float(ctx.baseline_mm), image_size_wh=list(ctx.image_size_wh), label=ctx.label)
    except Exception as ex:
        failures.append(_rec(FailureReason.INPUT_INVALID_CALIBRATION, f"{type(ex).__name__}: {ex}", "calibration"))
    try:
        if not failures:
            failures += inputs.problems(ctx)
        cal = None
        if not failures:
            try: cal = ctx.to_stereo_calibration()
            except Exception as ex: failures.append(_rec(FailureReason.INPUT_INVALID_CALIBRATION, f"{type(ex).__name__}: {ex}", "calibration"))
        if not failures:
            _run(inputs, ctx, cal, stages, limits, ev)
    except Exception as ex:                                          # nothing Method J raises may escape as an untyped exception
        failures.append(_rec(FailureReason.INTERNAL_ERROR, f"{type(ex).__name__}: {ex}", "internal"))
    return _finish(ev, t0)


def _finish(ev: dict, t0: float) -> dict:
    d = ev["dimensions"]; w, h = d["width"], d["height"]
    if ev["failures"]:
        ev["outcome"] = FAILED; ev["width_mm"] = ev["height_mm"] = None; ev["partial"] = False
        ev["dimensions"] = dict(width=dict(status="unavailable", value=None, n_sets=0), height=dict(status="unavailable", value=None, n_sets=0))
        ev["note"] = "no valid geometric reconstruction; no dimensions are reported"
    elif w["status"] == "available" and h["status"] == "available":
        o = VU.classify_measurement_outcome(w["value"], h["value"], quality_flags=ev["warnings"])        # the F8.30B policy, unchanged
        ev["outcome"] = o["outcome"]; ev["width_mm"], ev["height_mm"] = o["width_mm"], o["height_mm"]; ev["note"] = o["note"]
        if o["outcome"] == FAILED: ev["failures"] += o["diagnostics"]; ev["partial"] = False
    elif w["status"] == "available" or h["status"] == "available":
        ev["outcome"] = QUESTIONABLE; ev["partial"] = True; ev["width_mm"] = w["value"]; ev["height_mm"] = h["value"]
        ev["note"] = "only one dimension is mathematically available; the other is labelled unavailable (null). Estimates are unmodified"
    else:
        ev["failures"].append(_rec(FailureReason.NO_USABLE_FRAME_SETS, "no dimension could be computed", "dimensions")); ev["outcome"] = FAILED; ev["note"] = "no valid geometric reconstruction; no dimensions are reported"
    ev["evaluation"]["seconds"] = time.perf_counter() - t0
    return _jsonable(ev)


def _run(inputs, ctx, cal, stages, limits, ev):
    failures, flags, notes, diag = ev["failures"], ev["warnings"], ev["notes"], ev["diagnostics"]
    wh = tuple(ctx.image_size_wh); line_sets = list(inputs.line_sets)
    if not ctx.same_intrinsics(inputs.lines_calibration):             # re-express the fitted lines under THIS calibration (existing function)
        try:
            cal_from = inputs.lines_calibration.to_stereo_calibration()
            line_sets = [stages.retarget_lines(Ls, cal_from, cal) for Ls in line_sets]; ev["calibration"]["lines_retargeted"] = True
        except Exception as ex:
            failures.append(_rec(FailureReason.LINE_RETARGET_FAILED, f"{type(ex).__name__}: {ex}", "lines")); return
    try:
        D = stages.build_dataset(cal, list(inputs.match_sets))
    except Exception as ex:
        failures.append(_rec(FailureReason.DATASET_BUILD_FAILED, f"{type(ex).__name__}: {ex}", "dataset")); return
    try:
        pJ, pA, prep, _ = stages.estimate_pose(D)
    except Exception as ex:
        failures.append(_rec(FailureReason.POSE_SOLVER_EXCEPTION, f"{type(ex).__name__}: {ex}", "pose")); return
    mj, ma = prep["method_J"], prep["method_A"]["guard"]; fl = mj["flags"]
    diag["pose"] = dict(params=np.asarray(pJ, float).tolist(), robust_scale_px=fl["scale_px"], inlier_fraction=fl["inlier_fraction"], converged=fl["converged"], degenerate_flag=fl["degenerate"],
                        n_inlier=mj["n_inlier"], n_residuals=mj["n_residuals"], n_tracks=mj["n_tracks"], n_stereo_matches=int(len(D["n1"])), start=prep["method_A"]["start"],
                        epipolar_guard=dict(first_attempt=ma["first_attempt"], fallback_used=ma["fallback_used"], resolved=ma.get("resolved"), final=ma["final"], unresolved_degenerate=bool(ma["degenerate"])))
    if not np.isfinite(np.asarray(pJ, float)).all():
        failures.append(_rec(FailureReason.POSE_NON_FINITE, "the Method J pose is not finite", "pose")); return
    # The original pipeline (pipeline.py:48) raises on the joint degenerate flag = unresolved epipolar start OR joint not converged OR joint robust scale > 1 px.
    # Same decision here; every firing cause is typed, and the intermediate pose diagnostics stay under diagnostics["pose"] (no dimensions are computed or returned).
    causes = []
    if ma["degenerate"]: causes.append(_rec(FailureReason.POSE_DEGENERATE_UNRECOVERABLE, f"the epipolar start is degenerate after the fallback restart (flags {ma['final']})", "pose"))
    if not fl["converged"]: causes.append(_rec(FailureReason.POSE_JOINT_NOT_CONVERGED, "the joint pose refinement did not converge (the original pipeline rejects this)", "pose"))
    if fl["scale_px"] > P.DEGENERATE_SCALE_PX: causes.append(_rec(FailureReason.POSE_JOINT_SCALE_ABOVE_LIMIT, f"joint robust residual scale {fl['scale_px']:.3f} px exceeds Method J's own guard of {P.DEGENERATE_SCALE_PX} px (the original pipeline rejects this)", "pose", value=fl["scale_px"]))
    if bool(fl["degenerate"]) and not causes:                         # the original's flag is set but no cause is visible in the report: still the original's decision
        causes.append(_rec(FailureReason.POSE_DEGENERATE_UNRECOVERABLE, "Method J reports the joint pose as degenerate", "pose"))
    diag["original_pipeline_decision"] = dict(rule="pipeline.run_estimation raises when prep['method_J']['flags']['degenerate'] (pipeline.py:48)", rejected=bool(causes), causes=[c["code"] for c in causes])
    if causes:
        failures += causes; return
    if ma["fallback_used"]:                                           # resolved by the fallback and the final pose passes the original checks: SUCCESS plus a typed diagnostic note
        ev["diagnostic_notes"].append(_rec(NOTE_FALLBACK_START_USED, "the default epipolar start was degenerate; the generic parallel-rig fallback start recovered a non-degenerate solution and the final pose passes the original acceptance checks", "pose"))
    if fl["inlier_fraction"] < limits.pose_inlier_fraction_warn: flags.append(_rec(QualityFlag.POSE_LOW_INLIER_FRACTION, f"pose inlier fraction {fl['inlier_fraction']:.3f} < {limits.pose_inlier_fraction_warn} (advisory, visitor-backend level; not a Method J criterion)", "pose", value=fl["inlier_fraction"]))
    try:
        rows = stages.measure_sets(cal, pJ, list(inputs.match_sets), line_sets, wh)
    except Exception as ex:
        failures.append(_rec(FailureReason.GEOMETRY_FAILED, f"{type(ex).__name__}: {ex}", "geometry")); return
    ok = [(i, r) for i, r in enumerate(rows) if r.get("ok")]; bad = [(i, r) for i, r in enumerate(rows) if not r.get("ok")]
    known = {e.value for e in FailureReason}
    diag["set_failures"] = [dict(frame_set=i, code=r.get("code") if r.get("code") in known else FailureReason.GEOMETRY_FAILED.value, message=r.get("error", "")) for i, r in bad]
    diag["n_sets"], diag["n_sets_ok"] = len(rows), len(ok)
    if not ok:
        failures.append(_rec(FailureReason.NO_USABLE_FRAME_SETS, "no frame set produced a reconstruction", "geometry", causes=sorted({f["code"] for f in diag["set_failures"]}))); return
    if bad: flags.append(_rec(QualityFlag.SOME_FRAME_SETS_FAILED, f"{len(bad)} of {len(rows)} frame sets failed ({sorted({f['code'] for f in diag['set_failures']})}); the dimensions use the remaining sets", "geometry", frame_sets=[i for i, _ in bad]))
    bound = limits.max_dimension_mm
    Wv = [(i, float(r["width_mm"])) for i, r in ok if _finite_pos(r["width_mm"], bound)]; Hv = [(i, float(r["height_mm"])) for i, r in ok if _finite_pos(r["height_mm"], bound)]
    for name, vals in (("width", Wv), ("height", Hv)):
        ev["dimensions"][name] = dict(status="available", value=float(np.mean([v for _, v in vals])), n_sets=len(vals), sets=[i for i, _ in vals]) if vals else dict(status="unavailable", value=None, n_sets=0,
                                                                                                                                                                   reason=(FailureReason.WIDTH_UNAVAILABLE if name == "width" else FailureReason.HEIGHT_UNAVAILABLE).value)
    partial_sets = sorted({i for i, _ in ok} - {i for i, _ in Wv}) + sorted({i for i, _ in ok} - {i for i, _ in Hv})
    if partial_sets and (Wv or Hv):
        flags.append(_rec(QualityFlag.SET_DIMENSION_MISSING, "some frame sets lack a valid width or a valid height; each dimension averages only the sets where it is valid", "dimensions",
                          sets_without_width=sorted({i for i, _ in ok} - {i for i, _ in Wv}), sets_without_height=sorted({i for i, _ in ok} - {i for i, _ in Hv})))
    if bool(Wv) != bool(Hv):
        missing = "height" if Wv else "width"
        flags.append(_rec(QualityFlag.DIMENSION_PARTIAL, f"{missing} is not available (no frame set gave a finite positive {missing})", "dimensions", missing=missing))
    reg = [r["registration"] for _, r in ok]
    if reg:
        ni = min(r["n_inliers"] for r in reg); frac = min(r["n_inliers"] / max(r["n_pairs"], 1) for r in reg); rms = max(r["rms_mm"] for r in reg)
        if ni < limits.registration_min_inliers_warn: flags.append(_rec(QualityFlag.REGISTRATION_FEW_INLIERS, f"fewest registration inliers {ni} < {limits.registration_min_inliers_warn}", "registration", value=ni))
        if frac < limits.registration_min_inlier_fraction_warn: flags.append(_rec(QualityFlag.REGISTRATION_LOW_INLIER_FRACTION, f"lowest registration inlier fraction {frac:.2f} < {limits.registration_min_inlier_fraction_warn}", "registration", value=frac))
        if not math.isfinite(rms) or rms > limits.registration_rms_mm_warn: flags.append(_rec(QualityFlag.REGISTRATION_HIGH_RMS, f"worst registration rms {rms} mm > {limits.registration_rms_mm_warn}", "registration", value=rms))
        diag["registration"] = dict(min_inliers=ni, min_inlier_fraction=frac, max_rms_mm=rms)
    diag["per_set"] = [dict(frame_set=i, width_mm=r["width_mm"], height_mm=r["height_mm"], registration=r["registration"], vertical_error_deg=r.get("vertical_error_deg"), skew_mm=r.get("skew_mm")) for i, r in ok]
    diag["geometry_consistency_informational"] = dict(max_vertical_error_deg=max(max(r["vertical_error_deg"]) for _, r in ok if r.get("vertical_error_deg") is not None) if any(r.get("vertical_error_deg") is not None for _, r in ok) else None,
                                                      max_skew_mm=max(max(r["skew_mm"]) for _, r in ok if r.get("skew_mm") is not None) if any(r.get("skew_mm") is not None for _, r in ok) else None,
                                                      note="recorded only: the original pipeline does not use vertical-edge error or skew as acceptance criteria")
    notes.append("quality thresholds are the visitor backend's generic defaults (VM.QualityLimits); the pose scale threshold is Method J's own guard")


# ------------------------------------------------------------------ uncertainty contract
class EvaluationLog:
    """Every attempted evaluation, failures included.  Nothing is dropped; ``summary`` counts what happened."""
    def __init__(self): self.records: list = []

    def add(self, ev: dict) -> dict:
        rec = dict(index=len(self.records), kind=ev["evaluation"]["kind"], label=ev["evaluation"]["label"], outcome=ev["outcome"], partial=ev["partial"], width_mm=ev["width_mm"], height_mm=ev["height_mm"],
                   failure_codes=[f["code"] for f in ev["failures"]], warning_codes=[w["code"] for w in ev["warnings"]], calibration_digest=ev["calibration"]["digest"],
                   baseline_mm=ev["calibration"]["baseline_mm"], seconds=ev["evaluation"].get("seconds"))
        self.records.append(rec); return rec

    def summary(self) -> dict:
        codes: dict = {}
        for r in self.records:
            for c in r["failure_codes"]: codes[c] = codes.get(c, 0) + 1
        return dict(attempted=len(self.records), success=sum(r["outcome"] == SUCCESS for r in self.records), questionable=sum(r["outcome"] == QUESTIONABLE for r in self.records),
                    failed=sum(r["outcome"] == FAILED for r in self.records), partial=sum(bool(r["partial"]) for r in self.records), failure_codes=codes, records=list(self.records))


def make_estimator(inputs: ReconstructionInputs, base_ctx: VisitorCalibrationContext, *, log: EvaluationLog, stages: MethodJStages | None = None, limits: VM.QualityLimits | None = None,
                   include_questionable: bool = True, baseline_sensitivity_label: str | None = None):
    """Estimator callable for ``visitor_uncertainty`` (``propagate_monte_carlo`` / ``propagate_sigma_points``): params -> {width_mm, height_mm} or EstimatorFailure.

    * The WHOLE supplied (K, D) of both cameras is used for the evaluation, so correlations inside a calibration draw are preserved; nothing is perturbed here.
    * The baseline is the visitor's physical baseline.  A different baseline is rejected (logged, ``BASELINE_CHANGED_UNLABELLED``) unless the estimator was
      created with ``baseline_sensitivity_label``, in which case the evaluations are recorded as ``baseline_sensitivity``.
    * Every attempt, successful or not, is appended to ``log``.  Ground truth is never read or used.
    * Partial results cannot be propagated as a (width, height) pair: they are logged and raised as ``dimension_unavailable:<name>``.
    """
    def estimator(params: dict) -> dict:
        kind = "baseline_sensitivity" if baseline_sensitivity_label else "calibration_draw"; label = baseline_sensitivity_label or ""
        p = dict(params); p.setdefault("baseline_mm", base_ctx.baseline_mm)
        try:
            ctx = VisitorCalibrationContext.from_params(p, base_ctx.image_size_wh, label=f"{base_ctx.label}:{kind}")
            changed = not math.isclose(ctx.baseline_mm, base_ctx.baseline_mm, rel_tol=BASELINE_RTOL, abs_tol=0.0)
        except Exception as ex:
            ev = _finish(dict(schema=SCHEMA, ground_truth_used=False, evaluation=dict(kind=kind, label=label), outcome=FAILED, width_mm=None, height_mm=None, partial=False,
                              dimensions=dict(width=dict(status="unavailable", value=None, n_sets=0), height=dict(status="unavailable", value=None, n_sets=0)),
                              failures=[_rec(FailureReason.INPUT_INVALID_CALIBRATION, f"{type(ex).__name__}: {ex}", "calibration")], warnings=[], notes=[], diagnostics={},
                              calibration=dict(digest=None, baseline_mm=None, image_size_wh=None, label=None, lines_retargeted=False)), time.perf_counter())
            log.add(ev); raise VU.EstimatorFailure(FailureReason.INPUT_INVALID_CALIBRATION.value)
        if changed and not baseline_sensitivity_label:
            ev = _finish(dict(schema=SCHEMA, ground_truth_used=False, evaluation=dict(kind=kind, label=label), outcome=FAILED, width_mm=None, height_mm=None, partial=False,
                              dimensions=dict(width=dict(status="unavailable", value=None, n_sets=0), height=dict(status="unavailable", value=None, n_sets=0)),
                              failures=[_rec(FailureReason.BASELINE_CHANGED_UNLABELLED, f"baseline {ctx.baseline_mm} mm differs from the visitor's {base_ctx.baseline_mm} mm without a labelled sensitivity test", "calibration")],
                              warnings=[], notes=[], diagnostics={}, calibration=dict(digest=ctx.digest(), baseline_mm=ctx.baseline_mm, image_size_wh=list(ctx.image_size_wh), label=ctx.label, lines_retargeted=False)), time.perf_counter())
            log.add(ev); raise VU.EstimatorFailure(FailureReason.BASELINE_CHANGED_UNLABELLED.value)
        ev = evaluate(inputs, ctx, kind=kind, label=label, stages=stages, limits=limits); log.add(ev)
        if ev["outcome"] == FAILED:
            raise VU.EstimatorFailure(ev["failures"][0]["code"], ev["failures"][0]["message"])
        if ev["partial"]:
            raise VU.EstimatorFailure("dimension_unavailable:" + ("height" if ev["height_mm"] is None else "width"))
        if ev["outcome"] == QUESTIONABLE and not include_questionable:
            raise VU.EstimatorFailure("questionable_excluded:" + (ev["warnings"][0]["code"] if ev["warnings"] else ""))
        return dict(width_mm=ev["width_mm"], height_mm=ev["height_mm"], degenerate=False, outcome=ev["outcome"], quality_flags=[w["code"] for w in ev["warnings"]])
    return estimator


def measure_with_optional_uncertainty(inputs: ReconstructionInputs, ctx: VisitorCalibrationContext, uncertainty_runner=None, *, stages: MethodJStages | None = None,
                                      limits: VM.QualityLimits | None = None, include_questionable: bool = True) -> dict:
    """The original measurement is computed and frozen FIRST.  ``uncertainty_runner(estimator, log) -> result`` then runs in isolation: any exception, including
    ``UncertaintyUnavailable``, only changes the uncertainty status, never the measurement or its outcome."""
    measurement = evaluate(inputs, ctx, kind="original", stages=stages, limits=limits)
    log = EvaluationLog(); log.add(measurement)
    unc = dict(status="not_requested", result=None, reason=None)
    if uncertainty_runner is not None and measurement["outcome"] == FAILED:
        unc = dict(status="skipped_measurement_failed", result=None, reason="no valid measurement to attach an uncertainty to")
    elif uncertainty_runner is not None:
        try:
            est = make_estimator(inputs, ctx, log=log, stages=stages, limits=limits, include_questionable=include_questionable)
            unc = dict(status="computed", result=uncertainty_runner(est, log), reason=None)
        except VU.UncertaintyUnavailable as ex:
            unc = dict(status="unavailable", result=None, reason=str(ex))
        except Exception as ex:
            unc = dict(status="failed", result=None, reason=f"{type(ex).__name__}: {ex}")
    return dict(measurement=measurement, uncertainty=unc, evaluation_log=log.summary(), label=VU.LABEL,
                note="an uncertainty failure never invalidates the original measurement; uncertainty evaluations are logged separately from it")
