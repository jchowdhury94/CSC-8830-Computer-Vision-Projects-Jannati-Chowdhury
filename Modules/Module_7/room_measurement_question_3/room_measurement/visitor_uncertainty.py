"""F8.30B: visitor calibration uncertainty and synthetic propagation.  BACKEND ONLY, wired to nothing live.

What this module does
---------------------
1. Loads the F8.29 ``calibration_observations`` record (strictly validated) plus the stored distortion model.
2. Reproduces the saved calibration from those observations (tolerance check) with the UNCHANGED production solver
   ``stereo_measure.user_calibration.calibrate_stereo`` and the stored (never ``"auto"``) distortion model.
3. Pair-preserving bootstrap: resamples complete accepted left/right pairs with replacement, recalibrates each resample and keeps the whole
   18-parameter calibration vector, so correlations among fx, fy, cx, cy and the distortion terms are preserved.
4. Propagates calibration draws through an ESTIMATOR CALLABLE (a synthetic one in tests; Method J is never imported here) by full Monte Carlo and by a
   cheaper sigma-point approximation, and compares them.
5. Accounts for every failed draw (counts, reasons, fraction), reports SD stability, and classifies the result.
6. A separate, pure measurement outcome classifier (SUCCESS / QUESTIONABLE / FAILED).  It is not wired to the live visitor backend.

What it does NOT do
-------------------
It never reads video, never runs Method J, never changes the calibration solver or any accepted calibration, never invents missing observations, and never
claims a confidence interval or coverage probability.  All spreads are repeatability under resampling, not accuracy.  Thresholds marked PROVISIONAL are
explicit, configurable, and have no empirical justification yet.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_MODULE = Path(__file__).resolve().parents[1]
for _p in (str(_MODULE), str(_MODULE.parent)):          # _MODULE.parent: the shared stereo_measure package
    if _p not in sys.path:
        sys.path.insert(0, _p)
from stereo_measure import user_calibration as uc          # noqa: E402  (production solver; used unchanged)


def _load_observations_module():
    try:
        import calibration_observations as co               # type: ignore
        return co
    except ImportError:
        path = _MODULE / "f8_chunk_poc" / "calibration_observations.py"
        spec = importlib.util.spec_from_file_location("calibration_observations", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["calibration_observations"] = mod
        spec.loader.exec_module(mod)
        return mod


CO = _load_observations_module()

SCHEMA = "f8.visitor_uncertainty/1"
LABEL = "Indicative uncertainty"
MEASURED, SENSITIVITY, NOT_QUANTIFIED = "Measured", "Sensitivity-based", "Not quantified"

PARAM_NAMES = [f"{cam}_{n}" for cam in ("L", "R") for n in ("fx", "fy", "cx", "cy", "d0", "d1", "d2", "d3", "d4")]
N_PARAMS = len(PARAM_NAMES)
FIXED_PARAM_NOTE = "d2/d3 (tangential) are constant zero under the radial models; constant columns have zero spread by construction"


class UncertaintyUnavailable(Exception):
    """Calibration uncertainty cannot be estimated from what was supplied (the reason is the message). Never replaced by an assumed value."""


class EstimatorFailure(Exception):
    """Raised by an estimator for a draw it cannot reconstruct; ``code`` is the counted failure reason."""
    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


@dataclass
class UncertaintyConfig:
    seed: int = 2024                         # recorded in every output
    n_draws: int = 100
    max_draws: int = 2000                    # hard bound on memory and time, whatever the caller asks
    max_seconds: float = 120.0               # bootstrap stops (and says so) when exceeded; a bound, not a target
    min_pairs: int = uc.MIN_VALID_PAIRS      # the solver's own minimum
    # reproduction of the saved calibration (PROVISIONAL: OpenCV reruns differ ~1e-7 on one machine; other builds untested)
    repro_tol_K_px: float = 1e-2
    repro_tol_D: float = 1e-4
    require_reproduction: bool = True
    # classification thresholds (all PROVISIONAL, no empirical basis yet)
    max_failure_fraction: float = 0.20
    min_successful_draws: int = 30
    stability_rel_tol: float = 0.15          # |SD(n) - SD(n/2)| / SD(n)
    representativeness_z: float = 4.0        # pair-inclusion imbalance among successful draws, in standard errors
    approx_rel_tol: float = 0.25             # sigma-point SD vs Monte Carlo SD
    sigma_rank: int = 4

    PROVISIONAL = ("repro_tol_K_px", "repro_tol_D", "max_failure_fraction", "min_successful_draws", "stability_rel_tol", "representativeness_z",
                   "approx_rel_tol", "sigma_rank")

    def describe(self):
        d = asdict(self)
        d["provisional_thresholds"] = list(self.PROVISIONAL)
        d["provisional_note"] = "No threshold here is empirically justified; each is explicit and configurable."
        return d


# ------------------------------------------------------------------ parameter vectors
def params_to_vector(p: dict) -> np.ndarray:
    out = []
    for K, D in ((p["K_left"], p["D_left"]), (p["K_right"], p["D_right"])):
        K, D = np.asarray(K, float), np.asarray(D, float).ravel()
        if K.shape != (3, 3) or D.shape != (5,):
            raise ValueError("expected a 3x3 camera matrix and 5 distortion coefficients")
        out += [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D]
    return np.array(out, float)


def vector_to_params(v, baseline_mm=None) -> dict:
    v = np.asarray(v, float)
    p = {}
    for tag, o in (("left", 0), ("right", 9)):
        K = np.array([[v[o], 0, v[o + 2]], [0, v[o + 1], v[o + 3]], [0, 0, 1.0]])
        p["K_" + tag], p["D_" + tag] = K, v[o + 4:o + 9].copy()
    if baseline_mm is not None:
        p["baseline_mm"] = float(baseline_mm)
    return p


def _model_vector(m: dict) -> np.ndarray:
    return params_to_vector(dict(K_left=m["K1"], D_left=m["D1"], K_right=m["K2"], D_right=m["D2"]))


# ------------------------------------------------------------------ 1. loading and validation
@dataclass
class CalibrationInputs:
    pattern: tuple
    square_mm: float
    size_left: tuple
    size_right: tuple
    distortion: str
    left: np.ndarray                          # (n_pairs, N, 2)
    right: np.ndarray
    saved: np.ndarray | None                  # saved 18-vector, or None when not supplied
    observations_sha256: str

    @property
    def n_pairs(self):
        return len(self.left)


def _distortion_model(left, right):
    if not (isinstance(left, str) and isinstance(right, str)):
        raise UncertaintyUnavailable("the stored distortion model is missing; it is never guessed or replaced by 'auto'")
    if left != right:
        raise UncertaintyUnavailable(f"the stored per-camera models differ ({left} / {right}); calibrate_stereo takes one model for both cameras, so it cannot be reproduced")
    if left not in uc.DISTORTION_MODELS:
        raise UncertaintyUnavailable(f"the stored distortion model {left[:30]!r} is not one of {sorted(uc.DISTORTION_MODELS)}")
    return left


def load_inputs(obs: dict, *, distortion_left, distortion_right, saved_params: dict | None = None, config: UncertaintyConfig | None = None) -> CalibrationInputs:
    """Validated observation record + stored distortion model -> arrays.  Raises UncertaintyUnavailable with the reason; never fills anything in."""
    cfg = config or UncertaintyConfig()
    if obs is None:
        raise UncertaintyUnavailable("no calibration observations were saved with this calibration")
    try:
        CO.validate_observations(obs)
        left, right = CO.corner_arrays(obs)
    except CO.ObservationError as ex:
        raise UncertaintyUnavailable(f"the calibration observations are invalid: {ex}") from ex
    except Exception as ex:                                       # wrong container types
        raise UncertaintyUnavailable(f"the calibration observations could not be read ({type(ex).__name__})") from ex
    model = _distortion_model(distortion_left, distortion_right)
    if len(left) < cfg.min_pairs:
        raise UncertaintyUnavailable(f"{len(left)} accepted pairs; at least {cfg.min_pairs} are needed to recalibrate")
    saved = None
    if saved_params is not None:
        try:
            saved = params_to_vector(saved_params)
        except Exception as ex:
            raise UncertaintyUnavailable("the saved calibration parameters are malformed") from ex
        if not np.isfinite(saved).all():
            raise UncertaintyUnavailable("the saved calibration parameters are not finite")
    pat = obs["pattern"]
    return CalibrationInputs((pat["cols"], pat["rows"]), float(obs["square_mm"]), tuple(obs["image_size_left"]), tuple(obs["image_size_right"]), model,
                             left, right, saved, obs["observations_sha256"])


def inputs_from_document(doc: dict, config: UncertaintyConfig | None = None) -> CalibrationInputs:
    """From an ``f8.calibration_result/2`` document: observations validated against the same document's calibration, distortion model and saved K, D taken
    from the document.  A /1 document, or one without observations, is 'unavailable' (the F8.29 status text), never an error to paper over."""
    st = CO.uncertainty_status(doc)
    if not st["available"]:
        raise UncertaintyUnavailable(st["reason"])
    try:
        obs = CO.validate_document_observations(doc)
    except CO.ObservationError as ex:
        raise UncertaintyUnavailable(f"the calibration observations are invalid: {ex}") from ex
    cams = doc.get("cameras") or {}
    L, R = cams.get("left") or {}, cams.get("right") or {}
    saved = dict(K_left=L.get("K"), D_left=L.get("distortion"), K_right=R.get("K"), D_right=R.get("distortion"))
    return load_inputs(obs, distortion_left=L.get("distortion_assessment"), distortion_right=R.get("distortion_assessment"), saved_params=saved, config=config)


# ------------------------------------------------------------------ 2. calibration runs
def _degenerate_reason(m: dict, size_l, size_r) -> str | None:
    """Reasons a recalibration is counted as failed even though the solver returned.  The solver is not modified; this only classifies its output."""
    if m.get("errors"):
        return "solver_reported_errors"
    try:
        v = _model_vector(m)
    except Exception:
        return "unexpected_parameter_layout"
    if not np.isfinite(v).all():
        return "non_finite_parameters"
    for o, size in ((0, size_l), (9, size_r)):
        fx, fy, cx, cy = v[o:o + 4]
        if fx <= 0 or fy <= 0:
            return "non_positive_focal_length"
        if not (0 <= cx <= size[0] and 0 <= cy <= size[1]):
            return "principal_point_outside_image"
    return None


def recalibrate(inputs: CalibrationInputs, idx=None, solver=None):
    """Recalibrate from the stored pairs (or the pairs ``idx``, repeats allowed; left and right use the SAME indices).  Returns (vector, None) or (None, reason)."""
    solver = solver or uc.calibrate_stereo
    idx = np.arange(inputs.n_pairs) if idx is None else np.asarray(idx)
    try:
        m = solver([inputs.left[i] for i in idx], [inputs.right[i] for i in idx], inputs.pattern, inputs.square_mm, inputs.size_left, inputs.size_right,
                   distortion=inputs.distortion)
    except uc.CalibrationFailed:
        return None, "calibration_failed"
    except ValueError:
        return None, "invalid_input"
    except Exception as ex:                                      # cv2.error and anything unexpected is counted, never swallowed silently
        return None, "solver_exception:" + type(ex).__name__
    reason = _degenerate_reason(m, inputs.size_left, inputs.size_right)
    return (None, reason) if reason else (_model_vector(m), None)


def check_reproduction(inputs: CalibrationInputs, config: UncertaintyConfig | None = None, solver=None) -> dict:
    """Full-data recalibration versus the SAVED calibration, by tolerance (digests are not comparable: OpenCV is not bit-reproducible)."""
    cfg = config or UncertaintyConfig()
    if inputs.saved is None:
        return dict(performed=False, passed=False, reason="no saved calibration parameters were supplied to compare with")
    v, why = recalibrate(inputs, solver=solver)
    if v is None:
        return dict(performed=True, passed=False, reason=f"the full-data recalibration failed ({why})")
    d = np.abs(v - inputs.saved)
    kmask = np.array([i % 9 < 4 for i in range(N_PARAMS)])
    max_k, max_d = float(d[kmask].max()), float(d[~kmask].max())
    ok = max_k <= cfg.repro_tol_K_px and max_d <= cfg.repro_tol_D
    return dict(performed=True, passed=bool(ok), max_abs_diff_K_px=max_k, max_abs_diff_D=max_d, tol_K_px=cfg.repro_tol_K_px, tol_D=cfg.repro_tol_D,
                worst_parameter=PARAM_NAMES[int(np.argmax(d))], tolerances_provisional=True,
                reason=None if ok else "the recalibration from the stored observations does not reproduce the saved calibration within tolerance")


# ------------------------------------------------------------------ 3. pair bootstrap
@dataclass
class CalibrationDraws:
    seed: int
    attempted: int
    vectors: np.ndarray                       # (successful, 18) in draw order
    ok_index: list                            # draw numbers of the successes
    failures: list                            # [(draw number, reason)]
    counts: np.ndarray                        # (attempted, n_pairs) how often each pair entered each draw (all attempts)
    seconds: float
    truncated: bool                           # stopped early by max_seconds
    n_pairs: int
    distortion: str
    observations_sha256: str

    @property
    def successes(self):
        return len(self.vectors)

    def failure_reasons(self):
        out = {}
        for _, r in self.failures:
            out[r] = out.get(r, 0) + 1
        return out


def bootstrap_calibration(inputs: CalibrationInputs, config: UncertaintyConfig | None = None, solver=None) -> CalibrationDraws:
    """Resample complete accepted pairs (left+right together, with replacement), recalibrate each resample with the stored model, keep whole vectors."""
    cfg = config or UncertaintyConfig()
    n_req = max(0, min(int(cfg.n_draws), int(cfg.max_draws)))
    rng = np.random.default_rng(cfg.seed)
    n = inputs.n_pairs
    vecs, ok_idx, fails, counts = [], [], [], []
    t0 = time.perf_counter(); truncated = False
    for k in range(n_req):
        if time.perf_counter() - t0 > cfg.max_seconds:
            truncated = True; break
        idx = rng.integers(0, n, n)
        counts.append(np.bincount(idx, minlength=n).astype(np.int32))
        v, why = recalibrate(inputs, idx, solver)
        if v is None:
            fails.append((k, why))
        else:
            vecs.append(v); ok_idx.append(k)
    return CalibrationDraws(cfg.seed, len(counts), np.array(vecs).reshape(-1, N_PARAMS), ok_idx, fails,
                            np.array(counts).reshape(-1, n), time.perf_counter() - t0, truncated, n, inputs.distortion, inputs.observations_sha256)


def draw_summary(draws: CalibrationDraws) -> dict:
    V = draws.vectors
    sd = V.std(0, ddof=1) if len(V) > 1 else np.full(N_PARAMS, np.nan)
    free = [i for i in range(N_PARAMS) if len(V) > 1 and sd[i] > 1e-12]
    corr = None
    if len(free) > 1:
        corr = dict(names=[PARAM_NAMES[i] for i in free], matrix=np.corrcoef(V[:, free].T).round(4).tolist())
    return dict(attempted=draws.attempted, successful=draws.successes, failed=len(draws.failures), failure_fraction=(len(draws.failures) / draws.attempted if draws.attempted else None),
                failure_reasons=draws.failure_reasons(), seed=draws.seed, truncated_by_time_limit=draws.truncated, seconds=draws.seconds,
                distortion_model=draws.distortion, n_pairs=draws.n_pairs, parameter_sd={n: (None if not np.isfinite(s) else float(s)) for n, s in zip(PARAM_NAMES, sd)},
                correlation_free_parameters=corr, note=FIXED_PARAM_NOTE)


# ------------------------------------------------------------------ 4. stability and representativeness
def sd_stability(values, config: UncertaintyConfig | None = None) -> dict:
    """SD of ``values`` at growing sample sizes, and whether it has settled.  SD(n/2) vs SD(n) is a plain repeatability check, not a convergence proof."""
    cfg = config or UncertaintyConfig()
    x = np.asarray(values, float); n = len(x)
    if n < 4:
        return dict(n=n, stable=False, reason="fewer than 4 values", sd=None)
    half_k = max(2, n // 2)
    ks = sorted({max(2, int(round(n * f))) for f in (0.25, 0.75, 1.0)} | {half_k})
    sds = {k: float(x[:k].std(ddof=1)) for k in ks}
    full, half = sds[n], sds[half_k]
    rel = abs(full - half) / full if full > 0 else (0.0 if half == 0 else float("inf"))
    return dict(n=n, sd=full, sd_by_n={str(k): v for k, v in sds.items()}, rel_change_half_to_full=float(rel), rel_standard_error_normal_approx=float(1 / math.sqrt(2 * (n - 1))),
                stable=bool(rel <= cfg.stability_rel_tol), tolerance=cfg.stability_rel_tol, tolerance_provisional=True)


def parameter_stability(draws: CalibrationDraws, config: UncertaintyConfig | None = None) -> dict:
    out = {}
    for i, name in enumerate(PARAM_NAMES):
        col = draws.vectors[:, i] if draws.successes else np.array([])
        if len(col) > 3 and col.std(ddof=1) > 1e-12:
            out[name] = sd_stability(col, config)
    return out


def representativeness(draws: CalibrationDraws, config: UncertaintyConfig | None = None) -> dict:
    """Do the SUCCESSFUL draws still use every pair about as often as chance would?  A pair that is systematically over- or under-represented among successes
    means failures depend on which pairs were drawn, so the successes are not a fair resample.  Statistic: per-pair mean inclusion count (expected 1) in standard
    errors of a with-replacement draw (variance 1 - 1/n per count)."""
    cfg = config or UncertaintyConfig()
    ok = draws.counts[draws.ok_index] if draws.successes else np.zeros((0, draws.n_pairs))
    n_ok, n = len(ok), draws.n_pairs
    if n_ok < 2 or n < 2:
        return dict(assessed=False, representative=False, reason="too few successful draws")
    se = math.sqrt((1 - 1 / n) / n_ok)
    z = np.abs(ok.mean(0) - 1.0) / se
    return dict(assessed=True, max_abs_z=float(z.max()), worst_pair_index=int(z.argmax()), threshold_z=cfg.representativeness_z, threshold_provisional=True,
                representative=bool(z.max() <= cfg.representativeness_z))


# ------------------------------------------------------------------ 5. propagation (estimator callable)
def _eval_estimator(estimator, params: dict):
    """-> ((width, height), None) or (None, reason).  Never raises for an estimator failure."""
    try:
        r = estimator(params)
    except EstimatorFailure as ex:
        return None, ex.code
    except Exception as ex:
        return None, "estimator_exception:" + type(ex).__name__
    if not isinstance(r, dict) or "width_mm" not in r or "height_mm" not in r:
        return None, "estimator_output_malformed"
    if r.get("degenerate"):
        return None, "estimator_degenerate"
    try:
        w, h = float(r["width_mm"]), float(r["height_mm"])
    except (TypeError, ValueError):
        return None, "estimator_output_malformed"
    if not (math.isfinite(w) and math.isfinite(h)):
        return None, "non_finite_dimensions"
    if w <= 0 or h <= 0:
        return None, "non_positive_dimensions"
    return (w, h), None


@dataclass
class Propagation:
    method: str
    attempted: int
    outputs: np.ndarray                       # (successful, 2) width, height
    failures: list                            # [(label, reason)]
    seconds: float
    detail: dict = field(default_factory=dict)

    def failure_reasons(self):
        out = {}
        for _, r in self.failures:
            out[r] = out.get(r, 0) + 1
        return out


def propagate_monte_carlo(estimator, draws: CalibrationDraws, baseline_mm, config: UncertaintyConfig | None = None) -> Propagation:
    """Run the estimator once per successful calibration draw (the draws that already failed are accounted for by ``draws``, not re-run)."""
    t0 = time.perf_counter(); rows, fails = [], []
    for j, v in zip(draws.ok_index, draws.vectors):
        r, why = _eval_estimator(estimator, vector_to_params(v, baseline_mm))
        if r is None:
            fails.append((j, why))
        else:
            rows.append(r)
    return Propagation("monte_carlo", len(draws.vectors), np.array(rows).reshape(-1, 2), fails, time.perf_counter() - t0)


def sigma_points(center: np.ndarray, vectors: np.ndarray, rank: int):
    """Symmetric points center +/- sqrt(r) * sqrt(lambda_k) * v_k over the top ``rank`` principal directions of the draw covariance.

    Equal-weight points of this form reproduce the output variance exactly for a linear map when every retained direction is used (and the covariance has
    that rank); for a nonlinear estimator it is an approximation whose adequacy must be tested, never assumed."""
    C = np.cov(vectors.T, ddof=1) if len(vectors) > 1 else np.zeros((N_PARAMS, N_PARAMS))
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]; w, V = np.clip(w[order], 0, None), V[:, order]
    r = int(min(rank, (w > 1e-12 * max(w[0], 1e-300)).sum())) if w[0] > 0 else 0
    pts = []
    for k in range(r):
        step = math.sqrt(r * w[k]) * V[:, k]
        pts += [center + step, center - step]
    return pts, dict(rank_used=r, eigenvalues_top=w[:max(r, 1)].tolist(), variance_fraction_captured=float(w[:r].sum() / w.sum()) if w.sum() > 0 else None)


def propagate_sigma_points(estimator, draws: CalibrationDraws, center: np.ndarray, baseline_mm, config: UncertaintyConfig | None = None) -> Propagation:
    cfg = config or UncertaintyConfig()
    t0 = time.perf_counter()
    pts, info = sigma_points(np.asarray(center, float), draws.vectors, cfg.sigma_rank)
    rows, fails = [], []
    c, why = _eval_estimator(estimator, vector_to_params(center, baseline_mm))
    for i, p in enumerate(pts):
        r, why = _eval_estimator(estimator, vector_to_params(p, baseline_mm))
        if r is None:
            fails.append((i, why))
        else:
            rows.append(r)
    X = np.array(rows).reshape(-1, 2)
    info.update(center_ok=c is not None, center_output=None if c is None else list(c), runs=len(pts) + 1)
    if c is not None and len(X) == len(pts) and len(pts):                 # curvature diagnostic: the +/- pairs should average to the centre if linear
        info["nonlinearity_abs_mm"] = np.abs(X.reshape(-1, 2, 2).mean(1).mean(0) - np.array(c)).tolist()
    return Propagation("sigma_points", len(pts), X, fails, time.perf_counter() - t0, info)


def spread(P: Propagation, center=None, config: UncertaintyConfig | None = None) -> dict:
    """Width/height spread of a propagation.  Monte Carlo: sample SD about the sample mean.  Sigma points: sqrt(mean squared deviation from the centre run)
    (each +/- pair is a draw of the linearised map, so this is the matching estimator of the SD)."""
    n = len(P.outputs)
    d = dict(method=P.method, attempted=P.attempted, successful=n, failed=len(P.failures), failure_fraction=(len(P.failures) / P.attempted if P.attempted else None),
             failure_reasons=P.failure_reasons(), seconds=P.seconds)
    if n < 2:
        d.update(width_sd_mm=None, height_sd_mm=None); return d
    if P.method == "sigma_points":
        c = np.asarray(P.detail["center_output"], float) if P.detail.get("center_ok") else None
        if c is None or len(P.failures):
            d.update(width_sd_mm=None, height_sd_mm=None, note="sigma-point set incomplete (centre or a point failed); no spread reported"); return d
        sd = np.sqrt(((P.outputs - c) ** 2).mean(0)); mean = P.outputs.mean(0)
    else:
        sd = P.outputs.std(0, ddof=1); mean = P.outputs.mean(0)
    d.update(width_sd_mm=float(sd[0]), height_sd_mm=float(sd[1]), width_mean_mm=float(mean[0]), height_mean_mm=float(mean[1]))
    if P.method == "monte_carlo":
        d["stability"] = dict(width=sd_stability(P.outputs[:, 0], config), height=sd_stability(P.outputs[:, 1], config))
    return d


def compare_approximation(mc: dict, sp: dict, config: UncertaintyConfig | None = None) -> dict:
    """Does the sigma-point spread reproduce the Monte Carlo spread for THIS case?  Valid only for the case tested."""
    cfg = config or UncertaintyConfig()
    out = dict(tolerance=cfg.approx_rel_tol, tolerance_provisional=True, scope="this synthetic case only; says nothing about real Method J reconstruction")
    for k in ("width", "height"):
        a, b = mc.get(f"{k}_sd_mm"), sp.get(f"{k}_sd_mm")
        out[k] = dict(mc_sd_mm=a, approx_sd_mm=b, rel_error=(abs(b - a) / a if (a and b is not None) else None))
    errs = [out[k]["rel_error"] for k in ("width", "height")]
    out["adequate"] = bool(all(e is not None and e <= cfg.approx_rel_tol for e in errs))
    out["runtime_ratio_mc_over_approx"] = (mc["seconds"] / sp["seconds"]) if sp.get("seconds") else None
    out["runs_mc"], out["runs_approx"] = mc.get("successful"), sp.get("attempted")
    return out


# ------------------------------------------------------------------ 6. sensitivity (deterministic, assumed perturbations)
def sensitivity_analysis(estimator, base_params: dict, perturbations: dict) -> dict:
    """``perturbations``: name -> (description, function(params) -> params).  Each is a deterministic what-if with an ASSUMED magnitude.  Result is labelled
    Sensitivity-based; it is a change in output, not a measured uncertainty, and is never combined with measured spread."""
    base, why = _eval_estimator(estimator, base_params)
    out = dict(class_=SENSITIVITY, base_ok=base is not None, base=None if base is None else dict(width_mm=base[0], height_mm=base[1]), cases={},
               note="Assumed perturbation magnitudes. Not measured. Not a probability. Not added to measured components.")
    if base is None:
        out["base_failure"] = why; return out
    for name, (desc, fn) in perturbations.items():
        try:
            r, why = _eval_estimator(estimator, fn(dict(base_params)))
        except Exception as ex:
            r, why = None, "perturbation_exception:" + type(ex).__name__
        out["cases"][name] = (dict(assumed_perturbation=desc, failed=True, reason=why) if r is None else
                              dict(assumed_perturbation=desc, width_mm=r[0], height_mm=r[1], d_width_mm=r[0] - base[0], d_height_mm=r[1] - base[1]))
    return out


# ------------------------------------------------------------------ 7. uncertainty classification and report
NOT_QUANTIFIED_ALWAYS = [
    "systematic camera or lens-model error (resampling cannot see it)",
    "focus, zoom or stabilisation differences between calibration and room recordings",
    "correlation between consecutive calibration stills (pairs are treated as independent; the true spread is probably larger)",
    "variability of which pairs were selected (only the used pairs are stored)",
    "non-Gaussian and correlated corner-detection error",
    "boundary-annotation, floor-plane, registration and match-stability components (not part of F8.30B)",
    "generalisation beyond equipment on which Method J has been run",
]


def classify_calibration_uncertainty(summary: dict, mc: dict | None, reproduction: dict | None, representativeness_: dict | None, param_stability: dict | None,
                                     config: UncertaintyConfig | None = None) -> dict:
    """-> dict(status in {complete, incomplete, unavailable}, reasons).  'complete' means only that every provisional check passed; it is not validation."""
    cfg = config or UncertaintyConfig()
    reasons, unavailable = [], False
    if reproduction is not None and cfg.require_reproduction and not reproduction.get("passed"):
        unavailable = True; reasons.append("reproduction of the saved calibration failed: " + str(reproduction.get("reason")))
    if summary["successful"] < 2:
        unavailable = True; reasons.append("fewer than 2 successful calibration draws")
    if unavailable:
        return dict(status="unavailable", reasons=reasons)
    if summary["successful"] < cfg.min_successful_draws:
        reasons.append(f"only {summary['successful']} successful draws (provisional minimum {cfg.min_successful_draws})")
    ff = summary["failure_fraction"]
    if ff is not None and ff > cfg.max_failure_fraction:
        reasons.append(f"failure fraction {ff:.2f} exceeds the provisional limit {cfg.max_failure_fraction}")
    if summary.get("truncated_by_time_limit"):
        reasons.append("the bootstrap stopped at its time limit before the requested number of draws")
    if representativeness_ is not None and not representativeness_.get("representative"):
        reasons.append("successful draws are not representative of the pair resampling: " + str(representativeness_.get("reason") or f"max z {representativeness_.get('max_abs_z'):.1f}"))
    if mc is not None:
        if mc["successful"] < 2:
            reasons.append("fewer than 2 successful propagated draws"); return dict(status="unavailable", reasons=reasons + ["no propagated spread exists"])
        if mc["failure_fraction"] > cfg.max_failure_fraction:
            reasons.append(f"propagation failure fraction {mc['failure_fraction']:.2f} exceeds the provisional limit {cfg.max_failure_fraction}")
        for k in ("width", "height"):
            if not mc.get("stability", {}).get(k, {}).get("stable"):
                reasons.append(f"{k} SD has not settled across draw counts")
    if param_stability:
        unstable = sorted(n for n, s in param_stability.items() if not s["stable"])
        if unstable:
            reasons.append("calibration parameter spread not settled for: " + ", ".join(unstable) + " (reported, not hidden)")
    return dict(status="incomplete" if reasons else "complete", reasons=reasons)


def combine_measured(components: list, *, independence_justification: str | None = None) -> dict:
    """Quadrature combination of Measured components, refused unless the caller states why independence holds.  Sensitivity-based and unavailable parts are
    never accepted.  With one component there is nothing to combine."""
    meas = [c for c in components if c.get("class") == MEASURED]
    if len(meas) != len(components):
        return dict(combined=None, reason="only Measured components can be combined; Sensitivity-based and Not quantified parts are excluded")
    if any(c.get("status") != "complete" for c in meas):
        return dict(combined=None, reason="a Measured component is incomplete or unavailable; the combined estimate would be incomplete")
    if len(meas) < 2:
        return dict(combined=None, reason="a single Measured component: nothing to combine")
    if not independence_justification:
        return dict(combined=None, reason="independence of the components is not established; quadrature is refused (joint resampling through the estimator is preferred)")
    out = {}
    for k in ("width_sd_mm", "height_sd_mm"):
        out[k] = float(math.sqrt(sum(c[k] ** 2 for c in meas)))
    return dict(combined=out, method="quadrature", assumption=independence_justification, reason=None)


def build_report(*, summary, mc, approx_comparison, classification, reproduction, representativeness_, sensitivity=None, config: UncertaintyConfig, sigma=None) -> dict:
    """Class-labelled result block.  Wired to nothing live."""
    status = classification["status"]
    measured = dict(**{"class": MEASURED}, name="calibration parameter uncertainty propagated to width and height", status=status, reasons=classification["reasons"],
                    method="pair-preserving bootstrap of stored accepted checkerboard pairs, full recalibration with the stored distortion model, propagation of whole parameter vectors",
                    draws=dict(attempted=summary["attempted"], successful=summary["successful"], failed=summary["failed"], failure_fraction=summary["failure_fraction"],
                               failure_reasons=summary["failure_reasons"], seed=summary["seed"]),
                    limitations=["sampling variability only", "assumes independent pairs", "synthetic or real validation of coverage has not been performed"])
    if mc is not None and status != "unavailable":
        measured.update(width_sd_mm=mc.get("width_sd_mm"), height_sd_mm=mc.get("height_sd_mm"), propagation=mc)
    if status == "unavailable":
        measured.update(width_sd_mm=None, height_sd_mm=None)
    components = [measured]
    if sensitivity is not None:
        components.append(dict(sensitivity, **{"class": SENSITIVITY}))
    components.append({"class": NOT_QUANTIFIED, "items": list(NOT_QUANTIFIED_ALWAYS) + ([] if status != "unavailable" else ["calibration uncertainty: " + "; ".join(classification["reasons"])])})
    return dict(schema=SCHEMA, label=LABEL, claims=dict(confidence_interval=False, coverage_probability=None,
                                                         statement="Indicative uncertainty: repeatability under input resampling. Not a validated interval and no coverage is claimed."),
                combined_estimate_complete=False,
                combined_estimate_note="Only one Measured source exists in F8.30B (calibration); other Measured sources are not computed here, so the combined estimate is incomplete.",
                components=components, reproduction=reproduction, representativeness=representativeness_, approximation=approx_comparison, sigma_points=sigma,
                config=config.describe())


def estimate_calibration_uncertainty(inputs: CalibrationInputs, estimator, baseline_mm: float, config: UncertaintyConfig | None = None, *, solver=None,
                                     with_approximation: bool = True, sensitivity: dict | None = None) -> dict:
    """End to end for one visitor calibration.  Returns the class-labelled report; raises nothing for an estimable-or-not case (unavailable is a result)."""
    cfg = config or UncertaintyConfig()
    repro = check_reproduction(inputs, cfg, solver)
    if cfg.require_reproduction and not repro["passed"]:
        empty = CalibrationDraws(cfg.seed, 0, np.zeros((0, N_PARAMS)), [], [], np.zeros((0, inputs.n_pairs), np.int32), 0.0, False, inputs.n_pairs, inputs.distortion, inputs.observations_sha256)
        s = draw_summary(empty)
        return build_report(summary=s, mc=None, approx_comparison=None, classification=classify_calibration_uncertainty(s, None, repro, None, None, cfg), reproduction=repro,
                            representativeness_=None, sensitivity=sensitivity, config=cfg)
    draws = bootstrap_calibration(inputs, cfg, solver)
    s = draw_summary(draws)
    center = inputs.saved if inputs.saved is not None else None
    if center is None:
        v, _ = recalibrate(inputs, solver=solver); center = v
    rep, ps = representativeness(draws, cfg), parameter_stability(draws, cfg)
    mc = spread(propagate_monte_carlo(estimator, draws, baseline_mm, cfg), config=cfg) if draws.successes else None
    sp = cmp = None
    if with_approximation and draws.successes > 1 and center is not None:
        P = propagate_sigma_points(estimator, draws, center, baseline_mm, cfg); sp = spread(P, config=cfg)
        sp["detail"] = P.detail
        cmp = compare_approximation(mc, sp, cfg) if mc and mc.get("width_sd_mm") is not None else None
    cls = classify_calibration_uncertainty(s, mc, repro, rep, ps, cfg)
    rpt = build_report(summary=s, mc=mc, approx_comparison=cmp, classification=cls, reproduction=repro, representativeness_=rep, sensitivity=sensitivity, config=cfg, sigma=sp)
    rpt["parameter_stability_unsettled"] = sorted(n for n, x in ps.items() if not x["stable"])
    rpt["draw_summary"] = s
    return rpt


# ------------------------------------------------------------------ 8. measurement outcome policy (pure; not wired to the visitor backend)
SUCCESS, QUESTIONABLE, FAILED = "SUCCESS", "QUESTIONABLE", "FAILED"

# Codes are the CALLER's vocabulary; this classifier does not recompute any Method J guard and does not change one.
#   hard failure  -> no valid geometric solution (examples: no floor plane, wall corner not resolved, registration impossible, non-positive depth)
#   quality flag  -> a solution exists but a check raised a concern (examples: pose_degenerate, low inlier fraction, large robust scale, sparse registration)


def classify_measurement_outcome(width_mm, height_mm, *, hard_failures=(), quality_flags=(), uncertainty_status: str | None = None) -> dict:
    """SUCCESS / QUESTIONABLE / FAILED.  Estimates are passed through unmodified; FAILED never carries dimensions.

    ``hard_failures``: codes (or {code, message}) meaning no valid geometric solution exists.
    ``quality_flags``: codes (or {code, message}) for concerns about an otherwise valid solution.
    ``uncertainty_status`` is echoed; an incomplete or unavailable uncertainty does not by itself make a measurement QUESTIONABLE."""
    def norm(items):
        out = []
        for it in items:
            out.append(dict(it) if isinstance(it, dict) else dict(code=str(it), message=""))
        return out
    hard, flags = norm(hard_failures), norm(quality_flags)
    reasons = []
    for name, v in (("width", width_mm), ("height", height_mm)):
        ok = isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(float(v))
        if not ok:
            reasons.append(dict(code=f"{name}_not_finite", message=f"{name} is missing or not finite"))
        elif float(v) <= 0:
            reasons.append(dict(code=f"{name}_not_positive", message=f"{name} is not positive"))
    fail = hard + reasons
    base = dict(schema="f8.measurement_outcome/1", uncertainty_status=uncertainty_status)
    if fail:
        return dict(base, outcome=FAILED, width_mm=None, height_mm=None, diagnostics=fail, warnings=flags,
                    note="no valid geometric solution; no dimensions are reported")
    w, h = float(width_mm), float(height_mm)
    if flags:
        return dict(base, outcome=QUESTIONABLE, width_mm=w, height_mm=h, diagnostics=[], warnings=flags,
                    note="finite, geometrically valid dimensions exist; estimates are unmodified and the warnings must be shown with them")
    return dict(base, outcome=SUCCESS, width_mm=w, height_mm=h, diagnostics=[], warnings=[], note="valid dimensions, no quality concerns raised")
