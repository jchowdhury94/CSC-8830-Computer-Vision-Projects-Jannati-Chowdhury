"""F8.30B: calibration bootstrap, synthetic propagation, uncertainty classification and the measurement-outcome policy.

SYNTHETIC fixtures only: corners are rendered from known cameras; no video, no .MOV, no real calibration, no Method J, no experiment constant."""
import ast
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2]
for sub in ("", "f8_chunk_poc"):
    sys.path.insert(0, str(MODULE / sub) if sub else str(MODULE))

from room_measurement import visitor_uncertainty as V                       # noqa: E402
from room_measurement import visitor_uncertainty_synthetic as S             # noqa: E402

UC = V.uc
CFG = lambda **kw: V.UncertaintyConfig(**{**dict(n_draws=40), **kw})


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def scene():
    rig, obs, model, doc = S.make_calibration(n_pairs=24, seed=7, model="radial3", tilt_deg=35.0, noise_px=0.1)
    return dict(rig=rig, obs=obs, model=model, doc=doc, inputs=V.inputs_from_document(doc), est=S.SyntheticWallEstimator(rig))


@pytest.fixture(scope="module")
def draws(scene):
    return V.bootstrap_calibration(scene["inputs"], CFG(n_draws=60))


# ------------------------------------------------------------------ 1. loading and validation
def test_document_loads_with_stored_model_and_saved_parameters(scene):
    i = scene["inputs"]
    assert i.n_pairs == 24 and i.distortion == "radial3" and i.saved is not None and i.left.shape == (24, 48, 2) and i.right.shape == i.left.shape
    assert i.pattern == (8, 6) and i.square_mm == 25.0


def test_schema_v1_or_missing_observations_are_unavailable_not_invented(scene):
    d1 = copy.deepcopy(scene["doc"]); d1["schema"] = "f8.calibration_result/1"; d1.pop("calibration_observations")
    with pytest.raises(V.UncertaintyUnavailable, match="schema /1"):
        V.inputs_from_document(d1)
    d2 = copy.deepcopy(scene["doc"]); d2.pop("calibration_observations"); d2["calibration_observations_status"] = dict(reason="benchmark job")
    with pytest.raises(V.UncertaintyUnavailable, match="benchmark job"):
        V.inputs_from_document(d2)
    with pytest.raises(V.UncertaintyUnavailable, match="no calibration observations"):
        V.load_inputs(None, distortion_left="radial3", distortion_right="radial3")


@pytest.mark.parametrize("mutate", [
    lambda o: o["pairs"][0]["left_corners"].pop(),                                   # wrong corner count
    lambda o: o["pairs"][0]["right_corners"].__setitem__(3, [float("nan"), 1.0]),    # non-finite corner
    lambda o: o["pairs"][1].pop("right_corners"),                                    # missing field
    lambda o: o.__setitem__("n_pairs", 3),                                           # inconsistent count
    lambda o: o["pairs"][0]["left_corners"].__setitem__(0, [5.0, 5.0]),              # edited corner -> digest mismatch
    lambda o: o.__setitem__("schema", "f8.calibration_observations/9"),
])
def test_malformed_observations_are_rejected(scene, mutate):
    o = copy.deepcopy(scene["obs"]); mutate(o)
    with pytest.raises(V.UncertaintyUnavailable, match="invalid"):
        V.load_inputs(o, distortion_left="radial3", distortion_right="radial3")


def test_tampered_document_is_rejected(scene):
    d = copy.deepcopy(scene["doc"]); d["cameras"]["left"]["K"][0][0] += 5.0           # calibration no longer matches its observations
    with pytest.raises(V.UncertaintyUnavailable, match="invalid"):
        V.inputs_from_document(d)


def test_too_few_pairs_is_unavailable(scene):
    o = copy.deepcopy(scene["obs"]); o["pairs"] = o["pairs"][:5]; o["n_pairs"] = 5; o["observations_sha256"] = V.CO.pairs_digest(o["pairs"])
    with pytest.raises(V.UncertaintyUnavailable, match="at least 8"):
        V.load_inputs(o, distortion_left="radial3", distortion_right="radial3")


# ------------------------------------------------------------------ 2. fixed distortion model
@pytest.mark.parametrize("l,r,msg", [("auto", "auto", "not one of"), (None, None, "missing"), ("radial2", "radial3", "differ"), ("bogus", "bogus", "not one of")])
def test_distortion_model_must_be_stored_explicit_and_consistent(scene, l, r, msg):
    with pytest.raises(V.UncertaintyUnavailable, match=msg):
        V.load_inputs(scene["obs"], distortion_left=l, distortion_right=r)


def test_every_recalibration_uses_the_stored_model_never_auto(scene):
    seen = []

    def spy(*a, **k):
        seen.append(k.get("distortion")); return UC.calibrate_stereo(*a, **k)
    V.bootstrap_calibration(scene["inputs"], CFG(n_draws=3), solver=spy)
    V.check_reproduction(scene["inputs"], solver=spy)
    assert seen and set(seen) == {"radial3"}


# ------------------------------------------------------------------ 3. pair-preserving bootstrap
def test_resampling_keeps_left_right_pairing_and_draws_with_replacement(scene):
    inp, calls = scene["inputs"], []

    def spy(left, right, *a, **k):
        calls.append((left, right)); return UC.calibrate_stereo(left, right, *a, **k)
    V.bootstrap_calibration(inp, CFG(n_draws=5), solver=spy)
    assert len(calls) == 5
    repeats = 0
    for left, right in calls:
        assert len(left) == len(right) == inp.n_pairs
        used = []
        for l, r in zip(left, right):                                           # the i-th left board is paired with ITS right board, never another
            i = next(j for j in range(inp.n_pairs) if np.array_equal(inp.left[j], l))
            assert np.array_equal(inp.right[i], r)
            used.append(i)
        repeats += inp.n_pairs - len(set(used))
    assert repeats > 0, "with-replacement resampling should repeat pairs"


def test_draw_vector_equals_a_recalibration_of_its_own_resample(scene, draws):
    """Whole calibration solutions are kept: draw 0 is exactly the recalibration of the pair indices the seed produces for draw 0."""
    inp = scene["inputs"]
    idx = np.random.default_rng(draws.seed).integers(0, inp.n_pairs, inp.n_pairs)
    v, why = V.recalibrate(inp, idx)
    assert why is None and draws.ok_index[0] == 0
    assert np.allclose(v, draws.vectors[0], atol=1e-4, rtol=0)
    assert np.array_equal(draws.counts[0], np.bincount(idx, minlength=inp.n_pairs))


# ------------------------------------------------------------------ 4. correlations preserved
def test_whole_solutions_keep_parameter_correlations_that_independent_perturbation_would_lose(draws):
    V_ = draws.vectors; ix = {n: i for i, n in enumerate(V.PARAM_NAMES)}
    c = lambda a, b, M=V_: np.corrcoef(M[:, ix[a]], M[:, ix[b]])[0, 1]
    assert abs(c("L_d0", "L_d1")) > 0.6 and abs(c("L_d1", "L_d4")) > 0.6          # radial terms are strongly coupled
    rng = np.random.default_rng(0); shuffled = np.column_stack([rng.permutation(V_[:, j]) for j in range(V_.shape[1])])
    assert abs(c("L_d1", "L_d4", shuffled)) < 0.4                                   # per-parameter independent sampling destroys it
    s = V.draw_summary(draws)
    assert s["correlation_free_parameters"] is not None and "L_d2" not in s["correlation_free_parameters"]["names"]      # constant tangential columns excluded


# ------------------------------------------------------------------ 5. full-data recalibration vs saved
def test_full_data_recalibration_reproduces_saved_calibration(scene):
    r = V.check_reproduction(scene["inputs"])
    assert r["performed"] and r["passed"] and r["max_abs_diff_K_px"] < 1e-3 and r["max_abs_diff_D"] < 1e-6 and r["tolerances_provisional"] is True


def test_reproduction_failure_is_detected_by_tolerance_and_blocks_the_estimate(scene):
    inp = copy.copy(scene["inputs"]); saved = inp.saved.copy(); saved[0] += 1.0; inp.saved = saved       # saved fx off by 1 px
    r = V.check_reproduction(inp)
    assert not r["passed"] and r["worst_parameter"] == "L_fx" and r["max_abs_diff_K_px"] > 0.9
    assert V.check_reproduction(inp, CFG(repro_tol_K_px=2.0))["passed"]                # configurable
    rpt = V.estimate_calibration_uncertainty(inp, scene["est"], scene["rig"].baseline_mm, CFG(n_draws=5))
    assert rpt["components"][0]["status"] == "unavailable" and rpt["components"][0]["width_sd_mm"] is None
    assert any("reproduction" in x for x in rpt["components"][0]["reasons"])


def test_missing_saved_parameters_means_reproduction_not_performed(scene):
    inp = copy.copy(scene["inputs"]); inp.saved = None
    r = V.check_reproduction(inp)
    assert r["performed"] is False and r["passed"] is False


# ------------------------------------------------------------------ 6. seeded reproducibility
def test_same_seed_same_draws_different_seed_different_draws(scene):
    # resample indices are bit-identical; solver outputs agree to OpenCV's own run-to-run noise (~5e-7 measured, larger under load), hence atol 1e-4
    a = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=6, seed=5))
    b = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=6, seed=5))
    c = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=6, seed=6))
    assert np.array_equal(a.counts, b.counts) and np.allclose(a.vectors, b.vectors, atol=1e-4, rtol=0)
    assert not np.array_equal(a.counts, c.counts) and a.seed == 5 and c.seed == 6


def test_seed_is_recorded_in_report(scene):
    rpt = V.estimate_calibration_uncertainty(scene["inputs"], scene["est"], scene["rig"].baseline_mm, CFG(n_draws=8, seed=77))
    assert rpt["components"][0]["draws"]["seed"] == 77 and rpt["config"]["seed"] == 77


def test_draw_count_and_runtime_are_bounded(scene):
    d = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=10 ** 6, max_draws=4), solver=lambda *a, **k: UC.calibrate_stereo(*a, **k))
    assert d.attempted == 4
    t = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=50, max_seconds=0.0))
    assert t.truncated and t.attempted < 50
    s = V.draw_summary(t); assert s["truncated_by_time_limit"] is True


# ------------------------------------------------------------------ 7. failed-draw accounting
def _flaky_solver(fail_plan):
    state = dict(n=0)

    def solver(left, right, *a, **k):
        i = state["n"]; state["n"] += 1
        kind = fail_plan.get(i)
        if kind == "fail":
            raise UC.CalibrationFailed("x")
        if kind == "value":
            raise ValueError("x")
        if kind == "cv":
            import cv2
            raise cv2.error("boom")
        m = UC.calibrate_stereo(left, right, *a, **k)
        if kind == "errors":
            m["errors"] = ["bad"]
        if kind == "nan":
            m["K1"] = m["K1"] * np.nan
        if kind == "pp":
            m["K2"] = m["K2"].copy(); m["K2"][0, 2] = -50.0
        return m
    return solver


def test_failed_draws_are_counted_with_reasons_never_dropped_silently(scene):
    plan = {0: "fail", 1: "value", 2: "cv", 3: "errors", 4: "nan", 5: "pp", 6: "fail"}
    d = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=12), solver=_flaky_solver(plan))
    s = V.draw_summary(d)
    assert s["attempted"] == 12 and s["failed"] == 7 and s["successful"] == 5 and s["attempted"] == s["failed"] + s["successful"]
    assert s["failure_reasons"] == {"calibration_failed": 2, "invalid_input": 1, "solver_exception:error": 1, "solver_reported_errors": 1, "non_finite_parameters": 1, "principal_point_outside_image": 1}
    assert s["failure_fraction"] == pytest.approx(7 / 12) and d.counts.shape[0] == 12                  # counts kept for failed draws too
    assert [j for j, _ in d.failures] == [0, 1, 2, 3, 4, 5, 6]


def test_estimator_failures_are_counted_in_propagation(scene, draws):
    rig = scene["rig"]
    thr = np.median(draws.vectors[:, 0])
    est = S.SyntheticWallEstimator(rig, fail_when=lambda p: "focal_out_of_range" if p["K_left"][0, 0] > thr else None)
    P = V.propagate_monte_carlo(est, draws, rig.baseline_mm)
    sp = V.spread(P)
    assert sp["attempted"] == draws.successes and sp["successful"] + sp["failed"] == sp["attempted"] and sp["failed"] > 10
    assert set(sp["failure_reasons"]) == {"focal_out_of_range"} and sp["failure_fraction"] == pytest.approx(sp["failed"] / sp["attempted"])


@pytest.mark.parametrize("out,reason", [
    (dict(width_mm=float("nan"), height_mm=1.0), "non_finite_dimensions"), (dict(width_mm=-1.0, height_mm=1.0), "non_positive_dimensions"),
    ({"width_mm": 1.0}, "estimator_output_malformed"), (dict(width_mm=1.0, height_mm=1.0, degenerate=True), "estimator_degenerate"), ("junk", "estimator_output_malformed")])
def test_estimator_output_is_validated(draws, out, reason):
    P = V.propagate_monte_carlo(lambda p: out, draws, 100.0)
    assert P.failure_reasons() == {reason: draws.successes} and len(P.outputs) == 0


def test_estimator_exception_is_counted_not_raised(draws):
    def boom(p):
        raise RuntimeError("x")
    P = V.propagate_monte_carlo(boom, draws, 100.0)
    assert P.failure_reasons() == {"estimator_exception:RuntimeError": draws.successes}


# ------------------------------------------------------------------ 8. stability reporting
def test_sd_stability_distinguishes_settled_from_unsettled():
    rng = np.random.default_rng(1)
    good = V.sd_stability(rng.normal(0, 1, 400))
    assert good["stable"] and good["rel_change_half_to_full"] < 0.15 and set(good["sd_by_n"]) >= {"100", "200", "300", "400"}
    drifting = np.r_[rng.normal(0, 1, 100), rng.normal(0, 8, 100)]                    # SD changes as draws accumulate
    assert not V.sd_stability(drifting)["stable"]
    assert not V.sd_stability([1.0, 2.0])["stable"]
    assert V.sd_stability(rng.normal(0, 1, 50))["tolerance_provisional"] is True


def test_report_lists_unsettled_parameters(scene):
    rpt = V.estimate_calibration_uncertainty(scene["inputs"], scene["est"], scene["rig"].baseline_mm, CFG(n_draws=60))
    assert "parameter_stability_unsettled" in rpt and isinstance(rpt["parameter_stability_unsettled"], list)
    assert rpt["components"][0]["propagation"]["stability"]["width"]["n"] == 60


# ------------------------------------------------------------------ 9. sigma points versus Monte Carlo
def test_sigma_points_are_exact_for_a_linear_estimator(draws):
    rng = np.random.default_rng(3); center = draws.vectors.mean(0)
    est = S.linear_estimator(rng.normal(0, 1, V.N_PARAMS), rng.normal(0, 1, V.N_PARAMS), center)
    cfg = V.UncertaintyConfig(sigma_rank=V.N_PARAMS)
    mc = V.spread(V.propagate_monte_carlo(est, draws, 100.0)); sp = V.spread(V.propagate_sigma_points(est, draws, center, 100.0, cfg))
    cmp = V.compare_approximation(mc, sp, cfg)
    assert cmp["width"]["rel_error"] < 1e-6 and cmp["height"]["rel_error"] < 1e-6 and cmp["adequate"]
    assert sp["attempted"] == 2 * 9 or sp["attempted"] > 0


def test_sigma_points_vs_monte_carlo_on_synthetic_wall_records_everything(scene, draws):
    inp, rig = scene["inputs"], scene["rig"]; cfg = V.UncertaintyConfig(sigma_rank=4)
    P_mc = V.propagate_monte_carlo(scene["est"], draws, rig.baseline_mm); P_sp = V.propagate_sigma_points(scene["est"], draws, inp.saved, rig.baseline_mm, cfg)
    mc, sp = V.spread(P_mc, config=cfg), V.spread(P_sp, config=cfg); cmp = V.compare_approximation(mc, sp, cfg)
    assert mc["successful"] == 60 and sp["attempted"] == 8 and P_sp.detail["runs"] == 9 and P_sp.detail["rank_used"] == 4
    assert 0.5 < P_sp.detail["variance_fraction_captured"] <= 1.0 and mc["seconds"] > 0 and sp["seconds"] > 0
    assert cmp["runs_mc"] == 60 and cmp["runs_approx"] == 8 and cmp["runtime_ratio_mc_over_approx"] > 1
    # this benign case: the cheap approximation reproduces the Monte Carlo spread within the (provisional) tolerance
    assert cmp["width"]["rel_error"] < cfg.approx_rel_tol and cmp["height"]["rel_error"] < cfg.approx_rel_tol and cmp["adequate"]
    assert "synthetic case only" in cmp["scope"]


def test_sigma_points_are_flagged_inadequate_when_the_estimator_is_strongly_nonlinear():
    rig, obs, model, doc = S.make_calibration(n_pairs=12, seed=7, model="radial3", tilt_deg=10.0, noise_px=0.3)           # poorly constrained calibration
    inp = V.inputs_from_document(doc); est = S.SyntheticWallEstimator(rig); cfg = CFG(n_draws=60)
    d = V.bootstrap_calibration(inp, cfg)
    mc = V.spread(V.propagate_monte_carlo(est, d, rig.baseline_mm), config=cfg); sp = V.spread(V.propagate_sigma_points(est, d, inp.saved, rig.baseline_mm, cfg), config=cfg)
    assert not V.compare_approximation(mc, sp, cfg)["adequate"]


def test_incomplete_sigma_point_set_reports_no_spread(scene, draws):
    est = S.SyntheticWallEstimator(scene["rig"], fail_when=lambda p: "x" if p["K_left"][0, 0] > scene["inputs"].saved[0] else None)
    sp = V.spread(V.propagate_sigma_points(est, draws, scene["inputs"].saved, scene["rig"].baseline_mm))
    assert sp["width_sd_mm"] is None and sp["failed"] > 0
    assert V.compare_approximation(V.spread(V.propagate_monte_carlo(scene["est"], draws, 1.0)), sp)["adequate"] is False


# ------------------------------------------------------------------ 10. uncertainty classification
def _full(scene, cfg=None, solver=None, **kw):
    return V.estimate_calibration_uncertainty(scene["inputs"], scene["est"], scene["rig"].baseline_mm, cfg or CFG(n_draws=60), solver=solver, **kw)


def test_status_unavailable_when_nothing_usable():
    s = dict(attempted=0, successful=0, failed=0, failure_fraction=None)
    assert V.classify_calibration_uncertainty(s, None, dict(passed=True), None, None)["status"] == "unavailable"


def test_status_incomplete_on_high_failure_fraction(scene):
    plan = {i: "fail" for i in range(1, 61, 2)}                                        # half of the draws fail (call 0 is the reproduction check)
    rpt = _full(scene, solver=_flaky_solver(plan))
    c = rpt["components"][0]
    assert c["status"] == "incomplete" and c["draws"]["failure_fraction"] == pytest.approx(0.5)
    assert any("failure fraction" in r for r in c["reasons"]) and c["draws"]["failed"] > 0
    assert any("provisional" in r for r in c["reasons"])


def test_status_incomplete_when_too_few_successes_or_truncated(scene):
    assert _full(scene, CFG(n_draws=12, min_successful_draws=30))["components"][0]["status"] == "incomplete"
    t = _full(scene, CFG(n_draws=50, max_seconds=0.0))
    assert t["components"][0]["status"] in ("incomplete", "unavailable")


def test_unrepresentative_successes_are_detected(scene):
    """Failures that depend on a particular pair (draws lacking pair 0 fail) leave successes that over-represent it."""
    inp = scene["inputs"]; first = inp.left[0]

    def solver(left, right, *a, **k):
        if not any(np.array_equal(first, l) for l in left):
            raise UC.CalibrationFailed("pair 0 absent")
        return UC.calibrate_stereo(left, right, *a, **k)
    d = V.bootstrap_calibration(inp, CFG(n_draws=150), solver=solver)
    rep = V.representativeness(d, CFG())
    assert d.failures and rep["assessed"] and not rep["representative"] and rep["worst_pair_index"] == 0
    cls = V.classify_calibration_uncertainty(V.draw_summary(d), None, dict(passed=True), rep, None, CFG())
    assert cls["status"] == "incomplete" and any("not representative" in r for r in cls["reasons"])


def test_random_failures_do_not_look_unrepresentative(scene):
    d = V.bootstrap_calibration(scene["inputs"], CFG(n_draws=60), solver=_flaky_solver({i: "fail" for i in range(3, 60, 7)}))
    assert V.representativeness(d, CFG())["representative"]


def test_complete_status_only_when_all_provisional_checks_pass():
    rig, obs, model, doc = S.make_calibration(n_pairs=30, seed=9, model="radial2", tilt_deg=35.0, noise_px=0.05)
    inp = V.inputs_from_document(doc)
    rpt = V.estimate_calibration_uncertainty(inp, S.SyntheticWallEstimator(rig), rig.baseline_mm, CFG(n_draws=100))
    c = rpt["components"][0]
    assert c["status"] == "complete" and c["reasons"] == [] and c["width_sd_mm"] > 0 and rpt["approximation"]["adequate"]
    assert rpt["combined_estimate_complete"] is False                                   # only one measured source exists in this phase


def test_report_labelling_classes_and_no_interval_claims(scene):
    rig = scene["rig"]
    sens = V.sensitivity_analysis(scene["est"], V.vector_to_params(scene["inputs"].saved, rig.baseline_mm),
                                  {"baseline_plus_1pct": ("entered baseline x 1.01 (assumed tolerance)", lambda p: {**p, "baseline_mm": p["baseline_mm"] * 1.01})})
    rpt = _full(scene, CFG(n_draws=20), sensitivity=sens)
    classes = [c["class"] for c in rpt["components"]]
    assert classes == [V.MEASURED, V.SENSITIVITY, V.NOT_QUANTIFIED] and rpt["label"] == "Indicative uncertainty"
    m = rpt["components"][0]
    assert m["method"] and m["draws"]["successful"] > 0 and m["limitations"] and "width_sd_mm" in m
    assert rpt["claims"]["confidence_interval"] is False and rpt["claims"]["coverage_probability"] is None
    text = json.dumps(rpt, default=float)
    assert not re.search(r"95\s*%|95th|coverage of", text.lower()) and "confidence interval" not in text.lower().replace("not a validated interval", "")
    assert any("systematic" in x for x in rpt["components"][2]["items"])
    assert "width_sd_mm" not in rpt["components"][1] and rpt["components"][1]["note"].startswith("Assumed perturbation")   # sensitivity is never given a measured SD
    json.dumps(rpt, default=float)


def test_unavailable_report_lists_calibration_as_not_quantified(scene):
    inp = copy.copy(scene["inputs"]); inp.saved = inp.saved + 1.0
    rpt = V.estimate_calibration_uncertainty(inp, scene["est"], 120.0, CFG(n_draws=3))
    assert rpt["components"][0]["status"] == "unavailable"
    assert any(i.startswith("calibration uncertainty:") for i in rpt["components"][-1]["items"])


def test_sensitivity_baseline_scales_dimensions_proportionally(scene):
    rig = scene["rig"]; base = V.vector_to_params(scene["inputs"].saved, rig.baseline_mm)
    out = V.sensitivity_analysis(scene["est"], base, {"x0.9": ("baseline x0.9", lambda p: {**p, "baseline_mm": p["baseline_mm"] * 0.9}),
                                                      "x1.1": ("baseline x1.1", lambda p: {**p, "baseline_mm": p["baseline_mm"] * 1.1})})
    assert out["class_"] == V.SENSITIVITY
    for k, f in (("x0.9", 0.9), ("x1.1", 1.1)):
        assert out["cases"][k]["width_mm"] == pytest.approx(out["base"]["width_mm"] * f, rel=1e-6)
        assert out["cases"][k]["height_mm"] == pytest.approx(out["base"]["height_mm"] * f, rel=1e-6)


def test_sensitivity_failures_are_reported_not_hidden(scene):
    base = V.vector_to_params(scene["inputs"].saved, 120.0)
    out = V.sensitivity_analysis(scene["est"], base, {"bad": ("raises", lambda p: 1 / 0)})
    assert out["cases"]["bad"]["failed"] and "perturbation_exception" in out["cases"]["bad"]["reason"]


def test_combining_measured_components_is_conditional():
    a = {"class": V.MEASURED, "status": "complete", "width_sd_mm": 3.0, "height_sd_mm": 4.0}
    b = {"class": V.MEASURED, "status": "complete", "width_sd_mm": 4.0, "height_sd_mm": 3.0}
    assert V.combine_measured([a])["combined"] is None and "nothing to combine" in V.combine_measured([a])["reason"]
    assert V.combine_measured([a, b])["combined"] is None and "independence" in V.combine_measured([a, b])["reason"]
    ok = V.combine_measured([a, b], independence_justification="synthetic: disjoint random inputs")
    assert ok["combined"] == {"width_sd_mm": 5.0, "height_sd_mm": 5.0} and ok["method"] == "quadrature"
    assert V.combine_measured([a, dict(b, status="incomplete")], independence_justification="x")["combined"] is None
    assert V.combine_measured([a, {"class": V.SENSITIVITY}], independence_justification="x")["combined"] is None


# ------------------------------------------------------------------ 11. measurement outcome policy
def test_outcome_success():
    r = V.classify_measurement_outcome(5000.0, 2400.0, uncertainty_status="incomplete")
    assert r["outcome"] == "SUCCESS" and r["width_mm"] == 5000.0 and r["height_mm"] == 2400.0 and r["warnings"] == [] and r["uncertainty_status"] == "incomplete"


def test_outcome_questionable_preserves_unmodified_estimates_and_warnings():
    flags = [dict(code="pose_degenerate", message="robust scale 1.3 px"), "low_registration_coverage"]
    r = V.classify_measurement_outcome(5123.456789, 2401.5, quality_flags=flags)
    assert r["outcome"] == "QUESTIONABLE" and r["width_mm"] == 5123.456789 and r["height_mm"] == 2401.5            # not rounded, clipped or adjusted
    assert [w["code"] for w in r["warnings"]] == ["pose_degenerate", "low_registration_coverage"] and r["diagnostics"] == []


@pytest.mark.parametrize("w,h,hard,expected", [
    (5000.0, 2400.0, ["registration_insufficient"], "registration_insufficient"),
    (None, 2400.0, [], "width_not_finite"), (5000.0, float("nan"), [], "height_not_finite"), (float("inf"), 1.0, [], "width_not_finite"),
    (-3.0, 2400.0, [], "width_not_positive"), (5000.0, 0.0, [], "height_not_positive"), ("5000", 2400.0, [], "width_not_finite"), (True, 2400.0, [], "width_not_finite")])
def test_outcome_failed_reports_diagnostics_and_never_fabricates_dimensions(w, h, hard, expected):
    r = V.classify_measurement_outcome(w, h, hard_failures=hard, quality_flags=["also_noted"])
    assert r["outcome"] == "FAILED" and r["width_mm"] is None and r["height_mm"] is None
    assert expected in [d["code"] for d in r["diagnostics"]] and [x["code"] for x in r["warnings"]] == ["also_noted"]


def test_incomplete_uncertainty_alone_does_not_make_a_measurement_questionable():
    assert V.classify_measurement_outcome(1.0, 1.0, uncertainty_status="unavailable")["outcome"] == "SUCCESS"


def test_outcome_classifier_is_not_wired_to_the_visitor_backend():
    for f in ("visitor_workflow/f8_visitor_service.py", "room_measurement/visitor_measurement.py"):
        assert "visitor_uncertainty" not in (MODULE / f).read_text()


# ------------------------------------------------------------------ 12. isolation, constants, gate, unchanged code
def _imports(path):
    tree = ast.parse(Path(path).read_text()); out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out.add(("." * n.level) + (n.module or "")); out |= {("." * n.level) + (n.module or "") + "." + a.name for a in n.names}
    return out


@pytest.mark.parametrize("name", ["visitor_uncertainty.py", "visitor_uncertainty_synthetic.py"])
def test_no_experiment_constants_recipe_a_or_method_j_dependencies(name):
    p = MODULE / "room_measurement" / name; src = p.read_text(); code = re.sub(r'"""[\s\S]*?"""', "", src)
    forbidden_mods = ("pose", "measure", "pipeline", "geometry", "lines", "features", "frames", "validation", "stage_c", "stage_e1", "uncertainty", "display_data", "visitor_measurement", "f8_visitor_service")
    for imp in _imports(p):
        tail = imp.split(".")[-1]
        assert tail not in forbidden_mods or imp.endswith("visitor_uncertainty"), f"{name} imports {imp}"
    assert not re.search(r"\b22[56](\.0)?\b", code), "experiment baseline literal"
    assert not re.search(r"recipe[_ ]?a|method_j_experiment|frame_manifest|wall_boundary_annotations|\.mov\b|VideoCapture", code, re.I)


def test_room_processing_gate_stays_closed(monkeypatch):
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ENABLE_ROOM_PROCESSING", "1")
    sys.path.insert(0, str(MODULE / "visitor_workflow"))
    import f8_visitor_service as FS
    assert FS.ROOM_PROCESSING_ENABLED is False
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow/f8_visitor_service.py").read_text(), re.M)
    assert "ROOM_PROCESSING_ENABLED" not in (MODULE / "room_measurement/visitor_uncertainty.py").read_text()


def test_solver_and_method_j_sources_are_unmodified_in_git():
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=MODULE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pytest.skip("not a git checkout")
    files = [f"Modules/Module_7/{f}" for f in ("stereo_measure/user_calibration.py", "room_measurement/pose.py", "room_measurement/measure.py", "room_measurement/geometry.py",
                                               "room_measurement/lines.py", "room_measurement/features.py", "room_measurement/pipeline.py", "room_measurement/uncertainty.py")]
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files], cwd=root)
    assert r.returncode == 0, "calibration solver or Method J sources differ from HEAD"
