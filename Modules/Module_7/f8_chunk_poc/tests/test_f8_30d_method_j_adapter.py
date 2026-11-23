"""F8.30D-A: Method J adapter, failure classification and the uncertainty evaluation contract.

SYNTHETIC fixtures only: a rendered floor + wall scene (``visitor_method_j_adapter_synthetic``).  No video, no .MOV, no real calibration, no real room.
Method J itself (pose / geometry / measure) runs for real on the synthetic inputs; classification paths that cannot be reached naturally are driven by
substituting a stage (``MethodJStages``), which changes what the adapter is given, never any Method J mathematics."""
import ast
import copy
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE))

from room_measurement import geometry as G, measure as M, pose as P, visitor_measurement as VM, visitor_uncertainty as VU      # noqa: E402
from room_measurement import visitor_method_j_adapter as A                                                                   # noqa: E402
from room_measurement import visitor_method_j_adapter_synthetic as S                                                         # noqa: E402

ADAPTER_SRC = MODULE / "room_measurement/visitor_method_j_adapter.py"


# ------------------------------------------------------------------ fixtures and helpers
@pytest.fixture(scope="module")
def scene():
    ctx = S.make_context(); inp, truth = S.make_scene(ctx)
    return dict(ctx=ctx, inp=inp, truth=truth, base=A.evaluate(inp, ctx))


def codes(ev, key):
    return [x["code"] for x in ev[key]]


def real_stages(**over):
    return A.MethodJStages(**{**dict(build_dataset=P.build_dataset, estimate_pose=P.estimate_pose, measure_sets=VM._measure_sets), **over})


def pose_with(edit):
    """Real ``estimate_pose``, then ``edit(report)`` on the returned REPORT only (the pose itself is untouched)."""
    def f(D):
        pJ, pA, rep, extra = P.estimate_pose(D); rep = copy.deepcopy(rep); edit(rep); return pJ, pA, rep, extra
    return f


def rows_with(edit):
    def f(cal, p5, match_sets, line_sets, wh):
        rows = VM._measure_sets(cal, p5, match_sets, line_sets, wh); return edit(copy.deepcopy(rows))
    return f


def strip_time(ev):
    ev = copy.deepcopy(ev); ev["evaluation"].pop("seconds", None); return ev


def perturbed(ctx, dfx=6.0, dcx=-3.0, dk2=0.01):
    p = ctx.to_params(); p["K_left"][0, 0] += dfx; p["K_left"][1, 1] += dfx * 1.001; p["K_left"][0, 2] += dcx; p["D_left"][1] += dk2; p["K_right"][0, 0] -= dfx * 0.5; p["D_right"][4] += 0.004
    return p


# ------------------------------------------------------------------ 1-2: visitor K, D and baseline propagation
def test_visitor_k_and_d_reach_every_method_j_stage(scene):
    ctx, inp = scene["ctx"], scene["inp"]; p = perturbed(ctx); ctx2 = A.VisitorCalibrationContext.from_params(p, S.SIZE, "draw"); seen = []
    def bd(cal, ms): seen.append(("dataset", cal)); return P.build_dataset(cal, ms)
    def ms(cal, p5, m, l, wh): seen.append(("measure", cal)); return VM._measure_sets(cal, p5, m, l, wh)
    ev = A.evaluate(inp, ctx2, stages=real_stages(build_dataset=bd, measure_sets=ms))
    assert ev["outcome"] in (A.SUCCESS, A.QUESTIONABLE) and ev["calibration"]["lines_retargeted"] is True
    assert [s for s, _ in seen] == ["dataset", "measure"]
    for _, cal in seen:
        assert np.array_equal(cal.K_left, p["K_left"]) and np.array_equal(cal.D_left, p["D_left"]) and np.array_equal(cal.K_right, p["K_right"]) and np.array_equal(cal.D_right, p["D_right"])
    assert ev["calibration"]["digest"] == ctx2.digest() != ctx.digest()
    assert abs(ev["width_mm"] - scene["base"]["width_mm"]) > 0 and abs(ev["width_mm"] - scene["truth"]["width_mm"]) < 0.1 * scene["truth"]["width_mm"]   # a different calibration gives a different, still sane, number


def test_lines_are_not_retargeted_when_the_calibration_is_the_one_they_were_fitted_under(scene):
    assert scene["base"]["calibration"]["lines_retargeted"] is False


def test_visitor_baseline_reaches_method_j_and_scales_the_output(scene):
    ctx, inp = scene["ctx"], scene["inp"]; got = []
    def bd(cal, ms): got.append(cal.baseline_mm); return P.build_dataset(cal, ms)
    c2 = A.VisitorCalibrationContext.from_params({**ctx.to_params(), "baseline_mm": ctx.baseline_mm * 1.1}, S.SIZE, "b")
    ev = A.evaluate(inp, c2, stages=real_stages(build_dataset=bd))
    assert got == [ctx.baseline_mm * 1.1] and ev["calibration"]["baseline_mm"] == ctx.baseline_mm * 1.1
    assert ev["width_mm"] / scene["base"]["width_mm"] == pytest.approx(1.1, rel=2e-3) and ev["height_mm"] / scene["base"]["height_mm"] == pytest.approx(1.1, rel=2e-3)


def test_sane_dimensions_on_the_synthetic_scene(scene):
    b, t = scene["base"], scene["truth"]                                  # sanity only: the truth is never an input
    assert abs(b["width_mm"] - t["width_mm"]) < 0.03 * t["width_mm"] and abs(b["height_mm"] - t["height_mm"]) < 0.03 * t["height_mm"]
    assert b["ground_truth_used"] is False


# ------------------------------------------------------------------ 3: preservation of Method J numerical outputs
def test_adapter_reproduces_direct_method_j_calls_bit_for_bit(scene):
    ctx, inp, b = scene["ctx"], scene["inp"], scene["base"]
    cal = ctx.to_stereo_calibration(); D = P.build_dataset(cal, list(inp.match_sets)); pJ, pA, rep, _ = P.estimate_pose(D)
    rows = VM._measure_sets(cal, pJ, list(inp.match_sets), list(inp.line_sets), S.SIZE)
    assert all(r["ok"] for r in rows)
    assert b["width_mm"] == float(np.mean([r["width_mm"] for r in rows])) and b["height_mm"] == float(np.mean([r["height_mm"] for r in rows]))
    assert b["diagnostics"]["pose"]["params"] == np.asarray(pJ, float).tolist()
    assert [r["width_mm"] for r in b["diagnostics"]["per_set"]] == [r["width_mm"] for r in rows]


def test_adapter_agrees_with_the_original_phase1_measure_all_sets(scene):
    """At 3840 rows the visitor floor-row scaling is the identity, so the original ``measure.measure_all_sets`` must give the same numbers."""
    ctx, inp, b = scene["ctx"], scene["inp"], scene["base"]
    cal = ctx.to_stereo_calibration(); pJ = np.array(b["diagnostics"]["pose"]["params"]); rows = M.measure_all_sets(cal, pJ, list(inp.match_sets), list(inp.line_sets))
    assert VM.floor_row_threshold(S.SIZE[1]) == 2600.0
    assert b["width_mm"] == pytest.approx(float(np.mean([r["width_mm"] for r in rows])), abs=1e-9) and b["height_mm"] == pytest.approx(float(np.mean([r["height_mm"] for r in rows])), abs=1e-9)


def test_evaluation_is_deterministic_and_unrounded(scene):
    again = A.evaluate(scene["inp"], scene["ctx"])
    assert strip_time(again) == strip_time(scene["base"])
    assert scene["base"]["width_mm"] != round(scene["base"]["width_mm"], 3)                # not rounded


@pytest.mark.parametrize("edit", [lambda r: r["method_J"]["flags"].update(inlier_fraction=0.5)])          # F8.30D-B 1B: converged / scale flags are now FAILED, not warnings
def test_quality_warnings_never_change_the_numbers(scene, edit):
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=pose_with(edit)))
    assert ev["outcome"] == A.QUESTIONABLE and ev["width_mm"] == scene["base"]["width_mm"] and ev["height_mm"] == scene["base"]["height_mm"]


# ------------------------------------------------------------------ 4: SUCCESS
def test_success_classification(scene):
    b = scene["base"]
    assert b["outcome"] == A.SUCCESS and b["failures"] == [] and b["warnings"] == [] and not b["partial"]
    assert b["dimensions"]["width"]["status"] == b["dimensions"]["height"]["status"] == "available"
    assert b["dimensions"]["width"]["n_sets"] == 3 and b["diagnostics"]["pose"]["converged"] is True


# ------------------------------------------------------------------ 5: QUESTIONABLE
@pytest.mark.parametrize("edit, expected", [
    (lambda r: r["method_J"]["flags"].update(inlier_fraction=0.4), "POSE_LOW_INLIER_FRACTION"),
])
def test_pose_quality_issues_are_questionable_not_failed(scene, edit, expected):
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=pose_with(edit)))
    assert ev["outcome"] == A.QUESTIONABLE and expected in codes(ev, "warnings") and ev["failures"] == []
    assert ev["width_mm"] is not None and ev["height_mm"] is not None


def test_poor_registration_is_questionable(scene):
    def bad(rows):
        for r in rows: r["registration"] = dict(rms_mm=35.0, n_inliers=10, n_pairs=60)
        return rows
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(bad)))
    assert ev["outcome"] == A.QUESTIONABLE and {"REGISTRATION_FEW_INLIERS", "REGISTRATION_LOW_INLIER_FRACTION", "REGISTRATION_HIGH_RMS"} <= set(codes(ev, "warnings"))


def test_some_failed_frame_sets_are_questionable_and_use_the_rest(scene):
    def one_bad(rows):
        rows[1] = dict(ok=False, code="REGISTRATION_FAILED", error="only 3 registration pairs", width_mm=float("nan"), height_mm=float("nan")); return rows
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(one_bad)))
    ps = scene["base"]["diagnostics"]["per_set"]
    assert ev["outcome"] == A.QUESTIONABLE and "SOME_FRAME_SETS_FAILED" in codes(ev, "warnings") and ev["dimensions"]["width"]["n_sets"] == 2
    assert ev["width_mm"] == float(np.mean([ps[0]["width_mm"], ps[2]["width_mm"]]))
    assert ev["diagnostics"]["set_failures"] == [dict(frame_set=1, code="REGISTRATION_FAILED", message="only 3 registration pairs")]


# ------------------------------------------------------------------ 6-7: FAILED, degenerate pose
def test_unrecoverable_degenerate_pose_is_failed_and_geometry_is_not_run(scene):
    called = []
    def degenerate(rep): rep["method_A"]["guard"].update(degenerate=True, resolved=False, fallback_used=True)
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=pose_with(degenerate), measure_sets=lambda *a: called.append(1)))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["POSE_DEGENERATE_UNRECOVERABLE"] and not called
    assert ev["width_mm"] is None and ev["height_mm"] is None and ev["diagnostics"]["pose"]["epipolar_guard"]["unresolved_degenerate"] is True


def test_real_garbage_matches_give_a_typed_failure_not_a_number_or_an_exception(scene):
    rng = np.random.default_rng(0); sets = []
    for F in scene["inp"].match_sets:
        G_ = dict(F); G_["B1"] = rng.uniform(0, 2160, (len(F["A1"]), 2)).astype(np.float32); G_["B2"] = rng.uniform(0, 3840, (len(F["A2"]), 2)).astype(np.float32); sets.append(G_)
    with np.errstate(all="ignore"):
        ev = A.evaluate(A.ReconstructionInputs(tuple(sets), scene["inp"].line_sets, scene["ctx"]), scene["ctx"])
    assert ev["outcome"] == A.FAILED and ev["width_mm"] is None and codes(ev, "failures")[0] in {"POSE_SOLVER_EXCEPTION", "POSE_DEGENERATE_UNRECOVERABLE", "POSE_NON_FINITE", "NO_USABLE_FRAME_SETS"}


def test_non_finite_pose_is_failed(scene):
    def nan_pose(D): pJ, pA, rep, ex = P.estimate_pose(D); pJ = np.array(pJ, float); pJ[0] = np.nan; return pJ, pA, rep, ex
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=nan_pose))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["POSE_NON_FINITE"]


def test_solver_exception_is_typed(scene):
    def boom(D): raise np.linalg.LinAlgError("singular")
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=boom))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["POSE_SOLVER_EXCEPTION"] and "singular" in ev["failures"][0]["message"]


def test_all_frame_sets_failing_is_failed_with_causes(scene):
    def all_bad(rows): return [dict(ok=False, code="FLOOR_PLANE_FAILED", error="no floor", width_mm=float("nan"), height_mm=float("nan")) for _ in rows]
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(all_bad)))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["NO_USABLE_FRAME_SETS"] and ev["failures"][0]["causes"] == ["FLOOR_PLANE_FAILED"]


@pytest.mark.parametrize("make, code", [
    (lambda i, c: A.ReconstructionInputs((), (), c), "INPUT_NO_FRAME_SETS"),
    (lambda i, c: A.ReconstructionInputs(i.match_sets, i.line_sets[:1], c), "INPUT_LINES_MISSING"),
    (lambda i, c: A.ReconstructionInputs(i.match_sets, tuple({k: v for k, v in L.items() if k != "PR_ceilF"} for L in i.line_sets), c), "INPUT_LINES_MISSING"),
    (lambda i, c: A.ReconstructionInputs(tuple({k: v for k, v in F.items() if k != "MX"} for F in i.match_sets), i.line_sets, c), "INPUT_MALFORMED_FRAME_SET"),
    (lambda i, c: A.ReconstructionInputs(tuple({**F, "A1": F["A1"][:3], "B1": F["B1"][:3]} for F in i.match_sets), i.line_sets, c), "INPUT_MALFORMED_FRAME_SET"),
    (lambda i, c: A.ReconstructionInputs(tuple({**F, "A1": np.full_like(F["A1"], np.nan)} for F in i.match_sets), i.line_sets, c), "INPUT_MALFORMED_FRAME_SET"),
    (lambda i, c: A.ReconstructionInputs(i.match_sets, i.line_sets, A.VisitorCalibrationContext.from_params(c.to_params(), (1280, 720))), "INPUT_IMAGE_SIZE_MISMATCH"),
])
def test_invalid_inputs_are_failed_without_running_method_j(scene, make, code):
    called = []; st = real_stages(build_dataset=lambda *a: called.append(1), estimate_pose=lambda *a: called.append(1))
    ev = A.evaluate(make(scene["inp"], scene["ctx"]), scene["ctx"], stages=st)
    assert ev["outcome"] == A.FAILED and code in codes(ev, "failures") and not called and ev["width_mm"] is None


def test_invalid_calibration_is_failed(scene):
    p = scene["ctx"].to_params(); p["K_left"][0, 0] = -1.0
    ev = A.evaluate(scene["inp"], A.VisitorCalibrationContext.from_params(p, S.SIZE))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["INPUT_INVALID_CALIBRATION"]
    p = scene["ctx"].to_params(); p["baseline_mm"] = 0.0
    assert codes(A.evaluate(scene["inp"], A.VisitorCalibrationContext.from_params(p, S.SIZE)), "failures") == ["INPUT_INVALID_CALIBRATION"]


def test_retarget_failure_is_typed(scene):
    def boom(*a): raise ValueError("cannot refit")
    ev = A.evaluate(scene["inp"], A.VisitorCalibrationContext.from_params(perturbed(scene["ctx"]), S.SIZE), stages=real_stages(retarget_lines=boom))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["LINE_RETARGET_FAILED"]


# ------------------------------------------------------------------ 8: valid fallback
def test_fallback_start_that_recovers_a_valid_pose_is_success_with_a_diagnostic_note_and_unchanged_numbers(scene):
    def fb(rep): rep["method_A"]["guard"].update(fallback_used=True, resolved=True, degenerate=False)
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=pose_with(fb)))
    assert ev["outcome"] == A.SUCCESS and ev["warnings"] == [] and [n["code"] for n in ev["diagnostic_notes"]] == ["POSE_FALLBACK_START_USED"] and ev["failures"] == []
    assert ev["width_mm"] == scene["base"]["width_mm"] and ev["height_mm"] == scene["base"]["height_mm"]


def test_naturally_occurring_fallback_is_flagged_consistently():
    ctx = S.make_context(baseline_mm=250.0); inp, _ = S.make_scene(ctx, noise_px=0.1, seed=11)         # this seed trips the epipolar guard's fallback restart on OpenCV 5.0.0
    ev = A.evaluate(inp, ctx); g = ev["diagnostics"]["pose"]["epipolar_guard"]
    assert ("POSE_FALLBACK_START_USED" in [n["code"] for n in ev["diagnostic_notes"]]) == bool(g["fallback_used"] and ev["outcome"] != A.FAILED)
    if g["fallback_used"] and not g["unresolved_degenerate"] and ev["outcome"] != A.FAILED: assert ev["warnings"] == [] or "POSE_FALLBACK_START_USED" not in codes(ev, "warnings")


# ------------------------------------------------------------------ 9: partial dimensions
NAN = float("nan")


def test_width_available_height_unavailable(scene):
    def no_h(rows):
        for r in rows: r["height_mm"] = NAN
        return rows
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(no_h)))
    assert ev["outcome"] == A.QUESTIONABLE and ev["partial"] is True and ev["height_mm"] is None and ev["width_mm"] == scene["base"]["width_mm"]
    assert ev["dimensions"]["height"] == dict(status="unavailable", value=None, n_sets=0, reason="HEIGHT_UNAVAILABLE") and ev["dimensions"]["width"]["status"] == "available"
    assert "DIMENSION_PARTIAL" in codes(ev, "warnings")


def test_height_available_width_unavailable(scene):
    def no_w(rows):
        for r in rows: r["width_mm"] = -5.0                              # non-positive is not a valid dimension
        return rows
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(no_w)))
    assert ev["outcome"] == A.QUESTIONABLE and ev["partial"] and ev["width_mm"] is None and ev["height_mm"] == scene["base"]["height_mm"]
    assert ev["dimensions"]["width"]["reason"] == "WIDTH_UNAVAILABLE"


def test_a_set_missing_one_dimension_only_drops_out_of_that_dimension(scene):
    def mixed(rows): rows[0]["height_mm"] = NAN; return rows
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(mixed))); ps = scene["base"]["diagnostics"]["per_set"]
    assert ev["outcome"] == A.QUESTIONABLE and not ev["partial"] and "SET_DIMENSION_MISSING" in codes(ev, "warnings")
    assert ev["dimensions"]["width"]["n_sets"] == 3 and ev["dimensions"]["height"]["n_sets"] == 2
    assert ev["width_mm"] == scene["base"]["width_mm"] and ev["height_mm"] == float(np.mean([ps[1]["height_mm"], ps[2]["height_mm"]]))


def test_no_valid_dimension_at_all_is_failed(scene):
    def none(rows):
        for r in rows: r["width_mm"] = NAN; r["height_mm"] = float("inf")
        return rows
    ev = A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(measure_sets=rows_with(none)))
    assert ev["outcome"] == A.FAILED and ev["width_mm"] is None and ev["height_mm"] is None and not ev["partial"] and codes(ev, "failures") == ["NO_USABLE_FRAME_SETS"]


# ------------------------------------------------------------------ 10: uncertainty-only failure isolation
@pytest.mark.parametrize("exc, status", [(RuntimeError("bootstrap exploded"), "failed"), (VU.UncertaintyUnavailable("no observations stored"), "unavailable")])
def test_uncertainty_failure_never_changes_the_measurement(scene, exc, status):
    def runner(est, log): raise exc
    out = A.measure_with_optional_uncertainty(scene["inp"], scene["ctx"], runner)
    assert out["uncertainty"]["status"] == status and out["uncertainty"]["result"] is None
    assert strip_time(out["measurement"]) == strip_time(scene["base"]) and out["measurement"]["outcome"] == A.SUCCESS
    assert out["label"] == "Indicative uncertainty"


def test_uncertainty_is_skipped_and_not_attached_when_the_measurement_failed(scene):
    called = []
    out = A.measure_with_optional_uncertainty(A.ReconstructionInputs((), (), scene["ctx"]), scene["ctx"], lambda e, l: called.append(1))
    assert out["measurement"]["outcome"] == A.FAILED and out["uncertainty"]["status"] == "skipped_measurement_failed" and not called


def test_all_failing_uncertainty_draws_leave_the_original_untouched(scene):
    def runner(est, log):
        p = perturbed(scene["ctx"]); outcomes = []
        for _ in range(3):
            try: est(p)
            except VU.EstimatorFailure as ex: outcomes.append(ex.code)
        return outcomes
    st = real_stages(estimate_pose=pose_with(lambda r: r["method_A"]["guard"].update(degenerate=True)))
    out = A.measure_with_optional_uncertainty(scene["inp"], scene["ctx"], runner, stages=A.MethodJStages())
    assert out["measurement"]["outcome"] == A.SUCCESS
    out2 = A.measure_with_optional_uncertainty(scene["inp"], scene["ctx"], lambda e, l: None, stages=st)
    assert out2["measurement"]["outcome"] == A.FAILED and out2["uncertainty"]["status"] == "skipped_measurement_failed"


# ------------------------------------------------------------------ 11: typed reasons
def test_every_reported_code_is_a_declared_typed_reason(scene):
    failure_values = {e.value for e in A.FailureReason}; flag_values = {e.value for e in A.QualityFlag}
    assert len(failure_values) == len(list(A.FailureReason)) and len(flag_values) == len(list(A.QualityFlag)) and not failure_values & flag_values
    evs = [A.evaluate(A.ReconstructionInputs((), (), scene["ctx"]), scene["ctx"]),
           A.evaluate(scene["inp"], scene["ctx"], stages=real_stages(estimate_pose=pose_with(lambda r: r["method_J"]["flags"].update(converged=False))))]
    for ev in evs:
        assert set(codes(ev, "failures")) <= failure_values and set(codes(ev, "warnings")) <= flag_values
    assert all(isinstance(e, str) for e in A.FailureReason)


# ------------------------------------------------------------------ 12-: uncertainty contract: each draw's own calibration, correlations, logging of failures
def make_draws(ctx, n=4, seed=3):
    rng = np.random.default_rng(seed); base = VU.params_to_vector(ctx.to_params()); vecs = []
    for _ in range(n):
        z = rng.normal(size=2); v = base.copy()
        v[0] += 5.0 * z[0]; v[2] += -0.8 * 5.0 * z[0] + 1.0 * z[1]; v[5] += 0.004 * z[1]; v[6] -= 0.003 * z[1]          # correlated shifts: fx~cx, d1~d2 of the left camera
        vecs.append(v)
    return VU.CalibrationDraws(seed=seed, attempted=n, vectors=np.array(vecs), ok_index=list(range(n)), failures=[], counts=np.zeros((n, 1), np.int32), seconds=0.0, truncated=False, n_pairs=1,
                               distortion="radial3", observations_sha256="synthetic")


def test_estimator_uses_each_draws_whole_calibration_and_the_visitor_baseline(scene):
    ctx, inp = scene["ctx"], scene["inp"]; draws = make_draws(ctx, 3); log = A.EvaluationLog(); seen = []
    def bd(cal, ms): seen.append(cal); return P.build_dataset(cal, ms)
    est = A.make_estimator(inp, ctx, log=log, stages=real_stages(build_dataset=bd))
    prop = VU.propagate_monte_carlo(est, draws, ctx.baseline_mm)
    assert len(seen) == 3 and prop.outputs.shape == (3, 2) and not prop.failures
    for v, cal in zip(draws.vectors, seen):
        got = VU.params_to_vector(dict(K_left=cal.K_left, D_left=cal.D_left, K_right=cal.K_right, D_right=cal.D_right))
        assert np.array_equal(got, v) and cal.baseline_mm == ctx.baseline_mm                       # the whole 18-vector, unmodified: correlations are those of the draw
    assert [r["kind"] for r in log.records] == ["calibration_draw"] * 3 and len({r["calibration_digest"] for r in log.records}) == 3
    assert not np.allclose(prop.outputs, [scene["base"]["width_mm"], scene["base"]["height_mm"]], rtol=0, atol=1e-9)


def test_sigma_point_propagation_runs_through_the_adapter(scene):
    ctx, inp = scene["ctx"], scene["inp"]; draws = make_draws(ctx, 4); log = A.EvaluationLog()
    prop = VU.propagate_sigma_points(A.make_estimator(inp, ctx, log=log), draws, VU.params_to_vector(ctx.to_params()), ctx.baseline_mm, VU.UncertaintyConfig(sigma_rank=2))
    assert log.summary()["attempted"] == prop.detail["runs"] == 5 and prop.detail["center_ok"]
    c = prop.detail["center_output"]; assert c[0] == scene["base"]["width_mm"] and c[1] == scene["base"]["height_mm"]          # centre run == the original measurement


def test_failed_reconstruction_evaluations_are_recorded_and_counted(scene):
    ctx, inp = scene["ctx"], scene["inp"]; draws = make_draws(ctx, 4); log = A.EvaluationLog(); n = []
    def flaky(D):
        n.append(1); pJ, pA, rep, ex = P.estimate_pose(D)
        if len(n) in (2, 4): rep = copy.deepcopy(rep); rep["method_A"]["guard"].update(degenerate=True)
        return pJ, pA, rep, ex
    prop = VU.propagate_monte_carlo(A.make_estimator(inp, ctx, log=log, stages=real_stages(estimate_pose=flaky)), draws, ctx.baseline_mm)
    s = log.summary()
    assert s["attempted"] == 4 and s["failed"] == 2 and s["success"] + s["questionable"] == 2 and s["failure_codes"] == {"POSE_DEGENERATE_UNRECOVERABLE": 2}
    assert [j for j, _ in prop.failures] == [1, 3] and {w for _, w in prop.failures} == {"POSE_DEGENERATE_UNRECOVERABLE"} and len(prop.outputs) == 2
    assert [r["outcome"] for r in log.records][1] == A.FAILED and log.records[1]["width_mm"] is None                                 # kept, never discarded


def test_input_failure_inside_the_estimator_is_also_logged(scene):
    log = A.EvaluationLog(); est = A.make_estimator(scene["inp"], scene["ctx"], log=log); p = scene["ctx"].to_params(); p["K_left"][0, 0] = NAN
    with pytest.raises(VU.EstimatorFailure) as e: est(p)
    assert e.value.code == "INPUT_INVALID_CALIBRATION" and log.summary()["failed"] == 1


def test_questionable_draws_are_included_by_default_and_excludable(scene):
    st = real_stages(estimate_pose=pose_with(lambda r: r["method_J"]["flags"].update(inlier_fraction=0.5)))
    inc = A.make_estimator(scene["inp"], scene["ctx"], log=A.EvaluationLog(), stages=st)(scene["ctx"].to_params())
    assert inc["outcome"] == A.QUESTIONABLE and inc["quality_flags"] == ["POSE_LOW_INLIER_FRACTION"] and inc["degenerate"] is False
    with pytest.raises(VU.EstimatorFailure) as e: A.make_estimator(scene["inp"], scene["ctx"], log=A.EvaluationLog(), stages=st, include_questionable=False)(scene["ctx"].to_params())
    assert e.value.code.startswith("questionable_excluded")


def test_partial_results_cannot_enter_a_width_height_pair_but_are_logged(scene):
    def no_h(rows):
        for r in rows: r["height_mm"] = NAN
        return rows
    log = A.EvaluationLog()
    with pytest.raises(VU.EstimatorFailure) as e: A.make_estimator(scene["inp"], scene["ctx"], log=log, stages=real_stages(measure_sets=rows_with(no_h)))(scene["ctx"].to_params())
    assert e.value.code == "dimension_unavailable:height" and log.records[0]["partial"] is True and log.records[0]["width_mm"] is not None


def test_baseline_is_preserved_unless_a_labelled_sensitivity_test_is_requested(scene):
    ctx, inp = scene["ctx"], scene["inp"]; p = {**ctx.to_params(), "baseline_mm": ctx.baseline_mm * 1.05}
    log = A.EvaluationLog()
    with pytest.raises(VU.EstimatorFailure) as e: A.make_estimator(inp, ctx, log=log)(p)
    assert e.value.code == "BASELINE_CHANGED_UNLABELLED" and log.records[0]["failure_codes"] == ["BASELINE_CHANGED_UNLABELLED"]
    out = A.make_estimator(inp, ctx, log=log, baseline_sensitivity_label="baseline +5 % what-if")(p)
    assert log.records[1]["kind"] == "baseline_sensitivity" and log.records[1]["label"] == "baseline +5 % what-if" and log.records[1]["baseline_mm"] == ctx.baseline_mm * 1.05
    assert out["width_mm"] / scene["base"]["width_mm"] == pytest.approx(1.05, rel=2e-3)
    assert A.make_estimator(inp, ctx, log=A.EvaluationLog())({k: v for k, v in ctx.to_params().items() if k != "baseline_mm"})["width_mm"] == scene["base"]["width_mm"]   # default = the visitor's baseline


def test_ground_truth_never_adjusts_anything(scene):
    assert scene["base"]["ground_truth_used"] is False
    tree = ast.parse(ADAPTER_SRC.read_text()); names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not {"validation", "ground_truth", "truth", "expected_width_mm"} & names


# ------------------------------------------------------------------ 13-15: constants, original videos, gate
@pytest.mark.parametrize("name", ["room_measurement/visitor_method_j_adapter.py", "room_measurement/visitor_method_j_adapter_synthetic.py"])
def test_no_recipe_a_hardcoded_baseline_or_experiment_constants(name):
    code = (MODULE / name).read_text()
    assert not re.search(r"\b22[56](\.0)?\b", code), "experiment baseline literal"
    assert not re.search(r"\b(2879|2380|2840|2347)\b", code), "experiment result literal"
    assert not re.search(r"recipe[_ ]?a|room_video_recipe|frame_manifest|wall_boundary_annotations|ground_truth(?!_used)|load_calibration|DEFAULT_CALIBRATION_PATH", code, re.I)


def test_adapter_has_no_video_frame_feature_or_original_data_dependency():
    tree = ast.parse(ADAPTER_SRC.read_text()); imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom): imported |= {(n.module or "") + "." + a.name for a in n.names} | {n.module or ""}
        elif isinstance(n, ast.Import): imported |= {a.name for a in n.names}
    flat = " ".join(sorted(imported))
    for banned in ("frames", "features", "pipeline", "validation", "stage_c", "stage_e1", "VideoCapture", "visitor_video_validation", "visitor_frame_selection"):
        assert banned not in flat, banned
    src = ADAPTER_SRC.read_text(); assert not re.search(r"VideoCapture|\.mov\b|\.mp4\b|extract_frames|match_set\(|open\(|read_text|read_bytes", src, re.I)


def test_nothing_reads_video_or_matches_features_while_the_adapter_runs(scene, monkeypatch):
    import cv2
    from room_measurement import features, frames
    def forbid(*a, **k): raise AssertionError("original-video / feature-extraction code was called")
    monkeypatch.setattr(cv2, "VideoCapture", forbid); monkeypatch.setattr(features, "match_set", forbid); monkeypatch.setattr(frames, "extract_frames", forbid)
    monkeypatch.setattr(frames, "load_frame_sets", forbid); monkeypatch.setattr(VM, "extract_frames", forbid)
    ev = A.evaluate(scene["inp"], scene["ctx"])
    assert ev["outcome"] == A.SUCCESS


def test_room_processing_gate_stays_closed(monkeypatch):
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ENABLE_ROOM_PROCESSING", "1")
    sys.path.insert(0, str(MODULE / "visitor_workflow"))
    import f8_visitor_service as FS
    assert FS.ROOM_PROCESSING_ENABLED is False
    assert re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow/f8_visitor_service.py").read_text(), re.M)
    assert "ROOM_PROCESSING_ENABLED" not in ADAPTER_SRC.read_text() and "environ" not in ADAPTER_SRC.read_text()


def test_adapter_is_not_imported_by_the_live_visitor_code():
    BACKEND = {"room_reconstruction.py", "reconstruction_worker.py"}          # F8.31C-1: the only two files that may use the adapter; no page, service or app does
    for f in (MODULE / "visitor_workflow").glob("*.py"):
        if f.name not in BACKEND:
            assert "visitor_method_j_adapter" not in f.read_text(), f.name
    assert "visitor_method_j_adapter" not in (MODULE / "room_measurement/visitor_measurement.py").read_text()


def test_method_j_and_solver_sources_are_unmodified_in_git():
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=MODULE, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pytest.skip("not a git checkout")
    files = [f"Modules/Module_7/{f}" for f in ("stereo_measure/user_calibration.py", "room_measurement/pose.py", "room_measurement/measure.py", "room_measurement/geometry.py", "room_measurement/lines.py",
                                               "room_measurement/features.py", "room_measurement/pipeline.py", "room_measurement/uncertainty.py", "room_measurement/calibration.py")]
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files], cwd=root).returncode == 0


def test_f830b_uncertainty_module_and_visitor_measurement_are_unmodified_by_this_phase():
    """visitor_measurement / visitor_uncertainty are untracked (earlier phases); they must still not reference the adapter (checked above) and keep their policy API."""
    assert VU.classify_measurement_outcome(1.0, 2.0)["outcome"] == "SUCCESS" and VU.classify_measurement_outcome(1.0, 2.0, quality_flags=["x"])["outcome"] == "QUESTIONABLE"
