"""F8.30D-B Stage 1: audit of the F8.30D-A adapter against the original Method J.  SYNTHETIC fixtures only.

No video, no .MOV, no real calibration, no real room.  Method J (pose / geometry / measure / lines) runs unmodified; where a test calls a Method J function
directly it passes the arguments Method J itself uses (e.g. ``robust_lm`` with the solver's own tolerances).  The known wall size is a sanity reference for
assertions only and is never an input to a reconstruction.

Each test answers one audit question; the answers are summarised in ARCHITECTURE_PLAN.md section 16.  Numeric expectations were measured on OpenCV 5.0.0 / NumPy 2.5.3.
"""
import copy
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE))

from room_measurement import geometry as G, lines as L, measure as M, pose as P, visitor_measurement as VM, visitor_uncertainty as VU      # noqa: E402
from room_measurement import visitor_method_j_adapter as A                                                                              # noqa: E402
from room_measurement import visitor_method_j_adapter_synthetic as S                                                                    # noqa: E402

TW, TH = S.WALL_WIDTH_MM, S.WALL_HEIGHT_MM


# ------------------------------------------------------------------ helpers
def with_focal(ctx, fac):
    p = ctx.to_params()
    for k in ("K_left", "K_right"): p[k][0, 0] *= fac; p[k][1, 1] *= fac
    return A.VisitorCalibrationContext.from_params(p, ctx.image_size_wh, label=f"focal x{fac}")


def with_calibration(ctx, df=0.0, dk1=0.0, dcx=0.0):
    p = ctx.to_params()
    for k in ("K_left", "K_right"): p[k][0, 0] *= 1 + df; p[k][1, 1] *= 1 + df; p[k][0, 2] += dcx
    p["D_left"][0] += dk1; p["D_right"][0] += dk1
    return A.VisitorCalibrationContext.from_params(p, ctx.image_size_wh, label="B")


def pose_report(ctx, inp):
    """Real Method J pose estimation; returns (pJ, report, extra)."""
    D = P.build_dataset(ctx.to_stereo_calibration(), list(inp.match_sets)); pJ, pA, rep, extra = P.estimate_pose(D); return pJ, rep, extra


def original_pipeline_would_raise(rep):
    """The exact condition of pipeline.run_estimation (pipeline.py:48)."""
    return bool(rep["method_J"]["flags"]["degenerate"])


def direct_lines(s):
    """The line sets Method J receives for scenario ``s``: re-expressed under the evaluation calibration with the original ``retarget_lines`` when it differs from the true one."""
    if "true" not in s: return list(s["inp"].line_sets)
    return [L.retarget_lines(Ls, s["true"].to_stereo_calibration(), s["ctx"].to_stereo_calibration()) for Ls in s["inp"].line_sets]


def rel_err(v, truth): return abs(v / truth - 1.0)


def worst(ev, key): return max(max(p[key]) for p in ev["diagnostics"]["per_set"])


@pytest.fixture(scope="module")
def clean():
    ctx = S.make_context(); inp, _ = S.make_scene(ctx); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def noisy():        # C1: epipolar start degenerate after the fallback (robust scale > 1 px from noise alone; the pose is in fact fine)
    ctx = S.make_context(); inp, _ = S.make_scene(ctx, noise_px=3.0); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def fallback_ok():  # C2: default start lands in a cheirality-failed basin, the fallback start recovers a valid pose (a natural occurrence on this fixture)
    ctx = S.make_context(baseline_mm=250.0); inp, _ = S.make_scene(ctx, noise_px=0.1, seed=5); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def joint_not_converged():   # C3: joint refinement stops at its iteration cap; epipolar start fine; calibration focal 20 % too short
    base = S.make_context(); inp, _ = S.make_scene(base); ctx = with_focal(base, 0.8); return dict(true=base, ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def joint_scale_high():      # C4: epipolar start resolved (by the fallback), joint robust scale > 1 px; calibration focal 40 % too short
    base = S.make_context(); inp, _ = S.make_scene(base); ctx = with_focal(base, 0.6); return dict(true=base, ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


# ------------------------------------------------------------------ 1. degenerate epipolar start, fallback unsuccessful
def test_c1_epipolar_start_degenerate_and_fallback_unsuccessful_is_failed_with_no_dimensions(noisy):
    ev = noisy["ev"]; g = ev["diagnostics"]["pose"]["epipolar_guard"]
    assert g["first_attempt"]["degenerate"] and g["fallback_used"] and g["resolved"] is False and g["unresolved_degenerate"] is True and g["final"]["scale_px"] > P.DEGENERATE_SCALE_PX
    assert ev["outcome"] == A.FAILED and [f["code"] for f in ev["failures"]][0] == "POSE_DEGENERATE_UNRECOVERABLE" and "POSE_JOINT_SCALE_ABOVE_LIMIT" in [f["code"] for f in ev["failures"]] and ev["width_mm"] is None and ev["height_mm"] is None


def test_c1_the_original_pipeline_also_stops_here(noisy):
    pJ, rep, _ = pose_report(noisy["ctx"], noisy["inp"])
    assert original_pipeline_would_raise(rep) and rep["method_A"]["guard"]["degenerate"]


def test_c1_the_scale_rule_cannot_tell_noise_from_a_wrong_basin(noisy):
    """Finding: the same flag fires for uniformly noisy but geometrically sound data.  Method J's numbers (withheld by the adapter, as by the original pipeline)
    are accurate here, so the flag means 'cannot certify the pose', not 'the geometry is wrong'."""
    pJ, rep, _ = pose_report(noisy["ctx"], noisy["inp"]); assert np.isfinite(pJ).all()
    rows = VM._measure_sets(noisy["ctx"].to_stereo_calibration(), pJ, list(noisy["inp"].match_sets), list(noisy["inp"].line_sets), noisy["ctx"].image_size_wh)
    ok = [r for r in rows if r["ok"]]; assert len(ok) == len(rows)
    assert rel_err(np.mean([r["width_mm"] for r in ok]), TW) < 0.05 and rel_err(np.mean([r["height_mm"] for r in ok]), TH) < 0.05


# ------------------------------------------------------------------ 2. degenerate start recovered by the fallback
def test_c2_fallback_recovers_and_clears_the_degenerate_condition(fallback_ok):
    ev = fallback_ok["ev"]; g = ev["diagnostics"]["pose"]["epipolar_guard"]
    assert g["first_attempt"]["degenerate"] and g["first_attempt"]["positive_depth_fraction"] < P.MIN_POSITIVE_DEPTH_FRACTION       # the default start is in a cheirality-failed basin
    assert g["fallback_used"] and g["resolved"] is True and g["unresolved_degenerate"] is False and not g["final"]["degenerate"]
    assert ev["diagnostics"]["pose"]["degenerate_flag"] is False                                                                      # the joint flag inherits the FINAL epipolar verdict
    pJ, rep, _ = pose_report(fallback_ok["ctx"], fallback_ok["inp"]); assert not original_pipeline_would_raise(rep)


def test_c2_dimensions_after_a_recovered_fallback_are_as_accurate_as_without_it(fallback_ok):
    ev = fallback_ok["ev"]; assert ev["width_mm"] is not None and rel_err(ev["width_mm"], TW) < 0.02 and rel_err(ev["height_mm"], TH) < 0.02
    ctx = fallback_ok["ctx"]; inp, _ = S.make_scene(ctx, noise_px=0.1, seed=4); other = A.evaluate(inp, ctx)
    assert not other["diagnostics"]["pose"]["epipolar_guard"]["fallback_used"] and rel_err(other["width_mm"], TW) < 0.02


# ------------------------------------------------------------------ 3. joint degeneracy with finite outputs
def test_c3_joint_not_converged_is_an_iteration_cap_not_an_invalid_pose(joint_not_converged):
    ev = joint_not_converged["ev"]; p = ev["diagnostics"]["pose"]
    assert p["converged"] is False and p["degenerate_flag"] is True and p["epipolar_guard"]["unresolved_degenerate"] is False and p["robust_scale_px"] < P.DEGENERATE_SCALE_PX
    assert ev["outcome"] == A.FAILED and [f["code"] for f in ev["failures"]] == ["POSE_JOINT_NOT_CONVERGED"] and ev["width_mm"] is None and ev["height_mm"] is None          # F8.30D-B 1B (W3): the original raises, so does the adapter
    pJ, rep, extra = pose_report(joint_not_converged["ctx"], joint_not_converged["inp"]); assert rep["method_J"]["iterations"] == 60 and original_pipeline_would_raise(rep)     # the cap in fit_J
    info = extra["infoJ"]; p2, i2 = P.robust_lm(info["res"], info["p_full"], info["w0"], maxit=300, tol_p=1e-8, tol_c=1e-9)                                                 # same solver, same tolerances, more iterations
    assert i2["converged"] and np.abs(p2[:5] - pJ).max() < 1e-5          # continuing moves the pose by < 1e-5 (parameters are in radians): the stop was numerical


# ------------------------------------------------------------------ 4. joint degeneracy with invalid geometry
def test_c4_joint_scale_flag_can_accompany_finite_but_geometrically_invalid_dimensions(joint_scale_high):
    ev = joint_scale_high["ev"]; p = ev["diagnostics"]["pose"]; g = p["epipolar_guard"]
    assert g["fallback_used"] and not g["unresolved_degenerate"] and p["degenerate_flag"] is True and p["robust_scale_px"] > P.DEGENERATE_SCALE_PX
    assert ev["outcome"] == A.FAILED and [f["code"] for f in ev["failures"]] == ["POSE_JOINT_SCALE_ABOVE_LIMIT"] and ev["width_mm"] is None and ev["height_mm"] is None     # F8.30D-B 1B (W1)
    pJ, rep, _ = pose_report(joint_scale_high["ctx"], joint_scale_high["inp"]); assert original_pipeline_would_raise(rep)
    rows = VM._measure_sets(joint_scale_high["ctx"].to_stereo_calibration(), pJ, list(joint_scale_high["inp"].match_sets), direct_lines(joint_scale_high), joint_scale_high["ctx"].image_size_wh)
    W = np.mean([r["width_mm"] for r in rows if r["ok"]]); H = np.mean([r["height_mm"] for r in rows if r["ok"]])
    assert np.isfinite([W, H]).all() and rel_err(W, TW) > 0.10 and rel_err(H, TH) > 0.10                 # the withheld intermediate numbers are finite, positive and wrong by > 10 %
    assert max(max(r["vertical_error_deg"]) for r in rows if r["ok"]) > VM.QualityLimits().vertical_error_deg_warn


def test_finite_positive_dimensions_do_not_imply_a_valid_reconstruction_and_no_pose_flag_is_raised():
    """Calibration mismatches that leave every pose guard quiet: Method J returns finite, positive, wrong dimensions."""
    base = S.make_context(); inp, _ = S.make_scene(base)
    ev = A.evaluate(inp, with_focal(base, 1.8)); p = ev["diagnostics"]["pose"]
    assert p["converged"] and not p["degenerate_flag"] and p["robust_scale_px"] < P.DEGENERATE_SCALE_PX
    assert ev["width_mm"] > 0 and rel_err(ev["width_mm"], TW) > 1.0                                  # 3.9 x the true width
    assert worst(ev, "vertical_error_deg") > VM.QualityLimits().vertical_error_deg_warn and worst(ev, "skew_mm") > VM.QualityLimits().skew_mm_warn
    # ... and the adapter does not use those two visitor-backend checks: no vertical / skew warning code exists (audit finding, WARNING)
    assert not [w for w in ev["warnings"] if "VERTICAL" in w["code"] or "SKEW" in w["code"]] and not [e for e in A.QualityFlag if "VERTICAL" in e.value or "SKEW" in e.value]
    assert ev["outcome"] != A.SUCCESS                                                                 # here only because the registration checks also tripped


# ------------------------------------------------------------------ 5. calibration A versus B line geometry (rendered image evidence)
IMG_W, IMG_H = 1080, 1920


class _Cal:
    def __init__(self, K, D): self.K_left, self.D_left, self.K_right, self.D_right = K, D, K, D


def _K(f, cx=545.0, cy=960.0): return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def _compare_on_image(K_T, D_T, K_B, D_B, kind):
    """The original fit under A, the original fit again under B (image re-rectified with B), and retarget_lines A -> B, on one rendered distorted image."""
    gy, gx = np.mgrid[0:IMG_H, 0:IMG_W]; px = np.c_[gx.ravel(), gy.ravel()].astype(np.float64).reshape(-1, 1, 2)
    nrm = cv2.undistortPoints(px, K_T, D_T).reshape(IMG_H, IMG_W, 2); f = K_T[0, 0]
    d = f * (nrm[..., 0] - (0.05 + 0.02 * nrm[..., 1])) if kind == "v" else f * (nrm[..., 1] - (-0.30 + 0.03 * nrm[..., 0]))
    img = (255 / (1 + np.exp(-d / 1.2))).astype(np.uint8)

    def seed(cK, cD, sg):
        t = np.linspace(-0.35, 0.35, 60); n = np.c_[0.05 + 0.02 * t, t] if kind == "v" else np.c_[t, -0.30 + 0.03 * t]
        dist = cv2.projectPoints(np.c_[n, np.ones(len(n))].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_T, D_T)[0].reshape(-1, 2)
        u = cv2.undistortPoints(dist.reshape(-1, 1, 2), cK, cD, P=cK).reshape(-1, 2)
        if kind == "v": a, b = np.polyfit(u[:, 1], u[:, 0], 1); return dict(kind="v", method="edge_pts", c0=float(a * 960 + b), slope=float(a), ref=960, span=[450, 1450], half=25, sign=sg)
        a, b = np.polyfit(u[:, 0], u[:, 1], 1); return dict(kind="h", method="edge_pts", c0=float(a * 545 + b), slope=float(a), ref=545, span=[60, 1020], half=25, sign=sg)
    for sg in (1, -1):
        try:
            lA = L.fit_lines({"AL": img}, _Cal(K_T, D_T), dict(lines={"AL_x": seed(K_T, D_T, sg)}))["AL_x"]; lB = L.fit_lines({"AL": img}, _Cal(K_B, D_B), dict(lines={"AL_x": seed(K_B, D_B, sg)}))["AL_x"]; break
        except ValueError: continue
    else: pytest.fail("the synthetic edge could not be fitted")
    lR = L.retarget_lines({"AL_x": lA}, _Cal(K_T, D_T), _Cal(K_B, D_B))["AL_x"]
    lo = max(lB["span"][0], lR["span"][0]); hi = min(lB["span"][1], lR["span"][1]); s = np.linspace(lo, hi, 11); at = lambda l: l["a"] * s + l["b"]
    return dict(retarget_vs_refit=float(np.abs(at(lR) - at(lB)).max()), unretargeted_vs_refit=float(np.abs(at(lA) - at(lB)).max()), n=(lA["n"], lB["n"], lR["n"]))


DT = np.array([0.02, -0.06, 0, 0, 0.02]); DSTRONG = np.array([-0.25, 0.12, 0, 0, -0.02])


@pytest.mark.parametrize("kind", ["v", "h"])
def test_retargeting_matches_refitting_on_the_image_to_a_hundredth_of_a_pixel_and_is_needed(kind):
    r = _compare_on_image(_K(1500), DSTRONG, _K(1380, 557.0, 960.0), DSTRONG + [0.06, -0.05, 0, 0, 0], kind)
    assert r["retarget_vs_refit"] < 0.05 and r["n"][0] == r["n"][1] == r["n"][2]
    if kind == "h": assert r["unretargeted_vs_refit"] > 1.0                      # skipping the retarget would leave a > 1 px line error at this calibration difference


def test_retargeting_to_the_same_calibration_changes_nothing():
    r = _compare_on_image(_K(1500), DSTRONG, _K(1500), DSTRONG, "h"); assert r["retarget_vs_refit"] < 1e-3 and r["unretargeted_vs_refit"] < 1e-3      # residual is cv2.undistortPoints' iterative inverse (5 iterations), not the retarget


# ------------------------------------------------------------------ 6. retargeted versus independently refitted lines: final dimensions
def _refit(Ls, cal_true, ctx_to):
    out = {}
    for nm, l in Ls.items():
        left = nm[0] == "A"; K, D = (cal_true.K_left, cal_true.D_left) if left else (cal_true.K_right, cal_true.D_right); K2, D2 = (ctx_to.K_left, ctx_to.D_left) if left else (ctx_to.K_right, ctx_to.D_right)
        pts = l["pts"].astype(float); n = (np.linalg.inv(K) @ np.c_[pts, np.ones(len(pts))].T).T
        E = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)                    # the distorted edge evidence
        out[nm] = L.fit_line(cv2.undistortPoints(E.reshape(-1, 1, 2), K2, D2, P=K2).reshape(-1, 2), l["kind"])
    return out


@pytest.mark.parametrize("df,dk1,dcx", [(0.005, 0.01, 0.0), (0.03, 0.03, 10.0), (-0.05, -0.05, -20.0)])
def test_retargeted_lines_give_the_same_dimensions_as_independently_refitted_lines(clean, df, dk1, dcx):
    ctx, inp = clean["ctx"], clean["inp"]; B = with_calibration(ctx, df, dk1, dcx)
    ret = A.evaluate(inp, B); assert ret["calibration"]["lines_retargeted"] is True
    rf = A.evaluate(A.ReconstructionInputs(inp.match_sets, tuple(_refit(Ls, ctx, B) for Ls in inp.line_sets), B), B)
    assert abs(ret["width_mm"] - rf["width_mm"]) < 1e-6 and abs(ret["height_mm"] - rf["height_mm"]) < 1e-6 and ret["outcome"] == rf["outcome"]
    none = A.evaluate(A.ReconstructionInputs(inp.match_sets, inp.line_sets, B), B)                                          # lines NOT re-expressed
    assert abs(none["width_mm"] - rf["width_mm"]) > 1.0                                                                   # so the retarget does change the answer, and towards the refit


def test_retargeting_is_only_exercised_when_the_calibration_differs_from_the_one_the_lines_were_fitted_under(clean):
    assert clean["ev"]["calibration"]["lines_retargeted"] is False
    p = clean["ctx"].to_params(); p["baseline_mm"] = 123.0
    assert A.evaluate(clean["inp"], A.VisitorCalibrationContext.from_params(p, clean["ctx"].image_size_wh))["calibration"]["lines_retargeted"] is False      # a baseline change alone leaves the lines alone


# ------------------------------------------------------------------ 7. adapter versus the original pipeline's decisions and numbers
def test_adapter_numbers_equal_the_original_measure_all_sets_in_every_scenario_that_returns_numbers(clean, fallback_ok):
    for s in (clean, fallback_ok):
        ctx, inp, ev = s["ctx"], s["inp"], s["ev"]; pJ, rep, _ = pose_report(ctx, inp)
        rows = M.measure_all_sets(ctx.to_stereo_calibration(), pJ, list(inp.match_sets), direct_lines(s)); ok = [r for r in rows if r["ok"]]
        assert len(ok) == len(rows) and abs(ev["width_mm"] - float(np.mean([r["width_mm"] for r in ok]))) < 1e-9 and abs(ev["height_mm"] - float(np.mean([r["height_mm"] for r in ok]))) < 1e-9


def test_decision_table_adapter_versus_original_pipeline(clean, noisy, fallback_ok, joint_not_converged, joint_scale_high):
    """(scenario, original pipeline raises?, adapter outcome).  After F8.30D-B Stage 1B the adapter rejects exactly where the original raises (C1, C3, C4)."""
    rows = {}
    for name, s in (("clean", clean), ("C1", noisy), ("C2", fallback_ok), ("C3", joint_not_converged), ("C4", joint_scale_high)):
        _, rep, _ = pose_report(s["ctx"], s["inp"]); rows[name] = (original_pipeline_would_raise(rep), s["ev"]["outcome"])
    assert rows == {"clean": (False, A.SUCCESS), "C1": (True, A.FAILED), "C2": (False, A.SUCCESS), "C3": (True, A.FAILED), "C4": (True, A.FAILED)}      # F8.30D-B 1B: full agreement


def test_original_pipeline_still_treats_any_joint_degenerate_flag_as_fatal():
    src = (MODULE / "room_measurement/pipeline.py").read_text()
    assert 'if prep["method_J"]["flags"]["degenerate"]: raise RuntimeError' in src


def test_pose_guard_thresholds_are_in_pseudo_pixels_tied_to_f_px_not_to_the_cameras_focal_length():
    """Same TRUE pixel noise (0.7 px), different focal length: the 1 px rule is evaluated on F_PX * normalised residuals (F_PX = 3000)."""
    out = {}
    for f in (1500.0, 3000.0):
        ctx = S.make_context(f=f); inp, _ = S.make_scene(ctx, noise_px=0.7); ev = A.evaluate(inp, ctx); out[f] = (ev["outcome"], ev["diagnostics"]["pose"]["robust_scale_px"] if "pose" in ev["diagnostics"] else None)
    assert out[3000.0][0] != A.FAILED and out[3000.0][1] < 1.0
    assert out[1500.0][0] == A.FAILED and out[1500.0][1] > 1.0 and 1.8 < out[1500.0][1] / out[3000.0][1] < 2.4      # about 3000 / 1500 times larger
    assert G.F_PX == 3000.0


# ------------------------------------------------------------------ 8. typed failure propagation and untouched valid values
def test_typed_failure_reaches_the_uncertainty_engine_and_the_log(noisy):
    log = A.EvaluationLog(); est = A.make_estimator(noisy["inp"], noisy["ctx"], log=log)
    with pytest.raises(VU.EstimatorFailure) as ei: est(noisy["ctx"].to_params())
    assert getattr(ei.value, "code", None) == "POSE_DEGENERATE_UNRECOVERABLE" or "POSE_DEGENERATE_UNRECOVERABLE" in str(ei.value)
    assert log.summary()["failure_codes"] == {"POSE_DEGENERATE_UNRECOVERABLE": 1, "POSE_JOINT_SCALE_ABOVE_LIMIT": 1} and log.records[0]["outcome"] == A.FAILED


def test_valid_measurement_values_are_not_changed_by_any_classification(clean, fallback_ok):
    for s in (clean, fallback_ok):
        ev = s["ev"]; ctx, inp = s["ctx"], s["inp"]; pJ, _, _ = pose_report(ctx, inp)
        rows = VM._measure_sets(ctx.to_stereo_calibration(), pJ, list(inp.match_sets), direct_lines(s), ctx.image_size_wh)
        assert ev["width_mm"] == float(np.mean([r["width_mm"] for r in rows])) and ev["height_mm"] == float(np.mean([r["height_mm"] for r in rows]))      # bit for bit


def test_method_j_files_and_room_processing_gate_are_untouched_by_the_audit():
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain", "--", *(f"room_measurement/{f}.py" for f in ("pose", "measure", "geometry", "lines", "features", "pipeline", "calibration"))],
                         cwd=MODULE, capture_output=True, text=True).stdout
    assert out.strip() == ""
    for p in MODULE.rglob("*.py"):
        if "tests" in p.parts or ".venv" in p.parts: continue
        t = p.read_text(errors="ignore"); assert "ROOM_PROCESSING_ENABLED = True" not in t
