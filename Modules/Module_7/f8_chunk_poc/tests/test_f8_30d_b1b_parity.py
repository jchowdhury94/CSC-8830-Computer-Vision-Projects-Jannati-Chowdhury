"""F8.30D-B Stage 1B: behavioural parity of the Method J adapter with the ORIGINAL pipeline (``pipeline.run_estimation``).  SYNTHETIC fixtures only.

No video, no .MOV, no real calibration, no real room.  Method J (pose / geometry / measure / lines / pipeline) is unmodified and runs for real on the synthetic scene.
``original_decision`` replays the acceptance logic of ``pipeline.run_estimation`` (pipeline.py:47-51) by calling the same original functions: pose estimation, the
joint-degenerate raise, ``measure_all_sets`` and the "no frame set" raise.  The known wall size is a sanity reference only and is never a reconstruction input.
"""
import copy
import re
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE))

from room_measurement import geometry as G, lines as L, measure as M, pose as P, visitor_measurement as VM                  # noqa: E402
from room_measurement import visitor_method_j_adapter as A                                                                 # noqa: E402
from room_measurement import visitor_method_j_adapter_synthetic as S                                                       # noqa: E402

ADAPTER_SRC = MODULE / "room_measurement/visitor_method_j_adapter.py"
TW, TH = S.WALL_WIDTH_MM, S.WALL_HEIGHT_MM


# ------------------------------------------------------------------ helpers
def with_focal(ctx, fac):
    p = ctx.to_params()
    for k in ("K_left", "K_right"): p[k][0, 0] *= fac; p[k][1, 1] *= fac
    return A.VisitorCalibrationContext.from_params(p, ctx.image_size_wh, label=f"focal x{fac}")


def codes(ev, key): return [x["code"] for x in ev[key]]


def real_stages(**over):
    return A.MethodJStages(**{**dict(build_dataset=P.build_dataset, estimate_pose=P.estimate_pose), **over})


def original_decision(ctx, inp, lines=None):
    """What ``pipeline.run_estimation`` does after the data are prepared: (accepted?, width, height).  ``lines`` default to the supplied (same calibration)."""
    cal = ctx.to_stereo_calibration(); D = P.build_dataset(cal, list(inp.match_sets)); pJ, pA, prep, _ = P.estimate_pose(D)
    if prep["method_J"]["flags"]["degenerate"]: return False, None, None, "joint degenerate"             # pipeline.py:48
    rows = M.measure_all_sets(cal, pJ, list(inp.match_sets), list(lines if lines is not None else inp.line_sets)); ok = [r for r in rows if r["ok"]]
    if not ok: return False, None, None, "no frame set"                                                    # pipeline.py:51
    return True, float(np.mean([r["width_mm"] for r in ok])), float(np.mean([r["height_mm"] for r in ok])), "accepted"


def sparse_set(F, n_floor, n_other):
    """A frame set cut down to ``n_floor`` floor-region matches (rows > 2600) plus ``n_other`` others, per view; MX kept only where both keypoints survive."""
    out = dict(F)
    for a, b, m in (("A1", "B1", "M1"), ("A2", "B2", "M2")):
        fl = np.where(F[a][:, 1] > 2600)[0]; ot = np.where(F[a][:, 1] <= 2600)[0]; idx = np.sort(np.r_[fl[:n_floor], ot[:n_other]])
        out[a], out[b], out[m] = F[a][idx], F[b][idx], F[m][idx]
    q1 = set(out["M1"][:, 0].tolist()); q2 = set(out["M2"][:, 0].tolist())
    out["MX"] = np.array([(q, v) for q, v in F["MX"] if q in q1 and v in q2]).reshape(-1, 2)
    return out


@pytest.fixture(scope="module")
def clean():
    ctx = S.make_context(); inp, _ = S.make_scene(ctx); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def epipolar_failed():       # robust scale > 1 px from noise alone: default start degenerate, fallback does not resolve it
    ctx = S.make_context(); inp, _ = S.make_scene(ctx, noise_px=3.0); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def fallback_ok():           # default start in a cheirality-failed basin, fallback recovers a valid pose
    ctx = S.make_context(baseline_mm=250.0); inp, _ = S.make_scene(ctx, noise_px=0.1, seed=5); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx))


@pytest.fixture(scope="module")
def not_converged():         # joint refinement stops at its iteration cap (calibration focal 20 % too short)
    base = S.make_context(); inp, _ = S.make_scene(base); ctx = with_focal(base, 0.8); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx), base=base)


@pytest.fixture(scope="module")
def scale_high():            # epipolar start resolved, joint robust scale > 1 px (calibration focal 40 % too short)
    base = S.make_context(); inp, _ = S.make_scene(base); ctx = with_focal(base, 0.6); return dict(ctx=ctx, inp=inp, ev=A.evaluate(inp, ctx), base=base)


def retargeted(s): return [L.retarget_lines(Ls, s["base"].to_stereo_calibration(), s["ctx"].to_stereo_calibration()) for Ls in s["inp"].line_sets]


# ------------------------------------------------------------------ W1: joint scale above 1 px
def test_w1_joint_scale_above_one_pixel_is_failed_like_the_original(scale_high):
    ev = scale_high["ev"]; p = ev["diagnostics"]["pose"]
    assert p["robust_scale_px"] > P.DEGENERATE_SCALE_PX == 1.0 and not p["epipolar_guard"]["unresolved_degenerate"] and p["converged"]
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["POSE_JOINT_SCALE_ABOVE_LIMIT"] and ev["failures"][0]["value"] == p["robust_scale_px"]
    assert ev["width_mm"] is None and ev["height_mm"] is None and ev["warnings"] == [] and not ev["partial"]
    assert original_decision(scale_high["ctx"], scale_high["inp"], retargeted(scale_high))[0] is False


def test_w1_the_numerical_threshold_is_the_original_one():
    src = (MODULE / "room_measurement/pose.py").read_text()
    assert re.search(r"^DEGENERATE_SCALE_PX = 1\.0\b", src, re.M) and "P.DEGENERATE_SCALE_PX" in ADAPTER_SRC.read_text()
    assert not re.search(r"scale_px\"?\]?\s*>\s*\d", ADAPTER_SRC.read_text()), "the adapter must not carry its own numeric scale threshold"


# ------------------------------------------------------------------ epipolar degeneracy after a failed fallback
def test_epipolar_degeneracy_after_failed_fallback_is_failed_like_the_original(epipolar_failed):
    ev = epipolar_failed["ev"]; g = ev["diagnostics"]["pose"]["epipolar_guard"]
    assert g["first_attempt"]["degenerate"] and g["fallback_used"] and g["resolved"] is False and g["unresolved_degenerate"]
    assert ev["outcome"] == A.FAILED and codes(ev, "failures")[0] == "POSE_DEGENERATE_UNRECOVERABLE" and ev["width_mm"] is None and ev["height_mm"] is None
    assert original_decision(epipolar_failed["ctx"], epipolar_failed["inp"])[0] is False


# ------------------------------------------------------------------ W2: successful fallback
def test_w2_successful_fallback_is_success_with_a_typed_note_and_original_numbers(fallback_ok):
    ev = fallback_ok["ev"]; g = ev["diagnostics"]["pose"]["epipolar_guard"]
    assert g["fallback_used"] and g["resolved"] is True and not g["unresolved_degenerate"]
    assert ev["outcome"] == A.SUCCESS and ev["failures"] == [] and ev["warnings"] == []
    assert [n["code"] for n in ev["diagnostic_notes"]] == [A.NOTE_FALLBACK_START_USED] and ev["diagnostic_notes"][0]["stage"] == "pose"
    ok, w, h, why = original_decision(fallback_ok["ctx"], fallback_ok["inp"])
    assert ok and ev["width_mm"] == w and ev["height_mm"] == h                                              # bit for bit


def test_w2_a_fallback_that_does_not_pass_the_original_checks_gets_no_success_note(epipolar_failed):
    assert epipolar_failed["ev"]["diagnostic_notes"] == []


# ------------------------------------------------------------------ joint non-convergence
def test_joint_non_convergence_is_failed_like_the_original(not_converged):
    ev = not_converged["ev"]; p = ev["diagnostics"]["pose"]
    assert p["converged"] is False and p["robust_scale_px"] < P.DEGENERATE_SCALE_PX and not p["epipolar_guard"]["unresolved_degenerate"]
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["POSE_JOINT_NOT_CONVERGED"] and ev["width_mm"] is None
    assert original_decision(not_converged["ctx"], not_converged["inp"], retargeted(not_converged))[0] is False


def test_every_firing_cause_is_typed_when_several_apply(clean):
    def edit(D):
        pJ, pA, rep, ex = P.estimate_pose(D); rep = copy.deepcopy(rep); rep["method_A"]["guard"].update(degenerate=True, resolved=False, fallback_used=True)
        rep["method_J"]["flags"].update(converged=False, scale_px=2.5, degenerate=True); return pJ, pA, rep, ex
    ev = A.evaluate(clean["inp"], clean["ctx"], stages=real_stages(estimate_pose=edit))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["POSE_DEGENERATE_UNRECOVERABLE", "POSE_JOINT_NOT_CONVERGED", "POSE_JOINT_SCALE_ABOVE_LIMIT"]
    assert ev["diagnostics"]["original_pipeline_decision"]["rejected"] is True


def test_a_degenerate_flag_without_a_visible_cause_is_still_the_originals_decision(clean):
    def edit(D):
        pJ, pA, rep, ex = P.estimate_pose(D); rep = copy.deepcopy(rep); rep["method_J"]["flags"]["degenerate"] = True; return pJ, pA, rep, ex
    ev = A.evaluate(clean["inp"], clean["ctx"], stages=real_stages(estimate_pose=edit))
    assert ev["outcome"] == A.FAILED and ev["width_mm"] is None


# ------------------------------------------------------------------ W3: intermediate dimensions are never returned for a rejected reconstruction
@pytest.mark.parametrize("name", ["epipolar_failed", "not_converged", "scale_high"])
def test_w3_rejected_reconstructions_return_no_dimensions_and_keep_pose_diagnostics_separately(name, request):
    ev = request.getfixturevalue(name)["ev"]
    assert ev["outcome"] == A.FAILED and ev["width_mm"] is None and ev["height_mm"] is None and not ev["partial"]
    assert ev["dimensions"] == dict(width=dict(status="unavailable", value=None, n_sets=0), height=dict(status="unavailable", value=None, n_sets=0))
    assert "per_set" not in ev["diagnostics"] and "set_failures" not in ev["diagnostics"]                     # geometry is not run for a pose the original rejects
    pose = ev["diagnostics"]["pose"]; assert np.isfinite(pose["robust_scale_px"]) and pose["params"] and "epipolar_guard" in pose and "n_inlier" in pose
    assert ev["diagnostics"]["original_pipeline_decision"]["rejected"] is True and ev["diagnostics"]["original_pipeline_decision"]["causes"] == codes(ev, "failures")


@pytest.mark.parametrize("stage, exc, code", [("build_dataset", ValueError("bad"), "DATASET_BUILD_FAILED"), ("estimate_pose", np.linalg.LinAlgError("singular"), "POSE_SOLVER_EXCEPTION"),
                                              ("measure_sets", RuntimeError("geometry"), "GEOMETRY_FAILED")])
def test_original_pipeline_exceptions_become_typed_failures(clean, stage, exc, code):
    def boom(*a, **k): raise exc
    ev = A.evaluate(clean["inp"], clean["ctx"], stages=real_stages(**{stage: boom}))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == [code] and ev["width_mm"] is None and ev["height_mm"] is None and exc.args[0] in ev["failures"][0]["message"]


def test_all_frame_sets_failing_is_failed_like_the_original_no_frame_set_raise(clean):
    def all_bad(cal, p5, ms, ls, wh, seed=3): return [dict(ok=False, code="FLOOR_PLANE_FAILED", error="x", width_mm=float("nan"), height_mm=float("nan")) for _ in ms]
    ev = A.evaluate(clean["inp"], clean["ctx"], stages=real_stages(measure_sets=all_bad))
    assert ev["outcome"] == A.FAILED and codes(ev, "failures") == ["NO_USABLE_FRAME_SETS"] and ev["width_mm"] is None


# ------------------------------------------------------------------ minimum-point edge cases (W6)
@pytest.mark.parametrize("k", [(3, 3), (4, 4), (5, 5), (6, 3), (10, 5), (40, 10), (1, 1), (2, 0), (0, 2), (0, 0)])
def test_w6_geometry_stage_accepts_and_rejects_the_same_sets_as_the_original_with_identical_numbers(clean, k):
    ctx, inp = clean["ctx"], clean["inp"]; cal = ctx.to_stereo_calibration(); pJ = np.array(clean["ev"]["diagnostics"]["pose"]["params"])
    sets = [sparse_set(inp.match_sets[0], *k)] + list(inp.match_sets[1:])
    orig = M.measure_all_sets(cal, pJ, sets, list(inp.line_sets))[0]; mine = A.measure_sets_original(cal, pJ, sets, list(inp.line_sets), ctx.image_size_wh)[0]
    assert orig["ok"] == mine["ok"]
    if orig["ok"]: assert mine["width_mm"] == orig["width_mm"] and mine["height_mm"] == orig["height_mm"]
    else: assert mine["code"] in {e.value for e in A.FailureReason} and np.isnan(mine["width_mm"])


def test_w6_the_visitor_mirror_was_stricter_than_the_original_and_the_adapter_no_longer_is(clean):
    ctx, inp = clean["ctx"], clean["inp"]; cal = ctx.to_stereo_calibration(); pJ = np.array(clean["ev"]["diagnostics"]["pose"]["params"])
    sets = [sparse_set(inp.match_sets[0], 3, 3)] + list(inp.match_sets[1:])
    assert M.measure_all_sets(cal, pJ, sets, list(inp.line_sets))[0]["ok"] is True
    assert VM._measure_sets(cal, pJ, sets, list(inp.line_sets), ctx.image_size_wh)[0]["ok"] is False             # the extra >= 10 gated points guard (visitor_measurement.py:350)
    assert A.measure_sets_original(cal, pJ, sets, list(inp.line_sets), ctx.image_size_wh)[0]["ok"] is True


def test_w6_a_sparse_frame_set_is_not_rejected_at_input_and_set_level_decisions_match_the_original(clean):
    ctx, inp = clean["ctx"], clean["inp"]; sets = (sparse_set(inp.match_sets[0], 3, 3),) + tuple(inp.match_sets[1:])
    new = A.ReconstructionInputs(sets, inp.line_sets, inp.lines_calibration); assert new.problems(ctx) == []
    ev = A.evaluate(new, ctx); assert ev["outcome"] != A.FAILED
    cal = ctx.to_stereo_calibration(); pJ = np.array(ev["diagnostics"]["pose"]["params"]); rows = M.measure_all_sets(cal, pJ, list(sets), list(inp.line_sets))
    got = {p["frame_set"] for p in ev["diagnostics"]["per_set"]}; assert got == {i for i, r in enumerate(rows) if r["ok"]}
    assert ev["width_mm"] == float(np.mean([r["width_mm"] for r in rows if r["ok"]]))


def test_w6_set_failures_where_the_original_raises_are_typed(clean):
    ctx, inp = clean["ctx"], clean["inp"]; cal = ctx.to_stereo_calibration(); pJ = np.array(clean["ev"]["diagnostics"]["pose"]["params"])
    F0 = sparse_set(inp.match_sets[0], 0, 2)                                                                  # < 3 floor candidates: the original plane fit raises
    r = A.measure_sets_original(cal, pJ, [F0], [inp.line_sets[0]], ctx.image_size_wh)[0]
    assert r["ok"] is False and r["code"] == "FLOOR_PLANE_FAILED" and "plane fit needs at least 3 points" in r["error"]
    F1 = dict(clean["inp"].match_sets[0]); F1["MX"] = F1["MX"][:3]                                          # < 6 registration pairs: the original raises
    r = A.measure_sets_original(cal, pJ, [F1], [inp.line_sets[0]], ctx.image_size_wh)[0]
    assert r["ok"] is False and r["code"] == "REGISTRATION_FAILED" and "registration pairs" in r["error"]


def test_basic_input_validation_is_structural_not_a_quality_threshold(clean):
    ctx, inp = clean["ctx"], clean["inp"]
    one = tuple({**F, "A1": F["A1"][:3], "B1": F["B1"][:3]} for F in inp.match_sets)                           # A1/B1 shorter than M1: malformed structure
    assert [p["code"] for p in A.ReconstructionInputs(one, inp.line_sets, inp.lines_calibration).problems(ctx)][0] == "INPUT_MALFORMED_FRAME_SET"
    few = (sparse_set(inp.match_sets[0], 1, 1),)                                                              # 4 matches in total: the 8-point initialisation is undetermined
    assert "only 4 stereo matches" in A.ReconstructionInputs(few, inp.line_sets[:1], inp.lines_calibration).problems(ctx)[0]["message"]
    ok_small = (sparse_set(inp.match_sets[0], 3, 3),) + tuple(inp.match_sets[1:]); assert A.ReconstructionInputs(ok_small, inp.line_sets, inp.lines_calibration).problems(ctx) == []
    assert not hasattr(A, "MIN_GATED_POINTS") and A.MIN_STEREO_MATCHES == 8


# ------------------------------------------------------------------ W5 / W6: vertical-edge and skew are recorded, never acceptance criteria
def test_w5_the_original_pipeline_does_not_use_vertical_edge_error_or_skew_for_acceptance():
    for f in ("pipeline.py", "pose.py", "uncertainty.py"):
        t = (MODULE / "room_measurement" / f).read_text(); assert "vertical_error" not in t and "skew" not in t, f
    m = (MODULE / "room_measurement/measure.py").read_text()
    assert "vertical_error_deg=vang" in m and "skew=skew" in m and not re.search(r"(vang|skew|vertical_error_deg)\s*[<>]", m)      # computed and returned, never compared


def test_w5_a_visibly_inconsistent_geometry_that_the_original_accepts_is_not_rejected_by_the_adapter(clean):
    """Focal x1.8: every pose guard stays quiet, the original returns (wrong) numbers; the adapter returns the same numbers and only records the consistency values."""
    ctx = with_focal(clean["ctx"], 1.8); inp = clean["inp"]; lines = [L.retarget_lines(Ls, clean["ctx"].to_stereo_calibration(), ctx.to_stereo_calibration()) for Ls in inp.line_sets]
    ok, w, h, why = original_decision(ctx, inp, lines); ev = A.evaluate(inp, ctx)
    assert ok and ev["outcome"] != A.FAILED and ev["width_mm"] == w and ev["height_mm"] == h and abs(w / TW - 1) > 1.0
    gc = ev["diagnostics"]["geometry_consistency_informational"]; assert gc["max_vertical_error_deg"] > VM.QualityLimits().vertical_error_deg_warn and "does not use" in gc["note"]
    assert not any("VERTICAL" in c or "SKEW" in c for c in codes(ev, "warnings") + codes(ev, "failures"))


# ------------------------------------------------------------------ numerical parity
@pytest.mark.parametrize("name", ["clean", "fallback_ok"])
def test_valid_dimensions_equal_direct_method_j_execution_exactly(name, request):
    s = request.getfixturevalue(name); ok, w, h, _ = original_decision(s["ctx"], s["inp"])
    assert ok and s["ev"]["outcome"] == A.SUCCESS and s["ev"]["width_mm"] == w and s["ev"]["height_mm"] == h


def test_pose_parameters_equal_direct_estimate_pose(clean):
    cal = clean["ctx"].to_stereo_calibration(); pJ = P.estimate_pose(P.build_dataset(cal, list(clean["inp"].match_sets)))[0]
    assert clean["ev"]["diagnostics"]["pose"]["params"] == np.asarray(pJ, float).tolist()


# ------------------------------------------------------------------ original versus adapter acceptance parity
def test_acceptance_parity_table(clean, epipolar_failed, fallback_ok, not_converged, scale_high):
    table = {}
    for name, s in (("clean", clean), ("epipolar_failed", epipolar_failed), ("fallback_ok", fallback_ok), ("not_converged", not_converged), ("scale_high", scale_high)):
        lines = retargeted(s) if "base" in s else None
        ok, w, h, why = original_decision(s["ctx"], s["inp"], lines); ev = s["ev"]
        assert (ev["outcome"] != A.FAILED) == ok, (name, why, ev["outcome"])
        assert (ev["width_mm"], ev["height_mm"]) == ((w, h) if ok else (None, None)), name
        table[name] = (ok, ev["outcome"])
    assert table == {"clean": (True, A.SUCCESS), "epipolar_failed": (False, A.FAILED), "fallback_ok": (True, A.SUCCESS), "not_converged": (False, A.FAILED), "scale_high": (False, A.FAILED)}


def test_acceptance_parity_for_garbage_matches(clean):
    rng = np.random.default_rng(0); sets = []
    for F in clean["inp"].match_sets:
        G_ = dict(F); G_["B1"] = rng.uniform(0, 2160, (len(F["A1"]), 2)).astype(np.float32); G_["B2"] = rng.uniform(0, 3840, (len(F["A2"]), 2)).astype(np.float32); sets.append(G_)
    inp = A.ReconstructionInputs(tuple(sets), clean["inp"].line_sets, clean["ctx"])
    with np.errstate(all="ignore"):
        ev = A.evaluate(inp, clean["ctx"])
        try: ok = original_decision(clean["ctx"], inp)[0]
        except Exception: ok = False                                                                          # the original raises: the adapter returns a typed failure instead
    assert ok is False and ev["outcome"] == A.FAILED and ev["width_mm"] is None and codes(ev, "failures")


# ------------------------------------------------------------------ typed diagnostics and safety gates
def test_failure_and_warning_codes_are_declared_and_disjoint(scale_high, epipolar_failed, not_converged, fallback_ok):
    fv = {e.value for e in A.FailureReason}; wv = {e.value for e in A.QualityFlag}; assert not fv & wv
    for ev in (scale_high["ev"], epipolar_failed["ev"], not_converged["ev"], fallback_ok["ev"]):
        assert set(codes(ev, "failures")) <= fv and set(codes(ev, "warnings")) <= wv
    assert {"POSE_ROBUST_SCALE_HIGH", "POSE_FALLBACK_START_USED"}.isdisjoint(wv) and "POSE_JOINT_NOT_CONVERGED" not in wv                     # no longer warnings


def test_rejected_pose_is_reported_to_the_uncertainty_engine_as_a_typed_failure(scale_high):
    from room_measurement import visitor_uncertainty as VU
    log = A.EvaluationLog(); est = A.make_estimator(scale_high["inp"], scale_high["ctx"], log=log)
    with pytest.raises(VU.EstimatorFailure): est(scale_high["ctx"].to_params())
    assert log.summary()["failure_codes"] == {"POSE_JOINT_SCALE_ABOVE_LIMIT": 1} and log.records[0]["outcome"] == A.FAILED


def test_room_processing_gate_stays_closed_and_the_adapter_is_not_wired(monkeypatch):
    monkeypatch.setenv("ROOM_PROCESSING_ENABLED", "1"); monkeypatch.setenv("F8_ENABLE_ROOM_PROCESSING", "1")
    sys.path.insert(0, str(MODULE / "visitor_workflow"))
    import f8_visitor_service as FS
    assert FS.ROOM_PROCESSING_ENABLED is False and re.search(r"^ROOM_PROCESSING_ENABLED = False\b", (MODULE / "visitor_workflow/f8_visitor_service.py").read_text(), re.M)
    assert "ROOM_PROCESSING_ENABLED" not in ADAPTER_SRC.read_text() and "environ" not in ADAPTER_SRC.read_text()
    for f in (MODULE / "visitor_workflow").glob("*.py"):                                                   # F8.31C-1: only the two backend files may use the adapter; no page, service or app does
        if f.name not in ("room_reconstruction.py", "reconstruction_worker.py"): assert "visitor_method_j_adapter" not in f.read_text(), f.name


def test_method_j_sources_are_unmodified():
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain", "--", *(f"room_measurement/{f}.py" for f in ("pose", "measure", "geometry", "lines", "features", "pipeline", "calibration"))],
                         cwd=MODULE, capture_output=True, text=True).stdout
    assert out.strip() == ""
