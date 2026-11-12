"""Measurement uncertainty budget (validated recipe).  Components are 1-sigma, combined in quadrature, reported at +-1.96 sigma.

  1. frame-set / matching sampling : stratified match+track bootstrap of the full Method J estimator
  2. method choice                 : half the gap between Method A and Method J
  3. boundary observations         : +-2 px (corner lines), +-3 px (top point) perturbations, both top-corner variants
  4. camera intrinsics             : random perturbation fx (Air 10 px, Pro 5 px), principal point 4 px, propagated through
                                     the whole joint estimator   <-- dominant term; the magnitudes are ASSUMPTIONS, not measurements.
Nothing here uses ground truth.
"""
import numpy as np
from . import geometry as G, lines as L, pose as P, measure as M

ASSUMPTIONS = dict(
    intrinsics_fx_sd_px=dict(air=10.0, pro=5.0), intrinsics_principal_point_sd_px=4.0,
    boundary_line_sd_px=2.0, top_point_sd_px=3.0, interval="+-1.96 sigma of the quadrature total (normal approximation)",
    intrinsics_note="Assumed magnitudes inherited from the exploratory audit; Recipe B (an alternative fit) moves the width by about -9 %, so the true calibration error may be larger or smaller.",
    not_included=["systematic error in the manual boundary seeds shared by all frame sets", "lens-model error beyond the perturbed parameters", "rig motion between non-synchronised Air/Pro frames"])

def _dims(rows):
    W = np.array([r["width_mm"] for r in rows]); H = np.array([r["height_mm"] for r in rows]); ok = np.isfinite(W) & np.isfinite(H)
    return (float(W[ok].mean()), float(H[ok].mean())) if ok.sum() else (np.nan, np.nan)

def _estimate(cal, D, match_sets, line_sets):
    pA, infoA, recA = P.fit_A_guarded(D)
    if recA["degenerate"]: return None
    pJ, infoJ = P.fit_J(D, pA); return pJ, pA

def match_bootstrap(cal, D, match_sets, line_sets, n=40, seed=2024):
    rng = np.random.default_rng(seed); rows = []; failed = 0; fallback = 0
    for _ in range(n):
        Db = P.resample_dataset(D, rng)
        try:
            pA, infoA, recA = P.fit_A_guarded(Db); fallback += recA["fallback_used"]
            if recA["degenerate"]: failed += 1; continue
            pJ, _ = P.fit_J(Db, pA); w, h = _dims(M.measure_all_sets(cal, pJ, match_sets, line_sets))
            if np.isfinite(w): rows.append((w, h, G.yaw_deg(pJ)))
            else: failed += 1
        except Exception: failed += 1
    r = np.array(rows)
    return dict(n_requested=n, n_ok=len(r), n_failed=failed, n_fallback_used=fallback, width_sd_mm=float(r[:, 0].std(ddof=1)), height_sd_mm=float(r[:, 1].std(ddof=1)), yaw_sd_deg=float(r[:, 2].std(ddof=1)),
                width_p2_5_p97_5=np.percentile(r[:, 0], [2.5, 97.5]).tolist(), height_p2_5_p97_5=np.percentile(r[:, 1], [2.5, 97.5]).tolist())

def boundary_perturbation(cal, pJ, match_sets, line_sets, n=300, seed=21, set_index=0):
    R, T = G.pose_from_params(pJ, cal.baseline_mm); rng = np.random.default_rng(seed); g = M.reconstruct_geometry(cal, R, T, match_sets[set_index], rng); W = []; H = []
    for _ in range(n):
        dx = {"V1": (rng.normal(0, 2), rng.normal(0, 2)), "V2": (rng.normal(0, 2), rng.normal(0, 2)), "dy": {"V1": rng.normal(0, 3), "V2": rng.normal(0, 3)}}
        o = M.measure_wall(cal, R, T, g, line_sets[set_index], int(rng.integers(2)), dx); W.append(o["width_mm"]); H.append(o["height_mm"])
    return dict(n=n, width_sd_mm=float(np.std(W, ddof=1)), height_sd_mm=float(np.std(H, ddof=1)))

def perturb_intrinsics(cal, rng):
    K1 = cal.K_left.copy(); K2 = cal.K_right.copy()
    K1[0, 0] += rng.normal(0, 10); K1[1, 1] = K1[0, 0] * cal.K_left[1, 1] / cal.K_left[0, 0]; K2[0, 0] += rng.normal(0, 5); K2[1, 1] = K2[0, 0] * cal.K_right[1, 1] / cal.K_right[0, 0]
    K1[:2, 2] += rng.normal(0, 4, 2); K2[:2, 2] += rng.normal(0, 4, 2); return cal.with_intrinsics(K_left=K1, K_right=K2)

def intrinsics_perturbation(cal, match_sets, line_sets, n=30, seed=77):
    rng = np.random.default_rng(seed); rows = []; failed = 0
    for _ in range(n):
        c = perturb_intrinsics(cal, rng)
        try:
            Lr = [L.retarget_lines(l, cal, c) for l in line_sets]; D = P.build_dataset(c, match_sets); est = _estimate(c, D, match_sets, Lr)
            if est is None: failed += 1; continue
            w, h = _dims(M.measure_all_sets(c, est[0], match_sets, Lr))
            if np.isfinite(w): rows.append((w, h, G.yaw_deg(est[0])))
            else: failed += 1
        except Exception: failed += 1
    r = np.array(rows)
    return dict(n_requested=n, n_ok=len(r), n_failed=failed, width_sd_mm=float(r[:, 0].std(ddof=1)), height_sd_mm=float(r[:, 1].std(ddof=1)), yaw_sd_deg=float(r[:, 2].std(ddof=1)),
                width_p2_5_p97_5=np.percentile(r[:, 0], [2.5, 97.5]).tolist(), height_p2_5_p97_5=np.percentile(r[:, 1], [2.5, 97.5]).tolist())

def combine(components, width_mm, height_mm, z=1.96):
    sw = float(np.sqrt(sum(c["width_sd_mm"] ** 2 for c in components.values()))); sh = float(np.sqrt(sum(c["height_sd_mm"] ** 2 for c in components.values())))
    return dict(width=dict(estimate_mm=width_mm, sd_mm=sw, rel_sd=sw / width_mm, interval_95_mm=[width_mm - z * sw, width_mm + z * sw], half_width_mm=z * sw),
                height=dict(estimate_mm=height_mm, sd_mm=sh, rel_sd=sh / height_mm, interval_95_mm=[height_mm - z * sh, height_mm + z * sh], half_width_mm=z * sh))

def estimate_uncertainty(cal, D, match_sets, line_sets, pJ, pA, width_mm, height_mm, n_bootstrap=40, n_intrinsics=30, n_boundary=300):
    wa, ha = _dims(M.measure_all_sets(cal, pA, match_sets, line_sets)); comp = {}; detail = {}
    detail["match_bootstrap"] = match_bootstrap(cal, D, match_sets, line_sets, n_bootstrap); comp["frame_set_and_matching"] = dict(width_sd_mm=detail["match_bootstrap"]["width_sd_mm"], height_sd_mm=detail["match_bootstrap"]["height_sd_mm"])
    comp["method_choice_A_vs_J"] = dict(width_sd_mm=abs(wa - width_mm) / 2, height_sd_mm=abs(ha - height_mm) / 2); detail["method_A"] = dict(width_mm=wa, height_mm=ha)
    detail["boundary"] = boundary_perturbation(cal, pJ, match_sets, line_sets, n_boundary); comp["boundary_observations"] = dict(width_sd_mm=detail["boundary"]["width_sd_mm"], height_sd_mm=detail["boundary"]["height_sd_mm"])
    detail["intrinsics"] = intrinsics_perturbation(cal, match_sets, line_sets, n_intrinsics); comp["camera_intrinsics"] = dict(width_sd_mm=detail["intrinsics"]["width_sd_mm"], height_sd_mm=detail["intrinsics"]["height_sd_mm"])
    return dict(components_1sigma=comp, total=combine(comp, width_mm, height_mm), detail=detail, assumptions=ASSUMPTIONS)
