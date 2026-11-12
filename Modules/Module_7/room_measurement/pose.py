"""Method J -- jointly refined shared stereo-rig pose.

Method A (epipolar only) and Method J (A + rigid four-view track reprojection) share one deterministic robust
Gauss-Newton/LM solver (Tukey IRLS).  No random numbers are used anywhere in this module.

Objective of J (parameters: 5 stereo pose params + 6 rig-motion params per frame set):
    sum_over_matches  w_i * rho( Sampson_i )                      # epipolar term, all frame sets pooled
  + sum_over_tracks   w_t * rho( reprojection of the 4-view track in AL, PL, AR, PR )   # rigid-scene term
with rho = Tukey biweight (c = 4.685, scale = max(1.4826*MAD, 0.15 px)).  The physical baseline is fixed.
No wall dimension and no camera height enter the objective.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from . import geometry as G
from .features import build_tracks

SCALE_FLOOR_PX = 0.15
DEGENERATE_SCALE_PX = 1.0          # robust residual scale above this => wrong basin (validated rule)
MIN_INLIER_FRACTION = 0.85
MIN_POSITIVE_DEPTH_FRACTION = 0.98
GRID_PX = 128
# validated fallback start: parallel cameras, baseline along -x (a generic rig prior; no data-specific numbers)
FALLBACK_START = G.params_from_pose(np.eye(3), np.array([-1.0, 0.0, 0.001]))

# ---------------------------------------------------------------- dataset
def build_dataset(cal, match_sets, grid=GRID_PX):
    """Pool the stereo matches of all frame sets (normalised coords) with spatial de-clustering and per-set equal weight,
    and assemble the cross-view four-view tracks."""
    n1, n2, sid, w = [], [], [], []
    for si, F in enumerate(match_sets):
        for A, B in (("A1", "B1"), ("A2", "B2")):
            a, b = F[A], F[B]; n1.append(G.undistort_normalized(a, cal.K_left, cal.D_left)); n2.append(G.undistort_normalized(b, cal.K_right, cal.D_right)); sid += [si] * len(a)
            cell = (a[:, 0] // grid).astype(int) * 1000 + (a[:, 1] // grid).astype(int); _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True); w.append(1.0 / cnt[inv])
    n1, n2, sid, w = np.vstack(n1), np.vstack(n2), np.array(sid), np.concatenate(w)
    for si in range(len(match_sets)): m = sid == si; w[m] = w[m] / w[m].sum() * (len(w) / len(match_sets))
    tsid, obs = [], []
    for si, F in enumerate(match_sets):
        T = build_tracks(F)
        for t in T:
            obs.append([G.undistort_normalized(t[0], cal.K_left, cal.D_left)[0], G.undistort_normalized(t[1], cal.K_right, cal.D_right)[0],
                        G.undistort_normalized(t[2], cal.K_left, cal.D_left)[0], G.undistort_normalized(t[3], cal.K_right, cal.D_right)[0]]); tsid.append(si)
    return dict(n1=n1, n2=n2, sid=sid, w=w, T=dict(sid=np.array(tsid, int), obs=np.array(obs, np.float64).reshape(-1, 4, 2)), nsets=len(match_sets), baseline=cal.baseline_mm)

def resample_dataset(D, rng):
    """Stratified (per frame set) bootstrap resample of stereo matches and tracks."""
    idx = np.concatenate([rng.choice(np.where(D["sid"] == s)[0], size=(D["sid"] == s).sum()) for s in range(D["nsets"])]); T = D["T"]
    ti = np.concatenate([rng.choice(np.where(T["sid"] == s)[0], size=(T["sid"] == s).sum()) for s in range(D["nsets"])])
    return dict(n1=D["n1"][idx], n2=D["n2"][idx], sid=D["sid"][idx], w=D["w"][idx], T=dict(sid=T["sid"][ti], obs=T["obs"][ti]), nsets=D["nsets"], baseline=D["baseline"])

# ---------------------------------------------------------------- initialisation (IRLS 8-point: no RANSAC, no RNG, no stored extrinsics)
def init_pose(n1, n2, w0, baseline_mm, iters=40):
    def cond(x):
        c = x.mean(0); s = np.sqrt(2) / np.mean(np.linalg.norm(x - c, axis=1)); return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    T1, T2 = cond(n1), cond(n2); x1 = np.c_[n1, np.ones(len(n1))] @ T1.T; x2 = np.c_[n2, np.ones(len(n2))] @ T2.T
    A = np.einsum("ni,nj->nij", x2, x1).reshape(len(n1), 9); w = w0.copy(); x1h = np.c_[n1, np.ones(len(n1))]; x2h = np.c_[n2, np.ones(len(n2))]
    for _ in range(iters):
        _, _, Vt = np.linalg.svd((A * w[:, None]).T @ A); Ec = Vt[-1].reshape(3, 3); u, _, vt = np.linalg.svd(Ec); Ec = u @ np.diag([1, 1, 0]) @ vt
        E = T2.T @ Ec @ T1; E /= np.linalg.norm(E); Ex1 = x1h @ E.T; Etx2 = x2h @ E; e = (Ex1 * x2h).sum(1)
        r = G.F_PX * e / np.maximum(np.sqrt(Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2), 1e-12)
        wn = w0 * G.tukey_weight(r, G.mad_scale(r, SCALE_FLOOR_PX))
        if np.abs(wn - w).max() < 1e-6: w = wn; break
        w = wn
    ok = w > 0
    _, R, t, _ = cv2.recoverPose(E, n1[ok], n2[ok], np.eye(3))
    return G.params_from_pose(R, t.ravel())

# ---------------------------------------------------------------- robust LM
def robust_lm(resfun, p0, w0, fixed=(), maxit=100, tol_p=1e-9, tol_c=1e-10):
    p = np.array(p0, float); free = [i for i in range(len(p)) if i not in fixed]; lam = 1e-3; hist = []
    def jac(p, r):
        J = np.empty((len(r), len(free)))
        for k, i in enumerate(free): d = np.zeros_like(p); d[i] = 1e-7; J[:, k] = (resfun(p + d) - r) / 1e-7
        return J
    for it in range(maxit):
        r = resfun(p); s = G.mad_scale(r[w0 > 0], SCALE_FLOOR_PX); cost = (w0 * G.tukey_rho(r, s)).sum(); J = jac(p, r)
        w = w0 * G.tukey_weight(r, s); H = (J * w[:, None]).T @ J; g = (J * w[:, None]).T @ r
        while True:
            dp = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g); pn = p.copy(); pn[free] += dp; cn = (w0 * G.tukey_rho(resfun(pn), s)).sum()
            if cn <= cost: lam = max(lam / 3, 1e-9); break
            lam *= 5
            if lam > 1e8: dp = dp * 0; pn = p; cn = cost; break
        p = pn; hist.append((it, cost, cn, float(np.abs(dp).max()), s))
        if np.abs(dp).max() < tol_p or abs(cost - cn) < tol_c * max(cost, 1e-12): break
    r = resfun(p); s = G.mad_scale(r[w0 > 0], SCALE_FLOOR_PX); w = w0 * G.tukey_weight(r, s); J = jac(p, r)
    return p, dict(res=resfun, w0=w0, iters=len(hist), hist=hist, scale=s, H=(J * w[:, None]).T @ J, free=free, n_inlier=int((w > 0).sum()), n_res=len(r),
                   converged=bool(hist and (hist[-1][3] < tol_p or abs(hist[-1][1] - hist[-1][2]) < tol_c * max(hist[-1][1], 1e-12))))

# ---------------------------------------------------------------- Method A (epipolar only)
def fit_A(D, p0=None, fixed=()):
    n1, n2, w0, B = D["n1"], D["n2"], D["w"], D["baseline"]
    start = "irls8pt" if p0 is None else "given"
    if p0 is None: p0 = init_pose(n1, n2, w0, B)
    p, info = robust_lm(lambda q: G.sampson_residual(q, n1, n2, B), p0, w0, fixed=fixed); info["p_init"] = p0; info["start"] = start; return p, info

def diagnose(D, p5, info):
    """Degeneracy diagnostics of a converged epipolar solution."""
    B = D["baseline"]; R, T = G.pose_from_params(p5, B); r = G.sampson_residual(p5, D["n1"], D["n2"], B); inl = np.abs(r) < 4.685 * info["scale"]
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))]); P2 = np.hstack([R, T.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, D["n1"][inl].T, D["n2"][inl].T); X = (X[:3] / X[3]).T; z2 = (X @ R.T + T)[:, 2]
    pos = float(np.mean((X[:, 2] > 0) & (z2 > 0)))
    flags = dict(scale_px=float(info["scale"]), inlier_fraction=float(inl.mean()), positive_depth_fraction=pos, converged=bool(info["converged"]))
    flags["degenerate"] = bool((not flags["converged"]) or flags["scale_px"] > DEGENERATE_SCALE_PX or flags["inlier_fraction"] < MIN_INLIER_FRACTION or pos < MIN_POSITIVE_DEPTH_FRACTION)
    return flags

def fit_A_guarded(D):
    """Default IRLS-8-point start; if the result is degenerate, restart from the generic parallel-rig fallback.  Returns p, info, record."""
    p, info = fit_A(D); fl = diagnose(D, p, info); rec = dict(first_attempt=fl, fallback_used=False)
    if fl["degenerate"]:
        p2, info2 = fit_A(D, p0=FALLBACK_START); info2["start"] = "fallback_parallel_rig"; fl2 = diagnose(D, p2, info2); rec.update(fallback_used=True, second_attempt=fl2, resolved=not fl2["degenerate"])
        p, info, fl = p2, info2, fl2
    rec["final"] = fl; rec["degenerate"] = fl["degenerate"]; return p, info, rec

# ---------------------------------------------------------------- Method J (rigid four-view tracks)
def _cams(p5, q6, baseline):
    Rs, ts = G.pose_from_params(p5, baseline); R0 = cv2.Rodrigues(q6[:3])[0]; t0 = q6[3:]
    return [(np.eye(3), np.zeros(3)), (Rs, ts), (R0, t0), (Rs @ R0, Rs @ t0 + ts)]

def _triangulate_gn(cm, obs, X0, it=4):
    X = X0.copy()
    for _ in range(it):
        Hh = np.zeros((len(X), 3, 3)); gg = np.zeros((len(X), 3))
        for c, (Rc, tc) in enumerate(cm):
            Xc = X @ Rc.T + tc; z = np.maximum(Xc[:, 2], 1.0); u = Xc[:, :2] / z[:, None]; r = G.F_PX * (u - obs[:, c])
            Ju = G.F_PX * (Rc[None, :2, :] - u[:, :, None] * Rc[None, 2:3, :]) / z[:, None, None]
            Hh += np.einsum("nki,nkj->nij", Ju, Ju); gg += np.einsum("nki,nk->ni", Ju, r)
        X = X - np.linalg.solve(Hh + 1e-9 * np.eye(3)[None], gg[:, :, None])[:, :, 0]
    return X

def _track_residuals(p5, Q, T, X0, baseline):
    out = np.empty((len(T["obs"]), 4, 2))
    for si in np.unique(T["sid"]):
        m = T["sid"] == si; cm = _cams(p5, Q[si], baseline); X = _triangulate_gn(cm, T["obs"][m], X0[m])
        for c, (Rc, tc) in enumerate(cm):
            Xc = X @ Rc.T + tc; out[m, c] = G.F_PX * (Xc[:, :2] / np.maximum(Xc[:, 2:3], 1.0) - T["obs"][m][:, c])
    return out.reshape(-1)

def _init_tracks(D, p5):
    T = D["T"]; B = D["baseline"]; Rs, ts = G.pose_from_params(p5, B); P1 = np.hstack([np.eye(3), np.zeros((3, 1))]); P2 = np.hstack([Rs, ts.reshape(3, 1)])
    Xa = cv2.triangulatePoints(P1, P2, T["obs"][:, 0].T, T["obs"][:, 1].T); Xa = (Xa[:3] / Xa[3]).T
    Xb = cv2.triangulatePoints(P1, P2, T["obs"][:, 2].T, T["obs"][:, 3].T); Xb = (Xb[:3] / Xb[3]).T; Q = np.zeros((D["nsets"], 6))
    for si in range(D["nsets"]):
        m = (T["sid"] == si) & (Xa[:, 2] > 800) & (Xb[:, 2] > 800) & (Xa[:, 2] < 20000) & (Xb[:, 2] < 20000)
        R, t = G.umeyama_irls(Xa[m], Xb[m]); Q[si] = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
    return Q, Xa.copy()

def fit_J(D, p0, kappa=1.0, fixed=()):
    """Joint fit started from the (guarded) epipolar solution p0.  Returns the 5 stereo params and an info dict (incl. rig motion Q)."""
    n1, n2, w0, B, T, ns = D["n1"], D["n2"], D["w"], D["baseline"], D["T"], D["nsets"]; Q0, X0 = _init_tracks(D, p0); wt = np.zeros(len(T["obs"]))
    for si in range(ns):
        m = T["sid"] == si
        if m.sum(): wt[m] = kappa * (w0[D["sid"] == si].sum() / (m.sum() * 8.0))      # track block carries the same total weight as the epipolar block, per set
    wall = np.r_[w0, np.repeat(wt, 8)]
    def res(p): return np.r_[G.sampson_residual(p[:5], n1, n2, B), _track_residuals(p[:5], p[5:].reshape(ns, 6), T, X0, B)]
    p, info = robust_lm(res, np.r_[p0, Q0.ravel()], wall, fixed=fixed, maxit=60, tol_p=1e-8, tol_c=1e-9)
    info.update(Q=p[5:].reshape(ns, 6), p_init=p0, n_epipolar=len(n1), n_tracks=len(T["obs"]), p_full=p); return p[:5], info

def estimate_pose(D):
    """Method J end to end: guarded epipolar start -> joint refinement.  Returns (p5_J, p5_A, report dict)."""
    pA, infoA, recA = fit_A_guarded(D); pJ, infoJ = fit_J(D, pA)
    flagsJ = dict(converged=infoJ["converged"], scale_px=float(infoJ["scale"]), inlier_fraction=infoJ["n_inlier"] / infoJ["n_res"])
    flagsJ["degenerate"] = bool(recA["degenerate"] or (not flagsJ["converged"]) or flagsJ["scale_px"] > DEGENERATE_SCALE_PX)
    rep = dict(method_A=dict(guard=recA, iterations=infoA["iters"], robust_scale_px=infoA["scale"], n_inlier=infoA["n_inlier"], n_residuals=infoA["n_res"], start=infoA["start"]),
               method_J=dict(flags=flagsJ, iterations=infoJ["iters"], robust_scale_px=infoJ["scale"], n_inlier=infoJ["n_inlier"], n_residuals=infoJ["n_res"],
                             n_epipolar=infoJ["n_epipolar"], n_tracks=infoJ["n_tracks"], rig_motion_rotvec_deg=np.degrees(infoJ["Q"][:, :3]).tolist(), rig_motion_t_mm=infoJ["Q"][:, 3:].tolist()))
    return pJ, pA, rep, dict(infoA=infoA, infoJ=infoJ)
