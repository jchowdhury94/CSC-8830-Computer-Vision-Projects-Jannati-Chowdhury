"""Floor reconstruction, two-view registration and wall width/height (validated geometry, unchanged)."""
import numpy as np
from . import geometry as G

# (view) -> (Air corner line, Pro corner line, Pro ceiling front, Pro ceiling side)
VIEW_LINES = {"V1": ("AL_corner", "PL_corner", "PL_ceilF", "PL_ceilS"), "V2": ("AR_corner", "PR_corner", "PR_ceilF", "PR_ceilS")}
REG_ITERS, REG_THR_MM = 1500, 40.0
DEPTH_RANGE_MM = (800.0, 20000.0)
GATE_PX = 1.5

def view_points(cal, R, T, a, b, mt):
    """Triangulate one stereo view and keep well-conditioned points (depth range, reprojection and epipolar gates)."""
    X = G.triangulate(cal, R, T, a, b); e1, e2 = G.reprojection_errors(cal, R, T, X, a, b); d = G.epipolar_distances(cal, R, T, a, b)
    g = (X[:, 2] > DEPTH_RANGE_MM[0]) & (X[:, 2] < DEPTH_RANGE_MM[1]) & (e1 < GATE_PX) & (e2 < GATE_PX) & (d < GATE_PX)
    return X[g], a[g], mt[g]

def register_views(X1, i1, X2, i2, MX, rng):
    """Rigid V1->V2 registration from cross-view matches that have stereo 3D in both views (seeded RANSAC at 40 mm + refit)."""
    d1 = {k: i for i, k in enumerate(i1)}; d2 = {k: i for i, k in enumerate(i2)}
    pr = [(d1[q], d2[w]) for q, w in MX if q in d1 and w in d2]
    if len(pr) < 6: raise ValueError(f"only {len(pr)} registration pairs")
    p, q = map(np.array, zip(*pr)); return G.rigid_ransac(X1[p], X2[q], rng, REG_ITERS, REG_THR_MM)

def reconstruct_geometry(cal, R, T, F, rng):
    """Per frame set: triangulate both views, robust IRLS floor planes, V1->V2 registration."""
    X1, a1, i1 = view_points(cal, R, T, F["A1"], F["B1"], F["M1"]); X2, a2, i2 = view_points(cal, R, T, F["A2"], F["B2"], F["M2"])
    f1, f2 = G.fit_floor_plane(X1, a1), G.fit_floor_plane(X2, a2); R0, t0, rms, ni, nt = register_views(X1, i1[:, 0], X2, i2[:, 0], F["MX"], rng)
    return dict(f1=f1, f2=f2, R0=R0, t0=t0, registration=dict(rms_mm=rms, n_inliers=ni, n_pairs=nt), n_points=(len(X1), len(X2)), X=(X1, a1, X2, a2))

def measure_wall(cal, R, T, geom, lines, top_variant=0, perturb=None):
    """Bottom corners = stereo-derived vertical edge (Air+Pro planes) meets the floor; top corners = Pro ray through the ceiling
    intersection meets that vertical; register view 1 into view 2; width = bottom-corner separation, height = mean top height.
    ``perturb`` (uncertainty analysis only): {'V1':(dx_air,dx_pro),'V2':(..), 'dy':{'V1':..,'V2':..}} pixel offsets of the observations."""
    out = {}
    for v, (cl, cp, ce, cs) in VIEW_LINES.items():
        fl = geom["f1"] if v == "V1" else geom["f2"]; lL = G.line_to_h(lines[cl]).copy(); lP = G.line_to_h(lines[cp]).copy()
        if perturb is not None: lL[2] -= lL[0] * perturb[v][0]; lP[2] -= lP[0] * perturb[v][1]
        B, u = G.corner_line_on_floor(cal, R, T, lL, lP, fl); n = fl[0]; vang = float(np.degrees(np.arccos(min(1, abs(u @ n)))))
        if top_variant == 0: pt = np.cross(lP, G.line_to_h(lines[ce])); pt = pt[:2] / pt[2]
        else: pt = G.line_intersection(lines[ce], lines[cs])
        if perturb is not None and "dy" in perturb: pt = pt + np.array([0, perturb["dy"][v]])
        H, skew = G.top_point_on_vertical(cal, R, T, pt, B, n); out[v] = dict(B=B, H=float(H), skew=skew, vertical_error_deg=vang, top_px=pt, floor_height_mm=float(fl[1]))
    Bw = geom["R0"] @ out["V1"]["B"] + geom["t0"]; out["width_mm"] = float(np.linalg.norm(out["V2"]["B"] - Bw)); out["height_mm"] = 0.5 * (out["V1"]["H"] + out["V2"]["H"])
    out["bottom_corner_V1_registered_into_V2"] = Bw; return out

def measure_all_sets(cal, p5, match_sets, line_sets, top_variant=0, seed=3):
    """Run the geometry + measurement for every frame set under one shared pose.  Failures are recorded, not hidden."""
    R, T = G.pose_from_params(p5, cal.baseline_mm); rows = []
    for i, (F, L) in enumerate(zip(match_sets, line_sets)):
        try:
            g = reconstruct_geometry(cal, R, T, F, np.random.default_rng(seed + i)); o = measure_wall(cal, R, T, g, L, top_variant)
            rows.append(dict(ok=True, width_mm=o["width_mm"], height_mm=o["height_mm"], H1=o["V1"]["H"], H2=o["V2"]["H"], vertical_error_deg=(o["V1"]["vertical_error_deg"], o["V2"]["vertical_error_deg"]),
                             floor_height_mm=(o["V1"]["floor_height_mm"], o["V2"]["floor_height_mm"]), floor_normals=(g["f1"][0], g["f2"][0]), registration=g["registration"],
                             R0=g["R0"], t0=g["t0"], bottom_corners=dict(V1=o["V1"]["B"], V2=o["V2"]["B"]), skew_mm=(o["V1"]["skew"], o["V2"]["skew"]), n_points=g["n_points"]))
        except Exception as e: rows.append(dict(ok=False, error=repr(e), width_mm=float("nan"), height_mm=float("nan")))
    return rows
