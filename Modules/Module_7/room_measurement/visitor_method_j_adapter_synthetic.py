"""F8.30D-A: SYNTHETIC reconstruction inputs for ``visitor_method_j_adapter`` (tests only; never imported by production code).

A rendered scene: a planar floor, a rectangular wall with two end corners, two rig placements, four cameras with radial distortion, seeded noise.
It produces exactly the structures Method J consumes (match sets and fitted boundary lines) WITHOUT any video, SIFT, annotation or file.
Every number is invented here; none comes from the original experiment, and the known wall size is only used by tests as a sanity reference,
never as an input to the reconstruction.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import lines as L
from .visitor_method_j_adapter import ReconstructionInputs, VisitorCalibrationContext

SIZE = (2160, 3840)                      # portrait, same orientation as the visitor path expects (floor rows lie in the lower part)
CAMERA_HEIGHT_MM = 1600.0
WALL_WIDTH_MM, WALL_HEIGHT_MM = 3000.0, 2400.0
BASELINE_MM = 400.0                      # synthetic rig; arbitrary


def make_context(label="synthetic", baseline_mm=BASELINE_MM, f=3000.0) -> VisitorCalibrationContext:
    W, H = SIZE
    K1 = np.array([[f, 0, W / 2 + 5.0], [0, f * 1.001, H / 2 - 4.0], [0, 0, 1.0]])
    K2 = np.array([[f * 1.012, 0, W / 2 - 6.0], [0, f * 1.013, H / 2 + 5.0], [0, 0, 1.0]])
    return VisitorCalibrationContext(K1, np.array([0.02, -0.06, 0, 0, 0.02]), K2, np.array([0.015, -0.05, 0, 0, 0.015]), SIZE, float(baseline_mm), label)


def _proj(X, R, t, K, D=None):
    return cv2.projectPoints(X.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], np.asarray(t, float), K, np.zeros(5) if D is None else D)[0].reshape(-1, 2)


def _line(P3, R, t, K, kind):
    """Fitted-line dict (undistorted pixels) of a 3D segment, built with the production ``lines.fit_line``."""
    return L.fit_line(_proj(P3, R, t, K), kind)


def make_scene(ctx: VisitorCalibrationContext | None = None, n_sets=3, n_pts=2600, noise_px=0.1, outlier=0.02, seed=3):
    """-> (ReconstructionInputs, truth dict).  ``ctx`` renders the images (the 'true' cameras); the inputs record the lines under the same intrinsics."""
    ctx = ctx or make_context(); rng = np.random.default_rng(seed); W, H = SIZE
    Rr = cv2.Rodrigues(np.radians([0.2, -1.5, 0.1]))[0]; Tr = np.array([-ctx.baseline_mm, 0.4, 0.8])             # right frame = Rr * left + Tr
    R2 = cv2.Rodrigues(np.radians([0.0, -9.0, 0.0]))[0]; o2 = np.array([1300.0, 0.0, 250.0])                    # placement 2: world -> left frame R2 (X - o2)
    a = np.radians(8.0); d = np.array([np.cos(a), 0.0, np.sin(a)]); c1 = np.array([-1500.0, CAMERA_HEIGHT_MM, 5600.0]); c2 = c1 + WALL_WIDTH_MM * d
    up = np.array([0.0, -1.0, 0.0])
    tops = {k: c + WALL_HEIGHT_MM * up for k, c in (("c1", c1), ("c2", c2))}
    # world points: floor (planar), wall plane, and scattered room points
    nf, nw, ns = n_pts // 2, n_pts // 4, n_pts - n_pts // 2 - n_pts // 4
    floor = np.c_[rng.uniform(-3500, 3500, nf), np.full(nf, CAMERA_HEIGHT_MM), rng.uniform(2800, 6800, nf)]
    wp = c1 + rng.uniform(-300, WALL_WIDTH_MM + 300, nw)[:, None] * d + rng.uniform(0, WALL_HEIGHT_MM, nw)[:, None] * up
    sc = np.c_[rng.uniform(-3000, 3500, ns), rng.uniform(-900, CAMERA_HEIGHT_MM, ns), rng.uniform(3000, 6800, ns)]
    Xw = np.vstack([floor, wp, sc]); sets = []
    for s in range(n_sets):
        Rs = cv2.Rodrigues(rng.normal(0, 0.003, 3))[0]; ts = rng.normal(0, 6, 3)                              # small per-set rig motion between the two placements
        X1 = Xw; X2 = (Xw - o2) @ R2.T @ Rs.T + ts
        cams = dict(AL=(X1, np.eye(3), np.zeros(3), ctx.K_left, ctx.D_left), PL=(X1, Rr, Tr, ctx.K_right, ctx.D_right),
                    AR=(X2, np.eye(3), np.zeros(3), ctx.K_left, ctx.D_left), PR=(X2, Rr, Tr, ctx.K_right, ctx.D_right))
        px, ok = {}, np.ones(len(Xw), bool)
        for k, (X, R, t, K, D) in cams.items():
            Xc = X @ R.T + t; p = _proj(X, R, t, K, D); px[k] = p
            ok &= (Xc[:, 2] > 500) & (p[:, 0] > 5) & (p[:, 0] < W - 5) & (p[:, 1] > 5) & (p[:, 1] < H - 5)
        idx = np.where(ok)[0]; q = {k: v[idx] + rng.normal(0, noise_px, (len(idx), 2)) for k, v in px.items()}
        for k in ("PL", "PR"):
            m = rng.random(len(idx)) < outlier; q[k][m] = np.c_[rng.uniform(0, W, m.sum()), rng.uniform(0, H, m.sum())]
        ar = np.arange(len(idx)); pair = np.c_[ar, ar]
        sets.append(dict(A1=q["AL"].astype(np.float32), B1=q["PL"].astype(np.float32), M1=pair, A2=q["AR"].astype(np.float32), B2=q["PR"].astype(np.float32), M2=pair, MX=pair))
    # boundary lines (undistorted pixels; the same noise-free line is used for every set, as a fixed rig + static scene would give)
    ls = {}
    for v, (cb, cams) in (("1", (c1, dict(A=(np.eye(3), np.zeros(3), ctx.K_left), P=(Rr, Tr, ctx.K_right)))),
                          ("2", (c2, dict(A=(np.eye(3), np.zeros(3), ctx.K_left), P=(Rr, Tr, ctx.K_right))))):
        place = (lambda X: X) if v == "1" else (lambda X: (X - o2) @ R2.T)
        edge = np.array([cb + s * up for s in np.linspace(0, WALL_HEIGHT_MM, 40)]); topc = cb + WALL_HEIGHT_MM * up
        ceil = np.array([topc + s * d for s in np.linspace(-1500, 1500, 40)])
        for cam, (R, t, K) in cams.items():
            side = "L" if v == "1" else "R"
            ls[f"{'A' if cam == 'A' else 'P'}{side}_corner"] = _line(place(edge), R, t, K, "v")
            if cam == "P": ls[f"P{side}_ceilF"] = _line(place(ceil), R, t, K, "h"); ls[f"P{side}_ceilS"] = _line(place(np.array([topc - s * np.array([d[2], 0, -d[0]]) for s in np.linspace(0, 1500, 40)])), R, t, K, "h")
    inputs = ReconstructionInputs(tuple(sets), tuple(dict(ls) for _ in range(n_sets)), ctx)
    truth = dict(width_mm=WALL_WIDTH_MM, height_mm=WALL_HEIGHT_MM, baseline_mm=ctx.baseline_mm, note="sanity reference for tests only; never given to the reconstruction")
    return inputs, truth
