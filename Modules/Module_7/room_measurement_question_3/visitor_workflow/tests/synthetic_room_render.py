"""F8.31C-1 TEST HELPER (synthetic only, never imported by production code): renders photometric stereo frames of a made-up room.

A textured floor, a textured wall with two end corners and a textured ceiling are ray-cast through four distorted pinhole cameras (two rig placements,
a left and a right camera each, small per-frame-set rig motion).  The frames are real images, so the ORIGINAL ``features.match_set`` (SIFT, ratio test,
mutual check) runs on them for real; nothing about the matches is mocked.  The boundary lines are exact projections of the true 3D wall edges (the
same construction as ``visitor_method_j_adapter_synthetic``), because boundary refinement is covered by the F8.31B / F8.31B-V work, not here.

Every number is invented here.  Nothing comes from the original experiment.  The known wall size is a sanity reference for assertions only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

MODULE = Path(__file__).resolve().parents[2]
if str(MODULE) not in sys.path:
    sys.path.insert(0, str(MODULE))

from room_measurement import lines as L                                    # noqa: E402
from room_measurement.visitor_method_j_adapter import VisitorCalibrationContext   # noqa: E402

SIZE = (1080, 1920)                       # portrait, half of the original 2160 x 3840 with the focal length halved too (same geometry, a quarter of the pixels)
F = 1500.0
BASELINE_MM = 400.0                       # an arbitrary synthetic rig
CAMERA_HEIGHT_MM = 1600.0
WALL_WIDTH_MM, WALL_HEIGHT_MM = 3000.0, 2400.0
WALL_ANGLE = np.radians(8.0)
C1 = np.array([-1500.0, CAMERA_HEIGHT_MM, 5600.0])
D_WALL = np.array([np.cos(WALL_ANGLE), 0.0, np.sin(WALL_ANGLE)])
N_WALL = np.array([-np.sin(WALL_ANGLE), 0.0, np.cos(WALL_ANGLE)])
UP = np.array([0.0, -1.0, 0.0])
C2 = C1 + WALL_WIDTH_MM * D_WALL
R_RIGHT = cv2.Rodrigues(np.radians([0.2, -1.5, 0.1]))[0]
RIGHT_T = lambda baseline: np.array([-baseline, 0.4, 0.8])                 # noqa: E731  (right frame = R_RIGHT * left frame + T)
R2 = cv2.Rodrigues(np.radians([0.0, -9.0, 0.0]))[0]
O2 = np.array([1300.0, 0.0, 250.0])


def make_context(label="synthetic-visitor", baseline_mm=BASELINE_MM, f=F, size=SIZE) -> VisitorCalibrationContext:
    W, H = size
    K1 = np.array([[f, 0, W / 2 + 2.5], [0, f * 1.001, H / 2 - 2.0], [0, 0, 1.0]])
    K2 = np.array([[f * 1.012, 0, W / 2 - 3.0], [0, f * 1.013, H / 2 + 2.5], [0, 0, 1.0]])
    return VisitorCalibrationContext(K1, np.array([0.02, -0.06, 0, 0, 0.02]), K2, np.array([0.015, -0.05, 0, 0, 0.015]), tuple(size), float(baseline_mm), label)


def _texture(rng, n, mean, std=34.0):
    t = np.zeros((n, n), np.float32)
    for sigma, w in ((1.5, 0.7), (3, 1.0), (6, 1.0), (12, 0.8), (24, 0.5)):
        z = cv2.GaussianBlur(rng.standard_normal((n, n)).astype(np.float32), (0, 0), sigma); t += w * z / (z.std() + 1e-9)
    return (mean + std * t / (t.std() + 1e-9)).astype(np.float32)


class Room:
    """The textures are made once (seeded) and shared by every frame."""
    FLOOR_X, FLOOR_Z, RES = (-8000.0, 8000.0), (-1000.0, 14000.0), 5.0

    def __init__(self, seed=11):
        rng = np.random.default_rng(seed)
        self.floor = _texture(rng, 3200, 140.0); self.ceiling = _texture(rng, 3200, 232.0, 18.0)
        self.wall_in = _texture(rng, 2400, 188.0); self.wall_out = _texture(rng, 2400, 92.0)
        self.wall_origin = (-3000.0, -400.0)                      # (s, t) of texel (0, 0), mm; the wall texture covers s in [-3000, 4200], t in [-400, 2800]
        self.wall_res = 3.0

    def render(self, R, t, ctx_side_K, ctx_side_D, size, rng, noise=1.0) -> np.ndarray:
        W, H = size
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        xy = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2), ctx_side_K, ctx_side_D).reshape(-1, 2)
        dc = np.c_[xy, np.ones(len(xy))]; dw = dc @ R                       # R^T applied to each row vector
        o = -R.T @ t
        with np.errstate(divide="ignore", invalid="ignore"):
            lf = (CAMERA_HEIGHT_MM - o[1]) / dw[:, 1]; lc = (-(WALL_HEIGHT_MM - CAMERA_HEIGHT_MM) - o[1]) / dw[:, 1]; lw = (N_WALL @ (C1 - o)) / (dw @ N_WALL)
        Pf, Pc, Pw = o + lf[:, None] * dw, o + lc[:, None] * dw, o + lw[:, None] * dw
        front = lambda P: (P - C1) @ N_WALL <= 0                          # noqa: E731
        floor_ok = (lf > 0) & front(Pf) & (dw[:, 1] > 0)
        ceil_ok = ~floor_ok & (lc > 0) & front(Pc) & (dw[:, 1] < 0)
        wall_ok = ~floor_ok & ~ceil_ok & (lw > 0) & (Pw[:, 1] <= CAMERA_HEIGHT_MM) & (Pw[:, 1] >= -(WALL_HEIGHT_MM - CAMERA_HEIGHT_MM))
        img = np.full(len(dc), 128.0, np.float32)

        def sample(tex, u, v, mask):
            m = mask & (u >= 0) & (v >= 0) & (u < tex.shape[1] - 1) & (v < tex.shape[0] - 1)
            out = cv2.remap(tex, u.astype(np.float32).reshape(H, W), v.astype(np.float32).reshape(H, W), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
            return out, m
        fu = (Pf[:, 0] - self.FLOOR_X[0]) / self.RES; fv = (Pf[:, 2] - self.FLOOR_Z[0]) / self.RES
        v_, m_ = sample(self.floor, fu, fv, floor_ok); img[m_] = v_[m_]
        cu = (Pc[:, 0] - self.FLOOR_X[0]) / self.RES; cv_ = (Pc[:, 2] - self.FLOOR_Z[0]) / self.RES
        v_, m_ = sample(self.ceiling, cu, cv_, ceil_ok); img[m_] = v_[m_]
        s = (Pw - C1) @ D_WALL; tt = CAMERA_HEIGHT_MM - Pw[:, 1]
        wu = (s - self.wall_origin[0]) / self.wall_res; wv = (tt - self.wall_origin[1]) / self.wall_res
        inside = (s >= 0) & (s <= WALL_WIDTH_MM)
        v_in, m_in = sample(self.wall_in, wu, wv, wall_ok & inside); v_out, m_out = sample(self.wall_out, wu, wv, wall_ok & ~inside)
        img[m_in] = v_in[m_in]; img[m_out] = v_out[m_out]
        img = img.reshape(H, W) + rng.normal(0, noise, (H, W)).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)


def camera_extrinsics(ctx, set_rng):
    """(R, t) of AL, PL, AR, PR in the world frame (= left camera, placement 1), with a small per-set rig motion at placement 2."""
    Rs = cv2.Rodrigues(set_rng.normal(0, 0.003, 3))[0]; ts = set_rng.normal(0, 6, 3)
    Rr, Tr = R_RIGHT, RIGHT_T(ctx.baseline_mm)
    R_ar = Rs @ R2; t_ar = -R_ar @ O2 + ts
    return dict(AL=(np.eye(3), np.zeros(3)), PL=(Rr, Tr), AR=(R_ar, t_ar), PR=(Rr @ R_ar, Rr @ t_ar + Tr))


def render_set(room: Room, ctx: VisitorCalibrationContext, set_index: int, seed=5) -> dict:
    rng = np.random.default_rng(seed * 1000 + set_index); ext = camera_extrinsics(ctx, np.random.default_rng(seed * 10 + set_index)); out = {}
    for k, (R, t) in ext.items():
        K, D = (ctx.K_left, ctx.D_left) if k[0] == "A" else (ctx.K_right, ctx.D_right)
        out[k] = room.render(R, t, K, D, ctx.image_size_wh, rng)
    return out


def _proj(P3, R, t, K):
    return cv2.projectPoints(np.asarray(P3, float).reshape(-1, 1, 3), cv2.Rodrigues(R)[0], np.asarray(t, float), K, np.zeros(5))[0].reshape(-1, 2)


def true_lines(ctx: VisitorCalibrationContext, set_index: int, seed=5) -> dict:
    """The six Method J boundary lines (undistorted pixels, ``lines.fit_line`` dicts) of this set, projected from the true 3D wall edges."""
    ext = camera_extrinsics(ctx, np.random.default_rng(seed * 10 + set_index)); up = UP; out = {}
    for vk_air, vk_pro, corner, tag in (("AL", "PL", C1, "L"), ("AR", "PR", C2, "R")):
        edge = np.array([corner + s * up for s in np.linspace(0, WALL_HEIGHT_MM, 40)]); topc = corner + WALL_HEIGHT_MM * up
        ceil = np.array([topc + s * D_WALL for s in np.linspace(-1500, 1500, 40)])
        for vk in (vk_air, vk_pro):
            R, t = ext[vk]; K = ctx.K_left if vk[0] == "A" else ctx.K_right
            out[f"{vk}_corner"] = L.fit_line(_proj(edge, R, t, K), "v")
        R, t = ext[vk_pro]; out[f"P{tag}_ceilF"] = L.fit_line(_proj(ceil, R, t, ctx.K_right), "h")
    return out
