"""F8.30B: SYNTHETIC fixtures for ``visitor_uncertainty`` (tests and the benchmark only; never imported by production code).

Everything is rendered from known cameras with a seeded RNG.  No video, no .MOV, no real calibration, no Method J and no experiment constant is used.
The wall estimator below is a deliberately simple stand-in (triangulate four wall corners with the draw's K and D, fixed rig pose, entered baseline).
It is NOT Method J and says nothing about Method J's sensitivity to calibration.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import visitor_uncertainty as VU

CO, uc = VU.CO, VU.uc


class Rig:
    def __init__(self, size=(1280, 720), f=1000.0, baseline_mm=120.0, k=(0.05, -0.12, 0.08)):
        self.size, self.baseline_mm, self.f = size, float(baseline_mm), float(f)
        self.K1 = np.array([[f, 0, size[0] / 2 + 3.0], [0, f * 1.002, size[1] / 2 - 2.0], [0, 0, 1.0]])
        self.K2 = np.array([[f * 1.01, 0, size[0] / 2 - 4.0], [0, f * 1.012, size[1] / 2 + 3.0], [0, 0, 1.0]])
        self.D1 = np.array([k[0], k[1], 0, 0, k[2]])
        self.D2 = np.array([k[0] * 0.8, k[1] * 0.9, 0, 0, k[2] * 0.7])
        self.R = cv2.Rodrigues(np.radians([0.3, -1.0, 0.2]))[0]
        self.T = np.array([-baseline_mm, 0.5, 1.0])               # right frame = R * left + T


def _project(X, R, t, K, D):
    return cv2.projectPoints(X, cv2.Rodrigues(R)[0], t, K, D)[0].reshape(-1, 2)


def make_pairs(rig: Rig, n, seed, pattern=(8, 6), square_mm=25.0, tilt_deg=30.0, noise_px=0.2):
    """n board poses seen by both cameras, all corners inside both frames."""
    rng = np.random.default_rng(seed)
    sc = rig.f / 1000.0                                            # board distance and offset scale with the focal length (the board keeps its real size)
    obj = uc.board_object_points(pattern, square_mm).astype(np.float64)
    obj = obj - obj.mean(0)
    W, H = rig.size
    L, R, tries = [], [], 0
    while len(L) < n and tries < 50000:
        tries += 1
        Rb = cv2.Rodrigues(np.radians(rng.uniform(-tilt_deg, tilt_deg, 3) * np.array([1, 1, 0.5])))[0]
        t = np.array([rng.uniform(-60, 80), rng.uniform(-50, 50), rng.uniform(450, 800)]) * sc
        pl = _project(obj, Rb, t, rig.K1, rig.D1)
        pr = _project(obj, rig.R @ Rb, rig.R @ t + rig.T, rig.K2, rig.D2)
        if not all((p[:, 0] > 15).all() and (p[:, 0] < W - 15).all() and (p[:, 1] > 15).all() and (p[:, 1] < H - 15).all() for p in (pl, pr)):
            continue
        if Rb[2, 2] < 0.5:
            continue
        L.append(pl + rng.normal(0, noise_px, pl.shape)); R.append(pr + rng.normal(0, noise_px, pr.shape))
    if len(L) < n:
        raise RuntimeError(f"could only place {len(L)} of {n} board poses")
    return L, R


def make_calibration(n_pairs=20, seed=7, model="radial3", rig: Rig | None = None, tilt_deg=30.0, noise_px=0.2, pattern=(8, 6), square_mm=25.0):
    """-> (rig, observation record, saved model dict, document).  The 'saved' calibration is a real solve of these observations with the production solver."""
    rig = rig or Rig()
    L, R = make_pairs(rig, n_pairs, seed, pattern, square_mm, tilt_deg, noise_px)
    m = uc.calibrate_stereo(L, R, pattern, square_mm, rig.size, rig.size, distortion=model)
    obs = CO.build_observations(pattern=pattern, square_mm=square_mm, image_size_left=rig.size, image_size_right=rig.size, pair_ids=list(range(1, n_pairs + 1)),
                                pair_frames=[(7 * i, 7 * i + 1) for i in range(n_pairs)], left_corners=L, right_corners=R,
                                params=dict(K1=m["K1"], D1=m["D1"], K2=m["K2"], D2=m["D2"], R=m["R"], T=m["T"]))
    doc = dict(schema=CO.RESULT_SCHEMA_V2,
               cameras=dict(left=dict(K=m["K1"].tolist(), distortion=np.asarray(m["D1"]).ravel().tolist(), image_size_wh=list(rig.size), distortion_assessment=m["distortion_left"]),
                            right=dict(K=m["K2"].tolist(), distortion=np.asarray(m["D2"]).ravel().tolist(), image_size_wh=list(rig.size), distortion_assessment=m["distortion_right"])),
               extrinsics=dict(R=m["R"].tolist(), T=np.asarray(m["T"]).ravel().tolist()),
               settings=dict(cols=pattern[0], rows=pattern[1], square_mm=square_mm),
               selected_poses=dict(frame_pairs_left_right=[[p["left_frame"], p["right_frame"]] for p in obs["pairs"]]),
               calibration_observations=obs)
    return rig, obs, m, doc


class SyntheticWallEstimator:
    """Triangulates the four corners of a rectangular wall seen by the rig, using the DRAW's K and D, the fixed rig pose and the entered baseline."""

    def __init__(self, rig: Rig, width_mm=1400.0, height_mm=900.0, depth_mm=1800.0, noise_px=0.3, seed=5, fail_when=None):
        self.rig = rig
        w2, h2 = width_mm / 2, height_mm / 2
        X = np.array([[-w2, h2, depth_mm], [w2, h2, depth_mm], [-w2, -h2, depth_mm], [w2, -h2, depth_mm]], float)       # TL TR BL BR
        rng = np.random.default_rng(seed)
        self.pl = _project(X, np.eye(3), np.zeros(3), rig.K1, rig.D1) + rng.normal(0, noise_px, (4, 2))
        self.pr = _project(X, rig.R, rig.T, rig.K2, rig.D2) + rng.normal(0, noise_px, (4, 2))
        self.fail_when = fail_when                                  # optional predicate(params) -> failure code, for failure-accounting tests
        self.calls = 0

    def __call__(self, p):
        self.calls += 1
        if self.fail_when is not None:
            code = self.fail_when(p)
            if code:
                raise VU.EstimatorFailure(code)
        rig = self.rig
        scale = p.get("baseline_mm", rig.baseline_mm) / rig.baseline_mm
        nl = cv2.undistortPoints(self.pl.reshape(-1, 1, 2), p["K_left"], p["D_left"]).reshape(-1, 2)
        nr = cv2.undistortPoints(self.pr.reshape(-1, 1, 2), p["K_right"], p["D_right"]).reshape(-1, 2)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))]); P2 = np.hstack([rig.R, (rig.T * scale).reshape(3, 1)])
        Xh = cv2.triangulatePoints(P1, P2, nl.T, nr.T)
        if not np.isfinite(Xh).all() or np.abs(Xh[3]).min() < 1e-12:
            raise VU.EstimatorFailure("triangulation_degenerate")
        X = (Xh[:3] / Xh[3]).T
        if (X[:, 2] <= 0).any():
            raise VU.EstimatorFailure("point_behind_camera")
        return dict(width_mm=0.5 * (np.linalg.norm(X[1] - X[0]) + np.linalg.norm(X[3] - X[2])), height_mm=0.5 * (np.linalg.norm(X[0] - X[2]) + np.linalg.norm(X[1] - X[3])))


def linear_estimator(weights_w, weights_h, center, w0=2400.0, h0=1500.0):
    """Exactly linear map of the 18-vector: used to verify that sigma points reproduce Monte Carlo spread when the estimator is linear."""
    ww, wh, c = np.asarray(weights_w, float), np.asarray(weights_h, float), np.asarray(center, float)

    def est(p):
        v = VU.params_to_vector(p) - c
        return dict(width_mm=w0 + float(ww @ v), height_mm=h0 + float(wh @ v))
    return est
