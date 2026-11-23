"""TEST HELPER (synthetic only, never imported by production code): a made-up three-viewpoint Wall B scene with KNOWN geometry.

World frame: x right, y DOWN, z forward (the camera convention).  Floor is the plane y = FLOOR_Y.  Viewpoint v has its rig (left camera) at ``CENTERS[v]`` rotated about y by ``YAWS[v]``;
the right camera is the left one moved by the stereo pose (R_S, T_S) with the requested baseline.  The two wall-end vertical edges stand on the floor at (+-L/2, z=WALL_Z).
Every number is invented here; nothing comes from the original experiment.
"""
from __future__ import annotations

import cv2
import numpy as np

from room_measurement import three_view_wall_b as T

SIZE = (1080, 1920)
F = 1500.0
FLOOR_Y = 1500.0
WALL_Z = 7500.0
CENTERS = {1: np.array([-1500.0, 0.0, 0.0]), 2: np.array([0.0, 0.0, 200.0]), 3: np.array([1500.0, 0.0, 0.0])}
YAWS = {1: 4.0, 2: 0.0, 3: -4.0}
R_S = cv2.Rodrigues(np.radians([0.2, -1.5, 0.1]))[0]


def contract(baseline_mm=300.0, f=F, size=SIZE):
    W, H = size
    K1 = [[f, 0, W / 2 + 2.5], [0, f * 1.001, H / 2 - 2.0], [0, 0, 1.0]]
    K2 = [[f * 1.012, 0, W / 2 - 3.0], [0, f * 1.013, H / 2 + 2.5], [0, 0, 1.0]]
    return dict(K1=K1, D1=[0.02, -0.06, 0, 0, 0.02], K2=K2, D2=[0.015, -0.05, 0, 0, 0.015], image_size_wh=list(size), baseline_mm=float(baseline_mm))


def stereo_T(baseline_mm):
    d = np.array([-1.0, 0.0004, 0.0011]); return baseline_mm * d / np.linalg.norm(d)


def _rot(deg):
    return cv2.Rodrigues(np.array([0.0, np.radians(deg), 0.0]))[0]


def to_left(v, Xw):
    return (np.asarray(Xw) - CENTERS[v]) @ _rot(YAWS[v])          # = Rv^T (X - c)


def project(cal, Xl, cam, baseline_true):
    """Distorted pixels of left-camera-frame points Xl for the 'left' or 'right' camera, plus an in-image / in-front mask."""
    if cam == "left": X, K, D = Xl, cal.K_left, cal.D_left
    else: X, K, D = Xl @ R_S.T + stereo_T(baseline_true), cal.K_right, cal.D_right
    px = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    ok = (X[:, 2] > 300) & (px[:, 0] > 5) & (px[:, 0] < cal.image_size[0] - 5) & (px[:, 1] > 5) & (px[:, 1] < cal.image_size[1] - 5)
    return px, ok


def scene_points(seed=0):
    rng = np.random.default_rng(seed)
    floor = np.c_[rng.uniform(-3200, 3200, 1500), np.full(1500, FLOOR_Y), rng.uniform(2200, 7000, 1500)]
    stuff = np.c_[rng.uniform(-3200, 3200, 500), rng.uniform(-900, 500, 500), rng.uniform(3000, 7400, 500)]       # furniture-like points above the floor rows
    return np.vstack([floor, stuff])


def build(length_mm=4000.0, baseline_true=300.0, noise_px=0.0, seed=0, cal=None):
    """Returns (cal, {(1,2): match_set, (3,2): match_set}, annotations doc, true length).  ``cal`` defaults to the contract with the TRUE baseline."""
    cal = cal or T.calibration_from_contract(contract(baseline_true))
    rng = np.random.default_rng(seed + 1); P = scene_points(seed)
    def view(v):
        Xl = to_left(v, P); pl, ol = project(cal, Xl, "left", baseline_true); pr, orr = project(cal, Xl, "right", baseline_true)
        n = lambda a: a + rng.normal(0, noise_px, a.shape) if noise_px else a
        ids = np.where(ol & orr)[0]; return dict(ids=ids, L=n(pl[ids]), R=n(pr[ids]))
    V = {v: view(v) for v in (1, 2, 3)}
    def pair(a, b):
        va, vb = V[a], V[b]; ma = {int(i): k for k, i in enumerate(va["ids"])}; mb = {int(i): k for k, i in enumerate(vb["ids"])}
        M1 = np.c_[np.arange(len(va["ids"])), np.arange(len(va["ids"]))]; M2 = np.c_[np.arange(len(vb["ids"])), np.arange(len(vb["ids"]))]
        MX = np.array([(ma[i], mb[i]) for i in ma if i in mb])
        return dict(A1=va["L"], B1=va["R"], M1=M1, A2=vb["L"], B2=vb["R"], M2=M2, MX=MX)
    ms = {(1, 2): pair(1, 2), (3, 2): pair(3, 2)}
    anns = []
    for v, x in ((1, -length_mm / 2), (3, length_mm / 2)):
        B = np.array([x, FLOOR_Y, WALL_Z]); two = np.array([B, B + [0, -1800.0, 0]]); Xl = to_left(v, two)
        for cam in ("left", "right"):
            px, ok = project(cal, Xl, cam, baseline_true); assert ok.all()
            anns.append(dict(viewpoint=f"V{v}", camera=cam, frame_index=60, boundary="corner", orientation="vertical", points=px.tolist()))
    for cam in ("right",):
        anns.append(dict(viewpoint="V2", camera=cam, frame_index=60, boundary="ceiling", orientation="horizontal", points=[[100.0, 300.0], [900.0, 320.0]]))
    return cal, ms, dict(schema=T.ANNOTATION_SCHEMA, annotations=anns), length_mm
