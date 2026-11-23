"""Stage F8.5: synthetic stereo checkerboard videos with KNOWN ground truth (test data only; no real or protected video is read).

A flat 8x6 inner-corner board (9x7 squares of 20 mm) is rendered into two virtual pinhole cameras (identical intrinsics, known baseline and a
small relative rotation, no lens distortion) while it holds ~1.2 s in each of N poses and glides between them, so the calibration code sees what
a real recording looks like: still holds, motion in between, left/right in step or with a time offset. mp4v is used because OpenCV can both write
and read it everywhere.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

PATTERN = (8, 6)                 # inner corners (columns, rows)
SQUARE_MM = 20.0
TEX_PX_PER_MM = 2.5
BOARD_W_MM, BOARD_H_MM, BORDER_MM = 9 * SQUARE_MM, 7 * SQUARE_MM, 20.0


def _rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def board_texture():
    s = TEX_PX_PER_MM; tw, th = int((BOARD_W_MM + 2 * BORDER_MM) * s), int((BOARD_H_MM + 2 * BORDER_MM) * s)
    tex = np.full((th, tw), 235, np.uint8)
    for r in range(7):
        for c in range(9):
            if (r + c) % 2 == 0:
                x0, y0 = int((BORDER_MM + c * SQUARE_MM) * s), int((BORDER_MM + r * SQUARE_MM) * s)
                tex[y0:y0 + int(SQUARE_MM * s), x0:x0 + int(SQUARE_MM * s)] = 25
    return tex


def make_poses(n, seed=7):
    """n distinct board poses (angles in degrees, offsets in mm) in the LEFT camera frame; the board sits between the two optical axes."""
    rng = np.random.default_rng(seed); poses = []
    tilts = [(-25, -20), (20, -25), (-22, 22), (25, 18), (0, 0), (-10, 30), (12, -30), (30, 5), (-30, -5), (8, 24), (-18, -28), (28, -12), (-5, 12), (15, 15), (-28, 25), (22, 28)]
    for k in range(n):
        ax, ay = tilts[k % len(tilts)]
        poses.append(dict(rx=ax + rng.uniform(-4, 4), ry=ay + rng.uniform(-4, 4), rz=rng.uniform(-18, 18), dx=rng.uniform(-30, 30), dy=rng.uniform(-25, 25), z=rng.uniform(335, 445)))
    return poses


def _interp(a, b, t):
    t = t * t * (3 - 2 * t)                                        # smoothstep
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def scene_timeline(poses, hold, trans, rng):
    """Per scene-frame (pose, held) pairs. Hold and glide lengths vary per pose (like a person moving a board), which also makes the motion
    signal aperiodic so the left/right time offset is identifiable."""
    out = []
    for i, p in enumerate(poses):
        out += [(p, True)] * int(rng.integers(max(hold - 6, 10), hold + 16))
        if i + 1 < len(poses):
            n = int(rng.integers(max(trans - 4, 4), trans + 7))
            out += [(_interp(p, poses[i + 1], (j + 1) / (n + 1)), False) for j in range(n)]
    return out


def make_stereo_pair(out_dir, n_poses=16, hold=36, trans=12, w=1280, h=720, fps=30.0, f=1000.0, baseline_mm=100.0, right_starts_later_by=0, seed=7,
                     noise=1.2, blur=0.7):
    """Write left.mp4 / right.mp4 into ``out_dir`` and return the ground truth.

    right_starts_later_by = k: the RIGHT recording starts k scene frames after the LEFT one, so F7's offset d (right frame i + d shows the instant of
    left frame i) is -k."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    Rc = _rot(math.radians(0.4), math.radians(1.0), math.radians(-0.3)); tc = np.array([-baseline_mm, 0.0, 0.0])   # left-camera frame -> right-camera frame
    tex = board_texture(); s = TEX_PX_PER_MM; tex_to_board = np.array([[1 / s, 0, -(BOARD_W_MM / 2 + BORDER_MM)], [0, 1 / s, -(BOARD_H_MM / 2 + BORDER_MM)], [0, 0, 1.0]])
    yy, xx = np.mgrid[0:h, 0:w]; bg = (95 + 25 * (xx / w) + 15 * (yy / h)).astype(np.uint8)
    timeline = scene_timeline(make_poses(n_poses, seed), hold, trans, np.random.default_rng(seed + 500))
    paths = {"left": out_dir / "left.mp4", "right": out_dir / "right.mp4"}
    writers = {k: cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) for k, p in paths.items()}
    if not all(wr.isOpened() for wr in writers.values()):
        raise RuntimeError("cv2.VideoWriter could not open mp4v")
    rng = np.random.default_rng(seed + 1000)
    n_frames = len(timeline)
    ranges = {"left": range(0, n_frames), "right": range(right_starts_later_by, n_frames)}
    rendered = 0
    for t in range(n_frames):
        pose, _ = timeline[t]
        Rb = _rot(*(math.radians(pose[k]) for k in ("rx", "ry", "rz"))); tb = np.array([baseline_mm / 2 + pose["dx"], pose["dy"], pose["z"]])   # centred between the two optical axes
        for cam in ("left", "right"):
            if t not in ranges[cam]:
                continue
            R, tt = (Rb, tb) if cam == "left" else (Rc @ Rb, Rc @ tb + tc)
            H = K @ np.column_stack([R[:, 0], R[:, 1], tt]) @ tex_to_board
            img = bg.copy()
            cv2.warpPerspective(tex, H, (w, h), dst=img, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            img = cv2.GaussianBlur(img, (0, 0), blur) if blur else img
            if noise:
                img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
            writers[cam].write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        rendered += 1
    for wr in writers.values():
        wr.release()
    return dict(left=str(paths["left"]), right=str(paths["right"]), K=K, R=Rc, T=tc.reshape(3, 1), baseline_mm=float(baseline_mm), size=(w, h), fps=fps, pattern=PATTERN,
                square_mm=SQUARE_MM, n_scene_frames=n_frames, n_left=n_frames, n_right=n_frames - right_starts_later_by, expected_offset=-int(right_starts_later_by),
                n_poses=n_poses, hold=hold, trans=trans)


if __name__ == "__main__":
    import sys, time
    t0 = time.time(); gt = make_stereo_pair(sys.argv[1] if len(sys.argv) > 1 else "synthetic_stereo_out")
    print({k: gt[k] for k in ("left", "right", "n_left", "n_right")}, f"{time.time() - t0:.1f}s")
