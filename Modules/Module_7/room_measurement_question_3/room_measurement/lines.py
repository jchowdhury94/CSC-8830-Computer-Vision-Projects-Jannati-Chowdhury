"""Wall-boundary line fitting from the MANUAL seed annotations (config/wall_boundary_annotations.json).
Seeds are manual observations; sub-pixel edge locations and line fits are computed automatically."""
import json
from pathlib import Path
import cv2
import numpy as np

DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent / "config" / "wall_boundary_annotations.json"

def load_annotations(path=DEFAULT_ANNOTATIONS):
    a = json.loads(Path(path).read_text())
    for nm, s in a["lines"].items():
        if s["kind"] not in ("v", "h") or s["method"] not in ("edge_pts", "track2", "track_ceilS"): raise ValueError(f"bad annotation {nm}")
    return a

def undistorted_blur(gray, K, D, sigma=2.0):
    return cv2.GaussianBlur(cv2.undistort(gray, K, D, None, K).astype(np.float32), (0, 0), sigma)

def _subpixel(d, i):
    a, b, c3 = d[i - 1], d[i], d[i + 1]; den = a - 2 * b + c3; return 0.5 * (a - c3) / den if abs(den) > 1e-9 else 0

def edge_points(g, kind, span, center, half, sign, avg=9, step=6):
    """Gradient-peak edge points along a seed.  kind 'v': center(y)->x ; 'h': center(x)->y.  Returns (x, y) points."""
    pts = []
    for s in range(span[0], span[1], step):
        c = center(s); lo, hi = int(c - half), int(c + half)
        if kind == "v": band = g[s - avg // 2:s + avg // 2 + 1, lo - 1:hi + 2].mean(0)
        else: band = g[lo - 1:hi + 2, s - avg // 2:s + avg // 2 + 1].mean(1)
        d = np.gradient(band)[1:-1]; i = int(np.argmax(sign * d))
        if sign * d[i] < 1.0 or i in (0, len(d) - 1): continue
        pos = lo + i + _subpixel(d, i); pts.append((pos, s) if kind == "v" else (s, pos))
    return np.array(pts)

def track_vertical(g, ys, xfun, sign, half, avg=25, thr=0.4):
    pts = []
    for y in ys:
        c = xfun(y); lo, hi = int(c - half), int(c + half); band = g[y - avg // 2:y + avg // 2 + 1, lo - 1:hi + 2].mean(0); d = np.gradient(band)[1:-1]; i = int(np.argmax(sign * d))
        if sign * d[i] < thr or i in (0, len(d) - 1): continue
        pts.append((lo + i + _subpixel(d, i), y))
    return np.array(pts)

def fit_line(P, kind, it=15):
    """Robust (Tukey IRLS) line fit.  'v': x = a*y + b ; 'h': y = a*x + b."""
    if len(P) < 8: raise ValueError(f"only {len(P)} edge points; cannot fit a line")
    u, w = (P[:, 1], P[:, 0]) if kind == "v" else (P[:, 0], P[:, 1]); W = np.ones(len(u))
    for _ in range(it):
        A = np.c_[u, np.ones(len(u))] * W[:, None]; a, b = np.linalg.lstsq(A, w * W, rcond=None)[0]
        r = w - (a * u + b); s = max(1.4826 * np.median(np.abs(r)), 0.3); z = np.abs(r) / (4.685 * s); W = np.where(z < 1, (1 - z ** 2), 0)
    inl = np.abs(r) < 3 * s + 0.5; A = np.c_[u[inl], np.ones(inl.sum())]; a, b = np.linalg.lstsq(A, w[inl], rcond=None)[0]; r = w[inl] - (a * u[inl] + b)
    return dict(kind=kind, a=float(a), b=float(b), n=int(inl.sum()), N=len(u), rms=float(np.sqrt((r ** 2).mean())), span=(float(u[inl].min()), float(u[inl].max())), pts=P[inl])

def fit_lines(frames, cal, annotations):
    """frames: dict AL,PL,AR,PR grayscale (distorted).  Returns {line name: fitted line dict} (undistorted pixel coordinates)."""
    G = {k: undistorted_blur(frames[k], *((cal.K_left, cal.D_left) if k[0] == "A" else (cal.K_right, cal.D_right))) for k in frames}; out = {}
    for nm, s in annotations["lines"].items():
        g = G[nm[:2]]; seed = (lambda t, s=s: s["c0"] + s["slope"] * (t - s["ref"])); span = tuple(s["span"])
        if s["method"] == "edge_pts": P = edge_points(g, s["kind"], span, seed, s["half"], s["sign"])
        elif s["method"] == "track2":
            P = track_vertical(g, range(span[0], span[1], 6), seed, s["sign"], 10); l = fit_line(P, "v")
            P = track_vertical(g, range(span[0], span[1], 6), (lambda y, l=l: l["a"] * y + l["b"]), s["sign"], 5)
        else: P = track_vertical(g, range(span[0], span[1], 4), seed, s["sign"], s["half"], avg=5, thr=0.8)
        out[nm] = fit_line(P, s["kind"])
    return out

def retarget_lines(lines, cal_from, cal_to):
    """Re-express fitted line points under other intrinsics (as if the image had been undistorted with cal_to) and refit."""
    out = {}
    for nm, l in lines.items():
        left = nm[0] == "A"; KA, DA = (cal_from.K_left, cal_from.D_left) if left else (cal_from.K_right, cal_from.D_right); K, D = (cal_to.K_left, cal_to.D_left) if left else (cal_to.K_right, cal_to.D_right)
        P = l["pts"].astype(np.float64); n = (np.linalg.inv(KA) @ np.c_[P, np.ones(len(P))].T).T
        dist = cv2.projectPoints(n.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), KA, DA)[0].reshape(-1, 2)
        out[nm] = fit_line(cv2.undistortPoints(dist.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2), l["kind"])
    return out
