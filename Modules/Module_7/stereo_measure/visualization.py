"""Final Model B visualizations (OpenCV only): annotated stereo pairs, top-down depth view, and error-vs-truth chart."""
import cv2
import numpy as np

from .config import CORNER_NAMES, EDGES, GT_HEIGHT_MM, GT_WIDTH_MM
from .measurement import left_photo, right_photo

COLORS = [(0, 200, 255), (0, 220, 0), (255, 120, 0), (255, 0, 200)]   # BGR for TL, TR, BR, BL


def _text(im, s, org, scale=0.6, color=(30, 30, 30), th=1):
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def _draw_view(path, px, labels, scale):
    im = cv2.imread(str(path))                                       # applies EXIF orientation -> 4284 x 5712
    if im is None:
        raise FileNotFoundError(path)
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    p = px * scale
    cv2.polylines(im, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
    cen = p.mean(0)
    for k, (x, y) in enumerate(p):
        cv2.circle(im, (int(round(x)), int(round(y))), 7, COLORS[k], 2, cv2.LINE_AA)
        v = np.array([x, y]) - cen
        v = v / (np.linalg.norm(v) + 1e-9)
        _text(im, labels[k], (int(x + v[0] * 40 - 30), int(y + v[1] * 40 + 6)), 0.6, COLORS[k], 2)
    return im


def annotated_pair(res, scale=0.3):
    """Left and right photographs side by side with the four approved corners, their Z-depths, and measured width/height."""
    X, s = res["X"], res["summary"]
    labels = [f"{c} Z={X[k, 2]:.0f}" for k, c in enumerate(CORNER_NAMES)]
    a = _draw_view(left_photo(res["pair"]), res["left_px"], labels, scale)
    b = _draw_view(right_photo(res["pair"]), res["right_px"], labels, scale)
    out = np.hstack([a, b])
    off = a.shape[1]
    for k in range(4):
        pa = (int(round(res["left_px"][k, 0] * scale)), int(round(res["left_px"][k, 1] * scale)))
        pb = (int(round(res["right_px"][k, 0] * scale)) + off, int(round(res["right_px"][k, 1] * scale)))
        cv2.line(out, pa, pb, COLORS[k], 1, cv2.LINE_AA)
    _text(out, f"Pair {res['pair']} (Model B)  mean Z = {s['mean_Z_depth_mm']:.1f} mm", (20, 40), 0.9, (0, 0, 0), 2)
    _text(out, f"width {s['width_mm']:.1f} mm ({s['width_err_mm']:+.1f} vs {GT_WIDTH_MM:.0f})   "
               f"height {s['height_mm']:.1f} mm ({s['height_err_mm']:+.1f} vs {GT_HEIGHT_MM:.0f})", (20, 80), 0.8, (0, 0, 0), 2)
    return out


def depth_topdown(results, size=(1000, 760)):
    """Top-down (X right, Z forward) view of the reconstructed corners of all pairs, with the camera at the bottom."""
    W, H = size
    im = np.full((H, W, 3), 255, np.uint8)
    allp = np.vstack([r["X"][:, [0, 2]] for r in results])
    x0, x1 = allp[:, 0].min() - 80, allp[:, 0].max() + 80
    z0, z1 = 0.0, allp[:, 1].max() + 100
    sc = min((W - 160) / (x1 - x0), (H - 140) / (z1 - z0))
    P = lambda x, z: (int(80 + (x - x0) * sc), int(H - 60 - (z - z0) * sc))
    cv2.line(im, P(x0, 0), P(x1, 0), (200, 200, 200), 1)
    cv2.drawMarker(im, P(0, 0), (0, 0, 0), cv2.MARKER_TRIANGLE_UP, 18, 2)
    _text(im, "left camera", (P(0, 0)[0] + 14, P(0, 0)[1] - 6), 0.5)
    pair_col = [(0, 140, 255), (0, 170, 0), (200, 60, 0)]
    for r, col in zip(results, pair_col):
        pts = [P(*r["X"][k][[0, 2]]) for k in range(4)]
        cv2.polylines(im, [np.array(pts, np.int32).reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
        for k, p in enumerate(pts):
            cv2.circle(im, p, 4, col, -1, cv2.LINE_AA)
            _text(im, f"{CORNER_NAMES[k]} {r['X'][k, 2]:.0f}", (p[0] + 6, p[1] - 6), 0.45, col)
        _text(im, f"Pair {r['pair']}  mean Z {r['summary']['mean_Z_depth_mm']:.1f} mm", (20, 30 + 24 * (r["pair"] - 1)), 0.6, col, 2)
    _text(im, "Z (forward, mm) vs X (right, mm); corner labels show Z-depth", (20, H - 15), 0.5)
    return im


def error_chart(summaries, size=(900, 520)):
    """Measured width/height per pair against the 305 x 215 mm truth."""
    W, H = size
    im = np.full((H, W, 3), 255, np.uint8)
    L, B, T = 80, 80, 70
    ph = H - B - T
    vmax = 330.0
    Y = lambda v: int(T + ph * (1 - v / vmax))
    _text(im, "Model B: measured size vs ground truth (mm)", (L, 35), 0.7, (0, 0, 0), 2)
    for v in range(0, 331, 50):
        cv2.line(im, (L, Y(v)), (W - 30, Y(v)), (225, 225, 225), 1)
        _text(im, str(v), (20, Y(v) + 5), 0.45)
    gw = (W - L - 40) / len(summaries)
    for j, s in enumerate(summaries):
        for k, (key, truth, col) in enumerate((("width_mm", GT_WIDTH_MM, (200, 100, 20)), ("height_mm", GT_HEIGHT_MM, (40, 150, 230)))):
            x0 = int(L + j * gw + gw * (0.12 + 0.4 * k)); x1 = int(x0 + gw * 0.34)
            cv2.rectangle(im, (x0, Y(s[key])), (x1, Y(0)), col, -1)
            cv2.line(im, (x0 - 4, Y(truth)), (x1 + 4, Y(truth)), (0, 0, 200), 2)
            _text(im, f"{s[key]:.1f}", (x0, Y(s[key]) - 8), 0.5)
        _text(im, f"Pair {s['pair']}", (int(L + j * gw + gw * 0.35), Y(0) + 25), 0.55)
    _text(im, "blue = width, orange = height, red line = ground truth (305 / 215)", (L, H - 20), 0.5)
    return im
