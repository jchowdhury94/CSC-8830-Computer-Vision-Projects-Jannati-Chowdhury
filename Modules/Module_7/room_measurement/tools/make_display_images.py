"""Build the small, deployment-safe images shown on the Room Measurement results page (Module 7 Part 3, Phase 2A).

    python Modules/Module_7/room_measurement/tools/make_display_images.py [--frame-set 0] [--out DIR]

Inputs are the four original wall videos, only ever READ (never modified, moved or re-encoded).  The estimator
(room_measurement.pipeline, unchanged) is re-run without the uncertainty analysis; the script refuses to write anything if the
re-run does not reproduce the committed estimate for the chosen frame set.  No ground truth is read.  Output (default
``Modules/Module_7/outputs/room_measurement/display/``): downscaled JPEGs plus ``display_manifest.json``.

Layers drawn on the frames (frames are undistorted with the Recipe A intrinsics, shown in grayscale as the estimator sees them):
  OBSERVED  cyan dots     edge points found automatically (sub-pixel gradient peaks) near a manually seeded boundary
  FITTED    yellow lines  robust line fits through the observed edge points
  INFERRED  red diamonds  corners computed geometrically from the fitted lines + stereo pose + floor plane (never seen directly)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE_DIR))
from room_measurement import geometry as G, measure as M, pipeline  # noqa: E402

OUT_DIR = MODULE_DIR / "outputs" / "room_measurement" / "display"
COMMITTED = MODULE_DIR / "outputs" / "room_measurement" / "estimation_result.json"
PANEL_W = 480                      # each camera panel is downscaled to this width (frames are 2160 x 3840)
CYAN, YELLOW, RED, GREEN, WHITE = (255, 220, 0), (0, 230, 255), (60, 60, 235), (90, 220, 90), (255, 255, 255)   # BGR


def undistorted(gray, K, D):
    return cv2.undistort(gray, K, D, None, K)


def panel(gray, K, D):
    img = cv2.cvtColor(undistorted(gray, K, D), cv2.COLOR_GRAY2BGR)
    s = PANEL_W / img.shape[1]
    return cv2.resize(img, (PANEL_W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA), s


def project(K, X):
    p = K @ X
    return p[:2] / p[2]


def draw_fitted_line(img, s, line, color=YELLOW):
    a, b = line["a"], line["b"]
    lo, hi = line["span"]
    if line["kind"] == "v":
        p0, p1 = (a * lo + b, lo), (a * hi + b, hi)
    else:
        p0, p1 = (lo, a * lo + b), (hi, a * hi + b)
    cv2.line(img, tuple(int(round(v * s)) for v in p0), tuple(int(round(v * s)) for v in p1), color, 2, cv2.LINE_AA)


def draw_points(img, s, pts, color=CYAN):
    for x, y in np.asarray(pts):
        cv2.circle(img, (int(round(x * s)), int(round(y * s))), 2, color, -1, cv2.LINE_AA)


def draw_diamond(img, s, xy, label):
    x, y = int(round(xy[0] * s)), int(round(xy[1] * s))
    if not (8 <= x < img.shape[1] - 8 and 8 <= y < img.shape[0] - 8):
        return False                              # projects outside this camera's frame: not drawn, never clamped
    pts = np.array([[x, y - 10], [x + 10, y], [x, y + 10], [x - 10, y]], np.int32)
    cv2.fillConvexPoly(img, pts, RED, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, WHITE, 2, cv2.LINE_AA)
    tx = x + 14 if x < img.shape[1] - 150 else x - 150
    cv2.putText(img, label, (tx, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (tx, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return True


def title_bar(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def save(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    assert ok
    Path(path).write_bytes(buf.tobytes())
    return len(buf)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frame-set", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k = a.frame_set

    res, it = pipeline.run_estimation(with_uncertainty=False, keep_internals=True)
    cal, pJ = it["cal"], it["pJ"]
    committed = json.loads(COMMITTED.read_text())
    if abs(res["summary"]["width_mm"] - committed["summary"]["width_mm"]) > 1e-3 or \
            abs(res["summary"]["height_mm"] - committed["summary"]["height_mm"]) > 1e-3:
        sys.exit("re-run does not reproduce the committed estimate; nothing written")

    R, T = G.pose_from_params(pJ, cal.baseline_mm)
    F, lines, frames = it["match_sets"][k], it["line_sets"][k], it["sets"][k]
    geom = M.reconstruct_geometry(cal, R, T, F, np.random.default_rng(3 + k))
    meas = M.measure_wall(cal, R, T, geom, lines)
    if abs(meas["width_mm"] - committed["per_set"][k]["width_mm"]) > 1e-3:
        sys.exit("frame-set measurement differs from the committed per-set value; nothing written")

    cams = {"AL": (cal.K_left, cal.D_left), "PL": (cal.K_right, cal.D_right), "AR": (cal.K_left, cal.D_left), "PR": (cal.K_right, cal.D_right)}
    bases = {key: panel(frames[key], *cams[key]) for key in cams}
    names = {"AL": "Left camera (iPhone Air) - placement 1", "PL": "Right camera (iPhone Pro) - placement 1",
             "AR": "Left camera (iPhone Air) - placement 2", "PR": "Right camera (iPhone Pro) - placement 2"}
    line_names = {"V1": ("AL_corner", "PL_corner", "PL_ceilF", "PL_ceilS"), "V2": ("AR_corner", "PR_corner", "PR_ceilF", "PR_ceilS")}
    sizes, drawn_corners = {}, {}

    for v, (left_key, right_key) in {"V1": ("AL", "PL"), "V2": ("AR", "PR")}.items():
        n = "1" if v == "V1" else "2"
        plain = [bases[left_key][0].copy(), bases[right_key][0].copy()]
        for im, key in zip(plain, (left_key, right_key)):
            title_bar(im, names[key])
        sizes[f"view{n}_stereo_pair.jpg"] = save(out / f"view{n}_stereo_pair.jpg", np.hstack(plain))

        ann = [bases[left_key][0].copy(), bases[right_key][0].copy()]
        for im, key in zip(ann, (left_key, right_key)):
            s = bases[key][1]
            for nm, ln in lines.items():
                if nm[:2] == key and nm in line_names[v]:
                    draw_points(im, s, ln["pts"])
            for nm, ln in lines.items():
                if nm[:2] == key and nm in line_names[v]:
                    draw_fitted_line(im, s, ln)
        o = meas[v]
        B = o["B"]
        top3d = B + o["H"] * geom["f1" if v == "V1" else "f2"][0]
        drawn = {}
        for im, key, (X_cam) in zip(ann, (left_key, right_key), (lambda X: X, lambda X: R @ X + T.reshape(3))):
            K = cams[key][0]
            s = bases[key][1]
            drawn[key] = {"bottom": draw_diamond(im, s, project(K, X_cam(B)), "bottom corner*")}
            if key == right_key:
                drawn[key]["top"] = draw_diamond(im, s, o["top_px"], "top corner*")
            else:
                drawn[key]["top"] = draw_diamond(im, s, project(K, X_cam(top3d)), "top corner*")
        drawn_corners[v] = drawn
        for im, key in zip(ann, (left_key, right_key)):
            title_bar(im, names[key])
        sizes[f"view{n}_boundaries_corners.jpg"] = save(out / f"view{n}_boundaries_corners.jpg", np.hstack(ann))

    # two overlapping wall views: the two left-camera frames with a sample of the cross-view SIFT matches (directly observed)
    pa, pb = bases["AL"][0].copy(), bases["AR"][0].copy()
    sa, sb = bases["AL"][1], bases["AR"][1]
    MX = F["MX"]
    # MX holds keypoint index pairs of AL <-> AR; recover their pixel positions by repeating the (deterministic, seeded) detection
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.04)
    kA = np.array([p.pt for p in sift.detectAndCompute(frames["AL"], None)[0]], np.float32)
    kB = np.array([p.pt for p in sift.detectAndCompute(frames["AR"], None)[0]], np.float32)
    uA = cv2.undistortPoints(kA[MX[:, 0]].reshape(-1, 1, 2), cal.K_left, cal.D_left, P=cal.K_left).reshape(-1, 2)
    uB = cv2.undistortPoints(kB[MX[:, 1]].reshape(-1, 1, 2), cal.K_left, cal.D_left, P=cal.K_left).reshape(-1, 2)
    sel = np.linspace(0, len(MX) - 1, min(24, len(MX))).astype(int)
    canvas = np.hstack([pa, pb])
    for i in sel:
        p0 = (int(uA[i, 0] * sa), int(uA[i, 1] * sa))
        p1 = (int(uB[i, 0] * sb) + PANEL_W, int(uB[i, 1] * sb))
        cv2.line(canvas, p0, p1, GREEN, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, GREEN, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, GREEN, -1, cv2.LINE_AA)
    title_bar(canvas[:, :PANEL_W], "Placement 1 (left camera)")
    title_bar(canvas[:, PANEL_W:], "Placement 2 (left camera)")
    sizes["overlapping_views.jpg"] = save(out / "overlapping_views.jpg", canvas)

    manifest = dict(
        schema="csc8830.module7.room_display_images.v1", frame_set=k,
        frame_indices=committed["inputs"]["frame_indices"][k], ground_truth_used=False,
        reproduces_committed_estimate=True, bytes=sizes, total_bytes=int(sum(sizes.values())),
        n_cross_view_matches_total=int(len(MX)), n_cross_view_matches_drawn=int(len(sel)),
        corners_drawn_in_frame=drawn_corners,
        layers=dict(observed="cyan dots: automatically located edge points near manually seeded boundaries",
                    fitted="yellow lines: robust line fits through the observed edge points",
                    inferred="red diamonds (labels end with *): corners computed geometrically (a corner outside a camera's frame is not drawn)",
                    matches="green: sample of cross-view SIFT matches between the two left-camera placements"),
        note="Grayscale frames undistorted with the Recipe A intrinsics, downscaled to 480 px per panel; derived from the original videos, which are not modified.")
    (out / "display_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"bytes": sizes, "total": manifest["total_bytes"], "corners_drawn": drawn_corners}, indent=1))


if __name__ == "__main__":
    main()
