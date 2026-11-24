"""VALIDATION ONLY.  The only module that knows the ground truth and the measured lens height.
It must never be imported by the estimation code (enforced by tests); call it after an estimate exists."""
import numpy as np
from . import geometry as G
from .lines import fit_line

GROUND_TRUTH = dict(width_mm=2900.0, height_mm=2400.0)
LENS_HEIGHT_MM = 640.0

def compare_to_ground_truth(result: dict) -> dict:
    """result: output of pipeline.run_estimation (already complete).  Pure comparison; does not alter the estimate."""
    s = result["summary"]; unc = result.get("uncertainty"); out = {}
    for key, est in (("width_mm", s["width_mm"]), ("height_mm", s["height_mm"])):
        gt = GROUND_TRUTH[key]; row = dict(estimate_mm=est, ground_truth_mm=gt, error_mm=est - gt, abs_error_mm=abs(est - gt), error_percent=100 * (est - gt) / gt)
        if unc is not None:
            iv = unc["total"]["width" if key == "width_mm" else "height"]["interval_95_mm"]; row.update(interval_95_mm=iv, ground_truth_inside_interval=bool(iv[0] <= gt <= iv[1]))
        out[key] = row
    out["lens_height_check"] = lens_height_check(result)
    out["note"] = "Ground truth was applied only after the estimate was complete; it never entered any fit, selection or tuning."
    return out

def lens_height_check(result):
    """Diagnostic: robust-IRLS floor height (distance from the Air optical centre) vs the measured lens height."""
    rows = [r for r in result["per_set"] if r.get("ok")]; h = np.array([r["floor_height_mm"] for r in rows])
    return dict(measured_lens_height_mm=LENS_HEIGHT_MM, estimated_floor_height_mm_view1_view2=h.mean(0).tolist(), difference_mm=(h.mean(0) - LENS_HEIGHT_MM).tolist(),
                note="Diagnostic only: the estimator never used the measured lens height.")

def carpet_edge(g, xr, yr, thr, xstep=4, sm=9):
    import cv2
    im = g.astype(np.float32); mm = cv2.blur(im, (sm, sm)); s = np.sqrt(np.maximum(cv2.blur(im * im, (sm, sm)) - mm * mm, 0)); s = cv2.blur(s, (5, 5)); pts = []
    for x in range(xr[0], xr[1], xstep):
        col = s[yr[0]:yr[1], x - 2:x + 3].mean(1); idx = np.where(col > thr)[0]
        for i in idx:
            if i + 15 < len(col) and (col[i:i + 15] > thr).mean() > 0.9: pts.append((x, yr[0] + i)); break
    return np.array(pts)

def floor_contact_check(cal, p5, frames_set, lines, geom_row, annotations):
    """View-2 diagnostic: carpet-edge/corner-line intersection rays meet a floor plane at the MEASURED lens height; compare with the
    estimator's line-and-floor bottom corner.  Uses the measured height, hence validation only."""
    import cv2
    R, T = G.pose_from_params(p5, cal.baseline_mm); win = annotations["carpet_edge_windows"]; chk = []
    for k in ("AR", "PR"):
        w = win[k]; K, D = (cal.K_left, cal.D_left) if w["camera"] == "left" else (cal.K_right, cal.D_right); und = cv2.undistort(frames_set[k], K, D, None, K)
        P = carpet_edge(und, tuple(w["x_range"]), tuple(w["y_range"]), win["texture_threshold"])
        if len(P) < 8: return dict(ok=False, reason=f"{k}: {len(P)} carpet-edge points")
        fl = fit_line(P, "h"); p = G.line_intersection(lines[w["line"]], fl); cam = "L" if w["camera"] == "left" else "R"
        o, d = G.camera_ray(cal, R, T, cam, p); chk.append(G.ray_plane_intersection(o, d, geom_row["floor_normals"][1], LENS_HEIGHT_MM))
    return dict(ok=True, bottom_corner_difference_mm=float(np.linalg.norm(np.mean(chk, 0) - geom_row["bottom_corners"]["V2"])))
