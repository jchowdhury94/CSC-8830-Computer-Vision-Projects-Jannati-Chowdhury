"""End-to-end Method J room measurement.  Estimation only: never imports ground truth (see validation.py).

Manual observations (inputs a person provided):  config/frame_manifest.json (which frames), config/wall_boundary_annotations.json
(boundary-line seeds, carpet-edge windows), the physical baseline, the Recipe A calibration.
Automatic values: features/matches, pose, floor planes, line fits, corners, registration, dimensions, uncertainty.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import calibration as C, features, frames as Fr, geometry as G, lines as L, measure as M, pose as P, uncertainty as U

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def to_jsonable(o):
    if isinstance(o, dict): return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return to_jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, float) and not np.isfinite(o): return None
    return o

def estimation_config():
    return dict(
        feature_matching=dict(detector="SIFT", nfeatures=features.SIFT_NFEATURES, contrast_threshold=0.04, ratio_air_pro=features.RATIO_STEREO, ratio_air_air_cross_view=features.RATIO_CROSS, mutual_check=True, rng_seed=0),
        pose=dict(method="J: robust epipolar (Sampson) + rigid four-view track reprojection, shared pose over all frame sets", parameters="3 rotation + 2 baseline direction (+6 rig-motion per set); |T| fixed at the physical baseline",
                  loss="Tukey biweight", tukey_c=G.TUKEY_C, scale="max(1.4826*MAD, floor)", scale_floor_px=P.SCALE_FLOOR_PX, solver="Levenberg-Marquardt on IRLS weights, forward-difference Jacobian",
                  convergence=dict(epipolar=dict(max_param_step=1e-9, rel_cost_change=1e-10, max_iter=100), joint=dict(max_param_step=1e-8, rel_cost_change=1e-9, max_iter=60)),
                  weighting=dict(spatial_declustering_grid_px=P.GRID_PX, per_frame_set_equal_weight=True, track_block_weight_per_set="equal to the epipolar block"),
                  initialization="IRLS 8-point essential matrix (no RANSAC, no RNG) + cheirality via recoverPose",
                  degenerate_rule=dict(robust_scale_px_gt=P.DEGENERATE_SCALE_PX, inlier_fraction_lt=P.MIN_INLIER_FRACTION, positive_depth_fraction_lt=P.MIN_POSITIVE_DEPTH_FRACTION, not_converged=True),
                  fallback_start="parallel cameras, baseline along -x (R = I, t = (-1, 0, 0.001))"),
        floor=dict(method="Tukey IRLS plane fit, PCA start, over triangulated points with left-image row > 2600", iterations=30),
        registration=dict(method="seeded 3-point RANSAC rigid fit, threshold mm", threshold_mm=M.REG_THR_MM, iterations=M.REG_ITERS, rng_seed_per_set="3 + set index"),
        point_gates=dict(depth_mm=M.DEPTH_RANGE_MM, reprojection_px=M.GATE_PX, epipolar_px=M.GATE_PX),
        top_corner_variant=0, units="mm", coordinate_frame="left (Air) camera frame of each view; view 1 registered into view 2")

def run_estimation(video_dir=Fr.DEFAULT_VIDEO_DIR, calibration_path=C.DEFAULT_CALIBRATION_PATH, manifest_path=Fr.DEFAULT_MANIFEST, annotations_path=L.DEFAULT_ANNOTATIONS,
                   verify_hashes=True, with_uncertainty=True, uncertainty_kwargs=None, keep_internals=False, log=print):
    cal = C.load_calibration(calibration_path); manifest = Fr.load_manifest(manifest_path); ann = L.load_annotations(annotations_path)
    ver = Fr.verify_videos(manifest, video_dir)
    if verify_hashes and not all(r["ok"] for r in ver): raise RuntimeError(f"video hash verification failed: {[r for r in ver if not r['ok']]}")
    log("decoding frames ..."); sets = Fr.load_frame_sets(manifest, video_dir)
    log("matching features and fitting boundary lines ..."); match_sets = [features.match_set(s) for s in sets]; line_sets = [L.fit_lines(s, cal, ann) for s in sets]
    D = P.build_dataset(cal, match_sets); log(f"pooled {len(D['n1'])} stereo matches, {len(D['T']['obs'])} four-view tracks; estimating the shared pose ...")
    pJ, pA, prep, _ = P.estimate_pose(D)
    if prep["method_J"]["flags"]["degenerate"]: raise RuntimeError(f"degenerate pose solution: {prep}")
    R, T = G.pose_from_params(pJ, cal.baseline_mm); rows = M.measure_all_sets(cal, pJ, match_sets, line_sets)
    ok = [r for r in rows if r["ok"]]
    if not ok: raise RuntimeError("no frame set produced a measurement")
    W = np.array([r["width_mm"] for r in ok]); H = np.array([r["height_mm"] for r in ok])
    rA = M.measure_all_sets(cal, pA, match_sets, line_sets)
    res = dict(
        schema="csc8830.module7.room_measurement_result.v1", ground_truth_used=False,
        inputs=dict(videos=ver, frame_indices=Fr.frame_index_table(manifest), calibration=dict(name=cal.name, file=str(Path(calibration_path).name), sha256=_sha(calibration_path), image_size=cal.image_size, physical_baseline_mm=cal.baseline_mm),
                    annotations=dict(file=str(Path(annotations_path).name), sha256=_sha(annotations_path), kind="MANUAL seeds; fitted lines are automatic"), frame_manifest_sha256=_sha(manifest_path)),
        config=estimation_config(),
        manual_vs_automatic=dict(manual=["frame selection rule (documented)", "boundary-line seeds and search windows", "carpet-edge diagnostic windows", "physical baseline 225 mm", "which wall corners/edges are used"],
                                 automatic=["SIFT matches, tracks", "shared stereo pose", "floor planes", "fitted lines", "corners, registration", "width/height", "uncertainty"]),
        pose=dict(R=R, T_mm=T, params=pJ, rodrigues_deg=np.degrees(pJ[:3]), yaw_deg=G.yaw_deg(pJ), baseline_mm=float(np.linalg.norm(T)), method_A_rodrigues_deg=np.degrees(pA[:3]), report=prep),
        per_set=[dict(frame_set=i, **{k: v for k, v in r.items()}) for i, r in enumerate(rows)],
        summary=dict(width_mm=float(W.mean()), height_mm=float(H.mean()), width_sd_across_sets_mm=float(W.std(ddof=1)) if len(W) > 1 else 0.0, height_sd_across_sets_mm=float(H.std(ddof=1)) if len(H) > 1 else 0.0,
                     n_sets_ok=len(ok), n_sets=len(rows), method_A_width_mm=float(np.nanmean([r["width_mm"] for r in rA])), method_A_height_mm=float(np.nanmean([r["height_mm"] for r in rA])),
                     note="Spread across sets is small by construction (one shared pose); it is not the measurement uncertainty. See 'uncertainty'."))
    if with_uncertainty:
        log("estimating uncertainty (bootstrap, boundary and intrinsics perturbations) ...")
        res["uncertainty"] = U.estimate_uncertainty(cal, D, match_sets, line_sets, pJ, pA, res["summary"]["width_mm"], res["summary"]["height_mm"], **(uncertainty_kwargs or {}))
    out = to_jsonable(res)
    if keep_internals: return out, dict(cal=cal, sets=sets, match_sets=match_sets, line_sets=line_sets, D=D, pJ=pJ, pA=pA, rows=rows, annotations=ann, rows_raw=rows)
    return out
