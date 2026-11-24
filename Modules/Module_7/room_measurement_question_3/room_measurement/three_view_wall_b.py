"""Three-viewpoint Wall B (room length) reconstruction -- isolated, no Streamlit, no I/O except reading the videos it is told to read.

Ported from the validated offline experiment (scratchpad ``common.py`` / ``recon.py`` / ``base.py``).  The mathematics is unchanged and uses the existing Method J
primitives (``features.match_set``, ``pose.fit_A_guarded``, ``geometry.*``, ``measure.view_points``) WITHOUT modifying them:

    1. per viewpoint v in (1, 2, 3): an independent stereo pose (R, T) from that viewpoint's Air<->Pro matches, with the baseline FIXED to the accepted value;
    2. triangulate each viewpoint's stereo matches and fit its floor plane (robust IRLS on the points in the lower part of the left image);
    3. register V1 -> V2 and V3 -> V2 with the seeded RANSAC rigid fit on cross-view (Air<->Air) matches that have stereo 3D in both views;
    4. V1's LEFT bottom wall corner and V3's RIGHT bottom wall corner = intersection of the two back-projected vertical-edge planes (Air + Pro) with that viewpoint's floor;
    5. both corners are moved into the V2 frame and Wall B's length is their 3D distance.

What is deliberately NOT here (Phase B, step 2): numerical pass/warn/fail gates.  This module reports the metrics a gate would need (``metrics`` in every frame result) and raises
``ThreeViewError`` only for STRUCTURAL problems (wrong calibration/resolution, missing or malformed annotations, too few registration pairs).  Quality classification is a later, separately
approved step.

Calibration: K, D, image size and baseline come from the accepted calibration CONTRACT (``calibration_from_contract``).  Nothing is read from a file, nothing is optimised or overwritten, and the
experiment's 225 mm is not assumed: whatever baseline the contract carries is the fixed baseline.

V2's horizontal (ceiling/wall) annotation is part of the annotation schema and is carried through, but it is NOT used by the reconstruction (it has not been validated as a constraint).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

import cv2
import numpy as np

from . import calibration as C, features as Fe, geometry as G, measure as M, pose as P, three_view_quality as Q
from .visitor_method_j_adapter import VisitorCalibrationContext

ANNOTATION_SCHEMA = "wall_b_three_viewpoint_annotations.v1"
VIEWPOINTS = (1, 2, 3)
FLOOR_ROW_REFERENCE = (2600, 3840)         # the Phase 1 floor-row threshold (2600 px) at the 3840 px frame height it was chosen for; scaled to other heights (an unvalidated assumption, as in Method J)
REG_PAIRS = ((1, 2), (3, 2))               # (moving viewpoint, reference viewpoint): V1 -> V2 and V3 -> V2
MIN_REG_PAIRS = 6                          # structural minimum of cross-view pairs with stereo 3D in both views (as in the reference); not a quality threshold
GRID = 4                                   # coverage grid (GRID x GRID cells of the left image) used only for reporting


class ThreeViewError(Exception):
    """A structural failure with a stable ``code`` and a visitor-safe ``message``."""
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}"); self.code, self.message = code, message


@dataclass(frozen=True)
class ThreeViewConfig:
    """Initial configuration taken from the offline experiment.  NOT universally validated settings."""
    contrast: float = 0.01          # SIFT contrastThreshold ("richer" matching)
    nfeatures: int = 40000
    ratio_cross: float = 0.9        # Air<->Air cross-view ratio test
    seed: int = 3                   # registration RANSAC seed (the reference used 3)
    floor_row_fraction: float = FLOOR_ROW_REFERENCE[0] / FLOOR_ROW_REFERENCE[1]   # floor rows = lower (1 - fraction) of the frame
    share_detections: bool = True   # FAST MATCHING (default for three-viewpoint Wall B only): detect SIFT once per image and match V2's stereo pair once (bit-identical output; see compute_match_sets_shared)
    detect_only_sharing: bool = False   # benchmarking aid: share the detections but still repeat V2's stereo match
    subset_mutual_check: bool = True    # mutual-consistency pass searches only the candidate rows (same per-row search; output asserted identical by tests and the real-data benchmark)
    thresholds: Q.QualityThresholds = Q.DEFAULT_THRESHOLDS

    @classmethod
    def original_matcher(cls, **kw) -> "ThreeViewConfig":
        """Regression / debugging: the original ``features.match_set`` path (per-pair detection, repeated V2 stereo match, full mutual-check pass).  Output is identical to the fast default."""
        return cls(share_detections=False, subset_mutual_check=False, **kw)

    def uses_fast_matcher(self) -> bool:
        return bool(self.share_detections or self.subset_mutual_check)

    def match_kwargs(self) -> dict:
        return dict(contrast=self.contrast, nfeatures=self.nfeatures, ratio_cross=self.ratio_cross)


# ------------------------------------------------------------------ calibration
def calibration_from_contract(contract: Mapping) -> C.StereoCalibration:
    """Accepted calibration contract (``K1, D1`` = LEFT/Air, ``K2, D2`` = RIGHT/Pro, ``image_size_wh``, ``baseline_mm``) -> validated Method J calibration.  Values are used as given."""
    try:
        ctx = VisitorCalibrationContext.from_params(dict(K_left=contract["K1"], D_left=contract["D1"], K_right=contract["K2"], D_right=contract["D2"], baseline_mm=contract["baseline_mm"]),
                                                    tuple(contract["image_size_wh"]), label="visitor")
        return ctx.to_stereo_calibration()
    except (KeyError, TypeError, ValueError) as e:
        raise ThreeViewError("CALIBRATION_INVALID", f"The calibration contract cannot be used ({type(e).__name__}: {e}).") from e


def floor_y_min(cal: C.StereoCalibration) -> float:
    """Left-image row above which points count as floor, scaled from the Phase 1 value to the calibration frame height (2600 px at 3840 px: unchanged for the original recordings)."""
    return FLOOR_ROW_REFERENCE[0] * (cal.image_size[1] / FLOOR_ROW_REFERENCE[1])


def check_frame_size(cal: C.StereoCalibration, image_wh, what: str):
    w, h = int(image_wh[0]), int(image_wh[1])
    if (w, h) != tuple(int(v) for v in cal.image_size):
        raise ThreeViewError("RESOLUTION_MISMATCH", f"{what} is {w} x {h} px but the accepted calibration is for {cal.image_size[0]} x {cal.image_size[1]} px.")


# ------------------------------------------------------------------ annotations
@dataclass(frozen=True)
class Annotations:
    """Per (viewpoint, frame, camera, orientation) two-point lines in ORIGINAL full-resolution, lens-DISTORTED pixels (``points`` = ((x0, y0), (x1, y1)))."""
    lines: dict
    image_wh: tuple | None = None

    def get(self, v: int, frame: int, camera: str, orientation: str) -> np.ndarray:
        try:
            return np.array(self.lines[(v, int(frame), camera, orientation)], float)
        except KeyError:
            raise ThreeViewError("ANNOTATION_MISSING", f"Viewpoint {v}, frame {frame}: the {'Air' if camera == 'left' else 'Pro'} {orientation} boundary has not been marked.") from None

    def has(self, v: int, frame: int, camera: str, orientation: str) -> bool:
        return (v, int(frame), camera, orientation) in self.lines


REQUIRED_VERTICAL = ((1, "left", "vertical"), (1, "right", "vertical"), (3, "left", "vertical"), (3, "right", "vertical"))


def parse_annotations(doc: Mapping, image_wh=None) -> Annotations:
    """Validate a ``wall_b_three_viewpoint_annotations.v1`` document.  Every point must be finite and (when ``image_wh`` is given) inside the frame; a line needs two distinct points.
    Duplicate entries for one (viewpoint, frame, camera, orientation) are refused, never merged."""
    if not isinstance(doc, Mapping) or doc.get("schema") != ANNOTATION_SCHEMA:
        raise ThreeViewError("ANNOTATION_SCHEMA", f"Annotations must use the {ANNOTATION_SCHEMA} schema.")
    out = {}
    for i, a in enumerate(doc.get("annotations") or []):
        try:
            v = int(str(a["viewpoint"]).lstrip("Vv")); f = int(a["frame_index"]); cam = a["camera"]; ori = a["orientation"]; pts = np.array(a["points"], float)
        except (KeyError, TypeError, ValueError):
            raise ThreeViewError("ANNOTATION_MALFORMED", f"Annotation #{i + 1} is incomplete.") from None
        if v not in VIEWPOINTS or cam not in ("left", "right") or ori not in ("vertical", "horizontal"):
            raise ThreeViewError("ANNOTATION_MALFORMED", f"Annotation #{i + 1} names an unknown viewpoint, camera or orientation.")
        if pts.shape != (2, 2) or not np.isfinite(pts).all() or np.linalg.norm(pts[0] - pts[1]) < 1.0:
            raise ThreeViewError("ANNOTATION_MALFORMED", f"Annotation #{i + 1} must be two distinct finite points.")
        if image_wh is not None and not ((pts[:, 0] >= 0) & (pts[:, 0] <= image_wh[0]) & (pts[:, 1] >= 0) & (pts[:, 1] <= image_wh[1])).all():
            raise ThreeViewError("ANNOTATION_OUTSIDE_FRAME", f"Annotation #{i + 1} lies outside the {image_wh[0]} x {image_wh[1]} frame.")
        key = (v, f, cam, ori)
        if key in out:
            raise ThreeViewError("ANNOTATION_DUPLICATE", f"Viewpoint {v}, frame {f}: the {cam} {ori} boundary was marked twice.")
        out[key] = pts.tolist()
    return Annotations(out, None if image_wh is None else tuple(image_wh))


def check_annotations_complete(ann: Annotations, frames) -> None:
    """The four vertical corner lines of V1 and V3 are required for every frame.  V2 needs nothing for the length (its horizontal line is stored, not used)."""
    missing = [f"V{v} {'Air' if cam == 'left' else 'Pro'} vertical corner (frame {f})" for f in frames for v, cam, ori in REQUIRED_VERTICAL if not ann.has(v, f, cam, ori)]
    if missing:
        raise ThreeViewError("ANNOTATION_MISSING", "Missing: " + "; ".join(missing) + ".")


# ------------------------------------------------------------------ frames and matches
def read_gray_frame(path, index: int, cal: C.StereoCalibration | None = None) -> np.ndarray:
    """Decode one frame as 8-bit gray (the same call sequence as the experiment).  Read-only.  Checks the decoded size against the calibration when one is given."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ThreeViewError("VIDEO_UNREADABLE", "A video could not be opened.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index)); ok, fr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ThreeViewError("FRAME_UNREADABLE", f"Frame {index} could not be read from a video.")
    if cal is not None:
        check_frame_size(cal, (fr.shape[1], fr.shape[0]), "A video frame")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def compute_match_sets(grays: Mapping, frame: int, cfg: ThreeViewConfig, log: Callable = lambda *a: None) -> dict:
    """``grays[(v, "L"|"R")]`` -> {(1, 2): match_set, (3, 2): match_set} for one frame index.  Each pair is one ``features.match_set`` (Air_a, Pro_a, Air_b, Pro_b)."""
    out = {}
    for a, b in REG_PAIRS:
        t = time.time()
        out[(a, b)] = Fe.match_set(dict(AL=grays[(a, "L")], PL=grays[(a, "R")], AR=grays[(b, "L")], PR=grays[(b, "R")]), **cfg.match_kwargs())
        log(f"matched V{a}/V{b} frame {frame}: stereo {len(out[(a, b)]['M1'])}/{len(out[(a, b)]['M2'])}, cross {len(out[(a, b)]['MX'])} ({time.time() - t:.1f}s)")
    return out


def compute_match_sets_shared(grays: Mapping, frame: int, cfg: ThreeViewConfig, log: Callable = lambda *a: None) -> dict:
    """Same output as ``compute_match_sets``, with less repeated work and NO change to the mathematics.

    ``features.match_set`` is called once per pair and detects all four of its images, so the middle viewpoint's two images are detected twice and its Air<->Pro match is computed twice
    (identically: SIFT and the brute-force matcher are deterministic and the V2 inputs are the same).  Here every image is detected once and V2's stereo match is computed once.  The matching
    code below is ``features.match_set``'s, line for line (ratio test + mutual check); the equality of the two outputs is asserted by the tests and by the real-data benchmark.
    """
    t = time.time(); cv2.setRNGSeed(0); sift = cv2.SIFT_create(nfeatures=cfg.nfeatures, contrastThreshold=cfg.contrast); kp, ds = {}, {}
    for v in VIEWPOINTS:
        for s in ("L", "R"):
            kk, d = sift.detectAndCompute(grays[(v, s)], None); kp[(v, s)] = np.array([p.pt for p in kk], np.float32); ds[(v, s)] = d
    bf = cv2.BFMatcher()
    def match(a, b, rt):
        m = bf.knnMatch(ds[a], ds[b], k=2); good = [x[0] for x in m if len(x) == 2 and x[0].distance < rt * x[1].distance]
        if cfg.subset_mutual_check:
            rows = np.unique(np.array([x.trainIdx for x in good], np.int64))
            back = {int(rows[x[0].queryIdx]): x[0].trainIdx for x in bf.knnMatch(np.ascontiguousarray(ds[b][rows]), ds[a], k=1)} if len(rows) else {}
        else:
            back = {x[0].queryIdx: x[0].trainIdx for x in bf.knnMatch(ds[b], ds[a], k=1)}
        return np.array([(x.queryIdx, x.trainIdx) for x in good if back.get(x.trainIdx) == x.queryIdx])
    log(f"detected 6 images, frame {frame} ({time.time() - t:.1f}s)")
    stereo = {}
    def stereo_match(v):
        if v not in stereo or cfg.detect_only_sharing:
            stereo[v] = match((v, "L"), (v, "R"), Fe.RATIO_STEREO)
        return stereo[v]
    out = {}
    for a, b in REG_PAIRS:
        t = time.time(); M1, M2, MX = stereo_match(a), stereo_match(b), match((a, "L"), (b, "L"), cfg.ratio_cross)
        out[(a, b)] = dict(A1=kp[(a, "L")][M1[:, 0]], B1=kp[(a, "R")][M1[:, 1]], M1=M1, A2=kp[(b, "L")][M2[:, 0]], B2=kp[(b, "R")][M2[:, 1]], M2=M2, MX=MX)
        log(f"matched V{a}/V{b} frame {frame}: stereo {len(M1)}/{len(M2)}, cross {len(MX)} ({time.time() - t:.1f}s)")
    return out


# ------------------------------------------------------------------ reconstruction pieces (a faithful port of recon.py)
def _vp_feats(match_sets: Mapping, v: int):
    if v == 1: F = match_sets[(1, 2)]; return F["A1"], F["B1"]
    if v == 2: F = match_sets[(1, 2)]; return F["A2"], F["B2"]
    F = match_sets[(3, 2)]; return F["A1"], F["B1"]


def fit_pose(cal, match_sets, v):
    """Independent stereo pose of viewpoint ``v`` (fixed baseline, fixed K/D).  Returns the 5-parameter pose and the final pose diagnostics plus the in-lier residual RMS."""
    A, B = _vp_feats(match_sets, v); Z = np.zeros((0, 2), int)
    D = P.build_dataset(cal, [dict(A1=A, B1=B, A2=A, B2=B, M1=Z, M2=Z, MX=Z)])
    p, info, rec = P.fit_A_guarded(D)
    r = G.sampson_residual(p, D["n1"], D["n2"], cal.baseline_mm)
    fin = dict(rec["final"]); fin["resid_rms_px_inliers"] = float(np.sqrt(np.mean(r[np.abs(r) < 4.685 * info["scale"]] ** 2)))
    fin["yaw_deg"] = G.yaw_deg(p); fin["n_matches"] = int(len(A)); fin["fallback_used"] = bool(rec.get("fallback_used", False))
    return p, fin


def view_floor(cal, p, match_sets, v, y_min):
    """Triangulate viewpoint ``v`` and fit its floor plane (n, d) with n.x + d = 0.  Returns the plane and descriptive statistics."""
    R_, T_ = G.pose_from_params(p, cal.baseline_mm); A, B = _vp_feats(match_sets, v); mt = np.zeros((len(A), 1), int)
    X, a, _ = M.view_points(cal, R_, T_, A, B, mt)
    sel = a[:, 1] > y_min; n, d = G.fit_floor_plane(X, a, y_min=y_min)
    res = X[sel] @ n + d
    return (n, d), dict(n_pts=int(len(X)), n_floor=int(sel.sum()), floor_rms_mm=float(np.sqrt(np.mean(res ** 2))), floor_mad_mm=float(1.4826 * np.median(np.abs(res))),
                        cam_height_mm=float(d), floor_inl20=float(np.mean(np.abs(res) < 20)))


def _coverage(px, size_wh) -> dict:
    """Spatial distribution of registration inliers in the left image: occupied cells of a GRID x GRID grid, and the normalised bounding extent.  Reporting only."""
    w, h = size_wh; px = np.asarray(px, float)
    if len(px) == 0: return dict(cells=0, cells_total=GRID * GRID, cell_fraction=0.0, x_extent=0.0, y_extent=0.0)
    cx = np.clip((px[:, 0] / w * GRID).astype(int), 0, GRID - 1); cy = np.clip((px[:, 1] / h * GRID).astype(int), 0, GRID - 1)
    cells = len(set(zip(cx.tolist(), cy.tolist())))
    return dict(cells=cells, cells_total=GRID * GRID, cell_fraction=cells / (GRID * GRID), x_extent=float(np.ptp(px[:, 0]) / w), y_extent=float(np.ptp(px[:, 1]) / h))


def register(cal, match_sets, a, b, pa, pb, seed):
    """Rigid ``a`` -> ``b`` registration from the (a, b) pair's cross-view matches.  Returns R0, t0 and descriptive statistics (inlier count/pairs, RMS, coverage, 3D spread of the inliers)."""
    F = match_sets[(a, b)]
    Ra, Ta = G.pose_from_params(pa, cal.baseline_mm); Rb, Tb = G.pose_from_params(pb, cal.baseline_mm)
    X1, a1, i1 = M.view_points(cal, Ra, Ta, F["A1"], F["B1"], F["M1"]); X2, a2, i2 = M.view_points(cal, Rb, Tb, F["A2"], F["B2"], F["M2"])
    d1 = {k: i for i, k in enumerate(i1[:, 0])}; d2 = {k: i for i, k in enumerate(i2[:, 0])}
    pr = [(d1[q], d2[w]) for q, w in F["MX"] if q in d1 and w in d2]
    if len(pr) < MIN_REG_PAIRS:
        raise ThreeViewError("REGISTRATION_PAIRS", f"Viewpoints {a} and {b} share only {len(pr)} usable matched points (at least {MIN_REG_PAIRS} are needed); the views may not overlap enough.")
    p_, q_ = map(np.array, zip(*pr)); R0, t0, rms, ni, nt = G.rigid_ransac(X1[p_], X2[q_], np.random.default_rng(seed), M.REG_ITERS, M.REG_THR_MM)
    err = np.linalg.norm(X2[q_] - (X1[p_] @ R0.T + t0), axis=1); inl = err < M.REG_THR_MM
    ev = np.sort(np.linalg.eigvalsh(np.cov(X2[q_][inl].T))) if inl.sum() >= 4 else np.zeros(3)
    spread = dict(sd_mm=[float(np.sqrt(max(x, 0.0))) for x in ev[::-1]], planarity=float(np.sqrt(max(ev[0], 0.0) / max(ev[2], 1e-12))))   # smallest/largest principal SD: ~0 = planar or collinear inliers
    return dict(R0=R0, t0=t0, rms=rms, ni=ni, nt=nt, rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(R0) - 1) / 2, -1, 1)))),
                coverage_left_a=_coverage(a1[p_][inl], cal.image_size), coverage_left_b=_coverage(a2[q_][inl], cal.image_size), spread=spread)


def _h_line(cal, pts, cam):
    K, Dd = (cal.K_left, cal.D_left) if cam == "left" else (cal.K_right, cal.D_right)
    u = G.undistort_pixels(pts, K, Dd); return np.cross(np.r_[u[0], 1], np.r_[u[1], 1])


def corner(cal, p, floor, ann: Annotations, v, frame):
    """Bottom wall corner of viewpoint ``v`` (V1: left corner, V3: right corner) on that viewpoint's floor, and the angle (deg) between the reconstructed vertical edge and the floor normal."""
    R_, T_ = G.pose_from_params(p, cal.baseline_mm)
    lA = _h_line(cal, ann.get(v, frame, "left", "vertical"), "left"); lP = _h_line(cal, ann.get(v, frame, "right", "vertical"), "right")
    B, u = G.corner_line_on_floor(cal, R_, T_, lA, lP, floor)
    return B, float(np.degrees(np.arccos(min(1.0, abs(float(u @ floor[0]))))))


def _angle(a, b) -> float:
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def reconstruct_frame(cal: C.StereoCalibration, match_sets: Mapping, ann: Annotations, frame: int, cfg: ThreeViewConfig = ThreeViewConfig(), *, poses: Mapping | None = None) -> dict:
    """One frame index of Wall B: the length in the V2 frame and every metric a quality gate would use.  ``poses`` (v -> 5-vector) may be supplied to reuse earlier fits (sensitivity work);
    by default each viewpoint's pose is fitted independently from this frame's own matches."""
    check_annotations_complete(ann, [frame])
    y_min = floor_y_min(cal)
    pose_p, pose_d = {}, {}
    for v in VIEWPOINTS:
        if poses is not None and v in poses: pose_p[v], pose_d[v] = poses[v], None
        else: pose_p[v], pose_d[v] = fit_pose(cal, match_sets, v)
    r12 = register(cal, match_sets, 1, 2, pose_p[1], pose_p[2], cfg.seed); r32 = register(cal, match_sets, 3, 2, pose_p[3], pose_p[2], cfg.seed)
    fl, fq = {}, {}
    for v in VIEWPOINTS: fl[v], fq[v] = view_floor(cal, pose_p[v], match_sets, v, y_min)
    B1, ve1 = corner(cal, pose_p[1], fl[1], ann, 1, frame); B3, ve3 = corner(cal, pose_p[3], fl[3], ann, 3, frame)
    B1r = r12["R0"] @ B1 + r12["t0"]; B3r = r32["R0"] @ B3 + r32["t0"]
    L = float(np.linalg.norm(B3r - B1r)); n2 = fl[2][0]
    n1r = r12["R0"] @ fl[1][0]; n3r = r32["R0"] @ fl[3][0]
    d1r = fl[1][1] - n1r @ r12["t0"]; d3r = fl[3][1] - n3r @ r32["t0"]
    if n1r @ n2 < 0: n1r, d1r = -n1r, -d1r
    if n3r @ n2 < 0: n3r, d3r = -n3r, -d3r
    dv = B3r - B1r
    def reg_report(r): return dict(n_inliers=r["ni"], n_pairs=r["nt"], inlier_fraction=r["ni"] / max(r["nt"], 1), rms_mm=r["rms"], rot_deg=r["rot_deg"], coverage_a=r["coverage_left_a"], coverage_b=r["coverage_left_b"], spread=r["spread"])
    return dict(
        frame=int(frame), length_mm=L, corner_v1_in_v2=B1r.tolist(), corner_v3_in_v2=B3r.tolist(),
        metrics=dict(
            pose={v: pose_d[v] for v in VIEWPOINTS}, reg_v1_v2=reg_report(r12), reg_v3_v2=reg_report(r32), floor=fq,
            floor_consistency=dict(ang_v1_v2_deg=_angle(n1r, n2), ang_v3_v2_deg=_angle(n3r, n2), ang_v1_v3_deg=_angle(n1r, n3r), offset_v1_v2_mm=float(d1r - fl[2][1]), offset_v3_v2_mm=float(d3r - fl[2][1])),
            corner=dict(vertical_edge_vs_floor_normal_deg=dict(v1=ve1, v3=ve3), b1_off_v2_floor_mm=float(B1r @ n2 + fl[2][1]), b3_off_v2_floor_mm=float(B3r @ n2 + fl[2][1]),
                        b1_off_v3_floor_mm=float(abs(B1r @ n3r + d3r)), b3_off_v1_floor_mm=float(abs(B3r @ n1r + d1r)), length_vertical_component_mm=float(abs(dv @ n2))),
            baseline_mm=float(cal.baseline_mm), image_size_wh=list(cal.image_size), floor_row_min_px=float(y_min)),
        poses={v: np.asarray(pose_p[v]).tolist() for v in VIEWPOINTS})


def reconstruct(cal: C.StereoCalibration, match_sets_by_frame: Mapping, ann: Annotations, cfg: ThreeViewConfig = ThreeViewConfig()) -> dict:
    """All frames from precomputed match sets, each classified (``three_view_quality``).  A frame that cannot be reconstructed is reported as FAILED, not raised.  The estimate is the mean of
    the usable frames (>= ``thresholds.min_usable_frames``, else INSUFFICIENT_EVIDENCE and no length); warnings are always returned with it.  Input errors (missing annotations) still raise."""
    frames = sorted(match_sets_by_frame); check_annotations_complete(ann, frames); per = []
    for f in frames:
        try:
            r = reconstruct_frame(cal, match_sets_by_frame[f], ann, f, cfg)
            r["quality"] = Q.classify_frame(r, cfg.thresholds)
        except ThreeViewError as e:
            if e.code != "REGISTRATION_PAIRS": raise
            r = Q.failed_frame(f, "REG_PAIRS", e.message)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, IndexError, ZeroDivisionError) as e:
            r = Q.failed_frame(f, "RECONSTRUCTION_FAILED", f"The reconstruction could not be completed ({type(e).__name__}).")
        per.append(r)
    agg = Q.aggregate(per, cfg.thresholds)
    return dict(method="three_viewpoint_wall_b", n_viewpoints=3, frames=frames, per_frame=per, quality=agg, length_mm=agg["length_mm"], length_spread_mm=agg["length_spread_mm"],
                thresholds_note="Provisional engineering checks, not validated accuracy guarantees.")


def measure_from_grays(cal: C.StereoCalibration, gray_of: Callable, keys, ann: Annotations, cfg: ThreeViewConfig = ThreeViewConfig(), log: Callable = lambda *a: None) -> dict:
    """The measurement from frames that are already decoded.  ``gray_of(key, viewpoint, "L"|"R")`` returns the 8-bit gray frame (original size) of frame set ``key``; annotations are looked up
    under ``key``.  The six frames of one frame set are matched, then released, before the next set is requested."""
    keys = sorted(keys); check_annotations_complete(ann, keys)
    if ann.image_wh is not None: check_frame_size(cal, ann.image_wh, "The annotation frame size")
    t0 = time.time(); ms = {}
    for k in keys:
        grays = {(v, s): gray_of(k, v, s) for v in VIEWPOINTS for s in ("L", "R")}
        for g in grays.values(): check_frame_size(cal, (g.shape[1], g.shape[0]), "A frame")
        ms[k] = (compute_match_sets_shared if cfg.uses_fast_matcher() else compute_match_sets)(grays, k, cfg, log)
        del grays
    t1 = time.time(); out = reconstruct(cal, ms, ann, cfg); out["timing_s"] = dict(matching=t1 - t0, reconstruction=time.time() - t1)
    return out


def measure_frame_sets(cal: C.StereoCalibration, videos: Mapping, frame_sets: Mapping, ann: Annotations, cfg: ThreeViewConfig = ThreeViewConfig(), log: Callable = lambda *a: None) -> dict:
    """``videos[(v, "L"|"R")]`` = path of viewpoint v's left (Air) / right (Pro) video.  ``frame_sets[key][(v, "L"|"R")]`` = the frame number to read from that video for frame set ``key``.
    The annotations of a frame set are looked up under ``key`` (the experiment used the frame number itself as the key, the visitor workflow uses the frame-set number).  Read-only."""
    missing = [f"V{v} {s}" for v in VIEWPOINTS for s in ("L", "R") if (v, s) not in videos]
    if missing: raise ThreeViewError("VIDEO_MISSING", "Missing videos: " + ", ".join(missing) + ".")
    for k in sorted(frame_sets):
        lack = [f"V{v} {s}" for v in VIEWPOINTS for s in ("L", "R") if (v, s) not in frame_sets[k]]
        if lack: raise ThreeViewError("FRAME_MISSING", f"Frame set {k}: no frame chosen for " + ", ".join(lack) + ".")
    return measure_from_grays(cal, lambda k, v, s: read_gray_frame(videos[(v, s)], frame_sets[k][(v, s)], cal), frame_sets.keys(), ann, cfg, log)


def measure_from_videos(cal: C.StereoCalibration, videos: Mapping, frames, ann: Annotations, cfg: ThreeViewConfig = ThreeViewConfig(), log: Callable = lambda *a: None) -> dict:
    """The validated experiment's form: frame number ``f`` is read from all six videos and is also the annotation key."""
    return measure_frame_sets(cal, videos, {f: {(v, s): f for v in VIEWPOINTS for s in ("L", "R")} for f in frames}, ann, cfg, log)
