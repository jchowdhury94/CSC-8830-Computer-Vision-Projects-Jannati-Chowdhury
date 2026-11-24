"""Visitor automatic stereo frame-set selection (Phase 2B, Stage C).

Input : four videos -- left/right for placement 1 and left/right for placement 2 (Stage B validates them).
Output: ``n_sets`` frame sets.  Each set is one left+right frame pair in placement 1 and one in placement 2, plus diagnostics.

Nothing here estimates pose or measures anything, and no ground truth is used.  Source videos are only ever read.

Algorithm (all steps bounded, deterministic, no random numbers)
---------------------------------------------------------------
1. Candidate sampling  -- one sequential ``grab``/``retrieve`` pass per video over ``n_candidates`` evenly spaced bin centres.  Only a
   reduced-resolution grayscale preview is kept; full-resolution frames are never cached (see ``extract_selected_frames``).
2. Sharpness           -- variance of the Laplacian of the preview.  Blank frames and frames far blurrier than the video's median are rejected.
3. Features            -- SIFT (capped) on the preview.  Too few keypoints => rejected as low-texture.
4. Stereo pairing      -- left candidate <-> right candidate by ratio-test + mutual SIFT matching, verified by a RANSAC fundamental matrix.
   Only a bounded number of right candidates is tried per left candidate (nearest to the predicted time).
5. Selection           -- left-time bins give temporal coverage; in each bin the pair with the best quality score is a candidate
   (quality = sharpness + features + verified stereo matches, so the sharpest frame is NOT automatically chosen).  Frames are never reused.
6. Cross-placement     -- for each set the top-K pair choices of placement 1 are combined with those of placement 2 and the combination with
   the most verified four-view tracks (AL-PL-AR-PR, built with ``features.build_tracks``) wins.  Too few tracks => failure, not a weak pick.

Synchronisation
---------------
The two phones of a placement were not started together.  Exact timestamp synchronisation is neither required nor assumed.  The pairing is
justified only for a STATIC scene and a rigid stereo rig: then any left frame and any right frame view the same geometry.  The module
measures whether that holds (stereo match counts vs. time offset).  If the counts peak at a consistent offset, right candidates near
``t_left + offset`` are tried; if the counts are flat (matches are equally good at every offset) the offset is reported as
"not identifiable" and pairing falls back to equal fractions of each video's duration.  Nothing here shows that asynchronous frames are
suitable for a moving scene or a moving rig; if the match counts collapse away from one offset, the scene was not static.
Timestamps are nominal (``index / fps``); variable-frame-rate files are not corrected.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from . import features as _features
from . import frames as _frames
from .visitor_calibration import Issue, Severity, worst
from .visitor_video_validation import ValidationResult, validate_visitor_videos

PLACEMENTS = ("placement_1", "placement_2")
SOURCE_NOTE = ("Frames are selected from reduced-resolution previews; the indices refer to 0-based sequential decoding of the full video "
               "(the Phase 1 convention) and the full-resolution frames can be re-extracted with extract_selected_frames().")
SYNC_NOTE = ("Left/right frames are paired by stereo-match evidence, not by timestamp.  This is only defensible for a static scene and a "
             "rigid rig; it says nothing about moving scenes.")


class FrameSelectionError(RuntimeError):
    """A video could not be decoded as required."""


@dataclass(frozen=True)
class SelectionConfig:
    """Generic engineering limits.  None of these were fitted to the original room or to its ground truth; see tests and the Stage C report."""
    n_candidates: int = 16               # per video; raised to 3 * n_sets when that is larger
    preview_long_side: int = 800         # reduced resolution used for all quality evaluation
    n_features: int = 2000               # SIFT cap on previews
    contrast_threshold: float = 0.04     # SIFT, same value the Phase 1 matcher uses
    min_features: int = 150              # below this a frame is "low texture"
    min_gray_std: float = 3.0            # below this a frame is "blank"
    min_relative_sharpness: float = 0.25  # reject when Laplacian variance < this * median of the video's valid candidates
    ransac_px: float = 1.5               # fundamental-matrix inlier threshold on the preview
    min_stereo_verified: int = 40        # verified (RANSAC-inlier) stereo matches needed for a usable pair
    min_stereo_inlier_ratio: float = 0.25
    min_cross_tracks: int = 20           # verified four-view tracks needed between the two placements
    right_per_left: int = 3              # right candidates tried per left candidate
    n_probes: int = 4                    # left probes used to test whether a time offset is identifiable
    offset_contrast: float = 1.5         # best / median probe match count needed to call the offset identifiable
    top_k: int = 3                       # pair choices per set that enter the cross-placement search
    weight_sharpness: float = 0.2
    weight_features: float = 0.2
    weight_stereo: float = 0.6
    max_sets: int = 20


DEFAULT_CONFIG = SelectionConfig()


# ---------------------------------------------------------------- result types
@dataclass
class CandidateFrame:
    video: str
    position: int                 # position in the candidate list
    index: int                    # 0-based sequential frame index in the full video
    timestamp_s: float
    sharpness: float | None = None
    sharpness_percentile: float | None = None
    n_features: int | None = None
    valid: bool = True
    reject_reason: str | None = None

    def to_dict(self):
        return dict(video=self.video, position=self.position, index=self.index, timestamp_s=round(self.timestamp_s, 4),
                    sharpness=None if self.sharpness is None else round(self.sharpness, 3),
                    n_features=self.n_features, valid=self.valid, reject_reason=self.reject_reason)


@dataclass
class StereoPair:
    placement: str
    left: CandidateFrame
    right: CandidateFrame
    raw_matches: int
    verified_matches: int
    inlier_ratio: float
    usable: bool
    quality: float = 0.0

    def to_dict(self):
        return dict(left_frame_index=self.left.index, right_frame_index=self.right.index, left_timestamp_s=round(self.left.timestamp_s, 4),
                    right_timestamp_s=round(self.right.timestamp_s, 4), raw_matches=self.raw_matches, verified_matches=self.verified_matches,
                    inlier_ratio=round(self.inlier_ratio, 4), usable=self.usable, quality=round(self.quality, 4))


@dataclass
class PlacementFrame:
    """One stereo frame pair of one placement, in the form the measurement backend needs."""
    placement: str
    set_index: int
    left_frame_index: int
    right_frame_index: int
    left_timestamp_s: float
    right_timestamp_s: float
    left_sharpness: float
    right_sharpness: float
    left_features: int
    right_features: int
    stereo_raw_matches: int
    stereo_verified_matches: int
    stereo_inlier_ratio: float
    quality_score: float

    @property
    def time_offset_s(self):
        return self.right_timestamp_s - self.left_timestamp_s

    def to_dict(self):
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}
        d["time_offset_s"] = round(self.time_offset_s, 4)
        return d


@dataclass
class CrossPlacementDiagnostics:
    set_index: int
    left_left_raw_matches: int        # placement-1 left <-> placement-2 left
    left_left_verified_matches: int
    four_view_tracks: int
    sufficient: bool
    combination_score: float

    def to_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class FrameSet:
    set_index: int
    placement_1: PlacementFrame
    placement_2: PlacementFrame
    cross: CrossPlacementDiagnostics
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return dict(set_index=self.set_index, placement_1=self.placement_1.to_dict(), placement_2=self.placement_2.to_dict(),
                    cross_placement=self.cross.to_dict(), warnings=list(self.warnings))


@dataclass
class FrameSelectionResult:
    status: str                       # "SUCCESS" or "FAILED"
    requested_sets: int
    sets: list                        # [FrameSet]; EMPTY unless status == "SUCCESS" -- never a partial selection
    issues: list                      # [Issue]
    placements: dict                  # placement -> diagnostics (offset/sync, candidates, pair tables)
    timings_s: dict
    memory: dict
    config: dict
    note: str = SOURCE_NOTE + "  " + SYNC_NOTE

    @property
    def can_proceed(self):
        return self.status == "SUCCESS"

    @property
    def errors(self):
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def indices(self, placement):
        """{'left': [...], 'right': [...]} frame indices of one placement, in set order."""
        key = "placement_1" if placement in (1, "placement_1") else "placement_2"
        return dict(left=[getattr(s, key).left_frame_index for s in self.sets], right=[getattr(s, key).right_frame_index for s in self.sets])

    def to_dict(self):
        return dict(status=self.status, can_proceed=self.can_proceed, requested_sets=self.requested_sets, note=self.note,
                    sets=[s.to_dict() for s in self.sets], errors=[i.to_dict() for i in self.errors], warnings=[i.to_dict() for i in self.warnings],
                    placements=self.placements, timings_s={k: round(v, 3) for k, v in self.timings_s.items()}, memory=self.memory, config=self.config)


# ---------------------------------------------------------------- internals
@dataclass
class _Feat:
    kp: np.ndarray       # (n,2) float32 preview pixels
    ds: np.ndarray       # (n,128) float32


@dataclass
class _Video:
    label: str
    path: str
    frame_count: int
    fps: float
    cands: list = field(default_factory=list)
    feats: dict = field(default_factory=dict)       # position -> _Feat
    decoded_frames: int = 0

    @property
    def duration(self):
        return self.frame_count / self.fps if self.fps else 0.0

    def valid(self):
        return [c for c in self.cands if c.valid]


class _Timers(dict):
    def add(self, key, t0):
        self[key] = self.get(key, 0.0) + (time.perf_counter() - t0)


def candidate_indices(frame_count: int, n_candidates: int):
    """Bin centres: distinct, ascending, away from the start/stop jitter.  Fewer than requested only when the video is shorter."""
    n = max(0, min(int(n_candidates), int(frame_count)))
    idx = [int((i + 0.5) * frame_count / n) for i in range(n)] if n else []
    return sorted(set(min(i, frame_count - 1) for i in idx))


def _preview(gray, long_side):
    s = long_side / max(gray.shape)
    return gray if s >= 1 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def sample_candidate_frames(path, frame_count: int, indices, preview_long_side: int = DEFAULT_CONFIG.preview_long_side):
    """One sequential pass; returns ({index: preview gray}, frames_decoded).  Indices that cannot be decoded are simply absent.
    ``grab`` skips colour conversion for frames that are not candidates.  Raises FrameSelectionError if the file cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FrameSelectionError(f"cannot open {path}")
    want = sorted(set(int(i) for i in indices)); out = {}; pos = 0
    try:
        while want and pos <= want[-1]:
            if not cap.grab():
                break
            if pos in want:
                ok, f = cap.retrieve()
                if ok and f is not None and f.size:
                    out[pos] = _preview(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f, preview_long_side)
            pos += 1
    finally:
        cap.release()
    return out, pos


def sharpness(gray) -> float:
    """Variance of the Laplacian (higher = sharper).  Comparable only between frames of the same resolution."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _detect(sift, gray):
    kp, ds = sift.detectAndCompute(gray, None)
    if ds is None or not len(kp):
        return _Feat(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    return _Feat(np.array([p.pt for p in kp], np.float32), ds.astype(np.float32))


def _ratio_mutual(fa: _Feat, fb: _Feat, ratio):
    """Ratio test + mutual consistency (same rule as features.match_set).  Returns (m,2) int array of keypoint-index pairs."""
    if len(fa.ds) < 2 or len(fb.ds) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher()
    m = bf.knnMatch(fa.ds, fb.ds, k=2); good = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
    if not good:
        return np.zeros((0, 2), int)
    back = {x[0].queryIdx: x[0].trainIdx for x in bf.knnMatch(fb.ds, fa.ds, k=1)}
    return np.array([(x.queryIdx, x.trainIdx) for x in good if back.get(x.trainIdx) == x.queryIdx], int).reshape(-1, 2)


def _verify(fa, fb, M, px, seed=0):
    """RANSAC fundamental-matrix inlier mask for matches M (static scene).  Fewer than 8 matches => all rejected."""
    if len(M) < 8:
        return np.zeros(len(M), bool)
    cv2.setRNGSeed(seed)
    F, mask = cv2.findFundamentalMat(fa.kp[M[:, 0]], fb.kp[M[:, 1]], cv2.FM_RANSAC, px, 0.999)
    return np.zeros(len(M), bool) if mask is None else mask.ravel().astype(bool)


def match_frames(fa: _Feat, fb: _Feat, ratio, px):
    """Returns (verified index pairs, raw match count)."""
    M = _ratio_mutual(fa, fb, ratio); ok = _verify(fa, fb, M, px)
    return M[ok], len(M)


def _issue(sev, code, msg, subject=""):
    return Issue(sev, code, msg, subject)


# ---------------------------------------------------------------- per-video evaluation
def _prepare_video(v: _Video, cfg: SelectionConfig, n_cand: int, timers: _Timers, issues: list):
    """Candidate extraction, sharpness and features for one video.  Fills v.cands / v.feats; previews are dropped before returning."""
    idx = candidate_indices(v.frame_count, n_cand)
    t0 = time.perf_counter()
    previews, v.decoded_frames = sample_candidate_frames(v.path, v.frame_count, idx, cfg.preview_long_side)
    timers.add("candidate_extraction", t0)
    if idx and v.decoded_frames <= idx[-1]:
        issues.append(_issue(Severity.WARNING, "FRAME_COUNT_MISMATCH",
                             f"{v.label}: container reports {v.frame_count} frames but only {v.decoded_frames} decoded; later candidates are missing.", v.label))
    for pos, i in enumerate(idx):
        v.cands.append(CandidateFrame(v.label, pos, i, i / v.fps if v.fps else 0.0))
    t0 = time.perf_counter()
    for c in v.cands:
        g = previews.get(c.index)
        if g is None:
            c.valid, c.reject_reason = False, "missing_frame"
            continue
        c.sharpness = sharpness(g)
        if float(g.std()) < cfg.min_gray_std:
            c.valid, c.reject_reason = False, "blank"
    timers.add("sharpness", t0)
    ok = [c for c in v.cands if c.valid]
    if ok:
        med = float(np.median([c.sharpness for c in ok]))
        order = sorted(c.sharpness for c in ok)
        for c in ok:
            c.sharpness_percentile = (order.index(c.sharpness) / (len(order) - 1)) if len(order) > 1 else 1.0
            if med > 0 and c.sharpness < cfg.min_relative_sharpness * med:
                c.valid, c.reject_reason = False, "blurry"
    t0 = time.perf_counter()
    sift = cv2.SIFT_create(nfeatures=cfg.n_features, contrastThreshold=cfg.contrast_threshold)
    for c in v.cands:
        if not c.valid:
            continue
        f = _detect(sift, previews[c.index]); c.n_features = len(f.kp)
        if len(f.kp) < cfg.min_features:
            c.valid, c.reject_reason = False, "low_texture"
        else:
            v.feats[c.position] = f
    timers.add("features", t0)
    del previews
    if not v.valid():
        issues.append(_issue(Severity.ERROR, "DEGENERATE_CANDIDATES", f"{v.label}: no usable candidate frame ({_reasons(v)}).", v.label))


def _reasons(v):
    r = {}
    for c in v.cands:
        if not c.valid:
            r[c.reject_reason] = r.get(c.reject_reason, 0) + 1
    return ", ".join(f"{n} {k}" for k, n in sorted(r.items())) or "none sampled"


# ---------------------------------------------------------------- stereo pairing for one placement
class _Placement:
    def __init__(self, name, L: _Video, R: _Video, cfg: SelectionConfig, timers: _Timers):
        self.name, self.L, self.R, self.cfg, self.timers = name, L, R, cfg, timers
        self.pairs: dict = {}            # (posL, posR) -> StereoPair
        self.matches: dict = {}          # (posL, posR) -> verified index pairs
        self.sync: dict = {}

    def pair(self, i, j) -> StereoPair:
        key = (i, j)
        if key not in self.pairs:
            t0 = time.perf_counter(); cfg = self.cfg
            M, raw = match_frames(self.L.feats[i], self.R.feats[j], _features.RATIO_STEREO, cfg.ransac_px)
            self.timers.add("stereo_matching", t0)
            ratio = len(M) / raw if raw else 0.0
            self.matches[key] = M
            self.pairs[key] = StereoPair(self.name, self.L.cands[i], self.R.cands[j], raw, len(M), ratio,
                                         len(M) >= cfg.min_stereo_verified and ratio >= cfg.min_stereo_inlier_ratio)
        return self.pairs[key]

    def estimate_offset(self):
        """Probe a few left candidates against every right candidate.  Sets self.offset (seconds, or None) and self.sync diagnostics."""
        cfg = self.cfg; vl, vr = self.L.valid(), self.R.valid(); n = min(cfg.n_probes, len(vl))
        probes = [vl[int((k + 0.5) * len(vl) / n)] for k in range(n)] if n else []
        contrasts, offs = [], []
        for p in probes:
            counts = np.array([self.pair(p.position, r.position).verified_matches for r in vr], float)
            if not len(counts):
                continue
            contrasts.append(float(counts.max() / max(np.median(counts), 1.0)))
            offs.append(vr[int(np.argmax(counts))].timestamp_s - p.timestamp_s)
        spacing = max(self.R.duration / max(len(self.R.cands), 1), 1e-9)
        mad = float(np.median(np.abs(np.array(offs) - np.median(offs)))) if offs else None
        ident = bool(offs) and float(np.median(contrasts)) >= cfg.offset_contrast and mad is not None and mad <= 1.5 * spacing
        self.offset = float(np.median(offs)) if ident else None
        self.sync = dict(probe_best_offsets_s=[round(o, 3) for o in offs], probe_match_contrast=[round(c, 3) for c in contrasts],
                         offset_identifiable=ident, estimated_offset_s=None if self.offset is None else round(self.offset, 3),
                         pairing="time_offset" if ident else "equal_fraction_of_duration",
                         duration_left_s=round(self.L.duration, 3), duration_right_s=round(self.R.duration, 3), fps_left=self.L.fps, fps_right=self.R.fps,
                         interpretation=("match counts peak at a consistent offset" if ident else
                                         "match counts do not depend on the offset: consistent with a static scene and rig; offset not identifiable"))

    def predicted_time(self, tl):
        if self.offset is not None:
            return tl + self.offset
        return tl / self.L.duration * self.R.duration if self.L.duration else tl

    def evaluate_pairs(self):
        """Bounded pairing: each valid left candidate against the right candidates nearest to its predicted time."""
        vr = self.R.valid()
        for l in self.L.valid():
            tp = self.predicted_time(l.timestamp_s)
            for r in sorted(vr, key=lambda r: (abs(r.timestamp_s - tp), r.position))[: self.cfg.right_per_left]:
                self.pair(l.position, r.position)
        usable = [p for p in self.pairs.values() if p.usable]
        if usable:                                                # score: sharpness + features + stereo evidence, all relative within this placement
            cfg = self.cfg; mx = max(p.verified_matches for p in usable); wsum = cfg.weight_sharpness + cfg.weight_features + cfg.weight_stereo
            for p in usable:
                sh = min(p.left.sharpness_percentile, p.right.sharpness_percentile); ft = min(p.left.n_features, p.right.n_features) / cfg.n_features
                p.quality = (cfg.weight_sharpness * sh + cfg.weight_features * ft + cfg.weight_stereo * p.verified_matches / mx) / wsum

    def slot_pool(self, slot, n_sets):
        """Usable pairs whose left candidate lies in the temporal bin ``slot``, best quality first."""
        n = len(self.L.cands)
        sel = [p for p in self.pairs.values() if p.usable and int(p.left.position * n_sets / n) == slot]
        return sorted(sel, key=lambda p: (-p.quality, p.left.index, p.right.index))

    def relaxed_pool(self, used_l, used_r, min_sep):
        sel = [p for p in self.pairs.values() if p.usable and p.left.position not in used_l and p.right.position not in used_r
               and all(abs(p.left.position - u) >= min_sep for u in used_l)]
        return sorted(sel, key=lambda p: (-p.quality, p.left.index, p.right.index))


def _cross(P1: _Placement, p1: StereoPair, P2: _Placement, p2: StereoPair, cfg, timers, cache):
    """Verified left-left matches between the placements and the number of four-view tracks they close."""
    key = (p1.left.position, p1.right.position, p2.left.position, p2.right.position)
    if key in cache:
        return cache[key]
    t0 = time.perf_counter()
    fa, fb = P1.L.feats[p1.left.position], P2.L.feats[p2.left.position]
    MX_raw = _ratio_mutual(fa, fb, _features.RATIO_CROSS); ok = _verify(fa, fb, MX_raw, cfg.ransac_px); MX = MX_raw[ok]
    M1, M2 = P1.matches[(p1.left.position, p1.right.position)], P2.matches[(p2.left.position, p2.right.position)]
    f1b, f2b = P1.R.feats[p1.right.position], P2.R.feats[p2.right.position]
    F = dict(A1=fa.kp[M1[:, 0]], B1=f1b.kp[M1[:, 1]], M1=M1, A2=fb.kp[M2[:, 0]], B2=f2b.kp[M2[:, 1]], M2=M2, MX=MX)
    tracks = len(_features.build_tracks(F)) if len(M1) and len(M2) and len(MX) else 0
    timers.add("cross_matching", t0)
    cache[key] = (len(MX_raw), len(MX), tracks)
    return cache[key]


# ---------------------------------------------------------------- orchestration
def _fail(requested, issues, placements, timers, t_start, cfg, mem0):
    timers["total"] = time.perf_counter() - t_start
    return FrameSelectionResult("FAILED", requested, [], issues, placements, dict(timers), _memory(mem0), _cfg_dict(cfg))


def _cfg_dict(cfg):
    return {k: v for k, v in cfg.__dict__.items()}


def _peak_rss_mb():
    try:
        import resource, sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024
    except Exception:                                    # pragma: no cover - platform without resource
        return None


def _memory(mem0):
    now = _peak_rss_mb()
    return dict(peak_rss_mb=None if now is None else round(now, 1), peak_rss_growth_mb=None if now is None or mem0 is None else round(now - mem0, 1),
                note="process peak RSS (monotonic); growth is 0 when an earlier call already reached a higher peak.")


def _placement_diag(P: _Placement):
    return dict(sync=P.sync, left_candidates=[c.to_dict() for c in P.L.cands], right_candidates=[c.to_dict() for c in P.R.cands],
                evaluated_pairs=[p.to_dict() for p in sorted(P.pairs.values(), key=lambda p: (p.left.index, p.right.index))],
                n_valid_left=len(P.L.valid()), n_valid_right=len(P.R.valid()), n_usable_pairs=sum(p.usable for p in P.pairs.values()))


def _frame(p: StereoPair, set_index):
    return PlacementFrame(p.placement, set_index, p.left.index, p.right.index, p.left.timestamp_s, p.right.timestamp_s, p.left.sharpness,
                          p.right.sharpness, p.left.n_features, p.right.n_features, p.raw_matches, p.verified_matches, p.inlier_ratio, p.quality)


def select_frame_sets(left_video_1, right_video_1, left_video_2, right_video_2, n_sets: int = 3, config: SelectionConfig | None = None,
                      validation: ValidationResult | None = None) -> FrameSelectionResult:
    """Select ``n_sets`` stereo frame sets from four videos.  Never raises for bad input: returns status FAILED with Issues.

    ``validation`` is the Stage B result; when omitted Stage B is run here.  A FAILED result carries no frame sets."""
    cfg = config or DEFAULT_CONFIG; t_start = time.perf_counter(); timers = _Timers(); issues: list = []; placements = {}; mem0 = _peak_rss_mb()
    paths = dict(left_video_1=left_video_1, right_video_1=right_video_1, left_video_2=left_video_2, right_video_2=right_video_2)
    if not isinstance(n_sets, int) or isinstance(n_sets, bool) or not 1 <= n_sets <= cfg.max_sets:
        issues.append(_issue(Severity.ERROR, "INVALID_FRAME_SET_COUNT", f"n_sets must be an integer from 1 to {cfg.max_sets}, got {n_sets!r}.", "n_sets"))
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)
    if n_sets == 1:
        issues.append(_issue(Severity.WARNING, "SINGLE_FRAME_SET", "One frame set gives no temporal diversity.", "n_sets"))

    t0 = time.perf_counter()
    val = validation or validate_visitor_videos(*paths.values())
    timers.add("validation", t0)
    if not val.can_proceed:
        issues += [_issue(Severity.ERROR, "VIDEO_NOT_PROCESSABLE", f"{i.subject}: {i.message}", i.subject) for i in val.errors] or \
                  [_issue(Severity.ERROR, "VIDEO_NOT_PROCESSABLE", "video validation did not pass", "videos")]
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)

    n_cand = max(cfg.n_candidates, 3 * n_sets)
    vids = {}
    for label, key in zip(("left_video_1", "right_video_1", "left_video_2", "right_video_2"), paths):
        r = val.videos[label]
        fc, fps = r.frame_count, r.fps
        if not (isinstance(fc, (int, np.integer)) and not isinstance(fc, bool) and fc >= 1) or not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
            issues.append(_issue(Severity.ERROR, "INVALID_VIDEO_METADATA", f"{label}: frame count {fc!r} / fps {fps!r} are not usable for frame selection.", label))
            continue
        vids[label] = _Video(label, str(paths[key]), int(fc), float(fps))
    if len(vids) < 4:
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)
    try:
        for v in vids.values():
            _prepare_video(v, cfg, n_cand, timers, issues)
    except FrameSelectionError as e:
        issues.append(_issue(Severity.ERROR, "VIDEO_NOT_PROCESSABLE", str(e), "videos"))
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)
    short = [v.label for v in vids.values() if len(v.valid()) < n_sets]
    if any(i.code == "DEGENERATE_CANDIDATES" for i in issues):
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)
    if short:
        for lb in short:
            v = vids[lb]
            issues.append(_issue(Severity.ERROR, "TOO_FEW_CANDIDATES",
                                 f"{lb}: {len(v.valid())} usable candidate frame(s) ({len(v.cands)} sampled; rejected: {_reasons(v)}) but {n_sets} distinct frames are needed.", lb))
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)

    P = [_Placement(PLACEMENTS[0], vids["left_video_1"], vids["right_video_1"], cfg, timers),
         _Placement(PLACEMENTS[1], vids["left_video_2"], vids["right_video_2"], cfg, timers)]
    for p in P:
        p.estimate_offset(); p.evaluate_pairs()
        placements[p.name] = _placement_diag(p)
        if not any(x.usable for x in p.pairs.values()):
            best = max((x.verified_matches for x in p.pairs.values()), default=0)
            issues.append(_issue(Severity.ERROR, "STEREO_INSUFFICIENT",
                                 f"{p.name}: no left/right pair reached {cfg.min_stereo_verified} verified matches with inlier ratio >= {cfg.min_stereo_inlier_ratio} "
                                 f"(best verified count {best}); the two videos may not show the same scene.", p.name))
    if any(i.code == "STEREO_INSUFFICIENT" for i in issues):
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)

    # ---- assemble sets slot by slot; frames are never reused within a video
    used = [dict(l=set(), r=set()) for _ in P]; cache = {}; sets = []; sep = max(1, len(P[0].L.cands) // (2 * n_sets))
    for s in range(n_sets):
        best = None; relaxed = False; max_cross = 0
        for attempt in ("bin", "relaxed"):
            pools = []
            for k, p in enumerate(P):
                if attempt == "bin":
                    pool = [x for x in p.slot_pool(s, n_sets) if x.left.position not in used[k]["l"] and x.right.position not in used[k]["r"]]
                else:
                    pool = p.relaxed_pool(used[k]["l"], used[k]["r"], sep)
                pools.append(pool[: cfg.top_k])
            combos = []
            for a in pools[0]:
                for b in pools[1]:
                    nm, nv, tr = _cross(P[0], a, P[1], b, cfg, timers, cache)
                    combos.append((a, b, nm, nv, tr))
            max_cross = max([c[4] for c in combos], default=0)
            good = [c for c in combos if c[4] >= cfg.min_cross_tracks]
            if good:
                mt = max(c[4] for c in good); max_cross = mt
                best = max(good, key=lambda c: (0.5 * (c[0].quality + c[1].quality) / 2 + 0.5 * c[4] / mt, -c[0].left.index, -c[1].left.index))
                relaxed = attempt == "relaxed"
                break
        if best is None:
            sufficient_pairs = all(len(p.relaxed_pool(used[k]["l"], used[k]["r"], sep)) for k, p in enumerate(P))
            code = "CROSS_PLACEMENT_INSUFFICIENT" if sufficient_pairs else "SETS_NOT_FORMABLE"
            msg = (f"set {s}: best four-view track count between the placements was {max_cross} (< {cfg.min_cross_tracks}); the two views may not overlap enough."
                   if sufficient_pairs else f"set {s}: not enough distinct usable stereo pairs remain to form {n_sets} frame sets without reusing frames.")
            issues.append(_issue(Severity.ERROR, code, msg, f"set_{s}"))
            return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)
        a, b, nm, nv, tr = best
        for k, x in enumerate((a, b)):
            used[k]["l"].add(x.left.position); used[k]["r"].add(x.right.position)
        warn = ["temporal bin had no workable pair; frames taken from another part of the recording"] if relaxed else []
        if relaxed:
            issues.append(_issue(Severity.WARNING, "COVERAGE_RELAXED", f"set {s}: {warn[0]}.", f"set_{s}"))
        sets.append(FrameSet(s, _frame(a, s), _frame(b, s), CrossPlacementDiagnostics(s, nm, nv, tr, True, 0.5 * (a.quality + b.quality) / 2 + 0.5 * tr / max(max_cross, 1)), warn))
    sets.sort(key=lambda fs: fs.placement_1.left_timestamp_s)       # time order of placement 1; set_index renumbered below
    for i, fs in enumerate(sets):
        fs.set_index = i; fs.placement_1.set_index = i; fs.placement_2.set_index = i; fs.cross.set_index = i
    for p in P:
        if not p.sync.get("offset_identifiable"):
            issues.append(_issue(Severity.WARNING, "OFFSET_NOT_IDENTIFIABLE",
                                 f"{p.name}: stereo match counts do not reveal a time offset; pairing assumes a static scene and rig.", p.name))
    for k, name in enumerate(PLACEMENTS):
        if len({fs.__dict__[("placement_1", "placement_2")[k]].left_frame_index for fs in sets}) < len(sets):   # defensive; cannot happen
            issues.append(_issue(Severity.ERROR, "DUPLICATE_FRAMES", f"{name}: duplicate frame indices.", name))
    if any(i.code == "DUPLICATE_FRAMES" for i in issues):
        return _fail(n_sets, issues, placements, timers, t_start, cfg, mem0)
    issues.append(_issue(Severity.WARNING, "SETS_ARE_CORRELATED",
                         "Frames come from one continuous recording per camera; sets are not independent captures.", "selection"))
    timers["total"] = time.perf_counter() - t_start
    return FrameSelectionResult("SUCCESS", n_sets, sets, issues, placements, dict(timers), _memory(mem0), _cfg_dict(cfg))


def extract_selected_frames(result: FrameSelectionResult, left_video_1, right_video_1, left_video_2, right_video_2):
    """Re-extract the selected FULL-resolution grayscale frames (Phase 1 convention: sequential decode, BGR2GRAY), decoding each video once.
    Returns [{'placement_1': {'left','right'}, 'placement_2': {...}} per set].  Raises ValueError for a FAILED result."""
    if not result.can_proceed:
        raise ValueError("no frames were selected (result status is FAILED)")
    ix = lambda key, side: [getattr(s, key).__dict__[f"{side}_frame_index"] for s in result.sets]
    got = {(k, side): _frames.extract_frames(path, ix(k, side))
           for k, side, path in (("placement_1", "left", left_video_1), ("placement_1", "right", right_video_1),
                                 ("placement_2", "left", left_video_2), ("placement_2", "right", right_video_2))}
    return [{k: dict(left=got[(k, "left")][getattr(s, k).left_frame_index], right=got[(k, "right")][getattr(s, k).right_frame_index])
             for k in PLACEMENTS} for s in result.sets]
