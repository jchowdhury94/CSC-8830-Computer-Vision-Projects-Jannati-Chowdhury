"""Stage F2 (Module 7 Part 3, Phase 2B): AUTOMATIC room-boundary candidate detection -- isolated prototype.

Finds, in Stage C selected frames, candidate lines for the six boundaries the unchanged Phase 1 reconstruction needs
(top_variant=0):  placement_1: AL_corner PL_corner PL_ceilF ;  placement_2: AR_corner PR_corner PR_ceilF.

Method (classical, CPU, numpy + OpenCV only -- no model weights, no new dependency)
-----------------------------------------------------------------------------------
1. Work at reduced scale (``SCALE``) on the lens-undistorted frame (the same ``cv2.undistort`` Stage D uses).  Invalid (black) strips
   left by undistortion are filled by row-wise replication and excluded, so they cannot create edges.
2. Response = gradient of the lightly blurred image minus gradient of a heavily blurred image: a band-pass step detector that removes the
   smooth wall shading that otherwise dominates (wall corners here are 8-10 gray-level steps).  Clipped, so a strong edge cannot outweigh
   a long faint one.
3. "Sheared Kadane" line search: for a family of near-vertical (corner) or near-horizontal (ceiling) slopes the response is sheared so the
   lines become axis-parallel, and for every column/row the maximum-sum contiguous segment of (response - tau) is found.  This is a
   gradient-weighted Hough transform that returns SEGMENTS: the visible part of a partly hidden edge is a first-class result, gaps are
   tolerated, and an occluded stretch is never bridged silently (the segment ends there).
4. Confuser analysis on the cross-line profile of every candidate (mean response across the segment):
     * BAND   -- an opposite-polarity edge of comparable strength a few pixels away: a thin bright/dark strip (door casing, baseboard, trim,
                 a leg, a shadow) rather than a wall-plane step;
     * SHOULDER -- a much stronger edge very close by: the weak candidate may be a shadow/shoulder of it or may be real; it is flagged,
                 not trusted;
     * ISOLATED -- neither: a single step between two surfaces, as a wall corner is.
5. Joint wall-ceiling junction: a corner and a ceilF candidate are scored together (PL/PR only; the Air camera cannot see the ceiling).
   The ceilF edge must lie on the protocol side of the corner (placement 1: to the right of the left corner; placement 2: to the left of
   the right corner) and must meet the corner line close to the corner's top end.
6. Status (descriptive, not a probability):
     DETECTED     enough support and length, ISOLATED, clearly ahead of the runner-up, junction consistent (when a ceiling is visible)
     UNCERTAIN    a plausible candidate exists but any of those conditions fails; several candidates are returned
     NOT_DETECTED no candidate with enough support/length.

Coordinates: candidate lines are UNDISTORTED FULL-RESOLUTION pixels (``coordinate_space='undistorted_full'``), a space Stage D accepts
directly.  Nothing here reads ground truth, annotations, line equations or dimensions of the original room.
Protocol priors used (documented, room-independent): placement 1 shows the LEFT corner, placement 2 the RIGHT corner of the same wall;
ceilF extends from the apex toward the interior of the wall that carries the corner on its far side (right for placement 1, left for 2).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Mapping

import cv2
import numpy as np

from . import visitor_annotations as V

SCHEMA = "csc8830.module7.stage_f2_boundary_detection.v1"
SCALE = 0.25
MAX_GAIN = 3.0                     # auto-contrast gain is bounded: a frame with almost no dynamic range must not have its noise/shading stretched without limit
NORM_RANGE = 200.0                 # the 2nd-98th percentile range of every frame is stretched to this many gray levels (auto-contrast)
STATUSES = ("DETECTED", "UNCERTAIN", "NOT_DETECTED")
# (view key, boundary type) required for top_variant=0, in Stage D naming
REQUIRED_BOUNDARIES = tuple((vk, bt) for vk, bt in (("AL", "corner"), ("PL", "corner"), ("PL", "ceilF"), ("AR", "corner"), ("PR", "corner"), ("PR", "ceilF")))
PLACEMENT_OF = {"AL": "placement_1", "PL": "placement_1", "AR": "placement_2", "PR": "placement_2"}


@dataclass(frozen=True)
class DetectorLimits:
    """A-priori thresholds from generic image-processing considerations; none was tuned on dimensions or on a reference line."""
    blur_fine: float = 1.5                     # sigma at SCALE
    blur_coarse: float = 10.0
    cap: float = 2.5                           # response clip (gray levels / px at SCALE)
    tau: float = 0.8                           # a row must exceed this to extend a segment
    min_length_px_full: float = 500.0          # visible extent for DETECTED (full-resolution px)
    min_length_px_candidate: float = 250.0     # shorter segments are not even candidates
    slope_v: float = 0.10                      # |dx/dy| of near-vertical corner lines
    step_v: float = 0.01
    slope_h: float = 0.80                      # |dy/dx| of ceiling lines (the ceiling/side-wall edge is oblique)
    step_h: float = 0.03
    pool: int = 40                             # candidates kept per boundary after NMS (before ranking); top ``n_return`` are reported
    n_return: int = 8
    nms_px: float = 8.0
    nms_slope: float = 0.06
    margin_px: int = 14                        # response zeroed this close to the image border (SCALE px)
    corner_rows: tuple = (0.08, 0.78)          # rows searched for corner segments (fraction of height)
    ceil_rows: float = 0.35                    # rows searched for ceiling lines
    own_halfwidth: int = 2                     # SCALE px: own edge position tolerance across the line
    partner_window: tuple = (3, 22)            # SCALE px
    band_ratio: tuple = (0.6, 2.5)             # partner/own strength within this range -> BAND (isolated steps show a band-pass side lobe of ~0.3-0.45)
    shoulder_ratio: float = 2.5                # partner/own above this -> SHOULDER
    band_penalty: float = 0.2
    shoulder_penalty: float = 0.5
    prior_penalty: float = 0.3                 # candidate on the wrong side of the frame for the placement
    corner_prior: float = 0.55                 # placement 1: x <= 0.55 W ; placement 2: x >= 0.45 W
    min_support: float = 0.60                  # fraction of rows of the segment above tau
    margin_ratio: float = 1.25                 # best / runner-up (distinct line) score for DETECTED
    junction_extrap_ratio: float = 2.0         # corner top end -> corner/ceilF intersection, in corner-segment lengths
    junction_penalty: float = 0.3
    max_relative_step: float = 0.09            # photometric prior: a step between two same-paint wall planes is a small fraction of the frame's dynamic range (chosen after
                                               # inspecting the ORIGINAL-ROOM frames: wall corners 0.03-0.06, casing/trim 0.12-0.26; not validated on other rooms)
    contrast_penalty: float = 0.25             # (accent-wall rooms violate this prior; such edges are demoted, never silently trusted)
    step_gain: float = 3.76                    # peak of the band-pass response / step height for a step blurred by blur_fine (1 / (sigma*sqrt(2*pi)))
    vertex_tol_px_full: float = 60.0           # ceiling edges left and right of a corner must intersect this close to the corner line
    vertex_min_angle_deg: float = 8.0          # ... with at least this change of direction (a real wall-wall-ceiling vertex)
    vertex_penalty: float = 0.35
    vertex_end_tol_px_full: float = 160.0      # each ceiling segment must end/start within this of the vertex ...
    vertex_min_seg_px_full: float = 150.0      # ... and extend at least this far away from it
    vertex_min_rel_score: float = 0.08         # vertex members must be at least this fraction of the best ceiling-edge score (no vertex from weak spurious lines)
    ceil_end_gap_px_full: float = 300.0        # ceilF end nearest the corner must be this close (along x) to the corner line


DEFAULT_LIMITS = DetectorLimits()


# ------------------------------------------------------------------ image preparation / response
@dataclass
class Prepared:
    view_key: str
    size_wh: tuple
    u: np.ndarray            # undistorted full-resolution gray (uint8)
    s: np.ndarray            # SCALE image, float32, gap-filled
    valid: np.ndarray        # SCALE-resolution validity (eroded)
    scale: float
    resp_v: np.ndarray | None = None
    resp_h: np.ndarray | None = None
    runtime_s: float = 0.0
    contrast_gain: float = 1.0
    raw_range: float = 0.0


def prepare(gray, K, D, view_key, limits: DetectorLimits = DEFAULT_LIMITS) -> Prepared:
    t0 = time.perf_counter(); g = np.asarray(gray)
    if g.ndim == 3: g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    K = np.asarray(K, float); D = np.asarray(D, float)
    u = cv2.undistort(g, K, D, None, K); v = cv2.undistort(np.full(g.shape, 255, np.uint8), K, D, None, K)
    s = cv2.resize(u, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)
    vm = cv2.resize(v, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) > 250
    vm = cv2.erode(vm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    for r in range(s.shape[0]):                                   # fill invalid strips by row-wise edge replication
        idx = np.nonzero(vm[r])[0]
        if len(idx): s[r, :idx[0]] = s[r, idx[0]]; s[r, idx[-1] + 1:] = s[r, idx[-1]]
    lo, hi = np.percentile(s[vm], [2, 98]); gain = float(np.clip(NORM_RANGE / max(float(hi - lo), 20.0), 0.5, MAX_GAIN)); s = ((s - lo) * gain).astype(np.float32)          # auto-contrast: thresholds are in normalised gray levels
    p = Prepared(view_key=view_key, size_wh=(g.shape[1], g.shape[0]), u=u, s=s, valid=vm, scale=SCALE, contrast_gain=float(gain), raw_range=float(hi - lo))
    p.runtime_s = time.perf_counter() - t0; return p


def _response(p: Prepared, axis: int, limits):
    """Band-pass signed step response (positive = intensity increases along +x for axis 0, along +y for axis 1)."""
    a = cv2.GaussianBlur(p.s, (0, 0), limits.blur_fine); b = cv2.GaussianBlur(p.s, (0, 0), limits.blur_coarse); dx, dy = (1, 0) if axis == 0 else (0, 1)
    r = cv2.Sobel(a, cv2.CV_32F, dx, dy, ksize=3) / 8 - cv2.Sobel(b, cv2.CV_32F, dx, dy, ksize=3) / 8
    r[~p.valid] = 0; m = limits.margin_px; r[:, :m] = 0; r[:, -m:] = 0; r[:m] = 0; r[-m:] = 0
    return r


def _kadane(G, tau):
    """Per column of G (rows x cols): max-sum contiguous segment of (G - tau).  -> score, start, end."""
    R, Cn = G.shape; best = np.zeros(Cn, np.float32); cur = np.zeros(Cn, np.float32); cs = np.zeros(Cn, np.int32); bs = np.zeros(Cn, np.int32); be = np.zeros(Cn, np.int32)
    for r in range(R):
        x = G[r] - tau; reset = cur <= 0; cs = np.where(reset, r, cs); cur = np.where(reset, 0, cur) + x
        upd = cur > best; best = np.where(upd, cur, best); bs = np.where(upd, cs, bs); be = np.where(upd, r, be)
    return best, bs, be


def _shear(G, slope, kind):
    H, W = G.shape
    if kind == "v": return cv2.warpAffine(G, np.float32([[1, -slope, slope * H / 2], [0, 1, 0]]), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return cv2.warpAffine(G, np.float32([[1, 0, 0], [-slope, 1, slope * W / 2]]), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)


# ------------------------------------------------------------------ candidates
@dataclass
class Candidate:
    kind: str                       # 'v' corner-like / 'h' ceiling-like
    sign: int                       # gradient polarity along +x ('v') or +y ('h')
    pos: float                      # offset of the line at the image centre row/col (SCALE px)
    slope: float
    seg: tuple                      # (start, end) along the line, SCALE px
    raw_score: float
    # features (filled by ``analyse``)
    length_px_full: float = 0.0
    support: float = 0.0
    own_strength: float = 0.0
    partner_strength: float = 0.0
    partner_ratio: float = 0.0
    partner_offset_px_full: float = 0.0
    relative_step: float = 0.0
    confuser: str = "ISOLATED"      # ISOLATED | BAND | SHOULDER
    rank_score: float = 0.0
    notes: list = field(default_factory=list)
    flags: dict = field(default_factory=dict)

    def line_small(self, H, W):
        """(x or y at coordinate u) as dict a,b in SCALE px: v: x = a*y + b ; h: y = a*x + b."""
        c = H / 2 if self.kind == "v" else W / 2
        return dict(kind=self.kind, a=self.slope, b=self.pos - self.slope * c)

    def full_line(self, H, W, scale=SCALE):
        """Line in undistorted full-resolution px, in the dict form used by Stage D (kind, a, b, span)."""
        l = self.line_small(H, W); a = l["a"]; b = l["b"] / scale
        return dict(kind=self.kind, a=a, b=b, span=(self.seg[0] / scale, self.seg[1] / scale))


def _search(resp, kind, sign, slopes, rows, tau, H, W):
    out = []
    z = np.clip(sign * resp, -2.5, 2.5)
    for t in slopes:
        S = _shear(z, float(t), kind)
        if kind == "v": sc, a, b = _kadane(S[rows[0]:rows[1]], tau); off = rows[0]
        else: sc, a, b = _kadane(np.ascontiguousarray(S[:rows[1]].T), tau); off = 0
        for x in range(len(sc)):
            if sc[x] > 0: out.append((float(sc[x]), float(t), x, int(a[x]) + off, int(b[x]) + off))
    return out


def _nms(items, limits, kind, H, W):
    items.sort(key=lambda c: -c.raw_score); kept = []
    for c in items:
        dup = False
        for d in kept:
            if c.sign != d.sign or abs(c.slope - d.slope) >= limits.nms_slope: continue
            m = 0.5 * ((c.seg[0] + c.seg[1]) / 2 + (d.seg[0] + d.seg[1]) / 2); cen = H / 2 if kind == "v" else W / 2
            if abs((c.pos + c.slope * (m - cen)) - (d.pos + d.slope * (m - cen))) < limits.nms_px: dup = True; break
        if not dup: kept.append(c)
        if len(kept) >= limits.pool: break
    return kept


def generate_candidates(p: Prepared, boundary_type, limits: DetectorLimits = DEFAULT_LIMITS):
    """Raw candidate pool for 'corner' (near-vertical) or 'ceilF' (near-horizontal) before ranking."""
    H, W = p.s.shape; items = []
    if boundary_type == "corner":
        if p.resp_v is None: p.resp_v = _response(p, 0, limits)
        rows = (int(limits.corner_rows[0] * H), int(limits.corner_rows[1] * H)); slopes = np.arange(-limits.slope_v, limits.slope_v + 1e-9, limits.step_v)
        for sign in (1, -1):
            for sc, t, x, a, b in _search(p.resp_v, "v", sign, slopes, rows, limits.tau, H, W):
                if (b - a) * 4 >= limits.min_length_px_candidate: items.append(Candidate("v", sign, float(x), t, (float(a), float(b)), sc))
        return _nms(items, limits, "v", H, W)
    if p.resp_h is None: p.resp_h = _response(p, 1, limits)
    slopes = np.arange(-limits.slope_h, limits.slope_h + 1e-9, limits.step_h); rows = (0, int(limits.ceil_rows * H))
    for sign in (1, -1):
        for sc, t, y, a, b in _search(p.resp_h, "h", sign, slopes, rows, limits.tau, H, W):
            if (b - a) * 4 >= limits.min_length_px_candidate: items.append(Candidate("h", sign, float(y), t, (float(a), float(b)), sc))
    return _nms(items, limits, "h", H, W)


def _sample(resp, kind, c: Candidate, offsets, H, W, step=2.0):
    """Cross profile samples: rows = positions along the segment, cols = offsets (SCALE px)."""
    u = np.arange(c.seg[0], c.seg[1] + 1e-6, step, dtype=np.float32); cen = H / 2 if kind == "v" else W / 2; w = (c.pos + c.slope * (u - cen)).astype(np.float32)[:, None] + np.asarray(offsets, np.float32)[None, :]
    uu = np.repeat(u[:, None], len(offsets), 1); mx, my = (w, uu) if kind == "v" else (uu, w)
    return cv2.remap(resp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def analyse(p: Prepared, c: Candidate, limits: DetectorLimits = DEFAULT_LIMITS, side_prior=None):
    """Fill the evidence features of one candidate and its rank score."""
    H, W = p.s.shape; kind = c.kind; resp = p.resp_v if kind == "v" else p.resp_h; lo, hi = limits.partner_window
    offs = np.arange(-hi - 2, hi + 3); S = _sample(resp, kind, c, offs, H, W) * c.sign; m = S.mean(0)
    own_mask = np.abs(offs) <= limits.own_halfwidth; c.own_strength = float(m[own_mask].max())
    per_row_own = S[:, own_mask].max(1); c.support = float((per_row_own >= limits.tau).mean())
    part_mask = (np.abs(offs) >= lo) & (np.abs(offs) <= hi); opp = -m
    j = int(np.argmax(np.where(part_mask, opp, -1e9))); c.partner_strength = float(max(opp[j], 0.0)); c.partner_offset_px_full = float(offs[j] / SCALE)
    c.partner_ratio = c.partner_strength / max(c.own_strength, 1e-6); c.length_px_full = float((c.seg[1] - c.seg[0]) / SCALE)
    if c.own_strength <= 0.2: c.confuser = "ISOLATED"; c.notes.append("own edge strength negligible")
    elif limits.band_ratio[0] <= c.partner_ratio <= limits.band_ratio[1]: c.confuser = "BAND"; c.notes.append(f"opposite-polarity edge {c.partner_ratio:.2f}x as strong at {c.partner_offset_px_full:+.0f} px: thin band (casing/baseboard/trim/shadow)")
    elif c.partner_ratio > limits.shoulder_ratio: c.confuser = "SHOULDER"; c.notes.append(f"a {c.partner_ratio:.1f}x stronger opposite edge {c.partner_offset_px_full:+.0f} px away: weak edge beside a dominant one")
    else: c.confuser = "ISOLATED"
    f = 1.0 if c.confuser == "ISOLATED" else (limits.band_penalty if c.confuser == "BAND" else limits.shoulder_penalty)
    prior = 1.0
    if side_prior is not None and kind == "v":
        xm = (c.pos + c.slope * (((c.seg[0] + c.seg[1]) / 2) - H / 2)) / W
        ok = xm <= limits.corner_prior if side_prior == "left" else xm >= 1 - limits.corner_prior
        prior = 1.0 if ok else limits.prior_penalty
        if not ok: c.notes.append(f"corner at {xm:.2f} W is on the wrong side of the frame for this placement")
    # photometric plausibility: relative step height of the edge (mean brightness of the strips either side of the line)
    c.relative_step = float(limits.step_gain * c.own_strength / NORM_RANGE)           # step height as a fraction of the frame's (normalised) dynamic range
    cf = 1.0
    if c.relative_step > limits.max_relative_step:
        cf = limits.contrast_penalty; c.notes.append(f"high-contrast edge (step ~{c.relative_step:.0%} of the dynamic range): more typical of trim/furniture/door than of a wall-wall step")
    c.flags.update(prior_factor=prior, confuser_factor=f, contrast_factor=cf, relative_step=c.relative_step)
    c.rank_score = c.raw_score * f * prior * cf * min(1.0, c.support / limits.min_support)
    return c


# ------------------------------------------------------------------ geometry between candidates
def _line_pt(c: Candidate, u, H, W):
    cen = H / 2 if c.kind == "v" else W / 2; w = c.pos + c.slope * (u - cen); return (w, u) if c.kind == "v" else (u, w)


def _intersect(c1: Candidate, c2: Candidate, H, W):
    """Intersection (SCALE px, (x, y)) of a near-vertical and a near-horizontal candidate: x = a1*y + b1, y = a2*x + b2."""
    l1 = c1.line_small(H, W); l2 = c2.line_small(H, W); den = 1 - l1["a"] * l2["a"]
    if abs(den) < 1e-6: return None
    y = (l2["a"] * l1["b"] + l2["b"]) / den; x = l1["a"] * y + l1["b"]; return float(x), float(y)


def junction(corner: Candidate, ceil: Candidate, protocol_side, H, W, limits: DetectorLimits = DEFAULT_LIMITS):
    """Wall-ceiling junction consistency of a (corner, ceilF) pair.  -> (factor in (0,1], diagnostics dict)."""
    d = dict(factor=1.0, issues=[]); X = _intersect(corner, ceil, H, W)
    if X is None: d.update(factor=limits.junction_penalty); d["issues"].append("corner and ceiling lines are parallel"); return d["factor"], d
    x, y = X; d["intersection_px_full"] = [x / SCALE, y / SCALE]; top = min(corner.seg); length = abs(corner.seg[1] - corner.seg[0])
    ext = max(0.0, top - y) / max(length, 1e-6); d["corner_top_to_junction_extrapolation_ratio"] = ext
    # side of the corner on which the ceiling segment lies (mid x relative to the junction x)
    mid = (ceil.seg[0] + ceil.seg[1]) / 2; side_ok = (mid >= x) if protocol_side == "right" else (mid <= x); d["ceilF_on_protocol_side"] = bool(side_ok)
    # ceiling segment must reach (nearly) the junction: gap along x between the junction and the nearest end of the segment
    near_end = min(abs(ceil.seg[0] - x), abs(ceil.seg[1] - x)) if not (ceil.seg[0] <= x <= ceil.seg[1]) else 0.0; d["ceilF_end_gap_px_full"] = float(near_end / SCALE)
    if not side_ok: d["issues"].append("ceilF segment lies on the wrong side of the corner")
    if ext > limits.junction_extrap_ratio: d["issues"].append(f"corner top end is {ext:.1f} segment-lengths from the ceiling junction")
    if near_end / SCALE > limits.ceil_end_gap_px_full: d["issues"].append(f"ceilF segment ends {near_end / SCALE:.0f} px from the junction")
    if not (0 <= x <= W and -0.05 * H <= y <= H): d["issues"].append("junction lies outside the image");
    if d["issues"]: d["factor"] = limits.junction_penalty
    return d["factor"], d


# ------------------------------------------------------------------ detection per boundary / per set
@dataclass
class BoundaryResult:
    boundary_id: str                 # e.g. 'PL_corner'
    view_key: str
    boundary_type: str
    placement: str
    camera: str
    set_index: int | None
    frame_index: int | None
    frame_fingerprint: str | None
    image_wh: tuple
    status: str
    failure_reason: str | None
    method: str
    candidates: list                 # ranked dicts (rank 1 first)
    selected_rank: int | None
    diagnostics: dict
    reasons: list = field(default_factory=list)

    @property
    def selected(self):
        return None if self.selected_rank is None else self.candidates[self.selected_rank - 1]

    def to_dict(self): return asdict(self)


def _cand_dict(rank, c: Candidate, H, W, p: Prepared):
    fl = c.full_line(H, W); (x1, y1), (x2, y2) = [(_line_pt(c, u, H, W)) for u in (c.seg[0], c.seg[1])]
    pts_full = [[x1 / SCALE, y1 / SCALE], [x2 / SCALE, y2 / SCALE]]
    return dict(rank=rank, line=dict(kind=fl["kind"], a=fl["a"], b=fl["b"], span=list(fl["span"])), endpoints_undistorted_full=pts_full, polarity=c.sign, visible_length_px=c.length_px_full,
                raw_edge_score=c.raw_score, rank_score=c.rank_score, support_fraction=c.support, own_edge_strength=c.own_strength, partner_strength=c.partner_strength,
                partner_ratio=c.partner_ratio, partner_offset_px=c.partner_offset_px_full, confuser_class=c.confuser, notes=list(c.notes), factors=dict(c.flags))


def _status(top: Candidate | None, second: Candidate | None, limits, junction_ok=None, ceiling_visible=False, extra_issues=()):
    reasons = list(extra_issues)
    if top is None or top.support < limits.min_support * 0.5 or top.length_px_full < limits.min_length_px_candidate: return "NOT_DETECTED", ["no candidate with enough supported visible length"] + reasons
    ok = True
    if top.length_px_full < limits.min_length_px_full: ok = False; reasons.append(f"visible length {top.length_px_full:.0f} px < {limits.min_length_px_full:.0f}")
    if top.support < limits.min_support: ok = False; reasons.append(f"support {top.support:.2f} < {limits.min_support}")
    if top.confuser != "ISOLATED": ok = False; reasons.append(f"top candidate is {top.confuser}: " + "; ".join(top.notes))
    if second is not None and second.rank_score > 0 and top.rank_score / second.rank_score < limits.margin_ratio: ok = False; reasons.append(f"runner-up is within {limits.margin_ratio}x ({top.rank_score / second.rank_score:.2f}x)")
    if top.flags.get("prior_factor", 1.0) < 1.0: ok = False; reasons.append("corner is on the wrong side of the frame for the protocol")
    if junction_ok is False: ok = False; reasons.append("wall-ceiling junction inconsistent")
    return ("DETECTED", reasons) if ok else ("UNCERTAIN", reasons)


def detect_view_boundaries(p: Prepared, placement: str, limits: DetectorLimits = DEFAULT_LIMITS) -> dict:
    """All required boundaries visible in ONE camera: corner (always) and ceilF (right/Pro camera).  Returns {boundary_type: (ranked Candidate list, diag)}."""
    H, W = p.s.shape; side = "left" if placement == "placement_1" else "right"; out = {}
    t0 = time.perf_counter(); pool = generate_candidates(p, "corner", limits); [analyse(p, c, limits, side) for c in pool]; pool.sort(key=lambda c: -c.rank_score)
    out["corner"] = pool; p.runtime_s += time.perf_counter() - t0
    if p.view_key in ("PL", "PR"):
        t0 = time.perf_counter(); cp = generate_candidates(p, "ceilF", limits); [analyse(p, c, limits) for c in cp]; cp.sort(key=lambda c: -c.rank_score); out["ceilF"] = cp; p.runtime_s += time.perf_counter() - t0
    return out


def _hline_angle(c: Candidate): return math.degrees(math.atan(c.slope))


def ceiling_vertex(corner: Candidate, ceils, H, W, limits: DetectorLimits = DEFAULT_LIMITS):
    """Is there a wall-wall-ceiling VERTEX at this corner?  A real room corner is where the ceiling edge changes direction: an edge
    segment to the left and one to the right of the corner line must meet at the corner line (within ``vertex_tol_px_full``) with a
    change of direction >= ``vertex_min_angle_deg``.  A door casing or any mid-wall vertical edge meets a straight ceiling edge.
    -> (found, diag, left candidate, right candidate)."""
    cen = H / 2; xtop = corner.pos + corner.slope * (min(corner.seg) - cen); best = None; tol = limits.vertex_tol_px_full * SCALE
    top_raw = max([c.raw_score for c in ceils] or [0.0]); strong = [c for c in ceils if c.raw_score >= limits.vertex_min_rel_score * top_raw]
    for a in strong:
        for b in strong:
            if a is b or a.sign != b.sign: continue
            ma = (a.seg[0] + a.seg[1]) / 2; mb = (b.seg[0] + b.seg[1]) / 2
            if not (ma < xtop < mb): continue                       # a to the left of the corner line, b to the right
            la, lb = a.line_small(H, W), b.line_small(H, W); den = la["a"] - lb["a"]
            if abs(den) < 1e-9: continue
            x = (lb["b"] - la["b"]) / den; ang = abs(_hline_angle(a) - _hline_angle(b))
            # the corner line at the junction row must pass near the vertex
            y = la["a"] * x + la["b"]; xc = corner.pos + corner.slope * (y - cen)
            if abs(xc - x) > tol or ang < limits.vertex_min_angle_deg: continue
            # each ceiling segment must TERMINATE at the vertex (left one ends there, right one starts there) and extend away from it
            et = limits.vertex_end_tol_px_full * SCALE; ml = limits.vertex_min_seg_px_full * SCALE
            if not (abs(a.seg[1] - x) <= et and abs(b.seg[0] - x) <= et and (a.seg[1] - a.seg[0]) >= ml and (b.seg[1] - b.seg[0]) >= ml): continue
            gap = max(abs(a.seg[1] - x), abs(b.seg[0] - x))
            sc = min(a.raw_score, b.raw_score) * min(1.0, ang / 20.0)             # a sharper change of direction is stronger vertex evidence
            if best is None or sc > best[0]: best = (sc, a, b, dict(vertex_px_full=[x / SCALE, y / SCALE], corner_x_at_vertex_row_px_full=xc / SCALE, offset_px_full=(xc - x) / SCALE, direction_change_deg=ang, end_gap_px_full=gap / SCALE))
    if best is None: return False, dict(found=False, note="no ceiling-edge vertex at this corner"), None, None
    return True, dict(found=True, **best[3]), best[1], best[2]


def rank_with_junction(p: Prepared, placement: str, cands: dict, limits: DetectorLimits = DEFAULT_LIMITS):
    """Joint (corner, ceilF) ranking for a Pro-camera view.  Corner candidates without a ceiling vertex are demoted; the ceilF candidate is the
    member of the top corner's vertex pair on the protocol side (placement 1: right, placement 2: left), otherwise ranked by junction consistency."""
    H, W = p.s.shape; protocol_side = "right" if placement == "placement_1" else "left"; ceils = cands.get("ceilF", []); corners = cands["corner"]
    for c in corners[:limits.n_return * 2]:
        found, d, a, b = ceiling_vertex(c, ceils, H, W, limits); c.flags.update(ceiling_vertex=d, vertex_factor=1.0 if found else limits.vertex_penalty)
        c._pair = (a, b); c.rank_score *= c.flags["vertex_factor"]
        if not found: c.notes.append("no wall-wall-ceiling vertex at this corner (ceiling edge does not change direction here)")
    corners.sort(key=lambda c: -c.rank_score); top = corners[0] if corners else None; info = {}
    for cf in ceils:
        if top is None: break
        f, d = junction(top, cf, protocol_side, H, W, limits); member = getattr(top, "_pair", (None, None))[1 if protocol_side == "right" else 0] is cf
        vf = 1.0 if member else (1.0 if top.flags.get("vertex_factor", 1.0) < 1.0 else 0.4)
        cf.flags.update(junction_factor_vs_top_corner=f, junction_issues=d["issues"], vertex_pair_member=bool(member)); cf.rank_score *= f * vf
        if member: cf.notes.append("member of the top corner's wall-wall-ceiling vertex pair")
    ceils.sort(key=lambda c: -c.rank_score); return top


def detect_set(frames: Mapping, cal, *, set_index=None, frame_indices: Mapping | None = None, fingerprints: Mapping | None = None, limits: DetectorLimits = DEFAULT_LIMITS) -> dict:
    """Automatic detection for ONE Stage C frame set.  ``frames``: {AL,PL,AR,PR: grayscale full-res (lens-distorted) frame}.
    Returns {'boundaries': {boundary_id: BoundaryResult}, 'timing_s': ..., 'n_candidates': ...}.  Does not touch ground truth."""
    t_all = time.perf_counter(); res = {}; prepared = {}; timing = {}
    for vk in ("AL", "PL", "AR", "PR"):
        if vk not in frames: continue
        K, D = (cal.K_left, cal.D_left) if vk[0] == "A" else (cal.K_right, cal.D_right)
        prepared[vk] = prepare(frames[vk], K, D, vk, limits)
    for vk, p in prepared.items():
        placement = PLACEMENT_OF[vk]; H, W = p.s.shape; cands = detect_view_boundaries(p, placement, limits); junction_best = None
        if vk in ("PL", "PR"): junction_best = rank_with_junction(p, placement, cands, limits)
        for bt, pool in cands.items():
            extra = []; jok = None; diag = dict(n_pool=len(pool), runtime_s=p.runtime_s)
            top = pool[0] if pool else None; second = None
            if top is not None:
                for c in pool[1:]:                                   # runner-up must be a distinct line (NMS already guarantees)
                    second = c; break
            if bt == "corner" and vk in ("PL", "PR") and top is not None:
                vf = top.flags.get("vertex_factor", 1.0); diag["ceiling_vertex"] = top.flags.get("ceiling_vertex"); jok = (vf >= 1.0)
                if not jok: extra.append("no wall-wall-ceiling vertex at the top corner candidate: the ceiling edge does not change direction there, so it may be a door casing or any mid-wall vertical edge")
            if bt == "ceilF" and top is not None and cands["corner"]:
                tc = cands["corner"][0]; f, d = junction(tc, top, "right" if placement == "placement_1" else "left", H, W, limits); diag["junction"] = d
                member = bool(top.flags.get("vertex_pair_member")); jok = (f >= 1.0) and member; diag["corroborated_by_corner_vertex"] = member
                if not member: extra.append("ceilF is not corroborated by a wall-wall-ceiling vertex with the top corner candidate: a long near-horizontal edge alone cannot be told from a door head casing, shelf or light edge")
                extra += d["issues"]
            if vk in ("AL", "AR"): diag["junction"] = "not available: the Air camera cannot see the ceiling"
            if vk in ("AL", "AR"): extra = extra + ["Air camera: no ceiling or vertex evidence, so the corner rests on monocular edge evidence only; it needs stereo corroboration (pose) or visitor confirmation"]
            status, reasons = _status(top, second, limits, jok, vk in ("PL", "PR"), extra)
            if vk in ("AL", "AR") and status == "DETECTED": status = "UNCERTAIN"
            if bt == "corner" and vk in ("AL", "AR"): diag["monocular_only_cap"] = True
            cd = [_cand_dict(i + 1, c, H, W, p) for i, c in enumerate(pool[:limits.n_return])]
            sel = 1 if (status != "NOT_DETECTED" and cd) else None
            res[f"{vk}_{bt}"] = BoundaryResult(boundary_id=f"{vk}_{bt}", view_key=vk, boundary_type=bt, placement=placement, camera="left" if vk[0] == "A" else "right", set_index=set_index,
                                              frame_index=None if frame_indices is None else frame_indices.get(vk), frame_fingerprint=None if fingerprints is None else fingerprints.get(vk),
                                              image_wh=p.size_wh, status=status, failure_reason=None if status == "DETECTED" else ("; ".join(reasons) or None), method="sheared-Kadane band-pass step lines + confuser/junction ranking",
                                              candidates=cd, selected_rank=sel, diagnostics=diag, reasons=reasons)
    missing = [f"{vk}_{bt}" for vk, bt in REQUIRED_BOUNDARIES if f"{vk}_{bt}" not in res]
    return dict(set_index=set_index, boundaries=res, missing_views=missing, timing_s=time.perf_counter() - t_all, prepared_runtime_s={k: v.runtime_s for k, v in prepared.items()},
                n_candidates={k: len(v.candidates) for k, v in res.items()}, _prepared=prepared)


# ------------------------------------------------------------------ Stage D integration
def candidate_to_annotation(result: BoundaryResult, rank: int | None = None, *, set_index=None, annotation_id=None) -> dict:
    """Convert a candidate (default: the selected one) into the EXISTING Stage D two-point annotation (undistorted_full).  Frame identity is
    carried unchanged; no second line-fitting pipeline is created -- Stage D's refinement runs on this."""
    rank = rank or result.selected_rank
    if rank is None: raise ValueError(f"{result.boundary_id}: no candidate to convert ({result.status})")
    c = result.candidates[rank - 1]; pts = c["endpoints_undistorted_full"]
    return V.build_annotation(set_index=result.set_index if set_index is None else set_index, placement=result.placement, camera=result.camera, frame_index=result.frame_index, boundary_type=result.boundary_type,
                              image_wh=result.image_wh, coordinate_space="undistorted_full", points=pts, annotation_id=annotation_id or f"auto:set{result.set_index}:{result.boundary_id}:cand{rank}",
                              source="stage_f2_automatic_detection", frame_fingerprint=result.frame_fingerprint)


def run_stage_d(result: BoundaryResult, ctx, cal, rank=None, limits=V.DEFAULT_LIMITS, **kw):
    """Run Stage D (``process_annotation``) on a candidate; the candidate itself is never substituted for a failed refinement."""
    return V.process_annotation(candidate_to_annotation(result, rank), ctx, cal, limits, **kw)
