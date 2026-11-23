"""Stage F8.31B: visitor frame selection and wall-boundary annotation for the four room videos (NO Streamlit import, NO reconstruction).

What the ORIGINAL Method J experiment took as input (audited from ``room_measurement/frames.py``, ``lines.py``, ``measure.py``, ``pipeline.py``,
``config/frame_manifest.json`` and ``config/wall_boundary_annotations.json``; see ARCHITECTURE_PLAN.md section 21):

* MANUAL: which frames (``frame_indices_per_set``: five sets; for each of the four videos an INDEPENDENT 0-based sequential-decode index, chosen by a documented
  rule: set 0 = the sharpest frame, sets 1-4 = the sharpest frame within +-15 frames of 20/40/60/80 % of that video); the boundary-line SEEDS (one seed line per
  boundary, expressed in undistorted pixels, applied to ALL five frame sets) with a search half-window, an expected edge polarity and a tracking method.
* AUTOMATIC: SIFT matches and four-view tracks, sub-pixel edge points along each seed, the Tukey line fit, floor planes, corners, registration, dimensions.
* Left/right frames of a viewpoint are NOT timestamp-synchronised: the pairing is justified by a static scene and a rigid rig; the right-minus-left index
  differs from set to set (original: 45, 41, 41, 57 and 97 frames at placement 1).
* Method J (``top_variant = 0``) reads six lines per frame set: ``AL_corner PL_corner PL_ceilF AR_corner PR_corner PR_ceilF``.  The two ceiling/side-wall lines
  (``PL_ceilS``, ``PR_ceilS``) exist in the original annotation file but are only read by ``top_variant = 1``; they are NOT requested here.  The original eight
  entries are therefore not eight clicks, and Method J needs six.  Floor geometry and viewpoint registration are computed, never annotated.

This module reproduces the MANUAL part in the browser with components that already exist: the frame choice with an explicit right-minus-left offset per pair,
and, for each boundary, TWO clicks that become a seed line which ``visitor_annotations.process_annotation`` (Stage D) refines with the unchanged Method J
functions ``lines.edge_points`` and ``lines.fit_line``.  Differences from the owner's hand-typed seeds (search half-window, polarity, tracking method) are listed
in the plan and returned by ``METHOD_J_INPUT_NOTES``.

Coordinates.  Every stored coordinate is in the ORIGINAL full-resolution, lens-DISTORTED frame (``distorted_full``: OpenCV pixel centres at integers).  Previews
are reduced copies for DISPLAY only; ``preview_to_full`` / ``full_to_preview`` convert, and nothing that is measured is ever read from a preview.

State is bound to ``Binding`` (owner, accepted calibration, the sha256 of every video).  ``sync`` compares the current binding with the one the work was done under
and DISCARDS everything that no longer matches (new calibration: everything; replaced or expired video: that viewpoint).  Nothing is executed: no feature
matching, no pose, no measurement.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from room_measurement import visitor_annotations as V                      # noqa: E402
from room_measurement.visitor_measurement import JobLimits                 # noqa: E402
import frame_worker as FW                                                  # noqa: E402
import room_videos as RV                                                   # noqa: E402

SCHEMA = "csc8830.module7.visitor_frame_selection.v1"
VIEWPOINTS = (1, 2)
SIDES = ("left", "right")
VIEW_KEY = {(1, "left"): "AL", (1, "right"): "PL", (2, "left"): "AR", (2, "right"): "PR"}      # = visitor_annotations.VIEW_KEY by placement
VIEW_KEYS = tuple(VIEW_KEY.values())
SLOT_OF_VIEW = {"AL": "p1_left", "PL": "p1_right", "AR": "p2_left", "PR": "p2_right"}
VIEW_OF_SLOT = {v: k for k, v in SLOT_OF_VIEW.items()}
VIEWPOINT_OF_VIEW = {"AL": 1, "PL": 1, "AR": 2, "PR": 2}
SIDE_OF_VIEW = {"AL": "left", "PL": "right", "AR": "left", "PR": "right"}
REQUIRED_LINES = tuple(V.REQUIRED_LINES)               # AL_corner PL_corner PL_ceilF AR_corner PR_corner PR_ceilF -- what measure_wall reads with top_variant 0
LINES_OF_VIEW = {vk: tuple(n for n in REQUIRED_LINES if n.startswith(vk + "_")) for vk in VIEW_KEYS}
NOT_REQUESTED_LINES = tuple(V.OPTIONAL_LINES)          # PL_ceilS, PR_ceilS: used only by top_variant 1
DEFAULT_SETS, MIN_SETS = 5, 2                          # the original experiment used five frame sets
ORIGINAL_WINDOW_FRAMES = 15                            # "sharpest frame within +-15 frames of 20/40/60/80 %"
COORDINATE_SPACE = "distorted_full"
SEED_SET = 0                                            # F8.31D-6: the sharpest frame (original set 0) of each camera view carries the visitor's six seed lines
MIN_USABLE_SETS = MIN_SETS      # LEGACY, opt-in only (ReconstructionLimits(min_usable_sets=...)); the visitor default is STRICT (all five sets), like the original experiment.                              # fewer refined frame sets than this is a clear failure (the original used five; a floor of two is the workflow's own minimum)

# F8.31D-6: how the ORIGINAL per-line refinement settings (config/wall_boundary_annotations.json) map onto visitor boundaries.  Read from the original file, never imported from it:
# every entry says what was generalised and what was NOT.  ``search_half_window_px`` is the only setting that is applied (ceilF); the rest keep the validated Stage D defaults.
ORIGINAL_REFINEMENT_MAP = {
    "corner": dict(
        search_half_window_px=None,
        why="Original corner half-windows were 45 px (viewpoint 1 seeds, read visually) and 10 px (viewpoint 2 seeds, from a Hough-style scan, tracked in two passes): they encode how "
            "accurate each hand-typed seed was, not a property of a wall corner, so no single value generalises. The Stage D default is kept."),
    "ceilF": dict(
        search_half_window_px=30,
        why="Both original ceiling/front-wall seeds used the same method (edge_pts) with half-windows of 30 px and 28 px; 30 px is applied to this boundary type only."),
}
NOT_GENERALISED_SETTINGS = (
    "Edge polarity (`sign`): +1/-1 depended on which side of the edge was brighter in the owner's room (corners: +1, +1, -1, -1; ceiling: -1, -1), so it is inferred by Stage D, not copied.",
    "Tracking method: the original used single-pass `edge_pts` for viewpoint-1 lines and two-pass `track2` for viewpoint-2 corners; Stage D always does a wide pass then a narrow (6 px) pass around the fit, "
    "which is the `track2` idea applied to every boundary.",
    "Seed position, slope and span: room-specific pixel values; the visitor's two clicks replace them.",
)
CAMERA_ORDER = ("left", "right")
DISTORTION_MODEL = "OpenCV plumb-bob: k1, k2, p1, p2, k3 (5 coefficients)"

METHOD_J_INPUT_NOTES = (
    "Original: five frame sets, each with an independent frame index per video (set 0 = sharpest frame, sets 1-4 = sharpest within +-15 frames of 20/40/60/80 % of the video).",
    "Original: left and right frames are not timestamp-synchronised (right minus left index differed from set to set); a static scene and a rigid rig are assumed.",
    "Original: six boundary lines are used by Method J (top_variant 0); the ceiling/side-wall lines of the original file are not read, so they are not requested.",
    "Original: boundary seeds were typed by the owner once per camera view (position, slope, span, search half-window, edge polarity, tracking method) and applied to all frame sets.",
    "Browser: two clicks per boundary give the seed line and extent; the search half-window and polarity are the existing Stage D defaults (inferred, not hand-tuned) and the "
    "tracking method is Stage D's two-pass edge search.  These are the same Method J functions, driven by different seed parameters.",
)


class WorkflowError(RuntimeError):
    """A refusal with a visitor-safe message."""


class OffsetError(WorkflowError):
    pass


class FrameIntegrityError(WorkflowError):
    pass


# ------------------------------------------------------------------ coordinates
def preview_mapping(full_wh, preview_wh) -> V.DisplayMapping:
    """A preview is the whole frame uniformly reduced and drawn from the canvas origin: no padding, no crop.  Uses the existing ``DisplayMapping``
    (half-pixel convention, per-axis scale, aspect check)."""
    return V.DisplayMapping(int(full_wh[0]), int(full_wh[1]), float(preview_wh[0]), float(preview_wh[1]), 0.0, 0.0, float(preview_wh[0]), float(preview_wh[1]))


def preview_to_full(preview_pts, full_wh, preview_wh) -> np.ndarray:
    """Points clicked on the preview (continuous, edge-aligned, origin top-left) -> original full-resolution pixel coordinates."""
    m = preview_mapping(full_wh, preview_wh)
    bad = m.validate()
    if bad:
        raise WorkflowError("The preview is not a uniform reduction of the frame; refusing to convert coordinates.")
    return m.to_full(preview_pts)


def full_to_preview(full_pts, full_wh, preview_wh) -> np.ndarray:
    return preview_mapping(full_wh, preview_wh).from_full(full_pts)


# ------------------------------------------------------------------ synchronisation / pairing helpers (pure)
def offset_frames(left_index: int, right_index: int) -> int:
    """Explicit right-minus-left frame offset of a pair."""
    return int(right_index) - int(left_index)


def offset_seconds(left_index, right_index, left_fps, right_fps) -> float:
    """Nominal time difference (right time minus left time); each index is divided by ITS OWN video's frame rate (variable frame rate is not corrected)."""
    return float(right_index) / float(right_fps) - float(left_index) / float(left_fps)


def right_index_from_offset(left_index: int, offset: int, right_frame_count: int) -> int:
    """The right frame that goes with ``left_index`` under an explicit offset.  Raises ``OffsetError`` instead of clamping."""
    r = int(left_index) + int(offset)
    if not 0 <= r < int(right_frame_count):
        raise OffsetError(f"Left frame {left_index} with an offset of {offset:+d} frames points at right frame {r}, but the right video has frames 0 to {int(right_frame_count) - 1}.")
    return r


def original_rule_indices(sharpness, n_sets: int, window: int = ORIGINAL_WINDOW_FRAMES) -> list:
    """The original experiment's frame-selection rule for ONE video: set 0 = the sharpest frame; set k (1 <= k < n_sets) = the sharpest frame within +-``window`` frames
    of the point k / n_sets of the way through the video (20/40/60/80 % for five sets).  A frame is never chosen twice (the original reused one frame and noted that
    those two sets were therefore correlated).  The original's "small penalty on frame-to-frame motion" is not recorded in the repository and is not applied.
    ``sharpness``: Laplacian variance per frame (None for an undecodable frame)."""
    n = len(sharpness)
    if n < n_sets:
        raise WorkflowError(f"The video has only {n} decodable frames; {n_sets} frame sets need at least {n_sets}.")
    s = [(-1.0 if v is None else float(v)) for v in sharpness]
    chosen = [int(np.argmax(s))]
    for k in range(1, n_sets):
        c = int(round(k / n_sets * n)); lo, hi = max(0, c - window), min(n - 1, c + window)
        pool = [i for i in range(lo, hi + 1) if i not in chosen and sharpness[i] is not None]
        if not pool:
            pool = [i for i in range(n) if i not in chosen and sharpness[i] is not None]
        chosen.append(max(pool, key=lambda i: (s[i], -abs(i - c))))
    return chosen


# ------------------------------------------------------------------ identity
def calibration_id(contract: dict) -> str:
    """A stable identifier of the ACCEPTED calibration: a digest of the numbers frame selection and annotation depend on."""
    keys = ("K1", "D1", "K2", "D2", "image_size_wh", "baseline_mm", "capture_mode")
    blob = json.dumps({k: contract.get(k) for k in keys}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def owner_digest(owner: str) -> str:
    return hashlib.sha256(("f8.31b|" + str(owner)).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class SourceVideo:
    """One uploaded video as the workflow sees it.  ``path`` is read-only for the workflow; ``sha256`` / ``upload_id`` identify exactly what was uploaded."""
    slot: str
    path: str
    upload_id: str
    sha256: str
    size_bytes: int
    frame_count: int
    fps: float
    wh: tuple


@dataclass(frozen=True)
class Binding:
    owner: str
    calibration: str
    videos: tuple             # ((slot, sha256, upload_id), ...) sorted by slot


def make_binding(owner: str, contract: dict, sources: dict) -> Binding:
    return Binding(owner_digest(owner), calibration_id(contract), tuple(sorted((s, v.sha256, v.upload_id) for s, v in sources.items() if v is not None)))


@dataclass
class Readiness:
    ready: bool
    missing: list = field(default_factory=list)      # input the visitor has not provided yet
    problems: list = field(default_factory=list)     # input that was provided but is not usable / no longer valid
    notes: list = field(default_factory=list)

    def to_dict(self):
        return dict(ready=self.ready, missing=list(self.missing), problems=list(self.problems), notes=list(self.notes),
                    kind="ready" if self.ready else "missing_input" if self.missing and not self.problems else "invalid_input")


@dataclass
class AnnState:
    points: list = field(default_factory=list)       # up to two [x, y], original full-resolution distorted pixels
    edit_index: int | None = None
    revision: int = 0
    result: object = None                            # visitor_annotations.AnnotationResult (in memory only)
    accepted: bool = False
    acknowledged_codes: tuple = ()
    frame_fingerprint: str | None = None
    frame_index: int | None = None
    auto: bool = False                               # F8.31D-6: accepted by the automatic seed propagation (warnings were recorded, not asked about)
    error: str | None = None                         # why an automatic refinement could not run at all (no frame, bad point)


class _LRU(OrderedDict):
    def __init__(self, cap):
        super().__init__(); self.cap = cap

    def put(self, k, v):
        self[k] = v; self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


# ------------------------------------------------------------------ the per-visitor workflow
class RoomFrameWorkflow:
    """ONE visitor's frame selection and annotations.  Owns a private ``FrameWorkspace``; ``close`` removes it.  Holds no video bytes and at most a few frames."""

    def __init__(self, owner: str, contract: dict, cal, sources: dict, limits: JobLimits | None = None, *, n_sets: int = DEFAULT_SETS, worker_cmd=None, workspace_root=None):
        self.limits = limits if limits is not None else JobLimits(video=RV.room_video_limits())
        self.max_sets = int(self.limits.max_frame_sets)
        if not MIN_SETS <= n_sets <= self.max_sets:
            raise WorkflowError(f"The number of frame sets must be from {MIN_SETS} to {self.max_sets}.")
        self.job_limits = FW.FrameJobLimits.from_job_limits(self.limits)
        self.worker_cmd = worker_cmd
        self.ws = FW.FrameWorkspace(workspace_root)
        self.n_sets = int(n_sets)
        self.cal, self.contract, self.sources = cal, contract, dict(sources)
        self.binding = make_binding(owner, contract, self.sources)
        self.pairs: dict = {}                        # (viewpoint, set) -> dict(left, right, confirmed)
        self.previews: dict = {}                     # (slot, index) -> dict(path, wh, full_wh, job_dir)
        self.sharp: dict = {}                        # slot -> [sharpness per frame]
        self.full: dict = {}                         # (set, view_key) -> dict(path, index, fingerprint, wh, job_dir)
        self.ann: dict = {}                          # (set, line) -> AnnState
        self.notes: list = []                        # why something was discarded (shown to the visitor)
        self._ctx = _LRU(2)
        self.closed = False
        self.log: list = []                          # job records (op, slot, seconds): no paths, no pixels

    # ---- ownership / binding ------------------------------------------------------------------------
    def assert_owner(self, owner: str):
        if self.closed or owner_digest(owner) != self.binding.owner:
            raise WorkflowError("This frame selection belongs to another session.")

    def close(self):
        if not self.closed:
            self.closed = True
            self.pairs.clear(); self.previews.clear(); self.full.clear(); self.ann.clear(); self._ctx.clear(); self.ws.close()

    def sync(self, owner: str, contract: dict, cal, sources: dict) -> list:
        """Compare the CURRENT calibration and videos with those the work was done under and discard whatever no longer matches.  Returns the notes added.
        A different calibration discards everything; a replaced, re-uploaded or expired video discards that viewpoint's pairs, frames and annotations."""
        self.assert_owner(owner)
        added = []
        new = make_binding(owner, contract, sources)
        if new.calibration != self.binding.calibration:
            self._discard_all(); added.append("The accepted calibration changed, so your frame selections and annotations were discarded.")
        old_v = {s: (h, u) for s, h, u in self.binding.videos}
        for vp in VIEWPOINTS:
            for side in SIDES:
                slot = f"p{vp}_{side}"
                cur = sources.get(slot)
                prev = old_v.get(slot)
                if cur is None and prev is not None:
                    self._discard_viewpoint(vp); added.append(f"The {RV.SLOT_INFO[slot][2]} is no longer available (it expired or was removed), so Viewpoint {vp} frame selections and annotations were discarded.")
                elif cur is not None and prev is not None and (cur.sha256, cur.upload_id) != prev:
                    self._discard_viewpoint(vp); added.append(f"The {RV.SLOT_INFO[slot][2]} was replaced, so Viewpoint {vp} frame selections and annotations were discarded.")
                elif cur is not None and prev is None:
                    self._discard_viewpoint(vp)                       # a first video in a slot that held nothing when the work began: nothing valid can depend on it
        self.cal, self.contract, self.sources, self.binding = cal, contract, dict(sources), new
        self.notes.extend(added)
        return added

    def _discard_all(self):
        for vp in VIEWPOINTS:
            self._discard_viewpoint(vp)
        self.ann.clear(); self._ctx.clear()

    def _discard_viewpoint(self, vp: int):
        for k in [k for k in self.pairs if k[0] == vp]:
            del self.pairs[k]
        for side in SIDES:
            slot = f"p{vp}_{side}"
            self.sharp.pop(slot, None)
            for k in [k for k in self.previews if k[0] == slot]:
                _unlink(self.previews[k]["path"]); del self.previews[k]
        for vk in VIEW_KEYS:
            if VIEWPOINT_OF_VIEW[vk] == vp:
                self._drop_view(vk)
        self._prune_job_dirs()

    def _drop_view(self, vk: str):
        """Forget the extracted frames and annotations of one camera view (all sets)."""
        for k in [k for k in self.full if k[1] == vk]:
            _unlink(self.full[k]["path"]); del self.full[k]
        for k in [k for k in self.ann if k[1].startswith(vk + "_")]:
            del self.ann[k]
        for k in [k for k in self._ctx if k[1] == vk]:
            del self._ctx[k]

    def _drop_set_view(self, i: int, vk: str, prune: bool = True):
        k = (i, vk)
        if k in self.full:
            _unlink(self.full[k]["path"]); del self.full[k]
        for n in LINES_OF_VIEW[vk]:
            self.ann.pop((i, n), None)
            if i == SEED_SET:                                     # the seed was drawn on this frame: every set's refinement derived from it is no longer valid
                for j in range(self.n_sets):
                    self.ann.pop((j, n), None)
        self._ctx.pop(k, None)
        if prune:
            self._prune_job_dirs()

    # ---- sources --------------------------------------------------------------------------------------
    def source(self, slot: str) -> SourceVideo:
        s = self.sources.get(slot)
        if s is None:
            raise WorkflowError(f"The {RV.SLOT_INFO[slot][2]} is not available. It may have expired: upload it again.")
        return s

    def _job(self, op, slot, **kw) -> dict:
        src = self.source(slot)
        try:
            r = FW.run_frame_job(self.ws, src.path, self.job_limits, worker_cmd=self.worker_cmd, **kw)
        except FW.FrameJobError as e:
            self.log.append(dict(op=op, slot=slot, error=e.code)); raise WorkflowError(e.message) from None
        self.log.append(dict(op=op, slot=slot, seconds=r["seconds"], decoded=r["decoded_frames"]))
        return r

    # ---- previews (display only) ---------------------------------------------------------------------
    def _check_index(self, slot: str, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise WorkflowError("A frame number must be a whole number.")
        n = self.source(slot).frame_count
        if not 0 <= int(index) < n:
            raise WorkflowError(f"Frame {index} does not exist: the video has frames 0 to {n - 1}.")
        return int(index)

    def preview(self, slot: str, index: int) -> dict:
        """A reduced-size copy of one frame for DISPLAY (cached).  Returns dict(path, wh, full_wh)."""
        index = self._check_index(slot, index)
        if (slot, index) not in self.previews:
            r = self._job("preview", slot, preview_indices=[index])
            rec = r["written"][f"prev_{index}"]
            self.previews[(slot, index)] = dict(path=rec["path"], wh=tuple(rec["wh"]), full_wh=tuple(rec["full_wh"]), job_dir=r["job_dir"])
        p = self.previews[(slot, index)]
        return dict(path=p["path"], wh=p["wh"], full_wh=p["full_wh"])

    def contact_sheet(self, slot: str, n: int = 8) -> list:
        """``n`` evenly spaced preview frames of one video in ONE sequential pass (an overview for choosing frames).  Returns the indices."""
        n = max(1, min(int(n), self.job_limits.max_indices))
        count = self.source(slot).frame_count
        idx = sorted({min(count - 1, int((k + 0.5) * count / n)) for k in range(n)})
        todo = [i for i in idx if (slot, i) not in self.previews]
        if todo:
            r = self._job("contact_sheet", slot, preview_indices=todo)
            for i in todo:
                rec = r["written"][f"prev_{i}"]
                self.previews[(slot, i)] = dict(path=rec["path"], wh=tuple(rec["wh"]), full_wh=tuple(rec["full_wh"]), job_dir=r["job_dir"])
        return idx

    # ---- the original selection rule ------------------------------------------------------------------
    def scan_sharpness(self, slot: str) -> list:
        """Laplacian variance of every frame of one video (one bounded, sandboxed pass).  Cached."""
        if slot not in self.sharp:
            r = self._job("sharpness", slot, sharpness_all=True)
            self.workspace_drop(r["job_dir"])
            self.sharp[slot] = r["sharpness"] or []
        return self.sharp[slot]

    def workspace_drop(self, job_dir):
        self.ws.drop(job_dir)

    def propose_original_rule(self, viewpoint: int) -> list:
        """Fill every set of one viewpoint with the original rule (each video independently).  The pairs are PROPOSALS: unconfirmed, and the visitor can change any of them."""
        out = []
        idx = {side: original_rule_indices(self.scan_sharpness(f"p{viewpoint}_{side}"), self.n_sets) for side in SIDES}
        for i in range(self.n_sets):
            self.set_pair(viewpoint, i, idx["left"][i], idx["right"][i]); out.append(self.pairs[(viewpoint, i)])
        return out

    # ---- pairs ------------------------------------------------------------------------------------------
    def set_n_sets(self, n: int):
        if not MIN_SETS <= int(n) <= self.max_sets:
            raise WorkflowError(f"The number of frame sets must be from {MIN_SETS} to {self.max_sets}.")
        if int(n) < self.n_sets:
            for vp in VIEWPOINTS:
                for i in range(int(n), self.n_sets):
                    self.drop_pair(vp, i)
        self.n_sets = int(n)

    def _check_set(self, vp, i):
        if vp not in VIEWPOINTS:
            raise WorkflowError("Unknown viewpoint.")
        if not 0 <= int(i) < self.n_sets:
            raise WorkflowError(f"Frame set {i} does not exist (there are {self.n_sets}).")

    def set_pair(self, vp: int, i: int, left: int, right: int):
        """Create or REPLACE the frame pair of one set.  A changed frame discards that camera's extracted frame and annotations; the pair becomes unconfirmed."""
        self._check_set(vp, i)
        left, right = self._check_index(f"p{vp}_left", left), self._check_index(f"p{vp}_right", right)
        old = self.pairs.get((vp, i))
        if old is None or old["left"] != left:
            self._drop_set_view(i, VIEW_KEY[(vp, "left")])
        if old is None or old["right"] != right:
            self._drop_set_view(i, VIEW_KEY[(vp, "right")])
        self.pairs[(vp, i)] = dict(left=left, right=right, confirmed=False)

    def set_pair_by_offset(self, vp: int, i: int, left: int, offset: int):
        self._check_set(vp, i)
        self.set_pair(vp, i, left, right_index_from_offset(left, offset, self.source(f"p{vp}_right").frame_count))

    def drop_pair(self, vp: int, i: int):
        if (vp, i) in self.pairs:
            for side in SIDES:
                self._drop_set_view(i, VIEW_KEY[(vp, side)])
            del self.pairs[(vp, i)]

    def pair_info(self, vp: int, i: int) -> dict | None:
        p = self.pairs.get((vp, i))
        if p is None:
            return None
        L_, R_ = self.source(f"p{vp}_left"), self.source(f"p{vp}_right")
        return dict(left=p["left"], right=p["right"], confirmed=p["confirmed"], offset_frames=offset_frames(p["left"], p["right"]),
                    offset_seconds=offset_seconds(p["left"], p["right"], L_.fps, R_.fps), previews_viewed=self.previews_viewed(vp, i))

    def previews_viewed(self, vp: int, i: int) -> bool:
        p = self.pairs.get((vp, i))
        return bool(p and (f"p{vp}_left", p["left"]) in self.previews and (f"p{vp}_right", p["right"]) in self.previews)

    def viewpoint_offset(self, vp: int):
        """The common right-minus-left offset of a viewpoint's pairs, or None when they differ (as in the original data, where they did)."""
        offs = {offset_frames(p["left"], p["right"]) for (v, _), p in self.pairs.items() if v == vp}
        return offs.pop() if len(offs) == 1 else None

    def confirm_pair(self, vp: int, i: int):
        """Confirm only after BOTH frames of the pair have been shown to the visitor."""
        self._check_set(vp, i)
        if (vp, i) not in self.pairs:
            raise WorkflowError("Choose a left and a right frame first.")
        if not self.previews_viewed(vp, i):
            raise WorkflowError("Preview both frames of this pair before confirming it.")
        self.pairs[(vp, i)]["confirmed"] = True

    def unconfirm_pair(self, vp: int, i: int):
        if (vp, i) in self.pairs:
            self.pairs[(vp, i)]["confirmed"] = False

    def duplicate_frames(self) -> list:
        """Frames used by more than one set of the same video (the original experiment had one such reuse and noted that the sets were correlated).  A note, not a block."""
        seen, out = {}, []
        for (vp, i), p in sorted(self.pairs.items()):
            for side in SIDES:
                k = (f"p{vp}_{side}", p[side])
                if k in seen:
                    out.append(f"{RV.SLOT_INFO[k[0]][2]}: frame {k[1]} is used by frame sets {seen[k] + 1} and {i + 1}; those sets are correlated, not independent.")
                else:
                    seen[k] = i
        return out

    def all_pairs_confirmed(self) -> bool:
        return all(self.pairs.get((vp, i), {}).get("confirmed") for vp in VIEWPOINTS for i in range(self.n_sets))

    # ---- full-resolution extraction ------------------------------------------------------------------
    def extract_confirmed(self) -> int:
        """Decode the CONFIRMED frames at full resolution (one bounded sandboxed pass per video, only for frames not yet extracted).  Frames are written losslessly
        and re-read to verify; a frame whose size differs from the calibration is refused (nothing is resized).  Returns how many frames were written."""
        W, H = (int(v) for v in self.cal.image_size)
        todo = {}
        for (vp, i), p in self.pairs.items():
            if not p["confirmed"]:
                continue
            for side in SIDES:
                vk = VIEW_KEY[(vp, side)]
                cur = self.full.get((i, vk))
                if cur is None or cur["index"] != p[side]:
                    todo.setdefault(f"p{vp}_{side}", []).append((i, vk, p[side]))
        n = 0
        for slot, items in todo.items():
            r = self._job("extract", slot, full_indices=[ix for _, _, ix in items])
            try:
                for i, vk, ix in items:
                    rec = r["written"][f"full_{ix}"]
                    g = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
                    if g is None:
                        raise WorkflowError("A decoded frame could not be read back.")
                    if (g.shape[1], g.shape[0]) != (W, H):
                        raise WorkflowError(f"Frame {ix} of the {RV.SLOT_INFO[slot][2]} is {g.shape[1]} × {g.shape[0]} px but the accepted calibration is {W} × {H} px. "
                                            "Camera intrinsics are not rescaled and nothing is resized: record at the calibrated resolution.")
                    self._drop_set_view(i, vk, prune=False)         # the job directory being filled must not be pruned mid-loop
                    self.full[(i, vk)] = dict(path=rec["path"], index=ix, fingerprint=V.frame_fingerprint(g), wh=(W, H), job_dir=r["job_dir"], slot=slot)
                    n += 1
                    del g
            except BaseException:
                self.ws.drop(r["job_dir"])
                for k in [k for k, v in self.full.items() if v["job_dir"] == r["job_dir"]]:
                    del self.full[k]
                raise
            used = {v["job_dir"] for v in self.full.values()}
            if r["job_dir"] not in used:
                self.ws.drop(r["job_dir"])
        self._prune_job_dirs()
        return n

    def _prune_job_dirs(self):
        """Delete extraction job directories that no longer hold any referenced frame (their other frames were replaced)."""
        live = {v["job_dir"] for v in self.full.values()} | {v["job_dir"] for v in self.previews.values()}
        for d in Path(self.ws.path).glob("job_*"):
            if str(d) not in live and d.is_dir():
                shutil_rm(d)

    def frame_gray(self, i: int, vk: str) -> np.ndarray:
        """The EXACT extracted frame, re-fingerprinted on every load (a mismatch raises; nothing is substituted)."""
        rec = self.full.get((i, vk))
        if rec is None:
            raise WorkflowError("This frame has not been extracted yet. Confirm its frame pair first.")
        g = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
        if g is None or V.frame_fingerprint(g) != rec["fingerprint"]:
            raise FrameIntegrityError("An extracted frame no longer matches its fingerprint; select the frames again.")
        return g

    # ---- annotations ----------------------------------------------------------------------------------
    def state(self, i: int, line: str) -> AnnState:
        if line not in REQUIRED_LINES:
            raise WorkflowError("This boundary is not needed.")
        return self.ann.setdefault((i, line), AnnState())

    def _fresh(self, st: AnnState):
        st.revision += 1; st.result = None; st.accepted = False; st.acknowledged_codes = ()

    def _check_point(self, x, y):
        W, H = (int(v) for v in self.cal.image_size)
        if not (isinstance(x, (int, float, np.floating, np.integer)) and isinstance(y, (int, float, np.floating, np.integer))) or not (np.isfinite(x) and np.isfinite(y)):
            raise WorkflowError("A point must be two finite numbers.")
        if not (0 <= x <= W - 1 and 0 <= y <= H - 1):
            raise WorkflowError(f"The point ({x:.1f}, {y:.1f}) is outside the {W} × {H} px frame.")
        return [float(x), float(y)]

    def set_points(self, i: int, line: str, points):
        st = self.state(i, line)
        pts = [self._check_point(*p) for p in points]
        if len(pts) > 2:
            raise WorkflowError("A boundary has exactly two points.")
        st.points = pts; st.edit_index = None; self._fresh(st)

    def click(self, i: int, line: str, x: float, y: float) -> str:
        """A click in ORIGINAL full-resolution coordinates.  First two clicks place the points; later clicks move the point chosen with ``begin_edit`` or, failing that, the nearest."""
        st = self.state(i, line); p = self._check_point(x, y)
        if st.edit_index is not None and st.edit_index < len(st.points):
            st.points[st.edit_index] = p; msg = f"Point {st.edit_index + 1} moved."
        elif len(st.points) < 2:
            st.points.append(p); msg = f"Point {len(st.points)} placed."
        else:
            k = int(np.argmin([np.hypot(p[0] - q[0], p[1] - q[1]) for q in st.points])); st.points[k] = p; msg = f"Point {k + 1} moved."
        st.edit_index = None; self._fresh(st)
        return msg

    def begin_edit(self, i: int, line: str, index: int):
        st = self.state(i, line)
        if index >= len(st.points):
            raise WorkflowError("That point has not been placed yet.")
        st.edit_index = int(index)

    def clear_points(self, i: int, line: str):
        st = self.state(i, line); st.points = []; st.edit_index = None; self._fresh(st)

    def _context(self, i: int, vk: str):
        key = (i, vk)
        if key not in self._ctx:
            self._ctx.put(key, V.CameraContext(self.frame_gray(i, vk), *((self.cal.K_left, self.cal.D_left) if SIDE_OF_VIEW[vk] == "left" else (self.cal.K_right, self.cal.D_right))))
        return self._ctx[key]

    def refine(self, i: int, line: str):
        """Run the existing Stage D refinement (the unchanged Method J ``edge_points`` + ``fit_line``) on this boundary's two points.  Returns the ``AnnotationResult``."""
        st = self.state(i, line)
        if len(st.points) != 2:
            raise WorkflowError("Place two points on the boundary first.")
        vk = line[:2]; placement, camera = V.VIEW_PLACEMENT[vk]; rec = self.full.get((i, vk))
        if rec is None:
            raise WorkflowError("Confirm this set's frame pair first: its full-resolution frame has not been extracted.")
        W, H = (int(v) for v in self.cal.image_size)
        raw = V.build_annotation(set_index=i, placement=placement, camera=camera, frame_index=rec["index"], boundary_type=line.split("_")[1], image_wh=(W, H),
                                 coordinate_space=COORDINATE_SPACE, points=st.points, annotation_id=f"f8_31b_s{i}_{line}_r{st.revision}", source="f8_31b_visitor_browser",
                                 frame_fingerprint=rec["fingerprint"])
        res = V.process_annotation(raw, self._context(i, vk), self.cal, self.limits_for(line), expected_set_index=i, expected_frame_index=rec["index"])
        st.result, st.accepted, st.acknowledged_codes, st.auto, st.error = res, False, (), False, None; st.frame_fingerprint, st.frame_index = rec["fingerprint"], rec["index"]
        return res

    @staticmethod
    def limits_for(line: str):
        """Stage D limits for this boundary: the validated defaults, with the ORIGINAL half-window where ``ORIGINAL_REFINEMENT_MAP`` generalises one.  Nothing else is changed."""
        w = ORIGINAL_REFINEMENT_MAP[line.split("_")[1]]["search_half_window_px"]
        return V.DEFAULT_LIMITS if w is None else replace(V.DEFAULT_LIMITS, search_half_window_px=int(w))

    def accept(self, i: int, line: str, acknowledged: bool = False):
        """The visitor accepts the refined line after looking at it.  FAIL can never be accepted; WARNING needs an explicit acknowledgement of exactly the current codes."""
        st = self.state(i, line); res = st.result
        if res is None:
            raise WorkflowError("Refine the boundary first.")
        if res.status is V.Status.FAIL or not res.downstream_permitted or res.downstream_line is None:
            raise WorkflowError("This boundary cannot be used: " + ("; ".join(m.message for m in res.issues if m.status is V.Status.FAIL) or "the refinement did not produce a usable line") + ".")
        if res.status is V.Status.WARNING and not acknowledged:
            raise WorkflowError("Acknowledge the warnings before accepting this boundary.")
        st.accepted = True; st.acknowledged_codes = tuple(res.warning_codes) if res.status is V.Status.WARNING else ()

    def reject(self, i: int, line: str):
        st = self.state(i, line); st.accepted = False; st.acknowledged_codes = ()

    def copy_points(self, src_set: int, dst_set: int, line: str):
        """Use the points of another set as the STARTING seed for this one (the original applied one seed to all sets).  Nothing is refined or accepted by copying."""
        src = self.ann.get((src_set, line))
        if src is None or len(src.points) != 2:
            raise WorkflowError("The source set has no complete boundary to copy.")
        self.set_points(dst_set, line, src.points)

    def propagate(self, line: str, src_set: int) -> dict:
        """Apply one set's seed to every other set that has its frame extracted and refine each (automatic, like the original); the results are PROPOSALS: nothing is accepted.
        Returns {set: status or error text}."""
        out = {}
        for i in range(self.n_sets):
            if i == src_set or (i, line[:2]) not in self.full:
                continue
            try:
                self.copy_points(src_set, i, line); out[i] = self.refine(i, line).status.value
            except WorkflowError as e:
                out[i] = str(e)
        return out

    # ---- F8.31D-6: automatic frames, six seeds, automatic propagation ---------------------------------------
    def auto_select(self) -> int:
        """Step 1 of the visitor flow: the original rule for BOTH viewpoints (five sets, an independent index per video), the pairs confirmed by this automatic procedure (the visitor
        may look at them or change them, but is not asked to accept each one), then the full-resolution frames are extracted.  Raises ``WorkflowError`` with the reason; no index is invented."""
        if self.n_sets != DEFAULT_SETS:
            self.set_n_sets(DEFAULT_SETS)
        for vp in VIEWPOINTS:
            self.propose_original_rule(vp)
        for vp in VIEWPOINTS:
            for i in range(self.n_sets):
                self.pairs[(vp, i)]["confirmed"] = True; self.pairs[(vp, i)]["auto"] = True
        return self.extract_confirmed()

    def frames_ready(self) -> bool:
        return all(self.pairs.get((vp, i), {}).get("confirmed") and (i, VIEW_KEY[(vp, s)]) in self.full for vp in VIEWPOINTS for i in range(self.n_sets) for s in SIDES)

    def seed_points(self, line: str) -> list:
        st = self.ann.get((SEED_SET, line))
        return [] if st is None else [list(p) for p in st.points]

    def seeds_complete(self) -> bool:
        return all(len(self.seed_points(n)) == 2 for n in REQUIRED_LINES)

    def seed_click(self, line: str, x: float, y: float) -> str:
        msg = self.click(SEED_SET, line, x, y); self._clear_derived(line); return msg

    def seed_clear(self, line: str):
        self.clear_points(SEED_SET, line); self._clear_derived(line)

    def seed_set(self, line: str, points):
        """F8.31F: the visitor CONFIRMED two points (a suggested line they accepted).  Same storage and invalidation as two clicks: nothing is refined or accepted here."""
        self.set_points(SEED_SET, line, points); self._clear_derived(line)

    def seeds_reopen(self):
        """F8.31F: the visitor wants to review or change the boundaries.  The six seeds are kept; every derived refinement (sets 2..5) is dropped so the boundaries are applied again."""
        for line in REQUIRED_LINES:
            self._clear_derived(line)

    def reopen_boundary(self, line: str):
        """F8.31G: the visitor wants to correct ONE boundary (for example one that could not be refined).  Its two points are kept so they can be seen and replaced; its refinement
        results in every set are dropped.  Every other boundary keeps its seed and its results; nothing is refined here."""
        if line not in REQUIRED_LINES:
            raise WorkflowError("This boundary is not needed.")
        st = self.ann.get((SEED_SET, line))
        if st is not None:
            self._fresh(st); st.error = None; st.auto = False
        self._clear_derived(line)

    def seed_begin_edit(self, line: str, index: int):
        self.begin_edit(SEED_SET, line, index)

    def seed_check(self, line: str):
        """Refine the seed on its own frame (set 0) only, so the visitor can SEE the line the original functions find.  Nothing is propagated or accepted."""
        res = self.refine(SEED_SET, line); self._clear_derived(line); return res

    def _clear_derived(self, line: str):
        """A changed seed invalidates every other set's refinement of that boundary (they were derived from the old points)."""
        for j in range(1, self.n_sets):
            self.ann.pop((j, line), None)

    def apply_seeds(self) -> dict:
        """The original behaviour: ONE seed per boundary, the same seed applied to every frame set, refined by the unchanged Stage D / Method J line functions.
        Every (set, line) is refined and, when it yields a usable line (PASS, or WARNING), accepted automatically with its warnings recorded; a FAIL stays unaccepted and is reported.
        Nothing is retried, loosened or tuned.  Returns ``{(set, line): 'PASS'|'WARNING'|'FAIL'|'ERROR'}``."""
        if not self.seeds_complete():
            raise WorkflowError("Mark all six wall boundaries first.")
        if not self.frames_ready():
            raise WorkflowError("The frames have not been selected yet.")
        out = {}
        for line in REQUIRED_LINES:
            pts = self.seed_points(line)
            for i in range(self.n_sets):
                try:
                    if i != SEED_SET:
                        self.set_points(i, line, pts)
                    res = self.refine(i, line); st = self.ann[(i, line)]
                    if res.status in (V.Status.PASS, V.Status.WARNING) and res.downstream_permitted and res.downstream_line is not None:
                        st.accepted = True; st.auto = True; st.acknowledged_codes = tuple(res.warning_codes) if res.status is V.Status.WARNING else ()
                    out[(i, line)] = res.status.value if res.status in (V.Status.PASS, V.Status.WARNING) and st.accepted else "FAIL"
                except WorkflowError as e:
                    st = self.ann.setdefault((i, line), AnnState()); st.error = str(e); out[(i, line)] = "ERROR"
        return out

    def line_report(self, i: int, line: str) -> dict:
        """What happened to one boundary in one set: ``status`` is ok | ok_warning | failed | not_run, with the Stage D messages."""
        st = self.ann.get((i, line))
        if st is None or st.result is None and st.error is None:
            return dict(status="not_run", issues=[])
        if st.error:
            return dict(status="failed", issues=[dict(code="NOT_RUN", message=st.error)])
        res = st.result; issues = [dict(code=m.code, message=m.message, level=m.status.value) for m in res.issues if m.status in (V.Status.FAIL, V.Status.WARNING)]
        if st.accepted:
            return dict(status="ok_warning" if res.status is V.Status.WARNING else "ok", issues=issues)
        return dict(status="failed", issues=issues)

    def set_usable(self, i: int) -> bool:
        return all(self.ann.get((i, n)) is not None and self.ann[(i, n)].accepted for n in REQUIRED_LINES)

    def usable_sets(self) -> list:
        return [i for i in range(self.n_sets) if self.set_usable(i)]

    def seeds_applied(self) -> bool:
        return all(self.line_report(i, n)["status"] != "not_run" for i in range(self.n_sets) for n in REQUIRED_LINES)

    def annotation_status(self, i: int, line: str) -> str:
        st = self.ann.get((i, line))
        if st is None or not st.points:
            return "not_started"
        if st.accepted:
            return "accepted"
        if st.result is None:
            return "points_placed" if len(st.points) == 2 else "placing_points"
        return "refine_failed" if st.result.status is V.Status.FAIL else "needs_review"

    # ---- readiness and export -------------------------------------------------------------------------
    def readiness(self, *, deep: bool = False, min_usable_sets: int | None = None) -> Readiness:
        """Is every input Method J needs present and still valid?  This only checks INPUT.  It runs no feature matching, pose or geometry, so it can say nothing about whether a
        reconstruction will succeed; a reconstruction failure is a different, later outcome."""
        r = Readiness(True)
        miss, bad = r.missing, r.problems
        for slot in RV.ROOM_UPLOAD_SLOTS:
            if slot not in self.sources or self.sources[slot] is None:
                miss.append(f"{RV.SLOT_INFO[slot][2]} is not available (upload it again).")
        if self.binding.calibration != calibration_id(self.contract):
            bad.append("The calibration changed after the frames were selected.")
        for vp in VIEWPOINTS:
            for i in range(self.n_sets):
                p = self.pairs.get((vp, i))
                if p is None:
                    miss.append(f"Viewpoint {vp}, frame set {i + 1}: choose a left and a right frame."); continue
                if not p["confirmed"]:
                    miss.append(f"Viewpoint {vp}, frame set {i + 1}: confirm the frame pair."); continue
                for side in SIDES:
                    vk = VIEW_KEY[(vp, side)]; f = self.full.get((i, vk))
                    if f is None or f["index"] != p[side]:
                        miss.append(f"Viewpoint {vp}, frame set {i + 1}: the {side} frame has not been extracted."); continue
                    if deep:
                        try:
                            self.frame_gray(i, vk)
                        except WorkflowError as e:
                            bad.append(f"Viewpoint {vp}, frame set {i + 1}, {side} frame: {e}")
        W, H = (int(v) for v in self.cal.image_size)
        skipped = []
        for i in range(self.n_sets):
            if min_usable_sets is not None and not self.set_usable(i):
                skipped.append(i + 1); continue                      # reported, and the measurement uses the other sets (the original skipped a failed set)
            for line in REQUIRED_LINES:
                label = f"Frame set {i + 1}, {line}"
                st = self.ann.get((i, line))
                if st is None or not st.accepted:
                    miss.append(f"{label}: mark and accept the boundary."); continue
                f = self.full.get((i, line[:2]))
                if f is None or st.frame_fingerprint != f["fingerprint"] or st.frame_index != f["index"]:
                    bad.append(f"{label}: made on a frame that is no longer the selected one."); continue
                res = st.result
                if res is None or not res.downstream_permitted or res.downstream_line is None:
                    bad.append(f"{label}: the refined line is not usable."); continue
                parsed, issues = V.parse_annotation(res_raw(st, i, line, f, self.cal))
                if parsed is None or any(m.status is V.Status.FAIL for m in issues):
                    bad.append(f"{label}: the annotation has an invalid structure."); continue
                if any(not (0 <= x <= W - 1 and 0 <= y <= H - 1) for x, y in st.points):
                    bad.append(f"{label}: a point lies outside the {W} × {H} px frame.")
        if min_usable_sets is not None:
            n_ok = self.n_sets - len(skipped)
            if n_ok < min_usable_sets:
                miss.append(f"Only {n_ok} of {self.n_sets} frame sets have all six boundaries refined; at least {min_usable_sets} are needed.")
            elif skipped:
                r.notes.append("Frame set(s) " + ", ".join(str(k) for k in skipped) + " will be skipped because not every boundary could be refined in them.")
        if self.duplicate_frames():
            r.notes.extend(self.duplicate_frames())
        r.ready = not miss and not bad
        return r

    def lines_for_adapter(self) -> list:
        """Per frame set, ``{line name: fitted-line dict}`` in the form ``lines.fit_lines`` returns and ``ReconstructionInputs.line_sets`` expects (keys kind, a, b, n, N, rms,
        span, pts; undistorted pixels of the accepted calibration).  Raises unless the work is READY (deep check)."""
        rd = self.readiness(deep=True)
        if not rd.ready:
            raise WorkflowError("Frame selection and annotation are not complete.")
        return [{n: dict(self.ann[(i, n)].result.downstream_line) for n in REQUIRED_LINES} for i in range(self.n_sets)]

    def frame_table(self) -> list:
        """Per set: the original full-resolution frame index of each of the four views (the analogue of the original ``frame_index_table``)."""
        return [{VIEW_KEY[(vp, side)]: self.pairs[(vp, i)][side] for vp in VIEWPOINTS for side in SIDES} for i in range(self.n_sets)]

    def calibration_record(self) -> dict:
        """What every selection and annotation stays associated with: the accepted calibration, never a default and never the owner's."""
        c = self.contract
        return dict(calibration_id=self.binding.calibration, K_left=c["K1"], K_right=c["K2"], D_left=c["D1"], D_right=c["D2"], distortion_model=DISTORTION_MODEL,
                    image_size_wh=list(self.cal.image_size), baseline_mm=float(self.cal.baseline_mm), camera_order=list(CAMERA_ORDER), capture_mode=dict(c.get("capture_mode") or {}))

    def export_state(self) -> dict:
        """A JSON-safe summary (no pixels, no paths).  Valid only for the binding it carries."""
        rd = self.readiness()
        return dict(schema=SCHEMA, ready=rd.ready, calibration=self.calibration_record(), coordinate_space=COORDINATE_SPACE, n_sets=self.n_sets,
                    videos={s: dict(sha256=v.sha256, frame_count=v.frame_count, fps=v.fps, wh=list(v.wh)) for s, v in sorted(self.sources.items()) if v is not None},
                    frame_sets=[dict(set_index=i, frames={VIEW_KEY[(vp, side)]: dict(index=self.pairs[(vp, i)][side], fingerprint=self.full[(i, VIEW_KEY[(vp, side)])]["fingerprint"])
                                                             for vp in VIEWPOINTS for side in SIDES if (vp, i) in self.pairs and (i, VIEW_KEY[(vp, side)]) in self.full},
                                     offsets={f"viewpoint_{vp}": offset_frames(self.pairs[(vp, i)]["left"], self.pairs[(vp, i)]["right"]) for vp in VIEWPOINTS if (vp, i) in self.pairs})
                                for i in range(self.n_sets)],
                    annotations=[dict(set_index=i, line=n, points=[list(p) for p in self.ann[(i, n)].points], accepted=self.ann[(i, n)].accepted,
                                      status=None if self.ann[(i, n)].result is None else self.ann[(i, n)].result.status.value)
                                 for i in range(self.n_sets) for n in REQUIRED_LINES if (i, n) in self.ann],
                    method_j_input_notes=list(METHOD_J_INPUT_NOTES), not_requested_lines=list(NOT_REQUESTED_LINES), problems=rd.to_dict())


def res_raw(st: AnnState, i: int, line: str, f: dict, cal) -> dict:
    """The annotation payload that the stored result was made from (rebuilt from the stored points and frame identity; used to re-validate the structure)."""
    vk = line[:2]; placement, camera = V.VIEW_PLACEMENT[vk]
    return V.build_annotation(set_index=i, placement=placement, camera=camera, frame_index=f["index"], boundary_type=line.split("_")[1], image_wh=tuple(int(v) for v in cal.image_size),
                              coordinate_space=COORDINATE_SPACE, points=st.points, annotation_id=f"f8_31b_s{i}_{line}", source="f8_31b_visitor_browser", frame_fingerprint=f["fingerprint"])


def shutil_rm(p):
    import shutil
    shutil.rmtree(p, ignore_errors=True)


def _unlink(path):
    try:
        Path(path).unlink()
    except OSError:
        pass
