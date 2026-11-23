"""Three-viewpoint Wall B: frame sets and the seven-line annotation workflow of ONE visitor (NO Streamlit import, NO reconstruction).

What it collects (the schema of the validated offline experiment, ``wall_b_three_viewpoint_annotations.v1``):

    Viewpoint 1 (left end of the wall)   left-camera vertical corner · right-camera vertical corner · right-camera horizontal ceiling/wall boundary
    Viewpoint 2 (middle)                 right-camera horizontal ceiling/wall boundary ONLY (no vertical line)
    Viewpoint 3 (right end of the wall)  left-camera vertical corner · right-camera vertical corner · right-camera horizontal ceiling/wall boundary

A line is TWO clicks.  Coordinates are stored in ORIGINAL full-resolution, lens-DISTORTED frame pixels (the extracted frame is never resized; zoom and pan only crop it, see
``annotation_ui.core``).  Only the four vertical corner lines are used by the reconstruction; the three horizontal lines are collected and carried, not used (they have not been validated
as a geometric constraint), and may be marked unavailable.  A frame set whose four vertical lines are not all marked is NOT USABLE and is excluded (reported, never silently dropped);
the reconstruction needs at least ``QualityThresholds.min_usable_frames`` usable frame sets.

A FRAME SET is one frame number per video (default: the same number in all six videos, as in the validated experiment; left/right of a viewpoint are paired by number under the static-scene,
rigid-rig assumption).  Annotations are tied to the exact extracted frame: choosing a different frame, replacing a video or changing the calibration discards the work that depended on it.
Nothing from an earlier set of videos or a different viewpoint count is ever reused.

Previews and full-resolution extraction run only in the sandboxed ``frame_worker`` (shared with the two-viewpoint workflow); this module decodes nothing itself.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent
for _p in (str(MODULE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from room_measurement import visitor_annotations as V                      # noqa: E402  (frame_fingerprint only)
from room_measurement import three_view_wall_b as T                        # noqa: E402
from room_measurement import three_view_quality as Q                       # noqa: E402
from room_measurement.visitor_measurement import JobLimits                 # noqa: E402
import frame_worker as FW                                                  # noqa: E402
import room_frames as RF                                                   # noqa: E402
import room_videos as RV                                                   # noqa: E402

SCHEMA = "csc8830.module7.three_view_frame_selection.v1"
VIEWPOINTS = (1, 2, 3)
MIN_SETS, DEFAULT_SETS, MAX_SETS = 2, 3, 5
DEFAULT_FRACTIONS = (0.2, 0.5, 0.8)             # of the shortest video (the validated experiment used frames 60/150/240 of a 336-frame video: 18 % / 45 % / 71 %)
MIN_LINE_LENGTH_PX = 25.0                       # two clicks closer than this do not define a direction worth using
SLOTS = RV.ROOM_UPLOAD_SLOTS_3

# name -> (viewpoint, camera, orientation, required, boundary, role)
LINE_INFO = {
    "V1_left_vertical": (1, "left", "vertical", True, "corner", "left_wall_corner"),
    "V1_right_vertical": (1, "right", "vertical", True, "corner", "left_wall_corner"),
    "V1_right_horizontal": (1, "right", "horizontal", False, "ceiling_wall", "ceiling_wall_boundary"),
    "V2_right_horizontal": (2, "right", "horizontal", False, "ceiling_wall", "ceiling_wall_boundary"),
    "V3_left_vertical": (3, "left", "vertical", True, "corner", "right_wall_corner"),
    "V3_right_vertical": (3, "right", "vertical", True, "corner", "right_wall_corner"),
    "V3_right_horizontal": (3, "right", "horizontal", False, "ceiling_wall", "ceiling_wall_boundary"),
}
LINE_NAMES = tuple(LINE_INFO)                                    # the seven lines, in the order the visitor marks them
REQUIRED_LINES = tuple(n for n, i in LINE_INFO.items() if i[3])


def slot_of(line: str) -> str:
    v, cam = LINE_INFO[line][0], LINE_INFO[line][1]
    return f"p{v}_{cam}"


def label(line: str) -> str:
    v, cam, ori, req, bd, role = LINE_INFO[line]
    side = {1: "left end", 2: "middle", 3: "right end"}[v]
    what = "vertical wall corner" if ori == "vertical" else "ceiling / wall boundary"
    return f"Viewpoint {v} ({side}) · {cam} camera · {what}"


class ThreeViewError(RF.WorkflowError):
    """A visitor-safe refusal (a ``room_frames.WorkflowError``, so the shared preview / extraction plumbing and this class fail the same way)."""


@dataclass
class LineState:
    points: list = field(default_factory=list)       # up to two [x, y], original full-resolution distorted pixels
    unavailable: bool = False
    revision: int = 0
    fingerprint: str | None = None                   # of the extracted frame the points were marked on


class ThreeViewWorkflow:
    """ONE visitor's three-viewpoint Wall B frame sets and annotations.  Owns a private ``FrameWorkspace``; ``close`` removes it."""

    # The preview / extraction plumbing is the two-viewpoint workflow's own (same sandboxed worker, same limits, same cache rules).  These functions only use the attributes
    # ``sources, ws, job_limits, worker_cmd, log, previews, full`` and ``RV.SLOT_INFO``, all of which this class provides under the same names.
    source = RF.RoomFrameWorkflow.source
    _job = RF.RoomFrameWorkflow._job
    _check_index = RF.RoomFrameWorkflow._check_index
    preview = RF.RoomFrameWorkflow.preview
    workspace_drop = RF.RoomFrameWorkflow.workspace_drop
    _prune_job_dirs = RF.RoomFrameWorkflow._prune_job_dirs

    def __init__(self, owner: str, contract: dict, cal, sources: dict, limits: JobLimits | None = None, *, n_sets: int = DEFAULT_SETS, worker_cmd=None, workspace_root=None):
        self.limits = limits if limits is not None else JobLimits(video=RV.room_video_limits())
        self.job_limits = FW.FrameJobLimits.from_job_limits(self.limits)
        self.worker_cmd = worker_cmd
        self.ws = FW.FrameWorkspace(workspace_root)
        self.cal, self.contract, self.sources = cal, contract, dict(sources)
        self.binding = RF.make_binding(owner, contract, self.sources)
        self.previews: dict = {}                     # (slot, index) -> dict(path, wh, full_wh, job_dir)
        self.full: dict = {}                         # (set, slot) -> dict(path, index, fingerprint, wh, job_dir)
        self.sets: list = []                         # [{slot: frame number}] one dict per frame set
        self.confirmed = False                       # the visitor accepted the frame sets (their frames are extracted)
        self.ann: dict = {}                          # (set, line) -> LineState
        self.notes: list = []
        self.closed = False
        self.log: list = []
        self.view_limits = self.job_limits
        self.auto_select(n_sets)

    # ---- ownership / binding ------------------------------------------------------------------------
    def assert_owner(self, owner: str):
        if self.closed or RF.owner_digest(owner) != self.binding.owner:
            raise ThreeViewError("This frame selection belongs to another session.")

    def close(self):
        if not self.closed:
            self.closed = True; self.ws.close(); self.full.clear(); self.previews.clear(); self.ann.clear()

    def sync(self, owner: str, contract: dict, cal, sources: dict) -> list:
        """Compare the current inputs with those the work was done under and DISCARD what no longer matches: a different calibration discards everything; a replaced or expired video
        discards its viewpoint's frames and boundaries.  Returns visitor-readable notes."""
        self.assert_owner(owner)
        new = RF.make_binding(owner, contract, dict(sources)); added = []
        if new.calibration != self.binding.calibration:
            self._discard_all(); added.append("Your calibration changed, so your frame selections and boundaries were discarded.")
        else:
            old, cur = {s: (h, u) for s, h, u in self.binding.videos}, {s: (h, u) for s, h, u in new.videos}
            for v in VIEWPOINTS:
                for side in ("left", "right"):
                    slot = f"p{v}_{side}"
                    if slot in old and old[slot] != cur.get(slot):
                        self._discard_viewpoint(v)
                        added.append(f"The {RV.SLOT_INFO[slot][2]} " + ("is no longer available" if slot not in cur else "was replaced") + f", so Viewpoint {v} frames and boundaries were discarded.")
                        break
        self.cal, self.contract, self.sources, self.binding = cal, contract, dict(sources), new
        self.notes.extend(added)
        return added

    def _discard_all(self):
        for v in VIEWPOINTS:
            self._discard_viewpoint(v)
        self.confirmed = False

    def _discard_viewpoint(self, v: int):
        for k in [k for k in self.full if k[1].startswith(f"p{v}_")]:
            del self.full[k]
        for k in [k for k in self.ann if LINE_INFO[k[1]][0] == v]:
            del self.ann[k]
        for k in [k for k in self.previews if k[0].startswith(f"p{v}_")]:
            del self.previews[k]
        self.confirmed = False
        self._prune_job_dirs()

    def removal_impact(self, slot: str) -> list:
        """What removing this video would discard besides the video itself (for the page's confirmation)."""
        v = RV.SLOT_INFO[slot][0]; out = []
        n = sum(1 for (i, line), a in self.ann.items() if LINE_INFO[line][0] == v and (a.points or a.unavailable))
        if self.confirmed or self.full:
            out.append(f"the extracted frames of Viewpoint {v}")
        if n:
            out.append(f"{n} marked boundaries of Viewpoint {v}")
        return out

    # ---- frame sets -----------------------------------------------------------------------------------
    @property
    def n_sets(self) -> int:
        return len(self.sets)

    def shortest(self) -> int:
        return min(self.source(s).frame_count for s in SLOTS)

    def auto_select(self, n_sets: int = DEFAULT_SETS) -> list:
        """Propose ``n_sets`` frame numbers spread over the shortest video (the same number in all six videos).  A proposal is not verified: the visitor checks the previews."""
        n = int(n_sets)
        if not MIN_SETS <= n <= MAX_SETS:
            raise ThreeViewError(f"Choose from {MIN_SETS} to {MAX_SETS} frame sets.")
        count = self.shortest(); fr = DEFAULT_FRACTIONS if n == 3 else tuple((k + 1) / (n + 1) for k in range(n))
        idx = [min(count - 1, max(0, int(round(f * count)))) for f in fr]
        if len(set(idx)) < n:
            raise ThreeViewError("The videos are too short to choose different frames.")
        self._reset_sets([{s: ix for s in SLOTS} for ix in idx])
        return idx

    def _reset_sets(self, sets: list):
        self.sets = sets; self.full.clear(); self.ann.clear(); self.confirmed = False; self._prune_job_dirs()

    def set_frame(self, i: int, index: int):
        """Use frame number ``index`` in all six videos for frame set ``i``.  That set's extracted frames and boundaries are discarded."""
        self._check_set(i)
        for s in SLOTS:
            self._check_index(s, index)
        self._clear_set(i); self.sets[i] = {s: int(index) for s in SLOTS}

    def set_slot_frame(self, i: int, slot: str, index: int):
        """Advanced: a different frame number for ONE video (left/right of a viewpoint need not have the same number).  The set's boundaries are discarded."""
        self._check_set(i)
        if slot not in SLOTS:
            raise ThreeViewError("Unknown video.")
        self._check_index(slot, index); self._clear_set(i); self.sets[i][slot] = int(index)

    def _check_set(self, i: int):
        if isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < self.n_sets:
            raise ThreeViewError("Unknown frame set.")

    def _clear_set(self, i: int):
        for k in [k for k in self.full if k[0] == i]:
            del self.full[k]
        for k in [k for k in self.ann if k[0] == i]:
            del self.ann[k]
        self.confirmed = False

    def duplicate_frames(self) -> list:
        """The same frame of one video used by two frame sets makes those sets correlated, not independent (a warning, not a refusal)."""
        seen, out = {}, []
        for i, st in enumerate(self.sets):
            for s, ix in st.items():
                k = (s, ix)
                if k in seen and seen[k] != i:
                    out.append(f"{RV.SLOT_INFO[s][2]}: frame {ix} is used by frame sets {seen[k] + 1} and {i + 1}; those sets are correlated, not independent.")
                seen.setdefault(k, i)
        return list(dict.fromkeys(out))

    def preview_set(self, i: int) -> dict:
        """Previews (display only) of the six frames of set ``i``.  ONE sandboxed pass per video for the frames it still lacks.  Returns slot -> preview record."""
        self._check_set(i)
        todo = {s: [self.sets[i][s]] for s in SLOTS if (s, self.sets[i][s]) not in self.previews}
        for s, idx in todo.items():
            self.preview(s, idx[0])
        return {s: self.preview(s, self.sets[i][s]) for s in SLOTS}

    def extract_frames(self) -> int:
        """Decode the chosen frames at full resolution (one bounded sandboxed pass per video; lossless; read back and fingerprinted).  A frame whose size differs from the accepted calibration is
        refused (nothing is resized).  Marks the frame sets confirmed.  Returns how many frames were written."""
        W, H = (int(v) for v in self.cal.image_size); todo = {}
        for i, st in enumerate(self.sets):
            for s, ix in st.items():
                cur = self.full.get((i, s))
                if cur is None or cur["index"] != ix:
                    todo.setdefault(s, []).append((i, ix))
        n = 0
        for s, items in todo.items():
            r = self._job("extract", s, full_indices=sorted({ix for _, ix in items}))
            try:
                for i, ix in items:
                    rec = r["written"][f"full_{ix}"]; g = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
                    if g is None:
                        raise ThreeViewError("A decoded frame could not be read back.")
                    if (g.shape[1], g.shape[0]) != (W, H):
                        raise ThreeViewError(f"Frame {ix} of the {RV.SLOT_INFO[s][2]} is {g.shape[1]} × {g.shape[0]} px but the accepted calibration is {W} × {H} px. "
                                             "Camera intrinsics are not rescaled and nothing is resized: record at the calibrated resolution.")
                    self.full[(i, s)] = dict(path=rec["path"], index=ix, fingerprint=V.frame_fingerprint(g), wh=(W, H), job_dir=r["job_dir"], slot=s); n += 1
                    del g
            except BaseException:
                self.ws.drop(r["job_dir"])
                for k in [k for k, v in self.full.items() if v["job_dir"] == r["job_dir"]]:
                    del self.full[k]
                raise
        self._prune_job_dirs(); self.confirmed = all((i, s) in self.full for i in range(self.n_sets) for s in SLOTS)
        return n

    # ---- annotations ----------------------------------------------------------------------------------
    def state(self, i: int, line: str) -> LineState:
        self._check_set(i)
        if line not in LINE_INFO:
            raise ThreeViewError("Unknown boundary.")
        return self.ann.setdefault((i, line), LineState())

    def _frame_of(self, i: int, line: str) -> dict:
        rec = self.full.get((i, slot_of(line)))
        if rec is None:
            raise ThreeViewError("Confirm the frame selection first: this frame has not been extracted yet.")
        return rec

    def _frame_size(self):
        return tuple(int(v) for v in self.cal.image_size)

    def click(self, i: int, line: str, x: float, y: float) -> str:
        """Add one point (original full-resolution distorted pixels).  Returns 'first' / 'second'.  A third click is refused: undo or reset first."""
        rec = self._frame_of(i, line); st = self.state(i, line); W, H = self._frame_size()
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and np.isfinite([x, y]).all()) or not (0 <= x <= W and 0 <= y <= H):
            raise ThreeViewError("That point is outside the frame.")
        if st.unavailable:
            raise ThreeViewError("This boundary is marked as unavailable. Mark it as available again to draw it.")
        if len(st.points) >= 2:
            raise ThreeViewError("This line already has two points. Undo the last point or reset the line first.")
        if len(st.points) == 1 and float(np.hypot(x - st.points[0][0], y - st.points[0][1])) < MIN_LINE_LENGTH_PX:
            raise ThreeViewError(f"The second point is too close to the first (at least {MIN_LINE_LENGTH_PX:g} px apart). Click farther along the line.")
        st.points.append([float(x), float(y)]); st.revision += 1; st.fingerprint = rec["fingerprint"]
        return "first" if len(st.points) == 1 else "second"

    def undo(self, i: int, line: str):
        st = self.state(i, line)
        if st.points:
            st.points.pop(); st.revision += 1

    def reset(self, i: int, line: str):
        st = self.state(i, line); st.points = []; st.unavailable = False; st.revision += 1

    def set_unavailable(self, i: int, line: str, flag: bool = True):
        """Mark a boundary as not visible in this frame (clears its points).  An unavailable VERTICAL corner makes the whole frame set unusable."""
        st = self.state(i, line); st.points = []; st.unavailable = bool(flag); st.revision += 1

    def copy_points(self, src_set: int, dst_set: int, line: str):
        """Use another frame set's line as the starting line (the visitor must check it: the cameras may have moved)."""
        self._check_set(src_set); self._check_set(dst_set)
        src = self.state(src_set, line)
        if len(src.points) != 2:
            raise ThreeViewError("The other frame set has no marked line to copy.")
        rec = self._frame_of(dst_set, line); dst = self.state(dst_set, line)
        dst.points = [list(p) for p in src.points]; dst.unavailable = False; dst.revision += 1; dst.fingerprint = rec["fingerprint"]

    def line_status(self, i: int, line: str) -> str:
        st = self.ann.get((i, line))
        if st is None: return "empty"
        if st.unavailable: return "unavailable"
        if len(st.points) == 2:
            rec = self.full.get((i, slot_of(line)))
            return "marked" if rec is not None and st.fingerprint == rec["fingerprint"] else "stale"
        return "partial" if st.points else "empty"

    def set_usable(self, i: int) -> bool:
        return all(self.line_status(i, n) == "marked" for n in REQUIRED_LINES)

    def usable_sets(self) -> list:
        return [i for i in range(self.n_sets) if self.set_usable(i)]

    def set_problems(self, i: int) -> list:
        out = []
        for n in REQUIRED_LINES:
            s = self.line_status(i, n)
            if s != "marked":
                out.append(f"{label(n)}: " + {"empty": "not marked", "partial": "only one point", "unavailable": "marked as unavailable", "stale": "its frame changed; mark it again"}[s])
        return out

    def frame_gray(self, i: int, slot: str):
        """The EXACT extracted frame of set ``i`` from upload ``slot``, re-fingerprinted on every load (the shared implementation; its view key is the upload slot here)."""
        return RF.RoomFrameWorkflow.frame_gray(self, i, slot)

    # ---- readiness / hand-off -------------------------------------------------------------------------
    def readiness(self, min_usable_sets: int = Q.DEFAULT_THRESHOLDS.min_usable_frames) -> dict:
        problems, missing = [], []
        if not self.confirmed:
            missing.append("Confirm the frame selection.")
        usable = self.usable_sets()
        if self.confirmed and len(usable) < min_usable_sets:
            missing.append(f"Mark the four vertical wall corners in at least {min_usable_sets} frame sets ({len(usable)} complete).")
        for i in range(self.n_sets):
            if i not in usable and self.confirmed and any(self.line_status(i, n) != "empty" for n in LINE_NAMES):
                problems.append(f"Frame set {i + 1} cannot be used yet: " + "; ".join(self.set_problems(i)))
        for m in self.duplicate_frames():
            problems.append(m)
        return dict(ready=bool(self.confirmed and len(usable) >= min_usable_sets), missing=missing, notes=problems, usable_sets=usable, n_sets=self.n_sets)

    def frame_sets_for_run(self) -> dict:
        """{set number: {(viewpoint, 'L'|'R'): frame number}} of the USABLE frame sets (what ``three_view_wall_b.measure_frame_sets`` reads)."""
        return {i: {(RV.SLOT_INFO[s][0], "L" if RV.SLOT_INFO[s][1] == "left" else "R"): ix for s, ix in self.sets[i].items()} for i in self.usable_sets()}

    def annotation_document(self, only_usable: bool = True) -> dict:
        """The ``wall_b_three_viewpoint_annotations.v1`` document of the marked lines.  ``frame_index`` is the FRAME-SET number (the key the reconstruction looks lines up by); the real
        frame numbers are in ``frame_sets``.  Unavailable lines are simply absent (the horizontal lines are not used by the reconstruction)."""
        W, H = self._frame_size(); sets = self.usable_sets() if only_usable else range(self.n_sets); anns = []
        for i in sets:
            for line in LINE_NAMES:
                if self.line_status(i, line) != "marked":
                    continue
                v, cam, ori, req, bd, role = LINE_INFO[line]
                anns.append(dict(viewpoint=f"V{v}", camera=cam, video=slot_of(line), frame_index=int(i), video_frame=int(self.sets[i][slot_of(line)]), boundary=bd, orientation=ori, role=role,
                                 points=[list(p) for p in self.ann[(i, line)].points]))
        return dict(schema=T.ANNOTATION_SCHEMA, coordinate_space=f"original {W}x{H} frame, lens-distorted pixels", annotations=anns,
                    frame_sets={str(i): dict(self.sets[i]) for i in sets}, notes=["V2's horizontal line is stored but not used by the reconstruction."])

    def annotations(self) -> "T.Annotations":
        return T.parse_annotations(self.annotation_document(), self._frame_size())

    def fingerprint(self) -> dict:
        """Digest of exactly what a measurement would use (no pixels, no paths)."""
        sets = [dict(set=i, frames={s: self.sets[i][s] for s in SLOTS}, fp={s: self.full[(i, s)]["fingerprint"] for s in SLOTS if (i, s) in self.full}) for i in self.usable_sets()]
        anns = [(i, n, [list(map(float, p)) for p in self.ann[(i, n)].points]) for i in self.usable_sets() for n in LINE_NAMES if self.line_status(i, n) == "marked"]
        return dict(sets=hashlib.sha256(json.dumps(sets, sort_keys=True).encode()).hexdigest(), annotations=hashlib.sha256(json.dumps(anns, sort_keys=True).encode()).hexdigest())
