"""Stage F1 annotation prototype: presentation-independent core (NO Streamlit import).

* ``View`` / viewport maths: zoom and pan are done by CROPPING the full-resolution frame, never by permanently resizing it, so a displayed pixel maps
  back to the original 2160 x 3840 frame with a known, exactly invertible transform.
* ``render_view``: draws the crop, the two clicked points, the clicked line and the refined fitted line.
* ``AnnotationSession``: collects exactly two points per required boundary and calls the EXISTING backend -- ``visitor_measurement.evaluate_set`` (Stage D
  refinement) and ``visitor_measurement.gate_set`` / ``Confirmation`` (the Stage E2 identity, digest and warning rules).  No line fitting, no confirmation
  format and no annotation schema is defined here.

Coordinates
-----------
Annotation points are FULL-RESOLUTION, lens-DISTORTED frame pixels, OpenCV convention (pixel centres at integers) = Stage D coordinate space ``distorted_full``.
Display coordinates are continuous, edge-aligned, origin at the top-left of the displayed crop:  full = X0 + u / sx - 0.5  and  u = (full + 0.5 - X0) * sx.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np

from room_measurement import visitor_annotations as V, visitor_measurement as VM

ZOOM_MIN, ZOOM_MAX = 1.0, 32.0
DEFAULT_MAX_CANVAS = (720, 900)
SOURCE = "local_annotation_ui_prototype"
BOUNDARY_ORDER = tuple((n[:2], n.split("_")[1], n) for n in VM.REQUIRED_LINES)        # (view_key, boundary_type, line name): the verified Phase 1 order, top_variant 0
COLORS = dict(clicked=(255, 190, 0), refined=(40, 255, 110), manual=(255, 40, 40), point=(255, 255, 255), point_edit=(255, 80, 80))     # RGB


# ------------------------------------------------------------------ association (checked through the backend tables, never assumed)
def association(line_name: str) -> dict:
    """view key, placement, camera and boundary of a required line, verified against Stage D's own tables."""
    if line_name not in VM.REQUIRED_LINES: raise ValueError(f"{line_name!r} is not a required boundary")
    vk = line_name[:2]; bt = line_name.split("_")[1]; placement, camera = V.VIEW_PLACEMENT[vk]
    if V.VIEW_KEY[(placement, camera)] != vk or bt not in V.CAMERA_BOUNDARIES[camera]: raise ValueError(f"{line_name}: the {camera} camera of {placement} has no {bt} boundary")
    return dict(view_key=vk, placement=placement, camera=camera, boundary_type=bt, line_name=line_name)


# ------------------------------------------------------------------ viewport
@dataclass(frozen=True)
class ViewState:
    """The only zoom / pan state: zoom factor (1 = whole frame) and the full-resolution point at the centre of the view."""
    zoom: float = 1.0
    cx: float | None = None
    cy: float | None = None


@dataclass(frozen=True)
class View:
    full_wh: tuple
    zoom: float
    crop: tuple          # integer full-resolution crop box (X0, Y0, X1, Y1), edge coordinates, X1/Y1 exclusive
    shown_wh: tuple      # size of the rendered crop in display pixels
    canvas_wh: tuple     # size of the whole frame shown at zoom 1 (fits the maximum canvas)

    @property
    def sx(self): return self.shown_wh[0] / (self.crop[2] - self.crop[0])
    @property
    def sy(self): return self.shown_wh[1] / (self.crop[3] - self.crop[1])

    def to_full(self, u, v):
        return (self.crop[0] + np.asarray(u, float) / self.sx - 0.5, self.crop[1] + np.asarray(v, float) / self.sy - 0.5)

    def to_view(self, x, y):
        return ((np.asarray(x, float) + 0.5 - self.crop[0]) * self.sx, (np.asarray(y, float) + 0.5 - self.crop[1]) * self.sy)

    def contains(self, x, y) -> bool:
        u, v = self.to_view(x, y); return bool(0 <= u <= self.shown_wh[0] and 0 <= v <= self.shown_wh[1])

    def click_to_full(self, click: Mapping) -> tuple:
        """Component click -> full-resolution point.  ``click`` = {x, y, width, height} as returned by streamlit-image-coordinates: x, y are integer offsets inside
        the image as the BROWSER shows it (width, height); rescale to the rendered size first (identity when the component is not shrunk), centre in the pixel."""
        u = (float(click["x"]) + 0.5) * self.shown_wh[0] / float(click["width"]); v = (float(click["y"]) + 0.5) * self.shown_wh[1] / float(click["height"])
        x, y = self.to_full(u, v); return float(x), float(y)


def canvas_size(full_wh, max_wh=DEFAULT_MAX_CANVAS) -> tuple:
    W, H = full_wh; base = min(max_wh[0] / W, max_wh[1] / H); return max(1, round(W * base)), max(1, round(H * base))


def clamp_state(full_wh, st: ViewState) -> ViewState:
    W, H = full_wh; z = float(min(max(st.zoom, ZOOM_MIN), ZOOM_MAX)); vw, vh = W / z, H / z
    cx = W / 2 if st.cx is None else st.cx; cy = H / 2 if st.cy is None else st.cy
    return ViewState(z, float(min(max(cx, vw / 2), W - vw / 2)), float(min(max(cy, vh / 2), H - vh / 2)))


def make_view(full_wh, st: ViewState = ViewState(), max_wh=DEFAULT_MAX_CANVAS) -> View:
    W, H = full_wh; st = clamp_state(full_wh, st); vw, vh = W / st.zoom, H / st.zoom; cw, ch = canvas_size(full_wh, max_wh)
    X0 = int(min(max(round(st.cx - vw / 2), 0), W - 1)); Y0 = int(min(max(round(st.cy - vh / 2), 0), H - 1)); X1 = int(min(W, X0 + max(1, round(vw)))); Y1 = int(min(H, Y0 + max(1, round(vh))))
    s = cw / vw; shown = (max(1, round((X1 - X0) * s)), max(1, round((Y1 - Y0) * s)))
    return View((W, H), st.zoom, (X0, Y0, X1, Y1), shown, (cw, ch))


def zoom_by(full_wh, st: ViewState, factor: float, anchor=None) -> ViewState:
    """Zoom about the view centre, or about a full-resolution ``anchor`` point (which then stays where it is on screen as far as clamping allows)."""
    st = clamp_state(full_wh, st); z = min(max(st.zoom * factor, ZOOM_MIN), ZOOM_MAX)
    if anchor is None: return clamp_state(full_wh, ViewState(z, st.cx, st.cy))
    ax, ay = anchor; k = st.zoom / z; return clamp_state(full_wh, ViewState(z, ax + (st.cx - ax) * k, ay + (st.cy - ay) * k))


def pan_by(full_wh, st: ViewState, dx_frac: float, dy_frac: float) -> ViewState:
    """Move the view by a fraction of the visible full-resolution rectangle.  Never changes the zoom or the image."""
    st = clamp_state(full_wh, st); W, H = full_wh; return clamp_state(full_wh, ViewState(st.zoom, st.cx + dx_frac * W / st.zoom, st.cy + dy_frac * H / st.zoom))


def center_on(full_wh, st: ViewState, point) -> ViewState:
    return clamp_state(full_wh, ViewState(st.zoom, float(point[0]), float(point[1])))


# ------------------------------------------------------------------ rendering
def line_polyline(line: Mapping, K, D, n=80) -> np.ndarray:
    """A Stage D line (undistorted pixels, 'v': x = a*y + b or 'h': y = a*x + b) over its span, mapped into the DISTORTED frame the visitor looks at."""
    u = np.linspace(line["span"][0], line["span"][1], n); w = line["a"] * u + line["b"]; und = np.c_[w, u] if line["kind"] == "v" else np.c_[u, w]
    return V.distort_pixels(und, np.asarray(K, float), np.asarray(D, float))


def _poly(img, view, pts_full, color, thickness, dashed=False):
    p = np.c_[view.to_view(pts_full[:, 0], pts_full[:, 1])]; q = np.round(p * 16).astype(np.int32)
    if not dashed: cv2.polylines(img, [q.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA, shift=4); return
    for i in range(0, len(q) - 1, 4): cv2.polylines(img, [q[i:i + 2].reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA, shift=4)


def render_view(gray: np.ndarray, view: View, points=(), clicked_poly=None, refined_poly=None, edit_index=None, manual_pts=None) -> np.ndarray:
    """RGB image of the crop at the view's scale with overlays.  ``gray`` is never modified; points / polylines are FULL-RESOLUTION coordinates."""
    X0, Y0, X1, Y1 = view.crop; crop = gray[Y0:Y1, X0:X1]; interp = cv2.INTER_AREA if view.sx < 1 else cv2.INTER_LINEAR
    img = cv2.cvtColor(cv2.resize(crop, view.shown_wh, interpolation=interp), cv2.COLOR_GRAY2RGB)
    if clicked_poly is not None: _poly(img, view, np.asarray(clicked_poly, float), COLORS["clicked"], 2, dashed=True)
    if refined_poly is not None: _poly(img, view, np.asarray(refined_poly, float), COLORS["refined"], 2)
    if manual_pts is not None and len(manual_pts) == 2: _poly(img, view, np.asarray(manual_pts, float), COLORS["manual"], 2)       # F8.31G: solid red segment between the two SAVED points (full-resolution coordinates)
    for i, p in enumerate(points):
        u, v = (float(c) for c in view.to_view(p[0], p[1]))
        if not (-20 <= u <= view.shown_wh[0] + 20 and -20 <= v <= view.shown_wh[1] + 20): continue
        col = COLORS["point_edit"] if edit_index == i else COLORS["point"]; c = (int(round(u * 16)), int(round(v * 16)))
        cv2.circle(img, c, 9 * 16, (0, 0, 0), 3, cv2.LINE_AA, shift=4); cv2.circle(img, c, 9 * 16, col, 1, cv2.LINE_AA, shift=4)
        for dx, dy in ((14, 0), (-14, 0), (0, 14), (0, -14)): cv2.line(img, (c[0] + dx * 16 // 2, c[1] + dy * 16 // 2), (c[0] + dx * 16, c[1] + dy * 16), col, 1, cv2.LINE_AA, shift=4)
        cv2.putText(img, str(i + 1), (int(u) + 11, int(v) - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(img, str(i + 1), (int(u) + 11, int(v) - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------ plain-language messages (backend codes are always kept alongside)
EXPLANATIONS = {
    "COMPETING_EDGE_NEARBY": "Another nearby edge was detected. Check that the highlighted line follows the intended wall boundary.",
    "WEAK_EDGE": "The selected boundary is difficult to detect. Review the highlighted line.",
    "REFINEMENT_FAILED": "The boundary could not be refined. Try selecting two clearer points.",
    "LARGE_REFINEMENT_DISPLACEMENT": "The fitted line moved a long way from your line. Check that it is on the boundary you meant.",
    "REFINEMENT_AT_WINDOW_LIMIT": "The edge found is at the limit of the search area, so it may be a different edge or too far from your clicks. Place the two points closer to the boundary.",
    "TOO_FEW_EDGE_POINTS": "Only a few edge points supported the line. Select a longer, clearer part of the boundary.",
    "LOW_EDGE_COVERAGE": "The edge was found along only part of your line. Review the highlighted line.",
    "LOW_INLIER_FRACTION": "Many edge points disagreed with the fitted line. Review the highlighted line.",
    "HIGH_FIT_RMS": "The edge points scatter around the fitted line. Review the highlighted line.",
    "LARGE_ANGLE_CHANGE": "The fitted line points in a noticeably different direction than your line. Review it.",
    "EDGE_POINTS_AT_WINDOW_RIM": "Many edge points sit at the edge of the search area; the true boundary may lie outside it.",
    "AMBIGUOUS_EDGE_POLARITY": "Edges of both light/dark directions were found close together. Check that the highlighted line is the intended boundary.",
    "SEARCH_SPAN_CLIPPED": "Part of your line was too close to the image border to be searched.",
    "WINDOW": "The search area lies outside the image.", "SEARCH_WINDOW_OUT_OF_IMAGE": "The search area lies outside the image.",
    "LINE_TOO_SHORT": "The two points are too close together. Select points further apart along the boundary.",
    "DUPLICATE_ENDPOINTS": "The two points are at the same place. Select two different points.",
    "UNREASONABLE_ORIENTATION": "This line does not have the expected direction for this boundary (corners are vertical, ceiling edges near horizontal).",
    "FRAME_FINGERPRINT_MISMATCH": "These points were made on a different image than the one shown. Nothing was substituted; please annotate again.",
    "FRAME_MISMATCH": "The frame identity of this annotation does not match the frame shown.",
}
FAIL_NOTE = "A good fit does not prove this is the wall boundary. Check the green line by eye."


def explain(code: str, backend_message: str = "") -> str:
    return EXPLANATIONS.get(code) or backend_message or code


# ------------------------------------------------------------------ frames
@dataclass(frozen=True)
class FrameRecord:
    set_index: int
    view_key: str
    frame_index: int
    frame_fingerprint: str
    image_wh: tuple
    path: str

    @property
    def placement(self): return V.VIEW_PLACEMENT[self.view_key][0]
    @property
    def camera(self): return V.VIEW_PLACEMENT[self.view_key][1]
    @property
    def label(self): return f"{self.placement.replace('_', ' ')} · {self.camera} camera ({self.view_key}) · frame {self.frame_index}"


class FrameIntegrityError(RuntimeError):
    pass


def load_frame(rec: FrameRecord) -> np.ndarray:
    """The EXACT frame: re-read from its file and re-fingerprinted every time it is loaded; a mismatch raises instead of substituting anything."""
    g = cv2.imread(rec.path, cv2.IMREAD_GRAYSCALE)
    if g is None: raise FrameIntegrityError(f"frame file for set {rec.set_index} {rec.view_key} is missing")
    if (g.shape[1], g.shape[0]) != tuple(rec.image_wh) or V.frame_fingerprint(g) != rec.frame_fingerprint: raise FrameIntegrityError(f"frame set {rec.set_index} {rec.view_key} frame {rec.frame_index} no longer matches its fingerprint")
    return g


# ------------------------------------------------------------------ the annotation session (no Streamlit)
@dataclass
class BoundaryState:
    set_index: int
    line_name: str
    points: list = field(default_factory=list)       # up to two [x, y] in full-resolution distorted pixels
    edit_index: int | None = None                    # point that the next click replaces
    revision: int = 0                                # bumped on EVERY change of the points
    raw: dict | None = None                          # submitted Stage D payload
    result: dict | None = None                       # JSON-able refinement summary
    confirmation: object | None = None               # visitor_measurement.Confirmation
    accepted: bool = False
    _ev: dict | None = None                          # Stage D evaluation (re-gated, never persisted)


class AnnotationSession:
    """One annotation job.  Holds per-(set, boundary) state; all refinement / gating / confirmation goes through the backend."""

    def __init__(self, frames: Mapping, cal, limits: V.AnnotationLimits = V.DEFAULT_LIMITS, job_id: str | None = None, set_index: int = 0, loader: Callable = load_frame):
        self.job_id = job_id or uuid.uuid4().hex[:12]; self.frames = dict(frames); self.cal = cal; self.limits = limits; self.loader = loader; self.set_index = set_index
        self.states: dict = {}; self.position = 0; self._cache: dict = {}; self.sets = sorted({k[0] for k in self.frames})
        for si in self.sets:
            for vk, bt, ln in BOUNDARY_ORDER:
                if (si, vk) not in self.frames: raise ValueError(f"missing frame for set {si} view {vk}")
                a = association(ln); r = self.frames[(si, vk)]
                if (r.placement, r.camera) != (a["placement"], a["camera"]): raise ValueError(f"frame record {si}/{vk} is not the {a['placement']} {a['camera']} frame")
        if set_index not in self.sets: raise ValueError(f"unknown frame set {set_index}")

    # ---- navigation
    @property
    def lines(self) -> tuple: return tuple(b[2] for b in BOUNDARY_ORDER)
    @property
    def line(self) -> str: return self.lines[self.position]
    def progress(self) -> tuple: return self.position + 1, len(self.lines)
    def goto(self, i: int): self.position = int(min(max(i, 0), len(self.lines) - 1))
    def next(self): self.goto(self.position + 1)
    def prev(self): self.goto(self.position - 1)

    def reachable(self, i: int) -> bool:
        """A boundary may be opened only if every earlier boundary of the current set is accepted (earlier ones can always be reviewed)."""
        return all(self.state(ln).accepted for ln in self.lines[:i])

    def select_set(self, si: int):
        if si not in self.sets: raise ValueError(f"unknown frame set {si}")
        self.set_index = si; self.position = 0

    def state(self, line: str | None = None, set_index: int | None = None) -> BoundaryState:
        si = self.set_index if set_index is None else set_index; ln = line or self.line
        return self.states.setdefault((si, ln), BoundaryState(si, ln))

    def record(self, line: str | None = None) -> FrameRecord:
        a = association(line or self.line); return self.frames[(self.set_index, a["view_key"])]

    def frame(self, line: str | None = None) -> np.ndarray:
        rec = self.record(line); key = (self.set_index, rec.view_key)
        if key not in self._cache:
            if len(self._cache) >= 2: self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self.loader(rec)
        return self._cache[key]

    # ---- point collection
    def _invalidate(self, st: BoundaryState):
        st.revision += 1; st.raw = None; st.result = None; st._ev = None; st.confirmation = None; st.accepted = False

    def click(self, x: float, y: float) -> str:
        """First two clicks add points; after that a click replaces the point chosen with ``begin_edit`` (otherwise it is ignored).  -> what happened."""
        st = self.state(); W, H = self.record().image_wh
        if not (np.isfinite(x) and np.isfinite(y)) or not (-0.5 <= x <= W - 0.5 and -0.5 <= y <= H - 0.5): return "ignored: outside the image"
        x, y = float(min(max(x, 0), W - 1)), float(min(max(y, 0), H - 1))
        if len(st.points) < 2: st.points.append([x, y]); self._invalidate(st); return f"point {len(st.points)} placed"
        if st.edit_index is None: return "ignored: choose which point to move first"
        st.points[st.edit_index] = [x, y]; moved = st.edit_index; st.edit_index = None; self._invalidate(st); return f"point {moved + 1} moved"

    def begin_edit(self, index: int):
        st = self.state()
        if index not in (0, 1) or index >= len(st.points): raise ValueError("no such point")
        st.edit_index = index

    def clear(self):
        st = self.state(); st.points = []; st.edit_index = None; self._invalidate(st)

    # ---- refinement (Stage D) and gate (Stage E2)
    def can_submit(self) -> bool: return len(self.state().points) == 2

    def build_payload(self, st: BoundaryState) -> dict:
        a = association(st.line_name); rec = self.frames[(st.set_index, a["view_key"])]
        return V.build_annotation(set_index=st.set_index, placement=a["placement"], camera=a["camera"], frame_index=rec.frame_index, boundary_type=a["boundary_type"], image_wh=rec.image_wh, coordinate_space="distorted_full",
                                  points=st.points, annotation_id=f"f1:{self.job_id}:set{st.set_index}:{st.line_name}:r{st.revision}", source=SOURCE, frame_fingerprint=rec.frame_fingerprint)

    def submit(self) -> dict:
        """Run the existing Stage D refinement on the two points of the CURRENT boundary and gate it with the Stage E2 rules.  Refuses with fewer than two points."""
        st = self.state()
        if len(st.points) != 2: raise ValueError("exactly two points are required before refinement")
        a = association(st.line_name); rec = self.frames[(st.set_index, a["view_key"])]; gray = self.frame(st.line_name); vk = a["view_key"]
        raw = self.build_payload(st); fps = {vk: V.frame_fingerprint(gray)}
        ev = VM.evaluate_set([raw], {vk: gray}, fps, {vk: rec.frame_index}, self.cal, st.set_index, self.limits)
        st.raw, st._ev = raw, ev; self._regate(st); return st.result

    def _regate(self, st: BoundaryState):
        confs = [st.confirmation] if st.confirmation is not None else []
        g = VM.gate_set(st._ev, confs); d = g["decisions"][st.line_name]; e = [x for x in st._ev["entries"] if x["line_name"] == st.line_name][0]; r = e["result"]
        req = [q for q in g["confirmation_requests"] if q["line_name"] == st.line_name and q["scope"] == "annotation"]
        issues = [i.to_dict() for i in r.issues if i.status in (V.Status.WARNING, V.Status.FAIL)]
        st.result = dict(status=e["status"].value, decision=d["decision"], decision_reasons=d["reasons"], warning_codes=e["warning_codes"], failure_codes=e["failure_codes"], issues=issues, identity_ok=e["identity"]["ok"], identity_reasons=e["identity"]["reasons"],
                         initial_line=r.initial_line, refined_line=r.refined_line, refined_downstream=r.downstream_line is not None,
                         diagnostics={k: r.diagnostics.get(k) for k in ("displacement_max_px", "angle_change_deg", "fit_rms_px", "n_edge_points", "inlier_fraction", "edge_strength", "competitor_ratio", "competitor_offset_px", "pass_used")},
                         annotation_id=r.annotation_id, annotation_digest=e["annotation_digest"], refined_line_digest=e["refined_line_digest"], frame_fingerprint=e["identity"].get("annotation_fingerprint"),
                         confirmation_request=req[0] if req else None, confirmation_valid=d.get("confirmation_valid"), checks_not_evaluated=d.get("checks_not_evaluated", []))
        st.accepted = d["decision"] in ("ALLOWED", "ALLOWED_CONFIRMED")

    def state_summary(self, st: BoundaryState | None = None) -> str:
        """READY (PASS) / NEEDS_CONFIRMATION / CONFIRMED / BLOCKED / NOT_REFINED / INCOMPLETE."""
        st = st or self.state()
        if len(st.points) < 2: return "INCOMPLETE"
        if st.result is None: return "NOT_REFINED"
        d = st.result["decision"]
        return {"ALLOWED": "READY", "ALLOWED_CONFIRMED": "CONFIRMED", "BLOCKED_UNCONFIRMED_WARNING": "NEEDS_CONFIRMATION"}.get(d, "BLOCKED")

    # ---- confirmation (explicit; WARNING only; FAIL can never be confirmed)
    def can_confirm(self) -> bool:
        st = self.state(); return bool(st.result and st.result["decision"] == "BLOCKED_UNCONFIRMED_WARNING" and st.result["confirmation_request"])

    def confirm(self, confirmed_by: str = "local_ui_visitor", reason: str = "the visitor checked the highlighted line by eye") -> bool:
        st = self.state()
        if not self.can_confirm(): return False
        st.confirmation = VM.Confirmation.from_request(st.result["confirmation_request"], confirmed_by, reason); self._regate(st); return st.accepted

    def accept(self) -> bool:
        """PASS: allow proceeding (no confirmation needed).  Returns False for anything that is not ALLOWED."""
        st = self.state(); return bool(st.result and st.result["decision"] in ("ALLOWED", "ALLOWED_CONFIRMED"))

    def reject(self):
        """Throw the refinement and any confirmation away but keep the clicked points so they can be corrected."""
        st = self.state(); st.revision += 1; st.raw = None; st.result = None; st._ev = None; st.confirmation = None; st.accepted = False

    # ---- results
    def completed(self, set_index: int | None = None) -> list:
        si = self.set_index if set_index is None else set_index
        return [ln for ln in self.lines if (si, ln) in self.states and self.states[(si, ln)].accepted]

    def export(self) -> dict:
        """Accepted annotations (exact Stage D payloads) and their confirmation records, ready for the Stage E2 backend contract.  Nothing is measured here."""
        anns, confs = [], []
        for (si, ln), st in sorted(self.states.items()):
            if st.accepted and st.raw is not None:
                anns.append(st.raw)
                if st.confirmation is not None: confs.append(st.confirmation.to_dict())
        return dict(schema="csc8830.module7.f1_annotation_export.v1", job_id=self.job_id, annotations=anns, confirmations=confs, note="Stage F1 prototype export; no measurement has been run.")
