"""F8.31F: OPTIONAL boundary suggestions for the six ``top_variant = 0`` wall boundaries (no Streamlit import; no reconstruction).

The suggestions come from the EXISTING classical Stage F2 detector (``room_measurement.stage_f2_boundary_detection``: band-pass edge response, sheared-Kadane line search,
confuser and junction ranking).  Nothing new is detected here and no ML/DL is used.  This module only

* runs F2 on the visitor's OWN set-0 frames of the four videos (each video keeps its own frame index),
* converts the best candidate of each boundary into the two points the visitor workflow stores (original full-resolution, lens-DISTORTED pixels), and
* reports honestly what F2 said (DETECTED / UNCERTAIN / NOT_DETECTED and its reasons).

A suggestion is a PROPOSAL.  It is never accepted here: the page shows it on the frame and the visitor presses *Looks Correct* or corrects it with two clicks.  A rank-1 candidate
is not "verified": on the one room F2 was developed on it was the right line in only 4 of 12 cases (outputs/stage_f2_boundary_detection/summary.json), and the AL/AR corners
are capped at UNCERTAIN because the Air camera sees no ceiling.  Whatever F2 returns, the status is carried unchanged so the page can say so.

Nothing here reads an original-room coordinate or annotation.  A failure of F2 (an exception, an empty result, points outside the frame) yields NO suggestion for the affected
boundaries and the manual two-click method stays available.
"""
from __future__ import annotations

import numpy as np

from room_measurement import stage_f2_boundary_detection as F2
from room_measurement import visitor_annotations as V

REQUIRED = ("AL_corner", "PL_corner", "PL_ceilF", "AR_corner", "PR_corner", "PR_ceilF")
VIEW_KEYS = ("AL", "PL", "AR", "PR")


def _none(line: str, status: str, reasons) -> dict:
    return dict(line=line, available=False, status=status, reasons=list(reasons), points_distorted_full=None, line_undistorted=None, f2=None)


def _distort(points_und, K, D):
    return V.distort_pixels(np.asarray(points_und, float).reshape(-1, 2), np.asarray(K, float), np.asarray(D, float))


def suggest(frames: dict, cal) -> dict:
    """``frames``: ``{AL, PL, AR, PR: grayscale full-resolution lens-distorted frame}`` (set 0 of each video).  Returns ``{line: suggestion}`` for the six required boundaries.

    ``suggestion['available']`` is True only when F2 produced a candidate whose two end points lie inside the frame.  ``status`` is F2's own word.  Never raises."""
    missing = [v for v in VIEW_KEYS if v not in frames]
    if missing:
        return {n: _none(n, "NOT_DETECTED", [f"no frame for {', '.join(missing)}"]) for n in REQUIRED}
    try:
        det = F2.detect_set({k: frames[k] for k in VIEW_KEYS}, cal, set_index=0)
    except Exception as e:                                                # F2 is optional: a failure must never stop the manual path
        return {n: _none(n, "NOT_DETECTED", [f"automatic suggestion failed ({type(e).__name__})"]) for n in REQUIRED}
    out = {}
    for name in REQUIRED:
        br = det["boundaries"].get(name)
        if br is None or br.selected_rank is None or not br.candidates:
            out[name] = _none(name, "NOT_DETECTED" if br is None else br.status, (br.reasons if br is not None else []) or ["no candidate line was found"]); continue
        vk = name[:2]; K, D = ((cal.K_left, cal.D_left) if vk[0] == "A" else (cal.K_right, cal.D_right))
        c = br.candidates[br.selected_rank - 1]
        try:
            pts = _distort(c["endpoints_undistorted_full"], K, D)
        except Exception:
            out[name] = _none(name, "NOT_DETECTED", ["the candidate could not be mapped back to the video frame"]); continue
        W, H = (int(v) for v in br.image_wh)
        if not np.all(np.isfinite(pts)) or np.any(pts[:, 0] < 0) or np.any(pts[:, 0] > W - 1) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > H - 1):
            out[name] = _none(name, "NOT_DETECTED", ["the candidate line leaves the frame"]); continue
        out[name] = dict(line=name, available=True, status=br.status, reasons=list(br.reasons), points_distorted_full=[[float(x), float(y)] for x, y in pts],
                         line_undistorted=dict(c["line"]), f2=dict(visible_length_px=c["visible_length_px"], confuser_class=c["confuser_class"], support_fraction=c["support_fraction"],
                                                                 rank_score=c["rank_score"], n_candidates=len(br.candidates)))
    return out


def fingerprints(wf) -> tuple:
    """What the suggestions were computed from: (view, frame index, fingerprint) of set 0 for the four videos.  A suggestion whose key differs from the current one is stale."""
    return tuple((vk, int(wf.full[(0, vk)]["index"]), wf.full[(0, vk)]["fingerprint"]) for vk in VIEW_KEYS if (0, vk) in wf.full)


def suggest_for_workflow(wf) -> dict:
    """Run F2 on the workflow's extracted set-0 frames.  ``{'key': fingerprints, 'by_line': {...}}``; the caller stores it and compares ``key`` with ``fingerprints(wf)``."""
    key = fingerprints(wf)
    if len(key) != 4:
        return dict(key=key, by_line={n: _none(n, "NOT_DETECTED", ["the frames are not available yet"]) for n in REQUIRED})
    frames = {vk: wf.frame_gray(0, vk) for vk in VIEW_KEYS}
    return dict(key=key, by_line=suggest(frames, wf.cal))
