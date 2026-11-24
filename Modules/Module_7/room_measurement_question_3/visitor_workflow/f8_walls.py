"""Two-wall helpers for the visitor pages (Streamlit session-state side of ``VisitorSession.switch_wall``).

The service swaps the per-wall DATA (uploads, frames, boundaries, measurement record).  The frames page also keeps per-wall WIDGET state in ``st.session_state`` (current boundary, zoom, marking mode,
automatic-selection memo); it is parked with the wall it belongs to and restored when that wall becomes active again, so the two walls' boundary editors never see each other's state.
"""
from __future__ import annotations

import f8_visitor_service as S

UI_PREFIXES = ("f8i_fs_", "f8i_bd_", "f8i_an_", "f8i_rm_pending", "f8i_m_new_ask", "f8i_m_recalc", "f8i_m_err", "f8i_vp_pending", "f8i_tv_")
WALL_NAME = {"A": "First wall", "B": "Second wall"}
SECOND_WALL_ACTION = "Continue to Wall B — Room Length"
SECOND_WALL_HINT = "Upload stereo videos of the perpendicular wall to estimate the room's length."


# Two-step wording (presentation only).  Wall A gives the room's width and height; Wall B (the perpendicular wall) gives its length.
WALL_STEP = {"A": "Step 1 of 2 — Measure Wall A (Room Width)", "B": "Step 2 of 2 — Measure Wall B (Room Length)"}
WALL_HEADING = {"A": "Wall A — Room Width", "B": "Wall B — Room Length"}
WALL_UPLOAD_HINT = {"A": "Upload four stereo videos of the first wall to estimate the room's width and height.",
                    "B": "Upload four stereo videos of the perpendicular wall to estimate the room's length."}
WALL_UPLOAD_HINT_3 = {"B": "Upload six stereo videos (three viewpoints: left end, middle and right end of the wall) to estimate the room's length."}
WALL_A_DONE_TITLE = "Wall A Measured — Step 1 of 2"
WALL_A_DONE_TEXT = "The first wall has been measured. Next, measure the perpendicular wall to estimate the room's length."
WALL_STEPS = ("room", "frames", "measure")                      # the pages that belong to one wall (every earlier page belongs to the shared calibration)


def suffix(sess) -> str:
    """Widget-key suffix: '' for Wall A (its keys are exactly what they always were), '_b' for Wall B (so no widget of one wall can answer for the other)."""
    return "" if getattr(sess, "active_wall", "A") == "A" else "_b"


def switch_wall(sess, ss, wall: str, step: str | None = None):
    """Make ``wall`` active (a no-op when it already is), swapping the frames-page widget state with it, then optionally go to ``step``.  Raises ServiceError (nothing changed) when refused."""
    if wall != sess.active_wall:
        saved = {k: ss.pop(k) for k in list(ss) if str(k).startswith(UI_PREFIXES)}
        sess._ui = saved
        try:
            sess.switch_wall(wall)
        except S.ServiceError:
            ss.update(saved); sess._ui = {}
            raise
        ss.update(sess._ui); sess._ui = {}
    if step is not None:
        sess.go(step)


def start_second_wall(sess, ss):
    """Begin the second wall: keeps Wall A's videos, boundaries, result and the shared calibration; Wall B starts at its own four uploads."""
    switch_wall(sess, ss, "B", "room")


def go(sess, ss, step: str):
    """``sess.go`` that is wall-aware: pages of the shared calibration (home, ready, ...) are reached from the first wall; room / frames / measure stay on the active wall."""
    if sess.active_wall == "B" and step not in ("room", "frames", "measure"):
        switch_wall(sess, ss, "A", step)
    else:
        sess.go(step)
