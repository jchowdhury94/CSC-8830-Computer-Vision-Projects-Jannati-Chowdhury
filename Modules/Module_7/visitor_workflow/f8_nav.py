"""F8.31D-2: page navigation row and scroll-to-top for the integrated visitor app (H2, M1).

The row has the same structure and shared stylesheet as every Module 7 page (``home_navigation``: the compact back arrow, the Home button, a breadcrumb), so no style is invented.
Both buttons call ``VisitorSession.go``, which re-checks every readiness rule: navigation can never skip a step whose requirements are not met, and it never discards progress
(calibration, uploads, frame selections and annotations are only discarded by the actions that really replace them: a new calibration, Start over, Lock).
"Start over" and "Lock this session" are separate buttons at the bottom of the page and are not part of this row.
"""
from __future__ import annotations

import streamlit as st

import f8_visitor_service as S
import f8_walls as W
import home_navigation as HN

ROOT_LABEL = "Measure a Room"
# Pin the dark navigation colours: the shared stylesheet picks its text colour from the OS preference, but this app is always dark.
_DARK_PIN = "<style>:root:root:root { --nav-muted: #9AA1AC; --nav-text: #FFFFFF; }</style>"


def back_target(sess):
    """The previous step for the current one, or None on the first page.  Pure routing: it grants nothing (``go`` still validates)."""
    step = sess.step
    if getattr(sess, "active_wall", "A") == "B":
        return {"room": "wallA", "frames": "room", "measure": "frames"}.get(step, "wallA")     # the second wall's first page leads back to the first wall's results
    if step in ("setup", "import", "ready"):
        return "home"
    if step == "review":
        rev = sess.review
        return "setup" if sess.runner is not None else ("import" if rev is not None and rev.kind != "f8_recovered" else "home")
    if step in ("room", "frames", "measure") and sess.accepted is None:
        return "home"                                   # no accepted calibration: the only place to go is the start
    return {"room": "ready", "frames": "room", "measure": "frames"}.get(step)


def blocked_reason(sess):
    """Why the navigation row is disabled right now (shown as the button help), or None."""
    if sess.measurement.running:
        return "A measurement is running. Cancel it or wait for it to finish."
    if sess.busy and sess.step == "setup":
        return "A calibration is running. Cancel it or wait for it to finish."
    return None


def _nav(sess, step):
    try:
        if step == "wallA":                               # back to the first wall's results (nothing is discarded)
            W.switch_wall(sess, st.session_state, "A", "measure")
        elif sess.active_wall == "B" and step not in ("room", "frames", "measure"):
            W.switch_wall(sess, st.session_state, "A", step)          # Home etc. act on the shared calibration, i.e. from the first wall
        else:
            sess.go(step)
    except S.ServiceError as e:
        st.session_state["f8i_msg"] = str(e)


def render_nav(sess, current_label):
    st.html(HN._CSS + _DARK_PIN)
    back = back_target(sess)
    why = blocked_reason(sess)
    with st.container(key="nav_row", horizontal=True, vertical_alignment="center"):
        with st.container(key="nav_arrow"):
            st.button("Go back", key="f8i_nav_back", icon=":material/arrow_back:", on_click=_nav, args=(sess, back or "home"), disabled=back is None or why is not None, help=why)
        with st.container(key="nav_home"):
            st.button("Home", key="f8i_nav_home", icon=":material/home:", on_click=_nav, args=(sess, "home"), disabled=sess.step == "home" or why is not None,
                      help=why or "Back to the first page. Your calibration and uploads are kept.")
        with st.container(key="nav_current"):
            st.markdown(current_label)


def scroll_to_top_if_new_view(ss, view_key):
    """One-shot: when the page (or the frames sub-step) changes, scroll to the top.  Nothing is emitted on an ordinary rerun, so scrolling within a page is never disturbed."""
    if ss.get("f8i_last_view") == view_key:
        return
    ss["f8i_last_view"] = view_key
    n = ss["f8i_scroll_n"] = ss.get("f8i_scroll_n", 0) + 1
    with st.container(key="nav_scroll_reset"):
        st.html(HN._SCROLL_TOP_JS.replace("__N__", str(n)), unsafe_allow_javascript=True)
