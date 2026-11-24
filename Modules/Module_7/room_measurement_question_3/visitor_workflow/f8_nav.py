"""F8.31D-2: page navigation row and scroll-to-top for the integrated visitor app (H2, M1).

Breadcrumb: Home > Module 7 > Run Your Own Experiment > Room Measurement > Camera Calibration > (current step).  Back (on the first page), Home and the first three
crumbs are plain same-tab links to the main website (no target attribute, so the browser stays in this tab); the website's own pages are on another port, so a link is the
only way back.  Back on every later page, and the Camera Calibration crumb, are the buttons that move inside this application.

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
import local_mode

ROOT_LABEL = "Measure a Room"
CALIBRATION_LABEL = "Camera Calibration"
SITE_LEVELS = (("Module 7", "module-7", "home"), ("Run Your Own Experiment", "module-7", "run_own"), ("Room Measurement", "module-7", "3_visitor"))   # (label, website page, ?view=)
# Pin the dark navigation colours: the shared stylesheet picks its text colour from the OS preference, but this app is always dark.
_DARK_PIN = "<style>:root:root:root { --nav-muted: #9AA1AC; --nav-text: #FFFFFF; }</style>"
def _mask(path):
    """A white-on-currentColor icon drawn with a CSS mask (st.html strips inline <svg>)."""
    return ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='" + path + "'/%3E%3C/svg%3E\")")


_ARROW_MASK = _mask("M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z")
_HOME_MASK = _mask("M12 5.69l5 4.5V18h-2v-6H9v6H7v-7.81l5-4.5M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3z")
# Link versions of the shared navigation row's buttons (home_navigation._CSS styles buttons only): same sizes, colours and separators.
_LINK_CSS = f"""<style>
.st-key-nav_row a.nl {{ display: inline-flex; align-items: center; gap: 0.25rem; box-sizing: border-box; text-decoration: none; cursor: pointer; border-radius: 6px;
    padding: 0.15rem 0.1rem; min-height: 1.75rem; font-size: 0.85rem; font-weight: 500; line-height: 1.2; color: var(--nav-muted); }}
.st-key-nav_row [data-testid="stHtml"] {{ width: auto; }}
.st-key-nav_row [class*="st-key-nav_crumb_"] a.nl:hover {{ color: var(--nav-text); text-decoration: underline; }}
.st-key-nav_row .st-key-nav_home a.nl {{ color: {HN.ACCENT}; font-weight: 600; }}
.st-key-nav_row .st-key-nav_home a.nl:hover {{ color: {HN.ACCENT_DARK}; text-decoration: underline; }}
.st-key-nav_row .st-key-nav_arrow a.nl {{ width: 2rem; height: 2rem; padding: 0; justify-content: center; color: var(--nav-text); border: 1px solid rgba(154, 161, 172, 0.55);
    border-radius: 8px; transition: background-color 0.15s ease, border-color 0.15s ease; }}
.st-key-nav_row .st-key-nav_arrow a.nl:hover {{ background: rgba(154, 161, 172, 0.18); border-color: var(--nav-text); }}
.st-key-nav_row a.ic-arrow {{ --ic: {_ARROW_MASK}; }}
.st-key-nav_row a.ic-home {{ --ic: {_HOME_MASK}; }}
.st-key-nav_row a.nl.ic::before {{ content: ""; flex: 0 0 auto; background-color: currentColor; -webkit-mask: var(--ic) center / contain no-repeat; mask: var(--ic) center / contain no-repeat; }}
.st-key-nav_row .st-key-nav_arrow a.nl.ic::before {{ width: 1.2rem; height: 1.2rem; }}
.st-key-nav_row .st-key-nav_home a.nl.ic::before {{ width: 1.05rem; height: 1.05rem; }}
.st-key-nav_row a.nl:focus-visible {{ outline: 2px solid {HN.ACCENT}; outline-offset: 2px; }}
</style>"""


def site_link(path="", view=None):
    """URL of a website page, always on the loopback address and the launcher's website port."""
    return local_mode.website_url(path, view)


def _link(href, label=None, aria=None, icon=None):
    """A plain same-tab link (no target attribute)."""
    return f'<a class="nl{" ic ic-" + icon if icon else ""}" href="{href}"' + (f' aria-label="{aria}"' if aria else "") + f">{label or ''}</a>"


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
    st.html(HN._CSS + _DARK_PIN + _LINK_CSS)
    first = sess.step == "home"
    back = back_target(sess)
    why = blocked_reason(sess)
    with st.container(key="nav_row", horizontal=True, vertical_alignment="center"):
        with st.container(key="nav_arrow"):
            if first:                                    # first page: Back leaves this application for the Room Measurement page of the website (same tab)
                st.html(_link(site_link("module-7", "3_visitor"), aria="Go back", icon="arrow"))
            else:
                st.button("Go back", key="f8i_nav_back", icon=":material/arrow_back:", on_click=_nav, args=(sess, back or "home"), disabled=back is None or why is not None, help=why)
        with st.container(key="nav_home"):
            st.html(_link(site_link(), "Home", icon="home"))                         # the website's homepage, not this application's first page
        for index, (label, path, view) in enumerate(SITE_LEVELS):
            with st.container(key=f"nav_crumb_{index}"):
                st.html(_link(site_link(path, view), label))
        if first:
            with st.container(key="nav_current"):
                st.markdown(CALIBRATION_LABEL)
        else:
            with st.container(key=f"nav_crumb_{len(SITE_LEVELS)}"):
                st.button(CALIBRATION_LABEL, key="f8i_nav_calibration", on_click=_nav, args=(sess, "home"), disabled=why is not None,
                          help=why or "Back to the calibration choices. Your calibration and uploads are kept.")
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
