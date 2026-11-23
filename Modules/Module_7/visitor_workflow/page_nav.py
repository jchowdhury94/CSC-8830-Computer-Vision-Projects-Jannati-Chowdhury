"""Page-level navigation (Home / Back) for the visitor workflow when it runs STANDALONE.

It is the same row Module 7 shows on every page (home_navigation.nav_row): the compact "back" arrow, the Home button and the breadcrumb.
It reuses that row's container keys and the shared stylesheet (home_navigation._CSS), so it looks identical and no style is invented.

Why not call nav_row itself: its buttons use st.switch_page() to a page registered by home.py, which does not exist in this standalone
process.  Here Home and Back instead open the REAL routes of the main app, whose base URL is MAIN_APP_URL_ENV (default http://localhost:8501,
what `streamlit run home.py` serves):   Home -> <base>/        Back / "Module 7" -> <base>/module-7

When the workflow is integrated into Pages/app_module7.py, that page already draws nav_row for the view, so the integration calls
render_workflow() and never this module (host_nav=False): Home and Back then use the host's own session-state handling.

Page navigation (this row, top of the page) is separate from the wizard's own Back / Continue buttons (inside the centred column)."""
import json
import os

import streamlit as st

import home_navigation as HN

MAIN_APP_URL_ENV = "F6_MAIN_APP_URL"
DEFAULT_MAIN_APP_URL = "http://localhost:8501"
MODULE7_PATH = "/module-7"          # url_path of Module 7 in home.py's MODULES
LEAVE_KEY = "_f6_leave_to"
CURRENT_LABEL = "Your Own Stereo Videos"   # the same label Module 7's breadcrumb uses for this view


def main_app_url() -> str:
    return (os.environ.get(MAIN_APP_URL_ENV) or DEFAULT_MAIN_APP_URL).rstrip("/")


def home_url() -> str:
    return main_app_url() + "/"


def back_url() -> str:
    return main_app_url() + MODULE7_PATH


def _leave(url):
    st.session_state[LEAVE_KEY] = url


# The standalone app is always dark (run_stage_f6_visitor_app.py passes --theme.base dark), but the shared stylesheet picks its nav colours from the
# OS light/dark preference; pin its own dark values here so the arrow and the current-page label stay readable on a light-mode OS.
_DARK_PIN = "<style>:root:root:root { --nav-muted: #9AA1AC; --nav-text: #FFFFFF; }</style>"


def standalone_nav():
    """Same widget structure as home_navigation.nav_row(trail=(Module 7, current)); nothing else is drawn above the row."""
    st.html(HN._CSS + _DARK_PIN)
    with st.container(key="nav_row", horizontal=True, vertical_alignment="center"):
        with st.container(key="nav_arrow"):
            st.button("Go back", key="nav_arrow_button", icon=":material/arrow_back:", on_click=_leave, args=(back_url(),))
        with st.container(key="nav_home"):
            st.button("Home", key="nav_home_button", icon=":material/home:", on_click=_leave, args=(home_url(),))
        with st.container(key="nav_crumb_0"):
            st.button("Module 7", key="navbtn_0", on_click=_leave, args=(back_url(),))
        with st.container(key="nav_current"):
            st.markdown(CURRENT_LABEL)
    url = st.session_state.pop(LEAVE_KEY, None)
    if url:   # one-shot: the click's rerun opens the main app's route in this tab (the element is hidden by the shared CSS)
        with st.container(key="nav_scroll_reset"):
            st.html(f"<script>window.top.location.href = {json.dumps(url)};</script>", unsafe_allow_javascript=True)
