"""Shared top navigation row (Back left, Home right) and Back button for every module page and nested view (the sidebar
navigation is hidden app-wide)."""
import streamlit as st

ACCENT = "#C9363E"
ACCENT_DARK = "#A62A31"
HOME_PAGE_KEY = "_home_page"  # the Home st.Page, registered by home.py on every run
BUTTON_WIDTH = "110px"
BUTTON_HEIGHT = "40px"

# Back and Home are two identical 110 x 40 buttons in one horizontal st.container, vertically centred, with Home pushed to
# the right edge (margin-left: auto) so a module homepage with no Back still shows Home top-right. Every page-navigation Back
# (top and bottom) is wrapped in a "navback_<key>" container so only those buttons get the shared style; workflow buttons such
# as "Back to Scale" are left untouched. The row is in the normal flow, so nothing is overlapped and nothing is fixed-position.
_BUTTON = """
    width: {w}; min-width: {w}; max-width: {w}; height: {h}; min-height: {h}; padding: 0 0.5rem; gap: 0.3rem;
    border-radius: 8px; transition: background-color 0.15s ease, border-color 0.15s ease;
"""
_TEXT = "font-size: 0.9rem; font-weight: 600; line-height: 1; margin: 0;"
_CSS = f"""<style>
[data-testid="stLayoutWrapper"]:has(> .st-key-nav_scroll_reset), .st-key-nav_scroll_reset {{ display: none; }}
.st-key-nav_row {{ gap: 0.5rem; flex-wrap: nowrap; }}
.st-key-nav_row > div {{ flex: 0 0 auto; width: auto; }}
.st-key-nav_row > div:has(> .st-key-nav_home) {{ margin-left: auto; }}
.st-key-nav_home, [class*="st-key-navback_"] {{ width: {BUTTON_WIDTH}; }}
[class*="st-key-navback_"] button, .st-key-nav_home button {{ {_BUTTON.format(w=BUTTON_WIDTH, h=BUTTON_HEIGHT)} }}
[class*="st-key-navback_"] button p, .st-key-nav_home button p {{ {_TEXT} }}
[class*="st-key-navback_"] button {{
    background: transparent; border: 1px solid rgba(255, 255, 255, 0.45); color: #FFFFFF;
}}
[class*="st-key-navback_"] button p {{ color: #FFFFFF; }}
[class*="st-key-navback_"] button:hover, [class*="st-key-navback_"] button:active {{
    background: rgba(255, 255, 255, 0.1); border-color: #FFFFFF; color: #FFFFFF;
}}
.st-key-nav_home button {{ border: 1px solid {ACCENT}; background: {ACCENT}; color: #FFFFFF; }}
.st-key-nav_home button p, .st-key-nav_home button span {{ color: #FFFFFF; }}
.st-key-nav_home button:hover, .st-key-nav_home button:active {{ background: {ACCENT_DARK}; border-color: {ACCENT_DARK}; color: #FFFFFF; }}
[class*="st-key-navback_"] button:focus-visible, .st-key-nav_home button:focus-visible {{
    outline: 3px solid {ACCENT}; outline-offset: 2px; box-shadow: 0 0 0 2px #FFFFFF;
}}
@media (prefers-color-scheme: light) {{
    :root:not([data-theme="dark"]) [class*="st-key-navback_"] button {{ border-color: rgba(20, 23, 28, 0.45); color: #14171C; }}
    :root:not([data-theme="dark"]) [class*="st-key-navback_"] button p {{ color: #14171C; }}
    :root:not([data-theme="dark"]) [class*="st-key-navback_"] button:hover {{ background: rgba(20, 23, 28, 0.08); border-color: #14171C; }}
}}
</style>"""


LAST_VIEW_KEY = "_nav_last_view"
VIEW_COUNT_KEY = "_nav_view_count"
# Streamlit keeps the scroll offset of the main container across reruns of the same script, so a view that is a little taller
# than the viewport opens part-way down after Back. This one-shot script resets it and is emitted only when the view changes.
_SCROLL_TOP_JS = """<script>
(function () {  // view change #__N__ (a different body each time, so the element is re-run instead of being kept as is)
  var top = function () {
    var main = document.querySelector('[data-testid="stMain"]');
    if (main) { main.scrollTo({top: 0, left: 0, behavior: "instant"}); }
    window.scrollTo(0, 0);
  };
  top();
  setTimeout(top, 120);
})();
</script>"""


# Module number -> the session-state key that selects its current view, and the value of its homepage. Only these keys (and
# the UI-only workflow selector below) are touched by a reset; uploads, calibrations, results and experiment data are not.
MODULE_VIEW_KEYS = {"2": "m2_page", "3": "m3_page", "4": "module4_section", "6": "module6_section", "7": "module7_part"}
HOME_VIEW = "home"
# Temporary visitor workflow selection (Module 2's "Load Saved Calibration" / "Calibrate New Camera" choice), UI only.
VISITOR_WORKFLOW_KEYS = {"2": ("m2_cal_workflow",)}


def reset_module_navigation(modules=None):
    """Send the given modules (default: all five) back to their homepage and forget the last-shown view, so the next page opens
    at the top. Called only at the navigation boundaries: the Home button and the dashboard (page and cards)."""
    ss = st.session_state
    for module in (MODULE_VIEW_KEYS if modules is None else modules):
        ss[MODULE_VIEW_KEYS[module]] = HOME_VIEW
        for key in VISITOR_WORKFLOW_KEYS.get(module, ()):
            ss[key] = None
    ss.pop(LAST_VIEW_KEY, None)


def _go_home():
    reset_module_navigation()
    st.switch_page(st.session_state[HOME_PAGE_KEY])


def back_button(key, on_click, args=()):
    """One page-navigation Back button (label exactly "Back"); the widget key is the caller's own, unchanged. Styled by the
    CSS nav_row injects, which runs first on every view."""
    with st.container(key=f"navback_{key}"):
        st.button("Back", key=key, on_click=on_click, args=args)


def nav_row(back=None, page_css="", view=None):
    """Render the single top row for the current view. Call once per run, at the top of a module's page dispatcher.

    back is None on a module homepage (Home only) or (key, on_click, args) for a nested view. page_css is the module's own
    page-level <style> block; it shares this call's single style element because every extra element above the row would add
    a flow gap and push the row down.

    view identifies the view being shown (for example "m2:experiment"). The page opens at the top only when it differs from
    the view of the previous run, so reruns caused by widgets, uploads or calculations never move the scroll position."""
    ss = st.session_state
    new_view = view is not None and ss.get(LAST_VIEW_KEY) != view
    if view is not None:
        ss[LAST_VIEW_KEY] = view
    if new_view:
        ss[VIEW_COUNT_KEY] = ss.get(VIEW_COUNT_KEY, 0) + 1
    st.html(_CSS + page_css)
    with st.container(key="nav_row", horizontal=True, vertical_alignment="center"):
        if back is not None:
            key, on_click, args = back
            back_button(key, on_click, args)
        with st.container(key="nav_home"):
            st.button("Home", key="nav_home_button", icon=":material/home:", on_click=_go_home)
    if new_view:
        # After the row, in a wrapper hidden by CSS: a script element would otherwise add a flow gap on this first run only.
        with st.container(key="nav_scroll_reset"):
            st.html(_SCROLL_TOP_JS.replace("__N__", str(ss[VIEW_COUNT_KEY])), unsafe_allow_javascript=True)
