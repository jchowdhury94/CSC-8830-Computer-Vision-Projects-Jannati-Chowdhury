"""Shared breadcrumb navigation (Home > Module > ... > current page) at the top of every module page and nested view, plus the
bottom Back button (the sidebar navigation is hidden app-wide)."""
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
.st-key-nav_row {{ gap: 0.1rem 0; flex-wrap: wrap; align-items: center; margin: 0 0 0.25rem; }}
.st-key-nav_row > div {{ flex: 0 0 auto; width: auto; max-width: 100%; min-width: 0; }}
.st-key-nav_row .st-key-nav_arrow {{ display: flex; align-items: center; width: auto; height: 2rem; margin-right: 0.6rem; }}
.st-key-nav_row .st-key-nav_arrow button {{
    box-sizing: border-box; width: 2rem; min-width: 2rem; max-width: 2rem; height: 2rem; min-height: 2rem; max-height: 2rem; padding: 0; gap: 0; line-height: 1;
    justify-content: center; border: 1px solid rgba(154, 161, 172, 0.55); border-radius: 8px; color: var(--nav-text);
    background: transparent; transition: background-color 0.15s ease, border-color 0.15s ease; }}
.st-key-nav_row .st-key-nav_arrow button p {{ font-size: 0; line-height: 0; width: 0; overflow: hidden; margin: 0; }}
.st-key-nav_row .st-key-nav_arrow button span[data-testid="stIconMaterial"] {{ font-size: 1.2rem; color: var(--nav-text); margin: 0; }}
.st-key-nav_row .st-key-nav_arrow button:hover {{ background: rgba(154, 161, 172, 0.18); border-color: var(--nav-text); }}
.st-key-nav_row .st-key-nav_arrow {{ position: relative; }}
.st-key-nav_row .st-key-nav_arrow::after {{
    content: "Back"; position: absolute; left: 0; top: calc(100% + 4px); z-index: 10; padding: 0.15rem 0.45rem;
    border-radius: 4px; background: #14171C; color: #FFFFFF; font-size: 0.75rem; line-height: 1.2; pointer-events: none;
    opacity: 0; transition: opacity 0.12s ease 0.3s; }}
.st-key-nav_row .st-key-nav_arrow:hover::after, .st-key-nav_row .st-key-nav_arrow:focus-within::after {{ opacity: 1; }}
.st-key-nav_row .st-key-nav_arrow button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
.st-key-nav_row [class*="st-key-nav_crumb_"], .st-key-nav_row .st-key-nav_home, .st-key-nav_row .st-key-nav_current {{
    display: flex; flex-direction: row; align-items: center; gap: 0; width: auto; }}
.st-key-nav_row [data-testid="stElementContainer"], .st-key-nav_row [data-testid="stMarkdownContainer"] {{ width: auto; }}
.st-key-nav_row .st-key-nav_home::after, .st-key-nav_row [class*="st-key-nav_crumb_"]::after {{
    content: "\\203A"; color: var(--nav-muted); font-size: 1.15rem; line-height: 1; margin: 0 0.4rem; }}
.st-key-nav_row button {{
    width: auto; min-width: 0; min-height: 1.75rem; height: auto; padding: 0.15rem 0.1rem; gap: 0.25rem; border: 0;
    background: transparent; box-shadow: none; border-radius: 6px; }}
.st-key-nav_row button p, .st-key-nav_row button span {{
    font-size: 0.85rem; font-weight: 500; line-height: 1.2; margin: 0; white-space: normal; text-align: left; }}
.st-key-nav_row [class*="st-key-nav_crumb_"] button, .st-key-nav_row [class*="st-key-nav_crumb_"] button p {{ color: var(--nav-muted); }}
.st-key-nav_row [class*="st-key-nav_crumb_"] button:hover p {{ color: var(--nav-text); text-decoration: underline; }}
.st-key-nav_row .st-key-nav_home button, .st-key-nav_row .st-key-nav_home button p, .st-key-nav_row .st-key-nav_home button span {{
    color: {ACCENT}; font-weight: 600; }}
.st-key-nav_row .st-key-nav_home button span[data-testid="stIconMaterial"] {{ font-size: 1.05rem; }}
.st-key-nav_row .st-key-nav_home button:hover p {{ color: {ACCENT_DARK}; text-decoration: underline; }}
.st-key-nav_row button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
.st-key-nav_current, .st-key-nav_current > *, .st-key-nav_current [data-testid="stMarkdownContainer"] {{ height: auto; min-height: 1.75rem; }}
.st-key-nav_current [data-testid="stMarkdownContainer"] {{ display: flex; align-items: center; }}
.st-key-nav_current p {{ color: var(--nav-text); font-size: 0.85rem; font-weight: 700; line-height: 1.2; margin: 0; padding: 0.15rem 0.1rem; }}
:root {{ --nav-muted: #9AA1AC; --nav-text: #FFFFFF; }}
[class*="st-key-navback_"] {{ width: {BUTTON_WIDTH}; }}
[class*="st-key-navback_"] button {{ {_BUTTON.format(w=BUTTON_WIDTH, h=BUTTON_HEIGHT)} }}
[class*="st-key-navback_"] button p {{ {_TEXT} }}
[class*="st-key-navback_"] button {{
    background: transparent; border: 1px solid rgba(255, 255, 255, 0.45); color: #FFFFFF;
}}
[class*="st-key-navback_"] button p {{ color: #FFFFFF; }}
[class*="st-key-navback_"] button:hover, [class*="st-key-navback_"] button:active {{
    background: rgba(255, 255, 255, 0.1); border-color: #FFFFFF; color: #FFFFFF;
}}
[class*="st-key-navback_"] button:focus-visible {{
    outline: 3px solid {ACCENT}; outline-offset: 2px; box-shadow: 0 0 0 2px #FFFFFF;
}}
@media (prefers-color-scheme: light) {{
    :root:not([data-theme="dark"]) {{ --nav-muted: #5B6470; --nav-text: #14171C; }}
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


def nav_row(trail=(), navigate=None, page_css="", view=None):
    """Render the breadcrumb for the current view. Call once per run, at the top of a module's page dispatcher.

    trail lists the levels below Home, outermost first, as (label, target) pairs; the last pair is the current page and its
    target is ignored (it is not clickable). Every earlier pair is a clickable ancestor that calls navigate(target), the
    module's own page-switch function, so the module's existing session-state transitions run unchanged. Home resets module
    navigation exactly as before. page_css is the module's own page-level <style> block; it shares this call's single style
    element because every extra element above the row would add a flow gap and push the row down.

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
        if trail:  # compact arrow: nested view -> immediate parent (the previous breadcrumb); module homepage -> dashboard
            nested = len(trail) >= 2
            with st.container(key="nav_arrow"):
                st.button("Go back", key="nav_arrow_button", icon=":material/arrow_back:",
                          on_click=navigate if nested else _go_home, args=(trail[-2][1],) if nested else ())
        with st.container(key="nav_home"):
            st.button("Home", key="nav_home_button", icon=":material/home:", on_click=_go_home)
        for index, (label, target) in enumerate(trail):
            if index == len(trail) - 1:
                with st.container(key="nav_current"):
                    st.markdown(label)
            else:
                with st.container(key=f"nav_crumb_{index}"):
                    st.button(label, key=f"navbtn_{index}", on_click=navigate, args=(target,))
    if new_view:
        # After the row, in a wrapper hidden by CSS: a script element would otherwise add a flow gap on this first run only.
        with st.container(key="nav_scroll_reset"):
            st.html(_SCROLL_TOP_JS.replace("__N__", str(ss[VIEW_COUNT_KEY])), unsafe_allow_javascript=True)
