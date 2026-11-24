"""Selection-card design shared by the Module 7 pages of the website (Pages/app_module7.py) and of the standalone room measurement
application (Modules/Module_7/visitor_workflow): a centered heading and stacked dark-gray cards with a red icon tile, a title, a
subtitle and a chevron.  A card is a Streamlit button, a same-tab link (``card_link``) or an expander summary; the CSS is keyed by the
st.container key that wraps the card, so both applications draw the identical card.  Display only: no navigation logic lives here.
"""
import html

ACCENT = "#C9363E"


def svg_icon(path):
    return ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            f"fill='white'%3E%3Cpath d='{path}'/%3E%3C/svg%3E\")")


ICON_CHEVRON = svg_icon("M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z")
ICON_GUIDE = svg_icon("M19 3H8C6.9 3 6 3.9 6 5v14c0 1.1.9 2 2 2h11V3zM8 5h9v10H8c-.37 0-.7.1-1 .27V5zm0 14c-.55 0-1-.45-1-1s.45-1 1-1h9v2H8z")
ICON_PLAY = svg_icon("M8 5v14l11-7z")
ICON_CAMERA = svg_icon("M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4zM9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z")
ICON_UPLOAD = svg_icon("M9 16h6v-6h4l-7-7-7 7h4v6zm-4 2h14v2H5v-2z")
PANEL_KEYS = {"m7_open_room_setup": ICON_GUIDE, "m7_open_room_run": ICON_PLAY}                       # Room Measurement page
CALIBRATION_PANEL_KEYS = {"m7_pick_new": ICON_CAMERA, "m7_pick_import": ICON_UPLOAD}                 # Camera Calibration page

SELECT_CSS = """
<style>
.st-key-m7_sel_layout {
    max-width: 500px; width: 100%; margin: 0 auto; gap: 0; min-height: calc(100vh - 14rem);
    justify-content: center; align-items: stretch;
}
@media (min-width: 641px) {
    .st-key-m7_sel_layout { box-sizing: border-box; padding-bottom: 270px; }  /* centers the group ~135px above the middle */
}
.st-key-m7_sel_label { margin-bottom: 24px; }  /* spacing lives on the wrappers, not on the text elements */
.st-key-m7_sel_heading { margin-bottom: 44px; }
.st-key-m7_sel_heading_tight { margin-bottom: 12px; }
.st-key-m7_sel_sub { margin-bottom: 44px; }
.st-key-m7_sel_sub [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-m7_sel_sub p { margin: 0; text-align: center; font-size: 1.05rem; line-height: 1.4; opacity: 0.75; }
.st-key-m7_sel_heading [data-testid="stHeading"] > div, .st-key-m7_sel_heading_tight [data-testid="stHeading"] > div { margin: 0; }  /* Streamlit's default -16px bottom margin */
.st-key-m7_sel_label [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-m7_sel_label p {
    margin: 0; text-align: center; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.18em; opacity: 0.55;
}
.st-key-m7_sel_heading h2, .st-key-m7_sel_heading_tight h2 {
    font-size: clamp(1.5rem, 6.5vw, 2.25rem); font-weight: 800; line-height: 1.15; text-align: center;
    padding: 0; margin: 0; color: inherit; overflow-wrap: anywhere;
}
@media (min-width: 801px) {
    .st-key-m7_sel_heading, .st-key-m7_sel_heading [data-testid="stHeading"], .st-key-m7_sel_heading_tight, .st-key-m7_sel_heading_tight [data-testid="stHeading"] { display: block; width: 100%; }
    .st-key-m7_sel_heading h2, .st-key-m7_sel_heading_tight h2 {  /* one line, centered on the panels' axis even when wider than the 500px column */
        position: relative; left: 50%; transform: translateX(-50%); width: max-content; max-width: 94vw; white-space: nowrap;
    }
}
.st-key-m7_sel_panels { gap: 14px; }
__PANEL_SELECTOR__ { width: 100%; }
__PANEL_BUTTON__ {
    width: 100%; height: auto; min-height: 84px; padding: 14px 16px; border-radius: 14px;
    background-color: #262A33; border: 1px solid #3A3F4B; color: #F2F3F5;
    display: flex; align-items: center; justify-content: flex-start; gap: 16px; text-align: left;
    transition: border-color 0.15s, background-color 0.15s, transform 0.15s;
}
__PANEL_BUTTON__::before {
    content: ""; flex: 0 0 48px; height: 48px; border-radius: 12px; background-color: __ACCENT__;
    background-repeat: no-repeat; background-position: center; background-size: 26px;
}
__PANEL_BUTTON__::after {
    content: ""; flex: 0 0 28px; height: 28px; margin-left: auto; opacity: 0.7;
    background-image: __CHEVRON__; background-repeat: no-repeat; background-position: center; background-size: 28px;
}
__PANEL_BUTTON__ > div { flex: 1 1 auto; min-width: 0; justify-content: flex-start; }
__PANEL_BUTTON_SPAN__ { justify-content: flex-start; width: 100%; }
__PANEL_BUTTON__ p {
    margin: 0; text-align: left; font-size: 0.85rem; font-weight: 400; line-height: 1.45; color: #B4B9C4;
}
__PANEL_BUTTON__ p strong { display: block; font-size: 1.05rem; font-weight: 700; color: #FFFFFF; }
__PANEL_BUTTON_BR__ { display: none; }  /* the title is a block, so the label's line break would add a blank line */
__PANEL_BUTTON_HOVER__ { background-color: #2F343F; border-color: __ACCENT__; color: #F2F3F5; transform: translateY(-1px); }
__PANEL_BUTTON_FOCUS__ { outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px __ACCENT__; }
__PANEL_SELECTOR__ a { text-decoration: none; box-sizing: border-box; }
.st-key-m7_open_room_run button:disabled { opacity: 1; cursor: not-allowed; transform: none; background-color: #2B2F38; border: 1px dashed #5A5F6B; }
.st-key-m7_open_room_run button:disabled:hover { background-color: #2B2F38; border-color: #5A5F6B; transform: none; }
.st-key-m7_open_room_run button:disabled::before { background-color: #5A5F6B; }
.st-key-m7_open_room_run button:disabled::after { opacity: 0.25; }
.st-key-m7_open_room_run button:disabled p, .st-key-m7_open_room_run button:disabled p strong { color: #9AA0AC; }
.st-key-m7_room_run_note { padding: 2px 4px 0; }
.st-key-m7_room_run_note p { font-size: 0.85rem; text-align: center; margin: 0; opacity: 0.85; }
__PANEL_ICONS__
/* link cards: the <a> fills the st.html wrapper (a Streamlit button's width comes from its own wrapper) */
__PANEL_SELECTOR__ [data-testid="stHtml"], __PANEL_SELECTOR__ [data-testid="stElementContainer"] { width: 100%; }
__PANEL_SELECTOR__ a, __PANEL_SELECTOR__ a:visited { color: #F2F3F5; cursor: pointer; }
/* expander cards: the summary is the card; its own arrow is hidden and the chevron (::after) points down when closed and up when open */
__PANEL_SELECTOR__ [data-testid="stExpander"], __PANEL_SELECTOR__ details { border: 0; background: transparent; box-shadow: none; border-radius: 14px; }
__PANEL_SELECTOR__ summary { list-style: none; cursor: pointer; box-sizing: border-box; }
__PANEL_SELECTOR__ summary::-webkit-details-marker { display: none; }
__PANEL_SELECTOR__ summary > span { display: flex; flex: 1 1 auto; min-width: 0; align-items: center; }
__PANEL_SELECTOR__ summary > span > span { display: none; }
__PANEL_SELECTOR__ summary > span > div { flex: 1 1 auto; min-width: 0; }
__PANEL_SELECTOR__ summary::after { transform: rotate(90deg); transition: transform 0.15s; }
__PANEL_SELECTOR__ details[open] > summary::after { transform: rotate(-90deg); }
__PANEL_SELECTOR__ [data-testid="stExpanderDetails"] { padding: 18px 2px 4px; border: 0; }
@media (max-width: 640px) {
    .st-key-m7_sel_layout { min-height: calc(100vh - 12rem); }
    __PANEL_BUTTON__ { padding: 12px; gap: 12px; min-height: 76px; }
    __PANEL_BUTTON__::before { flex-basis: 42px; height: 42px; background-size: 22px; }
    __PANEL_BUTTON__ p strong { font-size: 0.98rem; }
}
</style>
"""


def select_css(panels=None):
    """The page's stylesheet for the cards whose wrapper keys are ``panels`` (key -> icon).  The same rules style a Streamlit button,
    a same-tab link (<a>) and an expander summary inside the keyed wrapper."""
    panels = PANEL_KEYS if panels is None else panels

    def selector(suffix):
        return ", ".join(f".st-key-{key} {tag}{suffix}" for key in panels for tag in ("button", "a", "summary"))

    icons = "\n".join(f".st-key-{key} {tag}::before {{ background-image: {icon}; }}" for key, icon in panels.items() for tag in ("button", "a", "summary"))
    css = SELECT_CSS
    for token, value in (("__PANEL_SELECTOR__", ":is(" + ", ".join(f".st-key-{key}" for key in panels) + ")"),
                         ("__PANEL_BUTTON_HOVER__", selector(":hover")),
                         ("__PANEL_BUTTON_FOCUS__", selector(":focus-visible")),
                         ("__PANEL_BUTTON__::before", selector("::before")),
                         ("__PANEL_BUTTON__::after", selector("::after")),
                         ("__PANEL_BUTTON_SPAN__", selector(" > div > span")),
                         ("__PANEL_BUTTON__ > div", selector(" > div")),
                         ("__PANEL_BUTTON__ p strong", selector(" p strong")),
                         ("__PANEL_BUTTON_BR__", selector(" p br")),
                         ("__PANEL_BUTTON__ p", selector(" p")),
                         ("__PANEL_BUTTON__", selector("")),
                         ("__PANEL_ICONS__", icons), ("__CHEVRON__", ICON_CHEVRON), ("__ACCENT__", ACCENT)):
        css = css.replace(token, value)
    return css


def panel_label(title, subtitle):
    return f"**{title}**  \n{subtitle}"


def card_link(url, title, subtitle):
    """HTML of a card that is a plain same-tab link (no target attribute, so the browser navigates in the current tab).  Render it with
    st.html inside the keyed wrapper.  The inner structure matches a Streamlit button's label, so one stylesheet styles both."""
    return f'<a href="{html.escape(url, quote=True)}"><p><strong>{html.escape(title)}</strong><br>{html.escape(subtitle)}</p></a>'
