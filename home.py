import streamlit as st

from home_navigation import HOME_PAGE_KEY, reset_module_navigation


ACCENT = "#C9363E"


def _icon(path):
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' "
        "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>" + path + "</svg>"
    )
    return "url(\"data:image/svg+xml;utf8," + svg.replace("#", "%23") + "\")"


ICONS = {
    "camera": "<path d='M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3z'/><circle cx='12' cy='13' r='3'/>",
    "sliders": "<path d='M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6'/>",
    "scan": "<path d='M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2'/><circle cx='12' cy='10' r='3'/><path d='M7 19c1-3 3-4 5-4s4 1 5 4'/>",
    "video": "<path d='m16 13 5.2 3.5a.5.5 0 0 0 .8-.4V7.9a.5.5 0 0 0-.8-.4L16 11'/><rect x='2' y='6' width='14' height='12' rx='2'/>",
    "stereo": "<circle cx='7' cy='12' r='4'/><circle cx='17' cy='12' r='4'/><circle cx='7' cy='12' r='1'/><circle cx='17' cy='12' r='1'/><path d='M11 12h2'/>",
    "arrow": "<path d='M7 17 17 7M8 7h9v9'/>",
}

# (key, number, title, description, icon, page file, url path)
MODULES = [
    ("2", "MODULE 02", "Camera Calibration", "2D Object Measurement", "camera", "Pages/app_module2.py", "module-2",
     "Module 2 - 2D Object Measurement"),
    ("3", "MODULE 03", "Image Filtering", "Spatial & FFT Gaussian Blur", "sliders", "Pages/app_module3.py", "module-3",
     "Module 3 - Image Filtering"),
    ("4", "MODULE 04", "Human Boundary Detection", "RGB & Thermal Images", "scan", "Pages/app_module4.py", "module-4",
     "Module 4 - Edge Detection, Boundary Detection, and Features"),
    ("6", "MODULE 06", "Optical Flow & SfM", "Motion Tracking & 3D Reconstruction", "video", "Pages/app_module6.py",
     "module-6", "Module 6 - Optical Flow and Structure from Motion"),
    ("7", "MODULE 07", "Stereo Vision", "Stereo Calibration & Measurement", "stereo", "Pages/app_module7.py",
     "module-7", "Module 7 - Stereo Vision"),
]

MODULE_PAGES = {
    m[0]: st.Page(m[5], title=m[7], url_path=m[6]) for m in MODULES
}


def _home_css():
    icon_rules = "".join(
        f".st-key-home_card_{m[0]} button::before {{ background-image: {_icon(ICONS[m[4]])}; }}\n"
        for m in MODULES
    )
    return f"""<style>
:root {{
    --home-card-bg: #1E2127; --home-card-bg-hover: #262A31; --home-border: #343A43; --home-border-hover: #5A626E;
    --home-text: #FFFFFF; --home-muted: #9AA1AC; --home-tile: rgba(201, 54, 62, 0.16);
}}
@media (prefers-color-scheme: light) {{
    :root:not([data-theme="dark"]) {{
        --home-card-bg: #F3F4F6; --home-card-bg-hover: #E9EBEF; --home-border: #D9DDE3; --home-border-hover: #A3A9B3;
        --home-text: #14171C; --home-muted: #5B6470;
    }}
}}
.stMainBlockContainer {{ max-width: 1000px; padding-top: 3rem; }}
.st-key-home_intro, .st-key-home_intro * {{ text-align: center !important; }}
.st-key-home_intro {{ align-items: center; gap: 0; margin-bottom: 2.5rem; }}
.home-label {{ color: {ACCENT}; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; margin: 0 0 1.15rem; }}
.home-title {{ color: var(--home-text); font-size: clamp(2rem, 6vw, 2.75rem) !important; font-weight: 800; line-height: 1.15; margin: 0 0 0.95rem; padding: 0; }}
.home-desc {{ color: var(--home-muted); font-size: 1.05rem; margin: 0 0 0.95rem; }}
.home-student {{ color: var(--home-muted); font-size: 0.9rem; margin: 0; }}
.st-key-home_cards {{ display: flex; flex-direction: row; flex-wrap: wrap; justify-content: center; gap: 24px; }}
.st-key-home_cards > [data-testid="stElementContainer"], .st-key-home_cards > div {{
    width: calc((100% - 48px) / 3); flex: 0 0 auto; }}
@media (max-width: 900px) {{
    .st-key-home_cards > [data-testid="stElementContainer"], .st-key-home_cards > div {{ width: calc((100% - 24px) / 2); }}
}}
@media (max-width: 600px) {{
    .home-label {{ letter-spacing: 0.08em; font-size: 0.72rem; }}
    .st-key-home_cards > [data-testid="stElementContainer"], .st-key-home_cards > div {{ width: 100%; }}
}}
.st-key-home_cards [data-testid="stButton"] {{ width: 100%; }}
.st-key-home_cards button {{
    position: relative; width: 100%; min-height: 190px; box-sizing: border-box; display: block; text-align: left;
    padding: 5.2rem 1.5rem 1.5rem; background: var(--home-card-bg); border: 1px solid var(--home-border);
    border-radius: 14px; color: var(--home-text); transition: background-color 0.15s ease, border-color 0.15s ease;
}}
.st-key-home_cards button > div {{ width: 100%; justify-content: flex-start; }}
.st-key-home_cards button p {{ color: var(--home-muted); font-size: 0.9rem; line-height: 1.45; text-align: left; margin: 0; }}
.st-key-home_cards button p::first-line {{ color: {ACCENT}; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; line-height: 2; margin-bottom: 0.15rem; }}
.st-key-home_cards button p strong {{ display: inline-block; color: var(--home-text); font-size: 1.1rem; font-weight: 700; margin: 0.2rem 0 0.4rem; }}
.st-key-home_cards button:hover {{ background: var(--home-card-bg-hover); border-color: var(--home-border-hover); color: var(--home-text); }}
.st-key-home_cards button:active {{ background: var(--home-card-bg); border-color: var(--home-border-hover); color: var(--home-text); }}
.st-key-home_cards button:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; border-color: {ACCENT}; box-shadow: none; }}
.st-key-home_cards button::before {{
    content: ""; position: absolute; left: 1.5rem; top: 1.5rem; width: 40px; height: 40px; border-radius: 10px;
    box-sizing: border-box; background-color: {ACCENT}; background-repeat: no-repeat; background-position: center; background-size: 22px;
}}
.st-key-home_cards button::after {{
    content: ""; position: absolute; right: 1.5rem; top: 1.7rem; width: 18px; height: 18px; background-color: var(--home-muted);
    -webkit-mask-image: {_icon(ICONS["arrow"])}; mask-image: {_icon(ICONS["arrow"])};
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-size: contain; mask-size: contain;
    transition: background-color 0.15s ease;
}}
.st-key-home_cards button:hover::after {{ background-color: var(--home-text); }}
{icon_rules}</style>"""


def _open_module(key):
    reset_module_navigation([key])  # always the module's homepage, whatever it showed before
    st.switch_page(MODULE_PAGES[key])


def show_home():
    reset_module_navigation()  # also covers reaching the dashboard by browser history rather than the Home button
    st.markdown(_home_css(), unsafe_allow_html=True)
    with st.container(key="home_intro"):
        st.markdown(
            '<p class="home-label">Georgia State University · CSC 8830</p>'
            '<div class="home-title" role="heading" aria-level="1">Computer Vision Projects</div>'
            '<p class="home-desc">Interactive experiments in image processing, motion analysis, and 3D reconstruction.</p>'
            '<p class="home-student">Jannati Chowdhury</p>',
            unsafe_allow_html=True,
        )
    with st.container(key="home_cards"):
        for key, number, title, desc, *_ in MODULES:
            st.button(
                f"{number}  \n**{title}**  \n{desc}",
                key=f"home_card_{key}",
                on_click=_open_module,
                args=(key,),
            )


st.set_page_config(
    page_title="CSC 8830 - Computer Vision Projects",
    layout="centered",
)

HOME_PAGE = st.Page(show_home, title="Home", default=True)
st.session_state[HOME_PAGE_KEY] = HOME_PAGE

page = st.navigation(
    [
        HOME_PAGE,
        MODULE_PAGES["2"],
        MODULE_PAGES["3"],
        MODULE_PAGES["4"],
        MODULE_PAGES["6"],
        MODULE_PAGES["7"],
    ],
    position="hidden",
)
page.run()
