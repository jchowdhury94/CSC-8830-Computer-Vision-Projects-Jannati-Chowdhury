"""Module 2 - Monocular camera calibration and 2D object measurement.

Layout follows Module 7: a clean home page with two navigation buttons, a visitor workflow (calibrate your camera, then
measure an object) and a separate, read-only "My Experiment & Results" view. The page state is st.session_state["m2_page"]
("home" / "visitor" / "experiment"). Session state uses the m2_ prefix.
"""
import csv
import hashlib
import io
import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates


# Locates Module 2 data before and after moving this file into Pages.
APP_DIR = Path(__file__).resolve().parent
MODULE_DIR = (
    APP_DIR.parent / "Modules" / "Module_2"
    if APP_DIR.name == "Pages"
    else APP_DIR
)
DISPLAY_DIR = MODULE_DIR / "display"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from home_navigation import back_button, nav_row  # noqa: E402
import calibration_file as cf  # noqa: E402  (visitor calibration files; never touches camera_calibration.npz)
import visitor_calibration as vc  # noqa: E402  (visitor monocular calibration; independent of my saved calibration)

# Loads my saved experiment calibration. It is read-only reference data for the
# experiment view only and is never used for visitor measurements.
calibration = np.load(MODULE_DIR / "camera_calibration.npz")

experiment_camera_matrix = calibration["camera_matrix"]
experiment_distortion_coefficients = calibration["distortion_coefficients"]
rms_error = calibration["rms_error"]


# Calculates the real-world length between two image points
def calculate_length(point1, point2, distance, fx, fy):
    u1, v1 = point1
    u2, v2 = point2
    delta_x = (u2 - u1) / fx
    delta_y = (v2 - v1) / fy
    return distance * np.sqrt(delta_x ** 2 + delta_y ** 2)


CSV_COLUMNS = [
    "Trial", "Image", "Distance_cm", "Actual_Width_cm", "Estimated_Width_cm",
    "Width_Error_cm", "Width_Error_Percent", "Actual_Height_cm",
    "Estimated_Height_cm", "Height_Error_cm", "Height_Error_Percent"
]
RESULTS_PATH = MODULE_DIR / "measurement_results.csv"

ACCENT = "#C9363E"
MEAN_REPROJECTION_ERROR_PX = 2.358   # from my original calibration report; it is not stored in the NPZ file
CALIBRATION_PHOTOS_CAPTURED = 20     # the NPZ stores only the accepted count (successful_images)
# st.session_state["m2_visitor_calibration"] is the active visitor calibration (supplied or computed by the visitor); it never
# falls back to my saved experiment calibration. Measurement is enabled exactly while a visitor calibration is active.
SCROLL_TO_PHOTO_JS = """
<script>
setTimeout(function () {
  var anchor = document.getElementById("m2-cal-photo-anchor");
  if (!anchor) { return; }
  anchor.style.scrollMarginTop = "72px";
  anchor.scrollIntoView({behavior: "smooth", block: "start"});
}, 150);
</script>
"""
# Widget values Streamlit drops while the visitor widgets are not rendered (home page or experiment view).
VISITOR_INPUT_KEYS = ("m2_object_name", "m2_actual_width", "m2_actual_height", "m2_distance", "m2_planned")


def _svg_mask(path):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='{path}'/></svg>"
    url = "url(\"data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23") + "\")"
    return f"-webkit-mask-image: {url}; mask-image: {url};"


# Same Material Symbols icons and styling as the Module 7 page, scoped by widget keys (st-key-<key>).
CHEVRON_MASK = _svg_mask("M16.59 8.59 12 13.17 7.41 8.59 6 10l6 6 6-6z")

MODULE_TITLE = "Module 2 — Camera Calibration & Object Measurement"
HOME_TITLE = "Module 2"
HOME_SUBTITLE = "Camera Calibration & 2D Object Measurement"
VISITOR_TITLE = "Calibrate Your Camera & Measure an Object"

HEADING_CSS = """
.st-key-m2_headings { gap: 0.25rem; }
/* Home page: mirrors Module 3's homepage. Title, subtitle and nav are separate containers; the layout's own gap is
   zeroed so the explicit padding-top below is the only space between them (44px title -> subtitle, 64px subtitle -> buttons). */
.st-key-m2_home_layout { gap: 0; padding-top: 1rem; }  /* equals the 16px gap Module 7's injected style block adds above its title */
.st-key-m2_home_title, .st-key-m2_home_subtitle { gap: 0; }
.st-key-m2_home_title, .st-key-m2_home_subtitle, .st-key-m2_home_nav { flex-shrink: 0; }
/* Streamlit's markdown wrapper carries a -16px bottom margin that would eat into the explicit gaps. */
.st-key-m2_home_title [data-testid="stMarkdownContainer"], .st-key-m2_home_subtitle [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
.st-key-m2_home_subtitle { padding-top: 44px; }
.st-key-m2_home_nav { padding-top: 64px; }
.st-key-m2_home_title h1 {
    font-size: clamp(2.5rem, 13vw, 80px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}
.st-key-m2_home_subtitle h2 {
    font-size: clamp(1.1rem, 4.6vw, 32px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2;
    text-align: center; padding: 0; margin: 0;
}
@media (min-width: 801px) { .st-key-m2_home_subtitle h2 { white-space: nowrap; } }
.st-key-m2_headings h1 {
    font-size: clamp(1.15rem, 3.2vw + 0.3rem, 1.9rem); line-height: 1.25; text-align: left;
    padding: 0.25rem 0 0; margin: 0; overflow-wrap: anywhere;
}
.st-key-m2_headings h2 {
    font-size: clamp(1rem, 1.6vw + 0.5rem, 1.3rem); font-weight: 500; line-height: 1.3; text-align: left;
    opacity: 0.8; padding: 0; margin: 0; overflow-wrap: anywhere;
}
/* Visitor only: larger main heading; clamp keeps it from overflowing on narrow screens. */
.st-key-m2_headings_visitor h1 {
    font-size: clamp(1.75rem, 5vw + 0.5rem, 3rem); font-weight: 800; line-height: 1.15;
}
"""

NAV_KEYS = ("m2_open_visitor", "m2_open_experiment")
_BUTTON_TEMPLATE = f"""
{{k}}, {{k}} [data-testid="stButton"] {{ width: 100%; }}
{{k}} button {{
    width: 100%; background-color: {ACCENT}; border: 2px solid {ACCENT}; border-radius: 12px;
    padding: 0.9rem 1rem; color: #FFFFFF;
}}
{{k}} button p {{ color: #FFFFFF; font-weight: 700; font-size: 1.15rem; }}
{{k}} button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
{{k}} button:focus-visible {{ outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT}; }}
{{k}} button:active {{ background-color: #8E2229; border-color: #8E2229; }}
{{k}} button {{ position: relative; }}
{{k}} button::after {{
    content: ""; position: absolute; right: 1.2rem; top: 50%; width: 2rem; height: 2rem; margin-top: -1rem;
    background-color: #FFFFFF; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-position: center;
    mask-position: center; -webkit-mask-size: contain; mask-size: contain; transition: transform 0.25s ease; {CHEVRON_MASK}
}}
@media (prefers-reduced-motion: reduce) {{ {{k}} button::after {{ transition: none; }} }}
"""
# Both home buttons share Module 7's "View My Experiment & Results" button style; the chevron points right (it opens a view).
BUTTON_CSS = "\n".join(
    _BUTTON_TEMPLATE.replace("{k}", f".st-key-{key}") + f".st-key-{key} button::after {{ transform: rotate(-90deg); }}"
    for key in NAV_KEYS)
MODE_CSS = f"""
.st-key-m2_cal_mode_run button, .st-key-m2_cal_mode_load button {{
    width: 100%; min-height: 4rem; border-radius: 14px; border: 2px solid {ACCENT}; font-size: 1.1rem; font-weight: 700;
    background-color: transparent; color: {ACCENT};
}}
.st-key-m2_cal_mode_run button p, .st-key-m2_cal_mode_load button p {{ font-size: 1.1rem; font-weight: 700; color: inherit; }}
.st-key-m2_cal_mode_run button[kind="primary"], .st-key-m2_cal_mode_load button[kind="primary"] {{ background-color: {ACCENT}; color: #FFFFFF; }}
.st-key-m2_cal_mode_run button:focus-visible, .st-key-m2_cal_mode_load button:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
"""
# Load Saved Calibration extras: the details expander header and the file-removal (×) button (scoped by container/button key).
LOADED_CSS = f"""
.st-key-m2_loaded_notes details {{
    border: 1px solid color-mix(in srgb, {ACCENT} 55%, transparent) !important; border-radius: 10px; overflow: hidden;
}}
.st-key-m2_loaded_notes details > summary {{
    background: color-mix(in srgb, {ACCENT} 9%, #1E2127); padding: 0.7rem 1rem; cursor: pointer;
}}
.st-key-m2_loaded_notes details > summary:hover {{ background: color-mix(in srgb, {ACCENT} 18%, #1E2127); }}
.st-key-m2_loaded_notes details > summary:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: -3px; }}
.st-key-m2_loaded_notes details > summary [data-testid="stMarkdownContainer"] p {{ font-weight: 700; font-size: 1.05rem; margin: 0; }}
[class*="st-key-m2_cal_legacy_w_"] [data-testid="InputInstructions"], [class*="st-key-m2_cal_legacy_h_"] [data-testid="InputInstructions"] {{ display: none; }}
.st-key-m2_npz_label p {{ font-size: 0.875rem; margin: 0; }}
.st-key-m2_loaded_file [data-testid="stHorizontalBlock"] {{
    align-items: center; min-height: 4.5rem; padding: 0.5rem 1rem;
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 30%, transparent);
    border-radius: 0.5rem;
}}
.st-key-m2_res_section, .st-key-m2_npz_box {{
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    border-radius: 12px; padding: 1.25rem 1.25rem 1rem; gap: 0.9rem;
}}
.st-key-m2_res_title p {{ font-size: 1.4rem; font-weight: 700; margin: 0; }}
.st-key-m2_res_section [data-testid="stTextInputRootElement"], .st-key-m2_res_section [data-testid="stFileUploaderDropzone"],
.st-key-m2_npz_box [data-testid="stFileUploaderDropzone"] {{
    border: 1px solid color-mix(in srgb, currentColor 30%, transparent);
}}
.st-key-m2_clear_loaded button {{
    background-color: transparent; border: 1px solid transparent; border-radius: 50%; color: inherit;
    min-height: 2.25rem; width: 2.25rem; padding: 0;
}}
.st-key-m2_clear_loaded button:hover {{ background-color: color-mix(in srgb, {ACCENT} 14%, transparent); border-color: {ACCENT}; color: {ACCENT}; }}
.st-key-m2_clear_loaded button:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
"""
# Home buttons: same stacked, centered layout and values as the Module 7 homepage (Pages/app_module7.py HOME_CSS + M7_STACK_CSS).
BLUE, BLUE_DARK = "#1D5FD1", "#174DA8"   # Module 7's blue; white text on these is above 6:1
HOME_CARD_CSS = f"""
.st-key-m2_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 24px; max-width: 560px; width: 100%; margin: 0 auto; }}
.st-key-m2_open_visitor, .st-key-m2_open_experiment {{ flex: 0 0 auto; width: 100%; min-width: 0; max-width: 100%; }}
.st-key-m2_open_visitor button, .st-key-m2_open_experiment button {{
    height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
.st-key-m2_open_visitor button p, .st-key-m2_open_experiment button p {{
    font-size: 18px; font-weight: 500; line-height: 1.3; text-align: center; text-wrap: balance;   /* Module 7 homepage buttons use weight 500 */
}}
.st-key-m2_open_visitor button::after, .st-key-m2_open_experiment button::after {{ content: none; display: none; }}
.st-key-m2_open_experiment button {{ background-color: {BLUE}; border-color: {BLUE}; }}
.st-key-m2_open_experiment button:hover {{ background-color: {BLUE_DARK}; border-color: {BLUE_DARK}; }}
.st-key-m2_open_experiment button:active {{ background-color: #123F8F; border-color: #123F8F; }}
.st-key-m2_open_experiment button:focus-visible {{ box-shadow: 0 0 0 6px {BLUE}; }}
@media (max-width: 640px) {{ .st-key-m2_home_nav {{ gap: 16px; }} }}
"""
RESULTS_CSS = """
.st-key-m2_res_head { gap: 0; padding-bottom: 28px; }
.st-key-m2_res_head h1 {
    font-size: clamp(2.5rem, 8vw, 60px); font-weight: 800; line-height: 1.1; text-align: center; padding: 0; margin: 0;
    color: inherit; overflow-wrap: anywhere;
}
.st-key-m2_res_head h2 {
    font-size: clamp(1.4rem, 4.6vw, 2.4rem); font-weight: 600; line-height: 1.2; text-align: center; padding: 0; margin: 0;
    color: inherit; opacity: 0.85; overflow-wrap: anywhere;
}
.st-key-m2_res_title { padding-bottom: 28px; }
.st-key-m2_res_body h2 { text-align: center; font-size: clamp(1.7rem, 5.2vw, 2.4rem); font-weight: 800; line-height: 1.2; padding: 1.5rem 0 0.25rem; overflow-wrap: anywhere; }
.st-key-m2_res_body h3 { text-align: center; font-size: clamp(1.4rem, 4.5vw, 1.9rem); font-weight: 700; padding-top: 1.5rem; }
.st-key-m2_res_body h4 { text-align: left; font-size: 1.15rem; padding: 1.1rem 0 0.2rem; }
.st-key-m2_res_body h5 { text-align: left; font-weight: 700; margin: 0; padding: 0.5rem 0 0.25rem; }
.st-key-m2_res_body [data-testid="stMetricValue"] { font-size: 2rem; }
.st-key-m2_res_body [data-testid="stMetric"] { text-align: left; }
.st-key-m2_res_body [data-testid="stMetricLabel"] p, .st-key-m2_res_body [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
.st-key-m2_res_body [data-testid="stTable"] { overflow-x: auto; }
.st-key-m2_res_body img { max-width: 100%; height: auto; }
/* each numbered section is set off by a thin divider; the conclusion shares Module 7's shaded, bordered panel look */
[class*="st-key-m2_res_sec_"] { border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent); margin-top: 1.5rem; padding-top: 0.5rem; }
.st-key-m2_res_conclusion {
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 16px; padding: 0.5rem 28px 28px; margin-top: 2rem;
}
@media (max-width: 640px) { .st-key-m2_res_conclusion { padding: 0.25rem 14px 18px; } }
"""
PAGE_CSS = "<style>" + HEADING_CSS + MODE_CSS + BUTTON_CSS + HOME_CARD_CSS + LOADED_CSS + RESULTS_CSS + "</style>"


# Reads and checks the existing validation trials (read-only)
def read_validation_rows():
    with RESULTS_PATH.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("The validation CSV columns do not match the expected format.")
        rows = list(reader)
    for row in rows:
        for column in CSV_COLUMNS:
            if column != "Image":
                if not np.isfinite(float(row[column])):
                    raise ValueError("The validation CSV contains invalid numeric values.")
    return rows


# ---------------------------------------------------------------- session state

def init_state():
    ss = st.session_state
    ss.setdefault("m2_visitor_calibration", None)   # active visitor calibration (never my saved calibration)
    ss.setdefault("m2_cal_workflow", None)          # selected calibration workflow: None / "load" / "run" (UI only)
    ss.setdefault("m2_page", "home")                # "home" / "visitor" / "experiment"
    ss.setdefault("m2_saved_inputs", {})            # visitor inputs kept while the visitor widgets are not rendered
    ss.setdefault("m2_cal", vc.new_state())         # Calibrate New Camera: board settings, photo slots, current photo
    ss.setdefault("m2_computed", None)              # calibration computed from the visitor's photos (not necessarily active)
    ss.setdefault("m2_load_pending", None)          # calibration read from an uploaded NPZ (not necessarily active)
    ss.setdefault("m2_load_name", None)
    ss.setdefault("m2_load_error", None)
    ss.setdefault("m2_load_base", None)             # the legacy NPZ model exactly as loaded (no image size); sized copies derive from it
    ss.setdefault("m2_legacy_photo", None)          # detected resolution of the original calibration photograph: name, width, height
    ss.setdefault("m2_legacy_photo_error", None)
    ss.setdefault("m2_cal_error", None)
    ss.setdefault("m2_cal_notice", None)
    ss.setdefault("m2_cal_validation", [])
    ss.setdefault("m2_cal_run_requested", False)
    ss.setdefault("m2_npz_gen", 0)                  # suffix of the NPZ uploader key; bumped so the same file can be chosen again
    ss.setdefault("m2_reset_count", 0)              # suffix of every calibration widget key; bumped by Reset All
    ss.setdefault("m2_unit_shown", "mm")
    ss.setdefault("m2_exp", new_experiment())       # the visitor's experiment: plan, saved measurements (session only)
    ss.setdefault("m2_planned", 20)                 # planned number of measurements
    ss.setdefault("m2_img_gen", 0)                  # suffix of the measurement-image uploader key; bumped for each new trial
    ss.setdefault("m2_confirm", None)               # pending confirmation: None / "restart" / "change_cal" / "reset_all"
    ss.setdefault("m2_row_action", None)            # edit/delete in progress in the results table: None / "edit" / "delete"


# Clears the temporary result and its confirmation
def clear_live_result():
    st.session_state.pop("m2_live_result", None)


# Discards the result and resets the selected corners
def discard_result():
    reset_points()
    st.session_state.m2_result_notice = "Live result discarded."


# Clears the corners and starts a fresh click component
def reset_points():
    clear_live_result()
    st.session_state.m2_selected_points = []
    st.session_state.m2_click_generation += 1


# Resets the typed object and measurement information
def reset_inputs():
    clear_live_result()
    st.session_state.m2_object_name = ""
    st.session_state.m2_actual_width = None
    st.session_state.m2_actual_height = None
    st.session_state.m2_distance = None


def select_workflow(workflow):
    """Only records which calibration workflow is shown; never loads or computes a calibration."""
    st.session_state["m2_cal_workflow"] = workflow


def reset_all():
    """Returns the visitor's calibration and measurement workflow to first-visit defaults.

    Touches visitor session state only: my saved calibration, the validation CSV and the experiment view are never read or written.
    """
    ss = st.session_state
    for key in list(ss.keys()):
        if str(key).startswith("m2_") and key not in ("m2_page", "m2_reset_count"):
            del ss[key]
    init_state()
    ss["m2_reset_count"] += 1    # new widget keys: the browser would otherwise send back the old checkerboard values and files


def go_to(page):
    """Switches the page. Streamlit forgets widgets that are not rendered, so the visitor's typed values are kept when leaving
    the visitor workflow and restored when entering it again; nothing else is cleared."""
    ss = st.session_state
    if ss["m2_page"] == "visitor" and page != "visitor":
        ss["m2_saved_inputs"] = {key: ss[key] for key in VISITOR_INPUT_KEYS if key in ss}
    elif page == "visitor":
        for key, value in ss["m2_saved_inputs"].items():
            ss[key] = value
    ss["m2_page"] = page


def kv_table(index_name, rows, columns):
    return st.table(pd.DataFrame(rows, columns=[index_name] + columns).set_index(index_name))


# ---------------------------------------------------------------- visitor section

def workflow_selector():
    """The two large workflow buttons at the top of the expander; they only select a workflow and never run anything."""
    workflow = st.session_state["m2_cal_workflow"]
    for column, (name, key, label) in zip(st.columns(2), (("load", "m2_cal_mode_load", "Load Saved Calibration"),
                                                          ("run", "m2_cal_mode_run", "Calibrate New Camera"))):
        with column:
            st.button(label, key=key, type="primary" if workflow == name else "secondary", on_click=select_workflow,
                      args=(name,), width="stretch")
    return workflow


# ---------------------------------------------------------------- visitor calibration: state changes

def fk(name):
    """Calibration widget key; the reset counter makes every widget brand new after Reset All."""
    return f"m2_cal_{name}_{st.session_state['m2_reset_count']}"


def model_token(model):
    """Identity of a calibration, from its numbers: whether a shown calibration is the active one."""
    raw = np.asarray(model["camera_matrix"], np.float64).tobytes() + np.asarray(model["distortion_coefficients"], np.float64).tobytes()
    return hashlib.sha256(raw + f"{model.get('image_width')}x{model.get('image_height')}".encode()).hexdigest()


def is_active(model):
    active = st.session_state["m2_visitor_calibration"]
    return active is not None and model is not None and model_token(active) == model_token(model)


def set_visitor_calibration(model):
    """Makes a calibration the active one and drops the measurement that was based on the previous one."""
    ss = st.session_state
    ss["m2_visitor_calibration"] = model
    ss["m2_cal_notice"] = None
    if "m2_click_generation" in ss:
        reset_points()
    else:
        clear_live_result()


def activate(model):
    set_visitor_calibration(dict(model))


def clear_calibration():
    ss = st.session_state
    ss["m2_computed"] = ss["m2_load_pending"] = ss["m2_load_name"] = ss["m2_load_error"] = ss["m2_cal_error"] = None
    ss["m2_load_base"] = ss["m2_legacy_photo"] = ss["m2_legacy_photo_error"] = None
    ss["m2_cal_validation"] = []
    set_visitor_calibration(None)


def on_unit_change():
    ss = st.session_state
    new = ss[fk("unit")]
    try:
        ss[fk("square")] = vc.convert_square(ss[fk("square")], ss["m2_unit_shown"], new)   # same physical size in the new unit
    except (KeyError, TypeError, ValueError):
        pass


def on_photo_upload(index):
    ss = st.session_state
    uploaded = ss.get(fk(f"up{index}_{ss['m2_cal']['slot_gen'].get(index, 0)}"))
    if uploaded is not None:
        vc.store_photo(ss["m2_cal"], index, uploaded.name, uploaded.getvalue())


def on_photo_remove(index):
    vc.remove_photo(st.session_state["m2_cal"], index)


def clear_all_photos():
    vc.remove_all_photos(st.session_state["m2_cal"])


def change_photo(move):
    """Apply a photo change and request the one-shot scroll only if the selected photo really changed."""
    ss = st.session_state
    before = ss["m2_cal"]["current"]
    move(ss["m2_cal"])
    if ss["m2_cal"]["current"] != before:
        ss["m2_scroll_to_cal_photo"] = True


def on_step(delta):
    change_photo(lambda state: vc.step(state, delta))


def on_goto():
    change_photo(lambda state: vc.go_to(state, st.session_state["m2_cal_goto"]))


def request_run():
    st.session_state["m2_cal_run_requested"] = True


def on_calibration_file():
    ss = st.session_state
    uploaded = ss.get(f"m2_cal_npz_{ss['m2_reset_count']}_{ss['m2_npz_gen']}")
    if uploaded is None:
        return
    ss["m2_npz_gen"] += 1                       # fresh widget keys for everything that belongs to the file
    try:
        model = cf.parse_calibration(uploaded.getvalue())
    except cf.CalibrationFileError as error:
        ss["m2_load_error"] = f"'{uploaded.name}' was not loaded: {error}"   # a file that was already loaded stays as it was
        return
    old, old_name = ss["m2_load_pending"], ss["m2_load_name"]
    was_active = old is not None and is_active(old)
    if was_active:
        set_visitor_calibration(None)           # replacing the calibration in use is explicit: say so below
    ss["m2_load_pending"], ss["m2_load_name"], ss["m2_load_error"] = model, uploaded.name, None   # loaded, NOT activated
    ss["m2_load_base"] = model if model["needs_image_size"] else None
    ss["m2_legacy_photo"] = ss["m2_legacy_photo_error"] = None
    ss["m2_cal_notice"] = (f"'{old_name}' was replaced, so it is no longer the active calibration. Activate the new file to measure."
                           if was_active else None)


def clear_loaded_calibration():
    """Clear Calibration for a loaded file: forget the file and everything built on it, and restore the NPZ uploader.

    Only a calibration that is the loaded file is deactivated; a separately computed new-camera calibration stays as it is.
    """
    ss = st.session_state
    pending = ss["m2_load_pending"]
    if pending is not None and is_active(pending):
        set_visitor_calibration(None)
    gen, count = ss["m2_npz_gen"], ss["m2_reset_count"]
    for key in (f"m2_cal_npz_{count}_{gen}", f"m2_cal_photo_{count}_{gen}", fk(f"photo_ok_{gen}"), fk(f"legacy_ok_{gen}"),
                fk(f"legacy_w_{gen}"), fk(f"legacy_h_{gen}")):
        ss.pop(key, None)
    ss["m2_load_pending"] = ss["m2_load_name"] = ss["m2_load_error"] = None
    ss["m2_load_base"] = ss["m2_legacy_photo"] = ss["m2_legacy_photo_error"] = None
    ss["m2_npz_gen"] += 1                       # fresh widget keys: empty uploaders, unticked boxes, empty size fields
    ss["m2_cal_notice"] = None


def legacy_base():
    """The legacy (size-less) model of the loaded file, or None when the file states its own resolution."""
    ss = st.session_state
    pending = ss["m2_load_pending"]
    if pending is None:
        return None
    if pending["needs_image_size"]:
        return pending
    return ss["m2_load_base"]


def set_legacy_size(model):
    """Replace the loaded model by a resized (or size-less) copy. A calibration that was active at the old size is deactivated."""
    ss = st.session_state
    if is_active(ss["m2_load_pending"]):
        set_visitor_calibration(None)
    ss["m2_load_pending"] = model


def revoke_legacy_size():
    """Drop any resolution applied so far; the visitor must confirm and activate again."""
    base = legacy_base()
    if base is not None:
        set_legacy_size(base)


def on_legacy_photo():
    """Photograph uploaded, replaced or removed: forget the previous detection and any confirmation built on it."""
    ss = st.session_state
    gen = ss["m2_npz_gen"]
    uploaded = ss.get(f"m2_cal_photo_{ss['m2_reset_count']}_{gen}")
    ss["m2_legacy_photo"] = ss["m2_legacy_photo_error"] = None
    ss[fk(f"photo_ok_{gen}")] = False
    revoke_legacy_size()
    if uploaded is None:
        return
    try:
        width, height = vc.detect_photo_resolution(uploaded.getvalue())
    except vc.CalibrationError as error:
        ss["m2_legacy_photo_error"] = f"'{uploaded.name}' could not be used: {error}"
        return
    ss["m2_legacy_photo"] = {"name": uploaded.name, "width": width, "height": height}


def remove_legacy_photo():
    """Remove Image: forget the photograph, its detected resolution and confirmation. The NPZ stays loaded; a file that
    lacked a resolution goes back to the missing-resolution state, and a calibration that was active is deactivated."""
    ss = st.session_state
    gen, count = ss["m2_npz_gen"], ss["m2_reset_count"]
    ss.pop(f"m2_cal_photo_{count}_{gen}", None)
    ss.pop(fk(f"photo_ok_{gen}"), None)
    ss["m2_legacy_photo"] = ss["m2_legacy_photo_error"] = ss["m2_load_error"] = None
    revoke_legacy_size()


def apply_detected_size():
    ss = st.session_state
    photo = ss["m2_legacy_photo"]
    try:
        if photo is None:
            raise cf.CalibrationFileError("Upload an original calibration photograph first.")
        if not ss.get(fk(f"photo_ok_{ss['m2_npz_gen']}")):
            raise cf.CalibrationFileError("Tick the confirmation box first.")
        set_legacy_size(cf.with_image_size(legacy_base(), photo["width"], photo["height"], "detected_from_photo"))
        ss["m2_load_error"] = None
    except (cf.CalibrationFileError, TypeError, ValueError) as error:
        ss["m2_load_error"] = f"The detected resolution was not applied: {error}"


def parse_pixels(text, label):
    """A typed pixel count: digits only (no sign, decimal point or spaces). The range check stays in cf.with_image_size."""
    text = (text or "").strip()
    if not (text.isascii() and text.isdigit()):
        raise cf.CalibrationFileError(f"{label} must be a whole number of pixels (digits only).")
    return int(text)


def apply_legacy_size():
    ss = st.session_state
    gen = ss["m2_npz_gen"]
    try:
        if not ss.get(fk(f"legacy_ok_{gen}")):
            raise cf.CalibrationFileError("Tick the confirmation box first.")
        width = parse_pixels(ss.get(fk(f"legacy_w_{gen}")), "The image width")
        height = parse_pixels(ss.get(fk(f"legacy_h_{gen}")), "The image height")
        set_legacy_size(cf.with_image_size(legacy_base(), width, height))
        ss["m2_load_error"] = None
    except (cf.CalibrationFileError, TypeError, ValueError) as error:
        ss["m2_load_error"] = f"The image size was not applied: {error}"


# ---------------------------------------------------------------- visitor calibration: shared results

def matrix_rows(matrix, digits):
    return [[f"{value:.{digits}f}" for value in row] for row in matrix]


def calibration_results(model, key_prefix):
    """Results of a computed or loaded calibration, with activation. Computed, warned and active are shown as separate states."""
    ss = st.session_state
    computed = model["source"] == vc.SOURCE_COMPUTED
    st.success("Calibration computed from your photographs." if computed else "Calibration file loaded and validated.")
    if not computed:
        st.write(f"**File:** {ss['m2_load_name']}")
        st.warning("The file was checked for structure, plausible values and internal consistency only. It is **not authenticated**, "
                   "so its accuracy and origin cannot be confirmed, and it is valid only for the camera, lens and settings it came from.")
        if model.get("size_source") == "detected_from_photo":
            st.warning(f"The file did not state its image size. The resolution {model['image_width']} × {model['image_height']} px was "
                       "detected from the original calibration photograph you uploaded and confirmed. Measurement photographs must "
                       "have exactly this resolution.")
        if model.get("size_source") == "confirmed_by_visitor":
            st.warning(f"The file did not state its image size. You confirmed {model['image_width']} × {model['image_height']} px as the "
                       "resolution of the calibration photographs. Measurement photographs must have exactly this resolution.")
    K = model["camera_matrix"]
    if computed:
        c = st.columns(3)
        c[0].metric("Uploaded photographs", model["photos_uploaded"])
        c[1].metric("Accepted", model["successful_images"])
        c[2].metric("Rejected", model["photos_rejected"])
    c = st.columns(2)
    c[0].metric("Calibration image resolution", f"{model['image_width']} × {model['image_height']} px")
    c[1].metric("RMS reprojection error", "not stored" if model["rms_error"] is None else f"{model['rms_error']:.3f} px")
    c = st.columns(4)
    for column, (label, value) in zip(c, (("fx", K[0, 0]), ("fy", K[1, 1]), ("cx", K[0, 2]), ("cy", K[1, 2]))):
        column.metric(label, f"{value:.2f} px")
    if model["checkerboard_columns"] is not None and model["checkerboard_rows"] is not None:
        square = "" if model["square_size_mm"] is None else f", {model['square_size_mm']:g} mm squares"
        st.write(f"**Checkerboard:** {model['checkerboard_columns']} × {model['checkerboard_rows']} inner corners{square}")
    st.caption("The reprojection error describes how well the model reproduces the checkerboard corners in the calibration photos. "
               "It is not the physical error of a later object measurement, which also depends on the distance, the corner clicks and "
               "how closely the measurement setup matches the calibration. Warning thresholds are heuristics, not accuracy guarantees.")
    warnings = model.get("warnings") or []
    if warnings:
        st.write("**Calibration quality warnings**")
        for item in warnings:
            st.warning(item["message"] + " (heuristic)")
    elif computed:
        st.info("No quality warnings were raised. This does not guarantee measurement accuracy.")
    for note in model.get("camera_notes") or []:
        st.caption(note)

    if is_active(model):
        st.success("This calibration is active for measurement (Step 2).")
    else:
        st.info("This calibration is not active yet. Activate it to use it for measurement.")
    st.button("Activate Calibration", key=f"m2_activate_{key_prefix}", type="primary", on_click=activate, args=(model,),
              disabled=is_active(model))
    if st.toggle("View Calibration Details", key=fk(f"details_{key_prefix}")):
        calibration_details(model)
    st.button("Clear Calibration", key=f"m2_clear_{key_prefix}", on_click=clear_calibration)


def calibration_details(model):
    st.write("**Camera intrinsic matrix K (3×3, pixels)**")
    st.table(matrix_rows(model["camera_matrix"], 4))
    st.write("**Distortion coefficients**")
    st.table({"Coefficient": ["k1", "k2", "p1", "p2", "k3"], "Value": [f"{v:.6f}" for v in model["distortion_coefficients"].ravel()]})
    if model.get("per_image"):
        st.write("**Per-image reprojection error**")
        st.dataframe(pd.DataFrame([{"Photo": e["number"], "File": e["name"], "Mean error (px)": round(e["error_px"], 4)}
                                   for e in model["per_image"]]).set_index("Photo"), width="stretch")
        coverage = model["coverage"]
        st.write(f"**Board coverage:** {coverage['area_fraction'] * 100:.0f}% of the frame, {coverage['cells_used']} of 9 image regions. "
                 f"**Board tilt range:** {min(model['tilts_deg']):.0f}° to {max(model['tilts_deg']):.0f}°.")
    else:
        st.caption("Per-image errors are not stored in calibration files.")
    meta = {"Source": "computed in this session" if model["source"] == vc.SOURCE_COMPUTED else "loaded from a file (not authenticated)",
            "Image size source": {"calibration_photos": "calibration photographs", "file": "stored in the file",
                                  "confirmed_by_visitor": "confirmed by you",
                                  "detected_from_photo": "detected from your original calibration photograph"}.get(model.get("size_source"), "-"),
            "Created (UTC)": model.get("created_utc", "-"),
            "Accepted photographs": model.get("successful_images", "-")}
    camera = model.get("camera") or {}
    meta["Camera (from photo metadata)"] = ", ".join(f"{k}: {v}" for k, v in camera.items() if k != "n_photos") or "no metadata"
    st.table({"Item": list(meta), "Value": [str(v) for v in meta.values()]})


def calibration_download(model):
    """Download of a newly computed calibration; always shown, enabled only once there is one."""
    data = None
    if model is not None:
        try:
            data = cf.export_calibration(model)
        except ValueError as error:
            st.error(f"This calibration cannot be exported, so downloading is disabled: {error}")
    st.download_button("Download Calibration (.npz)", data if data is not None else b"", file_name="camera_calibration_visitor.npz",
                       mime="application/octet-stream", key="m2_cal_download", icon=":material/download:", disabled=data is None)
    if data is None:
        st.caption("Run the calibration to enable downloading.")
    else:
        st.caption("The file holds calibration numbers only, no photographs. It can be reloaded with 'Load Saved Calibration' and is "
                   "valid only for the same camera, lens, zoom and settings at the same resolution.")


# ---------------------------------------------------------------- Workflow A: load a saved calibration

def status_line(state, text):
    """One compact status indicator per meaningful state of a loaded file."""
    color = {"needs_resolution": "orange", "ready": "blue", "active": "green"}[state]
    st.markdown(f":{color}-badge[{text}]")


def size_text(model):
    return f"{model['image_width']} × {model['image_height']} px"


RES_AUTO = "Detect Automatically from Calibration Image"
RES_MANUAL = "Enter Resolution Manually"


def on_resolution_method():
    """Switching method forgets whatever the other method had collected, so nothing unconfirmed can be applied later."""
    ss = st.session_state
    remove_legacy_photo()
    for name in ("legacy_w", "legacy_h", "legacy_ok"):
        ss.pop(fk(f"{name}_{ss['m2_npz_gen']}"), None)


def legacy_resolution(pending):
    """Older files have no image size: detect it from an original calibration photograph or enter it manually.
    The whole section is shown only while the loaded file still lacks a resolution; only the chosen method's controls are drawn."""
    ss = st.session_state
    gen = ss["m2_npz_gen"]
    with st.container(key="m2_res_section"):
        with st.container(key="m2_res_title"):
            st.markdown("**Calibration Image Resolution**")
        st.write("Your saved calibration file does not include the original image resolution. Select how you would like to provide it.")
        method = st.radio("Resolution method", [RES_AUTO, RES_MANUAL], key=fk(f"res_method_{gen}"), on_change=on_resolution_method,
                          label_visibility="collapsed")
        if method == RES_AUTO:
            st.file_uploader("Upload an Original Calibration Image", type=["jpg", "jpeg", "png"],
                             key=f"m2_cal_photo_{ss['m2_reset_count']}_{gen}", on_change=on_legacy_photo)
            if ss["m2_legacy_photo_error"]:
                st.error(ss["m2_legacy_photo_error"])
            photo = ss["m2_legacy_photo"]
            if photo is not None:
                st.markdown(f"**{photo['name']}**  \nDetected: **{photo['width']} × {photo['height']} px**")
                st.checkbox("I confirm this photograph was used for the original calibration and has not been resized or cropped since "
                            "that calibration.", key=fk(f"photo_ok_{gen}"))
                st.button("Use Detected Resolution", key="m2_photo_apply", on_click=apply_detected_size)
        else:
            st.write("Enter the original calibration image dimensions in pixels, after EXIF rotation.")
            c = st.columns(2)
            c[0].text_input("Image width (px)", placeholder="width", key=fk(f"legacy_w_{gen}"))
            c[1].text_input("Image height (px)", placeholder="height", key=fk(f"legacy_h_{gen}"))
            st.checkbox("I confirm these are the original calibration image dimensions and that I will use the same camera, lens, "
                        "and settings.", key=fk(f"legacy_ok_{gen}"))
            st.button("Apply Image Size", key="m2_legacy_apply", on_click=apply_legacy_size)


def loaded_results(model):
    """Validated loaded file. Results (resolution, RMS, status) are revealed only by activation, not by validation."""
    for item in model.get("warnings") or []:
        st.warning(item["message"] + " (heuristic)")
    if not is_active(model):
        st.button("Activate Calibration", key="m2_activate_loaded", type="primary", on_click=activate, args=(model,))
        return
    rms = "RMS not stored" if model["rms_error"] is None else f"RMS {model['rms_error']:.3f} px"
    st.markdown(f"**{size_text(model)}** | {rms}")
    status_line("active", ":material/check: Active for Measurement")


def loaded_notes(model):
    K = model["camera_matrix"]
    st.write("**Intrinsic parameters:** " + ", ".join(f"{label} = {value:.2f} px" for label, value in
                                                      (("fx", K[0, 0]), ("fy", K[1, 1]), ("cx", K[0, 2]), ("cy", K[1, 2]))))
    if model["checkerboard_columns"] is not None and model["checkerboard_rows"] is not None:
        square = "" if model["square_size_mm"] is None else f", {model['square_size_mm']:g} mm squares"
        st.write(f"**Checkerboard:** {model['checkerboard_columns']} × {model['checkerboard_rows']} inner corners{square}")
    source = {"detected_from_photo": "detected from your original calibration photograph",
              "confirmed_by_visitor": "confirmed by you", "file": "stored in the file"}.get(model.get("size_source"))
    if source:
        st.write(f"**Resolution:** {size_text(model)} ({source})")
    st.markdown("- Use the same camera, lens, zoom, settings and image resolution used for calibration.\n"
                "- For older NPZ files, confirm the original calibration image resolution.\n"
                "- File validation checks structure and plausible values, not accuracy or origin.")
    for note in model.get("camera_notes") or []:
        st.caption(note)


def loaded_file_card():
    """Replaces the uploader's dropzone, inside the same container and under the same label: icon, filename, status and a ×."""
    with st.container(key="m2_loaded_file"):
        info, remove = st.columns([12, 1], vertical_alignment="center")
        with info:
            st.markdown(f":material/description: **{st.session_state['m2_load_name']}**")
            st.caption(":material/check_circle: File uploaded and validated")
        remove.button(":material/close:", key="m2_clear_loaded", on_click=ask_confirm, args=("remove_file",),
                      help="Remove this file and clear its calibration", type="tertiary")


def photo_summary(pending):
    """Compact record of the confirmed calibration photograph (evidence of resolution only), with Remove Image."""
    photo = st.session_state["m2_legacy_photo"]
    if photo is None or pending.get("size_source") != "detected_from_photo":
        return
    with st.container(border=True):
        st.markdown("**Upload Calibration Image**")
        st.markdown(f":material/image: **{photo['name']}**  \nResolution: {photo['width']} × {photo['height']} px")
        st.button("Remove Image", key="m2_photo_remove", on_click=remove_legacy_photo)


def load_workflow():
    ss = st.session_state
    pending = ss["m2_load_pending"]
    with st.container(key="m2_npz_box"):            # one container for both states, so it looks the same before and after upload
        if pending is None:
            st.file_uploader("Upload Calibration File (.npz)", type=["npz"],
                             key=f"m2_cal_npz_{ss['m2_reset_count']}_{ss['m2_npz_gen']}", on_change=on_calibration_file)
            st.caption("The file is validated before it can be activated.")
        else:
            with st.container(key="m2_npz_label"):
                st.markdown("Upload Calibration File (.npz)")
            loaded_file_card()
    confirm_box("remove_file")
    if ss["m2_load_error"]:
        st.error(ss["m2_load_error"])
    if pending is None:
        return
    if pending["needs_image_size"]:
        legacy_resolution(pending)
    else:
        photo_summary(pending)
        loaded_results(pending)
        if is_active(pending):
            with st.container(key="m2_loaded_notes"), st.expander("Calibration Details & Important Notes"):
                loaded_notes(pending)
                calibration_details(pending)


# ---------------------------------------------------------------- Workflow B: calibrate a new camera

def board_settings():
    """Checkerboard inputs. Returns the validated settings (or None) and applies them to the wizard."""
    ss = st.session_state
    state = ss["m2_cal"]
    saved = state["settings"]
    st.subheader("1. Checkerboard settings")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    # The widgets start from the last accepted settings and are only given a start value while they have no state of their own
    # (Streamlit warns when a widget has both), so they survive a trip away from this page.
    start = lambda name, value: {} if fk(name) in ss else {"value": value}   # noqa: E731
    cols = c1.number_input("Inner corners (columns)", min_value=0, max_value=vc.CORNER_LIMITS[1], step=1, key=fk("cols"),
                           **start("cols", saved["cols"]))
    rows = c2.number_input("Inner corners (rows)", min_value=0, max_value=vc.CORNER_LIMITS[1], step=1, key=fk("rows"),
                           **start("rows", saved["rows"]))
    square = c3.number_input("Square size", min_value=0.0, step=0.5, format="%.2f", key=fk("square"),
                             **start("square", float(saved["square_value"])))
    unit = c4.selectbox("Unit", list(vc.UNIT_TO_MM), index=None if fk("unit") in ss else list(vc.UNIT_TO_MM).index(saved["unit"]),
                        key=fk("unit"), on_change=on_unit_change, filter_mode=None)
    ss["m2_unit_shown"] = unit
    st.caption(
        "**Inner corners** are the points where four squares meet inside the board, not the number of squares. A board with "
        "9 × 7 squares has 8 × 6 inner corners. The **square size** is the printed side length of one square; measure it on your "
        "printout. It is converted to millimetres, which sets the scale of the calibration. 8 × 6 corners and 20 mm are only defaults.")
    settings, errors = vc.validate_settings(cols, rows, square, unit)
    if errors:
        for error in errors:
            st.error(error)
        return None
    if vc.apply_settings(state, settings):
        ss["m2_cal_notice"] = "The checkerboard settings changed, so the photographs analysed earlier were cleared. Upload them again."
        st.rerun()     # the computed calibration (if any) depends on those photos; the next run clears it and shows the notice
    return settings


def photo_guidance():
    with st.container(border=True):
        st.markdown("**How to photograph the checkerboard**")
        st.markdown(
            "- Print a checkerboard and keep it **flat** (on a rigid board).\n"
            "- Use the **same camera, lens, zoom, focus mode and resolution** you will use to photograph the object.\n"
            "- Keep the **entire board visible**, sharp and evenly lit, with a white border around it.\n"
            "- Vary the **position, distance and tilt**, and put the board near the corners and edges of the frame in some photos.\n"
            "- Do not resize or crop the photographs. Use one resolution and orientation for all of them.")


def photo_slot(index, enabled):
    ss = st.session_state
    state = ss["m2_cal"]
    rec = state["photos"].get(index)
    if rec is not None:
        if rec.get("thumbnail"):
            st.image(rec["thumbnail"], width="stretch")
        st.caption(f"{rec['name']} · {rec['bytes'] / 1e6:.1f} MB" + (f" · {rec['size'][0]} × {rec['size'][1]} px" if rec.get("size") else ""))
        st.button("Remove photo", key=f"m2_cal_rm_{index}", on_click=on_photo_remove, args=(index,))
    else:
        st.file_uploader(f"Upload calibration photo {index + 1}", type=["jpg", "jpeg", "png"],
                         key=fk(f"up{index}_{state['slot_gen'].get(index, 0)}"), on_change=on_photo_upload, args=(index,),
                         disabled=not enabled, label_visibility="collapsed")


def photo_navigator(evaluation, enabled):
    ss = st.session_state
    state = ss["m2_cal"]
    n, current = state["n_slots"], state["current"]
    slot = evaluation["slots"][current]
    st.html('<div id="m2-cal-photo-anchor"></div>')
    st.markdown(f"#### Photo {current + 1} of {n} · {slot['status']}")
    photo_slot(current, enabled)
    if slot["status"] == vc.ACCEPTED:
        st.caption("Checkerboard detected; the corners are drawn on the preview.")
    for message in slot["messages"]:
        st.error(f"Photo {current + 1}: {message}")
    ss["m2_cal_goto"] = current
    st.selectbox("Go to photo", list(range(n)), key="m2_cal_goto", on_change=on_goto, label_visibility="collapsed", filter_mode=None,
                 format_func=lambda i: f"Photo {i + 1} – {evaluation['slots'][i]['status']}")
    nav = st.columns(2)
    nav[0].button("Previous", key="m2_cal_prev", on_click=on_step, args=(-1,), disabled=current == 0, width="stretch")
    nav[1].button("Next", key="m2_cal_next", on_click=on_step, args=(1,), disabled=current == n - 1, width="stretch")
    if ss.pop("m2_scroll_to_cal_photo", False):     # last element, so the photo section has rendered before the script runs
        st.html(SCROLL_TO_PHOTO_JS, unsafe_allow_javascript=True)


def photo_overview(evaluation):
    st.dataframe(pd.DataFrame([{"Photo": s["number"], "Status": s["status"], "File": "-" if s["rec"] is None else s["rec"]["name"]}
                               for s in evaluation["slots"]]).set_index("Photo"), width="stretch")


def run_validation(settings_ok, evaluation, max_listed=3):
    """Actionable reasons why a calibration cannot run yet. Empty list = ready."""
    if not settings_ok:
        return ["Enter valid checkerboard settings before running calibration."]
    if evaluation["n_uploaded"] == 0:
        return [f"No calibration photographs uploaded. Upload at least {vc.MIN_ACCEPTED} photographs in which the checkerboard is detected."]
    issues = [f"Photo {s['number']}: {' '.join(s['messages'])}" for s in evaluation["slots"] if s["status"] == vc.REJECTED]
    shown = issues[:max_listed]
    if len(issues) > max_listed:
        shown.append(f"{len(issues) - max_listed} more photo(s) have problems; see the photo list above.")
    missing = vc.MIN_ACCEPTED - evaluation["n_accepted"]
    if missing > 0:
        shown.append(f"Only {evaluation['n_accepted']} photograph{'s are' if evaluation['n_accepted'] != 1 else ' is'} accepted. At least "
                     f"{vc.MIN_ACCEPTED} are required. Add {missing} more accepted photograph{'s' if missing != 1 else ''}.")
    return shown


def run_step(evaluation, enabled):
    ss = st.session_state
    st.subheader("3. Run calibration")
    st.button("Run Camera Calibration", key="m2_cal_run", type="primary", on_click=request_run)
    if ss.pop("m2_cal_run_requested", False):
        problems = run_validation(enabled, evaluation)
        ss["m2_cal_validation"] = problems
        ss["m2_cal_validation_state"] = (evaluation["signature"], enabled, evaluation["n_accepted"], evaluation["n_uploaded"])
        if not problems:
            try:
                with st.spinner("Calibrating the camera..."):
                    result = vc.run_calibration(ss["m2_cal"])
            except vc.CalibrationError as error:
                ss["m2_cal_error"] = str(error)        # a failed attempt leaves any computed or active calibration untouched
            else:
                ss["m2_computed"], ss["m2_cal_error"] = result, None
    if ss["m2_cal_validation"] and ss.get("m2_cal_validation_state") != (evaluation["signature"], enabled, evaluation["n_accepted"],
                                                                         evaluation["n_uploaded"]):
        ss["m2_cal_validation"] = []                   # inputs changed since the refused click: the message would be stale
    for problem in ss["m2_cal_validation"]:
        st.warning(problem)
    if ss["m2_cal_error"]:
        st.error(ss["m2_cal_error"])
    computed = ss["m2_computed"]
    if computed is not None:
        calibration_results(computed, "computed")
    calibration_download(computed)


def new_camera_steps(evaluation):
    ss = st.session_state
    state = ss["m2_cal"]
    with st.expander("How to Calibrate Your Camera", expanded=False):
        st.write("Photograph a checkerboard with your smartphone camera from several positions and orientations, upload the photographs, "
                 "then run the calibration. Your photographs are analysed in this session only and are not saved.")
        photo_guidance()
    settings = board_settings()
    enabled = settings is not None

    st.subheader("2. Upload calibration photographs")
    choice = st.number_input("Number of photo slots", min_value=vc.MIN_SLOTS, max_value=vc.MAX_SLOTS, value=state["n_slots"], step=1,
                             key=fk("nslots"), help=f"{vc.MIN_SLOTS} to {vc.MAX_SLOTS} photographs. Raise it to add more photographs; "
                             "lowering it hides the last slots but keeps their photos.")
    vc.set_photo_count(state, int(choice))
    evaluation = vc.evaluate(state)                       # recompute: the slot count may just have changed
    st.progress(min(1.0, evaluation["n_accepted"] / vc.MIN_ACCEPTED))
    st.write(f"**{evaluation['n_uploaded']} uploaded** · **{evaluation['n_accepted']} accepted** · **{evaluation['n_rejected']} rejected**")
    st.caption(f"At least {vc.MIN_ACCEPTED} accepted photographs are required (a project rule, not a mathematical limit); "
               f"{vc.RECOMMENDED_ACCEPTED[0]}-{vc.RECOMMENDED_ACCEPTED[1]} give a more robust result.")
    photo_navigator(evaluation, enabled)
    with st.popover("All photos at a glance"):
        photo_overview(evaluation)
    st.button("Remove all calibration photos", key="m2_cal_clear_photos", on_click=clear_all_photos, disabled=not state["photos"])
    run_step(evaluation, enabled)


# ---------------------------------------------------------------- Step 1

def calibrate_workflow():
    """Step 1: the two workflows. Selecting one never activates a calibration; activation is always an explicit button."""
    ss = st.session_state
    state = ss["m2_cal"]
    evaluation = vc.evaluate(state)
    computed = ss["m2_computed"]
    if computed is not None and computed["input_signature"] != evaluation["signature"]:
        # the photographs or board settings behind it changed: the computed result (and the active copy of it) no longer applies
        if is_active(computed):
            set_visitor_calibration(None)
        ss["m2_computed"] = None
        ss["m2_cal_notice"] = ((ss["m2_cal_notice"] + " ") if ss["m2_cal_notice"] else "") + (
            "The photographs or checkerboard settings changed, so the previously computed calibration was cleared.")
    workflow = workflow_selector()
    if ss["m2_cal_notice"]:
        st.info(ss["m2_cal_notice"])
    if workflow == "run":
        new_camera_steps(evaluation)
    elif workflow == "load":
        load_workflow()


def geometry_assumptions():
    with st.expander("What a valid measurement requires"):
        st.markdown(
            "- The photograph comes from the **same camera, lens and settings** that were calibrated, at the **same image "
            "resolution**. The calibration is not rescaled.\n"
            "- The object is **flat**, and its face lies in a plane **perpendicular to the camera's optical axis** "
            "(fronto-parallel). A tilted or turned object is not measured correctly.\n"
            "- The **camera-to-object distance Z** is the known depth of that plane along the optical axis, measured carefully. "
            "Distance alone is not enough: with an unknown orientation or depth, a single image cannot give metric size.\n"
            "- The four corners you select are the object's outer corners, in the order top-left, top-right, bottom-right, "
            "bottom-left."
        )


# ---------------------------------------------------------------- visitor experiment (Steps 2-4)

VALIDATION_MIN_MEASUREMENTS = 20     # a complete validation experiment: at least this many measurements ...
VALIDATION_MIN_DISTANCE_CM = 200     # ... at camera-to-object distances greater than this
MAX_PLANNED = 100
NEW_OBJECT = "New Object"
EXPORT_COLUMNS = [CSV_COLUMNS[0], "Object"] + CSV_COLUMNS[1:]
CONFIRMS = {
    "restart": ("Restart the experiment? This deletes your {n} saved measurement{s} and keeps your calibration.", "Restart Experiment"),
    "change_cal": ("Change the calibration? This restarts the experiment and deletes your {n} saved measurement{s}.", "Restart and Change Calibration"),
    "remove_file": ("Remove the calibration file? This clears its calibration, restarts the experiment and deletes your {n} saved measurement{s}.", "Remove File and Restart"),
    "reset_all": ("Reset everything? This deletes your calibration and your {n} saved measurement{s}.", "Yes, Reset Everything"),
}


def request_scroll(anchor_id):
    """Asks the next run to scroll to an anchor element: the existing one-shot scroll script, pointed at another anchor."""
    st.session_state["m2_scroll_anchor"] = anchor_id


def new_experiment():
    return {"started": False, "finished": False, "results": [], "next_uid": 1}


def normalized(name):
    return " ".join(str(name).split()).casefold()


def compute_measurement(points, distance, actual_width, actual_height, fx, fy):
    """Estimated size and errors from four corners (top-left, top-right, bottom-right, bottom-left); values rounded to 2 decimals
    like the validation CSV. The same equations as the original experiment."""
    top_left, top_right, bottom_right, bottom_left = points
    top_width = calculate_length(top_left, top_right, distance, fx, fy)
    bottom_width = calculate_length(bottom_left, bottom_right, distance, fx, fy)
    estimated_width = (top_width + bottom_width) / 2
    left_height = calculate_length(top_left, bottom_left, distance, fx, fy)
    right_height = calculate_length(top_right, bottom_right, distance, fx, fy)
    estimated_height = (left_height + right_height) / 2
    width_error = abs(estimated_width - actual_width)
    height_error = abs(estimated_height - actual_height)
    return {
        "Estimated_Width_cm": round(float(estimated_width), 2), "Width_Error_cm": round(float(width_error), 2),
        "Width_Error_Percent": round(float(width_error / actual_width * 100), 2),
        "Estimated_Height_cm": round(float(estimated_height), 2), "Height_Error_cm": round(float(height_error), 2),
        "Height_Error_Percent": round(float(height_error / actual_height * 100), 2),
    }


def visitor_rows():
    """The saved measurements as numbered rows (the same column names as the validation CSV, plus Object)."""
    rows = []
    for number, record in enumerate(st.session_state["m2_exp"]["results"], start=1):
        row = {column: record[column] for column in CSV_COLUMNS if column != "Trial"}
        row.update(Trial=number, Object=record["Object"])
        rows.append(row)
    return rows


def results_csv(rows):
    def safe(value):    # a leading = + - @ would be read as a formula by spreadsheet programs
        return "'" + value if isinstance(value, str) and value[:1] in ("=", "+", "-", "@") else value
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for row in rows:
        writer.writerow([safe(row[column]) for column in EXPORT_COLUMNS])
    return buffer.getvalue().encode("utf-8")


# ---- callbacks

def clear_measurement_form():
    """Empties the object, distance and image of the current trial; the calibration and saved measurements are kept."""
    ss = st.session_state
    ss["m2_object_name"] = ""
    ss["m2_actual_width"] = None
    ss["m2_actual_height"] = None
    ss["m2_distance"] = None
    ss["m2_same_object"] = NEW_OBJECT
    ss["m2_img_gen"] += 1
    if "m2_click_generation" in ss:         # a statement, not a conditional expression: Streamlit would print its None result
        reset_points()
    else:
        clear_live_result()
    ss.pop("m2_result_notice", None)


def start_experiment():
    st.session_state["m2_exp"]["started"] = True


def finish_experiment():
    """Calculate Experiment Results / Finish Early & View Results: reveals Step 4."""
    st.session_state["m2_exp"]["finished"] = True
    request_scroll("m2-results-anchor")


def resume_experiment():
    ss = st.session_state
    ss["m2_exp"]["finished"] = False
    if len(ss["m2_exp"]["results"]) >= ss["m2_planned"] and ss["m2_planned"] < MAX_PLANNED:
        ss["m2_planned"] += 1
    request_scroll("m2-object-anchor")


def add_measurement():
    ss = st.session_state
    ss["m2_planned"] = min(MAX_PLANNED, ss["m2_planned"] + 1)
    request_scroll("m2-object-anchor")


def save_result():
    """Saves the calculated measurement once: the live result is taken out of the session first, and its id is checked."""
    ss = st.session_state
    result = ss.pop("m2_live_result", None)
    if result is None:
        return
    exp = ss["m2_exp"]
    if any(saved["uid"] == result["uid"] for saved in exp["results"]):
        return
    exp["results"].append(result)
    clear_measurement_form()
    ss["m2_saved_notice"] = f"Measurement {len(exp['results'])} saved."
    request_scroll("m2-object-anchor")


def apply_same_object():
    ss = st.session_state
    choice = ss.get("m2_same_object")
    if choice is None:
        return
    if choice == NEW_OBJECT:                # start a new object: empty the object fields (the distance and image are kept)
        ss["m2_object_name"] = ""
        ss["m2_actual_width"] = None
        ss["m2_actual_height"] = None
        clear_live_result()
        return
    for record in reversed(ss["m2_exp"]["results"]):
        if normalized(record["Object"]) == choice:
            ss["m2_object_name"] = record["Object"]
            ss["m2_actual_width"] = record["Actual_Width_cm"]
            ss["m2_actual_height"] = record["Actual_Height_cm"]
            break
    clear_live_result()


def run_confirmed(kind):
    ss = st.session_state
    if kind == "reset_all":
        reset_all()
        return
    if kind == "remove_file":
        clear_loaded_calibration()
    ss["m2_exp"] = new_experiment()         # restart: saved measurements and the plan go, the calibration stays
    ss["m2_confirm"] = None
    ss["m2_row_action"] = None
    clear_measurement_form()


def ask_confirm(kind):
    ss = st.session_state
    if ss["m2_exp"]["results"]:
        ss["m2_confirm"] = kind
    else:
        run_confirmed(kind)                 # nothing saved yet, so nothing to lose


def cancel_confirm():
    st.session_state["m2_confirm"] = None


def confirm_box(kind):
    ss = st.session_state
    if ss.get("m2_confirm") != kind:
        return
    n = len(ss["m2_exp"]["results"])
    message, label = CONFIRMS[kind]
    st.warning(message.format(n=n, s="" if n == 1 else "s"))
    yes, no = st.columns(2)
    yes.button(label, key=f"m2_confirm_yes_{kind}", type="primary", on_click=run_confirmed, args=(kind,), width="stretch")
    no.button("Cancel", key=f"m2_confirm_no_{kind}", on_click=cancel_confirm, width="stretch")


def pick_row(action):
    st.session_state["m2_row_action"] = action


def save_edit(uid):
    ss = st.session_state
    record = next((r for r in ss["m2_exp"]["results"] if r["uid"] == uid), None)
    if record is None:
        return
    name = " ".join(str(ss.get(f"m2_edit_name_{uid}") or "").split())
    values = [ss.get(f"m2_edit_{key}_{uid}") for key in ("width", "height", "distance")]
    if not name or any(v is None or v <= 0 for v in values):
        ss["m2_row_error"] = "Enter an object name and positive width, height and distance."
        return
    width, height, distance = values
    record.update(Object=name, Actual_Width_cm=width, Actual_Height_cm=height, Distance_cm=distance,
                  **compute_measurement(record["points"], distance, width, height, record["fx"], record["fy"]))
    ss["m2_row_error"] = None
    ss["m2_row_action"] = None


def delete_row(uid):
    ss = st.session_state
    ss["m2_exp"]["results"] = [r for r in ss["m2_exp"]["results"] if r["uid"] != uid]
    ss["m2_row_action"] = None
    ss.pop("m2_row_pick", None)
    if not ss["m2_exp"]["results"]:
        ss["m2_exp"]["finished"] = False    # nothing left to show results for


# ---- Step 2

def experiment_section():
    ss = st.session_state
    calibration = ss["m2_visitor_calibration"]
    exp = ss["m2_exp"]
    st.subheader("Step 2: Configure Your Experiment")
    if calibration is None:
        st.info("Load or calibrate your camera in Step 1 first. Once a calibration is active, you can set up and run your experiment.")
        return
    size = (f"{calibration['image_width']} × {calibration['image_height']} px" if calibration.get("image_width") else "unknown resolution")
    st.caption(f"Using active calibration: {size}")
    st.write(f"Measure a single object, or run a complete validation experiment. A complete validation experiment needs at least "
             f"{VALIDATION_MIN_MEASUREMENTS} measurements at camera-to-object distances greater than {VALIDATION_MIN_DISTANCE_CM} cm. "
             "Shorter distances are fine for general object measurement.")
    done = len(exp["results"])
    ss["m2_planned"] = max(ss["m2_planned"], done, 1)
    st.number_input("Number of measurements", min_value=max(1, done), max_value=MAX_PLANNED, step=1, key="m2_planned")
    if not exp["started"]:
        st.button("Start Experiment", key="m2_start", type="primary", on_click=start_experiment)
        return
    st.button("Restart Experiment", key="m2_restart", on_click=ask_confirm, args=("restart",))
    confirm_box("restart")
    measure_step(calibration)
    if exp["finished"] and exp["results"]:
        results_section()


# ---- Step 3

def measure_step(calibration):
    ss = st.session_state
    exp = ss["m2_exp"]
    done, planned = len(exp["results"]), ss["m2_planned"]
    st.html('<div id="m2-object-anchor"></div>')
    st.subheader("Step 3: Measure Your Objects")
    if "m2_saved_notice" in ss:
        st.success(ss.pop("m2_saved_notice"))
    st.progress(min(1.0, done / planned))
    st.caption(f"{done} of {planned} measurements completed")
    if exp["finished"]:
        st.info("Your experiment results are shown below. Use Continue Measuring at the bottom of Step 4 to add more measurements.")
    elif done >= planned:
        st.markdown("#### Measurements Completed")
        st.write("All planned measurements have been saved. You can now calculate your experiment results or add another measurement.")
        calculate, add = st.columns(2)
        calculate.button("Calculate Experiment Results", key="m2_calc_results", type="primary", on_click=finish_experiment, width="stretch")
        add.button("Add Another Measurement", key="m2_add_more", on_click=add_measurement, disabled=planned >= MAX_PLANNED, width="stretch")
    else:
        st.markdown(f"#### Measurement {done + 1} of {planned}")
        geometry_assumptions()
        visitor_measurement(calibration)
        if done:
            st.button("Finish Early & View Results", key="m2_finish", on_click=finish_experiment)


# ---- Step 4

def results_section():
    ss = st.session_state
    exp = ss["m2_exp"]
    rows = visitor_rows()
    done, planned = len(rows), ss["m2_planned"]
    qualifying = sum(1 for row in rows if row["Distance_cm"] > VALIDATION_MIN_DISTANCE_CM)
    stats = percent_stats(rows)
    objects = {normalized(row["Object"]) for row in rows}
    st.html('<div id="m2-results-anchor"></div>')
    st.subheader("Step 4: Experiment Results")
    with st.container(key="m2_res_body"):
        if done >= planned:
            plan = f"complete ({done} of {planned} measurements)"
        elif exp["finished"]:
            plan = f"finished early ({done} of {planned} planned measurements)"
        else:
            plan = f"in progress ({done} of {planned} planned measurements)"
        st.write(f"**Measurement plan:** {plan}")
        if qualifying >= VALIDATION_MIN_MEASUREMENTS:
            st.success(f"Full validation requirement met: {qualifying} measurements at distances greater than {VALIDATION_MIN_DISTANCE_CM} cm.")
        for column, card in zip(st.columns(2), (("Measurements Completed", f"{done}", f"of {planned} planned"),
                                                ("Distinct Objects", f"{len(objects)}", "by object name"))):
            with column:
                metric_card(*card)
        for column, card in zip(st.columns(2), (("Mean Width Percentage Error", f"{stats['Width']['mpe']:.2f}%", f"across {done} measurement{'' if done == 1 else 's'}"),
                                                ("Mean Height Percentage Error", f"{stats['Height']['mpe']:.2f}%", f"across {done} measurement{'' if done == 1 else 's'}"))):
            with column:
                metric_card(*card)

        st.markdown("#### Measurement Results")
        kv_table("Trial", [[
            r["Trial"], r["Object"], r["Image"], f"{r['Distance_cm']:g}",
            f"{r['Actual_Width_cm']:.2f}", f"{r['Estimated_Width_cm']:.2f}", f"{r['Width_Error_cm']:.2f}", f"{r['Width_Error_Percent']:.2f}",
            f"{r['Actual_Height_cm']:.2f}", f"{r['Estimated_Height_cm']:.2f}", f"{r['Height_Error_cm']:.2f}", f"{r['Height_Error_Percent']:.2f}",
        ] for r in rows], ["Object", "Image", "Distance (cm)", "Actual width (cm)", "Estimated width (cm)", "Width error (cm)",
                           "Width error (%)", "Actual height (cm)", "Estimated height (cm)", "Height error (cm)", "Height error (%)"])
        st.download_button("Download Results (CSV)", results_csv(rows), file_name="module2_experiment_results.csv", mime="text/csv",
                           key="m2_results_download", icon=":material/download:", on_click="ignore")

        st.markdown("#### Percentage Error per Measurement")
        trial_chart(rows)

        st.markdown("#### Error Statistics")
        st.table({"Statistic": ["Mean Absolute Error", "Mean Percentage Error", "Standard Deviation of Percentage Error", "RMSE"],
                  **{dimension: [f"{stats[dimension]['mae']:.2f} cm", f"{stats[dimension]['mpe']:.2f} %",
                                 f"{stats[dimension]['sd']:.2f} %", f"{stats[dimension]['rmse']:.2f} cm"]
                     for dimension in ("Width", "Height")}})

        st.markdown("#### Edit or Delete a Measurement")
        uids = [record["uid"] for record in exp["results"]]
        labels = {record["uid"]: f"Measurement {number} · {record['Object']} · {record['Distance_cm']:g} cm"
                  for number, record in enumerate(exp["results"], start=1)}
        uid = st.selectbox("Measurement", uids, format_func=labels.get, key="m2_row_pick", on_change=pick_row, args=(None,))
        edit, delete = st.columns(2)
        edit.button("Edit", key="m2_row_edit", on_click=pick_row, args=("edit",), width="stretch")
        delete.button("Delete", key="m2_row_delete", on_click=pick_row, args=("delete",), width="stretch")
        record = next((r for r in exp["results"] if r["uid"] == uid), None)
        action = ss["m2_row_action"]
        if record is not None and action == "edit":
            with st.container(border=True):
                st.text_input("Object Name", value=record["Object"], key=f"m2_edit_name_{uid}")
                width_column, height_column = st.columns(2)
                width_column.number_input("Actual Width (cm)", value=float(record["Actual_Width_cm"]), min_value=0.0, step=0.1,
                                          key=f"m2_edit_width_{uid}")
                height_column.number_input("Actual Height (cm)", value=float(record["Actual_Height_cm"]), min_value=0.0, step=0.1,
                                           key=f"m2_edit_height_{uid}")
                st.number_input("Camera-to-Object Distance (cm)", value=float(record["Distance_cm"]), min_value=0.0, step=1.0,
                                key=f"m2_edit_distance_{uid}")
                if ss.get("m2_row_error"):
                    st.error(ss["m2_row_error"])
                save, cancel = st.columns(2)
                save.button("Save Changes", key="m2_edit_save", type="primary", on_click=save_edit, args=(uid,), width="stretch")
                cancel.button("Cancel", key="m2_edit_cancel", on_click=pick_row, args=(None,), width="stretch")
                st.caption("The estimated size and errors are recalculated from the corners you selected. To change the corners, "
                           "delete this measurement and measure it again.")
        elif record is not None and action == "delete":
            st.warning(f"Delete {labels[uid]}? This cannot be undone.")
            yes, no = st.columns(2)
            yes.button("Delete Measurement", key="m2_delete_yes", type="primary", on_click=delete_row, args=(uid,), width="stretch")
            no.button("Cancel", key="m2_delete_no", on_click=pick_row, args=(None,), width="stretch")


def visitor_measurement(visitor_calibration):
    """One measurement trial: object information, image upload, four-corner selection and calculation (Step 3)."""
    ss = st.session_state
    visitor_camera_matrix = visitor_calibration["camera_matrix"]
    visitor_distortion_coefficients = visitor_calibration["distortion_coefficients"]
    visitor_fx = visitor_camera_matrix[0, 0]
    visitor_fy = visitor_camera_matrix[1, 1]

    st.subheader("Object Information")

    # Reuses the name and size of an object that was already measured (for example at another distance)
    saved = {}
    for record in ss["m2_exp"]["results"]:
        saved.setdefault(normalized(record["Object"]), record["Object"])
    if saved:
        st.caption("Select a previously measured object to reuse its dimensions, or select New Object.")
        st.selectbox("Select Object", [NEW_OBJECT] + list(saved), format_func=lambda key: saved.get(key, key), key="m2_same_object",
                     on_change=apply_same_object)

    # Gets the object information
    object_name = st.text_input("Object Name", key="m2_object_name", on_change=clear_live_result)

    width_column, height_column = st.columns(2)
    actual_width = width_column.number_input(
        "Actual Width (cm)",
        value=None,
        placeholder="0.00",
        min_value=0.0,
        step=0.1,
        key="m2_actual_width",
        on_change=clear_live_result
    )

    actual_height = height_column.number_input(
        "Actual Height (cm)",
        value=None,
        placeholder="0.00",
        min_value=0.0,
        step=0.1,
        key="m2_actual_height",
        on_change=clear_live_result
    )

    st.subheader("Measurement Information")

    # Gets the camera-to-object distance
    distance = st.number_input(
        "Camera-to-Object Distance (cm)",
        value=None,
        placeholder="0.00",
        min_value=0.0,
        step=1.0,
        key="m2_distance",
        on_change=clear_live_result
    )
    if distance is not None and 0 < distance <= VALIDATION_MIN_DISTANCE_CM:
        st.caption(f"Fine for general measurement. Only distances greater than {VALIDATION_MIN_DISTANCE_CM} cm count toward the full "
                   "validation requirement.")

    st.button("Reset Inputs", on_click=reset_inputs)

    st.caption("Photograph the object with the **same camera, lens, zoom, focus mode and settings** as the calibration, at exactly the "
               "calibration resolution. A matching resolution alone does not guarantee the calibration applies.")

    # Stores the selected points
    if "m2_selected_points" not in st.session_state:
        st.session_state.m2_selected_points = []
        st.session_state.m2_click_generation = 0

    # Gets the measurement image
    uploaded_image = st.file_uploader(
        "Upload Measurement Image",
        type=["jpg", "jpeg", "png"],
        key=f"m2_meas_img_{ss['m2_reset_count']}_{ss['m2_img_gen']}",
        on_change=reset_points
    )
    st.caption(f"JPG, JPEG or PNG, up to {st.get_option('server.maxUploadSize')} MB. The photograph must be exactly "
               f"{visitor_calibration['image_width']} × {visitor_calibration['image_height']} px.")

    if uploaded_image is not None:

        # Reads the uploaded image
        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            st.error("The measurement image could not be decoded. Upload a JPG or PNG photograph.")
            return

        # Blocks measurement unless the photograph has the calibration resolution (the intrinsics are never rescaled)
        compatibility_error = vc.check_measurement_image(visitor_calibration, (image.shape[1], image.shape[0]))
        if compatibility_error is not None:
            st.error(compatibility_error)
            return

        # Corrects lens distortion
        undistorted = cv2.undistort(
            image,
            visitor_camera_matrix,
            visitor_distortion_coefficients
        )

        # Converts the image from BGR to RGB
        undistorted_rgb = cv2.cvtColor(
            undistorted,
            cv2.COLOR_BGR2RGB
        )

        st.subheader("Select Object Corners")

        st.write(
            "Click the four corners of the object in this exact order:"
        )

        st.write("1. **Top-left**")
        st.write("2. **Top-right**")
        st.write("3. **Bottom-right**")
        st.write("4. **Bottom-left**")

        st.info(
            "Click as close as possible to each outer corner of the object."
        )

        # Defines the display width
        display_width = 500

        # Calculates the display scale
        original_width = undistorted_rgb.shape[1]
        display_scale = display_width / original_width

        # Resizes the image for display
        display_height = int(
            undistorted_rgb.shape[0] * display_scale
        )

        display_image = cv2.resize(
            undistorted_rgb,
            (display_width, display_height)
        )

        corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        points = st.session_state.m2_selected_points

        st.button("Reset Points", on_click=reset_points)

        # Displays the next corner and confirms the selected corners
        if len(points) < 4:
            st.info(
                f"Select point {len(points) + 1} of 4: "
                f"{corner_names[len(points)].upper()}"
            )
        else:
            st.success("All 4 corners selected.")

        for corner_name in corner_names[:len(points)]:
            st.write(f"✓ {corner_name} selected")

        # Draws numbered markers at the selected display positions
        for point_number, (original_x, original_y) in enumerate(points, start=1):
            display_x = round(original_x * display_scale)
            display_y = round(original_y * display_scale)
            marker_position = (display_x, display_y)
            cv2.circle(display_image, marker_position, 7, (0, 0, 0), -1)
            cv2.circle(display_image, marker_position, 5, (255, 255, 0), -1)

            # Keeps the label inside the displayed image
            label = str(point_number)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            label_position = (
                max(2, min(display_x + 10, display_width - label_width - 3)),
                max(label_height + 3, min(display_y - 10, display_height - baseline - 3))
            )
            cv2.putText(
                display_image, label, label_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA
            )
            cv2.putText(
                display_image, label, label_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
            )

        # Displays the resized clickable image. The component keeps one stable key, so it is never re-created (a new key for every
        # click re-mounted it and the image vanished); a click is recognised as new by its timestamp.
        coordinates = streamlit_image_coordinates(
            display_image,
            key="m2_measurement_image"
        )

        # Converts display coordinates to original coordinates
        if coordinates is not None and coordinates.get("unix_time") != st.session_state.get("m2_last_click"):
            # The component keeps returning its last click, so each click is handled once (Reset Points never re-adds an old one)
            st.session_state["m2_last_click"] = coordinates.get("unix_time")
            if len(points) < 4:
                original_x = coordinates["x"] / display_scale
                original_y = coordinates["y"] / display_scale
                points.append((original_x, original_y))
                st.rerun()

        if st.button("Calculate Measurement", disabled=len(points) != 4):
            if not str(object_name or "").strip():
                st.error("Enter an object name.")
            elif any(value is None or value <= 0 for value in (distance, actual_width, actual_height)):
                st.error("Enter a positive distance, actual width, and actual height.")
            else:
                # Stores a temporary snapshot of the completed measurement; Save & Next Measurement keeps it
                measured = compute_measurement(points, distance, actual_width, actual_height, visitor_fx, visitor_fy)
                clear_live_result()
                exp = ss["m2_exp"]
                uid = exp["next_uid"]
                exp["next_uid"] += 1
                ss.m2_live_result = {
                    "uid": uid,
                    "Object": " ".join(object_name.split()),
                    "Image": uploaded_image.name,
                    "Distance_cm": distance,
                    "Actual_Width_cm": actual_width,
                    "Actual_Height_cm": actual_height,
                    **measured,
                    "points": [tuple(point) for point in points],
                    "fx": float(visitor_fx),
                    "fy": float(visitor_fy),
                }

        if "m2_result_notice" in st.session_state:
            st.success(st.session_state.pop("m2_result_notice"))

        if "m2_live_result" in st.session_state:
            result = st.session_state.m2_live_result

            # Displays the measurement results
            st.subheader("Current Measurement Results")
            width_column, height_column = st.columns(2)
            with width_column:
                st.metric("Estimated Width (cm)", f"{result['Estimated_Width_cm']:.2f}")
                st.metric("Actual Width (cm)", f"{result['Actual_Width_cm']:.2f}")
                st.metric("Width Error (cm)", f"{result['Width_Error_cm']:.2f}")
                st.metric("Width Error (%)", f"{result['Width_Error_Percent']:.2f}")
            with height_column:
                st.metric("Estimated Height (cm)", f"{result['Estimated_Height_cm']:.2f}")
                st.metric("Actual Height (cm)", f"{result['Actual_Height_cm']:.2f}")
                st.metric("Height Error (cm)", f"{result['Height_Error_cm']:.2f}")
                st.metric("Height Error (%)", f"{result['Height_Error_Percent']:.2f}")

            save_column, discard_column = st.columns(2)
            save_column.button("Save & Next Measurement", key="m2_save_result", type="primary", on_click=save_result, width="stretch")
            discard_column.button("Discard Result", on_click=discard_result, width="stretch")


def page_headings(title, subtitle=None, anchor=None, variant=None):
    """One prominent page heading, with an optional secondary subtitle."""
    with st.container(key=f"m2_headings_{variant}" if variant else "m2_headings_default"):
        with st.container(key="m2_headings"):
            st.title(title, anchor=anchor or False)
            if subtitle:
                st.header(subtitle, anchor=False)


def locked_calibration():
    """Step 1 once measurements are saved: the calibration stays fixed; changing it restarts the experiment."""
    active = st.session_state["m2_visitor_calibration"]
    n = len(st.session_state["m2_exp"]["results"])
    size = f"{active['image_width']} × {active['image_height']} px" if active and active.get("image_width") else "unknown resolution"
    st.info(f"Calibration in use for this experiment: {size}. Changing it restarts the experiment and deletes your {n} saved measurement"
            f"{'' if n == 1 else 's'}.")
    st.button("Change Calibration", key="m2_change_cal", on_click=ask_confirm, args=("change_cal",))
    confirm_box("change_cal")


def visitor_page():
    page_headings(VISITOR_TITLE, anchor="m2-visitor-heading", variant="visitor")
    st.subheader("Step 1: Calibrate Your Camera")
    if st.session_state["m2_exp"]["results"] and st.session_state["m2_visitor_calibration"] is not None:
        locked_calibration()
    else:
        calibrate_workflow()

    experiment_section()
    exp = st.session_state["m2_exp"]
    if exp["finished"] and exp["results"] and st.session_state["m2_visitor_calibration"] is not None:
        continue_column, reset_column = st.columns(2)       # bottom of Step 4
        continue_column.button("Continue Measuring", key="m2_continue", type="primary", on_click=resume_experiment, width="stretch")
        reset_column.button("Reset All", key="m2_reset_all", on_click=ask_confirm, args=("reset_all",), width="stretch")
    else:
        st.button("Reset All", key="m2_reset_all", on_click=ask_confirm, args=("reset_all",))
    confirm_box("reset_all")
    target = st.session_state.pop("m2_scroll_anchor", None)
    if target:                                  # last element, so the anchors have rendered before the script runs
        st.html(SCROLL_TO_PHOTO_JS.replace("m2-cal-photo-anchor", target), unsafe_allow_javascript=True)


def home_page():
    with st.container(key="m2_home_layout"):
        with st.container(key="m2_home_title"):
            st.title(HOME_TITLE, anchor=False)
        with st.container(key="m2_home_subtitle"):
            st.header(HOME_SUBTITLE, anchor=False)
        with st.container(key="m2_home_nav"):
            home_buttons()


def home_buttons():
    st.button("View My Experiments & Results", key="m2_open_experiment", on_click=go_to, args=("experiment",), width="stretch")
    st.button("Run Your Own Experiment", key="m2_open_visitor", on_click=go_to, args=("visitor",), width="stretch")


# ---------------------------------------------------------------- saved experiment (read-only)

def display_image(name, caption):
    """One saved display photograph (original, EXIF-oriented copy). A missing file is reported instead of hidden."""
    path = DISPLAY_DIR / name
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Display image {name} is unavailable.")


def image_row(items, per_row=2):
    """items: (file name in display/, caption)."""
    for column, (name, caption) in zip(st.columns(per_row), items):
        with column:
            display_image(name, caption)


def metric_card(label, value, note=None):
    with st.container(border=True):
        st.metric(label, value)
        if note:
            st.caption(note)


def percent_stats(rows):
    """Mean absolute error, mean percentage error, SD of the percentage error and RMSE per dimension, from the saved CSV."""
    out = {}
    for dimension in ("Width", "Height"):
        errors = np.array([float(row[f"{dimension}_Error_cm"]) for row in rows])
        percent_errors = np.array([float(row[f"{dimension}_Error_Percent"]) for row in rows])
        out[dimension] = {"mae": np.mean(errors), "mpe": np.mean(percent_errors), "sd": np.std(percent_errors),
                          "rmse": np.sqrt(np.mean(errors ** 2)),
                          "under": sum(float(r[f"Estimated_{dimension}_cm"]) < float(r[f"Actual_{dimension}_cm"]) for r in rows)}
    return out


def trial_chart(rows):
    """Grouped (side-by-side) bars: the width and the height percentage error of every trial, both measured from zero."""
    import altair as alt          # ships with Streamlit
    trials = [int(r["Trial"]) for r in rows]
    series = ["Width error (%)", "Height error (%)"]
    data = pd.DataFrame([{"Trial": int(r["Trial"]), "Error type": name, "Percentage error (%)": float(r[column])}
                         for r in rows for name, column in zip(series, ("Width_Error_Percent", "Height_Error_Percent"))])
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Trial:O", sort=trials, title="Trial", axis=alt.Axis(labelAngle=0, labelOverlap=False, labelFontSize=11)),
        xOffset=alt.XOffset("Error type:N", sort=series),
        y=alt.Y("Percentage error (%):Q", title="Percentage error (%)", scale=alt.Scale(domainMin=0, nice=True)),
        color=alt.Color("Error type:N", sort=series, scale=alt.Scale(domain=series, range=["#0068C9", "#83C9FF"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Trial:O"), alt.Tooltip("Error type:N"), alt.Tooltip("Percentage error (%):Q", format=".2f")])
    st.altair_chart(chart.properties(height=340).configure(background="rgba(0,0,0,0)"), width="stretch")


def show_experiment():
    with st.container(key="m2_res_head"):
        with st.container(key="m2_res_title"):
            st.title("My Experiment & Results", anchor=False)
        st.header("Camera Calibration & 2D Object Measurement", anchor=False)

    with st.container(key="m2_res_body"):
        try:
            rows = read_validation_rows()
        except (OSError, ValueError, csv.Error, TypeError) as error:
            st.error(f"Validation results could not be loaded: {error}")
            return
        trial_count = len(rows)
        distances = sorted({float(row["Distance_cm"]) for row in rows})
        objects = {}
        for row in rows:
            objects.setdefault((float(row["Actual_Width_cm"]), float(row["Actual_Height_cm"])), len(objects) + 1)
        stats = percent_stats(rows)
        accepted = int(calibration["successful_images"])
        columns_n, rows_n = int(calibration["checkerboard_columns"]), int(calibration["checkerboard_rows"])
        square_mm = float(calibration["square_size_mm"])
        distance_text = ", ".join(f"{d:g}" for d in distances)

        # ---- objective and summary
        with st.container(key="m2_res_objective"):
            st.markdown("##### Objective")
            st.write(
                "The objective of this experiment is to calibrate a smartphone camera using OpenCV and estimate the real-world 2D "
                "dimensions of objects using perspective projection. The method is validated through 20 measurements of five objects "
                "at four camera-to-object distances (210, 220, 230, and 250 cm). The estimated dimensions are compared with the "
                "actual measured dimensions, and error statistics are used to evaluate the accuracy of the method."
            )
        for column, card in zip(st.columns(2), (
                ("Calibration Images", f"{accepted} / {CALIBRATION_PHOTOS_CAPTURED}", "successfully accepted"),
                ("Validation Measurements", f"{trial_count}", f"{len(objects)} objects at {len(distances)} distances"))):
            with column:
                metric_card(*card)
        for column, card in zip(st.columns(2), (
                ("Mean Width Percentage Error", f"{stats['Width']['mpe']:.2f}%", f"across {trial_count} measurements"),
                ("Mean Height Percentage Error", f"{stats['Height']['mpe']:.2f}%", f"across {trial_count} measurements"))):
            with column:
                metric_card(*card)

        # ---- 1. experimental setup
        with st.container(key="m2_res_sec_1"):
            st.header("1. Experimental Setup", anchor=False)
            st.subheader("Camera", anchor=False)
            st.table(pd.DataFrame({"Item": ["Device", "Camera", "Focal length", "Aperture", "Photograph resolution"],
                                   "Value": ["iPhone 17 Pro Max", "1× main camera", "6.765 mm", "f/1.78",
                                             "4284 × 5712 pixels (portrait, EXIF orientation applied)"]}).set_index("Item"))
            st.caption("The photographs' EXIF data lists the iPhone 17 Pro Max main lens (6.765 mm, f/1.78) for both the calibration and "
                       "the measurement photographs. The measurement photographs use the same resolution, so the saved focal lengths "
                       "apply to them unchanged.")

            st.subheader("Checkerboard", anchor=False)
            image_column, spec_column = st.columns(2)
            with image_column:
                display_image("calibration_IMG_7374.jpg", "IMG_7374: my printed checkerboard, photographed for calibration. "
                                                         "The checkerboard was detected in this photograph.")
            with spec_column:
                st.table(pd.DataFrame({"Specification": ["Physical squares", "Inner corners", "Square size"],
                                       "Value": [f"{columns_n + 1} × {rows_n + 1}", f"{columns_n} × {rows_n}", f"{square_mm:g} mm"]}
                                      ).set_index("Specification"))
                st.caption("Inner corners are the points where four squares meet, so a board with "
                           f"{columns_n + 1} × {rows_n + 1} squares has {columns_n} × {rows_n} inner corners.")

            st.subheader("Calibration Photographs", anchor=False)
            for column, card in zip(st.columns(2), (("Captured", f"{CALIBRATION_PHOTOS_CAPTURED}", "calibration photographs"),
                                                    ("Accepted", f"{accepted}", "checkerboard corners detected"))):
                with column:
                    metric_card(*card)
            st.write(f"{CALIBRATION_PHOTOS_CAPTURED} calibration photographs were captured and {accepted} were accepted: the "
                     f"checkerboard corners were detected in {accepted} photographs. OpenCV did not detect the checkerboard in "
                     "IMG_7390 or IMG_7391, so those two were not used.")
            image_row([("calibration_IMG_7388.jpg", "IMG_7388: checkerboard detected (accepted photograph)."),
                       ("calibration_rejected_IMG_7390.jpg", "IMG_7390: OpenCV did not detect the checkerboard in this photograph "
                                                              "(not accepted).")])

        # ---- 2. calibration and results
        with st.container(key="m2_res_sec_2"):
            st.header("2. Camera Calibration & Results", anchor=False)
            st.write("OpenCV detected the checkerboard corners in the accepted photographs, refined them to sub-pixel accuracy, and "
                     "estimated the camera intrinsic matrix and the lens distortion coefficients from them. The values below are the "
                     "saved calibration used for all 20 validation measurements.")
            st.subheader("Reprojection Error", anchor=False)
            for column, card in zip(st.columns(2), (
                    ("RMS reprojection error", f"{float(rms_error):.4f} px", "saved in the calibration file"),
                    ("Mean reprojection error", f"{MEAN_REPROJECTION_ERROR_PX:.3f} px", "from my original calibration report"))):
                with column:
                    metric_card(*card)
            st.subheader("Camera Intrinsic Matrix", anchor=False)
            K = experiment_camera_matrix
            st.latex(r"K = \small\begin{bmatrix} %.2f & 0 & %.2f \\ 0 & %.2f & %.2f \\ 0 & 0 & 1 \end{bmatrix}"
                     % (K[0, 0], K[0, 2], K[1, 1], K[1, 2]))
            st.table(pd.DataFrame({"Parameter": ["fx", "fy", "cx", "cy"],
                                   "Meaning": ["Focal length along x", "Focal length along y", "Principal point, x", "Principal point, y"],
                                   "Value (pixels)": [f"{K[0, 0]:.2f}", f"{K[1, 1]:.2f}", f"{K[0, 2]:.2f}", f"{K[1, 2]:.2f}"]}
                                  ).set_index("Parameter"))
            st.write("**Original image resolution:** 4284 × 5712 pixels (portrait, EXIF orientation applied)")
            st.subheader("Distortion Coefficients", anchor=False)
            st.table(pd.DataFrame({"Coefficient": ["k1", "k2", "p1", "p2", "k3"],
                                   "Type": ["Radial", "Radial", "Tangential", "Tangential", "Radial"],
                                   "Value": [f"{value:.5f}" for value in experiment_distortion_coefficients.flatten()]}
                                  ).set_index("Coefficient"))
            st.caption("These parameters are from my original smartphone calibration and were used for all 20 validation measurements. "
                       "Visitor measurements use their own activated calibration.")

        # ---- 3. methodology
        with st.container(key="m2_res_sec_3"):
            st.header("3. Object Measurement Methodology", anchor=False)
            st.markdown("#### 1. Image acquisition at a known distance")
            st.markdown("Each object was photographed at a measured camera-to-object distance $Z$, measured from the camera to the "
                        "object's plane and entered for the calculation.")
            st.markdown("#### 2. Lens distortion correction")
            st.markdown("Each photograph is undistorted with the saved camera matrix and distortion coefficients.")
            st.markdown("#### 3. Four-corner selection")
            st.markdown("On the corrected photograph, I selected the object's four corners by hand, clicking them in the order top-left, "
                        "top-right, bottom-right, bottom-left.")
            st.markdown("#### 4. Perspective projection and dimension estimation")
            st.markdown("For two image points the real-world length between them is")
            st.latex(r"L = Z\,\sqrt{\left(\dfrac{u_2-u_1}{f_x}\right)^2 + \left(\dfrac{v_2-v_1}{f_y}\right)^2}")
            st.markdown("- **Width** is the mean of the top-edge and bottom-edge lengths.\n"
                        "- **Height** is the mean of the left-edge and right-edge lengths.\n"
                        "- **Averaging opposite edges** reduces the effect of small errors in individual corner positions.")
            st.markdown("#### 5. Comparison with the measured dimensions")
            st.markdown("**Errors** are the absolute difference from the measured dimension, and that difference as a percentage of it.")
            st.caption("The method treats the object as a flat face perpendicular to the optical axis at the entered distance. All lengths "
                       "are in centimetres.")

        # ---- 4. validation
        with st.container(key="m2_res_sec_4"):
            st.header("4. Validation Experiment", anchor=False)
            for column, card in zip(st.columns(3), (("Objects", f"{len(objects)}", None), ("Distances", f"{len(distances)}", None),
                                                    ("Trials", f"{trial_count}", None))):
                with column:
                    metric_card(*card)
            st.write(f"**Camera-to-object distances:** {', '.join(f'{d:g} cm' for d in distances)}")
            st.write(f"{len(objects)} objects were each measured at {len(distances)} distances, all greater than 2 m: "
                     f"{trial_count} measurements in total.")

            st.subheader("Validation Photographs", anchor=False)
            first_trials = {}
            for row in rows:
                first_trials.setdefault((float(row["Actual_Width_cm"]), float(row["Actual_Height_cm"])), row)
            shown = []
            for number, row in enumerate(first_trials.values(), start=1):
                name = Path(row["Image"]).stem
                if name in ("IMG5", "IMG9"):
                    shown.append((f"measurement_{name}.jpg", f"Object {number} ({row['Actual_Width_cm']} × {row['Actual_Height_cm']} cm), "
                                                             f"{name}, {float(row['Distance_cm']):g} cm"))
            image_row(shown)

            st.subheader("Complete Validation Results", anchor=False)
            kv_table("Trial", [[
                r["Trial"], f"Object {objects[(float(r['Actual_Width_cm']), float(r['Actual_Height_cm']))]}",
                f"{float(r['Distance_cm']):g}",
                f"{float(r['Actual_Width_cm']):.2f}", f"{float(r['Estimated_Width_cm']):.2f}", f"{float(r['Width_Error_Percent']):.2f}",
                f"{float(r['Actual_Height_cm']):.2f}", f"{float(r['Estimated_Height_cm']):.2f}", f"{float(r['Height_Error_Percent']):.2f}",
            ] for r in rows], ["Object", "Distance (cm)", "Actual width (cm)", "Estimated width (cm)", "Width error (%)",
                               "Actual height (cm)", "Estimated height (cm)", "Height error (%)"])
            trial_chart(rows)

        # ---- 5. error analysis
        with st.container(key="m2_res_sec_5"):
            st.header("5. Error Analysis & Statistics", anchor=False)
            st.table({"Statistic": ["Mean Absolute Error", "Mean Percentage Error", "Standard Deviation of Percentage Error", "RMSE"],
                      **{dimension: [f"{stats[dimension]['mae']:.2f} cm", f"{stats[dimension]['mpe']:.2f} %",
                                     f"{stats[dimension]['sd']:.2f} %", f"{stats[dimension]['rmse']:.2f} cm"]
                         for dimension in ("Width", "Height")}})
            st.subheader("Mean Percentage Error", anchor=False)
            st.bar_chart(pd.DataFrame({"Dimension": ["Width", "Height"],
                                       "Mean percentage error (%)": [stats["Width"]["mpe"], stats["Height"]["mpe"]]}).set_index("Dimension"),
                         y_label="Mean percentage error (%)")

        # ---- 6. discussion
        with st.container(key="m2_res_sec_6"):
            st.header("6. Discussion & Limitations", anchor=False)
            st.markdown("#### Interpretation of the results")
            st.write(
                f"Across the {trial_count} measurements the mean percentage error is {stats['Width']['mpe']:.2f} % for width and "
                f"{stats['Height']['mpe']:.2f} % for height. "
                f"The estimate was smaller than the measured size in {stats['Width']['under']} of {trial_count} width measurements and "
                f"{stats['Height']['under']} of {trial_count} height measurements, so the error is mostly in one direction rather than "
                "scattered both ways. I did not isolate the cause; the factors below are plausible contributors, not proven ones."
            )
            st.markdown("#### Measurement factors")
            st.markdown(
                "- **Manual corner selection:** corners are clicked by hand, so a few pixels of placement error changes each edge. At "
                "these distances one pixel corresponds to roughly 0.5 to 0.6 mm on the object.\n"
                "- **Camera-to-object distance:** the length scales linearly with the entered distance Z, so a distance error of a "
                "few centimetres gives a proportional size error. The distance is a hand measurement, so it carries "
                "its own uncertainty."
            )
            st.markdown("#### Camera and calibration factors")
            st.markdown(
                "- **Residual lens distortion:** the calibration RMS reprojection error is about 3 px, so some distortion may remain "
                "after correction, especially away from the image centre."
            )
            st.markdown("#### Geometric assumptions")
            st.markdown(
                "- **Camera and object alignment:** the method assumes the camera axis is perpendicular to the object face. Any "
                "tilt or rotation changes the apparent size.\n"
                "- **Fronto-parallel plane assumption:** the object is treated as a flat face at a single depth. Depth variation "
                "across the object, or any thickness of the object, is not modelled.\n"
                "- **Limits of monocular metric measurement:** one image fixes direction but not scale. Metric size comes only from the "
                "known distance and the assumed plane orientation, so the result is valid only when those assumptions hold, for this "
                "camera, lens and resolution."
            )

        # ---- 7. conclusion
        with st.container(key="m2_res_conclusion"):
            st.header("7. Conclusion", anchor=False)
            st.markdown(
                f"I calibrated my smartphone camera with OpenCV from {accepted} accepted checkerboard photographs and used the "
                f"perspective-projection equations to estimate the width and height of {len(objects)} objects at {distance_text} cm. "
                f"Across the {trial_count} measurements the mean percentage error was {stats['Width']['mpe']:.2f} % for width and "
                f"{stats['Height']['mpe']:.2f} % for height."
            )
            st.markdown(
                "These results are valid only under the stated assumptions: the same camera, lens and resolution as the calibration, "
                "and a flat object face perpendicular to the optical axis at the measured distance."
            )


# ---------------------------------------------------------------- page

# Configures the webpage
st.set_page_config(
    page_title="2D Object Measurement",
    page_icon="📐",
    layout="centered"
)

init_state()
page = st.session_state["m2_page"]
# breadcrumb levels below Home: (label, page); the last is the current page
CRUMBS = {"home": (("Module 2", "home"),),
          "visitor": (("Module 2", "home"), ("Calibrate Your Camera & Measure an Object", "visitor")),
          "experiment": (("Module 2", "home"), ("My Experiment & Results", "experiment"))}
nav_row(CRUMBS[page], go_to, PAGE_CSS, view=f"m2:{page}")
if page == "visitor":
    visitor_page()
elif page == "experiment":
    show_experiment()
else:
    home_page()
