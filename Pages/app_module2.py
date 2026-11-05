"""Module 2 - Monocular camera calibration and 2D object measurement.

Layout follows Module 7: a clean home page with two navigation buttons, a visitor workflow (calibrate your camera, then
measure an object) and a separate, read-only "My Experiment & Results" view. The page state is st.session_state["m2_page"]
("home" / "visitor" / "experiment"). Session state uses the m2_ prefix.
"""
import csv
import hashlib
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

import calibration_file as cf  # noqa: E402  (visitor calibration files; never touches camera_calibration.npz)
import visitor_calibration as vc  # noqa: E402  (visitor monocular calibration; independent of my saved calibration)

# Loads my saved experiment calibration. It is read-only reference data for the
# experiment view only and is never used for visitor measurements.
calibration = np.load(MODULE_DIR / "camera_calibration.npz")

experiment_camera_matrix = calibration["camera_matrix"]
experiment_distortion_coefficients = calibration["distortion_coefficients"]
rms_error = calibration["rms_error"]

# Gets the experiment's calibrated focal lengths
experiment_fx = experiment_camera_matrix[0, 0]
experiment_fy = experiment_camera_matrix[1, 1]


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
CALIB_EXPANDER_LABEL = "**My Camera Calibration Reference**  \n:small[Click to View My Smartphone Calibration Parameters]"
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
VISITOR_INPUT_KEYS = ("m2_object_name", "m2_actual_width", "m2_actual_height", "m2_distance")


def _svg_mask(path):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='{path}'/></svg>"
    url = "url(\"data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23") + "\")"
    return f"-webkit-mask-image: {url}; mask-image: {url};"


# Same Material Symbols icons and styling as the Module 7 page, scoped by widget keys (st-key-<key>).
CAMERA_MASK = _svg_mask("M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4zM9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6"
                        "c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z")
CHEVRON_MASK = _svg_mask("M16.59 8.59 12 13.17 7.41 8.59 6 10l6 6 6-6z")

MODULE_TITLE = "Module 2 — Camera Calibration & Object Measurement"
HOME_TITLE = "Module 2"
HOME_SUBTITLE = "Camera Calibration & 2D Object Measurement"
VISITOR_TITLE = "Calibrate Your Camera & Measure an Object"

HEADING_CSS = """
.st-key-m2_headings { gap: 0.25rem; }
/* Home page: mirrors Module 3's homepage. Title, subtitle and nav are separate containers; the layout's own gap is
   zeroed so the explicit padding-top below is the only space between them (44px title -> subtitle, 64px subtitle -> buttons). */
.st-key-m2_home_layout { gap: 0; padding-top: 4vh; }
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
@media (min-width: 641px) { .st-key-m2_home_subtitle h2 { white-space: nowrap; } }
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

EXPANDER_CSS = f"""
.st-key-m2_calib_ref details {{
    border: 2px solid {ACCENT} !important; border-radius: 14px; overflow: hidden;
}}
.st-key-m2_calib_ref details > summary {{
    position: relative; padding: 1.15rem 3.6rem; cursor: pointer; width: 100%; box-sizing: border-box;
}}
.st-key-m2_calib_ref details > summary:hover {{ background: color-mix(in srgb, {ACCENT} 12%, transparent); }}
.st-key-m2_calib_ref details > summary:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: -5px; }}
.st-key-m2_calib_ref details > summary [data-testid="stMarkdownContainer"] p {{ margin: 0; line-height: 1.55; }}
.st-key-m2_calib_ref details > summary strong {{ font-size: 1.35rem; font-weight: 700; }}
/* Streamlit's single icon slot swaps the left icon for a chevron when open; hide it and draw both icons here. */
.st-key-m2_calib_ref details > summary [data-testid="stExpanderIcon"],
.st-key-m2_calib_ref details > summary [data-testid="stExpanderToggleIcon"],
.st-key-m2_calib_ref details > summary [data-testid="stIconMaterial"],
.st-key-m2_calib_ref details > summary > span > span:first-child {{ display: none; }}
.st-key-m2_calib_ref details > summary::before, .st-key-m2_calib_ref details > summary::after {{
    content: ""; position: absolute; top: 50%; width: 1.9rem; height: 1.9rem; margin-top: -0.95rem;
    background-color: {ACCENT}; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-position: center; mask-position: center; -webkit-mask-size: contain; mask-size: contain;
}}
.st-key-m2_calib_ref details > summary::before {{
    left: 1.2rem; {CAMERA_MASK}
}}
.st-key-m2_calib_ref details > summary::after {{
    right: 1.2rem; transition: transform 0.2s ease; {CHEVRON_MASK}
}}
.st-key-m2_calib_ref details[open] > summary::after {{ transform: rotate(180deg); }}
.st-key-m2_calib_ref [data-testid="stExpanderDetails"] {{ padding: 0.5rem 1.35rem 1.25rem; }}
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
# Load Saved Calibration extras: the details expander header and the Clear Calibration button (scoped by container/button key).
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
.st-key-m2_clear_loaded button {{
    background-color: transparent; border: 2px solid {ACCENT}; border-radius: 10px; color: {ACCENT};
}}
.st-key-m2_clear_loaded button p {{ color: {ACCENT}; font-weight: 700; }}
.st-key-m2_clear_loaded button:hover {{ background-color: color-mix(in srgb, {ACCENT} 14%, transparent); border-color: {ACCENT}; color: {ACCENT}; }}
.st-key-m2_clear_loaded button:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
@media (max-width: 640px) {{ .st-key-m2_clear_loaded, .st-key-m2_clear_loaded button {{ width: 100%; }} }}
"""
# Home buttons: the nav container becomes a row of two equal ~110px rectangles that stack on narrow screens.
HOME_CARD_CSS = """
.st-key-m2_home_nav { flex-direction: row; flex-wrap: wrap; gap: 24px; }
.st-key-m2_open_visitor, .st-key-m2_open_experiment { flex: 1 1 280px; width: auto; min-width: 0; max-width: 100%; }
.st-key-m2_open_visitor button, .st-key-m2_open_experiment button {
    height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; text-align: center;
}
.st-key-m2_open_visitor button p, .st-key-m2_open_experiment button p {
    font-size: 21px; font-weight: 700; line-height: 1.3; text-align: center; text-wrap: balance;
}
.st-key-m2_open_visitor button::after, .st-key-m2_open_experiment button::after { content: none; display: none; }
@media (max-width: 640px) {
    .st-key-m2_home_nav { flex-direction: column; flex-wrap: nowrap; gap: 16px; }
    .st-key-m2_open_visitor, .st-key-m2_open_experiment { flex: 0 0 auto; width: 100%; }
}
"""
PAGE_CSS = "<style>" + HEADING_CSS + MODE_CSS + EXPANDER_CSS + BUTTON_CSS + HOME_CARD_CSS + LOADED_CSS + "</style>"


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
    ss.setdefault("m2_scroll_to", None)             # one-shot: page whose heading to scroll to after a view switch
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
        if str(key).startswith("m2_") and key not in ("m2_page", "m2_scroll_to", "m2_reset_count"):
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
    ss["m2_scroll_to"] = None if page == "home" else page   # one-shot: only when opening a view


# Runs once, on the render right after a view opens, and brings its heading into view (same approach as Module 7).
SCROLL_TO_HEADING_JS = """
<script>
setTimeout(function () {
  var heading = document.getElementById("HEADING_ID");
  if (!heading) { return; }
  heading.style.scrollMarginTop = "24px";
  heading.scrollIntoView({behavior: "smooth", block: "start"});
}, 150);
</script>
"""


def scroll_to_heading(page):
    if st.session_state.pop("m2_scroll_to", None) == page:
        st.html(SCROLL_TO_HEADING_JS.replace("HEADING_ID", f"m2-{page}-heading"), unsafe_allow_javascript=True)


def matrix_table(matrix, digits):
    return st.table([[f"{value:.{digits}f}" for value in row] for row in matrix])


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


def apply_legacy_size():
    ss = st.session_state
    gen = ss["m2_npz_gen"]
    width, height = ss.get(fk(f"legacy_w_{gen}")), ss.get(fk(f"legacy_h_{gen}"))
    try:
        if not ss.get(fk(f"legacy_ok_{gen}")):
            raise cf.CalibrationFileError("Tick the confirmation box first.")
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


def legacy_resolution(pending):
    """Older files have no image size: detect it from an original calibration photograph (primary) or enter it manually (fallback).
    The whole section is shown only while the loaded file still lacks a resolution."""
    ss = st.session_state
    gen = ss["m2_npz_gen"]
    st.warning("Missing image resolution. Choose one option below.")
    option_a, option_b = st.columns(2)
    with option_a.container(border=True):
        st.markdown("**Upload Calibration Image**")
        st.caption("Detect resolution automatically")
        st.file_uploader("Upload Calibration Image", type=["jpg", "jpeg", "png"], key=f"m2_cal_photo_{ss['m2_reset_count']}_{gen}",
                         on_change=on_legacy_photo, label_visibility="collapsed")
        if ss["m2_legacy_photo_error"]:
            st.error(ss["m2_legacy_photo_error"])
        photo = ss["m2_legacy_photo"]
        if photo is not None:
            st.markdown(f"**{photo['name']}**  \nDetected: **{photo['width']} × {photo['height']} px**")
            st.checkbox("I confirm this photograph was used for the original calibration and has not been resized or cropped since "
                        "that calibration.", key=fk(f"photo_ok_{gen}"))
            st.button("Use Detected Resolution", key="m2_photo_apply", on_click=apply_detected_size)
    with option_b.container(border=True):
        st.markdown("**Enter Resolution Manually**")
        st.caption("Specify width and height")
        with st.expander("Show manual entry"):
            st.write("Enter the exact pixel size of the photographs the calibration was computed from, after EXIF rotation.")
            c = st.columns(2)
            c[0].number_input("Calibration image width (px)", min_value=cf.IMAGE_SIDE_LIMITS[0], max_value=cf.IMAGE_SIDE_LIMITS[1],
                              value=None, step=1, placeholder="width", key=fk(f"legacy_w_{gen}"))
            c[1].number_input("Calibration image height (px)", min_value=cf.IMAGE_SIDE_LIMITS[0], max_value=cf.IMAGE_SIDE_LIMITS[1],
                              value=None, step=1, placeholder="height", key=fk(f"legacy_h_{gen}"))
            st.checkbox("I confirm these are the exact dimensions of the photographs used for this calibration, and that I will "
                        "measure with the same camera, lens and settings.", key=fk(f"legacy_ok_{gen}"))
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
    """Compact summary that replaces the uploader once a file is loaded; the name comes from session state, so it survives reruns."""
    st.markdown(f":material/description: **{st.session_state['m2_load_name']}**")
    st.caption(":material/check_circle: File uploaded and validated")


def photo_summary(pending):
    """Compact record of the confirmed calibration photograph (evidence of resolution only), with Remove Image."""
    photo = st.session_state["m2_legacy_photo"]
    if photo is None or pending.get("size_source") != "detected_from_photo":
        return
    with st.container(border=True):
        st.markdown("**Upload Calibration Image**")
        st.markdown(f":material/image: **{photo['name']}**  \nResolution: {photo['width']} × {photo['height']} px")
        st.button("Remove Image", key="m2_photo_remove", on_click=remove_legacy_photo)


def clear_calibration_button():
    st.button("Clear Calibration", key="m2_clear_loaded", on_click=clear_loaded_calibration,
              help="Remove this file and reset the Load Saved Calibration workflow.")


def load_workflow():
    ss = st.session_state
    pending = ss["m2_load_pending"]
    if pending is None:
        st.file_uploader("Upload Calibration File (.npz)", type=["npz"], key=f"m2_cal_npz_{ss['m2_reset_count']}_{ss['m2_npz_gen']}",
                         on_change=on_calibration_file)
        st.caption("The file is validated before it can be activated.")
    else:
        loaded_file_card()
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
    clear_calibration_button()                  # always the last control of Step 1


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


def measurement_section():
    ss = st.session_state
    visitor_calibration = ss["m2_visitor_calibration"]
    st.subheader("Step 2: Measure an Object")
    geometry_assumptions()
    if visitor_calibration is not None:
        size = (f"{visitor_calibration['image_width']} × {visitor_calibration['image_height']} px"
                if visitor_calibration.get("image_width") else "unknown resolution")
        st.caption(f"Using active calibration: {size}")
    if visitor_calibration is None:
        st.info("First load a saved calibration or calibrate your camera in Step 1 above. After a calibration is active, "
                "upload a photograph of the object, enter the camera-to-object distance and select its four corners. "
                "The measurement controls stay disabled until then.")
        return
    visitor_measurement(visitor_calibration)


def visitor_measurement(visitor_calibration):
    """The existing measurement workflow, shown only while a visitor calibration is active."""
    visitor_camera_matrix = visitor_calibration["camera_matrix"]
    visitor_distortion_coefficients = visitor_calibration["distortion_coefficients"]
    visitor_fx = visitor_camera_matrix[0, 0]
    visitor_fy = visitor_camera_matrix[1, 1]

    st.subheader("Object Information")

    # Gets the object information
    object_name = st.text_input("Object Name", key="m2_object_name", on_change=clear_live_result)

    actual_width = st.number_input(
        "Actual Width (cm)",
        value=None,
        placeholder="0.00",
        min_value=0.0,
        step=0.1,
        key="m2_actual_width",
        on_change=clear_live_result
    )

    actual_height = st.number_input(
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
        on_change=reset_points
    )

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

        # Displays the resized clickable image
        coordinates = streamlit_image_coordinates(
            display_image,
            key=f"m2_measurement_image_{st.session_state.m2_click_generation}"
        )

        # Converts display coordinates to original coordinates
        if coordinates is not None and len(points) < 4:
            original_x = coordinates["x"] / display_scale
            original_y = coordinates["y"] / display_scale
            points.append((original_x, original_y))

            # Prevents the consumed click from being reused on reruns
            st.session_state.m2_click_generation += 1
            st.rerun()

        if st.button("Calculate Measurement", disabled=len(points) != 4):
            if any(value is None or value <= 0 for value in (distance, actual_width, actual_height)):
                st.error("Enter a positive distance, actual width, and actual height.")
            else:
                top_left, top_right, bottom_right, bottom_left = points

                # Calculates the estimated width
                top_width = calculate_length(top_left, top_right, distance, visitor_fx, visitor_fy)
                bottom_width = calculate_length(bottom_left, bottom_right, distance, visitor_fx, visitor_fy)
                estimated_width = (top_width + bottom_width) / 2

                # Calculates the estimated height
                left_height = calculate_length(top_left, bottom_left, distance, visitor_fx, visitor_fy)
                right_height = calculate_length(top_right, bottom_right, distance, visitor_fx, visitor_fy)
                estimated_height = (left_height + right_height) / 2

                # Calculates the absolute and percentage errors
                width_error = abs(estimated_width - actual_width)
                height_error = abs(estimated_height - actual_height)
                width_error_percent = width_error / actual_width * 100
                height_error_percent = height_error / actual_height * 100

                # Stores a temporary snapshot of the completed measurement
                clear_live_result()
                st.session_state.m2_live_result = {
                    "Image": uploaded_image.name,
                    "Distance_cm": distance,
                    "Actual_Width_cm": actual_width,
                    "Estimated_Width_cm": estimated_width,
                    "Width_Error_cm": width_error,
                    "Width_Error_Percent": width_error_percent,
                    "Actual_Height_cm": actual_height,
                    "Estimated_Height_cm": estimated_height,
                    "Height_Error_cm": height_error,
                    "Height_Error_Percent": height_error_percent
                }

        if "m2_result_notice" in st.session_state:
            st.success(st.session_state.pop("m2_result_notice"))

        if "m2_live_result" in st.session_state:
            result = st.session_state.m2_live_result

            # Displays the measurement results
            st.subheader("Measurement Results")
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

            st.button("Discard Result", on_click=discard_result)


def page_headings(title, subtitle=None, anchor=None, variant=None):
    """One prominent page heading, with an optional secondary subtitle."""
    with st.container(key=f"m2_headings_{variant}" if variant else "m2_headings_default"):
        with st.container(key="m2_headings"):
            st.title(title, anchor=anchor or False)
            if subtitle:
                st.header(subtitle, anchor=False)


def visitor_page():
    st.button("← Back", key="m2_back_visitor_top", on_click=go_to, args=("home",))
    page_headings(VISITOR_TITLE, anchor="m2-visitor-heading", variant="visitor")
    scroll_to_heading("visitor")
    st.subheader("Step 1: Calibrate Your Camera")
    calibrate_workflow()

    measurement_section()
    st.button("Reset All", key="m2_reset_all", on_click=reset_all)


def home_page():
    with st.container(key="m2_home_layout"):
        with st.container(key="m2_home_title"):
            st.title(HOME_TITLE, anchor=False)
        with st.container(key="m2_home_subtitle"):
            st.header(HOME_SUBTITLE, anchor=False)
        with st.container(key="m2_home_nav"):
            home_buttons()


def home_buttons():
    st.button("Calibrate Your Camera & Measure an Object", key="m2_open_visitor", on_click=go_to, args=("visitor",), width="stretch")
    st.button("View My Experiment & Results", key="m2_open_experiment", on_click=go_to, args=("experiment",), width="stretch")


# ---------------------------------------------------------------- saved experiment (read-only)

def show_calibration_reference():
    with st.container(key="m2_calib_ref"), st.expander(CALIB_EXPANDER_LABEL):
        st.write(
            "These parameters come from my own smartphone calibration and produced the 20 validation measurements shown "
            "below. They are reference data only and are not used for visitor measurements."
        )
        st.subheader("Intrinsic Parameters")
        st.write("**Camera intrinsic matrix K (3×3, pixels)**")
        matrix_table(experiment_camera_matrix, 2)
        st.table({
            "Parameter": ["fx", "fy", "cx", "cy"],
            "Value (pixels)": [f"{experiment_fx:.2f}", f"{experiment_fy:.2f}",
                               f"{experiment_camera_matrix[0, 2]:.2f}", f"{experiment_camera_matrix[1, 2]:.2f}"],
        })
        st.write("**Distortion coefficients**")
        st.table({
            "Coefficient": ["k1", "k2", "p1", "p2", "k3"],
            "Value": [f"{value:.5f}" for value in experiment_distortion_coefficients.flatten()],
        })
        st.subheader("Calibration Error")
        st.table({
            "Error measure (pixels)": ["RMS reprojection error (saved in the calibration file)",
                                       "Mean reprojection error (from my original report)"],
            "Value": [f"{float(rms_error):.4f}", f"{MEAN_REPROJECTION_ERROR_PX:.3f}"],
        })
        st.caption("The RMS error weights large corner errors more heavily than the mean error, so it is always at least as large "
                   "as the mean. The two values describe the same calibration and must not be confused. The mean error is "
                   "not stored in the calibration file; it is quoted from my original calibration report.")


def image_row(items, per_row=3):
    """items: (file name in display/, caption). Missing files are reported instead of hidden."""
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        for column, (name, caption) in zip(st.columns(per_row), chunk):
            with column:
                path = DISPLAY_DIR / name
                if path.is_file():
                    st.image(str(path), caption=caption, width="stretch")
                else:
                    st.warning(f"Display image {name} is unavailable.")


def show_experiment():
    st.header("My Experiment & Results", anchor="m2-experiment-heading")
    scroll_to_heading("experiment")
    st.button("Back to Module 2", key="m2_back_top", on_click=go_to, args=("home",))
    st.caption("Read-only. These results come from my saved calibration and validation files; nothing on this page can change them.")

    try:
        rows = read_validation_rows()
    except (OSError, ValueError, csv.Error, TypeError) as error:
        st.error(f"Validation results could not be loaded: {error}")
        st.button("Back to Module 2", key="m2_back_bottom", on_click=go_to, args=("home",))
        return
    trial_count = len(rows)
    distances = sorted({float(row["Distance_cm"]) for row in rows})

    # A. Experiment overview
    st.subheader("A. Experiment Overview")
    st.write(
        "**Assignment objective:** (1) calibrate a smartphone camera with OpenCV, (2) implement perspective projection "
        "equations that estimate the real-world 2D dimensions of an object from an image, and (3) validate the method at a "
        "measured camera-to-object distance greater than 2 m using 20 different object measurements, reporting error statistics."
    )
    st.write(
        f"**Completed experiment:** I calibrated my smartphone camera from checkerboard photographs, then photographed five "
        f"objects at four distances ({', '.join(f'{d:g}' for d in distances)} cm), giving {trial_count} measurements. For each "
        "photograph I selected the object's four corners by hand, computed width and height with the perspective-projection "
        "equations, and compared them with the measured dimensions of the object."
    )

    # B. Calibration setup
    st.subheader("B. Camera Calibration Setup")
    st.write("**Camera:** iPhone 17 Pro Max, 1× main camera")
    st.caption("The photographs' EXIF data lists the iPhone 17 Pro Max main lens (6.765 mm, f/1.78) for both the calibration and "
               "the measurement photographs.")
    columns_n, rows_n = int(calibration["checkerboard_columns"]), int(calibration["checkerboard_rows"])
    st.write(f"**Checkerboard:** {columns_n + 1} × {rows_n + 1} physical squares, {columns_n} × {rows_n} inner corners, "
             f"{float(calibration['square_size_mm']):g} mm squares")
    st.write(f"**Calibration photographs:** {CALIBRATION_PHOTOS_CAPTURED} captured, {int(calibration['successful_images'])} accepted "
             "(the checkerboard corners were detected in 18 photographs)")
    st.write("**Photograph resolution:** 4284 × 5712 pixels (portrait, EXIF orientation applied)")
    st.caption("The measurement photographs use the same resolution, so the saved focal lengths apply to them unchanged.")

    # C. Calibration results
    st.subheader("C. Calibration Results")
    show_calibration_reference()

    # D. Methodology
    st.subheader("D. Measurement Methodology")
    st.markdown(
        "1. **Lens distortion correction:** each photograph is undistorted with the saved camera matrix and distortion "
        "coefficients.\n"
        "2. **Four-corner selection:** the object's corners are clicked in the order top-left, top-right, bottom-right, "
        "bottom-left.\n"
        "3. **Known camera-to-object distance:** the distance Z from the camera to the object's plane was measured and entered.\n"
        "4. **Perspective projection:** for two image points the real-world length between them is"
    )
    st.latex(r"L = Z\,\sqrt{\left(\dfrac{u_2-u_1}{f_x}\right)^2 + \left(\dfrac{v_2-v_1}{f_y}\right)^2}")
    st.markdown(
        "5. **Width** is the mean of the top-edge and bottom-edge lengths.\n"
        "6. **Height** is the mean of the left-edge and right-edge lengths.\n"
        "7. **Averaging opposite edges** reduces the effect of small errors in individual corner positions.\n"
        "8. **Errors** are the absolute difference from the measured dimension, and that difference as a percentage of it."
    )
    st.caption("The method treats the object as a flat face perpendicular to the optical axis at the entered distance. All lengths "
               "are in centimetres.")

    # E. Validation experiment
    st.subheader("E. Validation Experiment")
    objects = {}
    for row in rows:
        objects.setdefault((float(row["Actual_Width_cm"]), float(row["Actual_Height_cm"])), len(objects) + 1)
    st.write(f"{len(objects)} objects were each measured at {len(distances)} distances, all greater than 2 m: "
             f"{trial_count} measurements in total.")
    kv_table("Trial", [[
        r["Trial"], f"Object {objects[(float(r['Actual_Width_cm']), float(r['Actual_Height_cm']))]}",
        f"{float(r['Distance_cm']):g}",
        f"{float(r['Actual_Width_cm']):.2f}", f"{float(r['Estimated_Width_cm']):.2f}", f"{float(r['Width_Error_Percent']):.2f}",
        f"{float(r['Actual_Height_cm']):.2f}", f"{float(r['Estimated_Height_cm']):.2f}", f"{float(r['Height_Error_Percent']):.2f}",
    ] for r in rows], ["Object", "Distance (cm)", "Actual width (cm)", "Estimated width (cm)", "Width error (%)",
                       "Actual height (cm)", "Estimated height (cm)", "Height error (%)"])
    st.caption("Values are read from measurement_results.csv, which this page cannot modify. Objects are numbered in the order "
               "they were measured.")
    st.bar_chart(pd.DataFrame({
        "Trial": [int(r["Trial"]) for r in rows],
        "Width error (%)": [float(r["Width_Error_Percent"]) for r in rows],
        "Height error (%)": [float(r["Height_Error_Percent"]) for r in rows],
    }).set_index("Trial"), x_label="Trial", y_label="Percentage error (%)")

    # F. Error statistics
    st.subheader("F. Error Statistics")
    statistics = {"Statistic": ["Mean Absolute Error", "Mean Percentage Error", "Standard Deviation of Percentage Error", "RMSE"]}
    underestimated = {}
    for dimension in ("Width", "Height"):
        errors = np.array([float(row[f"{dimension}_Error_cm"]) for row in rows])
        percent_errors = np.array([float(row[f"{dimension}_Error_Percent"]) for row in rows])
        statistics[dimension] = [f"{np.mean(errors):.2f} cm", f"{np.mean(percent_errors):.2f} %",
                                 f"{np.std(percent_errors):.2f} %", f"{np.sqrt(np.mean(errors ** 2)):.2f} cm"]
        underestimated[dimension] = sum(float(r[f"Estimated_{dimension}_cm"]) < float(r[f"Actual_{dimension}_cm"]) for r in rows)
    st.table(statistics)

    # G. Visual evidence
    st.subheader("G. Visual Evidence")
    st.write("**Calibration photographs** (accepted)")
    image_row([
        ("calibration_IMG_7374.jpg", "IMG_7374: checkerboard detected."),
        ("calibration_IMG_7380.jpg", "IMG_7380: checkerboard detected."),
        ("calibration_IMG_7388.jpg", "IMG_7388: checkerboard detected."),
        ("calibration_IMG_7394.jpg", "IMG_7394: checkerboard detected."),
    ], per_row=2)
    st.write("**Calibration photographs** (not accepted)")
    image_row([
        ("calibration_rejected_IMG_7390.jpg", "IMG_7390: OpenCV did not detect the checkerboard in this photograph."),
        ("calibration_rejected_IMG_7391.jpg", "IMG_7391: OpenCV did not detect the checkerboard in this photograph."),
    ], per_row=2)
    st.write("**Object photographs** (one per object, nearest distance)")
    first_trials = {}
    for row in rows:
        first_trials.setdefault((float(row["Actual_Width_cm"]), float(row["Actual_Height_cm"])), row)
    shown = []
    for number, row in enumerate(first_trials.values(), start=1):
        name = Path(row["Image"]).stem
        shown.append((f"measurement_{name}.jpg",
                      f"Object {number} ({row['Actual_Width_cm']} × {row['Actual_Height_cm']} cm), {name}, {float(row['Distance_cm']):g} cm"))
    image_row(shown, per_row=3)
    st.caption("Optimized display copies of the original photographs (EXIF orientation applied). They are shown without annotations "
               "because the clicked corner coordinates were not saved with the results.")

    # H. Discussion and limitations
    st.subheader("H. Discussion and Limitations")
    st.write(
        f"Across the {trial_count} measurements the mean percentage error is "
        f"{np.mean([float(r['Width_Error_Percent']) for r in rows]):.2f} % for width and "
        f"{np.mean([float(r['Height_Error_Percent']) for r in rows]):.2f} % for height. "
        f"The estimate was smaller than the measured size in {underestimated['Width']} of {trial_count} width measurements and "
        f"{underestimated['Height']} of {trial_count} height measurements, so the error is mostly in one direction rather than "
        "scattered both ways. I did not isolate the cause; the factors below are plausible contributors, not proven ones."
    )
    st.markdown(
        "- **Manual corner selection:** corners are clicked by hand, so a few pixels of placement error changes each edge. At "
        "these distances one pixel corresponds to roughly 0.5 to 0.6 mm on the object.\n"
        "- **Camera-to-object distance:** the length scales linearly with the entered distance Z, so a distance error of a "
        "few centimetres gives a proportional size error. The distance is a hand measurement, so it carries "
        "its own uncertainty.\n"
        "- **Camera and object alignment:** the method assumes the camera axis is perpendicular to the object face. Any "
        "tilt or rotation changes the apparent size.\n"
        "- **Residual lens distortion:** the calibration RMS reprojection error is about 3 px, so some distortion may remain "
        "after correction, especially away from the image centre.\n"
        "- **Fronto-parallel plane assumption:** the object is treated as a flat face at a single depth. Depth variation "
        "across the object, or any thickness of the object, is not modelled.\n"
        "- **Limits of monocular metric measurement:** one image fixes direction but not scale. Metric size comes only from the "
        "known distance and the assumed plane orientation, so the result is valid only when those assumptions hold, for this "
        "camera, lens and resolution.\n"
        f"- **Scope of validity:** five objects and distances from {min(distances):g} to {max(distances):g} cm are too few to "
        "claim accuracy outside this range."
    )
    st.button("Back to Module 2", key="m2_back_bottom", on_click=go_to, args=("home",))


# ---------------------------------------------------------------- page

# Configures the webpage
st.set_page_config(
    page_title="2D Object Measurement",
    page_icon="📐",
    layout="centered"
)

init_state()
st.markdown(PAGE_CSS, unsafe_allow_html=True)
page = st.session_state["m2_page"]
if page == "visitor":
    visitor_page()
elif page == "experiment":
    page_headings(MODULE_TITLE)
    show_experiment()
else:
    home_page()
