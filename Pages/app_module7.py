"""Module 7 - Stereo Vision: calibrated stereo object measurement (Part 1) and room measurement from stereo videos (Part 3: saved experiment results, and a guide to running the visitor application locally)."""

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

MODULE_DIR = Path(__file__).resolve().parents[1] / "Modules" / "Module_7"
Q3_DIR = MODULE_DIR / "room_measurement_question_3"
for _p in (str(MODULE_DIR), str(Q3_DIR)):    # MODULE_DIR: the shared stereo_measure package; Q3_DIR: room_measurement (results page reader)
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import local_mode  # noqa: E402  (set by run_local.py only; never true on the public deployment)
from home_navigation import nav_row  # noqa: E402
from select_cards import card_link, panel_label, select_css  # noqa: E402  (card design shared with the standalone room measurement app)
from stereo_measure import calibration_wizard as wiz  # noqa: E402
from stereo_measure import load_model_b, load_selection  # noqa: E402
from stereo_measure import selection as sel  # noqa: E402
from stereo_measure import corner_drag as cd  # noqa: E402
from stereo_measure.calibration_io import CalibrationFileError, export_calibration, load_calibration  # noqa: E402
from stereo_measure.config import (CORNER_NAMES, IMAGE_SIZE, NOTEBOOK_PAIR_IDS, OUTPUT_DIR,  # noqa: E402
                                   PATTERN, SQUARE_MM)
from stereo_measure.live import measure_points  # noqa: E402
from room_measurement import display_data as room_data  # noqa: E402  (JSON/image reader only: no estimator, no ground truth)

Q1_DIR = MODULE_DIR / "object_measurement_question_1"
DISPLAY_DIR = Q1_DIR / "display"
CALIB_SAMPLE_IMAGE = DISPLAY_DIR / "calibration_sample_left_pair_8.jpg"   # optimized crop of left calibration photo 8
CALIB_SAMPLE_PAIR = 8
ACCENT = "#C9363E"
CALIB_EXPANDER_LABEL = "**Click to View My Camera Calibration Parameters**"
CALIBRATE_EXPANDER_LABEL = ("**Calibrate Your Stereo Cameras**  \n:small[Upload matching checkerboard photos from your own "
                            "stereo camera setup to calculate the calibration parameters.]")
# st.session_state["m7_visitor_model"] is the active visitor calibration (computed here or loaded from a file); it never falls back
# to Model B. Object measurement is enabled exactly while a visitor calibration is active (m7_measurement_enabled mirrors that).

# Styles are scoped by widget keys (Streamlit adds the class st-key-<key>), so no other expander or button is affected.
def _svg_mask(path):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='{path}'/></svg>"
    url = "url(\"data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23") + "\")"
    return f"-webkit-mask-image: {url}; mask-image: {url};"


# Material Symbols "photo_camera" and "expand_more" outlines (Apache 2.0)
CAMERA_MASK = _svg_mask("M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4zM9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6"
                        "c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z")
CHEVRON_MASK = _svg_mask("M16.59 8.59 12 13.17 7.41 8.59 6 10l6 6 6-6z")

EXPANDER_CSS = f"""
.st-key-m7_calib_ref details {{
    border: 2px solid {ACCENT} !important; border-radius: 14px; overflow: hidden;
}}
.st-key-m7_calib_ref details > summary {{
    position: relative; padding: 1.15rem 3.6rem; cursor: pointer; width: 100%; box-sizing: border-box;
}}
.st-key-m7_calib_ref details > summary:hover {{ background: color-mix(in srgb, {ACCENT} 12%, transparent); }}
.st-key-m7_calib_ref details > summary:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: -5px; }}
.st-key-m7_calib_ref details > summary [data-testid="stMarkdownContainer"] p {{ margin: 0; line-height: 1.55; }}
.st-key-m7_calib_ref details > summary strong {{ font-size: 1.35rem; font-weight: 700; }}
/* Streamlit's single icon slot swaps the left icon for a chevron when open; hide it and draw both icons here. */
.st-key-m7_calib_ref details > summary [data-testid="stExpanderIcon"],
.st-key-m7_calib_ref details > summary [data-testid="stExpanderToggleIcon"],
.st-key-m7_calib_ref details > summary [data-testid="stIconMaterial"],
.st-key-m7_calib_ref details > summary > span > span:first-child {{ display: none; }}
.st-key-m7_calib_ref details > summary::before, .st-key-m7_calib_ref details > summary::after {{
    content: ""; position: absolute; top: 50%; width: 1.9rem; height: 1.9rem; margin-top: -0.95rem;
    background-color: {ACCENT}; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-position: center; mask-position: center; -webkit-mask-size: contain; mask-size: contain;
}}
.st-key-m7_calib_ref details > summary::before {{
    left: 1.2rem; {CAMERA_MASK}
}}
.st-key-m7_calib_ref details > summary::after {{
    right: 1.2rem; transition: transform 0.2s ease; {CHEVRON_MASK}
}}
.st-key-m7_calib_ref details[open] > summary::after {{ transform: rotate(180deg); }}
.st-key-m7_calib_ref [data-testid="stExpanderDetails"] {{ padding: 0.5rem 1.35rem 1.25rem; }}
"""
# The "Calibrate Your Stereo Cameras" expander shares the reference expander's look.
BUTTON_CSS = f"""
/* "Measure Another Image" under the results: solid blue in both themes (Reset All beside it keeps the default look). */
.st-key-m7_another_image button {{ background-color: #2563EB; border: 1px solid #2563EB; color: #FFFFFF; }}
.st-key-m7_another_image button p {{ color: #FFFFFF; font-weight: 700; }}
.st-key-m7_another_image button:hover {{ background-color: #3B82F6; border-color: #3B82F6; color: #FFFFFF; }}
.st-key-m7_another_image button:active {{ background-color: #1D4ED8; border-color: #1D4ED8; color: #FFFFFF; }}
.st-key-m7_another_image button:focus-visible {{ outline: 3px solid #93C5FD; outline-offset: 2px; }}
"""
NAV_CSS = f"""
.st-key-m7_cal_prev button, .st-key-m7_cal_next button {{ background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF; }}
.st-key-m7_cal_prev button p, .st-key-m7_cal_next button p {{ color: #FFFFFF; font-weight: 600; }}
.st-key-m7_cal_prev button:hover:not(:disabled), .st-key-m7_cal_next button:hover:not(:disabled) {{ background-color: #A62A31; border-color: #A62A31; }}
.st-key-m7_cal_prev button:focus-visible, .st-key-m7_cal_next button:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
.st-key-m7_cal_prev button:disabled, .st-key-m7_cal_next button:disabled {{ opacity: 0.45; }}
"""
MODE_CSS = f"""
.st-key-m7_cal_mode_run button, .st-key-m7_cal_mode_load button {{
    width: 100%; min-height: 4rem; border-radius: 14px; border: 2px solid {ACCENT}; font-size: 1.1rem; font-weight: 700;
    background-color: transparent; color: {ACCENT};
}}
.st-key-m7_cal_mode_run button p, .st-key-m7_cal_mode_load button p {{ font-size: 1.1rem; font-weight: 700; color: inherit; }}
.st-key-m7_cal_mode_run button[kind="primary"], .st-key-m7_cal_mode_load button[kind="primary"] {{ background-color: {ACCENT}; color: #FFFFFF; }}
.st-key-m7_cal_mode_run button:focus-visible, .st-key-m7_cal_mode_load button:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
"""
DOWNLOAD_BLUE = "#2563EB"
DOWNLOAD_CSS = f"""
.st-key-m7_cal_download button {{ border-radius: 12px; padding: 0.6rem 1.4rem; font-weight: 700; }}
.st-key-m7_cal_download button:not(:disabled) {{
    background-color: {DOWNLOAD_BLUE}; border: 2px solid {DOWNLOAD_BLUE}; color: #FFFFFF;
}}
.st-key-m7_cal_download button:not(:disabled), .st-key-m7_cal_download button:not(:disabled) p,
.st-key-m7_cal_download button:not(:disabled) span {{ color: #FFFFFF; font-weight: 700; }}
.st-key-m7_cal_download button:not(:disabled):hover {{ background-color: #1D4ED8; border-color: #1D4ED8; color: #FFFFFF; }}
.st-key-m7_cal_download button:not(:disabled):active {{ background-color: #1E40AF; border-color: #1E40AF; }}
.st-key-m7_cal_download button:not(:disabled):focus-visible {{ outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {DOWNLOAD_BLUE}; }}
/* disabled: Streamlit's own grey disabled look is kept (only the corners and weight above apply) */
"""
# Calibration reference only: subtler than the shared look above (thin border, light tint, modestly larger bold label).
CALIB_REF_CSS = f"""
.st-key-m7_calib_ref details {{
    border: 1px solid color-mix(in srgb, {ACCENT} 55%, transparent) !important;
    background: color-mix(in srgb, {ACCENT} 6%, transparent);
}}
.st-key-m7_calib_ref details > summary {{ padding-top: 0.8rem; padding-bottom: 0.8rem; }}
.st-key-m7_calib_ref details > summary strong {{ font-size: 1.15rem; }}
"""
PART_ONE_CSS = ("<style>" + MODE_CSS + NAV_CSS + EXPANDER_CSS + EXPANDER_CSS.replace("m7_calib_ref", "m7_calibrate") + CALIB_REF_CSS
                + BUTTON_CSS + DOWNLOAD_CSS + "</style>")
ZOOM_BUTTON_CSS = """<style>
.st-key-m7_zoom_accept button { background: #1b8a3a !important; border: 1px solid #14692c !important; color: #fff !important; border-radius: 10px !important; }
.st-key-m7_zoom_accept button p { color: #fff !important; font-weight: 700 !important; }
.st-key-m7_zoom_accept button:hover { background: #167532 !important; }
.st-key-m7_zoom_cancel button { background: #3a3f47 !important; border: 1px solid #2a2e34 !important; color: #fff !important; border-radius: 10px !important; }
.st-key-m7_zoom_cancel button p { color: #fff !important; }
.st-key-m7_zoom_cancel button:hover { background: #2a2e34 !important; }
</style>"""
CALIBRATION_SUMMARY_CSS = """<style>
.st-key-m7_calibration_summary [data-testid="stMetricValue"],
.st-key-m7_calibration_summary [data-testid="stMetricValue"] > div,
.st-key-m7_calibration_summary [data-testid="stMetricLabel"],
.st-key-m7_calibration_summary [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; }
.st-key-m7_calibration_summary [data-testid="stMetricValue"] { font-size: 1.6rem; line-height: 1.25; }
</style>"""
# Selected-pair results panel: Streamlit follows prefers-color-scheme (no custom theme is configured), so the media query matches its theme.
SELECTED_PAIR_CSS = """<style>
.st-key-m7_selected_pair_results { background: #F3F4F6; border: 1px solid #D9DDE3; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1rem; }
@media (prefers-color-scheme: dark) {
    .st-key-m7_selected_pair_results { background: #253240; border-color: #405466; }
}
/* heading hierarchy inside the panel: "Notebook Stereo Pair" (h3) is larger than the selected pair's photograph heading (h4); both bold */
.st-key-m7_selected_pair_results h3 { font-size: 1.6rem; font-weight: 700; line-height: 1.25; padding: 0 0 0.5rem; }
.st-key-m7_selected_pair_results h4 { font-size: 1.2rem; font-weight: 700; line-height: 1.3; padding: 1rem 0 0.5rem; overflow-wrap: anywhere; }
</style>"""
# Objective and Experimental setup: same look as the Question 3 page (bold left-aligned h5 headings, plain paragraphs)
Q1_INTRO_CSS = """<style>
.st-key-m7_q1_objective h5 { text-align: left; font-weight: 700; margin: 0; padding: 0.5rem 0 0.25rem; }
</style>"""
LEFT_CAMERA, RIGHT_CAMERA = "iPhone Air", "iPhone 17 Pro Max"
MEASURED_BASELINE_MM = 225
SIDES = ("left", "right")
SIDE_LABEL = {"left": "Left", "right": "Right"}
PREVIEW_WIDTH = 700
ZOOM_HALF = 80          # zoom crop is 160 x 160 original pixels, shown 4x larger


# ---------------------------------------------------------------- cached data

@st.cache_resource
def get_model():
    """Model B is read from the stored npz once; the app never recalibrates."""
    return load_model_b()


@st.cache_resource(max_entries=4, ttl=3600)
def prepare_image(digest, _data, size):
    """Decode an upload once at the calibration's frame size for its side. Cached objects are shared: treat them as read-only."""
    full = sel.decode_upload(_data, size)
    return {"full": full, "size": tuple(size), "preview": sel.make_preview(full, PREVIEW_WIDTH), "large": sel.make_preview(full, 1000)}


@st.cache_data
def load_saved_results():
    """Final saved notebook pipeline outputs (written by run_measurement.py)."""
    return {
        "summary": json.loads((OUTPUT_DIR / "measurement_summary.json").read_text()),
        "pairs": pd.read_csv(OUTPUT_DIR / "pair_summary.csv"),
        "corners": pd.read_csv(OUTPUT_DIR / "corner_coordinates_and_depths.csv"),
        "edges": pd.read_csv(OUTPUT_DIR / "edge_length_calculations.csv"),
    }


# ---------------------------------------------------------------- session state

MODULE7_RETURN_VIEWS = ("home", "run_own", "3_visitor")   # views the standalone room measurement app's Back / breadcrumb links return to (?view=...)


def requested_view():
    """The landing view named by a ?view= query parameter (one of MODULE7_RETURN_VIEWS), else "home".  The parameter is consumed so the URL stays clean."""
    view = st.query_params.get("view")
    if view is not None:
        del st.query_params["view"]
    return view if view in MODULE7_RETURN_VIEWS else "home"


def init_state():
    ss = st.session_state
    if "module7_part" not in ss:
        ss["module7_part"] = requested_view()
    ss.setdefault("m7_upload_gen", 0)
    ss.setdefault("m7_gen", 0)                  # bumped whenever a click is consumed, so each click event is used once
    ss.setdefault("m7_file_left", None)
    ss.setdefault("m7_file_right", None)
    ss.setdefault("m7_pair_id", None)
    ss.setdefault("m7_dialog_opened", None)
    ss.setdefault("m7_ack", None)
    ss.setdefault("m7_scroll_to_cal_pair", False)
    ss.setdefault("m7_points", {"left": [], "right": []})
    ss.setdefault("m7_pending", None)           # (side, coarse point) awaiting zoom refinement
    ss.setdefault("m7_refined", None)           # dragged position of the pending corner (original pixels), uncommitted
    ss.setdefault("m7_left_confirmed", False)   # the four left corners were reviewed and confirmed
    ss.setdefault("m7_right_confirmed", False)
    ss.setdefault("m7_stage", "left")           # current selection stage: "left" / "right" / "ready"
    ss.setdefault("m7_ack_right", None)         # pair id whose RIGHT instructions were dismissed with Start
    ss.setdefault("m7_dialog_right", None)      # pair id whose RIGHT instructions were already opened (once per transition)
    ss.setdefault("m7_calculated", False)
    ss.setdefault("m7_visitor_model", None)     # active visitor calibration (never Model B)
    ss.setdefault("m7_cal", wiz.new_state())    # calibration wizard: settings, pair slots, per-photo results
    ss.setdefault("m7_cal_error", None)
    ss.setdefault("m7_cal_notice", None)
    ss.setdefault("m7_cal_load_error", None)
    ss.setdefault("m7_cal_loaded_name", None)
    ss.setdefault("m7_cal_run_requested", False)
    ss.setdefault("m7_cal_validation", [])      # messages of the last refused Run Stereo Calibration click
    ss.setdefault("m7_cal_workflow", None)      # selected Section 3 workflow: None / "run" / "load" (UI only)
    ss.setdefault("m7_cal_source", "none")      # source of the active calibration: "none" / "computed" / "loaded"
    ss.setdefault("m7_cal_npz_gen", 0)
    ss.setdefault("m7_cal_form_gen", 0)         # suffix of the checkerboard / pair-count widget keys; bumped by Reset All
    ss.setdefault("m7_measurement_enabled", False)
    ss.setdefault("m7_saved_inputs", {})        # widget values kept across navigation (see save_inputs)


def reset_selection():
    ss = st.session_state
    ss["m7_points"] = {"left": [], "right": []}
    ss["m7_pending"] = None                     # (side, coarse point) awaiting zoom refinement
    ss["m7_refined"] = None
    ss["m7_left_confirmed"] = ss["m7_right_confirmed"] = False
    ss["m7_calculated"] = False
    ss["m7_stage"] = "left"
    ss["m7_ack_right"] = ss["m7_dialog_right"] = None   # right instructions reappear on the next transition into the right stage


def invalidate_confirmation(side=None):
    """A changed corner invalidates its image's confirmation and any result. Changing the left image also sends the workflow back
    to the left stage, so both images must be confirmed again; the right corners themselves are kept."""
    ss = st.session_state
    if side in (None, "left"):
        ss["m7_left_confirmed"] = False
        ss["m7_ack_right"] = ss["m7_dialog_right"] = None
    ss["m7_right_confirmed"] = False
    ss["m7_calculated"] = False


# Widget-owned values that Streamlit drops while their widget is not rendered. Navigation keeps a copy in m7_saved_inputs and puts it back
# (only if the key is missing) before the widget is created. The corner-click and zoom components (m7_coarse_*, m7_zoom_*) are deliberately
# not restored: replaying a stored click could be consumed twice, and the refined corner already lives in m7_pending / m7_refined.
SAVED_INPUT_KEYS = ("m7_exp_pair", "m7_cal_details")
SAVED_INPUT_PREFIXES = ("m7_gt_w_", "m7_gt_h_")


def _is_saved_input(key):
    key = str(key)
    if key in SAVED_INPUT_KEYS:
        return True
    return key.startswith(SAVED_INPUT_PREFIXES) and key.endswith(f"_{st.session_state['m7_upload_gen']}")


def save_inputs():
    ss = st.session_state
    saved = ss.setdefault("m7_saved_inputs", {})
    for key in list(ss.keys()):
        if _is_saved_input(key):
            saved[key] = ss[key]


def restore_inputs(*, experiment, measurement):
    """Called before the widgets of the view are created; never overwrites a value that is already present."""
    ss = st.session_state
    for key, value in ss.get("m7_saved_inputs", {}).items():
        wanted = experiment if key == "m7_exp_pair" else measurement
        if wanted and key not in ss and _is_saved_input(key):
            ss[key] = value


def forget_saved_inputs(prefixes=SAVED_INPUT_PREFIXES):
    saved = st.session_state.get("m7_saved_inputs", {})
    for key in [k for k in saved if str(k).startswith(prefixes)]:
        del saved[key]


def select_part(part):
    save_inputs()
    st.session_state["module7_part"] = part


def store_upload(side):
    uploaded = st.session_state.get(f"m7_up_{side}_{st.session_state['m7_upload_gen']}")
    st.session_state[f"m7_file_{side}"] = (uploaded.name, uploaded.getvalue()) if uploaded is not None else None


def remove_uploads():
    ss = st.session_state
    ss["m7_upload_gen"] += 1                    # fresh uploader and actual-size widgets: the browser cannot restore old values
    ss["m7_file_left"] = ss["m7_file_right"] = None
    forget_saved_inputs()
    reset_selection()


def measure_another_image():
    """Drop the current object photographs, corners and results; the active calibration is kept."""
    remove_uploads()
    st.session_state["m7_pair_id"] = st.session_state["m7_ack"] = st.session_state["m7_dialog_opened"] = None
    st.session_state["m7_ack_right"] = st.session_state["m7_dialog_right"] = None


def undo_point():
    ss = st.session_state
    if ss["m7_pending"] is not None:
        ss["m7_pending"] = ss["m7_refined"] = None
        ss["m7_gen"] += 1
        return
    side = "right" if ss["m7_left_confirmed"] and ss["m7_points"]["right"] else "left"
    if ss["m7_points"][side]:
        ss["m7_points"][side].pop()
        invalidate_confirmation(side)


def reset_side(side):
    st.session_state["m7_points"][side] = []
    st.session_state["m7_pending"] = st.session_state["m7_refined"] = None
    st.session_state["m7_gen"] += 1
    invalidate_confirmation(side)


def reset_both():
    reset_selection()


def confirm_side(side):
    """Confirm a complete set of four corners. Confirming the left image moves on to the right stage (and its instructions)."""
    ss = st.session_state
    if len(ss["m7_points"][side]) != 4 or ss["m7_pending"] is not None:
        return
    if side == "left":
        ss["m7_left_confirmed"] = True
        ss["m7_ack_right"] = ss["m7_dialog_right"] = None
    elif ss["m7_left_confirmed"]:
        ss["m7_right_confirmed"] = True


def calculate_now():
    st.session_state["m7_calculated"] = True


# ---------------------------------------------------------------- visitor calibration state

def reset_measurement():
    """Drop everything derived from the previous visitor calibration (the saved notebook experiment is untouched)."""
    remove_uploads()
    st.session_state["m7_pair_id"] = st.session_state["m7_ack"] = st.session_state["m7_dialog_opened"] = None
    st.session_state["m7_ack_right"] = st.session_state["m7_dialog_right"] = None


def set_visitor_model(model, loaded_name=None):
    st.session_state["m7_visitor_model"] = model
    st.session_state["m7_cal_loaded_name"] = loaded_name
    st.session_state["m7_cal_source"] = "none" if model is None else ("loaded" if loaded_name is not None else "computed")
    st.session_state["m7_cal_error"] = st.session_state["m7_cal_load_error"] = None
    st.session_state["m7_cal_validation"] = []
    reset_measurement()


def clear_calibration():
    set_visitor_model(None)
    st.session_state["m7_cal_validation"] = []
    st.session_state["m7_cal_notice"] = None


def select_workflow(workflow):
    """Only records which Section 3 workflow is shown; never runs a calibration or touches the active model."""
    st.session_state["m7_cal_workflow"] = workflow


def request_run():
    """Callback of the Run Stereo Calibration button: selects the Run workflow and asks for ONE calibration attempt.

    Callbacks fire once per click, so reruns, navigation and uploads never re-trigger it; the flag is consumed in run_workflow.
    """
    st.session_state["m7_cal_workflow"] = "run"
    st.session_state["m7_cal_run_requested"] = True


def clear_all_photos():
    wiz.remove_all_photos(st.session_state["m7_cal"])


def form_key(name):
    """Key of a checkerboard / pair-count widget in the current generation (see reset_all)."""
    return f"m7_cal_{name}_{st.session_state['m7_cal_form_gen']}"


RESET_WIDGET_KEYS = ("m7_cal_goto", "m7_cal_details", "m7_cal_download")
OLD_FORM_PREFIXES = ("m7_cal_cols_", "m7_cal_rows_", "m7_cal_square_", "m7_cal_npairs_")
RESET_WIDGET_PREFIXES = ("m7_cal_up_", "m7_cal_npz_", "m7_up_", "m7_coarse_", "m7_zoom_", "m7_gt_")


def reset_all():
    """Single reset for both 'Reset All' buttons: the visitor's calibration and measurement workflow back to first-visit defaults.

    Runs as a button callback (before the next script run), so widget keys may be deleted safely. Touches session state only: Model B,
    the notebook selections and results, and the experiment view are never read or written here.
    """
    ss = st.session_state
    wiz.reset_all(ss["m7_cal"])
    set_visitor_model(None)                     # clears the model, load errors and, via reset_measurement, uploads and selections
    reset_selection()
    ss["m7_cal_error"] = ss["m7_cal_notice"] = ss["m7_cal_load_error"] = ss["m7_cal_loaded_name"] = None
    ss["m7_cal_workflow"] = None
    ss["m7_cal_run_requested"] = False
    ss["m7_cal_validation"] = []
    ss["m7_cal_source"] = "none"
    ss["m7_cal_npz_gen"] += 1
    # Deleting a number_input's key is not enough: the browser still holds the old value and sends it back on the next rerun. New keys
    # make the four inputs brand-new widgets that start from the reset wizard state.
    ss["m7_cal_form_gen"] += 1
    ss["m7_gen"] += 1
    ss["m7_measurement_enabled"] = False
    ss["m7_saved_inputs"] = {}
    for key in list(ss.keys()):
        if key in RESET_WIDGET_KEYS or str(key).startswith(OLD_FORM_PREFIXES) or (str(key).startswith(RESET_WIDGET_PREFIXES) and not str(key).endswith("_gen")):
            del ss[key]


def on_photo_upload(index, side):
    ss = st.session_state
    uploaded = ss.get(f"m7_cal_up_{wiz.slot_key(index, side)}_{ss['m7_cal']['slot_gen'].get(wiz.slot_key(index, side), 0)}")
    if uploaded is not None:
        wiz.store_photo(ss["m7_cal"], index, side, uploaded.name, uploaded.getvalue())


def on_photo_remove(index, side):
    wiz.remove_photo(st.session_state["m7_cal"], index, side)


def on_exclude(index):
    wiz.toggle_exclude(st.session_state["m7_cal"], index)


def change_pair(move):
    """Apply a pair change and request the one-shot scroll only if the selected pair really changed."""
    ss = st.session_state
    before = ss["m7_cal"]["current"]
    move(ss["m7_cal"])
    if ss["m7_cal"]["current"] != before:
        ss["m7_scroll_to_cal_pair"] = True


def on_step(delta):
    change_pair(lambda state: wiz.step(state, delta))


def on_goto():
    change_pair(lambda state: wiz.go_to(state, st.session_state["m7_cal_goto"]))


def on_calibration_file():
    ss = st.session_state
    uploaded = ss.get(f"m7_cal_npz_{ss['m7_cal_npz_gen']}")
    if uploaded is None:
        return
    ss["m7_cal_npz_gen"] += 1                   # empties the uploader so the same file can be chosen again
    try:
        model = load_calibration(uploaded.getvalue())
    except CalibrationFileError as error:
        ss["m7_cal_load_error"] = f"'{uploaded.name}' was not loaded: {error}"
        return
    set_visitor_model(model, uploaded.name)
    ss["m7_cal_notice"] = None


# One-shot: runs on the render right after Previous / Next / the pair selector changed the pair.
# It brings the pair heading (the start of the left/right upload area) into view without going to the top of the module.
SCROLL_TO_CAL_PAIR_JS = """
<script>
setTimeout(function () {
  var anchor = document.getElementById("m7-cal-pair-anchor");
  if (!anchor) { return; }
  anchor.style.scrollMarginTop = "72px";
  anchor.scrollIntoView({behavior: "smooth", block: "start"});
}, 150);
</script>
"""


# ---------------------------------------------------------------- calibration reference

def matrix_table(matrix, digits):
    return st.table([[f"{value:.{digits}f}" for value in row] for row in matrix])


def show_calibration_reference(model):
    with st.container(key="m7_calib_ref"), st.expander(CALIB_EXPANDER_LABEL):
        st.write(
            "These parameters come from the stored Model B stereo calibration and are used to undo lens distortion "
            "and to triangulate matching points from the two cameras into 3D coordinates."
        )
        width, height = model["image_size"]
        st.write(f"**Left camera:** {LEFT_CAMERA}")
        st.write(f"**Right camera:** {RIGHT_CAMERA}")
        st.write(f"**Calibration image resolution:** {width} × {height} pixels (portrait, EXIF orientation applied)")
        st.write(f"**Checkerboard:** {PATTERN[0]} × {PATTERN[1]} inner corners, {SQUARE_MM:g} mm squares")
        st.write(f"**Calibration pairs:** {len(model['pairs'])}")
        st.write(f"**Measured camera baseline:** {MEASURED_BASELINE_MM} mm")
        st.write(f"**Estimated stereo baseline (|T|):** {model['baseline_mm']:.4f} mm")
        st.caption("Camera names, checkerboard layout and the measured baseline are the documented setup of this experiment; "
                   "the matrices below are read from the saved calibration file.")

        st.subheader("Intrinsic Parameters")
        st.write("**Left camera intrinsic matrix K (3×3)**")
        matrix_table(model["K1"], 4)
        st.write("**Right camera intrinsic matrix K (3×3)**")
        matrix_table(model["K2"], 4)
        d_left, d_right = model["D1"].ravel(), model["D2"].ravel()
        st.write("**Distortion coefficients**")
        st.table({
            "Coefficient": ["k1", "k2", "p1", "p2", "k3"],
            "Left": [f"{v:.6f}" for v in d_left],
            "Right": [f"{v:.6f}" for v in d_right],
        })
        st.caption("The calibration estimated k1 and k2 only; p1, p2 and k3 were held at zero.")

        st.subheader("Extrinsic Parameters (stereo)")
        st.write("**Rotation matrix R (right camera relative to left)**")
        matrix_table(model["R"], 6)
        st.write("**Translation vector T (mm)**")
        st.table({"Component": ["Tx", "Ty", "Tz"], "Value (mm)": [f"{v:.4f}" for v in model["T"].ravel()]})

        st.subheader("Calibration Error")
        st.table({
            "RMS reprojection error (pixels)": ["Left camera (single-camera)", "Right camera (single-camera)", "Stereo calibration"],
            "Value": [f"{model['rms_left']:.4f}", f"{model['rms_right']:.4f}", f"{model['stereo_rms']:.4f}"],
        })
        st.caption("The stereo RMS is noticeably larger than the single-camera values, which limits the accuracy that "
                   "can be expected from this calibration.")

        st.subheader("Intrinsic and Extrinsic Calibration")
        st.write(
            "Intrinsic calibration describes each camera on its own: focal lengths, principal point and lens distortion. "
            "Extrinsic (stereo) calibration describes how the two cameras are placed relative to each other: the rotation R "
            "and translation T that map left-camera coordinates to right-camera coordinates. The fixed distance between the "
            "cameras (the baseline) is what lets two views determine depth."
        )

        st.subheader("Sample Stereo Calibration Image")
        if CALIB_SAMPLE_IMAGE.is_file():
            st.image(str(CALIB_SAMPLE_IMAGE), width="stretch", caption=(
                f"Captured by the left camera ({LEFT_CAMERA}), calibration pair {CALIB_SAMPLE_PAIR} of {len(model['pairs'])}. "
                f"Checkerboard: {PATTERN[0]} × {PATTERN[1]} inner corners. Square size: {SQUARE_MM:g} mm. "
                "Cropped around the board for display; EXIF orientation applied."))
        else:
            st.warning("The sample calibration image is unavailable.")


# ---------------------------------------------------------------- visitor calibration wizard

def f4(value, digits=4):
    return f"{value:.{digits}f}"


def stats_text(stats, unit="px"):
    return "not available" if stats is None else f"mean {stats['mean']:.2f} · median {stats['median']:.2f} · max {stats['max']:.2f} {unit}"


def calibration_settings():
    """Checkerboard inputs. Returns the validated settings (or None) and applies them to the wizard."""
    wiz_state = st.session_state["m7_cal"]
    st.subheader("1. Checkerboard settings")
    c1, c2, c3 = st.columns(3)
    # 0 means "not configured". The widgets start from the last accepted settings (0 on first use), so they survive a trip away from
    # this page, and photos are only cleared when new valid settings really differ (apply_settings), never by an unrelated rerun.
    saved = wiz_state["settings"]
    cols = c1.number_input("Inner corners (columns)", min_value=0, max_value=wiz.CORNER_LIMITS[1], value=saved["cols"], step=1, key=form_key("cols"))
    rows = c2.number_input("Inner corners (rows)", min_value=0, max_value=wiz.CORNER_LIMITS[1], value=saved["rows"], step=1, key=form_key("rows"))
    square = c3.number_input("Square size (mm)", min_value=0.0, max_value=wiz.SQUARE_LIMITS_MM[1], value=float(saved["square_mm"]), step=0.5,
                             format="%.2f", key=form_key("square"))
    st.caption(
        "**Inner corners** are the points where four squares meet inside the board, not the number of squares. A board with "
        "9 × 7 squares has 8 × 6 inner corners. The **square size** is the printed side length of one square; it sets the metric "
        "scale of every 3D measurement, so measure it on your printout.")
    settings, errors = wiz.validate_settings(cols, rows, square)
    if errors:
        show = st.warning if 0 in (cols, rows, square) else st.error     # missing values: yellow; invalid values: red
        for error in errors:
            show(error)
        return None
    if wiz.apply_settings(wiz_state, settings):
        st.warning("The checkerboard settings changed, so the photos analysed earlier were cleared from the pair slots. "
                   "Upload them again.")
    return settings


def calibration_guidance():
    with st.container(border=True):
        st.markdown("**How to photograph the checkerboard**")
        st.markdown(
            "- Use two cameras mounted in a fixed stereo setup, and do not move them relative to each other.\n"
            "- For each pair, photograph the **same board position** with both cameras.\n"
            "- Keep the **entire checkerboard visible** in both photos, sharp and evenly lit.\n"
            "- Vary the position, distance and tilt between pairs.\n"
            "- Keep the same lenses, zoom, image resolution and orientation throughout.\n"
            "- Do not change the stereo rig after calibration; the result only describes this setup.")


def photo_slot(index, side, enabled):
    ss = st.session_state
    wiz_state = ss["m7_cal"]
    key = wiz.slot_key(index, side)
    rec = wiz_state["photos"].get(key)
    st.markdown(f"**{SIDE_LABEL[side]} camera image**")
    if rec is not None:
        if rec.get("thumbnail"):
            st.image(rec["thumbnail"], width="stretch")
        st.caption(f"{rec['name']} · {rec['bytes'] / 1e6:.1f} MB" + (f" · {rec['size'][0]} × {rec['size'][1]} px" if rec.get("size") else ""))
        if rec["status"] == "ok":
            st.caption("Checkerboard detected.")
        else:
            st.error(f"Pair {index + 1}, {side} camera: {rec['message']}")
    if rec is None:
        st.file_uploader(f"Upload {side} camera image (pair {index + 1})", type=["jpg", "jpeg", "png"],
                         key=f"m7_cal_up_{key}_{wiz_state['slot_gen'].get(key, 0)}", on_change=on_photo_upload, args=(index, side),
                         disabled=not enabled, label_visibility="collapsed")
    else:
        st.button(f"Remove {side} photo", key=f"m7_cal_rm_{key}", on_click=on_photo_remove, args=(index, side))


def pair_navigator(evaluation, enabled):
    ss = st.session_state
    wiz_state = ss["m7_cal"]
    n, current = wiz_state["n_pairs"], wiz_state["current"]
    row = evaluation["pairs"][current]
    st.html('<div id="m7-cal-pair-anchor"></div>')
    st.markdown(f"#### Pair {current + 1} of {n} · {row['status']}")
    for side_col, side in zip(st.columns(2), SIDES):
        with side_col:
            photo_slot(current, side, enabled)
    shown_in_slots = {f"Pair {current + 1}, {side} camera: {row[side]['message']}" for side in SIDES
                      if row[side] is not None and row[side]["status"] != "ok"}
    for message in row["messages"]:
        if message in shown_in_slots:
            continue
        (st.caption if row["status"] == wiz.VALID else st.info if row["status"] == wiz.INCOMPLETE else st.warning)(message)
    excluded = current in wiz_state["excluded"]
    top = st.columns([1, 1])
    top[0].button("Include this pair again" if excluded else "Exclude this pair from calibration", key="m7_cal_exclude",
                  on_click=on_exclude, args=(current,), width="stretch")
    ss["m7_cal_goto"] = current
    # filter_mode=None turns the dropdown into a plain, non-editable select (no text cursor); arrow keys and Enter still work
    top[1].selectbox("Go to pair", list(range(n)), key="m7_cal_goto", on_change=on_goto, label_visibility="collapsed", filter_mode=None,
                     format_func=lambda i: f"Pair {i + 1} – {evaluation['pairs'][i]['status']}")
    nav = st.columns(2)
    nav[0].button("Previous", key="m7_cal_prev", on_click=on_step, args=(-1,), disabled=current == 0, width="stretch")
    nav[1].button("Next", key="m7_cal_next", on_click=on_step, args=(1,), disabled=current == n - 1, width="stretch")
    if ss.pop("m7_scroll_to_cal_pair", False):       # last element, so the pair section has rendered before the script runs
        st.html(SCROLL_TO_CAL_PAIR_JS, unsafe_allow_javascript=True)


def pair_overview(evaluation):
    def side_text(rec):
        return "-" if rec is None else ("OK" if rec["status"] == "ok" else rec["status"].replace("_", " "))
    st.dataframe(pd.DataFrame([{"Pair": r["number"], "Status": r["status"], "Left photo": side_text(r["left"]),
                                "Right photo": side_text(r["right"])} for r in evaluation["pairs"]]).set_index("Pair"),
                 width="stretch")


def calibration_results(model):
    st.success("Stereo Calibration Completed" if model.get("source") == wiz.SOURCE_NEW else "Stereo Calibration Loaded")
    if model.get("source") == wiz.SOURCE_NEW:
        st.write("**Active calibration:** newly computed from your photographs in this session.")
    else:
        st.write(f"**Active calibration:** loaded from the file '{st.session_state['m7_cal_loaded_name']}'.")
        st.warning("The file was checked for structure and internal consistency only. It is **not authenticated**, so its accuracy "
                   "and origin cannot be confirmed.")
    for text in model.get("load_warnings", []):
        st.warning(text)
    metrics = model["metrics"]
    c = st.columns(3)
    c[0].metric("Uploaded pairs", model.get("pairs_uploaded", "-"))
    c[1].metric("Valid pairs used", model.get("pairs_valid", len(model["pairs"])))
    c[2].metric("Baseline", f"{model['baseline_mm']:.1f} mm")
    st.metric("Distortion models (left / right)", f"{model['distortion_left']} / {model['distortion_right']}")   # full width: never truncated
    c = st.columns(3)
    c[0].metric("Left camera RMS", f"{model['rms_left']:.3f} px")
    c[1].metric("Right camera RMS", f"{model['rms_right']:.3f} px")
    c[2].metric("Stereo RMS", f"{model['stereo_rms']:.3f} px")
    st.write(f"**Epipolar residual:** {stats_text(metrics['epipolar_px'])}")
    st.write(f"**Rectification residual (vertical misalignment):** {stats_text(metrics['rectification_vertical_px'])}")
    st.caption("Reprojection RMS and residuals describe how well the model reproduces the checkerboard corners in the calibration "
               "photos. They are **not** the physical error of a later object measurement, which also depends on corner picking, "
               "distance and the baseline. The warning thresholds are heuristics that scale with image resolution, not guaranteed "
               "accuracy limits.")
    st.write("**Selected distortion models:** left " + model["distortion_left"] + ", right " + model["distortion_right"] +
             " (radial2 = k1, k2; radial3 adds k3; full adds tangential terms).")
    warnings = model.get("warnings", [])
    if warnings:
        st.write("**Calibration quality warnings**")
        for item in warnings:
            st.warning(item["message"] + (" (heuristic)" if item.get("heuristic") else ""))
    else:
        st.info("No quality warnings were raised. This does not guarantee measurement accuracy.")
    for note in model.get("camera_notes", []):
        st.caption(note)


def calibration_details(model):
    st.write("**Left intrinsic matrix K**")
    matrix_table(model["K1"], 4)
    st.write("**Right intrinsic matrix K**")
    matrix_table(model["K2"], 4)
    st.write("**Distortion coefficients**")
    st.table({"Coefficient": ["k1", "k2", "p1", "p2", "k3"], "Left": [f"{v:.6f}" for v in model["D1"].ravel()],
              "Right": [f"{v:.6f}" for v in model["D2"].ravel()]})
    st.write("**Rotation matrix R (right camera relative to left)**")
    matrix_table(model["R"], 6)
    st.write("**Translation vector T (mm)**")
    st.table({"Component": ["Tx", "Ty", "Tz"], "Value (mm)": [f"{v:.4f}" for v in model["T"].ravel()]})
    st.write(f"**Image resolutions:** left {model['image_size_left'][0]} × {model['image_size_left'][1]} px, "
             f"right {model['image_size_right'][0]} × {model['image_size_right'][1]} px")
    st.write(f"**Checkerboard:** {model['pattern'][0]} × {model['pattern'][1]} inner corners, {model['square_mm']:g} mm squares")
    st.write("**Pair numbers used:** " + ", ".join(str(p) for p in model["pairs"]))

    st.write("**Per-pair error information**")
    if model.get("per_pair"):
        st.dataframe(pd.DataFrame([{"Pair": p["pair"], "Left mono (px)": p["mono_left_px"], "Right mono (px)": p["mono_right_px"],
                                    "Left stereo (px)": p["stereo_left_px"], "Right stereo (px)": p["stereo_right_px"],
                                    "Mean epipolar (px)": p["epipolar_mean_px"], "Outlier": p["outlier"]} for p in model["per_pair"]]
                                  ).set_index("Pair").round(3), width="stretch")
    else:
        st.caption("Per-pair errors are not stored in calibration files.")

    st.write("**Distortion-model cross-validation** (held-out reprojection RMS; the simplest model within 10% of the best is chosen)")
    tables = model.get("distortion_cv", {})
    if tables.get("left"):
        st.dataframe(pd.DataFrame([{"Model": a["model"], "Left held-out RMS (px)": "failed" if not np.isfinite(a["cv_rms"]) else round(a["cv_rms"], 3),
                                    "Right held-out RMS (px)": "failed" if not np.isfinite(b["cv_rms"]) else round(b["cv_rms"], 3),
                                    "Chosen": "left" * a["chosen"] + (" + " if a["chosen"] and b["chosen"] else "") + "right" * b["chosen"]}
                                   for a, b in zip(tables["left"], tables["right"])]).set_index("Model"), width="stretch")
    else:
        st.caption("No cross-validation table is available for this calibration.")

    st.write("**Calibration metadata**")
    camera = model.get("camera") or {}
    meta = {"Source": "newly computed" if model.get("source") == wiz.SOURCE_NEW else "loaded from file (not authenticated)",
            "Created (UTC)": model.get("created_utc", "-"), "Software": model.get("software", "this app (session)"),
            "Pairs uploaded / valid": f"{model.get('pairs_uploaded', '-')} / {model.get('pairs_valid', '-')}"}
    for side in SIDES:
        info = camera.get(side) or {}
        meta[f"{SIDE_LABEL[side]} camera"] = ", ".join(f"{k}: {v}" for k, v in info.items() if k != "n_photos") or "no metadata in the photos"
    st.table({"Item": list(meta), "Value": [str(v) for v in meta.values()]})


def calibration_download(model):
    """Download button of the Calibrate New Camera Setup workflow; always visible, enabled only for a newly computed calibration."""
    data = None
    if model is not None and st.session_state["m7_cal_source"] == "computed":
        try:
            data = export_calibration(model, model.get("pairs_uploaded"), model.get("camera"), model.get("created_utc"))
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"This calibration cannot be exported, so downloading is disabled: {error}")
    st.download_button("Download Calibration (.npz)", data if data is not None else b"", file_name="stereo_calibration.npz",
                       mime="application/octet-stream", key="m7_cal_download", icon=":material/download:", disabled=data is None)
    if data is None:
        st.caption("Run the stereo calibration to enable downloading.")
    else:
        st.caption("The file holds calibration numbers only, no photographs. Keep it with your camera setup; it is valid only for the "
                   "same cameras, lenses and settings.")


def active_calibration_message(workflow):
    """Short note when the active calibration came from the other workflow than the one being shown."""
    source = st.session_state["m7_cal_source"]
    if source == "computed" and workflow != "run":
        st.info("A calibration computed from your photographs is currently active. Loading a file will replace it.")
    elif source == "loaded" and workflow != "load":
        st.info(f"A calibration loaded from the file '{st.session_state['m7_cal_loaded_name']}' is currently active. "
                "Running a new calibration will replace it.")


def calibration_actions(model, details_key="m7_cal_details"):
    if st.toggle("View Calibration Details", key=details_key):
        calibration_details(model)
    st.button("Clear Calibration", key="m7_cal_clear", on_click=clear_calibration)
    st.button("Reset All", key="m7_cal_reset_bottom", on_click=reset_all)


def load_workflow(model):
    ss = st.session_state
    st.write("**Load Saved Calibration (.npz)**")
    st.file_uploader("Load Saved Calibration (.npz)", type=["npz"], key=f"m7_cal_npz_{ss['m7_cal_npz_gen']}",
                     on_change=on_calibration_file, label_visibility="collapsed")
    st.caption("Loading a file replaces the active calibration. Files are validated for structure and consistency, not authenticity: "
               "a valid file does not guarantee that it is genuine or that measurements made with it are accurate.")
    if ss["m7_cal_load_error"]:
        st.error(ss["m7_cal_load_error"])
    active_calibration_message("load")
    if model is not None and ss["m7_cal_source"] == "loaded":
        calibration_results(model)
        calibration_actions(model)


def run_workflow(model, evaluation, enabled):
    ss = st.session_state
    wiz_state = ss["m7_cal"]
    if ss.pop("m7_cal_run_requested", False):
        problems = wiz.validate_run(enabled, evaluation)
        ss["m7_cal_validation"] = problems
        ss["m7_cal_validation_state"] = (evaluation["signature"], enabled, evaluation["n_valid"], evaluation["n_uploaded"])
        if not problems:
            try:
                with st.spinner("Calibrating the stereo cameras (this can take a minute)..."):
                    result = wiz.run_calibration(wiz_state)
            except wiz.WizardError as error:
                ss["m7_cal_error"] = str(error)       # a failed attempt leaves any active model untouched
            else:
                set_visitor_model(result)             # clears errors and dependent measurements
                ss["m7_cal_notice"] = None
                model = result
    if ss["m7_cal_validation"] and ss.get("m7_cal_validation_state") != (evaluation["signature"], enabled, evaluation["n_valid"],
                                                                         evaluation["n_uploaded"]):
        ss["m7_cal_validation"] = []                  # inputs changed since the refused click: the message would be stale
    for problem in ss["m7_cal_validation"]:
        st.warning(problem)
    if ss["m7_cal_error"]:
        st.error(ss["m7_cal_error"])
    active_calibration_message("run")
    if model is not None and ss["m7_cal_source"] == "computed":
        calibration_results(model)
    calibration_download(model)
    if model is not None and ss["m7_cal_source"] == "computed":
        calibration_actions(model)


def workflow_selector():
    """The two large workflow buttons at the top of the expander; they only select a workflow and never run anything."""
    workflow = st.session_state["m7_cal_workflow"]
    columns = st.columns(2)
    for column, (name, key, label) in zip(columns, (("load", "m7_cal_mode_load", "Load Saved Calibration"),
                                                    ("run", "m7_cal_mode_run", "Calibrate New Camera Setup"))):
        with column:
            st.button(label, key=key, type="primary" if workflow == name else "secondary", on_click=select_workflow, args=(name,),
                      width="stretch")
    return workflow


def new_calibration_steps(model):
    """Workflow B: the three-step calibration process (checkerboard settings, stereo pairs, run)."""
    ss = st.session_state
    wiz_state = ss["m7_cal"]
    st.write("Photograph a checkerboard with both cameras, upload each stereo pair, then run the calibration. "
             "Your photos are analysed in this session only and are not saved.")
    calibration_guidance()
    settings = calibration_settings()
    enabled = settings is not None

    st.subheader("2. Upload stereo pairs")
    choice = st.number_input("Number of stereo pairs", min_value=wiz.MIN_PAIRS, max_value=wiz.MAX_PAIRS, value=wiz_state["n_pairs"], step=1,
                             key=form_key("npairs"), help=f"{wiz.MIN_PAIRS} to {wiz.MAX_PAIRS} pairs. Lowering the count hides the last "
                             "slots but keeps their photos; raising it brings them back.")
    wiz.set_pair_count(wiz_state, int(choice))
    evaluation = wiz.evaluate(wiz_state)
    st.progress(evaluation["n_valid"] / wiz_state["n_pairs"])
    st.write(f"**{evaluation['n_uploaded']} of {wiz_state['n_pairs']} pairs uploaded** · **{evaluation['n_valid']} valid stereo pairs**")
    st.caption(f"At least {wiz.uc.MIN_VALID_PAIRS} valid pairs are required; {wiz.uc.RECOMMENDED_PAIRS[0]}-{wiz.uc.RECOMMENDED_PAIRS[1]} "
               "give a more robust result. Pairs are matched by their slot number, never by file name.")
    pair_navigator(evaluation, enabled)
    with st.popover("All pairs at a glance"):
        pair_overview(evaluation)
    st.button("Remove all calibration photos", key="m7_cal_clear_photos", on_click=clear_all_photos,
              disabled=not wiz_state["photos"])
    st.button("Reset All", key="m7_cal_reset_top", on_click=reset_all)

    st.subheader("3. Run calibration")
    st.button("Run Stereo Calibration", key="m7_cal_run", type="primary", on_click=request_run)
    run_workflow(model, evaluation, enabled)


def calibrate_workflow():
    """The content of the 'Calibrate Your Stereo Cameras' expander."""
    ss = st.session_state
    wiz_state = ss["m7_cal"]
    model = ss["m7_visitor_model"]
    if model is not None and not wiz.is_current(wiz_state, model):
        set_visitor_model(None)
        ss["m7_cal_notice"] = "The photos, exclusions or checkerboard settings changed, so the previous calibration was cleared."
        model = None
    workflow = workflow_selector()
    if ss["m7_cal_notice"]:
        st.info(ss["m7_cal_notice"])
    if workflow == "run":
        new_calibration_steps(model)
    elif workflow == "load":
        load_workflow(model)
    else:
        active_calibration_message("none")
        st.caption("Choose how to set up your stereo cameras.")


# ---------------------------------------------------------------- upload handling

def calibration_summary(model):
    st.subheader("Upload Stereo Images")
    loaded = model.get("source") != wiz.SOURCE_NEW
    left, right = model["image_size_left"], model["image_size_right"]
    st.markdown(CALIBRATION_SUMMARY_CSS, unsafe_allow_html=True)
    with st.container(key="m7_calibration_summary"):
        row1 = st.columns(2)       # two wide columns; Streamlit stacks them on narrow screens
        row1[0].metric("Calibration source", "Loaded from NPZ" if loaded else "Newly computed")
        row1[1].metric("Baseline", f"{model['baseline_mm']:.1f} mm")
        row2 = st.columns(2)
        row2[0].metric("Left image resolution", f"{left[0]} × {left[1]} px")
        row2[1].metric("Right image resolution", f"{right[0]} × {right[1]} px")
    if model.get("pairs_valid") is not None:
        st.caption(f"Valid calibration pairs: {model['pairs_valid']}")
    st.write("Object measurements will use this active stereo calibration.")
    if loaded:
        st.caption("This calibration was validated structurally but not independently verified for accuracy.")


def upload_section(model):
    """Uploaders and validation against the active calibration. Returns side -> {name, data, digest, warnings, size} when both
    uploads are usable, else None."""
    calibration_summary(model)
    st.write("Upload two corresponding photographs of the **same stationary rectangular object**, taken at the same time "
             "from the two calibrated cameras. Each image must have the same resolution and orientation as that camera's "
             "calibration photos; images are never resized or rotated to fit.")
    generation = st.session_state["m7_upload_gen"]
    columns = st.columns(2)
    for column, side in zip(columns, SIDES):
        with column:
            st.file_uploader(
                f"{SIDE_LABEL[side]} camera image", type=["jpg", "jpeg", "png"],
                key=f"m7_up_{side}_{generation}", on_change=store_upload, args=(side,),
            )
            stored = st.session_state[f"m7_file_{side}"]
            if stored is not None:
                st.caption(f"Using: {stored[0]} ({len(stored[1]) / 1e6:.1f} MB)")

    files = {side: st.session_state[f"m7_file_{side}"] for side in SIDES}
    if None in files.values():
        st.info("Upload both the left and the right image to continue.")
        return None

    uploads, blocked = {}, False
    for side in SIDES:
        name, data = files[side]
        try:
            report = sel.check_upload_for_calibration(data, side, model)
        except sel.UploadError as error:
            st.error(f"{SIDE_LABEL[side]} image '{name}' was rejected: {error}")
            blocked = True
            continue
        warnings = [w["message"] for w in report["warnings"]]
        for warning in warnings:
            st.warning(f"{SIDE_LABEL[side]} image: {warning}")
        uploads[side] = {"name": name, "data": data, "digest": hashlib.sha256(data).hexdigest(), "warnings": warnings,
                         "size": tuple(report["size"])}
    if blocked:
        return None
    if uploads["left"]["digest"] == uploads["right"]["digest"]:
        st.error("The left and right uploads are the same file. Upload two different photographs.")
        return None
    return uploads


def upload_placeholder():
    """No active visitor calibration: measurement uploaders are not shown."""
    st.subheader("Upload Stereo Images")
    st.info(
        "First calibrate your stereo cameras using the section above, or load a saved calibration (.npz). After a calibration is "
        "active, upload one left and one right photograph of the object you want to measure."
    )
    return None


def corner_instructions(side, pair_id):
    """Body of the per-image instruction popup; Start records that this side's instructions were read for this pair."""
    st.write(f"**You are selecting corners on the {side.upper()} camera image.**")
    st.write("Mark the **outermost physical corners** of the rectangular object, in this order:")
    st.markdown("1. **Top-left (TL)**  \n2. **Top-right (TR)**  \n3. **Bottom-right (BR)**  \n4. **Bottom-left (BL)**")
    st.image(sel.guide_image(), caption="Corner order", width="content")
    st.write(
        "Click near each corner on the photograph. A zoomed-in view then opens: drag the marker onto the exact corner and "
        "click **Accept Corner** (or **Cancel**). Use the same four physical corners in the same order on both images."
    )
    if st.button("Start Corner Selection", type="primary", key=f"m7_start_{side}"):
        st.session_state["m7_ack" if side == "left" else "m7_ack_right"] = pair_id
        st.rerun()


@st.dialog("Select Corners — Left Image")
def left_instructions(pair_id):
    corner_instructions("left", pair_id)


@st.dialog("Select Corners — Right Image")
def right_instructions(pair_id):
    corner_instructions("right", pair_id)


INSTRUCTIONS = {"left": left_instructions, "right": right_instructions}
ACK_KEY = {"left": "m7_ack", "right": "m7_ack_right"}
OPENED_KEY = {"left": "m7_dialog_opened", "right": "m7_dialog_right"}


# ---------------------------------------------------------------- corner selection

def to_preview_points(points, preview, full_size):
    """Original-frame points -> positions in the preview, using the preview's actual size."""
    shown = (preview.shape[1], preview.shape[0])
    return [sel.source_to_shown(p, (0, 0), full_size, shown) for p in points]


COMPACT_WIDTH = 360     # CSS pixels per image during corner selection; same for the active and the inactive image
COMPACT_CSS = """<style>
/* Never give the click component's iframe `height: auto`: an iframe has no intrinsic height, so that collapses it to the 150 px
   default and clips the photograph. Its height is set by the component from the image it renders. */
.st-key-m7_compact_pair iframe { max-width: 100% !important; }
.st-key-m7_compact_pair img { max-width: 100% !important; height: auto; }
.st-key-m7_compact_pair [data-testid="stImage"] { width: auto !important; }
</style>"""


def compact_height(preview, width=COMPACT_WIDTH):
    """Whole-pixel display height that keeps the preview's aspect ratio at the given width. The click component sizes its frame
    from an integer image height (st.image scales with CSS instead), so the rounding is done here once, explicitly."""
    return max(1, round(width * preview.shape[0] / preview.shape[1]))


def compact_pair(images, active=None):
    """Both photographs side by side at one fixed compact size (each fills half the page, aspect ratio kept, the same in every
    stage). Only the active image takes clicks; it is returned as the component's click dict, else None. Display only: a click is
    mapped by the rendered size reported with it, so corners stay in original-image coordinates."""
    ss = st.session_state
    click = None
    st.markdown(COMPACT_CSS, unsafe_allow_html=True)
    with st.container(key="m7_compact_pair"):
        for column, side in zip(st.columns(2), SIDES):
            with column:
                preview = images[side]["preview"]
                pts = ss["m7_points"][side]
                drawn = sel.draw_points(preview, to_preview_points(pts, preview, images[side]["size"]), close=True)
                st.caption(f"{SIDE_LABEL[side]} image: {len(pts)}/4 corners")
                if side == active:
                    click = streamlit_image_coordinates(
                        drawn, width=COMPACT_WIDTH, height=compact_height(preview), cursor="crosshair", image_format="JPEG", jpeg_quality=90,
                        key=f"m7_coarse_{side}_{ss['m7_gen']}",
                    )
                else:
                    st.image(drawn, width=COMPACT_WIDTH)
    return click


def zoom_token(side):
    ss = st.session_state
    return f"{side}-{ss['m7_gen']}-{len(ss['m7_points'][side])}"


def zoom_key(side):
    return f"m7_zoom_{side}_{st.session_state['m7_gen']}"


def accept_corner(side):
    """Commit the dragged position (or the clicked one if the marker was never moved) of the pending corner."""
    ss = st.session_state
    pending = ss["m7_pending"]
    if pending is None or pending[0] != side or len(ss["m7_points"][side]) >= 4:
        return
    point = ss["m7_refined"]
    value = ss.get(zoom_key(side))
    if value is not None:       # the component value is current even if the last drag's rerun has not rendered yet
        full = ss["m7_images_size"][side]
        window = sel.zoom_window(pending[1], ZOOM_HALF, full)
        fresh = cd.refined_point(value, zoom_token(side), (window[0], window[1]), (window[2], window[3]), full)
        point = fresh if fresh is not None else point
    ss["m7_points"][side].append(point if point is not None else pending[1])
    ss["m7_pending"] = ss["m7_refined"] = None
    ss["m7_gen"] += 1
    invalidate_confirmation(side)


def cancel_corner():
    """Discard the pending corner and every unaccepted drag; accepted corners are untouched."""
    ss = st.session_state
    ss["m7_pending"] = ss["m7_refined"] = None
    ss["m7_gen"] += 1


def corner_selection(images):
    ss = st.session_state
    points = ss["m7_points"]
    counts = {side: len(points[side]) for side in SIDES}
    st.write(f"**Progress:** left image {counts['left']}/4 · right image {counts['right']}/4")

    cols = st.columns(4)
    cols[0].button("Undo last point", on_click=undo_point, disabled=not (ss["m7_pending"] or counts["left"] or counts["right"]))
    cols[1].button("Reset left", on_click=reset_side, args=("left",), disabled=not counts["left"])
    cols[2].button("Reset right", on_click=reset_side, args=("right",), disabled=not counts["right"])
    cols[3].button("Reset both", on_click=reset_both, disabled=not (counts["left"] or counts["right"] or ss["m7_pending"]))

    active = "left" if not ss["m7_left_confirmed"] else "right" if not ss["m7_right_confirmed"] else None
    ss["m7_stage"] = active or "ready"
    if active is None:
        compact_pair(images)
        return True
    if counts[active] == 4 and ss["m7_pending"] is None:
        compact_pair(images)
        st.success(f"All four {active} image corners are accepted. Review the markers above, then confirm.")
        st.button(f"Confirm {SIDE_LABEL[active]} Image Selections", key=f"m7_confirm_{active}", type="primary",
                  on_click=confirm_side, args=(active,))
        return False
    st.markdown(ZOOM_BUTTON_CSS, unsafe_allow_html=True)

    full = images[active]["full"]
    full_size = (full.shape[1], full.shape[0])
    ss["m7_images_size"] = {side: (images[side]["full"].shape[1], images[side]["full"].shape[0]) for side in SIDES}
    k = counts[active]
    pending = ss["m7_pending"]

    if pending is None or pending[0] != active:
        st.info(f"**{SIDE_LABEL[active]} image:** click corner {k + 1} of 4 - **{CORNER_NAMES[k]}**.")
        click = compact_pair(images, active)
        if click is not None:
            try:
                point = sel.click_to_source(click, (0, 0), full_size)
            except (KeyError, ValueError, TypeError):
                st.warning("Could not read that click. Please click the image again.")
            else:
                ss["m7_pending"] = (active, point)
                ss["m7_refined"] = None
                ss["m7_gen"] += 1
                st.rerun()
        return False

    # Zoom refinement of the pending coarse point: the marker is dragged; nothing is committed until Accept Corner
    coarse = pending[1]
    window = sel.zoom_window(coarse, ZOOM_HALF, full_size)
    zoom, crop_size = sel.make_zoom(full, window, 640)
    zoom_shown = (zoom.shape[1], zoom.shape[0])
    origin = (window[0], window[1])
    inside = [(i, p) for i, p in enumerate(points[active])
              if window[0] <= p[0] < window[0] + window[2] and window[1] <= p[1] < window[1] + window[3]]
    drawn = sel.draw_points(zoom, [sel.source_to_shown(p, origin, crop_size, zoom_shown) for _, p in inside],
                            radius=8, ids=[i for i, _ in inside], lines=False)
    st.info(f"**Zoom refinement, {SIDE_LABEL[active]} image, corner {k + 1} - {CORNER_NAMES[k]}.** "
            "Drag the yellow marker onto the exact corner, then press **Accept Corner** (or **Cancel**).")
    token = zoom_token(active)
    value = cd.corner_drag(drawn, cd.source_to_fraction(ss["m7_refined"] or coarse, origin, crop_size),
                           cd.fraction_limits(origin, crop_size, full_size), token, zoom_key(active))
    dragged = cd.refined_point(value, token, origin, crop_size, full_size)
    if dragged is not None:
        ss["m7_refined"] = dragged
    with st.container(key="m7_zoom_actions"):
        accept_col, cancel_col = st.columns(2)
        accept_col.button("Accept Corner", key="m7_zoom_accept", on_click=accept_corner, args=(active,), width="stretch")
        cancel_col.button("Cancel", key="m7_zoom_cancel", on_click=cancel_corner, width="stretch")
    return False


# ---------------------------------------------------------------- results

def kv_table(index_name, rows, columns):
    return st.table(pd.DataFrame(rows, columns=[index_name] + columns).set_index(index_name))


def show_results(result, images, truth):
    st.subheader("Results")
    for warning in result["warnings"]:
        st.warning(warning)
    st.write("**Annotated stereo images**")
    for column, side in zip(st.columns(2), SIDES):
        with column:
            large = images[side]["large"]
            pts = sel.draw_points(large, [sel.source_to_shown(p, (0, 0), images[side]["size"], (large.shape[1], large.shape[0]))
                                          for p in result[f"{side}_px"]], close=True, radius=8, names=True)
            st.image(pts, caption=f"{SIDE_LABEL[side]} image with selected corners", width="stretch")

    if result["errors"]:
        for error in result["errors"]:
            st.error(error)
        st.error("Dimensions are not shown because the reconstructed geometry is invalid. "
                 "Reset the corners and select them again.")
        return

    s = result["summary"]
    st.write("**Measurement summary**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Z-depth", f"{s['mean_Z_depth_mm']:.1f} mm")
    c2.metric("Estimated width", f"{s['width_mm']:.1f} mm")
    c3.metric("Estimated height", f"{s['height_mm']:.1f} mm")
    st.caption("Z-depth is each corner's coordinate along the left camera's optical axis. It is not the straight-line "
               "distance from the camera centre, which is √(X² + Y² + Z²) and is listed separately below.")

    st.write("**Detailed measurements**")
    names = {"top_width": "Top width (TL-TR)", "bottom_width": "Bottom width (BL-BR)",
             "left_height": "Left height (TL-BL)", "right_height": "Right height (TR-BR)"}
    edge_rows = {e["edge"]: e for e in result["edges"]}
    kv_table("Edge", [[names[k], f"{edge_rows[k]['length_3D_mm']:.2f}", f"{edge_rows[k]['dX_mm']:.2f}",
                       f"{edge_rows[k]['dY_mm']:.2f}", f"{edge_rows[k]['dZ_mm']:.2f}"] for k in names],
             ["3D length (mm)", "dX (mm)", "dY (mm)", "dZ (mm)"])

    st.write("**Reconstructed 3D coordinates** (left-camera frame, mm; X right, Y down, Z forward)")
    kv_table("Corner", [[c["corner"], f"{c['X_mm']:.2f}", f"{c['Y_mm']:.2f}", f"{c['Z_depth_mm']:.2f}",
                         f"{c['dist_from_left_camera_mm']:.2f}"] for c in result["corners"]],
             ["X (mm)", "Y (mm)", "Z-depth (mm)", "Distance from left camera centre (mm)"])

    if any(v is not None for v in truth):
        st.write("**Comparison with the actual size**")
        rows = []
        for label, key, actual in (("Width", "width", truth[0]), ("Height", "height", truth[1])):
            if actual is not None and actual > 0:
                rows.append([label, f"{actual:.2f}", f"{s[key + '_mm']:.2f}", f"{s[key + '_err_mm']:+.2f}",
                             f"{s[key + '_abs_err_mm']:.2f}", f"{s[key + '_err_pct']:+.2f}%"])
        kv_table("Dimension", rows, ["Actual (mm)", "Estimated (mm)", "Error (mm, estimate - actual)",
                                     "Absolute error (mm)", "Percentage error"])
    else:
        st.caption("No actual dimensions were entered, so no measurement error is reported.")

    with st.expander("How stereo triangulation estimates depth and size"):
        st.write(
            "Each selected pixel is first corrected for its own camera's lens distortion and converted into a viewing ray "
            "using that camera's intrinsic matrix. A corner seen in both images defines two rays; because the right camera "
            "is placed relative to the left by the calibrated rotation R and translation T, the two rays intersect at the "
            "corner's 3D position (X, Y, Z) in the left-camera frame. Z is an output of this intersection: no distance is "
            "assumed and nothing is rescaled to a known size.\n\n"
            "The 3D Euclidean length of each edge is √(dX² + dY² + dZ²). The estimated width is the mean of the top and "
            "bottom edges and the estimated height is the mean of the left and right edges. Z-depth is measured along the "
            "left camera's optical axis, whereas the distance to the camera centre is √(X² + Y² + Z²)."
        )

    left, right = st.columns(2)
    left.button("Measure Another Image", key="m7_another_image", on_click=measure_another_image, width="stretch",
                help="Clears these photographs, corners and results. The active calibration is kept.")
    right.button("Reset All", key="m7_results_reset_all", on_click=reset_all, width="stretch")


def part_one():
    ss = st.session_state
    restore_inputs(experiment=False, measurement=True)
    st.header("Object Size Estimation Using Calibrated Stereo")
    st.write(
        "Estimate the distance (Z-depth) of an object and calculate its real-world width and height "
        "using calibrated stereo camera geometry."
    )
    st.markdown(PART_ONE_CSS, unsafe_allow_html=True)
    with st.container(key="m7_calibrate"), st.expander(CALIBRATE_EXPANDER_LABEL):
        calibrate_workflow()

    model = ss["m7_visitor_model"]              # read after the workflow, which may have just set or cleared it
    ss["m7_measurement_enabled"] = model is not None
    uploads = upload_section(model) if model is not None else upload_placeholder()
    if uploads is not None:
        pair_id = hashlib.sha256((uploads["left"]["digest"] + uploads["right"]["digest"]).encode()).hexdigest()[:16]
        if ss["m7_pair_id"] != pair_id:
            reset_selection()
            ss["m7_pair_id"] = pair_id
        try:
            with st.spinner("Preparing images..."):
                images = {side: prepare_image(uploads[side]["digest"], uploads[side]["data"], uploads[side]["size"]) for side in SIDES}
        except (sel.UploadError, OSError, ValueError) as error:
            st.error(f"The images could not be decoded: {error}")
            images = None

        if images is not None:
            stage = "left" if not ss["m7_left_confirmed"] else "right" if not ss["m7_right_confirmed"] else "ready"
            ss["m7_stage"] = stage
            if stage != "ready" and ss[ACK_KEY[stage]] != pair_id:
                st.subheader(f"Step {1 if stage == 'left' else 2}: Select {SIDE_LABEL[stage]} Image Corners")
                if ss[OPENED_KEY[stage]] != pair_id:
                    ss[OPENED_KEY[stage]] = pair_id
                    INSTRUCTIONS[stage](pair_id)
                st.info(f"Read the {stage}-image corner-selection instructions to begin.")
                if st.button("Show corner-selection instructions"):
                    INSTRUCTIONS[stage](pair_id)
            else:
                if stage != "ready":
                    st.subheader(f"Step {1 if stage == 'left' else 2}: Select {SIDE_LABEL[stage]} Image Corners")
                    if st.button("Show instructions again"):
                        INSTRUCTIONS[stage](pair_id)
                else:
                    st.subheader("Corner Selections")
                complete = corner_selection(images)

                st.subheader("Actual Size (optional)")
                st.caption("Enter the object's real dimensions to compare with the estimate. Leave blank to see the estimate only.")
                c1, c2 = st.columns(2)
                actual_width = c1.number_input("Actual Width (mm)", min_value=0.0, value=None, step=1.0, format="%.2f", key=f"m7_gt_w_{ss['m7_upload_gen']}")
                actual_height = c2.number_input("Actual Height (mm)", min_value=0.0, value=None, step=1.0, format="%.2f", key=f"m7_gt_h_{ss['m7_upload_gen']}")
                truth = (actual_width or None, actual_height or None)

                if complete:
                    st.success("Left and right image selections are confirmed.")
                    st.button("Calculate Dimensions", type="primary", on_click=calculate_now)
                else:
                    st.caption("Accept and confirm all four corners on the left image, then on the right image, to enable measurement.")
                if complete and ss["m7_calculated"]:
                    result = measure_points(model, np.array(ss["m7_points"]["left"]), np.array(ss["m7_points"]["right"]), truth)
                    show_results(result, images, truth)



# ---------------------------------------------------------------- saved notebook experiment

def experiment_overlay(pair_id, side, selection):
    """Final approved corners drawn on the optimized display copy, scaled by the copies' actual size."""
    path = DISPLAY_DIR / f"pair_{pair_id}_{side}.jpg"
    if not path.is_file():
        return None
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    src_size = tuple(selection["image_size_wh"])
    shown = (image.shape[1], image.shape[0])
    pts = [sel.source_to_shown(p, (0, 0), src_size, shown) for p in selection[f"{side}_px"]]
    return sel.draw_points(image, pts, close=True, radius=9, font_scale=0.9, names=True)


def show_experiment():
    """The "experiment" view (My Experiments & Results > Question 1); the top navigation Back arrow is its only Back."""
    st.header("My Experiment & Results", anchor="m7-experiment-heading")
    try:
        saved = load_saved_results()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Saved experiment results are unavailable: {error}")
        return
    truth = saved["summary"]["ground_truth_mm"]
    st.markdown(Q1_INTRO_CSS, unsafe_allow_html=True)
    with st.container(key="m7_q1_objective"):
        st.markdown("##### Objective")
        st.write("Estimate the real-world width and height of a notebook using calibrated stereo images and classical computer vision techniques.")
        st.markdown("##### Experimental setup")
        st.write(f"Two phones were used as a stereo pair: an {LEFT_CAMERA} as the left camera and an {RIGHT_CAMERA} as the right camera, with a "
                 f"measured baseline of {MEASURED_BASELINE_MM} mm. The pair was calibrated with {len(get_model()['pairs'])} matched checkerboard image pairs "
                 f"({PATTERN[0]} × {PATTERN[1]} inner corners, {SQUARE_MM:g} mm squares); this calibration is called Model B. A notebook measuring "
                 f"{truth['width']:g} × {truth['height']:g} mm, with the red spine included in the width, was then photographed in three stereo image pairs, "
                 "and its width and height were estimated from each pair. Only classical computer vision is used.")

    show_calibration_reference(get_model())

    st.write("The corners shown below are the final approved selections, and the measurements are read from the saved pipeline outputs.")

    st.markdown(SELECTED_PAIR_CSS, unsafe_allow_html=True)
    with st.container(key="m7_selected_pair_results"):
        st.subheader("Notebook Stereo Pair", anchor=False)
        pair_id = int(st.radio("Notebook stereo pair", NOTEBOOK_PAIR_IDS, format_func=lambda i: f"Pair {i}", horizontal=True,
                               key="m7_exp_pair", label_visibility="collapsed"))
        pairs, corners, edges = saved["pairs"], saved["corners"], saved["edges"]
        row = pairs[pairs["pair"] == pair_id].iloc[0]
        try:
            selection = json.loads((Q1_DIR / "selections" / f"pair_{pair_id}_corners.json").read_text())
            assert selection["confirmed"] is True and selection["corner_order"] == list(CORNER_NAMES)
        except (OSError, ValueError, AssertionError, KeyError) as error:
            st.error(f"Final corner selection for pair {pair_id} is unavailable: {error}")
            return
        st.markdown(f"#### Pair {pair_id}: Photographs with the Final Approved Corners")
        for column, side in zip(st.columns(2), SIDES):
            with column:
                overlay = experiment_overlay(pair_id, side, selection)
                if overlay is None:
                    st.warning(f"Display image for pair {pair_id} ({side}) is unavailable.")
                else:
                    st.image(overlay, caption=f"{SIDE_LABEL[side]} photograph ({LEFT_CAMERA if side == 'left' else RIGHT_CAMERA})",
                             width="stretch")
        st.caption("Numbered markers (1 TL, 2 TR, 3 BR, 4 BL) with the boundary connecting the corners, drawn from the final "
                   "approved selection on an optimized display copy of each original photograph.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Z-depth", f"{row['mean_Z_depth_mm']:.2f} mm")
        c2.metric("Width", f"{row['width_mm']:.2f} mm")
        c3.metric("Height", f"{row['height_mm']:.2f} mm")

        st.write("**Reconstructed corners** (left-camera frame, mm)")
        pc = corners[corners["pair"] == pair_id]
        kv_table("Corner", [[r.corner, f"{r.X_mm:.2f}", f"{r.Y_mm:.2f}", f"{r.Z_depth_mm:.2f}", f"{r.dist_from_left_camera_mm:.2f}"]
                            for r in pc.itertuples()], ["X (mm)", "Y (mm)", "Z-depth (mm)", "Distance from camera centre (mm)"])
        st.write("**Edge lengths** (3D Euclidean)")
        pe = edges[edges["pair"] == pair_id]
        kv_table("Edge", [[f"{r.edge.replace('_', ' ').title()} ({r.endpoints})", f"{r.length_3D_mm:.2f}", f"{r.truth_mm:.0f}"]
                          for r in pe.itertuples()], ["Length (mm)", "Ground truth (mm)"])
        st.write("**Final dimensions versus ground truth**")
        kv_table("Dimension", [
            ["Width", f"{truth['width']:g}", f"{row['width_mm']:.2f}", f"{row['width_err_mm']:+.2f}", f"{abs(row['width_err_mm']):.2f}",
             f"{row['width_err_pct']:+.2f}%"],
            ["Height", f"{truth['height']:g}", f"{row['height_mm']:.2f}", f"{row['height_err_mm']:+.2f}", f"{abs(row['height_err_mm']):.2f}",
             f"{row['height_err_pct']:+.2f}%"],
        ], ["Ground truth (mm)", "Measured (mm)", "Error (mm, measured - truth)", "Absolute error (mm)", "Percentage error"])

        figure = OUTPUT_DIR / "visualizations" / f"pair_{pair_id}_measurement.jpg"
        if figure.is_file():
            st.image(str(figure), caption=f"Pair {pair_id}: final annotated stereo pair (corner correspondences linked)", width="stretch")

    st.subheader("Summary of all three pairs")
    kv_table("Pair", [[f"Pair {int(r.pair)}", f"{r.mean_Z_depth_mm:.4f}", f"{r.width_mm:.4f}", f"{r.height_mm:.4f}",
                       f"{r.width_err_mm:+.2f} ({r.width_err_pct:+.2f}%)", f"{r.height_err_mm:+.2f} ({r.height_err_pct:+.2f}%)"]
                      for r in pairs.itertuples()],
             ["Mean Z-depth (mm)", "Width (mm)", "Height (mm)", "Width error mm (%)", "Height error mm (%)"])
    for name, caption in (("depth_topdown.png", "Top-down view of the reconstructed corners (Z forward, X right)"),
                          ("error_vs_ground_truth.png", "Measured width and height against ground truth")):
        figure = OUTPUT_DIR / "visualizations" / name
        if figure.is_file():
            st.image(str(figure), caption=caption, width="stretch")

    model = get_model()
    abs_pct = pd.concat([pairs["width_err_pct"].abs(), pairs["height_err_pct"].abs()])
    st.subheader("Accuracy and limitations")
    st.write(
        f"Across the three pairs the estimated sizes are within {abs_pct.min():.1f}% to {abs_pct.max():.1f}% of the ground truth "
        f"(width {pairs['width_err_pct'].abs().mean():.1f}% and height {pairs['height_err_pct'].abs().mean():.1f}% mean absolute "
        f"error). Every width and height is below the true size "
        f"({'yes' if (pairs['width_err_mm'] < 0).all() and (pairs['height_err_mm'] < 0).all() else 'not for all pairs'}), "
        "so the error is systematic rather than random. The cause was not isolated in this project."
    )
    st.markdown(
        f"- **Calibration quality:** the stereo RMS reprojection error is {model['stereo_rms']:.2f} px (single cameras "
        f"{model['rms_left']:.2f} and {model['rms_right']:.2f} px). At a baseline of about {model['baseline_mm']:.0f} mm and "
        "depths of roughly 700-850 mm, a few pixels of correspondence error translate into millimetres of depth error.\n"
        "- **Manual corner selection:** corners are picked by hand, so a few pixels of placement error in either image "
        "changes the triangulated depth and the edge lengths.\n"
        "- **Scope of validity:** results apply to the calibrated phones, lenses and settings, one object and a limited "
        "distance range. Only three pairs were measured, which is too few for statistical claims.\n"
        "- **Object definition:** the width includes the red spine, and the object is assumed to be a flat rectangle.\n"
        "- **No correction applied:** nothing was rescaled or adjusted to match the known size; the ground truth is used only "
        "to report errors."
    )


# ---------------------------------------------------------------- Part 3: Room Measurement Using Stereo Videos

ROOM_VISITOR_BUTTON = "Measure a Room Using Your Own Stereo Videos"
ROOM_RESULTS_BUTTON = "View My Experiment & Results"
ROOM_CSS = """
<style>
.st-key-m7_room_head { gap: 0; padding-bottom: 28px; }
.st-key-m7_room_head h1 {
    font-size: clamp(2.5rem, 8vw, 60px); font-weight: 800; line-height: 1.1; text-align: center; padding: 0; margin: 0;
    color: inherit; overflow-wrap: anywhere;
}
.st-key-m7_room_head h2 {
    font-size: clamp(1.4rem, 4.6vw, 2.4rem); font-weight: 600; line-height: 1.2; text-align: center; padding: 0; margin: 0;
    color: inherit; opacity: 0.85; overflow-wrap: anywhere;
}
.st-key-m7_room_title { padding-bottom: 28px; }
.st-key-m7_room_intro { padding: 4px 0 8px; }
.st-key-m7_room_intro p { text-align: center; font-size: 1.1rem; margin: 0; }
.st-key-m7_room_body h3 { text-align: center; font-size: clamp(1.4rem, 4.5vw, 1.9rem); font-weight: 700; padding-top: 1.5rem; }
.st-key-m7_room_body h5 { text-align: center; font-weight: 700; margin: 0; padding: 0.5rem 0 0.25rem; }
.st-key-m7_room_body [data-testid="stMetricValue"] { font-size: 2rem; }
.st-key-m7_room_body [data-testid="stMetric"] { text-align: left; }
.st-key-m7_room_body video { width: 100%; max-height: 72vh; object-fit: contain; background: #000; border-radius: 8px; }
.st-key-m7_room_legend { font-size: 0.9rem; }
.st-key-m7_room_body [data-testid="stExpander"] :is(h3, h4, h5), .st-key-m7_room_objective h5 { text-align: left; }
.st-key-m7_room_methodology [data-testid="stExpander"] h4 { font-size: 1.15rem; padding: 1.1rem 0 0.2rem; }
.st-key-m7_room_body h2 { text-align: center; font-size: clamp(1.7rem, 5.2vw, 2.4rem); font-weight: 800; line-height: 1.2; padding: 1.5rem 0 0.25rem; overflow-wrap: anywhere; }
/* the three experiment panels (Wall A, Wall B, Final Room Dimensions) share one look: subtle shade, thin border, rounded corners */
.st-key-m7_room_wall_a, .st-key-m7_room_wall_b, .st-key-m7_room_final {
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 16px; padding: 0.5rem 28px 28px; margin-top: 2rem;
}
/* Camera Calibration: red outline and a bolder heading, scoped to the container that holds only this expander */
.st-key-m7_room_calibration [data-testid="stExpander"] { border: 1px solid #C9363E; border-radius: 10px; }
.st-key-m7_room_calibration [data-testid="stExpander"] summary p { font-weight: 700; }
[class*="st-key-m7_room_wall_note"] p { text-align: center; opacity: 0.85; }
.st-key-m7_room_final [data-testid="stMetricValue"] { font-size: 2.6rem; }
.st-key-m7_room_final [data-testid="stMetricLabel"] p, .st-key-m7_room_final [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
/* "What the cameras saw": button-style tabs, scoped to this one container only (the panel behind them is the experiment container) */
.st-key-m7_room_seen [data-baseweb="tab-list"] { gap: 10px; flex-wrap: wrap; padding: 4px 0 12px; }
.st-key-m7_room_seen [data-baseweb="tab-highlight"], .st-key-m7_room_seen [data-baseweb="tab-border"] { display: none; }
.st-key-m7_room_seen button[data-baseweb="tab"] {
    height: auto; padding: 10px 22px; border-radius: 10px; cursor: pointer;
    background: color-mix(in srgb, currentColor 10%, transparent); border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
    transition: background 0.15s, border-color 0.15s;
}
.st-key-m7_room_seen button[data-baseweb="tab"] p { font-size: 1.05rem; font-weight: 600; margin: 0; }
.st-key-m7_room_seen button[data-baseweb="tab"]:hover { background: color-mix(in srgb, currentColor 18%, transparent); border-color: color-mix(in srgb, currentColor 35%, transparent); }
.st-key-m7_room_seen button[data-baseweb="tab"][aria-selected="true"],
.st-key-m7_room_seen button[data-baseweb="tab"][aria-selected="true"]:hover { background: #C9363E; border-color: #C9363E; color: #FFFFFF; }
.st-key-m7_room_seen button[data-baseweb="tab"][aria-selected="true"] p { color: #FFFFFF; }
.st-key-m7_room_seen button[data-baseweb="tab"]:focus-visible { outline: 2px solid #FFFFFF; outline-offset: 2px; }
@media (min-width: 641px) { .st-key-m7_room_stats [data-testid="stMetricLabel"] { min-height: 2.7em; } }
@media (max-width: 640px) {
    .st-key-m7_room_wall_a, .st-key-m7_room_wall_b, .st-key-m7_room_final { padding: 0.25rem 14px 18px; }
    .st-key-m7_room_final [data-testid="stMetricValue"] { font-size: 2.1rem; }
    .st-key-m7_room_seen button[data-baseweb="tab"] { padding: 8px 14px; flex: 1 1 auto; }
    .st-key-m7_room_seen button[data-baseweb="tab"] p { font-size: 0.95rem; }
}
</style>
"""


REPO_URL = "https://github.com/jchowdhury94/CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury"
LOCAL_GUIDE_CLONE = f"git clone {REPO_URL}.git\ncd CSC-8830-Computer-Vision-Projects-Jannati-Chowdhury"
LOCAL_GUIDE_VENV_UNIX = "python3.12 -m venv .venv\nsource .venv/bin/activate"
LOCAL_GUIDE_VENV_WINDOWS = "py -3.12 -m venv .venv\n.venv\\Scripts\\Activate.ps1"
LOCAL_GUIDE_INSTALL = "python -m pip install -r Modules/Module_7/visitor_workflow/requirements.txt"
LOCAL_GUIDE_RUN = "python run_local.py"
SETUP_BUTTON = "Setup Instructions"
SETUP_SUBTITLE = "How to run the experiment on your computer"
RUN_BUTTON = "Run Your Experiment (Local Only)"
RUN_SUBTITLE = "Measure your own stereo videos (Wall A and Wall B)"
RUN_CLOUD_NOTE = "This experiment runs only on your own computer, not on this website. Open **Setup Instructions** above to see how to run it locally."
RUN_NOT_RUNNING_NOTE = ("The room measurement application is not running at {url}. Open **Setup Instructions** above, start everything with "
                        "`python run_local.py`, then press Check again.")


def room_visitor_choice():
    """Room Measurement landing page, shared by the Cloud website and the local website.  Setup Instructions is an inline expander (collapsed
    by default).  Run Your Experiment (Local Only) is enabled only in local mode (run_local.py) while the standalone application answers on
    its loopback port; it is then a plain link without a target attribute, so the standalone application opens in THIS browser tab
    (on its Camera Calibration page).  Nothing is uploaded, decoded or measured by this website."""
    st.markdown(select_css(), unsafe_allow_html=True)
    with st.container(key="m7_sel_layout"):
        with st.container(key="m7_sel_label"):
            st.markdown("MODULE 7")
        with st.container(key="m7_sel_heading"):
            st.header(ROOM_VISITOR_BUTTON, anchor=False)
        with st.container(key="m7_sel_panels"):
            with st.container(key="m7_open_room_setup"):
                with st.expander(panel_label(SETUP_BUTTON, SETUP_SUBTITLE), expanded=False, key="m7_setup_expander"):
                    room_setup_instructions()
            note = None
            if not local_mode.is_local_mode():
                st.button(panel_label(RUN_BUTTON, RUN_SUBTITLE), key="m7_open_room_run", disabled=True, width="stretch")
                note = RUN_CLOUD_NOTE
            elif local_mode.visitor_app_running():
                with st.container(key="m7_open_room_run"):
                    st.html(card_link(local_mode.visitor_url(), RUN_BUTTON, RUN_SUBTITLE))
            else:
                st.button(panel_label(RUN_BUTTON, RUN_SUBTITLE), key="m7_open_room_run", disabled=True, width="stretch")
                note = RUN_NOT_RUNNING_NOTE.format(url=local_mode.visitor_url())
            if note:
                with st.container(key="m7_room_run_note"):
                    st.markdown(note)
                    if local_mode.is_local_mode():
                        st.button("Check again", key="m7_room_run_recheck")


def room_setup_instructions():
    """Content of the Setup Instructions expander.  It never uploads, decodes or measures anything: the visitor application is a separate
    program that runs only on the visitor's own computer."""
    st.markdown("**This feature runs on your own computer, not on this website.** To measure a room from your own stereo videos, "
                "run the visitor application locally using the steps below. Nothing can be uploaded or measured on this page.")
    st.markdown("#### Why not here?")
    st.write("The measurement works on full-resolution (4K) stereo videos. The free hosting plan used for this website does not provide "
             "enough memory to process them reliably, so uploads are not accepted here. Running the same application on your own "
             "computer has no such limit. To see my own validated room experiment, go back to **My Experiments & Results** and open **Room Measurement Using Stereo Videos**.")
    st.markdown("#### Run it locally")
    st.markdown("**1. Get the code** (requires [Git](https://git-scm.com/downloads), or download the repository as a ZIP from "
                f"[GitHub]({REPO_URL}) and open a terminal in the extracted folder):")
    st.code(LOCAL_GUIDE_CLONE, language="bash")
    st.markdown("**2. Create and activate a virtual environment** with Python 3.12 (recommended):")
    st.markdown("macOS or Linux:")
    st.code(LOCAL_GUIDE_VENV_UNIX, language="bash")
    st.markdown("Windows (PowerShell):")
    st.code(LOCAL_GUIDE_VENV_WINDOWS, language="powershell")
    st.markdown("**3. Install the dependencies:**")
    st.code(LOCAL_GUIDE_INSTALL, language="bash")
    st.markdown("**4. Start everything with one command:**")
    st.code(LOCAL_GUIDE_RUN, language="bash")
    st.write("This starts two separate Streamlit applications on your computer, both reachable only from it: this website at "
             "http://127.0.0.1:8501 and the visitor measurement application at http://127.0.0.1:8640. Open the website, go to "
             "**Module 7 > Run Your Own Experiment > Room Measurement Using Stereo Videos** and press **Run Your Experiment (Local Only)**; "
             "it opens the visitor application in this same browser tab, and its Back arrow and Home link bring you back to this website. Press Ctrl+C in the terminal to stop both.")
    st.markdown("**5. Measure a wall.** In the application, accept a stereo calibration (**Upload Saved Calibration**), upload the stereo videos "
                "of the wall, choose the frames, mark the wall boundaries, and press **Calculate Wall Dimensions**. **Wall A** (two viewpoints, four videos) "
                "gives the room width and height. After it, **Continue to Wall B \u2014 Room Length** (three viewpoints, six videos) and press "
                "**Calculate Room Length**.")
    st.markdown("#### Before you record")
    st.markdown("- Use **two cameras on a fixed rig** and the **same cameras and settings** (resolution, orientation, frame rate, zoom, focus, stabilization) "
                "as in your calibration. Do not move one camera relative to the other.\n"
                "- The videos must have **exactly the frame size and orientation of your accepted calibration**. Videos are never resized or rotated. "
                "My validated recordings are 2160 \u00d7 3840 pixels, portrait.\n"
                "- Each video: MP4, MOV, M4V, Matroska/WebM or AVI, 1 to 120 seconds, 1 to 480 fps, up to 1024 MB. "
                "H.265/HEVC videos are accepted by the file check, but decoding depends on your computer's OpenCV build.\n"
                "- Choose a wall with **visible corners and boundaries**, a floor with **visible texture**, and keep the room still while you record.")
    st.markdown("#### What to expect")
    st.write("These are the application's checks and my recommendations, not a promise of accuracy. I validated the method on one room with one pair of "
             "phones, and arbitrary videos may give inaccurate results or be refused with an explanation. Processing of 4K videos can take "
             "a long time and a lot of memory on your computer.")
    st.markdown("#### Memory and temporary files")
    st.write("Nothing has to be configured. The application uses your computer's memory as its budget, keeps part of it in reserve and checks at every start that enough is free right now. "
             "Keep at least 4 GB of memory free (the Wall A calculation used about 2.0 GB at its peak on my recordings) and close other large programs first. "
             "The extracted frames are kept in a marked temporary folder while a measurement is open and are removed when you press New Measurement, when your session ends, "
             "when the application stops, or at the next start after a crash.")
    st.markdown("#### Troubleshooting")
    st.markdown("- **Command not found:** install Python 3.12 and check that the virtual environment is activated.\n"
                "- **A package fails to install:** upgrade pip (`python -m pip install --upgrade pip`) and retry.\n"
                "- **Port already in use:** `run_local.py` names the busy port and starts nothing; close the program using it, or choose other ports, for example `python run_local.py --website-port 8502 --visitor-port 8641`.\n"
                "- **Run Your Experiment (Local Only) is greyed out:** you are on the public website, or the visitor application is not running; start it with `python run_local.py` and press **Check again**.\n"
                "- **A video is rejected:** read the message; it names the check that failed (size, length, frame rate, or a different size or orientation than the calibration).\n"
                "- **Not enough memory or a stopped calculation:** close other large programs so that at least 4 GB is free and press Try again; the application names the limit that was reached.")
    st.markdown(f"Source code and full instructions: [GitHub repository]({REPO_URL}).")


def _dimension_card(title, d):
    cm, signed = room_data.cm, room_data.signed_cm
    low, high = d["interval_95_mm"]
    with st.container(border=True):
        st.markdown(f"#### {title}")
        left, right = st.columns(2)
        left.metric("Estimated dimension", f"{cm(d['estimate_mm'])} cm")
        right.metric("Actual dimension (tape measurement)", f"{cm(d['actual_mm'])} cm")
        st.write(f"**Signed error:** {signed(d['error_mm'])} cm")
        st.write(f"**Absolute error:** {abs(d['abs_error_mm']) / 10:.1f} cm")
        st.write(f"**Absolute percentage error:** {abs(d['error_percent']):.2f}%")
        st.write(f"**Approximate 95% uncertainty interval:** {cm(low)} to {cm(high)} cm")


def _room_images():
    status = room_data.image_status()
    tabs = st.tabs(["Stereo views", "Boundaries and corners", "Overlapping views"])
    for tab, name in zip(tabs, ("stereo", "corners", "overlap")):
        with tab:
            for path, group, caption, exists in status:
                if group != name:
                    continue
                if exists:
                    st.image(str(path), caption=caption, width="stretch")
                else:
                    st.warning(f"Visualization not available: {path.name} is missing from the committed outputs.")
            if name == "corners":
                with st.container(key="m7_room_legend"):
                    st.markdown(":cyan[**Cyan dots**] directly observed edge points (found automatically near manually placed "
                                "seeds) &nbsp;|&nbsp; :orange[**Yellow lines**] fitted wall-boundary lines &nbsp;|&nbsp; "
                                ":red[**Red diamonds (*)**] corners inferred geometrically, never seen directly")
                st.caption("Upper corners are inferred from the right (Pro) camera only. One floor corner is partly hidden, so "
                           "bottom corners come from where the reconstructed vertical edge meets the floor plane.")


def _viewpoint_counts():
    a, b = len(room_data.VIDEO_GROUPS), len(room_data.WALL_B_GROUPS)
    return a, b, a + b


def _objective_and_setup(results):
    a, b, total = _viewpoint_counts()
    word = lambda n: _WORDS.get(n, n)
    st.markdown("##### Objective")
    st.write("Estimate the width, length and height of a room using calibrated stereo video recordings and classical computer vision techniques.")
    st.markdown("##### Experimental setup")
    st.write(f"Two phones were used as a stereo pair: an iPhone Air as the left camera and an iPhone 17 Pro Max as the right camera, with a measured "
             f"baseline of {results['pose']['baseline_mm']:.0f} mm. The room was recorded from {word(total)} viewpoints in total, giving "
             f"{word(2 * total)} video recordings that form {word(total)} stereo pairs. Two measurement stages use them: Wall A, from {word(a)} "
             f"viewpoints, estimates the width and height, and Wall B, from {word(b)} viewpoints, estimates the length. Only classical computer "
             "vision is used.")


def _methodology(results, wb):
    """The approved eleven-step methodology. Every number comes from the saved results; equations use Streamlit's LaTeX support."""
    pose, a = results["pose"], results["assumptions"]
    sets_a = _WORDS.get(results["n_frame_sets"], results["n_frame_sets"]).capitalize()
    if wb:
        frames = wb["per_frame"]
        frame_text = f"Frames {', '.join(str(f['frame']) for f in frames[:-1])} and {frames[-1]['frame']}"
        spread = (max(f["length_mm"] for f in frames) - min(f["length_mm"] for f in frames)) / 10
        consistency = f"The measurements differ by approximately {spread:.1f} cm, showing consistency across the selected video frames."
    else:
        frame_text, consistency = "Three frame indices", "The frame sets show the consistency of the measurement across the selected video frames."
    step = lambda n, title: st.markdown(f"#### {n}. {title}")

    step(1, "Stereo Video Acquisition and Frame Selection")
    st.markdown("Individual image frames are extracted from the stereo video recordings. Each frame set contains a left and right image from every viewpoint.\n\n"
                f"- **Wall A:** {sets_a} frame sets are selected from different parts of the videos. The Laplacian variance method is used to select sharp "
                "images and reduce the effect of motion blur.\n"
                f"- **Wall B:** {frame_text} are extracted from all six videos, creating {_WORDS.get(len(wb['per_frame']), 'three') if wb else 'three'} frame sets for measurement.")

    step(2, "Camera Model and Calibration")
    st.markdown("Both cameras are calibrated using checkerboard videos to determine their focal lengths, principal points and lens distortion parameters.\n\n"
                "The relationship between the two cameras is represented by a rotation matrix $R$ and a translation vector $T$. The measured distance between "
                f"the cameras, known as the baseline, is {pose['baseline_mm']:.0f} mm.\n\nThe perspective projection equation is:")
    st.latex(r"s\,\tilde{x} = K\,(R X + T)")
    st.markdown("where:\n\n- $K$: camera intrinsic matrix\n- $R$: rotation matrix\n- $T$: translation vector\n- $X$: 3D point in the left camera coordinate system\n"
                "- $\\tilde{x}$: projected image point in the right camera\n- $s$: scale factor")

    step(3, "SIFT Feature Detection and Matching")
    st.markdown("The Scale-Invariant Feature Transform (SIFT) algorithm detects distinctive features in the extracted frames and matches corresponding points "
                "between images.\n\nLowe's ratio test and mutual matching are used to remove unreliable feature matches.\n\n"
                "- **Wall A:** A ratio threshold of 0.75 is used between stereo images and 0.85 between viewpoints. A total of "
                f"{pose['n_tracks']} feature tracks are established across four images.\n"
                "- **Wall B:** A ratio threshold of 0.75 is used between stereo images and 0.90 between neighboring viewpoints.")

    step(4, "Epipolar Geometry and Stereo Pose Estimation")
    st.markdown("Epipolar geometry describes the relationship between corresponding points in the left and right camera images. It is used to estimate the "
                "relative position and orientation of the cameras.\n\nThe essential matrix $E$ represents this relationship:")
    st.latex(r"\tilde{x}_R^{T} E\,\tilde{x}_L = 0")
    st.latex(r"E = [T]_{\times} R")
    st.markdown("where $\\tilde{x}_L$ and $\\tilde{x}_R$ are normalized image coordinates, and $[T]_{\\times}$ is the cross-product matrix of the translation vector.\n\n"
                "The eight-point algorithm provides an initial estimate. Sampson distance and Tukey biweight optimization are then used to improve the estimate "
                "and reduce the influence of incorrect matches. The baseline length remains fixed at its measured value.\n\n"
                "**Wall A:** A shared stereo camera pose is estimated and jointly refined using stereo matches and four-view feature tracks.")
    st.table(pd.DataFrame({"Pose refinement metric": ["Stereo feature matches", "Four-view feature tracks", "Robust residual scale", "Inlier percentage"],
                           "Result": [f"{pose['n_epipolar']:,}", f"{pose['n_tracks']:,}", f"{pose['robust_scale_px']:.2f} px", f"{pose['inlier_fraction'] * 100:.1f}%"]}
                          ).set_index("Pose refinement metric"))
    st.markdown("**Wall B:** A separate stereo camera pose is estimated for each of the three viewpoints.")

    step(5, "3D Triangulation")
    st.markdown("Corresponding feature points from the left and right images are reconstructed into 3D coordinates using the Direct Linear Transform (DLT) "
                "triangulation method.\n\nTo remove unreliable points, only points with depths between 0.8 m and 20 m and reprojection and epipolar errors below "
                "1.5 pixels are retained.\n\nThese 3D points are used to estimate the floor plane and align different viewpoints. They are not directly used as "
                "wall corners.")

    step(6, "Floor-Plane Estimation")
    st.markdown("The reconstructed 3D points near the bottom of each image are used to estimate the floor surface.\n\nPrincipal Component Analysis (PCA) provides "
                "an initial plane estimate, which is refined using Iteratively Reweighted Least Squares (IRLS) to reduce the influence of outliers.\n\n"
                "The floor plane is represented by:")
    st.latex(r"n^{T}X + d = 0, \qquad \|n\| = 1")
    st.markdown("where $n$ is the unit normal vector of the floor plane, $X$ is a 3D point, and $d$ is the plane offset (the distance of the camera from the floor).")

    step(7, "Wall-Boundary Annotation and Line Fitting")
    st.markdown("The visible wall boundaries are marked to identify the edges needed for room measurement.\n\n"
                "- **Wall A:** I manually marked approximate vertical wall edges and ceiling edges. Image gradients and robust line fitting are then used to "
                "automatically refine these boundaries with sub-pixel precision.\n"
                "- **Wall B:** I manually marked two points along each required vertical wall edge. These points directly define the boundary lines. The middle "
                "viewpoint is used for registration and does not require boundary annotations.")

    step(8, "Geometric Corner Estimation")
    st.markdown("The wall corners are estimated using geometric relationships between the detected wall boundaries and the reconstructed floor plane.\n\n"
                "Each image boundary line is projected back into 3D space to form a plane. The intersection of the left and right camera planes determines the "
                "vertical wall edge. The intersection of this edge with the floor plane determines the bottom corner.")
    st.latex(r"n_L = K_L^{T} l_L")
    st.latex(r"n_R = R^{T} K_R^{T} l_R")
    st.markdown("The bottom corner $B$ is obtained by solving:")
    st.latex(r"\begin{bmatrix} n_L^{T} \\ n_R^{T} \\ n^{T} \end{bmatrix} B = \begin{bmatrix} 0 \\ n_R^{T} C_R \\ -d \end{bmatrix}")
    st.markdown("where:\n\n- $K_L$ and $K_R$ are the left and right camera intrinsic matrices.\n- $l_L$ and $l_R$ are the image boundary lines.\n"
                "- $n_L$ and $n_R$ are the corresponding plane normals.\n- $C_R$ is the right camera center in the left camera coordinate system.\n"
                "- $B$ is the estimated bottom corner.\n\n"
                "**Wall A:** Both bottom and top corners are estimated to determine width and height. The top corner is estimated using the viewing ray through "
                "the ceiling intersection and the reconstructed vertical wall edge.\n\n"
                "**Wall B:** Only the two bottom end corners are required to determine length.\n\n"
                "This approach allows partially hidden wall corners to be estimated without directly observing them in both cameras.")

    step(9, "Multi-View Registration Using RANSAC")
    st.markdown("Because the videos are recorded from different viewpoints, their reconstructed 3D points initially belong to different coordinate systems.\n\n"
                "Random Sample Consensus (RANSAC) identifies reliable matching 3D points. Singular Value Decomposition (SVD) is then used to calculate the rotation "
                "and translation needed to align the viewpoints.")
    st.latex(r"(R_0, t_0) = \arg\min_{R,\,t} \sum_i \left\| Q_i - (R P_i + t) \right\|^2")
    st.markdown("where $P_i$ and $Q_i$ are corresponding 3D points, $R_0$ is the estimated rotation, and $t_0$ is the estimated translation.\n\n"
                "The RANSAC algorithm uses 1,500 iterations, three-point samples and a 40 mm inlier threshold.\n\n"
                "- **Wall A:** Viewpoint 1 is aligned with Viewpoint 2.\n- **Wall B:** Viewpoints 1 and 3 are aligned with Viewpoint 2.")

    step(10, "Room Dimension Estimation")
    st.markdown("Once the wall corners are represented in a common coordinate system, the room dimensions are calculated using Euclidean distance.\n\n"
                "**Wall A \u2014 Width and Height**\n\nThe width is the distance between the two registered bottom corners:")
    st.latex(r"W = \left\| B_2 - (R_0 B_1 + t_0) \right\|")
    st.markdown("The height is the average of the two estimated corner heights:")
    st.latex(r"H = \frac{H_1 + H_2}{2}")
    st.markdown("Here, $B_1$ and $B_2$ are the bottom corners, and $H_1$ and $H_2$ are the estimated heights.\n\n**Wall B \u2014 Length**\n\n"
                "The length is the distance between the two end corners after aligning them with the middle viewpoint:")
    st.latex(r"L = \left\| (R_{32} B_3 + t_{32}) - (R_{12} B_1 + t_{12}) \right\|")
    st.markdown("Here, $B_1$ and $B_3$ are the end corners, and the rotation and translation parameters transform them into Viewpoint 2's coordinate system.\n\n"
                "The final dimensions are obtained by averaging the measurements across the successful frame sets.")

    step(11, "Measurement Uncertainty and Consistency")
    st.markdown("Several factors can affect the estimated room dimensions, including frame selection, feature matching, camera calibration and wall-boundary "
                "placement.\n\n**Wall A \u2014 Uncertainty Analysis**\n\nFour sources of uncertainty are evaluated:")
    comp = results["components"]
    st.table(pd.DataFrame({"Source": ["Frame selection and feature matching", "Camera pose estimation method", "Wall-boundary placement",
                                      "Camera intrinsic parameters (assumed)"],
                           "Width (cm)": [f"{comp[k]['width_sd_mm'] / 10:.1f}" for k in ("frame_set_and_matching", "method_choice_A_vs_J", "boundary_observations", "camera_intrinsics")],
                           "Height (cm)": [f"{comp[k]['height_sd_mm'] / 10:.1f}" for k in ("frame_set_and_matching", "method_choice_A_vs_J", "boundary_observations", "camera_intrinsics")]}
                          ).set_index("Source"))
    st.markdown("Each value represents one standard deviation.\n\nThe individual standard deviations are combined to estimate the overall uncertainty:")
    st.latex(r"\sigma = \sqrt{\sum_k \sigma_k^{2}}")
    st.markdown("The approximate 95% uncertainty interval is calculated as:")
    st.latex(r"\text{Interval} = \text{Estimate} \pm 1.96\,\sigma")
    st.markdown("where $\\sigma_k$ represents the standard deviation of each uncertainty source and $\\sigma$ is the combined standard deviation.\n\n"
                "Camera calibration contributes the largest uncertainty under the assumed parameter variations. Systematic annotation errors, additional "
                "lens-model errors and camera movement between unsynchronized frames are not included.\n\n"
                f"**Wall B \u2014 Measurement Consistency**\n\nThe room length is calculated using three different frame sets. {consistency}\n\n"
                "Since the same boundary annotations are used for all three frame sets, this variation represents frame-to-frame consistency rather than overall "
                "measurement uncertainty. No confidence interval is calculated for Wall B.")


def _matrix_frame(matrix):
    return pd.DataFrame([[f"{v:.4f}" for v in row] for row in matrix], index=["row 1", "row 2", "row 3"], columns=["col 1", "col 2", "col 3"])


def _room_calibration():
    """Collapsed by default: what calibration is, and the stored Recipe A numbers for both phones."""
    with st.expander("Camera Calibration"):
        try:
            cal = room_data.load_calibration()
        except room_data.ResultsUnavailable as error:
            st.error(f"Camera calibration details are unavailable. {error}")
            return
        st.write("Camera calibration measures each phone's internal geometry: its focal length, where the optical axis meets the "
                 "image, and how the lens bends straight lines. Stereo measurement needs these numbers to turn image pixels into "
                 "accurate viewing rays; without them, distances computed from the two cameras would be distorted.")
        cols, rows = cal["pattern"]
        st.table(pd.DataFrame({"Setting": ["Left camera", "Right camera", "Video resolution", "Camera baseline",
                                           "Calibration checkerboard", "Checkerboard square size"],
                               "Value": [cal["left"]["camera"].split(",")[0], cal["right"]["camera"].split(",")[0],
                                         f"{cal['width']} \u00d7 {cal['height']} px", f"{cal['baseline_mm']:.0f} mm",
                                         f"{cols} \u00d7 {rows} inner corners", f"{cal['square_mm']:.0f} mm"]}).set_index("Setting"))
        st.markdown("**Intrinsic matrices** (pixels)")
        for column, (side, title) in zip(st.columns(2), (("left", "Left camera"), ("right", "Right camera"))):
            with column:
                st.caption(title)
                st.table(_matrix_frame(cal[side]["K"]))
        st.markdown("**Lens distortion coefficients** (OpenCV order)")
        names = ["k1", "k2", "p1", "p2", "k3"]
        st.table(pd.DataFrame({"Coefficient": names,
                               "Left camera": [f"{cal['left']['distortion'][n]:.5f}" for n in names],
                               "Right camera": [f"{cal['right']['distortion'][n]:.5f}" for n in names]}).set_index("Coefficient"))
        st.caption("Fitted from checkerboard corners in separate calibration videos (the 8 \u00d7 6 board and 20 mm squares are from my "
                   "capture setup). The 225 mm baseline is the physical spacing of the phones; the cameras' relative rotation for "
                   "the wall videos is estimated from the wall videos themselves.")


def _room_videos():
    status = room_data.video_status()
    for group, title in room_data.VIDEO_GROUPS.items():
        st.markdown(f"##### {title}")
        members = [item for item in status if item[1] == group]
        for column, (path, _, label, exists) in zip(st.columns(2), members):
            with column:
                st.caption(label)
                if exists:
                    st.video(str(path), format="video/mp4")
                else:
                    st.info(f"Video not available: {path.name} is not included in this deployment. "
                            "The measurement results below do not depend on it.")
    st.caption("Resized copies of my original stereo video recordings for web playback.")


def _wall_b_videos():
    status = room_data.wall_b_video_status()
    for n, title in room_data.WALL_B_GROUPS.items():
        st.markdown(f"##### {title}")
        members = [item for item in status if item[1] == n]
        if not any(item[3] for item in members):
            st.info(f"Playback copies of the {title.lower()} recordings are not included in this version. "
                    "The measurements below do not depend on them.")
            continue
        for column, (path, _, label, exists) in zip(st.columns(2), members):
            with column:
                st.caption(label)
                if exists:
                    st.video(str(path), format="video/mp4")
                else:
                    st.info(f"Playback copy not available: {path.name}.")
    if any(item[3] for item in status):
        st.caption("Resized copies of my original stereo video recordings for web playback.")


def _wall_b_section():
    """Wall B (room length): the accepted three-viewpoint results, read from the tracked wall_b_accepted_results.json."""
    cm, cm4 = room_data.cm, room_data.cm4
    with st.container(key="m7_room_wall_b"):
        st.header("Wall B \u2014 Length Measurement", anchor=False)
        with st.container(key="m7_room_wall_note_b"):
            st.markdown("Three viewpoints, six video recordings. This stage estimates the room's length.")
        try:
            wb = room_data.load_wall_b()
        except room_data.ResultsUnavailable as error:
            st.error(f"Wall B results are unavailable. {error}")
            return None
        with st.container(key="m7_room_recordings_b"):
            st.subheader("The Stereo Recordings", anchor=False)
            _wall_b_videos()

        st.subheader("Results", anchor=False)
        for column, item in zip(st.columns(3), wb["per_frame"]):
            with column:
                with st.container(border=True):
                    st.markdown(f"#### Frame {item['frame']}")
                    st.metric("Estimated length", f"{cm4(item['length_mm'])} cm")
        with st.container(border=True):
            st.markdown("#### Accepted estimate: mean of the three frame sets")
            left, right = st.columns(2)
            left.metric("Estimated dimension", f"{cm(wb['mean_mm'])} cm")
            right.metric("Actual dimension (tape measurement)", f"{wb['tape_mm'] / 10:.1f} cm")
            st.write(f"**Mean of the three frame sets:** {cm4(wb['mean_mm'])} cm")
            st.write(f"**Signed error:** {_signed(wb['difference_mm'] / 10, 1)} cm")
            st.write(f"**Absolute error:** {abs(wb['difference_mm']) / 10:.1f} cm")
            st.write(f"**Absolute percentage error:** {abs(wb['difference_percent']):.2f}%")
        return wb


_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _signed(value, digits=2):
    return f"{value:+.{digits}f}".replace("-", "\u2212")


def _error_chart(rows):
    """Grouped bars (estimated next to the tape measurement, in cm) for width, length and height."""
    import altair as alt          # ships with Streamlit; imported here so the page loads without it elsewhere
    order = [r["dimension"] for r in rows]
    series = ["Estimated dimension", "Actual dimension (tape measurement)"]
    data = pd.DataFrame([dict(Dimension=r["dimension"], Series=name, Centimetres=value)
                         for r in rows for name, value in ((series[0], r["estimate_cm"]), (series[1], r["reference_cm"]))])
    base = alt.Chart(data).encode(
        x=alt.X("Dimension:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=14)),
        xOffset=alt.XOffset("Series:N", sort=series),
        y=alt.Y("Centimetres:Q", title="Dimension (cm)"),
        color=alt.Color("Series:N", sort=series, scale=alt.Scale(domain=series, range=["#C9363E", "#8F99A8"]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("Dimension:N"), alt.Tooltip("Series:N"), alt.Tooltip("Centimetres:Q", format=".2f", title="cm")])
    st.altair_chart((base.mark_bar() + base.mark_text(dy=-8, fontSize=13).encode(text=alt.Text("Centimetres:Q", format=".1f"))).properties(height=360).configure(background="rgba(0,0,0,0)"),
                    width="stretch")


def _error_analysis(results, wb):
    """Quantitative error analysis of the three final dimensions (full-precision accepted values, no hard-coded statistics)."""
    analysis = room_data.error_analysis(results, wb)
    rows, stats = analysis["rows"], analysis["stats"]
    st.subheader("Validation & Error Analysis", anchor=False)
    st.write("After the estimation was completed, the estimated dimensions were compared with the actual dimensions measured with a tape measure. "
             "The tape measurements were not used in any calculation.")
    st.table(pd.DataFrame({
        "Dimension": [f"{r['dimension']} ({r['wall']})" for r in rows],
        "Estimated dimension (cm)": [f"{r['estimate_cm']:.2f}" for r in rows],
        "Actual dimension, tape measurement (cm)": [f"{r['reference_cm']:.2f}" for r in rows],
        "Signed error (cm)": [_signed(r["error_cm"]) for r in rows],
        "Absolute error (cm)": [f"{r['abs_error_cm']:.2f}" for r in rows],
        "Absolute percentage error": [f"{r['abs_percent']:.2f}%" for r in rows]}).set_index("Dimension"))
    st.caption("All three actual dimensions were measured with a tape measure. Signed error = estimated dimension \u2212 actual dimension "
               "(negative means the estimate is shorter).")

    st.markdown("##### Summary error statistics")
    with st.container(key="m7_room_stats"):
        for column, (label, value) in zip(st.columns(3), (("Mean Absolute Error (MAE)", f"{stats['mae_cm']:.2f} cm"),
                                                          ("Mean Absolute Percentage Error (MAPE)", f"{stats['mape_percent']:.2f} %"),
                                                          ("Root Mean Square Error (RMSE)", f"{stats['rmse_cm']:.2f} cm"))):
            with column:
                with st.container(border=True):
                    st.metric(label, value)

    st.markdown("##### Estimated vs. actual dimensions")
    _error_chart(rows)
    st.caption("Grouped bars in centimetres: the estimated dimension (red) next to the actual dimension from the tape measurement (grey).")

    with st.expander("Error calculation formulas"):
        st.markdown("For each dimension *i*, with estimated dimension $\\hat{x}_i$ and actual dimension (tape measurement) $x_i$, over $n$ dimensions (here $n = 3$):")
        for formula, text in (
                (r"e_i = \hat{x}_i - x_i", "**Signed error:** estimated minus actual dimension; negative means the estimate is shorter than the tape measurement."),
                (r"|e_i| = \left|\hat{x}_i - x_i\right|", "**Absolute error:** the size of the error, in the unit of the measurement (cm)."),
                (r"\mathrm{APE}_i = \frac{|e_i|}{x_i} \times 100\%", "**Absolute percentage error:** the absolute error relative to the actual dimension."),
                (r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} |e_i|", "**Mean absolute error:** the average size of the errors (cm)."),
                (r"\mathrm{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n} \frac{|e_i|}{x_i}", "**Mean absolute percentage error:** the average relative error."),
                (r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^{2}}", "**Root mean square error:** like MAE but squaring first, so the largest error weighs most (cm).")):
            st.markdown(text)
            st.latex(formula)

    worst = max(rows, key=lambda r: r["abs_error_cm"])
    wall_a = [r for r in rows if r["wall"] == "Wall A"]
    inside = results["width"]["actual_inside_interval"] and results["height"]["actual_inside_interval"]
    a_err, a_pct = max(r["abs_error_cm"] for r in wall_a), max(r["abs_percent"] for r in wall_a)
    st.markdown("##### Interpretation")
    st.markdown(
        ("- All three estimated dimensions are slightly smaller than their tape measurements.\n" if all(r["error_cm"] < 0 for r in rows)
         else "- The estimated dimensions differ from their tape measurements; the signed errors are listed in the table above.\n")
        + f"- The width and height estimates differ from the measured values by approximately {a_err:.0f} cm, with percentage errors "
        + (f"below {a_pct:.0f}%" if a_pct < 1 else f"up to {a_pct:.2f}%") + ". "
        + ("Both tape measurements fall within their respective approximate 95% uncertainty intervals.\n" if inside
           else "At least one tape measurement lies outside its approximate 95% uncertainty interval.\n")
        + f"- The {worst['dimension'].lower()} has the largest observed error of approximately {abs(worst['error_cm']):.1f} cm ({worst['abs_percent']:.2f}%).\n"
        f"- The overall mean absolute error is {stats['mae_cm']:.2f} cm, with a mean absolute percentage error of {stats['mape_percent']:.2f}%.\n"
        f"- The RMSE of {stats['rmse_cm']:.2f} cm reflects the greater influence of the {worst['dimension'].lower()} measurement error.")


def _final_dimensions(results, wb):
    """The combined summary: width and height from Wall A, length from Wall B, with their tape measurements."""
    cm = room_data.cm
    with st.container(key="m7_room_final"):
        st.header("Final Room Dimensions", anchor=False)
        with st.container(key="m7_room_wall_note_final"):
            st.markdown("Estimates obtained from the Wall A and Wall B measurements above.")
        cards = [("Width", results["width"]["estimate_mm"], "From Wall A", f"Actual dimension (tape measurement): {cm(results['width']['actual_mm'])} cm"),
                 ("Length", wb["mean_mm"] if wb else None, "From Wall B", f"Actual dimension (tape measurement): {wb['tape_mm'] / 10:.1f} cm" if wb else "Wall B results are unavailable."),
                 ("Height", results["height"]["estimate_mm"], "From Wall A", f"Actual dimension (tape measurement): {cm(results['height']['actual_mm'])} cm")]
        for column, (title, mm, source, reference) in zip(st.columns(3), cards):
            with column:
                with st.container(border=True):
                    st.metric(title, f"\u2248 {cm(mm)} cm" if mm is not None else "\u2014")
                    st.caption(f"**{source}**")
                    st.caption(reference)
        if wb is not None:
            _error_analysis(results, wb)
        else:
            st.info("The error analysis needs both the Wall A and the Wall B results; the Wall B results are unavailable.")
        st.markdown("##### Conclusion")
        if wb is not None:
            viewpoints = len(room_data.VIDEO_GROUPS) + len(room_data.WALL_B_GROUPS)
            stats = room_data.error_analysis(results, wb)["stats"]
            st.markdown(f"Using calibrated stereo video recordings captured from {_WORDS.get(viewpoints, viewpoints)} viewpoints, the room "
                        f"dimensions were estimated to be approximately {cm(results['width']['estimate_mm'])} cm wide, {cm(wb['mean_mm'])} cm long and "
                        f"{cm(results['height']['estimate_mm'])} cm high, with a mean absolute percentage error of {stats['mape_percent']:.2f}% across the "
                        f"three measured dimensions.")
            st.markdown("These results demonstrate how classical stereo vision and geometric reconstruction can be applied to video recordings to estimate real-world room dimensions.")
        else:
            st.markdown("The Wall B results are unavailable, so the room length and the combined error analysis cannot be shown.")


def room_results():
    st.markdown(ROOM_CSS, unsafe_allow_html=True)
    with st.container(key="m7_room_head"):
        with st.container(key="m7_room_title"):
            st.title("My Experiment & Results", anchor=False)
        st.header(Q3_BUTTON, anchor=False)
    with st.container(key="m7_room_body"):
        try:
            results = room_data.load_results()
        except room_data.ResultsUnavailable as error:
            st.error(f"Room measurement results are unavailable. {error} The saved results are read from "
                     "Modules/Module_7/outputs/room_measurement/.")
            return
        with st.container(key="m7_room_intro"):
            st.write("This page reports one room measurement experiment, carried out in two measurement stages on the same room: Wall A, from two "
                     "viewpoints, gives the width and height, and Wall B, from three viewpoints, gives the length.")
        wall_b_data = None
        try:
            wall_b_data = room_data.load_wall_b()
        except room_data.ResultsUnavailable:
            pass
        with st.container(key="m7_room_objective"):
            _objective_and_setup(results)
        with st.container(key="m7_room_calibration"):
            _room_calibration()
        with st.container(key="m7_room_methodology"):
            with st.expander("How It Works \u2014 Methodology"):
                _methodology(results, wall_b_data)

        with st.container(key="m7_room_wall_a"):
            st.header("Wall A \u2014 Width and Height Measurement", anchor=False)
            with st.container(key="m7_room_wall_note_a"):
                st.markdown("Two viewpoints, four video recordings. This stage estimates the room's width and height.")

            with st.container(key="m7_room_recordings"):
                st.subheader("The Stereo Recordings", anchor=False)
                _room_videos()

            st.subheader("Results", anchor=False)
            for column, (title, key) in zip(st.columns(2), (("Wall width", "width"), ("Wall height", "height"))):
                with column:
                    _dimension_card(title, results[key])
            st.caption("The approximate 95% intervals combine four error sources at \u00b11.96 standard deviations (normal approximation). "
                       "The camera-intrinsics uncertainty is assumed rather than measured and dominates them, as explained under How It Works \u2014 Methodology.")
            if not (results["width"]["actual_inside_interval"] and results["height"]["actual_inside_interval"]):
                st.warning("At least one actual dimension lies outside its saved uncertainty interval.")

            with st.container(key="m7_room_seen"):
                st.subheader("What the cameras saw", anchor=False)
                _room_images()

        wall_b = _wall_b_section()
        _final_dimensions(results, wall_b)


# ---------------------------------------------------------------- page

BLUE, BLUE_DARK = "#1D5FD1", "#174DA8"   # white text on these is above 6:1
HOME_TITLE = "Module 7"
HOME_SUBTITLE = "Stereo Vision"
Q1_TITLE = "Calibrated Stereo Object Measurement"
Q3_BUTTON = "Room Measurement Using Stereo Videos"
HOME_BUTTON_KEYS = ("m7_open_results", "m7_open_run", "m7_res_q1", "m7_res_q3", "m7_run_q1", "m7_run_q3")


def _buttons(suffix=""):
    return ", ".join(f".st-key-{key}{suffix}" for key in HOME_BUTTON_KEYS)


HOME_CSS = f"""
<style>
.st-key-m7_home_layout {{ gap: 0; padding-top: 0; }}  /* the nav row above already supplies the top space */
.st-key-m7_home_title, .st-key-m7_home_subtitle {{ gap: 0; flex-shrink: 0; }}
.st-key-m7_home_nav {{ flex-shrink: 0; }}
.st-key-m7_home_title [data-testid="stMarkdownContainer"],
.st-key-m7_home_subtitle [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.st-key-m7_home_subtitle {{ padding-top: 44px; }}
.st-key-m7_home_nav {{ padding-top: 64px; }}
.st-key-m7_home_title h1 {{
    font-size: clamp(2.5rem, 13vw, 80px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}}
.st-key-m7_home_subtitle h2 {{
    font-size: clamp(1.1rem, 4.6vw, 32px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2;
    text-align: center; padding: 0; margin: 0;
}}
@media (min-width: 801px) {{ .st-key-m7_home_subtitle h2 {{ white-space: nowrap; }} }}
.st-key-m7_home_nav {{ flex-direction: row; flex-wrap: wrap; gap: 24px; }}
{_buttons()} {{ flex: 1 1 280px; width: auto; min-width: 0; max-width: 100%; }}
{_buttons(" button")} {{
    width: 100%; height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
{_buttons(" button p")} {{
    font-size: 21px; font-weight: 700; color: #FFFFFF; line-height: 1.3; text-align: center; text-wrap: balance;
}}
{_buttons(" button:hover")} {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
{_buttons(" button:focus-visible")} {{
    outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT};
}}
{_buttons(" button:active")} {{ background-color: #8E2229; border-color: #8E2229; }}
@media (max-width: 640px) {{
    .st-key-m7_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 16px; }}
    {_buttons()} {{ flex: 0 0 auto; width: 100%; }}
}}
</style>
"""


# Module 7 homepage and its two selection pages: buttons stacked vertically; the selection buttons carry a smaller "(Question N)" line
# (a strong title block plus the label's second line) and the first homepage button is blue. Scoped to the m7_home_nav container.
M7_STACK_CSS = f"""
<style>
.st-key-m7_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 24px; max-width: 560px; width: 100%; margin: 0 auto; }}
{_buttons()} {{ flex: 0 0 auto; width: 100%; }}
{_buttons(" button p")} {{ font-size: 0.95rem; font-weight: 500; }}
{_buttons(" button p strong")} {{ display: block; font-size: 21px; font-weight: 700; line-height: 1.3; color: #FFFFFF; }}
{_buttons(" button p br")} {{ display: none; }}
.st-key-m7_open_results button p, .st-key-m7_open_run button p {{ font-size: 18px; }}   /* the two homepage buttons only */
.st-key-m7_open_results button {{ background-color: {BLUE}; border-color: {BLUE}; }}
.st-key-m7_open_results button:hover {{ background-color: {BLUE_DARK}; border-color: {BLUE_DARK}; }}
.st-key-m7_open_results button:active {{ background-color: #123F8F; border-color: #123F8F; }}
.st-key-m7_open_results button:focus-visible {{ box-shadow: 0 0 0 6px {BLUE}; }}
@media (max-width: 640px) {{ .st-key-m7_home_nav {{ gap: 16px; }} }}
</style>
"""


# the Calibrated Stereo landing page only: smaller heading, wraps instead of overflowing
Q1_TITLE_CSS = """
<style>
.st-key-m7_home_title h1 {
    font-size: clamp(2.5rem, 8vw, 60px); font-weight: 800; text-align: center; color: inherit;
    overflow-wrap: anywhere; max-width: 100%;
}
</style>
"""


# Room Measurement page only: spacing between the heading and the placeholder box (same gap as the landing buttons)
Q3_BODY_CSS = """
<style>
.st-key-m7_q3_body { padding-top: 64px; }
</style>
"""


def landing(title, subtitle, buttons):
    """Centered title, optional subtitle and red navigation buttons; the Module 7 and Question 1 pages share the markup and CSS."""
    st.markdown(HOME_CSS + M7_STACK_CSS, unsafe_allow_html=True)
    if subtitle is None:
        st.markdown(Q1_TITLE_CSS, unsafe_allow_html=True)
    with st.container(key="m7_home_layout"):
        with st.container(key="m7_home_title"):
            st.title(title, anchor=False)
        if subtitle is not None:
            with st.container(key="m7_home_subtitle"):
                st.header(subtitle, anchor=False)
        with st.container(key="m7_home_nav"):
            for target, key, label in buttons:
                st.button(label, key=key, on_click=select_part, args=(target,), width="stretch")


# breadcrumb levels below Home for every view: (label, view); the last is the current page, the one before it is the Back target
_M7, _RESULTS, _RUN = ("Module 7", "home"), ("My Experiments & Results", "my_results"), ("Run Your Own Experiment", "run_own")
_Q1_NAME, _Q3_NAME = "Calibrated Stereo Object Measurement", "Room Measurement"
CRUMBS = {"home": (_M7,),
          "my_results": (_M7, _RESULTS), "run_own": (_M7, _RUN),
          "experiment": (_M7, _RESULTS, (_Q1_NAME, "experiment")),
          "3_results": (_M7, _RESULTS, (_Q3_NAME, "3_results")),
          "1": (_M7, _RUN, (_Q1_NAME, "1")),
          "3_visitor": (_M7, _RUN, (_Q3_NAME, "3_visitor"))}


def question_label(title, number):
    """Button label: the title (bold block) and a smaller "(Question N)" line, both inside the one clickable button."""
    return f"**{title}**  \n(Question {number})"


init_state()
view = st.session_state["module7_part"]
if view not in CRUMBS:
    select_part("home")
    view = "home"
nav_row(CRUMBS[view], select_part, view=f"m7:{view}")
if view == "home":
    landing(HOME_TITLE, HOME_SUBTITLE, (("my_results", "m7_open_results", "View My Experiments & Results"),
                                        ("run_own", "m7_open_run", "Run Your Own Experiment")))
elif view == "my_results":
    landing("My Experiments & Results", None, (("experiment", "m7_res_q1", question_label(Q1_TITLE, 1)),
                                               ("3_results", "m7_res_q3", question_label(Q3_BUTTON, 3))))
elif view == "run_own":
    landing("Run Your Own Experiment", None, (("1", "m7_run_q1", question_label(Q1_TITLE, 1)),
                                              ("3_visitor", "m7_run_q3", question_label(Q3_BUTTON, 3))))
elif view == "1":
    part_one()
elif view == "experiment":
    restore_inputs(experiment=True, measurement=False)
    show_experiment()
elif view == "3_visitor":
    room_visitor_choice()
elif view == "3_results":
    room_results()
save_inputs()
