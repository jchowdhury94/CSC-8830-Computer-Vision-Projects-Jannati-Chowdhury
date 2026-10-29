"""Module 7 - Stereo Vision: calibrated stereo object measurement (Part 1) and the Part 3 placeholder."""

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
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from stereo_measure import calibration_wizard as wiz  # noqa: E402
from stereo_measure import load_model_b, load_selection  # noqa: E402
from stereo_measure import selection as sel  # noqa: E402
from stereo_measure import corner_drag as cd  # noqa: E402
from stereo_measure.calibration_io import CalibrationFileError, export_calibration, load_calibration  # noqa: E402
from stereo_measure.config import (CORNER_NAMES, IMAGE_SIZE, NOTEBOOK_PAIR_IDS, OUTPUT_DIR,  # noqa: E402
                                   PATTERN, SQUARE_MM)
from stereo_measure.live import measure_points  # noqa: E402

DISPLAY_DIR = MODULE_DIR / "display"
CALIB_SAMPLE_IMAGE = DISPLAY_DIR / "calibration_sample_left_pair_8.jpg"   # optimized crop of left calibration photo 8
CALIB_SAMPLE_PAIR = 8
ACCENT = "#C9363E"
CALIB_EXPANDER_LABEL = "**My Camera Calibration Reference**  \n:small[Click to View My Stereo Calibration Parameters]"
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
.st-key-m7_experiment_btn, .st-key-m7_experiment_btn [data-testid="stButton"] {{ width: 100%; }}
.st-key-m7_experiment_btn button {{
    width: 100%; background-color: {ACCENT}; border: 2px solid {ACCENT}; border-radius: 12px;
    padding: 0.9rem 1rem; color: #FFFFFF;
}}
.st-key-m7_experiment_btn button p {{ color: #FFFFFF; font-weight: 700; font-size: 1.15rem; }}
.st-key-m7_experiment_btn button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.st-key-m7_experiment_btn button:focus-visible {{ outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT}; }}
.st-key-m7_experiment_btn button:active {{ background-color: #8E2229; border-color: #8E2229; }}
/* Same chevron as the calibration header, drawn white at the far right; the label stays centred (the chevron is absolute). */
.st-key-m7_experiment_btn button {{ position: relative; }}
.st-key-m7_experiment_btn button::after {{
    content: ""; position: absolute; right: 1.2rem; top: 50%; width: 2rem; height: 2rem; margin-top: -1rem;
    background-color: #FFFFFF; -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-position: center;
    mask-position: center; -webkit-mask-size: contain; mask-size: contain; transition: transform 0.25s ease; {CHEVRON_MASK}
}}
@media (prefers-reduced-motion: reduce) {{ .st-key-m7_experiment_btn button::after {{ transition: none; }} }}
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
PART_ONE_CSS = ("<style>" + MODE_CSS + NAV_CSS + EXPANDER_CSS + EXPANDER_CSS.replace("m7_calib_ref", "m7_calibrate") + BUTTON_CSS
                + DOWNLOAD_CSS + "</style>")
# Drawn from the existing m7_show_experiment state: pointing up while the experiment is open.
ZOOM_BUTTON_CSS = """<style>
.st-key-m7_zoom_accept button { background: #1b8a3a !important; border: 1px solid #14692c !important; color: #fff !important; border-radius: 10px !important; }
.st-key-m7_zoom_accept button p { color: #fff !important; font-weight: 700 !important; }
.st-key-m7_zoom_accept button:hover { background: #167532 !important; }
.st-key-m7_zoom_cancel button { background: #3a3f47 !important; border: 1px solid #2a2e34 !important; color: #fff !important; border-radius: 10px !important; }
.st-key-m7_zoom_cancel button p { color: #fff !important; }
.st-key-m7_zoom_cancel button:hover { background: #2a2e34 !important; }
</style>"""
EXPERIMENT_OPEN_CSS = "<style>.st-key-m7_experiment_btn button::after { transform: rotate(180deg); }</style>"
CALIBRATION_SUMMARY_CSS = """<style>
.st-key-m7_calibration_summary [data-testid="stMetricValue"],
.st-key-m7_calibration_summary [data-testid="stMetricValue"] > div,
.st-key-m7_calibration_summary [data-testid="stMetricLabel"],
.st-key-m7_calibration_summary [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; }
.st-key-m7_calibration_summary [data-testid="stMetricValue"] { font-size: 1.6rem; line-height: 1.25; }
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

def init_state():
    ss = st.session_state
    ss.setdefault("module7_part", "home")
    ss.setdefault("m7_upload_gen", 0)
    ss.setdefault("m7_gen", 0)                  # bumped whenever a click is consumed, so each click event is used once
    ss.setdefault("m7_file_left", None)
    ss.setdefault("m7_file_right", None)
    ss.setdefault("m7_pair_id", None)
    ss.setdefault("m7_dialog_opened", None)
    ss.setdefault("m7_ack", None)
    ss.setdefault("m7_show_experiment", False)
    ss.setdefault("m7_scroll_to_experiment", False)
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


def select_part(part):
    st.session_state["module7_part"] = part


def store_upload(side):
    uploaded = st.session_state.get(f"m7_up_{side}_{st.session_state['m7_upload_gen']}")
    st.session_state[f"m7_file_{side}"] = (uploaded.name, uploaded.getvalue()) if uploaded is not None else None


def remove_uploads():
    ss = st.session_state
    ss["m7_upload_gen"] += 1                    # fresh uploader and actual-size widgets: the browser cannot restore old values
    ss["m7_file_left"] = ss["m7_file_right"] = None
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


def on_step(delta):
    wiz.step(st.session_state["m7_cal"], delta)


def on_goto():
    wiz.go_to(st.session_state["m7_cal"], st.session_state["m7_cal_goto"])


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


def toggle_experiment():
    ss = st.session_state
    ss["m7_show_experiment"] = not ss["m7_show_experiment"]
    ss["m7_scroll_to_experiment"] = ss["m7_show_experiment"]  # one-shot: only when opening


# Runs once, on the render right after the experiment opens. It scrolls the heading into view with a margin that keeps the
# "View My Experiment & Results" button (found by its widget-key class) on screen when there is room.
SCROLL_TO_EXPERIMENT_JS = """
<script>
setTimeout(function () {
  var heading = document.getElementById("m7-experiment-heading");
  if (!heading) { return; }
  var margin = 24;
  var button = document.querySelector(".st-key-m7_experiment_btn");
  if (button) {
    margin = heading.getBoundingClientRect().top - button.getBoundingClientRect().top + 16;
    margin = Math.min(margin, window.innerHeight * 0.4);
  }
  heading.style.scrollMarginTop = margin + "px";
  heading.scrollIntoView({behavior: "smooth", block: "start"});
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
    st.file_uploader(f"{'Replace' if rec else 'Upload'} {side} camera image (pair {index + 1})", type=["jpg", "jpeg", "png"],
                     key=f"m7_cal_up_{key}_{wiz_state['slot_gen'].get(key, 0)}", on_change=on_photo_upload, args=(index, side),
                     disabled=not enabled, label_visibility="collapsed")
    if rec is not None:
        st.button(f"Remove {side} photo", key=f"m7_cal_rm_{key}", on_click=on_photo_remove, args=(index, side))


def pair_navigator(evaluation, enabled):
    ss = st.session_state
    wiz_state = ss["m7_cal"]
    n, current = wiz_state["n_pairs"], wiz_state["current"]
    row = evaluation["pairs"][current]
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
    st.header("Part 1 – Object Size Estimation Using Calibrated Stereo")
    st.write(
        "Estimate the distance (Z-depth) of an object and calculate its real-world width and height "
        "using calibrated stereo camera geometry."
    )
    st.markdown(PART_ONE_CSS + (EXPERIMENT_OPEN_CSS if ss["m7_show_experiment"] else ""), unsafe_allow_html=True)
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

    st.divider()
    st.button("View My Experiment & Results", key="m7_experiment_btn", on_click=toggle_experiment, width="stretch")
    if ss["m7_show_experiment"]:
        show_experiment()


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
    st.header("My Experiment & Results", anchor="m7-experiment-heading")
    if st.session_state.pop("m7_scroll_to_experiment", False):
        st.html(SCROLL_TO_EXPERIMENT_JS, unsafe_allow_javascript=True)
    try:
        saved = load_saved_results()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Saved experiment results are unavailable: {error}")
        st.button("Close experiment", on_click=toggle_experiment)
        return
    truth = saved["summary"]["ground_truth_mm"]
    st.write(
        "I calibrated a stereo camera setup using an iPhone Air (left) and an iPhone 17 Pro Max (right), "
        "with 30 matched checkerboard image pairs. The resulting Model B calibration parameters were used "
        "to reconstruct 3D points and estimate object dimensions."
    )

    show_calibration_reference(get_model())

    st.write(
        "**Experiment:** A notebook (305 × 215 mm, red spine included in the width) was photographed using "
        "three stereo image pairs and measured with the Model B calibration. The corners shown are the final "
        "approved selections, and the measurements are read from the saved pipeline outputs."
    )

    pair_id = int(st.radio("Notebook stereo pair", NOTEBOOK_PAIR_IDS, format_func=lambda i: f"Pair {i}", horizontal=True,
                           key="m7_exp_pair"))
    pairs, corners, edges = saved["pairs"], saved["corners"], saved["edges"]
    row = pairs[pairs["pair"] == pair_id].iloc[0]
    try:
        selection = json.loads((MODULE_DIR / "selections" / f"pair_{pair_id}_corners.json").read_text())
        assert selection["confirmed"] is True and selection["corner_order"] == list(CORNER_NAMES)
    except (OSError, ValueError, AssertionError, KeyError) as error:
        st.error(f"Final corner selection for pair {pair_id} is unavailable: {error}")
        st.button("Close experiment", on_click=toggle_experiment)
        return

    st.subheader(f"Pair {pair_id}: photographs with the final approved corners")
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
    st.button("Close experiment", on_click=toggle_experiment)


# ---------------------------------------------------------------- Part 3 placeholder

def part_three():
    st.header("Part 3 – Room Measurement from Video")
    st.info("**Placeholder - not implemented yet.** The room measurement application for Part 3 has not been built, "
            "and no Part 3 results are shown here.")


# ---------------------------------------------------------------- page

init_state()
st.title("Module 7")
if st.session_state["module7_part"] == "home":
    st.subheader("Stereo Vision")
    for column, part, label, description in zip(
        st.columns(2), ("1", "3"),
        ("Part 1 – Calibrated Stereo Object Measurement", "Part 3 – Room Measurement from Video"),
        ("Measure an object's Z-depth, width and height from two calibrated stereo photographs.",
         "Placeholder: not implemented yet."),
    ):
        with column:
            st.button(label, key=f"module7_open_{part}", on_click=select_part, args=(part,))
            st.caption(description)
else:
    st.button("Module 7 Home", on_click=select_part, args=("home",))
    if st.session_state["module7_part"] == "1":
        part_one()
    else:
        part_three()
