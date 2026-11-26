"""Module 4 landing page and internally navigated assignment experiments."""

import hashlib
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from home_navigation import nav_row  # noqa: E402
from Modules.Module_4.human_boundary import (
    find_human_boundary,
    normalize_image,
    validate_rectangle,
)
from Modules.Module_4.rgb_preprocessing import (
    prepared_rgb_upload,
)
from Modules.Module_4.thermal_boundary import (
    CLOSING_ITERATIONS,
    CLOSING_KERNEL_SIZE,
    HSV_RANGES,
    find_thermal_boundary,
)
from Modules.Module_4.thermal_preprocessing import prepared_thermal_upload
from Modules.Module_4.sam2_runtime import (
    CPU_MAX_PIXELS,
    SAM2ResourceLimitError,
    checkpoint_configuration,
    check_image_budget,
    cpu_pixel_limit,
)


SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def sam2_configuration():
    """Track source, model, requested device and CPU policy independently of classical settings."""
    return (checkpoint_configuration(), SAM2_CONFIG, os.environ.get("SAM2_DEVICE", "auto"),
            os.environ.get("SAM2_CPU_MAX_PIXELS", str(CPU_MAX_PIXELS)))


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_sam2_model(checkpoint_settings, model_config, device):
    """Cache only the backend's model resource, including its inference lock."""
    from Modules.Module_4.sam2_comparison import load_sam2_model
    from Modules.Module_4.sam2_runtime import resolve_checkpoint

    local_path, cache_dir, _, _ = checkpoint_settings
    checkpoint = resolve_checkpoint(local_path, cache_dir)
    return load_sam2_model(checkpoint, model_config=model_config, device=device)


def current_sam2_model(source, config, device):
    """The cached model resource, rebuilt if it was created by an earlier copy of the SAM2 module. Streamlit reloads the app's
    modules when a source file changes, which gives the resource class a new identity; run_sam2 then rejects the old resource."""
    from Modules.Module_4.sam2_comparison import SAM2ModelResource

    resource = cached_sam2_model(source, config, device)
    if not isinstance(resource, SAM2ModelResource):
        cached_sam2_model.clear()
        resource = cached_sam2_model(source, config, device)
    return resource


def clear_results(method=None):
    """Clear one method or both, and always invalidate the comparison."""
    st.session_state.pop("module4_comparison", None)
    if method in (None, "classical"):
        st.session_state.pop("module4_results", None)
        st.session_state.pop("module4_classical_identity", None)
        st.session_state.pop("module4_classical_error", None)
    if method in (None, "sam2"):
        st.session_state.pop("module4_sam2_results", None)
        st.session_state.pop("module4_sam2_identity", None)
        st.session_state.pop("module4_sam2_error", None)


def reset_classical_result():
    """Discard the Classical run while preserving shared inputs and SAM2."""
    clear_results("classical")


def reset_sam2_result():
    """Discard the SAM2 run while preserving shared inputs and Classical."""
    clear_results("sam2")


def reset_rectangle():
    """Clear corners and replace the component so old clicks cannot replay."""
    st.session_state["module4_points"] = []
    st.session_state["module4_click_generation"] += 1
    clear_results()


def original_point(click, image_shape):
    """Map rendered-image coordinates to original-image pixel coordinates."""
    display_width, display_height = click["width"], click["height"]
    if display_width <= 0 or display_height <= 0:
        raise ValueError("The image preview is not ready. Please click again.")
    height, width = image_shape[:2]
    x = int(np.clip(click["x"] * width / display_width, 0, width - 1))
    y = int(np.clip(click["y"] * height / display_height, 0, height - 1))
    return x, y


def update_rgb_upload():
    """Keep image data independent of Streamlit's hidden-widget cleanup."""
    uploaded = st.session_state.get(f"module4_upload_{st.session_state['module4_upload_generation']}")
    st.session_state["module4_rgb_bytes"] = uploaded.getvalue() if uploaded is not None else None
    st.session_state["module4_rgb_filename"] = uploaded.name if uploaded is not None else None
    st.session_state.pop("module4_rgb_prepared", None)
    reset_rectangle()


def clear_rgb_upload():
    st.session_state["module4_upload_generation"] += 1
    st.session_state.pop("module4_rgb_bytes", None)
    st.session_state.pop("module4_rgb_filename", None)
    st.session_state.pop("module4_rgb_prepared", None)
    reset_rectangle()


SETTING_KEYS = ("module4_gaussian_kernel", "module4_gaussian_sigma", "module4_canny_lower", "module4_canny_upper",
                "module4_grabcut_iterations", "module4_cleanup_enabled", "module4_morphology_kernel")


def reset_experiment():
    """Full reset of the RGB experiment: upload, rectangle, results, comparison and the classical settings (back to their defaults)."""
    clear_rgb_upload()
    for key in SETTING_KEYS:
        st.session_state.pop(key, None)


def navigate_module4(section):
    st.session_state["module4_section"] = section
    if section == "thermal":
        st.session_state["module4_q2_click_generation"] = st.session_state.get("module4_q2_click_generation", 0) + 1
    # Replace the click component when revisiting RGB so an old click cannot replay.
    if section == "rgb":
        st.session_state["module4_click_generation"] = st.session_state.get("module4_click_generation", 0) + 1


def render_classical_settings():
    """Render the single Q1 parameter editor."""
    st.markdown("##### Processing Settings")
    gaussian_column, canny_column = st.columns(2)
    with gaussian_column:
        st.selectbox("Gaussian kernel size", [1, 3, 5, 7], index=1, key="module4_gaussian_kernel",
                     help="Size of the smoothing window. A larger kernel blurs more.")
        st.slider("Gaussian sigma", 0.1, 3.0, 0.8, 0.1, key="module4_gaussian_sigma",
                  help="Spread of the Gaussian. A larger sigma blurs more.")
    with canny_column:
        st.slider("Canny lower threshold", 0, 255, 50, key="module4_canny_lower",
                  help="Edges weaker than this are discarded.")
        st.slider("Canny upper threshold", 1, 255, 150, key="module4_canny_upper",
                  help="Edges stronger than this are kept. Edges in between are kept only if they connect to strong edges.")
    grabcut_column, cleanup_column = st.columns(2)
    with grabcut_column:
        st.slider("GrabCut iterations", 1, 10, 5, key="module4_grabcut_iterations",
                  help="How many times GrabCut refines the foreground and background. More iterations take longer.")
    with cleanup_column:
        cleanup_enabled = st.checkbox("Apply Morphological Cleanup", value=False, key="module4_cleanup_enabled")
        st.caption("Morphological cleanup helps refine the foreground mask by reducing small imperfections before extracting "
                   "the final boundary.")
        if cleanup_enabled:
            st.selectbox(
                "Cleanup kernel size", [3, 5], key="module4_morphology_kernel",
                help="A larger kernel produces stronger cleanup but may remove fine details from the foreground.",
            )


# Instructional example for Step 2: the saved example photograph with an illustrative rectangle and two corners. It is
# separate from the visitor's upload. Coordinates are in that 1000 x 1400 photograph and leave a small margin around the person.
EXAMPLE_IMAGE = "step1_original.jpg"
EXAMPLE_CORNERS = ((470, 40), (940, 1375))


# The same example for the thermal page: the saved thermal photograph, with the black markers that page uses.
THERMAL_EXAMPLE_IMAGE = "thermal_step1_original.jpg"
THERMAL_EXAMPLE_CORNERS = ((250, 15), (845, 1438))


@st.cache_data(show_spinner=False)
def rectangle_example(name=EXAMPLE_IMAGE, corners=EXAMPLE_CORNERS, color=(255, 200, 0)):
    image = cv2.cvtColor(cv2.imread(str(DISPLAY_DIR / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    pad = 130
    image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(236, 238, 242))
    (x1, y1), (x2, y2) = ((x, y + pad) for x, y in corners)
    marker, ink = color, (30, 33, 40)
    cv2.rectangle(image, (x1, y1), (x2, y2), marker, 8)
    for x, y in ((x1, y1), (x2, y2)):
        cv2.circle(image, (x, y), 22, ink, -1)
        cv2.circle(image, (x, y), 16, marker, -1)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.7, 4
    cv2.putText(image, "1. First corner", (x1, y1 - 40), font, scale, ink, thickness, cv2.LINE_AA)
    label = "2. Second corner"
    width = cv2.getTextSize(label, font, scale, thickness)[0][0]
    cv2.putText(image, label, (x2 - width + 10, y2 + 75), font, scale, ink, thickness, cv2.LINE_AA)
    return image


VISITOR_CSS = """
<style>
.st-key-m4_vis_headings h1 {
    font-size: clamp(1.75rem, 5vw + 0.5rem, 3rem); font-weight: 800; line-height: 1.15; text-align: center;
    padding: 0.25rem 0 0; margin: 0; overflow-wrap: anywhere;
}
.st-key-m4_vis_headings h2 {
    font-size: clamp(1rem, 1.6vw + 0.5rem, 1.3rem); font-weight: 500; line-height: 1.3; text-align: center; opacity: 0.8;
    padding: 0.5rem 0 0; margin: 0; overflow-wrap: anywhere;
}
.st-key-m4_vis_headings { padding-bottom: 0.5rem; }
/* this page only: the same 900px content width as the Calibrate My Cameras workflow */
.stMainBlockContainer, .block-container { max-width: 900px !important; }
.st-key-m4_rect_view { align-items: center; text-align: center; }
/* Light theme only (Streamlit follows prefers-color-scheme): charcoal primary text, a readable darker gray for secondary text,
   darker green / red status marks, and locked steps that are muted rather than faded. The dark theme is not touched. */
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) .st-key-m4_vis_headings h2 { color: #4B5160; opacity: 1; }
    :root:not([data-theme="dark"]) .st-key-m4_vis_body :is([data-testid="stCaptionContainer"], [data-testid="stImageCaption"],
        [data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] *) { color: #4B5160 !important; opacity: 1; }
    :root:not([data-theme="dark"]) .st-key-m4_vis_body :is([data-testid="stTable"] th, [data-testid="stTable"] td) { color: #31333F !important; }
    :root:not([data-theme="dark"]) [class*="st-key-m4_stepbox_"]:has([class*="st-key-m4_stmark_"][class*="_locked"]) { opacity: 1; }
    :root:not([data-theme="dark"]) [class*="st-key-m4_stmark_"][class*="_locked"] h3 { color: #5B6270 !important; }
    :root:not([data-theme="dark"]) [class*="st-key-m4_stmark_"][class*="_locked"] :is([data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *) { color: #4B5160 !important; }
    :root:not([data-theme="dark"]) [class*="st-key-m4_stmark_"][class*="_active"] :is([data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *) { color: #B3262E !important; }
    :root:not([data-theme="dark"]) [class*="st-key-m4_stmark_"][class*="_done"] :is([data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *) { color: #1B7A3E !important; }
}
.st-key-m4_vis_body h3 { padding-top: 0.25rem; }
.st-key-m4_vis_body h5 { font-weight: 700; margin: 0; padding: 0.75rem 0 0.25rem; }
/* Step boxes, as in the Calibrate My Cameras workflow: locked = dimmed, current = red border, completed = green border. Only the
   small heading wrapper inside a box carries the state, so the widgets in the box keep their identity when the state changes. */
[class*="st-key-m4_stepbox_"]:has([class*="st-key-m4_stmark_"][class*="_locked"]) { opacity: 0.5; }
[class*="st-key-m4_stepbox_"]:has([class*="st-key-m4_stmark_"][class*="_active"]) { border-color: #C9363E !important; }
[class*="st-key-m4_stepbox_"]:has([class*="st-key-m4_stmark_"][class*="_done"]) { border-color: rgba(46, 158, 91, 0.6) !important; }
</style>
"""


STEP_TITLES = {1: "Upload RGB Image", 2: "Select Person Rectangle", 3: "Classical Method", 4: "SAM2", 5: "Compare Results"}
STEP_LOCKED = {n: f"Locked. Complete Step {n - 1} to continue." for n in (2, 3, 4, 5)}


THERMAL_STEP_TITLES = {1: "Upload Thermal Image", 2: "Classical Method", 3: "SAM2", 4: "Compare Results"}


def step_head(n, state, titles=STEP_TITLES):
    """Heading of one step box: title plus a lock message, a completed mark or the current-step mark."""
    st.subheader(f"Step {n} · {titles[n]}", anchor=False)
    if state == "locked":
        st.caption(f":material/lock: {STEP_LOCKED[n]}")
    elif state == "done":
        st.caption(":green[:material/check_circle: Completed]")
    else:
        st.caption(":red[:material/arrow_forward: Current step]")


def render_rgb():
    ss = st.session_state
    ss.setdefault("module4_points", [])
    ss.setdefault("module4_click_generation", 0)
    ss.setdefault("module4_upload_generation", 0)
    # Streamlit forgets a widget's value while the widget is not drawn (a locked step); reassigning the key keeps it.
    for key in SETTING_KEYS:
        if key in ss:
            ss[key] = ss[key]
    st.markdown(VISITOR_CSS, unsafe_allow_html=True)

    # ---- everything the step states depend on is worked out first, from the stored upload, rectangle and results
    image_bytes = ss.get("module4_rgb_bytes")
    image_digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes is not None else None
    if ss.get("module4_image_digest") != image_digest:
        reset_rectangle()
        ss["module4_image_digest"] = image_digest
    image = original_shape = preparation_error = None
    if image_bytes is not None:
        image, original_shape, preparation_error = prepared_rgb_upload(ss, image_bytes, image_digest)
    prepared_identity = (
        ss["module4_rgb_prepared"][0], tuple(image.shape) if image is not None else None,
    ) if image_bytes is not None else None
    if ss.get("module4_rgb_processing_identity") != prepared_identity:
        reset_rectangle()
        ss["module4_rgb_processing_identity"] = prepared_identity
    resized = image is not None and image.shape != original_shape

    points = ss["module4_points"]
    rectangle = rectangle_image = rectangle_error = None
    if image is not None:
        rectangle_image = image.copy()
        if len(points) == 2:
            left, right = sorted((points[0][0], points[1][0]))
            top, bottom = sorted((points[0][1], points[1][1]))
            try:
                rectangle = validate_rectangle((left, top, right - left, bottom - top), image.shape)
            except ValueError as error:
                rectangle_error = str(error)
            else:
                # The backend rectangle uses an exclusive right/bottom boundary.
                cv2.rectangle(rectangle_image, (left, top), (right - 1, bottom - 1), (255, 200, 0), 2)

    # Both methods use the same processing image and rectangle; a change to either, or to a method's own settings,
    # discards the results that depended on it.
    defaults = {
        "gaussian_kernel": 3, "gaussian_sigma": 0.8,
        "canny_lower": 50, "canny_upper": 150, "grabcut_iterations": 5,
        "cleanup_enabled": False, "morphology_kernel": 3,
    }
    settings = {
        "gaussian_kernel_size": ss.get("module4_gaussian_kernel", defaults["gaussian_kernel"]),
        "gaussian_sigma": ss.get("module4_gaussian_sigma", defaults["gaussian_sigma"]),
        "canny_lower": ss.get("module4_canny_lower", defaults["canny_lower"]),
        "canny_upper": ss.get("module4_canny_upper", defaults["canny_upper"]),
        "grabcut_iterations": ss.get("module4_grabcut_iterations", defaults["grabcut_iterations"]),
        "cleanup_enabled": ss.get("module4_cleanup_enabled", defaults["cleanup_enabled"]),
        "morphology_kernel_size": ss.get("module4_morphology_kernel", defaults["morphology_kernel"]),
    }
    cleanup_enabled = settings["cleanup_enabled"]
    shared_inputs = (image_digest, tuple(image.shape) if image is not None else None, rectangle)
    classical_inputs = (shared_inputs, tuple(settings.items()))
    sam2_inputs = (shared_inputs, sam2_configuration())
    if ss.get("module4_shared_inputs") != shared_inputs:
        clear_results()
        ss["module4_shared_inputs"] = shared_inputs
    if ss.get("module4_result_inputs") != classical_inputs:
        clear_results("classical")
        ss["module4_result_inputs"] = classical_inputs
    if ss.get("module4_sam2_inputs") != sam2_inputs:
        clear_results("sam2")
        ss["module4_sam2_inputs"] = sam2_inputs
    ready = rectangle is not None
    thresholds_valid = settings["canny_lower"] < settings["canny_upper"]
    results = ss.get("module4_results")
    sam2_results = ss.get("module4_sam2_results")
    comparison_ready = (
        ready and results is not None and sam2_results is not None
        and ss.get("module4_classical_identity") == classical_inputs
        and ss.get("module4_sam2_identity") == sam2_inputs
    )
    compared = comparison_ready and ss.get("module4_comparison") == (classical_inputs, sam2_inputs)
    state = {
        1: "done" if image is not None else "active",
        2: "locked" if image is None else "done" if ready else "active",
        3: "locked" if not ready else "done" if results is not None else "active",
        4: "locked" if results is None else "done" if sam2_results is not None else "active",
        5: "locked" if not comparison_ready else "done" if compared else "active",
    }

    def process_classical():
        clear_results("classical")
        try:
            with st.spinner("Detecting edges and segmenting the foreground…"):
                result = find_human_boundary(image, rectangle, **settings)
                # The original image can be decoded from the upload; avoid retaining another copy.
                result.pop("original_image", None)
                if not cleanup_enabled:
                    result["cleaned_mask"] = result["raw_mask"]
                ss["module4_results"] = result
                ss["module4_classical_identity"] = classical_inputs
        except Exception:
            logging.getLogger(__name__).exception("Module 4 classical inference failed")
            ss["module4_classical_error"] = "Classical processing could not complete. Check the image, rectangle, and settings, then try again."

    def process_sam2():
        clear_results("sam2")
        try:
            from Modules.Module_4.sam2_comparison import run_sam2, select_device

            source, config, requested_device, _ = sam2_inputs[1]
            device = select_device(requested_device)
            # Refuse oversized CPU requests before downloading or loading weights.
            check_image_budget(image.shape, device, cpu_pixel_limit())
            with st.spinner("Preparing SAM2 model…"):
                resource = current_sam2_model(source, config, device)
            with st.spinner("Segmenting the selected person with SAM2…"):
                result = run_sam2(image, rectangle, resource)
                ss["module4_sam2_results"] = result
                ss["module4_sam2_identity"] = sam2_inputs
        except SAM2ResourceLimitError as error:
            ss["module4_sam2_error"] = str(error)
        except Exception:
            logging.getLogger(__name__).exception("Module 4 SAM2 inference failed")
            ss["module4_sam2_error"] = (
                "SAM2 could not complete. Model preparation or processing may be temporarily "
                "unavailable; please try again. You can still use the classical method."
            )

    def centered(image_array, caption):
        with st.columns([1, 1.2, 1])[1]:
            st.image(image_array, caption=caption, width="stretch")

    def pair():
        """Two equal columns for side-by-side images, kept at a moderate size on the wider page."""
        return st.columns([1, 4, 4, 1])[1:3]

    with st.container(key="m4_vis_headings"):
        st.title("Human Boundary Detection in RGB Images", anchor=False)
        st.header("Upload an RGB image, select a person, and compare human boundary detection using classical computer "
                  "vision and SAM2.", anchor=False)

    with st.container(key="m4_vis_body"):
        # ---------------------------------------------------------------- Step 1
        with st.container(border=True, key="m4_stepbox_1"):
            with st.container(key=f"m4_stmark_1_{state[1]}"):
                step_head(1, state[1])
            st.write("Upload an RGB image of one person, preferably fully visible and distinguishable from the background.")
            st.file_uploader(
                "Upload RGB Image", type=["jpg", "jpeg", "png"],
                key=f"module4_upload_{ss['module4_upload_generation']}", on_change=update_rgb_upload,
            )
            st.caption(f"Supported formats: JPG, JPEG and PNG. Maximum upload size: {st.get_option('server.maxUploadSize')} MB per file.")
            if image_bytes is not None and ss.get(f"module4_upload_{ss['module4_upload_generation']}") is None:
                st.caption(f"Loaded RGB image: {ss.get('module4_rgb_filename', 'previous upload')}")
                st.button("Clear RGB Image", key="module4_clear_image", on_click=clear_rgb_upload)
            if preparation_error:
                st.error(preparation_error)
            if resized:
                st.caption(
                    f"Image resized from {original_shape[1]} × {original_shape[0]} to "
                    f"{image.shape[1]} × {image.shape[0]} pixels for processing."
                )

        # ---------------------------------------------------------------- Step 2
        with st.container(border=True, key="m4_stepbox_2"):
            with st.container(key=f"m4_stmark_2_{state[2]}"):
                step_head(2, state[2])
            if state[2] != "locked":
                with st.expander("How to select the rectangle", expanded=False):
                    st.write("Click two opposite corners to create a rectangle around the person. Select the first corner above and "
                             "to the left of the person, then select the second corner below and to the right.")
                    st.write("Include the entire person, with a small margin around the body. Leave some background outside the rectangle.")
                    with st.columns([1, 1.5, 1])[1]:
                        st.image(rectangle_example(), caption="Select the first corner, then the second corner.", width="stretch")
                if rectangle_error:
                    st.warning(f"{rectangle_error} Use Reset Rectangle to select again.")
                if rectangle is not None:
                    st.caption(f"Rectangle (x, y, width, height): {rectangle} pixels")
                # Limit preview payload size; click mapping uses actual rendered dimensions.
                height, width = image.shape[:2]
                scale = min(1.0, 700 / width, 700 / height)
                preview = cv2.resize(rectangle_image, (max(1, round(width * scale)), max(1, round(height * scale))))
                for x, y in points:
                    position = (round(x * preview.shape[1] / width), round(y * preview.shape[0] / height))
                    cv2.circle(preview, position, 5, (255, 200, 0), -1)
                if len(points) < 2:
                    st.info("Select the first corner." if not points else "Select the second corner.")
                    with st.container(key="m4_rect_view"):
                        click = streamlit_image_coordinates(
                            preview, width="content", cursor="crosshair",
                            key=f"module4_image_{ss['module4_click_generation']}",
                        )
                    if click is not None:
                        try:
                            point = original_point(click, image.shape)
                        except (KeyError, ValueError, TypeError, OverflowError):
                            st.warning("Could not read that click. Please click the image again.")
                        else:
                            points.append(point)
                            clear_results()
                            # A new component key consumes this event exactly once across reruns.
                            ss["module4_click_generation"] += 1
                            st.rerun()
                else:
                    with st.container(key="m4_rect_view"):
                        st.image(preview, caption="Selected corners and GrabCut rectangle", width="content")
                st.button("Reset Rectangle", key="module4_reset_rectangle", on_click=reset_rectangle, disabled=not points)

        # ---------------------------------------------------------------- Step 3
        with st.container(border=True, key="m4_stepbox_3"):
            with st.container(key=f"m4_stmark_3_{state[3]}"):
                step_head(3, state[3])
            if state[3] != "locked":
                st.write("The classical method uses Gaussian smoothing, Canny edge detection, GrabCut foreground segmentation, "
                         "and contour extraction to detect the human boundary.")
                render_classical_settings()
                if not thresholds_valid:
                    st.warning("Canny lower threshold must be less than the upper threshold.")
                if st.button("Apply Classical Method", key="module4_apply_classical", type="primary", disabled=not thresholds_valid):
                    process_classical()
                    if ss.get("module4_results") is not None:
                        st.rerun()
                if ss.get("module4_classical_error"):
                    st.error(ss["module4_classical_error"])
                if results is not None:
                    st.markdown("##### Gaussian Smoothing")
                    original_column, smoothed_column = pair()
                    with original_column:
                        st.image(rectangle_image, caption="Processing Image with GrabCut Rectangle" if resized else "Original Image with GrabCut Rectangle", width="stretch")
                    with smoothed_column:
                        st.image(results["smoothed_image"], caption="Gaussian-Smoothed Image", width="stretch")
                    st.write("Gaussian smoothing reduces image noise and small intensity variations before edge detection.")
                    st.markdown("##### Canny Edge Detection")
                    centered(results["canny_edges"], "Canny Edge Map")
                    st.write("Canny detects intensity changes throughout the image, including the person's outline, clothing details, "
                             "and background edges. It demonstrates edge detection and is not used as the input to GrabCut.")
                    st.markdown("##### GrabCut Segmentation")
                    if cleanup_enabled:
                        raw_column, cleaned_column = pair()
                        with raw_column:
                            st.image(results["raw_mask"], caption="Raw GrabCut Foreground Mask", width="stretch")
                        with cleaned_column:
                            st.image(results["cleaned_mask"], caption="Foreground Mask After Morphological Cleanup", width="stretch")
                    else:
                        centered(results["raw_mask"], "GrabCut Foreground Mask")
                    st.write("GrabCut uses the selected rectangle to separate the foreground from the background. White represents the "
                             "foreground, and black represents the background.")
                    if cleanup_enabled:
                        st.write("Morphological cleanup refines small imperfections in the foreground mask. It cannot reliably recover "
                                 "large missing body regions caused by incorrect segmentation.")
                    st.write("Similar foreground and background colors, shadows, and complex backgrounds can cause parts of the person "
                             "to be missed or background regions to be included.")
                    st.markdown("##### Final Human Boundary")
                    if results["contour"] is None:
                        st.warning("No foreground contour was found. Adjust the rectangle or settings and try again; the image below is unchanged.")
                    centered(results["boundary_overlay"], "External Boundary on the Processing Image" if resized else "External Boundary on the Original Image")
                    st.write("The human boundary is extracted from the largest external contour of the segmentation mask.")
                    st.button("Reset Classical Method", key="module4_reset_classical", on_click=reset_classical_result)

        # ---------------------------------------------------------------- Step 4
        with st.container(border=True, key="m4_stepbox_4"):
            with st.container(key=f"m4_stmark_4_{state[4]}"):
                step_head(4, state[4])
            if state[4] != "locked":
                st.write("SAM2 uses the selected rectangle to segment the person from the background.")
                if st.button("Run SAM2", key="module4_run_sam2", type="primary"):
                    process_sam2()
                    if ss.get("module4_sam2_results") is not None:
                        st.rerun()
                if ss.get("module4_sam2_error"):
                    st.error(ss["module4_sam2_error"])
                if sam2_results is not None:
                    mask_column, boundary_column = pair()
                    with mask_column:
                        st.image(sam2_results["binary_mask"], caption="SAM2 Binary Mask", width="stretch")
                    with boundary_column:
                        st.image(sam2_results["boundary_overlay"], caption="SAM2 Human Boundary", width="stretch")
                    metadata = sam2_results["metadata"]
                    if metadata["empty_mask"]:
                        st.info("SAM2 predicted no foreground. Try selecting another rectangle.")
                if sam2_results is not None or ss.get("module4_sam2_error"):
                    st.button("Reset SAM2", key="module4_reset_sam2", on_click=reset_sam2_result)

        # ---------------------------------------------------------------- Step 5
        with st.container(border=True, key="m4_stepbox_5"):
            with st.container(key=f"m4_stmark_5_{state[5]}"):
                step_head(5, state[5])
            if state[5] != "locked":
                if st.button("Compare Results", key="module4_compare", type="primary"):
                    ss["module4_comparison"] = (classical_inputs, sam2_inputs)
                    st.rerun()
                if compared:
                    st.markdown("##### Foreground Mask Comparison")
                    classical_column, sam2_column = pair()
                    with classical_column:
                        # the final classical foreground mask, the one the boundary was extracted from
                        st.image(results["cleaned_mask"], caption="Classical Method — Cleaned Foreground Mask" if cleanup_enabled
                                 else "Classical Method — GrabCut Foreground Mask", width="stretch")
                    with sam2_column:
                        st.image(sam2_results["binary_mask"], caption="SAM2 — Foreground Mask", width="stretch")
                    st.markdown("##### Human Boundary Comparison")
                    classical_column, sam2_column = pair()
                    with classical_column:
                        st.image(results["boundary_overlay"], caption="Classical Human Boundary", width="stretch")
                    with sam2_column:
                        st.image(sam2_results["boundary_overlay"], caption="SAM2 Human Boundary", width="stretch")
                    mask_agreement_section(results["cleaned_mask"], sam2_results["binary_mask"])
                    st.markdown("##### Observations & Limitations")
                    st.markdown(
                        "- The classical method depends on the selected rectangle and how clearly the person can be distinguished "
                        "from the background. Shadows and similar colors may affect segmentation.\n"
                        "- SAM2 results depend on image quality, background complexity, and rectangle selection.\n"
                        "- Measuring actual segmentation accuracy requires a ground-truth mask.")

        st.button("Reset Experiment", key="module4_reset_rgb", on_click=reset_experiment)


def clear_thermal_results(method=None):
    """Invalidate one Q2 method or both, and always invalidate comparison."""
    st.session_state.pop("module4_q2_comparison", None)
    if method in (None, "classical"):
        for key in ("module4_q2_results", "module4_q2_error", "module4_q2_classical_identity"):
            st.session_state.pop(key, None)
    if method in (None, "sam2"):
        for key in ("module4_q2_sam2_results", "module4_q2_sam2_error", "module4_q2_sam2_identity"):
            st.session_state.pop(key, None)


def reset_thermal_rectangle():
    """Replace Q2 click state; preserve classical thermal and all Q1 results."""
    st.session_state["module4_q2_points"] = []
    st.session_state["module4_q2_click_generation"] = st.session_state.get("module4_q2_click_generation", 0) + 1
    clear_thermal_results("sam2")


def record_thermal_corner(component_key, image_shape):
    """Consume each actual component click once; rerenders are not clicks."""
    click = st.session_state.get(component_key)
    if not isinstance(click, dict):
        return
    event_time = click.get("unix_time")
    if event_time is None or event_time == st.session_state.get("module4_q2_last_click_time"):
        return
    points = st.session_state.get("module4_q2_points", [])
    if len(points) >= 2:
        return
    try:
        point = original_point(click, image_shape)
    except (KeyError, ValueError, TypeError, OverflowError):
        return
    st.session_state["module4_q2_last_click_time"] = event_time
    st.session_state["module4_q2_points"] = [*points, point]
    clear_thermal_results("sam2")


def update_thermal_upload():
    """Persist Q2 bytes across navigation, independently of uploader widgets."""
    uploaded = st.session_state.get(f"module4_q2_upload_{st.session_state['module4_q2_upload_generation']}")
    st.session_state["module4_q2_bytes"] = uploaded.getvalue() if uploaded is not None else None
    st.session_state["module4_q2_filename"] = uploaded.name if uploaded is not None else None
    st.session_state.pop("module4_q2_prepared", None)
    st.session_state.pop("module4_q2_processing_identity", None)
    clear_thermal_results()
    reset_thermal_rectangle()


def reset_thermal_upload():
    """Replace the Q2 uploader and clear Q2 data without touching RGB state."""
    st.session_state["module4_q2_upload_generation"] = st.session_state.get("module4_q2_upload_generation", 0) + 1
    for key in ("module4_q2_bytes", "module4_q2_filename", "module4_q2_digest"):
        st.session_state.pop(key, None)
    st.session_state.pop("module4_q2_prepared", None)
    st.session_state.pop("module4_q2_processing_identity", None)
    clear_thermal_results()
    reset_thermal_rectangle()


def thermal_mask_iou(classical_mask, sam2_mask):
    """Compare same-size masks without modifying pixels; empty union is undefined."""
    if classical_mask.shape != sam2_mask.shape or classical_mask.ndim != 2:
        raise ValueError("IoU requires two masks with matching processing dimensions.")
    classical_foreground = classical_mask != 0
    sam2_foreground = sam2_mask != 0
    union = np.count_nonzero(classical_foreground | sam2_foreground)
    if union == 0:
        return None
    return float(np.count_nonzero(classical_foreground & sam2_foreground) / union)


# ---- Mask Agreement (IoU): one educational section shared by both visitor pages. The IoU itself comes from thermal_mask_iou
# above; the diagrams are static inline SVG (text and lines use the current text color, so they follow the theme).
IOU_BLUE, IOU_ORANGE, IOU_GREEN, IOU_INK = "#478BD1", "#E7A33D", "#459B70", "#10261B"
IOU_CSS = """
<style>
.st-key-m4_iou_box h5, .st-key-m4_iou_box h3, .st-key-m4_iou_box p, .st-key-m4_iou_box [data-testid="stCaptionContainer"] { text-align: center; }
.st-key-m4_iou_box h3 { font-size: clamp(2rem, 8vw, 2.75rem); font-weight: 800; padding: 0.25rem 0; }
.st-key-m4_iou_art svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
</style>
"""


def overlap_distance(iou):
    """Distance between the centres of two equal circles of radius 1 whose intersection over union equals iou."""
    def circles_iou(distance):
        lens = 2 * math.acos(distance / 2) - (distance / 2) * math.sqrt(4 - distance * distance)
        return lens / (2 * math.pi - lens)
    low, high = 0.0, 2.0
    for _ in range(60):
        middle = (low + high) / 2
        low, high = (middle, high) if circles_iou(middle) > iou else (low, middle)
    return (low + high) / 2


def svg_circles(cx_left, cx_right, cy, r, overlap):
    """Blue and orange circles; their lens-shaped intersection, if they overlap, in green."""
    shapes = [f'<circle cx="{cx_left:.1f}" cy="{cy}" r="{r}" fill="{IOU_BLUE}"/>',
              f'<circle cx="{cx_right:.1f}" cy="{cy}" r="{r}" fill="{IOU_ORANGE}"/>']
    if overlap:
        middle, half = (cx_left + cx_right) / 2, math.sqrt(r * r - ((cx_right - cx_left) / 2) ** 2)
        shapes.append(f'<path d="M {middle:.1f} {cy - half:.1f} A {r} {r} 0 0 1 {middle:.1f} {cy + half:.1f} '
                      f'A {r} {r} 0 0 1 {middle:.1f} {cy - half:.1f} Z" fill="{IOU_GREEN}"/>')
    return "".join(shapes)


def iou_diagram_svg():
    r, cy, gap = 75, 118, 56
    left = (380 - (2 * r + gap)) / 2 + r
    right = left + gap
    edge_left, edge_right = left - r, right + r
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 380 250" width="380" role="img" '
            f'aria-label="Two overlapping circles: the classical mask, the SAM2 mask and their intersection, with the union bracketed below">'
            f'{svg_circles(left, right, cy, r, True)}'
            f'<g fill="currentColor" font-size="14" font-weight="600" text-anchor="middle">'
            f'<text x="{left - 22:.1f}" y="30">Classical Mask</text><text x="{right + 22:.1f}" y="30">SAM2 Mask</text>'
            f'<text x="{(left + right) / 2:.1f}" y="{cy + 4}" font-size="12" fill="{IOU_INK}">Intersection</text>'
            f'<text x="190" y="238">Union — combined foreground area</text></g>'
            f'<path d="M {edge_left:.1f} 202 v10 H {edge_right:.1f} v-10" fill="none" stroke="currentColor" stroke-width="2"/></svg>')


def iou_example_svg(value, text, kind):
    r, cy = 26, 34
    gap = {"none": 2 * r + 10, "partial": overlap_distance(0.5) * r, "same": 0}[kind]
    left = 75 - gap / 2
    right = 75 + gap / 2
    shapes = (f'<circle cx="75" cy="{cy}" r="{r}" fill="{IOU_GREEN}"/>' if kind == "same"
              else svg_circles(left, right, cy, r, kind == "partial"))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 112" width="150" role="img" aria-label="Example: IoU {value}, {text}">'
            f'{shapes}<g fill="currentColor" text-anchor="middle">'
            f'<text x="75" y="88" font-size="18" font-weight="700">{value}</text>'
            f'<text x="75" y="106" font-size="12">{text}</text></g></svg>')


IOU_EXAMPLES = (("0.0", "No overlap", "none"), ("0.5", "Partial overlap", "partial"), ("1.0", "Identical masks", "same"))


def mask_agreement_section(classical_mask, sam2_mask):
    """The Mask Agreement (IoU) box: the IoU of the two current foreground masks plus a static explanation of what it measures."""
    iou = thermal_mask_iou(classical_mask, sam2_mask)
    with st.container(border=True, key="m4_iou_box"):
        st.markdown(IOU_CSS, unsafe_allow_html=True)
        st.markdown("##### Mask Agreement (IoU)")
        st.markdown(f"### {iou:.4f}" if iou is not None else "### N/A")
        if iou is None:
            st.caption("IoU is undefined because both masks contain no foreground.")
        st.write("Intersection over Union (IoU) measures how closely the foreground masks from the classical method and SAM2 overlap.")
        with st.container(key="m4_iou_art"):
            st.html(iou_diagram_svg())
            st.latex(r"\mathrm{IoU} = \dfrac{\text{Intersection}}{\text{Union}}")
            st.caption("Illustration of mask overlap.")
            for column, (value, text, kind) in zip(st.columns(3), IOU_EXAMPLES):
                with column:
                    st.html(iou_example_svg(value, text, kind))
        st.caption("Values closer to 1 indicate greater agreement between the two masks, not necessarily greater segmentation accuracy.")


# Thermal visitor page: black outlines and rectangle markers stand out against the false-color palette.
THERMAL_MARK = (0, 0, 0)
THERMAL_BOUNDARY_THICKNESS = 5
THERMAL_CSS = """
<style>
/* this page only: keep the click image and the completed preview centered whatever the width of the element wrapper */
.st-key-m4_rect_view [data-testid="stElementContainer"] { width: 100%; display: flex; justify-content: center; }
.st-key-m4_rect_view iframe, .st-key-m4_rect_view img { display: block; margin-left: auto; margin-right: auto; }
</style>
"""


def visitor_centered(image_array, caption):
    """One tall image at a moderate width, centered (as on the RGB visitor page)."""
    with st.columns([1, 1.2, 1])[1]:
        st.image(image_array, caption=caption, width="stretch")


def visitor_pair():
    """Two equal columns for side-by-side images, kept at a moderate size on the wide page."""
    return st.columns([1, 4, 4, 1])[1:3]


def hue_text(ranges):
    return " and ".join(f"{lower[0]}–{upper[0]}" for lower, upper in ranges)


def render_thermal():
    ss = st.session_state
    ss.setdefault("module4_q2_upload_generation", 0)
    ss.setdefault("module4_q2_points", [])
    ss.setdefault("module4_q2_click_generation", 0)
    st.markdown(VISITOR_CSS + THERMAL_CSS, unsafe_allow_html=True)

    # ---- everything the step states depend on is worked out first, from the stored upload, rectangle and results
    image_bytes = ss.get("module4_q2_bytes")
    digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes is not None else None
    if ss.get("module4_q2_digest") != digest:
        clear_thermal_results()
        reset_thermal_rectangle()
        ss["module4_q2_digest"] = digest
    image = original_shape = preparation_error = prepared_identity = None
    if image_bytes is not None:
        image, original_shape, preparation_error = prepared_thermal_upload(ss, image_bytes, digest)
        prepared_identity = ss["module4_q2_prepared"][0]
    if ss.get("module4_q2_processing_identity") != prepared_identity:
        clear_thermal_results()
        reset_thermal_rectangle()
        ss["module4_q2_processing_identity"] = prepared_identity
    resized = image is not None and image.shape != original_shape

    points = ss["module4_q2_points"]
    rectangle = rectangle_error = None
    if image is not None and len(points) == 2:
        left, right = sorted((points[0][0], points[1][0]))
        top, bottom = sorted((points[0][1], points[1][1]))
        try:
            rectangle = validate_rectangle((left, top, right - left, bottom - top), image.shape)
        except ValueError as error:
            rectangle_error = str(error)

    # The classical method needs only the processing image; SAM2 needs the image and its own rectangle.
    classical_inputs = (prepared_identity, tuple(image.shape) if image is not None else None)
    sam2_inputs = (classical_inputs, rectangle, sam2_configuration())
    if ss.get("module4_q2_sam2_inputs") != sam2_inputs:
        clear_thermal_results("sam2")
        ss["module4_q2_sam2_inputs"] = sam2_inputs
    results = ss.get("module4_q2_results")
    if results is not None and ss.get("module4_q2_classical_identity") != classical_inputs:
        clear_thermal_results("classical")
        results = None
    sam2_results = ss.get("module4_q2_sam2_results")
    comparison_ready = (
        image is not None and rectangle is not None and results is not None and sam2_results is not None
        and ss.get("module4_q2_sam2_identity") == sam2_inputs
    )
    compared = comparison_ready and ss.get("module4_q2_comparison") == (classical_inputs, sam2_inputs)
    state = {
        1: "done" if image is not None else "active",
        2: "locked" if image is None else "done" if results is not None else "active",
        3: "locked" if results is None else "done" if sam2_results is not None else "active",
        4: "locked" if not comparison_ready else "done" if compared else "active",
    }

    def process_classical():
        clear_thermal_results("classical")
        try:
            with st.spinner("Extracting the thermal human boundary…"):
                result = find_thermal_boundary(image)
                # Retain display stages and metrics; the upload preserves the original.
                for key in ("original_image", "hsv_image", "closed_mask"):
                    result.pop(key)
                # Display only: the same contour, drawn in black with the page's boundary thickness instead of the library's magenta.
                result["boundary_overlay"] = image.copy()
                cv2.drawContours(result["boundary_overlay"], [result["contour"]], -1, THERMAL_MARK, THERMAL_BOUNDARY_THICKNESS)
                ss["module4_q2_results"] = result
                ss["module4_q2_classical_identity"] = classical_inputs
        except ValueError as error:
            ss["module4_q2_error"] = str(error)
        except cv2.error:
            logging.getLogger(__name__).exception("Module 4 thermal processing failed")
            ss["module4_q2_error"] = "Thermal processing could not complete. Please try another image."

    def process_sam2():
        clear_thermal_results("sam2")
        try:
            from Modules.Module_4.sam2_comparison import run_sam2, select_device

            source, config, requested_device, _ = sam2_inputs[2]
            device = select_device(requested_device)
            check_image_budget(image.shape, device, cpu_pixel_limit())
            with st.spinner("Preparing SAM2 model…"):
                resource = current_sam2_model(source, config, device)
            with st.spinner("Segmenting the thermal person with SAM2…"):
                result = run_sam2(image, rectangle, resource)
                # Presentation only: preserve every SAM2 mask pixel/component.
                contours, _ = cv2.findContours(result["binary_mask"].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result["boundary_overlay"] = image.copy()
                cv2.drawContours(result["boundary_overlay"], contours, -1, THERMAL_MARK, THERMAL_BOUNDARY_THICKNESS)
                result["metadata"] = {**result["metadata"], "overlay": "All external contours, black, thickness 5; no mask editing"}
                ss["module4_q2_sam2_results"] = result
                ss["module4_q2_sam2_identity"] = sam2_inputs
        except SAM2ResourceLimitError as error:
            ss["module4_q2_sam2_error"] = str(error)
        except Exception:
            logging.getLogger(__name__).exception("Module 4 thermal SAM2 inference failed")
            ss["module4_q2_sam2_error"] = "SAM2 could not complete. Please try again. The classical thermal method remains available."

    with st.container(key="m4_vis_headings"):
        st.title("Human Boundary Detection in Thermal Images", anchor=False)
        st.header("Upload a thermal image and compare human boundary detection using classical computer vision and SAM2.",
                  anchor=False)

    with st.container(key="m4_vis_body"):
        # ---------------------------------------------------------------- Step 1
        with st.container(border=True, key="m4_stepbox_1"):
            with st.container(key=f"m4_stmark_1_{state[1]}"):
                step_head(1, state[1], THERMAL_STEP_TITLES)
            st.write("Upload a false-color thermal image of one person, preferably with warm body colors against a blue or purple "
                     "background.")
            st.file_uploader(
                "Upload Thermal Image", type=["jpg", "jpeg", "png"],
                key=f"module4_q2_upload_{ss['module4_q2_upload_generation']}", on_change=update_thermal_upload,
            )
            st.caption(f"Supported formats: JPG, JPEG and PNG. Maximum upload size: {st.get_option('server.maxUploadSize')} MB per file.")
            if image_bytes is not None and ss.get(f"module4_q2_upload_{ss['module4_q2_upload_generation']}") is None:
                st.caption(f"Loaded thermal image: {ss.get('module4_q2_filename', 'previous upload')}")
                st.button("Clear Thermal Image", key="module4_q2_clear_image", on_click=reset_thermal_upload)
            if preparation_error:
                st.error(preparation_error)
            if image is not None:
                visitor_centered(image, "Thermal Image for Processing" if resized else "Original Thermal Image")
                if resized:
                    st.caption(
                        f"Thermal image resized from {original_shape[1]} × {original_shape[0]} to "
                        f"{image.shape[1]} × {image.shape[0]} pixels for Classical and SAM2 processing."
                    )
                else:
                    st.caption(f"Original and processing size: {image.shape[1]} × {image.shape[0]} pixels.")

        # ---------------------------------------------------------------- Step 2
        with st.container(border=True, key="m4_stepbox_2"):
            with st.container(key=f"m4_stmark_2_{state[2]}"):
                step_head(2, state[2], THERMAL_STEP_TITLES)
            if state[2] != "locked":
                st.write("The classical method identifies the largest warm-colored region in the thermal image and extracts its "
                         "external boundary.")
                st.write("Unlike the RGB classical method, which uses a rectangle to guide GrabCut segmentation, the thermal "
                         "classical method identifies the person directly using warm-colored regions.")
                with st.expander("View Processing Parameters", expanded=False):
                    results_table("Parameter", [
                        ("HSV hue range", hue_text(HSV_RANGES)),
                        ("Minimum saturation", str(HSV_RANGES[0][0][1])),
                        ("Minimum value", str(HSV_RANGES[0][0][2])),
                        ("Morphological closing", f"{CLOSING_KERNEL_SIZE} × {CLOSING_KERNEL_SIZE} elliptical kernel"),
                        ("Closing iterations", str(CLOSING_ITERATIONS)),
                        ("Connected components", "8-connectivity")])
                    st.caption("These parameters are fixed; they are not adjustable settings.")
                if st.button("Apply Classical Method", key="module4_q2_run", type="primary"):
                    process_classical()
                    if ss.get("module4_q2_results") is not None:
                        st.rerun()
                if ss.get("module4_q2_error"):
                    st.error(ss["module4_q2_error"])
                if results is not None:
                    st.markdown("##### HSV Thresholding & Human Segmentation")
                    initial_column, cleaned_column = visitor_pair()
                    with initial_column:
                        st.image(results["initial_hsv_mask"], caption="Initial HSV Mask", width="stretch")
                    with cleaned_column:
                        st.image(results["cleaned_human_mask"], caption="Cleaned Human Mask", width="stretch")
                    st.write("HSV thresholding identifies warm-colored regions. Morphological closing fills small gaps, and "
                             "connected-component analysis retains the largest foreground region.")
                    st.write("White represents the foreground, and black represents the background.")
                    st.markdown("##### Final Human Boundary")
                    visitor_centered(results["boundary_overlay"], "Classical Human Boundary")
                    st.write("The human boundary is extracted from the cleaned mask and displayed in black.")
                    st.button("Reset Classical Method", key="module4_q2_reset_classical",
                              on_click=lambda: clear_thermal_results("classical"))

        # ---------------------------------------------------------------- Step 3
        with st.container(border=True, key="m4_stepbox_3"):
            with st.container(key=f"m4_stmark_3_{state[3]}"):
                step_head(3, state[3], THERMAL_STEP_TITLES)
            if state[3] != "locked":
                st.write("SAM2 uses the selected rectangle to segment the person from the background.")
                st.markdown("##### Select Person Rectangle")
                with st.expander("How to select the rectangle", expanded=False):
                    st.write("Click two opposite corners to create a rectangle around the person. Select the first corner above and "
                             "to the left of the person, then select the second corner below and to the right.")
                    st.write("Include the entire person, with a small margin around the body. Leave some background outside the rectangle.")
                    with st.columns([1, 1.5, 1])[1]:
                        st.image(rectangle_example(THERMAL_EXAMPLE_IMAGE, THERMAL_EXAMPLE_CORNERS, THERMAL_MARK),
                                 caption="Select the first corner, then the second corner.", width="stretch")
                if rectangle_error:
                    st.warning(f"{rectangle_error} Use Reset Rectangle to select again.")
                if rectangle is not None:
                    st.caption(f"Rectangle (x, y, width, height): {rectangle} pixels")
                height, width = image.shape[:2]
                # Preview clicks map to the shared processing image used for inference.
                scale = min(1.0, 700 / width, 700 / height)
                preview = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
                for x, y in points:
                    cv2.circle(preview, (round(x * preview.shape[1] / width), round(y * preview.shape[0] / height)), 5, THERMAL_MARK, -1)
                if len(points) == 2:
                    if rectangle is not None:
                        left, right = sorted((points[0][0], points[1][0]))
                        top, bottom = sorted((points[0][1], points[1][1]))
                        cv2.rectangle(preview, (round(left * scale), round(top * scale)),
                                      (round((right - 1) * scale), round((bottom - 1) * scale)), THERMAL_MARK, 2)
                    with st.container(key="m4_rect_view"):
                        st.image(preview, caption="Selected corners and rectangle on the thermal processing image", width="content")
                else:
                    st.info("Select the first corner." if not points else "Select the second corner.")
                    component_key = f"module4_q2_click_{ss['module4_q2_click_generation']}"
                    with st.container(key="m4_rect_view"):
                        streamlit_image_coordinates(
                            preview, width="content", cursor="crosshair", key=component_key,
                            on_click=lambda: record_thermal_corner(component_key, image.shape),
                        )
                st.button("Reset Rectangle", key="module4_q2_reset_rectangle", on_click=reset_thermal_rectangle, disabled=not points)
                if rectangle is not None:
                    if st.button("Run SAM2", key="module4_q2_run_sam2", type="primary"):
                        process_sam2()
                        if ss.get("module4_q2_sam2_results") is not None:
                            st.rerun()
                if ss.get("module4_q2_sam2_error"):
                    st.error(ss["module4_q2_sam2_error"])
                if sam2_results is not None:
                    mask_column, boundary_column = visitor_pair()
                    with mask_column:
                        st.image(sam2_results["binary_mask"], caption="SAM2 Binary Mask", width="stretch")
                    with boundary_column:
                        st.image(sam2_results["boundary_overlay"], caption="SAM2 Human Boundary", width="stretch")
                    metadata = sam2_results["metadata"]
                    if metadata["empty_mask"]:
                        st.info("SAM2 predicted no foreground. Try another rectangle.")
                if sam2_results is not None or ss.get("module4_q2_sam2_error"):
                    st.button("Reset SAM2", key="module4_q2_reset_sam2", on_click=lambda: clear_thermal_results("sam2"))

        # ---------------------------------------------------------------- Step 4
        with st.container(border=True, key="m4_stepbox_4"):
            with st.container(key=f"m4_stmark_4_{state[4]}"):
                step_head(4, state[4], THERMAL_STEP_TITLES)
            if state[4] != "locked":
                if st.button("Compare Results", key="module4_q2_compare", type="primary"):
                    ss["module4_q2_comparison"] = (classical_inputs, sam2_inputs)
                    st.rerun()
                if compared:
                    st.markdown("##### Foreground Mask Comparison")
                    classical_column, sam2_column = visitor_pair()
                    with classical_column:
                        st.image(results["cleaned_human_mask"], caption="Classical Method — Cleaned Human Mask", width="stretch")
                    with sam2_column:
                        st.image(sam2_results["binary_mask"], caption="SAM2 — Foreground Mask", width="stretch")
                    st.markdown("##### Human Boundary Comparison")
                    classical_column, sam2_column = visitor_pair()
                    with classical_column:
                        st.image(results["boundary_overlay"], caption="Classical Human Boundary", width="stretch")
                    with sam2_column:
                        st.image(sam2_results["boundary_overlay"], caption="SAM2 Human Boundary", width="stretch")
                    mask_agreement_section(results["cleaned_human_mask"], sam2_results["binary_mask"])
                    st.markdown("##### Observations & Limitations")
                    st.markdown(
                        "- The classical method relies on displayed thermal colors, not actual temperature measurements. Other "
                        "warm-colored objects may affect segmentation.\n"
                        "- SAM2 results depend on image quality, background complexity, and rectangle selection.\n"
                        "- Measuring actual segmentation accuracy requires a ground-truth mask.")

        st.button("Reset Experiment", key="module4_q2_reset", on_click=reset_thermal_upload)


ACCENT = "#C9363E"
HOME_TITLE = "Module 4"
HOME_SUBTITLE = "Human Boundary Detection"

BLUE, BLUE_DARK = "#1D5FD1", "#174DA8"   # same blue as Modules 2 and 7; white text on these is above 6:1
# Homepage and My Experiments & Results page: stacked, centered buttons using Module 2's finalized values. The three homepage buttons and
# the two selection buttons all use the same fixed 110px height, so labels that wrap onto two lines never change a button's size.
NAV_BUTTON_KEYS = ("m4_open_results", "m4_open_rgb", "m4_open_thermal", "m4_res_rgb", "m4_res_thermal")


def nav_selector(suffix):
    return ", ".join(f".st-key-{key}{suffix}" for key in NAV_BUTTON_KEYS)


HOME_CSS = f"""
<style>
.st-key-m4_home_layout {{ gap: 0; padding-top: 0; }}  /* the nav row above already supplies the top space */
.st-key-m4_home_title, .st-key-m4_home_subtitle {{ gap: 0; flex-shrink: 0; }}
.st-key-m4_home_nav {{ flex-shrink: 0; }}
.st-key-m4_home_title [data-testid="stMarkdownContainer"],
.st-key-m4_home_subtitle [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.st-key-m4_home_subtitle {{ padding-top: 44px; }}
.st-key-m4_home_nav {{ padding-top: 64px; }}
.st-key-m4_home_title h1 {{
    font-size: clamp(2.5rem, 13vw, 80px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}}
.st-key-m4_home_subtitle h2 {{
    font-size: clamp(1.1rem, 4.6vw, 32px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2;
    text-align: center; padding: 0; margin: 0;
}}
@media (min-width: 801px) {{ .st-key-m4_home_subtitle h2 {{ white-space: nowrap; }} }}
.st-key-m4_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 24px; max-width: 560px; width: 100%; margin: 0 auto; }}
{nav_selector("")}, {nav_selector(' [data-testid="stButton"]')} {{ flex: 0 0 auto; width: 100%; min-width: 0; max-width: 100%; }}
{nav_selector(" button")} {{
    width: 100%; height: 110px; min-height: 110px; max-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
{nav_selector(" button p")} {{
    font-size: 18px; font-weight: 500; color: #FFFFFF; line-height: 1.3; text-align: center; text-wrap: balance;
}}
{nav_selector(" button p strong")} {{ display: block; font-size: 21px; font-weight: 700; line-height: 1.3; color: #FFFFFF; }}
{nav_selector(" button p br")} {{ display: none; }}
.st-key-m4_res_rgb button p, .st-key-m4_res_thermal button p {{ font-size: 0.95rem; }}   /* the "(Question N)" line */
{nav_selector(" button:hover")} {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
{nav_selector(" button:focus-visible")} {{
    outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT};
}}
{nav_selector(" button:active")} {{ background-color: #8E2229; border-color: #8E2229; }}
.st-key-m4_open_results button {{ background-color: {BLUE}; border-color: {BLUE}; }}
.st-key-m4_open_results button:hover {{ background-color: {BLUE_DARK}; border-color: {BLUE_DARK}; }}
.st-key-m4_open_results button:active {{ background-color: #123F8F; border-color: #123F8F; }}
.st-key-m4_open_results button:focus-visible {{ box-shadow: 0 0 0 6px {BLUE}; }}
@media (max-width: 640px) {{ .st-key-m4_home_nav {{ gap: 16px; }} }}
</style>
"""


RESULTS_TITLE_CSS = """
<style>
.st-key-m4_home_title h1 { font-size: clamp(1.75rem, 7vw, 48px); }   /* My Experiments & Results page only */
@media (min-width: 801px) { .st-key-m4_home_title h1 { white-space: nowrap; } }
</style>
"""


def home_layout(title, subtitle, buttons, extra_css=""):
    """Centered title, subtitle and stacked buttons; (label, key, destination) triples. Shared by the homepage and the results page."""
    st.markdown(HOME_CSS + extra_css, unsafe_allow_html=True)
    with st.container(key="m4_home_layout"):
        with st.container(key="m4_home_title"):
            st.title(title, anchor=False)
        with st.container(key="m4_home_subtitle"):
            st.header(subtitle, anchor=False)
        with st.container(key="m4_home_nav"):
            for label, key, destination in buttons:
                st.button(label, key=key, on_click=navigate_module4, args=(destination,), width="stretch")


def question_label(title, number):
    """Button label: the title (bold block) and a smaller "(Question N)" line, both inside the one clickable button."""
    return f"**{title}**  \n(Question {number})"


DISPLAY_DIR = REPOSITORY_ROOT / "Modules" / "Module_4" / "display"

# Recorded values of my RGB experiment.
RGB_EXPERIMENT = {
    "kernel": 3, "sigma": "0.80", "canny": "50 / 150", "iterations": 5, "cleanup": "Disabled",
    "rectangle": "1766 × 5620 pixels", "contour_area": "6,574,847.0 pixels²",
    "sam2_score": "0.989319", "sam2_device": "MPS",
}

# Results-page layout, mirroring Module 2's finalized results page (RESULTS_CSS in Pages/app_module2.py) under Module 4 keys.
RGB_RESULTS_CSS = """
<style>
.st-key-m4_res_head { gap: 0; padding-bottom: 28px; }
.st-key-m4_res_head h1 {
    font-size: clamp(2.5rem, 8vw, 60px); font-weight: 800; line-height: 1.1; text-align: center; padding: 0; margin: 0;
    color: inherit; overflow-wrap: anywhere;
}
.st-key-m4_res_head h2 {
    font-size: clamp(1.4rem, 4.6vw, 2.4rem); font-weight: 600; line-height: 1.2; text-align: center; padding: 0; margin: 0;
    color: inherit; opacity: 0.85; overflow-wrap: anywhere;
}
.st-key-m4_res_title { padding-bottom: 28px; }
.st-key-m4_res_body h2 { text-align: center; font-size: clamp(1.7rem, 5.2vw, 2.4rem); font-weight: 800; line-height: 1.2; padding: 1.5rem 0 0.25rem; overflow-wrap: anywhere; }
.st-key-m4_res_body h3 { text-align: center; font-size: clamp(1.4rem, 4.5vw, 1.9rem); font-weight: 700; padding-top: 1.5rem; }
.st-key-m4_res_body h5 { text-align: left; font-weight: 700; margin: 0; padding: 0.5rem 0 0.25rem; }
.st-key-m4_res_body [data-testid="stTable"] { overflow-x: auto; }
.st-key-m4_res_body img { max-width: 100%; height: auto; }
[class*="st-key-m4_res_sec_"] { border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent); margin-top: 1.5rem; padding-top: 0.5rem; }
.st-key-m4_res_conclusion {
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 16px; padding: 0.5rem 28px 28px; margin-top: 2rem;
}
@media (max-width: 640px) { .st-key-m4_res_conclusion { padding: 0.25rem 14px 18px; } }
/* Light theme only (Streamlit follows prefers-color-scheme): every gray text on this page uses the primary body text color; sizes,
   weights and spacing are untouched. The red Home accent is not affected. */
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) { --nav-muted: #31333F; }  /* intermediate breadcrumb labels and separators */
    :root:not([data-theme="dark"]) .st-key-m4_res_head h2,
    :root:not([data-theme="dark"]) .st-key-m4_res_body :is([data-testid="stCaptionContainer"], [data-testid="stImageCaption"],
        [data-testid="stCaptionContainer"] *, [data-testid="stImageCaption"] *, [data-testid="stTable"] th, [data-testid="stTable"] td) {
        color: #31333F !important; opacity: 1;
    }
}
</style>
"""


SURROUND_BGR = (23, 17, 13)  # the dark page-colored frame around the screenshot-style outputs (SAM2 outputs and comparisons)


@st.cache_data(show_spinner=False)
def framed_output(name):
    """Display-only copy of a saved output that is a dark-framed screenshot with its own caption: the original file is never changed.
    Each picture (a composite yields two, left to right) is cropped out of the frame, so the embedded gray caption is dropped, and the
    frame's rounded corners become transparent, so the picture sits on the page background in either theme. The picture's own pixels,
    including the black background of a binary mask, are kept as they are."""
    bgr = cv2.imread(str(DISPLAY_DIR / name), cv2.IMREAD_COLOR)
    outside = (np.abs(bgr.astype(int) - SURROUND_BGR).sum(axis=2) <= 12).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(1 - outside, connectivity=8)
    boxes = sorted((tuple(int(v) for v in stats[i][:4]) for i in range(1, count) if stats[i][4] >= 0.05 * outside.size))
    pictures = []
    for x, y, w, h in boxes:
        crop, corner = bgr[y:y + h, x:x + w], outside[y:y + h, x:x + w]
        edge_count, edge_labels = cv2.connectedComponents(corner, connectivity=4)
        touching = np.unique(np.concatenate([edge_labels[0], edge_labels[-1], edge_labels[:, 0], edge_labels[:, -1]]))
        transparent = np.isin(edge_labels, touching[touching > 0]) & (corner > 0)
        rgba = np.dstack([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), np.where(transparent, 0, 255).astype(np.uint8)])
        pictures.append(rgba)
    return pictures



def results_image(name, caption=None):
    """One saved image at the width of its column; images that already carry their own label get no extra caption."""
    st.image(str(DISPLAY_DIR / name), caption=caption, width="stretch")


def results_framed(name, caption, index=0):
    """One picture of a dark-framed screenshot output (see framed_output), shown with a theme-aware caption."""
    st.image(framed_output(name)[index], caption=caption, width="stretch", output_format="PNG")


def results_centered(show, *args):
    """A single tall image, kept to a moderate width so it does not dominate the page."""
    with st.columns([1, 1.6, 1])[1]:
        show(*args)


def results_table(index_name, rows, value_name="Value"):
    st.table(pd.DataFrame({index_name: [k for k, _ in rows], value_name: [v for _, v in rows]}).set_index(index_name))


def rgb_results_page():
    r = RGB_EXPERIMENT
    st.markdown(RGB_RESULTS_CSS, unsafe_allow_html=True)

    with st.container(key="m4_res_head"):
        with st.container(key="m4_res_title"):
            st.title("My Experiment & Results", anchor=False)
        st.header("Human Boundary Detection in RGB Images", anchor=False)

    with st.container(key="m4_res_body"):
        st.markdown("##### Objective")
        st.write("Detect the human boundary in an RGB image using classical computer vision techniques and compare the result with SAM2.")

        with st.container(key="m4_res_sec_1"):
            st.header("Step 1 — Original RGB Image & Experimental Parameters", anchor=False)
            image_column, table_column = st.columns(2, vertical_alignment="center")
            with image_column:
                results_image("step1_original.jpg", "Original RGB Image")
            with table_column:
                results_table("Parameter", [
                    ("Gaussian kernel", f"{r['kernel']} × {r['kernel']}"), ("Gaussian sigma", r["sigma"]),
                    ("Canny thresholds", r["canny"]), ("GrabCut iterations", str(r["iterations"])),
                    ("Morphological cleanup", r["cleanup"]), ("GrabCut rectangle", r["rectangle"])])
                st.caption("Only the size of the GrabCut rectangle was recorded, so its position is not drawn on the image.")

        with st.container(key="m4_res_sec_2"):
            st.header("Step 2 — Gaussian Smoothing", anchor=False)
            st.write(
                f"Gaussian smoothing reduces noise and small intensity variations before edge detection. A {r['kernel']} × {r['kernel']} "
                f"kernel with a sigma of {r['sigma']} was used, so the change is subtle by design.")
            original_column, smoothed_column = st.columns(2)
            with original_column:
                results_image("step1_original.jpg", "Original RGB Image")
            with smoothed_column:
                results_image("step2_gaussian_reproduced.jpg", "Gaussian-Smoothed Image (regenerated from the same kernel and sigma)")

        with st.container(key="m4_res_sec_3"):
            st.header("Step 3 — Canny Edge Detection", anchor=False)
            st.write(
                "Canny detects strong intensity changes along the body, clothing, hair and face. It also marks internal and "
                "background edges and does not know which edges form the outer boundary, so edge detection alone cannot give a "
                "single human boundary.")
            results_centered(results_image, "step3_canny.jpg", "Canny Edge Map")

        with st.container(key="m4_res_sec_4"):
            st.header("Step 4 — GrabCut Segmentation", anchor=False)
            st.write(
                f"GrabCut separates the foreground from the background using the selected rectangle. {r['iterations']} iterations were used "
                "and morphological cleanup was disabled, so the foreground mask is the raw GrabCut result. In the binary mask, white "
                "is the foreground and black is the background.")
            results_centered(results_framed, "step4_grabcut_mask.png", "GrabCut Binary Mask")

        with st.container(key="m4_res_sec_5"):
            st.header("Step 5 — Classical Human Boundary Detection", anchor=False)
            st.write(
                "The largest external contour of the foreground mask was extracted as the human boundary. The contour follows much of "
                "the visible person, but it also includes a connected cast shadow near the lower body, which is not part of the person.")
            results_centered(results_framed, "step5_classical_boundary.png", "Classical Human Boundary")
            results_table("Measurement", [("Classical contour area", r["contour_area"])])
            st.caption("The contour area is the geometric area enclosed by the contour, not a count of foreground pixels.")

        with st.container(key="m4_res_sec_6"):
            st.header("Step 6 — SAM2 Segmentation & Boundary Detection", anchor=False)
            st.write(
                "SAM2 is a deep learning segmentation model, used here for comparison. SAM 2.1 Hiera Tiny was applied to the same RGB "
                "image with a bounding-box prompt around the person.")
            mask_column, boundary_column = st.columns(2)
            with mask_column:
                results_framed("step6_sam2_mask.png", "SAM2 Binary Mask")
            with boundary_column:
                results_framed("step6_sam2_boundary.png", "SAM2 Human Boundary")
            results_table("Parameter", [("Model", "SAM 2.1 Hiera Tiny"), ("Prompt", "Bounding box"), ("Device", r["sam2_device"]),
                                ("Predicted mask-quality score", r["sam2_score"])])
            st.caption(
                "The predicted mask-quality score is SAM2's own estimate of its mask, not a measured segmentation accuracy; no "
                "ground-truth mask was used. The device is the one used when this experiment was run on my computer.")

        with st.container(key="m4_res_sec_7"):
            st.header("Step 7 — Classical vs. SAM2 Comparison", anchor=False)
            st.subheader("Mask Comparison", anchor=False)
            for column, caption, index in zip(st.columns(2), ("Classical Binary Mask (GrabCut)", "SAM2 Binary Mask"), (0, 1)):
                with column:
                    results_framed("step7_mask_comparison.png", caption, index)
            st.subheader("Boundary Comparison", anchor=False)
            for column, caption, index in zip(st.columns(2), ("Classical Human Boundary", "SAM2 Human Boundary"), (0, 1)):
                with column:
                    results_framed("step7_boundary_comparison.png", caption, index)
            st.write(
                "In this experiment the classical method includes a large part of the connected cast shadow, in both its mask and its "
                "contour. SAM2 follows the visible human shape more closely, especially around the lower body, and isolates the "
                "person without the shadow.")

        with st.container(key="m4_res_conclusion"):
            st.header("Conclusion", anchor=False)
            st.markdown(
                "The classical approach detected much of the human boundary, but it was affected by the difficulty of separating "
                "foreground from background and by the connected cast shadow. SAM2 produced a cleaner human boundary and separated the "
                "person from the background more effectively in this experiment.")
            st.markdown(
                "These results show that classical human boundary detection is sensitive to scene characteristics and to how well the "
                "foreground can be separated from the background.")


# Recorded values of my thermal experiment.
THERMAL_EXPERIMENT = {
    "size": "1080 × 1440 pixels", "closing": "3 × 3 elliptical kernel", "connectivity": "8-connectivity",
    "components": "20", "foreground": "392,206 pixels", "occupancy": "25.22%",
    "sam2_box_xy": "(263, 57) pixels", "sam2_box_size": "574 × 1380 pixels", "sam2_score": "0.977383", "sam2_device": "MPS",
    "iou": "0.9813",
}


def thermal_results_page():
    r = THERMAL_EXPERIMENT
    st.markdown(RGB_RESULTS_CSS, unsafe_allow_html=True)  # the same results-page layout and light-theme text colors as the RGB page

    with st.container(key="m4_res_head"):
        with st.container(key="m4_res_title"):
            st.title("My Experiment & Results", anchor=False)
        st.header("Human Boundary Detection in Thermal Images", anchor=False)

    with st.container(key="m4_res_body"):
        st.markdown("##### Objective")
        st.write("Detect the human boundary in a thermal image using classical computer vision techniques and compare the results with SAM2.")

        with st.container(key="m4_res_sec_1"):
            st.header("Step 1 — Original Thermal Image & Experimental Parameters", anchor=False)
            image_column, table_column = st.columns(2, vertical_alignment="center")
            with image_column:
                results_image("thermal_step1_original.jpg", "Original Thermal Image")
            with table_column:
                results_table("Parameter", [
                    ("Image resolution", r["size"]), ("Color space", "HSV"), ("Segmentation method", "Color thresholding"),
                    ("Morphological closing", r["closing"]), ("Connected components", r["connectivity"]),
                    ("Region selection", "Largest foreground component")])
            st.write(
                "The thermal image uses colors to represent temperature differences. The classical method uses these colors to "
                "separate the person from the background.")

        with st.container(key="m4_res_sec_2"):
            st.header("Step 2 — Classical Thermal Segmentation", anchor=False)
            st.write(
                "The thermal image was converted to HSV color space to separate warm-colored regions from the cooler background. "
                "Morphological closing filled small gaps in the mask, and the largest connected foreground region was selected as "
                "the person.")
            st.write("In the binary mask, white represents the person and black represents the background.")
            results_centered(results_framed, "thermal_step4_mask_comparison.png", "Classical Binary Mask", 0)
            results_table("Measurement", [("Foreground components found", r["components"]),
                                          ("Selected foreground size", r["foreground"]), ("Image occupancy", r["occupancy"])], "Result")
            st.caption("Image occupancy represents the percentage of the image occupied by the selected foreground.")

        with st.container(key="m4_res_sec_3"):
            st.header("Step 3 — Classical Human Boundary Detection", anchor=False)
            st.write(
                "The outer contour of the foreground mask was extracted and drawn on the original thermal image to identify the "
                "human boundary.")
            results_centered(results_framed, "thermal_step5_boundary_comparison.png", "Classical Human Boundary", 0)

        with st.container(key="m4_res_sec_4"):
            st.header("Step 4 — SAM2 Segmentation & Boundary Detection", anchor=False)
            st.write(
                "SAM2 is a deep learning segmentation model. A bounding box around the person was used to guide SAM2 in generating "
                "the human mask and boundary.")
            mask_column, boundary_column = st.columns(2)
            with mask_column:
                results_framed("thermal_step4_mask_comparison.png", "SAM2 Binary Mask", 1)
            with boundary_column:
                results_framed("thermal_step5_boundary_comparison.png", "SAM2 Human Boundary", 1)
            results_table("Parameter", [
                ("Prompt", "Bounding box"), ("Bounding box (x, y)", r["sam2_box_xy"]),
                ("Bounding box (width × height)", r["sam2_box_size"]), ("Experimental device", r["sam2_device"]),
                ("Predicted mask-quality score", r["sam2_score"])])
            st.caption("The predicted mask-quality score represents SAM2's estimated segmentation quality.")

        with st.container(key="m4_res_sec_5"):
            st.header("Step 5 — Classical vs. SAM2 Comparison", anchor=False)
            st.subheader("Mask Comparison", anchor=False)
            for column, caption, index in zip(st.columns(2), ("Classical Binary Mask", "SAM2 Binary Mask"), (0, 1)):
                with column:
                    results_framed("thermal_step4_mask_comparison.png", caption, index)
            st.write("Both methods produced similar masks, with minor differences around the head, hands, waist, and lower body.")
            st.subheader("Boundary Comparison", anchor=False)
            for column, caption, index in zip(st.columns(2), ("Classical Human Boundary", "SAM2 Human Boundary"), (0, 1)):
                with column:
                    results_framed("thermal_step5_boundary_comparison.png", caption, index)
            st.write("Both methods closely followed the visible outline of the person, producing similar human boundaries.")

        with st.container(key="m4_res_sec_6"):
            st.header("Step 6 — Quantitative Comparison", anchor=False)
            st.write("Intersection over Union (IoU) measures how closely two segmentation masks overlap.")
            st.latex(r"\mathrm{IoU} = \dfrac{|A \cap B|}{|A \cup B|}")
            st.markdown(
                "- **A** is the classical foreground mask.\n"
                "- **B** is the SAM2 foreground mask.\n"
                "- The numerator is the number of pixels shared by both masks.\n"
                "- The denominator is the total number of pixels belonging to either mask.")
            results_table("Measurement", [("Intersection over Union (IoU)", r["iou"])], "Result")
            st.write(
                "An IoU of 1.0 represents perfect overlap. The experimental result of 0.9813 indicates very high agreement between "
                "the two methods.")

        with st.container(key="m4_res_conclusion"):
            st.header("Conclusion", anchor=False)
            st.markdown(
                "The classical computer vision method successfully detected the human boundary in the thermal image. The strong color "
                "contrast between the person and the background made color-based segmentation effective.")
            st.markdown(
                "Both the classical method and SAM2 produced similar results, with an IoU of 0.9813 indicating high agreement between "
                "their segmentation masks.")
            st.markdown(
                "This experiment demonstrates that classical computer vision techniques can effectively detect human boundaries when "
                "the person and background are clearly distinguishable.")


st.set_page_config(page_title="Module 4 - Human Boundary Detection", layout="centered")
st.session_state.setdefault("module4_section", "home")

# Keep RGB settings when their widgets are absent from the selected section.
for setting_key in (
    "module4_gaussian_kernel", "module4_gaussian_sigma", "module4_canny_lower",
    "module4_canny_upper", "module4_grabcut_iterations", "module4_cleanup_enabled",
    "module4_morphology_kernel",
):
    if st.session_state["module4_section"] != "rgb" and setting_key in st.session_state:
        st.session_state[setting_key] = st.session_state[setting_key]

section = st.session_state["module4_section"]
# breadcrumb levels below Home for every view: (label, section); the last is the current page, the one before it is the Back target
_M4, _RESULTS = ("Module 4", "home"), ("My Experiments & Results", "my_results")
CRUMBS = {"home": (_M4,), "my_results": (_M4, _RESULTS),
          "rgb": (_M4, ("Detect Human Boundaries in Your Own RGB Image", "rgb")),
          "thermal": (_M4, ("Detect Human Boundaries in Your Own Thermal Image", "thermal")),
          "rgb_results": (_M4, _RESULTS, ("Human Boundary Detection in RGB Images", "rgb_results")),
          "thermal_results": (_M4, _RESULTS, ("Human Boundary Detection in Thermal Images", "thermal_results"))}
if section not in CRUMBS:  # e.g. a session left on a removed selection page
    navigate_module4("home")
    section = "home"
nav_row(CRUMBS[section], navigate_module4, view=f"m4:{section}")
if section == "home":
    home_layout(HOME_TITLE, HOME_SUBTITLE, (
        ("View My Experiments & Results", "m4_open_results", "my_results"),
        ("Detect Human Boundaries in Your Own RGB Image", "m4_open_rgb", "rgb"),
        ("Detect Human Boundaries in Your Own Thermal Image", "m4_open_thermal", "thermal")))
elif section == "my_results":
    home_layout("My Experiments & Results", HOME_SUBTITLE, (
        (question_label("Human Boundary Detection in RGB Images", 1), "m4_res_rgb", "rgb_results"),
        (question_label("Human Boundary Detection in Thermal Images", 2), "m4_res_thermal", "thermal_results")),
        RESULTS_TITLE_CSS)
elif section == "rgb_results":
    rgb_results_page()
elif section == "rgb":
    render_rgb()
elif section == "thermal_results":
    thermal_results_page()
elif section == "thermal":
    render_thermal()
