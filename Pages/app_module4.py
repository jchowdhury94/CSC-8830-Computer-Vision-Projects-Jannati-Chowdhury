"""Module 4 landing page and internally navigated assignment experiments."""

import hashlib
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Modules.Module_4.human_boundary import (
    find_human_boundary,
    normalize_image,
    validate_rectangle,
)
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
    reset_rectangle()


def clear_rgb_upload():
    st.session_state["module4_upload_generation"] += 1
    st.session_state.pop("module4_rgb_bytes", None)
    st.session_state.pop("module4_rgb_filename", None)
    reset_rectangle()


def navigate_module4(section):
    st.session_state["module4_section"] = section
    # Replace the click component when revisiting RGB so an old click cannot replay.
    if section == "rgb":
        st.session_state["module4_click_generation"] = st.session_state.get("module4_click_generation", 0) + 1


def render_rgb():
    st.subheader("Question 1 – Human Boundary Detection in an RGB Image")
    st.caption(
        "Select one main person who is the dominant foreground component. "
        "Canny demonstrates edge detection; GrabCut segments foreground from "
        "background using your rectangle. Neither method recognizes a person."
    )

    st.session_state.setdefault("module4_points", [])
    st.session_state.setdefault("module4_click_generation", 0)
    st.session_state.setdefault("module4_upload_generation", 0)

    st.button("Reset RGB Experiment", key="module4_reset_rgb", on_click=clear_rgb_upload)
    uploaded = st.file_uploader(
        "Upload RGB Image", type=["jpg", "jpeg", "png"],
        key=f"module4_upload_{st.session_state['module4_upload_generation']}", on_change=update_rgb_upload,
    )
    image_bytes = uploaded.getvalue() if uploaded is not None else st.session_state.get("module4_rgb_bytes")
    if uploaded is not None:
        st.session_state["module4_rgb_bytes"] = image_bytes
        st.session_state["module4_rgb_filename"] = uploaded.name
    elif image_bytes is not None:
        st.caption(f"Loaded RGB image: {st.session_state.get('module4_rgb_filename', 'previous upload')}")
        st.button("Clear RGB Image", key="module4_clear_image", on_click=clear_rgb_upload)
    image_digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes is not None else None
    if st.session_state.get("module4_image_digest") != image_digest:
        reset_rectangle()
        st.session_state["module4_image_digest"] = image_digest

    image = None
    if image_bytes is not None:
        try:
            # Decode to BGR uint8, then let the backend normalize to its RGB convention.
            decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            image = normalize_image(decoded, color_order="BGR")
        except (ValueError, cv2.error):
            st.error("This file could not be read as an image. Please upload another JPG or PNG.")

    rectangle = None
    rectangle_image = None
    if image is None:
        st.info("Upload an image to select a person and run the experiment.")
    else:
        st.image(image, caption="Original RGB Image", width="stretch")
        st.subheader("Select Person Rectangle")
        st.write("Click two opposite corners to create a rectangle around the person.")
        st.caption("Include the entire person and leave some background outside the rectangle.")
        st.button("Reset Rectangle", key="module4_reset_rectangle", on_click=reset_rectangle)
        points = st.session_state["module4_points"]
        rectangle_image = image.copy()
        if len(points) == 2:
            left, right = sorted((points[0][0], points[1][0]))
            top, bottom = sorted((points[0][1], points[1][1]))
            try:
                rectangle = validate_rectangle((left, top, right - left, bottom - top), image.shape)
            except ValueError as error:
                st.warning(f"{error} Use Reset Rectangle to select again.")
            else:
                # The backend rectangle uses an exclusive right/bottom boundary.
                cv2.rectangle(rectangle_image, (left, top), (right - 1, bottom - 1), (255, 200, 0), 2)
                st.caption(f"Rectangle (x, y, width, height): {rectangle} pixels")

        # Limit preview payload size; click mapping uses actual rendered dimensions.
        height, width = image.shape[:2]
        scale = min(1.0, 700 / width, 700 / height)
        preview = cv2.resize(rectangle_image, (max(1, round(width * scale)), max(1, round(height * scale))))
        for x, y in points:
            position = (round(x * preview.shape[1] / width), round(y * preview.shape[0] / height))
            cv2.circle(preview, position, 5, (255, 200, 0), -1)
        if len(points) < 2:
            st.info("Select the first corner." if not points else "Select the opposite corner.")
            click = streamlit_image_coordinates(
                preview, width="content", cursor="crosshair",
                key=f"module4_image_{st.session_state['module4_click_generation']}",
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
                    st.session_state["module4_click_generation"] += 1
                    st.rerun()
        else:
            st.image(preview, caption="Selected corners and GrabCut rectangle", width="content")

    st.subheader("Processing Settings")
    gaussian_column, canny_column = st.columns(2)
    with gaussian_column:
        gaussian_kernel = st.selectbox("Gaussian kernel size", [1, 3, 5, 7], index=1, key="module4_gaussian_kernel")
        gaussian_sigma = st.slider("Gaussian sigma", 0.1, 3.0, 0.8, 0.1, key="module4_gaussian_sigma")
    with canny_column:
        canny_lower = st.slider("Canny lower threshold", 0, 255, 50, key="module4_canny_lower")
        canny_upper = st.slider("Canny upper threshold", 1, 255, 150, key="module4_canny_upper")
    grabcut_iterations = st.slider("GrabCut iterations", 1, 10, 5, key="module4_grabcut_iterations")
    cleanup_enabled = st.checkbox("Apply Morphological Cleanup", value=False, key="module4_cleanup_enabled")
    morphology_kernel = st.selectbox(
        "Cleanup kernel size", [3, 5], disabled=not cleanup_enabled, key="module4_morphology_kernel",
        help="One elliptical closing operation; small kernels help preserve thin body parts.",
    )
    settings = {
        "gaussian_kernel_size": gaussian_kernel,
        "gaussian_sigma": gaussian_sigma,
        "canny_lower": canny_lower,
        "canny_upper": canny_upper,
        "grabcut_iterations": grabcut_iterations,
        "cleanup_enabled": cleanup_enabled,
        "morphology_kernel_size": morphology_kernel,
    }
    shared_inputs = (image_digest, tuple(image.shape) if image is not None else None, rectangle)
    classical_inputs = (shared_inputs, tuple(settings.items()))
    sam2_inputs = (shared_inputs, sam2_configuration())
    if st.session_state.get("module4_shared_inputs") != shared_inputs:
        clear_results()
        st.session_state["module4_shared_inputs"] = shared_inputs
    if st.session_state.get("module4_result_inputs") != classical_inputs:
        clear_results("classical")
        st.session_state["module4_result_inputs"] = classical_inputs
    if st.session_state.get("module4_sam2_inputs") != sam2_inputs:
        clear_results("sam2")
        st.session_state["module4_sam2_inputs"] = sam2_inputs

    thresholds_valid = canny_lower < canny_upper
    if not thresholds_valid:
        st.warning("Canny lower threshold must be less than the upper threshold.")
    classical_button, sam2_button = st.columns(2)
    with classical_button:
        run_classical = st.button(
            "Run Classical Method", key="module4_detect", type="primary",
            disabled=image is None or rectangle is None or not thresholds_valid,
        )
    with sam2_button:
        run_sam = st.button(
            "Run SAM2", key="module4_run_sam2",
            disabled=image is None or rectangle is None,
        )
    def process_classical():
        clear_results("classical")
        try:
            with st.spinner("Detecting edges and segmenting the foreground…"):
                result = find_human_boundary(image, rectangle, **settings)
                # The original image can be decoded from the upload; avoid retaining another copy.
                result.pop("original_image", None)
                if not cleanup_enabled:
                    result["cleaned_mask"] = result["raw_mask"]
                st.session_state["module4_results"] = result
                st.session_state["module4_classical_identity"] = classical_inputs
        except Exception:
            logging.getLogger(__name__).exception("Module 4 classical inference failed")
            st.session_state["module4_classical_error"] = "Classical processing could not complete. Check the image, rectangle, and settings, then try again."

    def process_sam2():
        clear_results("sam2")
        try:
            from Modules.Module_4.sam2_comparison import run_sam2, select_device

            source, config, requested_device, _ = sam2_inputs[1]
            device = select_device(requested_device)
            # Refuse oversized CPU requests before downloading or loading weights.
            check_image_budget(image.shape, device, cpu_pixel_limit())
            with st.spinner("Preparing SAM2 model…"):
                resource = cached_sam2_model(source, config, device)
            with st.spinner("Segmenting the selected person with SAM2…"):
                result = run_sam2(image, rectangle, resource)
                st.session_state["module4_sam2_results"] = result
                st.session_state["module4_sam2_identity"] = sam2_inputs
        except SAM2ResourceLimitError as error:
            st.session_state["module4_sam2_error"] = str(error)
        except Exception:
            logging.getLogger(__name__).exception("Module 4 SAM2 inference failed")
            st.session_state["module4_sam2_error"] = (
                "SAM2 could not complete. Model preparation or processing may be temporarily "
                "unavailable; please try again. You can still use the classical method."
            )

    if run_classical:
        process_classical()
    if run_sam:
        process_sam2()

    def open_comparison():
        st.session_state["module4_comparison"] = (classical_inputs, sam2_inputs)

    results = st.session_state.get("module4_results")
    sam2_results = st.session_state.get("module4_sam2_results")
    comparison_ready = (
        image is not None and rectangle is not None
        and results is not None and sam2_results is not None
        and st.session_state.get("module4_classical_identity") == classical_inputs
        and st.session_state.get("module4_sam2_identity") == sam2_inputs
    )
    if st.button("Compare Results", key="module4_compare", disabled=not comparison_ready):
        open_comparison()
    if st.session_state.get("module4_classical_error"):
        st.error(st.session_state["module4_classical_error"])
    if results is not None:
        st.divider()
        st.subheader("Results")
        original_column, smoothed_column = st.columns(2)
        with original_column:
            st.image(rectangle_image, caption="Original Image with GrabCut Rectangle", width="stretch")
        with smoothed_column:
            st.image(results["smoothed_image"], caption="Gaussian Smoothed Image", width="stretch")
        st.image(results["canny_edges"], caption="Canny Edge Map", width="stretch")
        st.caption(
            "Canny detects intensity changes: outer edges of the person, internal "
            "clothing/body edges, and background edges. This edge map is not used as GrabCut input."
        )
        raw_column, cleaned_column = st.columns(2)
        with raw_column:
            st.image(results["raw_mask"], caption="Raw GrabCut Foreground Mask", width="stretch")
        with cleaned_column:
            mask_caption = "Cleaned Foreground Mask" if cleanup_enabled else "Foreground Mask — Unchanged (Cleanup Off)"
            st.image(results["cleaned_mask"], caption=mask_caption, width="stretch")
        st.caption("GrabCut uses the selected rectangle to initialize foreground/background segmentation. White = foreground; black = background.")
        st.subheader("Final Human Boundary")
        if results["contour"] is None:
            st.warning("No foreground contour was found. Adjust the rectangle or settings and try again; the image below is unchanged.")
        st.image(results["boundary_overlay"], caption="External Boundary on the Original Image", width="stretch")
        st.caption("Contour extraction selects the largest external foreground contour, assuming the person is the dominant foreground component.")
        st.metric("Selected contour area (pixels²)", f"{results['contour_area']:,.1f}")
        st.caption(f"Rectangle width × height: {rectangle[2]} × {rectangle[3]} pixels. Contour area is geometric area, not a foreground pixel count.")
        if sam2_results is None and st.button(
            "Run SAM2", key="module4_run_sam2_after_classical",
            disabled=image is None or rectangle is None,
        ):
            process_sam2()
            st.rerun()

    if sam2_results is not None or st.session_state.get("module4_sam2_error"):
        st.divider()
        st.subheader("SAM2 Results")
        if st.session_state.get("module4_sam2_error"):
            st.error(st.session_state["module4_sam2_error"])
        if sam2_results is not None:
            st.image(sam2_results["binary_mask"], caption="SAM2 Binary Mask", width="stretch")
            st.image(sam2_results["boundary_overlay"], caption="SAM2 Human Boundary", width="stretch")
            quality_column, device_column = st.columns(2)
            with quality_column:
                st.metric("Predicted Mask Quality", f"{sam2_results['predicted_mask_quality']:.6f}")
            with device_column:
                st.metric("Device Used", sam2_results["metadata"]["device"].upper())
            st.caption("This is SAM2's predicted mask-quality score, not measured segmentation accuracy.")
            st.caption("SAM 2.1 Hiera Tiny, box-only prompt. All external foreground boundaries are shown.")
            if sam2_results["metadata"]["empty_mask"]:
                st.info("SAM2 predicted no foreground. Try selecting another rectangle.")
            if results is None and st.button(
                "Run Classical Method", key="module4_detect_after_sam2",
                disabled=image is None or rectangle is None or not thresholds_valid,
            ):
                process_classical()
                st.rerun()

    if comparison_ready and st.button("Compare Results", key="module4_compare_after_results"):
        open_comparison()

    if comparison_ready and st.session_state.get("module4_comparison") == (classical_inputs, sam2_inputs):
        st.divider()
        st.subheader("Compare Results")
        classical_column, sam2_column = st.columns(2)
        with classical_column:
            st.markdown("**Classical Method**")
            st.caption("Canny + GrabCut + contour extraction")
            st.image(results["raw_mask"], caption="GrabCut Binary Mask", width="stretch")
            st.image(results["boundary_overlay"], caption="Classical Boundary", width="stretch")
            if cleanup_enabled:
                st.caption("The classical boundary includes the selected morphological cleanup.")
        with sam2_column:
            st.markdown("**SAM2**")
            st.caption("SAM 2.1 Hiera Tiny, box-only prompt")
            st.image(sam2_results["binary_mask"], caption="SAM2 Binary Mask", width="stretch")
            st.image(sam2_results["boundary_overlay"], caption="SAM2 Boundary", width="stretch")

st.set_page_config(page_title="Module 4 - Human Boundary Detection", layout="centered")
st.title("Module 4 – Human Boundary Detection")
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
if section == "home":
    st.write("Select an experiment:")
    for column, (label, destination, question) in zip(st.columns(3), (
        ("RGB Image", "rgb", "Question 1"),
        ("Thermal Image", "thermal", "Question 2"),
        ("Frequency Domain Analysis", "frequency", "Question 3"),
    )):
        with column:
            st.button(label, key=f"module4_open_{destination}", on_click=navigate_module4, args=(destination,))
            st.caption(question)
else:
    st.button("Module 4 Home", key="module4_home", on_click=navigate_module4, args=("home",))
    if section == "rgb":
        render_rgb()
    elif section == "thermal":
        st.subheader("Question 2 – Thermal Image Human Boundary Detection")
        st.info("Thermal image workflow will be implemented next.")
    elif section == "frequency":
        st.subheader("Question 3 – Frequency Domain Analysis")
        st.info("Frequency-domain analysis will be added after the experimental sections.")
