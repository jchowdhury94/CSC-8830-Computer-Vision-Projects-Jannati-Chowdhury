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
from Modules.Module_4.rgb_preprocessing import (
    prepared_rgb_upload,
)
from Modules.Module_4.thermal_boundary import (
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
    st.session_state["module4_classical_config_open"] = True
    if st.session_state.get("module4_sam2_results") is not None:
        st.session_state["module4_rgb_first_method"] = "sam2"


def open_classical_settings():
    """Select Classical without executing either processing method."""
    st.session_state["module4_classical_config_open"] = True


def reset_sam2_result():
    """Discard the SAM2 run while preserving shared inputs and Classical."""
    clear_results("sam2")
    if st.session_state.get("module4_results") is not None:
        st.session_state["module4_rgb_first_method"] = "classical"


def reset_rectangle():
    """Clear corners and replace the component so old clicks cannot replay."""
    st.session_state["module4_classical_config_open"] = False
    st.session_state.pop("module4_rgb_first_method", None)
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


def navigate_module4(section):
    st.session_state["module4_section"] = section
    if section == "thermal":
        st.session_state["module4_q2_click_generation"] = st.session_state.get("module4_q2_click_generation", 0) + 1
    # Replace the click component when revisiting RGB so an old click cannot replay.
    if section == "rgb":
        st.session_state["module4_click_generation"] = st.session_state.get("module4_click_generation", 0) + 1


def render_classical_settings():
    """Render the single Q1 parameter editor in the active method area."""
    st.subheader("Processing Settings")
    gaussian_column, canny_column = st.columns(2)
    with gaussian_column:
        st.selectbox("Gaussian kernel size", [1, 3, 5, 7], index=1, key="module4_gaussian_kernel")
        st.slider("Gaussian sigma", 0.1, 3.0, 0.8, 0.1, key="module4_gaussian_sigma")
    with canny_column:
        st.slider("Canny lower threshold", 0, 255, 50, key="module4_canny_lower")
        st.slider("Canny upper threshold", 1, 255, 150, key="module4_canny_upper")
    st.slider("GrabCut iterations", 1, 10, 5, key="module4_grabcut_iterations")
    cleanup_enabled = st.checkbox("Apply Morphological Cleanup", value=False, key="module4_cleanup_enabled")
    st.selectbox(
        "Cleanup kernel size", [3, 5], disabled=not cleanup_enabled, key="module4_morphology_kernel",
        help="One elliptical closing operation; small kernels help preserve thin body parts.",
    )


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
    original_shape = None
    if image_bytes is not None:
        image, original_shape, preparation_error = prepared_rgb_upload(
            st.session_state, image_bytes, image_digest,
        )
        if preparation_error:
            st.error(preparation_error)
    prepared_identity = (
        st.session_state["module4_rgb_prepared"][0],
        tuple(image.shape) if image is not None else None,
    ) if image_bytes is not None else None
    if st.session_state.get("module4_rgb_processing_identity") != prepared_identity:
        reset_rectangle()
        st.session_state["module4_rgb_processing_identity"] = prepared_identity
    resized = image is not None and image.shape != original_shape
    if resized:
        st.caption(
            f"Image resized from {original_shape[1]} × {original_shape[0]} to "
            f"{image.shape[1]} × {image.shape[0]} pixels for processing."
        )

    rectangle = None
    rectangle_image = None
    if image is None:
        st.info("Upload an image to select a person and run the experiment.")
    else:
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

    # Reassign the same widget keys to preserve values while settings are hidden.
    defaults = {
        "gaussian_kernel": 3, "gaussian_sigma": 0.8,
        "canny_lower": 50, "canny_upper": 150, "grabcut_iterations": 5,
        "cleanup_enabled": False, "morphology_kernel": 3,
    }
    for name, default in defaults.items():
        key = f"module4_{name}"
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    settings = {
        "gaussian_kernel_size": st.session_state.get("module4_gaussian_kernel", defaults["gaussian_kernel"]),
        "gaussian_sigma": st.session_state.get("module4_gaussian_sigma", defaults["gaussian_sigma"]),
        "canny_lower": st.session_state.get("module4_canny_lower", defaults["canny_lower"]),
        "canny_upper": st.session_state.get("module4_canny_upper", defaults["canny_upper"]),
        "grabcut_iterations": st.session_state.get("module4_grabcut_iterations", defaults["grabcut_iterations"]),
        "cleanup_enabled": st.session_state.get("module4_cleanup_enabled", defaults["cleanup_enabled"]),
        "morphology_kernel_size": st.session_state.get("module4_morphology_kernel", defaults["morphology_kernel"]),
    }
    cleanup_enabled = settings["cleanup_enabled"]
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

    thresholds_valid = settings["canny_lower"] < settings["canny_upper"]
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

    def open_comparison():
        st.session_state["module4_comparison"] = (classical_inputs, sam2_inputs)

    # Reserve contextual areas once, then render each settings widget at most once.
    top_actions = st.container()
    if st.session_state.get("module4_rgb_first_method") == "sam2":
        sam2_display = st.container()
        after_sam2_actions = st.container()
        classical_display = st.container()
        after_classical_actions = st.container()
    else:
        classical_display = st.container()
        after_classical_actions = st.container()
        sam2_display = st.container()
        after_sam2_actions = st.container()

    results = st.session_state.get("module4_results")
    sam2_results = st.session_state.get("module4_sam2_results")
    run_classical = run_sam = False
    if results is None:
        area = after_sam2_actions if sam2_results is not None else top_actions
        with area:
            if st.session_state.get("module4_classical_config_open", False):
                render_classical_settings()
                if not thresholds_valid:
                    st.warning("Canny lower threshold must be less than the upper threshold.")
                run_classical = st.button(
                    "Apply Classical Method", key="module4_apply_classical", type="primary",
                    disabled=image is None or rectangle is None or not thresholds_valid,
                )
            elif sam2_results is not None:
                st.button("Classical Method", key="module4_detect_after_sam2",
                          on_click=open_classical_settings)
            else:
                classical_button, sam2_button = st.columns(2)
                classical_button.button(
                    "Classical Method", key="module4_detect", on_click=open_classical_settings,
                    disabled=image is None or rectangle is None,
                )
                run_sam = sam2_button.button(
                    "Run SAM2", key="module4_run_sam2",
                    disabled=image is None or rectangle is None,
                )
    if sam2_results is None and (results is not None or st.session_state.get("module4_classical_config_open")):
        with after_classical_actions if results is not None else top_actions:
            run_sam = st.button(
                "Run SAM2", key="module4_run_sam2_after_classical" if results is not None else "module4_run_sam2",
                disabled=image is None or rectangle is None,
            )
    if run_classical:
        process_classical()
        if st.session_state.get("module4_results") is not None:
            st.session_state["module4_classical_config_open"] = False
            st.session_state.setdefault("module4_rgb_first_method", "classical")
            st.rerun()
    if run_sam:
        process_sam2()
        if st.session_state.get("module4_sam2_results") is not None:
            st.session_state.setdefault("module4_rgb_first_method", "sam2")
            st.rerun()

    comparison_ready = (
        image is not None and rectangle is not None
        and results is not None and sam2_results is not None
        and st.session_state.get("module4_classical_identity") == classical_inputs
        and st.session_state.get("module4_sam2_identity") == sam2_inputs
    )
    with classical_display:
        if st.session_state.get("module4_classical_error"):
            st.error(st.session_state["module4_classical_error"])
        if results is not None:
            st.divider()
            st.subheader("Results")
            original_column, smoothed_column = st.columns(2)
            with original_column:
                st.image(rectangle_image, caption="Processing Image with GrabCut Rectangle" if resized else "Original Image with GrabCut Rectangle", width="stretch")
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
            st.image(results["boundary_overlay"], caption="External Boundary on the Processing Image" if resized else "External Boundary on the Original Image", width="stretch")
            st.caption("Contour extraction selects the largest external foreground contour, assuming the person is the dominant foreground component.")
            st.metric("Selected contour area (pixels²)", f"{results['contour_area']:,.1f}")
            st.caption(f"Rectangle width × height: {rectangle[2]} × {rectangle[3]} pixels. Contour area is geometric area, not a foreground pixel count.")
            st.button("Reset Classical Method", key="module4_reset_classical", on_click=reset_classical_result)

    with sam2_display:
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
                st.button("Reset SAM2", key="module4_reset_sam2", on_click=reset_sam2_result)
    if comparison_ready and st.button("Compare Results", key="module4_compare_after_results"):
        open_comparison()

    if comparison_ready and st.session_state.get("module4_comparison") == (classical_inputs, sam2_inputs):
        st.divider()
        st.subheader("Compare Results")
        if resized:
            st.caption(
                f"Classical and SAM2: {image.shape[1]} × {image.shape[0]} pixels, "
                "using the same processing image and rectangle."
            )
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

    if results is not None or sam2_results is not None:
        st.button("Reset RGB Experiment", key="module4_reset_rgb_bottom", on_click=clear_rgb_upload)


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


def open_thermal_sam2_setup():
    """Reveal the optional SAM2 box prompt without changing processing state."""
    st.session_state["module4_q2_setup_open"] = True
    st.session_state.setdefault("module4_q2_workflow", "sam2")


def update_thermal_upload():
    """Persist Q2 bytes across navigation, independently of uploader widgets."""
    uploaded = st.session_state.get(f"module4_q2_upload_{st.session_state['module4_q2_upload_generation']}")
    st.session_state["module4_q2_bytes"] = uploaded.getvalue() if uploaded is not None else None
    st.session_state["module4_q2_filename"] = uploaded.name if uploaded is not None else None
    st.session_state.pop("module4_q2_prepared", None)
    st.session_state.pop("module4_q2_processing_identity", None)
    clear_thermal_results()
    reset_thermal_rectangle()
    st.session_state["module4_q2_setup_open"] = False
    st.session_state.pop("module4_q2_workflow", None)


def reset_thermal_upload():
    """Replace the Q2 uploader and clear Q2 data without touching RGB state."""
    st.session_state["module4_q2_upload_generation"] = st.session_state.get("module4_q2_upload_generation", 0) + 1
    for key in ("module4_q2_bytes", "module4_q2_filename", "module4_q2_digest"):
        st.session_state.pop(key, None)
    st.session_state.pop("module4_q2_prepared", None)
    st.session_state.pop("module4_q2_processing_identity", None)
    clear_thermal_results()
    reset_thermal_rectangle()
    st.session_state["module4_q2_setup_open"] = False
    st.session_state.pop("module4_q2_workflow", None)


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


def render_thermal():
    st.subheader("Question 2 – Thermal Image Human Boundary Detection")
    st.caption(
        "Upload a false-color thermal image containing one person. "
        "This method selects the largest warm-colored region, so use an image "
        "where the person is green/yellow/orange/red against a mostly blue background."
    )
    st.session_state.setdefault("module4_q2_upload_generation", 0)
    st.button("Reset Question 2", key="module4_q2_reset", on_click=reset_thermal_upload)
    uploaded = st.file_uploader(
        "Upload Thermal Image", type=["jpg", "jpeg", "png"],
        key=f"module4_q2_upload_{st.session_state['module4_q2_upload_generation']}",
        on_change=update_thermal_upload,
    )
    image_bytes = uploaded.getvalue() if uploaded is not None else st.session_state.get("module4_q2_bytes")
    if uploaded is not None:
        st.session_state["module4_q2_bytes"] = image_bytes
        st.session_state["module4_q2_filename"] = uploaded.name
    elif image_bytes is not None:
        st.caption(f"Loaded thermal image: {st.session_state.get('module4_q2_filename', 'previous upload')}")
    digest = hashlib.sha256(image_bytes).hexdigest() if image_bytes is not None else None
    if st.session_state.get("module4_q2_digest") != digest:
        clear_thermal_results()
        reset_thermal_rectangle()
        st.session_state["module4_q2_setup_open"] = False
        st.session_state.pop("module4_q2_workflow", None)
        st.session_state["module4_q2_digest"] = digest

    image = None
    original_shape = None
    prepared_identity = None
    if image_bytes is not None:
        image, original_shape, preparation_error = prepared_thermal_upload(
            st.session_state, image_bytes, digest,
        )
        prepared_identity = st.session_state["module4_q2_prepared"][0]
        if preparation_error:
            st.error(preparation_error)
    if st.session_state.get("module4_q2_processing_identity") != prepared_identity:
        clear_thermal_results()
        reset_thermal_rectangle()
        st.session_state["module4_q2_processing_identity"] = prepared_identity
    resized = image is not None and image.shape != original_shape
    if resized:
        st.caption(
            f"Thermal image resized from {original_shape[1]} × {original_shape[0]} to "
            f"{image.shape[1]} × {image.shape[0]} pixels for Classical and SAM2 processing."
        )
    if image is None:
        st.info("Upload a false-color thermal image to run the classical method.")
    else:
        st.image(image, caption="Thermal Image for Processing" if resized else "Original Thermal Image", width="stretch")

    st.session_state.setdefault("module4_q2_points", [])
    st.session_state.setdefault("module4_q2_click_generation", 0)
    # Presentation order follows the chosen starting method, not processing identity.
    workflow = st.session_state.get("module4_q2_workflow")
    initial_actions = st.container()
    if workflow == "classical":
        classical_display = st.container()
        sam2_setup_display = st.container()
        sam2_display = st.container()
    else:
        sam2_setup_display = st.container()
        sam2_display = st.container()
        classical_display = st.container()
    st.session_state.setdefault("module4_q2_setup_open", False)
    rectangle = None
    with sam2_setup_display:
        if image is not None and st.session_state["module4_q2_setup_open"]:
            st.subheader("Select SAM2 Rectangle")
            st.caption("Click two opposite corners around the person on the thermal processing image. This box prompts SAM2 only; the classical method does not use it.")
            st.button("Reset SAM2 Rectangle", key="module4_q2_reset_rectangle", on_click=reset_thermal_rectangle)
            points = st.session_state["module4_q2_points"]
            height, width = image.shape[:2]
            # Preview clicks map to the shared processing image used for inference.
            scale = min(1.0, 700 / width, 700 / height)
            preview = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
            for x, y in points:
                cv2.circle(preview, (round(x * preview.shape[1] / width), round(y * preview.shape[0] / height)), 5, (255, 0, 255), -1)
            if len(points) == 2:
                left, right = sorted((points[0][0], points[1][0]))
                top, bottom = sorted((points[0][1], points[1][1]))
                try:
                    rectangle = validate_rectangle((left, top, right - left, bottom - top), image.shape)
                except ValueError as error:
                    st.warning(f"{error} Use Reset SAM2 Rectangle to select again.")
                else:
                    cv2.rectangle(preview, (round(left * scale), round(top * scale)),
                                  (round((right - 1) * scale), round((bottom - 1) * scale)), (255, 0, 255), 2)
                    st.caption(f"SAM2 box (x, y, width, height): {rectangle} pixels")
                st.image(preview, caption="SAM2 box on thermal processing image", width="content")
            else:
                st.info("Select the first SAM2 corner." if not points else "Select the second SAM2 corner.")
                component_key = f"module4_q2_click_{st.session_state['module4_q2_click_generation']}"
                streamlit_image_coordinates(
                    preview, width="content", cursor="crosshair", key=component_key,
                    on_click=lambda: record_thermal_corner(component_key, image.shape),
                )

    classical_inputs = (prepared_identity, tuple(image.shape) if image is not None else None)
    sam2_inputs = (classical_inputs, rectangle, sam2_configuration())
    if st.session_state.get("module4_q2_sam2_inputs") != sam2_inputs:
        clear_thermal_results("sam2")
        st.session_state["module4_q2_sam2_inputs"] = sam2_inputs

    def process_classical():
        clear_thermal_results("classical")
        try:
            with st.spinner("Extracting the thermal human boundary…"):
                result = find_thermal_boundary(image)
                # Retain display stages and metrics; the upload preserves the original.
                for key in ("original_image", "hsv_image", "closed_mask"):
                    result.pop(key)
                st.session_state["module4_q2_results"] = result
                st.session_state["module4_q2_classical_identity"] = classical_inputs
        except ValueError as error:
            st.session_state["module4_q2_error"] = str(error)
        except cv2.error:
            logging.getLogger(__name__).exception("Module 4 thermal processing failed")
            st.session_state["module4_q2_error"] = "Thermal processing could not complete. Please try another image."

    def process_sam2():
        clear_thermal_results("sam2")
        try:
            from Modules.Module_4.sam2_comparison import run_sam2, select_device

            source, config, requested_device, _ = sam2_inputs[2]
            device = select_device(requested_device)
            check_image_budget(image.shape, device, cpu_pixel_limit())
            with st.spinner("Preparing SAM2 model…"):
                resource = cached_sam2_model(source, config, device)
            with st.spinner("Segmenting the thermal person with SAM2…"):
                result = run_sam2(image, rectangle, resource)
                # Q2 presentation only: preserve every SAM2 mask pixel/component.
                contours, _ = cv2.findContours(result["binary_mask"].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result["boundary_overlay"] = image.copy()
                cv2.drawContours(result["boundary_overlay"], contours, -1, (255, 0, 255), 3)
                result["metadata"] = {**result["metadata"], "overlay": "All external contours, magenta, thickness 3; no mask editing"}
                st.session_state["module4_q2_sam2_results"] = result
                st.session_state["module4_q2_sam2_identity"] = sam2_inputs
        except SAM2ResourceLimitError as error:
            st.session_state["module4_q2_sam2_error"] = str(error)
        except Exception:
            logging.getLogger(__name__).exception("Module 4 thermal SAM2 inference failed")
            st.session_state["module4_q2_sam2_error"] = "SAM2 could not complete. Please try again. The classical thermal method remains available."

    if workflow is None and image is not None:
        with initial_actions:
            classical_column, sam2_column = st.columns(2)
            if classical_column.button("Run Classical Method", key="module4_q2_run", type="primary"):
                st.session_state["module4_q2_workflow"] = "classical"
                process_classical()
                st.rerun()
            sam2_column.button("Set Up SAM2", key="module4_q2_setup",
                               on_click=open_thermal_sam2_setup)
    if rectangle is not None and st.session_state.get("module4_q2_sam2_results") is None:
        with sam2_setup_display:
            if st.button("Run SAM2", key="module4_q2_run_sam2"):
                process_sam2()
    results = st.session_state.get("module4_q2_results")
    sam2_results = st.session_state.get("module4_q2_sam2_results")
    comparison_ready = (
        image is not None and rectangle is not None
        and results is not None and sam2_results is not None
        and st.session_state.get("module4_q2_classical_identity") == classical_inputs
        and st.session_state.get("module4_q2_sam2_identity") == sam2_inputs
    )
    def open_thermal_comparison():
        st.session_state["module4_q2_comparison"] = (classical_inputs, sam2_inputs)

    with classical_display:
        if st.session_state.get("module4_q2_error"):
            st.error(st.session_state["module4_q2_error"])
        results = st.session_state.get("module4_q2_results")
        if results is not None and image is not None:
            st.subheader("Classical Thermal Results")
            initial_column, cleaned_column = st.columns(2)
            with initial_column:
                st.image(results["initial_hsv_mask"], caption="Initial HSV Mask", width="stretch")
            with cleaned_column:
                st.image(results["cleaned_human_mask"], caption="Cleaned Human Mask", width="stretch")
            st.image(results["boundary_overlay"], caption="Human Boundary", width="stretch")
            st.caption(f"Processing dimensions: {results['width']} × {results['height']} pixels. Masks, overlays, and measurements use this resolution.")
            count_column, area_column, percentage_column = st.columns(3)
            count_column.metric("Foreground components", results["foreground_components"])
            area_column.metric("Selected area (pixels)", f"{results['selected_component_area']:,}")
            percentage_column.metric("Selected image occupancy", f"{results['selected_component_percentage']:.2f}%")
            st.caption(
                "Warm-color HSV thresholding → 3×3 elliptical closing → largest "
                "8-connected foreground component → external contour. "
                "Occupancy is the fraction of image pixels selected, not segmentation accuracy."
            )

    with sam2_display:
        if st.session_state.get("module4_q2_sam2_error"):
            st.error(st.session_state["module4_q2_sam2_error"])
        if sam2_results is not None:
            st.subheader("SAM2 Results")
            st.image(sam2_results["binary_mask"], caption="SAM2 Binary Mask", width="stretch")
            st.image(sam2_results["boundary_overlay"], caption="SAM2 Human Boundary", width="stretch")
            quality_column, device_column = st.columns(2)
            quality_column.metric("Predicted Mask Quality", f"{sam2_results['predicted_mask_quality']:.6f}")
            device_column.metric("Device Used", sam2_results["metadata"]["device"].upper())
            st.caption("SAM 2.1 Hiera Tiny, box-only prompt. Predicted Mask Quality is the model's score, not measured segmentation accuracy.")
            if sam2_results["metadata"]["empty_mask"]:
                st.info("SAM2 predicted no foreground. Try another rectangle.")

    # Keep next actions below their corresponding workflow, with no top duplicates.
    if workflow is not None and results is None and (workflow == "classical" or sam2_results is not None):
        with classical_display:
            if st.button("Run Classical Method", key="module4_q2_classical_next", disabled=image is None):
                process_classical()
                st.rerun()
    if results is not None and not st.session_state["module4_q2_setup_open"]:
        with classical_display:
            st.button("Set Up SAM2", key="module4_q2_setup_after_classical",
                      on_click=open_thermal_sam2_setup)
    if comparison_ready:
        if st.button("Compare Results", key="module4_q2_compare_after_results"):
            open_thermal_comparison()

    if comparison_ready and st.session_state.get("module4_q2_comparison") == (classical_inputs, sam2_inputs):
        st.subheader("Compare Results")
        st.caption(
            f"Classical and SAM2: {image.shape[1]} × {image.shape[0]} pixels, "
            "using the same processing image. IoU uses masks at this resolution."
        )
        iou = thermal_mask_iou(results["cleaned_human_mask"], sam2_results["binary_mask"])
        st.metric("IoU with SAM2", f"{iou:.4f}" if iou is not None else "N/A")
        st.caption("IoU measures the overlap between the Classical and SAM2 segmentation masks. A value closer to 1 indicates greater agreement between the two segmentations.")
        st.caption("SAM2 is the comparison segmentation, not ground truth.")
        if iou is None:
            st.caption("IoU is undefined because both masks contain no foreground.")
        st.caption("Classical Thermal Method: HSV thresholding + morphology + largest connected component + contour. SAM2: SAM 2.1 Hiera Tiny, box-only prompt.")
        # Each equal-width row starts together, independently of caption wrapping.
        classical_column, sam2_column = st.columns(2)
        classical_column.image(results["cleaned_human_mask"], caption="Classical Mask", width="stretch")
        sam2_column.image(sam2_results["binary_mask"], caption="SAM2 Mask", width="stretch")
        classical_column, sam2_column = st.columns(2)
        classical_column.image(results["boundary_overlay"], caption="Classical Boundary", width="stretch")
        sam2_column.image(sam2_results["boundary_overlay"], caption="SAM2 Boundary", width="stretch")


ACCENT = "#C9363E"
HOME_TITLE = "Module 4"
HOME_SUBTITLE = "Human Boundary Detection"

HOME_CSS = f"""
<style>
.st-key-m4_home_layout {{ gap: 0; padding-top: 4vh; }}
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
@media (min-width: 641px) {{ .st-key-m4_home_subtitle h2 {{ white-space: nowrap; }} }}
.st-key-m4_home_nav {{ flex-direction: row; flex-wrap: wrap; gap: 24px; }}
.st-key-m4_open_rgb, .st-key-m4_open_thermal {{ flex: 1 1 280px; width: auto; min-width: 0; max-width: 100%; }}
.st-key-m4_open_rgb button, .st-key-m4_open_thermal button {{
    width: 100%; height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
.st-key-m4_open_rgb button p, .st-key-m4_open_thermal button p {{
    font-size: 21px; font-weight: 700; color: #FFFFFF; line-height: 1.3; text-align: center; text-wrap: balance;
}}
.st-key-m4_open_rgb button:hover, .st-key-m4_open_thermal button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.st-key-m4_open_rgb button:focus-visible, .st-key-m4_open_thermal button:focus-visible {{
    outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT};
}}
.st-key-m4_open_rgb button:active, .st-key-m4_open_thermal button:active {{ background-color: #8E2229; border-color: #8E2229; }}
@media (max-width: 640px) {{
    .st-key-m4_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 16px; }}
    .st-key-m4_open_rgb, .st-key-m4_open_thermal {{ flex: 0 0 auto; width: 100%; }}
}}
</style>
"""


DISPLAY_DIR = REPOSITORY_ROOT / "Modules" / "Module_4" / "display"

# Values below are the ones documented in my original Module 4 report.
RGB_EXPERIMENT = {
    "kernel": 3, "sigma": "0.80", "canny": "50 / 150", "iterations": 5, "cleanup": "Disabled",
    "rectangle": "1766 \u00d7 5620 pixels", "contour_area": "6,574,847.0 pixels\u00b2",
    "sam2_score": "0.989319", "sam2_device": "MPS",
}


def rgb_results_page():
    r = RGB_EXPERIMENT
    st.button("Back", key="m4_rgb_results_back_top", on_click=navigate_module4, args=("rgb_select",))
    st.header("My RGB Experiment & Results", anchor=False)

    def with_text(image, caption, *text):
        image_column, text_column = st.columns([3, 2], vertical_alignment="center")
        image_column.image(str(DISPLAY_DIR / image), caption=caption, width="stretch")
        with text_column:
            for line in text:
                st.write(line)

    st.subheader("Step 1 \u2014 Original RGB Image & Parameters", anchor=False)
    image_column, text_column = st.columns([3, 2], vertical_alignment="center")
    image_column.image(str(DISPLAY_DIR / "step1_original.jpg"), caption="Original RGB image", width="stretch")
    with text_column:
        st.metric("Gaussian kernel size", r["kernel"])
        st.metric("Sigma", r["sigma"])
        st.metric("Canny thresholds (lower / upper)", r["canny"])
        st.metric("GrabCut iterations", r["iterations"])
        st.metric("Morphological cleanup", r["cleanup"])
        st.metric("GrabCut rectangle (width \u00d7 height)", r["rectangle"])
    st.caption("The rectangle's position was not recorded in the report, so it is not drawn on the image.")

    st.subheader("Step 2 \u2014 Gaussian Smoothing", anchor=False)
    with_text(
        "step2_gaussian_reproduced.jpg",
        "Gaussian-smoothed image \u2014 reproduced using the documented settings (kernel 3, sigma 0.80)",
        "Smoothing reduces small variations and noise before edge detection. "
        "The visual change is subtle by design.",
    )

    st.subheader("Step 3 \u2014 Canny Edge Detection", anchor=False)
    with_text(
        "step3_canny.jpg", "Canny edge map (thresholds 50 / 150)",
        "Canny found strong intensity changes along the body, clothing, hair and face.",
        "It also marks internal and background edges and does not know which edges form the outer "
        "boundary, so edges alone cannot give a single human boundary.",
    )

    st.subheader("Step 4 \u2014 GrabCut Segmentation", anchor=False)
    with_text(
        "step4_grabcut_mask.png", "GrabCut binary mask (white = foreground)",
        "GrabCut separated foreground from background using the rectangle (5 iterations).",
        "Morphological cleanup was disabled, so the final foreground mask is the raw GrabCut mask.",
    )

    st.subheader("Step 5 \u2014 Classical Human Boundary", anchor=False)
    with_text(
        "step5_classical_boundary.png", "Classical boundary (largest external contour)",
        "The contour follows much of the visible person, but a large cast shadow near the lower right "
        "was included in the foreground. Because the shadow stays connected to the person's mask, "
        "the external contour follows both.",
    )
    st.metric("Classical contour area", r["contour_area"])
    st.caption("Geometric area enclosed by the contour, not a count of foreground pixels.")

    st.subheader("Step 6 \u2014 SAM2 Segmentation", anchor=False)
    st.write("SAM 2.1 Hiera Tiny, with a box-only prompt around the person on the same RGB image.")
    mask_column, boundary_column = st.columns(2)
    mask_column.image(str(DISPLAY_DIR / "step6_sam2_mask.png"), caption="SAM2 binary mask (from report Figure 6)", width="stretch")
    boundary_column.image(str(DISPLAY_DIR / "step6_sam2_boundary.png"), caption="SAM2 boundary (from report Figure 7)", width="stretch")
    score_column, device_column = st.columns(2)
    score_column.metric("Predicted mask-quality score", r["sam2_score"])
    device_column.metric("Device", r["sam2_device"])
    st.caption("The score is SAM2's own estimate of its mask, not measured accuracy; no ground-truth mask was used.")

    st.subheader("Step 7 \u2014 Classical vs. SAM2 Comparison", anchor=False)
    st.image(str(DISPLAY_DIR / "step7_mask_comparison.png"), caption="GrabCut mask vs. SAM2 mask", width="stretch")
    st.image(str(DISPLAY_DIR / "step7_boundary_comparison.png"), caption="Classical boundary vs. SAM2 boundary", width="stretch")
    st.write(
        "The classical mask and contour include a large part of the connected cast shadow. SAM2 follows "
        "the visible person more closely, especially around the lower body, and isolates the person "
        "without the shadow."
    )
    st.button("Back", key="m4_rgb_results_back_bottom", on_click=navigate_module4, args=("rgb_select",))


def panel_label(title, subtitle):
    return f"**{title}**  \n{subtitle}"


def module4_selection_page(back_key, heading, panels):
    st.button("Back", key=back_key, on_click=navigate_module4, args=("home",))
    st.header(heading, anchor=False)
    for key, title, subtitle, destination in panels:
        st.button(panel_label(title, subtitle), key=key, on_click=navigate_module4, args=(destination,))


def rgb_select_page():
    module4_selection_page("m4_rgb_select_back_top", "RGB Image Boundary Detection", (
        ("m4_rgb_visitor", "Detect Human Boundaries in Your Own RGB Image", "Upload and analyze your own RGB image", "rgb"),
        ("m4_rgb_results", "View My Experiment & Results", "Explore my RGB boundary detection results", "rgb_results")))


# Values below are the ones documented in my original Module 4 report (thermal experiment).
THERMAL_EXPERIMENT = {
    "size": "1080 × 1440 pixels", "closing": "3 × 3 elliptical", "connectivity": "8-connectivity",
    "components": "20", "foreground": "392,206 pixels", "occupancy": "25.22%",
    "sam2_box": "(263, 57, 574, 1380) pixels", "sam2_score": "0.977383", "sam2_device": "MPS",
    "iou": "0.9813",
}


def thermal_results_page():
    r = THERMAL_EXPERIMENT
    st.button("Back", key="m4_thermal_results_back_top", on_click=navigate_module4, args=("thermal_select",))
    st.header("My Thermal Experiment & Results", anchor=False)

    st.subheader("Step 1 — Original Thermal Image & Settings", anchor=False)
    image_column, text_column = st.columns([3, 2], vertical_alignment="center")
    image_column.image(str(DISPLAY_DIR / "thermal_step1_original.jpg"),
                       caption="Original thermal pseudocolor image", width="stretch")
    with text_column:
        st.metric("Image size (width × height)", r["size"])
        st.metric("Morphological closing", r["closing"])
        st.metric("Connected components", r["connectivity"])
    st.write(
        "The input is a colorized thermal image, not raw temperature data, so the classical method "
        "works on the displayed colors."
    )
    st.caption("The HSV threshold values were not recorded in the report, so they are not listed.")

    st.subheader("Step 2 — Classical Thermal Segmentation", anchor=False)
    st.write(
        "The image was converted to HSV and the warm colors of the person were thresholded away from the "
        "cooler background. A 3 × 3 elliptical closing connected small gaps, and connected-component "
        "analysis kept the largest warm region as the person."
    )
    components_column, foreground_column, occupancy_column = st.columns(3)
    components_column.metric("Foreground components found", r["components"])
    foreground_column.metric("Selected foreground size", r["foreground"])
    occupancy_column.metric("Image occupancy", r["occupancy"])
    st.caption("Occupancy is the share of all image pixels in the selected component, not segmentation accuracy.")

    st.subheader("Step 3 — SAM2 Thermal Segmentation", anchor=False)
    st.write("SAM2 was applied to the same thermal image with a box prompt around the person.")
    box_column, score_column, device_column = st.columns([2, 1, 1])
    box_column.metric("SAM2 box (x, y, width, height)", r["sam2_box"])
    score_column.metric("Predicted mask quality", r["sam2_score"])
    device_column.metric("Device", r["sam2_device"])
    st.caption(
        "The box only prompts SAM2; the classical method does not use it. The score is SAM2's own estimate "
        "of its mask, not measured accuracy."
    )

    st.subheader("Step 4 — Mask Comparison", anchor=False)
    st.image(str(DISPLAY_DIR / "thermal_step4_mask_comparison.png"),
             caption="Classical thermal mask vs. SAM2 mask (from report Figure 9)", width="stretch")
    st.write(
        "The two masks are visually very similar. Small differences appear around the head, hands, "
        "waist and lower body."
    )

    st.subheader("Step 5 — Boundary Comparison", anchor=False)
    st.image(str(DISPLAY_DIR / "thermal_step5_boundary_comparison.png"),
             caption="Classical thermal boundary vs. SAM2 boundary (from report Figure 10)", width="stretch")
    st.write(
        "Both boundaries closely follow the same person. The warm body is strongly separated from much of "
        "the cooler background, which makes the HSV-based classical method effective here."
    )

    st.subheader("Step 6 — Quantitative Comparison", anchor=False)
    st.metric("IoU with SAM2", r["iou"])
    st.write(
        "An IoU of 1.0 would mean the masks overlap perfectly, so 0.9813 indicates very high agreement "
        "between the classical thermal segmentation and SAM2."
    )
    st.caption("This is not 98.13% accuracy: SAM2 is a comparison segmentation, not manually labeled ground truth.")
    st.button("Back", key="m4_thermal_results_back_bottom", on_click=navigate_module4, args=("thermal_select",))


def thermal_select_page():
    module4_selection_page("m4_thermal_select_back_top", "Thermal Image Boundary Detection", (
        ("m4_thermal_visitor", "Detect Human Boundaries in Your Own Thermal Image", "Upload and analyze your own thermal image", "thermal"),
        ("m4_thermal_results", "View My Experiment & Results", "Explore my thermal boundary detection results", "thermal_results")))


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
if section == "home":
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    with st.container(key="m4_home_layout"):
        with st.container(key="m4_home_title"):
            st.title(HOME_TITLE, anchor=False)
        with st.container(key="m4_home_subtitle"):
            st.header(HOME_SUBTITLE, anchor=False)
        with st.container(key="m4_home_nav"):
            for label, destination in (("Detect Human Boundaries in RGB Images", "rgb"),
                                       ("Detect Human Boundaries in Thermal Images", "thermal")):
                st.button(label, key=f"m4_open_{destination}", on_click=navigate_module4,
                          args=("rgb_select" if destination == "rgb" else "thermal_select",), width="stretch")
elif section == "rgb_select":
    rgb_select_page()
elif section == "rgb_results":
    rgb_results_page()
elif section == "rgb":
    st.button("Back", key="m4_rgb_visitor_back_top", on_click=navigate_module4, args=("rgb_select",))
    st.title("Module 4 – Human Boundary Detection")
    render_rgb()
    st.button("Back", key="m4_rgb_visitor_back_bottom", on_click=navigate_module4, args=("rgb_select",))
elif section == "thermal_select":
    thermal_select_page()
elif section == "thermal_results":
    thermal_results_page()
elif section == "thermal":
    st.button("Back", key="m4_thermal_visitor_back_top", on_click=navigate_module4, args=("thermal_select",))
    st.title("Module 4 – Human Boundary Detection")
    render_thermal()
    st.button("Back", key="m4_thermal_visitor_back_bottom", on_click=navigate_module4, args=("thermal_select",))
