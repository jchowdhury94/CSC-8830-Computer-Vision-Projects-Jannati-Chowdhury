"""Streamlit demonstration of spatial and frequency-domain Gaussian filtering."""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


# Locate the repository from this file, independently of the working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from home_navigation import back_button, nav_row  # noqa: E402
from Modules.Module_3.image_filtering import (
    create_gaussian_kernel,
    gaussian_blur,
    gaussian_blur_frequency,
)


st.set_page_config(page_title="Module 3 - Image Filtering", layout="centered")

DISPLAY_DIR = REPOSITORY_ROOT / "Modules" / "Module_3" / "display"

EXPERIMENT_KERNEL_SIZE = 9
EXPERIMENT_SIGMA = 2.0
HOME_TITLE = "Module 3"
HOME_SUBTITLE = "Gaussian Blur \u2014 Spatial & Frequency Domain"
ACCENT = "#C9363E"
# Widget values are dropped while their widgets are not rendered; these are kept across a visit to the home page.
VISITOR_INPUT_KEYS = ("gaussian_kernel_size", "gaussian_sigma")

PAGE_CSS = f"""
<style>
.st-key-m3_home_layout {{ gap: 0; padding-top: 1rem; }}  /* the nav row above already supplies the top space */
.st-key-m3_home_title, .st-key-m3_home_subtitle {{ gap: 0; flex-shrink: 0; }}
.st-key-m3_home_nav {{ flex-shrink: 0; }}
.st-key-m3_home_title [data-testid="stMarkdownContainer"],
.st-key-m3_home_subtitle [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.st-key-m3_home_subtitle {{ padding-top: 44px; }}
.st-key-m3_home_nav {{ padding-top: 64px; }}
.st-key-m3_home_title h1 {{
    font-size: clamp(2.5rem, 13vw, 80px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}}
.st-key-m3_home_subtitle h2 {{
    font-size: clamp(1.1rem, 4.6vw, 32px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2;
    text-align: center; padding: 0; margin: 0;
}}
@media (min-width: 641px) {{ .st-key-m3_home_subtitle h2 {{ white-space: nowrap; }} }}
.st-key-m3_home_nav {{ flex-direction: row; flex-wrap: wrap; gap: 24px; }}
.st-key-m3_open_visitor, .st-key-m3_open_experiment {{ flex: 1 1 280px; width: auto; min-width: 0; max-width: 100%; }}
.st-key-m3_open_visitor button, .st-key-m3_open_experiment button {{
    width: 100%; height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
.st-key-m3_open_visitor button p, .st-key-m3_open_experiment button p {{
    font-size: 21px; font-weight: 700; color: #FFFFFF; line-height: 1.3; text-align: center; text-wrap: balance;
}}
.st-key-m3_open_visitor button:hover, .st-key-m3_open_experiment button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.st-key-m3_open_visitor button:focus-visible, .st-key-m3_open_experiment button:focus-visible {{
    outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT};
}}
.st-key-m3_open_visitor button:active, .st-key-m3_open_experiment button:active {{ background-color: #8E2229; border-color: #8E2229; }}
@media (max-width: 640px) {{
    .st-key-m3_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 16px; }}
    .st-key-m3_open_visitor, .st-key-m3_open_experiment {{ flex: 0 0 auto; width: 100%; }}
}}
</style>
"""


def init_state():
    st.session_state.setdefault("m3_page", "home")           # "home" / "visitor" / "experiment"
    st.session_state.setdefault("m3_saved_inputs", {})


def go_to(page):
    """Switches the page, keeping the sigma and kernel-size values while the visitor page is not rendered."""
    ss = st.session_state
    if ss["m3_page"] == "visitor" and page != "visitor":
        ss["m3_saved_inputs"] = {key: ss[key] for key in VISITOR_INPUT_KEYS if key in ss}
    elif page == "visitor":
        for key, value in ss["m3_saved_inputs"].items():
            ss[key] = value
    ss["m3_page"] = page


def home_page():
    with st.container(key="m3_home_layout"):
        with st.container(key="m3_home_title"):
            st.title(HOME_TITLE, anchor=False)
        with st.container(key="m3_home_subtitle"):
            st.header(HOME_SUBTITLE, anchor=False)
        with st.container(key="m3_home_nav"):
            st.button("Apply Gaussian Blur to Your Own Image", key="m3_open_visitor",
                      on_click=go_to, args=("visitor",), width="stretch")
            st.button("View My Experiment & Results", key="m3_open_experiment",
                      on_click=go_to, args=("experiment",), width="stretch")


def experiment_page():
    # Values below are the ones documented in my original Module 3 report.
    kernel = create_gaussian_kernel(EXPERIMENT_KERNEL_SIZE, EXPERIMENT_SIGMA)

    st.header("My Experiment & Results", anchor=False)

    st.subheader("Step 1 \u2014 Original Image & Parameters", anchor=False)
    image_column, text_column = st.columns([3, 2], vertical_alignment="center")
    with image_column:
        st.image(str(DISPLAY_DIR / "step_original.png"), caption="Original flower image", width="stretch")
    with text_column:
        st.metric("Kernel size", f"{EXPERIMENT_KERNEL_SIZE} \u00d7 {EXPERIMENT_KERNEL_SIZE}")
        st.metric("Sigma", f"{EXPERIMENT_SIGMA:.1f}")

    st.subheader("Step 2 \u2014 Gaussian Kernel", anchor=False)
    st.dataframe(kernel, width="stretch")
    st.metric("Kernel sum", f"{kernel.sum():.12f}")

    st.subheader("Step 3 \u2014 Spatial Gaussian Blur", anchor=False)
    image_column, text_column = st.columns([3, 2], vertical_alignment="center")
    with image_column:
        st.image(str(DISPLAY_DIR / "step_spatial_blur.png"), caption="Spatially blurred image", width="stretch")
    with text_column:
        st.write(
            "I slid the kernel across the image one pixel at a time, multiplying the neighboring pixels "
            "by the kernel weights and summing them, one color channel at a time."
        )

    st.subheader("Step 4 \u2014 Frequency-Domain Blur", anchor=False)
    image_column, text_column = st.columns([3, 2], vertical_alignment="center")
    with image_column:
        st.image(str(DISPLAY_DIR / "step_fft_blur.png"), caption="Frequency-domain result", width="stretch")
    with text_column:
        st.write(
            "I transformed the image and the kernel with the FFT, multiplied their spectra, "
            "and applied the inverse FFT to return the blurred image."
        )

    st.subheader("Step 5 \u2014 Compare & Validate Results", anchor=False)
    spatial_column, frequency_column = st.columns(2)
    with spatial_column:
        st.image(str(DISPLAY_DIR / "step_spatial_blur.png"), caption="Spatial Domain Result", width="stretch")
    with frequency_column:
        st.image(str(DISPLAY_DIR / "step_fft_blur.png"), caption="Frequency Domain Result", width="stretch")
    maximum_column, mean_column = st.columns(2)
    with maximum_column:
        st.metric("Maximum Absolute Difference", "5.115908 \u00d7 10\u207b\u00b9\u00b3")
    with mean_column:
        st.metric("Mean Absolute Difference", "8.668103 \u00d7 10\u207b\u00b9\u2074")
    st.write(
        "The two results are visually identical and differ only at floating-point precision, "
        "so the methods are numerically equivalent."
    )

    st.subheader("Conclusion", anchor=False)
    st.write(
        "Blurring with the same 9 \u00d7 9 Gaussian kernel (sigma = 2.0) in the spatial and frequency "
        "domains gave visually identical images, with a maximum absolute difference of "
        "5.115908 \u00d7 10\u207b\u00b9\u00b3. This confirms the convolution theorem for this filter."
    )
    back_button("m3_back_experiment_bottom", go_to, ("home",))


def visitor_page():
    # Seed the controls through session state only; passing value=/index= as well triggers Streamlit's duplicate-default warning.
    st.session_state.setdefault("gaussian_kernel_size", 5)
    st.session_state.setdefault("gaussian_sigma", 1.5)
    st.title("Module 3 - Image Filtering")
    st.subheader("Spatial Gaussian Blurring")

    st.session_state.setdefault("module3_uploader_version", 0)
    uploader_key = f"module3_upload_{st.session_state['module3_uploader_version']}"
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key=uploader_key
    )


    def clear_fourier_results():
        st.session_state.pop("module3_frequency_result", None)
        st.session_state["module3_show_comparison"] = False


    def reset_gaussian_controls():
        st.session_state["gaussian_kernel_size"] = 5
        st.session_state["gaussian_sigma"] = 1.5
        clear_fourier_results()


    def reset_all():
        reset_gaussian_controls()
        st.session_state.pop("module3_result_inputs", None)
        st.session_state.pop("module3_show_comparison", None)
        # A fresh widget key clears the visible uploader as well as its old value.
        old_key = f"module3_upload_{st.session_state['module3_uploader_version']}"
        st.session_state.pop(old_key, None)
        st.session_state["module3_uploader_version"] += 1


    kernel_column, sigma_column = st.columns(2)
    with kernel_column:
        kernel_size = st.selectbox(
            "Gaussian kernel size", options=list(range(1, 16, 2)),
            key="gaussian_kernel_size",
        )
    with sigma_column:
        sigma = st.slider(
            "Sigma", min_value=0.1, max_value=5.0, step=0.1,
            key="gaussian_sigma",
        )

    st.button("Reset", on_click=reset_gaussian_controls)

    # A result belongs to one uploaded image and one set of filter parameters.
    # Invalidate it even when the upload is replaced or removed.
    image_digest = (
        hashlib.sha256(uploaded_image.getvalue()).hexdigest()
        if uploaded_image is not None else None
    )
    result_inputs = (image_digest, kernel_size, sigma)
    if st.session_state.get("module3_result_inputs") != result_inputs:
        clear_fourier_results()
        st.session_state["module3_result_inputs"] = result_inputs

    kernel = create_gaussian_kernel(kernel_size, sigma)
    image = None

    if uploaded_image is None:
        st.info("Upload a JPG, JPEG, or PNG image to see the filtering result.")
    else:
        # Decode to 8-bit pixels, retaining grayscale or color as appropriate.
        file_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_ANYCOLOR) if file_bytes.size else None

        if image is None:
            st.error("This file could not be read as an image. Please upload another image.")
        else:
            # OpenCV decodes color as BGR; Streamlit displays color as RGB.
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            original_column, blurred_column = st.columns(2)
            with original_column:
                st.image(image, caption="Original image", width="stretch")

            with st.spinner("Applying spatial Gaussian convolution..."):
                filtered_image = gaussian_blur(image, kernel_size, sigma)

            # The filtering function returns float64 in the 0–255 intensity scale.
            # Round and clamp before converting to uint8 for correct image display.
            display_image = np.clip(np.rint(filtered_image), 0, 255).astype(np.uint8)
            with blurred_column:
                st.image(display_image, caption="Spatially blurred image", width="stretch")

    st.markdown("#### Gaussian Kernel")
    st.dataframe(kernel, width="stretch")
    st.metric("Kernel sum", f"{kernel.sum():.12f}")

    st.divider()
    st.subheader("Fourier / Frequency Domain Gaussian Blurring")
    apply_fourier = st.button("Apply Fourier Filter", disabled=image is None)

    if image is not None and apply_fourier:
        # Keep the raw float64 result across reruns for this image and parameters.
        with st.spinner("Applying frequency-domain Gaussian filtering..."):
            st.session_state["module3_frequency_result"] = gaussian_blur_frequency(
                image, kernel_size, sigma
            )

    frequency_result = st.session_state.get("module3_frequency_result")
    if image is not None and frequency_result is not None:
        frequency_display = np.clip(np.rint(frequency_result), 0, 255).astype(np.uint8)
        st.image(frequency_display, caption="Frequency Domain Result", width="stretch")
        if st.button("Compare Results"):
            st.session_state["module3_show_comparison"] = True

    show_comparison = st.session_state.get("module3_show_comparison", False)
    if image is not None and frequency_result is not None and show_comparison:
        # Compare raw float64 results, not the rounded display images.
        absolute_difference = np.abs(filtered_image - frequency_result)
        max_absolute_difference = float(absolute_difference.max())
        mean_absolute_difference = float(absolute_difference.mean())

        st.divider()
        st.subheader("Spatial vs. Frequency Domain Validation")
        st.caption(
            f"Both methods use the same uploaded image, {kernel_size} × {kernel_size} "
            f"Gaussian kernel, sigma = {sigma:.1f}, and reflected image boundaries."
        )
        spatial_column, frequency_column = st.columns(2)
        with spatial_column:
            st.image(display_image, caption="Spatial Domain Result", width="stretch")
        with frequency_column:
            st.image(frequency_display, caption="Frequency Domain Result", width="stretch")

        maximum_column, mean_column = st.columns(2)
        with maximum_column:
            st.metric("Maximum Absolute Difference", f"{max_absolute_difference:.6e}")
        with mean_column:
            st.metric("Mean Absolute Difference (MAE)", f"{mean_absolute_difference:.6e}")
        st.write(
            "Values extremely close to zero show that the two methods produce "
            "numerically equivalent results. Tiny differences are expected from "
            "floating-point computation."
        )

    st.button("Reset All", on_click=reset_all)
    back_button("m3_back_bottom", go_to, ("home",))


init_state()
page = st.session_state["m3_page"]
# breadcrumb levels below Home: (label, page); the last is the current page
CRUMBS = {"home": (("Module 3", "home"),),
          "visitor": (("Module 3", "home"), ("Apply Gaussian Blur to Your Own Image", "visitor")),
          "experiment": (("Module 3", "home"), ("My Experiment & Results", "experiment"))}
nav_row(CRUMBS[page], go_to, PAGE_CSS, view=f"m3:{page}")
if page == "visitor":
    visitor_page()
elif page == "experiment":
    experiment_page()
else:
    home_page()
