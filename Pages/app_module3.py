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

from Modules.Module_3.image_filtering import (
    create_gaussian_kernel,
    gaussian_blur,
    gaussian_blur_frequency,
)


st.set_page_config(page_title="Module 3 - Image Filtering", layout="centered")
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
        "Gaussian kernel size", options=list(range(1, 16, 2)), index=2,
        key="gaussian_kernel_size",
    )
with sigma_column:
    sigma = st.slider(
        "Sigma", min_value=0.1, max_value=5.0, value=1.5, step=0.1,
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
