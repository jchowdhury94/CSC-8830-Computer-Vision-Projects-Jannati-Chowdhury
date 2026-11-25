"""Streamlit demonstration of spatial and frequency-domain Gaussian filtering."""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# Locate the repository from this file, independently of the working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from home_navigation import nav_row  # noqa: E402
from Modules.Module_3.image_filtering import (
    create_gaussian_kernel,
    gaussian_blur,
    gaussian_blur_frequency,
)


st.set_page_config(page_title="Module 3 - Image Filtering", layout="centered")

DISPLAY_DIR = REPOSITORY_ROOT / "Modules" / "Module_3" / "display"

EXPERIMENT_KERNEL_SIZE = 9
EXPERIMENT_SIGMA = 2.0
# Metrics documented in my original Module 3 report (not recomputed by this page).
DEFAULT_KERNEL_SIZE, DEFAULT_SIGMA = 1, 0.1   # an identity filter: the educational baseline
LARGE_IMAGE_PIXELS = 4_000_000   # above this a warning says the spatial filter may take longer
AGREEMENT_THRESHOLD = 1e-6       # intensity levels; far above floating-point rounding, far below one gray level
MAX_DIFFERENCE_TEXT = "5.115908 \u00d7 10\u207b\u00b9\u00b3"
MEAN_DIFFERENCE_TEXT = "8.668103 \u00d7 10\u207b\u00b9\u2074"
HOME_TITLE = "Module 3"
HOME_SUBTITLE = "Gaussian Blur \u2014 Spatial & Frequency Domain"
ACCENT = "#C9363E"
# Widget values are dropped while their widgets are not rendered; these are kept across a visit to the home page.
VISITOR_INPUT_KEYS = ("gaussian_kernel_size", "gaussian_sigma")

BLUE, BLUE_DARK = "#1D5FD1", "#174DA8"   # same blue as the Module 2 / Module 7 homepages


def _svg_mask(path):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='{path}'/></svg>"
    url = "url(\"data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23") + "\")"
    return f"-webkit-mask-image: {url}; mask-image: {url};"


CHEVRON_MASK = _svg_mask("M16.59 8.59 12 13.17 7.41 8.59 6 10l6 6 6-6z")
NAV_KEYS = ("m3_open_visitor", "m3_open_experiment")
# Homepage styling mirrors Pages/app_module2.py (HEADING_CSS home rules, _BUTTON_TEMPLATE, HOME_CARD_CSS), scoped to m3_* keys.
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
BUTTON_CSS = "\n".join(
    _BUTTON_TEMPLATE.replace("{k}", f".st-key-{key}") + f".st-key-{key} button::after {{ transform: rotate(-90deg); }}"
    for key in NAV_KEYS)
HEADING_CSS = """
.st-key-m3_home_layout { gap: 0; padding-top: 1rem; }  /* the nav row above already supplies the top space */
.st-key-m3_home_title, .st-key-m3_home_subtitle { gap: 0; }
.st-key-m3_home_title, .st-key-m3_home_subtitle, .st-key-m3_home_nav { flex-shrink: 0; }
.st-key-m3_home_title [data-testid="stMarkdownContainer"], .st-key-m3_home_subtitle [data-testid="stMarkdownContainer"] { margin-bottom: 0; }
.st-key-m3_home_subtitle { padding-top: 44px; }
.st-key-m3_home_nav { padding-top: 64px; }
.st-key-m3_home_title h1 {
    font-size: clamp(2.5rem, 13vw, 80px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}
.st-key-m3_home_subtitle h2 {
    font-size: clamp(1.1rem, 4.6vw, 32px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2;
    text-align: center; padding: 0; margin: 0;
}
@media (min-width: 801px) { .st-key-m3_home_subtitle h2 { white-space: nowrap; } }
"""
HOME_CARD_CSS = f"""
.st-key-m3_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 24px; max-width: 560px; width: 100%; margin: 0 auto; }}
.st-key-m3_open_visitor, .st-key-m3_open_experiment {{ flex: 0 0 auto; width: 100%; min-width: 0; max-width: 100%; }}
.st-key-m3_open_visitor button, .st-key-m3_open_experiment button {{
    height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
.st-key-m3_open_visitor button p, .st-key-m3_open_experiment button p {{
    font-size: 18px; font-weight: 500; line-height: 1.3; text-align: center; text-wrap: balance;
}}
.st-key-m3_open_visitor button::after, .st-key-m3_open_experiment button::after {{ content: none; display: none; }}
.st-key-m3_open_experiment button {{ background-color: {BLUE}; border-color: {BLUE}; }}
.st-key-m3_open_experiment button:hover {{ background-color: {BLUE_DARK}; border-color: {BLUE_DARK}; }}
.st-key-m3_open_experiment button:active {{ background-color: #123F8F; border-color: #123F8F; }}
.st-key-m3_open_experiment button:focus-visible {{ box-shadow: 0 0 0 6px {BLUE}; }}
@media (max-width: 640px) {{ .st-key-m3_home_nav {{ gap: 16px; }} }}
"""
RESULTS_CSS = """
.st-key-m3_res_head { gap: 0; padding-bottom: 28px; }
.st-key-m3_res_head h1 {
    font-size: clamp(2.5rem, 8vw, 60px); font-weight: 800; line-height: 1.1; text-align: center; padding: 0; margin: 0;
    color: inherit; overflow-wrap: anywhere;
}
.st-key-m3_res_head h2 {
    font-size: clamp(1.4rem, 4.6vw, 2.4rem); font-weight: 600; line-height: 1.2; text-align: center; padding: 0; margin: 0;
    color: inherit; opacity: 0.85; overflow-wrap: anywhere;
}
.st-key-m3_res_title { padding-bottom: 28px; }
.st-key-m3_res_body h2 { text-align: center; font-size: clamp(1.7rem, 5.2vw, 2.4rem); font-weight: 800; line-height: 1.2; padding: 1.5rem 0 0.25rem; overflow-wrap: anywhere; }
.st-key-m3_res_body h3 { text-align: center; font-size: clamp(1.4rem, 4.5vw, 1.9rem); font-weight: 700; padding-top: 1.5rem; }
.st-key-m3_res_body h4 { text-align: left; font-size: 1.15rem; padding: 1.1rem 0 0.2rem; }
.st-key-m3_res_body h5 { text-align: left; font-weight: 700; margin: 0; padding: 0.5rem 0 0.25rem; }
.st-key-m3_res_body [data-testid="stMetric"] { text-align: left; }
.st-key-m3_res_body [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 5vw, 1.6rem); }
.st-key-m3_res_body [data-testid="stMetricValue"], .st-key-m3_res_body [data-testid="stMetricValue"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; }
.st-key-m3_res_body [data-testid="stMetricLabel"] p, .st-key-m3_res_body [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
.st-key-m3_res_body [data-testid="stTable"] { overflow-x: auto; }
.st-key-m3_res_body img { max-width: 100%; height: auto; }
[class*="st-key-m3_res_sec_"] { border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent); margin-top: 1.5rem; padding-top: 0.5rem; }
.st-key-m3_res_conclusion {
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 16px; padding: 0.5rem 28px 28px; margin-top: 2rem;
}
@media (max-width: 640px) { .st-key-m3_res_conclusion { padding: 0.25rem 14px 18px; } }
"""
VISITOR_CSS = f"""
.st-key-m3_headings h1 {{
    font-size: clamp(1.75rem, 5vw + 0.5rem, 3rem); font-weight: 800; line-height: 1.15; text-align: left;
    padding: 0.25rem 0 0; margin: 0; overflow-wrap: anywhere;
}}
.st-key-m3_headings h2 {{
    font-size: clamp(1rem, 1.6vw + 0.5rem, 1.3rem); font-weight: 500; line-height: 1.3; text-align: left; opacity: 0.8;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}}
.st-key-m3_vis_cards [data-testid="stMetric"] {{ text-align: left; }}
.st-key-m3_vis_cards [data-testid="stMetricValue"] {{ font-size: clamp(1.3rem, 5vw, 1.6rem); }}
.st-key-m3_vis_cards [data-testid="stMetricValue"], .st-key-m3_vis_cards [data-testid="stMetricValue"] * {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; }}
.st-key-m3_vis_cards [data-testid="stMetricLabel"] p, .st-key-m3_vis_cards [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }}
.st-key-m3_generate_kernel button, .st-key-m3_apply_filters button {{
    background-color: {ACCENT}; border: 2px solid {ACCENT}; border-radius: 12px; color: #FFFFFF; min-height: 3rem;
}}
.st-key-m3_generate_kernel button p, .st-key-m3_apply_filters button p {{ color: #FFFFFF; font-weight: 700; }}
.st-key-m3_generate_kernel button:hover, .st-key-m3_apply_filters button:hover {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
.st-key-m3_generate_kernel button:active, .st-key-m3_apply_filters button:active {{ background-color: #8E2229; border-color: #8E2229; }}
.st-key-m3_generate_kernel button:focus-visible, .st-key-m3_apply_filters button:focus-visible {{ outline: 3px solid {ACCENT}; outline-offset: 2px; }}
"""
PAGE_CSS = "<style>" + HEADING_CSS + BUTTON_CSS + HOME_CARD_CSS + RESULTS_CSS + VISITOR_CSS + "</style>"


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
            st.button("View My Experiments & Results", key="m3_open_experiment",
                      on_click=go_to, args=("experiment",), width="stretch")
            st.button("Apply Gaussian Blur to Your Own Image", key="m3_open_visitor",
                      on_click=go_to, args=("visitor",), width="stretch")


def metric_card(label, value, note=None):
    with st.container(border=True):
        st.metric(label, value)
        if note:
            st.caption(note)


def display_image(name, caption):
    """One saved display image of my original experiment, shown at the width of its column (aspect ratio preserved)."""
    st.image(str(DISPLAY_DIR / name), caption=caption, width="stretch")


def experiment_page():
    # Values below are the ones documented in my original Module 3 report.
    kernel = create_gaussian_kernel(EXPERIMENT_KERNEL_SIZE, EXPERIMENT_SIGMA)
    size_text = f"{EXPERIMENT_KERNEL_SIZE} × {EXPERIMENT_KERNEL_SIZE}"

    with st.container(key="m3_res_head"):
        with st.container(key="m3_res_title"):
            st.title("My Experiment & Results", anchor=False)
        st.header(HOME_SUBTITLE, anchor=False)

    with st.container(key="m3_res_body"):
        # ---- objective and summary
        with st.container(key="m3_res_objective"):
            st.markdown("##### Objective")
            st.write("I implemented Gaussian image blurring using spatial and frequency-domain filtering and experimentally "
                     "demonstrated their equivalence using the convolution theorem.")
        for column, card in zip(st.columns(2), (("Kernel Size", size_text, "Gaussian kernel"),
                                                ("Sigma", f"{EXPERIMENT_SIGMA:.1f}", "standard deviation of the Gaussian"))):
            with column:
                metric_card(*card)
        for column, card in zip(st.columns(2), (
                ("Maximum Absolute Difference", MAX_DIFFERENCE_TEXT, "spatial vs. frequency, original experiment"),
                ("Mean Absolute Difference (MAE)", MEAN_DIFFERENCE_TEXT, "spatial vs. frequency, original experiment"))):
            with column:
                metric_card(*card)

        # ---- 1. experimental setup
        with st.container(key="m3_res_sec_1"):
            st.header("1. Experimental Setup", anchor=False)
            st.write("I used the same image and Gaussian filter parameters for both filtering methods to ensure a consistent "
                     "comparison.")
            image_column, table_column = st.columns(2, vertical_alignment="center")
            with image_column:
                display_image("step_original.png", "Original flower image, used as the input for both filtering methods.")
            with table_column:
                st.table(pd.DataFrame({"Parameter": ["Kernel size", "Sigma (σ)", "Boundary handling", "Color processing"],
                                       "Value": [size_text, f"{EXPERIMENT_SIGMA:.1f}", "Reflection",
                                                 "Each channel independently"]}).set_index("Parameter"))

        # ---- 2. Gaussian kernel generation
        with st.container(key="m3_res_sec_2"):
            st.header("2. Gaussian Kernel Generation", anchor=False)
            st.write("A Gaussian filter gives greater weight to pixels near the center of the kernel and smaller weight to pixels "
                     "farther away.")
            st.latex(r"G(x, y) = \exp\!\left[-\dfrac{x^{2} + y^{2}}{2\sigma^{2}}\right]")
            st.latex(r"h(x, y) = \dfrac{G(x, y)}{\sum_{u}\sum_{v} G(u, v)}")
            st.write("Here $x$ and $y$ are spatial offsets from the kernel center, $\\sigma$ is the standard deviation, and $u$ and "
                     "$v$ are summation indices over the kernel. Normalization ensures that the kernel weights sum to 1.")
            offsets = [str(value) for value in range(-(EXPERIMENT_KERNEL_SIZE // 2), EXPERIMENT_KERNEL_SIZE // 2 + 1)]
            st.table(pd.DataFrame([[f"{value:.6f}" for value in row] for row in kernel], index=offsets, columns=offsets))
            st.caption(f"Normalized {size_text} Gaussian kernel h(x, y) with σ = {EXPERIMENT_SIGMA:.1f}. "
                       f"Kernel sum = {kernel.sum():.12f}.")

        # ---- 3. spatial-domain filtering
        with st.container(key="m3_res_sec_3"):
            st.header("3. Spatial-Domain Gaussian Filtering", anchor=False)
            st.write("In spatial filtering, the Gaussian kernel moves across the image. Each output pixel is calculated as a "
                     "weighted sum of its neighboring pixels.")
            st.latex(r"g(x, y) = \sum_{u}\sum_{v} h(u, v)\, f(x-u,\; y-v)")
            st.write("Here $f$ is the input image, $h$ is the Gaussian kernel, $g$ is the filtered image, $x$ and $y$ are pixel "
                     "coordinates, and $u$ and $v$ are kernel offsets.")
            _, middle, _ = st.columns([1, 3, 1])
            with middle:
                display_image("step_spatial_blur.png", "Spatially blurred image (9 × 9 Gaussian kernel, σ = 2.0).")

        # ---- 4. frequency-domain filtering
        with st.container(key="m3_res_sec_4"):
            st.header("4. Frequency-Domain Gaussian Filtering", anchor=False)
            st.write("In frequency-domain filtering, the image and Gaussian kernel are transformed into the Fourier domain. Their "
                     "transforms are multiplied, and the inverse Fourier transform produces the blurred image.")
            st.latex(r"F = \mathcal{F}\{f\}")
            st.latex(r"H = \mathcal{F}\{h\}")
            st.latex(r"g = \mathcal{F}^{-1}\{F \cdot H\}")
            st.write("Here $F$ and $H$ are the Fourier transforms of the image and the Gaussian kernel, respectively. The "
                     "implementation uses the same reflected boundaries as the spatial method, sufficient FFT zero-padding, and "
                     "output alignment, so that both methods calculate the equivalent linear convolution. The equations describe "
                     "the convolution mathematically; the implementation extracts the correctly aligned, original-sized output.")
            _, middle, _ = st.columns([1, 3, 1])
            with middle:
                display_image("step_fft_blur.png", "Frequency-domain blurred image (same kernel, same boundary).")

        # ---- 5. convolution theorem (theory and experimental validation)
        with st.container(key="m3_res_sec_5"):
            st.header("5. Convolution Theorem (Theory & Experimental Validation)", anchor=False)
            st.subheader("Mathematical Demonstration", anchor=False)
            st.write("The convolution theorem states that convolution in the spatial domain is equivalent to multiplication in the "
                     "frequency domain.")
            st.write("Spatial-domain filtering is")
            st.latex(r"g_{\mathrm{s}} = f * h")
            st.write("Taking the Fourier transform of both sides gives")
            st.latex(r"\mathcal{F}\{g_{\mathrm{s}}\} = \mathcal{F}\{f * h\}")
            st.write("According to the convolution theorem,")
            st.latex(r"\mathcal{F}\{g_{\mathrm{s}}\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{h\}")
            st.write("Applying the inverse Fourier transform gives")
            st.latex(r"g_{\mathrm{s}} = \mathcal{F}^{-1}\{\mathcal{F}\{f\} \cdot \mathcal{F}\{h\}\}")
            st.write("The right-hand side is frequency-domain filtering. With consistent boundary handling, sufficient padding and "
                     "output alignment, the two outputs are therefore mathematically equivalent:")
            st.latex(r"\boxed{\,g_{\mathrm{s}} = g_{\mathrm{f}}\,}")
            st.markdown(
                "- $f$: input image\n"
                "- $h$: Gaussian kernel\n"
                "- $g_{\\mathrm{s}}$: spatial-domain output\n"
                "- $g_{\\mathrm{f}}$: frequency-domain output (the subscript $\\mathrm{f}$ stands for frequency)\n"
                "- $*$: convolution\n"
                "- $\\mathcal{F}$: Fourier transform, and $\\mathcal{F}^{-1}$: inverse Fourier transform\n"
                "- $\\cdot$: element-wise multiplication in the frequency domain"
            )

            st.subheader("Experimental Evidence & Validation", anchor=False)
            st.write("I applied both methods to the same image using the same 9 × 9 Gaussian kernel and sigma of 2.0.")
            image_column, frequency_column = st.columns(2)
            with image_column:
                display_image("step_spatial_blur.png", "Spatial-domain result.")
            with frequency_column:
                display_image("step_fft_blur.png", "Frequency-domain result.")
            st.latex(r"D_i = \left|\, g_{\mathrm{s},i} - g_{\mathrm{f},i} \,\right|")
            st.latex(r"\text{Maximum Absolute Difference} = \max_i D_i")
            st.latex(r"\mathrm{MAE} = \dfrac{1}{N}\sum_{i=1}^{N} D_i")
            st.write("Here $i$ is the index of an individual pixel-channel value and $N$ is the total number of values compared.")
            for column, card in zip(st.columns(2), (
                    ("Maximum Absolute Difference", MAX_DIFFERENCE_TEXT, "original experiment"),
                    ("Mean Absolute Difference (MAE)", MEAN_DIFFERENCE_TEXT, "original experiment"))):
                with column:
                    metric_card(*card)
            st.caption("These values are the results documented in my original experiment.")
            st.write("The two outputs are visually indistinguishable, and the documented numerical differences are extremely close "
                     "to zero. These small differences are consistent with floating-point rounding.")
            st.write("The experimental results therefore support the convolution theorem and demonstrate the equivalence of "
                     "spatial and frequency-domain Gaussian filtering for the tested image and parameters.")

        # ---- 6. conclusion
        with st.container(key="m3_res_conclusion"):
            st.header("6. Conclusion", anchor=False)
            st.markdown("Gaussian image blurring was successfully implemented using spatial convolution and frequency-domain "
                        "multiplication. Both methods produced visually identical results with negligible numerical differences, "
                        "experimentally supporting the convolution theorem.")


def clear_visitor_results():
    """Forget the generated kernel and both filtering results (they belong to one image and one set of parameters)."""
    ss = st.session_state
    for key in ("module3_kernel_inputs", "module3_spatial_result", "module3_frequency_result"):
        ss.pop(key, None)


def reset_gaussian_controls():
    st.session_state["gaussian_kernel_size"] = DEFAULT_KERNEL_SIZE
    st.session_state["gaussian_sigma"] = DEFAULT_SIGMA
    clear_visitor_results()


def reset_all():
    """Back to Step 1: no image, default parameters, nothing generated."""
    ss = st.session_state
    reset_gaussian_controls()
    ss.pop("module3_result_inputs", None)
    # A fresh widget key clears the visible uploader as well as its old value.
    ss.pop(f"module3_upload_{ss['module3_uploader_version']}", None)
    ss["module3_uploader_version"] += 1


def decode_upload(uploaded_image):
    """Decode an upload to an 8-bit grayscale or RGB array; None when it is not a readable image."""
    file_bytes = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_ANYCOLOR) if file_bytes.size else None
    if image is not None and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     # OpenCV decodes BGR; Streamlit displays RGB
    return image


_SUPERSCRIPT = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def scientific_text(value):
    """Format a number like the documented results: 4.547474 \u00d7 10\u207b\u00b9\u00b3 (six decimals, superscript exponent)."""
    mantissa, exponent = f"{float(value):.6e}".split("e")
    return f"{mantissa} \u00d7 10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def to_display(values):
    """The filtering functions return float64 in the 0-255 scale; round and clamp only for display."""
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def visitor_page():
    ss = st.session_state
    # Seed the controls through session state only; passing value=/index= as well triggers Streamlit's duplicate-default warning.
    # The values are also remembered under a second key, because Streamlit drops a widget's value while the widget is not shown
    # (no valid image yet), and the settings should survive removing or replacing the upload.
    ss.setdefault("gaussian_kernel_size", ss.get("m3_remembered_size", DEFAULT_KERNEL_SIZE))
    ss.setdefault("gaussian_sigma", ss.get("m3_remembered_sigma", DEFAULT_SIGMA))
    ss.setdefault("module3_uploader_version", 0)

    with st.container(key="m3_headings"):
        st.title("Apply Gaussian Blur to Your Own Image", anchor=False)
        st.header(HOME_SUBTITLE, anchor=False)

    # ---- Step 1: upload
    st.subheader("Step 1: Upload Your Image", anchor=False)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],
                                      key=f"module3_upload_{ss['module3_uploader_version']}")
    st.caption(f"Supported formats: JPG, JPEG and PNG. Maximum upload size: {st.get_option('server.maxUploadSize')} MB per file.")

    image = None
    digest = None
    if uploaded_image is not None:
        image = decode_upload(uploaded_image)
        if image is None:
            st.error("This file could not be read as an image. Please upload another image.")
        else:
            digest = hashlib.sha256(uploaded_image.getvalue()).hexdigest()
            height, width = image.shape[:2]
            st.image(image, caption="Original image", width="stretch")
            st.caption(f"{uploaded_image.name} · {uploaded_image.size / 1e6:.2f} MB · {width} × {height} pixels")
    if image is None:
        clear_visitor_results()
        ss.pop("module3_result_inputs", None)
        return

    # ---- Step 2: configure
    st.subheader("Step 2: Configure Gaussian Filter", anchor=False)
    kernel_column, sigma_column = st.columns(2)
    with kernel_column:
        kernel_size = st.selectbox("Gaussian kernel size", options=list(range(1, 16, 2)),
                                   format_func=lambda size: f"{size} × {size}", key="gaussian_kernel_size")
    with sigma_column:
        sigma = st.slider("Sigma", min_value=0.1, max_value=5.0, step=0.1, key="gaussian_sigma")
    ss["m3_remembered_size"], ss["m3_remembered_sigma"] = kernel_size, sigma
    if kernel_size == 1:
        st.write("A 1 \u00d7 1 Gaussian kernel acts as an identity filter, leaving the image unchanged regardless of sigma. "
                 "Increase the kernel size and adjust sigma to explore the effects of Gaussian blurring.")
    st.button("Reset", on_click=reset_gaussian_controls)

    # Everything below belongs to one image and one set of parameters: a change to either hides Steps 3-5 again.
    inputs = (digest, kernel_size, sigma)
    if ss.get("module3_result_inputs") != inputs:
        clear_visitor_results()
        ss["module3_result_inputs"] = inputs
    if st.button("Generate Gaussian Kernel", key="m3_generate_kernel", type="primary"):
        ss["module3_kernel_inputs"] = inputs

    if ss.get("module3_kernel_inputs") == inputs:
        # ---- Step 3: kernel
        kernel = create_gaussian_kernel(kernel_size, sigma)
        st.subheader("Step 3: Gaussian Kernel", anchor=False)
        st.dataframe(kernel, width="stretch")
        st.metric("Kernel sum", f"{kernel.sum():.12f}")      # the full-precision kernel, not the rounded table
        st.write("Each entry is the weight given to a pixel at that offset from the center: the weights are largest at the center "
                 "and fall off with distance, and a larger sigma spreads them over a wider neighborhood. The kernel is normalized "
                 "so that its weights sum to 1, which preserves constant intensity regions and prevents unintended intensity "
                 "scaling.")
        if image.shape[0] * image.shape[1] > LARGE_IMAGE_PIXELS:
            st.warning("This is a large image. The spatial filter processes every pixel and may take considerably longer than "
                       "the frequency-domain filter.")
        if st.button("Apply Both Filters", key="m3_apply_filters", type="primary"):
            ss.pop("module3_spatial_result", None)
            ss.pop("module3_frequency_result", None)
            try:
                with st.spinner("Applying spatial and frequency-domain Gaussian filtering..."):
                    spatial = gaussian_blur(image, kernel_size, sigma)
                    frequency = gaussian_blur_frequency(image, kernel_size, sigma)
            except Exception as error:                        # nothing partial is stored when either method fails
                st.error(f"The filters could not be applied to this image: {error}")
            else:
                ss["module3_spatial_result"], ss["module3_frequency_result"] = spatial, frequency

    spatial = ss.get("module3_spatial_result")
    frequency = ss.get("module3_frequency_result")
    if spatial is not None and frequency is not None:
        # ---- Step 4: results
        st.subheader("Step 4: Filtering Results", anchor=False)
        spatial_column, frequency_column = st.columns(2)
        with spatial_column:
            st.image(to_display(spatial), caption="Spatial-Domain Result", width="stretch")
        with frequency_column:
            st.image(to_display(frequency), caption="Frequency-Domain Result", width="stretch")

        # ---- Step 5: compare and validate (raw float64 outputs, before display rounding)
        absolute_difference = np.abs(spatial - frequency)
        maximum = float(absolute_difference.max())
        mean = float(absolute_difference.mean())
        st.subheader("Step 5: Compare & Validate Results", anchor=False)
        with st.container(key="m3_vis_cards"):
            maximum_column, mean_column = st.columns(2)
            with maximum_column:
                metric_card("Maximum Absolute Difference", scientific_text(maximum))
            with mean_column:
                metric_card("Mean Absolute Difference (MAE)", scientific_text(mean))
        st.caption(f"Both methods used the same uploaded image, a {kernel_size} × {kernel_size} Gaussian kernel, "
                   f"sigma = {sigma:.1f}, and reflected image boundaries (FFT zero-padding for linear convolution).")
        if maximum < AGREEMENT_THRESHOLD:
            st.write("The convolution theorem predicts that spatial convolution and frequency-domain multiplication give the same "
                     "result. Here the largest difference is far below one intensity level, so the outputs agree to within "
                     "floating-point rounding, consistent with that prediction.")
        else:
            st.warning("The convolution theorem predicts that the two methods agree to within floating-point rounding, but the "
                       f"largest difference here is {maximum:.3e} intensity levels, larger than rounding alone would explain. "
                       "These results do not support numerical equivalence for this image and settings.")

    st.button("Reset All", key="m3_reset_all", on_click=reset_all)


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
