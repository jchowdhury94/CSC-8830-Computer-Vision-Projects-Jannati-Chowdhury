"""Module 6 navigation and saved Part A experimental evidence."""

import dataclasses
import hashlib
import importlib
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from home_navigation import back_button, nav_row  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "Modules" / "Module_6" / "Part_A" / "output"


def show_asset(filename, caption, video=False):
    path = OUTPUT / filename
    if not path.is_file():
        st.warning(f"Unavailable asset: {filename}")
        return
    try:
        if video:
            st.video(str(path), format="video/mp4", width=400)
            st.caption(caption)
        else:
            st.image(str(path), caption=caption, width="stretch")
    except (OSError, ValueError) as error:
        st.warning(f"Unable to display {filename}: {error}")


def select_part(section):
    st.session_state["module6_section"] = section


def show_part_a():
    st.header("Part A — Optical Flow and Motion Tracking")
    st.write("Two videos with different motion patterns were analyzed using sparse Pyramidal Lucas–Kanade optical flow.")

    st.subheader("Video 1 — Horizontal Motion")
    show_asset("video_1_optical_flow_safari.mp4", "Sparse feature tracks — Video 1", video=True)
    st.table({"Measure": ["Initial features", "Features tracked to end", "Mean u (px/frame)", "Mean v (px/frame)"],
              "Result": ["17", "17", "+0.818", "+0.013"]})
    st.caption("Motion is predominantly horizontal: u is much larger than v.")

    st.subheader("Video 2 — Diagonal Motion")
    show_asset("video_2_optical_flow_safari.mp4", "Sparse feature tracks — Video 2", video=True)
    st.table({"Measure": ["Initial features", "Features tracked to end", "Mean u (px/frame)", "Mean v (px/frame)", "Mean magnitude (px/frame)"],
              "Result": ["20", "20", "+0.939", "+0.260", "0.991"]})
    st.caption("Positive u and v indicate down-right diagonal motion in image coordinates.")

    st.subheader("What Can Be Inferred from Optical Flow?")
    st.write("Feature tracks reveal motion direction, horizontal displacement u, vertical displacement v, and motion magnitude over time. Video 1 has dominant u and nearly zero v; Video 2 has positive u and v. Image x increases rightward and y increases downward.")

    st.subheader("Tracking Validation with Consecutive Frames")
    for number, feature, frames, old, new, u, v, magnitude, interpretation in (
        (1, 14, "484 → 485", "(508.8892, 1131.7012)", "(509.8865, 1131.6842)", "+0.9973", "−0.0170", "0.9974",
         "The feature moves almost entirely rightward, with negligible vertical displacement."),
        (2, 12, "482 → 483", "(401.2964, 1644.0986)", "(402.1024, 1644.6052)", "+0.8061", "+0.5066", "0.9520",
         "Both displacement components are positive, confirming rightward and downward motion."),
    ):
        st.markdown(f"**Video {number} · Feature {feature} · Frames {frames}**")
        show_asset(f"video_{number}_validation_comparison.png", f"Original consecutive frames — Video {number}")
        st.table({"Old coordinate": [old], "New coordinate": [new], "u (px)": [u], "v (px)": [v], "Magnitude (px)": [magnitude]})
        st.caption(interpretation)

    st.subheader("Lucas-Kanade Validation")
    st.write("Lucas–Kanade assumes approximately constant motion in a small neighborhood. The optical-flow constraint produces a least-squares system:")
    st.latex(r"I_xu + I_yv + I_t = 0, \qquad A\begin{bmatrix}u\\v\end{bmatrix}\approx b")
    st.caption("Both manual calculations use a 21×21 window. The manual method is a one-level linear estimate; OpenCV uses iterative pyramidal Lucas–Kanade.")
    st.markdown("**Video 1**")
    st.table({"Method": ["Manual", "OpenCV"], "u (px)": ["+1.3060", "+0.9973"],
              "v (px)": ["−0.0129", "−0.0170"], "Magnitude (px)": ["1.3061", "0.9974"]})
    st.caption("Position error: 0.3088 px. Both methods identify almost purely horizontal motion.")
    st.markdown("**Video 2**")
    st.table({"Method": ["Manual", "OpenCV"], "u (px)": ["+1.0801", "+0.8061"],
              "v (px)": ["+0.6620", "+0.5066"], "Magnitude (px)": ["1.2668", "0.9520"],
              "Angle (degrees)": ["31.5061", "32.1485"]})
    st.caption("Position error: 0.3150 px; angle difference: 0.6424°. Both methods identify down-right motion with close direction estimates. Manual magnitudes are higher in both videos.")

    st.subheader("Bilinear Interpolation")
    st.write("Optical-flow tracking can produce subpixel coordinates. Bilinear interpolation estimates intensity at a fractional coordinate using the four surrounding integer pixels.")
    show_asset("video_2_bilinear_interpolation.png", "Feature 12: real pixel samples and the subpixel location P")
    st.caption("P = (401.29635620, 1644.09863281); dx = 0.29635620; dy = 0.09863281")
    st.table({"Pixel": ["Q11", "Q21", "Q12", "Q22"],
              "Coordinate": ["(401, 1644)", "(402, 1644)", "(401, 1645)", "(402, 1645)"],
              "Intensity": [83, 90, 104, 109],
              "Weight": ["0.634241434767", "0.267125755233", "0.069402365233", "0.029230444767"]})
    st.caption("Weight sum: 1.000000000000")
    st.latex(r"I(x,y)=(1-d_x)(1-d_y)I_{11}+d_x(1-d_y)I_{21}+(1-d_x)d_yI_{12}+d_xd_yI_{22}")
    st.table({"Calculation": ["Manual intensity", "OpenCV verification", "Absolute difference"],
              "Result": ["87.087321520467", "87.087326049805", "0.000004529338"]})
    st.caption("The near-identical values validate the manual bilinear interpolation calculation.")


PART_B_OUTPUT = OUTPUT.parent.parent / "Part_B" / "output"


def show_part_b_image(filename, caption, width=620):
    path = PART_B_OUTPUT / filename
    if not path.is_file():
        st.warning(f"Unavailable Part B asset: {filename}")
        return
    try:
        st.image(str(path), caption=caption, width=width)
    except (OSError, ValueError) as error:
        st.warning(f"Unable to display {filename}: {error}")


def part_b_table(filename):
    import csv

    path = PART_B_OUTPUT / filename
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError) as error:
        st.warning(f"Unable to load Part B table {filename}: {error}")
        return []


def show_part_b():
    st.header("Part B — Structure from Motion")
    st.caption("Reconstructing the front cover of a book from four camera viewpoints")
    st.write("The book remained stationary while four photographs were taken from different camera viewpoints. Eight corresponding cover points were manually identified, reconstructed using rank-3 factorization based on the Tomasi–Kanade Structure-from-Motion method, and scaled using the measured 14.5 × 21.7 cm front cover. The results below retain the recovered geometric errors.")

    st.subheader("Four Camera Viewpoints")
    show_part_b_image("part_b_four_views.png", "View 1: near-overhead. Views 2–4: different oblique camera viewpoints.", width=700)
    st.caption("The book remained stationary; the camera viewpoint changed. Camera: Apple iPhone 17 Pro Max. Originals: 5712 × 4284 stored pixels. Upright working images: 4284 × 5712 pixels.")

    st.subheader("Corresponding Feature Points")
    st.write("The same eight physical features keep IDs P1–P8 in every view. P1–P4 are the physical front-cover corners; P5–P8 are distinctive printed landmarks. Coordinates refer to full-resolution upright images, with the origin at the upper-left, x rightward, and y downward.")
    show_part_b_image("part_b_correspondence_comparison.png", "Manually identified correspondences across Views 1–4", width=700)
    with st.expander("Inspect landmark close-ups"):
        show_part_b_image("part_b_landmark_closeups.png", "Each row follows one physical landmark across all four views", width=700)
    with st.expander("View correspondence coordinates"):
        st.dataframe(part_b_table("part_b_correspondences.csv"), hide_index=True, width="stretch")

    st.subheader("Measurement Matrix and Rank Analysis")
    st.write("With four views and eight points, W has shape 2F × P = 8 × 8. Columns are P1–P8; rows are V1_x, V1_y, V2_x, V2_y, V3_x, V3_y, V4_x, V4_y. Each view was centered by subtracting its mean x and mean y.")
    for column, label, value in zip(st.columns(3),
                                    ("Centered W rank", "First-three energy", "Rank-3 relative error"),
                                    ("7", "99.9182%", "2.8605%")):
        column.metric(label, value)
    st.caption("First-three squared energy: 99.918174594472%. Relative approximation error: 2.860514036456%.")
    show_part_b_image("part_b_singular_values.png", "All eight singular values; the dashed line marks the theoretical rank-3 cutoff", width=650)
    st.write("The observed matrix is not exactly rank 3, but the first three singular values capture approximately 99.92% of its squared energy. Rank 3 provides a strong approximation to the measurements; perspective effects and manual localization error prevent exact agreement. A good matrix fit alone does not establish accurate depth.")
    with st.expander("Measurement matrix details"):
        st.dataframe(part_b_table("part_b_measurement_matrix_centered.csv"), hide_index=True, width="stretch")

    st.subheader("Rank-3 Factorization")
    st.latex(r"W_3=U_3\Sigma_3V_3^T")
    st.latex(r"\widehat{M}=U_3\Sigma_3^{1/2},\qquad\widehat{S}=\Sigma_3^{1/2}V_3^T")
    st.write("The initial motion factor M̂ describes the camera rows, and the structure factor Ŝ describes the eight reconstructed points. An invertible transformation can change both factors while preserving their product, so a metric upgrade seeks approximately unit, orthogonal camera rows.")
    st.latex(r"W_3=M_{\mathrm{metric}}S_{\mathrm{metric}}")
    with st.expander("Metric upgrade diagnostics"):
        st.table({"Diagnostic": ["L positive definite", "L eigenvalues", "Metric constraint system", "Constraint-system rank", "Relative reconstruction error"],
                  "Result": ["Yes — no eigenvalue correction", "0.00041291793866116627; 0.00070862269507240296; 0.0015861019373099266", "12 × 6", "6", "2.860514036456%"]})
        camera_rows = part_b_table("part_b_camera_row_checks.csv")
        st.table([{"View": row["view"], "‖i‖": f"{float(row['norm_i']):.9f}",
                   "‖j‖": f"{float(row['norm_j']):.9f}", "i · j": f"{float(row['i_dot_j']):.9f}"} for row in camera_rows])
        st.caption("The least-squares rows are approximately orthonormal. They are not exact calibrated perspective camera poses.")

    st.subheader("Unscaled 3D Reconstruction")
    st.write("P1–P4 form the reconstructed cover boundary; P5–P8 are printed landmarks. These coordinates initially use arbitrary SfM units. Because all points lie on the same physical cover, an ideal reconstruction would be nearly planar.")
    left, right = st.columns(2)
    with left:
        show_part_b_image("part_b_reconstruction_unscaled.png", "Unscaled metric reconstruction", width=340)
    with right:
        show_part_b_image("part_b_reconstruction_plane_aligned.png", "Best-fit-plane view; visualization only", width=340)
    st.table({"Planarity measure": ["RMS fitted-plane distance", "Maximum absolute fitted-plane distance"],
              "Arbitrary units": ["216.946625", "393.504402"]})
    st.caption("Apparent depth primarily reflects reconstruction/model error rather than actual cover relief.")

    st.subheader("Scaling to Physical Dimensions")
    st.write("Measured front-cover width: 14.5 cm; height: 21.7 cm. One joint least-squares isotropic scale was applied to all three coordinates. This preserves the recovered geometry; X and Y were not stretched independently.")
    st.table({"Scale": ["Width-only", "Height-only", "Chosen joint isotropic"],
              "cm per arbitrary unit": ["0.006096547854923", "0.006102218321958", "0.006100465747245"]})

    st.subheader("Scaled 3D Reconstruction")
    show_part_b_image("part_b_reconstruction_scaled_cm.png", "Scaled SfM reconstruction coordinates; equal axis scaling, dimensions in cm", width=650)
    show_part_b_image("part_b_reconstruction_top_view_cm.png", "Top view: recovered boundary and an ideal rectangle for reference only", width=560)

    st.subheader("Physical Validation")
    validation = {row["measurement"]: row for row in part_b_table("part_b_physical_validation.csv")}
    if validation:
        st.table([{"Measurement": label, "Reconstructed": f"{float(validation[key]['reconstructed_cm']):.4f} cm",
                   "Reference": f"{float(validation[key]['known_or_reference_cm']):.4f} cm",
                   "Error": f"{float(validation[key]['percentage_error']):.4f}%"}
                  for key, label in (("mean_width", "Mean width"), ("mean_height", "Mean height"), ("mean_diagonal", "Mean diagonal"))])
        st.write("The known width and height were used to determine the global scale, so their close agreement is not an independent validation.")
        st.table({"Opposite edges": ["Width: top / bottom", "Height: left / right"],
                  "Edge lengths (cm)": [f"{float(validation['top_width']['reconstructed_cm']):.4f} / {float(validation['bottom_width']['reconstructed_cm']):.4f}",
                                        f"{float(validation['left_height']['reconstructed_cm']):.4f} / {float(validation['right_height']['reconstructed_cm']):.4f}"],
                  "Difference (cm)": [f"{float(validation['opposite_width_discrepancy']['reconstructed_cm']):.4f}",
                                      f"{float(validation['opposite_height_discrepancy']['reconstructed_cm']):.4f}"]})
    st.caption("Opposite-edge discrepancies reveal residual geometric distortion, which was retained in the reconstruction.")

    st.subheader("Recovered Depth Error")
    show_part_b_image("part_b_depth_error_cm.png", "Book-frame Z coordinates and actual distances to the fitted cover plane", width=650)
    for column, label, value in zip(st.columns(3),
                                    ("RMS plane displacement", "Maximum displacement", "RMS / book height"),
                                    ("1.3235 cm", "2.4006 cm", "6.099%")):
        column.metric(label, value)
    st.caption("The origin is the four-corner centroid. The fitted plane is at Z = 0.602388 cm, so signed plane distance is Z − 0.602388 cm. The reported errors use distance to that plane.")
    st.write("These landmarks lie on the same physical front-cover surface. The large recovered out-of-plane displacement should be interpreted primarily as reconstruction/model error, not real surface depth.")

    st.subheader("Interpretation and Limitations")
    st.write("Rank 3 explains approximately 99.92% of measurement energy. The scaled reconstruction captures overall cover proportions reasonably well, and the mean diagonal differs from the ideal by about 1.75%. Significant boundary distortion and artificial depth remain.")
    st.write("The perspective smartphone photographs are approximated by an affine/orthographic camera model. Correspondences were manually localized, and all landmarks lie on one approximately planar surface, making true depth recovery weak or degenerate. This is an educational SfM reconstruction, not a calibrated metric 3D scan.")


ACCENT = "#C9363E"
NAV_BUTTON_KEYS = ("m6_open_A", "m6_open_B", "m6_a_visitor", "m6_a_results", "m6_b_visitor", "m6_b_results")


def nav_selector(suffix):
    return ", ".join(f".st-key-{key}{suffix}" for key in NAV_BUTTON_KEYS)


HOME_TITLE = "Module 6"
HOME_SUBTITLE = "Optical Flow and Structure from Motion"
PART_A_SUBTITLE = "Optical Flow and Motion Tracking"
PART_B_SUBTITLE = "Structure from Motion Reconstruction"

HOME_CSS = f"""
<style>
.st-key-m6_home_layout {{ gap: 0; padding-top: 0; }}  /* the nav row above already supplies the top space */
.st-key-m6_home_title, .st-key-m6_home_subtitle {{ gap: 0; flex-shrink: 0; }}
.st-key-m6_home_nav {{ flex-shrink: 0; }}
.st-key-m6_home_title [data-testid="stMarkdownContainer"],
.st-key-m6_home_subtitle [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.st-key-m6_home_subtitle {{ padding-top: 44px; }}
.st-key-m6_home_nav {{ padding-top: 64px; }}
.st-key-m6_home_title h1 {{
    font-size: clamp(2.5rem, 13vw, 80px); font-weight: 800; color: inherit; line-height: 1.1; text-align: center;
    padding: 0; margin: 0; overflow-wrap: anywhere;
}}
.st-key-m6_home_subtitle h2 {{
    font-size: clamp(1.1rem, 4.6vw, 32px); font-weight: 600; color: inherit; opacity: 0.85; line-height: 1.2;
    text-align: center; padding: 0; margin: 0;
}}
@media (min-width: 801px) {{ .st-key-m6_home_subtitle h2 {{ white-space: nowrap; }} }}
.st-key-m6_home_nav {{ flex-direction: row; flex-wrap: wrap; gap: 24px; }}
{nav_selector("")} {{ flex: 1 1 280px; width: auto; min-width: 0; max-width: 100%; }}
{nav_selector(" button")} {{
    width: 100%; height: 110px; min-height: 110px; padding: 0.75rem 1.25rem; border-radius: 12px;
    background-color: {ACCENT}; border: 2px solid {ACCENT}; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center; text-align: center;
}}
{nav_selector(" button p")} {{
    font-size: 21px; font-weight: 700; color: #FFFFFF; line-height: 1.3; text-align: center; text-wrap: balance;
}}
{nav_selector(" button:hover")} {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
{nav_selector(" button:focus-visible")} {{
    outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT};
}}
{nav_selector(" button:active")} {{ background-color: #8E2229; border-color: #8E2229; }}
@media (max-width: 640px) {{
    .st-key-m6_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 16px; }}
    {nav_selector("")} {{ flex: 0 0 auto; width: 100%; }}
}}
</style>
"""


def svg_icon(path):
    return ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
            f"fill='white'%3E%3Cpath d='{path}'/%3E%3C/svg%3E\")")


ICON_VIDEO = svg_icon("M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z")
ICON_IMAGE = svg_icon("M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z")
ICON_CHART = svg_icon("M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z")
ICON_CHEVRON = svg_icon("M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z")
PANEL_KEYS = {"m6_a_visitor": ICON_VIDEO, "m6_a_results": ICON_CHART, "m6_b_visitor": ICON_IMAGE, "m6_b_results": ICON_CHART}

SELECT_CSS = """
<style>
.st-key-m6_sel_layout {
    max-width: 500px; width: 100%; margin: 0 auto; gap: 0; min-height: calc(100vh - 14rem);
    justify-content: center; align-items: stretch;
}
@media (min-width: 641px) {
    .st-key-m6_sel_layout { box-sizing: border-box; padding-bottom: 270px; }  /* centers the group ~135px above the middle */
}
.st-key-m6_sel_label { margin-bottom: 24px; }  /* spacing lives on the wrappers, not on the text elements */
.st-key-m6_sel_heading { margin-bottom: 44px; }
.st-key-m6_sel_heading [data-testid="stHeading"] > div { margin: 0; }  /* Streamlit's default -16px bottom margin */
.st-key-m6_sel_label [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-m6_sel_label p {
    margin: 0; text-align: center; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.18em; opacity: 0.55;
}
.st-key-m6_sel_heading h2 {
    font-size: clamp(1.5rem, 6.5vw, 2.25rem); font-weight: 800; line-height: 1.15; text-align: center;
    padding: 0; margin: 0; color: inherit; overflow-wrap: anywhere;
}
@media (min-width: 801px) {
    .st-key-m6_sel_heading, .st-key-m6_sel_heading [data-testid="stHeading"] { display: block; width: 100%; }
    .st-key-m6_sel_heading h2 {  /* one line, centered on the panels' axis even when wider than the 500px column */
        position: relative; left: 50%; transform: translateX(-50%); width: max-content; max-width: 94vw; white-space: nowrap;
    }
}
.st-key-m6_sel_panels { gap: 14px; }
__PANEL_SELECTOR__ { width: 100%; }
__PANEL_BUTTON__ {
    width: 100%; height: auto; min-height: 84px; padding: 14px 16px; border-radius: 14px;
    background-color: #262A33; border: 1px solid #3A3F4B; color: #F2F3F5;
    display: flex; align-items: center; justify-content: flex-start; gap: 16px; text-align: left;
    transition: border-color 0.15s, background-color 0.15s, transform 0.15s;
}
__PANEL_BUTTON__::before {
    content: ""; flex: 0 0 48px; height: 48px; border-radius: 12px; background-color: __ACCENT__;
    background-repeat: no-repeat; background-position: center; background-size: 26px;
}
__PANEL_BUTTON__::after {
    content: ""; flex: 0 0 28px; height: 28px; margin-left: auto; opacity: 0.7;
    background-image: __CHEVRON__; background-repeat: no-repeat; background-position: center; background-size: 28px;
}
__PANEL_BUTTON__ > div { flex: 1 1 auto; min-width: 0; justify-content: flex-start; }
__PANEL_BUTTON_SPAN__ { justify-content: flex-start; width: 100%; }
__PANEL_BUTTON__ p {
    margin: 0; text-align: left; font-size: 0.85rem; font-weight: 400; line-height: 1.45; color: #B4B9C4;
}
__PANEL_BUTTON__ p strong { display: block; font-size: 1.05rem; font-weight: 700; color: #FFFFFF; }
__PANEL_BUTTON_BR__ { display: none; }  /* the title is a block, so the label's line break would add a blank line */
__PANEL_BUTTON_HOVER__ { background-color: #2F343F; border-color: __ACCENT__; color: #F2F3F5; transform: translateY(-1px); }
__PANEL_BUTTON_FOCUS__ { outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px __ACCENT__; }
__PANEL_ICONS__
@media (max-width: 640px) {
    .st-key-m6_sel_layout { min-height: calc(100vh - 12rem); }
    __PANEL_BUTTON__ { padding: 12px; gap: 12px; min-height: 76px; }
    __PANEL_BUTTON__::before { flex-basis: 42px; height: 42px; background-size: 22px; }
    __PANEL_BUTTON__ p strong { font-size: 0.98rem; }
}
</style>
"""


def select_css():
    def selector(suffix):
        return ", ".join(f".st-key-{key} button{suffix}" for key in PANEL_KEYS)

    icons = "\n".join(f".st-key-{key} button::before {{ background-image: {icon}; }}" for key, icon in PANEL_KEYS.items())
    css = SELECT_CSS
    for token, value in (("__PANEL_SELECTOR__", ", ".join(f".st-key-{key}" for key in PANEL_KEYS)),
                         ("__PANEL_BUTTON_HOVER__", selector(":hover")),
                         ("__PANEL_BUTTON_FOCUS__", selector(":focus-visible")),
                         ("__PANEL_BUTTON__::before", selector("::before")),
                         ("__PANEL_BUTTON__::after", selector("::after")),
                         ("__PANEL_BUTTON_SPAN__", selector(" > div > span")),
                         ("__PANEL_BUTTON__ > div", selector(" > div")),
                         ("__PANEL_BUTTON__ p strong", selector(" p strong")),
                         ("__PANEL_BUTTON_BR__", selector(" p br")),
                         ("__PANEL_BUTTON__ p", selector(" p")),
                         ("__PANEL_BUTTON__", selector("")),
                         ("__PANEL_ICONS__", icons), ("__CHEVRON__", ICON_CHEVRON), ("__ACCENT__", ACCENT)):
        css = css.replace(token, value)
    return css


def panel_label(title, subtitle):
    return f"**{title}**  \n{subtitle}"


def selection_page(heading, panels):
    st.markdown(select_css(), unsafe_allow_html=True)
    with st.container(key="m6_sel_layout"):
        with st.container(key="m6_sel_label"):
            st.markdown("MODULE 6")
        with st.container(key="m6_sel_heading"):
            st.header(heading, anchor=False)
        with st.container(key="m6_sel_panels"):
            for key, title, subtitle, destination in panels:
                st.button(panel_label(title, subtitle), key=key, on_click=select_part, args=(destination,), width="stretch")


def part_a_select_page():
    selection_page(PART_A_SUBTITLE, (
        ("m6_a_visitor", "Track Motion in Your Own Video", "Use your own recording", "a_visitor"),
        ("m6_a_results", "View My Experiment & Results", "Explore the completed experiment", "a_results")))


def part_b_select_page():
    selection_page(PART_B_SUBTITLE, (
        ("m6_b_visitor", "Reconstruct an Object from Your Own Images", "Upload images to reconstruct an object", "b_visitor"),
        ("m6_b_results", "View My Experiment & Results", "Explore the completed reconstruction", "b_results")))


B_KEYS = ("m6b_files", "m6b_infos", "m6b_error", "m6b_match", "m6b_match_error", "m6b_matching")
B_RESULT_KEYS = ("m6b_result", "m6b_result_sig", "m6b_failed", "m6b_run_error")
B_THUMBS_PER_ROW = 4
B_QUALITY_LABELS = {"good": "Good", "caution": "Caution", "poor": "Poor"}


def b_reset_results():
    for key in B_RESULT_KEYS:
        st.session_state.pop(key, None)


def b_reset_after_matching():
    """New or cleared matches invalidate the scale choice and every reconstruction."""
    st.session_state["m6b_scale"] = None
    st.session_state["m6b_recon_open"] = False
    st.session_state["m6b_gen"] = st.session_state.get("m6b_gen", 0) + 1
    b_reset_results()


def b_discard_images():
    for key in B_KEYS:
        st.session_state.pop(key, None)
    st.session_state["m6b_uploader_seen"] = False
    st.session_state.pop("m6b_scroll", None)
    b_reset_after_matching()


def b_upload_key():
    return f"m6b_upload_{st.session_state['m6b_upload_generation']}"


def b_scroll_to(step):
    """Request one smooth scroll to a step; consumed by b_scroll_script on the next completed run."""
    st.session_state["m6b_scroll"] = step


def b_new_upload():
    """Uploader on_change: any change (add, remove, replace) rebuilds the image set and drops all results."""
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    uploaded = st.session_state.get(b_upload_key()) or []
    b_discard_images()
    if not uploaded:
        return
    st.session_state["m6b_uploader_seen"] = True
    if len(uploaded) > vr.MAX_VIEWS:
        st.session_state["m6b_error"] = (f"Please upload at most {vr.MAX_VIEWS} photographs; "
                                         f"you have {len(uploaded)}. Remove some with the x.")
        return
    if sum(f.size for f in uploaded) > vr.MAX_TOTAL_BYTES:
        st.session_state["m6b_error"] = f"Total upload exceeds {vr.MAX_TOTAL_BYTES // 1024 // 1024} MB."
        return
    files, infos = [], []
    for item in uploaded:
        data = item.getvalue()  # bytes only; the filename is never used as a path
        try:
            infos.append(vr.image_summary(data, item.name))
        except vr.ReconstructionError as error:
            st.session_state["m6b_error"] = f"{error} Remove it with the x to continue."
            return
        files.append((data, item.name))
    st.session_state["m6b_files"] = files
    st.session_state["m6b_infos"] = infos
    if len(files) >= vr.MIN_VIEWS:
        b_scroll_to(2)  # once, right after a valid set is accepted


def b_begin_matching():
    """Find Matching Points on_click: the next run renders the button disabled and then does the work."""
    st.session_state["m6b_matching"] = True


def b_start_over():
    st.session_state["m6b_upload_generation"] += 1  # a fresh uploader key is the supported way to empty it
    b_discard_images()
    b_scroll_to(1)


B_SCROLL_JS = """<script>
(function () {
  var selector = ".st-key-m6b_step_%d", tries = 0;
  function go() {
    var target = document.querySelector(selector);
    if (target) {
      var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      target.scrollIntoView({behavior: calm ? "auto" : "smooth", block: "start"});
    } else if (tries++ < 20) {
      setTimeout(go, 100);
    }
  }
  setTimeout(go, 150);
})(); /* transition %d */
</script>"""
B_SCROLL_HOST_CSS = """<style>
.st-key-m6b_scroll_host { height: 0; margin: 0; overflow: hidden; }
</style>"""


def b_scroll_script():
    """Emit the pending scroll request exactly once, at the end of the run so every step already exists."""
    step = st.session_state.pop("m6b_scroll", None)
    if step not in (1, 2, 3, 4, 5):
        return
    st.session_state["m6b_scroll_count"] = st.session_state.get("m6b_scroll_count", 0) + 1
    with st.container(key="m6b_scroll_host"):
        st.html(B_SCROLL_HOST_CSS)
        st.html(B_SCROLL_JS % (step, st.session_state["m6b_scroll_count"]), unsafe_allow_javascript=True)


def b_size_text(size_bytes):
    return f"{size_bytes / 1024 / 1024:.1f} MB" if size_bytes >= 1024 * 1024 else f"{max(1, round(size_bytes / 1024))} KB"


def b_upload_section(vr):
    with st.expander("Capture and upload instructions"):
        st.markdown(
            f"- **Photos:** upload {vr.MIN_VIEWS} to {vr.MAX_VIEWS} photographs of one object; "
            f"**{vr.RECOMMENDED_VIEWS} is recommended**. Formats: JPG, PNG, HEIC/HEIF. Up to "
            f"{vr.MAX_FILE_BYTES // 1024 // 1024} MB per photo, {vr.MAX_TOTAL_BYTES // 1024 // 1024} MB in total, "
            f"and {vr.MAX_PIXELS // 1_000_000} megapixels per photo.\n"
            "- **Object:** it must stay still. Choose one with visible texture (printed patterns, labels, grain) "
            "that is not flat, such as a box, mug or shoe. Posters, pages and other flat items give weak depth; "
            "shiny, transparent or plain surfaces match poorly.\n"
            "- **Order:** move the camera around the object in small steps, roughly 10–20° between photos, and "
            "upload the photos in the order you took them. Each photo is matched to the next one, so "
            "neighbouring photos must overlap strongly.\n"
            "- **Distance:** keep the same distance and zoom, with the object near the center. Stepping back and "
            "zooming in works better than a close-up, because the method assumes a distant, orthographic camera.\n"
            "- **Light:** keep the lighting steady and avoid moving people or objects in the background.\n"
            "- Your photos are processed in temporary storage and are not kept.")
    uploaded = st.file_uploader(
        "Upload your photographs", type=[e.lstrip(".") for e in vr.ALLOWED_EXTENSIONS], accept_multiple_files=True,
        key=b_upload_key(), on_change=b_new_upload, max_upload_size=vr.MAX_FILE_BYTES // 1024 // 1024)
    if uploaded:
        st.session_state["m6b_uploader_seen"] = True
    elif st.session_state.get("m6b_uploader_seen"):
        # The widget lost its files without on_change (e.g. the page was left and reopened): drop stale data.
        b_discard_images()
    if st.session_state.get("m6b_error"):
        st.error(st.session_state["m6b_error"])
    infos = st.session_state.get("m6b_infos")
    if not infos:
        if not st.session_state.get("m6b_error"):
            st.caption(f"Upload {vr.MIN_VIEWS} to {vr.MAX_VIEWS} photographs ({vr.RECOMMENDED_VIEWS} recommended) to begin.")
        return
    for start in range(0, len(infos), B_THUMBS_PER_ROW):
        columns = st.columns(B_THUMBS_PER_ROW)
        for column, number in zip(columns, range(start, min(start + B_THUMBS_PER_ROW, len(infos)))):
            info = infos[number]
            column.image(info["thumbnail"], width="stretch")
            column.caption(f"**{number + 1}. {info['name']}**  \n{info['width']} × {info['height']} px · "
                           f"{b_size_text(info['size_bytes'])}")
    total = sum(info["size_bytes"] for info in infos)
    if len(infos) < vr.MIN_VIEWS:
        st.caption(f"{len(infos)} photo{'s' if len(infos) != 1 else ''} added; add at least {vr.MIN_VIEWS}.")
    else:
        st.caption(f"{len(infos)} photographs, {b_size_text(total)} in total. Photos are used in the order listed.")


B_STALE_MATCH_MESSAGE = "Your earlier matching results are out of date. Please run Find Matching Points again."


B_MATCH_REQUIRED = ("tracks", "marker_candidates", "marker_base", "overlays", "pair_stats")


def b_match_is_current(vr, match):
    """False when a stored match lacks anything this page reads.

    The names this page needs are listed here, not taken from vr.MatchResult: a long-running server can hold
    a stale vr module whose class lacks the same fields, and then a class-based check would pass wrongly.
    """
    names = set(B_MATCH_REQUIRED) | {field.name for field in dataclasses.fields(vr.MatchResult)}
    return all(getattr(match, name, None) is not None for name in names)


def b_drop_stale_match(vr):
    """Invalidate an incompatible match and everything downstream; the uploaded images are kept.

    Returns True when a match was dropped. A server started before the matching code changed still has the old
    module in memory; it is reloaded once so that new matches carry every field the page needs.
    """
    if not hasattr(vr.MatchResult, "marker_base"):
        importlib.reload(vr)
    match = st.session_state.get("m6b_match")
    if match is None or b_match_is_current(vr, match):
        return False
    st.session_state.pop("m6b_match", None)
    b_reset_after_matching()
    st.session_state["m6b_match_error"] = B_STALE_MATCH_MESSAGE
    return True


B_MATCH_ACTIONS_CSS = """<style>
.st-key-m6b_match_actions button[kind="secondary"] { border: 2px solid #C9363E; height: 2.75rem; min-height: 2.75rem; }
.st-key-m6b_match_actions button[kind="secondary"]:hover,
.st-key-m6b_match_actions button[kind="secondary"]:focus-visible { border-color: #E04850; }
.st-key-m6b_match_actions button[kind="primary"] { height: 2.75rem; min-height: 2.75rem; }
</style>"""
B_MATCH_FAIL_HELP = ("Use photos that overlap strongly, were taken in order, show a textured and non-flat object, "
                     "and are sharp and evenly lit. Remove any photo that does not show the same object.")


def b_stage_matching(vr):
    st.header("Step 2 — Find Matching Points", anchor=False)
    st.write("The app looks for distinctive features (SIFT) and follows the same physical points through every photo. "
             "Nothing is matched until you click the button.")
    matching = bool(st.session_state.get("m6b_matching"))
    st.html(B_MATCH_ACTIONS_CSS)
    with st.container(key="m6b_match_actions"):
        find_column, start_column = st.columns(2)
        if st.session_state.get("m6b_match") is None:  # hidden once matching has succeeded
            find_column.button("Find Matching Points", key="m6b_find", type="primary", disabled=matching,
                               on_click=b_begin_matching, width="stretch")
        start_column.button("Start Over", key="m6b_start_over_match", on_click=b_start_over, width="stretch")
    if matching:
        st.session_state["m6b_matching"] = False  # never auto-retry after an unexpected error
        b_reset_after_matching()
        st.session_state.pop("m6b_match", None)
        st.session_state.pop("m6b_match_error", None)
        bar = st.progress(0.0, text="Starting…")
        try:
            match = vr.find_matches(st.session_state["m6b_files"],
                                    lambda stage, fraction: bar.progress(min(1.0, max(0.0, fraction)), text=stage))
        except vr.ReconstructionError as error:
            st.session_state["m6b_match_error"] = str(error)
        except (cv2.error, OSError, ValueError, MemoryError):
            logging.getLogger(__name__).exception("Module 6 visitor matching failed")
            st.session_state["m6b_match_error"] = "Matching could not complete. Please try different photos."
        else:
            st.session_state["m6b_match"] = match
            b_scroll_to(3)
        bar.empty()
        st.rerun()
    if st.session_state.get("m6b_match_error"):
        st.error(st.session_state["m6b_match_error"])
        st.caption(B_MATCH_FAIL_HELP)
    match = st.session_state.get("m6b_match")
    if match is None:
        return
    first = st.columns(3)
    first[0].metric("Photos", len(match.filenames))
    first[1].metric("Points in every photo", match.n_tracks_found)
    first[2].metric("Consistent points kept", match.n_tracks)
    if match.n_tracks < vr.FEW_POINTS:
        st.caption(f"Only {match.n_tracks} points were kept, so the shape will be less reliable. {B_MATCH_FAIL_HELP}")
    st.caption("Points kept passed three checks: a distinctive match (ratio test), agreement with a single camera "
               "geometry between neighbouring photos (RANSAC), and a consistency fit across all photos. "
               "Lines below join the same physical point in neighbouring photos "
               f"(a spread-out sample of up to {vr.OVERLAY_MAX_LINES}).")
    st.image(match.overlays[0], caption=f"Matches between photos 1 and 2", width="stretch")
    if len(match.overlays) > 1:
        with st.expander("Matches for the other photo pairs"):
            for number, overlay in enumerate(match.overlays[1:], 2):
                st.image(overlay, caption=f"Matches between photos {number} and {number + 1}", width="stretch")
    with st.expander("Match statistics per photo pair"):
        st.dataframe(pd.DataFrame([{"Photos": f"{s.view_a}–{s.view_b}", "Features (first)": s.keypoints_a,
                                    "Features (second)": s.keypoints_b, "Ratio-test matches": s.ratio_matches,
                                    "RANSAC inliers": s.inliers, "Median shift (px)": round(s.median_displacement_px, 1)}
                                   for s in match.pair_stats]), hide_index=True, width="stretch")


def b_scale_key(scale):
    return None if scale is None else tuple(sorted(scale.items()))


def b_commit_scale(match_scale):
    """Button callback: store the visitor's scale choice (None = arbitrary units) and open Step 4."""
    gen = st.session_state["m6b_gen"]
    if match_scale:
        scale = dict(a=int(st.session_state[f"m6b_pa_{gen}"]) - 1, b=int(st.session_state[f"m6b_pb_{gen}"]) - 1,
                     distance=float(st.session_state[f"m6b_dist_{gen}"]), unit=st.session_state[f"m6b_unit_{gen}"])
    else:
        scale = None
    st.session_state["m6b_scale"] = scale
    st.session_state["m6b_recon_open"] = True
    b_reset_results()
    b_scroll_to(4)


def b_scale_text(scale):
    if scale is None:
        return "No scale: coordinates will be in arbitrary reconstruction units."
    return (f"Points {scale['a'] + 1} and {scale['b'] + 1} are {scale['distance']:g} {scale['unit']} apart; "
            f"coordinates will be in {scale['unit']}.")


def b_stage_scale(vr, match):
    gen = st.session_state["m6b_gen"]
    st.header("Step 3 — Set Scale (Optional)", anchor=False)
    st.write("You can skip this step. Without a scale, the object's shape and proportions are estimated in "
             "arbitrary units rather than centimeters or millimeters. Reconstruction accuracy depends on the "
             "images and camera geometry.")
    with st.expander("How scaling works"):
        st.markdown(
            "- Photo 1 below shows numbered points. Pick two whose real-world distance you know, for example two "
            "printed marks on the object, and measure that distance yourself.\n"
            "- One scale factor is applied to all three axes, so the shape does not change. Scaling only changes "
            "the units; it does not make the reconstruction more accurate.\n"
            "- The result is approximate: it depends on your measurement and on the camera model, which assumes "
            "a distant, orthographic camera.")
    if b_drop_stale_match(vr):  # validated right before the markers are drawn; never touches marker_base first
        st.rerun()
    options = [i + 1 for i in match.marker_candidates]
    # Widget values from the previous interaction are already in session state when this run starts.
    shown_a = st.session_state.get(f"m6b_pa_{gen}", options[0])
    shown_b = st.session_state.get(f"m6b_pb_{gen}", options[-1])
    marked = vr.render_point_markers(None, match.tracks[0], match.marker_candidates,
                                     selected=(shown_a - 1, shown_b - 1), base=match.marker_base)
    st.image(marked, caption="Numbered points on photo 1 (Point A yellow, Point B green)", width="stretch")
    first, second = st.columns(2)
    point_a = first.selectbox("Point A", options, index=0, key=f"m6b_pa_{gen}")
    point_b = second.selectbox("Point B", options, index=len(options) - 1, key=f"m6b_pb_{gen}")
    third, fourth = st.columns(2)
    distance = third.number_input("Known distance between A and B", min_value=0.0, value=None, step=0.5,
                                  format="%g", placeholder="e.g. 5", key=f"m6b_dist_{gen}")
    fourth.selectbox("Unit", vr.UNIT_OPTIONS, index=1, key=f"m6b_unit_{gen}")
    valid = point_a != point_b and distance is not None and distance > 0
    if point_a == point_b:
        st.caption("Choose two different points.")
    st.write("Continue without scaling to use arbitrary units, or enter a known distance between two points to "
             "estimate real-world measurements.")
    skip, use = st.columns(2)
    skip.button("Continue Without Scaling", key="m6b_skip_scale", on_click=b_commit_scale, args=(False,), width="stretch")
    use.button("Apply Scale & Continue", key="m6b_use_scale", type="primary", on_click=b_commit_scale, args=(True,),
               disabled=not valid, width="stretch")


def b_stage_reconstruct(vr, match):
    st.header("Step 4 — Reconstruct Your Object", anchor=False)
    scale = st.session_state.get("m6b_scale")
    st.markdown(f"**Photos:** {len(match.filenames)}  \n**Points:** {match.n_tracks}  \n**Scale:** {b_scale_text(scale)}")
    st.caption("The reconstruction uses rank-3 factorization of the point tracks (Tomasi–Kanade) with an "
               "affine/orthographic camera model.")
    signature = (st.session_state["m6b_gen"], b_scale_key(scale))
    if st.session_state.get("m6b_result_sig") != signature:
        b_reset_results()
    _, run_column, _ = st.columns([1, 2, 1])  # centered, moderate width; stacks on narrow screens
    clicked = run_column.button("Run Reconstruction", key="m6b_run", type="primary", width="stretch")
    if clicked:
        b_reset_results()
        bar = st.progress(0.0, text="Starting…")
        try:
            result = vr.reconstruct_matches(match, scale,
                                            lambda stage, fraction: bar.progress(min(1.0, max(0.0, fraction)), text=stage))
        except vr.ReconstructionError as error:
            st.session_state["m6b_run_error"] = str(error)
            st.session_state["m6b_result_sig"] = signature  # keeps the message until the inputs change
        except (ValueError, np.linalg.LinAlgError, MemoryError):
            logging.getLogger(__name__).exception("Module 6 visitor reconstruction failed")
            st.session_state["m6b_run_error"] = "Reconstruction could not complete. Please try different photos."
            st.session_state["m6b_result_sig"] = signature
        else:
            if result.status == "ok":
                st.session_state["m6b_result"] = result
                st.session_state["m6b_result_sig"] = signature
                b_scroll_to(5)
            else:
                st.session_state["m6b_failed"] = result
                st.session_state["m6b_result_sig"] = signature
        bar.empty()
        st.rerun()
    if st.session_state.get("m6b_run_error"):
        st.error(st.session_state["m6b_run_error"])
    failed = st.session_state.get("m6b_failed")
    if failed is not None:
        st.error(failed.message)
        with st.expander("Diagnostics"):
            d = failed.diagnostics
            st.table({"Measure": ["Points used", "First-three energy", "Rank-3 relative error", "L eigenvalues"],
                      "Value": [str(d["n_points_used"]), f"{100 * d['first_three_energy']:.2f}%",
                                f"{100 * d['rank3_relative_error']:.2f}%",
                                ", ".join(f"{v:.3g}" for v in d["L_eigenvalues"])]})
        st.download_button("Diagnostic report", vr.diagnostic_report(failed), "reconstruction_report.txt",
                           "text/plain", key="m6b_dl_failed_report", on_click="ignore")


B_RESULTS_NAV_CSS = """<style>
.st-key-m6b_results_nav { margin-top: 28px; }
.st-key-m6b_results_nav button { border: 2px solid #C9363E; height: 2.75rem; min-height: 2.75rem; }
.st-key-m6b_results_nav button:hover, .st-key-m6b_results_nav button:focus-visible { border-color: #E04850; }
</style>"""


# One chart treatment for the 3D cloud and projections: a fixed light-neutral plot area (identical in Streamlit
# light and dark mode) so dark photo colors stay visible, dark-outlined markers, and moderate grid contrast.
B_PLOT_BG = "rgb(238,240,244)"
B_GRID = "rgb(176,181,190)"
B_AXIS_LINE = "rgb(90,96,106)"
B_TICK_FONT = dict(size=10, color="rgb(40,44,52)")
B_MARKER_OUTLINE = "rgba(25,28,34,0.85)"
B_CLOUD_MARKER_SIZE = 5
B_PROJECTION_MARKER_SIZE = 8
B_CAMERA = dict(eye=dict(x=1.5, y=-1.5, z=1.0), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))


def b_cloud_axis(title):
    return dict(title=dict(text=title, font=dict(size=12, color="rgb(40,44,52)")), backgroundcolor=B_PLOT_BG,
                gridcolor=B_GRID, gridwidth=1, showline=True, linecolor=B_AXIS_LINE, linewidth=2,
                zeroline=False, tickfont=B_TICK_FONT)


def b_cloud_figure(payload):
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter3d(
        x=payload["x"], y=payload["y"], z=payload["z"], mode="markers", text=payload["hover"], hoverinfo="text",
        marker=dict(size=B_CLOUD_MARKER_SIZE, color=payload["colors"], opacity=1.0,
                    line=dict(color=B_MARKER_OUTLINE, width=1))))
    figure.update_layout(
        height=520, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=B_PLOT_BG, plot_bgcolor=B_PLOT_BG, font=dict(color="rgb(40,44,52)"),
        scene=dict(bgcolor=B_PLOT_BG, xaxis=b_cloud_axis("X"), yaxis=b_cloud_axis("Y"), zaxis=b_cloud_axis("Z"),
                   aspectmode=payload["aspectmode"], camera=B_CAMERA))
    return figure


def b_projection_figure(payload, horizontal, vertical):
    import plotly.graph_objects as go

    titles = payload["axis_titles"]
    figure = go.Figure(go.Scatter(x=payload[horizontal], y=payload[vertical], mode="markers", text=payload["hover"],
                                  hoverinfo="text",
                                  marker=dict(size=B_PROJECTION_MARKER_SIZE, color=payload["colors"], opacity=1.0,
                                              line=dict(color=B_MARKER_OUTLINE, width=1))))
    axis = dict(gridcolor=B_GRID, linecolor=B_AXIS_LINE, showline=True, zeroline=False, tickfont=B_TICK_FONT)
    figure.update_xaxes(title=dict(text=titles[horizontal], font=dict(size=10)), **axis)
    figure.update_yaxes(title=dict(text=titles[vertical], font=dict(size=10)), scaleanchor="x", scaleratio=1, **axis)
    figure.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor=B_PLOT_BG)
    return figure


def b_singular_figure(values):
    import plotly.graph_objects as go

    figure = go.Figure(go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="lines+markers",
                                  hovertemplate="σ%{x} = %{y:.4g}<extra></extra>"))
    figure.add_vline(x=3.5, line_dash="dash", annotation_text="rank-3 cutoff", annotation_position="top right")
    figure.update_yaxes(type="log", title=dict(text="Singular value (log scale)", font=dict(size=10)))
    figure.update_xaxes(title=dict(text="Index", font=dict(size=10)), dtick=1)
    figure.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0))
    return figure


def b_results(vr, result):
    st.header("Step 5 — View Results", anchor=False)
    d = result.diagnostics
    payload = vr.plotly_payload(result)
    st.markdown(f"**A. Reconstruction Quality: {B_QUALITY_LABELS[result.quality]}**")
    first = st.columns(4)
    first[0].metric("Points reconstructed", payload["valid_point_count"])
    first[1].metric("Reprojection error (RMS)", f"{d['reprojection_rms_px']:.2f} px")
    first[2].metric("Rank-3 energy", f"{100 * d['first_three_energy']:.2f}%")
    first[3].metric("Planarity ratio", f"{d['planarity_ratio']:.2f}")
    st.caption(f"Coordinate units: {result.units}. "
               + ("The reconstruction was scaled with one distance you supplied; it is approximate."
                  if result.scale_info else "No scale was set, so these are not centimeters or millimeters.")
               + f" Reprojection error is {100 * d['reprojection_rms_relative']:.2f}% of the image diagonal. "
               "The quality label is a heuristic from reprojection error, rank-3 energy, point count and flatness.")
    if result.warnings:
        st.markdown("**Notes**\n" + "\n".join(f"- {vr.WARNING_MESSAGES.get(w, w)}" for w in result.warnings))
    st.markdown("**B. Interactive 3D Point Cloud**")
    # theme=None: Streamlit's default theme="streamlit" re-applies its own template (dark scene in dark mode).
    st.plotly_chart(b_cloud_figure(payload), width="stretch", key="m6b_plot_3d", theme=None,
                    config={"displaylogo": False})
    st.caption("Coordinate units: " + (f"{result.units}. Scale is approximate." if result.scale_info
                                       else "arbitrary reconstruction units."))
    st.caption("Drag to rotate and scroll to zoom. Axes use equal scale; point colors come from photo 1 "
               "(shown unchanged, on a light plot area with dark marker outlines for visibility). "
               "Mirror image and orientation are not determined by the photos.")
    st.markdown("**C. Projections**")
    left, right = st.columns(2)
    with left:
        st.caption("Top view (X–Y)")
        st.plotly_chart(b_projection_figure(payload, "x", "y"), width="stretch", key="m6b_plot_xy",
                        config={"displaylogo": False})
    with right:
        st.caption("Side view (X–Z)")
        st.plotly_chart(b_projection_figure(payload, "x", "z"), width="stretch", key="m6b_plot_xz",
                        config={"displaylogo": False})
    st.markdown("**D. Singular-Value Analysis**")
    st.plotly_chart(b_singular_figure(d["singular_values"]), width="stretch", key="m6b_plot_sv",
                    config={"displaylogo": False})
    st.caption("Ideal orthographic data of a rigid object has only three non-zero singular values. "
               f"The first three capture {100 * d['first_three_energy']:.2f}% of the measurement energy "
               f"(rank-3 relative error {100 * d['rank3_relative_error']:.2f}%).")
    with st.expander("Reconstruction diagnostics"):
        st.table({"Measure": ["Photos", "Points in every photo", "Points discarded as inconsistent", "Points used",
                              "Metric constraint rank", "L eigenvalues (all positive)", "Planarity ratio"],
                  "Value": [str(d["n_views"]), str(d["n_points_initial"]), str(d["n_points_trimmed"]),
                            str(d["n_points_used"]), str(d["metric_constraint_rank"]), ", ".join(f"{v:.3g}" for v in d["L_eigenvalues"]),
                            f"{d['planarity_ratio']:.3f}"]})
        st.table([{"Photo": v + 1, "Reprojection RMS (px)": f"{e:.2f}",
                   "|i|": f"{c[0]:.4f}", "|j|": f"{c[1]:.4f}", "i · j": f"{c[2]:.4f}"}
                  for v, (e, c) in enumerate(zip(d["reprojection_rms_per_view_px"], d["camera_rows"]))])
        st.caption(result.model + ".")
    with st.expander("Reconstructed 3D points"):
        unit = result.units
        st.dataframe(pd.DataFrame({"Point": range(1, len(result.points) + 1),
                                   f"X ({unit})": result.points[:, 0], f"Y ({unit})": result.points[:, 1],
                                   f"Z ({unit})": result.points[:, 2],
                                   "Reprojection error (px)": result.point_errors}),
                     hide_index=True, width="stretch")
    st.markdown("**E. Downloads**")
    csv_column, ply_column, report_column = st.columns(3)
    csv_column.download_button("Points CSV", vr.points_csv(result), "reconstruction_points.csv", "text/csv",
                               key="m6b_dl_csv", on_click="ignore", width="stretch")
    ply_column.download_button("PLY point cloud", vr.points_ply(result), "reconstruction.ply", "text/plain",
                               key="m6b_dl_ply", on_click="ignore", width="stretch")
    report_column.download_button("Diagnostic report", vr.diagnostic_report(result), "reconstruction_report.txt",
                                  "text/plain", key="m6b_dl_report", on_click="ignore", width="stretch")


def part_b_visitor_page():
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    st.session_state.setdefault("m6b_upload_generation", 0)
    st.session_state.setdefault("m6b_gen", 0)
    st.session_state.setdefault("m6b_scale", None)
    st.session_state.setdefault("m6b_recon_open", False)

    st.header("Reconstruct an Object from Your Own Images", anchor=False)
    # One continuous page: each step stays rendered once its prerequisites exist. The uploader runs first
    # because a removed or replaced photo resets everything that depends on it within this same run.
    with st.container(key="m6b_step_1"):
        st.header("Step 1 — Upload Your Images", anchor=False)
        b_upload_section(vr)
    infos = st.session_state.get("m6b_infos")
    ready = bool(infos) and len(infos) >= vr.MIN_VIEWS
    if ready:
        b_drop_stale_match(vr)
        with st.container(key="m6b_step_2"):
            b_stage_matching(vr)
        match = st.session_state.get("m6b_match")
        if match is not None:
            with st.container(key="m6b_step_3"):
                b_stage_scale(vr, match)
            if st.session_state["m6b_recon_open"]:
                with st.container(key="m6b_step_4"):
                    b_stage_reconstruct(vr, match)
                result = st.session_state.get("m6b_result")
                if result is not None:
                    with st.container(key="m6b_step_5"):
                        b_results(vr, result)
                        st.html(B_RESULTS_NAV_CSS)
                        with st.container(key="m6b_results_nav"):
                            back, start = st.columns(2)
                            back.button("Back to Scale", key="m6b_results_back_scale", on_click=b_scroll_to,
                                        args=(3,), width="stretch")
                            start.button("Start Over", key="m6b_start_over", on_click=b_start_over, width="stretch")
        back_button("m6_b_visitor_back_bottom", select_part, ("b_select",))
    b_scroll_script()


VISITOR_CLICK_KEY = "m6v_click"
VISITOR_PREVIEW_MAX = 700
VISITOR_KEYS = ("m6v_bytes", "m6v_filename", "m6v_digest", "m6v_analysis", "m6v_error", "m6v_run_error")


def visitor_reset_results():
    st.session_state["m6v_results"] = None
    st.session_state["m6v_results_sig"] = None
    st.session_state.pop("m6v_run_error", None)


def visitor_reset_selection():
    """Clear corners. The click component keeps one stable key (remounting it blanks the preview), so
    the click it last reported is marked consumed instead; it can then never be replayed as a new corner."""
    st.session_state["m6v_points"] = []
    st.session_state["m6v_settings_open"] = False
    last = st.session_state.get(VISITOR_CLICK_KEY)
    st.session_state["m6v_last_click"] = last.get("unix_time") if isinstance(last, dict) else None
    visitor_reset_results()


# Streamlit has no option to hide the uploader's "+" (it is a fixed "Add files" button beside the file chip),
# so it is hidden by CSS. Scoped to this uploader's st-key class; the file's remove (x) button is a different
# button and stays visible. Only one file is accepted (accept_multiple_files is False), so replacing a video
# works by removing it with the x and choosing another.
VISITOR_UPLOADER_CSS = """<style>
[class*="st-key-m6v_upload_"] button[aria-label="Add files"] { display: none !important; }
</style>"""


def visitor_upload_key():
    return f"m6v_upload_{st.session_state['m6v_upload_generation']}"


def visitor_discard_video():
    """Drop the video and everything derived from it (analysis, ROI, results, notices); back to Upload."""
    for key in VISITOR_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("m6v_params_saved", None)
    st.session_state["m6v_uploader_seen"] = False
    st.session_state.pop("m6v_scroll", None)
    visitor_reset_selection()


def visitor_new_upload():
    """Uploader on_change: a new file replaces the old video entirely; the native x removes it entirely."""
    from Modules.Module_6.Part_A import visitor_tracking as vt

    uploaded = st.session_state.get(visitor_upload_key())
    visitor_discard_video()
    if uploaded is None:
        return
    st.session_state["m6v_uploader_seen"] = True
    try:
        vt.check_upload(uploaded.size, uploaded.name)
    except vt.VideoError as error:
        st.session_state["m6v_error"] = str(error)
        return
    data = uploaded.getvalue()  # bytes only; the filename is never used as a path
    st.session_state["m6v_bytes"] = data
    st.session_state["m6v_filename"] = uploaded.name
    st.session_state["m6v_digest"] = hashlib.sha256(data).hexdigest()


def visitor_clear_video():
    """Start Over: a fresh uploader key is the supported way to empty the native uploader from code."""
    st.session_state["m6v_upload_generation"] += 1
    visitor_discard_video()


def visitor_scroll_to(step):
    """Request one smooth scroll to a step; consumed by visitor_scroll_script on the next completed run."""
    st.session_state["m6v_scroll"] = step


def visitor_reset_clicked():
    visitor_reset_selection()
    visitor_scroll_to(2)


def visitor_continue():
    st.session_state["m6v_settings_open"] = True
    visitor_scroll_to(3)


# Streamlit has no scroll API, so a one-shot script runs only on workflow transitions. It touches nothing but
# the element with the step's st-key class, and a changing nonce makes the frontend execute it again.
VISITOR_SCROLL_JS = """<script>
(function () {
  var selector = ".st-key-m6v_step_%d", tries = 0;
  function go() {
    var target = document.querySelector(selector);
    if (target) {
      var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      target.scrollIntoView({behavior: calm ? "auto" : "smooth", block: "start"});
    } else if (tries++ < 20) {
      setTimeout(go, 100);
    }
  }
  setTimeout(go, 150);
})(); /* transition %d */
</script>"""
VISITOR_SCROLL_HOST_CSS = """<style>
.st-key-m6v_scroll_host { height: 0; margin: 0; overflow: hidden; }
</style>"""


def visitor_scroll_script():
    """Emit the pending scroll request exactly once, at the end of the run so every step already exists."""
    step = st.session_state.pop("m6v_scroll", None)
    if step not in (1, 2, 3, 4):
        return
    st.session_state["m6v_scroll_count"] = st.session_state.get("m6v_scroll_count", 0) + 1
    with st.container(key="m6v_scroll_host"):
        st.html(VISITOR_SCROLL_HOST_CSS)
        st.html(VISITOR_SCROLL_JS % (step, st.session_state["m6v_scroll_count"]), unsafe_allow_javascript=True)


def visitor_preview_scale(plan):
    return min(1.0, VISITOR_PREVIEW_MAX / plan.width, VISITOR_PREVIEW_MAX / plan.height)


def visitor_preview_width(plan):
    """Displayed width of the Step 3 preview in pixels."""
    return max(1, round(plan.width * visitor_preview_scale(plan)))


def visitor_record_corner(component_key, plan):
    """Consume each real component click once; reruns are not clicks."""
    from Modules.Module_6.Part_A import visitor_tracking as vt

    click = st.session_state.get(component_key)
    if not isinstance(click, dict):
        return
    event_time = click.get("unix_time")
    points = st.session_state["m6v_points"]
    if event_time is None or event_time == st.session_state.get("m6v_last_click") or len(points) >= 2:
        return
    try:
        point = vt.map_click_to_frame(click["x"], click["y"], click["width"], click["height"],
                                      plan.width, plan.height)
    except (KeyError, ValueError, TypeError, OverflowError):
        return
    st.session_state["m6v_last_click"] = event_time
    st.session_state["m6v_points"] = [*points, point]
    visitor_reset_results()


def visitor_parameters(vt):
    """Advanced settings expander; defaults are the original experiment's values."""
    defaults = vt.TrackingParams()
    saved = st.session_state.get("m6v_params_saved") or {}

    def value(name):
        # Widget state is dropped when the settings stage is left, so the last values are restored from here.
        return saved.get(name, getattr(defaults, name))

    with st.expander("Advanced tracking settings", expanded=True):
        st.caption("Defaults match the original experiment. Features can disappear during tracking because of "
                   "blur, occlusion, or motion that is too large; lost features are not replaced.")
        left, right = st.columns(2)
        with left:
            max_corners = st.number_input("Shi-Tomasi maximum corners", 1, 200, value('max_corners'), 1, key="m6v_p_corners")
            quality = st.number_input("Quality level", 0.01, 0.95, value('quality_level'), 0.05, key="m6v_p_quality")
            min_distance = st.number_input("Minimum distance (px)", 1, 50, value('min_distance'), 1, key="m6v_p_distance")
            block = st.number_input("Block size (odd)", 3, 31, value('block_size'), 2, key="m6v_p_block")
        with right:
            window = st.number_input("Lucas-Kanade window (odd, square)", 5, 51, value('window'), 2, key="m6v_p_window")
            level = st.number_input("Pyramid maximum level", 0, 5, value('max_level'), 1, key="m6v_p_level")
            iterations = st.number_input("Termination iterations", 1, 100, value('max_iterations'), 1, key="m6v_p_iter")
            epsilon = st.number_input("Termination epsilon", 0.001, 1.0, value('epsilon'), 0.005, format="%.3f", key="m6v_p_eps")
    params = vt.TrackingParams(int(max_corners), float(quality), int(min_distance), int(block),
                               int(window), int(level), int(iterations), float(epsilon))
    st.session_state["m6v_params_saved"] = vars(params).copy()
    try:
        params.validate()
    except ValueError as error:
        st.warning(str(error))
        return None
    return params


def padded_domain(values, floor=None):
    """Axis domain covering every value with 10% padding; `floor` pins the lower bound (e.g. 0 for counts)."""
    values = [float(v) for v in values if pd.notna(v)]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    if floor is not None:
        low = min(low, floor)
    span = high - low
    pad = span * 0.1 if span > 0 else max(abs(high) * 0.1, 1.0)
    return [floor if floor is not None else low - pad, high + pad]


def tracking_charts(frame, unit):
    """Return [(caption, altair chart)] for the three tracking graphs; plotted values are the CSV/summary values."""
    import altair as alt

    chart_height = 340

    def axis(title, **kwargs):
        # Regular-weight titles and labels. The Y axis gets no Vega axis title (title=None): axis titles are merged
        # across layers, which collapsed the multi-line list into one tall line. The Y label is instead a
        # chart-level title on the left (see line_chart), which is never merged and wraps one list entry per line.
        return alt.Axis(title=title, titleFontWeight="normal", labelFontWeight="normal", titleFontSize=12,
                        labelFontSize=11, labelLimit=0, titlePadding=10, **kwargs)

    time_axis = axis("Time (seconds)", format=".2f")
    time_scale = alt.Scale(domain=padded_domain(frame["time_s"]), nice=False)

    def line_chart(data, series, y_title, y_domain, y_axis, value_format, count=False):
        long = data[["time_s", *series]].melt("time_s", var_name="key", value_name="value")
        long["series"] = long["key"].map(series)
        long["value"] = long["value"].astype(float)
        value_title = " ".join(y_title)  # tooltips take a single string; axis titles take the line list
        base = alt.Chart(long).encode(
            x=alt.X("time_s:Q", axis=time_axis, scale=time_scale),
            y=alt.Y("value:Q", axis=y_axis, scale=alt.Scale(domain=y_domain, nice=False, zero=False)),
            color=alt.Color("series:N", title=None, legend=alt.Legend(orient="bottom") if len(series) > 1 else None),
            tooltip=[alt.Tooltip("time_s:Q", title="Time (seconds)", format=".2f"),
                     alt.Tooltip("series:N", title="Series"),
                     alt.Tooltip("value:Q", title=value_title, format=value_format)])
        layers = [base.mark_line(point=alt.OverlayMarkDef(size=18))]
        if y_domain[0] < 0 < y_domain[1]:
            layers.insert(0, alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(opacity=0.4, strokeDash=[4, 3])
                          .encode(y="zero:Q"))
        y_label = alt.TitleParams(text=list(y_title), orient="left", anchor="middle", angle=270, offset=8,
                                  fontSize=12, fontWeight="normal", lineHeight=16)
        chart = alt.layer(*layers).properties(height=chart_height, title=y_label,
                                              padding={"left": 8, "right": 16, "top": 8, "bottom": 8}).interactive()
        # config as well as per-axis settings, so no theme default (Vega's own axis titles are bold) can win
        return chart.configure_axis(titleFontWeight="normal", labelFontWeight="normal").configure_title(
            fontWeight="normal")

    tracked_max = int(frame["tracked"].max())
    tracked_top = tracked_max + max(1, -(-tracked_max // 10))
    tracked = line_chart(
        frame, {"tracked": "Tracked features"}, ["Number of", "Tracked Features"], [0, tracked_top],
        axis(None, format="d", tickMinStep=1), "d", count=True)

    step_title = ["Mean Displacement", "per Step (px)"]
    step_cols = {"mean_u": "u (horizontal)", "mean_v": "v (vertical)"}
    step_values = pd.concat([frame[c] for c in step_cols])
    step = line_chart(frame, step_cols, step_title, padded_domain(step_values),
                      axis(None, format=".2f"), ".3f")

    cumulative_title = ["Mean Cumulative", "Displacement (px)"]
    cumulative_cols = {"mean_cumulative_u": "cumulative u (horizontal)", "mean_cumulative_v": "cumulative v (vertical)"}
    cumulative_values = pd.concat([frame[c] for c in cumulative_cols])
    cumulative = line_chart(frame, cumulative_cols, cumulative_title, padded_domain(cumulative_values),
                            axis(None, format=".1f"), ".2f")
    return [("Number of tracked features over time", tracked),
            ("Mean per-step horizontal (u) and vertical (v) displacement over time", step),
            ("Mean cumulative horizontal (u) and vertical (v) displacement from the starting positions", cumulative)]


# Spacing below the downloads and a 2 px red border on the two navigation buttons; scoped to this container.
VISITOR_RESULTS_NAV_CSS = """<style>
.st-key-m6v_results_nav { margin-top: 28px; }
.st-key-m6v_results_nav button { border: 2px solid #C9363E; height: 2.75rem; min-height: 2.75rem; }
.st-key-m6v_results_nav button:hover, .st-key-m6v_results_nav button:focus-visible { border-color: #E04850; }
</style>"""


def visitor_results(results, plan):
    st.header("Step 4 — Your Tracking Results", anchor=False)
    summary = results.summary
    st.markdown("**A. Annotated Video**")
    # Same displayed width as the Step 3 preview, centered; the encoded bytes are untouched.
    with st.container(horizontal_alignment="center"):
        st.video(results.video_bytes, format="video/mp4", width=visitor_preview_width(plan))
    st.markdown("**B. Tracking Summary**")
    unit = "px (original frame)"
    first = st.columns(3)
    first[0].metric("Initial features", summary["initial_features"])
    first[1].metric("Final features", summary["final_features"])
    first[2].metric("Feature retention", f"{summary['feature_retention_percent']:.1f}%")
    second = st.columns(4)
    second[0].metric("Mean u per step", f"{summary['mean_u_per_step']:+.3f}")
    second[1].metric("Mean v per step", f"{summary['mean_v_per_step']:+.3f}")
    second[2].metric("Mean magnitude per step", f"{summary['mean_magnitude_per_step']:.3f}")
    second[3].metric("Motion direction", summary["motion_direction"])
    if summary["all_features_lost_at_frame"]:
        st.warning(f"All features were lost at processed frame {summary['all_features_lost_at_frame']}; "
                   "the remaining frames show no tracks.")
    st.caption(f"Per-step values are in {unit}. One step is the motion between consecutive processed frames "
               f"({summary['step_definition']}). Positive u is rightward and positive v is downward.")
    if summary["mean_cumulative_u_final"] is not None:
        st.caption(f"Cumulative displacement is measured from each feature's starting position: the surviving "
                   f"features moved on average u = {summary['mean_cumulative_u_final']:+.1f}, "
                   f"v = {summary['mean_cumulative_v_final']:+.1f} {unit} by their last tracked frame.")
    if plan.stride > 1:
        st.info(f"Frames were skipped (every {plan.stride}), so each step spans {plan.step_seconds:.3f} s "
                "instead of one source frame interval.")
    frame = pd.DataFrame(results.per_frame)
    if not frame.empty:
        with st.expander("View Tracking Graphs", expanded=False):
            for caption, chart in tracking_charts(frame, unit):
                st.caption(caption)
                st.altair_chart(chart, width="stretch", theme=None)  # Vega-Lite defaults, not Streamlit's theme fonts
    st.caption("Image motion can reflect movement of the object, movement of the camera, or both.")
    st.markdown("**D. Downloads**")
    first, second, third = st.columns(3)
    first.download_button("Annotated MP4", results.video_bytes, "tracked_video.mp4", "video/mp4",
                          key="m6v_dl_video", on_click="ignore", width="stretch")
    second.download_button("Tracking CSV", results.csv_bytes, "tracking.csv", "text/csv",
                           key="m6v_dl_csv", on_click="ignore", width="stretch")
    third.download_button("Summary JSON", results.summary_json(), "tracking_summary.json", "application/json",
                          key="m6v_dl_json", on_click="ignore", width="stretch")


def visitor_roi(points, plan):
    """Return (roi, error_message) for the clicked corners; roi is None until two valid corners exist."""
    from Modules.Module_6.Part_A import visitor_tracking as vt

    if len(points) != 2:
        return None, None
    try:
        return vt.roi_from_corners(points[0], points[1], plan.width, plan.height), None
    except ValueError as error:
        return None, str(error)


def visitor_upload_section(vt):
    """Persistent top section: instructions, the native uploader, and analysis of a newly uploaded video."""
    with st.expander("Upload instructions"):
        st.markdown(
            f"- Formats: {', '.join(e.lstrip('.').upper() for e in vt.ALLOWED_EXTENSIONS)}; up to "
            f"{vt.MAX_FILE_BYTES // 1024 // 1024} MB and {vt.MAX_DURATION_S:.0f} seconds.\n"
            "- Use a steady camera and an object with visible texture or edges.\n"
            "- Some HEVC/H.265 files cannot be decoded here; re-export as H.264 MP4 if that happens.\n"
            "- Your video is processed in temporary storage and is not kept.")
    st.html(VISITOR_UPLOADER_CSS)
    # max_upload_size is per widget, so only this uploader shows 50 MB; other modules keep Streamlit's default.
    uploaded = st.file_uploader("Upload a video", type=[e.lstrip(".") for e in vt.ALLOWED_EXTENSIONS],
                                key=visitor_upload_key(), on_change=visitor_new_upload,
                                max_upload_size=vt.MAX_FILE_BYTES // 1024 // 1024)
    if uploaded is not None:
        st.session_state["m6v_uploader_seen"] = True
    elif st.session_state.get("m6v_uploader_seen"):
        # The widget lost its file without on_change (e.g. the page was left and reopened): drop stale data.
        visitor_discard_video()
    if st.session_state.get("m6v_error"):
        st.error(st.session_state["m6v_error"])
    data = st.session_state.get("m6v_bytes")
    if data is not None and st.session_state.get("m6v_analysis") is None:
        try:
            with st.spinner("Inspecting your video…"):
                st.session_state["m6v_analysis"] = vt.analyze_upload(data, st.session_state["m6v_filename"])
        except vt.VideoError as error:
            st.session_state["m6v_error"] = str(error)
            st.session_state.pop("m6v_bytes", None)
        else:
            visitor_scroll_to(2)  # once, right after a successful analysis
        st.rerun()


# st.metric clips long values with an ellipsis in narrow columns ("1080 × 1..."). Scoped to this keyed container:
# smaller responsive value font, no clipping, and columns that wrap to two rows on narrow screens.
VISITOR_INFO_CSS = """<style>
.st-key-m6v_video_info [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.5rem 1rem; }
.st-key-m6v_video_info [data-testid="stColumn"] { flex: 1 1 8rem; min-width: 8rem; width: auto; }
.st-key-m6v_video_info [data-testid="stMetricValue"] { font-size: clamp(1.1rem, 3.2vw, 1.5rem); }
.st-key-m6v_video_info [data-testid="stMetricValue"],
.st-key-m6v_video_info [data-testid="stMetricValue"] > div { overflow: visible; text-overflow: clip; white-space: nowrap; }
</style>"""


def visitor_video_info(analysis):
    """Metadata and optimization notices, directly under the uploader."""
    if analysis is None:
        if not st.session_state.get("m6v_error"):
            st.info("Upload a video to begin.")
        return
    info, plan = analysis.info, analysis.plan
    st.html(VISITOR_INFO_CSS)
    with st.container(key="m6v_video_info"):
        columns = st.columns(4)
        columns[0].metric("Resolution", f"{info.width} × {info.height}")
        columns[1].metric("Duration", f"{info.duration:.1f} s" if info.duration else "Unknown")
        columns[2].metric("Frame rate", f"{info.fps:.2f} fps" + (" (assumed)" if info.fps_assumed else ""))
        columns[3].metric("Frames", info.frame_count or "Unknown")
    for note in plan.messages():
        st.info(note)


# Reset Selection and Continue share one compact, left-aligned size (fixed width and height, not full-width).
VISITOR_SELECT_ACTIONS_CSS = """<style>
.st-key-m6v_select_actions { justify-content: flex-start; align-items: flex-start; flex-wrap: wrap; }
.st-key-m6v_select_actions [data-testid="stElementContainer"] { width: auto; flex: 0 0 auto; }
.st-key-m6v_select_actions button { width: 11rem; height: 2.75rem; min-height: 2.75rem; padding: 0 0.75rem; }
</style>"""


def visitor_stage_selection(vt, analysis):
    plan = analysis.plan
    st.header("Step 2 — Select the Object", anchor=False)
    st.write("Click two opposite corners around the object you want to track.")
    points = st.session_state["m6v_points"]
    frame_rgb = cv2.cvtColor(analysis.first_frame, cv2.COLOR_BGR2RGB)
    scale = visitor_preview_scale(plan)
    preview = cv2.resize(frame_rgb, (visitor_preview_width(plan), max(1, round(plan.height * scale))))
    roi, error = visitor_roi(points, plan)
    for x, y in points:
        cv2.circle(preview, (round(x * scale), round(y * scale)), 5, (255, 0, 255), -1)
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(preview, (round(rx * scale), round(ry * scale)),
                      (round((rx + rw - 1) * scale), round((ry + rh - 1) * scale)), (255, 0, 255), 2)
    # One component with a stable key for every state, so Reset Selection only changes its image.
    streamlit_image_coordinates(preview, width="content", cursor="crosshair" if len(points) < 2 else "default",
                                key=VISITOR_CLICK_KEY,
                                on_click=lambda: visitor_record_corner(VISITOR_CLICK_KEY, plan))
    if error:
        st.warning(f"{error} Use Reset Selection to select again.")
    elif roi is not None:
        st.caption(f"Selected rectangle (x, y, width, height): {roi} pixels at processing resolution")
    st.html(VISITOR_SELECT_ACTIONS_CSS)
    with st.container(key="m6v_select_actions", horizontal=True, gap="small"):
        st.button("Reset Selection", key="m6v_reset_selection", on_click=visitor_reset_clicked)
        st.button("Continue", key="m6v_continue", type="primary", disabled=roi is None,
                  on_click=visitor_continue)


def visitor_stage_settings(vt, analysis, roi):
    plan = analysis.plan
    st.header("Step 3 — Tracking Settings", anchor=False)
    st.markdown(f"**Processed at:** {plan.width} × {plan.height}, {plan.frames_to_process} frames  \n"
                f"**Selected rectangle (x, y, width, height):** {roi}")
    params = visitor_parameters(vt)
    available, detail = vt.encoder_status()
    if not available:
        st.error(f"Video encoding is unavailable on this server: {detail} Tracking cannot produce a result video.")
    signature = (st.session_state["m6v_digest"], roi, params)
    if st.session_state.get("m6v_results_sig") != signature:
        visitor_reset_results()
    ready = params is not None and available
    back, run_column = st.columns(2)
    back.button("Back to Selection", key="m6v_back_selection", on_click=visitor_scroll_to, args=(2,), width="stretch")
    clicked = run_column.button("Run Tracking", key="m6v_run", type="primary", width="stretch", disabled=not ready)
    if st.session_state.get("m6v_run_error"):
        st.error(st.session_state["m6v_run_error"])
    if clicked:
        visitor_reset_results()
        progress = st.progress(0.0, text="Starting…")
        try:
            result = vt.run_tracking_job(st.session_state["m6v_bytes"], st.session_state["m6v_filename"], plan, roi,
                                         params, lambda fraction, text: progress.progress(fraction, text=text))
        except vt.VideoError as error:
            st.session_state["m6v_run_error"] = str(error)
        except (cv2.error, OSError, ValueError, MemoryError):
            logging.getLogger(__name__).exception("Module 6 visitor tracking failed")
            st.session_state["m6v_run_error"] = "Tracking could not complete. Please try a different video or settings."
        else:
            st.session_state["m6v_results"] = result
            st.session_state["m6v_results_sig"] = signature
            visitor_scroll_to(4)
        progress.empty()
        st.rerun()


def visitor_start_over():
    visitor_clear_video()
    visitor_scroll_to(1)


def part_a_visitor_page():
    from Modules.Module_6.Part_A import visitor_tracking as vt

    st.session_state.setdefault("m6v_upload_generation", 0)
    st.session_state.setdefault("m6v_points", [])
    st.session_state.setdefault("m6v_results", None)
    st.session_state.setdefault("m6v_settings_open", False)

    st.header("Track Motion in Your Own Video", anchor=False)
    # One continuous page: each step stays rendered once its prerequisites exist. The uploader runs first
    # because a removed or replaced video resets everything that depends on it within this same run.
    with st.container(key="m6v_step_1"):
        st.header("Step 1 — Upload Your Video", anchor=False)
        visitor_upload_section(vt)
        analysis = st.session_state.get("m6v_analysis")
        visitor_video_info(analysis)
    if analysis is not None:
        roi, _ = visitor_roi(st.session_state["m6v_points"], analysis.plan)
        with st.container(key="m6v_step_2"):
            visitor_stage_selection(vt, analysis)
        if st.session_state["m6v_settings_open"] and roi is not None:
            with st.container(key="m6v_step_3"):
                visitor_stage_settings(vt, analysis, roi)
            if st.session_state.get("m6v_results") is not None:
                with st.container(key="m6v_step_4"):
                    visitor_results(st.session_state["m6v_results"], analysis.plan)
                    st.html(VISITOR_RESULTS_NAV_CSS)
                    with st.container(key="m6v_results_nav"):
                        back, start = st.columns(2)
                        back.button("Back to Settings", key="m6v_back_settings", on_click=visitor_scroll_to,
                                    args=(3,), width="stretch")
                        start.button("Start Over", key="m6v_start_over", on_click=visitor_start_over,
                                     width="stretch")
    if analysis is not None:  # no bottom Back (and no spacing) until a valid video has been analyzed
        back_button("m6_a_visitor_back_bottom", select_part, ("a_select",))
    visitor_scroll_script()


st.set_page_config(page_title="Module 6", layout="centered")
st.session_state.setdefault("module6_section", "home")
section = st.session_state["module6_section"]
# breadcrumb levels below Home for every view: (label, section); the last is the current page
_M6, _A, _B = ("Module 6", "home"), ("Optical Flow", "a_select"), ("Structure from Motion", "b_select")
CRUMBS = {"home": (_M6,), "a_select": (_M6, _A), "b_select": (_M6, _B),
          "a_visitor": (_M6, _A, ("Your Own Video", "a_visitor")), "a_results": (_M6, _A, ("My Experiment & Results", "a_results")),
          "b_visitor": (_M6, _B, ("Your Own Images", "b_visitor")), "b_results": (_M6, _B, ("My Experiment & Results", "b_results"))}
nav_row(CRUMBS[section], select_part, view=f"m6:{section}")
if section == "home":
    st.markdown(HOME_CSS, unsafe_allow_html=True)
    with st.container(key="m6_home_layout"):
        with st.container(key="m6_home_title"):
            st.title(HOME_TITLE, anchor=False)
        with st.container(key="m6_home_subtitle"):
            st.header(HOME_SUBTITLE, anchor=False)
        with st.container(key="m6_home_nav"):
            for destination, key, label in (("a_select", "A", "View Optical Flow and Motion Tracking"),
                                            ("b_select", "B", "View Structure from Motion Reconstruction")):
                st.button(label, key=f"m6_open_{key}", on_click=select_part, args=(destination,), width="stretch")
elif section == "a_select":
    part_a_select_page()
elif section == "a_visitor":
    part_a_visitor_page()
elif section == "a_results":
    show_part_a()
    back_button("m6_a_results_back_bottom", select_part, ("a_select",))
elif section == "b_select":
    part_b_select_page()
elif section == "b_visitor":
    part_b_visitor_page()
else:
    show_part_b()
    back_button("m6_b_results_back_bottom", select_part, ("b_select",))
