"""Module 6 page: navigation, the saved Part A and Part B experiments and the two visitor workflows.

Run from the repository root with `.venv/bin/python -m streamlit run home.py` and open Module 6.
Inputs: saved results in Modules/Module_6/Part_A/output and Part_B/output/revised, the Part B submission
photographs, and visitor uploads. Outputs: the web page and visitor downloads; saved results are not changed.
Requires the packages in the repository requirements.txt.
"""

import hashlib
import io
import logging
import math
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

from home_navigation import nav_row  # noqa: E402
from Modules.Module_6.Part_A.optical_flow import ARROW_MAGNIFICATION, ARROW_MAX_LENGTH  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "Modules" / "Module_6" / "Part_A" / "output"


def show_asset(filename, caption=None, video=False):
    path = OUTPUT / filename
    if not path.is_file():
        st.warning(f"Unavailable asset: {filename}")
        return
    try:
        if video:
            st.video(str(path), format="video/mp4", width="stretch")
            if caption:
                st.caption(caption)
        else:
            st.image(str(path), caption=caption, width="stretch")
    except (OSError, ValueError) as error:
        st.warning(f"Unable to display {filename}: {error}")


def select_part(section):
    st.session_state["module6_section"] = section


# Results-page layout, mirroring the finalized results pages of Modules 3 and 4 under Module 6 keys. Long equations scroll
# inside their own block on narrow screens. Light-theme text colors follow the other results pages.
PART_A_CSS = """<style>
.st-key-m6a_head { gap: 0; padding-bottom: 28px; }
.st-key-m6a_label { padding-bottom: 12px; }
.st-key-m6a_label [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-m6a_label p { margin: 0; text-align: center; font-size: 12px; font-weight: 600; letter-spacing: 0.18em; opacity: 0.55; }
.st-key-m6a_title { padding-bottom: 16px; }
.st-key-m6a_head h1 {
    font-size: 36px; font-weight: 700; line-height: 1.2; text-align: center; padding: 0; margin: 0;
    color: inherit; overflow-wrap: anywhere;
}
.st-key-m6a_head h2 {
    font-size: 20px; font-weight: 600; line-height: 1.3; text-align: center; padding: 0; margin: 0;
    color: inherit; opacity: 0.85; overflow-wrap: anywhere;
}
.st-key-m6a_body h2 { text-align: center; font-size: 26px; font-weight: 800; line-height: 1.2; padding: 1.5rem 0 0.25rem; overflow-wrap: anywhere; }
.st-key-m6a_body h3 { text-align: center; font-size: 22px; font-weight: 700; padding-top: 1.5rem; }
.st-key-m6a_body h5 { text-align: left; font-size: 18px; font-weight: 700; margin: 0; padding: 0.5rem 0 0.25rem; }
.st-key-m6a_body [data-testid="stMetric"] { text-align: left; }
.st-key-m6a_body [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 5vw, 1.6rem); }
.st-key-m6a_body [data-testid="stMetricValue"], .st-key-m6a_body [data-testid="stMetricValue"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; overflow-wrap: anywhere; }
.st-key-m6a_body [data-testid="stMetricLabel"] p, .st-key-m6a_body [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
.st-key-m6a_body [data-testid="stTable"] { overflow-x: auto; }
.st-key-m6a_body img { max-width: 100%; height: auto; }
.st-key-m6a_body .katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2rem 0; margin: 0.2rem 0; }
[class*="st-key-m6a_flow_"] { gap: 0.25rem; }
[class*="st-key-m6a_flow_"] p { text-align: center; line-height: 2.1; overflow-x: auto; overflow-y: hidden; }
.st-key-m6a_body [class*="st-key-m6a_label_"] p { text-align: center; font-weight: 700; margin: 0.5rem 0 0; }
.st-key-m6a_body [data-testid="stExpander"] h3 { text-align: left; padding-top: 0.75rem; }
[class*="st-key-m6a_sec_"] { border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent); margin-top: 1.5rem; padding-top: 0.5rem; }
.st-key-m6a_conclusion {
    background: color-mix(in srgb, currentColor 5%, transparent); border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
    border-radius: 16px; padding: 0.5rem 28px 28px; margin-top: 2rem;
}
@media (max-width: 640px) { .st-key-m6a_conclusion { padding: 0.25rem 14px 18px; } }
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) :is(.st-key-m6a_label p, .st-key-m6a_head h2) { color: #31333F; opacity: 1; }
}
</style>"""

# Recorded values of my two validation feature pairs. A^T A, the manual (u, v) and the OpenCV values are the ones saved in
# Part_A/output (validation reports and tracking CSVs). A^T b was not saved; it was recovered from the saved validation frames
# with the method of the manual_lucas_kanade_validation scripts, and it reproduces the saved (u, v).
VALIDATIONS = (
    dict(video=1, title="Horizontal Motion", feature=14, frames=(484, 485), center=(509, 1132),
         old=(508.88922119, 1131.70117188),
         ata=((0.150347642734, -0.037866986255), (-0.037866986255, 0.534424680411)),
         atb=(0.196850249904, -0.056356209150),
         manual=(1.306048680516, -0.012911233275),
         opencv=(0.99725342, -0.01696777), opencv_new=(509.88647461, 1131.68420410)),
    dict(video=2, title="Diagonal Motion", feature=12, frames=(482, 483), center=(401, 1644),
         old=(401.29635620, 1644.09863281),
         ata=((0.813748257882, -0.308142961842), (-0.308142961842, 1.445424055652)),
         atb=(0.674911572472, 0.624096501346),
         manual=(1.080078063292, 0.662030599980),
         opencv=(0.80606079, 0.50659180), opencv_new=(402.10241699, 1644.60522461)),
)

# Saved bilinear interpolation example (video_2_bilinear_interpolation.txt): Video 2, Feature 12, old frame 482.
BILINEAR_POINT = (401.29635620, 1644.09863281)
BILINEAR_PIXELS = (("Q11", (401, 1644), 83), ("Q21", (402, 1644), 90), ("Q12", (401, 1645), 104), ("Q22", (402, 1645), 109))
BILINEAR_OPENCV = 87.087326049805


def tracking_comparison(data):
    """Derived quantities of one validation pair, all from the full-precision recorded values."""
    old, manual, opencv = (np.array(data[key], dtype=float) for key in ("old", "manual", "opencv"))
    new = old + manual
    angles = [float(np.degrees(np.arctan2(v, u))) for u, v in (manual, opencv)]
    return {"new": new, "magnitude": (float(np.hypot(*manual)), float(np.hypot(*opencv))), "angles": angles,
            "position_difference": float(np.hypot(*(new - np.array(data["opencv_new"])))),
            "angle_difference": abs((angles[0] - angles[1] + 180.0) % 360.0 - 180.0)}


BILINEAR_EQUATION = r"""\boxed{\begin{aligned}
I(x, y) ={} & (1-d_x)(1-d_y)\,I_{11} \\
 &+ d_x(1-d_y)\,I_{21} \\
 &+ (1-d_x)\,d_y\,I_{12} \\
 &+ d_x d_y\,I_{22}
\end{aligned}}"""


def bilinear_example():
    x, y = BILINEAR_POINT
    dx, dy = x - int(x), y - int(y)  # both coordinates are positive, so int() is floor()
    weights = ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy)
    intensity = sum(weight * pixel[2] for weight, pixel in zip(weights, BILINEAR_PIXELS))
    return {"dx": dx, "dy": dy, "weights": weights, "intensity": intensity,
            "difference": abs(intensity - BILINEAR_OPENCV)}


@st.cache_data(show_spinner=False)
def tracking_summary(path, modified):
    """Statistics of one saved tracking CSV (all features, all consecutive-frame transitions)."""
    frame = pd.read_csv(path)
    return {"initial": int((frame["frame"] == 2).sum()), "final": int((frame["frame"] == frame["frame"].max()).sum()),
            "u": float(frame["u"].mean()), "v": float(frame["v"].mean()), "magnitude": float(frame["magnitude"].mean())}


def metric_card(label, value, note=None, help_text=None):
    with st.container(border=True):
        st.metric(label, value, help=help_text)
        if note:
            st.caption(note)


def signed(value, places=4):
    return f"{value:+.{places}f}".replace("-", "−")


def simple_table(index_name, columns):
    st.table(pd.DataFrame(columns).set_index(index_name))


def equation_label(key, text):
    with st.container(key=f"m6a_label_{key}"):
        st.markdown(text)


def boxed(equation):
    return equation if equation.startswith(r"\boxed") else rf"\boxed{{\,{equation}\,}}"


def show_equation(equation):
    st.latex(boxed(equation))


def wrapping_equations(key, equations):
    """Centered display-style equations, one boxed per line, that scroll sideways on a narrow screen."""
    with st.container(key=f"m6a_flow_{key}"):
        for equation in equations:
            st.markdown(rf"$\displaystyle {boxed(equation)}$")


# Technical notes kept for the lab report (not shown on the results page).
# Setup: frame resolution 1080 × 1920 pixels; Shi–Tomasi corner detection; pyramidal Lucas–Kanade; 21 × 21 pixel tracking
#   window; maximum pyramid level 3. Lost features are removed without replacement.
# Inference: direction is determined by the signs and relative values of u and v; displacement is the movement between
#   consecutive frames; a trajectory is the feature positions tracked across frames; relative motion is image-space
#   movement, not physical distance or speed. The Video 1 averages confirm predominantly rightward motion.
# Validation methodology: OpenCV is the reference implementation for this comparison, not independently measured ground
#   truth, so e is a position difference relative to OpenCV. Gradients were computed on grayscale intensities scaled to
#   [0, 1]: I_x and I_y with a 3 × 3 Sobel operator (scale 1/8) on the average of the two frames, and I_t as the new frame
#   minus the old. The motion angle is θ = atan2(v, u) in degrees, measured from the rightward axis toward downward.
#   Calculations use full precision; displayed values are rounded. The manual calculation is a single-level, one-step
#   linear estimate: the normal equations are solved once over a 21 × 21 pixel window (n = 441) centered at P_1 rounded
#   to the nearest pixel. OpenCV uses iterative pyramidal Lucas–Kanade, which refines the estimate repeatedly over several
#   image scales, so the two methods are not algorithmically identical.
def part_a_setup():
    st.header("1. Experimental Setup", anchor=False)
    st.write("**Objective:** Analyze horizontal and diagonal motion using Lucas–Kanade optical flow and experimentally "
             "validate motion tracking and bilinear interpolation equations.")
    for column, card in zip(st.columns(2), (("Video 1 — Horizontal Motion", "≈ 31 s", "soda can moving horizontally"),
                                            ("Video 2 — Diagonal Motion", "≈ 32 s", "soda can moving diagonally"))):
        with column:
            metric_card(*card)
    st.write("Shi–Tomasi corner detection identifies features within the selected rectangle around the soda can. "
             "Pyramidal Lucas–Kanade optical flow tracks these features across consecutive grayscale frames using a "
             "21 × 21 pixel window.")


def part_a_flow():
    st.header("2. Optical Flow Visualization and Motion Analysis", anchor=False)
    st.subheader("Optical Flow Visualization", anchor=False)
    for column, (number, title) in zip(st.columns(2), ((1, "Horizontal Motion"), (2, "Diagonal Motion"))):
        with column:
            st.markdown(f"##### Video {number} — {title}")
            show_asset(f"video_{number}_optical_flow_arrows.mp4", video=True)
    st.caption("Yellow dots identify tracked features. Each arrow starts at a feature's current position and points in its "
               "direction of motion since the previous frame. Arrow length shows that displacement magnified "
               f"{ARROW_MAGNIFICATION}× (maximum {ARROW_MAX_LENGTH:.0f} px) in both videos; the reported measurements are not "
               "magnified.")

    st.subheader("Experimental Results", anchor=False)
    try:
        summaries = []
        for number in (1, 2):
            path = OUTPUT / f"video_{number}_tracking.csv"
            summaries.append(tracking_summary(str(path), path.stat().st_mtime))
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"Unable to load the saved tracking data: {error}")
    else:
        simple_table("Measurement", {
            "Measurement": ["Initial features", "Features tracked to end", "Mean horizontal displacement, u (px/frame)",
                            "Mean vertical displacement, v (px/frame)", "Mean motion magnitude (px/frame)"],
            **{f"Video {number}": [str(s["initial"]), str(s["final"]), signed(s["u"], 3), signed(s["v"], 3),
                                   f"{s['magnitude']:.3f}"] for number, s in enumerate(summaries, 1)}})
        st.caption("Displacements and motion magnitudes are averaged across tracked features and consecutive frame pairs. "
                   "Mean motion magnitude averages the individual magnitudes; it is not the magnitude of the mean "
                   "displacement (u, v).")

    st.subheader("Information Inferred from Optical Flow", anchor=False)
    show_equation(r"\text{motion magnitude} = \sqrt{u^{2} + v^{2}}")
    st.markdown(
        "**Video 1 — Horizontal Motion:** The horizontal displacement (0.818 px/frame) dominates the vertical "
        "displacement (0.013 px/frame), indicating rightward motion.\n\n"
        "**Video 2 — Diagonal Motion:** Positive horizontal (0.939 px/frame) and vertical (0.260 px/frame) displacements "
        "indicate downward-right motion.\n\n"
        "Optical flow reveals motion direction, displacement, magnitude and feature trajectories in image coordinates.")


def part_a_lucas_kanade_theory():
    st.header("3. Theory: Motion Tracking Equations", anchor=False)
    equation_label("constraint", "Optical Flow Constraint Equation")
    show_equation(r"I_x u + I_y v + I_t = 0")
    st.markdown(r"$I_x$ and $I_y$ are the spatial intensity gradients in the $x$ and $y$ directions, $I_t$ is the "
                r"temporal intensity gradient, and $u$ and $v$ are the horizontal and vertical displacements between "
                r"consecutive frames.")
    equation_label("displacement", "Lucas–Kanade Displacement Equation")
    show_equation(r"\boxed{\,\mathbf{d} = (A^{T}A)^{-1}A^{T}\mathbf{b}\,}")
    st.markdown(r"$\mathbf{d} = (u, v)^{T}$ is the displacement vector, $A$ is the spatial-gradient matrix, with one row "
                r"$(I_x, I_y)$ for each pixel in the local window, and $\mathbf{b}$ is the vector of negative temporal "
                r"gradients $-I_t$ at the same pixels.")
    equation_label("tracking", "Motion Tracking Equations")
    show_equation(r"\boxed{\,x_2 = x_1 + u, \qquad y_2 = y_1 + v\,}")
    st.markdown(r"$(x_1, y_1)$ are the original feature coordinates, $(x_2, y_2)$ are the new coordinates, and $u$ and "
                r"$v$ are the horizontal and vertical displacements.")
    with st.expander("View Mathematical Derivation", expanded=False):
        lucas_kanade_derivation()


def lucas_kanade_derivation():
    st.subheader("3.1 Brightness Constancy Assumption", anchor=False)
    st.write("A moving image feature is assumed to keep approximately the same intensity between consecutive frames:")
    show_equation(r"I(x, y, t) = I(x+u,\; y+v,\; t+1)")
    st.write("Here $I(x, y, t)$ is the image intensity at position $(x, y)$ in frame $t$, $u$ is the horizontal displacement "
             "and $v$ is the vertical displacement. The time interval is one frame.")

    st.subheader("3.2 Optical Flow Constraint Equation", anchor=False)
    st.write("A first-order Taylor expansion of the right-hand side about $(x, y, t)$, with steps $u$, $v$ and one frame, gives")
    show_equation(r"""\begin{aligned}
I(x+u,\; y+v,\; t+1) \approx{} & I(x, y, t) \\
 &+ I_x u + I_y v + I_t
\end{aligned}""")
    st.write("where $I_x = \\partial I / \\partial x$ and $I_y = \\partial I / \\partial y$ are the spatial intensity gradients "
             "and $I_t = \\partial I / \\partial t$ is the temporal intensity gradient, with time measured in frames. "
             "Substituting into the brightness constancy equation:")
    show_equation(r"I(x, y, t) \approx I(x, y, t) + I_x u + I_y v + I_t")
    st.write("Cancelling the common term $I(x, y, t)$ leaves the optical flow constraint equation:")
    show_equation(r"\boxed{\,I_x u + I_y v + I_t = 0\,}")
    st.write("This is a first-order approximation, appropriate for sufficiently small motion.")

    st.subheader("3.3 Lucas–Kanade Motion Estimation", anchor=False)
    st.write("The constraint has two unknowns, $u$ and $v$, but gives only one equation per pixel. Lucas–Kanade assumes the "
             "motion is approximately constant in a small neighborhood around a feature, so its $n$ pixels, indexed "
             "$i = 1, \\dots, n$, share the same $(u, v)$:")
    show_equation(r"""\begin{aligned}
I_{x1}u + I_{y1}v &= -I_{t1} \\
I_{x2}u + I_{y2}v &= -I_{t2} \\
&\;\;\vdots \\
I_{xn}u + I_{yn}v &= -I_{tn}
\end{aligned}""")
    st.write("Here $I_{xi}$, $I_{yi}$ and $I_{ti}$ are the gradients at pixel $i$. In matrix form, $A\\mathbf{d} \\approx "
             "\\mathbf{b}$, with")
    show_equation(r"""\begin{gathered} A = \begin{bmatrix} I_{x1} & I_{y1} \\ I_{x2} & I_{y2} \\ \vdots & \vdots \\ I_{xn} & I_{yn} \end{bmatrix},
\\[1ex] \mathbf{d} = \begin{bmatrix} u \\ v \end{bmatrix},
\qquad \mathbf{b} = \begin{bmatrix} -I_{t1} \\ -I_{t2} \\ \vdots \\ -I_{tn} \end{bmatrix} \end{gathered}""")
    st.write("$A$ is the $n \\times 2$ matrix of spatial gradients, $\\mathbf{d}$ is the displacement vector and $\\mathbf{b}$ "
             "is the negative temporal gradient vector. With more equations than unknowns, there is generally no exact "
             "solution, so we minimize the sum of squared residuals:")
    show_equation(r"""\begin{aligned}
E(\mathbf{d}) &= \lVert A\mathbf{d} - \mathbf{b} \rVert^{2}
 = (A\mathbf{d} - \mathbf{b})^{T}(A\mathbf{d} - \mathbf{b}) \\
&= \mathbf{d}^{T}A^{T}A\,\mathbf{d} - 2\,\mathbf{d}^{T}A^{T}\mathbf{b} + \mathbf{b}^{T}\mathbf{b}
\end{aligned}""")
    st.write("Here $E$ is the least-squares error and $T$ denotes the transpose. Differentiating with respect to $\\mathbf{d}$:")
    show_equation(r"""\begin{aligned}
\nabla E(\mathbf{d}) &= 2A^{T}A\,\mathbf{d} - 2A^{T}\mathbf{b} \\
&= 2A^{T}(A\mathbf{d} - \mathbf{b})
\end{aligned}""")
    st.write("Setting the gradient to zero at the minimum gives the normal equations:")
    show_equation(r"""\begin{aligned}
2A^{T}(A\mathbf{d} - \mathbf{b}) &= \mathbf{0} \\
A^{T}A\,\mathbf{d} &= A^{T}\mathbf{b}
\end{aligned}""")
    st.write("Expanded, with each sum $\\Sigma$ taken over the pixels of the local neighborhood:")
    show_equation(r"""\underbrace{\begin{bmatrix} \sum I_x^{2} & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^{2} \end{bmatrix}}_{A^{T}A}
\begin{bmatrix} u \\ v \end{bmatrix} =
\underbrace{\begin{bmatrix} -\sum I_x I_t \\ -\sum I_y I_t \end{bmatrix}}_{A^{T}\mathbf{b}}""")
    st.write("Multiplying both sides by $(A^{T}A)^{-1}$ gives the Lucas–Kanade displacement:")
    show_equation(r"\boxed{\,\mathbf{d} = (A^{T}A)^{-1}A^{T}\mathbf{b}\,}")
    st.write("This requires $A^{T}A$ to be invertible, and sufficiently well-conditioned for reliable tracking: the "
             "neighborhood needs intensity variation in two independent directions, as at a corner.")

    st.subheader("3.4 Motion Tracking Between Two Consecutive Frames", anchor=False)
    st.write("Given a feature at $P_1 = (x_1, y_1)$ in the first frame, the tracking problem is to find its position "
             "$P_2 = (x_2, y_2)$ in the second frame. With the estimated displacement $\\mathbf{d} = (u, v)$:")
    show_equation(r"P_2 = P_1 + \mathbf{d}")
    show_equation(r"\boxed{\,x_2 = x_1 + u, \qquad y_2 = y_1 + v\,}")

def part_a_bilinear():
    example = bilinear_example()
    st.header("4. Theory: Bilinear Interpolation", anchor=False)
    equation_label("bilinear", "Bilinear Interpolation Equation")
    show_equation(BILINEAR_EQUATION)
    st.markdown(r"$I(x, y)$ is the interpolated intensity at the subpixel coordinate $(x, y)$, and "
                r"$d_x = x - \lfloor x \rfloor$ and $d_y = y - \lfloor y \rfloor$ are the horizontal and vertical "
                r"fractional offsets. $I_{11}$, $I_{21}$, $I_{12}$ and $I_{22}$ are the intensities of the top-left, "
                r"top-right, bottom-left and bottom-right surrounding pixels (first index: column; second index: row).")
    with st.expander("4.1 Mathematical Derivation", expanded=False):
        bilinear_derivation()
    bilinear_experiment(example)


def bilinear_derivation():
    st.write("For a fractional location $P = (x, y)$, the fractional offsets from the top-left pixel are")
    show_equation(r"d_x = x - \lfloor x \rfloor, \qquad d_y = y - \lfloor y \rfloor")
    st.write("where $\\lfloor\\cdot\\rfloor$ rounds down to an integer and $0 \\le d_x, d_y < 1$. Let $I_{11}$, $I_{21}$, "
             "$I_{12}$ and $I_{22}$ be the intensities of the top-left, top-right, bottom-left and bottom-right surrounding "
             "pixels (the first index is the column, the second the row). Linear interpolation along the upper pair, one "
             "pixel apart, starts at $I_{11}$ and adds the fraction $d_x$ of the change:")
    show_equation(r"""\begin{aligned}
I_{\mathrm{top}} &= I_{11} + d_x(I_{21} - I_{11}) \\
&= (1-d_x)I_{11} + d_x I_{21}
\end{aligned}""")
    st.write("Along the lower pair:")
    show_equation(r"I_{\mathrm{bottom}} = (1-d_x)I_{12} + d_x I_{22}")
    st.write("Interpolating vertically between these two results:")
    show_equation(r"I(x, y) = (1-d_y)\,I_{\mathrm{top}} + d_y\,I_{\mathrm{bottom}}")
    st.write("Substituting the two horizontal results and expanding:")
    show_equation(BILINEAR_EQUATION)
    st.write("Each pixel contributes according to its horizontal and vertical proximity to the interpolation point. The four "
             "weights sum to 1.")


def bilinear_experiment(example):
    st.subheader("4.2 Experimental Validation", anchor=False)
    x, y = BILINEAR_POINT
    dx, dy, weights = example["dx"], example["dy"], example["weights"]
    st.write(rf"Video 2, Feature 12 (frame 482): $P = ({x:.8f},\; {y:.8f})$, so $d_x = {dx:.8f}$ and $d_y = {dy:.8f}$.")
    show_asset("video_2_bilinear_interpolation.png", "Feature 12, its four surrounding grayscale pixels and the point P.")
    names = (r"w_{11} = (1-d_x)(1-d_y)", r"w_{21} = d_x(1-d_y)", r"w_{12} = (1-d_x)\,d_y", r"w_{22} = d_x d_y")
    factors = ((1 - dx, 1 - dy), (dx, 1 - dy), (1 - dx, dy), (dx, dy))
    with st.expander("View Numerical Calculations", expanded=False):
        wrapping_equations("weights", [rf"{name} = ({a:.8f})({b:.8f}) \approx {w:.6f}"
                                       for name, (a, b), w in zip(names, factors, weights)])
        terms = " + ".join(rf"{w:.6f}({pixel[2]})" for w, pixel in zip(weights, BILINEAR_PIXELS))
        wrapping_equations("intensity", [r"w_{11} + w_{21} + w_{12} + w_{22} = 1",
                                         rf"I(P) = {terms}",
                                         rf"I(P) \approx {example['intensity']:.8f}"])
        st.caption("Weights are shown rounded to six decimals; the final intensity was calculated with the "
                   "full-precision weights.")
    simple_table("Pixel", {"Pixel": [name for name, _, _ in BILINEAR_PIXELS],
                           "Coordinate": [f"({px}, {py})" for _, (px, py), _ in BILINEAR_PIXELS],
                           "Intensity": [str(value) for _, _, value in BILINEAR_PIXELS],
                           "Weight": [f"{w:.6f}" for w in weights]})
    mantissa, exponent = f"{example['difference']:.3e}".split("e")
    superscript = str(int(exponent)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    for column, card in zip(st.columns(3), (("Manual Intensity", f"{example['intensity']:.8f}"),
                                            ("OpenCV Intensity", f"{BILINEAR_OPENCV:.8f}"),
                                            ("Absolute Difference", f"{mantissa} × 10{superscript}"))):
        with column:
            metric_card(*card)
    st.write("The manually calculated intensity closely matches OpenCV's interpolated intensity.")


def part_a_validation_video(data):
    result = tracking_comparison(data)
    number, (old_frame, new_frame) = data["video"], data["frames"]
    (x1, y1), (u, v), (a, b) = data["old"], data["manual"], data["atb"]
    (m11, m12), (m21, m22) = data["ata"]
    x2, y2 = result["new"]
    st.subheader(f"5.{number} Video {number} — {data['title']}", anchor=False)
    st.markdown(f"##### Feature {data['feature']} — Frames {old_frame} and {new_frame}")
    ou, ov = data["opencv"]
    show_asset(f"video_{number}_validation_comparison_arrows.png",
               f"Consecutive frames {old_frame} (left) and {new_frame} (right). The yellow arrow represents the OpenCV "
               f"displacement vector, magnified {ARROW_MAGNIFICATION}× (maximum {ARROW_MAX_LENGTH:.0f} px) for visibility.")
    with st.expander("View Numerical Calculations", expanded=False):
        show_equation(rf"P_1 = ({x1:.4f},\; {y1:.4f})")
        st.write(f"Normal equations $A^{{T}}A\\,\\mathbf{{d}} = A^{{T}}\\mathbf{{b}}$ over the 21 × 21 window centered at "
                 f"{data['center']}:")
        show_equation(rf"""\begin{{bmatrix}} {m11:.6f} & {m12:.6f} \\ {m21:.6f} & {m22:.6f} \end{{bmatrix}}
\begin{{bmatrix}} u \\ v \end{{bmatrix}} =
\begin{{bmatrix}} {a:.6f} \\ {b:.6f} \end{{bmatrix}}""")
        st.write("Solving with the 2 × 2 inverse:")
        determinant = m11 * m22 - m12 * m21
        wrapping_equations(f"inverse_{number}", [
            rf"\begin{{bmatrix}} u \\ v \end{{bmatrix}} = \frac{{1}}{{{determinant:.6f}}} \allowbreak "
            rf"\begin{{bmatrix}} {m22:.6f} & {-m12:.6f} \\ {-m21:.6f} & {m11:.6f} \end{{bmatrix}} \allowbreak "
            rf"\begin{{bmatrix}} {a:.6f} \\ {b:.6f} \end{{bmatrix}} "
            rf"\approx \begin{{bmatrix}} {u:+.4f} \\ {v:+.4f} \end{{bmatrix}}"])
        st.write("Applying the tracking equations:")
        sign = "+" if v >= 0 else "-"
        wrapping_equations(f"predicted_{number}", [
            rf"x_2 = x_1 + u = {x1:.6f} + {u:.6f} \approx {x2:.4f}",
            rf"y_2 = y_1 + v = {y1:.6f} {sign} {abs(v):.6f} \approx {y2:.4f}"])
    ox, oy = data["opencv_new"]
    rows = [("Horizontal displacement (px)", signed(u), signed(ou)), ("Vertical displacement (px)", signed(v), signed(ov)),
            ("New x-coordinate", f"{x2:.4f}", f"{ox:.4f}"), ("New y-coordinate", f"{y2:.4f}", f"{oy:.4f}"),
            ("Motion magnitude (px)", *(f"{m:.4f}" for m in result["magnitude"]))]
    if number == 2:
        rows.append(("Motion angle, downward from rightward (degrees)", *(f"{angle:.4f}" for angle in result["angles"])))
    simple_table("Measurement", dict(zip(("Measurement", "Manual Lucas–Kanade", "OpenCV"), map(list, zip(*rows)))))
    cards = [("Position Difference Relative to OpenCV", f"{result['position_difference']:.4f} px")]
    if number == 2:
        cards.append(("Angular Difference Relative to OpenCV", f"{result['angle_difference']:.4f}°"))
    for column, card in zip(st.columns(2), cards):
        with column:
            metric_card(*card)
    if number == 1:
        st.write("Both methods identify predominantly rightward motion.")
    else:
        st.write("Both methods identify downward-right motion.")


def part_a_validation():
    st.header("5. Experimental Validation of Motion Tracking", anchor=False)
    st.write("The derived Lucas–Kanade equations are applied to one feature in each video. The predicted positions are "
             "compared with OpenCV's tracked positions using the position difference e.")
    show_equation(r"""\begin{aligned}
\Delta x &= x_{\mathrm{manual}} - x_{\mathrm{OpenCV}} \\
\Delta y &= y_{\mathrm{manual}} - y_{\mathrm{OpenCV}} \\
e &= \sqrt{(\Delta x)^{2} + (\Delta y)^{2}}
\end{aligned}""")
    for data in VALIDATIONS:
        part_a_validation_video(data)
    st.subheader("Validation and Limitations", anchor=False)
    st.write("Both predicted positions differ from OpenCV by less than 0.32 pixels, although manual displacement "
             "magnitudes are approximately 31% and 33% larger. The manual method uses a single-level, one-step "
             "approximation, while "
             "OpenCV uses iterative pyramidal Lucas–Kanade. Validation is limited to one feature per video, with OpenCV "
             "serving as a reference rather than ground truth.")


def show_part_a():
    st.html(PART_A_CSS)
    with st.container(key="m6a_head"):
        with st.container(key="m6a_label"):
            st.markdown("MODULE 6 — PART A")
        with st.container(key="m6a_title"):
            st.title("Optical Flow and Motion Tracking", anchor=False)
        st.header("My Experiment & Results", anchor=False)
    with st.container(key="m6a_body"):
        for number, section in enumerate((part_a_setup, part_a_flow, part_a_lucas_kanade_theory, part_a_bilinear,
                                          part_a_validation), 1):
            with st.container(key=f"m6a_sec_{number}"):
                section()
        with st.container(key="m6a_conclusion"):
            st.header("6. Conclusion", anchor=False)
            st.markdown(
                "Optical flow successfully distinguished horizontal and diagonal motion. For the two validated features "
                "(one per video), the derived motion tracking equations produced predicted positions within 0.32 pixels "
                "of OpenCV's estimates, while bilinear interpolation at the tested point demonstrated close numerical "
                "agreement with OpenCV.")


# Part B results page. It reuses the Part A layout under its own keys, so nothing here reaches Part A or the visitor
# workflows. In the light theme every fill (conclusion, tables, dropdowns) is removed.
PART_B_RESULTS_CSS = PART_A_CSS.replace("m6a_", "m6bres_") + """<style>
.st-key-m6bres_body [data-testid="stExpander"] h5 { padding-top: 0.9rem; }
.st-key-m6bres_body [data-testid="stTable"] :is(th, td, th p, td p) { white-space: nowrap; overflow-wrap: normal; }
.st-key-m6bres_body [data-testid="stExpander"] [data-testid="stTable"] :is(th, td) { font-size: 0.875rem; }
.st-key-m6bres_body :is([class*="st-key-m6bres_math_"], [class*="st-key-m6bres_graph_"]) {
    background: color-mix(in srgb, currentColor 4%, transparent);
    border: 1px solid color-mix(in srgb, currentColor 14%, transparent); border-radius: 12px; padding: 0.9rem 1.25rem; gap: 0.6rem;
}
.st-key-m6bres_body :is([class*="st-key-m6bres_math_"], [class*="st-key-m6bres_graph_"])
    :is([data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] > :last-child) { margin-bottom: 0; }
.st-key-m6bres_body [class*="st-key-m6bres_math_"] li { font-size: 0.95rem; }
.st-key-m6bres_body [class*="st-key-m6bres_graph_"] hr { margin: 0.25rem 0; }
@media (max-width: 640px) {
    .st-key-m6bres_body :is([class*="st-key-m6bres_math_"], [class*="st-key-m6bres_graph_"]) { padding: 0.75rem; }
}
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) .st-key-m6bres_conclusion { background: transparent; }
    :root:not([data-theme="dark"]) .st-key-m6bres_body :is([data-testid="stTable"], [data-testid="stTable"] table,
        [data-testid="stTable"] thead, [data-testid="stTable"] tr, [data-testid="stTable"] th, [data-testid="stTable"] td,
        [data-testid="stExpander"] details, [data-testid="stExpander"] summary, [data-testid="stExpanderDetails"],
        [data-testid="stMetric"]) { background: transparent !important; }
}
</style>"""

# Mid-tone data colors that stay readable on both themes; backgrounds are transparent and text follows the theme.
PART_B_CORNER = "#2E7DD7"
PART_B_LANDMARK = "#E07A2F"
PART_B_REFERENCE = "#8A8F98"
PART_B_GRID = "rgba(128, 128, 128, 0.30)"
PART_B_CAPTIONS = ("View 1 — Near-overhead view", "View 2 — Oblique view, spine side visible",
                   "View 3 — Lower oblique view, bottom edge visible",
                   "View 4 — Spine-side oblique view, title inverted")


def num(value, places=4):
    """Fixed-point text with a true minus sign; values that round to zero carry no sign."""
    text = f"{value:.{places}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "−")


def tex(value, places=4):
    """A number for use inside an equation."""
    return num(value, places).replace("−", "-")


def tex_squared(value, places=4):
    text = tex(value, places)
    return f"({text})^2" if text.startswith("-") else f"{text}^2"


def tex_vector(values, places=4):
    return r"\begin{bmatrix} " + r" \\ ".join(num(v, places).replace("−", "-") for v in values) + r" \end{bmatrix}"


def matrix_table(index_name, labels, columns, matrix, places=4):
    simple_table(index_name, {index_name: list(labels),
                              **{column: [num(row[k], places) for row in matrix] for k, column in enumerate(columns)}})


@st.cache_data(show_spinner=False)
def part_b_photo_bytes(path, modified):
    """Upright, screen-sized copy of one submission photograph (no metadata other than its color profile)."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        profile = source.info.get("icc_profile")
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((1050, 1400), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90, icc_profile=profile)
    return buffer.getvalue()


def part_b_image(path, caption=None):
    if not path.is_file():
        st.warning(f"Unavailable Part B asset: {path.name}")
        return
    try:
        st.image(str(path), caption=caption, width="stretch")
    except (OSError, ValueError) as error:
        st.warning(f"Unable to display {path.name}: {error}")


def part_b_plot(figure, key):
    # Streamlit's chart theme supplies the light or dark text color.
    st.plotly_chart(figure, width="stretch", theme="streamlit", key=key, config={"displaylogo": False})


def part_b_top_view_figure(data):
    import plotly.graph_objects as go

    cm, w, h = data["cm"], data["width"] / 2, data["height"] / 2
    figure = go.Figure()
    figure.add_scatter(x=[-w, w, w, -w, -w], y=[-h, -h, h, h, -h], mode="lines", hoverinfo="skip",
                       name="Ideal rectangle (reference)", line=dict(color=PART_B_REFERENCE, width=2, dash="dash"))
    loop = [0, 1, 2, 3, 0]
    figure.add_scatter(x=cm[loop, 0], y=cm[loop, 1], mode="lines+markers+text", name="Estimated boundary (P1–P4)",
                       text=["P1", "P2", "P3", "P4", ""], hoverinfo="x+y",
                       textposition=["top left", "top right", "bottom right", "bottom left", "top left"],
                       line=dict(color=PART_B_CORNER, width=3), marker=dict(size=10, color=PART_B_CORNER))
    figure.add_scatter(x=cm[4:, 0], y=cm[4:, 1], mode="markers+text", name="Landmarks (P5–P8)",
                       text=list(("P5", "P6", "P7", "P8")), hoverinfo="x+y",
                       textposition=["bottom center", "middle right", "bottom center", "top right"],
                       marker=dict(size=10, color=PART_B_LANDMARK, symbol="diamond"))
    axis = dict(gridcolor=PART_B_GRID, zeroline=False, constrain="domain", showline=True, linecolor=PART_B_GRID)
    figure.update_xaxes(title_text="X (cm)", range=[cm[:, 0].min() - 2.5, cm[:, 0].max() + 2.5], **axis)
    figure.update_yaxes(title_text="Y (cm, downward)", range=[cm[:, 1].max() + 1.8, cm[:, 1].min() - 1.8],
                        scaleanchor="x", scaleratio=1, **axis)
    figure.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12, yanchor="top"))
    return figure


def part_b_3d_figure(data):
    import plotly.graph_objects as go

    cm = data["cm"]
    figure = go.Figure()
    loop = [0, 1, 2, 3, 0]
    figure.add_scatter3d(x=cm[loop, 0], y=cm[loop, 1], z=cm[loop, 2], mode="lines+markers+text",
                         name="Corners (P1–P4)", text=["P1", "P2", "P3", "P4", ""],
                         textposition=["top left", "top right", "middle right", "bottom left", "top left"],
                         line=dict(color=PART_B_CORNER, width=5), marker=dict(size=5, color=PART_B_CORNER))
    figure.add_scatter3d(x=cm[4:, 0], y=cm[4:, 1], z=cm[4:, 2], mode="markers+text", name="Landmarks (P5–P8)",
                         text=["P5", "P6", "P7", "P8"], textposition=["top center", "middle right", "top left", "top right"],
                         marker=dict(size=5, color=PART_B_LANDMARK, symbol="diamond"))

    # One margin on every axis and box sides proportional to the axis spans keep one centimeter the same length in all
    # three directions; the camera only chooses the viewpoint.
    ranges = [[float(low) - 1.0, float(high) + 1.0] for low, high in zip(cm.min(axis=0), cm.max(axis=0))]
    spans = [high - low for low, high in ranges]
    box = dict(zip("xyz", (1.1 * span / max(spans) for span in spans)))

    def axis(title, limits):
        return dict(title=dict(text=title), range=limits, showbackground=False, gridcolor=PART_B_GRID, zerolinecolor=PART_B_GRID,
                    showline=True, linecolor=PART_B_GRID)

    figure.update_layout(height=580, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                         scene=dict(xaxis=axis("X (cm)", ranges[0]), yaxis=axis("Y (cm)", ranges[1]),
                                    zaxis=axis("Z (cm)", ranges[2]), aspectmode="manual", aspectratio=box,
                                    bgcolor="rgba(0,0,0,0)",
                                    camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=1.25, y=-0.65, z=-1.35))),
                         legend=dict(orientation="h", x=0.5, xanchor="center", y=0.02))
    return figure


def part_b_math(key):
    """Shaded group holding one explanation, its equation and the variable definitions."""
    return st.container(key=f"m6bres_math_{key}")


def part_b_graph(key, title, introduction, figure, caption, observation):
    """One bordered group holding a graph with its title, introduction, caption and observation."""
    with st.container(key=f"m6bres_graph_{key}"):
        st.markdown(f"##### {title}")
        st.write(introduction)
        part_b_plot(figure, f"m6bres_plot_{key}")
        st.caption(caption)
        st.divider()
        st.markdown(f"**Observation**\n\n{observation}")


def part_b_setup(data, results):
    st.header("1. Experimental Setup", anchor=False)
    st.write("**Objective:** Reconstruct the front cover of a book from four camera viewpoints using structure from "
             "motion and estimate its boundary.")
    for column, card in zip(st.columns(3), (("Camera Viewpoints", "4"), ("Corresponding Points", "8"),
                                            ("Actual Cover Size", f"{data['width']:g} × {data['height']:g} cm"))):
        with column:
            metric_card(*card)
    st.write("The book remained stationary while the camera moved to capture four photographs from different "
             "viewpoints. The cover dimensions were measured with a ruler.")

    st.subheader("Camera Viewpoints", anchor=False)
    paths = [results.PHOTOS / name for name in results.PHOTO_NAMES]
    for first in (0, 2):
        for column, index in zip(st.columns(2), (first, first + 1)):
            with column:
                st.markdown(f"##### {PART_B_CAPTIONS[index]}")
                if paths[index].is_file():
                    st.image(part_b_photo_bytes(str(paths[index]), paths[index].stat().st_mtime), width="stretch")
                else:
                    st.warning(f"Unavailable Part B asset: {paths[index].name}")
    st.caption("Viewpoints are described qualitatively. Exact camera distances, heights and viewing angles were not "
               "measured.")

    st.subheader("Camera Parameters", anchor=False)
    simple_table("Parameter", {
        "Parameter": ["Camera", "Lens", "Focal length", "Image resolution"],
        "Value": ["Apple iPhone 17 Pro Max", "Rear wide-angle camera", "6.765 mm", "4284 × 5712 px"]})
    st.caption("An orthographic camera model was used without intrinsic calibration.")


def part_b_correspondences(data, results):
    st.header("2. Corresponding Feature Points", anchor=False)
    st.write("Eight corresponding points were used: four book cover corners (P1–P4) and four interior feature "
             "points (P5–P8). The four corners were selected in View 1, and the remaining corresponding points were "
             "identified automatically. All corresponding points were verified, and missing or inaccurate positions "
             "were manually selected or corrected before reconstruction.")
    part_b_image(results.REVISED / "part_b_correspondence_views.png",
                 "Corresponding points P1–P8 across four viewpoints.")
    st.subheader("Corresponding Landmark Close-ups", anchor=False)
    st.write("The close-ups show the same eight physical landmarks across all four viewpoints, demonstrating the "
             "point correspondences used for structure-from-motion reconstruction.")
    st.write("Each label gives the image coordinates (x, y) in pixels of the full-resolution upright 4284 × 5712 "
             "images, with the origin at the upper-left, x to the right and y downward.")
    part_b_image(results.REVISED / "part_b_landmark_closeups.png")


def part_b_reconstruction(data):
    st.header("3. Structure from Motion Reconstruction", anchor=False)
    st.write("Structure from motion recovers the 3D positions of points, together with the camera motion, from their "
             "image positions in several viewpoints.")

    st.subheader("Orthographic Projection Model", anchor=False)
    with part_b_math("projection"):
        st.write("The image coordinates of the eight selected points are known, but their 3D coordinates are unknown. "
                 "The orthographic projection equations establish the relationship between these coordinates.")
        show_equation(r"""\begin{aligned} x_{fp} &= \mathbf{i}_f^{T}\mathbf{s}_p + t_{x,f} \\
y_{fp} &= \mathbf{j}_f^{T}\mathbf{s}_p + t_{y,f} \end{aligned}""")
        st.markdown(r"""- $(x_{fp}, y_{fp})$: image coordinates of point $p = 1, \dots, 8$ in viewpoint $f = 1, \dots, 4$
- $\mathbf{s}_p = (X_p, Y_p, Z_p)^{T}$: 3D position of the point
- $\mathbf{i}_f, \mathbf{j}_f$: unit, mutually perpendicular camera x and y axes
- $(t_{x,f}, t_{y,f})$: image translation""")

    st.subheader("Measurement Matrix and Coordinate Centering", anchor=False)
    with part_b_math("centering"):
        st.write("The book appears at different positions in the four photographs. Coordinate centering removes these "
                 "position shifts by subtracting the average point position in each image.")
        show_equation(r"\tilde{x}_{fp} = x_{fp} - \bar{x}_f, \qquad \tilde{y}_{fp} = y_{fp} - \bar{y}_f")
        st.markdown(r"""- $(\bar{x}_f, \bar{y}_f)$: mean image position of the points in viewpoint $f$
- $\tilde{W}$: 8 × 8 measurement matrix of the centered coordinates, with two rows (x and y) per viewpoint and one column per point""")
    with st.expander("View Coordinate Centering Calculations", expanded=False):
        part_b_centering_calculations()

    st.subheader("Tomasi–Kanade Factorization", anchor=False)
    with part_b_math("factorization"):
        st.write("Tomasi–Kanade factorization separates the image measurements into camera motion and object structure, "
                 "allowing the shape of the book cover to be reconstructed.")
        show_equation(r"\tilde{W} \approx MS")
        st.markdown(r"""- $M$ (8 × 3): camera motion, stacking the axes $\mathbf{i}_f^{T}$ and $\mathbf{j}_f^{T}$ of every viewpoint
- $S$ (3 × 8): structure, whose columns are the reconstructed 3D points
- $\approx$: exact only for ideal orthographic images; the photographs contain perspective effects and point location errors""")

    st.subheader("Singular Value Decomposition (SVD)", anchor=False)
    with part_b_math("svd"):
        st.write("Singular value decomposition (SVD) is used to retain the three largest singular components of the "
                 "measurement matrix to obtain an initial 3D reconstruction of the eight points.")
        show_equation(r"\tilde{W} = U\Sigma V^{T} \approx U_3\Sigma_3V_3^{T} = \hat{M}\hat{S}")
        st.markdown(r"""- $U, V$: singular vectors
- $\Sigma$: singular values in decreasing order; the subscript 3 keeps the three largest
- $\hat{M}, \hat{S}$: initial motion and structure estimates""")

    st.subheader("Metric Upgrade", anchor=False)
    with part_b_math("metric"):
        st.write("The initial reconstruction may stretch or distort the book cover. The metric upgrade reduces "
                 "geometric ambiguity by making the recovered camera axes approximately perpendicular and unit length.")
        show_equation(r"M = \hat{M}Q, \qquad S = Q^{-1}\hat{S}")
        st.markdown(r"""- $Q$: 3 × 3 transformation matrix that makes the recovered camera axes approximately unit length and mutually perpendicular
- $M, S$: upgraded camera motion and 3D point structure""")
    with st.expander("View Orthographic Constraints", expanded=False):
        part_b_orthographic_constraints()
    st.markdown("The reconstructed points represent the shape of the book cover but are not yet aligned with its width "
                "and height or expressed in centimeters. Coordinate alignment and scaling are therefore needed to "
                "estimate its physical boundary.\n\n"
                "Since the book cover is approximately flat, its shape can be described using two dimensions. Variations "
                "in the reconstructed depth mainly represent reconstruction errors rather than actual physical depth.")


def part_b_centering_calculations():
    st.write("The arithmetic mean of the eight image coordinates gives the average position of the book cover points in "
             "each photograph. Coordinate centering subtracts this mean from every point in that view:")
    show_equation(r"\bar{x}_f = \frac{1}{8}\sum_{p=1}^{8} x_{fp}, \qquad \bar{y}_f = \frac{1}{8}\sum_{p=1}^{8} y_{fp}")
    st.markdown(r"$\sum$ adds the coordinates of points $p = 1, \dots, 8$ in viewpoint $f$.")
    st.write("Placing the 3D origin at the centroid of the eight points shows why coordinate centering removes the "
             "book's position shift. In the orthographic projection equations, each image mean then equals the image "
             "translation, which cancels:")
    show_equation(r"""\begin{aligned} \sum_{p=1}^{8}\mathbf{s}_p = \mathbf{0} \;&\Rightarrow\; \bar{x}_f = t_{x,f},
\quad \bar{y}_f = t_{y,f} \\
\tilde{x}_{fp} &= \mathbf{i}_f^{T}\mathbf{s}_p, \quad \tilde{y}_{fp} = \mathbf{j}_f^{T}\mathbf{s}_p \end{aligned}""")
    st.markdown(r"The centered coordinates $(\tilde{x}_{fp}, \tilde{y}_{fp})$ depend only on the camera axes and the "
                r"3D point, so coordinate centering removes the image translation.")


def part_b_orthographic_constraints():
    st.write("The orthographic constraints require the two camera axes recovered from each of the four photographs to "
             "have unit length and be perpendicular. These conditions determine the metric upgrade matrix Q:")
    show_equation(r"""\begin{aligned} \hat{\mathbf{m}}_{x f}\,L\,\hat{\mathbf{m}}_{x f}^{T} &= 1 \\
\hat{\mathbf{m}}_{y f}\,L\,\hat{\mathbf{m}}_{y f}^{T} &= 1 \\
\hat{\mathbf{m}}_{x f}\,L\,\hat{\mathbf{m}}_{y f}^{T} &= 0, \qquad L = QQ^{T} \end{aligned}""")
    st.markdown(r"""- $\hat{\mathbf{m}}_{x f}, \hat{\mathbf{m}}_{y f}$: rows of $\hat{M}$ for viewpoint $f$
- $L$: symmetric 3 × 3 matrix, estimated by least squares (the constraints are linear in $L$) and then factored to obtain $Q$""")
    st.write("The orthographic constraints are satisfied only approximately, so the recovered axes do not represent "
             "calibrated camera poses.")


def part_b_alignment_calculations(data):
    w, h = data["mean_width"], data["mean_height"]
    st.markdown("##### Scale Factor")
    st.write("The least-squares scaling equation combines the measured cover dimensions with the reconstructed mean "
             "width and height to find the single scale factor for this book cover:")
    show_equation(rf"""\begin{{aligned}} s &= \frac{{({data['width']:g})({w:.4f}) + ({data['height']:g})({h:.4f})}}{{{w:.4f}^2 + {h:.4f}^2}} \\
&\approx {data['scale']:.10f}\ \text{{cm/unit}} \end{{aligned}}""")
    st.markdown(rf"$\bar{{w}} \approx {w:.4f}$ and $\bar{{h}} \approx {h:.4f}$ are in reconstruction units.")
    st.markdown("##### Conversion of Point P1")
    st.write("Uniform scaling multiplies the aligned top-left corner P1 by this scale factor to give its position on "
             "the cover in centimeters:")
    show_equation(rf"""\begin{{aligned}} \mathbf{{p}}_1 = s\,\mathbf{{b}}_1 &\approx {data['scale']:.10f}\,{tex_vector(data['book'][0], 4)} \\
&\approx {tex_vector(data['cm'][0], 3)}\ \mathrm{{cm}} \end{{aligned}}""")
    st.markdown(r"$\mathbf{b}_1$ is the aligned position of P1 and $\mathbf{p}_1$ its position in cm.")


def part_b_boundary(data):
    st.header("4. Estimated Object Boundary", anchor=False)
    st.subheader("Coordinate Alignment", anchor=False)
    with part_b_math("alignment"):
        st.write("The reconstructed book cover may be tilted in 3D space. A coordinate transformation aligns the X and "
                 "Y axes with the width and height of the cover, while the Z-axis represents depth relative to its "
                 "surface.")
        show_equation(r"\mathbf{b}_p = B(\mathbf{s}_p - \mathbf{o})")
        st.markdown(r"""- $\mathbf{s}_p$: reconstructed point $p$ (column $p$ of $S$)
- $\mathbf{o}$: origin, at the center of the four corners
- $B$: 3 × 3 rotation matrix into the cover axes, obtained using the best-fit plane of the eight points
- $\mathbf{b}_p$: aligned point""")

    st.subheader("Physical Scaling", anchor=False)
    with part_b_math("scaling"):
        st.write("The reconstructed coordinates are initially in arbitrary units. A least-squares scaling equation uses the "
                 f"measured width of {data['width']:g} cm and height of {data['height']:g} cm to determine a single scale "
                 "factor.")
        show_equation(r"s = \frac{W_a\bar{w} + H_a\bar{h}}{\bar{w}^{2} + \bar{h}^{2}}")
        st.markdown(rf"""- $s$: single scale factor that best matches both measured dimensions in the least-squares sense
- $W_a = {data['width']:g}$ cm, $H_a = {data['height']:g}$ cm: measured cover width and height
- $\bar{{w}}, \bar{{h}}$: reconstructed mean width and height before scaling""")
        st.write("Uniform scaling converts each aligned point into centimeters by multiplying its coordinates by the "
                 "scale factor.")
        show_equation(r"\mathbf{p}_p = s\,\mathbf{b}_p")
        st.markdown(r"""- $\mathbf{p}_p$: position of point $p$ in cm
- The same factor applies to all three axes, so the shape is preserved and not flattened.""")
    with st.expander("View Coordinate Alignment and Scaling Calculations", expanded=False):
        part_b_alignment_calculations(data)

    st.subheader("Reconstructed Points", anchor=False)
    st.write("After coordinate alignment and scaling, the eight reconstructed points are expressed in centimeters. The "
             "X and Y coordinates represent positions along the width and height of the book cover, while the Z "
             "coordinates represent depth relative to the aligned cover plane. Points P1–P4 are used to estimate the "
             "book cover boundary, and P5–P8 represent the additional landmarks.")
    matrix_table("Point", [f"P{i}" for i in range(1, 9)], ["X (cm)", "Y (cm)", "Z (cm)"], data["cm"], 3)

    st.subheader("Estimated Boundary", anchor=False)
    part_b_graph(
        "top", "Estimated Object Boundary — Top View",
        "The top view shows how closely the reconstructed boundary matches the rectangular shape of the book cover.",
        part_b_top_view_figure(data),
        f"The dashed rectangle represents the measured cover dimensions of {data['width']:g} × {data['height']:g} cm "
        "and is shown for reference only. It was not fitted to the reconstructed points.",
        "The reconstructed boundary approximately preserves the rectangular shape of the book cover. The left edge "
        "lies close to the reference rectangle, but the right edge is noticeably tilted, narrowing the top edge and "
        "widening the bottom edge.")
    part_b_graph(
        "3d", "Scaled 3D Reconstruction",
        "The eight reconstructed points are displayed in 3D space to examine the recovered book cover shape and "
        "determine whether the points lie on a flat surface.",
        part_b_3d_figure(data), "Drag to rotate. All axes use the same scale.",
        "The reconstructed points do not lie perfectly on a flat surface. Since the actual book cover is approximately "
        "flat, the differences in Z-coordinates mainly represent reconstruction error rather than actual depth.")


def part_b_validation_calculations(data):
    cm = data["cm"]
    top = next(m for m in data["measurements"] if m["key"] == "top_width")
    st.markdown("##### Top Edge Length (P1 to P2)")
    differences = [tex_squared(cm[1, k] - cm[0, k], 3) for k in range(3)]
    st.write("The Euclidean distance between the reconstructed corners P1 and P2 gives the estimated length of the "
             "cover's top edge:")
    show_equation(rf"""\begin{{aligned}} d_{{12}} &= \sqrt{{{differences[0]} + {differences[1]} + {differences[2]}}} \\
&\approx {top['reconstructed']:.3f}\ \mathrm{{cm}} \end{{aligned}}""")
    st.markdown(r"$d_{12}$ is the distance between P1 and P2.")
    st.markdown("##### Top Edge Percentage Error")
    st.write("The percentage error compares this top edge length with the measured cover width of "
             f"{top['actual']:g} cm to evaluate the estimate:")
    show_equation(rf"\frac{{\lvert {top['reconstructed']:.3f} - {top['actual']:g} \rvert}}{{{top['actual']:g}}} "
                  rf"\times 100\% \approx {top['error']:.2f}\%")
    st.markdown("The numerator is the absolute difference between the reconstructed and measured lengths.")


def part_b_validation(data):
    st.header("5. Experimental Validation", anchor=False)
    st.subheader("Euclidean Distance", anchor=False)
    with part_b_math("distance"):
        st.write("The Euclidean distance equation calculates the distance between adjacent reconstructed corners to "
                 "estimate the four edge lengths of the book cover.")
        show_equation(r"d_{ab} = \sqrt{(X_a-X_b)^2 + (Y_a-Y_b)^2 + (Z_a-Z_b)^2}")
        st.markdown(r"""- $d_{ab}$: distance between corners $a$ and $b$
- $(X_a, Y_a, Z_a), (X_b, Y_b, Z_b)$: their reconstructed coordinates in cm""")
    st.subheader("Percentage Error", anchor=False)
    with part_b_math("error"):
        st.write("The percentage error equation compares each reconstructed edge length with its measured length to "
                 "evaluate how closely the estimated dimensions match the actual book cover dimensions.")
        show_equation(r"\text{Error} = \frac{\lvert d_{\mathrm{reconstructed}} - d_{\mathrm{ref}} \rvert}"
                      r"{d_{\mathrm{ref}}} \times 100\%")
        st.markdown(r"""- $d_{\mathrm{reconstructed}}$: reconstructed length
- $d_{\mathrm{ref}}$: measured reference length""")
    with st.expander("View Validation Calculations", expanded=False):
        part_b_validation_calculations(data)
    edges = [m for m in data["measurements"] if not m["key"].startswith("diagonal")]
    simple_table("Measurement", {
        "Measurement": [m["label"] for m in edges],
        "Reference (cm)": [num(m["actual"], 3) for m in edges],
        "Reconstructed (cm)": [num(m["reconstructed"], 3) for m in edges],
        "Error (%)": [num(m["error"], 2) for m in edges]})
    st.caption("Since the measured width and height are used for scaling, their agreement with the reconstructed "
               "dimensions is not an independent validation.")
    st.markdown("**Reconstruction limitations:**\n\n"
                "- Perspective effects not fully represented by the orthographic model.\n"
                "- Limited depth recovery from a planar object.\n"
                "- Point localization errors, especially at corners near the rounded spine.")


def show_part_b():
    from Modules.Module_6.Part_B import results_data

    st.html(PART_B_RESULTS_CSS)
    with st.container(key="m6bres_head"):
        with st.container(key="m6bres_label"):
            st.markdown("MODULE 6 — PART B")
        with st.container(key="m6bres_title"):
            st.title("Structure from Motion Reconstruction", anchor=False)
        st.header("My Experiment & Results", anchor=False)
    try:
        data = results_data.load()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Unable to load the saved Part B experiment: {error}")
        return
    with st.container(key="m6bres_body"):
        sections = (lambda: part_b_setup(data, results_data), lambda: part_b_correspondences(data, results_data),
                    lambda: part_b_reconstruction(data), lambda: part_b_boundary(data), lambda: part_b_validation(data))
        for number, section in enumerate(sections, 1):
            with st.container(key=f"m6bres_sec_{number}"):
                section()
        errors = [m["error"] for m in data["measurements"] if not m["key"].startswith("diagonal")]
        with st.container(key="m6bres_conclusion"):
            st.header("6. Conclusion", anchor=False)
            st.markdown(
                "The structure-from-motion experiment reconstructed eight corresponding points from four camera "
                "viewpoints and estimated the book cover boundary. Using the measured cover dimensions for scaling, "
                f"individual edge errors ranged from {min(errors):.2f}% to {max(errors):.2f}%. Reconstruction "
                "accuracy was limited by perspective effects, planar depth ambiguity, and point localization errors.")


ACCENT = "#C9363E"
BLUE, BLUE_DARK = "#1D5FD1", "#174DA8"   # same blue as Modules 2, 4 and 7
# Homepage and My Experiments & Results page: stacked, centered 110px buttons using Module 4's values.
NAV_BUTTON_KEYS = ("m6_open_results", "m6_open_a_visitor", "m6_open_b_visitor", "m6_res_a", "m6_res_b")


def nav_selector(suffix):
    return ", ".join(f".st-key-{key}{suffix}" for key in NAV_BUTTON_KEYS)


HOME_TITLE = "Module 6"
HOME_SUBTITLE = "Optical Flow and Structure from Motion"

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
.st-key-m6_home_nav {{ flex-direction: column; flex-wrap: nowrap; gap: 24px; max-width: 560px; width: 100%; margin: 0 auto; }}
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
.st-key-m6_res_a button p, .st-key-m6_res_b button p {{ font-size: 0.95rem; }}   /* the "(Part X)" line */
{nav_selector(" button:hover")} {{ background-color: #A62A31; border-color: #A62A31; color: #FFFFFF; }}
{nav_selector(" button:focus-visible")} {{
    outline: 3px solid #FFFFFF; outline-offset: 2px; box-shadow: 0 0 0 6px {ACCENT};
}}
{nav_selector(" button:active")} {{ background-color: #8E2229; border-color: #8E2229; }}
.st-key-m6_open_results button {{ background-color: {BLUE}; border-color: {BLUE}; }}
.st-key-m6_open_results button:hover {{ background-color: {BLUE_DARK}; border-color: {BLUE_DARK}; }}
.st-key-m6_open_results button:active {{ background-color: #123F8F; border-color: #123F8F; }}
.st-key-m6_open_results button:focus-visible {{ box-shadow: 0 0 0 6px {BLUE}; }}
@media (max-width: 640px) {{ .st-key-m6_home_nav {{ gap: 16px; }} }}
</style>
"""

# My Experiments & Results page only: the small module label above a smaller title (Module 4's results-title size).
RESULTS_TITLE_CSS = """
<style>
.st-key-m6_sel_label { padding-bottom: 24px; }
.st-key-m6_sel_label [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-m6_sel_label p {
    margin: 0; text-align: center; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.18em; opacity: 0.55;
}
.st-key-m6_home_title h1 { font-size: clamp(1.75rem, 7vw, 48px); }
@media (min-width: 801px) { .st-key-m6_home_title h1 { white-space: nowrap; } }
</style>
"""


# Light theme only (Streamlit follows prefers-color-scheme), using Module 4's text colors: primary charcoal for titles, breadcrumbs,
# captions and tables, and a readable darker gray for captions in the visitor workflows. Passed to nav_row on every Module 6 view.
LIGHT_TEXT_CSS = """<style>
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) { --nav-muted: #31333F; }  /* intermediate breadcrumb labels and separators */
    :root:not([data-theme="dark"]) :is(.st-key-m6_sel_label p, .st-key-m6_home_title h1, .st-key-m6_home_subtitle h2) {
        color: #31333F; opacity: 1;
    }
    :root:not([data-theme="dark"]) [data-testid="stMain"] :is([data-testid="stCaptionContainer"], [data-testid="stImageCaption"],
        [data-testid="stCaptionContainer"] *, [data-testid="stImageCaption"] *, [data-testid="stTable"] th, [data-testid="stTable"] td) {
        color: #31333F !important; opacity: 1;
    }
    :root:not([data-theme="dark"]) [data-testid="stMain"] :is([class*="st-key-m6v_step_"], [class*="st-key-m6b_step_"])
        :is([data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *,
        [data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] *) {
        color: #4B5160 !important; opacity: 1;
    }
}
</style>"""


def home_layout(title, buttons, subtitle=None, label=None, extra_css=""):
    """Centered label, title, subtitle and stacked buttons; (label, key, destination) triples."""
    st.markdown(HOME_CSS + extra_css, unsafe_allow_html=True)
    with st.container(key="m6_home_layout"):
        if label is not None:
            with st.container(key="m6_sel_label"):
                st.markdown(label)
        with st.container(key="m6_home_title"):
            st.title(title, anchor=False)
        if subtitle is not None:
            with st.container(key="m6_home_subtitle"):
                st.header(subtitle, anchor=False)
        with st.container(key="m6_home_nav"):
            for text, key, destination in buttons:
                st.button(text, key=key, on_click=select_part, args=(destination,), width="stretch")


def part_label(title, part):
    """Button label: the title (bold block) and a smaller "(Part X)" line, both inside the one clickable button."""
    return f"**{title}**  \n(Part {part})"


VISITOR_CLICK_KEY = "m6v_click"
VISITOR_PREVIEW_MAX = 700
VISITOR_KEYS = ("m6v_bytes", "m6v_filename", "m6v_digest", "m6v_analysis", "m6v_error")
# TrackingParams field -> widget key of the eight tracking settings
VISITOR_PARAM_KEYS = {"max_corners": "m6v_p_corners", "quality_level": "m6v_p_quality", "min_distance": "m6v_p_distance",
                      "block_size": "m6v_p_block", "window": "m6v_p_window", "max_level": "m6v_p_level",
                      "max_iterations": "m6v_p_iter", "epsilon": "m6v_p_eps"}
VISITOR_STALE_MESSAGE = "Settings changed. Run tracking again."
# Same 900px content width as the other visitor workflows, so the preview fits inside a bordered step. Title, step headings
# and subsection headings step down in size.
VISITOR_PAGE_CSS = """<style>
.stMainBlockContainer, .block-container { max-width: 900px !important; }
.st-key-m6v_head { gap: 0; padding-bottom: 0.75rem; }
.st-key-m6v_label { padding-bottom: 12px; }
.st-key-m6v_label [data-testid="stMarkdownContainer"] { margin: 0; }
.st-key-m6v_label p { margin: 0; text-align: center; font-size: 12px; font-weight: 600; letter-spacing: 0.18em; opacity: 0.55; }
.st-key-m6v_head h1 {
    font-size: clamp(1.75rem, 7vw, 40px); font-weight: 800; line-height: 1.15; text-align: center; padding: 0; margin: 0;
    overflow-wrap: anywhere;
}
[class*="st-key-m6v_step_"] h2 { font-size: 1.5rem; font-weight: 700; line-height: 1.25; padding: 0.25rem 0; }
[class*="st-key-m6v_step_"] h3 { font-size: 1.15rem; font-weight: 700; line-height: 1.3; padding: 0.5rem 0 0; }
.st-key-m6v_select_hint { min-height: 3.5rem; text-align: center; }
.st-key-m6v_select_hint :is([data-testid="stElementContainer"], [data-testid="stMarkdown"], [data-testid="stMarkdownContainer"]) {
    width: 100% !important; }
.st-key-m6v_select_hint p { text-align: center; }
.st-key-m6v_step_4 [data-testid="stMetricValue"] { font-size: clamp(1.2rem, 4vw, 1.6rem); }
.st-key-m6v_step_4 :is([data-testid="stMetricValue"], [data-testid="stMetricValue"] *, [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"]) {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]) :is(.st-key-m6v_label p, .st-key-m6v_head h1) { color: #31333F; opacity: 1; }
}
</style>"""


def visitor_clear_run_error():
    st.session_state.pop("m6v_run_error", None)
    st.session_state.pop("m6v_run_error_sig", None)


def visitor_reset_results():
    """Drop the results, any run error and the outdated-results notice."""
    st.session_state["m6v_results"] = None
    st.session_state["m6v_results_sig"] = None
    st.session_state.pop("m6v_stale", None)
    visitor_clear_run_error()


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
    for key in ("m6v_params_saved", *VISITOR_PARAM_KEYS.values()):
        st.session_state.pop(key, None)
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
    """Reset All: a fresh uploader key is the supported way to empty the native uploader from code."""
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
    """Displayed width of the first-frame preview and result video in pixels."""
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


def visitor_reset_parameters():
    """Reset to Defaults: restore all eight settings. Results are invalidated only if a value actually changes."""
    from Modules.Module_6.Part_A import visitor_tracking as vt

    defaults = vt.TrackingParams()
    for name, key in VISITOR_PARAM_KEYS.items():
        st.session_state[key] = getattr(defaults, name)
    st.session_state.pop("m6v_params_saved", None)
    visitor_clear_run_error()


def visitor_parameters(vt):
    """Advanced settings expander; defaults are the original experiment's values."""
    defaults = vt.TrackingParams()
    saved = st.session_state.get("m6v_params_saved") or {}
    for name, key in VISITOR_PARAM_KEYS.items():
        # Widget state is dropped while Step 3 is hidden, so the last values are restored from the saved copy.
        if key not in st.session_state:
            st.session_state[key] = saved.get(name, getattr(defaults, name))

    def setting(name, label, low, high, step, help_text, **options):
        return st.number_input(label, low, high, step=step, key=VISITOR_PARAM_KEYS[name], help=help_text, **options)

    with st.expander("Advanced Settings (Optional)", expanded=False):
        st.caption("The default settings match the original experiment.")
        left, right = st.columns(2)
        with left:
            max_corners = setting("max_corners", "Maximum corners", 1, 200, 1,
                                  "Largest number of features detected inside the selection.")
            quality = setting("quality_level", "Quality level", 0.01, 0.95, 0.05,
                              "Minimum corner strength, as a fraction of the strongest corner's.")
            min_distance = setting("min_distance", "Minimum distance (px)", 1, 50, 1,
                                   "Smallest spacing between features, in processed-video pixels.")
            block = setting("block_size", "Block size", 3, 31, 2,
                            "Neighborhood size used to score each corner (odd number of pixels).")
        with right:
            window = setting("window", "Lucas–Kanade window", 5, 51, 2,
                             "Side of the square window matched around each feature (odd number of pixels).")
            level = setting("max_level", "Pyramid maximum level", 0, 5, 1,
                            "Extra reduced-size image levels for larger motion. 0 uses only the processed frame.")
            iterations = setting("max_iterations", "Termination iterations", 1, 100, 1,
                                 "Maximum search iterations at each pyramid level.")
            epsilon = setting("epsilon", "Termination epsilon", 0.001, 1.0, 0.005,
                              "The search stops once its position update is smaller than this (pixels).",
                              format="%.3f")
        st.caption("Features lost to blur, occlusion or large motion are not replaced.")
        st.button("Reset to Defaults", key="m6v_reset_params", on_click=visitor_reset_parameters)
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


def time_axis_range(times):
    """(upper limit, tick count) of a time axis from zero, using 1-2-5 tick steps and at most 10 intervals."""
    longest = max((float(t) for t in times if pd.notna(t)), default=0.0) or 1.0
    magnitude = 10.0 ** math.floor(math.log10(longest / 10))
    step = next(m * magnitude for m in (1, 2, 5, 10) if longest / (m * magnitude) <= 10 + 1e-9)
    count = max(1, math.ceil(longest / step - 1e-9))
    return round(count * step, 10), count


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

    time_end, time_ticks = time_axis_range(frame["time_s"])
    # No fixed number format, so labels carry only the decimals their step needs; panning never labels negative times.
    time_axis = axis("Time (s)", tickCount=time_ticks, labelExpr="datum.value < 0 ? '' : datum.label")
    time_scale = alt.Scale(domain=[0, time_end], nice=False)

    def line_chart(data, series, y_title, value_title, y_domain, y_axis, value_format, count=False):
        long = data[["time_s", *series]].melt("time_s", var_name="key", value_name="value")
        long["series"] = long["key"].map(series)
        long["value"] = long["value"].astype(float)
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
        frame, {"tracked": "Tracked features"}, ["Features"], "Number of Tracked Features", [0, tracked_top],
        axis(None, format="d", tickMinStep=1), "d", count=True)

    displacement_title = ["Displacement (px)"]
    step_cols = {"mean_u": "u (horizontal)", "mean_v": "v (vertical)"}
    step_values = pd.concat([frame[c] for c in step_cols])
    step = line_chart(frame, step_cols, displacement_title, "Mean Displacement per Step (px)",
                      padded_domain(step_values), axis(None, format=".2f"), ".3f")

    cumulative_cols = {"mean_cumulative_u": "cumulative u (horizontal)", "mean_cumulative_v": "cumulative v (vertical)"}
    cumulative_values = pd.concat([frame[c] for c in cumulative_cols])
    cumulative = line_chart(frame, cumulative_cols, displacement_title, "Mean Cumulative Displacement (px)",
                            padded_domain(cumulative_values), axis(None, format=".1f"), ".2f")
    return [("Tracked Features", tracked), ("Mean Displacement per Step", step),
            ("Mean Cumulative Displacement", cumulative)]


# Spacing below the downloads and a 2 px red border on Reset All; scoped to this container.
VISITOR_RESULTS_NAV_CSS = """<style>
.st-key-m6v_results_nav { margin-top: 16px; }
.st-key-m6v_results_nav button { border: 2px solid #C9363E; width: 11rem; height: 2.75rem; min-height: 2.75rem; }
.st-key-m6v_results_nav button:hover, .st-key-m6v_results_nav button:focus-visible { border-color: #E04850; }
</style>"""


def visitor_results(results, plan):
    from Modules.Module_6.Part_A import visitor_tracking as vt

    st.header("Step 4 — Tracking Results", anchor=False)
    summary = results.summary
    st.subheader("Optical Flow Visualization", anchor=False)
    # Same displayed width as the first-frame preview, centered; the encoded bytes are untouched.
    with st.container(horizontal_alignment="center"):
        st.video(results.video_bytes, format="video/mp4", width=visitor_preview_width(plan))
    cap_unit = "processed-video px" if plan.resized else "px"
    st.caption("Yellow dots: tracked features. Arrows: motion direction and displacement magnified "
               f"{ARROW_MAGNIFICATION}× (maximum {ARROW_MAX_LENGTH:.0f} {cap_unit}).")

    st.subheader("Tracking Summary", anchor=False)
    cards = (("Initial Features", summary["initial_features"]), ("Final Features", summary["final_features"]),
             ("Retention", f"{summary['feature_retention_percent']:.1f}%"))
    for column, card in zip(st.columns(3), cards):
        with column:
            metric_card(*card)
    # Without a single tracked step there is no displacement to average.
    measured = any(f["tracked"] for f in results.per_frame)

    def value(text):
        return text if measured else "Not available"

    def card_row(row):
        for column, (label, text, help_text) in zip(st.columns(2), row):
            with column:
                metric_card(label, text, help_text=help_text)

    st.subheader("Mean Motion per Step", anchor=False,
                 help="Motion measured in the image can result from object motion, camera motion or both.")
    card_row((("Horizontal (u)", value(f"{summary['mean_u_per_step']:+.3f} px"), None),
              ("Vertical (v)", value(f"{summary['mean_v_per_step']:+.3f} px"), None)))
    card_row((("Magnitude", value(f"{summary['mean_magnitude_per_step']:.3f} px"),
               "Average of the individual displacement lengths, so it can exceed the length of the mean (u, v) vector."),
              ("Direction", value(summary["motion_direction"]), "Direction of the mean (u, v) displacement vector.")))

    st.subheader("Mean Cumulative Displacement", anchor=False)
    cumulative_help = "Mean displacement from the starting positions, at the last frame with tracked features."
    card_row(tuple(
        (label, "Not available" if summary[key] is None else f"{summary[key]:+.1f} px", cumulative_help)
        for label, key in (("Horizontal (u)", "mean_cumulative_u_final"), ("Vertical (v)", "mean_cumulative_v_final"))))
    lost = summary["all_features_lost_at_frame"]
    if lost:
        # The summary counts processed frames; the video overlay shows source frames, so report that number.
        shown = next((f["source_frame"] for f in results.per_frame if f["frame"] == lost), lost)
        name = "source frame" if plan.stride > 1 else "frame"
        st.warning(f"All features were lost at {name} {shown}. Later frames show no tracks.")
    skipped = f" (every {vt.ordinal(plan.stride)} frame tracked)" if plan.stride > 1 else ""
    st.caption("Original-frame pixels, without magnification. Positive u: rightward; positive v: downward. "
               f"Step interval: {plan.step_seconds:.3f} s{skipped}.")
    frame = pd.DataFrame(results.per_frame)
    if not frame.empty:
        with st.expander("View Tracking Graphs", expanded=False):
            for caption, chart in tracking_charts(frame, "px (original frame)"):
                st.caption(caption)
                st.altair_chart(chart, width="stretch", theme=None)  # Vega-Lite defaults, not Streamlit's theme fonts

    st.subheader("Download Results", anchor=False)
    first, second, third = st.columns(3)
    first.download_button("Video (MP4)", results.video_bytes, "tracked_video.mp4", "video/mp4",
                          key="m6v_dl_video", on_click="ignore", width="stretch")
    second.download_button("Tracking (CSV)", results.csv_bytes, "tracking.csv", "text/csv",
                           key="m6v_dl_csv", on_click="ignore", width="stretch")
    third.download_button("Summary (JSON)", results.summary_json(), "tracking_summary.json", "application/json",
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
    with st.expander("Video Recording Instructions", expanded=False):
        st.markdown(
            "- Keep the camera steady.\n"
            "- Choose an object with visible texture or edges.\n"
            f"- Only the first {vt.MAX_DURATION_S:.0f} seconds are analyzed.\n"
            "- If decoding fails, re-export as H.264 MP4.")
    st.html(VISITOR_UPLOADER_CSS)
    # max_upload_size is per widget, so only this uploader shows 50 MB; other modules keep Streamlit's default.
    # The native drop zone lists the size limit and formats.
    uploaded = st.file_uploader("Video file", type=[e.lstrip(".") for e in vt.ALLOWED_EXTENSIONS],
                                key=visitor_upload_key(), on_change=visitor_new_upload,
                                max_upload_size=vt.MAX_FILE_BYTES // 1024 // 1024, label_visibility="collapsed")
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
        except Exception:
            logging.getLogger(__name__).exception("Module 6 visitor video analysis failed")
            st.session_state["m6v_error"] = ("This video could not be analyzed. Re-export it as H.264 MP4 and "
                                             "upload it again.")
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
    """Metadata and the processing summary, directly under the uploader."""
    if analysis is None:
        return
    info, plan = analysis.info, analysis.plan
    st.html(VISITOR_INFO_CSS)
    with st.container(key="m6v_video_info"):
        columns = st.columns(4)
        columns[0].metric("Resolution", f"{info.width} × {info.height}")
        columns[1].metric("Duration", f"{info.duration:.1f} s" if info.duration else "Unknown")
        columns[2].metric("Frame Rate", f"{info.fps:.2f} fps" + (" (assumed)" if info.fps_assumed else ""))
        columns[3].metric("Frames", info.frame_count or "Unknown")
    st.caption(plan.summary_line())


# Selection guide image: a crop of an experiment frame; corners are in the full frame's pixels.
VISITOR_EXAMPLE_IMAGE = "video_1_validation_frame_old.png"
VISITOR_EXAMPLE_CROP = (200, 1050, 720, 1440)  # left, top, right, bottom
VISITOR_EXAMPLE_CORNERS = ((385, 1115), (530, 1362))


@st.cache_data(show_spinner=False)
def visitor_selection_example():
    """Two-corner guide in the style of Module 4's rectangle example; None if the saved frame is unavailable."""
    image = cv2.imread(str(OUTPUT / VISITOR_EXAMPLE_IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        return None
    left, top, right, bottom = VISITOR_EXAMPLE_CROP
    image = cv2.resize(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB), None, fx=2, fy=2,
                       interpolation=cv2.INTER_CUBIC)
    pad = 110
    image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=(236, 238, 242))
    (x1, y1), (x2, y2) = ((2 * (x - left), 2 * (y - top) + pad) for x, y in VISITOR_EXAMPLE_CORNERS)
    marker, ink = (255, 0, 255), (30, 33, 40)
    cv2.rectangle(image, (x1, y1), (x2, y2), marker, 6)
    for x, y in ((x1, y1), (x2, y2)):
        cv2.circle(image, (x, y), 18, ink, -1)
        cv2.circle(image, (x, y), 13, marker, -1)
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.7, 4
    # Labels sit in the padding above and below the photograph, next to their corners.
    cv2.putText(image, "1. First corner", (x1 - 90, 72), font, scale, ink, thickness, cv2.LINE_AA)
    cv2.putText(image, "2. Opposite corner", (x2 - 330, image.shape[0] - 42), font, scale, ink, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, 88), (x1, y1 - 20), ink, 3, cv2.LINE_AA)
    cv2.line(image, (x2, y2 + 20), (x2, image.shape[0] - 92), ink, 3, cv2.LINE_AA)
    return image


# Reset Selection and Continue split the width of the preview above them.
VISITOR_SELECT_ACTIONS_CSS = """<style>
.st-key-m6v_select_actions [data-testid="stElementContainer"] { flex: 1 1 0; min-width: 0; }
.st-key-m6v_select_actions button { height: 2.75rem; min-height: 2.75rem; padding: 0 0.75rem; }
</style>"""


def visitor_stage_selection(vt, analysis):
    plan = analysis.plan
    st.header("Step 2 — Select the Object", anchor=False)
    points = st.session_state["m6v_points"]
    frame_rgb = cv2.cvtColor(analysis.first_frame, cv2.COLOR_BGR2RGB)
    scale = visitor_preview_scale(plan)
    preview = cv2.resize(frame_rgb, (visitor_preview_width(plan), max(1, round(plan.height * scale))))
    roi, error = visitor_roi(points, plan)
    with st.expander("How to Select the Object", expanded=False):
        st.write("Keep the rectangle close to the object.")
        example = visitor_selection_example()
        if example is not None:
            with st.columns([1, 1.5, 1])[1]:
                st.image(example, width="stretch")
    # One instruction block of constant height, so the preview does not move between clicks.
    with st.container(key="m6v_select_hint"):
        if error:
            st.warning(f"{error} Click Reset Selection.")
        elif roi is not None:
            st.write("Object selected. Click Continue.")
        elif points:
            st.write("Click the opposite corner.")
        else:
            st.write("Click the first corner.")
    for x, y in points:
        cv2.circle(preview, (round(x * scale), round(y * scale)), 5, (255, 0, 255), -1)
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(preview, (round(rx * scale), round(ry * scale)),
                      (round((rx + rw - 1) * scale), round((ry + rh - 1) * scale)), (255, 0, 255), 2)
    st.html(VISITOR_SELECT_ACTIONS_CSS)
    width = preview.shape[1]
    with st.container(horizontal_alignment="center"):
        # The component's frame is left-aligned inside, so a box exactly as wide as the preview is what gets centered.
        with st.container(key="m6v_preview_box", width=width):
            # One component with a stable key for every state, so Reset Selection only changes its image.
            streamlit_image_coordinates(preview, width="content", cursor="crosshair" if len(points) < 2 else "default",
                                        key=VISITOR_CLICK_KEY,
                                        on_click=lambda: visitor_record_corner(VISITOR_CLICK_KEY, plan))
        # Never narrower than the two button labels need.
        with st.container(key="m6v_select_actions", horizontal=True, gap="small", width=max(width, 300)):
            st.button("Reset Selection", key="m6v_reset_selection", on_click=visitor_reset_clicked, width="stretch")
            st.button("Continue", key="m6v_continue", type="primary", disabled=roi is None,
                      on_click=visitor_continue, width="stretch")


def visitor_run_error_text(vt, error):
    """Short, actionable message for a tracking failure."""
    if isinstance(error, vt.NoFeaturesError):
        return ("No trackable features were found. Select an area with more texture or edges, or lower the quality "
                "level.")
    if isinstance(error, vt.BusyError):
        return "The server is busy. Wait a minute, then run tracking again."
    if isinstance(error, vt.EncodingError):
        return "The result video could not be created. Run tracking again."
    return str(error)


def visitor_stage_settings(vt, analysis, roi):
    plan = analysis.plan
    st.header("Step 3 — Run Tracking", anchor=False)
    st.write("Shi–Tomasi feature detection and pyramidal Lucas–Kanade optical flow.")
    params = visitor_parameters(vt)
    available, detail = vt.encoder_status()
    if not available:
        st.error(f"Tracking is unavailable because this server cannot encode video. {detail}")
    signature = (st.session_state["m6v_digest"], roi, params)
    # Results and run errors each belong to the inputs that produced them.
    if st.session_state.get("m6v_results") is not None and st.session_state.get("m6v_results_sig") != signature:
        visitor_reset_results()
        st.session_state["m6v_stale"] = True
    if st.session_state.get("m6v_run_error_sig", signature) != signature:
        visitor_clear_run_error()
    notice = st.empty()
    if st.session_state.get("m6v_stale"):
        notice.info(VISITOR_STALE_MESSAGE)
    clicked = st.button("Run Tracking", key="m6v_run", type="primary", disabled=params is None or not available,
                        width="stretch")
    if clicked:
        visitor_reset_results()
        notice.empty()
        progress = st.progress(0.0, text="Starting…")
        failure = None
        try:
            result = vt.run_tracking_job(st.session_state["m6v_bytes"], st.session_state["m6v_filename"], plan, roi,
                                         params, lambda fraction, text: progress.progress(fraction, text=text))
        except vt.VideoError as error:
            failure = visitor_run_error_text(vt, error)
        except Exception:
            logging.getLogger(__name__).exception("Module 6 visitor tracking failed")
            failure = "Tracking could not complete. Run it again, or try different settings or another video."
        else:
            st.session_state["m6v_results"] = result
            st.session_state["m6v_results_sig"] = signature
            visitor_scroll_to(4)
        if failure:
            st.session_state["m6v_run_error"] = failure
            st.session_state["m6v_run_error_sig"] = signature
        progress.empty()
        st.rerun()
    elif st.session_state.get("m6v_run_error"):
        st.error(st.session_state["m6v_run_error"])


def visitor_reset_all():
    visitor_clear_video()
    visitor_scroll_to(1)


def part_a_visitor_page():
    from Modules.Module_6.Part_A import visitor_tracking as vt

    st.session_state.setdefault("m6v_upload_generation", 0)
    st.session_state.setdefault("m6v_points", [])
    st.session_state.setdefault("m6v_results", None)
    st.session_state.setdefault("m6v_settings_open", False)

    st.html(VISITOR_PAGE_CSS)
    with st.container(key="m6v_head"):
        with st.container(key="m6v_label"):
            st.markdown("MODULE 6 — PART A")
        st.title("Optical Flow and Motion Tracking", anchor=False)
    # One continuous page: each step stays rendered once its prerequisites exist. The uploader runs first
    # because a removed or replaced video resets everything that depends on it within this same run.
    current = 1
    with st.container(border=True, key="m6v_step_1"):
        st.header("Step 1 — Upload Your Video", anchor=False)
        visitor_upload_section(vt)
        analysis = st.session_state.get("m6v_analysis")
        visitor_video_info(analysis)
    if analysis is not None:
        current = 2
        roi, _ = visitor_roi(st.session_state["m6v_points"], analysis.plan)
        with st.container(border=True, key="m6v_step_2"):
            visitor_stage_selection(vt, analysis)
        if st.session_state["m6v_settings_open"] and roi is not None:
            current = 3
            with st.container(border=True, key="m6v_step_3"):
                visitor_stage_settings(vt, analysis, roi)
            if st.session_state.get("m6v_results") is not None:
                current = 4
                with st.container(border=True, key="m6v_step_4"):
                    visitor_results(st.session_state["m6v_results"], analysis.plan)
                    st.html(VISITOR_RESULTS_NAV_CSS)
                    with st.container(key="m6v_results_nav", horizontal_alignment="center"):
                        st.button("Reset All", key="m6v_reset_all", on_click=visitor_reset_all)
    st.html(f"<style>.st-key-m6v_step_{current} {{ border-color: {ACCENT} !important; }}</style>")  # current step
    visitor_scroll_script()


# Part B visitor workflow: the experiment's correspondence method on four uploaded photographs. Correspondences and
# reconstruction come from Modules/Module_6/Part_B/sfm_pipeline.py, the same functions as the saved experiment.
B_PAGE_CSS = VISITOR_PAGE_CSS.replace("m6v_", "m6b_") + """<style>
.st-key-m6b_point_hint { min-height: 3.25rem; }
.st-key-m6b_point_hint p { margin: 0; }
.st-key-m6b_point_actions button, .st-key-m6b_confirm_row button { height: 2.75rem; min-height: 2.75rem; }
.st-key-m6b_counter p { margin: 0; }
[class*="st-key-m6b_upload_"] [data-testid="stFileUploaderDropzone"] button {
    font-size: 0; border: 2px solid #C9363E; min-height: 2.75rem; min-width: 12rem; }
[class*="st-key-m6b_upload_"] [data-testid="stFileUploaderDropzone"] button > * { display: none; }
[class*="st-key-m6b_upload_"] [data-testid="stFileUploaderDropzone"] button::after { font-size: 1rem; font-weight: 700; }
.st-key-m6b_results_nav { margin-top: 16px; }
.st-key-m6b_results_nav button { border: 2px solid #C9363E; width: 11rem; height: 2.75rem; min-height: 2.75rem; }
.st-key-m6b_results_nav button:hover, .st-key-m6b_results_nav button:focus-visible { border-color: #E04850; }
</style>"""
B_STEPS = {1: "Step 1 — Upload Your Images", 2: "Step 2 — Select Corresponding Points",
           3: "Step 3 — Enter Object Dimensions", 4: "Step 4 — Reconstruct Your Object", 5: "Step 5 — View Results"}
B_OVERVIEW_WIDTH = 400  # displayed width of the clickable photo, CSS pixels
B_ZOOM_HALF = 50  # zoom view: 100 x 100 working pixels around the point, shown 4x larger
B_ZOOM_WIDTH = 400
B_CORNERS = 4  # P1-P4, the only points selected before the automatic identification
B_EXAMPLE = (Path(__file__).resolve().parents[1] / "Modules" / "Module_6" / "Part_B" / "output" / "revised"
             / "part_b_selection_example.jpg")
B_IMAGE_KEYS = ("m6b_files", "m6b_infos", "m6b_images", "m6b_error")
B_POINT_KEYS = ("m6b_points", "m6b_status", "m6b_matches", "m6b_confirmed", "m6b_history", "m6b_view",
                "m6b_active", "m6b_focus", "m6b_find_error", "m6b_find_requested")
B_RESULT_KEYS = ("m6b_result", "m6b_result_sig", "m6b_run_error")
B_STATUS_TEXT = {"selected": "selected", "automatic": "automatic", "corrected": "corrected", None: "missing"}


def b_statuses(view, points):
    """Statuses of one photo's points as first shown after the automatic identification. A point without a
    position is missing; the photo 1 corners were selected by hand."""
    return [None if not np.isfinite(point).all() else "selected" if view == 0 and i < B_CORNERS else "automatic"
            for i, point in enumerate(points)]


def b_point_count():
    """P1-P4 until the corresponding points have been found, then P1-P8."""
    return B_CORNERS if st.session_state.get("m6b_matches") is None else b_sfm().POINT_COUNT


def b_sfm():
    from Modules.Module_6.Part_B import sfm_pipeline as sfm
    return sfm


def b_reset_results():
    for key in B_RESULT_KEYS:
        st.session_state.pop(key, None)


def b_reset_dimensions():
    st.session_state["m6b_dims"] = None
    b_reset_results()


def b_reset_points():
    """Empty P1-P8 in every photo; downstream dimensions confirmation and results are dropped."""
    sfm = b_sfm()
    for key in B_POINT_KEYS:
        st.session_state.pop(key, None)
    st.session_state["m6b_points"] = np.full((sfm.VIEW_COUNT, sfm.POINT_COUNT, 2), np.nan)
    st.session_state["m6b_status"] = [[None] * sfm.POINT_COUNT for _ in range(sfm.VIEW_COUNT)]
    st.session_state["m6b_confirmed"] = [False] * sfm.VIEW_COUNT
    st.session_state["m6b_history"] = []
    st.session_state["m6b_view"] = 0
    st.session_state["m6b_active"] = st.session_state["m6b_focus"] = 0
    for view in range(sfm.VIEW_COUNT):
        st.session_state.pop(f"m6b_pill_{view}", None)
    b_reset_dimensions()


def b_discard_images():
    for key in (*B_IMAGE_KEYS, "m6b_dims_typed", "m6b_width", "m6b_height", "m6b_view_choice", "m6b_clicks"):
        st.session_state.pop(key, None)
    st.session_state.pop("m6b_scroll", None)
    b_reset_points()


def b_upload_key():
    return f"m6b_upload_{st.session_state['m6b_upload_generation']}"


def b_scroll_to(step):
    """Request one smooth scroll to a step; consumed by b_scroll_script on the next completed run."""
    st.session_state["m6b_scroll"] = step


def b_set_photos(files, infos):
    """Replace the accepted photo set (upload order kept); everything derived from the old set is dropped."""
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    b_discard_images()
    st.session_state["m6b_files"] = files
    st.session_state["m6b_infos"] = infos
    if len(files) == vr.VIEW_COUNT:
        try:
            st.session_state["m6b_images"] = [vr.decode_image(data, name) for data, name in files]
        except vr.ReconstructionError as error:
            st.session_state["m6b_error"] = f"{error} Remove that photo to continue."
            return
        b_scroll_to(2)  # once, right after a complete set is accepted


def b_new_upload():
    """Uploader on_change: validate the new files, add them after the accepted ones and empty the uploader."""
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    uploaded = st.session_state.get(b_upload_key()) or []
    if not uploaded:
        return
    st.session_state["m6b_upload_generation"] += 1  # a fresh key empties the uploader for the next addition
    st.session_state.pop("m6b_error", None)
    files = list(st.session_state.get("m6b_files") or [])
    infos = list(st.session_state.get("m6b_infos") or [])
    free = vr.VIEW_COUNT - len(files)
    if len(uploaded) > free:
        st.session_state["m6b_error"] = (f"Exactly {vr.VIEW_COUNT} photographs are needed. You selected "
                                         f"{len(uploaded)}, but only {free} more can be added. None were added.")
        return
    if sum(len(data) for data, _ in files) + sum(f.size for f in uploaded) > vr.MAX_TOTAL_BYTES:
        st.session_state["m6b_error"] = (f"Total upload exceeds {vr.MAX_TOTAL_BYTES // 1024 // 1024} MB. "
                                         "The new photos were not added.")
        return
    errors = []
    for item in uploaded:
        data = item.getvalue()  # bytes only; the filename is never used as a path
        try:
            infos.append(vr.image_summary(data, item.name))
        except vr.ReconstructionError as error:
            errors.append(f"{error} It was not added.")
            continue
        files.append((data, item.name))
    if len(files) > vr.VIEW_COUNT - free:
        b_set_photos(files, infos)
    if errors:
        st.session_state["m6b_error"] = " ".join(errors)


def b_remove_photo(index):
    files, infos = list(st.session_state["m6b_files"]), list(st.session_state["m6b_infos"])
    del files[index], infos[index]
    b_set_photos(files, infos)


def b_start_over():
    st.session_state["m6b_upload_generation"] += 1  # a fresh uploader key is the supported way to empty it
    b_discard_images()
    b_scroll_to(1)


B_SCROLL_JS = VISITOR_SCROLL_JS.replace("m6v_step_", "m6b_step_")
B_SCROLL_HOST_CSS = VISITOR_SCROLL_HOST_CSS.replace("m6v_", "m6b_")


def b_scroll_script():
    """Emit the pending scroll request exactly once, at the end of the run so every step already exists."""
    step = st.session_state.pop("m6b_scroll", None)
    if step not in B_STEPS:
        return
    st.session_state["m6b_scroll_count"] = st.session_state.get("m6b_scroll_count", 0) + 1
    with st.container(key="m6b_scroll_host"):
        st.html(B_SCROLL_HOST_CSS)
        st.html(B_SCROLL_JS % (step, st.session_state["m6b_scroll_count"]), unsafe_allow_javascript=True)


def b_size_text(size_bytes):
    return f"{size_bytes / 1024 / 1024:.1f} MB" if size_bytes >= 1024 * 1024 else f"{max(1, round(size_bytes / 1024))} KB"


def b_upload_section(vr):
    with st.expander("Image Capture Instructions", expanded=False):
        st.markdown(
            "- **Object:** a stationary, approximately flat, rectangular object such as a book, box lid or "
            "poster. All four corners must be visible in every photo.\n"
            "- **Surface detail:** the object needs distinctive detail (letters, logos, pattern details) so "
            "that interior points can be found and checked.\n"
            "- **Viewpoints:** take four photos of the object from clearly different angles, without moving it. "
            "Photo 1 should show it most directly.\n"
            "- **Distance:** keep the whole object in frame and fairly large. Stepping back and zooming in works "
            "better than a close-up, because the method assumes a distant (orthographic) camera.\n"
            "- **Measure** the object's width and height in centimeters; you will need them in Step 3.\n"
            "- Your photos are processed in temporary storage and are not kept.")
    infos = st.session_state.get("m6b_infos") or []
    with st.container(key="m6b_counter"):
        st.markdown(f"**{len(infos)} of {vr.VIEW_COUNT} photos uploaded**")
    if len(infos) < vr.VIEW_COUNT:
        label = "+ Add More Photos" if infos else "+ Add Photos"
        st.html('<style>[class*="st-key-m6b_upload_"] [data-testid="stFileUploaderDropzone"] button::after '
                f'{{ content: "{label}"; }}</style>')
        st.file_uploader(
            label, type=[e.lstrip(".") for e in vr.ALLOWED_EXTENSIONS], accept_multiple_files=True,
            key=b_upload_key(), on_change=b_new_upload, max_upload_size=vr.MAX_FILE_BYTES // 1024 // 1024,
            label_visibility="collapsed")
    st.caption(f"Exactly {vr.VIEW_COUNT} photographs, added one at a time or together. Formats: JPG, PNG, "
               f"HEIC/HEIF. Up to {vr.MAX_FILE_BYTES // 1024 // 1024} MB per photo, "
               f"{vr.MAX_TOTAL_BYTES // 1024 // 1024} MB in total and {vr.MAX_PIXELS // 1_000_000} megapixels "
               "per photo.")
    if st.session_state.get("m6b_error"):
        st.error(st.session_state["m6b_error"])
    if not infos:
        return
    columns = st.columns(vr.VIEW_COUNT)
    for number, info in enumerate(infos):
        column = columns[number]
        column.image(info["thumbnail"], width="stretch")
        column.caption(f"**Photo {number + 1}**  \n{info['name']}  \n{info['width']} × {info['height']} px · "
                       f"{b_size_text(info['size_bytes'])}")
        column.button("Remove", key=f"m6b_remove_{number}", on_click=b_remove_photo, args=(number,),
                      width="stretch")
    if len(infos) == vr.VIEW_COUNT:
        st.caption("Photos are numbered in the order listed. Photo 1 is the reference photo.")


# ---------------------------------------------------------------- Step 2: corresponding points

def b_unconfirm(view):
    """A change to photo 1, the reference, requires every photo to be checked again."""
    confirmed = st.session_state["m6b_confirmed"]
    for other in range(len(confirmed)) if view == 0 else (view,):
        confirmed[other] = False


def b_record(view, index, point, status):
    """Set one point (original pixels), remember the previous value for Undo, invalidate dependent state."""
    points, statuses = st.session_state["m6b_points"], st.session_state["m6b_status"]
    st.session_state["m6b_history"].append((view, index, points[view, index].copy(), statuses[view][index]))
    points[view, index] = point
    statuses[view][index] = status
    b_unconfirm(view)
    b_reset_results()


def b_next_missing(view, start):
    points = st.session_state["m6b_points"][view][:b_point_count()]
    order = list(range(start + 1, len(points))) + list(range(start + 1))
    return next((i for i in order if not np.isfinite(points[i]).all()), None)


def b_set_focus(view, index):
    st.session_state["m6b_active"] = st.session_state["m6b_focus"] = index
    st.session_state[f"m6b_pill_{view}"] = index


def b_click(key, view, zoom):
    """Consume one real click on the photo (overview) or the zoom view; reruns are not clicks."""
    sfm = b_sfm()
    click = st.session_state.get(key)
    if not isinstance(click, dict):
        return
    stamp = click.get("unix_time")
    consumed = st.session_state.setdefault("m6b_clicks", {})
    if stamp is None or consumed.get(key) == stamp:
        return
    consumed[key] = stamp
    image = st.session_state["m6b_images"][view]
    try:
        if zoom:
            index = st.session_state["m6b_focus"]
            window = st.session_state.get("m6b_zoom_window")
            if window is None:
                return
            point = sfm.display_to_original(click["x"], click["y"], click["width"], click["height"], image,
                                            window[:2], window[2:])
        else:
            index = st.session_state["m6b_active"]
            point = sfm.display_to_original(click["x"], click["y"], click["width"], click["height"], image)
    except (KeyError, ValueError, TypeError, OverflowError):
        return
    b_record(view, index, point, "selected" if view == 0 else "corrected")
    if zoom:
        return
    st.session_state["m6b_focus"] = index  # the zoom view shows the point just placed
    following = b_next_missing(view, index)
    if following is not None:
        st.session_state["m6b_active"] = following
        st.session_state[f"m6b_pill_{view}"] = following
    else:
        st.session_state[f"m6b_pill_{view}"] = index


def b_pill_changed(view):
    index = st.session_state[f"m6b_pill_{view}"]
    st.session_state["m6b_active"] = st.session_state["m6b_focus"] = index


def b_select_view(view):
    st.session_state["m6b_view"] = view
    points = st.session_state["m6b_points"][view][:b_point_count()]
    # the first missing point, otherwise the first point to check (photo 1: the first automatic one)
    default = B_CORNERS if view == 0 and len(points) > B_CORNERS else 0
    b_set_focus(view, next((i for i in range(len(points)) if not np.isfinite(points[i]).all()), default))


def b_view_changed():
    b_select_view(int(st.session_state["m6b_view_choice"]))


def b_undo():
    history = st.session_state["m6b_history"]
    if not history:
        return
    view, index, point, status = history.pop()
    st.session_state["m6b_points"][view, index] = point
    st.session_state["m6b_status"][view][index] = status
    b_unconfirm(view)
    b_reset_results()
    st.session_state["m6b_view"] = view
    st.session_state["m6b_view_choice"] = view
    b_set_focus(view, index)


def b_reset_all_points():
    b_reset_points()
    b_scroll_to(2)


def b_restore_view(view):
    """Return one photo to the automatic points (photo 1 keeps the corners selected there)."""
    points = st.session_state["m6b_matches"].points[view]
    st.session_state["m6b_points"][view] = points
    st.session_state["m6b_status"][view] = b_statuses(view, points)
    st.session_state["m6b_history"] = [h for h in st.session_state["m6b_history"] if h[0] != view]
    b_unconfirm(view)
    b_reset_results()
    b_select_view(view)


def b_confirm_view(view):
    confirmed = st.session_state["m6b_confirmed"]
    confirmed[view] = True
    remaining = [v for v in range(len(confirmed)) if not confirmed[v]]
    if remaining:
        st.session_state["m6b_view_choice"] = remaining[0]
        b_select_view(remaining[0])
    else:
        b_scroll_to(3)


def b_overview(image, points, statuses, focus):
    """Photo shrunk to the display width with numbered markers (point coordinates in original pixels)."""
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    sfm = b_sfm()
    scale = B_OVERVIEW_WIDTH / image.size[0]
    height = max(1, round(image.size[1] * scale))
    small = cv2.resize(image.rgb, (B_OVERVIEW_WIDTH, height), interpolation=cv2.INTER_AREA)
    shown = (sfm.to_working(points, image) + 0.5) * scale - 0.5
    return vr.render_points(small, shown, statuses, active=focus, radius=5, font_scale=0.45)


def b_zoom(image, points, statuses, focus, center):
    """Magnified crop around `center` (original pixels); returns (rgb, window in working pixels)."""
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    sfm = b_sfm()
    working = sfm.to_working(points, image)
    window = vr.zoom_window(sfm.to_working(center, image), B_ZOOM_HALF, image.size)
    left, top, width, height = window
    crop = image.rgb[top:top + height, left:left + width]
    factor = B_ZOOM_WIDTH / width
    zoomed = cv2.resize(crop, (B_ZOOM_WIDTH, max(1, round(height * factor))), interpolation=cv2.INTER_CUBIC)
    shown = (working - [left, top] + 0.5) * factor - 0.5
    return vr.render_points(zoomed, shown, statuses, active=focus, outline=False, radius=9, font_scale=0.6), window


def b_request_find():
    st.session_state["m6b_find_requested"] = True


def b_find():
    """Find P1-P8 in all four photos from the photo 1 corners (shared pipeline)."""
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    sfm = b_sfm()
    st.session_state.pop("m6b_find_error", None)
    images = st.session_state["m6b_images"]
    try:
        with vr.job_slot(), st.spinner("Finding corresponding points…"):
            found = sfm.find_correspondences(images, st.session_state["m6b_points"][0, :B_CORNERS])
    except vr.ReconstructionError as error:
        st.session_state["m6b_find_error"] = str(error)
        return
    except (cv2.error, ValueError, MemoryError):
        logging.getLogger(__name__).exception("Module 6 Part B correspondence search failed")
        st.session_state["m6b_find_error"] = ("The corresponding points could not be found. Click the button "
                                              "again, or try different photos.")
        return
    st.session_state["m6b_matches"] = found
    for view in range(sfm.VIEW_COUNT):
        st.session_state["m6b_points"][view] = found.points[view]
        st.session_state["m6b_status"][view] = b_statuses(view, found.points[view])
        st.session_state["m6b_confirmed"][view] = False
        st.session_state.pop(f"m6b_pill_{view}", None)
    st.session_state["m6b_history"] = []
    st.session_state["m6b_view_choice"] = 0
    b_select_view(0)
    b_scroll_to(2)


def b_point_labels(statuses):
    """Pill labels; a point without a position is marked with a question mark."""
    return [f"P{i + 1}" + ("" if status is not None else " ?") for i, status in enumerate(statuses)]


def b_stage_points(vr):
    sfm = b_sfm()
    st.header(B_STEPS[2], anchor=False)
    st.write("Select the four corners of the object in Photo 1. Four additional feature points and their "
             "corresponding positions in Photos 2–4 will be identified automatically. Review and correct the "
             "points where necessary.")
    with st.expander("How to Select the Points", expanded=False):
        st.markdown(
            "- **P1–P4** are the object's four boundary corners, selected clockwise in Photo 1 starting at the "
            "**top-left** corner: top-left, top-right, bottom-right, bottom-left (for a book, with the cover "
            "upright).\n"
            "- **P5–P8** are interior feature points identified automatically.\n"
            "- Each name must mark the **same physical point** in all four photos, even when a photo is "
            "rotated.\n"
            "- Click the photo to place the highlighted point. Click the zoom view to position it precisely.\n"
            "- Check every photo. To correct a point, choose it and click its correct position. Points marked "
            "**?** must be selected by hand.")
        if B_EXAMPLE.is_file():
            st.image(str(B_EXAMPLE), width=360, caption="Example: corners P1–P4 selected on a book cover and "
                     "interior points P5–P8 identified automatically.")
    if st.session_state.pop("m6b_find_requested", False):
        b_find()  # before any point widget exists, so their state can be set
    images = st.session_state["m6b_images"]
    points, statuses = st.session_state["m6b_points"], st.session_state["m6b_status"]
    matches = st.session_state.get("m6b_matches")
    found = matches is not None
    count = b_point_count()
    if found:
        if matches.landmarks_found < sfm.POINT_COUNT - B_CORNERS:
            st.warning(f"{matches.landmarks_found} of the 4 interior points could be identified automatically. "
                       "Select each missing point (marked ?) on the same physical detail in all four photos.")
        st.session_state.setdefault("m6b_view_choice", st.session_state["m6b_view"])
        photo_labels = [f"Photo {v + 1}" + (" ✓" if done else "") for v, done in enumerate(st.session_state["m6b_confirmed"])]
        st.segmented_control("Photo", list(range(sfm.VIEW_COUNT)), key="m6b_view_choice", on_change=b_view_changed,
                             format_func=lambda v: photo_labels[v],
                             required=True, width="stretch")
    view = st.session_state["m6b_view"] if found else 0
    image = images[view]
    focus = st.session_state["m6b_focus"]
    active = st.session_state["m6b_active"]
    placed = int(np.isfinite(points[view][:count]).all(axis=1).sum())

    with st.container(key="m6b_point_hint"):
        if found and view > 0 and not matches.views[view - 1].reliable:
            st.warning(matches.views[view - 1].message)
        role = "interior point" if active >= B_CORNERS else sfm.POINT_ROLES[active].lower()
        if placed < count:
            kind = "points" if found else "corners"
            st.markdown(f"**Photo {view + 1}:** click **P{active + 1}** — {role}. ({placed} of {count} {kind} placed)")
        elif found:
            st.markdown(f"**Photo {view + 1}:** check all 8 points, correct any that are inaccurate, then confirm.")
        else:
            st.markdown("**Photo 1:** all 4 corners placed. Adjust any if needed, then find the corresponding points.")
    st.session_state.setdefault(f"m6b_pill_{view}", focus)
    point_labels = b_point_labels(statuses[view][:count])
    st.pills("Point", list(range(count)), key=f"m6b_pill_{view}", required=True,
             format_func=lambda i: point_labels[i], on_change=b_pill_changed, args=(view,),
             label_visibility="collapsed")

    overview_key, zoom_key = f"m6b_overview_{view}", f"m6b_zoom_{view}"
    left, right = st.columns(2)
    with left:
        st.caption(f"Photo {view + 1} — click to place P{active + 1}")
        streamlit_image_coordinates(b_overview(image, points[view], statuses[view], focus), width="content",
                                    cursor="crosshair", key=overview_key, image_format="JPEG", jpeg_quality=88,
                                    on_click=lambda: b_click(overview_key, view, False))
    with right:
        target = points[view, focus]
        if np.isfinite(target).all():
            zoomed, window = b_zoom(image, points[view], statuses[view], focus, target)
            st.session_state["m6b_zoom_window"] = window
            st.caption(f"Zoom on P{focus + 1} — click to position it precisely")
            streamlit_image_coordinates(zoomed, width="content", cursor="crosshair", key=zoom_key,
                                        image_format="JPEG", jpeg_quality=90,
                                        on_click=lambda: b_click(zoom_key, view, True))
            x, y = target
            st.caption(f"P{focus + 1}: ({x:.0f}, {y:.0f}) px in the original photo · "
                       f"{B_STATUS_TEXT.get(statuses[view][focus], 'missing')}")
        else:
            st.session_state["m6b_zoom_window"] = None
            st.info(f"P{focus + 1} has no position in photo {view + 1} yet. Click it on the photo.")
    st.caption(("Cyan: automatic · " if found else "") +
               "Yellow: selected or corrected by you · Red: point in the zoom view.")

    with st.container(key="m6b_point_actions", horizontal=True, gap="small"):
        st.button("Undo", key="m6b_undo", on_click=b_undo, disabled=not st.session_state["m6b_history"],
                  width="stretch")
        if found:
            st.button("Restore Automatic Points", key="m6b_restore", on_click=b_restore_view, args=(view,),
                      width="stretch")
        if view == 0:
            st.button("Reset All Points", key="m6b_reset_points", on_click=b_reset_all_points, width="stretch",
                      disabled=not found and placed == 0)
    problem = None
    if view == 0 and placed == count:
        try:
            if found:
                sfm.check_reference_points(points[0], image.original_size)
            else:
                sfm.check_corners(points[0, :B_CORNERS], image.original_size)
        except sfm.PointError as error:
            problem = str(error)
    if problem:
        st.warning(problem)
    if not found:
        with st.container(key="m6b_confirm_row"):
            st.button("Find Corresponding Points", key="m6b_find", type="primary", width="stretch",
                      on_click=b_request_find, disabled=placed < count or problem is not None)
        if st.session_state.get("m6b_find_error"):
            st.error(st.session_state["m6b_find_error"])
        return
    confirmed = st.session_state["m6b_confirmed"]
    with st.container(key="m6b_confirm_row"):
        if confirmed[view]:
            st.success(f"Photo {view + 1} points confirmed.")
        else:
            st.button(f"Confirm Photo {view + 1} Points", key="m6b_confirm", type="primary", width="stretch",
                      on_click=b_confirm_view, args=(view,), disabled=placed < count or problem is not None)
    remaining = [str(v + 1) for v in range(sfm.VIEW_COUNT) if not confirmed[v]]
    if remaining:
        st.caption("Still to confirm: photo " + ", ".join(remaining) + ".")


def b_points_ready():
    confirmed = st.session_state.get("m6b_confirmed")
    points = st.session_state.get("m6b_points")
    return (st.session_state.get("m6b_matches") is not None and confirmed is not None and all(confirmed)
            and points is not None and np.isfinite(points).all())


# ---------------------------------------------------------------- Step 3: dimensions

def b_dimensions_changed():
    st.session_state["m6b_dims_typed"] = (st.session_state.get("m6b_width"), st.session_state.get("m6b_height"))
    b_reset_dimensions()


def b_commit_dimensions():
    st.session_state["m6b_dims"] = (float(st.session_state["m6b_width"]), float(st.session_state["m6b_height"]))
    b_reset_results()
    b_scroll_to(4)


def b_stage_dimensions():
    st.header(B_STEPS[3], anchor=False)
    st.write("Enter the object's measured size. The width is the P1–P2 edge and the height the P1–P4 edge.")
    typed = st.session_state.get("m6b_dims_typed") or (None, None)
    for key, value in zip(("m6b_width", "m6b_height"), typed):
        if key not in st.session_state:  # widget state is dropped while Step 3 is hidden
            st.session_state[key] = value
    left, right = st.columns(2)
    width = left.number_input("Measured Width (cm)", min_value=0.0, step=0.1, format="%.2f",
                              key="m6b_width", on_change=b_dimensions_changed, placeholder="e.g. 14.5")
    height = right.number_input("Measured Height (cm)", min_value=0.0, step=0.1, format="%.2f",
                                key="m6b_height", on_change=b_dimensions_changed, placeholder="e.g. 21.7")
    valid = width is not None and height is not None and width > 0 and height > 0
    if (width is not None and width <= 0) or (height is not None and height <= 0):
        st.warning("Both dimensions must be greater than zero.")
    if st.session_state.get("m6b_dims") is None:
        st.button("Continue", key="m6b_dims_continue", type="primary", disabled=not valid, width="stretch",
                  on_click=b_commit_dimensions)


# ---------------------------------------------------------------- Step 4 and 5: reconstruction and results

def b_signature():
    points = st.session_state["m6b_points"]
    return hashlib.sha256(points.tobytes()).hexdigest(), st.session_state["m6b_dims"]


def b_stage_reconstruct():
    sfm = b_sfm()
    st.header(B_STEPS[4], anchor=False)
    width, height = st.session_state["m6b_dims"]
    st.markdown(f"**Points:** 8 in each of 4 photos (all confirmed)  \n**Measured size:** {width:g} × {height:g} cm")
    st.caption("The verified corresponding points and measured dimensions are used to reconstruct the object and "
               "estimate its boundary.")
    signature = b_signature()
    if st.session_state.get("m6b_result_sig") not in (None, signature):
        b_reset_results()
    if st.button("Reconstruct", key="m6b_run", type="primary", width="stretch"):
        b_reset_results()
        try:
            with st.spinner("Reconstructing…"):
                result = sfm.reconstruct(st.session_state["m6b_points"], width, height)
        except sfm.images.ReconstructionError as error:
            st.session_state["m6b_run_error"] = str(error)
        except (ValueError, np.linalg.LinAlgError, MemoryError):
            logging.getLogger(__name__).exception("Module 6 Part B visitor reconstruction failed")
            st.session_state["m6b_run_error"] = ("Reconstruction could not complete. Check that every point marks the "
                                                 "same physical location in all four photos.")
        else:
            st.session_state["m6b_result"] = result
            b_scroll_to(5)
        st.session_state["m6b_result_sig"] = signature  # keeps an error message until the inputs change
        st.rerun()
    if st.session_state.get("m6b_run_error"):
        st.error(st.session_state["m6b_run_error"])


def b_results(vr, result):
    st.header(B_STEPS[5], anchor=False)
    data = dict(cm=result.cm, width=result.width_cm, height=result.height_cm)
    st.subheader("Reconstructed Points", anchor=False)
    st.write("X and Y run along the object's width and height; Z is the depth relative to its aligned surface. "
             "P1–P4 give the boundary and P5–P8 are the landmarks.")
    matrix_table("Point", [f"P{i}" for i in range(1, 9)], ["X (cm)", "Y (cm)", "Z (cm)"], result.cm, 3)
    st.subheader("Estimated Object Boundary — Top View", anchor=False)
    part_b_plot(part_b_top_view_figure(data), "m6b_plot_top")
    st.caption(f"The dashed rectangle shows the measured size, {result.width_cm:g} × {result.height_cm:g} cm, for "
               "reference only. It was not fitted to the reconstructed points.")
    st.subheader("Scaled 3D Reconstruction", anchor=False)
    part_b_plot(part_b_3d_figure(data), "m6b_plot_3d")
    st.caption("Drag to rotate. All axes use the same scale. For a flat object, differences in Z mainly reflect "
               "reconstruction error rather than real depth.")
    st.subheader("Edge Lengths", anchor=False)
    edges = [m for m in result.measurements if not m["key"].startswith("diagonal")]
    simple_table("Measurement", {
        "Measurement": [m["label"] for m in edges],
        "Measured (cm)": [num(m["actual"], 3) for m in edges],
        "Reconstructed (cm)": [num(m["reconstructed"], 3) for m in edges],
        "Error (%)": [num(m["error"], 2) for m in edges]})
    st.caption("Your measured width and height set the single scale factor, so their agreement with the "
               "reconstructed edges is not an independent validation. The differences between opposite edges show "
               "the shape error.")
    st.subheader("Download Results", anchor=False)
    names = [name for _, name in st.session_state["m6b_files"]]
    st.download_button("Points and Edges (CSV)", vr.points_csv(result, names), "reconstruction_points.csv",
                       "text/csv", key="m6b_dl_csv", on_click="ignore", width="stretch")


def part_b_visitor_page():
    from Modules.Module_6.Part_B import visitor_reconstruction as vr

    st.session_state.setdefault("m6b_upload_generation", 0)
    st.session_state.setdefault("m6b_dims", None)
    if "m6b_points" not in st.session_state:
        b_reset_points()

    st.html(B_PAGE_CSS)
    with st.container(key="m6b_head"):
        with st.container(key="m6b_label"):
            st.markdown("MODULE 6 — PART B")
        st.title("Structure from Motion Reconstruction", anchor=False)
    # One continuous page: each step stays rendered once its prerequisites exist. The uploader runs first
    # because a removed or replaced photo resets everything that depends on it within this same run.
    current = 1
    with st.container(border=True, key="m6b_step_1"):
        st.header(B_STEPS[1], anchor=False)
        b_upload_section(vr)
    if st.session_state.get("m6b_images"):
        current = 2
        with st.container(border=True, key="m6b_step_2"):
            b_stage_points(vr)
        if b_points_ready():
            current = 3
            with st.container(border=True, key="m6b_step_3"):
                b_stage_dimensions()
            if st.session_state.get("m6b_dims") is not None:
                current = 4
                with st.container(border=True, key="m6b_step_4"):
                    b_stage_reconstruct()
                result = st.session_state.get("m6b_result")
                if result is not None and st.session_state.get("m6b_result_sig") == b_signature():
                    current = 5
                    with st.container(border=True, key="m6b_step_5"):
                        b_results(vr, result)
                        with st.container(key="m6b_results_nav", horizontal_alignment="center"):
                            st.button("Reset All", key="m6b_reset_all", on_click=b_start_over)
    st.html(f"<style>.st-key-m6b_step_{current} {{ border-color: {ACCENT} !important; }}</style>")  # current step
    b_scroll_script()


st.set_page_config(page_title="Module 6", layout="centered")
st.session_state.setdefault("module6_section", "home")
section = st.session_state["module6_section"]
# breadcrumb levels below Home for every view: (label, section); the last is the current page, the one before it is the Back target
_M6, _RESULTS = ("Module 6", "home"), ("My Experiments & Results", "my_results")
CRUMBS = {"home": (_M6,), "my_results": (_M6, _RESULTS),
          "a_visitor": (_M6, ("Optical Flow and Motion Tracking", "a_visitor")),
          "b_visitor": (_M6, ("Structure from Motion Reconstruction", "b_visitor")),
          "a_results": (_M6, _RESULTS, ("Optical Flow and Motion Tracking", "a_results")),
          "b_results": (_M6, _RESULTS, ("Structure from Motion Reconstruction", "b_results"))}
if section not in CRUMBS:  # e.g. a session left on a removed selection page
    select_part("home")
    section = "home"
nav_row(CRUMBS[section], select_part, page_css=LIGHT_TEXT_CSS, view=f"m6:{section}")
# Leaving the Part B visitor page discards its photographs and everything derived from them.
if st.session_state.get("m6_shown_section") == "b_visitor" and section != "b_visitor":
    b_discard_images()
st.session_state["m6_shown_section"] = section
if section == "home":
    home_layout(HOME_TITLE, (
        ("View My Experiments & Results", "m6_open_results", "my_results"),
        ("Optical Flow and Motion Tracking", "m6_open_a_visitor", "a_visitor"),
        ("Structure from Motion Reconstruction", "m6_open_b_visitor", "b_visitor")), subtitle=HOME_SUBTITLE)
elif section == "my_results":
    home_layout("My Experiments & Results", (
        (part_label("Optical Flow and Motion Tracking", "A"), "m6_res_a", "a_results"),
        (part_label("Structure from Motion Reconstruction", "B"), "m6_res_b", "b_results")),
        label="MODULE 6", extra_css=RESULTS_TITLE_CSS)
elif section == "a_visitor":
    part_a_visitor_page()
elif section == "a_results":
    show_part_a()
elif section == "b_visitor":
    part_b_visitor_page()
else:
    show_part_b()
