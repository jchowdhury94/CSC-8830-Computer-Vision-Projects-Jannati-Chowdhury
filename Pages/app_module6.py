"""Module 6 navigation and saved Part A experimental evidence."""

from pathlib import Path

import streamlit as st


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


def select_part(part):
    st.session_state["module6_part"] = part


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


st.set_page_config(page_title="Module 6", layout="centered")
st.title("Module 6")
st.session_state.setdefault("module6_part", "home")
if st.session_state["module6_part"] == "home":
    for column, part, description in zip(st.columns(2), ("A", "B"),
                                         ("Optical Flow and Motion Tracking", "Structure from Motion")):
        with column:
            st.button(f"Part {part}", key=f"module6_open_{part}", on_click=select_part, args=(part,))
            st.caption(description)
else:
    st.button("Module 6 Home", on_click=select_part, args=("home",))
    if st.session_state["module6_part"] == "A":
        show_part_a()
    else:
        st.header("Part B — Structure from Motion")
        st.write("This section will contain the Structure from Motion experiment.")
