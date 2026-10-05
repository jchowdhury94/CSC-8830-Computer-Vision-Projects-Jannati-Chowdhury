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
        show_part_b()
