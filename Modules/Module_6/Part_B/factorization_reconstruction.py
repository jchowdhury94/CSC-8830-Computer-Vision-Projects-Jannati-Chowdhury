"""Step 4: rank-3 factorization, explicit orthographic metric upgrade, unscaled 3D.

Only reads approved Step 3 input; only writes new Step 4 artifacts. No physical
dimensions, correspondence refinement, calibrated camera poses, or optimization.
"""
from pathlib import Path
import csv
import hashlib
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
POINTS = tuple(f"P{i}" for i in range(1, 9))
ROWS = tuple(f"V{v}_{a}" for v in range(1, 5) for a in ("x", "y"))
AXES = ("X", "Y", "Z")
ENTRIES = ("L11", "L12", "L13", "L22", "L23", "L33")


def read_matrix():
    with (OUT / "part_b_measurement_matrix_centered.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != 9 or rows[0] != ["row", *POINTS]:
        raise ValueError("Expected 8 x 8 matrix with P1–P8 columns")
    if any(len(row) != 9 for row in rows[1:]) or tuple(r[0] for r in rows[1:]) != ROWS:
        raise ValueError("Expected ordered V1_x, V1_y, ..., V4_y rows")
    matrix = np.array([[float(x) for x in row[1:]] for row in rows[1:]], dtype=np.float64)
    if matrix.shape != (8, 8) or not np.isfinite(matrix).all():
        raise ValueError("Invalid matrix shape or nonfinite values")
    return matrix


def save_csv(name, header, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([format(float(x), ".17g") if isinstance(x, (float, np.floating)) else x for x in row])


def save_matrix(name, matrix, row_labels, columns):
    save_csv(name, ("row", *columns), ([label, *row] for label, row in zip(row_labels, matrix)))


def symmetric_coefficients(a, b):
    """Coefficients of a L b^T in order L11,L12,L13,L22,L23,L33."""
    return [a[0]*b[0], a[0]*b[1] + a[1]*b[0], a[0]*b[2] + a[2]*b[0],
            a[1]*b[1], a[1]*b[2] + a[2]*b[1], a[2]*b[2]]


def solve_metric(initial_motion):
    coefficients, target, labels = [], [], []
    for v in range(4):
        a, b = initial_motion[2*v:2*v+2]
        coefficients.extend([symmetric_coefficients(a, a), symmetric_coefficients(b, b),
                             symmetric_coefficients(a, b)])
        target.extend([1., 1., 0.])
        labels.extend([f"V{v+1}_x_unit", f"V{v+1}_y_unit", f"V{v+1}_xy_orthogonal"])
    A, rhs = np.array(coefficients), np.array(target)
    entries, _, rank, singular_values = np.linalg.lstsq(A, rhs, rcond=None)
    L = np.array([[entries[0], entries[1], entries[2]],
                  [entries[1], entries[3], entries[4]],
                  [entries[2], entries[4], entries[5]]])
    residual = A @ entries - rhs
    eigenvalues = np.linalg.eigvalsh(L)
    # Scale-aware float64 tolerance; never clip eigenvalues or force a solution.
    tolerance = 100 * np.finfo(float).eps * np.linalg.norm(L, ord=2)
    return dict(A=A, rhs=rhs, labels=labels, L=L, residual=residual, rank=int(rank),
                condition=float(np.linalg.cond(A)), constraint_singular_values=singular_values,
                eigenvalues=eigenvalues, tolerance=tolerance,
                positive_definite=bool(np.min(eigenvalues) > tolerance),
                symmetric=bool(np.array_equal(L, L.T)))


def fit_plane(structure):
    points = structure.T
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, spread_singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    # Deterministic sign choice for the normal; sign has no geometric meaning.
    normal = vectors[-1].copy()
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    e1 = vectors[0].copy()
    e2 = np.cross(normal, e1)  # Right-handed orthonormal plane visualization basis.
    basis = np.vstack([e1, e2, normal])
    aligned = centered @ basis.T
    distances = centered @ normal
    return dict(centroid=centroid, normal=normal, singular_values=spread_singular_values,
                rms_spreads=spread_singular_values / np.sqrt(len(points)),
                distances=distances, rms_distance=float(np.sqrt(np.mean(distances**2))),
                max_distance=float(np.max(np.abs(distances))), basis=basis, aligned=aligned)


def plot_structure(points, title, filename, axis_names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*points[:4].T, color="#174b85", s=45, label="Front-cover corners")
    ax.scatter(*points[4:].T, color="#b54a25", s=40, label="Printed landmarks")
    boundary = points[[0, 1, 2, 3, 0]]
    ax.plot(*boundary.T, color="#174b85", linewidth=1.5)
    for name, point in zip(POINTS, points):
        label_point = point.copy()
        if name == "P7":
            # P7 is close to P2 in both views; offset its label with a leader.
            label_point += np.array([-180., -100., 350.])
            ax.plot(*np.vstack([point, label_point]).T, color="gray", linewidth=.7)
        ax.text(*label_point, " " + name, fontsize=10)
    # Equal limits and equal box aspect preserve geometry, including plane displacement.
    middle = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * .55, 1.)
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), middle):
        setter(center-radius, center+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(axis_names[0] + " (arbitrary units)")
    ax.set_ylabel(axis_names[1] + " (arbitrary units)")
    ax.set_zlabel(axis_names[2] + " (arbitrary units)")
    ax.set_title(title)
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc="upper left")
    fig.savefig(OUT / filename, dpi=200)
    plt.close(fig)


def format_array(matrix):
    return np.array2string(matrix, precision=12, suppress_small=False, max_line_width=150)


def main():
    # Protect all earlier Part B artifacts, originals, Part A, and Streamlit.
    repository = BASE.parents[2]
    protected = [p for p in BASE.rglob("*") if p.is_file()]
    protected += [p for p in (repository / "Modules/Module_6/Part_A").rglob("*") if p.is_file()]
    protected += [repository / "Pages/app_module6.py", repository / "home.py"]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    # Existing Step 4 outputs can be regenerated; all prior-step files remain protected.
    prior = {p: h for p, h in before.items() if not p.name.startswith("part_b_step4_")
             and p.name not in OUTPUT_NAMES}
    W = read_matrix()
    U, singular_values, Vt = np.linalg.svd(W, full_matrices=False)
    with (OUT / "part_b_singular_values.csv").open(newline="") as f:
        previous = np.array([float(r["singular_value"]) for r in csv.DictReader(f)])
    if previous.shape != (8,) or not np.allclose(singular_values, previous, rtol=1e-12, atol=1e-10):
        raise ArithmeticError("SVD disagrees with Step 3")
    U3, Sigma3, V3t = U[:, :3], np.diag(singular_values[:3]), Vt[:3, :]
    W3 = U3 @ Sigma3 @ V3t
    root = np.diag(np.sqrt(singular_values[:3]))
    M_hat, S_hat = U3 @ root, root @ V3t
    approximate_error = np.linalg.norm(W-W3, ord="fro")
    relative_error = approximate_error / np.linalg.norm(W, ord="fro")
    initial_error = np.linalg.norm(M_hat @ S_hat-W3, ord="fro")
    ambiguity_transform = np.array([[1.2, .1, 0.], [0., .8, .1], [0., 0., 1.1]])
    ambiguity_error = np.linalg.norm((M_hat @ ambiguity_transform) @
                                    np.linalg.solve(ambiguity_transform, S_hat)-W3, ord="fro")
    if not np.allclose(M_hat @ S_hat, W3, rtol=1e-12, atol=1e-10):
        raise ArithmeticError("Initial factorization verification failed")
    if not np.isclose(relative_error, np.linalg.norm(previous[3:])/np.linalg.norm(previous), rtol=1e-12):
        raise ArithmeticError("Approximation error disagrees with Step 3")
    save_matrix("part_b_measurement_matrix_rank3.csv", W3, ROWS, POINTS)
    save_matrix("part_b_motion_initial.csv", M_hat, ROWS, ("c1", "c2", "c3"))
    save_matrix("part_b_structure_initial.csv", S_hat, ("s1", "s2", "s3"), POINTS)
    metric = solve_metric(M_hat)
    save_csv("part_b_metric_constraints.csv", ("constraint", *ENTRIES, "rhs", "achieved", "residual"),
             ([label, *row, target, target+res, res] for label, row, target, res in
              zip(metric["labels"], metric["A"], metric["rhs"], metric["residual"])))
    save_matrix("part_b_metric_L.csv", metric["L"], ("c1", "c2", "c3"), ("c1", "c2", "c3"))
    residual_norm = np.linalg.norm(metric["residual"])
    lines = ["Module 6 Part B — Step 4: factorization and unscaled reconstruction",
             "W shape: (8, 8); row order: " + ", ".join(ROWS), "Point order: " + ", ".join(POINTS),
             "Input: approved centered CSV, used without modification or re-centering.",
             "Coordinate convention: full-resolution upright pixels, origin upper-left, +x right, +y down.",
             "All recomputed singular values:\n" + format_array(singular_values),
             "SVD agrees with Step 3: PASS (rtol=1e-12, atol=1e-10).",
             "Retained singular values:\n" + format_array(singular_values[:3]),
             f"W3 Frobenius approximation error: {approximate_error:.17g}",
             f"W3 relative approximation error: {relative_error:.17g}",
             "M_hat shape: (8, 3); S_hat shape: (3, 8)",
             f"Initial ||M_hat S_hat - W3||_F: {initial_error:.17g}",
             "", "AFFINE FACTORIZATION AMBIGUITY",
             "For any invertible 3 x 3 Q: M_hat S_hat = (M_hat Q)(Q^-1 S_hat).",
             "Initial factors are therefore not unique Euclidean camera motion and structure.",
             "The metric upgrade chooses a transformation consistent approximately with unit, orthogonal",
             "camera rows. Global orthogonal rotations/reflections remain ambiguous.",
             "Numerical demonstration with an arbitrary invertible transformation T:\n" + format_array(ambiguity_transform),
             f"||(M_hat T) solve(T, S_hat) - W3||_F: {ambiguity_error:.17g}",
             "", "EXPLICIT METRIC CONSTRAINTS",
             "Unknown order: L11,L12,L13,L22,L23,L33; L = Q Q^T.",
             "Per view: a L a^T = 1, b L b^T = 1, a L b^T = 0.",
             "r L r^T coefficients: [r1^2,2r1r2,2r1r3,r2^2,2r2r3,r3^2].",
             "a L b^T coefficients: [a1b1,a1b2+a2b1,a1b3+a3b1,a2b2,a2b3+a3b2,a3b3].",
             f"Constraint A shape: {metric['A'].shape}; rhs shape: {metric['rhs'].shape}",
             "A:\n" + format_array(metric["A"]), "rhs:\n" + format_array(metric["rhs"]),
             f"Constraint system rank: {metric['rank']}", f"Condition number (2-norm): {metric['condition']:.17g}",
             f"Least-squares residual norm ||A l - rhs||_2: {residual_norm:.17g}",
             f"Least-squares sum of squared residuals: {residual_norm**2:.17g}",
             f"Least-squares RMS constraint residual: {residual_norm/np.sqrt(12):.17g}",
             "Constraint residual vector:\n" + format_array(metric["residual"]),
             "L:\n" + format_array(metric["L"]), "L eigenvalues:\n" + format_array(metric["eigenvalues"]),
             f"L exactly symmetric: {metric['symmetric']}", f"Positive-definiteness tolerance: {metric['tolerance']:.17g}",
             f"L positive definite above tolerance: {metric['positive_definite']}"]
    if not metric["positive_definite"]:
        lines += ["STOP: L is not reliably positive definite. No eigenvalue correction was applied.",
                  "Q, metric factors, 3D points and plots were not produced."]
    else:
        Q = np.linalg.cholesky(metric["L"])  # NumPy returns lower triangular Q: L = Q Q.T.
        q_error = np.linalg.norm(Q @ Q.T-metric["L"], ord="fro")
        if not np.allclose(Q @ Q.T, metric["L"], rtol=1e-12, atol=metric["tolerance"]):
            raise ArithmeticError("Cholesky verification failed")
        M = M_hat @ Q
        S = np.linalg.solve(Q, S_hat)
        metric_product_error = np.linalg.norm(M @ S-W3, ord="fro")
        if not np.allclose(M @ S, W3, rtol=1e-12, atol=1e-10):
            raise ArithmeticError("Metric factorization product verification failed")
        save_matrix("part_b_metric_Q.csv", Q, ("c1", "c2", "c3"), AXES)
        save_matrix("part_b_motion_metric.csv", M, ROWS, AXES)
        save_matrix("part_b_structure_metric_unscaled.csv", S, AXES, POINTS)
        save_csv("part_b_reconstructed_points_unscaled.csv", ("point_id", *AXES),
                 ([p, *xyz] for p, xyz in zip(POINTS, S.T)))
        camera_rows = []
        for v in range(4):
            i, j = M[2*v:2*v+2]
            k = np.cross(i, j)
            k /= np.linalg.norm(k)
            camera_rows.append([v+1, np.linalg.norm(i), np.linalg.norm(j), i @ j, *k])
        save_csv("part_b_camera_row_checks.csv", ("view", "norm_i", "norm_j", "i_dot_j", "k_X", "k_Y", "k_Z"), camera_rows)
        plane = fit_plane(S)
        save_csv("part_b_planarity_analysis.csv", ("point_id", "signed_plane_distance", "in_plane_1", "in_plane_2", "normal_displacement"),
                 ([p, d, *xyz] for p, d, xyz in zip(POINTS, plane["distances"], plane["aligned"])))
        reprojection = W-M @ S
        error = np.linalg.norm(reprojection, ord="fro")
        view_rms = [np.sqrt(np.mean(np.sum(reprojection[2*v:2*v+2]**2, axis=0))) for v in range(4)]
        save_csv("part_b_matrix_fit_residuals.csv", ("view", "rms_2d_point_residual_pixels"),
                 ([v, e] for v, e in enumerate(view_rms, 1)))
        plot_structure(S.T, "Unscaled Rank-3 SfM Reconstruction", "part_b_reconstruction_unscaled.png", AXES)
        plot_structure(plane["aligned"], "Unscaled Rank-3 SfM Reconstruction — Best-fit-plane View",
                       "part_b_reconstruction_plane_aligned.png", ("In-plane 1", "In-plane 2", "Plane normal"))
        lines += ["No correction to L or its eigenvalues was used.", "", "METRIC UPGRADE",
                  "NumPy Cholesky returns lower-triangular Q satisfying L = Q Q^T.",
                  "Q:\n" + format_array(Q), f"||Q Q^T - L||_F: {q_error:.17g}",
                  "M_metric = M_hat Q; S_metric = solve(Q, S_hat).",
                  "M_metric shape: (8, 3); S_metric shape: (3, 8)",
                  "M_metric:\n" + format_array(M), "S_metric:\n" + format_array(S),
                  f"||M_metric S_metric - W3||_F: {metric_product_error:.17g}",
                  "", "CAMERA ROW CHECKS: view, ||i||, ||j||, i dot j, normalized cross-product k",
                  format_array(np.array(camera_rows)),
                  "k is for interpretation only; these are not exact calibrated perspective camera poses.",
                  "", "UNSCALED POINTS: arbitrary reconstruction units; columns X,Y,Z"]
        lines.extend(f"{p}: " + format_array(xyz) for p, xyz in zip(POINTS, S.T))
        lines += ["", "BEST-FIT PLANE (no raw-axis depth assumption)",
                  "3D centroid: " + format_array(plane["centroid"]),
                  "Plane normal (sign arbitrary): " + format_array(plane["normal"]),
                  "Plane equation: normal dot (point - centroid) = 0.",
                  "Centered 3D singular values, descending: " + format_array(plane["singular_values"]),
                  "RMS spreads (in-plane 1, in-plane 2, normal): " + format_array(plane["rms_spreads"])]
        lines.extend(f"{p} signed perpendicular distance: {d:.17g}" for p, d in zip(POINTS, plane["distances"]))
        lines += [f"RMS plane distance: {plane['rms_distance']:.17g}",
                  f"Maximum absolute plane distance: {plane['max_distance']:.17g}",
                  f"Normal RMS spread / in-plane RMS spreads: {format_array(plane['rms_spreads'][2]/plane['rms_spreads'][:2])}",
                  "The reconstructed nonplanarity is appreciable; it is not evidence that the real cover",
                  "has this thickness. Model mismatch and weak planar depth recovery can cause apparent depth.",
                  "Plane-aligned basis (visualization only):\n" + format_array(plane["basis"]),
                  "Saved metric coordinates were not translated, rotated, or forced into a rectangle.",
                  "", "MATRIX FIT / CENTERED REPROJECTION",
                  f"||W_centered - M_metric S_metric||_F: {error:.17g}",
                  f"Relative reconstruction error: {error/np.linalg.norm(W):.17g}",
                  f"Consistent with Step 3 rank-3 error: {bool(np.isclose(error, approximate_error, rtol=1e-12))}",
                  "Per-view RMS 2D point residual (sqrt(mean(dx^2+dy^2)), pixels): " + format_array(np.array(view_rms))]
    lines += ["", "LIMITATIONS",
              "These are perspective iPhone photographs; Tomasi-Kanade rank-3 factorization assumes",
              "an affine/orthographic-style camera model. The original centered matrix had numerical",
              f"rank {np.linalg.matrix_rank(W)}; rank 3 captured {100*np.sum(singular_values[:3]**2)/np.sum(singular_values**2):.12f}% of squared energy.",
              "Correspondences were manually estimated. All eight landmarks lie on an approximately",
              "planar cover, making depth recovery weak/degenerate compared with noncoplanar points.",
              "A visually plausible planar reconstruction does not prove exact 3D recovery.",
              "Least-squares camera rows need not be exactly orthonormal; do not interpret them as",
              "calibrated perspective camera rotations. No row re-normalization was used to conceal residuals.",
              "All 3D units are arbitrary. No known physical dimensions were used, no physical scaling",
              "or corner rectification was performed, and no bundle adjustment was performed."]
    if any(not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in prior.items()):
        raise RuntimeError("A protected preexisting file changed")
    lines += ["", f"Protected preexisting file hashes unchanged: PASS ({len(prior)} files checked).",
              "Part A, Streamlit, original JPEGs, correspondences, and Step 3 results were read only.",
              "No commit or push. STOP after Step 4."]
    report = "\n".join(lines) + "\n"
    (OUT / "part_b_step4_factorization_report.txt").write_text(report, encoding="utf-8")
    print(report)


OUTPUT_NAMES = {"part_b_measurement_matrix_rank3.csv", "part_b_motion_initial.csv", "part_b_structure_initial.csv",
                "part_b_metric_constraints.csv", "part_b_metric_L.csv", "part_b_metric_Q.csv", "part_b_motion_metric.csv",
                "part_b_structure_metric_unscaled.csv", "part_b_reconstructed_points_unscaled.csv",
                "part_b_camera_row_checks.csv", "part_b_planarity_analysis.csv", "part_b_matrix_fit_residuals.csv",
                "part_b_reconstruction_unscaled.png", "part_b_reconstruction_plane_aligned.png"}

if __name__ == "__main__":
    main()
