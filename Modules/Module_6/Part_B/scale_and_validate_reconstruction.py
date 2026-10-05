"""Step 5: rigid book alignment, one isotropic physical scale, honest validation.

Reads approved Step 4 points. No SfM, rectification, flattening, or point changes.
NumPy handles calculations; Matplotlib creates standalone report figures.
"""
from pathlib import Path
import csv
import hashlib
import re
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"
IDS = tuple(f"P{i}" for i in range(1, 9))
TRUE_WIDTH, TRUE_HEIGHT = 14.5, 21.7
CREATED = {
    "part_b_points_book_frame_unscaled.csv", "part_b_reconstructed_points_cm.csv",
    "part_b_physical_validation.csv", "part_b_plane_displacement_cm.csv",
    "part_b_interior_landmark_checks.csv", "part_b_book_frame_transform.csv",
    "part_b_reconstruction_scaled_cm.png", "part_b_reconstruction_top_view_cm.png",
    "part_b_depth_error_cm.png", "part_b_step5_physical_validation_report.txt",
    "part_b_step5_scope_verification.json",
}


def load_points():
    with (OUT / "part_b_reconstructed_points_unscaled.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["point_id", "X", "Y", "Z"]:
            raise ValueError("Expected point_id,X,Y,Z columns")
        rows = list(reader)
    if len(rows) != 8 or tuple(r["point_id"] for r in rows) != IDS:
        raise ValueError("Expected exactly P1–P8 in the approved order")
    if any(None in r or any(v is None or not v.strip() for v in r.values()) for r in rows):
        raise ValueError("Missing or extra coordinate fields")
    points = np.array([[float(r[a]) for a in ("X", "Y", "Z")] for r in rows])
    if not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite")
    return points


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length <= np.finfo(float).eps:
        raise ValueError("Degenerate frame direction")
    return vector / length


def align(points):
    centroid = points.mean(axis=0)
    _, spreads, vt = np.linalg.svd(points-centroid, full_matrices=False)
    normal = unit(vt[-1])
    if normal[np.argmax(np.abs(normal))] < 0:
        normal *= -1
    distances = (points-centroid) @ normal
    # Independently verify Step 4 normal and distances; a normal sign flip is equivalent.
    report = (OUT / "part_b_step4_factorization_report.txt").read_text()
    match = re.search(r"Plane normal \(sign arbitrary\):\s*\[([^\]]+)\]", report)
    if not match:
        raise ValueError("Step 4 plane normal unavailable for verification")
    old_normal = np.fromstring(match.group(1), sep=" ")
    with (OUT / "part_b_planarity_analysis.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if tuple(r["point_id"] for r in rows) != IDS:
        raise ValueError("Step 4 plane-distance IDs disagree")
    old_distances = np.array([float(r["signed_plane_distance"]) for r in rows])
    sign = 1 if normal @ old_normal >= 0 else -1
    if not (np.allclose(normal, sign*old_normal, atol=1e-10, rtol=0)
            and np.allclose(distances, sign*old_distances, atol=1e-9, rtol=1e-12)):
        raise ArithmeticError("Best-fit plane disagrees with Step 4")
    horizontal = unit((points[1]-points[0]) + (points[2]-points[3]))
    x = unit(horizontal - (horizontal @ normal)*normal)
    z = normal.copy()
    y = unit(np.cross(z, x))
    vertical = (points[3]-points[0]) + (points[2]-points[1])
    if y @ vertical < 0:
        y, z = -y, -z
    basis = np.vstack([x, y, z])
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(basis), 1., atol=1e-12):
        raise ArithmeticError("Frame is not orthonormal and right handed")
    origin = points[:4].mean(axis=0)
    book = (points-origin) @ basis.T
    # The prescribed corner centroid is generally NOT on the all-point fitted plane.
    plane_z = float((centroid-origin) @ z)
    plane_distances = (points-centroid) @ z
    if not np.allclose(book[:, 2]-plane_z, plane_distances, atol=1e-10):
        raise ArithmeticError("Plane/origin offset verification failed")
    return dict(centroid=centroid, normal=normal, basis=basis, origin=origin, book=book,
                plane_z=plane_z, distances=plane_distances, rms_spreads=spreads/np.sqrt(8),
                normal_step4_sign=sign, direction_dots=(x@horizontal, y@vertical))


def measure(points):
    pairs = ((0, 1), (3, 2), (0, 3), (1, 2), (0, 2), (1, 3))
    lengths = [float(np.linalg.norm(points[b]-points[a])) for a, b in pairs]
    top, bottom, left, right, diagonal1, diagonal2 = lengths
    return dict(top_width=top, bottom_width=bottom, mean_width=(top+bottom)/2,
                left_height=left, right_height=right, mean_height=(left+right)/2,
                diagonal_P1_P3=diagonal1, diagonal_P2_P4=diagonal2,
                mean_diagonal=(diagonal1+diagonal2)/2)


def scale_factor(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    return (width*TRUE_WIDTH + height*TRUE_HEIGHT)/(width**2 + height**2)


def save(name, header, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([format(float(v), ".17g") if isinstance(v, (float, np.floating)) else v for v in row])


def fmt(values):
    return np.array2string(np.asarray(values), precision=12, max_line_width=160)


def figures(cm, plane_z, distances):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    boundary = cm[[0, 1, 2, 3, 0]]
    fig = plt.figure(figsize=(8, 7), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cm[:4].T, color="#174b85", s=45, label="Reconstructed cover corners")
    ax.scatter(*cm[4:].T, color="#b54a25", s=40, label="Printed landmarks")
    ax.plot(*boundary.T, color="#174b85")
    for label, point in zip(IDS, cm):
        label_point = point.copy()
        if label == "P7":
            label_point += [-.8, -.5, 1.5]
            ax.plot(*np.vstack([point, label_point]).T, color="gray", linewidth=.7)
        ax.text(*label_point, " " + label, fontsize=10)
    midpoint = (cm.min(axis=0)+cm.max(axis=0))/2
    radius = float(np.ptp(cm, axis=0).max())*.55
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), midpoint):
        setter(center-radius, center+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("Book X (cm)"); ax.set_ylabel("Book Y (cm)"); ax.set_zlabel("Book Z (cm)")
    ax.set_title("Scaled SfM Reconstruction of Book Cover\nSingle isotropic scale; dimensions in cm")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=25, azim=-60)
    fig.savefig(OUT / "part_b_reconstruction_scaled_cm.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 8), layout="constrained")
    ax.plot(boundary[:, 0], boundary[:, 1], "o-", color="#174b85", label="Reconstructed cover boundary")
    ax.scatter(cm[4:, 0], cm[4:, 1], color="#b54a25", label="Printed landmarks")
    for label, (x, y, _) in zip(IDS, cm):
        offset = (8, -12) if label == "P7" else (5, 6)
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points")
    ax.add_patch(Rectangle((-TRUE_WIDTH/2, -TRUE_HEIGHT/2), TRUE_WIDTH, TRUE_HEIGHT,
                           fill=False, edgecolor="gray", linestyle="--", label="Ideal cover rectangle — reference only"))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xlabel("Book X (cm; left to right)"); ax.set_ylabel("Book Y (cm; top to bottom)")
    ax.set_title("Book Cover: Top-down Physical Validation")
    ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.savefig(OUT / "part_b_reconstruction_top_view_cm.png", dpi=200)
    plt.close(fig)
    # Distinguish origin-based Z from true plane distance rather than conflating them.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained", sharex=True)
    indices = np.arange(8)
    axes[0].bar(indices, cm[:, 2], color="#174b85")
    axes[0].axhline(plane_z, color="#b54a25", linestyle="--", label=f"Fitted plane at Z = {plane_z:.4f} cm")
    axes[0].axhline(0, color="gray", linewidth=.8)
    axes[0].set_ylabel("Book-frame Z (cm)")
    axes[0].set_title("Corner-centroid origin: signed Z coordinates")
    axes[0].legend()
    axes[1].bar(indices, distances, color="#b54a25")
    axes[1].axhline(0, color="black", linewidth=1, label="Fitted cover plane (zero distance)")
    axes[1].set_ylabel("Signed plane distance (cm)")
    axes[1].set_title("Actual out-of-plane displacement")
    axes[1].set_xticks(indices, IDS); axes[1].set_xlabel("Point ID"); axes[1].legend()
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.suptitle("Scaled Reconstruction: Depth / Planarity Error")
    fig.savefig(OUT / "part_b_depth_error_cm.png", dpi=200)
    plt.close(fig)


def main():
    root = BASE.parents[2]
    protected = [p for p in BASE.rglob("*") if p.is_file() and p.name not in CREATED]
    protected += [p for p in (root / "Modules/Module_6/Part_A").rglob("*") if p.is_file()]
    protected += [root / "Pages/app_module6.py", root / "home.py"]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    points = load_points()
    frame = align(points)
    raw = measure(frame["book"])
    width, height = raw["mean_width"], raw["mean_height"]
    s = scale_factor(width, height)
    cm = s*frame["book"]
    scaled = measure(cm)
    plane_z = s*frame["plane_z"]
    distances = s*frame["distances"]
    rms = float(np.sqrt(np.mean(distances**2)))
    maximum = float(np.max(np.abs(distances)))
    ideal_diagonal = float(np.hypot(TRUE_WIDTH, TRUE_HEIGHT))
    save("part_b_points_book_frame_unscaled.csv", ("point_id", "X_book", "Y_book", "Z_book"), ([p, *v] for p, v in zip(IDS, frame["book"])))
    save("part_b_reconstructed_points_cm.csv", ("point_id", "X_cm", "Y_cm", "Z_cm"), ([p, *v] for p, v in zip(IDS, cm)))
    save("part_b_book_frame_transform.csv", ("component", "X", "Y", "Z"),
         [["origin_unscaled", *frame["origin"]], ["plane_centroid_unscaled", *frame["centroid"]],
          *[[a, *v] for a, v in zip(("book_x_axis", "book_y_axis", "book_z_axis"), frame["basis"])],
          ["global_scale_cm_per_unit", s, "", ""], ["fitted_plane_Z_cm", plane_z, "", ""]])
    validation = []
    for name, value in scaled.items():
        reference = TRUE_WIDTH if "width" in name else TRUE_HEIGHT if "height" in name else ideal_diagonal
        error = abs(value-reference)
        validation.append([name, value, reference, error, 100*error/reference])
    validation += [["opposite_width_discrepancy", abs(scaled["top_width"]-scaled["bottom_width"]), "", "", ""],
                   ["opposite_height_discrepancy", abs(scaled["left_height"]-scaled["right_height"]), "", "", ""],
                   ["RMS_plane_displacement", rms, "", "", ""], ["max_plane_displacement", maximum, "", "", ""]]
    save("part_b_physical_validation.csv", ("measurement", "reconstructed_cm", "known_or_reference_cm", "absolute_error_cm", "percentage_error"), validation)
    save("part_b_plane_displacement_cm.csv", ("point_id", "Z_cm", "absolute_Z_cm", "signed_plane_distance_cm", "absolute_plane_distance_cm"),
         ([p, z, abs(z), d, abs(d)] for p, z, d in zip(IDS, cm[:, 2], distances)))
    low, high = cm[:4, :2].min(axis=0), cm[:4, :2].max(axis=0)
    interior = []
    for p, point, distance in zip(IDS[4:], cm[4:], distances[4:]):
        inside = bool(np.all(point[:2] >= low) and np.all(point[:2] <= high))
        interior.append([p, *point, distance, inside])
    save("part_b_interior_landmark_checks.csv", ("point_id", "X_cm", "Y_cm", "Z_cm", "signed_plane_distance_cm", "inside_corner_XY_extent"), interior)
    figures(cm, plane_z, distances)
    lines = ["Module 6 Part B — Step 5: physical alignment, scaling, and validation",
             "Input: output/part_b_reconstructed_points_unscaled.csv (approved Step 4, unchanged).",
             f"Known front-cover dimensions: width={TRUE_WIDTH} cm; height={TRUE_HEIGHT} cm.",
             "Best-fit plane normal in original reconstruction frame: " + fmt(frame["normal"]),
             "Plane agrees with Step 4 within sign/tolerance: PASS.",
             "Book axes are ROWS of basis B (original reconstruction coordinates):\n" + fmt(frame["basis"]),
             "Axis norms: " + fmt(np.linalg.norm(frame["basis"], axis=1)),
             "Basis Gram matrix (norms squared on diagonal; pairwise dot products off diagonal):\n" + fmt(frame["basis"] @ frame["basis"].T),
             f"Rotation determinant: {np.linalg.det(frame['basis']):.17g}",
             "Corner-centroid origin, arbitrary units: " + fmt(frame["origin"]),
             "Book coordinates = (original point - corner centroid) @ B.T.",
             "X points approximately left-to-right; Y top-to-bottom; Z completes a right-handed frame.",
             "No point projection/flattening: only the horizontal axis is projected into the plane.",
             "", "UNSCALED 3D COVER DISTANCES (arbitrary reconstruction units)"]
    lines += [f"{name}: {value:.17g}" for name, value in raw.items()]
    lines += [f"Reconstructed aspect ratio: {width/height:.17g}", f"Physical aspect ratio: {TRUE_WIDTH/TRUE_HEIGHT:.17g}",
              "", "ONE ISOTROPIC SCALE",
              f"Width-only scale (cm/unit): {TRUE_WIDTH/width:.17g}", f"Height-only scale (cm/unit): {TRUE_HEIGHT/height:.17g}",
              "s = (mean_width*true_width + mean_height*true_height)/(mean_width^2 + mean_height^2)",
              f"Chosen joint least-squares scale (cm/unit): {s:.17g}",
              "The same positive scale applies to X,Y,Z, preserving shape without stretching an axis.",
              "", "SCALED SfM RECONSTRUCTION COORDINATES (cm; not ground-truth 3D coordinates)"]
    lines += [f"{p}: " + fmt(v) for p, v in zip(IDS, cm)]
    lines += ["", "PHYSICAL VALIDATION: measurement, reconstructed cm, reference cm, absolute error cm, percentage error"]
    lines += [", ".join(format(x, ".12g") if isinstance(x, (float, np.floating)) else str(x) for x in row) for row in validation]
    lines += [f"Ideal rectangle diagonal (validation only): {ideal_diagonal:.17g}",
              "", "ORIGIN / FITTED-PLANE OFFSET",
              "The mandated four-corner centroid generally lies off the all-point fitted plane.",
              f"Fitted plane location in book frame: Z_book = {frame['plane_z']:.17g} arbitrary units.",
              f"Fitted plane location after scaling: Z_cm = {plane_z:.17g} cm.",
              "Thus Z_cm is the normal coordinate relative to the corner-centroid origin; it is NOT",
              "the signed distance to the fitted plane. Signed plane distance = Z_cm - fitted_plane_Z_cm.",
              "The corner-centroid origin is preserved; no hidden shift or flattening is applied.",
              "point_id, Z_cm, abs(Z_cm), signed_plane_distance_cm, abs(plane_distance_cm)"]
    lines += [f"{p}, {z:.12f}, {abs(z):.12f}, {d:.12f}, {abs(d):.12f}" for p, z, d in zip(IDS, cm[:, 2], distances)]
    lines += [f"RMS actual plane displacement (cm): {rms:.17g}", f"Maximum actual absolute plane displacement (cm): {maximum:.17g}",
              f"Normalized planarity error, RMS / known height (%): {100*rms/TRUE_HEIGHT:.17g}",
              f"RMS book-frame Z coordinate (distinct origin-based statistic, cm): {np.sqrt(np.mean(cm[:,2]**2)):.17g}",
              "", "INTERIOR LANDMARKS (broad corner XY bounding-box check, not a rectangle constraint)",
              "Corner XY lower extent (cm): " + fmt(low), "Corner XY upper extent (cm): " + fmt(high)]
    lines += [f"{row[0]}: coordinates={fmt(row[1:4])}, plane distance={row[4]:.12f} cm; inside broad extent={row[5]}" for row in interior]
    lines += ["", "INTERPRETATION",
              "One isotropic scale converts arbitrary SfM units to approximate cm and preserves all",
              "recovered geometry. This provides a quantitative reconstruction and dimension comparison.",
              "Width and height were used to determine scale; their agreement is not fully independent",
              "validation. The joint scale forces neither mean width nor mean height to match exactly.",
              "Diagonal and opposite-edge differences were used only for validation, not fitting.",
              "Perspective iPhone images are approximated by affine/orthographic factorization;",
              "correspondences were manually localized and all landmarks are on one nearly planar surface.",
              "Recovered depth is weak and should not be read as true cover relief. Apparent nonplanarity",
              "and boundary distortion remain visible; this is not a calibrated metric 3D scan.",
              "No new SfM, point adjustment, independent axis scaling, flattening, rectangle forcing,",
              "or homography rectification was performed."]
    if any(not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise RuntimeError("A protected preexisting file changed")
    lines += [f"Protected file hash checks: PASS ({len(hashes)} files).", "No commit, push, Part A, or Streamlit edits. STOP after Step 5."]
    report = "\n".join(lines) + "\n"
    (OUT / "part_b_step5_physical_validation_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
