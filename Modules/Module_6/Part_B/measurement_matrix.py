"""Step 3: approved CSV -> raw/centered measurements -> SVD/rank analysis.

No correspondence selection, motion/structure factors, or reconstruction.
Requires NumPy and Matplotlib; runs from any current directory.
"""
from pathlib import Path
import csv
import hashlib

import numpy as np

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
INPUT = OUTPUT / "part_b_correspondences.csv"
POINT_IDS = tuple(f"P{i}" for i in range(1, 9))
COORD_FIELDS = tuple(f"view{v}_{a}" for v in range(1, 5) for a in ("x", "y"))
ROW_LABELS = tuple(f"V{v}_{a}" for v in range(1, 5) for a in ("x", "y"))
CENTER_TOLERANCE = 1e-10  # Pixels; comfortably above float64 rounding noise.


def load_measurements(path=INPUT):
    """Validate exactly four views and eight ordered point records, without sorting."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_fields = {"point_id", "description", *COORD_FIELDS}
        if (reader.fieldnames is None or len(reader.fieldnames) != len(expected_fields)
                or set(reader.fieldnames) != expected_fields):
            raise ValueError("CSV must contain point_id, description, and exactly four x/y view pairs")
        records = list(reader)
    if len(records) != 8 or tuple(r["point_id"] for r in records) != POINT_IDS:
        raise ValueError("CSV must contain exactly P1 through P8 in that order")
    if any(None in r or any(value is None or not value.strip() for value in r.values())
           for r in records):
        raise ValueError("Missing or extra CSV values")
    try:
        matrix = np.array([[float(r[field]) for r in records] for field in COORD_FIELDS], dtype=np.float64)
    except ValueError as error:
        raise ValueError("All coordinate values must be numeric") from error
    if matrix.shape != (8, 8) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite 8 x 8 measurements")
    # Bounds of the previously verified full-resolution upright images.
    if np.any(matrix < 0) or np.any(matrix[::2] >= 4284) or np.any(matrix[1::2] >= 5712):
        raise ValueError("Coordinate outside upright image bounds")
    return matrix


def analyze(raw):
    centroids = raw.mean(axis=1, keepdims=True)
    centered = raw - centroids
    row_means = centered.mean(axis=1)
    if not np.allclose(row_means, 0.0, atol=CENTER_TOLERANCE, rtol=0.0):
        raise ArithmeticError("Centering failed the residual row mean check")
    U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
    squared = singular_values ** 2
    if squared.sum() == 0 or singular_values[2] == 0:
        raise ValueError("Degenerate measurements: requested singular-value ratios undefined")
    energy = squared / squared.sum()
    # Analysis only: reconstruct an 8 x 8 rank-3 measurement approximation.
    approximation = U[:, :3] @ np.diag(singular_values[:3]) @ Vt[:3, :]
    error = float(np.linalg.norm(centered - approximation, ord="fro"))
    rank_tolerance = float(singular_values[0] * max(centered.shape) * np.finfo(centered.dtype).eps)
    return dict(centroids=centroids.reshape(4, 2), centered=centered, row_means=row_means,
                singular_values=singular_values, energy=energy,
                cumulative_energy=np.cumsum(energy),
                numerical_rank=int(np.linalg.matrix_rank(centered)), rank_tolerance=rank_tolerance,
                sigma4_over_sigma3=float(singular_values[3] / singular_values[2]),
                sigma4_over_sigma1=float(singular_values[3] / singular_values[0]),
                first_three_energy=float(energy[:3].sum()), remaining_energy=float(energy[3:].sum()),
                frobenius_error=error,
                relative_error=error / float(np.linalg.norm(centered, ord="fro")))


def save_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        # Preserve float64 precision in saved matrices and statistics.
        for row in rows:
            writer.writerow([format(float(value), ".17g") if isinstance(value, (float, np.floating))
                             else value for value in row])


def matrix_text(matrix):
    lines = ["row        " + " ".join(f"{p:>14}" for p in POINT_IDS)]
    for label, row in zip(ROW_LABELS, matrix):
        lines.append(f"{label:<10} " + " ".join(f"{value:14.6f}" for value in row))
    return "\n".join(lines)


def report_text(raw, result, input_hash):
    r = result
    lines = ["Module 6 Part B — Step 3: measurement matrix and rank/SVD analysis",
             "Views F = 4; points P = 8; raw and centered W shape = (8, 8)",
             "Coordinates: full-resolution upright pixels; origin upper-left; +x right; +y down.",
             "Point order: " + " ".join(POINT_IDS), "Row order: " + " ".join(ROW_LABELS),
             "Ordering matches the lecture-style arrangement explicitly supplied in the Step 3 request.",
             "No course lecture files were available in the repository for independent verification.",
             f"Authoritative input: {INPUT.relative_to(BASE)}", f"Input SHA-256: {input_hash}",
             "Input CSV was read only; no point selection, refinement, or rematching.",
             "", "RAW W (pixels)", matrix_text(raw), "", "VIEW CENTROIDS (pixels)",
             "view          mean_x          mean_y"]
    lines.extend(f"{v:<6} {x:15.9f} {y:15.9f}" for v, (x, y) in enumerate(r["centroids"], 1))
    lines.extend(["", "CENTERED W (pixels)", matrix_text(r["centered"]), "", "CENTERED ROW MEANS (pixels)"])
    lines.extend(f"{label}: {mean:.17e}" for label, mean in zip(ROW_LABELS, r["row_means"]))
    lines.extend([f"Centering check: PASS; absolute tolerance = {CENTER_TOLERANCE:.1e} pixels; rtol = 0",
                  "No rounding was applied before SVD.", "", "ALL SINGULAR VALUES (descending)",
                  "index singular_value         normalized_to_largest  energy_fraction         cumulative_energy_fraction"])
    for i, (s, e, c) in enumerate(zip(r["singular_values"], r["energy"], r["cumulative_energy"]), 1):
        lines.append(f"{i:<5} {s:.17e} {s/r['singular_values'][0]:.17e} {e:.17e} {c:.17e}")
    lines.extend(["", f"NumPy numerical matrix rank: {r['numerical_rank']}",
                  f"Default rank tolerance: {r['rank_tolerance']:.17e}",
                  f"sigma4 / sigma3: {r['sigma4_over_sigma3']:.17g}",
                  f"sigma4 / sigma1: {r['sigma4_over_sigma1']:.17g}",
                  f"First-three squared-singular-value energy fraction: {r['first_three_energy']:.17g}",
                  f"First-three energy percentage: {100*r['first_three_energy']:.12f}%",
                  f"Remaining energy fraction: {r['remaining_energy']:.17g}",
                  f"Remaining energy percentage: {100*r['remaining_energy']:.12f}%",
                  f"Rank-3 Frobenius approximation error (pixels): {r['frobenius_error']:.17g}",
                  f"Relative rank-3 Frobenius error: {r['relative_error']:.17g}",
                  f"Relative error percentage: {100*r['relative_error']:.12f}%", "", "INTERPRETATION",
                  "The ideal orthographic/affine model predicts centered W = M S with rank <= 3.",
                  f"The observed NumPy numerical rank is {r['numerical_rank']}, so this data is not exactly rank 3.",
                  "The first three components capture most measured energy and the relative rank-3",
                  "error is small, making rank 3 a reasonable approximate measurement model for",
                  "exploratory later analysis. However, sigma4 is a substantial fraction of sigma3,",
                  "so the departure from the third component is appreciable; energy alone does",
                  "not establish the accuracy of an eventual reconstruction.",
                  "Perspective projection, manual localization, image noise, lens/camera effects,",
                  "and small cover/edge deviations from the affine/rigid model can contribute.",
                  "These diagnostics do not determine each source's individual contribution.",
                  "Centering guarantees W_centered @ ones(8) = 0, so its rank cannot exceed 7.",
                  "The eighth singular value is therefore numerical roundoff, not evidence for rank 3.",
                  "The cover points are largely coplanar; ideal planar affine data has rank <= 2.",
                  "A good rank-3 approximation does not by itself demonstrate recovered 3D depth.",
                  "", "SCOPE",
                  "W3 was computed only to measure approximation error; it was not saved as reconstruction.",
                  "No motion/structure factors, metric upgrade, camera poses, or 3D points calculated.",
                  "No physical cover dimensions used. No commit or push."])
    return "\n".join(lines) + "\n"


def plot_singular_values(values):
    # Agg produces a standalone report artifact without requiring a GUI.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    indices = np.arange(1, 9)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax in axes:
        ax.plot(indices, values, "o-", color="#174b85", linewidth=1.8, markersize=6)
        ax.axvline(3.5, color="#b54a25", linestyle="--", label="Theoretical rank-3 cutoff")
        ax.set_xticks(indices)
        ax.set_xlabel("Singular-value index")
        ax.grid(True, alpha=.25)
    axes[0].set_ylabel("Singular value (pixel units)")
    axes[0].set_title("All eight singular values — linear scale")
    axes[0].legend(fontsize=9)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Singular value (pixel units, logarithmic scale)")
    axes[1].set_title("Same values — logarithmic scale")
    axes[1].annotate(f"sigma8 = {values[-1]:.2e}\ncentering null direction / roundoff",
                     xy=(8, values[-1]), xytext=(3.8, 1e-8), fontsize=8,
                     arrowprops=dict(arrowstyle="->", color="gray"))
    fig.suptitle("Module 6 Part B: centered measurement matrix SVD")
    fig.savefig(OUTPUT / "part_b_singular_values.png", dpi=200)
    plt.close(fig)


def main():
    input_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    raw = load_measurements()
    result = analyze(raw)
    OUTPUT.mkdir(exist_ok=True)
    save_csv(OUTPUT / "part_b_measurement_matrix_raw.csv", ("row", *POINT_IDS),
             ([label, *row] for label, row in zip(ROW_LABELS, raw)))
    save_csv(OUTPUT / "part_b_measurement_matrix_centered.csv", ("row", *POINT_IDS),
             ([label, *row] for label, row in zip(ROW_LABELS, result["centered"])))
    save_csv(OUTPUT / "part_b_view_centroids.csv", ("view", "mean_x", "mean_y"),
             ([v, *xy] for v, xy in enumerate(result["centroids"], 1)))
    save_csv(OUTPUT / "part_b_singular_values.csv",
             ("index", "singular_value", "normalized_to_largest", "energy_fraction", "cumulative_energy_fraction"),
             ([i, s, s/result["singular_values"][0], e, c] for i, (s, e, c) in
              enumerate(zip(result["singular_values"], result["energy"], result["cumulative_energy"]), 1)))
    plot_singular_values(result["singular_values"])
    if hashlib.sha256(INPUT.read_bytes()).hexdigest() != input_hash:
        raise RuntimeError("Approved CSV changed during analysis")
    report = report_text(raw, result, input_hash)
    (OUTPUT / "part_b_step3_matrix_analysis.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
