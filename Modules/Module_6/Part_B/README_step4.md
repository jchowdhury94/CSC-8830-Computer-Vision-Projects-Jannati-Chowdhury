# Step 4 — unscaled factorization and metric upgrade

Run `factorization_reconstruction.py` with NumPy and Matplotlib. The script
reads the approved Step 3 centered matrix and singular-value CSV directly,
preserving their contents and point/row ordering. It does not re-center or
reselect measurements. Using the temporary plotting environment from Step 3:

```sh
PYTHONPATH=/private/tmp/module6-step3-plot-deps \
MPLCONFIGDIR=/private/tmp/module6-step3-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python Modules/Module_6/Part_B/factorization_reconstruction.py
```

Alternatively use an environment with NumPy and Matplotlib already installed.
The temporary package directory may eventually be removed by the system.
No Pandas or SciPy is required.

The script retains three SVD components, forms the square-root initial factors,
and builds the 12 × 6 symmetric metric-constraint system explicitly. The
unknown order is L11,L12,L13,L22,L23,L33; constraints per view are the two
squared row norms equal to one and their inner product equal to zero.
Least squares solves for L. Its eigenvalues must exceed a scale-aware float64
tolerance before proceeding; otherwise execution reports the diagnostics and
stops without generating new metric reconstruction outputs. Eigenvalues are
never clipped. When L is valid, NumPy's lower-triangular Cholesky factor Q
satisfies L = Q Q^T. Motion is upgraded with M_hat Q and structure with
solve(Q, S_hat), preserving their product.

The output constraint CSV includes the full linear system, target values,
achieved values, and signed residuals. Separate outputs record initial and
metric factors, L, Q, rank-3 W, unscaled XYZ points, camera-row diagnostics,
planarity, and per-view 2D matrix-fit residuals. All CSV floats retain 17
significant digits. The text report includes a numerical demonstration of the
invertible-transform factorization ambiguity and all relevant diagnostics.

The best-fit plane is obtained by SVD of centered reconstructed 3D points,
not by assuming Z is physical depth. Its normal has a deterministic but
arbitrary sign. RMS principal spreads are singular values divided by sqrt(8).
The plane-aligned view changes only the visualization coordinates. Both plots
use equal axis limits and box aspect and connect P1–P2–P3–P4–P1. The metric
coordinates are not forced into a rectangle or scaled to physical dimensions.

Per-view 2D residual is sqrt(mean(dx² + dy²)) over the eight points, in pixels.
Camera row norms and inner products retain their least-squares discrepancies;
normalized cross-product directions are for interpretation only, not calibrated
perspective poses. Significant apparent cover nonplanarity and imperfect row
norms highlight model mismatch and weak depth recovery from planar landmarks.

The photographs are perspective images; the rank-3 affine/orthographic model
is approximate. The input matrix had numerical rank 7, with rank 3 capturing
99.918174594472% of squared energy. Manually estimated, approximately planar
landmarks give weak/degenerate depth recovery. A plausible plot alone does not
prove accurate 3D reconstruction. No physical dimensions are used in Step 4.
