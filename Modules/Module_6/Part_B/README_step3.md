# Step 3 — measurement matrix and SVD analysis

`measurement_matrix.py` reads the approved `output/part_b_correspondences.csv`
directly. It validates P1–P8 in order, four view coordinate pairs, numeric finite
values, and image bounds. Rows are V1_x, V1_y, V2_x, V2_y, V3_x, V3_y, V4_x,
V4_y; columns are P1–P8. This follows the lecture-style convention supplied in
the Step 3 request. No lecture files were available locally to verify separately.

NumPy performs the calculations; Matplotlib produces the PNG plot. No Pandas
or new root dependency file is needed. Run with a Python environment containing
NumPy and Matplotlib:

```sh
python Modules/Module_6/Part_B/measurement_matrix.py
```

For this session, the repository environment lacked Matplotlib. Plotting
packages were downloaded into a temporary directory, leaving the existing
environment unchanged. The command used from the repository root was:

```sh
PYTHONPATH=/private/tmp/module6-step3-plot-deps \
MPLCONFIGDIR=/private/tmp/module6-step3-matplotlib \
.venv/bin/python Modules/Module_6/Part_B/measurement_matrix.py
```

The temporary package directory may eventually be removed by the system; use
an environment containing Matplotlib for future runs.

The script writes raw/centered matrix CSVs, centroids, all eight singular values,
the singular-value plot, and a detailed text report under `output/`. CSV floats
use 17 significant digits; computations use unrounded float64 data. Centered
row means are checked with absolute tolerance 1e-10 pixels. Matrix rank uses
NumPy's default tolerance, also reported numerically. Singular-value energy
uses squared values. The plot shows unchanged values on linear and logarithmic
axes, with the rank-3 cutoff between indices 3 and 4.

The best rank-3 measurement approximation is computed solely to measure its
Frobenius residual, not to calculate motion or structure factors. Centering
creates a null direction, so the eighth singular value is near roundoff and
the maximum possible rank here is 7. The rank-3 energy fraction and residual
support an approximation, not exact affine geometry or a validated 3D result.
The largely planar cover also warrants caution about interpreting depth.

No physical dimensions, correspondence changes, reconstruction, metric upgrade,
camera pose estimation, or 3D visualization are included.
