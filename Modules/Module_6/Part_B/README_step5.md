# Step 5 — book-frame alignment, physical scale, and validation

`scale_and_validate_reconstruction.py` reads the approved Step 4 point CSV;
there is no new reconstruction or point fitting. It refits the all-point plane
and verifies its normal/distances against the Step 4 report and planarity CSV,
allowing equivalent normal signs.

Run with NumPy and Matplotlib. Using the existing temporary plotting packages:

```sh
PYTHONPATH=/private/tmp/module6-step3-plot-deps \
MPLCONFIGDIR=/private/tmp/module6-step3-matplotlib \
PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python Modules/Module_6/Part_B/scale_and_validate_reconstruction.py
```

Alternatively use a Python environment containing NumPy and Matplotlib; the
temporary package directory may eventually be removed by the system. No new
dependencies were installed for Step 5, and Pandas is not required.

The coordinate origin is the centroid of P1–P4. The X direction uses the sum
of the top and bottom edges projected into the fitted plane. Y and Z are
chosen to give top-to-bottom Y and an orthonormal right-handed frame. Only the
axis direction is projected: no reconstructed point is flattened. The saved
transformation CSV records the origin, plane centroid, basis axes, single
scale, and fitted-plane offset.

Full 3D edge lengths determine mean width and height. The one joint scale
minimizes `(s*mean_width - 14.5)^2 + (s*mean_height - 21.7)^2`. The same scale
applies to X, Y, and Z. Individual edges, diagonals, and interior landmarks do
not change the scale. The centroid-origin outputs retain the original shape.
No additional P1-origin presentation table is produced.

## Distinguishing Z coordinates from true fitted-plane distances

The four-corner centroid is off the plane fitted to all eight points. These
two requested conventions therefore cannot both put the plane at Z=0. The
mandated corner-centroid origin is retained. The fitted plane is at
`Z_cm = 0.602387594027 cm`, so:

```text
signed_plane_distance_cm = Z_cm - 0.602387594027
```

`part_b_plane_displacement_cm.csv` records both the book-frame Z coordinate
and the actual signed plane distance, plus their absolute values. The actual
plane distances determine RMS/max planarity error. The depth figure displays
both conventions separately, with the proper fitted-plane reference line in
each panel. Treating raw corner-origin Z as plane distance would give an
incorrect planarity statistic.

The physical validation CSV uses blank references for planarity and opposite
edge discrepancies. Planarity is assessed against the recovered fitted plane,
without asserting exact ground-truth relief. The interior-landmark check is a
broad bounding-box check against the four reconstructed corner XY coordinates.
It does not force rectangular bounds or clip points.

The top view overlays a centered, axis-aligned ideal rectangle for reference
only. The 3D figure uses equal axis limits and box aspect; top-down Y increases
downward. Both retain boundary distortion and nonzero recovered depth.

Small errors in mean width/height are not independent validation because those
dimensions set the scale. The opposite-edge discrepancy and planarity error
remain substantial. Perspective imagery, affine/orthographic approximation,
manual localization, and planar landmarks limit depth reliability. These are
scaled SfM reconstruction coordinates, not a calibrated metric 3D scan or
exact ground-truth points.
