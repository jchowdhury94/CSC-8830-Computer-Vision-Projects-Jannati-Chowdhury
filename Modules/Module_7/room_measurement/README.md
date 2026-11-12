# Module 7 Part 3 — Room (wall) measurement from two stereo video views

**Status: Phase 1 (Python implementation only).** No Streamlit interface yet.
This is an approximate, educational experiment — **not a precision room-measurement system.**

```
python Modules/Module_7/run_room_measurement.py            # full run incl. uncertainty (~6 min)
python Modules/Module_7/run_room_measurement.py --quick    # fewer uncertainty replicates
python -m pytest tests/test_module7_room_measurement.py    # synthetic tests + real-video regression
```

The estimate is written to `outputs/room_measurement/estimation_result.json` **first**; only afterwards is it compared with the
ground truth (`validation.json`). The final dimensions are never hard-coded anywhere.

## Camera setup and data collection
| | |
|---|---|
| Left camera | iPhone Air, 1× rear |
| Right camera | iPhone 17 Pro Max, 1× rear |
| Video | 2160 × 3840 (portrait), 30 fps |
| Physical baseline | 225 mm (fixed in the estimator) |
| Lens height | 640 mm (**validation only**) |

The rig was recorded at two placements (wall-left, wall-right), each by both phones. Four protected videos are used
(`left_air_wall_left`, `right_pro_wall_left`, `left_air_wall_right`, `right_pro_wall_right`); their SHA-256 hashes, frame counts and the
5 frame sets (frame indices) are in `config/frame_manifest.json` and verified before every run. The videos live in
`data/room_measurement/` and are only ever read. Note that the Air and Pro frames of a set are *not* synchronised, and AL frame 247 appears in
both set 0 and set 4: the five sets are correlated captures of one rig placement, not independent experiments.

## Calibration
`calibration/room_video/room_video_recipe_a_calibration.json` ("Recipe A"): intrinsics and 5-term distortion of both phones at 2160 × 3840,
stored at full float64 precision with provenance. They were fitted from checkerboard corners of the calibration videos (not the old photo
Model B calibration, which is not reused). The legacy stereo extrinsics of that fit are kept in the file for provenance only and are **never used**:
the rig pose for the wall is re-estimated from the wall videos themselves.

## Method (all in `room_measurement/`)
1. **Features** (`features.py`): SIFT (12 000 features), Lowe ratio 0.75 Air↔Pro, mutual check; Air↔Air cross-view matches (ratio 0.85) link the two
   placements into four-view tracks (AL, PL, AR, PR).
2. **Shared stereo pose — Method J** (`pose.py`): 3 rotation + 2 baseline-direction parameters, |T| = 225 mm.
   Start: IRLS 8-point essential matrix (no RANSAC/RNG). Stage A: Tukey-robust Sampson epipolar cost over all 18 600 matches of all frame sets
   (per-set equal weight, spatial de-clustering). Stage J adds rigid four-view track reprojection with a free rig motion per set.
   Solver: Levenberg–Marquardt on Tukey IRLS weights (c = 4.685, scale = max(1.4826·MAD, 0.15 px)); stopping when the step < 1e-9 (A) / 1e-8 (J)
   or the relative cost change < 1e-10 / 1e-9. **Degenerate start** (robust scale > 1 px, inliers < 85 %, bad cheirality, or no convergence) →
   restart from the generic *parallel-camera* fallback. No measured dimension or camera height enters the fit.
3. **Floor** (`geometry.py`, `measure.py`): triangulate each view; deterministic Tukey-IRLS plane through points in the lower image (row > 2600).
4. **Boundary lines** (`lines.py`): *manual* seeds (`config/wall_boundary_annotations.json`) → automatic sub-pixel edge search → robust line fits.
5. **Corner inference**: bottom corner = (Air-plane ∩ Pro-plane of the vertical wall edge) ∩ floor plane. Top corner = point on that vertical closest
   to the Pro ray through the image intersection of the corner line with the ceiling line. Height = mean of the two views.
6. **Two-view alignment**: seeded RANSAC rigid registration (40 mm) of view 1 onto view 2 from cross-view matches with 3D in both.
7. **Width** = distance between the two registered bottom corners; **height** = mean top height.
8. **Uncertainty** (`uncertainty.py`): quadrature of (a) match/track bootstrap of the whole estimator, (b) ½|A − J|, (c) ±2 px / ±3 px boundary
   perturbations, (d) random intrinsics perturbation (fx 10/5 px, principal point 4 px) through the whole estimator; reported at ±1.96 σ.

**Coordinates / units:** millimetres; per view the left (Air) camera frame (x right, y down, z forward); pixels x right, y down.

## Manual vs automatic
Manual: frame selection, boundary-line seeds and windows, carpet-edge windows, the 225 mm baseline, the choice of corners. Everything else is computed.

## Validation
`validation.py` is the only code that knows the ground truth (2900 × 2400 mm) and the lens height (640 mm); estimation code cannot import it
(tested). It reports signed/absolute/percentage error, whether the truth lies inside the interval, and a lens-height diagnostic.

## Limitations — read before quoting a number
* **The Air cannot see the ceiling corners**; top corners are inferred from the Pro rays only.
* **One bottom corner is obstructed**; bottom corners are inferred from the stereo-derived vertical line and the floor plane.
* **Some corner positions are geometrically inferred** (vertical-wall assumption, floor-plane assumption).
* **Camera intrinsics dominate the uncertainty** (about 107 mm 1σ of the 114 mm total in width, 94 of 98 mm in height), and the assumed intrinsics perturbation sizes are not
  measured. A different calibration fit moves the width by several percent.
* Relative stereo yaw is only weakly observable from epipolar geometry; 0.1° of yaw error ≈ 80 mm of width. A low epipolar residual does **not**
  prove a correct metric reconstruction.
* The small spread across frame sets (a few mm) is an artefact of sharing one pose and one set of annotations; it is not the measurement uncertainty.
* Not a precision system; no claim beyond the stated interval.
