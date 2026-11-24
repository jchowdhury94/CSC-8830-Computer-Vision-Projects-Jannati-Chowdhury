# room_measurement – technical reference (Module 7, Question 3)

This package implements the room-measurement estimator of Question 3 (classical computer vision only): **Wall A** (width and height, two
viewpoints) and **Wall B** (length, three viewpoints). It is a short technical reference; the objective, the eleven-step methodology, the accepted
results and the limitations are in the [Module 7 README](../../README.md#5-question-3--room-measurement-using-stereo-videos).

**Base directory.** Relative paths below are relative to `Modules/Module_7/room_measurement_question_3/` (the *Question 3 directory*) unless
they start with `Modules/`. Commands are run from the repository root.

## Contents

| Path | Purpose |
|---|---|
| `features.py`, `pose.py`, `geometry.py`, `measure.py`, `lines.py`, `pipeline.py` | Wall A estimator: SIFT matching, shared stereo pose, triangulation and floor plane, boundary lines, corner inference and registration, end-to-end run. In the code the shared stereo-pose estimator is called "Method J". |
| `uncertainty.py` | Wall A uncertainty: quadrature of the frame-set/matching bootstrap, the pose-method difference, boundary perturbations and an assumed intrinsics perturbation; reported at ±1.96σ. |
| `three_view_wall_b.py`, `three_view_quality.py` | Wall B: per-viewpoint stereo pose and floor plane, registration of viewpoints 1 and 3 onto viewpoint 2, end-corner distance (the length), and quality classification of each frame set. |
| `calibration.py`, `frames.py` | Loading and validating the room-video calibration file; reading the private videos (hashing, manifest check, deterministic frame extraction by index). Videos are only read. |
| `validation.py` | The only code that knows the measured room size; the estimation code cannot import it (checked by a test). |
| `visitor_*.py` | Backends used by the standalone visitor application: video validation, frame selection, annotations, calibration adapter, measurement job, uncertainty. |
| `display_data.py` | Read-only loader for the website's results page (saved JSON, display images, playback videos). |
| `config/` | `frame_manifest.json` (Wall A videos with SHA-256 hashes, frame counts and the five frame sets) and `wall_boundary_annotations.json` (the author's boundary seeds). |
| `reference/` | `method_j_experiment_reference.json`, reference values from the original experiment used by regression tests. |
| `tools/` | Scripts that regenerate the display images, the playback videos and the exported calibration file from the private recordings. |

The visitor application is in `visitor_workflow/`, its upload and worker components in `f8_chunk_poc/`, and the boundary-annotation helper in
`annotation_ui/`; the calibration file is `calibration/room_video/room_video_recipe_a_calibration.json` and the saved results are in
`outputs/room_measurement/`.

## Wall A in brief

1. SIFT (up to 12,000 features) with Lowe's ratio test, 0.75 between the left and right camera and 0.85 between the two placements, and a
   mutual check; matches across placements link four-view tracks.
2. Shared stereo pose with the 225 mm baseline fixed: eight-point start, Tukey-robust Sampson epipolar cost over all stereo matches, then the
   four-view track reprojection cost. If the start is degenerate, a parallel-camera fallback is used. No measured dimension enters the fit.
3. Floor plane through reconstructed points in the lower image (robust, deterministic).
4. Boundary lines: manual seeds, then automatic sub-pixel edge search and robust line fits.
5. Bottom corner: intersection of the left- and right-camera planes of the vertical wall edge with the floor plane; top corner from the viewing ray
   through the ceiling intersection and the vertical edge.
6. Registration of placement 1 onto placement 2 (seeded RANSAC, 40 mm); width is the distance between the registered bottom corners, height the
   mean top height.

Manual inputs: frame selection, boundary seeds and windows, the 225 mm baseline and the choice of corners. Everything else is computed.
Coordinates are in millimetres in the left-camera frame (x right, y down, z forward).

## Wall B in brief

Three viewpoints (frames 60, 150 and 240 of six videos). Each viewpoint has its own stereo pose and floor plane; two points are marked along each
required vertical wall edge of viewpoints 1 and 3; viewpoints 1 and 3 are registered onto viewpoint 2 from cross-view matches (ratio 0.90
between neighbouring viewpoints); the length is the 3D distance between the two outer bottom corners. No Wall B confidence interval is computed.

## Private inputs

The original recordings are not in Git. To reproduce an experiment, place them under `data/room_measurement/`:

- Wall A: `calibration_videos/` containing `left_air_wall_left`, `right_pro_wall_left`, `left_air_wall_right` and `right_pro_wall_right`
  (checked against `config/frame_manifest.json` before every run).
- Wall B: `wall_b_three_viewpoints/` containing `left_air_2_{1,2,3}.MOV`, `right_pro_2_{1,2,3}.MOV` and the author's annotation file.

## Reproduction commands

```bash
# Wall A (--quick uses fewer uncertainty replicates). Existing results are not overwritten without --overwrite.
python Modules/Module_7/room_measurement_question_3/run_room_measurement.py --out DIR [--quick] [--skip-uncertainty]

# Wall B and the visitor production path, against the accepted values (opt-in, needs the private recordings)
M7_REAL_VIDEO_TESTS=1 python -m pytest Modules/Module_7/room_measurement_question_3/room_measurement/tests/test_three_view_real_data.py \
                                       Modules/Module_7/room_measurement_question_3/visitor_workflow/tests/test_f8_35_real_three_view.py

# Synthetic-data tests (no private data needed)
python -m pytest Modules/Module_7/room_measurement_question_3/room_measurement/tests tests/test_module7_room_measurement.py
```

`run_room_measurement.py` writes `estimation_result.json` first and only then compares with the measured size (`validation.json`); the final
dimensions are never hard-coded. The saved results shown on the website are not modified by these commands unless `--out` points at them with
`--overwrite`.
