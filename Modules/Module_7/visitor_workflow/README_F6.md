# Stage F6 — simplified classical visitor workflow (local prototype)

Local only. Not linked from `Pages/app_module7.py`, not deployed. **Classical computer vision only — no machine learning.**

Start: `python Modules/Module_7/run_stage_f6_visitor_app.py [port]` → <http://127.0.0.1:8630>

## What it reuses (unchanged)
| Step | Component |
|---|---|
| Calibration import | Stage A `visitor_calibration` (via `VisitorMeasurementJob.validate_calibration`) |
| Four-video validation | Stage B `visitor_video_validation` |
| Frame selection | Stage C `visitor_frame_selection` |
| Boundary suggestions | Stage F2 `stage_f2_boundary_detection.detect_set` (gradient / line search; OpenCV + NumPy) |
| Refinement | Stage D `visitor_annotations` (`lines.edge_points`, `lines.fit_line`) |
| Gate + measurement | Stage E2 `visitor_measurement` (identity, digests, confirmations, Phase 1 geometry, `top_variant=0`) |
| Manual correction | Stage F1 `annotation_ui/core.py` (clicks, zoom / pan mapping, refinement) |

## Calibration (Stage F7 wizard; the F6 import page is the alternative)
Calibration is a guided six-step wizard, one step at a time (`f7_calibration.py`, `f7_pages.py`): instructions, cameras and checkerboard, upload of the left and right calibration
videos, processing, quality assessment, accept or discard. It reuses the existing Module 7 solver (`stereo_measure.user_calibration`: `detect_board`, `resolve_pair_orderings`,
`calibrate_stereo`); the new code is video frame extraction, left/right time-offset estimation, board-still pose selection and the wizard rules. Nothing is accepted automatically:
FAILED can never be accepted, QUESTIONABLE needs an explicit acknowledgement. A visitor who already has a calibration file can import it instead (the Stage A contract or the saved
Phase 1 layout; the capture mode is declared by the visitor, never guessed).

## Review policy
F2's top candidate is a proposal. Each of the 6 boundaries in every frame set needs the visitor's explicit confirmation, bound to
set, boundary, source, candidate rank, annotation id + digest, refined-line digest, frame index and frame fingerprint. Changing the selection removes it;
`export_for_measurement` re-verifies each one. Stage D / E2 FAIL can never be confirmed; WARNING uses the exact Stage E2 `Confirmation`.

## Actual dimensions
Optional, entered after the estimate exists, converted to mm, used only for the comparison. They are never passed to the job. Percentage error is
`|estimate − actual| / actual × 100` (absolute); the signed difference is shown separately.
