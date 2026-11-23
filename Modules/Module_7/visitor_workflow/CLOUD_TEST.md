# Isolated Cloud test of the visitor room-measurement app

Nothing here changes the main app, `home.py`, `Pages/` or the root `requirements.txt`. Reconstruction stays **off**: `ROOM_PROCESSING_ENABLED = False` in `f8_visitor_service.py`.

| Setting | Value |
|---|---|
| Repository / branch | the repository and branch you choose when creating the app (nothing is pushed by this document) |
| Main file path | `Modules/Module_7/visitor_workflow/app_f8_integrated.py` |
| Python | 3.12 (Advanced settings) |
| Requirements | `Modules/Module_7/visitor_workflow/requirements.txt` (read before the root file; no ML packages) |
| Secrets (required) | `F8_PASSCODE = "<long random passcode, at least 8 characters>"`. Without it the app stays locked. |
| Optional environment | `F8_ENABLE_FULL_CALIBRATION=1` only to test calibration inside the app; otherwise use *Upload Saved Calibration*. |
| Theme | dark with the red accent (`#C9363E`), set by `Modules/Module_7/visitor_workflow/.streamlit/config.toml` (read from the main script's folder, so the root and the main app are unaffected) |
| Not needed | recordings, `outputs/`, `data/`, `.streamlit/` files, the experiment scripts |

## What can be tested with reconstruction off
1. Passcode gate, then **Upload Saved Calibration** (capture mode *video*). The validated calibration file
   `Modules/Module_7/calibration/room_video/room_video_recipe_a_calibration.json` is accepted with capture mode *video* (checked locally).
2. Room-video upload: real HEVC `.MOV` files, file/size/duration/frame-rate/resolution checks, the displayed requirements.
3. Frame extraction, frame selection and boundary annotation.
4. *Start over* / *Lock this session* (setup/review/ready pages, and *Session Options* on the measurement page) and the retention sweep.

The **Calculate** buttons are disabled with an explanation. Measurement needs an explicit, separately approved change of the constant.

## Workload (for the later reconstruction test)
* Wall A (two viewpoints): 4 videos, 5 frame sets (`room_frames.DEFAULT_SETS`), 4 full-resolution images per set.
* Wall B (three viewpoints): 6 videos, 3 frame sets by default (`three_view_workflow`: min 2, default 3, max 5), 6 images per set.
* The only Cloud figure for Wall B (about 2418 MiB worker peak) is for ONE synthetic frame set.
