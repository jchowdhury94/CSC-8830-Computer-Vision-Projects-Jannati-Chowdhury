# Isolated Cloud test of the visitor room-measurement app

Nothing here changes the main app, `home.py`, `Pages/` or the root `requirements.txt`. On this test branch **Wall A (two viewpoints: room width and height) calculation is on**: `WALL_A_PROCESSING_ENABLED = True` in `f8_visitor_service.py`. **Wall B (three viewpoints: room length) calculation is also on for this controlled test**: `WALL_B_PROCESSING_ENABLED = True`. The two walls are gated independently; the global `ROOM_PROCESSING_ENABLED` stays `False` and applies only to a call without a wall. All three are code constants, not settings. Opening a gate does not bypass any validation or resource-admission check: Wall B may still be refused (memory budget) or stopped (300 s timeout, 1800 CPU-second limit, memory guard) on this server, and the exact reason is shown.

| Setting | Value |
|---|---|
| Repository / branch | the repository and branch you choose when creating the app (nothing is pushed by this document) |
| Main file path | `Modules/Module_7/visitor_workflow/app_f8_integrated.py` |
| Python | 3.12 (Advanced settings) |
| Requirements | `Modules/Module_7/visitor_workflow/requirements.txt` (read before the root file; no ML packages) |
| Secrets | none. The app opens directly on its home page: there is no password or login screen, so anyone with the link can reach it. |
| Optional environment | `F8_ENABLE_FULL_CALIBRATION=1` only to test calibration inside the app; otherwise use *Upload Saved Calibration*. |
| Theme | dark with the red accent (`#C9363E`), set by `Modules/Module_7/visitor_workflow/.streamlit/config.toml` (read from the main script's folder, so the root and the main app are unaffected) |
| Not needed | recordings, `outputs/`, `data/`, `.streamlit/` files, the experiment scripts |

## What can be tested
1. Open the home page directly, then **Upload Saved Calibration** (capture mode *video*). The validated calibration file
   `Modules/Module_7/calibration/room_video/room_video_recipe_a_calibration.json` is accepted with capture mode *video* (checked locally).
2. Room-video upload: real HEVC `.MOV` files, file/size/duration/frame-rate/resolution checks, the displayed requirements.
3. Frame extraction, frame selection and boundary annotation.
4. *Start over* / *Lock this session* (setup/review/ready pages, and *Session Options* on the measurement page) and the retention sweep.

5. **Wall A Calculate** (real calibration, four real videos, the visitor's own frames and boundaries): the measurement is computed from those inputs only, never from stored experiment results.

6. **Wall B Calculate** (after an accepted Wall A result: six real videos, three frame sets, the seven-line boundaries): if the server refuses or stops it, note the exact reason shown. Nothing is retried automatically with other settings.

## Workload
* Wall A (two viewpoints): 4 videos, 5 frame sets (`room_frames.DEFAULT_SETS`), 4 full-resolution images per set.
* Wall B (three viewpoints): 6 videos, 3 frame sets by default (`three_view_workflow`: min 2, default 3, max 5), 6 images per set.
* The only Cloud figure for Wall B (about 2418 MiB worker peak) is for ONE synthetic frame set.
