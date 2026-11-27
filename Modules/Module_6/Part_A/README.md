# Part A - Optical Flow and Motion Tracking

Saved results are in `output/`; the page `Pages/app_module6.py` reads them. Each script's header lists its
inputs, outputs and requirements.

## Method

Shi-Tomasi features are detected inside a selected rectangle on the first frame and tracked frame to frame with
pyramidal Lucas-Kanade optical flow. The settings are in `optical_flow.py`:

| Setting | Value |
|---|---|
| Maximum corners, quality level | 50, 0.3 |
| Minimum distance, block size | 7 px, 7 px |
| Lucas-Kanade window, maximum pyramid level | 21 x 21, 3 |
| Termination | 30 iterations or epsilon 0.01 |

## Validation results

A single-level, one-step local least-squares estimate (21 x 21 window) is compared with OpenCV's iterative
pyramidal result for one feature on one pair of consecutive frames per video.

| Video | Frames | Feature | Manual (u, v) px | OpenCV (u, v) px | Position error (px) | Relative error (%) |
|---|---|---:|---|---|---:|---:|
| 1 (horizontal) | 484 to 485 | 14 | (1.306, -0.013) | (0.997, -0.017) | 0.309 | 30.96 |
| 2 (diagonal) | 482 to 483 | 12 | (1.080, 0.662) | (0.806, 0.507) | 0.315 | 33.09 |

The motion directions agree; the magnitudes differ by about 31-33 %. In the bilinear interpolation example
(Video 2, feature 12) the manual intensity is 87.087322 and OpenCV's is 87.087326.

## Files

| Path | Content |
|---|---|
| `optical_flow.py` | Original tracking; shared settings and arrow drawing |
| `render_arrow_visualization.py` | Draws the arrow videos from the saved tracking data |
| `extract_validation_frames.py`, `extract_validation_frames_video2.py` | Consecutive-frame validation images |
| `annotate_validation_frames.py` | Adds motion arrows to the validation comparisons |
| `manual_lucas_kanade_validation.py`, `manual_lucas_kanade_validation_video2.py` | Manual Lucas-Kanade estimates |
| `bilinear_interpolation_validation.py` | Worked bilinear interpolation example |
| `visitor_tracking.py` | Visitor workflow backend |
| `output/` | Tracking CSVs, arrow videos, validation images and reports |

## Dependencies

Python with the repository `requirements.txt` installed (NumPy, OpenCV, imageio-ffmpeg).

## Inputs

`input/video_1_horizontal.MOV` and `input/video_2_diagonal.MOV` are the original recordings. They are kept
locally and are not in Git.

## Reproduce the saved figures

From the repository root, in this order. These commands use the saved tracking data
(`output/video_<n>_tracking.csv`); the first three also need the recordings.

```sh
.venv/bin/python -m Modules.Module_6.Part_A.render_arrow_visualization
.venv/bin/python -m Modules.Module_6.Part_A.extract_validation_frames
.venv/bin/python -m Modules.Module_6.Part_A.extract_validation_frames_video2
.venv/bin/python -m Modules.Module_6.Part_A.manual_lucas_kanade_validation
.venv/bin/python -m Modules.Module_6.Part_A.manual_lucas_kanade_validation_video2
.venv/bin/python -m Modules.Module_6.Part_A.annotate_validation_frames
.venv/bin/python -m Modules.Module_6.Part_A.bilinear_interpolation_validation
```

Motion arrows are yellow, 1 px thick, magnified 30x (maximum 90 px), with 2 px feature markers. The
magnification affects drawing only.

## Tracking

`optical_flow.py --video 1` (or `2`) created the tracking data. It asks for a rectangle drawn by hand around
the can and overwrites the tracking CSV, so run it only to repeat the tracking itself; the rectangle is not
saved and a new selection can give different features. It needs an OpenCV build with window support
(`opencv-python`), while the repository requirements install the headless build.

## Visitor workflow

**Optical Flow and Motion Tracking** on the Module 6 page runs four steps: upload a video, select the object with
a rectangle, run tracking, and view the results. The default tracking settings match the experiment and can be
changed under **Advanced Settings (Optional)**. Downloads are the annotated video (MP4), tracking data (CSV) and
summary (JSON); **Reset All** starts over.

Accepted files are MP4, MOV, M4V, AVI, MKV and WebM up to 50 MB. Only the first 20 seconds are analyzed, at most
600 frames are tracked (longer selections skip frames evenly), and larger videos are reduced to 960 px on the
longest side.
