# Part B - Structure from Motion Reconstruction

Reconstruction of an approximately planar object (the front cover of *The Swan's Daughter*, measured
14.5 x 21.7 cm) from four viewpoints.

## Experiment

- **Photographs:** `IMG_7596`, `IMG_7597`, `IMG_7598`, `IMG_7603` (Views 1-4). iPhone 17 Pro Max rear wide
  camera, actual focal length 6.765 mm, upright resolution 4284 x 5712 px.
- **Corresponding points:** P1-P4 are the top-left, top-right, bottom-right and bottom-left cover corners (title
  upright). Their coordinates, manually reviewed and approved, are in `correspondence_selection.json`, which
  records how they were obtained. P5-P8 are interior feature points identified automatically from the four
  View 1 corners; all eight points were verified and no landmark needed correction. The automatic corner
  estimates in Views 2-4 are replaced by the verified corner coordinates.
- **Reconstruction:** measurement matrix, Tomasi-Kanade rank-3 factorization, metric upgrade, coordinate
  alignment, one least-squares scale from the measured width and height, and boundary edge lengths. The
  equations are shown on the results page and implemented in `sfm_pipeline.py`, which the visitor workflow
  uses as well.

| Edge | Reference (cm) | Reconstructed (cm) | Error (%) |
|---|---:|---:|---:|
| Top | 14.500 | 13.108 | 9.60 |
| Bottom | 14.500 | 15.948 | 9.98 |
| Left | 21.700 | 21.346 | 1.63 |
| Right | 21.700 | 22.016 | 1.46 |

Width and height set the scale, so this comparison is not an independent validation. The orthographic model,
the nearly planar object and point localization limit accuracy.

## Files

| Path | Content |
|---|---|
| `automatic_landmark_experiment.py` | Runs the experiment and draws its figures |
| `sfm_pipeline.py` | Correspondence method and reconstruction mathematics (shared) |
| `visitor_reconstruction.py` | Image decoding, features, point markers and CSV export (shared) |
| `results_data.py` | Loads the saved experiment for the results page |
| `correspondence_selection.json` | Verified corner coordinates and measured dimensions |
| `submission_images/` | The four photographs, full resolution, without location metadata |
| `sanitize_submission_images.py` | Recreates those copies from the local originals in `input/` (not in Git) |
| `output/revised/` | Final coordinates, reconstructed points, measurements, summary and figures |

## Visitor workflow

**Reconstruct an Object from Your Own Images** on the Module 6 page runs five steps:

1. Upload exactly four photographs: JPG, PNG or HEIC/HEIF, up to 25 MB each, 100 MB in total and 64 megapixels each.
2. Select the four corners in the first photograph, find the corresponding points automatically, then verify or
   correct all eight points in every photograph.
3. Enter the measured width and height in centimeters.
4. Reconstruct.
5. View the points, estimated boundary, 3D reconstruction and four edge measurements, and download the points and
   edge lengths (CSV).

Changing the photographs, points or dimensions discards the results that depend on them.

## Run

From the repository root, with the repository `requirements.txt` installed (NumPy, OpenCV, Pillow; pillow-heif for
HEIC photographs and Plotly for the page's plots):

```sh
.venv/bin/python Modules/Module_6/Part_B/automatic_landmark_experiment.py
```

The script reads the corner file and `submission_images/`, writes only `output/revised/`, and checks that no
other Part B file changed. Add `--figures-only` to redraw the two results-page figures from the saved
coordinates. Figure label fonts depend on the system (Arial on macOS, otherwise DejaVu).
