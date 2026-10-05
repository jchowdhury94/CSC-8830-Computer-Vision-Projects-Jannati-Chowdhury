# Part B — Step 2: correspondence preparation

The stationary book was photographed from four camera viewpoints. View order:
IMG_7596.jpeg, IMG_7597.jpeg, IMG_7598.jpeg, IMG_7603.jpeg.

Original JPEGs are read only. EXIF orientation is applied to separate PNG working
copies without resizing. All four saved working images are 4284 × 5712 pixels.
Coordinates are full-resolution upright pixels: origin upper-left, x right,
y down. The cover dimensions (14.5 × 21.7 cm) are recorded as metadata only;
they do not change the image coordinates.

## Regenerate the artifacts

From the repository root:

```sh
.venv/bin/python Modules/Module_6/Part_B/prepare_correspondences.py
```

The script reads the manually identified coordinates in
`correspondence_selection.json`, checks them, saves the CSV and annotations,
verifies the serialized CSV, and checks original JPEG hashes before/after.
The validation JSON records the hashes, dimensions, checks, method, and limits.
The default run does not select points through an automatic matcher.

## Review or reselect manually

```sh
.venv/bin/python Modules/Module_6/Part_B/prepare_correspondences.py --manual
```

This requires a desktop OpenCV GUI. For each view, follow the displayed physical
feature descriptions in the same P1–P8 order. Click the scaled overview to open
a 400 × 400 full-resolution detail crop, click the precise feature in that crop,
then press Enter to accept. Press U to undo the last accepted point in the
current view; Esc cancels without replacing the saved coordinates. The overview
mapping uses its actual horizontal/vertical scale and pixel-center convention;
the detail crop uses native pixels plus its full-resolution crop origin.
Working images and saved coordinates are never resized for calculations.

P1–P4 are the physical front-cover top-left, top-right, bottom-right, and
bottom-left corners, as defined with the title upright. In View 4 these appear
in different screen positions. P5 is the center of the diamond-shaped head of
the apostrophe in Swan’s. P6 is the dot above the final i in Roshani. P7 is the
tip of the upper-right gold border flourish. P8 is the center of the terminal
round loop on the lower-left gold border.

Review `output/part_b_correspondence_comparison.png` for the four-view overview
and `output/part_b_landmark_closeups.png` for individual landmarks. Each
`view_*_correspondences.png` also retains full resolution. Markers are open
rings, with labels offset from the feature.

The initial coordinates were manually estimated by the assistant through
visual inspection of full-resolution crops, without subpixel refinement.
Rounded/flexible cover corners have some localization uncertainty. The close-up
sheet shows the chosen front-cover perimeter locations rather than back-cover
corners. Inspect these before subsequent geometry. Illustrated facial details,
repeated stars/leaves, and water texture were excluded because precise
localization or identity was less clear; shadows, glare, and background were
also excluded. The manual GUI is provided for correction and has not been
interactively exercised in this session.

No observation matrix, centering, factorization, SfM, reconstruction, camera
pose estimation, or physical scaling is performed in this step.
