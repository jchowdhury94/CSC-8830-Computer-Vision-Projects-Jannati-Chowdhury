"""Image-side helpers for the interactive app: upload validation, coordinate mapping, zoom windows and overlays.

Coordinate conventions. "Original" coordinates are OpenCV continuous pixel coordinates of the camera's EXIF-oriented calibration
frame (pixel i is centred on i), exactly what triangulate() expects. A click reported by the image component is an
integer offset u inside a rendered image of reported width rw; it identifies the rendered pixel [u, u+1), whose centre u + 0.5
lies on the source region's edge-based axis. Every mapping uses the actual source size, crop size and rendered size of that
particular image; there is no shared scale factor.
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import CORNER_NAMES, IMAGE_SIZE
from .user_calibration import CalibrationPhotoError, inspect_photo

ACCEPTED_FORMATS = ("JPEG", "PNG")
# The author's Model B cameras, kept as reference data. No function here falls back to them: every upload check and decode
# must be given the active calibration's size (and camera metadata, or None to skip that comparison).
MODEL_B_CAMERAS = {"left": {"model": "iPhone Air", "focal_length_mm": 5.96},
                   "right": {"model": "iPhone 17 Pro Max", "focal_length_mm": 6.765}}
EXPECTED_MODEL = {s: c["model"] for s, c in MODEL_B_CAMERAS.items()}                    # kept for backward compatibility
EXPECTED_FOCAL_MM = {s: c["focal_length_mm"] for s, c in MODEL_B_CAMERAS.items()}
ORIENTATION_TAG, MODEL_TAG, EXIF_IFD, FOCAL_TAG, LENS_TAG = 274, 272, 0x8769, 0x920A, 0xA434
FOCAL_TOLERANCE_MM = 0.05            # absolute, or 1% of the calibrated focal length if larger
FOCAL_35MM_TOLERANCE = 1.0
COLORS_RGB = [(255, 200, 0), (0, 220, 0), (255, 120, 0), (255, 0, 200)]   # TL, TR, BR, BL
LINE_RGB = (0, 255, 255)
METADATA_CAVEAT = ("Matching camera metadata does not prove that the physical stereo rig, lenses and settings are unchanged, and "
                   "missing or different metadata does not by itself prove that they changed.")


class UploadError(ValueError):
    """The upload cannot be used with the active stereo calibration."""


def _size_message(side, oriented, stored, orientation, expected, other_size):
    (w, h), (ew, eh) = oriented, expected
    text = (f"The {side} image is {w} x {h} px after applying its EXIF orientation (stored {stored[0]} x {stored[1]}, orientation "
            f"tag {orientation}), but the {side} camera of the active calibration uses {ew} x {eh} px. "
            "The app does not resize or rotate uploads.")
    if oriented == (eh, ew):
        text += " The image looks rotated by 90 degrees relative to the calibration photos: use the same orientation as in calibration."
    elif other_size is not None and oriented == tuple(other_size) and tuple(other_size) != expected:
        text += " This size matches the other camera, so the left and right images may be swapped."
    elif abs(w / h - ew / eh) < 0.01 * ew / eh:
        text += (" The shape matches but the resolution differs: use the same camera mode and resolution as in calibration, "
                 "with no zoom, crop or resizing.")
    return text


COMPARED_FIELDS = ("make", "model", "lens_model", "focal_length_mm", "focal_length_35mm", "orientation")


def comparable_camera_fields(info):
    """The part of a camera-info dict that can actually be compared (known, non-empty fields)."""
    return {k: v for k, v in (info or {}).items() if k in COMPARED_FIELDS and v not in (None, "")}


def compare_camera_metadata(actual, expected, side, other_expected=None):
    """Compare an upload's EXIF summary with the calibration photos' summary. Returns a list of {code, message} WARNINGS.

    Only fields present in `expected` are compared (make, model, lens_model, focal_length_mm, focal_length_35mm, orientation).
    Nothing here is proof either way (see METADATA_CAVEAT); a mismatch or missing metadata never blocks an upload.
    """
    keys = COMPARED_FIELDS
    expected = comparable_camera_fields(expected)
    if not expected:
        return []
    actual = actual or {}
    present = lambda k: actual.get(k) not in (None, "")
    if not any(present(k) for k in keys[:5]):
        return [dict(code="no_exif", message=(
            f"The {side} image has no camera metadata (EXIF), so its camera model, lens and focal length could not be compared "
            "with the calibration photos. This is common for edited, exported or messaging-app images."))]
    out = []
    for key, label in (("make", "camera make"), ("model", "camera model"), ("lens_model", "lens")):
        if key in expected and present(key) and str(actual[key]).strip().lower() != str(expected[key]).strip().lower():
            hint = ""
            if key == "model" and other_expected and str(other_expected.get("model", "")).strip().lower() == str(actual[key]).strip().lower():
                hint = f" This matches the {'right' if side == 'left' else 'left'}-camera calibration setup, so the images may be swapped."
            out.append(dict(code=f"{key}_mismatch", message=(
                f"The {side} image's {label} is '{actual[key]}', but the calibration's {side} camera was '{expected[key]}'. "
                f"A different camera or lens would invalidate the calibration; EXIF alone cannot confirm either way.{hint}")))
    for key, label, unit, tol in (("focal_length_mm", "focal length", "mm", None), ("focal_length_35mm", "35 mm-equivalent focal length", "mm", FOCAL_35MM_TOLERANCE)):
        if key in expected and present(key):
            tolerance = tol if tol is not None else max(FOCAL_TOLERANCE_MM, 0.01 * float(expected[key]))
            if abs(float(actual[key]) - float(expected[key])) > tolerance:
                out.append(dict(code=f"{key}_mismatch", message=(
                    f"The {side} image's {label} is {float(actual[key]):.3f} {unit}, but the calibration photos used "
                    f"{float(expected[key]):.3f} {unit}; a different lens or zoom would invalidate the calibration.")))
    if "orientation" in expected and int(actual.get("orientation") or 1) != int(expected["orientation"]):
        out.append(dict(code="orientation_differs", message=(
            f"The {side} image's EXIF orientation tag is {actual.get('orientation') or 1}, but the calibration photos used "
            f"{expected['orientation']}. The image size matches, but the camera may have been held differently; confirm the rig "
            "and orientation are the same as in calibration.")))
    missing = [label for key, label in (("make", "make"), ("model", "model"), ("lens_model", "lens"), ("focal_length_mm", "focal length"),
                                        ("focal_length_35mm", "35 mm focal length")) if key in expected and not present(key)]
    if missing:
        out.append(dict(code="metadata_incomplete", message=f"The {side} image lacks metadata for: {', '.join(missing)}; those could not be compared."))
    return out


def check_upload(data, side, expected_size, expected_camera, other_size=None, other_camera=None):
    """Validate one object photograph against the camera it must match. Cheap (header only); raises UploadError if unusable.

    side: 'left' or 'right'. expected_size: (w, h) of that camera's calibration photos after EXIF orientation.
    expected_camera: that camera's EXIF summary from calibration (dict), or None to skip the metadata comparison. Both are
    required: there is no default, so a visitor measurement can never be checked against Model B by omission. other_size / other_camera: the opposite camera's values, used only to hint at swapped images.

    Returns dict(size, stored_size, orientation, exif, warnings=[{code, message}], metadata_compared). The size check is an error;
    metadata differences are warnings (see METADATA_CAVEAT).
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    expected = tuple(int(v) for v in expected_size)
    try:
        im, orientation, oriented, exif = inspect_photo(data, min_side=1)
    except CalibrationPhotoError as error:
        raise UploadError(str(error)) from error
    if oriented != expected:
        raise UploadError(_size_message(side, oriented, im.size, orientation, expected, other_size))
    warnings = compare_camera_metadata(exif, expected_camera, side, other_camera)
    return dict(size=oriented, stored_size=im.size, orientation=orientation, exif=exif, warnings=warnings,
                metadata_compared=bool(comparable_camera_fields(expected_camera)))


def inspect_upload(data, side, expected_size, expected_camera, other_size=None, other_camera=None):
    """check_upload() reduced to (oriented_size, [warning text]) for callers that only need the messages."""
    report = check_upload(data, side, expected_size, expected_camera, other_size, other_camera)
    return report["size"], [w["message"] for w in report["warnings"]]


def expected_for_side(cal, side):
    """What an object photo must match, read from a calibration dict: {'size': (w, h), 'camera': EXIF summary or None}.

    Uses the per-camera sizes of visitor calibrations (image_size_left/right) and falls back to a single image_size. The camera
    summary is cal['camera'][side] when the calibration recorded one.
    """
    size = cal.get(f"image_size_{side}") or cal.get("image_size")
    if size is None:
        raise ValueError("The calibration does not record its image size.")
    return dict(size=tuple(int(v) for v in size), camera=(cal.get("camera") or {}).get(side) or None)


def check_upload_for_calibration(data, side, cal):
    """check_upload() against the active calibration `cal` (computed or loaded); never consults Model B."""
    other = "right" if side == "left" else "left"
    mine, theirs = expected_for_side(cal, side), expected_for_side(cal, other)
    return check_upload(data, side, mine["size"], mine["camera"], other_size=theirs["size"], other_camera=theirs["camera"])


def decode_upload(data, expected_size):
    """Full decode to an RGB uint8 array in the camera's calibration frame (EXIF orientation applied explicitly, no resizing)."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    arr = np.asarray(im)
    if (arr.shape[1], arr.shape[0]) != tuple(expected_size):
        raise UploadError("Decoded image size does not match the calibration frame.")
    return arr


# ---- coordinate mapping -------------------------------------------------------------------------------------------------

def make_preview(image, max_width=700):
    """Uniformly downscaled copy; returns the array. The true per-axis scale is always recomputed from the actual shapes."""
    h, w = image.shape[:2]
    pw = min(w, max_width)
    ph = max(1, round(h * pw / w))
    return cv2.resize(image, (pw, ph), interpolation=cv2.INTER_AREA)


def click_to_source(click, src_origin, src_size):
    """Map a component click to source continuous pixel coordinates.

    src_origin (x0, y0): integer pixel of the source image shown at the rendered image's top-left; src_size (w, h): number of
    source pixels shown across/down the rendered image. The rendered size comes from the click itself.
    """
    rw, rh = float(click["width"]), float(click["height"])
    if rw <= 0 or rh <= 0:
        raise ValueError("The image is not rendered yet; click again.")
    u, v = float(click["x"]), float(click["y"])
    if not (0 <= u <= rw and 0 <= v <= rh):
        raise ValueError("The click is outside the image.")
    x = src_origin[0] + (u + 0.5) * src_size[0] / rw - 0.5
    y = src_origin[1] + (v + 0.5) * src_size[1] / rh - 0.5
    return x, y


def source_to_shown(pt, src_origin, src_size, shown_size):
    """Inverse of click_to_source: source continuous pixel coordinates -> position in an image of shown_size (float pixels)."""
    u = (pt[0] + 0.5 - src_origin[0]) * shown_size[0] / src_size[0] - 0.5
    v = (pt[1] + 0.5 - src_origin[1]) * shown_size[1] / src_size[1] - 0.5
    return u, v


def zoom_window(pt, half=80, size=IMAGE_SIZE):
    """Integer crop (x0, y0, w, h) of the original image around pt, shifted to stay entirely inside the image."""
    w = h = 2 * half
    x0 = int(np.clip(round(pt[0]) - half, 0, size[0] - w))
    y0 = int(np.clip(round(pt[1]) - half, 0, size[1] - h))
    return x0, y0, w, h


def make_zoom(image, window, out_width=640):
    """Crop (x0, y0, w, h) upscaled to out_width wide; returns (array, (w, h) of the crop)."""
    x0, y0, w, h = window
    crop = image[y0:y0 + h, x0:x0 + w]
    out_h = round(out_width * h / w)
    return cv2.resize(crop, (out_width, out_h), interpolation=cv2.INTER_CUBIC), (w, h)


# ---- drawing (RGB arrays) -----------------------------------------------------------------------------------------------

def _label(im, text, pos, color, scale):
    cv2.putText(im, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(im, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_points(img, shown_pts, close=False, radius=7, font_scale=0.7, names=False, ids=None, lines=True):
    """Numbered markers and connecting lines at already-converted positions in img. Returns a new array.

    ids gives each point's corner index (0-3) when only a subset is drawn; by default points are corners 0, 1, 2, ...
    """
    im = img.copy()
    pts = [(int(round(x)), int(round(y))) for x, y in shown_pts]
    ids = list(range(len(pts))) if ids is None else list(ids)
    if lines and len(pts) > 1:
        cv2.polylines(im, [np.array(pts, np.int32).reshape(-1, 1, 2)], close and len(pts) == 4, LINE_RGB, 2, cv2.LINE_AA)
    h, w = im.shape[:2]
    for k, (x, y) in zip(ids, pts):
        cv2.circle(im, (x, y), radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(im, (x, y), radius - 2, COLORS_RGB[k], -1, cv2.LINE_AA)
        text = f"{k + 1} {CORNER_NAMES[k]}" if names else str(k + 1)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        pos = (int(np.clip(x + radius + 3, 2, w - tw - 3)), int(np.clip(y - radius - 3, th + 3, h - base - 3)))
        _label(im, text, pos, COLORS_RGB[k], font_scale)
    return im


def draw_crosshair(img, pos, color=(255, 255, 255), arm=14):
    im = img.copy()
    x, y = int(round(pos[0])), int(round(pos[1]))
    for c, t in (((0, 0, 0), 3), (color, 1)):
        cv2.line(im, (x - arm, y), (x + arm, y), c, t, cv2.LINE_AA)
        cv2.line(im, (x, y - arm), (x, y + arm), c, t, cv2.LINE_AA)
    return im


def guide_image(size=(360, 260)):
    """Schematic of a rectangle with its four numbered corners."""
    w, h = size
    im = np.full((h, w, 3), 245, np.uint8)
    pts = [(70, 55), (w - 70, 55), (w - 70, h - 55), (70, h - 55)]
    cv2.rectangle(im, pts[0], pts[2], (190, 190, 200), -1)
    cv2.rectangle(im, pts[0], pts[2], (90, 90, 100), 2)
    im = draw_points(im, pts, close=True, radius=9, font_scale=0.6, names=True)
    return im
