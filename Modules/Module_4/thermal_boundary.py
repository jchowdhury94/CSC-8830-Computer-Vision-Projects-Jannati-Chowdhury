"""Classical false-color thermal segmentation using OpenCV and NumPy only.

Public color images are RGB uint8. The largest warm region is assumed to be
one person; this palette-dependent method does not recognize humans.
"""

import cv2
import numpy as np


# OpenCV hue is 0..179; saturation and value are 0..255 (inclusive).
HSV_RANGES = (((0, 60, 60), (85, 255, 255)),
              ((170, 60, 60), (179, 255, 255)))
CLOSING_KERNEL_SIZE = 3
CLOSING_ITERATIONS = 1


def normalize_thermal_image(image, color_order="RGB"):
    """Copy uint8 RGB/BGR color input to RGB without resizing; discard alpha."""
    if color_order not in ("RGB", "BGR"):
        raise ValueError("Color order must be RGB or BGR.")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.size == 0 or image.shape[2] not in (3, 4)):
        raise ValueError("Provide a nonempty uint8 false-color thermal image with 3 or 4 channels.")
    rgb = image[:, :, :3]
    if color_order == "BGR":
        rgb = rgb[:, :, ::-1]
    return np.array(rgb, dtype=np.uint8, order="C", copy=True)


def thermal_to_hsv(image):
    """Convert an RGB thermal image to OpenCV HSV."""
    return cv2.cvtColor(normalize_thermal_image(image), cv2.COLOR_RGB2HSV)


def warm_color_mask(hsv):
    """Threshold green through red, including red's hue wraparound."""
    if (not isinstance(hsv, np.ndarray) or hsv.dtype != np.uint8
            or hsv.ndim != 3 or hsv.size == 0 or hsv.shape[2] != 3):
        raise ValueError("HSV image must be a nonempty three-channel uint8 array.")
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(
            hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)
        ))
    return mask


def _validate_mask(mask):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8
            or mask.ndim != 2 or mask.size == 0
            or not np.all((mask == 0) | (mask == 255))):
        raise ValueError("Mask must be a nonempty 2D uint8 array containing only 0 and 255.")


def cleanup_thermal_mask(mask):
    """Fill small gaps with one 3x3 elliptical closing operation."""
    _validate_mask(mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSING_ITERATIONS)


def select_largest_component(mask):
    """Return selected mask, foreground count and pixel area (8-connectivity)."""
    _validate_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count == 1:
        raise ValueError("No foreground matched the warm-color thresholds. Try another false-color thermal image.")
    # Exclude background label zero; ties retain the first largest label.
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    human_mask = np.where(labels == selected, 255, 0).astype(np.uint8)
    return human_mask, count - 1, int(stats[selected, cv2.CC_STAT_AREA])


def extract_external_contour(mask):
    """Extract the external boundary, leaving interior holes unoutlined."""
    _validate_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("The selected foreground has no external contour.")
    return max(contours, key=cv2.contourArea)


def overlay_thermal_boundary(image, contour):
    """Draw a three-pixel bright magenta contour on an RGB image copy."""
    overlay = normalize_thermal_image(image)
    if (not isinstance(contour, np.ndarray) or contour.dtype != np.int32
            or contour.ndim != 3 or contour.shape[1:] != (1, 2) or contour.size == 0):
        raise ValueError("Contour must be a nonempty int32 array of shape (N, 1, 2).")
    cv2.drawContours(overlay, [contour], -1, (255, 0, 255), 3)
    return overlay


def find_thermal_boundary(image, *, color_order="RGB"):
    """Return full-resolution RGB images, HSV, binary masks and component metrics.

    Occupancy is the selected foreground pixel count / image pixel count,
    expressed as a percentage; it is not segmentation accuracy.
    """
    original = normalize_thermal_image(image, color_order)
    hsv = thermal_to_hsv(original)
    initial = warm_color_mask(hsv)
    closed = cleanup_thermal_mask(initial)
    human, count, area = select_largest_component(closed)
    contour = extract_external_contour(human)
    height, width = original.shape[:2]
    return {
        "original_image": original,
        "hsv_image": hsv,
        "initial_hsv_mask": initial,
        "closed_mask": closed,
        "cleaned_human_mask": human,
        "contour": contour,
        "boundary_overlay": overlay_thermal_boundary(original, contour),
        "width": width,
        "height": height,
        "foreground_components": count,
        "selected_component_area": area,
        "selected_component_percentage": 100.0 * area / (width * height),
    }
