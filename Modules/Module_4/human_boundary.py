"""Classical boundary extraction for a user-selected foreground region.

Public color images use RGB uint8 unless an input color order is specified.
GrabCut performs foreground/background segmentation from a user rectangle;
it does not recognize humans. Selecting the largest external contour assumes
one person is the dominant segmented foreground component.
"""

import cv2
import numpy as np


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _kernel_size(value: int) -> int:
    value = _integer(value, "Kernel size")
    if value % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    return value


def normalize_image(image: np.ndarray, color_order: str = "RGB") -> np.ndarray:
    """Return an independent RGB uint8 image; expand gray and discard alpha.

    Use color_order='BGR' for OpenCV-decoded BGR/BGRA input. Alpha is ignored,
    not composited. Other dtypes are rejected to avoid guessing their scale.
    """
    if color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be 'RGB' or 'BGR'.")
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Image must be a uint8 NumPy array with values 0–255.")
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("Image must be a nonempty grayscale or color image.")
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.shape[2] in (3, 4):
        rgb = image[:, :, :3]
        if color_order == "BGR":
            rgb = rgb[:, :, ::-1]
    else:
        raise ValueError("Image must have 1, 3, or 4 channels.")
    return np.array(rgb, dtype=np.uint8, order="C", copy=True)


def gaussian_preprocess(
    image: np.ndarray, kernel_size: int = 3, sigma: float = 0.8
) -> np.ndarray:
    """Smooth RGB input gently before edge detection and segmentation."""
    kernel_size = _kernel_size(kernel_size)
    if not isinstance(sigma, (int, float, np.number)) or not np.isreal(sigma):
        raise ValueError("Gaussian sigma must be a finite nonnegative number.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian sigma must be a finite nonnegative number.")
    return cv2.GaussianBlur(
        normalize_image(image), (kernel_size, kernel_size), float(sigma)
    )


def canny_edges(
    smoothed_image: np.ndarray, lower_threshold: float = 50,
    upper_threshold: float = 150,
) -> np.ndarray:
    """Return a uint8 edge map, including internal and background edges."""
    for threshold in (lower_threshold, upper_threshold):
        if not isinstance(threshold, (int, float, np.number)) or not np.isreal(threshold):
            raise ValueError("Canny thresholds must be finite nonnegative numbers.")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Canny thresholds must be finite nonnegative numbers.")
    if lower_threshold >= upper_threshold:
        raise ValueError("Canny lower threshold must be less than upper threshold.")
    gray = cv2.cvtColor(normalize_image(smoothed_image), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, float(lower_threshold), float(upper_threshold))


def validate_rectangle(
    rectangle: tuple[int, int, int, int], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Intersect (x, y, width, height) with image bounds and check samples."""
    if not isinstance(rectangle, (tuple, list, np.ndarray)) or len(rectangle) != 4:
        raise ValueError("Rectangle must contain (x, y, width, height).")
    for value in rectangle:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("Rectangle coordinates and dimensions must be integers.")
    x, y, width, height = map(int, rectangle)
    if width <= 0 or height <= 0:
        raise ValueError("Rectangle width and height must be positive.")
    image_height, image_width = image_shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(image_width, x + width), min(image_height, y + height)
    width, height = right - left, bottom - top
    if width < 2 or height < 2 or width * height < 5:
        raise ValueError("Clamped rectangle is too small; use at least 2×3 pixels.")
    # GrabCut fits five-component color models and needs samples on both sides.
    if image_width * image_height - width * height < 5:
        raise ValueError("Rectangle must leave at least 5 background pixels outside it.")
    return left, top, width, height


def grabcut_segment(
    image: np.ndarray, rectangle: tuple[int, int, int, int], iterations: int = 5
) -> np.ndarray:
    """Segment RGB color input using a rectangle; return a 0/255 uint8 mask."""
    rgb = normalize_image(image)
    rectangle = validate_rectangle(rectangle, rgb.shape)
    iterations = _integer(iterations, "GrabCut iterations")
    labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    # Segmentation uses color statistics, independently of the Canny edge map.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(
            bgr, labels, rectangle, background_model, foreground_model,
            iterations, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as error:
        raise ValueError(
            "GrabCut could not segment this image; try a different rectangle "
            "containing the person and leaving background outside."
        ) from error
    foreground = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    return foreground.astype(np.uint8) * 255


def _validate_mask(mask: np.ndarray) -> None:
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8
            or mask.ndim != 2 or mask.size == 0):
        raise ValueError("Mask must be a nonempty 2D uint8 array.")
    if not np.all((mask == 0) | (mask == 255)):
        raise ValueError("Binary mask must contain only 0 and 255.")


def cleanup_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply one small closing operation to fill narrow gaps in a binary mask."""
    _validate_mask(mask)
    kernel_size = _kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def extract_boundary(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return the largest external contour and geometric area, or (None, 0)."""
    _validate_mask(mask)
    contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    return contour, float(cv2.contourArea(contour))


def overlay_boundary(
    image: np.ndarray, contour: np.ndarray | None, thickness: int = 2
) -> np.ndarray:
    """Draw a green boundary on an RGB copy; no contour leaves it unchanged."""
    thickness = _integer(thickness, "Boundary thickness")
    overlay = normalize_image(image)
    if contour is not None:
        if (not isinstance(contour, np.ndarray) or contour.dtype != np.int32
                or contour.ndim != 3 or contour.shape[1:] != (1, 2)
                or len(contour) == 0):
            raise ValueError("Contour must be a nonempty int32 array of shape (N, 1, 2).")
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness)
    return overlay


def find_human_boundary(
    image: np.ndarray,
    rectangle: tuple[int, int, int, int],
    *,
    color_order: str = "RGB",
    gaussian_kernel_size: int = 3,
    gaussian_sigma: float = 0.8,
    canny_lower: float = 50,
    canny_upper: float = 150,
    grabcut_iterations: int = 5,
    cleanup_enabled: bool = False,
    morphology_kernel_size: int = 3,
    boundary_thickness: int = 2,
) -> dict:
    """Return RGB images, binary masks, contour, area, and clamped rectangle.

    Canny is an independent edge-detection demonstration. GrabCut receives the
    smoothed color image. With cleanup disabled, cleaned_mask copies raw_mask.
    Contour area is geometric area in squared pixels, not foreground pixel count.
    An empty segmentation yields contour=None, area=0, and an unchanged overlay.
    """
    if not isinstance(cleanup_enabled, (bool, np.bool_)):
        raise ValueError("cleanup_enabled must be True or False.")
    _kernel_size(morphology_kernel_size)
    _integer(boundary_thickness, "Boundary thickness")
    original = normalize_image(image, color_order)
    rectangle = validate_rectangle(rectangle, original.shape)
    smoothed = gaussian_preprocess(original, gaussian_kernel_size, gaussian_sigma)
    edges = canny_edges(smoothed, canny_lower, canny_upper)
    raw_mask = grabcut_segment(smoothed, rectangle, grabcut_iterations)
    # Cleanup is opt-in because even small morphology can alter thin body parts.
    cleaned_mask = (
        cleanup_mask(raw_mask, morphology_kernel_size)
        if cleanup_enabled else raw_mask.copy()
    )
    contour, area = extract_boundary(cleaned_mask)
    overlay = overlay_boundary(original, contour, boundary_thickness)
    return {
        "original_image": original,
        "smoothed_image": smoothed,
        "canny_edges": edges,
        "raw_mask": raw_mask,
        "cleaned_mask": cleaned_mask,
        "contour": contour,
        "contour_area": area,
        "boundary_overlay": overlay,
        "rectangle": rectangle,
    }
