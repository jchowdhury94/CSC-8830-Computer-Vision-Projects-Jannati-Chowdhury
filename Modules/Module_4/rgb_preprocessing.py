"""RGB upload preparation and coordinate mapping; no segmentation changes."""

import cv2
import numpy as np

from Modules.Module_4.human_boundary import normalize_image, validate_rectangle


MAX_RGB_DIMENSION = 1500


def decode_rgb(image_bytes):
    """Decode the original upload using the RGB experiment's conventions."""
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return normalize_image(decoded, color_order="BGR")


def prepare_rgb_upload(image_bytes):
    """Return a Classical processing image and the original image dimensions."""
    image = decode_rgb(image_bytes)
    original_shape = image.shape
    height, width = original_shape[:2]
    if max(height, width) > MAX_RGB_DIMENSION:
        scale = MAX_RGB_DIMENSION / max(height, width)
        image = cv2.resize(
            image, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image, original_shape


def prepared_rgb_upload(state, image_bytes, digest):
    """Reuse one prepared upload per session, including failed decode attempts."""
    key = "module4_rgb_prepared"
    cached = state.get(key)
    if cached is None or cached[0] != digest:
        try:
            image, original_shape = prepare_rgb_upload(image_bytes)
            cached = (digest, image, original_shape, None)
        except (ValueError, cv2.error):
            cached = (digest, None, None, "This file could not be read as an image. Please upload another JPG or PNG.")
        state[key] = cached
    return cached[1:]


def map_rectangle(rectangle, processing_shape, original_shape):
    """Map exclusive rectangle edges to original pixels, covering the selection."""
    x, y, width, height = validate_rectangle(rectangle, processing_shape)
    ph, pw = processing_shape[:2]
    oh, ow = original_shape[:2]
    left, top = x * ow // pw, y * oh // ph
    right = ((x + width) * ow + pw - 1) // pw
    bottom = ((y + height) * oh + ph - 1) // ph
    return validate_rectangle((left, top, right - left, bottom - top), original_shape)
