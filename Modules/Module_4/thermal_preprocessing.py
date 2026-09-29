"""Shared thermal upload preparation; no segmentation changes."""

import math

import cv2
import numpy as np

from Modules.Module_4.thermal_boundary import normalize_thermal_image
from Modules.Module_4.sam2_runtime import CPU_MAX_PIXELS, cpu_pixel_limit


MAX_THERMAL_DIMENSION = 1500
MAX_THERMAL_PIXELS = 500_000


def preprocessing_policy():
    """Version and limits participate in cache and selection invalidation."""
    configured = cpu_pixel_limit()
    budget = min(MAX_THERMAL_PIXELS, CPU_MAX_PIXELS, configured or CPU_MAX_PIXELS)
    return ("shared-thermal-area-floor-v1", MAX_THERMAL_DIMENSION, budget)


def decode_thermal(image_bytes):
    """Decode the original upload using the thermal experiment's stored orientation and color conventions."""
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return normalize_thermal_image(decoded, color_order="BGR")


def prepare_thermal_upload(image_bytes, policy=None):
    """Return one shared processing image and the original image dimensions."""
    image = decode_thermal(image_bytes)
    original_shape = image.shape
    height, width = original_shape[:2]
    _, longest_side, budget = policy or preprocessing_policy()
    scale = min(1.0, longest_side / max(height, width), math.sqrt(budget / (height * width)))
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, math.floor(width * scale)), max(1, math.floor(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image, original_shape


def prepared_thermal_upload(state, image_bytes, digest):
    """Reuse one prepared upload per session, including failed decode attempts."""
    key = "module4_q2_prepared"
    identity = (digest, preprocessing_policy())
    cached = state.get(key)
    if cached is None or cached[0] != identity:
        try:
            image, original_shape = prepare_thermal_upload(image_bytes, identity[1])
            cached = (identity, image, original_shape, None)
        except (ValueError, cv2.error):
            cached = (identity, None, None, "This file could not be read as an image. Please upload another JPG or PNG.")
        state[key] = cached
    return cached[1:]
