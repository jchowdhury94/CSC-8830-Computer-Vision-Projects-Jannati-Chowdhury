"""Draggable corner marker for the zoom refinement view: a tiny static Streamlit component (no build step, no extra dependency).

The frontend keeps the marker as a fraction (0..1, edge-based) of the displayed zoom image and reports it when the pointer is
released. The conversions below turn that fraction into original-image continuous pixel coordinates (pixel i centred on i).
"""
import base64
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO

_component = components.declare_component("m7_corner_drag", path=str(Path(__file__).parent / "corner_drag_frontend"))


def fraction_to_source(frac, src_origin, src_size, image_size):
    """Marker fraction (fx, fy) of the zoom crop -> original continuous pixel coordinates, clamped to the image's pixel centres.

    src_origin (x0, y0) and src_size (w, h) describe the crop in original pixels; the crop edge x0 - 0.5 is fraction 0.
    """
    x = src_origin[0] + frac[0] * src_size[0] - 0.5
    y = src_origin[1] + frac[1] * src_size[1] - 0.5
    return (float(np.clip(x, 0, image_size[0] - 1)), float(np.clip(y, 0, image_size[1] - 1)))


def source_to_fraction(pt, src_origin, src_size):
    """Inverse of fraction_to_source (no clamping)."""
    return ((pt[0] + 0.5 - src_origin[0]) / src_size[0], (pt[1] + 0.5 - src_origin[1]) / src_size[1])


def fraction_limits(src_origin, src_size, image_size):
    """(fx_min, fx_max, fy_min, fy_max): the marker range that stays on the image (pixel centres 0 .. size-1) inside the crop."""
    lo = source_to_fraction((0, 0), src_origin, src_size)
    hi = source_to_fraction((image_size[0] - 1, image_size[1] - 1), src_origin, src_size)
    return (max(0.0, lo[0]), min(1.0, hi[0]), max(0.0, lo[1]), min(1.0, hi[1]))


def refined_point(value, token, src_origin, src_size, image_size):
    """Original-image point for a component value, or None if there is no value or it belongs to another pending corner."""
    if not isinstance(value, dict) or value.get("token") != token:
        return None
    try:
        frac = (float(value["fx"]), float(value["fy"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(frac)):
        return None
    return fraction_to_source(frac, src_origin, src_size, image_size)


def corner_drag(image, marker, limits, token, key):
    """Show `image` (RGB array) with a draggable marker starting at `marker` (fractions). Returns the last reported value or None."""
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=92)
    src = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    return _component(src=src, marker=list(marker), limits=list(limits), token=token, key=key, default=None)
