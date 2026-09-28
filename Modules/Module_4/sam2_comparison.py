"""In-memory and offline SAM 2.1 inference, independent of the classical pipeline.

Box convention: --box X Y WIDTH HEIGHT in original decoded-image pixels,
with exclusive right/bottom boundaries. Supply the same box used for GrabCut.
The target is the visible person (including clothing and accessories), not the
cast shadow. No point prompts, mask prompts, or manual mask edits are used.

Use the validated csc8830-combined environment. In-memory usage:
    resource = load_sam2_model(checkpoint)
    result = run_sam2(image_rgb, (x, y, width, height), resource)

Cache/reuse the resource, not a predictor with image-specific state. No Streamlit
dependency, automatic downloads, or result-file writes are needed for this API.

Obtain sam2.1_hiera_tiny.pt separately from the official SAM2 repository's
checkpoint links and store it outside this repository. No automatic downloads.
After validating the environment, preserve pip freeze output for reproduction.

Example (replace image, box, and checkpoint with actual experiment inputs):
    conda run -n csc8830-combined python Modules/Module_4/sam2_comparison.py \
        --image /path/to/person.jpg --box 10 20 300 600 \
        --checkpoint /path/outside/repository/sam2.1_hiera_tiny.pt \
        --output-dir Modules/Module_4/comparison_results/experiment_01
"""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


MODEL_NAME = "sam2.1_hiera_tiny"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
OUTPUT_FILES = ("sam2_mask.png", "sam2_overlay.png", "metadata.json")


@dataclass(frozen=True)
class SAM2ModelResource:
    """Reusable model plus a lock shared by every request using this resource.

    Treat the model as read-only after loading; do not move it between devices.
    For a custom configuration, supply its corresponding model_name to the loader.
    """

    model: object = field(repr=False)
    model_name: str
    model_config: str
    checkpoint_path: Path
    device: str
    _lock: object = field(default_factory=Lock, repr=False, compare=False)


def read_image(path: Path) -> tuple[np.ndarray, str]:
    """Decode exactly as the Streamlit page does; return RGB and file SHA-256."""
    if not path.is_file():
        raise ValueError(f"Input image not found: {path}")
    content = path.read_bytes()
    if not content:
        raise ValueError(f"Input image is empty: {path}")
    bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hashlib.sha256(content).hexdigest()


def box_to_xyxy(box, image_shape) -> np.ndarray:
    """Validate XYWH without silently changing the recorded classical box."""
    if not isinstance(box, (tuple, list, np.ndarray)) or np.shape(box) != (4,) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer)) for value in box
    ):
        raise ValueError("Box must contain four integer values: x y width height.")
    x, y, width, height = map(int, box)
    image_height, image_width = image_shape[:2]
    if x < 0 or y < 0 or width < 2 or height < 2 or width * height < 5:
        raise ValueError("Box needs nonnegative coordinates and dimensions of at least 2×3 pixels.")
    if x + width > image_width or y + height > image_height:
        raise ValueError("Box extends outside the image. Supply the final clamped GrabCut box.")
    if image_width * image_height - width * height < 5:
        raise ValueError("Use the classical rectangle, which leaves background outside the box.")
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def binary_mask(prediction: np.ndarray, image_shape) -> np.ndarray:
    """Convert a single SAM2 binary prediction to a full-resolution 0/255 mask."""
    prediction = np.asarray(prediction)
    if prediction.shape != (1, *image_shape[:2]):
        raise ValueError("SAM2 must return exactly one mask at the original image dimensions.")
    if not np.all((prediction == 0) | (prediction == 1)):
        raise ValueError("Expected binary SAM2 output, not logits or probabilities.")
    return prediction[0].astype(np.uint8) * 255


def boundary_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw every external foreground boundary on a copy; preserve the mask."""
    if (mask.shape != rgb.shape[:2] or mask.dtype != np.uint8
            or not np.all((mask == 0) | (mask == 255))):
        raise ValueError("Overlay requires a matching uint8 binary 0/255 mask.")
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = rgb.copy()
    # Keep disconnected components visible rather than choosing the largest one.
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def package_metadata() -> dict:
    """Record installed versions and the SAM2 source revision when available."""
    versions = {}
    for name in ("SAM-2", "torch", "torchvision", "numpy", "opencv-python-headless", "opencv-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        source = importlib.metadata.distribution("SAM-2").read_text("direct_url.json")
        revision = json.loads(source or "{}").get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
        revision = None
    return {"package_versions": versions, "sam2_revision": revision}


def _sam2_dependencies():
    """Import heavy optional dependencies only when SAM2 is requested."""
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "SAM2/PyTorch could not be imported. Run in the validated "
            "csc8830-combined environment with the official SAM2 package. "
            f"Original error: {error}"
        ) from error
    return torch, build_sam2, SAM2ImagePredictor


def select_device(device: str = "auto") -> str:
    """Prefer MPS, then CUDA, then CPU; reject unavailable explicit devices."""
    if device not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError("Device must be auto, mps, cuda, or cpu.")
    torch, _, _ = _sam2_dependencies()
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if ((device == "mps" and not torch.backends.mps.is_available())
            or (device == "cuda" and not torch.cuda.is_available())):
        raise ValueError(f"Requested device {device} is unavailable. Use cpu or auto.")
    return device


def load_sam2_model(
    checkpoint: str | Path, model_config: str = MODEL_CONFIG,
    device: str = "auto", *, model_name: str = MODEL_NAME,
) -> SAM2ModelResource:
    """Load once for reuse or external resource caching; never download weights."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    if not isinstance(model_config, str) or not model_config.strip():
        raise ValueError("SAM2 model configuration must be a nonempty string.")
    torch, build_sam2, _ = _sam2_dependencies()
    device = select_device(device)
    try:
        model = build_sam2(
            model_config, str(checkpoint), device=device, apply_postprocessing=True,
        )
        model.eval()
    except Exception as error:
        raise RuntimeError(f"Could not load SAM2 model on {device}: {error}") from error
    return SAM2ModelResource(model, model_name, model_config, checkpoint, device)


def run_sam2(
    image_rgb: np.ndarray, rectangle_xywh, model_resource: SAM2ModelResource,
) -> dict:
    """Infer from original RGB uint8 pixels and one exact XYWH box, in memory.

    Returns binary_mask (uint8 0/255), boundary_overlay (RGB uint8),
    predicted_mask_quality (the predictor's scalar score), and metadata.
    A valid all-background prediction is returned unchanged with empty_mask=True.
    Invalid inputs/results raise ValueError; dependency/inference failures raise
    RuntimeError. No resizing, rectangle clamping, or file I/O occurs here beyond
    SAM2's standard internal image transform and installed-package metadata reads.
    """
    if (not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8
            or image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.size == 0):
        raise ValueError("Image must be a nonempty H×W×3 RGB uint8 NumPy array.")
    xyxy = box_to_xyxy(rectangle_xywh, image_rgb.shape)
    if not isinstance(model_resource, SAM2ModelResource):
        raise ValueError("model_resource must be returned by load_sam2_model().")
    torch, _, predictor_class = _sam2_dependencies()
    try:
        # Serialize shared-model access, but never retain image state in the resource.
        with model_resource._lock, torch.inference_mode():
            predictor = predictor_class(
                model_resource.model, max_hole_area=0, max_sprinkle_area=0,
            )
            predictor.set_image(image_rgb)
            prediction = predictor.predict(
                box=xyxy, point_coords=None, point_labels=None, mask_input=None,
                multimask_output=False, return_logits=False,
            )
    except Exception as error:
        raise RuntimeError(f"SAM2 inference failed on {model_resource.device}: {error}") from error
    if not isinstance(prediction, (tuple, list)) or len(prediction) != 3:
        raise ValueError("SAM2 must return masks, quality scores, and low-resolution logits.")
    masks, scores, _ = prediction
    mask = binary_mask(masks, image_rgb.shape)
    scores = np.asarray(scores)
    if (scores.shape != (1,) or scores.dtype.kind not in "fiu"
            or not np.all(np.isfinite(scores))):
        raise ValueError("SAM2 must return one finite numeric predicted mask quality score.")
    quality = float(scores[0])
    metadata = {
        "model_name": model_resource.model_name,
        "model_config": model_resource.model_config,
        "checkpoint_path": str(model_resource.checkpoint_path),
        "checkpoint_filename": model_resource.checkpoint_path.name,
        "device": model_resource.device,
        "rectangle_xywh": list(map(int, rectangle_xywh)),
        "box_xywh": list(map(int, rectangle_xywh)),
        "box_xyxy": xyxy.astype(int).tolist(),
        "image_dimensions": {"width": image_rgb.shape[1], "height": image_rgb.shape[0]},
        "prompt_type": "box-only", "point_prompts": None, "mask_prompt": None,
        "extra_point_prompts_used": False, "mask_prompt_used": False,
        "multimask_output": False,
        "predicted_mask_quality": quality,
        "empty_mask": not bool(np.any(mask)),
        "postprocessing": {"stability_fallback": True, "max_hole_area": 0, "max_sprinkle_area": 0},
        "overlay": "All external contours, green, thickness 2; no mask editing",
        **package_metadata(),
    }
    return {
        "binary_mask": mask, "boundary_overlay": boundary_overlay(image_rgb, mask),
        "predicted_mask_quality": quality, "metadata": metadata,
    }


def save_artifacts(output_dir: Path, mask: np.ndarray, overlay: np.ndarray, metadata: dict):
    """Write lossless images and metadata without replacing a previous result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in OUTPUT_FILES):
        raise ValueError("Output artifacts already exist. Choose a new experiment directory.")
    for name, pixels in (
        ("sam2_mask.png", mask),
        ("sam2_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)),
    ):
        if not cv2.imwrite(str(output_dir / name), pixels):
            raise OSError(f"Could not write {output_dir / name}")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline SAM 2.1 Tiny box-only comparison.")
    parser.add_argument("--image", type=Path, required=True, help="Same image file uploaded to Streamlit.")
    parser.add_argument("--box", type=int, nargs=4, required=True, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                        help="Exact classical rectangle in original-image pixels (XYWH).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Existing official sam2.1_hiera_tiny.pt outside the repository.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New experiment output directory.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto", help="Default: prefer MPS, then CUDA, otherwise CPU.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rgb, image_hash = read_image(args.image)
        box_to_xyxy(args.box, rgb.shape)
        if args.output_dir.exists() and not args.output_dir.is_dir():
            raise ValueError("Output directory path refers to a file.")
        if any((args.output_dir / name).exists() for name in OUTPUT_FILES):
            raise ValueError("Output artifacts already exist. Choose a new experiment directory.")
        resource = load_sam2_model(args.checkpoint, device=args.device)
        print(f"Running {resource.model_name} on {resource.device} with a box-only prompt.")
        result = run_sam2(rgb, args.box, resource)
        mask, overlay = result["binary_mask"], result["boundary_overlay"]
        metadata = {
            **result["metadata"],
            "input_filename": args.image.name,
            "input_sha256": image_hash,
            "image_decode": "OpenCV IMREAD_COLOR, BGR converted to RGB",
            "outputs": {"mask": OUTPUT_FILES[0], "overlay": OUTPUT_FILES[1], "metadata": OUTPUT_FILES[2]},
        }
        save_artifacts(args.output_dir, mask, overlay, metadata)
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved comparison artifacts to {args.output_dir}")
    if not np.any(mask):
        print("No foreground was predicted; the saved overlay is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
