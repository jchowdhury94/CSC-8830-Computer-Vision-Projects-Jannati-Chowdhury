"""Lightweight SAM2 checkpoint provisioning and CPU demo policy.

No torch/SAM2 imports or network activity occur on import. Operator overrides:
SAM2_CHECKPOINT (optional local file), SAM2_CACHE_DIR (writable runtime directory),
SAM2_DEVICE (auto/cpu/mps/cuda), SAM2_CPU_MAX_PIXELS (0 disables the CPU guard on
a separately measured, sufficiently provisioned host). Never resize inputs here.
"""

import hashlib
import os
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/" + CHECKPOINT_NAME
)
# Meta's immutable official Hugging Face LFS pointer publishes both values:
# https://huggingface.co/facebook/sam2.1-hiera-tiny/commit/36f406a75c9be63c7f429da63246273f028c6fd4
CHECKPOINT_SHA256 = "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69"
CHECKPOINT_BYTES = 156_008_466
# Local CPU measurements (2 threads): 1.47 MP peaked at 1286 MiB RSS; 29.04 MP
# at 1581 MiB, both excluding Streamlit and retained classical/session results.
# Bound per-session arrays to the tested moderate size; this is NOT a RAM guarantee.
CPU_MAX_PIXELS = 1_500_000
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_DEADLINE_SECONDS = 180


class CheckpointProvisionError(RuntimeError):
    """Recoverable checkpoint preparation failure; technical detail is for logs."""


class SAM2ResourceLimitError(ValueError):
    """An explicit size policy refused inference without changing the image."""


def checkpoint_configuration() -> tuple:
    """Stable cache identity: do not change it just because a download completes."""
    local = Path(os.environ.get(
        "SAM2_CHECKPOINT", str(Path.home() / "Models" / "sam2" / CHECKPOINT_NAME),
    )).expanduser().resolve()
    cache = Path(os.environ.get(
        "SAM2_CACHE_DIR", str(Path(tempfile.gettempdir()) / "csc8830-sam2"),
    )).expanduser().resolve()
    try:
        info = local.stat()
        identity = (info.st_size, info.st_mtime_ns)
    except OSError:
        identity = None
    return str(local), str(cache), identity, CHECKPOINT_SHA256


def validate_checkpoint(path: Path) -> None:
    """Verify exact published size and SHA-256 without deserializing weights."""
    if path.stat().st_size != CHECKPOINT_BYTES:
        raise CheckpointProvisionError("SAM2 checkpoint size does not match the official weights.")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINT_SHA256:
        raise CheckpointProvisionError("SAM2 checkpoint SHA-256 verification failed.")


def _download_checkpoint(destination: Path) -> None:
    """Write a unique partial file; publish it only after full verification."""
    partial = None
    try:
        started = time.monotonic()
        request = Request(CHECKPOINT_URL, headers={"User-Agent": "CSC8830-SAM2/1.0"})
        with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise CheckpointProvisionError("SAM2 download returned an unexpected HTTP status.")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != CHECKPOINT_BYTES:
                raise CheckpointProvisionError("SAM2 download has an unexpected Content-Length.")
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=CHECKPOINT_NAME + ".", suffix=".part", delete=False,
            ) as target:
                partial = Path(target.name)
                received = 0
                while True:
                    if time.monotonic() - started > DOWNLOAD_DEADLINE_SECONDS:
                        raise CheckpointProvisionError("SAM2 download exceeded its time limit.")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > CHECKPOINT_BYTES:
                        raise CheckpointProvisionError("SAM2 download exceeded the expected size.")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        validate_checkpoint(partial)
        os.replace(partial, destination)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def resolve_checkpoint(local_checkpoint: str | Path, cache_dir: str | Path) -> Path:
    """Reuse verified local weights, otherwise provision the official runtime cache.

    A cross-process file lock prevents concurrent downloads. Invalid runtime cache
    files are replaced, but a corrupt user-provided local file is never modified.
    Files are reverified on resolver calls; Streamlit caches the loaded model so
    ordinary page reruns do not call this function at all.
    """
    try:
        local = Path(local_checkpoint).expanduser().resolve()
        if local.is_file():
            validate_checkpoint(local)
            return local
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / CHECKPOINT_NAME
        from filelock import FileLock

        with FileLock(str(destination) + ".lock", timeout=240):
            if destination.is_file():
                try:
                    validate_checkpoint(destination)
                    return destination
                except CheckpointProvisionError:
                    destination.unlink()
            _download_checkpoint(destination)
            return destination
    except CheckpointProvisionError:
        raise
    except Exception as error:
        raise CheckpointProvisionError(f"Could not prepare the official SAM2 checkpoint: {error}") from error


def cpu_pixel_limit() -> int:
    """Default matches the measured moderate fixture; 0 is an explicit opt-out."""
    limit = int(os.environ.get("SAM2_CPU_MAX_PIXELS", str(CPU_MAX_PIXELS)))
    if limit < 0:
        raise ValueError("SAM2_CPU_MAX_PIXELS must be nonnegative.")
    return limit


def check_image_budget(image_shape, device: str, max_cpu_pixels: int) -> None:
    """Apply only to SAM2 on CPU; classical processing and accelerated runs remain unchanged."""
    height, width = image_shape[:2]
    if device == "cpu" and max_cpu_pixels and height * width > max_cpu_pixels:
        raise SAM2ResourceLimitError(
            f"SAM2 on this CPU host is limited to {max_cpu_pixels:,} pixels per image "
            f"to reduce memory use. Your image is {width:,} × {height:,} "
            f"({width * height:,} pixels). No resizing was performed. "
            "Upload a smaller image to run SAM2 and compare both methods on it. "
            "The classical method remains available for the current image."
        )
