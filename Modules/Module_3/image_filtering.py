"""Gaussian image blurring using a kernel and explicit spatial convolution."""

import cv2
import numpy as np


def create_gaussian_kernel(kernel_size, sigma):
    """Return a normalized square Gaussian kernel with an odd size.

    kernel_size must be a positive odd integer, and sigma must be positive.
    A larger sigma spreads the weights farther from the center.
    """
    if not isinstance(kernel_size, (int, np.integer)) or isinstance(kernel_size, bool):
        raise ValueError("kernel_size must be a positive odd integer.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive, finite number.")

    # Place the origin at the center of the kernel.
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(coordinates, coordinates)

    # Sample the 2D Gaussian. Normalizing below makes its prefactor unnecessary.
    with np.errstate(over="ignore", under="ignore"):
        kernel = np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2))

    # Weights sum to approximately 1, preserving constant image brightness.
    kernel = kernel / kernel.sum()
    return kernel


def spatial_convolution(image, kernel):
    """Convolve a grayscale (H, W) or color (H, W, 3) image with a 2D kernel.

    The kernel must have positive odd dimensions. Reflect padding keeps the
    output the same size and avoids introducing a dark border from zero padding.
    Returns float64 values in the input's intensity scale, without rounding or
    clipping. Color channels are filtered independently, in their original order.
    Explicit loops make the process clear, but can be slow for large images.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a nonempty grayscale or three-channel image.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Color images must have three channels.")
    if kernel.ndim != 2 or any(size == 0 or size % 2 == 0 for size in kernel.shape):
        raise ValueError("kernel must be a nonempty 2D array with odd dimensions.")

    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    padded_image = cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101
    )

    # Convolution flips the kernel vertically and horizontally before applying it.
    # A symmetric Gaussian looks the same after flipping.
    flipped_kernel = kernel[::-1, ::-1]
    if image.ndim == 3:
        # Broadcast the same spatial weights across all three color channels.
        flipped_kernel = flipped_kernel[:, :, np.newaxis]

    output = np.zeros_like(image, dtype=np.float64)
    height, width = image.shape[:2]
    for row in range(height):
        for col in range(width):
            neighborhood = padded_image[
                row : row + kernel_height, col : col + kernel_width
            ]
            # Multiply neighboring pixels by their weights, then sum spatially.
            output[row, col] = np.sum(neighborhood * flipped_kernel, axis=(0, 1))

    return output


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Build a Gaussian kernel and blur an image using spatial convolution.

    Returns a float64 array with the same shape and intensity scale as image.
    """
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return spatial_convolution(image, kernel)


def gaussian_blur_frequency(image, kernel_size=5, sigma=1.0):
    """Blur a grayscale or three-channel image using the convolution theorem.

    Uses the same Gaussian kernel and reflected boundaries as gaussian_blur().
    Returns float64 values with the input's shape and intensity scale, without
    rounding or clipping. Results can differ from spatial convolution by tiny
    floating-point roundoff errors.
    """
    kernel = create_gaussian_kernel(kernel_size, sigma)
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a nonempty grayscale or three-channel image.")
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Color images must have three channels.")

    # Match spatial_convolution's boundary pixels exactly before taking any FFT.
    radius = kernel.shape[0] // 2
    padded_image = cv2.copyMakeBorder(
        image, radius, radius, radius, radius, cv2.BORDER_REFLECT_101
    )
    height, width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape

    # FFT multiplication normally gives circular convolution (edges wrap).
    # Zero-padding BOTH arrays to this full convolution size prevents wrapping.
    fft_shape = (
        padded_image.shape[0] + kernel_height - 1,
        padded_image.shape[1] + kernel_width - 1,
    )

    # Kernel Fourier transform: s=fft_shape adds zeros below and to the right.
    # Keep the kernel in the top-left corner, without shifting or flipping it.
    # The crop below accounts for its center and the reflected image border.
    kernel_fft = np.fft.fft2(kernel, s=fft_shape)

    # Treat grayscale as one channel so the same steps work for both formats.
    channels = 1 if image.ndim == 2 else image.shape[2]
    output = np.empty_like(image, dtype=np.float64)
    for channel in range(channels):
        plane = padded_image if image.ndim == 2 else padded_image[:, :, channel]

        # Image Fourier transform, zero-padded to the same size as the kernel.
        image_fft = np.fft.fft2(plane, s=fft_shape)

        # Multiplication in the frequency domain equals spatial convolution.
        filtered_fft = image_fft * kernel_fft

        # Inverse Fourier transform brings the filtered image back to space.
        # The inputs are real; discard imaginary roundoff left by the FFT.
        full_convolution = np.fft.ifft2(filtered_fft).real

        # Full convolution includes extra pixels from both kinds of padding.
        # Starting at kernel_size - 1 selects the same neighborhoods used by
        # spatial_convolution (radius for reflection + radius for kernel center).
        start_y = kernel_height - 1
        start_x = kernel_width - 1
        filtered_plane = full_convolution[
            start_y : start_y + height, start_x : start_x + width
        ]
        if image.ndim == 2:
            output[:, :] = filtered_plane
        else:
            output[:, :, channel] = filtered_plane

    return output
