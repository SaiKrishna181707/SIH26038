"""Grad-CAM class-activation heat maps for the DR classifier.

The classifier head is ``GlobalAveragePooling2D -> Dropout -> Dense(softmax)``,
which makes Grad-CAM reduce to a closed form that needs no automatic
differentiation. For final feature maps ``A`` (shape ``H x W x K``), pooling
``P_k = mean_ij A_kij``, and dense kernel ``W``, the pre-softmax score is::

    logit_c = sum_k W_kc * P_k + b_c

so the gradient of the score with respect to every spatial position is constant::

    d logit_c / d A_kij = W_kc / (H * W)

Grad-CAM's channel weight is the spatial mean of that gradient, so
``alpha_kc = W_kc / (H * W)`` and::

    cam = relu(sum_k alpha_kc * A_k) = relu(sum_k W_kc * A_k) / (H * W)

The ``1 / (H * W)`` factor is removed by the [0, 1] normalisation below, so
``relu(A @ W[:, c])`` is *exactly* Grad-CAM for this architecture -- not an
approximation of it.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

OVERLAY_SIZE = (448, 448)
MAX_ALPHA = 0.55
WEBP_QUALITY = 85


def class_activation_map(features: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Return the normalised Grad-CAM map for one image."""
    if features.ndim != 3:
        raise ValueError(f"Expected feature maps with shape (H, W, K), got {features.shape}.")
    if class_weights.ndim != 1 or class_weights.shape[0] != features.shape[-1]:
        raise ValueError(
            f"Expected {features.shape[-1]} class weights, got shape {class_weights.shape}."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature maps contain non-finite values.")
    if not np.all(np.isfinite(class_weights)):
        raise ValueError("Class weights contain non-finite values.")

    cam = np.tensordot(features, class_weights, axes=([2], [0])).astype(np.float64)
    np.maximum(cam, 0.0, out=cam)

    peak = float(cam.max())
    if peak <= 0.0:
        return np.zeros(cam.shape, dtype=np.float64)
    return cam / peak


def jet_colors(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to the conventional Grad-CAM 'jet' ramp as uint8 RGB."""
    if not np.all(np.isfinite(values)):
        raise ValueError("Heat-map values contain non-finite values.")
    scaled = np.clip(values, 0.0, 1.0) * 4.0
    red = np.clip(1.5 - np.abs(scaled - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(scaled - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(scaled - 1.0), 0.0, 1.0)
    stacked = np.stack([red, green, blue], axis=-1)
    return np.round(stacked * 255.0).astype(np.uint8)


def upsample_map(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Smoothly resize a small CAM to ``size`` (width, height), clipped to [0, 1]."""
    if cam.ndim != 2 or not np.all(np.isfinite(cam)):
        raise ValueError("Expected a finite two-dimensional heat map.")
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Invalid target size {size!r}.")
    # Bicubic interpolation overshoots on sharp edges, so clip afterwards.
    resized = Image.fromarray(cam.astype(np.float32), mode="F").resize(
        size, Image.Resampling.BICUBIC
    )
    return np.clip(np.asarray(resized, dtype=np.float64), 0.0, 1.0)


def render_overlay(
    image: Image.Image,
    cam: np.ndarray,
    size: tuple[int, int] = OVERLAY_SIZE,
    max_alpha: float = MAX_ALPHA,
) -> Image.Image:
    """Blend a CAM over ``image`` with opacity proportional to attention."""
    if not 0.0 <= max_alpha <= 1.0:
        raise ValueError("max_alpha must be between 0 and 1.")
    base = np.asarray(
        image.convert("RGB").resize(size, Image.Resampling.LANCZOS), dtype=np.float64
    )
    heat = upsample_map(cam, size)
    colors = jet_colors(heat).astype(np.float64)

    alpha = (heat * max_alpha)[..., np.newaxis]
    blended = base * (1.0 - alpha) + colors * alpha
    return Image.fromarray(np.round(blended).astype(np.uint8), mode="RGB")


def encode_data_uri(image: Image.Image) -> str:
    """Encode an image as a browser-ready ``data:`` URI, preferring WebP."""
    for image_format, mime in (("WEBP", "image/webp"), ("PNG", "image/png")):
        buffer = BytesIO()
        try:
            if image_format == "WEBP":
                image.save(buffer, format=image_format, quality=WEBP_QUALITY)
            else:
                image.save(buffer, format=image_format, optimize=True)
        except (OSError, KeyError, ValueError):
            continue
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    raise RuntimeError("Pillow could not encode the overlay as WebP or PNG.")


def build_heatmap_uri(
    image: Image.Image, features: np.ndarray, class_weights: np.ndarray
) -> str:
    """Produce the overlay data URI for one image and target class."""
    cam = class_activation_map(features, class_weights)
    return encode_data_uri(render_overlay(image, cam))
