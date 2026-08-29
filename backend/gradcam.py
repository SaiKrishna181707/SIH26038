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
approximation of it. Consequences worth knowing:

* only a forward pass is required, so this runs on any Keras backend and adds
  no measurable cost on top of the prediction that already ran;
* the maths is plain linear algebra, so it is unit-testable without a model.

The identity holds only for a global-average-pool + single-dense head.
``model_service`` validates that shape at load time and disables explanations
rather than emitting a heat map from an architecture this does not describe.

Gradients are taken on the pre-softmax logit, which is standard: backpropagating
the softmax probability instead mixes in a ``-p_c * sum_k(...)`` term that can
flip the sign of the map.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

# The heat map is intrinsically 7x7, so rendering above the model's 224px input
# costs nothing and keeps the fundus itself from looking soft when the UI scales
# the overlay up into a large panel.
OVERLAY_SIZE = (448, 448)
# Peak attention reaches this opacity; low-attention areas stay fully transparent
# so the untouched retina remains readable underneath.
MAX_ALPHA = 0.55
# WebP keeps the response near ~30 KB instead of the ~250 KB a PNG of the same
# overlay costs. This project targets low-bandwidth rural deployments, and the
# overlay is a visualisation rather than source data, so lossy encoding is fine.
WEBP_QUALITY = 85


def class_activation_map(features: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Return the normalised Grad-CAM map for one image.

    Args:
        features: final feature maps, shape ``(H, W, K)``.
        class_weights: dense weights for the target class, shape ``(K,)``.

    Returns:
        Array of shape ``(H, W)`` scaled to [0, 1]. An all-zero map is returned
        when no position carries positive evidence, which is degenerate but not
        an error.
    """
    if features.ndim != 3:
        raise ValueError(f"Expected feature maps with shape (H, W, K), got {features.shape}.")
    if class_weights.ndim != 1 or class_weights.shape[0] != features.shape[-1]:
        raise ValueError(
            f"Expected {features.shape[-1]} class weights, got shape {class_weights.shape}."
        )

    cam = np.tensordot(features, class_weights, axes=([2], [0])).astype(np.float64)
    np.maximum(cam, 0.0, out=cam)

    peak = cam.max()
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(cam.shape, dtype=np.float64)
    return cam / peak


def jet_colors(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to the conventional Grad-CAM 'jet' ramp as uint8 RGB."""
    scaled = np.clip(values, 0.0, 1.0) * 4.0
    red = np.clip(1.5 - np.abs(scaled - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(scaled - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(scaled - 1.0), 0.0, 1.0)
    stacked = np.stack([red, green, blue], axis=-1)
    return np.round(stacked * 255.0).astype(np.uint8)


def upsample_map(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Smoothly resize a small CAM to ``size`` (width, height), clipped to [0, 1]."""
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
            # Pillow can be built without WebP support; fall through to PNG.
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
