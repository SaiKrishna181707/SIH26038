"""Tests for the Grad-CAM heat-map maths and rendering.

These are pure numpy/Pillow and need no model or ML backend installed.
"""

import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import gradcam


def textbook_gradcam(features, class_weights, bias=0.3):
    """Compute Grad-CAM the literal way, via numerical gradients of the logit.

    This deliberately does *not* use the closed form in :mod:`gradcam`. It
    differentiates the pre-softmax score with finite differences, averages the
    gradient spatially to get each channel weight, then combines and rectifies.
    Used to prove the closed form is equivalent rather than merely plausible.
    """

    def logit(feature_maps):
        pooled = feature_maps.mean(axis=(0, 1))
        return float(pooled @ class_weights + bias)

    step = 1e-5
    gradients = np.zeros_like(features, dtype=np.float64)
    for index in np.ndindex(features.shape):
        nudged_up = features.astype(np.float64).copy()
        nudged_down = features.astype(np.float64).copy()
        nudged_up[index] += step
        nudged_down[index] -= step
        gradients[index] = (logit(nudged_up) - logit(nudged_down)) / (2 * step)

    channel_weights = gradients.mean(axis=(0, 1))
    cam = np.maximum(np.tensordot(features, channel_weights, axes=([2], [0])), 0.0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def test_closed_form_matches_numerically_differentiated_gradcam():
    rng = np.random.default_rng(20260829)
    features = rng.normal(size=(3, 4, 6))
    class_weights = rng.normal(size=6)

    assert gradcam.class_activation_map(features, class_weights) == pytest.approx(
        textbook_gradcam(features, class_weights), abs=1e-6
    )


def test_closed_form_is_independent_of_the_output_bias():
    """The bias shifts the logit but not its gradient, so the map must not move."""
    rng = np.random.default_rng(7)
    features = rng.normal(size=(3, 3, 5))
    class_weights = rng.normal(size=5)

    assert textbook_gradcam(features, class_weights, bias=0.0) == pytest.approx(
        textbook_gradcam(features, class_weights, bias=12.5), abs=1e-6
    )


def test_map_is_normalised_and_rectified():
    features = np.array([[[1.0, 0.0], [0.0, 4.0]], [[2.0, 0.0], [0.0, 0.0]]])
    cam = gradcam.class_activation_map(features, np.array([1.0, -1.0]))

    assert cam.shape == (2, 2)
    assert cam.min() >= 0.0
    assert cam.max() == pytest.approx(1.0)
    # The position dominated by the negatively weighted channel is suppressed.
    assert cam[0, 1] == 0.0


def test_map_highlights_the_expected_region():
    features = np.zeros((4, 4, 2))
    features[3, 0, 0] = 5.0
    cam = gradcam.class_activation_map(features, np.array([1.0, 0.0]))

    assert np.unravel_index(int(np.argmax(cam)), cam.shape) == (3, 0)


def test_degenerate_map_is_all_zero_rather_than_nan():
    features = np.ones((2, 2, 3))
    cam = gradcam.class_activation_map(features, np.array([-1.0, -1.0, -1.0]))

    assert np.all(cam == 0.0)
    assert np.all(np.isfinite(cam))


@pytest.mark.parametrize(
    "features, weights",
    [
        (np.ones((2, 2)), np.ones(2)),           # feature maps not 3-D
        (np.ones((2, 2, 3)), np.ones(4)),        # weight count mismatch
        (np.ones((2, 2, 3)), np.ones((3, 1))),   # weights not 1-D
    ],
)
def test_invalid_shapes_are_rejected(features, weights):
    with pytest.raises(ValueError):
        gradcam.class_activation_map(features, weights)


def test_jet_ramp_spans_blue_to_red_and_clips():
    colors = gradcam.jet_colors(np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))

    assert colors.dtype == np.uint8
    assert colors.shape == (5, 3)
    # Out-of-range input clamps to the same colour as the range endpoints.
    assert tuple(colors[0]) == tuple(colors[1])
    assert tuple(colors[4]) == tuple(colors[3])
    # Low end is blue-dominant, high end red-dominant, middle is green-dominant.
    assert colors[1][2] > colors[1][0]
    assert colors[3][0] > colors[3][2]
    assert colors[2][1] >= max(colors[2][0], colors[2][2])


def test_upsampling_preserves_range_and_hits_target_size():
    rng = np.random.default_rng(3)
    cam = rng.random((7, 7))
    upsampled = gradcam.upsample_map(cam, (64, 48))

    assert upsampled.shape == (48, 64)  # numpy is (height, width)
    assert upsampled.min() >= 0.0
    assert upsampled.max() <= 1.0


def test_upsampling_keeps_the_hot_region_in_place():
    cam = np.zeros((7, 7))
    cam[0, 6] = 1.0  # top-right in (row, col)
    upsampled = gradcam.upsample_map(cam, (70, 70))
    row, col = np.unravel_index(int(np.argmax(upsampled)), upsampled.shape)

    assert row < 35 and col > 35


def test_overlay_tints_hot_regions_and_leaves_cold_ones_untouched():
    base = Image.new("RGB", (32, 32), (10, 20, 30))
    cam = np.zeros((4, 4))
    cam[0, 0] = 1.0  # hot in the top-left quadrant only

    overlay = gradcam.render_overlay(base, cam, size=(32, 32))
    pixels = np.asarray(overlay)

    assert overlay.size == (32, 32)
    assert pixels.dtype == np.uint8
    assert tuple(pixels[31, 31]) == (10, 20, 30)  # cold corner is the original
    assert tuple(pixels[0, 0]) != (10, 20, 30)  # hot corner is tinted


def test_overlay_respects_the_alpha_ceiling():
    """Even at peak attention the retina must stay partly visible underneath."""
    base = Image.new("RGB", (8, 8), (0, 0, 0))
    overlay = np.asarray(
        gradcam.render_overlay(base, np.ones((2, 2)), size=(8, 8), max_alpha=0.5)
    )
    pure_heat = gradcam.jet_colors(np.array(1.0)).astype(np.float64)

    assert np.all(overlay <= np.ceil(pure_heat * 0.5) + 1)


def test_overlay_resizes_to_the_requested_size():
    overlay = gradcam.render_overlay(
        Image.new("RGB", (100, 60), "white"), np.ones((7, 7)), size=(224, 224)
    )
    assert overlay.size == (224, 224)


def test_data_uri_decodes_back_to_an_image():
    uri = gradcam.encode_data_uri(Image.new("RGB", (16, 16), "red"))
    header, _, payload = uri.partition(",")

    assert header.startswith("data:image/")
    assert header.endswith(";base64")
    decoded = Image.open(BytesIO(base64.b64decode(payload)))
    assert decoded.size == (16, 16)


def test_build_heatmap_uri_end_to_end():
    rng = np.random.default_rng(11)
    features = rng.random((7, 7, 12))
    uri = gradcam.build_heatmap_uri(
        Image.new("RGB", (300, 300), (120, 40, 30)), features, rng.normal(size=12)
    )

    assert uri.startswith("data:image/")
    decoded = Image.open(BytesIO(base64.b64decode(uri.partition(",")[2])))
    assert decoded.size == gradcam.OVERLAY_SIZE
