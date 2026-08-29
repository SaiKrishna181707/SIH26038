"""Tests for image handling, probability validation, and the inference path.

The Keras model itself is faked here so these run without the ML stack. The real
model is exercised in ``test_real_model.py``.
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import model_service
from model_service import (
    CLASS_NAMES,
    DRModelService,
    InvalidImageError,
    Prediction,
    decode_image,
    normalize_probabilities,
    preprocess_image,
    to_numpy,
)

FEATURE_SHAPE = (7, 7, 8)


class FakeBackbone:
    """Stands in for the nested EfficientNetB0 feature extractor."""

    def __init__(self, features):
        self.features = features
        self.calls = 0

    def __call__(self, batch, training=None):
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        assert training is False, "inference must not run layers in training mode"
        self.calls += 1
        return self.features


class FakePool:
    def __call__(self, activations, training=None):
        assert training is False
        return activations.mean(axis=(1, 2))


class FakeDense:
    def __init__(self, kernel, bias):
        self.kernel = kernel
        self.bias = bias

    def __call__(self, activations, training=None):
        assert training is False
        logits = activations @ self.kernel + self.bias
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return shifted / shifted.sum(axis=-1, keepdims=True)


class FakeWholeModel:
    """A model whose head Grad-CAM cannot describe, so only predict() is used."""

    def __init__(self, output):
        self.output = output

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return self.output


def explainable_service(seed=0, peak_channel=2, peak_position=(1, 5)):
    """A service wired with fake layers that support Grad-CAM."""
    rng = np.random.default_rng(seed)
    features = rng.random((1, *FEATURE_SHAPE)) * 0.1
    features[0, peak_position[0], peak_position[1], peak_channel] = 9.0

    kernel = np.full((FEATURE_SHAPE[-1], len(CLASS_NAMES)), -0.1)
    kernel[peak_channel, 2] = 1.0

    service = DRModelService()
    service._model = object()
    service._backbone = FakeBackbone(features)
    service._head_layers = (FakePool(), FakeDense(kernel, np.zeros(len(CLASS_NAMES))))
    service._class_weights = kernel
    return service


def image_bytes(mode="RGB", image_format="PNG", size=(32, 48)):
    buffer = BytesIO()
    Image.new(mode, size, color=128).save(buffer, format=image_format)
    return buffer.getvalue()


# --- preprocessing ---------------------------------------------------------


def test_preprocess_produces_expected_shape_and_type():
    batch = preprocess_image(Image.new("L", (50, 30), color=100))

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_preprocess_does_not_rescale_pixels():
    """The model normalises internally, so values must stay in 0-255 here."""
    batch = preprocess_image(Image.new("RGB", (10, 10), (255, 255, 255)))

    assert batch.max() == pytest.approx(255.0)
    assert preprocess_image(Image.new("RGB", (10, 10), (0, 0, 0))).min() == 0.0


def test_preprocess_converts_grayscale_and_rgba_to_three_channels():
    for mode in ("L", "RGBA", "P"):
        assert preprocess_image(Image.new(mode, (20, 20))).shape == (1, 224, 224, 3)


# --- image decoding --------------------------------------------------------


def test_invalid_image_is_rejected():
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")


def test_empty_upload_is_rejected():
    with pytest.raises(InvalidImageError, match="empty"):
        decode_image(b"")


def test_oversized_image_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(model_service, "MAX_IMAGE_PIXELS", 100)

    with pytest.raises(InvalidImageError, match="too large"):
        decode_image(image_bytes(size=(64, 64)))


def test_supported_formats_decode():
    for image_format in ("PNG", "JPEG", "WEBP"):
        assert decode_image(image_bytes(image_format=image_format)).size == (32, 48)


# --- probability validation ------------------------------------------------


def test_probabilities_with_tiny_float_drift_are_renormalised():
    normalised = normalize_probabilities(
        np.array([[0.1999, 0.2, 0.2, 0.2, 0.2]])
    )

    assert normalised.sum() == pytest.approx(1.0)
    assert normalised.shape == (len(CLASS_NAMES),)


@pytest.mark.parametrize(
    "raw, message",
    [
        (np.ones((1, 3)), "one five-class model output"),
        (np.ones((5, 1)) / 5, "one five-class model output"),
        (np.array([[0.5, 0.5, np.nan, 0.0, 0.0]]), "non-finite"),
        (np.array([[-0.5, 0.5, 0.5, 0.5, 0.0]]), "outside"),
        (np.array([[1.1, 0.0, 0.0, 0.0, 0.0]]), "outside"),
        (np.zeros((1, 5)), "sum to 1"),
        (np.ones((1, 5)), "sum to 1"),
    ],
)
def test_malformed_model_output_is_rejected(raw, message):
    with pytest.raises(RuntimeError, match=message):
        normalize_probabilities(raw)


def test_to_numpy_passes_arrays_through():
    array = np.arange(4.0)
    assert to_numpy(array) is array


# --- prediction ------------------------------------------------------------


def test_prediction_maps_all_five_classes():
    result = explainable_service().predict_bytes(image_bytes())

    assert result.prediction == "Moderate DR"
    assert list(result.probabilities) == list(CLASS_NAMES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == pytest.approx(max(result.probabilities.values()))


def test_prediction_includes_a_gradcam_overlay():
    result = explainable_service().predict_bytes(image_bytes())

    assert result.heatmap is not None
    assert result.heatmap.startswith("data:image/")


def test_explanation_can_be_skipped():
    service = explainable_service()
    result = service.predict_bytes(image_bytes(), explain=False)

    assert result.heatmap is None
    assert result.prediction == "Moderate DR"
    assert service._backbone.calls == 1


def test_backbone_runs_once_per_prediction():
    service = explainable_service()
    service.predict_bytes(image_bytes())

    assert service._backbone.calls == 1


def test_heatmap_is_hottest_where_the_driving_channel_peaks():
    from base64 import b64decode

    service = explainable_service(peak_position=(0, 6))
    heatmap = service.predict_bytes(image_bytes()).heatmap
    overlay = np.asarray(
        Image.open(BytesIO(b64decode(heatmap.partition(",")[2]))).convert("RGB"),
        dtype=np.float64,
    )
    heat = overlay[..., 0] - overlay[..., 2]
    row, col = np.unravel_index(int(np.argmax(heat)), heat.shape)
    height, width = heat.shape

    assert row < height / 2
    assert col > width / 2


def test_service_without_explanation_support_falls_back_to_predict():
    service = DRModelService()
    service._model = FakeWholeModel(np.array([[0.02, 0.05, 0.03, 0.80, 0.10]]))

    result = service.predict_bytes(image_bytes())

    assert service.supports_explanations is False
    assert result.prediction == "Severe DR"
    assert result.heatmap is None


def test_heatmap_failure_degrades_to_a_prediction(monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("overlay backend unavailable")

    monkeypatch.setattr(model_service.gradcam, "build_heatmap_uri", explode)
    result = explainable_service().predict_bytes(image_bytes())

    assert result.prediction == "Moderate DR"
    assert result.heatmap is None


def test_predict_file_reads_from_disk(tmp_path):
    path = tmp_path / "fundus.png"
    path.write_bytes(image_bytes())

    assert explainable_service().predict_file(path).prediction == "Moderate DR"


def test_as_dict_exposes_the_api_contract():
    payload = Prediction(
        "No DR", 0.9, {"No DR": 0.9}, heatmap="data:image/webp;base64,AA"
    ).as_dict()

    assert set(payload) == {"prediction", "confidence", "probabilities", "heatmap"}


def test_is_loaded_and_supports_explanations_default_to_false():
    service = DRModelService()

    assert service.is_loaded is False
    assert service.supports_explanations is False
