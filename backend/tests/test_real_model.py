"""Integration tests that load the real Keras model.

These are the tests that catch the failure mode unit tests with fakes cannot:
a model whose architecture or preprocessing contract has silently drifted from
what ``model_service`` assumes.

They are skipped when the ML backend or the weights file is unavailable -- for
example on Windows ARM64, where no ``torch``/``jax``/``tensorflow`` wheel is
published. Run them on the machine you demo from:

    python -m pytest tests/test_real_model.py -v
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import model_service
from model_service import CLASS_NAMES, DEFAULT_LOCAL_MODEL, DRModelService, preprocess_image

keras = pytest.importorskip(
    "keras", reason="No Keras backend installed for this platform."
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_LOCAL_MODEL.is_file(),
    reason=f"Model weights not present at {DEFAULT_LOCAL_MODEL}.",
)


@pytest.fixture(scope="module")
def service() -> DRModelService:
    loaded = DRModelService()
    loaded.load()
    return loaded


@pytest.fixture(scope="module")
def fundus_bytes() -> bytes:
    """A synthetic fundus-like image: a bright disc on a dark background."""
    size = 512
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.hypot(yy - size / 2, xx - size / 2)
    disc = (radius < size * 0.45).astype(np.float64)
    image = np.zeros((size, size, 3), dtype=np.float64)
    image[..., 0] = disc * 170
    image[..., 1] = disc * 70
    image[..., 2] = disc * 45

    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_model_loads_and_supports_explanations(service):
    assert service.is_loaded is True
    assert service.supports_explanations is True, (
        "Grad-CAM was disabled, meaning the architecture no longer matches the "
        "global-average-pool + single-dense head the closed form requires."
    )


def test_model_matches_expected_architecture(service):
    """Pin the architecture that the preprocessing and Grad-CAM maths rely on."""
    backbone, head = model_service.split_backbone_and_head(service._model)

    assert service._model.input.shape[1:] == (224, 224, 3)
    assert len(backbone.output.shape) == 4
    assert isinstance(head[-1], keras.layers.Dense)
    assert head[-1].units == len(CLASS_NAMES)
    assert keras.activations.get(head[-1].activation) is keras.activations.softmax


def test_backbone_normalises_internally(service):
    """The reason preprocessing feeds raw 0-255 pixels rather than scaling them.

    If these layers ever disappear from the saved model, ``preprocess_image``
    must start normalising and this test is the tripwire.
    """
    backbone, _ = model_service.split_backbone_and_head(service._model)
    kinds = [type(layer).__name__ for layer in backbone.layers[:4]]

    assert "Rescaling" in kinds
    assert "Normalization" in kinds


def test_manual_head_matches_model_predict(service, fundus_bytes):
    """The Grad-CAM forward pass must be numerically identical to model.predict.

    ``_forward`` applies the head layers itself so it can keep the feature maps.
    This proves that shortcut has not changed the prediction.
    """
    batch = preprocess_image(model_service.decode_image(fundus_bytes))

    manual, features = service._forward(batch)
    reference = service._model.predict(batch, verbose=0)

    assert features.shape[-1] == service._class_weights.shape[0]
    assert np.asarray(manual).reshape(-1) == pytest.approx(
        np.asarray(reference).reshape(-1), abs=1e-5
    )


def test_end_to_end_prediction_is_well_formed(service, fundus_bytes):
    result = service.predict_bytes(fundus_bytes)

    assert result.prediction in CLASS_NAMES
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.confidence == pytest.approx(max(result.probabilities.values()))
    assert result.prediction == max(result.probabilities, key=result.probabilities.get)


def test_end_to_end_heatmap_is_a_decodable_overlay(service, fundus_bytes):
    from base64 import b64decode

    import gradcam

    result = service.predict_bytes(fundus_bytes)

    assert result.heatmap is not None and result.heatmap.startswith("data:image/")
    overlay = Image.open(BytesIO(b64decode(result.heatmap.partition(",")[2])))
    assert overlay.size == gradcam.OVERLAY_SIZE


def test_heatmap_is_not_uniform(service, fundus_bytes):
    """A constant map would mean the explanation carries no information."""
    from base64 import b64decode

    result = service.predict_bytes(fundus_bytes)
    overlay = np.asarray(
        Image.open(BytesIO(b64decode(result.heatmap.partition(",")[2]))).convert("RGB")
    )

    assert overlay.std() > 1.0


def test_repeated_predictions_are_deterministic(service, fundus_bytes):
    """Dropout must be inactive; two runs of the same image must agree exactly."""
    first = service.predict_bytes(fundus_bytes, explain=False)
    second = service.predict_bytes(fundus_bytes, explain=False)

    assert first.probabilities == second.probabilities
