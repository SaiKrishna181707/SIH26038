from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from model_service import CLASS_NAMES, DRModelService, InvalidImageError, decode_image, preprocess_image


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        return np.array([[0.02, 0.05, 0.90, 0.02, 0.01]], dtype=np.float32)


def image_bytes(mode="RGB", image_format="PNG"):
    buffer = BytesIO()
    Image.new(mode, (32, 48), color=128).save(buffer, format=image_format)
    return buffer.getvalue()


def test_preprocess_produces_expected_shape_and_type():
    batch = preprocess_image(Image.new("L", (50, 30), color=100))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32


def test_prediction_maps_all_five_classes():
    service = DRModelService()
    service._model = FakeModel()
    result = service.predict_bytes(image_bytes())
    assert result.prediction == "Moderate DR"
    assert result.confidence == pytest.approx(0.9)
    assert list(result.probabilities) == list(CLASS_NAMES)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_invalid_image_is_rejected():
    with pytest.raises(InvalidImageError):
        decode_image(b"not an image")

