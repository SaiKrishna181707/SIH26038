"""Inference service for the pretrained diabetic-retinopathy model."""

from __future__ import annotations

import os

# Keras reads this value during import, so it must be set first.
os.environ.setdefault("KERAS_BACKEND", "torch")

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import keras
import numpy as np
from PIL import Image, UnidentifiedImageError

MODEL_ID = "Aldahmashi/DR-EfficientNetB0"
DEFAULT_LOCAL_MODEL = Path(__file__).resolve().parent / "models" / "final_model.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = (
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
)


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a supported image."""


@dataclass(frozen=True)
class Prediction:
    prediction: str
    confidence: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an image to the model's expected batched RGB float32 tensor."""
    rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb_image, dtype=np.float32)
    return np.expand_dims(pixels, axis=0)


def decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc


class DRModelService:
    """Thread-safe lazy loader and predictor for the Hugging Face Keras model."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        configured_path = model_path or os.getenv("MODEL_PATH", str(DEFAULT_LOCAL_MODEL))
        self.model_path = Path(configured_path)
        self._model: Any | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                source = (
                    str(self.model_path)
                    if self.model_path.is_file()
                    else f"hf://{self.model_id}"
                )
                # Inference does not need the saved training compile state. Disabling
                # JIT also avoids requiring the Visual C++ compiler on Windows CPUs.
                self._model = keras.saving.load_model(source, compile=False)
                self._model.jit_compile = False

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        image = decode_image(image_bytes)
        batch = preprocess_image(image)
        self.load()
        with self._predict_lock:
            raw_output = self._model.predict(batch, verbose=0)

        probabilities = np.asarray(raw_output, dtype=np.float64).reshape(-1)
        if probabilities.size != len(CLASS_NAMES):
            raise RuntimeError(
                f"Expected {len(CLASS_NAMES)} model outputs, received {probabilities.size}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("The model returned non-finite probabilities.")

        # The published model has a softmax output. Normalize defensively to absorb
        # harmless floating-point drift without hiding malformed negative outputs.
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise RuntimeError("The model returned invalid probabilities.")
        probabilities = probabilities / probabilities.sum()
        best_index = int(np.argmax(probabilities))
        probability_map = {
            label: float(probabilities[index])
            for index, label in enumerate(CLASS_NAMES)
        }
        return Prediction(
            prediction=CLASS_NAMES[best_index],
            confidence=float(probabilities[best_index]),
            probabilities=probability_map,
        )

    def predict_file(self, image_path: str | Path) -> Prediction:
        return self.predict_bytes(Path(image_path).read_bytes())


model_service = DRModelService()
