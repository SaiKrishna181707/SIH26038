"""Inference service for the pretrained diabetic-retinopathy model."""

from __future__ import annotations

import logging
import os

# Keras reads this value during import, so it must be set before keras loads.
os.environ.setdefault("KERAS_BACKEND", "torch")

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import gradcam

logger = logging.getLogger(__name__)

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
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
# Fundus cameras top out well below this. The limit is checked against the image
# header before any pixels are decoded, so a decompression bomb is rejected
# before it can allocate memory.
MAX_IMAGE_PIXELS = 40_000_000
PROBABILITY_SUM_TOLERANCE = 1e-3


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as a supported image."""


class ExplanationUnsupportedError(RuntimeError):
    """Raised when the loaded architecture cannot produce a valid Grad-CAM map."""


@dataclass(frozen=True)
class Prediction:
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    heatmap: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "heatmap": self.heatmap,
        }


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert an image to the model's expected batched RGB float32 tensor.

    No rescaling or mean/std normalisation is applied here on purpose. This
    model wraps ``keras.applications.EfficientNetB0``, whose graph begins with
    ``Rescaling(1/255) -> Normalization(ImageNet stats) -> Rescaling``, so the
    network expects raw 0-255 values and normalising here would apply it twice.
    ``test_model_matches_expected_architecture`` pins that assumption.
    """
    rgb_image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(rgb_image, dtype=np.float32)
    return np.expand_dims(pixels, axis=0)


def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode one static JPEG/PNG/WebP and normalize its EXIF orientation."""
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty.")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            actual_format = (image.format or "").upper()
            if actual_format not in SUPPORTED_IMAGE_FORMATS:
                raise InvalidImageError(
                    "The uploaded file is not a supported JPEG, PNG, or WebP image."
                )
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                raise InvalidImageError("Animated images are not supported.")

            # Image.open only parses the header, so the size check happens
            # before the pixel buffer is allocated.
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(
                    f"The image is too large to process ({width}x{height} pixels)."
                )
            image.load()
            # Phone/camera files can store orientation in EXIF rather than in the
            # pixel matrix. Normalize it before preprocessing and Grad-CAM so the
            # displayed retina and model input have the same orientation.
            return ImageOps.exif_transpose(image).copy()
    except InvalidImageError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise InvalidImageError("The uploaded file is not a valid image.") from exc


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert a Keras backend tensor to numpy, passing numpy arrays straight through.

    The passthrough keeps the inference path usable with plain arrays, so the
    prediction and Grad-CAM logic can be exercised without the ML stack
    installed.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    import keras

    return keras.ops.convert_to_numpy(tensor)


def normalize_probabilities(raw_output: Any) -> np.ndarray:
    """Validate a single softmax output and absorb only tiny floating-point drift."""
    array = np.asarray(raw_output, dtype=np.float64)
    valid_shapes = {(len(CLASS_NAMES),), (1, len(CLASS_NAMES))}
    if array.shape not in valid_shapes:
        raise RuntimeError(
            "Expected one five-class model output with shape "
            f"(5,) or (1, 5), received {array.shape}."
        )

    probabilities = array.reshape(-1)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("The model returned non-finite probabilities.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("The model returned invalid probabilities outside [0, 1].")

    total = float(probabilities.sum())
    if total <= 0.0 or not np.isclose(
        total, 1.0, rtol=0.0, atol=PROBABILITY_SUM_TOLERANCE
    ):
        raise RuntimeError(
            f"The model probabilities must sum to 1; received {total:.6f}."
        )
    return probabilities / total


def split_backbone_and_head(model: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split a model into its convolutional backbone and classifier head.

    Grad-CAM's closed form (see :mod:`gradcam`) is only valid for a
    ``GlobalAveragePooling2D -> Dropout* -> Dense(softmax)`` head sitting on top
    of a single nested feature extractor. This validates that shape and raises
    :class:`ExplanationUnsupportedError` otherwise, so a mismatched architecture
    disables explanations instead of producing a meaningless map.
    """
    import keras

    layers = [
        layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)
    ]
    backbones = [layer for layer in layers if isinstance(layer, keras.Model)]
    if len(backbones) != 1:
        raise ExplanationUnsupportedError(
            f"Expected exactly one nested feature-extractor model, found {len(backbones)}."
        )

    backbone = backbones[0]
    head = tuple(layers[layers.index(backbone) + 1 :])
    if not head:
        raise ExplanationUnsupportedError("The model has no classifier head after the backbone.")

    feature_shape = backbone.output.shape
    if len(feature_shape) != 4:
        raise ExplanationUnsupportedError(
            f"Expected 4-D backbone feature maps, got shape {feature_shape}."
        )

    allowed = (
        keras.layers.GlobalAveragePooling2D,
        keras.layers.Dropout,
        keras.layers.Dense,
    )
    unexpected = [type(layer).__name__ for layer in head if not isinstance(layer, allowed)]
    if unexpected:
        raise ExplanationUnsupportedError(
            f"Unsupported head layers for Grad-CAM: {', '.join(unexpected)}."
        )

    pool_layers = [layer for layer in head if isinstance(layer, keras.layers.GlobalAveragePooling2D)]
    if len(pool_layers) != 1 or not isinstance(head[0], keras.layers.GlobalAveragePooling2D):
        raise ExplanationUnsupportedError(
            "Grad-CAM requires exactly one GlobalAveragePooling2D layer at the start of the head."
        )

    dense_layers = [layer for layer in head if isinstance(layer, keras.layers.Dense)]
    if len(dense_layers) != 1 or not isinstance(head[-1], keras.layers.Dense):
        raise ExplanationUnsupportedError(
            "Grad-CAM requires exactly one Dense layer, and it must be the output layer."
        )
    output_layer = dense_layers[0]
    if output_layer.units != len(CLASS_NAMES):
        raise ExplanationUnsupportedError(
            f"Expected a {len(CLASS_NAMES)}-unit output layer, got {output_layer.units}."
        )
    if output_layer.activation is not keras.activations.softmax:
        raise ExplanationUnsupportedError(
            "The classifier output must use softmax for the five-class probability contract."
        )
    return backbone, head


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
        self._backbone: Any | None = None
        self._head_layers: tuple[Any, ...] = ()
        self._class_weights: np.ndarray | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def supports_explanations(self) -> bool:
        return (
            self._backbone is not None
            and bool(self._head_layers)
            and self._class_weights is not None
        )

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # Imported lazily so the API, /health, and tests that inject a model
            # do not pay the multi-second keras + torch import, and so a broken
            # ML install degrades instead of preventing startup entirely.
            import keras

            source = (
                str(self.model_path)
                if self.model_path.is_file()
                else f"hf://{self.model_id}"
            )
            logger.info("Loading DR model from %s", source)
            # Inference does not need the saved training compile state. Disabling
            # JIT also avoids requiring the Visual C++ compiler on Windows CPUs.
            model = keras.saving.load_model(source, compile=False)
            model.jit_compile = False
            self._configure_explanations(model)
            self._model = model
            logger.info(
                "DR model ready (explanations %s)",
                "enabled" if self.supports_explanations else "disabled",
            )

    def _configure_explanations(self, model: Any) -> None:
        """Prepare Grad-CAM state, disabling explanations if setup is unsupported."""
        import keras

        try:
            backbone, head = split_backbone_and_head(model)
            class_weights = np.asarray(
                keras.ops.convert_to_numpy(head[-1].kernel), dtype=np.float64
            )
            channels = backbone.output.shape[-1]
            if channels is None or class_weights.shape != (channels, len(CLASS_NAMES)):
                raise ExplanationUnsupportedError(
                    f"Expected output weights of shape {(channels, len(CLASS_NAMES))}, "
                    f"got {class_weights.shape}."
                )
            if not np.all(np.isfinite(class_weights)):
                raise ExplanationUnsupportedError("The classifier weights contain non-finite values.")
        except ExplanationUnsupportedError as exc:
            logger.warning(
                "Grad-CAM explanations disabled for this model: %s "
                "Predictions will still be served.",
                exc,
            )
            self._backbone, self._head_layers, self._class_weights = None, (), None
            return
        except Exception:
            # Explanation setup is deliberately non-critical. A serialization or
            # backend-specific introspection failure should not make an otherwise
            # usable classifier unavailable.
            logger.exception(
                "Grad-CAM setup failed unexpectedly; predictions will continue without explanations."
            )
            self._backbone, self._head_layers, self._class_weights = None, (), None
            return

        self._backbone, self._head_layers = backbone, head
        self._class_weights = class_weights

    def _forward(self, batch: np.ndarray) -> tuple[Any, np.ndarray | None]:
        """Run inference, also returning backbone feature maps when available.

        When explanations are supported the head is applied to the backbone
        output layer-by-layer using the model's own layer objects, so the result
        is identical to ``model.predict`` by construction while making the
        feature maps Grad-CAM needs available from the same forward pass.
        """
        if not self.supports_explanations:
            return self._model.predict(batch, verbose=0), None

        features = self._backbone(batch, training=False)
        activations = features
        for layer in self._head_layers:
            activations = layer(activations, training=False)
        return (
            to_numpy(activations),
            np.asarray(to_numpy(features), dtype=np.float64)[0],
        )

    def predict_bytes(self, image_bytes: bytes, explain: bool = True) -> Prediction:
        image = decode_image(image_bytes)
        batch = preprocess_image(image)
        self.load()
        with self._predict_lock:
            raw_output, features = self._forward(batch)

        probabilities = normalize_probabilities(raw_output)
        best_index = int(np.argmax(probabilities))

        heatmap: str | None = None
        if explain and features is not None and self._class_weights is not None:
            try:
                heatmap = gradcam.build_heatmap_uri(
                    image, features, self._class_weights[:, best_index]
                )
            except Exception:
                # A missing heat map is a degraded answer; a failed request is not.
                logger.exception("Grad-CAM generation failed; returning prediction only.")

        return Prediction(
            prediction=CLASS_NAMES[best_index],
            confidence=float(probabilities[best_index]),
            probabilities={
                label: float(probabilities[index])
                for index, label in enumerate(CLASS_NAMES)
            },
            heatmap=heatmap,
        )

    def predict_file(self, image_path: str | Path, explain: bool = True) -> Prediction:
        return self.predict_bytes(Path(image_path).read_bytes(), explain=explain)


model_service = DRModelService()
