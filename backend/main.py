"""FastAPI application for the SIH26038 DR screening prototype."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_service import CLASS_NAMES, MODEL_ID, InvalidImageError, model_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    heatmap: str | None = Field(
        default=None,
        description=(
            "Grad-CAM overlay as a data: URI, or null when the loaded "
            "architecture does not support explanations."
        ),
    )


def configured_origins() -> list[str]:
    raw_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def _warm_up() -> None:
    """Load the model once at startup so the first request is not the slow one."""
    try:
        model_service.load()
    except Exception:
        # A failed warm-up must not stop the API from booting: /health reports
        # model_loaded=false and /predict retries the load per request.
        logger.exception("Model warm-up failed; the API will retry on first request.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("WARMUP_ON_STARTUP", "1") != "0":
        # Warm up off-thread so the server binds immediately instead of blocking
        # for the multi-second model load.
        Thread(target=_warm_up, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(
    title="SIH26038 Diabetic Retinopathy Prediction API",
    version="1.1.0",
    description=(
        "Prototype image-classification API. This model is not clinically validated "
        "and must not be used as a medical diagnosis."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": model_service.is_loaded,
        "explanations_available": model_service.supports_explanations,
        "classes": list(CLASS_NAMES),
        "clinical_use": False,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a JPEG, PNG, or WebP retinal image.",
        )

    image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        result = model_service.predict_bytes(image_bytes)
    except InvalidImageError as exc:
        logger.info("Rejected upload %r: %s", image.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed for upload %r", image.filename)
        raise HTTPException(
            status_code=503,
            detail="Model inference is temporarily unavailable.",
        ) from exc

    logger.info(
        "Predicted %s (%.3f) for upload %r", result.prediction, result.confidence, image.filename
    )
    return result.as_dict()
