"""FastAPI application for the SIH26038 DR screening prototype."""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_service import CLASS_NAMES, MODEL_ID, InvalidImageError, model_service

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


def configured_origins() -> list[str]:
    raw_origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    )
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


app = FastAPI(
    title="SIH26038 Diabetic Retinopathy Prediction API",
    version="1.0.0",
    description=(
        "Prototype image-classification API. This model is not clinically validated "
        "and must not be used as a medical diagnosis."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": model_service.is_loaded,
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
        return model_service.predict_bytes(image_bytes).as_dict()
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Model inference is temporarily unavailable.",
        ) from exc

