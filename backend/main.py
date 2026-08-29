"""FastAPI application for the SIH26038 DR screening prototype."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from threading import BoundedSemaphore, Thread
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from model_service import CLASS_NAMES, MODEL_ID, InvalidImageError, model_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive integer environment variable without making import fragile."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d.", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d.", name, raw, default)
        return default
    return value


MAX_IMAGE_BYTES = _positive_int_env("MAX_IMAGE_BYTES", 10 * 1024 * 1024)
MAX_CONCURRENT_PREDICTIONS = _positive_int_env("MAX_CONCURRENT_PREDICTIONS", 2)
# Browser multipart requests add only a small amount of framing around the file.
# This generous allowance lets legitimate MAX_IMAGE_BYTES uploads through while
# rejecting obviously oversized Content-Length requests before Starlette parses
# or spools the multipart body.
MAX_MULTIPART_OVERHEAD_BYTES = _positive_int_env(
    "MAX_MULTIPART_OVERHEAD_BYTES", 512 * 1024
)
MAX_PREDICT_REQUEST_BYTES = MAX_IMAGE_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
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


def _header_value(scope: Scope, name: bytes) -> bytes | None:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return value
    return None


class PredictionLoadSheddingMiddleware:
    """Bound concurrent /predict requests before multipart parsing begins.

    Inference is intentionally serialized inside ``DRModelService``. Without an
    admission limit, a burst can still queue many request bodies and worker
    threads around that lock. Rejecting excess work early keeps health checks
    responsive and prevents a small demo machine from being memory-pressure
    killed under load.
    """

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self._slots = BoundedSemaphore(limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/predict"
        ):
            await self.app(scope, receive, send)
            return

        content_length = _header_value(scope, b"content-length")
        if content_length is not None:
            try:
                declared_bytes = int(content_length)
            except ValueError:
                declared_bytes = -1
            if declared_bytes > MAX_PREDICT_REQUEST_BYTES:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request exceeds the {MAX_IMAGE_BYTES / (1024 * 1024):g} MB "
                            "image limit."
                        )
                    },
                )
                await response(scope, receive, send)
                return

        if not self._slots.acquire(blocking=False):
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "1"},
                content={
                    "detail": (
                        "The screening service is busy. Wait for the current analysis "
                        "to finish and try again."
                    )
                },
            )
            await response(scope, receive, send)
            return

        try:
            # Buffer at most the admitted request ceiling before handing the body to
            # Starlette's multipart parser. This also enforces the limit for chunked
            # requests that intentionally omit Content-Length. The endpoint later
            # reads the file into memory for Pillow anyway, so this does not change
            # the asymptotic per-request memory bound; the concurrency gate above
            # caps how many such buffers can coexist.
            messages: list[dict[str, Any]] = []
            received_bytes = 0
            while True:
                message = await receive()
                messages.append(message)
                if message["type"] == "http.request":
                    received_bytes += len(message.get("body", b""))
                    if received_bytes > MAX_PREDICT_REQUEST_BYTES:
                        response = JSONResponse(
                            status_code=413,
                            headers={"Connection": "close"},
                            content={
                                "detail": (
                                    f"Request exceeds the {MAX_IMAGE_BYTES / (1024 * 1024):g} MB "
                                    "image limit."
                                )
                            },
                        )
                        await response(scope, receive, send)
                        return
                    if not message.get("more_body", False):
                        break
                elif message["type"] == "http.disconnect":
                    break

            message_index = 0

            async def replay_receive():
                nonlocal message_index
                if message_index < len(messages):
                    message = messages[message_index]
                    message_index += 1
                    return message
                return {"type": "http.disconnect"}

            await self.app(scope, replay_receive, send)
        finally:
            self._slots.release()


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
    version="1.2.0",
    description=(
        "Prototype image-classification API. This model is not clinically validated "
        "and must not be used as a medical diagnosis."
    ),
    lifespan=lifespan,
)
# Add load shedding first and CORS second so CORS remains the outer middleware
# and busy/oversized responses still carry the expected browser headers.
app.add_middleware(PredictionLoadSheddingMiddleware, limit=MAX_CONCURRENT_PREDICTIONS)
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
        "max_concurrent_predictions": MAX_CONCURRENT_PREDICTIONS,
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
            status_code=413,
            detail=(
                f"Image exceeds the {MAX_IMAGE_BYTES / (1024 * 1024):g} MB limit."
            ),
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
        "Predicted %s (%.3f) for upload %r",
        result.prediction,
        result.confidence,
        image.filename,
    )
    return result.as_dict()
