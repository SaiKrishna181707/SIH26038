from __future__ import annotations

import asyncio
import os
os.environ.setdefault("WARMUP_ON_STARTUP", "0")

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event, Lock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import gradcam
import main
import model_service
from model_service import InvalidImageError, Prediction


def image_bytes(image_format="PNG", size=(32, 32), color="black"):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def result():
    return Prediction(
        prediction="No DR",
        confidence=0.8,
        probabilities={
            "No DR": 0.8,
            "Mild DR": 0.1,
            "Moderate DR": 0.05,
            "Severe DR": 0.03,
            "Proliferative DR": 0.02,
        },
    )


class ValidatingFakeService:
    is_loaded = True
    supports_explanations = False

    def predict_bytes(self, payload, explain=True):
        model_service.decode_image(payload)
        return result()


def test_valid_static_formats_decode():
    for fmt in ("JPEG", "PNG", "WEBP"):
        assert model_service.decode_image(image_bytes(fmt)).size == (32, 32)


def test_gif_is_rejected_even_if_bytes_are_decodable():
    with pytest.raises(InvalidImageError, match="supported JPEG"):
        model_service.decode_image(image_bytes("GIF"))


def test_api_rejects_mime_spoofed_gif(monkeypatch):
    monkeypatch.setattr(main, "model_service", ValidatingFakeService())
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"image": ("fake.png", image_bytes("GIF"), "image/png")},
        )
    assert response.status_code == 422


def test_exif_orientation_is_normalized():
    image = Image.new("RGB", (20, 10), "red")
    exif = image.getexif()
    exif[274] = 6  # 90 degrees clockwise
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    assert model_service.decode_image(buffer.getvalue()).size == (10, 20)


def test_animated_webp_is_rejected_when_encoder_supports_it():
    buffer = BytesIO()
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "blue")
    try:
        first.save(buffer, format="WEBP", save_all=True, append_images=[second], duration=100, loop=0)
    except OSError:
        pytest.skip("Pillow build has no animated WebP encoder")
    with pytest.raises(InvalidImageError, match="Animated"):
        model_service.decode_image(buffer.getvalue())


@pytest.mark.parametrize(
    "raw",
    [
        np.ones((5, 1)) / 5,
        np.ones((1, 1, 5)) / 5,
        np.ones((2, 5)) / 5,
    ],
)
def test_probability_shape_must_be_one_prediction(raw):
    with pytest.raises(RuntimeError, match="shape"):
        model_service.normalize_probabilities(raw)


@pytest.mark.parametrize(
    "raw",
    [
        [1, 1, 1, 1, 1],
        [0.4, 0.3, 0.2, 0.1, 0.1],
        [1.01, 0, 0, 0, 0],
        [-0.01, 0.51, 0.2, 0.2, 0.1],
        [np.nan, 0.25, 0.25, 0.25, 0.25],
        [np.inf, 0, 0, 0, 0],
    ],
)
def test_malformed_probability_values_fail_closed(raw):
    with pytest.raises(RuntimeError):
        model_service.normalize_probabilities(np.asarray(raw, dtype=float))


def test_small_probability_drift_is_normalized():
    raw = np.array([0.1999, 0.2, 0.2, 0.2, 0.2])
    normalized = model_service.normalize_probabilities(raw)
    assert normalized.sum() == pytest.approx(1.0)


def test_gradcam_rejects_nonfinite_features():
    features = np.ones((7, 7, 3))
    features[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        gradcam.class_activation_map(features, np.ones(3))


def test_gradcam_rejects_nonfinite_weights():
    weights = np.ones(3)
    weights[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        gradcam.class_activation_map(np.ones((7, 7, 3)), weights)


def test_render_overlay_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="max_alpha"):
        gradcam.render_overlay(Image.new("RGB", (10, 10)), np.ones((2, 2)), max_alpha=1.5)


def test_positive_int_env_falls_back_on_bad_values(monkeypatch):
    for raw in ("abc", "0", "-5"):
        monkeypatch.setenv("DEMO_LIMIT", raw)
        assert main._positive_int_env("DEMO_LIMIT", 7) == 7
    monkeypatch.setenv("DEMO_LIMIT", "3")
    assert main._positive_int_env("DEMO_LIMIT", 7) == 3


def test_declared_oversized_request_is_rejected_before_endpoint(monkeypatch):
    class MustNotRun:
        is_loaded = True
        supports_explanations = False
        def predict_bytes(self, *_args, **_kwargs):
            raise AssertionError("endpoint should not run")

    monkeypatch.setattr(main, "model_service", MustNotRun())
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            headers={"Content-Length": str(main.MAX_PREDICT_REQUEST_BYTES + 1)},
            files={"image": ("tiny.png", image_bytes(), "image/png")},
        )
    assert response.status_code == 413


def test_load_shedding_rejects_third_simultaneous_prediction(monkeypatch):
    entered = Event()
    release = Event()
    lock = Lock()
    state = {"count": 0}

    class BlockingService:
        is_loaded = True
        supports_explanations = False
        def predict_bytes(self, *_args, **_kwargs):
            with lock:
                state["count"] += 1
                if state["count"] >= main.MAX_CONCURRENT_PREDICTIONS:
                    entered.set()
            assert release.wait(timeout=5)
            return result()

    monkeypatch.setattr(main, "model_service", BlockingService())
    payload = image_bytes()

    with TestClient(main.app) as client, ThreadPoolExecutor(
        max_workers=main.MAX_CONCURRENT_PREDICTIONS
    ) as pool:
        futures = [
            pool.submit(
                client.post,
                "/predict",
                files={"image": (f"{index}.png", payload, "image/png")},
            )
            for index in range(main.MAX_CONCURRENT_PREDICTIONS)
        ]
        assert entered.wait(timeout=5), "concurrent requests never entered fake inference"
        busy = client.post(
            "/predict", files={"image": ("busy.png", payload, "image/png")}
        )
        assert busy.status_code == 429
        assert busy.headers.get("retry-after") == "1"
        release.set()
        assert [future.result(timeout=5).status_code for future in futures] == [200] * len(futures)


def test_health_stays_available_while_prediction_slots_are_full(monkeypatch):
    release = Event()
    started = Event()

    class BlockingService:
        is_loaded = True
        supports_explanations = False
        def predict_bytes(self, *_args, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return result()

    monkeypatch.setattr(main, "model_service", BlockingService())
    with TestClient(main.app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            client.post,
            "/predict",
            files={"image": ("one.png", image_bytes(), "image/png")},
        )
        assert started.wait(timeout=5)
        assert client.get("/health").status_code == 200
        release.set()
        assert pending.result(timeout=5).status_code == 200


def test_streamed_body_limit_works_without_content_length(monkeypatch):
    monkeypatch.setattr(main, "MAX_PREDICT_REQUEST_BYTES", 100)

    body = (
        b"--edge\r\n"
        b"Content-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\n"
        b"Content-Type: image/png\r\n\r\n"
        + b"x" * 200
        + b"\r\n--edge--\r\n"
    )

    def chunks():
        yield body[:80]
        yield body[80:160]
        yield body[160:]

    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            content=chunks(),
            headers={"Content-Type": "multipart/form-data; boundary=edge"},
        )
    assert response.status_code == 413


def test_load_slot_is_released_when_request_stream_fails():
    entered = []

    async def downstream(_scope, receive, _send):
        await receive()
        entered.append(True)

    middleware = main.PredictionLoadSheddingMiddleware(downstream, limit=1)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [],
    }

    async def scenario():
        async def broken_receive():
            raise RuntimeError("client stream failed")

        async def discard_send(_message):
            pass

        with pytest.raises(RuntimeError, match="client stream failed"):
            await middleware(scope, broken_receive, discard_send)

        messages = [{"type": "http.request", "body": b"", "more_body": False}]

        async def good_receive():
            return messages.pop(0)

        await middleware(scope, good_receive, discard_send)

    asyncio.run(scenario())
    assert entered == [True], "failed request leaked the only prediction slot"
