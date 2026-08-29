"""Tests for the HTTP contract the frontend depends on."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from model_service import Prediction

HEATMAP_URI = "data:image/webp;base64,AAAA"


class FakeService:
    is_loaded = True
    supports_explanations = True

    def __init__(self, heatmap=HEATMAP_URI):
        self.heatmap = heatmap

    def predict_bytes(self, image_bytes, explain=True):
        assert image_bytes
        return Prediction(
            prediction="Moderate DR",
            confidence=0.91,
            probabilities={
                "No DR": 0.02,
                "Mild DR": 0.04,
                "Moderate DR": 0.91,
                "Severe DR": 0.02,
                "Proliferative DR": 0.01,
            },
            heatmap=self.heatmap,
        )


def png_bytes(size=(16, 16)):
    buffer = BytesIO()
    Image.new("RGB", size, "black").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def fake_client(monkeypatch, client):
    monkeypatch.setattr(main, "model_service", FakeService())
    return client


def test_health_endpoint(client):
    response = client.get("/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["clinical_use"] is False
    assert payload["classes"] == [
        "No DR",
        "Mild DR",
        "Moderate DR",
        "Severe DR",
        "Proliferative DR",
    ]
    assert "model_loaded" in payload
    assert "explanations_available" in payload


def test_predict_endpoint(fake_client):
    response = fake_client.post(
        "/predict", files={"image": ("retina.png", png_bytes(), "image/png")}
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["prediction"] == "Moderate DR"
    assert payload["confidence"] == 0.91
    assert len(payload["probabilities"]) == 5
    assert payload["heatmap"] == HEATMAP_URI


def test_predict_response_omits_no_fields_when_heatmap_is_absent(monkeypatch, client):
    monkeypatch.setattr(main, "model_service", FakeService(heatmap=None))
    response = client.post(
        "/predict", files={"image": ("retina.png", png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json()["heatmap"] is None


@pytest.mark.parametrize("content_type", ["image/jpeg", "image/png", "image/webp"])
def test_predict_accepts_documented_content_types(fake_client, content_type):
    response = fake_client.post(
        "/predict", files={"image": ("retina", png_bytes(), content_type)}
    )
    assert response.status_code == 200


def test_predict_rejects_non_image_content_type(client):
    response = client.post(
        "/predict", files={"image": ("notes.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 415
    assert "JPEG" in response.json()["detail"]


def test_predict_requires_the_image_field(client):
    assert client.post("/predict", files={"photo": ("r.png", png_bytes(), "image/png")}).status_code == 422


def test_predict_rejects_undecodable_bytes(client):
    """An honest content type over garbage bytes must be a 422, not a 500."""
    response = client.post(
        "/predict", files={"image": ("retina.png", b"definitely not a png", "image/png")}
    )

    assert response.status_code == 422
    assert "not a valid image" in response.json()["detail"]


def test_predict_rejects_oversized_uploads(monkeypatch, client):
    monkeypatch.setattr(main, "MAX_IMAGE_BYTES", 512)
    response = client.post(
        "/predict", files={"image": ("big.png", png_bytes(size=(512, 512)), "image/png")}
    )

    assert response.status_code == 413


def test_inference_failure_returns_503_without_leaking_internals(monkeypatch, client):
    class BrokenService:
        is_loaded = False
        supports_explanations = False

        def predict_bytes(self, image_bytes, explain=True):
            raise RuntimeError("CUDA kernel exploded at 0xdeadbeef")

    monkeypatch.setattr(main, "model_service", BrokenService())
    response = client.post(
        "/predict", files={"image": ("retina.png", png_bytes(), "image/png")}
    )

    assert response.status_code == 503
    assert "deadbeef" not in response.text


def test_cors_allows_the_vite_dev_origin(client):
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_configured_origins_parses_and_trims(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", " https://a.example , https://b.example ,, ")

    assert main.configured_origins() == ["https://a.example", "https://b.example"]


def test_warm_up_failure_is_swallowed(monkeypatch):
    """Startup must survive a broken model install so /health can report it."""
    def explode():
        raise RuntimeError("no torch wheel for this platform")

    monkeypatch.setattr(main.model_service, "load", explode)
    main._warm_up()  # must not raise


def test_app_starts_with_warmup_disabled(monkeypatch):
    monkeypatch.setenv("WARMUP_ON_STARTUP", "0")

    with TestClient(main.app) as started:
        assert started.get("/health").status_code == 200
