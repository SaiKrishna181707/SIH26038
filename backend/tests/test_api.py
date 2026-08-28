from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import main
from model_service import Prediction


class FakeService:
    is_loaded = True

    def predict_bytes(self, image_bytes):
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
        )


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (16, 16), "black").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["clinical_use"] is False


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(main, "model_service", FakeService())
    client = TestClient(main.app)
    response = client.post(
        "/predict", files={"image": ("retina.png", png_bytes(), "image/png")}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "Moderate DR"
    assert response.json()["confidence"] == 0.91
    assert len(response.json()["probabilities"]) == 5


def test_predict_rejects_non_image_content_type():
    client = TestClient(main.app)
    response = client.post(
        "/predict", files={"image": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415

