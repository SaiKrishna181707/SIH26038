# SIH26038 AI Backend

FastAPI service and prediction CLI for five-class diabetic-retinopathy grading with the
pretrained `Aldahmashi/DR-EfficientNetB0` Keras model on the PyTorch backend, plus a
Grad-CAM overlay for each prediction.

> **Prototype limitation:** this model is not clinically validated and must not be used
> for diagnosis or patient-care decisions. `/health` reports `clinical_use: false`.

## Setup

```bash
python -m venv .venv
# Windows: .venv/Scripts/activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Python 3.11 or 3.12 is the safest choice for the pinned ML stack. The committed model is
`models/final_model.keras`; set `MODEL_PATH` to use another local copy. If the configured
file does not exist, Keras falls back to `hf://Aldahmashi/DR-EfficientNetB0`.

## Run the API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: `http://localhost:8000/docs`.

The model warms on a background thread. A failed warm-up does not prevent boot;
`/health` remains available and `/predict` retries model loading.

### `GET /health`

```json
{
  "status": "ok",
  "model": "Aldahmashi/DR-EfficientNetB0",
  "model_loaded": true,
  "explanations_available": true,
  "classes": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
  "clinical_use": false,
  "max_concurrent_predictions": 2
}
```

### `POST /predict`

`multipart/form-data`, field name exactly `image`. The declared MIME type and the actual
file must both resolve to a **static JPEG, PNG, or WebP**. Default image limit: 10 MB.
EXIF orientation is normalized before inference.

```json
{
  "prediction": "Moderate DR",
  "confidence": 0.91,
  "probabilities": {
    "No DR": 0.02,
    "Mild DR": 0.04,
    "Moderate DR": 0.91,
    "Severe DR": 0.02,
    "Proliferative DR": 0.01
  },
  "heatmap": "data:image/webp;base64,UklGR..."
}
```

`heatmap` is a 448×448 Grad-CAM overlay data URI. It is `null` if explanation setup or
rendering is unavailable; prediction still succeeds.

Failure modes: `415` wrong declared content type, `413` request/image too large, `422`
invalid/unsupported/animated image, `429` prediction capacity full, `503` model/inference
failure. Internal exception details are logged, not returned.

## Load protection

The model service serializes inference because the backend/model combination is not
assumed to be safe under parallel prediction. The HTTP layer additionally limits how many
`/predict` requests may be active/queued before multipart parsing. Excess requests get
`429 Too Many Requests` and `Retry-After: 1`; `/health` is not subject to this gate.

The request-size ceiling is enforced twice: obvious oversized `Content-Length` values are
rejected immediately, and streamed/chunked bodies are counted before Starlette parses the
multipart payload. This prevents an attacker from bypassing the ceiling by omitting
`Content-Length`.

## Grad-CAM

The classifier head is `GlobalAveragePooling2D → Dropout* → Dense(5, softmax)`. For final
feature maps `A` and dense class weights `W`, Grad-CAM reduces exactly to:

```
cam = relu(A @ W[:, class])
```

up to a positive spatial constant removed by `[0,1]` normalization. `model_service.py`
validates that runtime architecture before enabling explanations. Non-finite features or
weights are rejected and explanation failure degrades to `heatmap: null`.

## Preprocessing and output contract

The saved EfficientNet model normalizes internally, so preprocessing feeds raw `0..255`
`float32` RGB pixels resized to 224×224. The output must be exactly one five-class softmax
vector with shape `(5,)` or `(1, 5)`, finite values in `[0,1]`, and a sum within `1e-3` of
`1`. Invalid model output fails closed instead of being reshaped/renormalized into a
plausible answer.

## Local prediction

```bash
python predict.py path/to/retinal-image.jpg
python predict.py path/to/retinal-image.jpg --heatmap overlay.webp
```

## Configuration

| Variable | Default | Effect |
|---|---:|---|
| `MODEL_PATH` | `models/final_model.keras` | Local weights path. |
| `MAX_IMAGE_BYTES` | `10485760` | Image byte ceiling. Invalid/non-positive values fall back safely. |
| `MAX_CONCURRENT_PREDICTIONS` | `2` | Active/queued `/predict` admission limit. |
| `MAX_MULTIPART_OVERHEAD_BYTES` | `524288` | Multipart framing allowance above the image limit. |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed frontend origins. |
| `WARMUP_ON_STARTUP` | `1` | Set `0` to skip background model warm-up. |
| `LOG_LEVEL` | `INFO` | Root log level. |
| `KERAS_BACKEND` | `torch` | Keras backend selected before import. |

## Verification

```bash
python -m pytest -q
python -m pytest tests/test_real_model.py -v
```

Most tests run without loading Keras. `test_real_model.py` executes the committed network,
checks the preprocessing/architecture assumptions, confirms manual-head predictions match
`model.predict`, and verifies the overlay is decodable and non-uniform. GitHub Actions
installs the full pinned ML stack on Linux x64 so this test runs on every pull request.
