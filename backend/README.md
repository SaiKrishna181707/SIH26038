# SIH26038 AI Backend

FastAPI service and prediction CLI for five-class diabetic-retinopathy grading with the
pretrained [`Aldahmashi/DR-EfficientNetB0`](https://huggingface.co/Aldahmashi/DR-EfficientNetB0)
Keras model on the PyTorch backend, plus a Grad-CAM overlay for each prediction.

> **Prototype limitation:** this model is not clinically validated and must not be used
> for diagnosis or patient-care decisions. `/health` reports `clinical_use: false` for
> exactly this reason.

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate && python -m pip install -r requirements-dev.txt
```

On Linux/macOS the activate path is `.venv/bin/activate`. Python 3.11 or 3.12 is the
safe choice — the pinned `torch==2.8.0` has no wheel for every 3.13 platform, and none at
all for Windows on ARM64.

The model lives at `models/final_model.keras`. If that file is absent the first
prediction downloads it from Hugging Face. Set `MODEL_PATH` to use another local copy.

## Run the API

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs at `http://localhost:8000/docs`.

The model loads on a background thread at startup, so the server binds immediately and
the first request is not the slow one. A failed load does not stop the boot: `/health`
reports `model_loaded: false` and `/predict` retries the load per request.

### `GET /health`

```json
{
  "status": "ok",
  "model": "Aldahmashi/DR-EfficientNetB0",
  "model_loaded": true,
  "explanations_available": true,
  "classes": ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"],
  "clinical_use": false
}
```

The front end polls this to drive its status indicators, and uses `classes` to order the
probability bars.

### `POST /predict`

`multipart/form-data`, field name exactly `image`. JPEG, PNG, or WebP up to 10 MB.

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

`heatmap` is a 448×448 Grad-CAM overlay as a data URI — inlined rather than fetched
separately, because a second round trip is expensive on the connections this project
targets. It is `null` when the loaded architecture does not support explanations or when
overlay generation fails; a prediction is still returned in both cases.

Failure modes: `415` wrong content type, `413` over the byte limit, `422` undecodable or
oversized image, `503` inference failure (details are logged, not returned).

## Grad-CAM

`gradcam.py` needs no autograd, no GPU, and no ML framework — only numpy and Pillow.

The head of this network is `GlobalAveragePooling2D → Dropout → Dense(5)`, so for feature
map `A_k` of size H×W and class weight `W_kc`:

```
logit_c = Σ_k W_kc · mean(A_k) + b_c      ⇒      ∂logit_c/∂A_kij = W_kc / (H·W)
```

The gradient is spatially constant, so Grad-CAM's spatial average of gradients is
`α_kc = W_kc/(H·W)`, and after the standard `[0,1]` normalisation the `1/(H·W)` factor
cancels:

```
cam = relu(Σ_k W_kc · A_k)
```

That is not an approximation of Grad-CAM for this architecture — it is Grad-CAM, in
closed form. `tests/test_gradcam.py` verifies it against finite-difference numerical
gradients of the logit rather than taking the derivation on trust.

Two consequences worth stating: the weights come from the **pre-softmax logit** (softmax
gradients add a `−p_c·Σ` term that can flip the sign), and the map is intrinsically 7×7
because that is EfficientNetB0's final spatial resolution at 224×224 input. Upsampling to
448 makes it legible, not more precise.

`model_service.py` validates the architecture at load time. If the head is not a single
trailing `Dense` over a `GlobalAveragePooling2D`, it logs a warning, sets
`explanations_available: false`, and keeps serving predictions.

## Preprocessing

Feed **raw 0–255 float32**. The saved model normalises internally — its first layers are
`Rescaling(1/255) → Normalization(ImageNet stats) → Rescaling`. Dividing by 255 before
calling it applies the scaling twice and silently degrades accuracy.
`tests/test_real_model.py` asserts those layers are present, so this assumption fails
loudly if the weights are ever swapped for a model that expects normalised input.

## Local prediction

```bash
python predict.py path/to/retinal-image.jpg
```

```bash
python predict.py path/to/retinal-image.jpg --heatmap overlay.webp
```

Prints the class, confidence, and all five probabilities; `--heatmap` also writes the
decoded overlay. Exit codes: `0` success, `1` bad image, `2` inference failure.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `MODEL_PATH` | `models/final_model.keras` | Local weights to load instead of downloading. |
| `MAX_IMAGE_BYTES` | `10485760` | Upload size ceiling. |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed front-end origins. |
| `WARMUP_ON_STARTUP` | `1` | Set `0` to skip the background load (useful in tests). |
| `LOG_LEVEL` | `INFO` | Root log level. |
| `KERAS_BACKEND` | `torch` | Set before any Keras import; override only if you have a different backend installed. |

In production, set `CORS_ORIGINS` to the real front-end origin:

```bash
CORS_ORIGINS=https://frontend.example.com python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Verification

```bash
python -m pytest -q
```

`conftest.py` puts this directory on `sys.path`, so a bare `pytest` from the repo root
works too.

Most of the suite runs without a Keras backend at all — the ML imports are lazy and the
tests substitute fakes, so Grad-CAM maths, the API contract, image validation, and error
paths are all covered on any machine.

`tests/test_real_model.py` is the exception: it loads the actual weights and executes the
network. It skips cleanly when Keras or the weights are missing, so **run it at least
once on the machine you will demo from** — it is what proves that the manual head applied
in `_forward` is numerically identical to `model.predict`, and that the real overlay is
non-uniform.

```bash
python -m pytest tests/test_real_model.py -v
```
