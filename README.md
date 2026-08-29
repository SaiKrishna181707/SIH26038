# SIH26038 — Explainable AI for Diabetic Retinopathy Screening in Rural India

A two-part screening console: a FastAPI service that grades a fundus image into five
diabetic-retinopathy classes and returns a Grad-CAM overlay, and a React front end that
uploads the image and renders the grade, the class distribution, and the overlay.

> **Not a medical device.** The model is not clinically validated. Output is a
> screening aid for a trained reviewer, never a diagnosis.

```
SIH26038/
├── backend/     FastAPI service, Grad-CAM, prediction CLI, tests
└── frontend/    React + Vite screening console
```

## Run it

Two terminals. Backend first — the front end polls `/health` and will show
"Service offline" until it is up.

```bash
cd backend && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt && python -m uvicorn main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173. The Vite dev server proxies `/api` to
`http://localhost:8000`, so no CORS configuration is needed in development.

## How the two halves connect

| | |
|---|---|
| Contract | `POST /predict`, `multipart/form-data`, field name `image` |
| Response | `prediction`, `confidence`, `probabilities`, `heatmap` (data URI or `null`) |
| Dev routing | Vite proxy `/api/*` → `localhost:8000/*` ([frontend/vite.config.js](frontend/vite.config.js)) |
| Prod routing | set `VITE_API_BASE_URL` to the API origin at build time |
| Client | [frontend/src/api.js](frontend/src/api.js) — the only module that talks to the network |

`GET /health` reports `model_loaded` and `explanations_available`; the sidebar and the
top-bar pill reflect those fields rather than hard-coded text.

## Verify

```bash
cd backend && python -m pytest -q
```

55 tests. One (`tests/test_real_model.py`) is skipped unless a Keras backend and the
model weights are both present — it is the suite that actually executes the network, so
run it at least once on the demo machine.

```bash
cd frontend && npm run build
```

## What the model can and cannot do

It grades **the whole image** into `No DR`, `Mild DR`, `Moderate DR`, `Severe DR`,
`Proliferative DR`. It does not detect, localise, or count individual lesions —
no microaneurysm counts, no haemorrhage boundaries. The Grad-CAM overlay shows which
regions drove the grade; it is not a lesion segmentation and must not be read as one.

Per-component detail: [backend/README.md](backend/README.md),
[frontend/README.md](frontend/README.md).
