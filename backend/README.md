# SIH26038 AI Backend

FastAPI backend and local prediction CLI for five-class diabetic-retinopathy grading with the pretrained [`Aldahmashi/DR-EfficientNetB0`](https://huggingface.co/Aldahmashi/DR-EfficientNetB0) Keras model running on the PyTorch backend.

> Prototype limitation: this model is not clinically validated and must not be used for diagnosis or patient-care decisions.

## Setup (Windows PowerShell)

```powershell
cd D:\SIH\SIH26038-AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The prepared project keeps the model at `models/final_model.keras`. If that file is absent, the first prediction downloads it from Hugging Face. Set `MODEL_PATH` to use another local copy.

## Local prediction

```powershell
python predict.py C:\path\to\retinal-image.jpg
```

Output includes the predicted class, confidence, and all five probabilities.

## Run the API

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for interactive API documentation. Health is available at `GET /health`.

## Frontend integration

Send a `multipart/form-data` request whose field name is exactly `image`:

```javascript
const formData = new FormData();
formData.append("image", selectedFile);

const response = await fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
});

if (!response.ok) {
  throw new Error((await response.json()).detail ?? "Prediction failed");
}

const result = await response.json();
console.log(result.prediction, result.confidence, result.probabilities);
```

Default allowed frontend origins are `http://localhost:3000` and `http://localhost:5173`. Override them before startup for another frontend URL:

```powershell
$env:CORS_ORIGINS="https://frontend.example.com,http://localhost:5173"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Response shape:

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
  }
}
```

Accepted uploads: JPEG, PNG, or WebP, up to 10 MB. Set `MAX_IMAGE_BYTES` to change the byte limit.

## Verification

```powershell
python -m pytest -q
```

Grad-CAM is intentionally not included in this baseline: the required prediction path is complete first, and an unvalidated heatmap should not be represented as clinical explainability.
