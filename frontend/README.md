# RetinaCare AI — Frontend

React + Vite screening console for SIH26038. It uploads a fundus image to the FastAPI
backend, validates the returned contract, then renders the grade, probability distribution,
and Grad-CAM overlay.

## Run

```bash
npm install
npm run dev
```

The development server opens on `http://localhost:5173` and proxies `/api/*` to
`http://localhost:8000/*`.

For deployed builds set the API origin at build time:

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

The backend must allow the frontend origin in `CORS_ORIGINS`.

## Verify

```bash
npm test
npm run build
```

`npm test` uses Node's built-in test runner, so no additional test framework is required.
The tests pin response validation and fail-safe confidence/severity behavior. GitHub Actions
also runs the Vite production build on every pull request.

## Network safety

`src/api.js` is the only module that performs network calls. It enforces:

- client-side JPEG/PNG/WebP and 10 MB checks before upload;
- a 5-second `/health` timeout and a 120-second `/predict` timeout;
- safe extraction of FastAPI error details;
- structural validation of prediction, confidence, probability ranges/sum, and predicted
  class consistency before any result reaches the UI;
- `heatmap` restricted to bounded PNG/WebP `data:` URIs.

The backend independently revalidates all image bytes and model output; client checks are
for fast feedback and UI safety, not security boundaries.

## Fail-safe triage behavior

Confidence below 70% is treated as inconclusive. A low-confidence non-referable class is
**not** shown as cleared: the decision panel and exported note require manual review before
triage/clearance. Unknown grades and invalid confidence values also fail safe to review.

## Session-only state

Patient ID, name, age, and screening history stay in React memory and are not sent to the
backend; only the image is uploaded. The activity list resets on reload. Exports are
created entirely in the browser.

## Main modules

| Path | Role |
|---|---|
| `src/api.js` | Network calls, timeouts, upload and response validation. |
| `src/severity.js` | Grade metadata, fail-safe confidence handling, formatting. |
| `src/report.js` | Browser-side JSON and screening/review-note exports. |
| `src/components/Screening.jsx` | Upload/analyze/cancel flow and in-flight request guard. |
| `src/components/Assessment.jsx` | Probabilities, triage guidance, Grad-CAM display. |
| `src/components/ActivityPage.jsx` | Session metrics and history table. |
| `src/App.jsx` | Health polling, navigation, session screening list. |
