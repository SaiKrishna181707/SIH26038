# RetinaCare AI — Frontend

React + Vite screening console for SIH26038 (Explainable AI for Diabetic Retinopathy
Screening in Rural India). It uploads a fundus image to the backend at
[`../backend`](../backend), then renders the returned grade, class distribution, and
Grad-CAM overlay.

## Run

```bash
npm install
npm run dev
```

Opens on http://localhost:5173. Start the backend first (see
[../backend/README.md](../backend/README.md)) — without it the console loads but shows
"Service offline" and uploads fail with a message naming the uvicorn command.

```bash
npm run build      # production bundle into dist/
npm run preview    # serve the built bundle
```

## Backend wiring

In development, [vite.config.js](vite.config.js) proxies `/api/*` to
`http://localhost:8000/*`, which keeps the browser on a single origin and sidesteps CORS.

For any other deployment, point the app at the API origin at **build** time — Vite
inlines `import.meta.env` values into the bundle, so this is not a runtime setting:

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

The backend must then list that front-end origin in `CORS_ORIGINS`.

## Layout

| Path | Role |
|---|---|
| `src/api.js` | The only module that performs network calls. Validation, `POST /predict`, `GET /health`, error-message extraction. |
| `src/severity.js` | Grade → tone, referral flag, guidance text. Confidence formatting and ordering. |
| `src/report.js` | Client-side JSON export and plain-text referral note. No server round trip. |
| `src/components/Screening.jsx` | Upload, analyze, cancel. Owns the `AbortController` and the preview object URL. |
| `src/components/Assessment.jsx` | Grade, probability bars, referral guidance, Grad-CAM overlay. |
| `src/components/ActivityPage.jsx` | Session metrics and table, derived from real screenings. |
| `src/App.jsx` | `/health` polling, page routing, session screening list. |

## Session-only state

Nothing is persisted. Patient ID, name, and age stay in React state and are used only in
the exported referral note — the request body contains the image and nothing else. The
activity table resets on reload. There is no login and no database; adding either is a
deployment decision, not a front-end one.

## Constraints worth knowing

- Accepted uploads: JPEG, PNG, WebP, up to 10 MB. Checked client-side *and* server-side.
- The overlay arrives as a WebP data URI (~30 KB) rather than a second request, which
  matters on the low-bandwidth connections this project targets.
- If the backend cannot produce an overlay, `heatmap` is `null` and the explanation panel
  says so rather than showing a decorative substitute.
