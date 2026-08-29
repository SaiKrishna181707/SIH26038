// Client for the FastAPI backend in ../backend.
//
// In development requests go to /api, which vite.config.js proxies to
// http://localhost:8000. Set VITE_API_BASE_URL at build time to target a
// deployed backend directly (CORS_ORIGINS must then allow this origin).

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

// Kept in step with MAX_IMAGE_BYTES and ALLOWED_CONTENT_TYPES in backend/main.py
// so an oversized or unsupported file is reported instantly instead of after a
// wasted upload. The backend still enforces both.
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
export const ACCEPT_ATTRIBUTE = ACCEPTED_TYPES.join(',');

export class ApiError extends Error {}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// A dev-proxy or reverse proxy answers with these when the backend itself is
// not listening, so they mean the same thing to the user as a network failure.
const UPSTREAM_DOWN_STATUSES = new Set([502, 503, 504]);

/** Pull FastAPI's `detail` out of an error response, falling back to the status. */
async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    // Validation errors arrive as a list of objects.
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch {
    // Not JSON — fall through to the generic message.
  }
  if (response.status === 404) return 'Backend reached, but /predict was not found.';
  // No JSON body on a gateway status means nothing answered upstream. FastAPI's
  // own 503 does carry a detail, so it is handled above and never reaches here.
  if (UPSTREAM_DOWN_STATUSES.has(response.status)) return describeNetworkFailure();
  return `The screening service returned an error (HTTP ${response.status}).`;
}

function describeNetworkFailure() {
  return (
    'Cannot reach the screening service. Start the backend with ' +
    '"python -m uvicorn main:app --port 8000" in the backend folder.'
  );
}

/** Validate a file locally. Returns an error string, or null when acceptable. */
export function validateImage(file) {
  if (!file) return 'No file selected.';
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return 'Unsupported format. Upload a JPEG, PNG, or WebP fundus image.';
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return `Image is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_IMAGE_BYTES)}.`;
  }
  if (file.size === 0) return 'The selected file is empty.';
  return null;
}

/** GET /health — reports model readiness and the canonical class order. */
export async function fetchHealth({ signal } = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}/health`, { signal });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError(describeNetworkFailure());
  }
  if (!response.ok) throw new ApiError(await errorMessage(response));
  return response.json();
}

/**
 * POST /predict — multipart upload under the field name `image`.
 * Resolves to { prediction, confidence, probabilities, heatmap }.
 */
export async function requestPrediction(file, { signal } = {}) {
  const invalid = validateImage(file);
  if (invalid) throw new ApiError(invalid);

  const formData = new FormData();
  formData.append('image', file);

  let response;
  try {
    response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
      signal,
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError(describeNetworkFailure());
  }

  if (!response.ok) throw new ApiError(await errorMessage(response));

  const result = await response.json();
  if (!result?.prediction || !result?.probabilities) {
    throw new ApiError('The screening service returned an unexpected response.');
  }
  return result;
}
