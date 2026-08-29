// Client for the FastAPI backend in ../backend.

const ENV = import.meta.env ?? {};
const configuredBase = String(ENV.VITE_API_BASE_URL ?? '/api').trim();
const BASE_URL = configuredBase === '/' ? '' : configuredBase.replace(/\/+$/, '');

export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
export const ACCEPT_ATTRIBUTE = ACCEPTED_TYPES.join(',');
export const HEALTH_TIMEOUT_MS = 5000;
export const PREDICT_TIMEOUT_MS = 120000;
const MAX_HEATMAP_URI_CHARS = 2_000_000;

export class ApiError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ApiError';
  }
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const UPSTREAM_DOWN_STATUSES = new Set([502, 503, 504]);

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  } catch {
    // Not JSON — fall through to the generic message.
  }
  if (response.status === 404) return 'Backend reached, but /predict was not found.';
  if (UPSTREAM_DOWN_STATUSES.has(response.status)) return describeNetworkFailure();
  return `The screening service returned an error (HTTP ${response.status}).`;
}

function describeNetworkFailure() {
  return (
    'Cannot reach the screening service. Check that the backend is running and that ' +
    'VITE_API_BASE_URL points to it in deployed builds.'
  );
}

async function fetchWithTimeout(url, options, timeoutMs, timeoutMessage) {
  const timeoutController = new AbortController();
  const externalSignal = options.signal;
  let timedOut = false;

  const forwardAbort = () => timeoutController.abort();
  if (externalSignal?.aborted) {
    timeoutController.abort();
  } else {
    externalSignal?.addEventListener('abort', forwardAbort, { once: true });
  }

  const timer = setTimeout(() => {
    timedOut = true;
    timeoutController.abort();
  }, timeoutMs);

  try {
    return await fetch(url, { ...options, signal: timeoutController.signal });
  } catch (error) {
    if (timedOut) throw new ApiError(timeoutMessage);
    throw error;
  } finally {
    clearTimeout(timer);
    externalSignal?.removeEventListener('abort', forwardAbort);
  }
}

/** Validate a file locally. Returns an error string, or null when acceptable. */
export function validateImage(file) {
  if (!file) return 'No file selected.';
  if (!ACCEPTED_TYPES.includes(file.type)) {
    return 'Unsupported format. Upload a JPEG, PNG, or WebP fundus image.';
  }
  if (!Number.isFinite(file.size) || file.size < 0) {
    return 'The selected file has an invalid size.';
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return `Image is ${formatBytes(file.size)}. The limit is ${formatBytes(MAX_IMAGE_BYTES)}.`;
  }
  if (file.size === 0) return 'The selected file is empty.';
  return null;
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Return null for a valid /predict payload, otherwise a safe diagnostic string. */
export function predictionPayloadError(result) {
  if (!isRecord(result)) return 'response is not an object';
  if (typeof result.prediction !== 'string' || !result.prediction.trim() || result.prediction.length > 100) {
    return 'prediction is missing or invalid';
  }
  if (!Number.isFinite(result.confidence) || result.confidence < 0 || result.confidence > 1) {
    return 'confidence is outside [0, 1]';
  }
  if (!isRecord(result.probabilities)) return 'probabilities is not an object';

  const entries = Object.entries(result.probabilities);
  if (entries.length === 0 || entries.length > 50) return 'probability map has an invalid size';

  let total = 0;
  for (const [name, value] of entries) {
    if (!name || name.length > 100 || !Number.isFinite(value) || value < 0 || value > 1) {
      return 'probability map contains an invalid entry';
    }
    total += value;
  }
  if (Math.abs(total - 1) > 0.02) return 'probabilities do not sum to 1';

  const predictedProbability = result.probabilities[result.prediction];
  if (!Number.isFinite(predictedProbability)) return 'predicted class is missing from probabilities';
  if (Math.abs(predictedProbability - result.confidence) > 0.02) {
    return 'confidence does not match the predicted class';
  }

  if (result.heatmap !== null && result.heatmap !== undefined) {
    if (
      typeof result.heatmap !== 'string' ||
      result.heatmap.length > MAX_HEATMAP_URI_CHARS ||
      !/^data:image\/(?:webp|png);base64,[A-Za-z0-9+/=]+$/.test(result.heatmap)
    ) {
      return 'heatmap is not a supported image data URI';
    }
  }
  return null;
}

/** GET /health — reports model readiness and the canonical class order. */
export async function fetchHealth({ signal } = {}) {
  let response;
  try {
    response = await fetchWithTimeout(
      `${BASE_URL}/health`,
      { signal },
      HEALTH_TIMEOUT_MS,
      'The screening service health check timed out.',
    );
  } catch (error) {
    if (error instanceof ApiError || error.name === 'AbortError') throw error;
    throw new ApiError(describeNetworkFailure());
  }
  if (!response.ok) throw new ApiError(await errorMessage(response));
  return response.json();
}

/** POST /predict — multipart upload under the field name `image`. */
export async function requestPrediction(file, { signal } = {}) {
  const invalid = validateImage(file);
  if (invalid) throw new ApiError(invalid);

  const formData = new FormData();
  formData.append('image', file);

  let response;
  try {
    response = await fetchWithTimeout(
      `${BASE_URL}/predict`,
      { method: 'POST', body: formData, signal },
      PREDICT_TIMEOUT_MS,
      'The image analysis timed out. Please try again.',
    );
  } catch (error) {
    if (error instanceof ApiError || error.name === 'AbortError') throw error;
    throw new ApiError(describeNetworkFailure());
  }

  if (!response.ok) throw new ApiError(await errorMessage(response));

  const result = await response.json();
  if (predictionPayloadError(result)) {
    throw new ApiError('The screening service returned an unexpected response.');
  }
  return result;
}
