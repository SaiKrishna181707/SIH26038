import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MAX_IMAGE_BYTES,
  predictionPayloadError,
  validateImage,
} from '../src/api.js';

const validPayload = () => ({
  prediction: 'Moderate DR',
  confidence: 0.7,
  probabilities: {
    'No DR': 0.05,
    'Mild DR': 0.1,
    'Moderate DR': 0.7,
    'Severe DR': 0.1,
    'Proliferative DR': 0.05,
  },
  heatmap: null,
});

test('accepts the documented prediction payload', () => {
  assert.equal(predictionPayloadError(validPayload()), null);
});

test('rejects malformed probability containers', () => {
  for (const probabilities of [null, [], 'not-an-object']) {
    const payload = validPayload();
    payload.probabilities = probabilities;
    assert.ok(predictionPayloadError(payload));
  }
});

test('rejects non-finite and out-of-range confidence', () => {
  for (const confidence of [NaN, Infinity, -0.01, 1.01]) {
    const payload = validPayload();
    payload.confidence = confidence;
    assert.ok(predictionPayloadError(payload));
  }
});

test('rejects probability maps that do not sum to one', () => {
  const payload = validPayload();
  payload.probabilities['No DR'] = 0.5;
  assert.match(predictionPayloadError(payload), /sum to 1/);
});

test('rejects confidence that disagrees with predicted class', () => {
  const payload = validPayload();
  payload.confidence = 0.4;
  assert.match(predictionPayloadError(payload), /does not match/);
});

test('rejects dangerous or malformed heatmap URLs', () => {
  for (const heatmap of [
    'javascript:alert(1)',
    'data:text/html;base64,PGgxPmJvb208L2gxPg==',
    'data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=',
  ]) {
    const payload = validPayload();
    payload.heatmap = heatmap;
    assert.match(predictionPayloadError(payload), /heatmap/);
  }
});

test('accepts png/webp image data URIs', () => {
  for (const heatmap of [
    'data:image/png;base64,AAAA',
    'data:image/webp;base64,AAAA',
  ]) {
    const payload = validPayload();
    payload.heatmap = heatmap;
    assert.equal(predictionPayloadError(payload), null);
  }
});

test('local image validation rejects spoof-prone metadata and oversize files', () => {
  assert.equal(validateImage({ type: 'image/png', size: 10 }), null);
  assert.ok(validateImage({ type: 'image/gif', size: 10 }));
  assert.ok(validateImage({ type: 'image/png', size: 0 }));
  assert.ok(validateImage({ type: 'image/png', size: MAX_IMAGE_BYTES + 1 }));
  assert.ok(validateImage({ type: 'image/png', size: NaN }));
});
