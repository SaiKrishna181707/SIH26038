import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CLASS_ORDER,
  formatPercent,
  gradeInfo,
  isLowConfidence,
  orderedProbabilities,
} from '../src/severity.js';

test('invalid confidence fails safe to manual review', () => {
  for (const value of [undefined, null, NaN, Infinity, -1, 1.1, 0.69]) {
    assert.equal(isLowConfidence(value), true);
  }
  assert.equal(isLowConfidence(0.7), false);
});

test('formatPercent does not emit NaN or Infinity', () => {
  assert.equal(formatPercent(NaN), '—');
  assert.equal(formatPercent(Infinity), '—');
  assert.equal(formatPercent(0.812), '81.2%');
});

test('unknown grade is escalated instead of cleared', () => {
  assert.equal(gradeInfo('Unexpected Class').referral, true);
  assert.equal(gradeInfo(null).referral, true);
});

test('probability ordering survives malformed health class order', () => {
  const probabilities = Object.fromEntries(CLASS_ORDER.map((name, index) => [name, index / 10]));
  assert.deepEqual(
    orderedProbabilities(probabilities, 'not-an-array').map(([name]) => name),
    CLASS_ORDER,
  );
  assert.deepEqual(orderedProbabilities(null, CLASS_ORDER), []);
});
