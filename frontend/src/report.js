// Client-side report generation. The backend stores nothing, so "Export" and
// "Create referral" produce a downloadable record instead of pretending a
// records system exists behind them.

import { formatPercent, gradeInfo, isLowConfidence } from './severity.js';

function download(filename, mimeType, contents) {
  const url = URL.createObjectURL(new Blob([contents], { type: mimeType }));
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function slug(patient) {
  const base = patient.id || patient.name || 'screening';
  return base.trim().replace(/[^\w.-]+/g, '-').toLowerCase() || 'screening';
}

function timestampForFilename(isoTimestamp) {
  return isoTimestamp.slice(0, 19).replace(/[:T]/g, '-');
}

/** Machine-readable record of one screening, including the full class distribution. */
export function exportScreeningJson(screening) {
  const payload = {
    ...screening,
    disclaimer:
      'Prototype AI screening support. Not clinically validated and not a medical diagnosis.',
  };
  download(
    `screening-${slug(screening.patient)}-${timestampForFilename(screening.timestamp)}.json`,
    'application/json',
    JSON.stringify(payload, null, 2),
  );
}

/** Human-readable referral note for printing or attaching to a case file. */
export function exportReferralNote(screening) {
  const { patient, result, timestamp } = screening;
  const grade = gradeInfo(result.prediction);
  const distribution = Object.entries(result.probabilities)
    .map(([name, value]) => `  ${name.padEnd(20)} ${formatPercent(value)}`)
    .join('\n');

  const lines = [
    'RETINACARE AI — SCREENING REFERRAL NOTE',
    '=======================================',
    '',
    `Generated:      ${new Date(timestamp).toLocaleString('en-IN')}`,
    `Patient ID:     ${patient.id || '(not recorded)'}`,
    `Patient name:   ${patient.name || '(not recorded)'}`,
    `Age:            ${patient.age || '(not recorded)'}`,
    '',
    'AI SCREENING RESULT',
    `  Grade:        ${result.prediction}`,
    `  Confidence:   ${formatPercent(result.confidence)}`,
    `  Referral:     ${grade.referral ? 'Recommended' : 'Not indicated by this screening'}`,
    isLowConfidence(result.confidence)
      ? '  Note:         Low confidence — manual review required.'
      : null,
    '',
    'CLASS DISTRIBUTION',
    distribution,
    '',
    'GUIDANCE',
    `  ${grade.guidance}`,
    '',
    'IMPORTANT',
    '  Produced by an AI screening prototype that is not clinically validated.',
    '  This is decision support only and is not a medical diagnosis. All findings',
    '  must be confirmed by a qualified eye-care professional.',
    '',
  ].filter((line) => line !== null);

  download(
    `referral-${slug(patient)}-${timestampForFilename(timestamp)}.txt`,
    'text/plain',
    lines.join('\n'),
  );
}
