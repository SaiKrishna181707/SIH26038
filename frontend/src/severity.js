// Presentation rules for the five grades the backend can return.
//
// Wording is deliberately conservative: this is screening triage support, and
// every surface that shows it also shows the not-a-diagnosis notice.

export const CLASS_ORDER = [
  'No DR',
  'Mild DR',
  'Moderate DR',
  'Severe DR',
  'Proliferative DR',
];

// Below this the grade is treated as inconclusive and flagged for manual review
// rather than presented as a confident answer.
export const LOW_CONFIDENCE_THRESHOLD = 0.7;

const GRADES = {
  'No DR': {
    tone: 'clear',
    shortLabel: 'No DR',
    referral: false,
    guidance: 'No referable retinopathy detected. Continue routine re-screening.',
  },
  'Mild DR': {
    tone: 'mild',
    shortLabel: 'Mild',
    referral: false,
    guidance: 'Early changes detected. Schedule earlier re-screening and review glycaemic control.',
  },
  'Moderate DR': {
    tone: 'moderate',
    shortLabel: 'Moderate',
    referral: true,
    guidance: 'Findings that an eye-care professional should review.',
  },
  'Severe DR': {
    tone: 'severe',
    shortLabel: 'Severe',
    referral: true,
    guidance: 'Advanced changes indicated. Prompt specialist evaluation recommended.',
  },
  'Proliferative DR': {
    tone: 'proliferative',
    shortLabel: 'Proliferative',
    referral: true,
    guidance: 'Sight-threatening changes indicated. Urgent specialist evaluation recommended.',
  },
};

const UNKNOWN_GRADE = {
  tone: 'unknown',
  shortLabel: 'Unrecognised',
  referral: true,
  guidance: 'This grade is not recognised by the interface. Refer for manual review.',
};

export function gradeInfo(prediction) {
  return GRADES[prediction] ?? { ...UNKNOWN_GRADE, shortLabel: prediction };
}

export function isLowConfidence(confidence) {
  return typeof confidence === 'number' && confidence < LOW_CONFIDENCE_THRESHOLD;
}

export function formatPercent(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * Order the probability map for display.
 *
 * Prefers the class order reported by /health so the backend stays the single
 * source of truth, falls back to the built-in order, and appends any class the
 * frontend does not know about rather than silently dropping it.
 */
export function orderedProbabilities(probabilities, classOrder) {
  const preferred = classOrder?.length ? classOrder : CLASS_ORDER;
  const known = preferred.filter((name) => name in probabilities);
  const extra = Object.keys(probabilities).filter((name) => !preferred.includes(name));
  return [...known, ...extra].map((name) => [name, probabilities[name]]);
}
