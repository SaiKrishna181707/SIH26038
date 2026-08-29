// Presentation rules for the five grades the backend can return.

export const CLASS_ORDER = [
  'No DR',
  'Mild DR',
  'Moderate DR',
  'Severe DR',
  'Proliferative DR',
];

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
  if (typeof prediction !== 'string' || !prediction.trim()) return { ...UNKNOWN_GRADE };
  return GRADES[prediction] ?? { ...UNKNOWN_GRADE, shortLabel: prediction };
}

/** Invalid/missing confidence is treated conservatively as requiring review. */
export function isLowConfidence(confidence) {
  return !Number.isFinite(confidence) || confidence < LOW_CONFIDENCE_THRESHOLD || confidence > 1;
}

export function formatPercent(value, digits = 1) {
  if (!Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function orderedProbabilities(probabilities, classOrder) {
  const safeProbabilities =
    probabilities && typeof probabilities === 'object' && !Array.isArray(probabilities)
      ? probabilities
      : {};
  const preferred =
    Array.isArray(classOrder) && classOrder.every((name) => typeof name === 'string')
      ? classOrder
      : CLASS_ORDER;
  const known = preferred.filter((name) => Object.hasOwn(safeProbabilities, name));
  const extra = Object.keys(safeProbabilities).filter((name) => !preferred.includes(name));
  return [...known, ...extra].map((name) => [name, safeProbabilities[name]]);
}
