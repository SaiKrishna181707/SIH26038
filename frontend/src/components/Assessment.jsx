import React from 'react';
import { AlertTriangle, CheckCircle2, Download, FileText, Info, ShieldCheck } from 'lucide-react';
import { Panel } from './ui.jsx';
import { formatPercent, gradeInfo, isLowConfidence, orderedProbabilities } from '../severity.js';
import { exportReferralNote, exportScreeningJson } from '../report.js';

export function Assessment({ screening, classOrder }) {
  const { result } = screening;
  const grade = gradeInfo(result.prediction);
  const lowConfidence = isLowConfidence(result.confidence);
  const distribution = orderedProbabilities(result.probabilities, classOrder);
  const requiresReview = lowConfidence || grade.referral;

  let decisionTitle;
  let decisionGuidance;
  if (lowConfidence && !grade.referral) {
    decisionTitle = 'Manual review required before clearance';
    decisionGuidance =
      'The predicted grade is non-referable, but confidence is too low to clear this screening from the AI result alone.';
  } else if (lowConfidence && grade.referral) {
    decisionTitle = 'Specialist evaluation and manual review recommended';
    decisionGuidance =
      'The predicted grade is referable and confidence is low. Prioritize specialist evaluation and manually verify the image.';
  } else if (grade.referral) {
    decisionTitle = 'Specialist evaluation recommended';
    decisionGuidance = grade.guidance;
  } else {
    decisionTitle = 'No referral indicated by this screening';
    decisionGuidance = grade.guidance;
  }

  return (
    <>
      <Panel
        title="AI assessment"
        step="STEP 03"
        action={
          <button type="button" className="plain" onClick={() => exportScreeningJson(screening)}>
            <Download size={13} aria-hidden="true" /> JSON
          </button>
        }
      >
        <div className="assessment-head">
          <div>
            <small>SCREENING RESULT</small>
            <h2>{result.prediction}</h2>
            <span className={`severity ${grade.tone}`}>{grade.shortLabel}</span>
            <strong className="confidence">
              {formatPercent(result.confidence)} <em>confidence</em>
            </strong>
          </div>
          <div className="score">
            <b>{Number.isFinite(result.confidence) ? Math.round(result.confidence * 100) : '—'}</b>
            <span>CONF.</span>
          </div>
        </div>

        {lowConfidence && (
          <div className="warn-note">
            <AlertTriangle size={14} aria-hidden="true" />
            <span>
              Confidence is below {formatPercent(0.7, 0)}. Treat this grade as inconclusive
              and have the image reviewed manually.
            </span>
          </div>
        )}

        <div className="prob-title">
          <b>Class probability</b>
          <span>Model output</span>
        </div>
        {distribution.map(([name, value]) => (
          <div className={`prob ${name === result.prediction ? 'selected' : ''}`} key={name}>
            <div>
              <span>{name}</span>
              <b>{formatPercent(value)}</b>
            </div>
            <i>
              <u
                style={{
                  width: `${Math.min(
                    Math.max(Number.isFinite(value) ? value * 100 : 0, 0.5),
                    100,
                  )}%`,
                }}
              />
            </i>
          </div>
        ))}
      </Panel>

      <Panel title="Clinical decision support" step="REVIEW">
        <div className={`decision ${requiresReview ? 'refer' : 'clear'}`}>
          {requiresReview ? (
            <AlertTriangle size={17} aria-hidden="true" />
          ) : (
            <CheckCircle2 size={17} aria-hidden="true" />
          )}
          <div>
            <b>{decisionTitle}</b>
            <p>{decisionGuidance}</p>
          </div>
        </div>
        <button
          type="button"
          className="primary wide"
          onClick={() => exportReferralNote(screening)}
        >
          <FileText size={14} aria-hidden="true" />{' '}
          {requiresReview ? 'Generate review / referral note' : 'Generate screening note'}
        </button>
        <div className="safety">
          <ShieldCheck size={13} aria-hidden="true" /> Screening support only · not a diagnosis
        </div>
      </Panel>

      <Panel title="About this assessment" step="MODEL">
        <dl className="spec">
          <div>
            <dt>Model</dt>
            <dd>EfficientNet-B0, five-class DR severity</dd>
          </div>
          <div>
            <dt>Input</dt>
            <dd>224 × 224 RGB fundus image</dd>
          </div>
          <div>
            <dt>Explanation</dt>
            <dd>
              {result.heatmap
                ? 'Grad-CAM over final convolutional features'
                : 'Unavailable for this model'}
            </dd>
          </div>
        </dl>
        <div className="scope-note">
          <Info size={13} aria-hidden="true" />
          <span>
            This model grades the image as a whole. It does not localise or count individual
            lesions such as microaneurysms, haemorrhages, or exudates.
          </span>
        </div>
      </Panel>
    </>
  );
}

/** Side-by-side original and Grad-CAM overlay, or an honest note when absent. */
export function VisualExplanation({ previewUrl, heatmap }) {
  return (
    <Panel title="Visual explanation" step="STEP 04">
      {heatmap ? (
        <>
          <div className="explain">
            <div>
              <small>ORIGINAL</small>
              <img src={previewUrl} alt="Uploaded fundus image" />
            </div>
            <div>
              <small>GRAD-CAM ATTENTION</small>
              <img src={heatmap} alt="Grad-CAM attention overlay from the model" />
            </div>
          </div>
          <p className="muted">
            Warmer regions contributed most to the predicted grade. The map is computed from
            the model&apos;s own final convolutional features at 7 × 7 resolution, so it shows
            broad areas of influence rather than precise lesion boundaries.
          </p>
        </>
      ) : (
        <div className="explain-missing">
          <Info size={15} aria-hidden="true" />
          <div>
            <b>No attention map returned</b>
            <p>
              The backend reported that the loaded model does not support Grad-CAM, so no
              overlay is shown. The grade above is still the model&apos;s own output.
            </p>
          </div>
        </div>
      )}
    </Panel>
  );
}
