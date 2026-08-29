import React from 'react';
import { ClipboardList, Download } from 'lucide-react';
import { EmptyState, Panel } from './ui.jsx';
import { formatPercent, gradeInfo, isLowConfidence } from '../severity.js';
import { exportScreeningJson } from '../report.js';

const PAGE_COPY = {
  overview: {
    eyebrow: 'OVERVIEW',
    title: 'Overview',
    blurb: 'Screening activity recorded in this session.',
  },
  history: {
    eyebrow: 'HISTORY',
    title: 'Patient history',
    blurb: 'Every image graded since this console was opened.',
  },
  reports: {
    eyebrow: 'REPORTS',
    title: 'Screening reports',
    blurb: 'Download a machine-readable record of any screening.',
  },
};

function metrics(screenings) {
  const total = screenings.length;
  const referrals = screenings.filter((s) => gradeInfo(s.result.prediction).referral).length;
  const lowConfidence = screenings.filter((s) => isLowConfidence(s.result.confidence)).length;
  const meanConfidence = total
    ? screenings.reduce((sum, s) => sum + s.result.confidence, 0) / total
    : 0;

  return [
    ['Screenings this session', String(total), total === 1 ? '1 image graded' : `${total} images graded`],
    ['Referrals recommended', String(referrals), 'Moderate DR or worse'],
    ['Awaiting manual review', String(lowConfidence), 'Confidence below 70%'],
    ['Mean confidence', total ? formatPercent(meanConfidence) : '—', 'Across this session'],
  ];
}

export function ActivityPage({ page, screenings, onNavigate }) {
  const copy = PAGE_COPY[page] ?? PAGE_COPY.overview;

  return (
    <>
      <div className="page-head">
        <div>
          <small>{copy.eyebrow}</small>
          <h1>{copy.title}</h1>
          <p>{copy.blurb}</p>
        </div>
        <button type="button" className="primary" onClick={() => onNavigate('screening')}>
          New screening
        </button>
      </div>

      <div className="metric-strip">
        {metrics(screenings).map(([label, value, hint]) => (
          <div className="metric-card" key={label}>
            <small>{label}</small>
            <strong>{value}</strong>
            <span>{hint}</span>
          </div>
        ))}
      </div>

      <Panel title="Screening activity">
        {screenings.length === 0 ? (
          <EmptyState icon={ClipboardList} title="No screenings yet">
            Results appear here as soon as you grade an image. Nothing is stored on the
            server, so this list resets when the page reloads.
          </EmptyState>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Patient</th>
                  <th>Result</th>
                  <th>Confidence</th>
                  <th>Status</th>
                  <th aria-label="Export" />
                </tr>
              </thead>
              <tbody>
                {screenings.map((screening) => {
                  const { result, patient, timestamp } = screening;
                  const grade = gradeInfo(result.prediction);
                  const low = isLowConfidence(result.confidence);
                  return (
                    <tr key={screening.id}>
                      <td>{new Date(timestamp).toLocaleTimeString('en-IN')}</td>
                      <td>
                        <b>{patient.name || patient.id || 'Unidentified'}</b>
                      </td>
                      <td>
                        <span className={`table-tag ${grade.tone}`}>{result.prediction}</span>
                      </td>
                      <td className={low ? 'warn-text' : 'green-text'}>
                        {formatPercent(result.confidence)}
                      </td>
                      <td>{low ? 'Manual review' : grade.referral ? 'Referral' : 'Cleared'}</td>
                      <td>
                        <button
                          type="button"
                          className="plain"
                          onClick={() => exportScreeningJson(screening)}
                          aria-label={`Export screening for ${patient.name || 'unidentified patient'}`}
                        >
                          <Download size={13} aria-hidden="true" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}
