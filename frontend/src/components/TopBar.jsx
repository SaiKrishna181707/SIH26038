import React from 'react';
import { AlertTriangle, CheckCircle2, Globe2, Loader2, Menu } from 'lucide-react';

const PAGE_TITLES = {
  screening: 'Screening',
  overview: 'Overview',
  history: 'Patient history',
  reports: 'Reports',
};

/** Connection pill driven by /health rather than a fixed "Offline-ready" label. */
function ConnectionPill({ health }) {
  if (health.state === 'loading') {
    return (
      <span className="conn pending">
        <Loader2 size={13} className="spin" aria-hidden="true" /> Connecting
      </span>
    );
  }
  if (health.state === 'error') {
    return (
      <span className="conn down" title={health.error}>
        <AlertTriangle size={13} aria-hidden="true" /> Service offline
      </span>
    );
  }
  return (
    <span className="conn up">
      <CheckCircle2 size={13} aria-hidden="true" />{' '}
      {health.data.model_loaded ? 'AI service connected' : 'Model loading'}
    </span>
  );
}

export function TopBar({ page, onOpenMenu, health, screeningCount }) {
  return (
    <header className="topbar">
      <button type="button" className="menu" onClick={onOpenMenu} aria-label="Open menu">
        <Menu size={19} aria-hidden="true" />
      </button>
      <div>
        <b>{PAGE_TITLES[page] ?? 'Screening'}</b>
        <span>SIH26038 · Explainable AI for diabetic retinopathy screening</span>
      </div>
      <div className="top-actions">
        <ConnectionPill health={health} />
        <span>
          This session <b>{screeningCount}</b>
        </span>
        <span className="lang">
          <Globe2 size={13} aria-hidden="true" /> EN
        </span>
      </div>
    </header>
  );
}
