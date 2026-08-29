import React from 'react';
import {
  Activity,
  Eye,
  FileText,
  LayoutDashboard,
  Settings,
  ShieldCheck,
  Users,
  X,
} from 'lucide-react';
import { NavButton } from './ui.jsx';

const WORKSPACE_PAGES = [
  { id: 'screening', text: 'Screening', icon: Activity },
  { id: 'overview', text: 'Overview', icon: LayoutDashboard },
  { id: 'history', text: 'Patient history', icon: Users },
  { id: 'reports', text: 'Reports', icon: FileText },
];

/** Real backend state, replacing what used to be a hardcoded "ready" badge. */
function statusText(health) {
  if (health.state === 'loading') return { className: 'pending', label: 'Connecting to AI service' };
  if (health.state === 'error') return { className: 'down', label: 'AI service unreachable' };
  if (!health.data.model_loaded) return { className: 'pending', label: 'Model loading' };
  return { className: 'up', label: 'AI system ready' };
}

export function Sidebar({ page, onNavigate, open, onClose, health }) {
  const status = statusText(health);
  const modelName = health.data?.model ?? 'EfficientNet-B0';

  return (
    <aside className={`sidebar ${open ? 'open' : ''}`}>
      <div className="brand">
        <div className="brand-mark">
          <Eye size={18} aria-hidden="true" />
        </div>
        <div>
          <b>RetinaCare</b>
          <span>Clinical Screening Console</span>
        </div>
        <button type="button" className="mobile-close" onClick={onClose} aria-label="Close menu">
          <X size={18} aria-hidden="true" />
        </button>
      </div>

      <div className="workspace">
        <span className={`live-dot ${status.className}`} /> Rural screening workspace
      </div>

      <div className="section-label">WORKSPACE</div>
      <nav>
        {WORKSPACE_PAGES.map(({ id, text, icon }) => (
          <NavButton
            key={id}
            icon={icon}
            text={text}
            active={page === id}
            onClick={() => onNavigate(id)}
          />
        ))}
      </nav>

      <div className="section-label system-label">SYSTEM</div>
      <NavButton icon={Settings} text="Settings" disabled />
      <NavButton icon={ShieldCheck} text="Help &amp; guidance" disabled />

      <div className="sidebar-bottom">
        <div className="system-status">
          <b>
            <span className={`live-dot ${status.className}`} /> {status.label}
          </b>
          <span>Model · {modelName}</span>
          <span>
            Explanations ·{' '}
            {health.data?.explanations_available ? 'Grad-CAM enabled' : 'unavailable'}
          </span>
        </div>
        <div className="operator">
          <div>SK</div>
          <span>
            <b>Screening operator</b>
            <small>Rural care unit</small>
          </span>
        </div>
      </div>
    </aside>
  );
}
