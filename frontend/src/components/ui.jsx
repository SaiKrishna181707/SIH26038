import React from 'react';

/** Card with a title and optional step badge, used for every block on a page. */
export function Panel({ title, step, action, children }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <b>{title}</b>
          {step && <small>{step}</small>}
        </div>
        {action}
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function Field({ label, value, onChange, placeholder, type = 'text', inputMode }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type={type}
        inputMode={inputMode}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

export function NavButton({ icon: Icon, text, active, disabled, onClick }) {
  return (
    <button
      type="button"
      className={`nav ${active ? 'active' : ''}`}
      disabled={disabled}
      aria-current={active ? 'page' : undefined}
      onClick={onClick}
    >
      <Icon size={16} aria-hidden="true" />
      {text}
    </button>
  );
}

export function EmptyState({ icon: Icon, eyebrow, title, children }) {
  return (
    <div className="empty-result">
      <div>
        <Icon size={22} aria-hidden="true" />
      </div>
      {eyebrow && <small>{eyebrow}</small>}
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}
