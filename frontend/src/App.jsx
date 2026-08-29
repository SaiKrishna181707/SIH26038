import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { ApiError, fetchHealth } from './api.js';
import { Sidebar } from './components/Sidebar.jsx';
import { TopBar } from './components/TopBar.jsx';
import { Screening } from './components/Screening.jsx';
import { ActivityPage } from './components/ActivityPage.jsx';

// The model loads lazily in the backend, so /health is polled until it reports
// ready. This is what drives the connection indicators instead of fixed labels.
// Polling continues at a slower cadence afterwards: if the backend goes away
// mid-session the indicators must stop claiming a live connection.
const HEALTH_POLL_MS = 4000;
const READY_POLL_MS = 30000;

export default function App() {
  const [page, setPage] = useState('screening');
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [screenings, setScreenings] = useState([]);
  const [health, setHealth] = useState({ state: 'loading', data: null, error: '' });

  useEffect(() => {
    const controller = new AbortController();
    let timer;
    let active = true;

    const poll = async () => {
      try {
        const data = await fetchHealth({ signal: controller.signal });
        if (!active) return;
        setHealth({ state: 'ready', data, error: '' });
        // Back off once the model is warm; keep checking so a backend that dies
        // later is reflected instead of leaving a stale "connected" indicator.
        timer = setTimeout(poll, data.model_loaded ? READY_POLL_MS : HEALTH_POLL_MS);
      } catch (error) {
        if (!active || error.name === 'AbortError') return;
        setHealth({
          state: 'error',
          data: null,
          error: error instanceof ApiError ? error.message : 'Health check failed.',
        });
        timer = setTimeout(poll, HEALTH_POLL_MS);
      }
    };

    poll();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  const navigate = useCallback((next) => {
    setPage(next);
    setMobileNavOpen(false);
  }, []);

  const recordScreening = useCallback((screening) => {
    setScreenings((previous) => [screening, ...previous]);
  }, []);

  return (
    <div className="app">
      <Sidebar
        page={page}
        onNavigate={navigate}
        open={mobileNavOpen}
        onClose={() => setMobileNavOpen(false)}
        health={health}
      />
      {mobileNavOpen && <div className="overlay" onClick={() => setMobileNavOpen(false)} />}
      <main>
        <TopBar
          page={page}
          onOpenMenu={() => setMobileNavOpen(true)}
          health={health}
          screeningCount={screenings.length}
        />
        {health.state === 'error' && (
          <div className="service-banner" role="alert">
            <AlertTriangle size={15} aria-hidden="true" />
            <span>{health.error}</span>
          </div>
        )}
        <section className="page">
          {page === 'screening' ? (
            <Screening
              classOrder={health.data?.classes}
              onScreeningComplete={recordScreening}
            />
          ) : (
            <ActivityPage page={page} screenings={screenings} onNavigate={navigate} />
          )}
        </section>
      </main>
    </div>
  );
}
