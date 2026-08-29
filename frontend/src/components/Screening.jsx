import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  Loader2,
  UploadCloud,
} from 'lucide-react';
import { ACCEPT_ATTRIBUTE, ApiError, requestPrediction, validateImage } from '../api.js';
import { EmptyState, Field, Panel } from './ui.jsx';
import { Assessment, VisualExplanation } from './Assessment.jsx';

const EMPTY_PATIENT = { id: '', name: '', age: '' };

export function Screening({ classOrder, onScreeningComplete }) {
  const [patient, setPatient] = useState(EMPTY_PATIENT);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [screening, setScreening] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const fileInput = useRef(null);
  const requestRef = useRef(null);
  // Held in a ref so the unmount cleanup sees the current URL without making
  // the effect re-run (and revoke a live URL) on every selection.
  const previewRef = useRef('');

  useEffect(() => {
    previewRef.current = previewUrl;
  }, [previewUrl]);

  useEffect(
    () => () => {
      requestRef.current?.abort();
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    },
    [],
  );

  const updatePatient = (key) => (value) => setPatient((prev) => ({ ...prev, [key]: value }));

  const selectImage = useCallback((selected) => {
    if (!selected) return;
    const invalid = validateImage(selected);
    if (invalid) {
      setError(invalid);
      return;
    }
    requestRef.current?.abort();
    setError('');
    setScreening(null);
    setAnalyzing(false);
    setFile(selected);
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return URL.createObjectURL(selected);
    });
  }, []);

  const analyze = async () => {
    if (!file || analyzing) return;
    const controller = new AbortController();
    requestRef.current = controller;
    setAnalyzing(true);
    setError('');

    try {
      const result = await requestPrediction(file, { signal: controller.signal });
      const record = {
        id:
          globalThis.crypto?.randomUUID?.() ??
          `screening-${performance.now().toString(36)}`,
        timestamp: new Date().toISOString(),
        patient: { ...patient },
        image: { name: file.name, sizeBytes: file.size, type: file.type },
        result,
      };
      setScreening(record);
      onScreeningComplete(record);
    } catch (caught) {
      if (caught.name === 'AbortError') return;
      setError(
        caught instanceof ApiError ? caught.message : 'Screening failed. Please try again.',
      );
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setAnalyzing(false);
      }
    }
  };

  const clear = () => {
    requestRef.current?.abort();
    requestRef.current = null;
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return '';
    });
    setFile(null);
    setScreening(null);
    setAnalyzing(false);
    setError('');
  };

  return (
    <>
      <div className="page-head">
        <div>
          <small>SCREENING WORKSPACE</small>
          <h1>Retinal screening</h1>
          <p>Capture or upload a fundus image, then review the AI-assisted assessment.</p>
        </div>
      </div>

      <div className="screen-grid">
        <div>
          <Panel title="Patient details" step="STEP 01">
            <div className="fields">
              <Field
                label="Patient ID"
                value={patient.id}
                onChange={updatePatient('id')}
                placeholder="e.g. RC-10428"
              />
              <Field
                label="Patient name"
                value={patient.name}
                onChange={updatePatient('name')}
                placeholder="Full name"
              />
              <Field
                label="Age"
                value={patient.age}
                onChange={updatePatient('age')}
                placeholder="Years"
                inputMode="numeric"
              />
            </div>
            <p className="field-note">
              Recorded locally for the referral note only. Nothing is sent to the server
              except the image itself.
            </p>
          </Panel>

          <Panel
            title="Fundus image"
            step="STEP 02"
            action={
              file && (
                <button type="button" className="plain" onClick={clear}>
                  Clear
                </button>
              )
            }
          >
            {!file ? (
              <div
                className="drop"
                role="button"
                tabIndex={0}
                onClick={() => fileInput.current.click()}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    fileInput.current.click();
                  }
                }}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  selectImage(event.dataTransfer.files[0]);
                }}
              >
                <div className="upload-mark">
                  <UploadCloud size={20} aria-hidden="true" />
                </div>
                <b>Drop fundus image here</b>
                <span>JPEG, PNG or WebP · up to 10 MB</span>
                <button
                  type="button"
                  className="primary"
                  onClick={(event) => {
                    event.stopPropagation();
                    fileInput.current.click();
                  }}
                >
                  Choose image
                </button>
              </div>
            ) : (
              <div className="image-work">
                <img src={previewUrl} alt="Selected fundus image" />
                <div>
                  <em>
                    <CheckCircle2 size={13} aria-hidden="true" /> Image loaded
                  </em>
                  <b>{file.name}</b>
                  <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  <div className="actions">
                    <button
                      type="button"
                      className="primary"
                      onClick={analyze}
                      disabled={analyzing}
                    >
                      {analyzing ? (
                        <>
                          <Loader2 size={14} className="spin" aria-hidden="true" /> Analyzing…
                        </>
                      ) : (
                        'Analyze image'
                      )}
                    </button>
                    <button
                      type="button"
                      className="secondary"
                      onClick={() => fileInput.current.click()}
                      disabled={analyzing}
                    >
                      Change
                    </button>
                  </div>
                </div>
              </div>
            )}
            <input
              ref={fileInput}
              hidden
              type="file"
              accept={ACCEPT_ATTRIBUTE}
              onChange={(event) => {
                selectImage(event.target.files[0]);
                event.target.value = '';
              }}
            />
            {error && (
              <div className="error-note" role="alert">
                <AlertTriangle size={14} aria-hidden="true" />
                <span>{error}</span>
              </div>
            )}
          </Panel>

          {screening && (
            <VisualExplanation previewUrl={previewUrl} heatmap={screening.result.heatmap} />
          )}
        </div>

        <div>
          {screening ? (
            <Assessment screening={screening} classOrder={classOrder} />
          ) : (
            <EmptyState
              icon={analyzing ? Loader2 : Eye}
              eyebrow="AI ASSESSMENT"
              title={analyzing ? 'Running analysis…' : 'No screening result'}
            >
              {analyzing
                ? 'The image is being graded by the backend model.'
                : 'Upload a retinal image on the left to run the AI-assisted screening workflow.'}
            </EmptyState>
          )}
        </div>
      </div>
    </>
  );
}
