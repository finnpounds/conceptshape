"use client";

import { useTrainingStore } from "@/store/training";

const CATEGORY_LEGEND = [
  { id: "emotions",     label: "Emotions",  color: "#ff6b6b" },
  { id: "abstractions", label: "Abstract",  color: "#a8e6cf" },
  { id: "nature",       label: "Nature",    color: "#ff8c00" },
  { id: "mind",         label: "Mind",      color: "#b388ff" },
  { id: "relations",    label: "Relations", color: "#69d2e7" },
];

function fmtTokens(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

/** Map CKA [0,1] to dim red → bright green (same ramp as compare mode). */
function ckaColor(v: number): string {
  const r = Math.round(255 * (1 - v));
  const g = Math.round(180 * v + 60);
  return `rgb(${r},${g},60)`;
}

export default function TrainingControls() {
  const {
    data, isLoading, error,
    scrub, setScrub,
    isPlaying, setIsPlaying,
    playSpeed, setPlaySpeed,
    showTrails, setShowTrails,
    replay,
  } = useTrainingStore();

  const maxIdx = data ? data.steps.length - 1 : 0;
  const idx = Math.min(Math.round(scrub), maxIdx);
  const atEnd = data ? scrub >= maxIdx - 1e-6 : false;

  return (
    <div className="controls-panel">
      <div className="control-section">
        <label className="control-label">Watch a Model Learn</label>
        <p className="app-subtitle" style={{ marginBottom: 8 }}>
          Pythia-70M, re-loaded at {data ? data.steps.length : 9} checkpoints across
          its training run. Every point is a concept placed by its distance to the
          anchor concepts. At step 0 the geometry is random noise — scrub forward
          and watch the map of meaning crystallize.
        </p>
      </div>

      {error && (
        <div className="control-section">
          <p className="error-text">{error}</p>
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Scrubber */}
          <div className="control-section">
            <div className="slider-header">
              <label className="control-label">
                Step {data.steps[idx].toLocaleString()}
              </label>
              <div className="playback-controls">
                <button onClick={replay} className="playback-btn" title="Replay from step 0">⟲</button>
                <button
                  onClick={() => (atEnd ? replay() : setIsPlaying(!isPlaying))}
                  className="playback-btn"
                >
                  {isPlaying ? "⏸" : "▶"}
                </button>
              </div>
            </div>
            <input
              type="range" min={0} max={maxIdx} step={0.01} value={scrub}
              onChange={(e) => { setIsPlaying(false); setScrub(parseFloat(e.target.value)); }}
              className="layer-slider"
            />
            <div className="info-row">
              <span className="info-label">Tokens seen</span>
              <span className="info-value">{fmtTokens(data.tokens_seen[idx])}</span>
            </div>
            <div className="speed-row">
              <label className="info-label">Speed</label>
              <input type="range" min={0.2} max={3} step={0.1} value={playSpeed}
                onChange={(e) => setPlaySpeed(parseFloat(e.target.value))} className="speed-slider" />
              <span className="info-value">{playSpeed.toFixed(1)}x</span>
            </div>
          </div>

          {/* Crystallization curve */}
          <div className="control-section">
            <label className="control-label">Geometry match vs final model</label>
            <div className="recon-error-bar" style={{ height: 48 }}>
              {data.cka_to_final.map((v, i) => (
                <div
                  key={i}
                  className="recon-bar-seg"
                  style={{
                    height: `${Math.max(4, v * 100)}%`,
                    background: ckaColor(v),
                    opacity: i === idx ? 1 : 0.45,
                    outline: i === idx ? "1px solid #e0e0e8" : "none",
                  }}
                  title={`step ${data.steps[i].toLocaleString()}: CKA ${v.toFixed(2)}`}
                />
              ))}
            </div>
            <div className="info-row">
              <span className="info-label">← step 0 · step 143k →</span>
              <span className="info-value">CKA {data.cka_to_final[idx].toFixed(2)}</span>
            </div>
            <p className="app-subtitle" style={{ marginTop: 6 }}>
              The concept map stays near-random for the first ~8B tokens, then
              snaps into its final shape late in training.
            </p>
          </div>

          {/* Display options */}
          <div className="control-section">
            <label className="control-label">
              <input type="checkbox" checked={showTrails}
                onChange={(e) => setShowTrails(e.target.checked)} />{" "}
              Show Drift Trails
            </label>
          </div>

          {/* Legend */}
          <div className="control-section">
            <label className="control-label">Concept Categories</label>
            <div className="token-legend">
              {CATEGORY_LEGEND.map((c) => (
                <span key={c.id} className="token-chip"
                  style={{ borderColor: c.color, color: c.color }}>
                  {c.label}
                </span>
              ))}
            </div>
          </div>

          {/* Method note */}
          <div className="control-section">
            <div className="info-row">
              <span className="info-label">Model</span>
              <span className="info-value">{data.model}</span>
            </div>
            <div className="info-row">
              <span className="info-label">Checkpoints</span>
              <span className="info-value">{data.steps.length} (log-spaced)</span>
            </div>
            <div className="info-row">
              <span className="info-label">Var. explained</span>
              <span className="info-value">
                {(data.explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
