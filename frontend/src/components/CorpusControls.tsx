"use client";

import { useCallback, useRef, useState } from "react";
import { useCorpusStore, CORPUS_PRESETS } from "@/store/corpus";
import { compareShapes } from "@/lib/api";

// Per-text color palette — matches CorpusViewport
export const CORPUS_COLORS = [
  "#69d2e7", "#ff6b6b", "#a8e6cf", "#ffd700",
  "#ff8c00", "#b388ff", "#80cbc4", "#f032e6",
];

export default function CorpusControls() {
  const {
    corpusTexts, addCorpusText, removeCorpusText, updateLabel,
    corpusLayer, setCorpusLayer,
    corpusMaxPoints, setCorpusMaxPoints,
    corpusMetric, setCorpusMetric,
    corpusResults, setCorpusResults, clearCorpusResults,
    visibleShapes, toggleShapeVisibility,
    isCorpusLoading, setIsCorpusLoading,
    corpusError, setCorpusError,
    activeDiagramIndex, setActiveDiagramIndex,
    diagramView, setDiagramView,
  } = useCorpusStore();

  const [newText, setNewText] = useState("");
  const [newLabel, setNewLabel] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleAddText = () => {
    const t = newText.trim();
    if (!t) return;
    addCorpusText(t, newLabel.trim() || t.slice(0, 30));
    setNewText("");
    setNewLabel("");
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      if (text) addCorpusText(text, file.name.replace(/\.[^.]+$/, ""));
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const handlePreset = (preset: typeof CORPUS_PRESETS[0]) => {
    clearCorpusResults();
    for (const { text, label } of preset.texts) {
      addCorpusText(text, label);
    }
  };

  const handleAnalyze = useCallback(async () => {
    if (corpusTexts.length < 2) return;
    setIsCorpusLoading(true);
    setCorpusError(null);
    try {
      const data = await compareShapes(
        corpusTexts.map((t) => t.text),
        corpusTexts.map((t) => t.label),
        corpusLayer,
        corpusMaxPoints,
        corpusMetric,
      );
      setCorpusResults({
        shapes: data.shapes.map((s) => ({
          label: s.label,
          nTokensTotal: s.n_tokens_total,
          nPointsSampled: s.n_points_sampled,
          pointCloud3d: s.point_cloud_3d,
          persistenceDiagram: s.persistence_diagram,
          nComponents: s.n_components,
          nLoops: s.n_loops,
          explainedVariance: s.explained_variance,
        })),
        distanceMatrix: data.distance_matrix,
        labels: data.labels,
        metric: data.metric,
      });
    } catch (err) {
      setCorpusError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsCorpusLoading(false);
    }
  }, [corpusTexts, corpusLayer, corpusMaxPoints, corpusMetric]);

  const canAnalyze = corpusTexts.length >= 2 && !isCorpusLoading;

  return (
    <div className="controls-panel">
      {/* Presets */}
      <div className="control-section">
        <label className="control-label">Quick Presets</label>
        <div className="anchor-presets">
          {CORPUS_PRESETS.map((p) => (
            <button key={p.label} className="preset-btn" onClick={() => handlePreset(p)}>
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Add text */}
      <div className="control-section">
        <label className="control-label">Add Text</label>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <input
            type="text"
            value={newLabel}
            onChange={(e) => setNewLabel(e.target.value)}
            placeholder="Label (optional)"
            className="anchor-input"
          />
          <textarea
            value={newText}
            onChange={(e) => setNewText(e.target.value)}
            placeholder="Paste text here…"
            className="corpus-textarea"
            rows={4}
          />
          <div style={{ display: "flex", gap: 4 }}>
            <button
              className="anchor-compute-btn"
              style={{ flex: 1 }}
              onClick={handleAddText}
              disabled={!newText.trim()}
            >
              + Add
            </button>
            <button
              className="anchor-compute-btn"
              style={{ flex: 1 }}
              onClick={() => fileRef.current?.click()}
            >
              Upload .txt
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".txt"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />
          </div>
        </div>
      </div>

      {/* Text list */}
      {corpusTexts.length > 0 && (
        <div className="control-section">
          <label className="control-label">Texts ({corpusTexts.length})</label>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {corpusTexts.map((ct, i) => (
              <div key={ct.id} className="corpus-text-row">
                <span
                  className="corpus-text-dot"
                  style={{ background: CORPUS_COLORS[i % CORPUS_COLORS.length] }}
                />
                <input
                  type="text"
                  value={ct.label}
                  onChange={(e) => updateLabel(ct.id, e.target.value)}
                  className="anchor-input"
                  style={{ flex: 1, fontSize: 10 }}
                />
                <span className="corpus-text-count">
                  {ct.text.split(/\s+/).length}w
                </span>
                <button
                  className="anchor-remove-btn"
                  onClick={() => removeCorpusText(ct.id)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Analysis settings */}
      <div className="control-section">
        <label className="control-label">Settings</label>

        <div className="info-row">
          <span className="info-label">Layer</span>
          <span className="info-value">{corpusLayer === -1 ? "Last" : corpusLayer}</span>
        </div>
        <input
          type="range"
          min={-1}
          max={5}
          step={1}
          value={corpusLayer}
          onChange={(e) => setCorpusLayer(parseInt(e.target.value))}
          className="layer-slider"
        />

        <div className="info-row" style={{ marginTop: 4 }}>
          <span className="info-label">Max points</span>
          <span className="info-value">{corpusMaxPoints.toLocaleString()}</span>
        </div>
        <input
          type="range"
          min={200}
          max={5000}
          step={200}
          value={corpusMaxPoints}
          onChange={(e) => setCorpusMaxPoints(parseInt(e.target.value))}
          className="layer-slider"
        />

        <div className="info-row" style={{ marginTop: 4 }}>
          <span className="info-label">Diagram metric</span>
        </div>
        <div className="method-toggle">
          <button
            className={`method-btn ${corpusMetric === "wasserstein" ? "active" : ""}`}
            onClick={() => setCorpusMetric("wasserstein")}
            title="Earth-mover distance between diagrams — sensitive to persistence mass"
          >
            Wasserstein
          </button>
          <button
            className={`method-btn ${corpusMetric === "bottleneck" ? "active" : ""}`}
            onClick={() => setCorpusMetric("bottleneck")}
            title="Maximum matched point distance — robust to noise"
          >
            Bottleneck
          </button>
        </div>
      </div>

      {/* Run button */}
      <div className="control-section">
        <button
          className="anchor-compute-btn"
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          style={{ borderColor: "#69d2e7", color: "#69d2e7", background: "rgba(105,210,231,0.1)" }}
        >
          {isCorpusLoading
            ? `Analyzing ${corpusTexts.length} texts…`
            : `Analyze ${corpusTexts.length} Texts`}
        </button>
        {corpusError && <p className="error-text">{corpusError}</p>}
        {!canAnalyze && !isCorpusLoading && corpusTexts.length < 2 && (
          <p style={{ fontSize: 10, color: "var(--text-dim)", marginTop: 4 }}>
            Add at least 2 texts to compare.
          </p>
        )}
      </div>

      {/* Results */}
      {corpusResults && (
        <>
          {/* Shape visibility */}
          <div className="control-section">
            <label className="control-label">Point Clouds</label>
            {corpusResults.shapes.map((shape, i) => (
              <label key={i} className="model-checkbox-label" style={{ marginBottom: 3 }}>
                <input
                  type="checkbox"
                  checked={visibleShapes.has(i)}
                  onChange={() => toggleShapeVisibility(i)}
                  style={{ accentColor: CORPUS_COLORS[i % CORPUS_COLORS.length] }}
                />
                <span
                  className="model-checkbox-name"
                  style={{ color: CORPUS_COLORS[i % CORPUS_COLORS.length] }}
                >
                  {shape.label}
                </span>
                <span className="model-checkbox-note">
                  {shape.nPointsSampled.toLocaleString()}pts · H0:{shape.nComponents} H1:{shape.nLoops}
                </span>
              </label>
            ))}
          </div>

          {/* Distance matrix */}
          {corpusResults.distanceMatrix.length > 1 && (
            <div className="control-section">
              <label className="control-label">Topology Distance ({corpusResults.metric})</label>
              <div className="distance-table-wrap">
                <table className="distance-table">
                  <thead>
                    <tr>
                      <th className="dt-token-head" />
                      {corpusResults.labels.map((l, i) => (
                        <th key={i} className="dt-anchor-head" title={l}>
                          {l.length > 6 ? l.slice(0, 6) + "…" : l}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {corpusResults.distanceMatrix.map((row, i) => {
                      const maxD = Math.max(...corpusResults.distanceMatrix.flat(), 0.001);
                      return (
                        <tr key={i}>
                          <td className="dt-token" style={{ color: CORPUS_COLORS[i % CORPUS_COLORS.length] }}>
                            {corpusResults.labels[i].slice(0, 8)}
                          </td>
                          {row.map((d, j) => (
                            <td
                              key={j}
                              className="dt-dist"
                              style={{ opacity: i === j ? 0.15 : 0.4 + (1 - d / maxD) * 0.6 }}
                              title={`${corpusResults.labels[i]} vs ${corpusResults.labels[j]}: ${d.toFixed(3)}`}
                            >
                              {i === j ? "—" : d.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Persistence diagram selector */}
          <div className="control-section">
            <label className="control-label">Persistence Diagram</label>
            <div className="method-toggle" style={{ marginBottom: 4 }}>
              <button
                className={`method-btn ${diagramView === "scatter" ? "active" : ""}`}
                onClick={() => setDiagramView("scatter")}
              >
                Scatter
              </button>
              <button
                className={`method-btn ${diagramView === "barcode" ? "active" : ""}`}
                onClick={() => setDiagramView("barcode")}
              >
                Barcode
              </button>
            </div>
            <div className="anchor-presets" style={{ marginBottom: 4 }}>
              {corpusResults.shapes.map((s, i) => (
                <button
                  key={i}
                  className={`preset-btn ${activeDiagramIndex === i ? "active" : ""}`}
                  style={activeDiagramIndex === i
                    ? { borderColor: CORPUS_COLORS[i % CORPUS_COLORS.length], color: CORPUS_COLORS[i % CORPUS_COLORS.length] }
                    : {}}
                  onClick={() => setActiveDiagramIndex(i)}
                >
                  {s.label.slice(0, 8)}
                </button>
              ))}
            </div>
            <PersistenceDiagramPanel />
          </div>
        </>
      )}
    </div>
  );
}

/** Inline SVG persistence diagram (scatter or barcode). */
function PersistenceDiagramPanel() {
  const { corpusResults, activeDiagramIndex, diagramView } = useCorpusStore();
  if (!corpusResults) return null;

  const shape = corpusResults.shapes[activeDiagramIndex];
  if (!shape) return null;

  const W = 240, H = 180;
  const PAD = 24;

  const features = shape.persistenceDiagram;
  if (features.length === 0) {
    return <p style={{ fontSize: 10, color: "var(--text-dim)" }}>No finite features.</p>;
  }

  const DIM_COLORS: Record<number, string> = { 0: "#69d2e7", 1: "#a8e6cf", 2: "#ff6b6b" };

  if (diagramView === "barcode") {
    const sorted = [...features].sort((a, b) => b.persistence - a.persistence).slice(0, 40);
    const maxDeath = Math.max(...sorted.map((f) => f.death), 0.01);
    const barH = Math.max(3, Math.floor((H - PAD) / sorted.length) - 1);

    return (
      <svg width={W} height={H} style={{ display: "block" }}>
        {sorted.map((f, i) => {
          const x1 = PAD + (f.birth / maxDeath) * (W - PAD * 2);
          const x2 = PAD + (f.death / maxDeath) * (W - PAD * 2);
          const y = PAD / 2 + i * (barH + 1);
          return (
            <rect
              key={i}
              x={x1}
              y={y}
              width={Math.max(1, x2 - x1)}
              height={barH}
              fill={DIM_COLORS[f.dimension] ?? "#aaa"}
              opacity={0.75}
            >
              <title>H{f.dimension}: [{f.birth.toFixed(3)}, {f.death.toFixed(3)}] p={f.persistence.toFixed(3)}</title>
            </rect>
          );
        })}
        <text x={PAD} y={H - 4} fontSize={8} fill="#555566">birth →</text>
      </svg>
    );
  }

  // Scatter diagram
  const allBirths = features.map((f) => f.birth);
  const allDeaths = features.map((f) => f.death);
  const maxVal = Math.max(...allDeaths, 0.01);

  const toX = (v: number) => PAD + (v / maxVal) * (W - PAD * 1.5);
  const toY = (v: number) => H - PAD - (v / maxVal) * (H - PAD * 1.5);

  return (
    <svg width={W} height={H} style={{ display: "block" }}>
      {/* Diagonal (birth = death) */}
      <line
        x1={toX(0)} y1={toY(0)} x2={toX(maxVal)} y2={toY(maxVal)}
        stroke="#2a2a3a" strokeWidth={1}
      />
      {features.map((f, i) => (
        <circle
          key={i}
          cx={toX(f.birth)}
          cy={toY(f.death)}
          r={Math.max(2, Math.min(5, f.persistence * 12))}
          fill={DIM_COLORS[f.dimension] ?? "#aaa"}
          opacity={0.7}
        >
          <title>H{f.dimension}: birth={f.birth.toFixed(3)} death={f.death.toFixed(3)} p={f.persistence.toFixed(3)}</title>
        </circle>
      ))}
      {/* Axis labels */}
      <text x={PAD} y={H - 6} fontSize={8} fill="#555566">birth →</text>
      <text x={4} y={PAD} fontSize={8} fill="#555566" writingMode="vertical-lr" style={{ textOrientation: "mixed" }}>← death</text>
      {/* Legend */}
      {[0, 1].map((dim) => (
        <g key={dim} transform={`translate(${W - 55}, ${12 + dim * 12})`}>
          <circle cx={4} cy={4} r={3} fill={DIM_COLORS[dim]} opacity={0.8} />
          <text x={10} y={8} fontSize={8} fill="#888899">H{dim}</text>
        </g>
      ))}
    </svg>
  );
}
