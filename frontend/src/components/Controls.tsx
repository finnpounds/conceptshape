"use client";

import { useCallback } from "react";
import { useExplorerStore } from "@/store/explorer";
import { useState } from "react";
import { analyzeText, analyzeAnchors, runComparison, probeConceptsAPI } from "@/lib/api";

const TOKEN_COLORS = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
  "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
  "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
  "#000075", "#a9a9a9",
];

const ANCHOR_PRESETS: { label: string; anchors: string[] }[] = [
  { label: "self/world", anchors: ["self", "other", "world", "logic"] },
  { label: "valence",    anchors: ["positive", "negative", "neutral"] },
  { label: "cognition",  anchors: ["logic", "emotion", "perception", "memory"] },
  { label: "abstract",   anchors: ["concrete", "abstract", "specific", "general"] },
];

// Concept vocabulary categories with matching backend colors
const PROBE_CATEGORIES = [
  { id: "emotions",    label: "Emotions",    color: "#ff6b6b" },
  { id: "relations",   label: "Relations",   color: "#69d2e7" },
  { id: "abstractions",label: "Abstract",    color: "#a8e6cf" },
  { id: "states",      label: "States",      color: "#ffd700" },
  { id: "nature",      label: "Nature",      color: "#ff8c00" },
  { id: "mind",        label: "Mind",        color: "#b388ff" },
  { id: "qualities",   label: "Qualities",   color: "#80cbc4" },
];

// Full vocabulary matching backend CONCEPT_VOCABULARY
const CONCEPT_VOCABULARY: Record<string, string[]> = {
  emotions:    ["love","fear","joy","anger","sadness","hope","trust","surprise","guilt","pride","shame","compassion"],
  relations:   ["friend","enemy","parent","child","teacher","student","ally","rival","partner","stranger","leader","follower"],
  abstractions:["truth","justice","beauty","freedom","time","death","life","power","knowledge","meaning","order","chaos"],
  states:      ["alive","dead","happy","sad","sick","healthy","young","old","strong","weak","awake","asleep"],
  nature:      ["water","fire","earth","sky","sun","moon","stone","tree","mountain","ocean","wind","light"],
  mind:        ["thought","belief","desire","memory","dream","consciousness","reason","emotion","will","perception"],
  qualities:   ["good","evil","beautiful","ugly","true","false","real","imaginary","certain","uncertain","simple","complex"],
};

const AVAILABLE_MODELS = [
  { id: "pythia-70m",  label: "Pythia-70M",  note: "6L · 512d · fast" },
  { id: "gpt2",        label: "GPT-2 Small",  note: "12L · 768d" },
  { id: "pythia-160m", label: "Pythia-160M",  note: "12L · 768d" },
];

// Per-model color palette for the compare legend
const MODEL_COLORS = ["#ff6b6b", "#69d2e7", "#a8e6cf"];

export default function Controls() {
  const {
    inputText, setInputText,
    isLoading, setIsLoading,
    error, setError,
    setTrajectoryData,
    currentLayer, setCurrentLayer,
    nLayers,
    isPlaying, setIsPlaying,
    playSpeed, setPlaySpeed,
    showAttention, setShowAttention,
    attentionThreshold, setAttentionThreshold,
    projectionMethod, setProjectionMethod,
    tokens, explainedVariance, modelName,
    logitLens,
    hideBOS, setHideBOS,
    spread, setSpread,
    // Anchor
    viewMode, setViewMode,
    anchorInputs, setAnchorInputs,
    anchorLabels, anchorDistances, anchorExplainedVariance,
    anchorTrajectories,
    isAnchorLoading, setIsAnchorLoading,
    anchorError, setAnchorError,
    setAnchorData,
    // Compare
    selectedModels, setSelectedModels,
    compareData, comparePairwise, compareExplainedVariance,
    compareLayer, setCompareLayer,
    isCompareLoading, setIsCompareLoading,
    compareError, setCompareError,
    setCompareData,
    // Probe
    probeResults, probeReconstructionErrors, probeNLayers,
    isProbeLoading, setIsProbeLoading,
    probeError, setProbeError,
    setProbeData,
    probeSelectedCategories, setProbeSelectedCategories,
    probeCustomConcepts, setProbeCustomConcepts,
  } = useExplorerStore();

  const [probeCustomInput, setProbeCustomInput] = useState("");

  // --- Absolute analysis ---
  const handleAnalyze = useCallback(async () => {
    if (!inputText.trim()) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyzeText(inputText, projectionMethod);
      setTrajectoryData({
        tokens: data.tokens,
        trajectories: data.trajectories,
        attention: data.attention,
        logitLens: data.logit_lens,
        nLayers: data.n_layers,
        explainedVariance: data.explained_variance,
        modelName: data.model_name,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setIsLoading(false);
    }
  }, [inputText, projectionMethod]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleAnalyze(); }
  };

  // --- Anchor analysis ---
  const handleAnchorAnalyze = useCallback(async () => {
    const validAnchors = anchorInputs.map((a) => a.trim()).filter(Boolean);
    if (!inputText.trim() || validAnchors.length < 2) return;
    setIsAnchorLoading(true);
    setAnchorError(null);
    try {
      const data = await analyzeAnchors(inputText, validAnchors);
      setAnchorData({
        anchorTrajectories: data.trajectories,
        anchorMarkers: data.anchor_markers,
        anchorDistances: data.distances,
        anchorLabels: data.anchors,
        anchorExplainedVariance: data.explained_variance,
      });
    } catch (err) {
      setAnchorError(err instanceof Error ? err.message : "Anchor analysis failed");
    } finally {
      setIsAnchorLoading(false);
    }
  }, [inputText, anchorInputs]);

  // --- Compare ---
  const handleCompare = useCallback(async () => {
    const validAnchors = anchorInputs.map((a) => a.trim()).filter(Boolean);
    if (!inputText.trim() || selectedModels.length < 2 || validAnchors.length < 2) return;
    setIsCompareLoading(true);
    setCompareError(null);
    try {
      const data = await runComparison(inputText, selectedModels, validAnchors);
      setCompareData({
        compareData: data.models.map((m) => ({
          modelName: m.model_name,
          tokens: m.tokens,
          nLayers: m.n_layers,
          trajectories: m.trajectories,
        })),
        comparePairwise: data.pairwise,
        compareExplainedVariance: data.explained_variance,
      });
    } catch (err) {
      setCompareError(err instanceof Error ? err.message : "Comparison failed");
    } finally {
      setIsCompareLoading(false);
    }
  }, [inputText, selectedModels, anchorInputs]);

  // --- Probe ---
  const handleProbe = useCallback(async () => {
    const validAnchors = anchorInputs.map((a) => a.trim()).filter(Boolean);
    if (validAnchors.length < 2) return;

    // Collect probes from selected categories + custom concepts
    const probes: string[] = [];
    for (const cat of probeSelectedCategories) {
      probes.push(...(CONCEPT_VOCABULARY[cat] ?? []));
    }
    for (const c of probeCustomConcepts) {
      if (c.trim() && !probes.includes(c.trim())) probes.push(c.trim());
    }
    if (probes.length === 0) return;

    setIsProbeLoading(true);
    setProbeError(null);
    try {
      const data = await probeConceptsAPI(validAnchors, probes);
      setProbeData({
        probeResults: data.probes,
        probeAnchorMarkers: data.anchor_markers,
        probeReconstructionErrors: data.reconstruction_errors,
        probeExplainedVariance: data.explained_variance,
        probeNLayers: data.n_layers,
      });
    } catch (err) {
      setProbeError(err instanceof Error ? err.message : "Probe failed");
    } finally {
      setIsProbeLoading(false);
    }
  }, [anchorInputs, probeSelectedCategories, probeCustomConcepts]);

  const toggleProbeCategory = (id: string) => {
    setProbeSelectedCategories(
      probeSelectedCategories.includes(id)
        ? probeSelectedCategories.filter((c) => c !== id)
        : [...probeSelectedCategories, id]
    );
  };

  const addCustomConcept = () => {
    const trimmed = probeCustomInput.trim();
    if (!trimmed || probeCustomConcepts.includes(trimmed)) return;
    setProbeCustomConcepts([...probeCustomConcepts, trimmed]);
    setProbeCustomInput("");
  };

  const removeCustomConcept = (i: number) => {
    setProbeCustomConcepts(probeCustomConcepts.filter((_, idx) => idx !== i));
  };

  const toggleModel = (id: string) => {
    setSelectedModels(
      selectedModels.includes(id)
        ? selectedModels.filter((m) => m !== id)
        : [...selectedModels, id]
    );
  };

  const updateAnchorInput = (i: number, value: string) => {
    const next = [...anchorInputs];
    next[i] = value;
    setAnchorInputs(next);
  };

  const addAnchorInput = () => {
    if (anchorInputs.length < 8) setAnchorInputs([...anchorInputs, ""]);
  };

  const removeAnchorInput = (i: number) => {
    if (anchorInputs.length <= 2) return;
    setAnchorInputs(anchorInputs.filter((_, idx) => idx !== i));
  };

  const visibleTokens = tokens
    .map((tok, i) => ({ tok, i }))
    .filter(({ i }) => !(hideBOS && i === 0));

  const layerIdx = Math.min(Math.floor(currentLayer), nLayers);
  const hasAnchorData = anchorTrajectories.length > 0;
  const hasAbsoluteData = nLayers > 0;
  const hasCompareData = compareData.length > 0;
  const hasProbeData = probeResults.length > 0;
  const activeNLayers = viewMode === "probe" ? probeNLayers : nLayers;
  const probeCount = probeSelectedCategories.reduce(
    (n, cat) => n + (CONCEPT_VOCABULARY[cat]?.length ?? 0), 0
  ) + probeCustomConcepts.length;

  return (
    <div className="controls-panel">
      {/* Text Input */}
      <div className="control-section">
        <label className="control-label">Input Text</label>
        <div className="input-row">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter text to analyze..."
            className="text-input"
            disabled={isLoading}
          />
          <button onClick={handleAnalyze} disabled={isLoading || !inputText.trim()} className="analyze-btn">
            {isLoading ? "..." : "→"}
          </button>
        </div>
        {error && <p className="error-text">{error}</p>}
      </div>

      {/* View Mode */}
      {(hasAbsoluteData || hasAnchorData || hasCompareData || hasProbeData) && (
        <div className="control-section">
          <label className="control-label">View Mode</label>
          <div className="method-toggle">
            <button
              className={`method-btn ${viewMode === "absolute" ? "active" : ""}`}
              onClick={() => setViewMode("absolute")}
              disabled={!hasAbsoluteData}
              title="PCA of raw residual stream — global trajectory through layers"
            >
              Absolute
            </button>
            <button
              className={`method-btn ${viewMode === "anchor" ? "active" : ""}`}
              onClick={() => setViewMode("anchor")}
              disabled={!hasAnchorData}
              title="Token positions in anchor-concept distance space"
            >
              Anchor
            </button>
            <button
              className={`method-btn ${viewMode === "compare" ? "active" : ""}`}
              onClick={() => setViewMode("compare")}
              disabled={!hasCompareData}
              title="Multiple models in shared anchor-relative space"
            >
              Compare
            </button>
            <button
              className={`method-btn ${viewMode === "probe" ? "active" : ""}`}
              onClick={() => setViewMode("probe")}
              disabled={!hasProbeData}
              title="Concept vocabulary embedded in anchor-relative space"
            >
              Probe
            </button>
          </div>
        </div>
      )}

      {/* Model Info */}
      {modelName && viewMode !== "compare" && (
        <div className="control-section">
          <div className="info-row">
            <span className="info-label">Model</span>
            <span className="info-value">{modelName}</span>
          </div>
          {viewMode === "absolute" && explainedVariance.length > 0 && (
            <div className="info-row">
              <span className="info-label">Var. explained</span>
              <span className="info-value">
                {(explainedVariance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {viewMode === "anchor" && anchorExplainedVariance.length > 0 && (
            <div className="info-row">
              <span className="info-label">Anchor var.</span>
              <span className="info-value">
                {(anchorExplainedVariance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {viewMode === "probe" && probeResults.length > 0 && (
            <div className="info-row">
              <span className="info-label">Probes</span>
              <span className="info-value">{probeResults.length} concepts</span>
            </div>
          )}
        </div>
      )}

      {/* Compare model legend */}
      {viewMode === "compare" && hasCompareData && (
        <div className="control-section">
          {compareData.map((m, mi) => (
            <div key={m.modelName} className="info-row">
              <span className="model-legend-dot" style={{ color: MODEL_COLORS[mi % MODEL_COLORS.length] }}>■</span>
              <span className="info-value">{m.modelName}</span>
              <span className="info-label">{m.nLayers}L · {m.tokens.length}tok</span>
            </div>
          ))}
          {compareExplainedVariance.length > 0 && (
            <div className="info-row">
              <span className="info-label">Joint var.</span>
              <span className="info-value">
                {(compareExplainedVariance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      )}

      {/* Token Legend — not shown in compare mode */}
      {tokens.length > 0 && viewMode !== "compare" && (
        <div className="control-section">
          <label className="control-label">Tokens</label>
          <div className="token-legend">
            {visibleTokens.map(({ tok, i }) => (
              <span key={i} className="token-chip" style={{
                borderColor: TOKEN_COLORS[i % TOKEN_COLORS.length],
                color: TOKEN_COLORS[i % TOKEN_COLORS.length],
              }}>
                {tok}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Layer Scrubber — absolute / anchor / probe modes */}
      {activeNLayers > 0 && viewMode !== "compare" && (
        <div className="control-section">
          <div className="slider-header">
            <label className="control-label">
              Layer {Math.floor(currentLayer)} / {activeNLayers}
            </label>
            <div className="playback-controls">
              <button onClick={() => setCurrentLayer(0)} className="playback-btn" title="Reset">⟲</button>
              <button onClick={() => setIsPlaying(!isPlaying)} className="playback-btn">
                {isPlaying ? "⏸" : "▶"}
              </button>
            </div>
          </div>
          <input
            type="range" min={0} max={activeNLayers} step={0.05} value={currentLayer}
            onChange={(e) => { setIsPlaying(false); setCurrentLayer(parseFloat(e.target.value)); }}
            className="layer-slider"
          />
          <div className="speed-row">
            <label className="info-label">Speed</label>
            <input type="range" min={0.2} max={3} step={0.1} value={playSpeed}
              onChange={(e) => setPlaySpeed(parseFloat(e.target.value))} className="speed-slider" />
            <span className="info-value">{playSpeed.toFixed(1)}x</span>
          </div>
        </div>
      )}

      {/* Logit Lens — what each token predicts next, decoded from the current layer */}
      {viewMode === "absolute" && logitLens.length > 0 && (
        <div className="control-section">
          <label className="control-label">
            Logit Lens · layer {Math.min(Math.round(currentLayer), nLayers)}
          </label>
          <p className="lens-hint">
            what each token “expects” next, decoded from this layer — scrub the
            slider to watch guesses resolve
          </p>
          <div className="lens-table">
            {visibleTokens.map(({ tok, i }) => {
              const li = Math.min(Math.round(currentLayer), nLayers);
              const preds = logitLens[li]?.[i] ?? [];
              const top = preds[0];
              if (!top) return null;
              const show = (s: string) => (s.trim() === "" ? "␣" : s.trim());
              return (
                <div
                  key={i}
                  className="lens-row"
                  title={preds
                    .map((p) => `${show(p.token)}  ${(p.prob * 100).toFixed(0)}%`)
                    .join("   ·   ")}
                >
                  <span
                    className="lens-src"
                    style={{ color: TOKEN_COLORS[i % TOKEN_COLORS.length] }}
                  >
                    {show(tok)}
                  </span>
                  <span className="lens-arrow">→</span>
                  <span className="lens-pred">{show(top.token)}</span>
                  <span className="lens-prob">
                    <span
                      className="lens-prob-fill"
                      style={{ width: `${Math.max(3, top.prob * 100)}%` }}
                    />
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Compare Depth Scrubber */}
      {viewMode === "compare" && hasCompareData && (
        <div className="control-section">
          <div className="slider-header">
            <label className="control-label">
              Depth {(compareLayer * 100).toFixed(0)}%
            </label>
            <div className="playback-controls">
              <button onClick={() => { setIsPlaying(false); setCompareLayer(0); }} className="playback-btn" title="Reset">⟲</button>
              <button onClick={() => setIsPlaying(!isPlaying)} className="playback-btn">
                {isPlaying ? "⏸" : "▶"}
              </button>
            </div>
          </div>
          <input
            type="range" min={0} max={1} step={0.01} value={compareLayer}
            onChange={(e) => { setIsPlaying(false); setCompareLayer(parseFloat(e.target.value)); }}
            className="layer-slider"
          />
          <div className="speed-row">
            <label className="info-label">Speed</label>
            <input type="range" min={0.2} max={3} step={0.1} value={playSpeed}
              onChange={(e) => setPlaySpeed(parseFloat(e.target.value))} className="speed-slider" />
            <span className="info-value">{playSpeed.toFixed(1)}x</span>
          </div>
        </div>
      )}

      {/* Spread */}
      {(hasAbsoluteData || hasCompareData) && (
        <div className="control-section">
          <div className="speed-row">
            <label className="info-label">Spread</label>
            <input type="range" min={0.5} max={6} step={0.1} value={spread}
              onChange={(e) => setSpread(parseFloat(e.target.value))} className="speed-slider" />
            <span className="info-value">{spread.toFixed(1)}x</span>
          </div>
        </div>
      )}

      {/* Convergence Metrics — compare mode */}
      {viewMode === "compare" && Object.keys(comparePairwise).length > 0 && (
        <div className="control-section">
          <label className="control-label">Convergence (CKA)</label>
          {Object.entries(comparePairwise).map(([pair, metrics]) => (
            <div key={pair} className="convergence-block">
              <div className="convergence-pair">{pair}</div>
              <div className="convergence-row">
                {[0, 3, 6].map((fi) => (
                  <div key={fi} className="convergence-cell">
                    <span className="convergence-label">
                      {(metrics.fractions[fi] * 100).toFixed(0)}%
                    </span>
                    <span
                      className="convergence-value"
                      style={{ color: ckaColor(metrics.cka[fi]) }}
                    >
                      {metrics.cka[fi].toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
              <div className="convergence-bar">
                {metrics.cka.map((v, i) => (
                  <div
                    key={i}
                    className="convergence-bar-seg"
                    style={{
                      flex: 1,
                      background: ckaColor(v),
                      opacity: 0.4 + v * 0.6,
                    }}
                    title={`${(metrics.fractions[i] * 100).toFixed(0)}%: CKA ${v.toFixed(2)}`}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Attention Controls — absolute mode only */}
      {nLayers > 0 && viewMode === "absolute" && (
        <div className="control-section">
          <label className="control-label">
            <input type="checkbox" checked={showAttention}
              onChange={(e) => setShowAttention(e.target.checked)} />{" "}
            Show Attention Edges
          </label>
          {showAttention && (
            <div className="speed-row">
              <label className="info-label">Threshold</label>
              <input type="range" min={0.01} max={0.5} step={0.01} value={attentionThreshold}
                onChange={(e) => setAttentionThreshold(parseFloat(e.target.value))} className="speed-slider" />
              <span className="info-value">{attentionThreshold.toFixed(2)}</span>
            </div>
          )}
        </div>
      )}

      {/* BOS */}
      {(hasAbsoluteData || hasCompareData) && (
        <div className="control-section">
          <label className="control-label">
            <input type="checkbox" checked={hideBOS}
              onChange={(e) => setHideBOS(e.target.checked)} />{" "}
            Hide BOS Token
          </label>
        </div>
      )}

      {/* Projection — absolute mode only */}
      {viewMode === "absolute" && (
        <div className="control-section">
          <label className="control-label">Projection</label>
          <div className="method-toggle">
            <button
              className={`method-btn ${projectionMethod === "pca" ? "active" : ""}`}
              onClick={() => setProjectionMethod("pca")}
              title="PCA — preserves global structure. Best for watching tokens diverge through layers."
            >
              PCA
            </button>
            <button
              className={`method-btn ${projectionMethod === "umap" ? "active" : ""}`}
              onClick={() => setProjectionMethod("umap")}
              title="UMAP — preserves local neighborhoods. Best for seeing which tokens cluster together at a given layer."
            >
              UMAP
            </button>
          </div>
        </div>
      )}

      {/* Concept Probe (M5) */}
      <div className="control-section">
        <label className="control-label">Concept Probe</label>
        <div className="probe-categories">
          {PROBE_CATEGORIES.map((cat) => {
            const active = probeSelectedCategories.includes(cat.id);
            return (
              <button
                key={cat.id}
                className={`probe-cat-btn ${active ? "active" : ""}`}
                style={active ? { background: cat.color, borderColor: cat.color, color: "#0a0a0f" } : {}}
                onClick={() => toggleProbeCategory(cat.id)}
                title={CONCEPT_VOCABULARY[cat.id]?.slice(0, 4).join(", ") + "…"}
              >
                {cat.label}
              </button>
            );
          })}
        </div>

        <div className="anchor-input-row" style={{ marginBottom: 6 }}>
          <input
            type="text"
            value={probeCustomInput}
            onChange={(e) => setProbeCustomInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") addCustomConcept(); }}
            placeholder="add custom concept…"
            className="anchor-input"
          />
          <button className="anchor-add-btn" onClick={addCustomConcept}
            style={{ flexShrink: 0, padding: "0 8px" }}>+</button>
        </div>

        {probeCustomConcepts.length > 0 && (
          <div className="probe-chip-list">
            {probeCustomConcepts.map((c, i) => (
              <span key={i} className="probe-chip">
                {c}
                <button className="probe-chip-remove" onClick={() => removeCustomConcept(i)}>×</button>
              </span>
            ))}
          </div>
        )}

        <button
          className="probe-embed-btn"
          onClick={handleProbe}
          disabled={
            isProbeLoading ||
            probeCount === 0 ||
            anchorInputs.filter((a) => a.trim()).length < 2
          }
        >
          {isProbeLoading
            ? `Embedding ${probeCount} concepts…`
            : `Embed ${probeCount} Concepts`}
        </button>
        {probeError && <p className="error-text">{probeError}</p>}

        {probeReconstructionErrors.length > 0 && (
          <>
            <div className="info-row" style={{ marginTop: 8 }}>
              <span className="info-label">Recon. error vs anchors</span>
            </div>
            <div className="recon-error-bar">
              {probeReconstructionErrors.map((pt, i) => {
                const maxErr = Math.max(...probeReconstructionErrors.map((p) => p.error), 0.001);
                return (
                  <div
                    key={i}
                    className="recon-bar-seg"
                    style={{ height: `${Math.max(4, (pt.error / maxErr) * 100)}%` }}
                    title={`${pt.n_anchors} anchors: err ${pt.error.toFixed(3)}`}
                  />
                );
              })}
            </div>
            <div className="info-row">
              <span className="info-label">← fewer anchors · more →</span>
            </div>
          </>
        )}
      </div>

      {/* Anchor Concepts */}
      <div className="control-section">
        <label className="control-label">Anchor Concepts</label>
        <div className="anchor-presets">
          {ANCHOR_PRESETS.map((preset) => (
            <button key={preset.label} className="preset-btn"
              onClick={() => setAnchorInputs(preset.anchors)} title={preset.anchors.join(", ")}>
              {preset.label}
            </button>
          ))}
        </div>
        <div className="anchor-inputs">
          {anchorInputs.map((val, i) => (
            <div key={i} className="anchor-input-row">
              <input type="text" value={val}
                onChange={(e) => updateAnchorInput(i, e.target.value)}
                placeholder={`anchor ${i + 1}`} className="anchor-input" />
              <button className="anchor-remove-btn" onClick={() => removeAnchorInput(i)}
                disabled={anchorInputs.length <= 2}>×</button>
            </div>
          ))}
          {anchorInputs.length < 8 && (
            <button className="anchor-add-btn" onClick={addAnchorInput}>+ add anchor</button>
          )}
        </div>
        <button className="anchor-compute-btn" onClick={handleAnchorAnalyze}
          disabled={isAnchorLoading || !inputText.trim() || anchorInputs.filter((a) => a.trim()).length < 2}>
          {isAnchorLoading ? "Computing..." : "Compute Anchor View"}
        </button>
        {anchorError && <p className="error-text">{anchorError}</p>}
      </div>

      {/* Cross-Model Comparison */}
      <div className="control-section">
        <label className="control-label">Compare Models</label>
        <div className="model-checkboxes">
          {AVAILABLE_MODELS.map((m) => (
            <label key={m.id} className="model-checkbox-label">
              <input
                type="checkbox"
                checked={selectedModels.includes(m.id)}
                onChange={() => toggleModel(m.id)}
                disabled={selectedModels.includes(m.id) && selectedModels.length <= 2}
              />
              <span className="model-checkbox-name">{m.label}</span>
              <span className="model-checkbox-note">{m.note}</span>
            </label>
          ))}
        </div>
        <button className="anchor-compute-btn" onClick={handleCompare}
          disabled={
            isCompareLoading ||
            !inputText.trim() ||
            selectedModels.length < 2 ||
            anchorInputs.filter((a) => a.trim()).length < 2
          }>
          {isCompareLoading ? "Running (may take ~30s for new models)..." : "Run Comparison"}
        </button>
        {compareError && <p className="error-text">{compareError}</p>}
      </div>

      {/* Distance Table — anchor mode only */}
      {viewMode === "anchor" && hasAnchorData && anchorDistances.length > 0 && (
        <div className="control-section">
          <label className="control-label">Anchor Distances — Layer {layerIdx}</label>
          <div className="distance-table-wrap">
            <table className="distance-table">
              <thead>
                <tr>
                  <th className="dt-token-head"></th>
                  {anchorLabels.map((lbl) => (
                    <th key={lbl} className="dt-anchor-head" title={lbl}>
                      {lbl.length > 5 ? lbl.slice(0, 5) + "…" : lbl}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visibleTokens.map(({ tok, i }) => {
                  const row = anchorDistances[i]?.[layerIdx] ?? [];
                  return (
                    <tr key={i}>
                      <td className="dt-token" style={{ color: TOKEN_COLORS[i % TOKEN_COLORS.length] }}>
                        {tok}
                      </td>
                      {row.map((d, ai) => (
                        <td key={ai} className="dt-dist" style={{ opacity: 0.4 + (1 - d) * 0.6 }}>
                          {d.toFixed(2)}
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
    </div>
  );
}

/** Map CKA [0,1] to a color from dim red → bright green */
function ckaColor(cka: number): string {
  const r = Math.round(255 * (1 - cka));
  const g = Math.round(180 * cka + 60);
  return `rgb(${r},${g},60)`;
}
