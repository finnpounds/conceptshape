import { create } from "zustand";

export interface TokenTrajectory {
  token: string;
  positions: number[][]; // [n_layers+1][3]
}

export interface AttentionEdge {
  layer: number;
  head: number;
  weights: number[][]; // [n_tokens][n_tokens]
}

export interface LensPrediction {
  token: string;
  prob: number;
}

export interface AnchorMarker {
  label: string;
  positions: number[][]; // [n_layers+1][3]
}

export interface ProbeResult {
  label: string;
  category: string;
  positions: number[][]; // [n_layers+1][3]
  uncertainty: number;   // [0,1] — min cosine dist to nearest anchor at final layer
}

export interface CompareModelData {
  modelName: string;
  tokens: string[];
  nLayers: number;
  trajectories: TokenTrajectory[]; // anchor-relative, jointly projected
}

export interface PairMetrics {
  fractions: number[];
  cka: number[];
  procrustes: number[];
  cka_null?: number[];  // 95th-percentile CKA with rows shuffled (chance level)
  n_samples?: number;   // aligned words the metrics were computed on
}

// Position 0 is the BOS attention sink. The backend keeps it out of every
// PCA fit and scales the scene to the content tokens, so BOS can legitimately
// land outside the [-1, 1] box (≈2x at mid layers, ≈3.5x in the steer view).
// For DISPLAY only, pull it in to a fixed radius along its own direction so
// "show BOS" still puts a sphere on screen; nothing else reads these values.
const BOS_TOKEN = "<|endoftext|>";
const BOS_DISPLAY_RADIUS = 1.5;
function clampBosForDisplay(trajectories: TokenTrajectory[]): TokenTrajectory[] {
  if (trajectories.length === 0 || trajectories[0].token !== BOS_TOKEN) return trajectories;
  const bos = trajectories[0];
  const positions = bos.positions.map((p) => {
    const r = Math.hypot(p[0], p[1], p[2]);
    if (r <= BOS_DISPLAY_RADIUS) return p;
    const k = BOS_DISPLAY_RADIUS / r;
    return [p[0] * k, p[1] * k, p[2] * k];
  });
  return [{ ...bos, positions }, ...trajectories.slice(1)];
}

interface ExplorerState {
  // Absolute view data (M1)
  tokens: string[];
  trajectories: TokenTrajectory[];
  attention: AttentionEdge[];
  logitLens: LensPrediction[][][]; // [n_layers+1][n_tokens][k]
  nLayers: number;
  explainedVariance: number[];
  modelName: string;

  // Anchor-relative view data (M3)
  viewMode: "absolute" | "anchor" | "compare" | "probe" | "steer";
  anchorInputs: string[];
  anchorTrajectories: TokenTrajectory[];
  anchorMarkers: AnchorMarker[];
  anchorDistances: number[][][]; // [token][layer][anchor_idx]
  anchorLabels: string[];
  anchorExplainedVariance: number[];
  isAnchorLoading: boolean;
  anchorError: string | null;

  // Probe view data (M5)
  probeResults: ProbeResult[];
  probeAnchorMarkers: AnchorMarker[];
  probeReconstructionErrors: { n_anchors: number; error: number }[];
  probeExplainedVariance: number[];
  probeNLayers: number;
  isProbeLoading: boolean;
  probeError: string | null;
  probeSelectedCategories: string[];
  probeCustomConcepts: string[];

  // Steering playground data (representation engineering)
  steerPositive: string;
  steerNegative: string;
  steerLayer: number;
  steerStrength: number;
  steerOriginalTrajectories: TokenTrajectory[];
  steerSteeredTrajectories: TokenTrajectory[];
  steerGenerationOriginal: string;
  steerGenerationSteered: string;
  steerNLayers: number;
  steerExplainedVariance: number[];
  isSteerLoading: boolean;
  steerError: string | null;

  // Compare view data (M4)
  selectedModels: string[];
  compareData: CompareModelData[];
  comparePairwise: Record<string, PairMetrics>; // "a vs b" -> metrics
  compareExplainedVariance: number[];
  compareLayer: number; // 0-1 fraction (normalised depth across models)
  isCompareLoading: boolean;
  compareError: string | null;

  // UI state
  currentLayer: number;
  isPlaying: boolean;
  playSpeed: number;
  showAttention: boolean;
  attentionThreshold: number;
  activeHeads: Set<number>;
  projectionMethod: "pca" | "umap";
  inputText: string;
  isLoading: boolean;
  error: string | null;
  hideBOS: boolean;
  spread: number;

  // Actions — absolute
  setTrajectoryData: (data: {
    tokens: string[];
    trajectories: TokenTrajectory[];
    attention: AttentionEdge[];
    logitLens: LensPrediction[][][];
    nLayers: number;
    explainedVariance: number[];
    modelName: string;
  }) => void;
  setCurrentLayer: (layer: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaySpeed: (speed: number) => void;
  setShowAttention: (show: boolean) => void;
  setAttentionThreshold: (threshold: number) => void;
  toggleHead: (head: number) => void;
  setProjectionMethod: (method: "pca" | "umap") => void;
  setInputText: (text: string) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setHideBOS: (hide: boolean) => void;
  setSpread: (spread: number) => void;

  // Actions — probe
  setProbeData: (data: {
    probeResults: ProbeResult[];
    probeAnchorMarkers: AnchorMarker[];
    probeReconstructionErrors: { n_anchors: number; error: number }[];
    probeExplainedVariance: number[];
    probeNLayers: number;
  }) => void;
  setIsProbeLoading: (loading: boolean) => void;
  setProbeError: (error: string | null) => void;
  setProbeSelectedCategories: (cats: string[]) => void;
  setProbeCustomConcepts: (concepts: string[]) => void;

  // Actions — anchor
  setViewMode: (mode: "absolute" | "anchor" | "compare" | "probe" | "steer") => void;
  setAnchorInputs: (inputs: string[]) => void;
  setAnchorData: (data: {
    anchorTrajectories: TokenTrajectory[];
    anchorMarkers: AnchorMarker[];
    anchorDistances: number[][][];
    anchorLabels: string[];
    anchorExplainedVariance: number[];
  }) => void;
  setIsAnchorLoading: (loading: boolean) => void;
  setAnchorError: (error: string | null) => void;

  // Actions — steer
  setSteerInputs: (positive: string, negative: string) => void;
  setSteerLayer: (layer: number) => void;
  setSteerStrength: (strength: number) => void;
  setSteerData: (data: {
    steerOriginalTrajectories: TokenTrajectory[];
    steerSteeredTrajectories: TokenTrajectory[];
    steerGenerationOriginal: string;
    steerGenerationSteered: string;
    steerNLayers: number;
    steerExplainedVariance: number[];
  }) => void;
  setIsSteerLoading: (loading: boolean) => void;
  setSteerError: (error: string | null) => void;

  // Actions — compare
  setSelectedModels: (models: string[]) => void;
  setCompareData: (data: {
    compareData: CompareModelData[];
    comparePairwise: Record<string, PairMetrics>;
    compareExplainedVariance: number[];
  }) => void;
  setCompareLayer: (layer: number) => void;
  setIsCompareLoading: (loading: boolean) => void;
  setCompareError: (error: string | null) => void;
}

const DEFAULT_ANCHORS = ["self", "other", "world", "logic"];

export const useExplorerStore = create<ExplorerState>((set) => ({
  // Absolute data defaults
  tokens: [],
  trajectories: [],
  attention: [],
  logitLens: [],
  nLayers: 0,
  explainedVariance: [],
  modelName: "",

  // Anchor data defaults
  viewMode: "absolute",
  anchorInputs: DEFAULT_ANCHORS,
  anchorTrajectories: [],
  anchorMarkers: [],
  anchorDistances: [],
  anchorLabels: [],
  anchorExplainedVariance: [],
  isAnchorLoading: false,
  anchorError: null,

  // Probe data defaults
  probeResults: [],
  probeAnchorMarkers: [],
  probeReconstructionErrors: [],
  probeExplainedVariance: [],
  probeNLayers: 0,
  isProbeLoading: false,
  probeError: null,
  probeSelectedCategories: ["emotions", "abstractions", "mind"],
  probeCustomConcepts: [],

  // Steering defaults
  steerPositive: "love joy happiness",
  steerNegative: "hate fear anger",
  steerLayer: 3,
  steerStrength: 1.0,
  steerOriginalTrajectories: [],
  steerSteeredTrajectories: [],
  steerGenerationOriginal: "",
  steerGenerationSteered: "",
  steerNLayers: 0,
  steerExplainedVariance: [],
  isSteerLoading: false,
  steerError: null,

  // Compare data defaults
  selectedModels: ["pythia-70m", "gpt2"],
  compareData: [],
  comparePairwise: {},
  compareExplainedVariance: [],
  compareLayer: 0,
  isCompareLoading: false,
  compareError: null,

  // UI defaults
  currentLayer: 0,
  isPlaying: false,
  playSpeed: 1,
  showAttention: false,
  attentionThreshold: 0.15,
  activeHeads: new Set<number>(),
  projectionMethod: "pca",
  inputText: "I think therefore I am",
  isLoading: false,
  error: null,
  hideBOS: true,
  spread: 1.0,

  // Absolute actions
  setTrajectoryData: (data) =>
    set({
      tokens: data.tokens,
      trajectories: clampBosForDisplay(data.trajectories),
      attention: data.attention,
      logitLens: data.logitLens,
      nLayers: data.nLayers,
      explainedVariance: data.explainedVariance,
      modelName: data.modelName,
      currentLayer: data.nLayers,
      isPlaying: false,
      viewMode: "absolute",
    }),

  setCurrentLayer: (layer) => set({ currentLayer: layer }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),
  setPlaySpeed: (speed) => set({ playSpeed: speed }),
  setShowAttention: (show) => set({ showAttention: show }),
  setAttentionThreshold: (threshold) => set({ attentionThreshold: threshold }),
  toggleHead: (head) =>
    set((state) => {
      const next = new Set(state.activeHeads);
      if (next.has(head)) next.delete(head);
      else next.add(head);
      return { activeHeads: next };
    }),
  setProjectionMethod: (method) => set({ projectionMethod: method }),
  setInputText: (text) => set({ inputText: text }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  setHideBOS: (hide) => set({ hideBOS: hide }),
  setSpread: (spread) => set({ spread }),

  // Probe actions
  setProbeData: (data) =>
    set({
      probeResults: data.probeResults,
      probeAnchorMarkers: data.probeAnchorMarkers,
      probeReconstructionErrors: data.probeReconstructionErrors,
      probeExplainedVariance: data.probeExplainedVariance,
      probeNLayers: data.probeNLayers,
      viewMode: "probe",
      currentLayer: data.probeNLayers,
      isPlaying: false,
    }),
  setIsProbeLoading: (loading) => set({ isProbeLoading: loading }),
  setProbeError: (error) => set({ probeError: error }),
  setProbeSelectedCategories: (cats) => set({ probeSelectedCategories: cats }),
  setProbeCustomConcepts: (concepts) => set({ probeCustomConcepts: concepts }),

  // Anchor actions
  setViewMode: (mode) => set({ viewMode: mode }),
  setAnchorInputs: (inputs) => set({ anchorInputs: inputs }),
  setAnchorData: (data) =>
    set({
      anchorTrajectories: data.anchorTrajectories,
      anchorMarkers: data.anchorMarkers,
      anchorDistances: data.anchorDistances,
      anchorLabels: data.anchorLabels,
      anchorExplainedVariance: data.anchorExplainedVariance,
      viewMode: "anchor",
      // Open at the final layer like every other view (assembled meaning first).
      currentLayer: Math.max(0, (data.anchorTrajectories[0]?.positions.length ?? 1) - 1),
      isPlaying: false,
    }),
  setIsAnchorLoading: (loading) => set({ isAnchorLoading: loading }),
  setAnchorError: (error) => set({ anchorError: error }),

  // Steering actions
  setSteerInputs: (positive, negative) =>
    set({ steerPositive: positive, steerNegative: negative }),
  setSteerLayer: (layer) => set({ steerLayer: layer }),
  setSteerStrength: (strength) => set({ steerStrength: strength }),
  setSteerData: (data) =>
    set({
      steerOriginalTrajectories: clampBosForDisplay(data.steerOriginalTrajectories),
      steerSteeredTrajectories: clampBosForDisplay(data.steerSteeredTrajectories),
      steerGenerationOriginal: data.steerGenerationOriginal,
      steerGenerationSteered: data.steerGenerationSteered,
      steerNLayers: data.steerNLayers,
      steerExplainedVariance: data.steerExplainedVariance,
      viewMode: "steer",
      currentLayer: data.steerNLayers,
      isPlaying: false,
    }),
  setIsSteerLoading: (loading) => set({ isSteerLoading: loading }),
  setSteerError: (error) => set({ steerError: error }),

  // Compare actions
  setSelectedModels: (models) => set({ selectedModels: models }),
  setCompareData: (data) =>
    set({
      compareData: data.compareData,
      comparePairwise: data.comparePairwise,
      compareExplainedVariance: data.compareExplainedVariance,
      viewMode: "compare",
      compareLayer: 0,
      isPlaying: false,
    }),
  setCompareLayer: (layer) => set({ compareLayer: layer }),
  setIsCompareLoading: (loading) => set({ isCompareLoading: loading }),
  setCompareError: (error) => set({ compareError: error }),
}));
