import { API_BASE } from "./config";

export { API_BASE };

export interface AnalyzeResponse {
  tokens: string[];
  n_layers: number;
  trajectories: { token: string; positions: number[][] }[];
  attention: { layer: number; head: number; weights: number[][] }[];
  explained_variance: number[];
  projection_method: string;
  model_name: string;
}

export interface AnchorAnalyzeResponse {
  tokens: string[];
  n_layers: number;
  model_name: string;
  anchors: string[];
  trajectories: { token: string; positions: number[][] }[];
  anchor_markers: { label: string; positions: number[][] }[];
  distances: number[][][]; // [token][layer][anchor]
  explained_variance: number[];
}

export interface CompareResponse {
  models: {
    model_name: string;
    tokens: string[];
    n_layers: number;
    trajectories: { token: string; positions: number[][] }[];
  }[];
  pairwise: Record<string, {
    fractions: number[];
    cka: number[];
    procrustes: number[];
  }>;
  anchors: string[];
  explained_variance: number[];
}

export interface ProbeResponse {
  probes: {
    label: string;
    category: string;
    positions: number[][];
    uncertainty: number;
  }[];
  anchor_markers: { label: string; positions: number[][] }[];
  reconstruction_errors: { n_anchors: number; error: number }[];
  explained_variance: number[];
  n_layers: number;
}

export interface ModelsInfo {
  supported: string[];
  loaded: string[];
}

export interface ModelInfo {
  model_name: string;
  n_layers: number;
  d_model: number;
  n_heads: number;
}

export async function analyzeText(
  text: string,
  method: "pca" | "umap" = "pca"
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, method }),
  });
  if (!res.ok) throw new Error(`Analysis failed: ${await res.text()}`);
  return res.json();
}

export async function analyzeAnchors(
  text: string,
  anchors: string[]
): Promise<AnchorAnalyzeResponse> {
  const res = await fetch(`${API_BASE}/anchor-analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, anchors }),
  });
  if (!res.ok) throw new Error(`Anchor analysis failed: ${await res.text()}`);
  return res.json();
}

export async function runComparison(
  text: string,
  models: string[],
  anchors: string[]
): Promise<CompareResponse> {
  const res = await fetch(`${API_BASE}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, models, anchors }),
  });
  if (!res.ok) throw new Error(`Comparison failed: ${await res.text()}`);
  return res.json();
}

export async function probeConceptsAPI(
  anchors: string[],
  probes: string[],
  model = "pythia-70m"
): Promise<ProbeResponse> {
  const res = await fetch(`${API_BASE}/probe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ anchors, probes, model }),
  });
  if (!res.ok) throw new Error(`Probe failed: ${await res.text()}`);
  return res.json();
}

export interface BatchLineResult {
  line_index: number;
  text: string;
  tokens: string[];
  trajectories: { token: string; positions: number[][] }[];
}

export interface AnalyzeBatchResponse {
  results: BatchLineResult[];
  n_layers: number;
  explained_variance: number[];
  model_name: string;
}

export async function analyzeBatch(
  lines: string[],
  method: "pca" | "umap" = "pca"
): Promise<AnalyzeBatchResponse> {
  const res = await fetch(`${API_BASE}/analyze-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lines, method }),
  });
  if (!res.ok) throw new Error(`Batch analysis failed: ${await res.text()}`);
  return res.json();
}

export interface PersistenceFeatureAPI {
  dimension: number;
  birth: number;
  death: number;
  persistence: number;
}

export interface TextShapeResponse {
  label: string;
  n_tokens_total: number;
  n_points_sampled: number;
  point_cloud_3d: number[][];
  persistence_diagram: PersistenceFeatureAPI[];
  n_components: number;
  n_loops: number;
  explained_variance: number[];
}

export interface CorpusCompareResponse {
  shapes: TextShapeResponse[];
  distance_matrix: number[][];
  labels: string[];
  metric: string;
}

export async function analyzeTextShape(
  text: string,
  label = "",
  layer = -1,
  maxPoints = 2000
): Promise<TextShapeResponse> {
  const res = await fetch(`${API_BASE}/text-shape`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, label, layer, max_points: maxPoints }),
  });
  if (!res.ok) throw new Error(`Shape analysis failed: ${await res.text()}`);
  return res.json();
}

export async function compareShapes(
  texts: string[],
  labels: string[],
  layer = -1,
  maxPoints = 2000,
  metric = "wasserstein"
): Promise<CorpusCompareResponse> {
  const res = await fetch(`${API_BASE}/compare-shapes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts, labels, layer, max_points: maxPoints, metric }),
  });
  if (!res.ok) throw new Error(`Shape comparison failed: ${await res.text()}`);
  return res.json();
}

export async function getModelsInfo(): Promise<ModelsInfo> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to get models info");
  return res.json();
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model-info`);
  if (!res.ok) throw new Error("Failed to get model info");
  return res.json();
}
