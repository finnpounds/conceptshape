// Gallery: precomputed showcase examples that render with NO backend.
//
// Each example is a raw API response captured by backend/scripts/build_gallery.py.
// applyExample() replays it through the SAME zustand setters the live app uses,
// so the public site is fully interactive (orbit / scrub layers / toggle
// attention) even when no Python backend is running.

import { useExplorerStore } from "@/store/explorer";
import { useCorpusStore, type TextShapeResult } from "@/store/corpus";
import { useTrainingStore } from "@/store/training";
import { useUIStore } from "@/store/ui";
import type {
  AnalyzeResponse,
  AnchorAnalyzeResponse,
  CompareResponse,
  ProbeResponse,
  CorpusCompareResponse,
} from "./api";

export type GalleryMode = "explore" | "corpus" | "song" | "training";
export type GallerySubMode =
  | "absolute"
  | "anchor"
  | "compare"
  | "probe"
  | "corpus"
  | "song"
  | "training";

export interface GalleryEntry {
  id: string;
  mode: GalleryMode;
  subMode: GallerySubMode;
  title: string;
  hook: string;
  detail: string;
  params: Record<string, unknown>;
  file: string;
}

export interface GalleryIndex {
  models: { supported: string[]; loaded: string[] };
  examples: GalleryEntry[];
}

let _indexCache: GalleryIndex | null = null;
const _dataCache = new Map<string, unknown>();

export async function fetchGalleryIndex(): Promise<GalleryIndex> {
  if (_indexCache) return _indexCache;
  const res = await fetch("/gallery/index.json");
  if (!res.ok) throw new Error("Failed to load gallery index");
  _indexCache = (await res.json()) as GalleryIndex;
  return _indexCache;
}

async function fetchEntryData<T>(entry: GalleryEntry): Promise<T> {
  if (_dataCache.has(entry.file)) return _dataCache.get(entry.file) as T;
  const res = await fetch(`/gallery/${entry.file}`);
  if (!res.ok) throw new Error(`Failed to load gallery example: ${entry.file}`);
  const data = (await res.json()) as T;
  _dataCache.set(entry.file, data);
  return data;
}

function mapShape(s: CorpusCompareResponse["shapes"][number]): TextShapeResult {
  return {
    label: s.label,
    nTokensTotal: s.n_tokens_total,
    nPointsSampled: s.n_points_sampled,
    pointCloud3d: s.point_cloud_3d,
    persistenceDiagram: s.persistence_diagram.map((f) => ({
      dimension: f.dimension,
      birth: f.birth,
      death: f.death,
      persistence: f.persistence,
    })),
    nComponents: s.n_components,
    nLoops: s.n_loops,
    explainedVariance: s.explained_variance,
  };
}

/**
 * Load an example's precomputed data and apply it to the relevant stores,
 * switching the app into the right top-level mode + view.
 */
export async function applyExample(entry: GalleryEntry): Promise<void> {
  const explorer = useExplorerStore.getState();
  const corpus = useCorpusStore.getState();
  const ui = useUIStore.getState();

  if (entry.mode === "explore") {
    ui.setAppMode("explore");

    if (entry.subMode === "absolute") {
      const d = await fetchEntryData<AnalyzeResponse>(entry);
      if (typeof entry.params.text === "string")
        explorer.setInputText(entry.params.text);
      if (entry.params.method === "umap" || entry.params.method === "pca")
        explorer.setProjectionMethod(entry.params.method);
      explorer.setTrajectoryData({
        tokens: d.tokens,
        trajectories: d.trajectories,
        attention: d.attention,
        logitLens: d.logit_lens ?? [],
        nLayers: d.n_layers,
        explainedVariance: d.explained_variance,
        modelName: d.model_name,
      });
    } else if (entry.subMode === "anchor") {
      const d = await fetchEntryData<AnchorAnalyzeResponse>(entry);
      if (typeof entry.params.text === "string")
        explorer.setInputText(entry.params.text);
      if (Array.isArray(entry.params.anchors))
        explorer.setAnchorInputs(entry.params.anchors as string[]);
      // The token legend + distance table read the shared `tokens` field.
      useExplorerStore.setState({ tokens: d.tokens, modelName: d.model_name });
      explorer.setAnchorData({
        anchorTrajectories: d.trajectories,
        anchorMarkers: d.anchor_markers,
        anchorDistances: d.distances,
        anchorLabels: d.anchors,
        anchorExplainedVariance: d.explained_variance,
      });
    } else if (entry.subMode === "compare") {
      const d = await fetchEntryData<CompareResponse>(entry);
      if (typeof entry.params.text === "string")
        explorer.setInputText(entry.params.text);
      if (Array.isArray(entry.params.models))
        explorer.setSelectedModels(entry.params.models as string[]);
      if (Array.isArray(entry.params.anchors))
        explorer.setAnchorInputs(entry.params.anchors as string[]);
      explorer.setCompareData({
        compareData: d.models.map((m) => ({
          modelName: m.model_name,
          tokens: m.tokens,
          nLayers: m.n_layers,
          trajectories: m.trajectories,
        })),
        comparePairwise: d.pairwise,
        compareExplainedVariance: d.explained_variance,
      });
    } else if (entry.subMode === "probe") {
      const d = await fetchEntryData<ProbeResponse>(entry);
      if (Array.isArray(entry.params.anchors))
        explorer.setAnchorInputs(entry.params.anchors as string[]);
      explorer.setProbeData({
        probeResults: d.probes,
        probeAnchorMarkers: d.anchor_markers,
        probeReconstructionErrors: d.reconstruction_errors,
        probeExplainedVariance: d.explained_variance,
        probeNLayers: d.n_layers,
      });
    }
  } else if (entry.mode === "corpus") {
    ui.setAppMode("corpus");
    const d = await fetchEntryData<CorpusCompareResponse>(entry);
    corpus.setCorpusResults({
      shapes: d.shapes.map(mapShape),
      distanceMatrix: d.distance_matrix,
      labels: d.labels,
      metric: d.metric,
    });
  } else if (entry.mode === "training") {
    ui.setAppMode("training");
    const training = useTrainingStore.getState();
    await training.load();
    training.replay(); // restart the time-lapse from step 0 for the tour
  } else if (entry.mode === "song") {
    ui.setAppMode("song");
    // Song mode is interactive-with-audio and is served via the live backend.
    // (No static song example is bundled.)
  }
}
