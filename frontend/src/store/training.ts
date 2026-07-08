import { create } from "zustand";

// Data shape produced by backend/scripts/build_training_lapse.py.
// Positions are anchor-relative concept coordinates, jointly PCA'd across
// ALL training checkpoints so movement over training time is meaningful.

export interface TrainingConcept {
  label: string;
  category: string;
  positions: number[][]; // [n_steps][3]
}

export interface TrainingData {
  model: string;
  steps: number[];        // training step numbers (log-spaced)
  tokens_seen: number[];  // cumulative training tokens at each step
  anchors: string[];
  concepts: TrainingConcept[];
  anchor_markers: { label: string; positions: number[][] }[];
  cka_to_final: number[]; // geometry similarity to the final checkpoint
  explained_variance: number[];
}

interface TrainingState {
  data: TrainingData | null;
  isLoading: boolean;
  error: string | null;

  // Scrub position in CHECKPOINT-INDEX space (float, 0 .. n_steps-1).
  // Checkpoints are log-spaced in training steps, so index-space scrubbing
  // gives visually even pacing across the interesting part of training.
  scrub: number;
  isPlaying: boolean;
  playSpeed: number;
  showTrails: boolean;

  load: () => Promise<void>;
  replay: () => void;
  setScrub: (v: number) => void;
  setIsPlaying: (b: boolean) => void;
  setPlaySpeed: (v: number) => void;
  setShowTrails: (b: boolean) => void;
}

let _loading: Promise<void> | null = null;

export const useTrainingStore = create<TrainingState>((set, get) => ({
  data: null,
  isLoading: false,
  error: null,

  scrub: 0,
  isPlaying: true,
  playSpeed: 1,
  showTrails: true,

  load: () => {
    if (get().data) return Promise.resolve();
    if (_loading) return _loading;
    set({ isLoading: true, error: null });
    _loading = fetch("/gallery/training.json")
      .then((r) => {
        if (!r.ok) throw new Error("Training time-lapse data not found");
        return r.json();
      })
      .then((data: TrainingData) =>
        set({ data, isLoading: false, scrub: 0, isPlaying: true })
      )
      .catch((e) => {
        set({ error: e instanceof Error ? e.message : "Load failed", isLoading: false });
        _loading = null;
      });
    return _loading;
  },

  replay: () => set({ scrub: 0, isPlaying: true }),
  setScrub: (v) => set({ scrub: v }),
  setIsPlaying: (b) => set({ isPlaying: b }),
  setPlaySpeed: (v) => set({ playSpeed: v }),
  setShowTrails: (b) => set({ showTrails: b }),
}));
