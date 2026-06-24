import { create } from "zustand";
import type { ParsedLrc } from "@/lib/lrc-parser";
import type { BatchLineResult } from "@/lib/api";

interface SongState {
  // Loaded content
  lrcData: ParsedLrc | null;
  audioUrl: string | null;
  audioFilename: string;
  batchResults: BatchLineResult[] | null;
  songNLayers: number;
  songExplainedVariance: number[];
  songModelName: string;

  // Playback state
  isPlayingAudio: boolean;
  audioCurrentTime: number;
  audioDuration: number;
  currentLineIndex: number;  // -1 = before first line

  // Visual settings
  ghostOpacity: number;     // base opacity for ghost lines
  ghostDecay: number;       // opacity multiplier per line ago (0.85 = 15% decay/line)
  showCumulative: boolean;  // false = decaying ghosts, true = all lines at once
  songLayer: number;        // which transformer layer to show (-1 = last)

  // Pre-processing
  songIsPreprocessing: boolean;

  // Actions
  setLrcData: (data: ParsedLrc) => void;
  setAudioUrl: (url: string, filename: string) => void;
  setBatchResults: (
    results: BatchLineResult[],
    nLayers: number,
    ev: number[],
    model: string
  ) => void;
  setIsPlayingAudio: (playing: boolean) => void;
  setAudioCurrentTime: (time: number) => void;
  setAudioDuration: (dur: number) => void;
  setCurrentLineIndex: (index: number) => void;
  setGhostOpacity: (opacity: number) => void;
  setGhostDecay: (decay: number) => void;
  setShowCumulative: (show: boolean) => void;
  setSongLayer: (layer: number) => void;
  setSongIsPreprocessing: (loading: boolean) => void;
  reset: () => void;
}

export const useSongStore = create<SongState>((set) => ({
  lrcData: null,
  audioUrl: null,
  audioFilename: "",
  batchResults: null,
  songNLayers: 0,
  songExplainedVariance: [],
  songModelName: "",

  isPlayingAudio: false,
  audioCurrentTime: 0,
  audioDuration: 0,
  currentLineIndex: -1,

  ghostOpacity: 0.3,
  ghostDecay: 0.82,
  showCumulative: false,
  songLayer: -1,

  songIsPreprocessing: false,

  setLrcData: (data) => set({ lrcData: data }),
  setAudioUrl: (url, filename) => set({ audioUrl: url, audioFilename: filename }),
  setBatchResults: (results, nLayers, ev, model) =>
    set({
      batchResults: results,
      songNLayers: nLayers,
      songExplainedVariance: ev,
      songModelName: model,
      currentLineIndex: -1,
    }),
  setIsPlayingAudio: (playing) => set({ isPlayingAudio: playing }),
  setAudioCurrentTime: (time) => set({ audioCurrentTime: time }),
  setAudioDuration: (dur) => set({ audioDuration: dur }),
  setCurrentLineIndex: (index) => set({ currentLineIndex: index }),
  setGhostOpacity: (opacity) => set({ ghostOpacity: opacity }),
  setGhostDecay: (decay) => set({ ghostDecay: decay }),
  setShowCumulative: (show) => set({ showCumulative: show }),
  setSongLayer: (layer) => set({ songLayer: layer }),
  setSongIsPreprocessing: (loading) => set({ songIsPreprocessing: loading }),
  reset: () =>
    set({
      lrcData: null,
      audioUrl: null,
      audioFilename: "",
      batchResults: null,
      songNLayers: 0,
      isPlayingAudio: false,
      audioCurrentTime: 0,
      audioDuration: 0,
      currentLineIndex: -1,
    }),
}));
