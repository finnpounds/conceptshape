import { create } from "zustand";

export type AppMode = "explore" | "corpus" | "song";

// Backend reachability for the live "type your own" path.
//   unknown   — not yet checked
//   checking  — a health ping is in flight (possibly a cold start)
//   online    — backend responded ok
//   offline   — no backend reachable (gallery-only experience)
export type BackendStatus = "unknown" | "checking" | "online" | "offline";

interface UIState {
  appMode: AppMode;
  setAppMode: (m: AppMode) => void;

  // First-run hero / landing overlay
  showHero: boolean;
  setShowHero: (b: boolean) => void;

  // Guided tour — null when not running, else the current step index
  tourIndex: number | null;
  startTour: () => void;
  endTour: () => void;
  setTourIndex: (i: number) => void;

  // About / how-it-works modal
  showAbout: boolean;
  setShowAbout: (b: boolean) => void;

  // Live-backend status
  backendStatus: BackendStatus;
  setBackendStatus: (s: BackendStatus) => void;
}

export const useUIStore = create<UIState>((set) => ({
  appMode: "explore",
  setAppMode: (m) => set({ appMode: m }),

  showHero: true,
  setShowHero: (b) => set({ showHero: b }),

  tourIndex: null,
  startTour: () => set({ tourIndex: 0, showHero: false, showAbout: false }),
  endTour: () => set({ tourIndex: null }),
  setTourIndex: (i) => set({ tourIndex: i }),

  showAbout: false,
  setShowAbout: (b) => set({ showAbout: b }),

  backendStatus: "unknown",
  setBackendStatus: (s) => set({ backendStatus: s }),
}));
