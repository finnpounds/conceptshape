import { create } from "zustand";

export interface PersistenceFeature {
  dimension: number;   // 0=component, 1=loop, 2=void
  birth: number;
  death: number;
  persistence: number;
}

export interface TextShapeResult {
  label: string;
  nTokensTotal: number;
  nPointsSampled: number;
  pointCloud3d: number[][];   // [n_points, 3]
  persistenceDiagram: PersistenceFeature[];
  nComponents: number;
  nLoops: number;
  explainedVariance: number[];
}

export interface CorpusText {
  id: string;
  text: string;
  label: string;
}

interface CorpusState {
  // Loaded texts
  corpusTexts: CorpusText[];
  addCorpusText: (text: string, label: string) => void;
  removeCorpusText: (id: string) => void;
  updateLabel: (id: string, label: string) => void;

  // Analysis settings
  corpusLayer: number;
  setCorpusLayer: (layer: number) => void;
  corpusMaxPoints: number;
  setCorpusMaxPoints: (n: number) => void;
  corpusMetric: "wasserstein" | "bottleneck";
  setCorpusMetric: (metric: "wasserstein" | "bottleneck") => void;

  // Results
  corpusResults: {
    shapes: TextShapeResult[];
    distanceMatrix: number[][];
    labels: string[];
    metric: string;
  } | null;
  setCorpusResults: (results: CorpusState["corpusResults"]) => void;
  clearCorpusResults: () => void;

  // Per-shape visibility toggle
  visibleShapes: Set<number>;
  toggleShapeVisibility: (index: number) => void;

  // UI
  isCorpusLoading: boolean;
  setIsCorpusLoading: (loading: boolean) => void;
  corpusError: string | null;
  setCorpusError: (error: string | null) => void;

  // Active diagram: which shape's persistence diagram to show
  activeDiagramIndex: number;
  setActiveDiagramIndex: (i: number) => void;

  // Diagram view: scatter or barcode
  diagramView: "scatter" | "barcode";
  setDiagramView: (view: "scatter" | "barcode") => void;
}

let _nextId = 0;
const uid = () => String(++_nextId);

// Built-in preset texts for quick demos
export const CORPUS_PRESETS: { label: string; texts: { label: string; text: string }[] }[] = [
  {
    label: "Philosophy vs Recipe",
    texts: [
      {
        label: "Descartes",
        text: "I think, therefore I am. The mind is a substance that thinks. The body is a substance that is extended in space. These two substances are distinct and separate. Doubt is the beginning of wisdom. I will doubt everything that can be doubted, until I arrive at something certain.",
      },
      {
        label: "Recipe",
        text: "Preheat the oven to 375 degrees Fahrenheit. Mix the flour, sugar, and butter in a bowl until crumbly. Add the eggs and vanilla extract. Stir until smooth. Pour the batter into a greased pan. Bake for 30 minutes until golden brown. Let cool before serving.",
      },
    ],
  },
  {
    label: "Poetry vs Science",
    texts: [
      {
        label: "Poem",
        text: "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May, and summer's lease hath all too short a date. Sometime too hot the eye of heaven shines, and often is his gold complexion dimmed.",
      },
      {
        label: "Science",
        text: "The force between two masses is proportional to the product of their masses and inversely proportional to the square of the distance between them. This fundamental law governs the motion of planets, the tides of the ocean, and the fall of objects on Earth.",
      },
    ],
  },
  {
    label: "Emotion vs Logic",
    texts: [
      {
        label: "Emotional",
        text: "She felt a wave of grief wash over her, so deep and vast she could not breathe. The love she had carried for years collapsed into sorrow. Every memory was a wound. She wept until there were no more tears, and then sat in silence, holding the emptiness.",
      },
      {
        label: "Logical",
        text: "If all premises are true and the argument is valid, then the conclusion must be true. A deductive argument is sound if and only if it is valid and all its premises are actually true. Inductive reasoning moves from specific observations to general conclusions.",
      },
    ],
  },
];

export const useCorpusStore = create<CorpusState>((set, get) => ({
  corpusTexts: [],
  addCorpusText: (text, label) =>
    set((s) => ({
      corpusTexts: [...s.corpusTexts, { id: uid(), text, label }],
    })),
  removeCorpusText: (id) =>
    set((s) => ({ corpusTexts: s.corpusTexts.filter((t) => t.id !== id) })),
  updateLabel: (id, label) =>
    set((s) => ({
      corpusTexts: s.corpusTexts.map((t) => (t.id === id ? { ...t, label } : t)),
    })),

  corpusLayer: -1,
  setCorpusLayer: (layer) => set({ corpusLayer: layer }),
  corpusMaxPoints: 2000,
  setCorpusMaxPoints: (n) => set({ corpusMaxPoints: n }),
  corpusMetric: "wasserstein",
  setCorpusMetric: (metric) => set({ corpusMetric: metric }),

  corpusResults: null,
  setCorpusResults: (results) => set({
    corpusResults: results,
    visibleShapes: new Set(results?.shapes.map((_, i) => i) ?? []),
    activeDiagramIndex: 0,
  }),
  clearCorpusResults: () => set({ corpusResults: null }),

  visibleShapes: new Set(),
  toggleShapeVisibility: (index) =>
    set((s) => {
      const next = new Set(s.visibleShapes);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return { visibleShapes: next };
    }),

  isCorpusLoading: false,
  setIsCorpusLoading: (loading) => set({ isCorpusLoading: loading }),
  corpusError: null,
  setCorpusError: (error) => set({ corpusError: error }),

  activeDiagramIndex: 0,
  setActiveDiagramIndex: (i) => set({ activeDiagramIndex: i }),

  diagramView: "scatter",
  setDiagramView: (view) => set({ diagramView: view }),
}));
