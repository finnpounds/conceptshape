# Semantic Geometry Explorer
## Project Scope & Architecture

### Vision
An interactive 3D visualization tool that reveals how meaning takes shape inside
language models — showing the geometric trajectories of concepts as they evolve
through transformer layers, and exploring whether that geometry is universal
across models and predictive of unseen beliefs.

### Research Questions
1. **Trajectory visualization**: What does the "path to meaning" look like as
   tokens traverse transformer layers? How do token representations reorganize
   from surface-level similarity to semantic structure?
2. **Anchor-relative universality**: If we describe concepts by their distances
   to a fixed set of semantic anchors, do different models converge on similar
   geometries? (Visual test of the Platonic Representation Hypothesis)
3. **Geometric extrapolation**: Given a partial map of concept-anchor distances,
   can we predict where unobserved concepts fall — effectively reading "beliefs"
   from geometric structure?

### Key References
- Huh et al. (2024) — *The Platonic Representation Hypothesis*: convergence of
  representations across models/modalities toward shared structure
- Moschella et al. (2022) — *Relative Representations*: model-agnostic concept
  descriptions via distances to anchor points
- Zou et al. (2023) — *Representation Engineering*: linear directions in
  activation space encode high-level concepts; readable and steerable
- Nanda et al. — *TransformerLens*: library for hooking into transformer
  internals (residual stream, attention, MLP outputs)
- Linear Representation Hypothesis — concepts as directions in activation space,
  enabling vector arithmetic and geometric reasoning

---

## Milestone Breakdown

### Milestone 1 — Residual Stream Trajectories (MVP)
**Goal**: Type a sentence, watch each token's representation evolve through
layers in an interactive 3D scene.

**Backend (Python)**
- Model: Pythia-70M via TransformerLens (small, open, well-documented)
- For a given input string:
  - Tokenize and run forward pass with hooks on every layer's residual stream
  - Extract per-token activation vectors at each layer (shape: [n_layers, n_tokens, d_model])
  - Apply dimensionality reduction (PCA fit on all activations jointly, project to 3D)
  - Return JSON: `{ tokens: [...], layers: int, trajectories: [token][layer][x,y,z] }`
- Serve via FastAPI endpoint: `POST /analyze` accepts `{ text: string }`
- Option: pre-compute PCA basis on a corpus for stable axes across inputs

**Frontend (Next.js + React Three Fiber)**
- Text input field → sends to backend → receives trajectory data
- 3D scene:
  - Each token is a sphere, color-coded
  - Layer progression shown as connected path (tube or line) per token
  - Scrubber/slider to animate through layers (layer 0 → layer N)
  - At each layer, token spheres move to their projected position
  - Labels float near each token sphere
- Camera controls (orbit, zoom, pan)
- Layer slider with play/pause animation
- Token legend showing color mapping

**Deliverable**: Working local app — type "I think therefore I am", see five
tokens trace paths through 3D space as meaning assembles across layers.

**Key decisions**:
- PCA vs UMAP: PCA preserves global linear structure (better for showing
  directions), UMAP preserves local neighborhoods (better for clusters).
  Start with PCA, add UMAP toggle later.
- Axes: label PC1/PC2/PC3 with interpretable descriptions if possible
  (e.g., "PC1 correlates with concreteness")

---

### Milestone 2 — Attention Edges
**Goal**: Overlay attention patterns as directed edges between token nodes,
revealing how the model routes information.

**Backend additions**
- Extract attention weights per layer per head: [n_layers, n_heads, n_tokens, n_tokens]
- Include in response JSON: `attention: [layer][head][from][to]` (float weights)

**Frontend additions**
- At the currently selected layer, draw directed edges (lines/arrows) between
  token spheres, with opacity proportional to attention weight
- Color edges by attention head
- Controls:
  - Toggle individual heads on/off
  - Threshold slider (hide edges below weight X)
  - Head selector (show one head at a time, or all)
- Tooltip on edge hover: "Head 3: 'therefore' → 'think' (weight 0.72)"

**Deliverable**: See both WHERE tokens are (position) and HOW they relate
(attention edges) at each layer. "Therefore" attending strongly to "think"
becomes a visible bright edge.

---

### Milestone 3 — Anchor-Relative View
**Goal**: Re-render the geometry in a model-agnostic coordinate system defined
by distances to user-chosen anchor concepts.

**Backend additions**
- New endpoint: `POST /anchor-analyze`
  - Accepts: `{ text: string, anchors: string[] }`
  - For each anchor string, compute its mean representation at each layer
  - For each token at each layer, compute cosine distances to all anchors
  - Return anchor-relative coordinates: `[token][layer][anchor_distances]`
  - Also project anchor-distance vectors to 3D via PCA/MDS

**Frontend additions**
- Anchor definition UI: text inputs for 3-8 anchor concepts
  - Presets: "concrete/abstract", "positive/negative/neutral",
    "self/other/world", "logic/emotion/perception"
- Toggle between absolute (Milestone 1) and anchor-relative views
- In anchor-relative view:
  - Axes labeled by anchor concepts (or principal components of anchor distances)
  - Anchor points rendered as fixed reference markers
  - Token trajectories show movement relative to anchors across layers
- Side panel: distance table showing each token's distance to each anchor
  at the current layer

**Deliverable**: "I think therefore I am" viewed relative to anchors
["self", "existence", "logic", "doubt"]. Watch "I" cluster near "self",
"think" near "logic", and see how "am" moves toward "existence" in later
layers.

---

### Milestone 4 — Cross-Model Comparison
**Goal**: Run the same input through multiple models, show trajectories in
anchor-relative space, test whether they converge.

**Backend additions**
- Support multiple models: Pythia-70M, GPT-2 Small, Pythia-160M
  (same architecture family for clean comparison, then cross-architecture)
- For each model, compute anchor-relative representations
- Alignment: use Procrustes analysis or CKA (Centered Kernel Alignment)
  to quantitatively measure geometric similarity between models
- Return comparison metrics alongside trajectory data

**Frontend additions**
- Model selector: checkboxes for which models to display
- Side-by-side or overlaid 3D views
- Color-code trajectories by model (same token, different models)
- Convergence metrics panel: CKA score, Procrustes distance per layer
- Highlight: do models converge more in later layers? (test prediction
  of Platonic Representation Hypothesis)

**Deliverable**: Visual evidence of whether "I think therefore I am" takes
on similar geometric structure across different models when viewed through
anchor-relative coordinates. If trajectories converge, that's a visual
proof of representational universality.

---

### Milestone 5 — Geometric Extrapolation (Research Frontier)
**Goal**: Given observed anchor-concept geometry, predict where unobserved
concepts fall — reading "beliefs" from structure.

**Backend additions**
- Concept inventory: embed a vocabulary of ~100-500 concepts spanning
  semantic space (emotions, objects, abstractions, relations)
- For a given model + layer, compute full pairwise distance matrix
- Matrix completion: given a subset of known distances (the anchors),
  predict remaining distances using:
  - Linear interpolation in embedding space
  - Low-rank matrix factorization
  - Kernel-based regression on the distance matrix
- Evaluate: how well do predicted positions match actual positions?
  Report reconstruction error as a function of number of anchors.

**Frontend additions**
- "Probe mode": user picks N anchor concepts, system shows where it
  predicts other concepts would fall
- Predicted concepts shown as semi-transparent spheres with confidence
  halos (larger halo = less certain)
- Reveal button: show actual positions, color-code by prediction error
- Interactive: user adds more anchors, watch predictions improve

**Deliverable**: Choose 10 anchor concepts, see the system predict positions
of 50 others. Measure whether geometric structure is regular enough to
extrapolate from. If yes, you've demonstrated that partial concept-geometry
is sufficient to infer broader "belief structure."

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Frontend (Next.js + React Three Fiber)             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Text Input   │  │ 3D Viewport  │  │ Controls   │ │
│  │ + Anchors    │  │ (R3F/Three)  │  │ Panel      │ │
│  └──────┬───────┘  └──────▲───────┘  └────────────┘ │
│         │                 │                          │
│         ▼                 │                          │
│  ┌──────────────────────────────────────┐            │
│  │  State Manager (zustand or context)  │            │
│  │  - trajectories, attention, anchors  │            │
│  │  - current layer, active heads       │            │
│  │  - view mode (absolute/relative)     │            │
│  └──────┬───────────────────────────────┘            │
└─────────┼───────────────────────────────────────────┘
          │ HTTP / WebSocket
          ▼
┌─────────────────────────────────────────────────────┐
│  Backend (FastAPI + Python)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ TransformerL │  │ Dimensionali │  │ Anchor     │ │
│  │ ens Hooks    │  │ ty Reduction │  │ Relative   │ │
│  │ (residual,   │  │ (PCA, UMAP,  │  │ Transform  │ │
│  │  attention)  │  │  MDS)        │  │            │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│  ┌──────────────────────────────────────┐            │
│  │ Model Registry                       │            │
│  │ - Pythia-70M, GPT-2, etc.           │            │
│  │ - Cached activations                 │            │
│  └──────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

## Tech Stack
- **Frontend**: Next.js 14+, React Three Fiber, drei, zustand, Tailwind
- **Backend**: FastAPI, TransformerLens, PyTorch, scikit-learn (PCA/UMAP)
- **Models**: Pythia-70M (primary), GPT-2 Small, Pythia-160M (comparison)
- **Compute**: Google Colab for initial extraction experiments,
  local M3 MacBook for small models (Pythia-70M runs on CPU)
- **Deployment**: Vercel (frontend), Railway or Modal (backend)

## Development Phases
| Phase | Milestone | Effort | Dependencies |
|-------|-----------|--------|-------------|
| 1     | Residual stream trajectories | 2-3 weeks | TransformerLens setup, R3F scene |
| 2     | Attention edges | 1-2 weeks | Phase 1 backend |
| 3     | Anchor-relative view | 2-3 weeks | Phase 1 complete |
| 4     | Cross-model comparison | 2-3 weeks | Phase 3 complete |
| 5     | Geometric extrapolation | 3-4 weeks | Phase 3 complete |

## Open Questions
- [ ] Does PCA produce stable enough axes across different inputs, or do we
      need a fixed PCA basis from a reference corpus?
- [ ] How to handle tokenizer differences across models in Milestone 4?
      (Align at word level, not token level?)
- [ ] For Milestone 5, what's the minimum number of anchor concepts needed
      for useful extrapolation? (Empirical question — build and measure)
- [ ] Can we extract "belief directions" à la Representation Engineering and
      show them as axes in the visualization?
- [ ] Publication venue: NeurIPS workshop? Distill-style interactive essay?
      VIS conference?
