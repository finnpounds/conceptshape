# Claude Code Task: Semantic Geometry Explorer — Milestone 1 Frontend Bring-Up

## Context
This is a visualization tool that shows how token representations evolve through
transformer layers in 3D space. The backend is working — it extracts residual
stream activations and attention patterns from Pythia-70M via TransformerLens,
projects to 3D via PCA, and serves via FastAPI.

Proof-of-concept output confirms the pipeline works:
- 6 tokens, 7 layer steps (embed + 6 layers), 512 dims → PCA to 3D
- 95.9% variance explained in 3 PCs
- Attention patterns show expected structure (BOS sink + semantic edges)

## Project Structure
```
semantic-geometry-explorer/
  backend/
    app/
      main.py          # FastAPI — POST /analyze returns trajectories + attention
      extractor.py     # TransformerLens hooks (residual stream + attention)
      projector.py     # PCA/UMAP to 3D
    scripts/
      poc_extract.py   # Standalone test script (already validated)
  frontend/
    src/
      app/page.tsx     # Main page (sidebar + viewport layout)
      app/layout.tsx
      app/globals.css  # Dark theme, monospace, scientific aesthetic
      components/
        Viewport.tsx   # React Three Fiber 3D scene (token spheres, trails, attention edges)
        Controls.tsx   # Text input, layer slider, playback, attention toggles
      store/explorer.ts  # Zustand state
      lib/api.ts       # Backend API client
  tasks/
    todo.md            # Checklist — track progress here
    lessons.md         # Update after any corrections
  docs/
    project-scope.md   # Full milestone roadmap
```

## Your Task

### 1. Get the frontend running
- `cd frontend && npm install && npm run dev`
- Fix any TypeScript or import issues that come up
- Verify it renders the sidebar + 3D viewport (will be empty until backend connects)

### 2. Start the backend and test end-to-end
- `cd backend && pip install -r requirements.txt` (venv should already exist)
- `uvicorn app.main:app --reload --port 8000`
- Type "I think therefore I am" in the frontend, confirm trajectory data renders

### 3. Integrate these refinements based on initial data analysis

**BOS token handling:**
- Add a "Hide BOS" toggle to Controls.tsx (default: ON — hide the BOS token)
- When active, filter out token index 0 (`<|endoftext|>`) from both trajectory
  rendering and attention edges
- BOS dominates attention (every token attends to it at ~0.5-1.0 weight) which
  drowns out the meaningful semantic edges. Hiding it reveals the real structure.

**Initial view state:**
- Default `currentLayer` to the last layer (nLayers) instead of 0, so the user
  sees the final semantic arrangement first, then can scrub backward to watch
  it assemble
- Consider starting the camera position slightly closer: [4, 3, 4] instead of
  [6, 4, 6]

**Attention edge improvements:**
- Default `attentionThreshold` to 0.15 instead of 0.1 (reduces noise)
- Add edge labels on hover showing: "token_from → token_to (weight: 0.XX)"
  Use drei's Html component for hover tooltips
- Color attention edges with a gradient from dim (#333) to bright (#fff) based
  on weight, instead of uniform white

**Visual polish:**
- Add a subtle glow/bloom effect on token spheres (drei's Bloom or custom shader)
- Trail lines should fade in opacity from current layer (bright) to layer 0 (dim)
- Add axis labels at the ends of the grid showing PC1/PC2/PC3 with explained
  variance percentages (e.g., "PC1 (56.3%)")

### 4. Validation
- Test with these sentences and verify visually coherent results:
  - "I think therefore I am"
  - "The cat sat on the mat"
  - "Love is patient love is kind"
  - "To be or not to be that is the question"
- Confirm layer scrubber animation is smooth
- Confirm attention edges appear/disappear correctly when toggling
- Confirm BOS toggle works

### 5. Update tracking
- Mark completed items in tasks/todo.md
- If you hit any issues or make architectural decisions, log them in tasks/lessons.md

## Key Technical Notes
- Pythia-70M: 6 layers, 512 dim, 8 attention heads — runs on CPU (M3 MacBook)
- PCA is fit jointly across all layers so axes are consistent (same coordinate
  system at every layer step)
- The frontend uses React Three Fiber (not raw Three.js) — use R3F patterns
  (useFrame, declarative components)
- drei library is available for helpers (Text, Billboard, Line, Html, OrbitControls)
- State is in zustand, not React context — use `useExplorerStore` selectors
- The API response shape matches the zustand store types exactly
