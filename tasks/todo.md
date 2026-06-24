# Semantic Geometry Explorer — Tasks

## Milestone 1: Residual Stream Trajectories (MVP)

### Setup
- [x] Project scaffold (repo structure, configs, READMEs)
- [x] Backend: FastAPI app skeleton with CORS
- [x] Backend: TransformerLens extraction module
- [x] Backend: PCA projection pipeline
- [x] Backend: `/analyze` endpoint (text → trajectories JSON)
- [x] Backend: proof-of-concept extraction script (standalone, testable)
- [x] Frontend: Next.js + React Three Fiber scaffold
- [x] Frontend: 3D viewport with token trajectory rendering
- [x] Frontend: text input → API call → render pipeline
- [x] Frontend: layer scrubber/slider with animation
- [x] Frontend: token labels + color legend

### Refinements (completed)
- [x] BOS token hide toggle (default ON) — filters BOS from trajectories, attention edges, token legend
- [x] Default view starts at final layer (not layer 0) — user sees assembled meaning first
- [x] Camera moved closer: [4, 3, 4] instead of [6, 4, 6]
- [x] Attention threshold default raised to 0.15 (reduces noise)
- [x] Attention edge color gradient: dark (#333) → white based on weight
- [x] Attention edge hover tooltip (from → to, weight) via drei Html
- [x] Trail lines fade in opacity: dim at layer 0, bright at current layer
- [x] Token sphere glow halo (additive blending)
- [x] Axis labels: PC1/PC2/PC3 with explained variance percentages
- [x] Playback resumes from current scrubber position (not always layer 0)
- [x] Spread multiplier control (0.5–6x) — scales token positions to expose inter-token distances

### Validation
- [ ] Run extraction on "I think therefore I am" — verify shapes make sense
- [ ] Confirm PCA axes are stable across similar inputs
- [ ] Test with 5+ diverse sentences, check for visual coherence
- [ ] Profile: backend response time < 2s for short sentences
- [ ] Deploy locally, end-to-end demo working

### Stretch
- [ ] UMAP toggle (alternative to PCA)
- [ ] Export trajectory data as JSON for notebooks
- [ ] Dark/light theme toggle

---

## Milestone 2: Attention Edges
- [x] Backend already returns `[n_layers, n_heads, n_tokens, n_tokens]` attention data
- [x] Frontend renders attention edges with weight-based color gradient and hover tooltips
- [x] Threshold slider, BOS filtering — M2 complete as part of M1 refinements

---

## Milestone 3: Anchor-Relative View
- [x] Backend: `extract_concept_stream` on ModelExtractor (mean-pool all layers)
- [x] Backend: `anchor.py` — cosine descriptor computation and joint PCA projection
- [x] Backend: `POST /anchor-analyze` endpoint
- [x] Frontend: anchor state in zustand store (viewMode, inputs, trajectories, markers, distances)
- [x] Frontend: anchor UI in Controls (presets, add/remove inputs, compute button)
- [x] Frontend: view mode toggle (Absolute ↔ Anchor-Relative)
- [x] Frontend: AnchorMarkers in Viewport (wireframe octahedra with labels)
- [x] Frontend: distance table in sidebar at current layer
- [x] Frontend: axis labels switch to Anc-PC1/2/3 in anchor mode
- [x] README: PCA vs UMAP explanation table

### Validation
- [ ] Test "I think therefore I am" with anchors ["self", "other", "logic", "doubt"]
- [ ] Verify "I" clusters near "self", "think" near "logic"
- [ ] Test that anchor markers move across layers (anchors evolve too)
- [ ] Confirm distance table updates as layer scrubber moves

---

## Milestone 4: Cross-Model Comparison
- [x] Backend: `alignment.py` — `linear_cka`, `procrustes_similarity`, `pairwise_metrics_across_depth`
- [x] Backend: model registry (`_model_registry`) — lazy-loads additional models on demand, caches after first load
- [x] Backend: `GET /models` — lists supported + currently-loaded models
- [x] Backend: `POST /compare` — joint PCA across all models' anchor descriptors, pairwise CKA + Procrustes at 7 normalised depth fractions
- [x] Frontend: compare state in zustand (selectedModels, compareData, comparePairwise, compareLayer 0-1)
- [x] Frontend: model selector checkboxes in Controls (pythia-70m, gpt2, pythia-160m)
- [x] Frontend: "Run Comparison" button with loading state note about first-run latency
- [x] Frontend: compare depth scrubber (0–100%, normalised across model depths)
- [x] Frontend: convergence panel — CKA at 0%/50%/100% per pair with colour-coded values and sparkline bar
- [x] Frontend: CompareScene in Viewport — per-model color (red/cyan/green), token labels, fading trails
- [x] Frontend: ComparePlaybackController — separate from AbsolutePlaybackController, correct 0-1 range
- [x] Frontend: view mode toggle extended to Absolute | Anchor | Compare
- [x] Frontend: axis labels show Cmp-PC1/2/3 in compare mode

### Validation
- [ ] Run pythia-70m vs gpt2 on "I think therefore I am" with ["self","logic","existence","doubt"]
- [ ] Confirm CKA increases from early → late layers (Platonic Representation Hypothesis prediction)
- [ ] Visually check: do token clusters converge across models at depth 100%?
- [ ] Test with pythia-160m as third model

---

## Milestone 5: Geometric Extrapolation / Probe Mode
- [x] Backend: `probe.py` — CONCEPT_VOCABULARY (7 cats × 12 = 84 concepts), CATEGORY_COLORS, `get_concept_stream` with in-memory cache, `compute_probe_positions` (joint PCA, uncertainty, reconstruction error curve)
- [x] Backend: `POST /probe` endpoint — `ProbeRequest`, `SingleProbeResult`, `ProbeResponse` Pydantic models; loads model on demand
- [x] Frontend: `ProbeResult` interface in store; extend `viewMode` to include "probe"
- [x] Frontend: probe state in zustand store — `probeResults`, `probeAnchorMarkers`, `probeReconstructionErrors`, `probeExplainedVariance`, `probeNLayers`, `isProbeLoading`, `probeError`, `probeSelectedCategories`, `probeCustomConcepts`
- [x] Frontend: `probeConceptsAPI()` in `api.ts` with `ProbeResponse` interface
- [x] Frontend: Controls probe section — category toggle chips (7 categories), custom concept input + add, removable chip list, "Embed N Concepts" button, reconstruction error bar display
- [x] Frontend: `ProbeConceptSphere` — color by category, uncertainty halo (radius/opacity proportional to uncertainty), hover tooltip with label + category + uncertainty
- [x] Frontend: `ProbeScene` — renders all concept spheres + anchor octahedra
- [x] Frontend: View mode toggle gains "Probe" button (active when `hasProbeData`)
- [x] Frontend: Layer scrubber handles probe mode (uses `probeNLayers`), playback controller uses correct max layer
- [x] Frontend: AxisLabels shows "Prb-PC" prefix in probe mode
- [x] Frontend: probe CSS — `.probe-categories`, `.probe-cat-btn`, `.probe-chip-list`, `.probe-chip`, `.probe-chip-remove`, `.probe-embed-btn`, `.recon-error-bar`, `.recon-bar-seg`

### Validation
- [ ] Run with anchors ["self","other","logic","emotion"] — verify emotions cluster near "emotion" anchor, abstractions near "logic"
- [ ] Add custom concept "war" — verify it lands near "conflict"/"enemy" cluster
- [ ] Check reconstruction error curve decreases monotonically with more anchors
- [ ] Verify layer scrubber animates probe positions (concepts should move slightly layer-to-layer)

## Milestone 6: Corpus-Scale Shape Comparison (TDA)
- [x] Backend: `topology.py` — `TopologyAnalyzer` with farthest-point sampling, ripser persistent homology (H0+H1), persim Wasserstein/bottleneck distances, `pairwise_distances()`
- [x] Backend: `extractor.py` — `extract_text_cloud()` with overlapping chunk windows, center-priority assignment for overlap regions
- [x] Backend: `POST /text-shape` endpoint — `TextShapeRequest`, `PersistenceFeatureModel`, `TextShapeResponse`
- [x] Backend: `POST /compare-shapes` endpoint — `CorpusCompareRequest`, `CorpusCompareResponse`, pairwise distance matrix
- [x] Frontend: `store/corpus.ts` — `CorpusState` with texts, results, visibility, diagram view, metric settings; built-in presets (Philosophy vs Recipe, Poetry vs Science, Emotion vs Logic)
- [x] Frontend: `lib/api.ts` — `TextShapeResponse`, `CorpusCompareResponse`, `analyzeTextShape()`, `compareShapes()`
- [x] Frontend: `CorpusControls.tsx` — preset buttons, textarea + file upload (.txt), labeled text list with remove, layer/max-points/metric sliders, analyze button, distance matrix table, persistence diagram (scatter + barcode SVG views)
- [x] Frontend: `CorpusViewport.tsx` — `InstancedMesh` point clouds per text, centroid labels, per-cloud visibility toggle
- [x] Frontend: `page.tsx` — top-level Explore | Corpus mode tab bar, conditional routing to Controls/Viewport or CorpusControls/CorpusViewport
- [x] Frontend: CSS — `.corpus-textarea`, `.corpus-text-row`, `.corpus-text-dot`, `.corpus-text-count`

### Validation
- [ ] Preset "Philosophy vs Recipe" — verify H1 loop count differs between texts
- [ ] Same text twice → distance ≈ 0
- [ ] Distinct texts → distance clearly > 0
- [ ] Persistence diagram scatter: all points above diagonal (death > birth)
- [ ] File upload: drop a .txt file, verify it appears in the list

## Milestone 7: Semantic Song Visualizer (Song Mode)
- [x] Backend: `projector.py` — add `fit_basis()` (fit PCA without projecting) and `project_with_basis()` (project without re-normalizing, for caller-controlled joint normalization)
- [x] Backend: `POST /analyze-batch` — joint PCA across ALL lyric lines, joint normalization, returns shared-coordinate-system trajectories per line
- [x] Frontend: `lib/lrc-parser.ts` — `parseLrc()` (handles `[mm:ss.xx]`, multi-timestamp lines, metadata tags), `findCurrentLineIndex()` for O(n) playback sync
- [x] Frontend: `lib/api.ts` — `BatchLineResult`, `AnalyzeBatchResponse`, `analyzeBatch()`
- [x] Frontend: `store/song.ts` — full song state: lrcData, audioUrl, batchResults, playback, ghost settings, layer, preprocessing flag
- [x] Frontend: `SongControls.tsx` — drag-and-drop LRC + audio zones, hidden `<audio>` element with timeupdate sync, pre-analyze button, play/pause + seek bar, current lyric preview, ghost opacity/decay sliders, Current/Cumulative toggle, layer slider
- [x] Frontend: `SongViewport.tsx` — `CurrentToken` (full sphere + glow + label + trail), `GhostLine` (desaturated small spheres, opacity = ghostOpacity × ghostDecay^linesAgo), `CumulativeCloud` (all lines, cool→warm gradient), `AutoOrbitController` (5°/s Y-axis, pauses on interaction, resumes after 3s), `LyricsOverlay` (absolute-positioned pill at bottom-center)
- [x] Frontend: `page.tsx` — Explore | Corpus | Song three-tab bar
- [x] Frontend: CSS — `.song-dropzone`, `.song-dropzone--loaded`, `.song-current-line`

### Validation
- [ ] Parse a real .lrc file — verify timestamp sort order and endTime computation
- [ ] Pre-analyze 5 lines — verify shared coordinate system (semantically similar words appear near each other across lines)
- [ ] Audio playback sync — correct line highlights at correct timestamp
- [ ] Ghosts appear and fade with configured decay
- [ ] Cumulative mode: all points visible, cool→warm gradient
- [ ] Song end → cumulative view activates automatically, camera continues orbiting

## Review
_(to be filled after milestone completion)_
