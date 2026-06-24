# Lessons Learned

## Project-Specific
- BOS token (`<|endoftext|>`) dominates attention at all layers (~0.5–1.0 weight to itself).
  Hiding it is essential to reveal meaningful semantic attention edges. Default the toggle ON.
- Starting the view at the final layer (not layer 0) gives a better first impression — the user
  sees the assembled semantic structure, then can scrub backward to watch it form.
- PCA explained variance for Pythia-70M on short sentences: ~95%+ in 3 PCs — projection is
  very informative for this model/scale.

## Technical Notes
- Pythia-70M: 6 layers, 512 dim, 8 heads — small enough for CPU on M3
- TransformerLens caches all activations by default; use `names_filter`
  to avoid OOM on larger models
- PCA on residual streams: fit on ALL layer activations jointly so axes
  are comparable across layers (same coordinate system)
- React Three Fiber: use `useFrame` for animation, not `requestAnimationFrame`
- Trail fading: render individual Line segments per layer step with opacity
  proportional to position (dim at layer 0, bright at current). Each Pythia-70M
  run has ≤7 segments so this is cheap.
- Attention edge raycasting: drei's `Line` (uses LineSegments2 internally when
  lineWidth > 1) supports pointer events; onPointerEnter/Leave work for tooltips.
- Glow effect without postprocessing: additive-blended larger sphere behind the core
  sphere gives a convincing soft glow with zero extra dependencies.
- PlaybackController: sync `currentLayerRef` from store's `currentLayer` when not
  playing, so resuming play starts from wherever the scrubber is.

## BOS Contamination in Concept Embeddings (affects M3, M4, M5)
- `extract_concept_stream()` must skip token index 0 (BOS) before mean-pooling.
  BOS is an attention-sink token that dominates activations at all layers. Including it
  pulls all concept vectors toward the same direction → cosine distances between
  concepts become near-identical → PCA finds near-zero variance → everything collapses.
- Symptom: anchor markers all converge to one point; probe concepts all land on one point.
- Fix: `stream[:, 1:, :].mean(axis=1)` — pool only content tokens.
- Note: the in-memory concept cache (`_concept_cache` in probe.py) holds embeddings from
  the current server session. Restart the server after this fix to clear stale values.

## Milestone 3 — Anchor-Relative View
- PCA vs UMAP trade-off: PCA for global trajectory story (tokens diverging through
  layers), UMAP for local clustering story (which tokens are near each other). Document
  this clearly — users need to know when to switch.
- Anchor-relative computation: fit PCA on ALL token-layer descriptors jointly so axes
  are consistent across layers, same as absolute mode. This is the key design decision
  that makes layer scrubbing meaningful.
- Anchor self-distances: project anchors into the same space by using their pairwise
  cosine-distance vectors as input to the fitted PCA transform. This places anchor
  markers in a geometrically sensible position relative to the tokens.
- n_anchors < 3: guard with `min(n_components, n_anchors)` and pad projected output
  to 3D with zeros — prevents crash when user tries 2 anchors.
- Attention edges in anchor mode: disabled (viewMode check in AttentionEdges) since
  attention is computed on raw residual stream, not anchor-relative positions. Could
  add later if useful.
- Distance table: opacity-encode cells by (1 - distance) to make "close" anchors
  visually pop without adding color complexity.

## Milestone 7 — Song Mode
- Joint PCA is the non-negotiable design constraint: fit on ALL lines' activations before
  projecting any of them. Per-line normalization after that would break comparability —
  must normalize jointly across all projected outputs before returning.
- The `<audio>` element lives in SongControls (controls the playback UI), but the
  viewport doesn't need it — just reads `currentLineIndex` from the store.
  Keep the audio element co-located with playback controls, not in the viewport.
- LRC multi-timestamp lines: one line of text can appear at multiple times (`[00:05.00][00:35.00] chorus`).
  The parser correctly expands these into separate events.
- Auto-orbit via programmatic camera mutation in `useFrame` conflicts with OrbitControls
  if you try to set `camera.position` while OrbitControls is listening. Solution: use the
  `AutoOrbitController` pattern that only mutates position when autoOrbit=true, and let
  OrbitControls win otherwise. OrbitControls will re-sync on the next user drag.
- Ghost desaturation: compute HSL-style gray blend in RGB space (weighted luminance).
  Desaturating by 65% gives visually distinct ghosts without looking fully gray.
- Cumulative mode cool→warm gradient: interpolate from #4363d8 (blue) to #e6194b (red)
  by line index fraction. Gives the "shape of the song" a temporal color story.
- Individual mesh per ghost token is fine at this scale (~50 lines × 10 tokens = 500 meshes).
  For song with 200+ lines, switch to InstancedMesh with per-instance color attribute.
- Song end auto-triggers cumulative view via `onEnded` → `setShowCumulative(true)`.
  Camera keeps orbiting. This is the "hero moment" — the full shape of the song revealed.

## Milestone 5 — Probe Mode
- Concept cache keyed by (model_name, concept_text) → [n_layers+1, d_model]: single forward
  pass per concept, cached forever in the process. First run of 84 concepts is slow (~10-30s);
  subsequent calls hit cache instantly — worth communicating in the UI loading label.
- Uncertainty = min(cosine_distances to all anchors) at the final layer. High uncertainty means
  the concept is equidistant from all anchors — genuinely ambiguous or off-manifold. Low uncertainty
  means it's well-anchored near one concept.
- Reconstruction error curve (zero-pad partial descriptors → project → compare): measures info
  value of each additional anchor. Expect steep drop from K=2 to K=3, plateau after K=5-6.
  Display as a bar chart — bars get taller left-to-right as more anchors are used; height = error,
  so shorter bars to the right show improvement.
- ProbeConceptSphere hover uses Html from drei with pointer-events: none — the same pattern as
  AttentionEdges tooltip. Works reliably in R3F.
- 84 spheres renders fine at 60fps on M3 (no instanced mesh needed). If scaling to 500+ concepts,
  switch to InstancedMesh with per-instance color via vertex attribute.
- Category colors must match backend CATEGORY_COLORS exactly — both defined separately in
  probe.py and Viewport.tsx. If adding categories, update both.

## Milestone 4 — Cross-Model Comparison
- Joint PCA across ALL models' anchor descriptors is the key design decision: it puts
  different models in the same coordinate system without any explicit alignment step.
  Anchor-relative coordinates do the heavy lifting — joint PCA just handles the 3D
  projection consistently.
- Different tokenizations (GPT-2 vs Pythia) produce different n_tokens. CKA
  comparison truncates to min(n_tokens_a, n_tokens_b) — a limitation noted in code.
  For a research writeup, word-level alignment would be cleaner.
- Model loading is blocking (HookedTransformer.from_pretrained). For a multi-user
  deployment, this needs asyncio.run_in_executor or a worker queue. Fine for
  single-user local use.
- Depth normalisation (0→1 fraction) across models with different layer counts
  lets the compare scrubber have a coherent semantic: "how far through processing?"
  Works well as long as both models are autoregressive transformers.
- Two complementary metrics: CKA (rotation-invariant, literature-standard) and
  Procrustes (geometric, sensitive to arrangement). Show CKA prominently; Procrustes
  available in the API for deeper analysis.
- PlaybackController must be split: AbsolutePlaybackController (range 0→nLayers) and
  ComparePlaybackController (range 0→1). Rendering both simultaneously would fight
  each other — render conditionally based on viewMode.
