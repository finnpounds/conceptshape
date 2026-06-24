# Milestone 7 — Semantic Song Visualizer ("Song Mode")

## Instructions for Claude Code

1. Read this file fully before starting
2. Read `docs/project-scope.md` for full project context
3. If M6 has been implemented, the mode toggle (Explore | Corpus | Song)
   already exists — add Song as the third tab. If not, create the mode
   toggle with Explore and Song only.
4. Execute the file checklist at the bottom in order
5. Run validation steps before marking complete

---

## Concept

An iTunes-visualizer-style mode where lyrics animate through the 3D semantic
geometry in real time as a song plays. Instead of mapping audio frequencies to
visuals, this maps *semantic structure* to visuals — you see meaning take shape
as each line is sung. Ghost trails accumulate over the course of the song, and
when it ends you're looking at the shape of the song's meaning.

---

## Architecture

### Data Flow

```
LRC file (timestamped lyrics) + audio file
  → Parse LRC: extract lines + timestamps
  → Pre-process: send ALL lines to POST /analyze-batch
    → Backend fits single shared PCA basis across all lines
    → Returns all trajectory data in one response
  → Frontend stores pre-computed trajectories per line
  → Audio playback starts
  → requestAnimationFrame loop checks currentTime vs timestamps
  → Current line: render bright token spheres + trails
  → Previous lines: render as fading ghost particles
  → Song ends: freeze on cumulative point cloud (shape of the song)
```

### Backend Additions

**New endpoint: `POST /analyze-batch`**

This is the critical architectural piece. Each line MUST be projected using
a shared PCA basis, otherwise positions aren't comparable across lines.

```python
class AnalyzeBatchRequest(BaseModel):
    lines: list[str] = Field(..., min_length=1, max_length=200)
    method: str = Field(default="pca", pattern="^(pca|umap)$")

class BatchLineResult(BaseModel):
    line_index: int
    text: str
    tokens: list[str]
    trajectories: list[TokenTrajectory]  # reuse existing type

class AnalyzeBatchResponse(BaseModel):
    results: list[BatchLineResult]
    n_layers: int
    explained_variance: list[float]
    model_name: str
```

**Implementation in `main.py`:**

```python
@app.post("/analyze-batch")
async def analyze_batch(req: AnalyzeBatchRequest):
    # 1. Extract activations for ALL lines
    all_extractions = []
    for line in req.lines:
        raw = _extractor.extract(line)
        all_extractions.append(raw)

    # 2. Collect ALL residual streams into one array for joint PCA
    all_residuals = []
    for raw in all_extractions:
        all_residuals.append(raw["residual_stream"])

    # 3. Flatten all activations, fit PCA once
    all_flat = np.concatenate([
        r.reshape(-1, r.shape[-1]) for r in all_residuals
    ], axis=0)

    projector = Projector(method=req.method)
    projector.fit_basis(all_flat)

    # 4. Project each line's activations using shared basis
    results = []
    for i, raw in enumerate(all_extractions):
        positions = projector.project_with_basis(raw["residual_stream"])
        results.append(BatchLineResult(
            line_index=i,
            text=req.lines[i],
            tokens=raw["tokens"],
            trajectories=[...],
        ))

    return AnalyzeBatchResponse(results=results, ...)
```

**Add to `projector.py`:**

```python
def fit_basis(self, all_activations: np.ndarray) -> None:
    """
    Fit PCA on combined activation matrix WITHOUT projecting.
    Call once on union of all data, then use project_with_basis()
    for each individual item.

    Args:
        all_activations: [n_total_points, d_model]
    """
    mean = all_activations.mean(axis=0)
    centered = all_activations - mean
    self._pca = PCA(n_components=self.n_components)
    self._pca.fit(centered)
    self._pca_mean = mean

def project_with_basis(self, residual_stream: np.ndarray) -> np.ndarray:
    """
    Project activations using previously fit PCA basis.

    Args:
        residual_stream: [n_layers+1, n_tokens, d_model]

    Returns:
        positions: [n_layers+1, n_tokens, 3] normalized to [-1, 1]
    """
    n_layer_steps, n_tokens, d_model = residual_stream.shape
    flat = residual_stream.reshape(-1, d_model)
    centered = flat - self._pca_mean
    projected = self._pca.transform(centered)
    positions = projected.reshape(n_layer_steps, n_tokens, self.n_components)

    max_abs = np.abs(positions).max(axis=(0, 1), keepdims=True)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
    positions = positions / max_abs
    return positions
```

### Frontend: New Components

**1. `LrcParser.ts` — utility module**

```typescript
export interface LrcLine {
  startTime: number;  // seconds
  endTime: number;    // seconds (= next line's startTime, or Infinity)
  text: string;
}

export interface LrcMetadata {
  title?: string;
  artist?: string;
  album?: string;
}

export interface ParsedLrc {
  metadata: LrcMetadata;
  lines: LrcLine[];
}

export function parseLrc(content: string): ParsedLrc {
  // Parse format: [mm:ss.xx] lyric text
  // Handle metadata tags: [ti:Title], [ar:Artist], [al:Album]
  // Sort by time, compute endTime from next line's startTime
  // Skip empty lines and instrumental markers
}
```

**2. `SongControls.tsx`**

```
Layout (top to bottom):
┌──────────────────────────┐
│ MODE: [Explore] [Song]   │  ← tab bar
├──────────────────────────┤
│ LYRICS FILE              │
│ [Drop .lrc file here]    │  ← drag-and-drop zone
│ song-title.lrc ✓         │  ← loaded file indicator
├──────────────────────────┤
│ AUDIO FILE               │
│ [Drop audio file here]   │  ← mp3, wav, ogg, m4a
│ song-title.mp3 ✓         │
├──────────────────────────┤
│ STATUS                   │
│ Pre-analyzing lyrics...  │  ← progress during /analyze-batch
│ ████████░░ 34/50 lines   │
│ Ready ✓                  │
├──────────────────────────┤
│ ▶  ━━━━━●━━━━━  2:34     │  ← playback controls + seek bar
│ ♪ "current lyric line"   │  ← current line preview
├──────────────────────────┤
│ GHOST TRAILS             │
│ Opacity ━━●━━━  0.3      │
│ Decay   ━━━●━━  0.85     │
├──────────────────────────┤
│ VIEW                     │
│ [Current] [Cumulative]   │  ← decaying ghosts vs all visible
├──────────────────────────┤
│ LAYER  5 / 6             │
│ ━━━━━━━━━━━━●            │  ← fixed during playback
└──────────────────────────┘
```

Audio playback: use HTML5 `<audio>` element with ref. Sync via
timeupdate event + requestAnimationFrame for smooth tracking.

**3. `SongViewport.tsx`**

Key differences from Explore mode:

**Current line rendering:**
- Same as Explore: colored token spheres with labels + trails
- Emissive glow (emissiveIntensity: 0.5)
- Show at target layer only (no layer scrubbing during playback)

**Ghost line rendering:**
- InstancedMesh for performance (small spheres, no labels)
- Opacity per ghost line = `ghostOpacity * (ghostDecay ^ linesAgo)`
- Color: desaturated version of token colors (HSL: -60% sat, -20% light)
- Sphere radius: 0.06 (half of current line's 0.12)
- No trails for ghosts — just final-layer positions

**Cumulative cloud mode:**
- Toggle: show ALL lines at full opacity (no decay)
- "Shape of the song" view
- Color by line index: gradient from cool (#4363d8) to warm (#e6194b)
  across song timeline

**Camera behavior:**
- During playback: slow auto-orbit (Y axis, ~5°/second)
- On user mouse interaction: pause orbit
- Resume orbit after 3s of no input
- On song end: continue orbiting cumulative cloud

**Lyrics overlay:**
- Current line at bottom-center of viewport
- Font: system-ui, 18px, white, `rgba(0,0,0,0.5)` background pill
- 0.3s fade in/out crossfade between lines

**Line transitions:**
- Current line: 0.3s ease-out fade
- Next line: 0.3s ease-in
- Slight overlap (crossfade)

### Frontend: State Additions

Add to `store/explorer.ts` or create `store/song.ts`:

```typescript
interface SongState {
  // Data
  lrcData: ParsedLrc | null;
  audioUrl: string | null;
  batchResults: BatchLineResult[] | null;

  // Playback
  isPlayingAudio: boolean;
  audioCurrentTime: number;
  audioDuration: number;
  currentLineIndex: number;

  // Ghost settings
  ghostOpacity: number;       // default 0.3
  ghostDecay: number;         // default 0.85
  showCumulative: boolean;    // false = decaying, true = all visible

  // Target layer (fixed during playback)
  songLayer: number;          // default: last layer

  // Pre-processing
  songIsPreprocessing: boolean;
  songPreprocessProgress: number;  // 0-1

  // Actions
  setLrcData: (data: ParsedLrc) => void;
  setAudioUrl: (url: string) => void;
  setBatchResults: (results: BatchLineResult[]) => void;
  setIsPlayingAudio: (playing: boolean) => void;
  setAudioCurrentTime: (time: number) => void;
  setCurrentLineIndex: (index: number) => void;
  setGhostOpacity: (opacity: number) => void;
  setGhostDecay: (decay: number) => void;
  setShowCumulative: (show: boolean) => void;
  setSongLayer: (layer: number) => void;
}
```

### API Client (`lib/api.ts`)

```typescript
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
```

---

## Visual Design

Song mode should feel cinematic — more music visualizer than research tool.

- **Background**: `#050508` (deeper than explore mode's `#0a0a0f`)
- **Current line spheres**: full brightness + emissive glow
- **Ghost spheres**: desaturated, small, no labels
- **Lyrics text**: white, semi-transparent, bottom-center, sans-serif
- **Camera**: slow orbit = meditative, matches music tempo
- **End state**: cumulative cloud slowly rotating = the hero moment

---

## Performance

- **Pre-processing**: ~50 lines × Pythia-70M on M3 CPU → ~25 seconds
  - Show progress bar during this phase
- **Rendering**: ~500 ghost spheres as InstancedMesh → trivial for Three.js
- **Audio sync**: requestAnimationFrame → ~16ms precision (good enough
  for line-level sync where lines are 2-5 seconds apart)

---

## Validation

1. **LRC parsing**: Parse a real .lrc file, verify timestamps and sort order
2. **Batch analysis**: Send 5 test lines, verify shared coordinate system
   (positions should be spatially coherent across lines)
3. **Playback sync**: Play audio, verify correct line at correct timestamp
4. **Ghost rendering**: Verify ghosts appear, fade with configured decay
5. **Cumulative mode**: Toggle on, verify all points visible
6. **End state**: Song finishes → cloud persists, camera orbits

---

## File Checklist

- [ ] `backend/app/main.py` — add `POST /analyze-batch` endpoint
- [ ] `backend/app/projector.py` — add `fit_basis()` and `project_with_basis()`
- [ ] `frontend/src/lib/lrc-parser.ts` — LRC file parser
- [ ] `frontend/src/lib/api.ts` — add `analyzeBatch()` function
- [ ] `frontend/src/store/explorer.ts` — add song mode state (or `store/song.ts`)
- [ ] `frontend/src/components/SongControls.tsx` — file upload, playback, ghosts
- [ ] `frontend/src/components/SongViewport.tsx` — ghosts, lyrics overlay, orbit
- [ ] `frontend/src/app/page.tsx` — add Song to mode toggle
- [ ] `docs/project-scope.md` — add M7, update dev phases table
- [ ] `tasks/todo.md` — add M7 checklist
- [ ] `README.md` — add M7 to milestones list
- [ ] Run validation tests listed above
