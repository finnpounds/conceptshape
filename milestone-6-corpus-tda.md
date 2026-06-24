# Milestone 6 — Corpus-Scale Shape Comparison (TDA)

## Instructions for Claude Code

1. Read this file fully before starting
2. Read `docs/project-scope.md` for full project context (Milestone 6 section)
3. Read `backend/app/topology.py` — stub already exists with interface defined
4. Execute the file checklist at the bottom in order
5. Run validation steps before marking complete

---

## Concept

A text is not a single point — it's thousands of tokens, each producing an
activation vector. That cloud of points IS the geometric shape of the text's
meaning. This milestone uses Topological Data Analysis (persistent homology)
to extract structural fingerprints from those point clouds and compare them
across texts. The question: does philosophy have a different shape than poetry?
Can you identify an author from geometry alone?

---

## Architecture

### Data Flow

```
Input text (book, essay, article — any length)
  → Chunk into segments (512 tokens max per forward pass)
  → Run each chunk through ModelExtractor
  → Collect all token activations at target layer
  → Union into single point cloud [n_total_tokens, d_model]
  → Subsample to max_points (persistence is O(n³))
  → Compute persistent homology (H0, H1, H2)
  → Return persistence diagram + point cloud (projected to 3D)

For comparison:
  → Repeat for each text
  → Compute pairwise distances on persistence diagrams
  → Return distance matrix + per-text diagrams
```

### Backend: Implement `topology.py`

The stub at `backend/app/topology.py` defines the interface. Implement
these methods:

```python
class TopologyAnalyzer:

    def compute_shape(self, activations: np.ndarray, label: str = "") -> TextShape:
        """
        1. Subsample if n_points > self.max_points (use farthest point sampling
           for better coverage than random sampling)
        2. Compute persistence diagram using ripser:
           diagrams = ripser(point_cloud, maxdim=self.max_dim)['dgms']
        3. Convert to structured format:
           - H0: connected components (birth=0, death=merge distance)
           - H1: loops (birth=loop forms, death=loop fills)
           - H2: voids (birth=void forms, death=void fills)
        4. Return TextShape with point_cloud and persistence_diagram
        """

    def compare(self, shape_a: TextShape, shape_b: TextShape, metric: str) -> float:
        """
        Use persim library:
        - wasserstein: persim.wasserstein(dgm_a, dgm_b)
        - bottleneck: persim.bottleneck(dgm_a, dgm_b)
        Compare H0 and H1 separately, return weighted sum or dict.
        """

    def pairwise_distances(self, shapes: list[TextShape], metric: str) -> np.ndarray:
        """
        NxN symmetric matrix. Use compare() for each pair.
        """
```

**Additional method to add:**

```python
    def farthest_point_sample(self, points: np.ndarray, n: int) -> np.ndarray:
        """
        Subsample n points using farthest point sampling.
        Better than random — preserves geometric coverage of the manifold.

        Algorithm:
        1. Pick a random starting point
        2. For each subsequent point, pick the one farthest from all
           already-selected points
        3. Repeat until n points selected

        Returns indices of selected points.
        """
```

**Chunked extraction — add to `extractor.py`:**

```python
    def extract_text_cloud(
        self,
        text: str,
        layer: int = -1,
        chunk_size: int = 512,
        stride: int = 256,
    ) -> np.ndarray:
        """
        Process a long text by chunking with overlap, extract activations
        at target layer, return union of all token activations.

        Args:
            text: full text (can be book-length)
            layer: which layer to extract from (-1 = last)
            chunk_size: max tokens per forward pass
            stride: step size between chunks (< chunk_size = overlap)

        Returns:
            np.ndarray of shape [n_total_tokens, d_model]

        Implementation:
        1. Tokenize full text
        2. Split token IDs into overlapping windows of chunk_size
        3. For each window, run forward pass, extract residual stream at layer
        4. For overlapping regions, keep only the activations from the window
           where each token is closest to center (avoids edge effects)
        5. Concatenate all activations
        """
```

### Backend: New Endpoints in `main.py`

**1. `POST /text-shape`**

```python
class TextShapeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500000)  # up to ~book length
    layer: int = Field(default=-1)  # -1 = last layer
    max_points: int = Field(default=2000, ge=100, le=10000)
    method: str = Field(default="pca", pattern="^(pca|umap)$")

class PersistenceFeature(BaseModel):
    dimension: int      # 0=component, 1=loop, 2=void
    birth: float
    death: float
    persistence: float  # death - birth

class TextShapeResponse(BaseModel):
    label: str
    n_tokens_total: int
    n_points_sampled: int
    point_cloud_3d: list[list[float]]  # [n_points, 3] projected
    persistence_diagram: list[PersistenceFeature]
    # Summary stats
    n_components: int     # H0 features
    n_loops: int          # H1 features
    n_voids: int          # H2 features
    explained_variance: list[float]
```

**2. `POST /compare-shapes`**

```python
class CompareRequest(BaseModel):
    texts: list[str] = Field(..., min_length=2, max_length=10)
    labels: list[str] = Field(default=[])  # human names for each text
    layer: int = Field(default=-1)
    max_points: int = Field(default=2000)
    metric: str = Field(default="wasserstein", pattern="^(wasserstein|bottleneck)$")

class CompareResponse(BaseModel):
    shapes: list[TextShapeResponse]        # per-text results
    distance_matrix: list[list[float]]     # NxN pairwise distances
    labels: list[str]
    metric: str
```

### Frontend: New Components

**1. Mode toggle — update `page.tsx`**

Add a top-level tab bar with three modes:
- **Explore** — current single-sentence mode (M1-M5)
- **Corpus** — text comparison mode (M6)
- **Song** — song visualizer mode (M7, separate milestone)

Only implement Explore and Corpus for this milestone.

**2. `CorpusControls.tsx`**

```
- Text input area: large textarea for pasting text, or file upload (.txt)
- "Add text" button — adds to a list of texts to compare
- Text list: shows all loaded texts with labels, remove button per text
- Layer selector: dropdown or slider for target layer
- Max points slider: 500-5000
- "Analyze" button → calls /text-shape or /compare-shapes
- Presets dropdown (stretch goal):
  - "Philosophy: Descartes vs Hume vs Kant" (bundled excerpts)
  - "Genre: Novel vs Poetry vs Science" (bundled excerpts)
```

**3. `CorpusViewport.tsx`**

Renders point clouds instead of token trajectories:

```
- Each text rendered as a point cloud in 3D (using R3F Points or InstancedMesh)
- Color-coded by text (text A = blue cloud, text B = red cloud, etc.)
- Density visualization: points colored by local density
  (hot = dense cluster, cool = sparse frontier)
- Opacity per-cloud controlled by sidebar toggles
- Camera: orbit controls, same as Explore mode

- Side panel or overlay: persistence diagram
  - Standard birth-death scatter plot
  - Points colored by dimension (H0=blue, H1=green, H2=red)
  - Toggle between individual texts or overlay mode
  - Barcode view toggle (horizontal bars instead of scatter)

- Distance matrix panel (when comparing multiple texts):
  - Small heatmap showing pairwise distances
  - Darker = more similar topology
```

**4. Point cloud rendering approach:**

For up to 2000 points, use `InstancedMesh` with sphere geometry:

```tsx
// Pseudocode for point cloud rendering
function PointCloud({ points, color, opacity }) {
  const meshRef = useRef();
  const tempObject = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    points.forEach((point, i) => {
      tempObject.position.set(point[0], point[1], point[2]);
      tempObject.scale.setScalar(0.03); // small spheres
      tempObject.updateMatrix();
      meshRef.current.setMatrixAt(i, tempObject.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [points]);

  return (
    <instancedMesh ref={meshRef} args={[null, null, points.length]}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial color={color} transparent opacity={opacity} />
    </instancedMesh>
  );
}
```

**5. Persistence diagram component:**

Use a 2D overlay (HTML/CSS positioned over or beside the 3D viewport):

```tsx
function PersistenceDiagram({ features, width, height }) {
  // SVG scatter plot
  // X axis = birth, Y axis = death
  // Diagonal line (birth = death) shown as reference
  // Points above diagonal = features with positive persistence
  // Color by dimension: H0, H1, H2
  // Tooltip on hover: "H1 loop: born at 0.34, died at 0.89, persistence 0.55"
}
```

### Frontend: State Additions (`store/explorer.ts`)

```typescript
// Add to existing store or create store/corpus.ts

interface CorpusState {
  mode: "explore" | "corpus" | "song";
  setMode: (mode: "explore" | "corpus" | "song") => void;

  // Corpus mode
  corpusTexts: { id: string; text: string; label: string }[];
  addCorpusText: (text: string, label: string) => void;
  removeCorpusText: (id: string) => void;

  corpusResults: {
    shapes: TextShapeResponse[];
    distanceMatrix: number[][];
    labels: string[];
  } | null;
  setCorpusResults: (results: any) => void;

  corpusLayer: number;
  setCorpusLayer: (layer: number) => void;
  corpusMaxPoints: number;
  setCorpusMaxPoints: (n: number) => void;
  corpusIsLoading: boolean;
}
```

### API Client Additions (`lib/api.ts`)

```typescript
export async function analyzeTextShape(
  text: string,
  layer: number = -1,
  maxPoints: number = 2000,
  method: string = "pca"
): Promise<TextShapeResponse> { ... }

export async function compareShapes(
  texts: string[],
  labels: string[],
  layer: number = -1,
  maxPoints: number = 2000,
  metric: string = "wasserstein"
): Promise<CompareResponse> { ... }
```

---

## Processing Limits & Performance

- **Pythia-70M on M3 CPU**: ~50 tokens/second inference
- **Book-length text** (~80,000 tokens): ~27 minutes to process all chunks
  - This is too slow for interactive use
  - Solution: process in background, show progress bar
  - Or: start with essays/chapters (~2,000-5,000 tokens, ~1-2 minutes)
- **Persistence computation** (ripser): O(n³) worst case
  - 2,000 points: ~2-5 seconds
  - 5,000 points: ~30-60 seconds
  - 10,000 points: minutes — avoid unless needed
- **Recommendation**: default max_points to 2,000 for interactive use.
  Offer "high fidelity" mode at 5,000 for final comparisons.

---

## Validation

Test with these inputs to verify correct behavior:

1. **Sanity check**: Feed the same text twice with different labels.
   Distance should be ~0. Persistence diagrams should be identical.
2. **Distinct texts**: Feed a paragraph of philosophy and a paragraph of
   a cooking recipe. Distance should be high. Point clouds should
   occupy different regions.
3. **Related texts**: Feed two paragraphs from the same philosophy book.
   Distance should be moderate — lower than philosophy vs cooking.
4. **Persistence diagram check**: Verify H0 features have birth=0
   (components exist from the start). Verify H1 features have
   birth > 0 and death > birth. No features below the diagonal.
5. **Point cloud rendering**: Verify clouds are visually distinct
   in 3D. Rotation should reveal structure, not a uniform blob.

---

## File Checklist

- [ ] `backend/app/topology.py` — implement all methods (stub exists)
- [ ] `backend/app/extractor.py` — add `extract_text_cloud()` method
- [ ] `backend/app/main.py` — add `/text-shape` and `/compare-shapes` endpoints
- [ ] `frontend/src/store/explorer.ts` — add corpus mode state (or new store)
- [ ] `frontend/src/components/CorpusControls.tsx` — text input, file upload, presets
- [ ] `frontend/src/components/CorpusViewport.tsx` — point cloud + persistence diagram
- [ ] `frontend/src/lib/api.ts` — add `analyzeTextShape()` and `compareShapes()`
- [ ] `frontend/src/app/page.tsx` — add mode toggle tabs
- [ ] `docs/project-scope.md` — verify M6 section is present and current
- [ ] `tasks/todo.md` — add M6 checklist
- [ ] Run validation tests listed above
