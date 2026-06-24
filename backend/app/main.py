"""
main.py — FastAPI backend for Semantic Geometry Explorer.

Endpoints:
    POST /analyze   — text → 3D trajectories + attention patterns
    GET  /model-info — model metadata (layers, dims, etc.)
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

from app.extractor import ModelExtractor
from app.projector import Projector
from app.anchor import (
    compute_anchor_descriptors,
    compute_anchor_self_descriptors,
    project_to_3d,
)
from app.alignment import pairwise_metrics_across_depth
from app.probe import compute_probe_positions, get_concept_category
from app.topology import TopologyAnalyzer


# --- Global state ---
_extractor: ModelExtractor | None = None
_projector: Projector | None = None

# Model registry for M4 — keyed by TransformerLens model name.
# The default model is added on startup; additional models load on demand.
_model_registry: dict[str, ModelExtractor] = {}

SUPPORTED_MODELS = ["pythia-70m", "gpt2", "pythia-160m"]


def _get_or_load_model(name: str) -> ModelExtractor:
    """Return a cached extractor, loading it if needed. Blocking."""
    if name not in _model_registry:
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{name}'. Choose from: {SUPPORTED_MODELS}")
        _model_registry[name] = ModelExtractor(model_name=name)
    return _model_registry[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default model on startup."""
    global _extractor, _projector
    model_name = os.environ.get("SGE_MODEL", "pythia-70m")
    _extractor = ModelExtractor(model_name=model_name)
    _model_registry[model_name] = _extractor
    _projector = Projector(method="pca")
    yield
    _extractor = None
    _projector = None
    _model_registry.clear()


app = FastAPI(
    title="Semantic Geometry Explorer",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS. Allow local dev plus any *.vercel.app deployment, and any explicit
# origins supplied via SGE_ALLOWED_ORIGINS (comma-separated) for a custom domain.
_default_origins = ["http://localhost:3000", "http://localhost:3001"]
_env_origins = [
    o.strip()
    for o in os.environ.get("SGE_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _env_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Models ---

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    method: str = Field(default="pca", pattern="^(pca|umap)$")


class TokenTrajectory(BaseModel):
    token: str
    positions: list[list[float]]  # [n_layers+1][3] — 3D position at each layer


class AttentionEdge(BaseModel):
    """Attention weights for one layer, one head."""
    layer: int
    head: int
    weights: list[list[float]]  # [n_tokens][n_tokens]


class AnalyzeResponse(BaseModel):
    tokens: list[str]
    n_layers: int
    trajectories: list[TokenTrajectory]
    attention: list[AttentionEdge]
    explained_variance: list[float]
    projection_method: str
    model_name: str


class ModelInfo(BaseModel):
    model_name: str
    n_layers: int
    d_model: int
    n_heads: int


# --- Anchor-relative models (Milestone 3) ---

class AnchorAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    anchors: list[str] = Field(..., min_length=2, max_length=8)


class AnchorMarker(BaseModel):
    label: str
    positions: list[list[float]]  # [n_layers+1][3]


class AnchorAnalyzeResponse(BaseModel):
    tokens: list[str]
    n_layers: int
    model_name: str
    anchors: list[str]
    trajectories: list[TokenTrajectory]   # positions in anchor-distance space
    anchor_markers: list[AnchorMarker]    # anchor reference points in same space
    distances: list[list[list[float]]]    # [n_tokens][n_layers+1][n_anchors]
    explained_variance: list[float]


# --- Endpoints ---

@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    if _extractor is None:
        raise HTTPException(503, "Model not loaded")
    return ModelInfo(
        model_name=_extractor.model_name,
        n_layers=_extractor.n_layers,
        d_model=_extractor.d_model,
        n_heads=_extractor.n_heads,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if _extractor is None or _projector is None:
        raise HTTPException(503, "Model not loaded")

    # Extract activations
    raw = _extractor.extract(req.text)

    # Project to 3D
    _projector.method = req.method
    projection = _projector.project_trajectories(raw["residual_stream"])

    positions = projection["positions"]  # [n_layers+1, n_tokens, 3]
    n_layer_steps, n_tokens, _ = positions.shape

    # Build token trajectories
    trajectories = []
    for t in range(n_tokens):
        trajectories.append(TokenTrajectory(
            token=raw["tokens"][t],
            positions=positions[:, t, :].tolist(),
        ))

    # Build attention edges
    attention_data = raw["attention"]  # [n_layers, n_heads, n_tokens, n_tokens]
    attention_edges = []
    n_layers, n_heads = attention_data.shape[0], attention_data.shape[1]
    for layer in range(n_layers):
        for head in range(n_heads):
            attention_edges.append(AttentionEdge(
                layer=layer,
                head=head,
                weights=attention_data[layer, head].tolist(),
            ))

    return AnalyzeResponse(
        tokens=raw["tokens"],
        n_layers=n_layers,
        trajectories=trajectories,
        attention=attention_edges,
        explained_variance=projection["explained_variance"],
        projection_method=projection["method"],
        model_name=_extractor.model_name,
    )


@app.post("/anchor-analyze", response_model=AnchorAnalyzeResponse)
async def anchor_analyze(req: AnchorAnalyzeRequest):
    """
    Compute anchor-relative trajectories (Milestone 3).
    Each token is described by its cosine distances to anchor concept embeddings
    at each layer, then projected to 3D via PCA for visualization.
    """
    if _extractor is None:
        raise HTTPException(503, "Model not loaded")

    # Extract residual stream for the input text
    raw = _extractor.extract(req.text)
    residual_stream = raw["residual_stream"]  # [n_layers+1, n_tokens, d_model]

    # Extract mean-pooled anchor streams (one forward pass per anchor)
    anchor_streams = np.stack([
        _extractor.extract_concept_stream(anchor)
        for anchor in req.anchors
    ], axis=0)  # [n_anchors, n_layers+1, d_model]

    # Compute anchor-relative descriptors
    descriptors = compute_anchor_descriptors(residual_stream, anchor_streams)
    # shape: [n_layers+1, n_tokens, n_anchors]

    anchor_self_dists = compute_anchor_self_descriptors(anchor_streams)
    # shape: [n_layers+1, n_anchors, n_anchors]

    # Project to 3D (jointly across all layers)
    projection = project_to_3d(descriptors, anchor_self_dists)
    positions = projection["positions"]           # [n_layers+1, n_tokens, 3]
    anchor_positions = projection["anchor_positions"]  # [n_layers+1, n_anchors, 3]

    n_layer_steps, n_tokens, _ = positions.shape
    n_layers = n_layer_steps - 1

    trajectories = [
        TokenTrajectory(token=raw["tokens"][t], positions=positions[:, t, :].tolist())
        for t in range(n_tokens)
    ]

    anchor_markers = [
        AnchorMarker(label=label, positions=anchor_positions[:, a, :].tolist())
        for a, label in enumerate(req.anchors)
    ]

    # distances shape: [n_tokens][n_layers+1][n_anchors]
    distances = descriptors.transpose(1, 0, 2).tolist()

    return AnchorAnalyzeResponse(
        tokens=raw["tokens"],
        n_layers=n_layers,
        model_name=_extractor.model_name,
        anchors=req.anchors,
        trajectories=trajectories,
        anchor_markers=anchor_markers,
        distances=distances,
        explained_variance=projection["explained_variance"],
    )


@app.get("/models")
async def list_models():
    """List supported models and which are currently loaded (cached)."""
    return {
        "supported": SUPPORTED_MODELS,
        "loaded": list(_model_registry.keys()),
    }


# --- Compare models (Milestone 4) ---

class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    models: list[str] = Field(..., min_length=2, max_length=3)
    anchors: list[str] = Field(..., min_length=2, max_length=8)


class ModelComparison(BaseModel):
    model_name: str
    tokens: list[str]
    n_layers: int
    trajectories: list[TokenTrajectory]  # anchor-relative, jointly projected


class PairMetrics(BaseModel):
    fractions: list[float]    # [0.0, ..., 1.0] — normalised depth
    cka: list[float]          # linear CKA at each fraction
    procrustes: list[float]   # Procrustes similarity at each fraction


class CompareResponse(BaseModel):
    models: list[ModelComparison]
    pairwise: dict[str, PairMetrics]  # "model_a vs model_b" -> metrics
    anchors: list[str]
    explained_variance: list[float]


@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """
    Run the same text through multiple models, compute anchor-relative
    trajectories in a jointly projected 3D space, and return pairwise
    CKA / Procrustes convergence metrics across normalised depth.

    Note: first call for a new model is slow (~30s to download + load).
    Subsequent calls use the cached extractor.
    """
    # Validate model names before any loading
    for name in req.models:
        if name not in SUPPORTED_MODELS:
            raise HTTPException(400, f"Unsupported model '{name}'. Choose from: {SUPPORTED_MODELS}")

    # Load / retrieve extractors (blocking — acceptable for single-user tool)
    try:
        extractors = {name: _get_or_load_model(name) for name in req.models}
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {e}")

    # For each model: extract residual stream + anchor descriptor vectors
    model_raw: dict[str, dict] = {}
    for name, ext in extractors.items():
        raw = ext.extract(req.text)
        anchor_streams = np.stack([
            ext.extract_concept_stream(anchor) for anchor in req.anchors
        ], axis=0)  # [n_anchors, n_layers+1, d_model]

        descriptors = compute_anchor_descriptors(raw["residual_stream"], anchor_streams)
        # [n_layers+1, n_tokens, n_anchors]

        model_raw[name] = {
            "tokens": raw["tokens"],
            "n_layers": ext.n_layers,
            "descriptors": descriptors,
        }

    # Joint PCA across ALL models' anchor-descriptor data so axes are shared
    from sklearn.decomposition import PCA as _PCA
    all_flat = np.concatenate([
        d["descriptors"].reshape(-1, d["descriptors"].shape[-1])
        for d in model_raw.values()
    ], axis=0)

    n_anchors = all_flat.shape[1]
    n_comp = max(1, min(3, n_anchors, all_flat.shape[0] - 1))
    joint_pca = _PCA(n_components=n_comp)
    joint_pca.fit(all_flat)

    # Project each model into the shared space
    for data in model_raw.values():
        desc = data["descriptors"]
        flat = desc.reshape(-1, n_anchors)
        proj = joint_pca.transform(flat)
        if n_comp < 3:
            proj = np.concatenate([proj, np.zeros((proj.shape[0], 3 - n_comp))], axis=1)
        n_ls, n_tok = desc.shape[:2]
        data["positions"] = proj.reshape(n_ls, n_tok, 3)

    # Normalize all positions jointly to [-1, 1]
    all_pos = np.concatenate([d["positions"].reshape(-1, 3) for d in model_raw.values()], axis=0)
    max_abs = np.abs(all_pos).max(axis=0)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
    for data in model_raw.values():
        data["positions"] = data["positions"] / max_abs

    # Pairwise metrics (CKA + Procrustes) across normalised depth fractions
    names = list(model_raw.keys())
    pairwise: dict[str, PairMetrics] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            metrics = pairwise_metrics_across_depth(
                model_raw[a]["descriptors"],
                model_raw[b]["descriptors"],
                n_fractions=7,
            )
            pairwise[f"{a} vs {b}"] = PairMetrics(
                fractions=metrics["fractions"],
                cka=metrics["cka"],
                procrustes=metrics["procrustes"],
            )

    # Build response
    model_comparisons = []
    for name, data in model_raw.items():
        positions = data["positions"]
        n_ls, n_tok, _ = positions.shape
        trajectories = [
            TokenTrajectory(token=data["tokens"][t], positions=positions[:, t, :].tolist())
            for t in range(n_tok)
        ]
        model_comparisons.append(ModelComparison(
            model_name=name,
            tokens=data["tokens"],
            n_layers=data["n_layers"],
            trajectories=trajectories,
        ))

    ev = joint_pca.explained_variance_ratio_.tolist()
    while len(ev) < 3:
        ev.append(0.0)

    return CompareResponse(
        models=model_comparisons,
        pairwise=pairwise,
        anchors=req.anchors,
        explained_variance=ev,
    )


# --- Probe mode (Milestone 5) ---

class ProbeRequest(BaseModel):
    anchors: list[str] = Field(..., min_length=2, max_length=8)
    probes: list[str] = Field(..., min_length=1, max_length=100)
    model: str = Field(default="pythia-70m")


class SingleProbeResult(BaseModel):
    label: str
    category: str
    positions: list[list[float]]  # [n_layers+1][3]
    uncertainty: float             # [0,1] — min cosine dist to nearest anchor at final layer


class ProbeResponse(BaseModel):
    probes: list[SingleProbeResult]
    anchor_markers: list[AnchorMarker]
    reconstruction_errors: list[dict]  # [{n_anchors: int, error: float}]
    explained_variance: list[float]
    n_layers: int


@app.post("/probe", response_model=ProbeResponse)
async def probe(req: ProbeRequest):
    """
    Embed a vocabulary of probe concepts in anchor-relative space (Milestone 5).

    Runs forward passes for each unique concept (cached after first call),
    projects all probes + anchors into a jointly fitted 3D PCA space,
    and returns per-probe positions across layers plus a reconstruction error
    curve measuring how much positional accuracy is lost with fewer anchors.
    """
    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(400, f"Unsupported model '{req.model}'. Choose from: {SUPPORTED_MODELS}")
    try:
        ext = _get_or_load_model(req.model)
    except Exception as e:
        raise HTTPException(500, f"Model loading failed: {e}")

    result = compute_probe_positions(ext, req.anchors, req.probes)

    probe_positions = result["probe_positions"]   # [n_layers+1, N, 3]
    anchor_positions = result["anchor_positions"] # [n_layers+1, K, 3]
    n_layers = int(probe_positions.shape[0]) - 1
    N = len(req.probes)
    K = len(req.anchors)

    probe_results = [
        SingleProbeResult(
            label=req.probes[i],
            category=get_concept_category(req.probes[i]),
            positions=probe_positions[:, i, :].tolist(),
            uncertainty=float(result["uncertainty"][i]),
        )
        for i in range(N)
    ]

    anchor_markers = [
        AnchorMarker(label=req.anchors[k], positions=anchor_positions[:, k, :].tolist())
        for k in range(K)
    ]

    return ProbeResponse(
        probes=probe_results,
        anchor_markers=anchor_markers,
        reconstruction_errors=result["reconstruction_errors"],
        explained_variance=result["explained_variance"],
        n_layers=n_layers,
    )


# --- Batch analysis for song mode (Milestone 7) ---

class AnalyzeBatchRequest(BaseModel):
    lines: list[str] = Field(..., min_length=1, max_length=200)
    method: str = Field(default="pca", pattern="^(pca|umap)$")


class BatchLineResult(BaseModel):
    line_index: int
    text: str
    tokens: list[str]
    trajectories: list[TokenTrajectory]


class AnalyzeBatchResponse(BaseModel):
    results: list[BatchLineResult]
    n_layers: int
    explained_variance: list[float]
    model_name: str


@app.post("/analyze-batch", response_model=AnalyzeBatchResponse)
async def analyze_batch(req: AnalyzeBatchRequest):
    """
    Extract residual streams for every lyric line and project them into a
    SHARED 3D coordinate system via joint PCA.

    All positions are comparable across lines — a token cluster in line 12
    occupies the same region as a semantically similar cluster in line 3.
    This shared basis is what makes the ghost-trail accumulation meaningful.
    """
    if _extractor is None:
        raise HTTPException(503, "Model not loaded")

    # 1. Extract residual streams for all lines
    all_extractions = []
    for line in req.lines:
        if not line.strip():
            continue
        try:
            all_extractions.append(_extractor.extract(line.strip()))
        except Exception as e:
            raise HTTPException(500, f"Extraction failed for line '{line[:30]}': {e}")

    if not all_extractions:
        raise HTTPException(400, "No non-empty lines provided")

    # 2. Fit a joint PCA basis on ALL activations, per-layer centered.
    # Per-layer centering removes inter-layer norm growth so PCA captures
    # within-layer token differences rather than the layer-to-layer magnitude shift.
    all_centered = []
    for raw in all_extractions:
        rs = raw["residual_stream"]  # [L+1, T, D]
        layer_means = rs.mean(axis=1, keepdims=True)
        all_centered.append((rs - layer_means).reshape(-1, rs.shape[-1]))
    all_flat = np.concatenate(all_centered, axis=0)

    n_comp = min(3, all_flat.shape[1], all_flat.shape[0] - 1)
    from sklearn.decomposition import PCA as _PCA
    mean = all_flat.mean(axis=0)
    pca = _PCA(n_components=n_comp)
    pca.fit(all_flat - mean)

    # 3. Project each line using the shared per-layer-centered basis
    all_positions = []
    for raw in all_extractions:
        rs = raw["residual_stream"]
        layer_means = rs.mean(axis=1, keepdims=True)
        flat = (rs - layer_means).reshape(-1, rs.shape[-1])
        proj = pca.transform(flat - mean)
        if n_comp < 3:
            proj = np.concatenate([proj, np.zeros((proj.shape[0], 3 - n_comp))], axis=1)
        n_ls, n_tok = rs.shape[:2]
        all_positions.append(proj.reshape(n_ls, n_tok, 3))

    # 4. Normalize JOINTLY across all lines so scale is consistent
    all_cat = np.concatenate([p.reshape(-1, 3) for p in all_positions], axis=0)
    max_abs = np.abs(all_cat).max(axis=0)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
    all_positions = [p / max_abs for p in all_positions]

    # 5. Build response
    results = []
    for i, (raw, positions) in enumerate(zip(all_extractions, all_positions)):
        n_ls, n_tok, _ = positions.shape
        trajectories = [
            TokenTrajectory(token=raw["tokens"][t], positions=positions[:, t, :].tolist())
            for t in range(n_tok)
        ]
        results.append(BatchLineResult(
            line_index=i,
            text=req.lines[i],
            tokens=raw["tokens"],
            trajectories=trajectories,
        ))

    ev = pca.explained_variance_ratio_.tolist()
    while len(ev) < 3:
        ev.append(0.0)

    return AnalyzeBatchResponse(
        results=results,
        n_layers=_extractor.n_layers,
        explained_variance=ev,
        model_name=_extractor.model_name,
    )


# --- Corpus topology (Milestone 6) ---

class TextShapeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500000)
    layer: int = Field(default=-1)
    max_points: int = Field(default=2000, ge=100, le=10000)
    label: str = Field(default="")
    method: str = Field(default="pca", pattern="^(pca|umap)$")


class PersistenceFeatureModel(BaseModel):
    dimension: int
    birth: float
    death: float
    persistence: float


class TextShapeResponse(BaseModel):
    label: str
    n_tokens_total: int
    n_points_sampled: int
    point_cloud_3d: list[list[float]]   # [n_points, 3]
    persistence_diagram: list[PersistenceFeatureModel]
    n_components: int   # H0 features
    n_loops: int        # H1 features
    explained_variance: list[float]


class CorpusCompareRequest(BaseModel):
    texts: list[str] = Field(..., min_length=2, max_length=10)
    labels: list[str] = Field(default=[])
    layer: int = Field(default=-1)
    max_points: int = Field(default=2000, ge=100, le=10000)
    metric: str = Field(default="wasserstein", pattern="^(wasserstein|bottleneck)$")


class CorpusCompareResponse(BaseModel):
    shapes: list[TextShapeResponse]
    distance_matrix: list[list[float]]
    labels: list[str]
    metric: str


def _shape_to_response(shape, n_points_sampled: int) -> TextShapeResponse:
    features = [
        PersistenceFeatureModel(
            dimension=f.dimension,
            birth=f.birth,
            death=f.death,
            persistence=f.persistence,
        )
        for f in shape.persistence_diagram
    ]
    return TextShapeResponse(
        label=shape.label,
        n_tokens_total=shape.n_tokens_total,
        n_points_sampled=n_points_sampled,
        point_cloud_3d=shape.point_cloud_3d.tolist(),
        persistence_diagram=features,
        n_components=sum(1 for f in shape.persistence_diagram if f.dimension == 0),
        n_loops=sum(1 for f in shape.persistence_diagram if f.dimension == 1),
        explained_variance=shape.explained_variance,
    )


@app.post("/text-shape", response_model=TextShapeResponse)
async def text_shape(req: TextShapeRequest):
    """
    Extract token activations at a target layer, subsample for performance,
    compute persistent homology, and return a topological shape fingerprint.

    Processing time: ~1-5s for ≤2000-token texts + 2-5s for persistence.
    Long texts use overlapping chunks and take proportionally longer.
    """
    if _extractor is None:
        raise HTTPException(503, "Model not loaded")

    label = req.label or req.text[:40].strip()

    try:
        activations = _extractor.extract_text_cloud(req.text, layer=req.layer)
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {e}")

    analyzer = TopologyAnalyzer(max_points=req.max_points)
    try:
        shape = analyzer.compute_shape(activations, label=label)
    except Exception as e:
        raise HTTPException(500, f"Topology computation failed: {e}")

    n_sampled = min(len(activations), req.max_points)
    return _shape_to_response(shape, n_sampled)


@app.post("/compare-shapes", response_model=CorpusCompareResponse)
async def compare_shapes(req: CorpusCompareRequest):
    """
    Compute topological fingerprints for multiple texts and return pairwise
    Wasserstein or bottleneck distances between persistence diagrams.

    Used to answer: do philosophy texts have different topological shape than poetry?
    """
    if _extractor is None:
        raise HTTPException(503, "Model not loaded")

    labels = req.labels if len(req.labels) == len(req.texts) else [
        t[:30].strip() for t in req.texts
    ]

    analyzer = TopologyAnalyzer(max_points=req.max_points)
    shapes = []
    for text, label in zip(req.texts, labels):
        try:
            activations = _extractor.extract_text_cloud(text, layer=req.layer)
            shape = analyzer.compute_shape(activations, label=label)
            shapes.append(shape)
        except Exception as e:
            raise HTTPException(500, f"Failed processing '{label}': {e}")

    dist_matrix = analyzer.pairwise_distances(shapes, metric=req.metric)

    n_sampled = [min(s.n_tokens_total, req.max_points) for s in shapes]
    shape_responses = [_shape_to_response(s, n) for s, n in zip(shapes, n_sampled)]

    return CorpusCompareResponse(
        shapes=shape_responses,
        distance_matrix=dist_matrix.tolist(),
        labels=labels,
        metric=req.metric,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _extractor is not None}
