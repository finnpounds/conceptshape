"""
probe.py — Geometric extrapolation / probe mode (Milestone 5).

Core idea: embed a vocabulary of ~100 concepts in anchor-relative space,
show where they fall, and measure how much positional information is
recoverable from a partial set of anchor distances.

The reconstruction error curve — error vs number of anchors — is the
empirical answer to "is partial concept geometry sufficient to infer
the rest of the belief structure?"

Caching: concept streams ([n_layers+1, d_model]) are cached in memory by
(model_name, concept_text) so repeated calls are near-instant after
the first embedding run.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.extractor import ModelExtractor

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

CONCEPT_VOCABULARY: dict[str, list[str]] = {
    "emotions": [
        "love", "fear", "joy", "anger", "sadness", "hope",
        "trust", "surprise", "guilt", "pride", "shame", "compassion",
    ],
    "relations": [
        "friend", "enemy", "parent", "child", "teacher", "student",
        "ally", "rival", "partner", "stranger", "leader", "follower",
    ],
    "abstractions": [
        "truth", "justice", "beauty", "freedom", "time", "death",
        "life", "power", "knowledge", "meaning", "order", "chaos",
    ],
    "states": [
        "alive", "dead", "happy", "sad", "sick", "healthy",
        "young", "old", "strong", "weak", "awake", "asleep",
    ],
    "nature": [
        "water", "fire", "earth", "sky", "sun", "moon",
        "stone", "tree", "mountain", "ocean", "wind", "light",
    ],
    "mind": [
        "thought", "belief", "desire", "memory", "dream",
        "consciousness", "reason", "emotion", "will", "perception",
    ],
    "qualities": [
        "good", "evil", "beautiful", "ugly", "true", "false",
        "real", "imaginary", "certain", "uncertain", "simple", "complex",
    ],
}

CATEGORY_COLORS: dict[str, str] = {
    "emotions":    "#ff6b6b",
    "relations":   "#69d2e7",
    "abstractions":"#a8e6cf",
    "states":      "#ffd700",
    "nature":      "#ff8c00",
    "mind":        "#b388ff",
    "qualities":   "#80cbc4",
    "custom":      "#e0e0e8",
}


def get_concept_category(concept: str) -> str:
    for cat, words in CONCEPT_VOCABULARY.items():
        if concept in words:
            return cat
    return "custom"


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

# Keyed (model_name, concept_text) → np.ndarray [n_layers+1, d_model]
_concept_cache: dict[tuple[str, str], np.ndarray] = {}


def get_concept_stream(
    extractor: "ModelExtractor",
    concept: str,
) -> np.ndarray:
    """Return mean-pooled residual stream for a concept, cached after first call."""
    key = (extractor.model_name, concept)
    if key not in _concept_cache:
        _concept_cache[key] = extractor.extract_concept_stream(concept)
    return _concept_cache[key]


# ---------------------------------------------------------------------------
# Core probe computation
# ---------------------------------------------------------------------------

def compute_probe_positions(
    extractor: "ModelExtractor",
    anchors: list[str],
    probes: list[str],
) -> dict:
    """
    Embed anchor and probe concepts in anchor-relative space and project
    jointly to 3D. Also compute a reconstruction error curve measuring
    how much positional accuracy is lost when fewer anchor distances are used.

    Returns:
        probe_positions:   [n_layers+1, N_probes, 3]
        anchor_positions:  [n_layers+1, K_anchors, 3]
        uncertainty:       [N_probes]   — min cosine distance to nearest anchor at final layer
        reconstruction_errors: list of {n_anchors: int, error: float}
        explained_variance: list[float]
    """
    K = len(anchors)
    N = len(probes)

    # Fetch (cached) concept streams
    anchor_streams = np.stack([
        get_concept_stream(extractor, a) for a in anchors
    ], axis=0)  # [K, n_layers+1, d_model]

    probe_streams = np.stack([
        get_concept_stream(extractor, p) for p in probes
    ], axis=0)  # [N, n_layers+1, d_model]

    n_layer_steps = anchor_streams.shape[1]

    # All concept streams combined — used to compute a per-layer mean for centering.
    # Per-layer centering removes the dominant "background" language representation
    # at each depth so cosine distances capture directional variation rather than
    # similarity to a shared residual-norm direction (same fix as absolute PCA view).
    all_concept_streams = np.concatenate(
        [probe_streams, anchor_streams], axis=0
    )  # [N+K, n_layers+1, d_model]

    # Probe descriptors: cosine distance to each anchor at each layer
    probe_descriptors = np.zeros((n_layer_steps, N, K))
    for layer in range(n_layer_steps):
        probe_vecs  = probe_streams[:, layer, :]   # [N, d]
        anchor_vecs = anchor_streams[:, layer, :]  # [K, d]
        layer_mean  = all_concept_streams[:, layer, :].mean(axis=0)  # [d]
        probe_descriptors[layer] = cosine_distances(
            probe_vecs  - layer_mean,
            anchor_vecs - layer_mean,
        )

    # Anchor self-descriptors: pairwise distances between anchors at each layer
    anchor_self = np.zeros((n_layer_steps, K, K))
    for layer in range(n_layer_steps):
        av = anchor_streams[:, layer, :]
        layer_mean = all_concept_streams[:, layer, :].mean(axis=0)
        av_c = av - layer_mean
        anchor_self[layer] = cosine_distances(av_c, av_c)

    # Joint PCA on probes + anchors (all layers) so axes are consistent
    all_data = np.concatenate([
        probe_descriptors.reshape(-1, K),
        anchor_self.reshape(-1, K),
    ], axis=0)

    n_comp = max(1, min(3, K, all_data.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    pca.fit(all_data)

    def _project(data: np.ndarray) -> np.ndarray:
        """Project [M, K] data with padding to 3 components."""
        proj = pca.transform(data)
        if n_comp < 3:
            proj = np.concatenate([proj, np.zeros((proj.shape[0], 3 - n_comp))], axis=1)
        return proj

    probe_proj  = _project(probe_descriptors.reshape(-1, K)).reshape(n_layer_steps, N, 3)
    anchor_proj = _project(anchor_self.reshape(-1, K)).reshape(n_layer_steps, K, 3)

    # Normalize jointly to [-1, 1]
    all_pos = np.concatenate([probe_proj.reshape(-1, 3), anchor_proj.reshape(-1, 3)], axis=0)
    max_abs = np.abs(all_pos).max(axis=0)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
    probe_proj  = probe_proj  / max_abs
    anchor_proj = anchor_proj / max_abs

    # Uncertainty = min cosine distance to nearest anchor at the final layer
    # (small = probe is close to some anchor = well-anchored = confident)
    final_desc = probe_descriptors[-1]           # [N, K]
    uncertainty = np.clip(final_desc.min(axis=1), 0.0, 1.0)  # [N]

    # Reconstruction error curve
    # How much positional accuracy is lost when only k < K anchor distances are used?
    # Method: zero-pad partial descriptor → project → compare with full projection.
    full_final = probe_proj[-1]  # [N, 3] at final layer (normalized)
    reconstruction_errors = []
    for k in range(2, K + 1):
        partial = np.zeros_like(probe_descriptors[-1])  # [N, K]
        partial[:, :k] = probe_descriptors[-1, :, :k]
        partial_proj = _project(partial) / max_abs
        mean_err = float(np.mean(np.linalg.norm(full_final - partial_proj, axis=1)))
        reconstruction_errors.append({"n_anchors": k, "error": mean_err})

    ev = pca.explained_variance_ratio_.tolist()
    while len(ev) < 3:
        ev.append(0.0)

    return {
        "probe_positions":   probe_proj,
        "anchor_positions":  anchor_proj,
        "uncertainty":       uncertainty.tolist(),
        "reconstruction_errors": reconstruction_errors,
        "explained_variance": ev,
    }
