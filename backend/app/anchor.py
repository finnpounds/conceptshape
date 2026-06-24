"""
anchor.py — Anchor-relative coordinate system for Milestone 3.

Core idea (Moschella et al. 2022 — Relative Representations):
  Describe each token embedding not by its raw coordinates, but by its
  cosine distances to a fixed set of "anchor" concept embeddings.
  This produces a model-agnostic descriptor that enables cross-model comparison.

Pipeline:
  1. Extract residual stream for input text: [n_layers+1, n_tokens, d_model]
  2. Extract mean-pooled anchor streams: [n_anchors, n_layers+1, d_model]
  3. Compute cosine distances: [n_layers+1, n_tokens, n_anchors]
  4. Project to 3D via PCA (fit on all token-layer data jointly)
  5. Project anchor self-distances into the same space for reference markers
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances


def compute_anchor_descriptors(
    residual_stream: np.ndarray,  # [n_layers+1, n_tokens, d_model]
    anchor_streams: np.ndarray,   # [n_anchors, n_layers+1, d_model]
) -> np.ndarray:
    """
    For each (token, layer), compute cosine distance to each anchor at that layer.
    Distances are computed in the same layer — so 'anchor' at layer l means
    the anchor concept's representation after l transformer blocks.

    Per-layer centering: subtract the mean token vector at each layer before
    computing cosine distances. Without this, all vectors at a given depth share
    a large common component (the accumulated residual norm), making all cosine
    distances approach 0 regardless of semantic content — the same issue that
    required per-layer centering in the absolute PCA view.

    Returns: [n_layers+1, n_tokens, n_anchors]
    """
    n_layer_steps, n_tokens, _ = residual_stream.shape
    n_anchors = anchor_streams.shape[0]
    descriptors = np.zeros((n_layer_steps, n_tokens, n_anchors))

    for layer in range(n_layer_steps):
        token_vecs = residual_stream[layer]        # [n_tokens, d_model]
        anchor_vecs = anchor_streams[:, layer, :]  # [n_anchors, d_model]
        # Center by the token mean at this layer so cosine distances capture
        # directional differences rather than similarity to the shared background
        layer_mean = token_vecs.mean(axis=0)       # [d_model]
        descriptors[layer] = cosine_distances(
            token_vecs - layer_mean,
            anchor_vecs - layer_mean,
        )

    return descriptors


def compute_anchor_self_descriptors(
    anchor_streams: np.ndarray,  # [n_anchors, n_layers+1, d_model]
) -> np.ndarray:
    """
    Pairwise cosine distances between anchors at each layer.
    Used to place anchor reference markers in the projected space.

    Per-layer centering applied for the same reason as compute_anchor_descriptors:
    shared residual norm collapses pairwise cosine distances toward zero.

    Returns: [n_layers+1, n_anchors, n_anchors]
    """
    n_anchors, n_layer_steps, _ = anchor_streams.shape
    result = np.zeros((n_layer_steps, n_anchors, n_anchors))

    for layer in range(n_layer_steps):
        anchor_vecs = anchor_streams[:, layer, :]  # [n_anchors, d_model]
        anchor_vecs_c = anchor_vecs - anchor_vecs.mean(axis=0)
        result[layer] = cosine_distances(anchor_vecs_c, anchor_vecs_c)

    return result


def project_to_3d(
    descriptors: np.ndarray,        # [n_layers+1, n_tokens, n_anchors]
    anchor_self_dists: np.ndarray,  # [n_layers+1, n_anchors, n_anchors]
    n_components: int = 3,
) -> dict:
    """
    Project anchor-relative descriptors to 3D via PCA.
    PCA is fit on ALL token-layer combinations jointly so that axes are
    consistent across layers (same coordinate system, meaningful movement).
    Anchor points are projected into the identical space.

    Returns:
        positions:          np.ndarray [n_layers+1, n_tokens, 3]
        anchor_positions:   np.ndarray [n_layers+1, n_anchors, 3]
        explained_variance: list[float] — per component
    """
    n_layer_steps, n_tokens, n_anchors = descriptors.shape

    # Flatten all token-layer descriptors for joint PCA fit
    all_token_data = descriptors.reshape(-1, n_anchors)

    # Handle case where n_anchors < n_components
    n_comp = min(n_components, n_anchors, all_token_data.shape[0] - 1)
    n_comp = max(n_comp, 1)

    pca = PCA(n_components=n_comp)
    token_proj_flat = pca.fit_transform(all_token_data)

    # Pad to 3 columns if we have fewer than 3 components
    if n_comp < n_components:
        pad = np.zeros((token_proj_flat.shape[0], n_components - n_comp))
        token_proj_flat = np.concatenate([token_proj_flat, pad], axis=1)

    positions = token_proj_flat.reshape(n_layer_steps, n_tokens, n_components)

    # Project anchor self-descriptors into the same space
    anchor_flat = anchor_self_dists.reshape(-1, n_anchors)
    anchor_proj = pca.transform(anchor_flat)
    if n_comp < n_components:
        pad = np.zeros((anchor_proj.shape[0], n_components - n_comp))
        anchor_proj = np.concatenate([anchor_proj, pad], axis=1)
    anchor_positions = anchor_proj.reshape(n_layer_steps, n_anchors, n_components)

    # Normalize both to [-1, 1] using combined scale
    combined = np.concatenate([
        positions.reshape(-1, n_components),
        anchor_positions.reshape(-1, n_components),
    ], axis=0)
    max_abs = np.abs(combined).max(axis=0)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)

    positions = positions / max_abs
    anchor_positions = anchor_positions / max_abs

    ev = pca.explained_variance_ratio_.tolist()
    while len(ev) < n_components:
        ev.append(0.0)

    return {
        "positions": positions,
        "anchor_positions": anchor_positions,
        "explained_variance": ev,
    }
