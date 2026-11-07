"""
anchor.py — Anchor-relative coordinate system for Milestone 3.

Core idea (Moschella et al. 2022 — Relative Representations):
  Describe each token embedding not by its raw coordinates, but by its
  cosine distances to a fixed set of "anchor" concept embeddings.
  This produces a model-agnostic descriptor that enables cross-model comparison.

Pipeline:
  1. Extract residual stream for input text: [n_layers+1, n_tokens, d_model]
  2. Extract mean-pooled anchor streams: [n_anchors, n_layers+1, d_model]
  3. Centre tokens AND anchors by ONE shared per-layer mean (content tokens +
     anchors), then compute cosine distances: [n_layers+1, n_tokens, n_anchors]
  4. Project to 3D via PCA (fit on all content-token/layer rows jointly)
  5. Project anchor self-distances into the same space for reference markers

Why the shared mean matters: anchors are mean-pooled single words from a
different context, so relative to the *sentence's* token mean they all point
the same way, and a token's K anchor distances collapse to one number repeated
K times (the joint PCA then put 99.8% of its variance on PC1 — a line, not a
map; the four anchor columns were correlated 1.00). Centering by the mean over
content tokens + anchors (the recipe probe.py already used) removes that shared
offset. BOS is excluded from the mean for the reason given in projector.py.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from app.projector import content_slice, unit_scale


def joint_layer_means(
    residual_stream: np.ndarray,  # [n_layers+1, n_tokens, d_model]
    anchor_streams: np.ndarray,   # [n_anchors, n_layers+1, d_model]
    has_bos: bool = True,
) -> np.ndarray:
    """Per-layer mean over content tokens and anchors together: [n_layers+1, d_model]."""
    content = residual_stream[:, content_slice(residual_stream.shape[1], has_bos), :]
    anchors = anchor_streams.transpose(1, 0, 2)  # [n_layers+1, n_anchors, d_model]
    return np.concatenate([content, anchors], axis=1).mean(axis=1)


def compute_anchor_descriptors(
    residual_stream: np.ndarray,  # [n_layers+1, n_tokens, d_model]
    anchor_streams: np.ndarray,   # [n_anchors, n_layers+1, d_model]
    layer_means: np.ndarray | None = None,  # [n_layers+1, d_model]
    has_bos: bool = True,
) -> np.ndarray:
    """
    For each (token, layer), compute cosine distance to each anchor at that layer.
    Distances are computed in the same layer — so 'anchor' at layer l means
    the anchor concept's representation after l transformer blocks.

    Per-layer centering: tokens and anchors are both shifted by `layer_means`
    (default: joint_layer_means over content tokens + anchors) before the cosine.
    Without centering, all vectors at a given depth share a large common
    component (the accumulated residual norm), making all cosine distances
    approach 0 regardless of semantic content.

    Returns: [n_layers+1, n_tokens, n_anchors]
    """
    if layer_means is None:
        layer_means = joint_layer_means(residual_stream, anchor_streams, has_bos)

    n_layer_steps, n_tokens, _ = residual_stream.shape
    n_anchors = anchor_streams.shape[0]
    descriptors = np.zeros((n_layer_steps, n_tokens, n_anchors))

    for layer in range(n_layer_steps):
        m = layer_means[layer]
        descriptors[layer] = cosine_distances(
            residual_stream[layer] - m,
            anchor_streams[:, layer, :] - m,
        )

    return descriptors


def compute_anchor_self_descriptors(
    anchor_streams: np.ndarray,  # [n_anchors, n_layers+1, d_model]
    layer_means: np.ndarray | None = None,  # [n_layers+1, d_model]
) -> np.ndarray:
    """
    Pairwise cosine distances between anchors at each layer, centred by the SAME
    `layer_means` used for the token descriptors so the reference markers live
    in the same descriptor space. Without `layer_means` (callers that have no
    token stream) the anchors' own mean is used.

    Returns: [n_layers+1, n_anchors, n_anchors]
    """
    n_anchors, n_layer_steps, _ = anchor_streams.shape
    result = np.zeros((n_layer_steps, n_anchors, n_anchors))

    for layer in range(n_layer_steps):
        anchor_vecs = anchor_streams[:, layer, :]  # [n_anchors, d_model]
        m = layer_means[layer] if layer_means is not None else anchor_vecs.mean(axis=0)
        centered = anchor_vecs - m
        result[layer] = cosine_distances(centered, centered)

    return result


def project_to_3d(
    descriptors: np.ndarray,        # [n_layers+1, n_tokens, n_anchors]
    anchor_self_dists: np.ndarray,  # [n_layers+1, n_anchors, n_anchors]
    n_components: int = 3,
    has_bos: bool = True,
) -> dict:
    """
    Project anchor-relative descriptors to 3D via PCA.
    PCA is fit on ALL content-token/layer rows jointly so that axes are
    consistent across layers (same coordinate system, meaningful movement).
    The BOS row (if present) and the anchor self-descriptors are transformed
    with the same basis but do not shape it.

    Returns:
        positions:          np.ndarray [n_layers+1, n_tokens, 3]
        anchor_positions:   np.ndarray [n_layers+1, n_anchors, 3]
        explained_variance: list[float] — per component
    """
    n_layer_steps, n_tokens, n_anchors = descriptors.shape
    content = content_slice(n_tokens, has_bos)

    fit_data = descriptors[:, content, :].reshape(-1, n_anchors)
    n_comp = max(1, min(n_components, n_anchors, fit_data.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    pca.fit(fit_data)

    def _project(flat: np.ndarray) -> np.ndarray:
        proj = pca.transform(flat)
        if n_comp < n_components:
            pad = np.zeros((proj.shape[0], n_components - n_comp))
            proj = np.concatenate([proj, pad], axis=1)
        return proj

    positions = _project(descriptors.reshape(-1, n_anchors)).reshape(
        n_layer_steps, n_tokens, n_components
    )
    anchor_positions = _project(anchor_self_dists.reshape(-1, n_anchors)).reshape(
        n_layer_steps, n_anchors, n_components
    )

    # Normalize both to [-1, 1] on the extent of content tokens + anchors
    scale = unit_scale(positions[:, content, :], anchor_positions)
    positions = positions / scale
    anchor_positions = anchor_positions / scale

    ev = pca.explained_variance_ratio_.tolist()
    while len(ev) < n_components:
        ev.append(0.0)

    return {
        "positions": positions,
        "anchor_positions": anchor_positions,
        "explained_variance": ev,
    }
