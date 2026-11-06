"""
projector.py — Dimensionality reduction for visualizing high-dimensional
activation vectors in 3D space.

Key design decisions:
  * PCA is fit on ALL layers jointly, so the 3D axes are consistent across
    layers and movement through the stack is meaningful.
  * Each layer is centred by its mean over CONTENT tokens, and the PCA is fit
    on content tokens only. The BOS token (position 0, the attention sink) has
    a norm up to ~10x any content token at mid layers; if it takes part in the
    centering or the fit, PC1 becomes "the BOS direction" and every real token
    is squashed onto a plane (measured in docs/review-2026-11-04.md: 87% of
    variance on PC1, and on-screen distances at layers 2-3 uncorrelated with
    the true ones). BOS is still *transformed* with the same basis so the UI's
    "show BOS" toggle works — it just no longer defines the axes.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional


def unit_scale(*extent_arrays: np.ndarray) -> np.ndarray:
    """
    Per-axis max-abs over the given arrays (any shape [..., 3]).
    Divide positions by this so that the given arrays fill [-1, 1].
    """
    allp = np.concatenate([a.reshape(-1, a.shape[-1]) for a in extent_arrays], axis=0)
    max_abs = np.abs(allp).max(axis=0)
    return np.where(max_abs < 1e-8, 1.0, max_abs)


def content_slice(n_tokens: int, has_bos: bool) -> slice:
    """Index of the content tokens: everything after BOS when BOS is present."""
    return slice(1, None) if (has_bos and n_tokens > 1) else slice(0, None)


class Projector:
    """Project high-dimensional activations to 3D for visualization."""

    def __init__(self, method: str = "pca", n_components: int = 3):
        self.method = method
        self.n_components = n_components
        self._pca: Optional[PCA] = None

    def project_trajectories(
        self,
        residual_stream: np.ndarray,
        has_bos: bool = True,
    ) -> dict:
        """
        Project residual stream activations to 3D.

        Args:
            residual_stream: shape [n_layers+1, n_tokens, d_model]
            has_bos: whether position 0 is the BOS token (kept out of the fit)

        Returns:
            {
                "positions": np.ndarray of shape [n_layers+1, n_tokens, 3],
                "explained_variance": list[float] (per component),
                "method": str,
            }
        """
        if self.method == "pca":
            return self._project_pca(residual_stream, has_bos)
        elif self.method == "umap":
            return self._project_umap(residual_stream, has_bos)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def _project_pca(self, residual_stream: np.ndarray, has_bos: bool) -> dict:
        """PCA projection — fast, preserves global linear structure."""
        n_layer_steps, n_tokens, d_model = residual_stream.shape
        content = content_slice(n_tokens, has_bos)

        # Per-layer centering: subtract each layer's mean across CONTENT tokens.
        #
        # Without this, PC1 is dominated by inter-layer norm growth: residual
        # connections accumulate vector magnitude across layers (~10x from embedding
        # to final layer), so all tokens at the final layer project to the same
        # large PC1 value and appear as a single cluster regardless of their
        # semantic content. Per-layer centering focuses PCA on *how tokens differ
        # from each other* at each depth. The mean must exclude BOS: its norm is so
        # large that a BOS-inclusive mean is mostly BOS/n_tokens, which adds the
        # same huge offset to every content token.
        layer_means = residual_stream[:, content, :].mean(axis=1, keepdims=True)  # [L+1, 1, D]
        stream_centered = residual_stream - layer_means                           # [L+1, T, D]
        fit_data = stream_centered[:, content, :].reshape(-1, d_model)

        n_comp = max(1, min(self.n_components, fit_data.shape[0] - 1, d_model))
        pca = PCA(n_components=n_comp)
        pca.fit(fit_data)

        projected_flat = pca.transform(stream_centered.reshape(-1, d_model))
        if n_comp < self.n_components:
            pad = np.zeros((projected_flat.shape[0], self.n_components - n_comp))
            projected_flat = np.concatenate([projected_flat, pad], axis=1)
        positions = projected_flat.reshape(n_layer_steps, n_tokens, self.n_components)

        # Normalize to [-1, 1] per axis on the CONTENT tokens' extent, so the BOS
        # outlier cannot shrink the scene. BOS itself may land outside [-1, 1].
        positions = positions / unit_scale(positions[:, content, :])

        self._pca = pca
        ev = pca.explained_variance_ratio_.tolist()
        while len(ev) < self.n_components:
            ev.append(0.0)

        return {
            "positions": positions,
            "explained_variance": ev,
            "method": "pca",
        }

    def _project_umap(self, residual_stream: np.ndarray, has_bos: bool) -> dict:
        """UMAP projection — preserves local neighborhoods / clusters."""
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        n_layer_steps, n_tokens, d_model = residual_stream.shape
        content = content_slice(n_tokens, has_bos)
        all_activations = residual_stream.reshape(-1, d_model)
        fit_data = residual_stream[:, content, :].reshape(-1, d_model)

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=min(15, len(fit_data) - 1),
            min_dist=0.1,
            metric="cosine",
        )
        # Fit on content tokens only (BOS is an extreme outlier even under
        # cosine), then place every token — BOS included — with the fitted map.
        reducer.fit(fit_data)
        projected_flat = reducer.transform(all_activations)

        positions = projected_flat.reshape(n_layer_steps, n_tokens, self.n_components)
        positions = positions / unit_scale(positions[:, content, :])

        return {
            "positions": positions,
            "explained_variance": [],  # UMAP doesn't have this
            "method": "umap",
        }
