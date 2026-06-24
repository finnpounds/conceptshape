"""
projector.py — Dimensionality reduction for visualizing high-dimensional
activation vectors in 3D space.

Key design decision: PCA is fit on ALL activations across ALL layers jointly,
so the 3D axes are consistent across layers. This lets you see meaningful
movement as tokens traverse layers — they're all in the same coordinate system.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional


class Projector:
    """Project high-dimensional activations to 3D for visualization."""

    def __init__(self, method: str = "pca", n_components: int = 3):
        self.method = method
        self.n_components = n_components
        self._pca: Optional[PCA] = None

    def project_trajectories(
        self,
        residual_stream: np.ndarray,
    ) -> dict:
        """
        Project residual stream activations to 3D.

        Args:
            residual_stream: shape [n_layers+1, n_tokens, d_model]

        Returns:
            {
                "positions": np.ndarray of shape [n_layers+1, n_tokens, 3],
                "explained_variance": list[float] (per component),
                "method": str,
            }
        """
        n_layer_steps, n_tokens, d_model = residual_stream.shape

        if self.method == "pca":
            return self._project_pca(residual_stream)
        elif self.method == "umap":
            return self._project_umap(residual_stream)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

    def _project_pca(self, residual_stream: np.ndarray) -> dict:
        """PCA projection — fast, preserves global linear structure."""
        n_layer_steps, n_tokens, d_model = residual_stream.shape

        # Per-layer centering: subtract each layer's mean across tokens before PCA.
        #
        # Without this, PC1 is dominated by inter-layer norm growth: residual
        # connections accumulate vector magnitude across layers (~10x from embedding
        # to final layer), so all tokens at the final layer project to the same
        # large PC1 value and appear as a single cluster regardless of their
        # semantic content. Per-layer centering removes this shared growth component
        # and focuses PCA on *how tokens differ from each other* at each depth —
        # which is the semantically meaningful structure.
        layer_means = residual_stream.mean(axis=1, keepdims=True)  # [L+1, 1, D]
        stream_centered = residual_stream - layer_means             # [L+1, T, D]
        all_activations = stream_centered.reshape(-1, d_model)

        # Fit PCA on the per-layer-centered data
        pca = PCA(n_components=self.n_components)
        projected_flat = pca.fit_transform(all_activations)

        # Reshape back to [n_layer_steps, n_tokens, 3]
        positions = projected_flat.reshape(n_layer_steps, n_tokens, self.n_components)

        # Normalize to [-1, 1] per axis
        max_abs = np.abs(positions).max(axis=(0, 1), keepdims=True)
        max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
        positions = positions / max_abs

        self._pca = pca
        self._pca_mean = layer_means.mean(axis=0).squeeze()  # global approx for project_points

        return {
            "positions": positions,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "method": "pca",
        }

    def fit_basis(self, all_activations: np.ndarray) -> None:
        """
        Fit PCA on a combined activation matrix WITHOUT projecting.
        Call once on the union of all data, then use project_with_basis()
        for each individual item to keep positions in a shared coordinate system.

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
        Project activations using a previously fitted PCA basis (from fit_basis).
        Does NOT re-normalize — caller is responsible for joint normalization.

        Args:
            residual_stream: [n_layers+1, n_tokens, d_model]

        Returns:
            positions: [n_layers+1, n_tokens, 3] in PCA space (not yet normalized)
        """
        if self._pca is None or not hasattr(self, "_pca_mean"):
            raise ValueError("No PCA basis — call fit_basis() first.")
        n_layer_steps, n_tokens, d_model = residual_stream.shape
        flat = residual_stream.reshape(-1, d_model)
        projected = self._pca.transform(flat - self._pca_mean)
        if projected.shape[1] < 3:
            projected = np.concatenate(
                [projected, np.zeros((projected.shape[0], 3 - projected.shape[1]))], axis=1
            )
        return projected.reshape(n_layer_steps, n_tokens, 3)

    def _project_umap(self, residual_stream: np.ndarray) -> dict:
        """UMAP projection — preserves local neighborhoods / clusters."""
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")

        n_layer_steps, n_tokens, d_model = residual_stream.shape
        all_activations = residual_stream.reshape(-1, d_model)

        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=min(15, len(all_activations) - 1),
            min_dist=0.1,
            metric="cosine",
        )
        projected_flat = reducer.fit_transform(all_activations)

        positions = projected_flat.reshape(n_layer_steps, n_tokens, self.n_components)

        # Normalize
        max_abs = np.abs(positions).max(axis=(0, 1), keepdims=True)
        max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
        positions = positions / max_abs

        return {
            "positions": positions,
            "explained_variance": [],  # UMAP doesn't have this
            "method": "umap",
        }

    def project_points(
        self,
        vectors: np.ndarray,
        pca_basis: Optional[PCA] = None,
    ) -> np.ndarray:
        """
        Project arbitrary vectors using a previously fit PCA basis.
        Useful for projecting anchor concept embeddings into the same
        coordinate system as the trajectories.

        Args:
            vectors: shape [n_points, d_model]
            pca_basis: a fitted PCA object (default: use last fit)

        Returns:
            np.ndarray of shape [n_points, 3]
        """
        pca = pca_basis or self._pca
        if pca is None:
            raise ValueError("No PCA basis available. Run project_trajectories first.")

        return pca.transform(vectors - pca.mean_)
