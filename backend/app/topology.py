"""
topology.py — Topological Data Analysis for corpus-scale shape comparison (Milestone 6).

Core idea: a text's activations at a given layer form a point cloud in d_model-dimensional
space. That cloud has geometric shape — connected components, loops, voids — that can be
fingerprinted using persistent homology. Comparing fingerprints (persistence diagrams)
across texts reveals structural similarity independent of coordinate systems.

Libraries: ripser (fast Vietoris-Rips persistence), persim (diagram distances).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PersistenceFeature:
    dimension: int      # 0 = component, 1 = loop, 2 = void
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.death - self.birth


@dataclass
class TextShape:
    label: str
    point_cloud: np.ndarray               # [n_points, d_model] subsampled
    point_cloud_3d: np.ndarray            # [n_points, 3] PCA projected
    persistence_diagram: list[PersistenceFeature]
    n_tokens_total: int
    explained_variance: list[float]


# ---------------------------------------------------------------------------
# TopologyAnalyzer
# ---------------------------------------------------------------------------

class TopologyAnalyzer:
    """
    Compute persistent homology of text-activation point clouds and compare
    them using Wasserstein or bottleneck distance on persistence diagrams.
    """

    def __init__(self, max_points: int = 2000, max_dim: int = 1):
        """
        Args:
            max_points: subsample to this many points before persistence
                        (ripser is O(n³), 2000 pts ~ 2-5s on CPU)
            max_dim:    max homology dimension (0=components, 1=loops)
                        dim 2 (voids) is expensive; default 1 is safe
        """
        self.max_points = max_points
        self.max_dim = max_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_shape(self, activations: np.ndarray, label: str = "") -> TextShape:
        """
        Compute the topological shape of a text's activation point cloud.

        Args:
            activations: [n_tokens, d_model] — raw activations at target layer
            label: human-readable name for this text

        Returns:
            TextShape with subsampled point cloud + persistence diagram
        """
        from ripser import ripser
        from sklearn.decomposition import PCA

        n_tokens_total = len(activations)

        # 1. Subsample using farthest point sampling for geometric coverage
        if len(activations) > self.max_points:
            indices = self.farthest_point_sample(activations, self.max_points)
            point_cloud = activations[indices]
        else:
            point_cloud = activations.copy()

        # Normalize to unit variance for stable distance computation
        std = point_cloud.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        point_cloud_norm = (point_cloud - point_cloud.mean(axis=0)) / std

        # 2. Persistent homology via Vietoris-Rips complex (cosine metric)
        result = ripser(point_cloud_norm, maxdim=self.max_dim, metric="cosine")
        diagrams = result["dgms"]  # list: dgms[0]=H0, dgms[1]=H1, ...

        # 3. Convert to structured features, filtering out infinite deaths in H0
        features: list[PersistenceFeature] = []
        for dim, dgm in enumerate(diagrams):
            for birth, death in dgm:
                if not np.isfinite(death):
                    continue  # skip the one infinite H0 bar (connected component)
                if death > birth:  # guard against numerical noise
                    features.append(PersistenceFeature(dim, float(birth), float(death)))

        # 4. PCA projection to 3D for visualization (fit on subsampled cloud)
        n_comp = min(3, point_cloud.shape[0] - 1, point_cloud.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(point_cloud)
        if n_comp < 3:
            proj = np.concatenate([proj, np.zeros((proj.shape[0], 3 - n_comp))], axis=1)

        # Normalize to [-1, 1]
        max_abs = np.abs(proj).max(axis=0)
        max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
        proj = proj / max_abs

        ev = pca.explained_variance_ratio_.tolist()
        while len(ev) < 3:
            ev.append(0.0)

        return TextShape(
            label=label,
            point_cloud=point_cloud,
            point_cloud_3d=proj,
            persistence_diagram=features,
            n_tokens_total=n_tokens_total,
            explained_variance=ev,
        )

    def compare(
        self,
        shape_a: TextShape,
        shape_b: TextShape,
        metric: Literal["wasserstein", "bottleneck"] = "wasserstein",
    ) -> dict:
        """
        Compare two text shapes via persistence diagram distance.

        Returns dict with H0 and H1 distances plus a weighted combined score.
        Higher distance = more different topological structure.
        """
        import persim

        distances = {}
        for dim in range(min(self.max_dim + 1, 2)):
            dgm_a = self._get_diagram(shape_a, dim)
            dgm_b = self._get_diagram(shape_b, dim)

            if len(dgm_a) == 0 and len(dgm_b) == 0:
                distances[f"H{dim}"] = 0.0
                continue

            # persim needs at least one point per diagram; pad with trivial feature
            if len(dgm_a) == 0:
                dgm_a = np.array([[0.0, 0.0]])
            if len(dgm_b) == 0:
                dgm_b = np.array([[0.0, 0.0]])

            if metric == "wasserstein":
                d = float(persim.wasserstein(dgm_a, dgm_b))
            else:
                d = float(persim.bottleneck(dgm_a, dgm_b))

            distances[f"H{dim}"] = d

        # Combined: weight H1 (loops) more than H0 (components) — loops capture richer structure
        combined = distances.get("H0", 0.0) * 0.3 + distances.get("H1", 0.0) * 0.7
        distances["combined"] = combined
        return distances

    def pairwise_distances(
        self,
        shapes: list[TextShape],
        metric: Literal["wasserstein", "bottleneck"] = "wasserstein",
    ) -> np.ndarray:
        """
        Compute NxN symmetric distance matrix across all shapes.

        Returns:
            [N, N] float array — distances[i][j] = topological distance between text i and j
        """
        N = len(shapes)
        matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = self.compare(shapes[i], shapes[j], metric)["combined"]
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def farthest_point_sample(self, points: np.ndarray, n: int) -> np.ndarray:
        """
        Subsample n points using farthest point sampling (greedy).
        Better than random — preserves geometric coverage of the manifold.

        Returns indices of selected points.
        """
        N = len(points)
        if n >= N:
            return np.arange(N)

        selected = [np.random.randint(N)]
        # Distances from each point to the nearest already-selected point
        min_dists = np.full(N, np.inf)

        for _ in range(n - 1):
            last = points[selected[-1]]
            # Update min distances using squared Euclidean (fast)
            dists = np.sum((points - last) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return np.array(selected)

    def _get_diagram(self, shape: TextShape, dim: int) -> np.ndarray:
        """Extract birth-death pairs for a given homology dimension as [N, 2] array."""
        pts = [
            [f.birth, f.death]
            for f in shape.persistence_diagram
            if f.dimension == dim
        ]
        return np.array(pts) if pts else np.empty((0, 2))
