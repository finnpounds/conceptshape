"""
alignment.py — Representation similarity metrics for cross-model comparison (M4).

Two complementary metrics:

  linear_cka(A, B)
    Centered Kernel Alignment — measures similarity between representation
    spaces regardless of rotation or scale. 1.0 = identical structure,
    0.0 = no shared structure. Standard metric in representation similarity
    literature (Kornblith et al. 2019).

  procrustes_similarity(A, B)
    Orthogonal Procrustes — measures similarity after optimal rotation /
    reflection. Also in [0, 1]. More geometric than CKA; sensitive to
    the relative arrangement of points.

Both operate on [n_samples, n_features] matrices (token x anchor_descriptor
vectors at a given layer). They give complementary evidence of convergence.
"""

import numpy as np


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices.

    Args:
        A: [n_samples, n_features_a]
        B: [n_samples, n_features_b]

    Returns:
        float in [0, 1]
    """
    # Center columns
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)

    # HSIC via Frobenius norms of cross-products
    dot = np.linalg.norm(A.T @ B, "fro") ** 2
    norm_a = np.linalg.norm(A.T @ A, "fro")
    norm_b = np.linalg.norm(B.T @ B, "fro")

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return float(np.clip(dot / (norm_a * norm_b), 0.0, 1.0))


def procrustes_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Procrustes similarity after optimal orthogonal alignment.

    Args:
        A: [n_samples, n_features]
        B: [n_samples, n_features]  (same n_samples)

    Returns:
        float in [0, 1]
    """
    A = A / (np.linalg.norm(A, "fro") + 1e-12)
    B = B / (np.linalg.norm(B, "fro") + 1e-12)

    # Singular values of cross-covariance = how well they align after rotation
    _, s, _ = np.linalg.svd(A.T @ B, full_matrices=False)
    return float(np.clip(s.sum(), 0.0, 1.0))


def pairwise_metrics_across_depth(
    descriptors_a: np.ndarray,  # [n_layers_a+1, n_tokens_a, n_anchors]
    descriptors_b: np.ndarray,  # [n_layers_b+1, n_tokens_b, n_anchors]
    n_fractions: int = 7,
) -> dict:
    """
    Compute CKA and Procrustes at evenly-spaced depth fractions (0..1),
    mapping each fraction to the nearest layer in each model.

    Different models may have different layer counts and tokenizations.
    We compare at min(n_tokens_a, n_tokens_b) tokens.

    Returns:
        {
            "fractions": [float],         # [n_fractions]
            "cka":        [float],         # [n_fractions]
            "procrustes": [float],         # [n_fractions]
        }
    """
    n_steps_a = descriptors_a.shape[0]
    n_steps_b = descriptors_b.shape[0]
    min_tokens = min(descriptors_a.shape[1], descriptors_b.shape[1])

    fractions = [i / (n_fractions - 1) for i in range(n_fractions)]
    cka_scores = []
    proc_scores = []

    for frac in fractions:
        la = round(frac * (n_steps_a - 1))
        lb = round(frac * (n_steps_b - 1))

        A = descriptors_a[la, :min_tokens, :]  # [min_tokens, n_anchors]
        B = descriptors_b[lb, :min_tokens, :]

        cka_scores.append(linear_cka(A, B))
        proc_scores.append(procrustes_similarity(A, B))

    return {
        "fractions": fractions,
        "cka": cka_scores,
        "procrustes": proc_scores,
    }
