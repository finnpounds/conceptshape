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
    # Center columns first (as CKA does). Anchor-distance descriptors are all
    # positive and near 1, so without centering both matrices are dominated by
    # the same constant component and the score is ≈1.0 for any two models.
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
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


def cka_permutation_null(
    descriptors_a: np.ndarray,  # [n_layers_a+1, n_tokens_a, n_anchors]
    descriptors_b: np.ndarray,  # [n_layers_b+1, n_tokens_b, n_anchors]
    n_fractions: int = 7,
    n_perm: int = 200,
    seed: int = 0,
    quantile: float = 0.95,
) -> list[float]:
    """
    Chance level for the CKA curve from `pairwise_metrics_across_depth`.

    At each depth fraction, shuffle the token rows of B (destroying the token
    correspondence that CKA is supposed to detect) and take the `quantile` of
    the resulting CKA distribution. A real CKA at or below this value is not
    distinguishable from chance. With a handful of tokens the bar is high
    (≈0.5 at the 95th percentile for 5 tokens), which is why a single short
    sentence cannot support claims about cross-model convergence — pool tokens
    from several sentences before reading anything into the curve.
    """
    rng = np.random.default_rng(seed)
    n_steps_a, n_steps_b = descriptors_a.shape[0], descriptors_b.shape[0]
    min_tokens = min(descriptors_a.shape[1], descriptors_b.shape[1])

    null = []
    for i in range(n_fractions):
        frac = i / (n_fractions - 1)
        la = round(frac * (n_steps_a - 1))
        lb = round(frac * (n_steps_b - 1))
        A = descriptors_a[la, :min_tokens, :]
        B = descriptors_b[lb, :min_tokens, :]
        vals = [linear_cka(A, B[rng.permutation(min_tokens)]) for _ in range(n_perm)]
        null.append(float(np.quantile(vals, quantile)))
    return null


def word_groups(token_strs: list[str], has_bos: bool = True) -> list[list[int]]:
    """
    Group content-token indices into whitespace-delimited words.

    Different tokenizers split the same text into different tokens, so
    comparing descriptor matrices row-by-row (token position) misaligns them
    as soon as one tokenizer splits a word the other keeps whole. Decoded token
    strings carry their leading whitespace (" think"), so a token that starts
    with whitespace starts a new word and everything else (sub-word pieces,
    trailing punctuation) attaches to the current word. Both models then yield
    one row per word, in the same order.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for i in range(1 if has_bos else 0, len(token_strs)):
        starts_word = token_strs[i][:1].isspace() or not current
        if starts_word and current:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def word_level(descriptors: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    """Average token-level descriptors [L+1, n_tokens, K] within each word → [L+1, n_words, K]."""
    return np.stack([descriptors[:, g, :].mean(axis=1) for g in groups], axis=1)
