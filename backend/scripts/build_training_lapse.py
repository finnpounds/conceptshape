"""
build_training_lapse.py — precompute concept geometry across Pythia-70m
TRAINING CHECKPOINTS for the "watch a model learn" time-lapse.

For each checkpoint (log-spaced from step 0 to step 143000):
  1. Load pythia-70m at that training step (EleutherAI ships 154 revisions).
  2. Extract mean-pooled final-layer streams for a fixed concept vocabulary
     + 4 anchor concepts (same per-layer-centering recipe as probe mode).
  3. Compute anchor-relative descriptors [N_concepts, K_anchors].

Then jointly:
  4. Fit ONE PCA across all checkpoints' descriptors so positions are
     comparable across training time — movement means geometry change.
  5. CKA(step, final) — how early does the final concept geometry emerge?

Output: frontend/public/gallery/training.json (small, static, replayable).

Usage:
    cd backend && source .venv/bin/activate
    python scripts/build_training_lapse.py

Note: each checkpoint downloads ~150MB from HF on first run.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from app.alignment import linear_cka  # noqa: E402

OUT = BACKEND_DIR.parent / "frontend" / "public" / "gallery" / "training.json"

MODEL = "pythia-70m"
# Log-spaced training steps (Pythia revisions: 0,1,2,...,512, then every 1000)
STEPS = [0, 16, 128, 512, 1000, 4000, 16000, 64000, 143000]
# Pythia batch = 1024 seqs x 2048 tokens ≈ 2.097M tokens per step
TOKENS_PER_STEP = 2_097_152

ANCHORS = ["self", "other", "world", "logic"]
CONCEPTS: list[tuple[str, str]] = [
    # (label, category) — matches frontend CATEGORY_COLORS
    ("love", "emotions"), ("fear", "emotions"), ("joy", "emotions"),
    ("anger", "emotions"), ("hope", "emotions"), ("sadness", "emotions"),
    ("truth", "abstractions"), ("justice", "abstractions"), ("freedom", "abstractions"),
    ("time", "abstractions"), ("death", "abstractions"), ("power", "abstractions"),
    ("water", "nature"), ("fire", "nature"), ("sun", "nature"),
    ("moon", "nature"), ("tree", "nature"), ("ocean", "nature"),
    ("thought", "mind"), ("memory", "mind"), ("dream", "mind"), ("reason", "mind"),
    ("friend", "relations"), ("enemy", "relations"),
]


def concept_stream_final_layer(model, text: str) -> np.ndarray:
    """Mean-pooled final-layer representation, excluding BOS (attention sink)."""
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name.endswith(
            f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
        ),
    )
    acts = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"][0]
    acts = acts.detach().cpu().numpy()  # [seq, d_model]
    if acts.shape[0] > 1:
        acts = acts[1:]  # drop BOS
    return acts.mean(axis=0)  # [d_model]


def main() -> None:
    from transformer_lens import HookedTransformer

    labels = [c[0] for c in CONCEPTS]
    all_texts = labels + ANCHORS
    N, K = len(labels), len(ANCHORS)

    per_step_descriptors: list[np.ndarray] = []  # each [N, K]
    per_step_anchor_self: list[np.ndarray] = []  # each [K, K]

    for step in STEPS:
        print(f"=== checkpoint step {step} ===", flush=True)
        model = HookedTransformer.from_pretrained(
            MODEL, checkpoint_value=step, device="cpu"
        )
        model.eval()

        with torch.no_grad():
            vecs = np.stack(
                [concept_stream_final_layer(model, t) for t in all_texts], axis=0
            )  # [N+K, d_model]

        # Center by the mean over all concepts+anchors (same recipe as probe.py)
        mean = vecs.mean(axis=0)
        concept_vecs = vecs[:N] - mean
        anchor_vecs = vecs[N:] - mean

        per_step_descriptors.append(cosine_distances(concept_vecs, anchor_vecs))
        per_step_anchor_self.append(cosine_distances(anchor_vecs, anchor_vecs))

        del model
        gc.collect()

    S = len(STEPS)
    desc = np.stack(per_step_descriptors, axis=0)     # [S, N, K]
    anch = np.stack(per_step_anchor_self, axis=0)     # [S, K, K]

    # Joint PCA across ALL checkpoints so axes are shared over training time
    all_rows = np.concatenate(
        [desc.reshape(-1, K), anch.reshape(-1, K)], axis=0
    )
    n_comp = max(1, min(3, K, all_rows.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    pca.fit(all_rows)

    def project(rows: np.ndarray) -> np.ndarray:
        p = pca.transform(rows)
        if n_comp < 3:
            p = np.concatenate([p, np.zeros((p.shape[0], 3 - n_comp))], axis=1)
        return p

    concept_pos = project(desc.reshape(-1, K)).reshape(S, N, 3)
    anchor_pos = project(anch.reshape(-1, K)).reshape(S, K, 3)

    # Joint normalization to [-1, 1]
    allp = np.concatenate([concept_pos.reshape(-1, 3), anchor_pos.reshape(-1, 3)])
    max_abs = np.abs(allp).max(axis=0)
    max_abs = np.where(max_abs < 1e-8, 1.0, max_abs)
    concept_pos /= max_abs
    anchor_pos /= max_abs

    # CKA of each step's concept descriptor matrix vs the final step's
    cka_to_final = [float(linear_cka(desc[s], desc[-1])) for s in range(S)]

    ev = pca.explained_variance_ratio_.tolist()
    while len(ev) < 3:
        ev.append(0.0)

    payload = {
        "model": MODEL,
        "steps": STEPS,
        "tokens_seen": [int(s * TOKENS_PER_STEP) for s in STEPS],
        "anchors": ANCHORS,
        "concepts": [
            {
                "label": labels[i],
                "category": CONCEPTS[i][1],
                "positions": concept_pos[:, i, :].tolist(),  # [S][3]
            }
            for i in range(N)
        ],
        "anchor_markers": [
            {"label": ANCHORS[k], "positions": anchor_pos[:, k, :].tolist()}
            for k in range(K)
        ],
        "cka_to_final": cka_to_final,
        "explained_variance": ev,
    }

    OUT.write_text(json.dumps(payload))
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"CKA to final across steps: {[round(c, 3) for c in cka_to_final]}")


if __name__ == "__main__":
    main()
