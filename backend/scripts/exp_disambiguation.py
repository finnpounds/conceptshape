"""
exp_disambiguation.py — At which layer does a transformer disambiguate
a polysemous word?

Design:
  Ambiguous condition: "The river bank was steep but the money bank was closed"
    → the two " bank" tokens SHOULD diverge (different senses).
  Control condition:  "The first bank was closed and the second bank was closed"
    → the two " bank" tokens should stay close (same sense).

Metric: cosine distance between the two " bank" occurrences at each layer,
after per-layer centering across the sentence's tokens (same recipe the app
uses — removes the shared residual-norm component).

Runs on pythia-70m (6 layers) and gpt2 (12 layers); reports the normalized
depth at which the ambiguous/control gap opens.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from app.extractor import ModelExtractor  # noqa: E402

AMBIG = "The river bank was steep but the money bank was closed"
CONTROL = "The first bank was closed and the second bank was closed"


def bank_distance_per_layer(extractor: ModelExtractor, text: str) -> list[float]:
    """Cosine distance between the two ' bank' tokens at each layer."""
    raw = extractor.extract(text)
    tokens = raw["tokens"]
    idxs = [i for i, t in enumerate(tokens) if t.strip().lower() == "bank"]
    assert len(idxs) == 2, f"expected 2 'bank' tokens, got {idxs} in {tokens}"

    rs = raw["residual_stream"]  # [L+1, T, D]
    dists = []
    for layer in range(rs.shape[0]):
        vecs = rs[layer]
        centered = vecs - vecs.mean(axis=0)  # per-layer centering
        a, b = centered[idxs[0]], centered[idxs[1]]
        cos = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        )
        dists.append(1.0 - cos)
    return dists


def main() -> None:
    results = {}
    for model_name in ["pythia-70m", "gpt2"]:
        ext = ModelExtractor(model_name=model_name, device="cpu")
        with torch.no_grad():
            ambig = bank_distance_per_layer(ext, AMBIG)
            control = bank_distance_per_layer(ext, CONTROL)
        gap = [a - c for a, c in zip(ambig, control)]
        # Divergence layer: first layer where gap exceeds half its max
        gmax = max(gap)
        div_layer = next(i for i, g in enumerate(gap) if g >= gmax / 2)
        n_layers = len(ambig) - 1
        results[model_name] = {
            "n_layers": n_layers,
            "ambiguous": [round(d, 4) for d in ambig],
            "control": [round(d, 4) for d in control],
            "gap": [round(g, 4) for g in gap],
            "max_gap": round(gmax, 4),
            "divergence_layer": div_layer,
            "divergence_depth_frac": round(div_layer / n_layers, 3),
        }
        del ext
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
