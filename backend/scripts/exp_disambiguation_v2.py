"""
exp_disambiguation_v2.py — Sense-anchor version of the disambiguation probe.

Instead of raw distance between the two "bank" tokens (confounded at late
layers by next-token / position features), measure each occurrence's SENSE
PREFERENCE at every layer:

    pref(bank_i, layer) = d(bank_i, money_anchor) - d(bank_i, river_anchor)

positive → closer to the river sense; negative → closer to the money sense.
Anchors are mean-pooled concept streams ("river water shore", "money finance
loans") extracted per layer — exactly the app's anchor-relative machinery.

Disambiguation depth = the layer where the two occurrences' preferences
split apart (river-bank goes positive, money-bank goes negative).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from app.extractor import ModelExtractor  # noqa: E402

SENTENCE = "The river bank was steep but the money bank was closed"
RIVER_ANCHOR = "river water shore stream"
MONEY_ANCHOR = "money finance loans deposit"


def main() -> None:
    results = {}
    for model_name in ["pythia-70m", "gpt2"]:
        ext = ModelExtractor(model_name=model_name, device="cpu")
        with torch.no_grad():
            raw = ext.extract(SENTENCE)
            river_stream = ext.extract_concept_stream(RIVER_ANCHOR)  # [L+1, D]
            money_stream = ext.extract_concept_stream(MONEY_ANCHOR)  # [L+1, D]

        tokens = raw["tokens"]
        idxs = [i for i, t in enumerate(tokens) if t.strip().lower() == "bank"]
        assert len(idxs) == 2, f"expected 2 banks, got {tokens}"
        rs = raw["residual_stream"]  # [L+1, T, D]
        n_steps = rs.shape[0]

        prefs = {0: [], 1: []}  # occurrence -> per-layer preference
        for layer in range(n_steps):
            vecs = rs[layer]
            mean = vecs.mean(axis=0)
            for occ, ti in enumerate(idxs):
                v = vecs[ti] - mean
                r = river_stream[layer] - mean
                m = money_stream[layer] - mean

                def cosd(a, b):
                    return 1.0 - float(
                        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                    )

                prefs[occ].append(cosd(v, m) - cosd(v, r))

        # separation = river-bank pref minus money-bank pref (should go positive)
        separation = [prefs[0][l] - prefs[1][l] for l in range(n_steps)]
        smax = max(separation)
        div_layer = (
            next(i for i, s in enumerate(separation) if s >= smax / 2)
            if smax > 0 else -1
        )
        n_layers = n_steps - 1
        results[model_name] = {
            "n_layers": n_layers,
            "river_bank_pref": [round(p, 4) for p in prefs[0]],
            "money_bank_pref": [round(p, 4) for p in prefs[1]],
            "separation": [round(s, 4) for s in separation],
            "max_separation": round(smax, 4),
            "divergence_layer": div_layer,
            "divergence_depth_frac": round(div_layer / n_layers, 3) if div_layer >= 0 else None,
        }
        del ext
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
