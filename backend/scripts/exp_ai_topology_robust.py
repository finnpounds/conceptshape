"""
exp_ai_topology_robust.py — robustness sweep for the AI-vs-human topology result.

Re-runs the separation analysis across transformer LAYERS and both diagram
METRICS, to check the headline result isn't an artifact of picking the final
layer + Wasserstein. Reuses helpers + corpus loading from exp_ai_topology.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_ai_topology import (  # noqa: E402
    loo_nn_accuracy, separation_stat, exact_permutation_test,
)

API = "http://localhost:8000"
BACKEND_DIR = Path(__file__).resolve().parents[1]
OUT = BACKEND_DIR / "scripts" / "ai_topology_robust.json"


def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    corpus = json.loads((BACKEND_DIR / "scripts" / "ai_topology_corpus.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ids = [tok(c["text"])["input_ids"] for c in corpus]
    target = min(len(x) for x in ids)
    texts = [tok.decode(x[:target]) for x in ids]
    names = [c["id"] for c in corpus]
    labels = [c["author_type"] for c in corpus]

    configs = [(-1, "wasserstein"), (3, "wasserstein"), (1, "wasserstein"),
               (-1, "bottleneck"), (3, "bottleneck")]
    rows = []
    for layer, metric in configs:
        resp = post("/compare-shapes", {
            "texts": texts, "labels": names,
            "layer": layer, "max_points": 2000, "metric": metric,
        })
        dist = np.array(resp["distance_matrix"])
        acc = loo_nn_accuracy(dist, labels)
        sep, p, n_perm, _ = exact_permutation_test(dist, labels, separation_stat)
        rows.append({"layer": layer, "metric": metric,
                     "separation": round(sep, 4), "p_exact": round(p, 4),
                     "loo_nn_accuracy": acc})
        print(f"layer={layer:>2} {metric:<11} sep={sep:+.3f} p={p:.3f} NN={acc*100:.0f}%")

    OUT.write_text(json.dumps({"target_tokens": target, "sweep": rows}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
