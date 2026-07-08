"""
exp_ai_topology_verify.py — verify the adversarial panel's deflation:
  "The separation is H0 (cloud scale), not H1 (real loops); a single scalar
   (mean token-merge distance) reproduces the whole result."

We pull the per-text persistence diagrams from /compare-shapes and, using
persim directly, decompose the Wasserstein separation into:
  * H0-only  (connected components = merge-distance distribution ~ cloud scale)
  * H1-only  (loops = genuine topological structure)
  * a 1-D "cloud spread" scalar = mean finite H0 death per text.

If H0 carries the signal and H1 is at chance, the "topology detects AI" story
collapses to "modern activation clouds are a bit more spread out at the final
layer" — and that scalar is itself era-confounded.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import persim

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_ai_topology import loo_nn_accuracy, separation_stat, exact_permutation_test  # noqa: E402

API = "http://localhost:8000"
BACKEND_DIR = Path(__file__).resolve().parents[1]
OUT = BACKEND_DIR / "scripts" / "ai_topology_verify.json"


def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def dim_diagram(pd, dim):
    pts = [[f["birth"], f["death"]] for f in pd if f["dimension"] == dim]
    return np.array(pts) if pts else np.empty((0, 2))


def wass_matrix(diagrams):
    n = len(diagrams)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a = diagrams[i] if len(diagrams[i]) else np.array([[0.0, 0.0]])
            b = diagrams[j] if len(diagrams[j]) else np.array([[0.0, 0.0]])
            d = float(persim.wasserstein(a, b))
            M[i, j] = M[j, i] = d
    return M


def report(name, dist, labels):
    acc = loo_nn_accuracy(dist, labels)
    sep, p, _, _ = exact_permutation_test(dist, labels, separation_stat)
    print(f"  {name:16s} sep={sep:+.4f}  p={p:.3f}  LOO-NN={acc*100:3.0f}%")
    return {"separation": sep, "p_exact": p, "loo_nn": acc}


def main():
    corpus = json.loads((BACKEND_DIR / "scripts" / "ai_topology_corpus.json").read_text())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ids = [tok(c["text"])["input_ids"] for c in corpus]
    target = min(len(x) for x in ids)
    texts = [tok.decode(x[:target]) for x in ids]
    names = [c["id"] for c in corpus]
    labels = [c["author_type"] for c in corpus]

    resp = post("/compare-shapes", {"texts": texts, "labels": names,
                                    "layer": -1, "max_points": 2000, "metric": "wasserstein"})
    combined = np.array(resp["distance_matrix"])
    diags = [s["persistence_diagram"] for s in resp["shapes"]]
    h0 = [dim_diagram(pd, 0) for pd in diags]
    h1 = [dim_diagram(pd, 1) for pd in diags]

    print("SEPARATION BY COMPONENT:")
    r_comb = report("combined (H0+H1)", combined, labels)
    r_h0 = report("H0 only (scale)", wass_matrix(h0), labels)
    r_h1 = report("H1 only (loops)", wass_matrix(h1), labels)

    # 1-D "cloud spread" scalar: mean finite H0 death per text
    spread = np.array([np.mean(d[:, 1]) if len(d) else 0.0 for d in h0])
    scalar_dist = np.abs(spread[:, None] - spread[None, :])
    print("\n1-D SCALAR BASELINE (mean H0 death = cloud spread):")
    r_scalar = report("cloud-spread 1D", scalar_dist, labels)

    # correlation of the scalar-distance with the full combined distance
    iu = np.triu_indices(len(names), k=1)
    r_corr = float(np.corrcoef(scalar_dist[iu], combined[iu])[0, 1])

    # group overlap on the raw scalar
    hs = [spread[i] for i in range(len(labels)) if labels[i] == "human"]
    as_ = [spread[i] for i in range(len(labels)) if labels[i] == "ai"]
    overlap = max(hs) < min(as_) or max(as_) < min(hs)

    print(f"\nscalar-dist vs combined-dist correlation r = {r_corr:.3f}")
    print(f"human cloud-spread range [{min(hs):.3f},{max(hs):.3f}]  "
          f"ai [{min(as_):.3f},{max(as_):.3f}]  separable-by-1-number: {overlap}")

    OUT.write_text(json.dumps({
        "combined": r_comb, "h0_only": r_h0, "h1_only": r_h1,
        "scalar_1d": r_scalar, "scalar_vs_combined_r": r_corr,
        "spread_by_text": {names[i]: float(spread[i]) for i in range(len(names))},
        "spread_separable_by_one_number": bool(overlap),
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
