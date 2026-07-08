"""
exp_ai_topology.py — Can persistent homology of a language model's activation
clouds separate AI-written text from human-written text?

Design (controls a skeptic would demand):
  * MATCHED TOPICS: 4 themes, each with one human (public-domain) and one AI text,
    so topic cannot explain any human/AI separation.
  * LENGTH CONTROL: every text is trimmed to the SAME token count before analysis,
    so point-cloud size (which drives the number of topological features) is constant.
  * EXACT PERMUTATION TEST: with n=8 there are only C(8,4)/2 = 35 balanced label
    splits — we enumerate all of them for an exact p-value instead of sampling.
  * CONFOUND BASELINE: a nearest-neighbour classifier using ONLY surface features
    (type-token ratio, mean word length, punctuation density, mean sentence length).
    If topology beats this, it captured structure beyond trivial statistics.
  * DETERMINISM: texts are < max_points tokens, so TopologyAnalyzer does NO random
    subsampling — the persistence diagrams are reproducible.

Honest confound NOT controlled: era/register. The human texts are 19th-century
literary prose; the AI texts are modern. Any separation conflates "human vs AI"
with "1850s vs 2020s". Reported explicitly.

Usage:
    # corpus produced by the ai-topology-corpus workflow, saved as JSON:
    #   [{ "id","topic","author_type","text", ... }, ...]
    python scripts/exp_ai_topology.py scripts/ai_topology_corpus.json
"""

import itertools
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

API = "http://localhost:8000"
BACKEND_DIR = Path(__file__).resolve().parents[1]
OUT = BACKEND_DIR / "scripts" / "ai_topology_results.json"


def post(path, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---- surface (confound) features -------------------------------------------

def surface_features(text: str) -> dict:
    words = re.findall(r"[A-Za-z']+", text.lower())
    n = max(1, len(words))
    puncts = re.findall(r"[.,;:!?—\-()\"']", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "type_token_ratio": len(set(words)) / n,
        "mean_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "punct_density": len(puncts) / n,
        "mean_sentence_len_words": n / max(1, len(sentences)),
    }


# ---- classification / stats helpers ----------------------------------------

def loo_nn_accuracy(dist: np.ndarray, labels: list[str]) -> float:
    """Leave-one-out nearest-neighbour accuracy on a precomputed distance matrix."""
    n = len(labels)
    correct = 0
    for i in range(n):
        order = [j for j in np.argsort(dist[i]) if j != i]
        if labels[order[0]] == labels[i]:
            correct += 1
    return correct / n


def separation_stat(dist: np.ndarray, mask_group: np.ndarray) -> float:
    """mean(cross-group dist) - mean(within-group dist). Larger = more separated."""
    n = len(mask_group)
    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if mask_group[i] == mask_group[j] else cross).append(dist[i, j])
    return float(np.mean(cross) - np.mean(within))


def exact_permutation_test(dist, true_labels, stat_fn):
    """Exact test over all balanced 4/4 relabellings (n=8). Returns (observed, p)."""
    n = len(true_labels)
    idx = list(range(n))
    uniq = sorted(set(true_labels))
    assert len(uniq) == 2
    true_bool = np.array([1 if l == uniq[1] else 0 for l in true_labels])
    observed = stat_fn(dist, true_bool)

    group_size = int(true_bool.sum())
    seen = set()
    stats = []
    for combo in itertools.combinations(idx, group_size):
        b = np.zeros(n, dtype=int)
        for c in combo:
            b[c] = 1
        # canonicalize so a split and its complement aren't double counted
        key = tuple(b) if b[0] == 1 else tuple(1 - b)
        if key in seen:
            continue
        seen.add(key)
        stats.append(stat_fn(dist, b))
    stats = np.array(stats)
    p = float(np.mean(stats >= observed))
    return observed, p, len(stats), stats.tolist()


def main():
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BACKEND_DIR / "scripts" / "ai_topology_corpus.json"
    corpus = json.loads(corpus_path.read_text())
    print(f"loaded {len(corpus)} texts from {corpus_path}")

    # --- length control: trim all texts to a common token count ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    ids = [tok(c["text"])["input_ids"] for c in corpus]
    raw_tok_counts = [len(x) for x in ids]
    target = min(raw_tok_counts)
    print(f"raw token counts: {raw_tok_counts} -> trimming all to {target}")
    for c, x in zip(corpus, ids):
        c["_trimmed"] = tok.decode(x[:target])

    texts = [c["_trimmed"] for c in corpus]
    labels = [c["author_type"] for c in corpus]
    labels_bool = np.array([1 if l == "ai" else 0 for l in labels])
    names = [c["id"] for c in corpus]

    # --- topology distances via the live backend ---
    resp = post("/compare-shapes", {
        "texts": texts, "labels": names,
        "layer": -1, "max_points": 2000, "metric": "wasserstein",
    })
    dist = np.array(resp["distance_matrix"])
    shapes = resp["shapes"]

    # --- topology-based classification + exact permutation test ---
    topo_acc = loo_nn_accuracy(dist, labels)
    sep_obs, sep_p, n_perm, _ = exact_permutation_test(dist, labels, separation_stat)

    # null distribution of LOO-NN accuracy over the same 35 splits
    nn_null = []
    for combo in itertools.combinations(range(len(labels)), int(labels_bool.sum())):
        b = np.zeros(len(labels), dtype=int)
        for c in combo:
            b[c] = 1
        nn_null.append(loo_nn_accuracy(dist, ["ai" if v else "human" for v in b]))
    nn_p = float(np.mean(np.array(nn_null) >= topo_acc))

    # --- confound baseline: NN on surface features only ---
    feats = [surface_features(t) for t in texts]
    fkeys = list(feats[0].keys())
    F = np.array([[f[k] for k in fkeys] for f in feats])
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    fdist = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            fdist[i, j] = np.linalg.norm(Fz[i] - Fz[j])
    confound_acc = loo_nn_accuracy(fdist, labels)

    # correlation between topology distances and confound distances (Mantel-ish)
    iu = np.triu_indices(len(texts), k=1)
    mantel_r = float(np.corrcoef(dist[iu], fdist[iu])[0, 1])

    # group distance means
    def group_means(d):
        within, cross = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                (within if labels[i] == labels[j] else cross).append(d[i, j])
        return float(np.mean(within)), float(np.mean(cross))
    win_h = [d for i in range(len(labels)) for j in range(i+1, len(labels))
             if labels[i]==labels[j]=="human" for d in [dist[i,j]]]
    win_a = [d for i in range(len(labels)) for j in range(i+1, len(labels))
             if labels[i]==labels[j]=="ai" for d in [dist[i,j]]]
    within_mean, cross_mean = group_means(dist)

    results = {
        "n_texts": len(corpus),
        "target_tokens": target,
        "raw_token_counts": raw_tok_counts,
        "ids": names,
        "labels": labels,
        "topics": [c["topic"] for c in corpus],
        "distance_matrix": dist.tolist(),
        "shapes_summary": [
            {"id": names[i], "author_type": labels[i], "topic": corpus[i]["topic"],
             "n_components": s["n_components"], "n_loops": s["n_loops"],
             "n_points": s["n_points_sampled"]}
            for i, s in enumerate(shapes)
        ],
        "surface_features": [dict(id=names[i], author_type=labels[i], **feats[i])
                             for i in range(len(texts))],
        "results": {
            "within_human_mean_dist": float(np.mean(win_h)),
            "within_ai_mean_dist": float(np.mean(win_a)),
            "within_all_mean_dist": within_mean,
            "cross_mean_dist": cross_mean,
            "separation_stat": sep_obs,
            "separation_p_exact": sep_p,
            "n_permutations": n_perm,
            "topology_loo_nn_accuracy": topo_acc,
            "topology_nn_p_exact": nn_p,
            "confound_loo_nn_accuracy": confound_acc,
            "topology_vs_confound_mantel_r": mantel_r,
        },
    }
    OUT.write_text(json.dumps(results, indent=2))

    r = results["results"]
    print("\n================ AI-vs-HUMAN TOPOLOGY RESULTS ================")
    print(f"texts: {len(corpus)} ({labels.count('human')} human / {labels.count('ai')} ai), "
          f"trimmed to {target} tokens each")
    print(f"within-human mean dist : {r['within_human_mean_dist']:.4f}")
    print(f"within-AI    mean dist : {r['within_ai_mean_dist']:.4f}")
    print(f"cross (H-AI) mean dist : {r['cross_mean_dist']:.4f}")
    print(f"separation (cross-within): {r['separation_stat']:+.4f}  "
          f"exact p = {r['separation_p_exact']:.3f} (over {r['n_permutations']} splits)")
    print(f"TOPOLOGY  LOO-NN accuracy: {r['topology_loo_nn_accuracy']*100:.0f}%  "
          f"(exact p = {r['topology_nn_p_exact']:.3f})")
    print(f"CONFOUND  LOO-NN accuracy: {r['confound_loo_nn_accuracy']*100:.0f}%  "
          f"(surface features only)")
    print(f"topology<->confound Mantel r: {r['topology_vs_confound_mantel_r']:+.3f}")
    print("H0/H1 per text:")
    for s in results["shapes_summary"]:
        print(f"  {s['id']:22s} {s['author_type']:5s} H0={s['n_components']:3d} H1={s['n_loops']:3d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
