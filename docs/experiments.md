# Experiments

Small, honest studies run with the Semantic Geometry Explorer's own backend.
These are **demo-scale** (a 70M–124M parameter model, a handful of texts) — fun
probes of representation geometry, not publishable claims. Each reports its
controls and confounds explicitly.

Reproduce any of these with the scripts in [`backend/scripts/`](../backend/scripts)
against a running backend (`uvicorn app.main:app --port 8000`).

---

## 1. Where does a transformer disambiguate a word?

**Question.** A polysemous word like *bank* has two senses. At which layer does
the model split them apart?

**Method** ([`exp_disambiguation_v2.py`](../backend/scripts/exp_disambiguation_v2.py)).
Sentence: *"The river bank was steep but the money bank was closed."* We track
each of the two ` bank` tokens and, at every layer, measure its **sense
preference** — its cosine distance to a *money* anchor (`"money finance loans
deposit"`) minus its distance to a *river* anchor (`"river water shore
stream"`), using the same per-layer centering the app applies everywhere. The
disambiguation layer is where the two occurrences' preferences pull apart
(river-bank leans river, money-bank leans money).

**Result.**

| Model | Layers | Divergence layer | Depth | Max separation |
|-------|:------:|:----------------:|:-----:|:--------------:|
| Pythia-70M | 6 | **3** | 0.50 | 0.28 |
| GPT-2 small | 12 | **12** | 1.00 | 0.39 |

Pythia-70M commits to the two senses in the **middle** of the stack and holds
it; GPT-2 keeps the two `bank`s nearly identical until the **final** layer, then
splits them hard (separation +0.39) — consistent with late-layer features being
organized around next-token prediction.

**Confound noted.** A naive version ([`exp_disambiguation.py`](../backend/scripts/exp_disambiguation.py))
just measured raw cosine distance between the two `bank` tokens. That signal is
contaminated at late layers by position / next-token features (the two `bank`s
sit in different sentence positions), so it *looks* like late disambiguation for
the wrong reason. The sense-anchor version above controls for that by measuring
*direction toward a sense*, not raw distance. Worth remembering: "the vectors
moved apart" is not the same as "the meanings separated."

---

## 2. Can activation topology tell AI text from human text?

**tl;dr — a cautionary tale.** In one specific setting the pipeline separates AI
from human text *perfectly*. Then every control I added tore that result down.
It's the most useful experiment in this file precisely because it *failed* the
right way.

**Question.** A text's final-layer token activations form a point cloud. Persistent
homology fingerprints that cloud's shape ([topology.py](../backend/app/topology.py)).
Do AI and human texts have distinguishable fingerprints?

**Corpus** ([`ai_topology_corpus.json`](../backend/scripts/ai_topology_corpus.json),
built by the `ai-topology-corpus` agent workflow). 4 verbatim public-domain human
passages (Emerson, Darwin, Thoreau, Twain) vs 4 LLM-written essays on the **same
four topics**, each trimmed to **exactly 316 tokens** so length can't drive the
result. Analysis: [`exp_ai_topology.py`](../backend/scripts/exp_ai_topology.py),
sweep [`..._robust.py`](../backend/scripts/exp_ai_topology_robust.py), decomposition
[`..._verify.py`](../backend/scripts/exp_ai_topology_verify.py).

**The headline result (final layer, Wasserstein):** within-group distances
(human 4.05, AI 5.32) < cross-group (6.90); leave-one-out nearest-neighbour =
**8/8 (100%)**; exact permutation p = 0.029. It even beats a 4-feature stylometry
baseline (75%).

**Then it fell apart under three controls:**

1. **It's not robust.** Swept across layers and metrics, the separation appears
   in **1 of 5 settings**. At intermediate/early layers NN drops to 38% (*below*
   chance); the bottleneck metric gives 75%. And p = 0.029 is the *floor* of a
   35-split test (1/35) — after correcting for 5 configs it's not significant.

2. **It's not topology.** Splitting the diagram by homology dimension: H0
   (connected components) alone gives 100%, but **H1 — the actual loops — gives
   only 75%**. H0 here is just the distribution of token-merge distances, i.e.
   **overall cloud spread**. A single scalar (mean merge distance) reproduces the
   full 100% separation and correlates r = **0.98** with the whole Wasserstein
   matrix. Persistent homology added nothing over "how spread out is the cloud."

3. **It's confounded.** That one scalar cleanly separates the groups (human ≤
   0.328 < AI ≥ 0.344, zero overlap) — but human = 1850s literary prose, AI =
   2020s expository prose. Their sentence-length distributions barely touch
   (human 16–52 words, AI 15–19), and the topology-distance matrix correlates
   r = 0.69 with pure surface stylometry. The pipeline is almost certainly
   reading **era/register, not authorship**.

**Verdict:** *likely confounded* (3 of 4 adversarial reviewers; the statistician
said *spurious*). A genuine, deterministic in-sample separation that, on
inspection, is a single activation-scale number tracking 19th-century-vs-modern
register — not an AI detector.

**What would make it real.** Hold register constant (modern human expository
essays vs AI, sentence-length matched), pre-register the single final-layer +
Wasserstein config, include the one-scalar baseline as a control, and grow to
≥7 texts per group (so the permutation floor drops below 0.01). Expected outcome:
it collapses to chance. If it *survived* that, it would be worth a second look.

*Methodological note:* the numbers above were computed, not assumed. The
adversarial panel correctly predicted the H0/scalar deflation but estimated the
scalar correlation at 0.996 and H1 at "chance"; direct computation gave 0.98 and
75%. Verify, don't trust — even your own critics.
