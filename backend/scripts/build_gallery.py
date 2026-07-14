"""
build_gallery.py — precompute a curated set of showcase examples and save the
raw API responses as static JSON for the deployed frontend.

The frontend replays these via the same zustand setters used for live calls, so
the public site is fully interactive (rotate / scrub layers / toggle attention)
with NO backend running. The live backend is only needed for custom input.

Usage:
    # with the API server already running on :8000
    python scripts/build_gallery.py
"""

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

API = "http://localhost:8000"
OUT = Path(__file__).resolve().parents[2] / "frontend" / "public" / "gallery"


def post(path: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(f"{API}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


# Each entry: how to call the backend + the human-facing framing the tour uses.
# `mode`/`subMode` tell the frontend loader which store setter to apply.
EXAMPLES = [
    {
        "id": "cogito",
        "mode": "explore",
        "subMode": "absolute",
        "title": "Cogito, ergo sum",
        "hook": "Watch a sentence fan out through 6 layers of a transformer.",
        "detail": (
            "Every word starts as a point. As it flows through the model's layers, "
            "it moves — picking up context from its neighbours. Each colored trail "
            "is one word's journey through the model's internal space. Drag to orbit; "
            "scrub the layer slider to replay the motion."
        ),
        "call": ("/analyze", {"text": "I think therefore I am", "method": "pca"}),
    },
    {
        "id": "river",
        "mode": "explore",
        "subMode": "absolute",
        "title": "Context bends meaning",
        "hook": "The same word lands in different places depending on its sentence.",
        "detail": (
            "Turn on attention edges to see which words are 'looking at' which other "
            "words at each layer — the wiring the model uses to mix meaning together."
        ),
        "call": ("/analyze", {"text": "The river bank was steep but the money bank was closed", "method": "pca"}),
    },
    {
        "id": "anchors",
        "mode": "explore",
        "subMode": "anchor",
        "title": "Words as distances to ideas",
        "hook": "Re-measure every word by how close it sits to reference concepts.",
        "detail": (
            "Instead of raw coordinates, each word is described by its distance to a few "
            "'anchor' concepts — here self, other, world, and logic. This turns an opaque "
            "512-dimensional space into something you can actually read."
        ),
        "call": ("/anchor-analyze", {"text": "I think therefore I am", "anchors": ["self", "other", "world", "logic"]}),
    },
    {
        "id": "convergence",
        "mode": "explore",
        "subMode": "compare",
        "title": "Do different models agree?",
        "hook": "Two models, same sentence — do they build the same geometry?",
        "detail": (
            "The Platonic Representation Hypothesis says different models converge on the "
            "same internal structure. Here Pythia-70M and GPT-2 process the same sentence; "
            "the CKA bars measure how aligned their geometry is at each depth."
        ),
        "call": ("/compare", {
            "text": "I think therefore I am",
            "models": ["pythia-70m", "gpt2"],
            "anchors": ["self", "other", "world", "logic"],
        }),
    },
    {
        "id": "concept-map",
        "mode": "explore",
        "subMode": "probe",
        "title": "A map of concepts",
        "hook": "Embed dozens of concepts in the same space and watch them sort themselves.",
        "detail": (
            "Each point is a concept (love, fear, truth, time...) placed by its distance to "
            "the anchors. Emotions cluster with emotions, abstractions with abstractions — "
            "structure the model was never explicitly taught."
        ),
        "call": ("/probe", {
            "anchors": ["self", "other", "world", "logic"],
            "probes": [
                "love", "fear", "joy", "anger", "sadness", "hope", "trust", "guilt", "pride", "shame",
                "truth", "justice", "beauty", "freedom", "time", "death", "life", "power", "knowledge", "meaning",
                "thought", "belief", "desire", "memory", "dream", "consciousness", "reason", "will", "perception",
            ],
            "model": "pythia-70m",
        }),
    },
    {
        "id": "text-shapes",
        "mode": "corpus",
        "subMode": "corpus",
        "title": "The shape of a text",
        "hook": "Different kinds of writing have measurably different geometric shape.",
        "detail": (
            "Each text becomes a cloud of token activations. Persistent homology measures the "
            "cloud's topology — its connected pieces and loops — producing a 'shape fingerprint'. "
            "The distance matrix shows philosophy, recipe, poetry, and physics are genuinely distinct shapes."
        ),
        "call": ("/compare-shapes", {
            "texts": [
                "I think, therefore I am. The mind is a substance that thinks. The body is a substance that is extended in space. These two substances are distinct and separate. Doubt is the beginning of wisdom.",
                "Preheat the oven to 375 degrees. Mix the flour, sugar, and butter in a bowl until crumbly. Add the eggs and vanilla. Stir until smooth. Pour the batter into a greased pan. Bake for 30 minutes until golden brown.",
                "Shall I compare thee to a summer's day? Thou art more lovely and more temperate. Rough winds do shake the darling buds of May, and summer's lease hath all too short a date.",
                "The force between two masses is proportional to the product of their masses and inversely proportional to the square of the distance between them. This law governs the motion of planets and the tides of the ocean.",
            ],
            "labels": ["Philosophy", "Recipe", "Poetry", "Physics"],
            "layer": -1,
            "max_points": 1500,
            "metric": "wasserstein",
        }),
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # sanity: backend up?
    try:
        h = get("/health")
        print(f"backend health: {h}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: backend not reachable at {API} ({e}). Start it first.", file=sys.stderr)
        sys.exit(1)

    manifest = {"models": get("/models"), "examples": []}

    for ex in EXAMPLES:
        path, payload = ex["call"]
        t0 = time.time()
        print(f"  -> {ex['id']:14s} POST {path} ...", end="", flush=True)
        try:
            resp = post(path, payload)
        except urllib.error.HTTPError as e:
            print(f" FAILED ({e.code}): {e.read().decode()[:200]}")
            continue
        dt = time.time() - t0
        fname = f"{ex['id']}.json"
        (OUT / fname).write_text(json.dumps(resp))
        size_kb = (OUT / fname).stat().st_size / 1024
        print(f" ok  {dt:4.1f}s  {size_kb:6.0f} KB")
        manifest["examples"].append({
            "id": ex["id"],
            "mode": ex["mode"],
            "subMode": ex["subMode"],
            "title": ex["title"],
            "hook": ex["hook"],
            "detail": ex["detail"],
            "params": payload,
            "file": fname,
        })

    # Attention-sink tour step — reuses river.json but with attention edges on
    # and the BOS token shown, so the "everything attends to token 0" sink is visible.
    if any(e["id"] == "river" for e in manifest["examples"]):
        river_idx = next(i for i, e in enumerate(manifest["examples"]) if e["id"] == "river")
        manifest["examples"].insert(river_idx + 1, {
            "id": "attention-sink",
            "mode": "explore",
            "subMode": "absolute",
            "title": "The attention sink",
            "hook": "Almost every token secretly stares at the very first one.",
            "detail": (
                "With attention edges on and the start token shown, a pattern jumps out: "
                "most tokens pour a chunk of their attention into position 0 — the invisible "
                "begin-of-sequence marker. Models use it as a no-op 'sink,' a place to dump "
                "attention when a head has nothing useful to point at. It's why the other "
                "views hide that token by default."
            ),
            "params": {"text": "The river bank was steep but the money bank was closed", "method": "pca"},
            "file": "river.json",
            "viewState": {"showAttention": True, "hideBOS": False, "currentLayer": 3},
        })

    # Static tour entries whose data files are produced by other scripts
    # (no API call needed here — just referenced in the manifest).
    if (OUT / "training.json").exists():
        manifest["examples"].append({
            "id": "training-lapse",
            "mode": "training",
            "subMode": "training",
            "title": "Watch a model learn",
            "hook": "300 billion tokens of training, compressed into ten seconds.",
            "detail": (
                "The same model, re-loaded at nine checkpoints across its training "
                "run. Every point is a concept placed by its distance to the anchors. "
                "At step 0 the geometry is random noise — press play and watch the "
                "map of meaning crystallize. The bars show how similar each "
                "checkpoint's geometry is to the finished model: it stays near-random "
                "for the first ~8 billion tokens, then snaps into place late."
            ),
            "params": {},
            "file": "training.json",
        })
    else:
        print("note: training.json not found — run build_training_lapse.py to include it")

    (OUT / "index.json").write_text(json.dumps(manifest, indent=2))
    total_kb = sum((OUT / e["file"]).stat().st_size for e in manifest["examples"]) / 1024
    print(f"\nwrote {len(manifest['examples'])} examples -> {OUT}  ({total_kb:.0f} KB total)")


if __name__ == "__main__":
    main()
