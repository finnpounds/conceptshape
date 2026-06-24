"""
poc_extract.py — Proof-of-concept: extract residual stream from Pythia-70M,
project to 3D, and print/save results.

Usage:
    python scripts/poc_extract.py "I think therefore I am"
    python scripts/poc_extract.py --save output.json "The cat sat on the mat"

Run this first to validate the pipeline before wiring up the API.
"""

import sys
import json
import argparse
import numpy as np

# Add parent dir to path so we can import app modules
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from app.extractor import ModelExtractor
from app.projector import Projector


def main():
    parser = argparse.ArgumentParser(description="Extract semantic trajectories")
    parser.add_argument("text", type=str, help="Input text to analyze")
    parser.add_argument("--model", type=str, default="pythia-70m", help="Model name")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "umap"])
    parser.add_argument("--save", type=str, default=None, help="Save JSON to file")
    args = parser.parse_args()

    # Extract
    extractor = ModelExtractor(model_name=args.model)
    raw = extractor.extract(args.text)

    print(f"\n{'='*60}")
    print(f"Input: {args.text!r}")
    print(f"Tokens ({len(raw['tokens'])}): {raw['tokens']}")
    print(f"Residual stream shape: {raw['residual_stream'].shape}")
    print(f"  → [{extractor.n_layers + 1} layer steps, "
          f"{len(raw['tokens'])} tokens, {extractor.d_model} dims]")
    print(f"Attention shape: {raw['attention'].shape}")
    print(f"  → [{extractor.n_layers} layers, {extractor.n_heads} heads, "
          f"{len(raw['tokens'])} tokens, {len(raw['tokens'])} tokens]")

    # Project
    projector = Projector(method=args.method)
    projection = projector.project_trajectories(raw["residual_stream"])
    positions = projection["positions"]

    print(f"\n{'='*60}")
    print(f"Projection method: {args.method}")
    if projection["explained_variance"]:
        ev = projection["explained_variance"]
        print(f"Explained variance: {ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f} "
              f"(total: {sum(ev):.3f})")

    print(f"\nToken trajectories (3D positions at each layer step):")
    print(f"{'Token':<15} {'Embed':<25} {'Final Layer':<25} {'Distance':<10}")
    print("-" * 75)
    for t, token in enumerate(raw["tokens"]):
        start = positions[0, t]
        end = positions[-1, t]
        dist = np.linalg.norm(end - start)
        start_str = f"({start[0]:+.3f}, {start[1]:+.3f}, {start[2]:+.3f})"
        end_str = f"({end[0]:+.3f}, {end[1]:+.3f}, {end[2]:+.3f})"
        print(f"{token!r:<15} {start_str:<25} {end_str:<25} {dist:<10.3f}")

    # Attention summary: strongest attention edges at last layer
    last_attn = raw["attention"][-1]  # [n_heads, seq, seq]
    avg_attn = last_attn.mean(axis=0)  # [seq, seq] — averaged across heads
    print(f"\nStrongest attention edges (last layer, head-averaged):")
    tokens_list = raw["tokens"]
    flat_indices = np.argsort(avg_attn.ravel())[::-1]
    for rank in range(min(10, len(flat_indices))):
        idx = flat_indices[rank]
        from_tok = idx // len(tokens_list)
        to_tok = idx % len(tokens_list)
        weight = avg_attn[from_tok, to_tok]
        print(f"  {tokens_list[from_tok]!r:>12} → {tokens_list[to_tok]!r:<12} "
              f"weight: {weight:.3f}")

    # Save if requested
    if args.save:
        output = {
            "text": args.text,
            "model": args.model,
            "tokens": raw["tokens"],
            "n_layers": extractor.n_layers,
            "projection_method": args.method,
            "explained_variance": projection["explained_variance"],
            "trajectories": [
                {
                    "token": raw["tokens"][t],
                    "positions": positions[:, t, :].tolist(),
                }
                for t in range(len(raw["tokens"]))
            ],
        }
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
