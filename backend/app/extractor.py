"""
extractor.py — Extract residual stream trajectories and attention patterns
from transformer models using TransformerLens.

Core module for Semantic Geometry Explorer.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Optional


class ModelExtractor:
    """Extracts internal representations from a hooked transformer model."""

    def __init__(self, model_name: str = "pythia-70m", device: str = "auto"):
        if device == "auto":
            if torch.backends.mps.is_available():
                # MPS can be flaky with some ops; fall back to CPU for safety
                device = "cpu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        print(f"Loading {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.model.eval()
        print(f"Loaded. Layers: {self.model.cfg.n_layers}, "
              f"d_model: {self.model.cfg.d_model}, "
              f"heads: {self.model.cfg.n_heads}")

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.model.cfg.d_model

    @property
    def n_heads(self) -> int:
        return self.model.cfg.n_heads

    def extract(self, text: str) -> dict:
        """
        Run a forward pass and extract:
        - residual stream activations at each layer (including embedding layer)
        - attention patterns at each layer

        Returns:
            {
                "tokens": list[str],        # decoded token strings
                "token_ids": list[int],
                "residual_stream": np.ndarray,  # shape: [n_layers+1, n_tokens, d_model]
                    # index 0 = post-embedding, index i = post-layer-(i-1)
                "attention": np.ndarray,     # shape: [n_layers, n_heads, n_tokens, n_tokens]
            }
        """
        # Tokenize
        tokens = self.model.to_tokens(text)  # [1, seq_len]
        token_strs = self.model.to_str_tokens(text)

        # Run with cache — get residual stream and attention
        # names_filter limits what we cache to avoid memory issues
        _, cache = self.model.run_with_cache(
            tokens,
            names_filter=lambda name: (
                "hook_resid_post" in name  # residual stream after each layer
                or "hook_resid_pre" in name  # residual stream before first layer
                or "hook_pattern" in name    # attention patterns
                or name == "hook_embed"      # embedding layer output
            ),
        )

        n_layers = self.model.cfg.n_layers
        seq_len = tokens.shape[1]

        # Collect residual stream: embedding + each layer's output
        # Shape: [n_layers+1, seq_len, d_model]
        residual_layers = []

        # Layer 0: post-embedding (before any transformer blocks)
        embed_key = "hook_embed"
        if embed_key in cache:
            residual_layers.append(cache[embed_key][0].detach().cpu().numpy())
        else:
            # Fallback: use resid_pre of first block
            residual_layers.append(
                cache["blocks.0.hook_resid_pre"][0].detach().cpu().numpy()
            )

        # Layers 1..n_layers: post each transformer block
        for layer in range(n_layers):
            key = f"blocks.{layer}.hook_resid_post"
            residual_layers.append(cache[key][0].detach().cpu().numpy())

        residual_stream = np.stack(residual_layers, axis=0)
        # Shape: [n_layers+1, seq_len, d_model]

        # Collect attention patterns
        # Shape: [n_layers, n_heads, seq_len, seq_len]
        attention_layers = []
        for layer in range(n_layers):
            key = f"blocks.{layer}.attn.hook_pattern"
            attention_layers.append(cache[key][0].detach().cpu().numpy())

        attention = np.stack(attention_layers, axis=0)

        return {
            "tokens": token_strs,
            "token_ids": tokens[0].tolist(),
            "residual_stream": residual_stream,
            "attention": attention,
        }

    def logit_lens(self, residual_stream: np.ndarray, k: int = 3) -> list:
        """
        Logit lens (nostalgebraist, 2020): decode the residual stream at EVERY
        layer through the model's final layernorm + unembedding, revealing what
        the model would predict as the next token "as if" it stopped reading at
        that layer. At the final layer this reproduces the model's real
        next-token distribution; at earlier layers you watch the guess resolve.

        Args:
            residual_stream: [n_layers+1, n_tokens, d_model] (from extract()).
            k: top-k predictions per (layer, position).

        Returns:
            nested list [n_layers+1][n_tokens][k] of {"token": str, "prob": float}.
            Entry [L][i] is the model's top-k next-token guesses decoded from
            layer L at position i.
        """
        rs = torch.tensor(residual_stream, dtype=torch.float32, device=self.device)
        out: list = []
        with torch.no_grad():
            for layer in range(rs.shape[0]):
                normed = self.model.ln_final(rs[layer])   # [T, d_model]
                logits = self.model.unembed(normed)       # [T, d_vocab]
                probs = torch.softmax(logits, dim=-1)
                topv, topi = probs.topk(k, dim=-1)        # [T, k]
                layer_preds = []
                for t in range(rs.shape[1]):
                    layer_preds.append([
                        {
                            "token": self.model.tokenizer.decode([int(topi[t, j])]),
                            "prob": float(topv[t, j]),
                        }
                        for j in range(k)
                    ])
                out.append(layer_preds)
        return out

    def extract_concept_stream(self, text: str) -> np.ndarray:
        """
        Get mean-pooled embedding at every layer, excluding the BOS token.

        BOS (<|endoftext|>) dominates activations at all layers via the attention
        sink pattern — including it in the mean collapses all concept embeddings
        toward the same direction, making cosine distances between concepts nearly
        identical and causing anchor-relative / probe views to collapse to one point.

        Returns:
            np.ndarray of shape [n_layers+1, d_model]
        """
        result = self.extract(text)
        stream = result["residual_stream"]  # [n_layers+1, n_tokens, d_model]
        # Skip index 0 (BOS) — pool only over content tokens
        if stream.shape[1] > 1:
            stream = stream[:, 1:, :]
        return stream.mean(axis=1)  # [n_layers+1, d_model]

    def extract_text_cloud(
        self,
        text: str,
        layer: int = -1,
        chunk_size: int = 512,
        stride: int = 256,
    ) -> np.ndarray:
        """
        Process text of arbitrary length by chunking with overlap, extracting
        residual stream activations at the target layer, and returning the union
        of all token activations.

        For overlapping regions, keeps activations from the window where each
        token is closest to the center (reduces edge effects).

        Args:
            text:       input text (can be long — book-length)
            layer:      which layer to extract (-1 = last)
            chunk_size: max tokens per forward pass
            stride:     step size between windows (< chunk_size = overlap)

        Returns:
            np.ndarray of shape [n_total_tokens, d_model]
        """
        # Tokenize full text once
        token_ids = self.model.to_tokens(text)[0]  # [total_seq]
        total_len = token_ids.shape[0]

        if total_len <= chunk_size:
            # Short enough to process in one pass
            result = self.extract(text)
            layer_idx = layer if layer >= 0 else self.n_layers + 1 + layer
            return result["residual_stream"][layer_idx]

        # Map each token to the window that covers it most centrally
        # assigned_window[i] = window index responsible for token i
        assigned_window = np.full(total_len, -1, dtype=int)
        window_starts = list(range(0, total_len - chunk_size + 1, stride))
        if not window_starts or window_starts[-1] + chunk_size < total_len:
            window_starts.append(max(0, total_len - chunk_size))

        for wi, start in enumerate(window_starts):
            end = min(start + chunk_size, total_len)
            center = (start + end) / 2
            for pos in range(start, end):
                dist_to_center = abs(pos - center)
                if assigned_window[pos] < 0:
                    assigned_window[pos] = wi
                else:
                    prev_start = window_starts[assigned_window[pos]]
                    prev_end = min(prev_start + chunk_size, total_len)
                    prev_center = (prev_start + prev_end) / 2
                    if dist_to_center < abs(pos - prev_center):
                        assigned_window[pos] = wi

        # Extract activations per window and collect
        layer_idx = layer if layer >= 0 else self.n_layers + 1 + layer
        all_activations: list[np.ndarray] = [None] * total_len  # type: ignore

        for wi, start in enumerate(window_starts):
            end = min(start + chunk_size, total_len)
            chunk_ids = token_ids[start:end].unsqueeze(0)
            _, cache = self.model.run_with_cache(
                chunk_ids,
                names_filter=lambda name: (
                    "hook_resid_post" in name
                    or name == "hook_embed"
                    or "hook_resid_pre" in name
                ),
            )
            # Collect residual at target layer
            if layer_idx == 0:
                key = "hook_embed"
                if key not in cache:
                    key = "blocks.0.hook_resid_pre"
                acts = cache[key][0].detach().cpu().numpy()  # [chunk_len, d_model]
            else:
                key = f"blocks.{layer_idx - 1}.hook_resid_post"
                acts = cache[key][0].detach().cpu().numpy()

            # Assign activations to tokens where this window is responsible
            for local_pos in range(end - start):
                global_pos = start + local_pos
                if assigned_window[global_pos] == wi:
                    all_activations[global_pos] = acts[local_pos]

        # Stack (all positions should be filled)
        filled = [a for a in all_activations if a is not None]
        return np.stack(filled, axis=0)  # [n_tokens, d_model]

    def extract_concept_embedding(
        self, text: str, layer: Optional[int] = None
    ) -> np.ndarray:
        """
        Get a single vector representing a concept (mean-pooled across tokens).
        Used for anchor-relative computations (Milestone 3).

        Args:
            text: concept string (e.g., "justice", "love")
            layer: which layer to extract from (default: last layer)

        Returns:
            np.ndarray of shape [d_model]
        """
        result = self.extract(text)
        if layer is None:
            layer = -1  # last layer
        # Mean-pool across tokens (excluding BOS if present)
        activations = result["residual_stream"][layer]  # [seq_len, d_model]
        return activations.mean(axis=0)
