"""
extractor.py — Extract residual stream trajectories and attention patterns
from transformer models using TransformerLens.

Core module for Semantic Geometry Explorer.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer


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

    @property
    def prepends_bos(self) -> bool:
        """
        Whether to_tokens() puts a BOS token at position 0. That token is the
        attention sink: its activations are an extreme outlier (norm up to ~10x
        any content token at mid layers), so every centering / PCA fit in this
        codebase excludes it. Callers get the flag as `has_bos` in extract().
        """
        return bool(getattr(self.model.cfg, "default_prepend_bos", True))

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
                "has_bos": bool,             # position 0 is the BOS attention sink
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
            "has_bos": self.prepends_bos,
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

    def steer(
        self,
        text: str,
        positive: str,
        negative: str,
        layer: int,
        strength: float,
        n_generate: int = 14,
    ) -> dict:
        """
        Representation engineering (ActAdd, Turner et al. 2023): build a steering
        direction from the difference of two concept activations, inject it into
        the residual stream at `layer` with coefficient `strength`, and return
        both the ORIGINAL and STEERED residual streams (for trajectory overlay)
        plus greedy generations with and without the intervention.

        direction = mean_act(positive) - mean_act(negative)   (at `layer`)
        steered residual at `layer` += strength * direction    (all positions)

        Returns dict with residual streams [n_layers+1, n_tokens, d_model] for
        both conditions, the two generations, and the direction norm.
        """
        tokens = self.model.to_tokens(text)
        pos = self.extract_concept_stream(positive)  # [n_layers+1, d_model]
        neg = self.extract_concept_stream(negative)
        # Concept streams are indexed like residual_stream: index 0 = embedding,
        # index i = blocks.{i-1}.hook_resid_post. The hook below is
        # blocks.{layer}.hook_resid_post, i.e. stream index layer + 1 — read the
        # direction at the SAME point it is injected (ActAdd), not one block earlier.
        direction = torch.tensor(
            pos[layer + 1] - neg[layer + 1], dtype=torch.float32, device=self.device
        )
        steering = strength * direction
        hook_name = f"blocks.{layer}.hook_resid_post"

        def steer_hook(value, hook):  # noqa: ANN001
            return value + steering

        filt = lambda n: (  # noqa: E731
            "hook_resid_post" in n or n == "hook_embed" or "hook_resid_pre" in n
        )

        with torch.no_grad():
            _, cache_orig = self.model.run_with_cache(tokens, names_filter=filt)
            gen_orig = self.model.generate(
                tokens, max_new_tokens=n_generate, do_sample=False, verbose=False
            )
            with self.model.hooks(fwd_hooks=[(hook_name, steer_hook)]):
                _, cache_steer = self.model.run_with_cache(tokens, names_filter=filt)
                gen_steer = self.model.generate(
                    tokens, max_new_tokens=n_generate, do_sample=False, verbose=False
                )

        def build_rs(cache) -> np.ndarray:  # noqa: ANN001
            layers = []
            first = "hook_embed" if "hook_embed" in cache else "blocks.0.hook_resid_pre"
            layers.append(cache[first][0].detach().cpu().numpy())
            for l in range(self.n_layers):
                layers.append(cache[f"blocks.{l}.hook_resid_post"][0].detach().cpu().numpy())
            return np.stack(layers, axis=0)

        return {
            "tokens": self.model.to_str_tokens(text),
            "residual_original": build_rs(cache_orig),
            "residual_steered": build_rs(cache_steer),
            "generation_original": self.model.to_string(gen_orig[0]),
            "generation_steered": self.model.to_string(gen_steer[0]),
            "direction_norm": float(direction.norm()),
            "has_bos": self.prepends_bos,
        }

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

        BOS handling: the BOS token is NOT part of the returned cloud (it is the
        attention-sink outlier, not text — in the old version it sat at the
        corner of every short text's point cloud), but every window is run WITH
        a BOS prepended so each chunk sees the in-distribution context the model
        was trained on.

        Args:
            text:       input text (can be long — book-length)
            layer:      which layer to extract (-1 = last, else 0..n_layers)
            chunk_size: max tokens per forward pass (including the BOS)
            stride:     step size between windows (< chunk_size = overlap)

        Returns:
            np.ndarray of shape [n_content_tokens, d_model]
        """
        if layer < -1 or layer > self.n_layers:
            raise ValueError(f"layer must be -1 (last) or 0..{self.n_layers}")
        layer_idx = layer if layer >= 0 else self.n_layers + 1 + layer

        # Tokenize full text once; split off the BOS the tokenizer prepended.
        token_ids = self.model.to_tokens(text)[0]  # [total_seq]
        if self.prepends_bos and token_ids.shape[0] > 1:
            bos, content_ids = token_ids[:1], token_ids[1:]
        else:
            bos, content_ids = None, token_ids
        total_len = int(content_ids.shape[0])
        win = chunk_size - (1 if bos is not None else 0)  # content tokens per window

        names_filter = lambda name: (  # noqa: E731
            "hook_resid_post" in name or name == "hook_embed" or "hook_resid_pre" in name
        )

        def window_acts(ids: torch.Tensor) -> np.ndarray:
            """Target-layer activations for one window of content ids (BOS row dropped)."""
            if bos is not None:
                ids = torch.cat([bos, ids])
            _, cache = self.model.run_with_cache(ids.unsqueeze(0), names_filter=names_filter)
            if layer_idx == 0:
                key = "hook_embed" if "hook_embed" in cache else "blocks.0.hook_resid_pre"
            else:
                key = f"blocks.{layer_idx - 1}.hook_resid_post"
            acts = cache[key][0].detach().cpu().numpy()  # [len(ids), d_model]
            return acts[1:] if bos is not None else acts

        if total_len <= win:
            return window_acts(content_ids)

        # Overlapping windows. Each token keeps the activation from the window
        # whose centre is nearest to it.
        window_starts = list(range(0, total_len - win + 1, stride))
        if window_starts[-1] + win < total_len:
            window_starts.append(total_len - win)
        starts = np.array(window_starts)
        positions = np.arange(total_len)
        covered = (positions[None, :] >= starts[:, None]) & (positions[None, :] < starts[:, None] + win)
        dist_to_center = np.abs(positions[None, :] - (starts[:, None] + win / 2))
        assigned = np.where(covered, dist_to_center, np.inf).argmin(axis=0)  # [total_len]

        out = np.zeros((total_len, self.d_model), dtype=np.float32)
        for wi, start in enumerate(window_starts):
            end = start + win
            acts = window_acts(content_ids[start:end])
            mask = assigned[start:end] == wi
            out[start:end][mask] = acts[mask]
        return out
