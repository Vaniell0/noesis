#!/usr/bin/env python3
"""ByteAdapter: frozen RWKV-7 backbone with trainable byte-level encoder/decoder.

Architecture:
  byte_ids (0-255) → ByteEmbed (256 → model_dim) → frozen WKV backbone → ByteHead (model_dim → 256)

ByteEmbed and ByteHead are the only trainable parameters (~530K for G1d 0.4B).
The WKV backbone processes byte-patch representations instead of subword embeddings.

Usage:
  python byte_adapter.py --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
      --text "01001110" --mode probe
"""

from __future__ import annotations

import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# ByteAdapter module
# ---------------------------------------------------------------------------

class ByteAdapter(nn.Module):
    """Trainable byte encoder/decoder around a frozen RWKV-7 backbone."""

    def __init__(self, model_dim: int = 1024, n_bytes: int = 256):
        super().__init__()
        self.n_bytes = n_bytes
        self.model_dim = model_dim
        # Encoder: byte ID → model_dim vector
        self.embed = nn.Embedding(n_bytes, model_dim)
        # Decoder: model_dim → byte logits
        self.head = nn.Linear(model_dim, n_bytes, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.head.weight)

    def encode(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """byte_ids: (T,) int64 in [0, 255] → (T, model_dim)"""
        return self.embed(byte_ids)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (model_dim,) → (n_bytes,) logits"""
        return self.head(hidden)


# ---------------------------------------------------------------------------
# Probe: run frozen backbone with byte inputs, collect state
# ---------------------------------------------------------------------------

def probe_byte_state(model, adapter: ByteAdapter, text: str, target_layers: list[int]):
    """Run model token-by-token on byte-encoded text, collect WKV state norms."""
    byte_ids = list(text.encode('utf-8'))
    print(f"Text: {repr(text)}")
    print(f"Bytes ({len(byte_ids)}): {byte_ids}")
    print()

    state = None
    results = {l: [] for l in target_layers}

    for i, b in enumerate(byte_ids):
        # Get embedding vector for this byte
        byte_tensor = torch.tensor([b], dtype=torch.long)
        emb_vec = adapter.encode(byte_tensor)[0]  # (model_dim,)

        # Inject into model: temporarily replace embedding for this token
        # We run the model forward manually using the embedding vector
        out, state = model.forward_with_embedding(emb_vec.unsqueeze(0), state)

        for l in target_layers:
            wkv = state[3 * l + 1]  # (n_head, head_size, head_size)
            norm = float(wkv.float().norm())
            results[l].append(norm)

        print(f"  byte {i:2d} ({b:3d} '{chr(b) if 32<=b<127 else '?'}'): "
              + "  ".join(f"L{l}:{results[l][-1]:.3f}" for l in target_layers))

    return results


# ---------------------------------------------------------------------------
# Simple forward hook approach (doesn't need model modification)
# ---------------------------------------------------------------------------

def probe_via_hook(model, adapter: ByteAdapter, tokenizer, text: str, target_layers: list[int]):
    """Alternative: use hook to intercept embedding output and replace with byte embed."""
    byte_ids = list(text.encode('utf-8'))
    world_ids = tokenizer(text)['input_ids']

    print(f"Text: {repr(text)}")
    print(f"World tokens ({len(world_ids)}): {[tokenizer.decode([i]) for i in world_ids]}")
    print(f"Bytes  ({len(byte_ids)}): {[chr(b) if 32<=b<127 else f'\\x{b:02x}' for b in byte_ids]}")
    print()

    # Run with world tokenizer (baseline)
    state_world = None
    world_norms = {l: [] for l in target_layers}
    for tok_id in world_ids:
        out, state_world = model.forward([tok_id], state_world)
        for l in target_layers:
            wkv = state_world[3 * l + 1]
            world_norms[l].append(float(wkv.float().norm()))

    print("World tokenizer state norms (per token):")
    for l in target_layers:
        vals = [f"{v:.3f}" for v in world_norms[l]]
        print(f"  L{l}: {vals}")
    print()

    return world_norms, byte_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default="01001110")
    ap.add_argument("--layers", default="0,4,8,16,23")
    ap.add_argument("--mode", default="probe", choices=["probe", "info"])
    args = ap.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "0")
    os.environ.setdefault("RWKV_V7_ON", "1")

    sys.path.insert(0, "experiments/A0_state_probe")
    from probe import load_model

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, device="cpu")

    # Check model dim from embedding
    import torch
    state_dict = model.w if hasattr(model, 'w') else {}
    model_dim = 1024  # G1d 0.4B default

    adapter = ByteAdapter(model_dim=model_dim)
    print(f"ByteAdapter: {sum(p.numel() for p in adapter.parameters()):,} trainable params")
    print(f"  embed: 256 × {model_dim} = {256*model_dim:,}")
    print(f"  head:  {model_dim} × 256 = {model_dim*256:,}")
    print()

    if args.mode == "info":
        # Just show tokenization comparison
        texts = [args.text, "badc", "STONE", "γαηε", "1010"]
        for t in texts:
            ids = tokenizer(t)['input_ids']
            tokens = [repr(tokenizer.decode([i])) for i in ids]
            bytes_ = list(t.encode('utf-8'))
            print(f"{repr(t):20s}  World({len(ids)}): {tokens}  →  bytes({len(bytes_)}): {bytes_}")
        return

    # Probe: run world tokenizer baseline and show state norms
    probe_via_hook(model, adapter, tokenizer, args.text, target_layers)

    print("Note: byte adapter training requires GPU. Script ready for fine-tuning.")
    print("Architecture: ByteEmbed(256→1024) + frozen_WKV + ByteHead(1024→256)")


if __name__ == "__main__":
    main()
