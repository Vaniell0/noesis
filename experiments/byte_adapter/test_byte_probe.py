#!/usr/bin/env python3
"""test_byte_probe.py — probe G1d 0.4B with byte-level inputs via ByteAdapter.

Patches model.z['emb.weight'][:256] with ByteAdapter embeddings (random init),
then runs forward on byte-encoded text. Compares state norms vs WorldTokenizer.

Usage:
    python experiments/byte_adapter/test_byte_probe.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
"""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "experiments/A0_state_probe"))
sys.path.insert(0, str(ROOT / "experiments/byte_adapter"))

from byte_adapter import ByteAdapter


def wkv_norm(state, layer: int) -> float:
    return float(state[3 * layer + 1].float().norm())


def run_world(model, tokenizer, text: str, layers: list[int]) -> dict:
    ids = tokenizer(text)["input_ids"]
    state = None
    norms = {l: [] for l in layers}
    for tok_id in ids:
        _, state = model.forward([tok_id], state)
        for l in layers:
            norms[l].append(wkv_norm(state, l))
    return {"n_tokens": len(ids), "norms": norms}


def run_bytes(model, adapter: ByteAdapter, text: str, layers: list[int]) -> dict:
    byte_ids = list(text.encode("utf-8"))
    state = None
    norms = {l: [] for l in layers}
    for b in byte_ids:
        _, state = model.forward([b], state)
        for l in layers:
            norms[l].append(wkv_norm(state, l))
    return {"n_tokens": len(byte_ids), "norms": norms}


def patch_model(model, adapter: ByteAdapter):
    """Replace first 256 rows of emb.weight with ByteAdapter embeddings."""
    emb = model.z["emb.weight"]  # (65536, dim), bf16
    with torch.no_grad():
        byte_emb = adapter.embed.weight.to(emb.dtype)  # (256, dim)
        emb[:256] = byte_emb
    print(f"Patched emb[:256] with ByteAdapter (dtype={emb.dtype})")


def print_comparison(name: str, world: dict, byte: dict, layers: list[int]):
    print(f"\n=== {name} ===")
    print(f"  World: {world['n_tokens']} tokens  |  Byte: {byte['n_tokens']} bytes")
    for l in layers:
        w_final = world["norms"][l][-1] if world["norms"][l] else 0.0
        b_final = byte["norms"][l][-1] if byte["norms"][l] else 0.0
        w_mean  = float(np.mean(world["norms"][l])) if world["norms"][l] else 0.0
        b_mean  = float(np.mean(byte["norms"][l])) if byte["norms"][l] else 0.0
        print(f"  L{l:2d}  World final={w_final:.3f} mean={w_mean:.3f} | "
              f"Byte  final={b_final:.3f} mean={b_mean:.3f}  ratio={b_final/(w_final+1e-9):.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="0,4,8,16,23")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ.setdefault("RWKV_CUDA_ON", "0")
    os.environ.setdefault("RWKV_V7_ON", "1")

    from probe import load_model
    model, tokenizer = load_model(args.model, device="cpu")

    torch.manual_seed(args.seed)
    adapter = ByteAdapter(model_dim=1024)
    patch_model(model, adapter)

    test_cases = [
        ("bit_book_01 input", "01001110"),
        ("bit_book_01 answer", "badc"),
        ("word-search row (sp)",  "S T O N E"),
        ("word-search row (nsp)", "STONE"),
        ("word-search 5x5 nsp",
         "STONE\nHIGHT\nABCDE\nFGHIJ\nKLMNO"),
        ("Greek substitution", "γαηε"),
    ]

    print("\n" + "="*60)
    print("ByteAdapter probe — G1d 0.4B (random byte embeddings)")
    print("="*60)

    for name, text in test_cases:
        # Reset state for each test
        world = run_world(model, tokenizer, text, layers)

        # Re-patch before byte run (world run doesn't touch patch)
        byte = run_bytes(model, adapter, text, layers)
        print_comparison(name, world, byte, layers)

    print("\nNote: byte embeddings are random — state norms reflect backbone")
    print("dynamics under unstructured input, not trained representations.")
    print("Meaningful comparison requires trained ByteAdapter.")


if __name__ == "__main__":
    main()
