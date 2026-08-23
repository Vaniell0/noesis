#!/usr/bin/env python3
"""merge_lora.py — merge a train_think_distill.py LoRA checkpoint into
its base weights, producing a plain .pth loadable by the standard
CPU/blink probe battery (experiments/run.py, experiments/A0_state_probe/*).

Promoted 2026-08-23 from an ephemeral, hardcoded-paths VM-only script
(there were already two: one for g1i_think_distill_dynstop32_full/step200,
one improvised for g1i_think_distill_zlk_phase1_v3/step500) into a real,
parameterized, repo-tracked utility — this operation is needed repeatedly,
not once: every LoRA checkpoint that needs the standard battery, and
Phase 1.5 itself starts with exactly this merge before continuing on
full-FT.

Also writes a "think_marker-only" checkpoint directory (real ThinkChain
marker weights, param_names=[] so a --resume load only restores
mlp_delta) for continuity with peft-backend scripts — the standard
battery probes themselves don't consume it (they run over plain text,
no <think> embedding markers; see this file's own note on
think_geometry.py's incompatibility with ThinkChain, docs/rl-track.md).

Usage:
    python experiments/rl/merge_lora.py \\
        --base-model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \\
        --lora-ckpt experiments/rl/runs/archive/g1i_think_distill_zlk_phase1_v3/ckpt_step000500 \\
        --lora-r 32 --lora-alpha 64 --chain-phases 1 \\
        --out-model models/rwkv7-g1i-2.9b-zlk_phase1_v3-step500-merged.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import load_checkpoint
from experiments.rl.train_think_distill import ThinkChain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora-ckpt", required=True, type=Path)
    ap.add_argument("--lora-r", type=int, required=True)
    ap.add_argument("--lora-alpha", type=int, required=True)
    ap.add_argument("--chain-phases", type=int, default=1,
                     help="Must match the checkpoint's trained M — mismatched "
                          "shape silently falls back to random markers for the "
                          "think_marker-only sidecar (does NOT affect the merged "
                          "base weights, which don't depend on the marker at all).")
    ap.add_argument("--out-model", required=True, type=Path)
    ap.add_argument("--out-thinkmarker-ckpt", type=Path, default=None,
                     help="Defaults to <out-model's parent>/ckpt_thinkmarker_only "
                          "next to --lora-ckpt's directory.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    loaded = load_rwkv7(args.base_model, device=args.device, backend="peft",
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    think_marker = ThinkChain(loaded.n_embd, args.chain_phases).to(args.device)
    try:
        step = load_checkpoint(args.lora_ckpt, loaded, mlp_delta=think_marker)
        marker_loaded = True
    except RuntimeError as e:
        step = load_checkpoint(args.lora_ckpt, loaded, mlp_delta=None)
        print(f"[merge_lora] WARNING: think_marker NOT loaded (shape mismatch, "
              f"check --chain-phases): {e}")
        marker_loaded = False
    print(f"[merge_lora] loaded checkpoint at step={step}")

    n_merged = 0
    for name, mod in loaded.model.named_modules():
        if hasattr(mod, "merge") and hasattr(mod, "lora_A") and not getattr(mod, "merged", False):
            mod.merge()
            n_merged += 1
    print(f"[merge_lora] merged {n_merged} LoRA-adapted modules")
    if n_merged == 0:
        print("[merge_lora] WARNING: zero modules merged -- lora_r/alpha mismatch, "
              "or the checkpoint has no LoRA delta at all?")

    raw_sd = loaded.model.state_dict()
    clean_sd = {}
    for k, v in raw_sd.items():
        if "lora_A" in k or "lora_B" in k:
            continue
        k2 = k.replace(".base_layer.", ".")
        clean_sd[k2] = v.detach().cpu()

    orig_sd = torch.load(args.base_model, map_location="cpu", weights_only=True, mmap=True)
    orig_keys = set(orig_sd.keys())
    clean_keys = set(clean_sd.keys())
    missing = orig_keys - clean_keys
    extra = clean_keys - orig_keys
    print(f"[merge_lora] orig keys={len(orig_keys)} clean keys={len(clean_keys)} "
          f"missing={len(missing)} extra={len(extra)}")
    if missing:
        print("[merge_lora] MISSING (first 10):", list(missing)[:10])
    if extra:
        print("[merge_lora] EXTRA (first 10):", list(extra)[:10])
    assert not missing and not extra, "key mismatch -- aborting, not saving"

    sample_key = next((k for k in orig_keys if "receptance.weight" in k or "key.weight" in k),
                       next(iter(orig_keys)))
    diff = (clean_sd[sample_key].float() - orig_sd[sample_key].float()).abs().max().item()
    print(f"[merge_lora] sample key {sample_key}: max abs diff from base = {diff:.6f} "
          f"(should be > 0)")
    assert diff > 0, "merged weight identical to base -- merge had no effect, aborting"

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_sd, args.out_model)
    print(f"[merge_lora] saved merged model -> {args.out_model}")

    if marker_loaded:
        out_dir = args.out_thinkmarker_ckpt or (args.lora_ckpt.parent / "ckpt_thinkmarker_only")
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = torch.load(args.lora_ckpt / "meta.pt", map_location="cpu", weights_only=False)
        new_meta = {"step": step, "param_names": [], "mlp_delta": meta["mlp_delta"]}
        torch.save(new_meta, out_dir / "meta.pt")
        print(f"[merge_lora] saved think_marker-only checkpoint -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
