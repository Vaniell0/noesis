#!/usr/bin/env python3
"""Rebuild a flat LoRA state dict from the RL stack's per-parameter
checkpoint directory (checkpoint.py::save_checkpoint format), then merge
it into a base model via training/merge_lora.py::merge().

training/merge_lora.py itself expects a single flat .pth dict (its own
docstring: "lora keys: base_model.model.blocks.X.att.Y.lora_{A|B}...");
checkpoint.py saves one file per param (avoids a second full-model CPU
copy that OOM-killed a run on 2026-08-18) plus meta.pt. This script
bridges the two formats without duplicating merge_lora.py's merge logic
-- written 2026-09-05 to fix a corrupt merged checkpoint locally, kept
here (not left in job-tmp scratch) so the same merge can be re-run on a
rented GPU VM directly from the small checkpoint dir + base model,
instead of transferring an already-merged multi-GB file over the network.

Only lora_A/lora_B entries are merged -- ThinkChain's `chain` embedding
(meta.pt's mlp_delta) is a new parameter, not a delta on an existing base
weight, so it isn't part of this merge; it loads separately at the start
of the next training run (see checkpoint.py::load_checkpoint and the
.yaml note alongside the checkpoint directory).

Usage:
    training/.venv/bin/python experiments/rl/merge_checkpoint_dir.py \\
        --ckpt-dir experiments/rl/checkpoints/g1i_zlk_phase1_v3_step500 \\
        --base models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \\
        --out models/rwkv7-g1i-2.9b-zlk_phase1_v3-step500-merged.pth
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from training.merge_lora import merge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(REPO_ROOT / "experiments/rl/checkpoints/g1i_zlk_phase1_v3_step500"))
    ap.add_argument("--base", default=str(REPO_ROOT / "models/rwkv7-g1i-2.9b-20260805-ctx16384.pth"))
    ap.add_argument("--out", default=str(REPO_ROOT / "models/rwkv7-g1i-2.9b-zlk_phase1_v3-step500-merged.pth"))
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                     help="'cuda' to merge on GPU instead of host RAM -- "
                          "needed on Brenna (8GB RAM / 24GB VRAM), see "
                          "training/merge_lora.py::merge()'s 2026-09-10 note.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu", weights_only=False)
    param_names = meta["param_names"]
    lora_names = [n for n in param_names if "lora_A" in n or "lora_B" in n]
    print(f"[merge_checkpoint_dir] {len(param_names)} total trainable params, {len(lora_names)} are lora_A/B")

    flat = {n: torch.load(ckpt_dir / "model" / f"{n}.pt", map_location="cpu") for n in lora_names}

    lora_tmp = ckpt_dir / "_flat_lora_tmp.pth"
    torch.save(flat, lora_tmp)
    print(f"[merge_checkpoint_dir] wrote flat LoRA dict: {lora_tmp} ({lora_tmp.stat().st_size/1e6:.1f} MB, {len(flat)} tensors)")

    merge(args.base, str(lora_tmp), args.out, args.lora_r, args.lora_alpha, args.device)
    lora_tmp.unlink()


if __name__ == "__main__":
    main()
