"""Merge a LoRA adapter into the base RWKV-7 model weights.

Usage:
    python training/merge_lora.py \
        --base  ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        --lora  /tmp/step4_lora_step3500.pth \
        --out   /tmp/step4_merged_step3500.pth \
        --rank  16 --lora-alpha 32
"""
import argparse, torch
from pathlib import Path


def merge(base_path, lora_path, out_path, rank, lora_alpha):
    scale = lora_alpha / rank
    print(f"Loading base model from {base_path} ...")
    base = torch.load(base_path, map_location="cpu")
    print(f"Loading LoRA weights from {lora_path} ...")
    lora = torch.load(lora_path, map_location="cpu")

    # lora keys: base_model.model.blocks.X.att.Y.lora_{A|B}.default.weight
    # base keys: blocks.X.att.Y.weight
    lora_a = {k: v for k, v in lora.items() if "lora_A" in k}
    lora_b = {k: v for k, v in lora.items() if "lora_B" in k}

    merged = 0
    for key_a, A in lora_a.items():
        key_b = key_a.replace("lora_A", "lora_B")
        B = lora_b[key_b]
        # strip PEFT prefix + lora_A.default.weight suffix → base key
        base_key = (key_a
                    .replace("base_model.model.", "")
                    .replace(".lora_A.default.weight", ".weight"))
        if base_key not in base:
            print(f"  [warn] {base_key} not in base — skipping")
            continue
        delta = (B @ A).to(base[base_key].dtype) * scale
        base[base_key] = base[base_key] + delta
        merged += 1

    print(f"Merged {merged} LoRA pairs (scale={scale})")
    print(f"Saving to {out_path} ...")
    torch.save(base, out_path)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args()
    merge(args.base, args.lora, args.out, args.rank, args.lora_alpha)
