"""Merge a LoRA adapter into the base RWKV-7 model weights.

Usage:
    python training/merge_lora.py \
        --base  ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        --lora  /tmp/step4_lora_step3500.pth \
        --out   /tmp/step4_merged_step3500.pth \
        --rank  16 --lora-alpha 32

Note: output is always saved in bfloat16 to halve disk usage vs. fp32 base models.
"""
import argparse, torch
from pathlib import Path


def merge(base_path, lora_path, out_path, rank, lora_alpha, device="cpu"):
    """device="cuda" (found 2026-09-10, needed on RAM-constrained boxes):
    load + do the fp32-upcast compute on the GPU instead of host RAM. On an
    8GB-RAM VM, loading the 5.8GB base as CPU tensors and then upcasting
    each merged layer to fp32 ON TOP of that (the old CPU-only path)
    pegged host RAM at 100% and had to be killed — verified on the real
    Brenna A30 box (8GB RAM / 24GB VRAM). GPU has the headroom; host RAM
    doesn't. Tensors move back to CPU only once, right before
    torch.save — that's still ~base-model-sized resident RAM at the very
    end (unavoidable with a single torch.save of the whole dict), but
    without the extra per-layer fp32 temporaries stacking on top of it.
    """
    scale = lora_alpha / rank
    print(f"Loading base model from {base_path} (device={device}) ...")
    base = torch.load(base_path, map_location=device)
    print(f"Loading LoRA weights from {lora_path} ...")
    lora = torch.load(lora_path, map_location=device)

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
        delta = (B.float() @ A.float()) * scale
        base[base_key] = (base[base_key].float() + delta).to(torch.bfloat16)
        merged += 1

    # Cast all remaining fp32 tensors to bf16 (embedding, head, ln weights, etc.)
    for k in base:
        if base[k].dtype == torch.float32:
            base[k] = base[k].to(torch.bfloat16)

    print(f"Merged {merged} LoRA pairs (scale={scale}), output dtype=bfloat16")
    if device != "cpu":
        print("Moving merged state dict to CPU for saving ...")
        base = {k: v.cpu() for k, v in base.items()}
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
    p.add_argument("--device", default="cpu",
                    help="'cuda' to do the merge compute on GPU instead of "
                         "host RAM (see merge()'s 2026-09-10 docstring note).")
    args = p.parse_args()
    merge(args.base, args.lora, args.out, args.rank, args.lora_alpha, args.device)
