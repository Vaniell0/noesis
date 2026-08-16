#!/usr/bin/env python3
"""train_wordsearch.py — GRPO RL training on word-search curriculum.

Stack:
    model loading  — RWKV-PEFT LoRA (training/rwkv-peft)
    rollout        — experiments/rl/rollout.py (rwkv inference, no grad)
    rewards        — r_correct + r_clipo + r_entropy (rewards.py)
    update         — GRPO (grpo.py)
    probing        — R-lens after checkpoint 1 (TransformerLens hook)

Usage:
    python3 experiments/rl/train_wordsearch.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1i-2.9b-20260805-ctx16384.pth \
        --tasks experiments/A0_eval/tasks_matrix_wordsearch.jsonl \
        --out experiments/rl/runs/ws_grpo_01 \
        --G 8 --max-new 256 --lr 1e-5 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/rwkv-peft"))
sys.path.insert(0, str(ROOT / "experiments/A0_state_probe"))

from experiments.rl.rollout import generate_rollouts
from experiments.rl.rewards import compute_rewards
from experiments.rl.clipo_head import CLIPOHead
from experiments.rl.grpo import grpo_loss


def load_tasks(path: str) -> list:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_inference_model(model_path: str, device: str):
    from probe import load_model
    model, tokenizer = load_model(model_path, device=device)
    return model, tokenizer


def load_train_model(model_path: str, device: str, lora_r: int = 32):
    """Load RWKV-PEFT model for gradient-based update."""
    from rwkv.model import RWKV
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_CUDA_ON"] = "1" if device == "cuda" else "0"
    model = RWKV(model=model_path, strategy=f"{device} fp32")
    model.train()
    return model


def curriculum_level(step: int, accuracies: dict) -> int:
    """Return current max curriculum level based on recent accuracy."""
    for lvl in range(7, 0, -1):
        if accuracies.get(lvl, 0.0) >= 0.8:
            return min(lvl + 1, 7)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--G", type=int, default=8, help="Rollouts per prompt")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4, help="Prompts per update")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-clipo", action="store_true")
    ap.add_argument("--byte-adapter", action="store_true",
                    help="Use byte-level tokenization + ByteAdapter embedding patch. "
                         "Requires nsp-format tasks (--format nsp). "
                         "Trains only ByteAdapter weights; WKV backbone frozen.")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = load_tasks(args.tasks)
    print(f"[train] {len(all_tasks)} tasks, G={args.G}, device={args.device}")

    # Inference model for rollout (no grad)
    inf_model, tokenizer = load_inference_model(args.model, args.device)
    inf_model.eval()

    # Byte adapter setup (replaces tokenizer + patches embedding table)
    byte_adapter = None
    if args.byte_adapter:
        sys.path.insert(0, str(ROOT / "experiments/byte_adapter"))
        from byte_adapter import ByteAdapter
        import torch
        byte_adapter = ByteAdapter(model_dim=1024).to(args.device)
        # Patch inference model embedding
        with torch.no_grad():
            emb = inf_model.z["emb.weight"]
            emb[:256] = byte_adapter.embed.weight.to(emb.dtype)
        tokenizer = __import__("rollout").ByteTokenizer()
        print(f"[train] byte-adapter mode: vocab=256, trainable params={sum(p.numel() for p in byte_adapter.parameters()):,}")

    # Training model for gradient update
    train_model = load_train_model(args.model, args.device)

    clipo_head = None if args.no_clipo else CLIPOHead(out_dim=512).to(args.device)

    params = list(train_model.parameters())
    if clipo_head is not None:
        params += list(clipo_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    accuracies: dict = {}   # level → recent accuracy
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(args.epochs):
        random.shuffle(all_tasks)
        max_lvl = curriculum_level(epoch, accuracies)
        tasks_this_epoch = [t for t in all_tasks
                            if t.get("level", 1) <= max_lvl]
        if not tasks_this_epoch:
            tasks_this_epoch = all_tasks

        print(f"\n[epoch {epoch}] max_level={max_lvl} tasks={len(tasks_this_epoch)}")

        for batch_start in range(0, len(tasks_this_epoch), args.batch):
            batch = tasks_this_epoch[batch_start: batch_start + args.batch]

            # 1. Rollout (inference model, no grad)
            groups = generate_rollouts(
                inf_model, tokenizer, batch,
                G=args.G, max_new_tokens=args.max_new,
                temperature=0.7, device=args.device,
                byte_mode=args.byte_adapter,
            )

            # 2. Rewards
            rewards_per_group = []
            for group in groups:
                r = compute_rewards(
                    group, clipo_head=clipo_head,
                    clipo_weight=0.0 if args.no_clipo else 1.0,
                )
                rewards_per_group.append(r)

            # 3. GRPO update (training model, with grad)
            optimizer.zero_grad()
            loss = grpo_loss(train_model, groups, rewards_per_group)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # Logging
            batch_correct = sum(
                (r > 0).float().mean().item()
                for r in rewards_per_group
            ) / len(rewards_per_group)

            log_entry = {
                "epoch": epoch,
                "batch_start": batch_start,
                "loss": float(loss),
                "accuracy": batch_correct,
                "max_level": max_lvl,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if batch_start % (args.batch * 5) == 0:
                print(f"  step {batch_start}: loss={loss:.4f} acc={batch_correct:.2%}")

        # Checkpoint
        ckpt_path = out_dir / f"epoch{epoch}.pth"
        torch.save(train_model.state_dict(), ckpt_path)
        print(f"[epoch {epoch}] checkpoint → {ckpt_path}")
        print("  NOTE: Run R-lens probe before advancing curriculum (rl-track.md §mandatory)")


if __name__ == "__main__":
    main()
