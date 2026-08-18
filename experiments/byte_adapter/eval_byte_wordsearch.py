#!/usr/bin/env python3
"""eval_byte_wordsearch.py — word-search accuracy with byte tokenizer on frozen model.

Patches model.z['emb.weight'][:256] with random ByteAdapter embeddings,
then runs greedy decode on nsp word-search tasks and checks rubric regex.

Usage:
    python experiments/byte_adapter/eval_byte_wordsearch.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        --tasks experiments/A0_eval/tasks_matrix_wordsearch_nsp.jsonl \
        --max-new 128 --out experiments/A0_eval/results/g1d_wordsearch_nsp_byte.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "experiments/A0_state_probe"))
sys.path.insert(0, str(ROOT / "experiments/byte_adapter"))

os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")
os.environ.setdefault("RWKV_V7_ON", "1")


def greedy_decode(model, input_ids: list[int], max_new: int) -> list[int]:
    with torch.no_grad():
        logits, state = model.forward(input_ids, None)
        out = []
        for _ in range(max_new):
            if logits.dim() > 1:
                logits = logits[-1]
            next_id = int(logits.argmax().item())
            if next_id == 0:
                break
            out.append(next_id)
            logits, state = model.forward([next_id], state)
    return out


def check_rubric(text: str, rubric: dict) -> bool:
    if rubric.get("type") == "regex":
        return bool(re.search(rubric["value"], text, re.IGNORECASE))
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="experiments/A0_eval/tasks_matrix_wordsearch_nsp.jsonl")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/A0_eval/results/g1d_wordsearch_nsp_byte.json")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="was hardcoded to cpu — fine for G1d (0.4B) but too slow "
                         "for 2.9B models over many tasks (found 2026-08-18)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    from probe import load_model
    from byte_adapter import ByteAdapter

    print(f"Loading {args.model} ...")
    model, _ = load_model(args.model, device=args.device)

    with torch.no_grad():
        emb = model.z["emb.weight"]
        # model_dim=1024 was hardcoded to G1d's n_embd — broke on any other
        # model size (found running G1i, n_embd=2560, for the first time,
        # 2026-08-18: "expanded size 2560 must match existing size 1024").
        adapter = ByteAdapter(model_dim=emb.shape[1])
        emb[:256] = adapter.embed.weight.to(emb.dtype)
    print(f"Patched emb[:256] with random ByteAdapter (dtype={emb.dtype})")

    tasks = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    print(f"Tasks: {len(tasks)}")

    results = []
    per_level: dict[int, dict] = {}
    t0 = time.time()

    for i, task in enumerate(tasks):
        prompt_bytes = list(task["prompt"].encode("utf-8"))
        output_ids = greedy_decode(model, prompt_bytes, args.max_new)
        text = bytes([b for b in output_ids if 0 <= b < 256]).decode("utf-8", errors="replace")
        correct = check_rubric(text, task.get("rubric", {}))

        lvl = task.get("level", 0)
        if lvl not in per_level:
            per_level[lvl] = {"n": 0, "correct": 0}
        per_level[lvl]["n"] += 1
        per_level[lvl]["correct"] += int(correct)

        results.append({"id": task["id"], "level": lvl, "correct": correct,
                        "output": text[:200]})

        if (i + 1) % 8 == 0:
            so_far = sum(r["correct"] for r in results)
            print(f"  [{i+1}/{len(tasks)}] correct={so_far}  "
                  f"elapsed={time.time()-t0:.0f}s")

    n_correct = sum(r["correct"] for r in results)
    elapsed = time.time() - t0
    overall_acc = n_correct / len(tasks) if tasks else 0.0

    print(f"\n=== G1d 0.4B + byte tokenizer (random ByteAdapter) ===")
    print(f"Overall: {n_correct}/{len(tasks)} = {overall_acc:.1%}")
    for lvl in sorted(per_level):
        d = per_level[lvl]
        print(f"  L{lvl}: {d['correct']}/{d['n']} = {d['correct']/d['n']:.1%}")

    out = {
        "model": args.model,
        "tasks": args.tasks,
        "byte_adapter": "random_init",
        "max_new": args.max_new,
        "n_total": len(tasks),
        "n_correct": n_correct,
        "overall_accuracy": overall_acc,
        "elapsed_s": elapsed,
        "per_level": {str(k): v for k, v in sorted(per_level.items())},
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
