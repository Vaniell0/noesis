#!/usr/bin/env python3
"""H12b — run multi-slot probe on RWKV model.

Usage:
    python experiments/A0_H12b_multislot/run_probe.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        --probes /tmp/h12b_probes.jsonl \
        --out    /tmp/h12b_results.json

Output: per-cell accuracy table (K × P) + cross-contamination rate.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.model import load_model
from experiments._common.results import save_result


def greedy_decode(model, tokenizer, prompt: str, max_tokens: int = 16) -> str:
    import torch
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    logits, state = model.forward(ids, None)
    out_ids = []
    for _ in range(max_tokens):
        if logits.dim() > 1:
            logits = logits.reshape(-1)
        nxt = int(torch.argmax(logits).item())
        if nxt == 0:
            break
        out_ids.append(nxt)
        logits, state = model.forward([nxt], state)
    return tokenizer.decode(out_ids).strip().lower()


def score(response: str, answer: str) -> bool:
    return answer.lower() in response.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = os.environ.get("NOESIS_EVAL_DEVICE", "cpu")
    print(f"[H12b] loading model {args.model} on {device}", file=sys.stderr)
    mdl, tok = load_model(args.model, device)

    with open(args.probes) as f:
        probes = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        probes = probes[:args.limit]

    results = []
    # cell key: (n_slots, n_pairs)
    by_cell: dict[tuple, list[bool]] = defaultdict(list)

    for i, p in enumerate(probes):
        resp = greedy_decode(mdl, tok, p["prompt"])
        correct = score(resp, p["answer"])
        results.append({**p, "response": resp, "correct": correct})
        by_cell[(p["n_slots"], p["n_pairs"])].append(correct)
        mark = "OK" if correct else "FAIL"
        print(f"[H12b] {mark} {p['id']} expected={p['answer']!r} got={resp!r}",
              file=sys.stderr, flush=True)

    # Accuracy table
    all_k = sorted({k for k, _ in by_cell})
    all_p = sorted({p for _, p in by_cell})
    print(f"\n{'K\\P':<6}", end="")
    for p in all_p:
        print(f"  P={p}", end="")
    print()
    for k in all_k:
        print(f"K={k:<4}", end="")
        for p in all_p:
            vals = by_cell.get((k, p), [])
            acc = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {acc:.3f}", end="")
        print()

    # Verdict
    k2_acc = {p: sum(by_cell.get((2, p), [])) / max(len(by_cell.get((2, p), [1])), 1)
               for p in all_p}
    k8_acc = {p: sum(by_cell.get((8, p), [])) / max(len(by_cell.get((8, p), [1])), 1)
               for p in all_p}
    drop = {p: k2_acc.get(p, 0) - k8_acc.get(p, 0) for p in all_p}
    print(f"\nDrop K=2→K=8: {drop}")
    if any(d > 0.15 for d in drop.values()):
        print("→ H12b: SLOT CONTAMINATION DETECTED (>0.15 drop)")
    else:
        print("→ H12b: NO CONTAMINATION (multi-slot holds)")

    by_cell_summary = {f"k{k}_p{p}": {"n": len(v), "correct": sum(v),
                                        "accuracy": sum(v) / len(v) if v else 0.0}
                       for (k, p), v in by_cell.items()}
    payload = {
        "model": args.model,
        "n_probes": len(results),
        "by_cell": by_cell_summary,
        "results": results,
        "_summary": {cell: f"{s['accuracy']:.3f} ({s['correct']}/{s['n']})"
                     for cell, s in by_cell_summary.items()},
    }
    out_path = save_result(
        args.out, payload, experiment="h12b_multislot", hypothesis=["H12b"],
        model=args.model, script=__file__,
    )
    print(f"\n[H12b] results -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
