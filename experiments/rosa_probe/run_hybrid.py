"""run_hybrid.py — eval G1h 2.9B + ROSA additive branch (pseudo RWKV-8 probe).

Phase 0 (always):  attach ROSA with zero-init output weights → contributes
                   nothing → scores must match stock G1h exactly.
Phase 1 (--train-pt):  train ROSA addon params on a tokenised .pt corpus
                   for --train-steps gradient steps, then re-eval.

Training uses BPTT-1 (detach WKV state per token): valid for ROSA since it
lives in the feedforward path, not in the recurrent state.

IMPORTANT: Phase 1 requires RWKV_JIT_ON=0 for autograd through hooks.
           Set it in environment before running, or pass --no-jit.
           On CPU with 2.9B this is slow (~5–10 s/step). Use --device cuda.

Usage:
    # Phase 0 only (sanity check — verifies zero-init ROSA doesn't break model)
    python experiments/rosa_probe/run_hybrid.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1h-2.9b.pth \\
        --tasks experiments/A0_eval/tasks.jsonl \\
        --out /tmp/hybrid_p0.json

    # Phase 0 + Phase 1 (train ROSA 200 steps on RFC QA, then re-eval)
    python experiments/rosa_probe/run_hybrid.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1h-2.9b.pth \\
        --tasks experiments/A0_eval/tasks.jsonl \\
        --train-pt training/tokenised/step9_rfc_train.pt \\
        --train-steps 200 --lr 1e-4 --device cuda \\
        --out /tmp/hybrid_p1.json --no-jit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))


# ── Rubric judge ──────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_task(task: Dict[str, Any], response: str) -> bool:
    rubric = task["rubric"]
    rt, rv = rubric["type"], rubric["value"]
    if rt == "exact":
        return _norm(response) == _norm(rv)
    if rt == "contains":
        return _norm(rv) in _norm(response)
    if rt == "regex":
        return bool(re.search(rv, response, re.IGNORECASE))
    return False


# ── Model loading ─────────────────────────────────────────────────────────────

def load_rwkv(model_path: str, device: str, enable_jit: bool) -> Tuple[Any, Any]:
    """Load G1h model via rwkv package.

    Returns (model, pipeline) where pipeline has .encode() / .decode().
    """
    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1" if device.startswith("cuda") else "0"
    os.environ["RWKV_JIT_ON"]  = "1" if enable_jit else "0"

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE

    path = model_path
    if path.endswith(".pth"):
        path = path[:-4]
    if not os.path.exists(path + ".pth"):
        raise FileNotFoundError(f"Not found: {path}.pth")

    dtype_str = "bf16"
    strategy = f"{device} {dtype_str}"
    print(f"[hybrid] loading model {path}.pth  strategy={strategy}")
    model = RWKV(model=path, strategy=strategy)

    vocab_file = os.path.join(os.path.dirname(__import__("rwkv").__file__),
                              "rwkv_vocab_v20230424")
    pipeline = PIPELINE(model, vocab_file)
    return model, pipeline


# ── Greedy decode (torch.no_grad) ─────────────────────────────────────────────

def greedy_decode(model, pipeline, prompt: str, max_tokens: int = 120) -> str:
    ids = pipeline.encode(prompt)
    with torch.no_grad():
        logits, state = model.forward(ids, None)
        out_ids: List[int] = []
        for _ in range(max_tokens):
            if logits.dim() > 1:
                logits = logits.reshape(-1)
            nxt = int(torch.argmax(logits).item())
            if nxt == 0:
                break
            out_ids.append(nxt)
            logits, state = model.forward([nxt], state)
    return pipeline.decode(out_ids)


# ── Eval loop ─────────────────────────────────────────────────────────────────

def run_eval(
    model,
    pipeline,
    tasks: List[Dict[str, Any]],
    max_tokens: int,
    label: str,
) -> Dict[str, Any]:
    print(f"\n[hybrid] === {label} ===")
    results = []
    by_cat: Dict[str, List[bool]] = {}
    t_start = time.time()

    for task in tasks:
        t0 = time.time()
        resp = greedy_decode(model, pipeline, task["prompt"], max_tokens)
        ok = score_task(task, resp)
        cat = task.get("category", "?")
        by_cat.setdefault(cat, []).append(ok)
        results.append({
            "id": task["id"], "category": cat, "correct": ok,
            "response": resp[:300], "elapsed": round(time.time() - t0, 1),
        })
        mark = "OK" if ok else "FAIL"
        print(f"  {mark} [{cat}] {task['id']:30s}  {resp[:60]!r}")

    elapsed = time.time() - t_start
    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)
    print(f"\n  Overall: {n_correct}/{n_total} = {100*n_correct//max(n_total,1)}%  ({elapsed:.1f}s)")
    for cat, vals in sorted(by_cat.items()):
        print(f"    {cat}: {sum(vals)}/{len(vals)}")

    return {
        "label": label,
        "n": n_total,
        "correct": n_correct,
        "accuracy": round(n_correct / max(n_total, 1), 4),
        "by_category": {c: {"correct": sum(v), "total": len(v)} for c, v in by_cat.items()},
        "results": results,
        "wall_s": round(elapsed, 1),
    }


# ── ROSA training ─────────────────────────────────────────────────────────────

def _load_rollouts(pt_path: str) -> List[Dict[str, Tensor]]:
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids = blob["ids"]
    lm  = blob["loss_mask"]
    starts = blob["starts"].tolist()
    rollouts = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        rollouts.append({"ids": ids[s:e], "loss_mask": lm[s:e]})
    return rollouts


def train_rosa(
    model,
    addons: List[nn.Module],
    train_pt: str,
    n_steps: int,
    lr: float,
    device: str,
    log_interval: int = 20,
) -> None:
    """Train ROSA addon params via BPTT-1 CE loss on tokenised corpus.

    BPTT-1: WKV state is detached at each token boundary. Gradients flow
    through the feedforward path (block outputs), which includes rosa_out.
    This is valid because ROSA lives in the feedforward path.
    """
    print(f"\n[hybrid] training ROSA ({n_steps} steps, lr={lr}, device={device})")
    rollouts = _load_rollouts(train_pt)
    rng = torch.Generator()
    rng.manual_seed(42)

    # Freeze base model, unfreeze ROSA only
    for p in model.parameters():
        p.requires_grad_(False)
    rosa_params = [p for a in addons for p in a.parameters()]
    for p in rosa_params:
        p.requires_grad_(True)
    n_params = sum(p.numel() for p in rosa_params)
    print(f"[hybrid] trainable ROSA params: {n_params:,}")

    opt = torch.optim.AdamW(rosa_params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0)

    step = 0
    loss_acc = 0.0
    n_acc = 0

    while step < n_steps:
        # Sample a random rollout
        idx = int(torch.randint(len(rollouts), (1,), generator=rng).item())
        roll = rollouts[idx]
        seq_ids = roll["ids"].tolist()
        lm      = roll["loss_mask"].tolist()

        if len(seq_ids) < 4:
            continue

        opt.zero_grad(set_to_none=True)

        state = None
        loss_terms: List[Tensor] = []

        # BPTT-1: forward token by token, detach state at each step
        with torch.enable_grad():
            for t in range(len(seq_ids)):
                logits, state = model.forward([seq_ids[t]], state)
                # Detach WKV state to truncate BPTT
                if state is not None:
                    state = [s.detach() if isinstance(s, Tensor) else s for s in state]

                if t + 1 < len(seq_ids) and lm[t + 1]:
                    target = seq_ids[t + 1]
                    log_probs = torch.log_softmax(logits.reshape(-1).float(), dim=-1)
                    loss_terms.append(-log_probs[target])

        if not loss_terms:
            continue

        loss = torch.stack(loss_terms).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(rosa_params, 1.0)
        opt.step()

        loss_acc += loss.item()
        n_acc += 1
        step += 1

        if step % log_interval == 0:
            print(f"  step {step:4d}/{n_steps}  loss={loss_acc/n_acc:.4f}")
            loss_acc = 0.0
            n_acc = 0

    print(f"[hybrid] training done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True, help="Path to G1h .pth checkpoint")
    ap.add_argument("--tasks",       required=True, help="Path to tasks.jsonl")
    ap.add_argument("--out",         required=True, help="Output JSON path")
    ap.add_argument("--limit",       type=int, default=None, help="Cap number of tasks")
    ap.add_argument("--max-tokens",  type=int, default=120, help="Greedy decode budget")
    ap.add_argument("--device",      default="cpu")
    ap.add_argument("--no-jit",      action="store_true",
                    help="Disable RWKV JIT (required for Phase 1 training)")
    # ROSA hyperparams
    ap.add_argument("--n-rosa",      type=int, default=256, help="ROSA inner dim")
    ap.add_argument("--qk-bits",     type=int, default=4,   help="Bits per ROSA head")
    ap.add_argument("--max-suffix",  type=int, default=32,  help="ROSA max_suffix_length")
    # Phase 1 training
    ap.add_argument("--train-pt",    default=None, help="Tokenised .pt for ROSA training")
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--log-interval",type=int, default=20)
    args = ap.parse_args()

    enable_jit = not args.no_jit
    if args.train_pt and enable_jit:
        print("[hybrid] WARNING: Phase 1 training requires --no-jit for autograd. "
              "Adding --no-jit automatically.")
        enable_jit = False

    # Load model
    model, pipeline = load_rwkv(args.model, args.device, enable_jit)

    # Attach ROSA (zero-init output → Phase 0 is baseline)
    sys.path.insert(0, str(_HERE))
    from g1h_rosa_block import attach_rosa
    addons = attach_rosa(
        model,
        n_rosa=args.n_rosa,
        qk_bits=args.qk_bits,
        max_suffix_length=args.max_suffix,
        device=args.device,
    )

    # Load tasks
    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"[hybrid] {len(tasks)} tasks")

    phases: List[Dict[str, Any]] = []

    # Phase 0: eval with zero-init ROSA (should match stock G1h)
    p0 = run_eval(model, pipeline, tasks, args.max_tokens,
                  label="Phase0 — zero-init ROSA (baseline sanity)")
    p0["phase"] = 0
    phases.append(p0)

    # Phase 1: train ROSA, then re-eval
    if args.train_pt:
        train_rosa(
            model, addons, args.train_pt,
            n_steps=args.train_steps,
            lr=args.lr,
            device=args.device,
            log_interval=args.log_interval,
        )
        p1 = run_eval(model, pipeline, tasks, args.max_tokens,
                      label=f"Phase1 — after {args.train_steps} ROSA steps")
        p1["phase"] = 1
        p1["train_steps"] = args.train_steps
        p1["lr"] = args.lr
        phases.append(p1)

    out = {
        "model": args.model,
        "n_rosa": args.n_rosa,
        "qk_bits": args.qk_bits,
        "device": args.device,
        "phases": phases,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[hybrid] → {args.out}")


if __name__ == "__main__":
    main()
