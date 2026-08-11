"""Generate self-CoT corpus from a trained checkpoint.

Runs the step-9 (or any) checkpoint on A0-style tasks with N=2, K=512,
filters rollouts where the final answer is correct, and emits a JSONL
suitable for tokenize_plain_cot.py.

Self-CoT principle: correct <think> trace → strong L_state training signal.
Incorrect traces are discarded. This teaches the model to produce think-tokens
that actually help, not DSL noise.

Usage:
  training/.venv/bin/python experiments/A0.8_refine/generate_cot_corpus.py \\
      --model  training/runs/pilot_g1h_step9/merged.pth \\
      --tasks  experiments/A0_eval/tasks.jsonl \\
      --n-variants 4 \\
      --max-think 256 \\
      --out    training/corpus_open/selfcot.jsonl

Output JSONL fields: id, system, user, think, answer (plain-CoT format,
compatible with tokenize_plain_cot.py).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")

sys.path.insert(0, str(HERE.parents[1] / "training"))


# ── Judge (same logic as run_rosa_eval.py) ────────────────────────────────────

def judge(response: str, task: dict) -> bool:
    rubric = task.get("rubric", {})
    rtype  = rubric.get("type", "")
    resp   = response.strip().lower()
    if rtype == "exact":
        return rubric.get("value", "").lower() in resp
    if rtype == "contains":
        val = rubric.get("value", "")
        return all(w.lower() in resp for w in val.split())
    if rtype == "regex":
        return bool(re.search(rubric.get("value", ""), response, re.IGNORECASE))
    expected = str(task.get("expected", "")).strip().lower()
    return expected != "" and expected in resp


# ── RWKV-7 inference (minimal, CPU or GPU) ───────────────────────────────────

class RWKVInfer:
    """Thin wrapper: load RWKV-7 via the rwkv pip package for inference."""

    def __init__(self, model_path: str, strategy: str = "cuda fp16"):
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE, PIPELINE_ARGS
        self.model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self._PIPELINE_ARGS = PIPELINE_ARGS

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.8, top_p: float = 0.9) -> str:
        out = []
        state = None

        def callback(token):
            out.append(token)

        self.pipeline.generate(
            prompt,
            token_count=max_tokens,
            args=self._PIPELINE_ARGS(
                temperature=temperature,
                top_p=top_p,
                alpha_frequency=0.0,
                alpha_presence=0.0,
                token_stop=[0],
                token_ban=[],
            ),
            callback=callback,
            state=state,
        )
        return "".join(out)


# ── Self-CoT extraction ───────────────────────────────────────────────────────

def extract_think_answer(response: str) -> tuple[str, str]:
    """Parse <think>...</think> and answer from model output."""
    m = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if m:
        think  = m.group(1).strip()
        answer = m.group(2).strip()
    else:
        think  = ""
        answer = response.strip()
    return think, answer


def make_cot_item(task: dict, think: str, answer: str) -> dict:
    uid = hashlib.md5(f"{task['id']}:{think[:40]}".encode()).hexdigest()[:10]
    return {
        "id":       f"selfcot_{task['id']}_{uid}",
        "system":   "You are a precise reasoning assistant. Work step by step.",
        "user":     task["prompt"],
        "think":    think,
        "answer":   answer,
        "source":   "self_cot",
        "base_task": task["id"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       required=True,
                    help="Merged checkpoint (.pth) for inference")
    ap.add_argument("--tasks",       required=True,
                    help="tasks.jsonl path")
    ap.add_argument("--n-variants",  type=int, default=4,
                    help="Samples per task (temperature sampling)")
    ap.add_argument("--max-think",   type=int, default=256,
                    help="Max tokens for <think> section")
    ap.add_argument("--categories",  nargs="*", default=None,
                    help="Task categories to include (default: all)")
    ap.add_argument("--strategy",    default="cuda fp16")
    ap.add_argument("--out",         required=True)
    args = ap.parse_args()

    print(f"[generate_cot] loading model {args.model} …")
    infer = RWKVInfer(args.model, strategy=args.strategy)

    all_tasks = [json.loads(l) for l in open(args.tasks)]
    if args.categories:
        all_tasks = [t for t in all_tasks if t.get("category") in args.categories]
    print(f"[generate_cot] {len(all_tasks)} tasks × {args.n_variants} variants")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total, kept = 0, 0
    with open(out_path, "w") as fout:
        for task in all_tasks:
            prompt = task["prompt"] + "\n\n<think>\n"
            task_kept = 0
            for v in range(args.n_variants):
                t0 = time.time()
                raw = infer.generate(prompt, max_tokens=args.max_think + 80,
                                     temperature=0.8 if v > 0 else 0.0)
                total += 1
                # Reconstruct full response for parsing
                full = "<think>\n" + raw
                think, answer = extract_think_answer(full)
                if not think:
                    continue
                if judge(answer, task):
                    item = make_cot_item(task, think, answer)
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    kept += 1
                    task_kept += 1
                    elapsed = time.time() - t0
                    print(f"  ✓ {task['id']} v{v} ({elapsed:.0f}s) kept={task_kept}")
                else:
                    print(f"  ✗ {task['id']} v{v} wrong: {answer[:60]!r}")

    print(f"\n[generate_cot] {kept}/{total} kept → {args.out}")
    if kept < 20:
        print("[generate_cot] WARNING: <20 items — model may not yet produce correct CoT")


if __name__ == "__main__":
    main()
