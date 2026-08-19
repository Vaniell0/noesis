#!/usr/bin/env python3
"""gen_selfcot_fast.py — self-CoT generation via the peft/FLA backend.

Same judge()/extract_think_answer()/make_cot_item() logic as
experiments/A0.8_refine/generate_cot_corpus.py (not importable directly —
"A0.8_refine" has a dot in the dirname, not a valid Python package path —
so duplicated here rather than fought with importlib for ~20 lines).

Why this exists (2026-08-19): the original script's `RWKVInfer` wraps the
`rwkv` pip package's per-token CUDA kernel (no chunk-parallelism) — slow
enough that a 585-task x 1-variant run pegged one vCPU at 100% and made
barely any progress. Everything else this session (diagnostics, training)
went through experiments/rl/loader.py's peft backend + FLA chunk kernels,
which is what actually ran fast. This script reuses that path for
generation instead — greedy argmax loop, matching the exact mechanism
used in today's _diag_step9.py / _diag_think_content.py one-offs.

Runs on any model loadable via loader.py's peft backend — pass --model
step9b-e1 or the G1i checkpoint to compare the two directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path("/root/noesis")
sys.path.insert(0, str(ROOT))

import torch

from experiments.rl.loader import load_rwkv7
from experiments.rl.wkv_loop import _last_vec


def judge(response: str, task: dict) -> bool:
    rubric = task.get("rubric", {})
    rtype = rubric.get("type", "")
    resp = response.strip().lower()
    if rtype == "exact":
        return rubric.get("value", "").lower() in resp
    if rtype == "contains":
        val = rubric.get("value", "")
        return all(w.lower() in resp for w in val.split())
    if rtype == "regex":
        return bool(re.search(rubric.get("value", ""), response, re.IGNORECASE))
    expected = str(task.get("expected", "")).strip().lower()
    return expected != "" and expected in resp


def extract_think_answer(response: str) -> tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", response.strip()


def make_cot_item(task: dict, think: str, answer: str) -> dict:
    uid = hashlib.md5(f"{task['id']}:{think[:40]}".encode()).hexdigest()[:10]
    return {
        "id": f"selfcot_{task['id']}_{uid}",
        "system": "You are a precise reasoning assistant. Work step by step.",
        "user": task["prompt"],
        "think": think,
        "answer": answer,
        "source": "self_cot",
        "base_task": task["id"],
    }


def generate(loaded, prompt: str, max_tokens: int) -> str:
    tok = loaded.tokenizer
    ids = tok.encode(prompt)
    eos_id = 0
    with torch.no_grad():
        state = loaded.new_state(batch=1)
        inp = torch.tensor([ids], dtype=torch.long, device=loaded.device)
        logits, state = loaded.forward_stateful(inp, state)
        out = []
        for _ in range(max_tokens):
            v = _last_vec(logits)
            nid = int(v.argmax().item())
            if nid == eos_id:
                break
            out.append(nid)
            step_inp = torch.tensor([[nid]], dtype=torch.long, device=loaded.device)
            logits, state = loaded.forward_stateful(step_inp, state)
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--max-think", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loaded = load_rwkv7(args.model, device="cuda", backend="peft")
    tasks = [json.loads(l) for l in open(args.tasks)]
    print(f"[gen_selfcot_fast] {len(tasks)} tasks, model={args.model}")

    kept = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for i, task in enumerate(tasks):
            prompt = task["prompt"] + "\n\n<think>\n"
            t0 = time.time()
            raw = generate(loaded, prompt, args.max_think + 80)
            full = "<think>\n" + raw
            think, answer = extract_think_answer(full)
            elapsed = time.time() - t0
            if not think:
                print(f"  - {task['id']} ({elapsed:.1f}s) no <think> found")
                continue
            if judge(answer, task):
                item = make_cot_item(task, think, answer)
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                kept += 1
                print(f"  ✓ {task['id']} ({elapsed:.1f}s) kept={kept}/{i+1}")
            else:
                print(f"  ✗ {task['id']} ({elapsed:.1f}s) wrong: {answer[:60]!r}")

    print(f"\n[gen_selfcot_fast] {kept}/{len(tasks)} kept → {args.out}")


if __name__ == "__main__":
    main()
