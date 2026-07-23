#!/usr/bin/env python3
"""A0.H12a — probe runner.

Runs the ``gen_triples.py`` outputs against an Ollama-served model.
Default target is ``mollysama/rwkv-7-g1d:0.4b`` which the running
``noesis-runtime.service`` keeps resident, so cold-load latency is
avoided.

Two invocation modes:

- ``--width`` — sweep ``tasks-N*.jsonl`` files.
- ``--dist``  — sweep ``tasks-dist-*.jsonl`` files.

Both are optional and independent; combine to run everything.

Emits one ``results/{stem}.json`` per input file: per-task response +
match booleans + aggregate accuracy / precision / recall / F1.

Reuses the ``call_ollama`` pattern from
``experiments/A0_eval/eval.py`` (kept inline to avoid an import path
hop).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Set, Tuple


ITEM_RE = re.compile(r"item-[a-z0-9]+-\d{2}")


def call_ollama(host: str, model: str, prompt: str, num_predict: int,
                timeout_s: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())
    # Reasoning-model tags (e.g. rwkv-7-g1d) emit chain-of-thought into a
    # separate `thinking` field; the final answer lands in `response` only
    # after CoT terminates. For H12a we care whether the item IDs surface
    # anywhere in the completion, so we concatenate both — the pair
    # extractor is regex-based and language-agnostic.
    thinking = data.get("thinking") or ""
    response = data.get("response") or ""
    return f"{thinking}\n{response}" if thinking else response


def extract_pairs(text: str) -> Set[Tuple[str, str]]:
    """Return the set of item-pairs the model output mentions.

    We accept any line that contains two item-<prefix>-NN identifiers.
    Each pair is normalised to a lexicographically-ordered tuple so
    ``(a, b)`` and ``(b, a)`` count as the same pair.
    """
    out: Set[Tuple[str, str]] = set()
    for line in text.splitlines():
        ids = ITEM_RE.findall(line)
        if len(ids) < 2:
            continue
        uniq: List[str] = []
        seen: Set[str] = set()
        for i in ids:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                p = tuple(sorted((uniq[a], uniq[b])))
                out.add(p)
    return out


def score_task(task: Dict[str, Any], response: str) -> Dict[str, Any]:
    expected: Set[Tuple[str, str]] = {tuple(sorted(p)) for p in task["expected_pairs"]}
    predicted = extract_pairs(response)
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    exact = predicted == expected and len(expected) > 0
    return {
        "expected": [list(p) for p in sorted(expected)],
        "predicted": [list(p) for p in sorted(predicted)],
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "exact_match": bool(exact),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    acc = [1 if r["exact_match"] else 0 for r in rows]
    f1 = [r["f1"] for r in rows]
    return {
        "n": len(rows),
        "accuracy_exact": sum(acc) / len(acc),
        "mean_f1": statistics.fmean(f1),
        "mean_precision": statistics.fmean(r["precision"] for r in rows),
        "mean_recall": statistics.fmean(r["recall"] for r in rows),
        "mean_word_gap": statistics.fmean(r["mean_word_gap"] for r in rows),
    }


def run_one_file(host: str, model: str, num_predict: int, timeout_s: int,
                 tasks_path: pathlib.Path, results_dir: pathlib.Path,
                 tag: str, num_predict_per_n: int = 0,
                 num_predict_cap: int = 3000) -> Dict[str, Any]:
    with tasks_path.open() as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    print(f"[probe {tag}] {tasks_path.name}: {len(tasks)} tasks", file=sys.stderr, flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        if num_predict_per_n > 0:
            per_task_budget = min(
                max(num_predict, num_predict_per_n * int(task.get("n", 1))),
                num_predict_cap,
            )
        else:
            per_task_budget = num_predict
        try:
            resp = call_ollama(host, model, task["prompt"], per_task_budget, timeout_s)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            resp = ""
            print(f"[probe {tag}]   task {i} ({task['id']}) failed: {e}",
                  file=sys.stderr, flush=True)
        scored = score_task(task, resp)
        rows.append({
            "id": task["id"],
            "n": task["n"],
            "mean_word_gap": task["mean_word_gap"],
            "seed": task["seed"],
            "variant": task["variant"],
            "response": resp,
            **scored,
        })
        mark = "OK" if scored["exact_match"] else f"F1={scored['f1']:.2f}"
        print(f"[probe {tag}]  {mark:>7}  {task['id']}", file=sys.stderr, flush=True)

    payload = {
        "tasks_file": str(tasks_path),
        "model": model,
        "num_predict": num_predict,
        "num_predict_per_n": num_predict_per_n,
        "num_predict_cap": num_predict_cap,
        "n_tasks": len(rows),
        "elapsed_s": time.time() - t0,
        "aggregate": _aggregate(rows),
        "results": rows,
    }
    out_path = results_dir / f"{tasks_path.stem.replace('tasks-', '')}.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[probe {tag}]  wrote {out_path.name}  acc_exact={payload['aggregate']['accuracy_exact']:.2f}",
          file=sys.stderr, flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.H12a probe runner.")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="mollysama/rwkv-7-g1d:0.4b")
    ap.add_argument("--num-predict", type=int, default=512,
                    help="Max tokens per response. Kept small — task answer "
                         "is a short pair list. Also used as the floor when "
                         "--num-predict-per-n scales below it.")
    ap.add_argument("--num-predict-per-n", type=int, default=0,
                    help="If >0, per-task budget = min(cap, max(--num-predict, "
                         "this * task.n)). Intended for the damped-search "
                         "variant of H12a (approximation as N grows) — bigger "
                         "N gets more test-time compute so the model has room "
                         "to reason rather than diverge into code-mode.")
    ap.add_argument("--num-predict-cap", type=int, default=3000,
                    help="Absolute ceiling on per-task budget so context + gen "
                         "fits within Ollama's -c 4096.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--tasks-dir", type=pathlib.Path, default=None,
                    help="Directory holding tasks-*.jsonl (default: ./tasks).")
    ap.add_argument("--results-dir", type=pathlib.Path, default=None,
                    help="Directory to write per-file results (default: ./results).")
    ap.add_argument("--width", action="store_true", help="Run the width sweep files (tasks-N*.jsonl).")
    ap.add_argument("--dist", action="store_true", help="Run the distance sweep files (tasks-dist-*.jsonl).")
    ap.add_argument("--only", type=str, default=None,
                    help="Run exactly one file (e.g. tasks-N4.jsonl or tasks-dist-50.jsonl).")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    tasks_dir = args.tasks_dir or (here / "tasks")
    results_dir = args.results_dir or (here / "results")
    results_dir.mkdir(exist_ok=True)

    if not (args.width or args.dist or args.only):
        print("Nothing selected. Pass --width and/or --dist, or --only <file>.", file=sys.stderr)
        return 2

    files: List[Tuple[pathlib.Path, str]] = []
    if args.only:
        p = tasks_dir / args.only
        files.append((p, "single"))
    else:
        if args.width:
            for p in sorted(tasks_dir.glob("tasks-N*.jsonl")):
                files.append((p, "width"))
        if args.dist:
            for p in sorted(tasks_dir.glob("tasks-dist-*.jsonl")):
                files.append((p, "dist"))

    for p, tag in files:
        run_one_file(args.host, args.model, args.num_predict, args.timeout,
                     p, results_dir, tag,
                     num_predict_per_n=args.num_predict_per_n,
                     num_predict_cap=args.num_predict_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
