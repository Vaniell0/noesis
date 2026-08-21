#!/usr/bin/env python3
"""A0.2 — held-out reasoning eval scorer.

Reads ``tasks.jsonl``, calls an Ollama endpoint (or local rwkv package
if ``--backend rwkv``), collects a single-shot response per task, and
scores against the per-task rubric.

Emits ``<out>.json`` (per-task results + aggregate) and prints a
Markdown summary to stdout.

Backends
--------

- ``--backend ollama`` (default): POSTs to ``{host}/api/generate`` with
  ``stream=false``. Uses ``options.temperature=0.0`` for determinism,
  ``options.num_predict`` capped so runaway generations don't hang the
  eval.
- ``--backend rwkv``: loads a local ``.pth`` via the BlinkDL rwkv
  package (same as A0.5 ``probe.load_model``). Greedy decode until
  newline or max_tokens. Only suitable for models that ship as
  ``.pth`` (no Ollama registry hop).

Rubrics
-------

Case-insensitive by default. See ``README.md`` for the taxonomy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments._common.results import save_result


# --------------------------------------------------------------------------- #
# Rubric scoring
# --------------------------------------------------------------------------- #

def _strip_tool_use(text: str) -> str:
    """Strip <tool_use> wrappers injected by glaive-SFT format bleeding.

    Handles both well-formed <tool_use>...</tool_use> and the unclosed
    <tool_use>{...} variant the Step-5 model produces.

    Priority:
      1. Free text outside all <tool_use> blocks → return that.
      2. Tool-call JSON with "name"+"input" keys → return json.dumps(input)
         so json_subset rubrics fire against the extracted payload.
      3. Raw inner JSON (non-tool-call) → return as-is.
      4. No <tool_use> found → return original text.
    """
    chunks = re.split(r'<tool_use>', text)
    if len(chunks) <= 1:
        return text

    outer_text = chunks[0].strip()
    if outer_text:
        return outer_text

    # Try to parse the first well-formed tool_use block
    for chunk in chunks[1:]:
        chunk = re.sub(r'</tool_use>.*', '', chunk, flags=re.DOTALL).strip()
        start = chunk.find("{")
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(chunk)):
            if chunk[i] == "{":
                depth += 1
            elif chunk[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(chunk[start : i + 1])
                        if isinstance(obj, dict) and "name" in obj and "input" in obj:
                            return json.dumps(obj["input"])
                        return chunk[start : i + 1]
                    except json.JSONDecodeError:
                        break
    return text


def _norm(s: str) -> str:
    return s.strip().lower()


def _json_subset_match(expected: Any, actual: Any) -> bool:
    """Deep check that every key/value in ``expected`` is present and equal
    (loosely: numeric compared numerically, strings case-insensitively)
    inside ``actual``."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(k in actual and _json_subset_match(v, actual[k])
                   for k, v in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return False
        return all(_json_subset_match(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 1e-6
    if isinstance(expected, str) and isinstance(actual, str):
        return _norm(expected) == _norm(actual)
    return expected == actual


def _first_json_object(text: str) -> Optional[Any]:
    """Extract the first balanced {...} block and try to parse. Handles
    common wrapper prose like 'Here is the JSON: {...}'."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def score_task(task: Dict[str, Any], response: str) -> Dict[str, Any]:
    response = _strip_tool_use(response)
    rubric = task["rubric"]
    rt = rubric["type"]
    rv = rubric["value"]

    correct = False
    detail = ""

    if rt == "exact":
        correct = _norm(response) == _norm(rv)
        detail = "exact match"
    elif rt == "contains":
        correct = _norm(rv) in _norm(response)
        detail = "substring match"
    elif rt == "regex":
        correct = bool(re.search(rv, response, re.IGNORECASE))
        detail = "regex match"
    elif rt == "json_subset":
        parsed = _first_json_object(response)
        if parsed is None:
            correct, detail = False, "no valid JSON found in response"
        else:
            correct = _json_subset_match(rv, parsed)
            detail = "json subset match" if correct else f"subset check failed against {parsed}"
    elif rt == "manual":
        correct = False
        detail = "manual review required"
    else:
        correct, detail = False, f"unknown rubric type {rt}"

    return {"correct": correct, "detail": detail}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

def call_ollama(host: str, model: str, prompt: str,
                num_predict: int, timeout_s: int,
                chat_wrap: bool = False) -> str:
    if chat_wrap:
        prompt = f"User: {prompt}\n\nAssistant:"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict,
                    **({"stop": ["\n\nUser", "\n\nHuman"]} if chat_wrap else {})},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "")


def call_rwkv_mloop(loaded, prompt: str, num_predict: int, m_max: int,
                    feed_mode: str = "discrete") -> str:
    """M-loop decode via experiments.rl.wkv_loop.generate_rollout.

    H10's N/K/readout_mode axes collapse to this single M axis (internal
    WKV-loop steps, no decoded think-tokens) — see hypotheses/H10.md's
    stale-marker note. `loaded` comes from experiments.rl.loader.load_rwkv7,
    not experiments._common.model.load_model (different LoadedModel
    interface — this is the same loader/rollout machinery
    train_wkv_loop.py uses for RL, applied here as a pure eval with no
    reward/gradient computation).
    """
    from experiments.rl.wkv_loop import generate_rollout
    rollout = generate_rollout(
        loaded, prompt, feed_mode=feed_mode, M_max=m_max,
        max_answer_tokens=num_predict, answer_temperature=0.0,
    )
    return rollout.text


def call_rwkv_native(loaded, prompt: str, num_predict: int,
                     system_prompt: str = "You are a precise reasoning assistant. Work step by step.",
                     think_prefix: bool = True) -> str:
    """Greedy decode using the model's own training-corpus template, not
    chat_wrap's generic "User:/Assistant:" markup (call_ollama-only —
    never applied to this backend at all, --chat-wrap silently does
    nothing for --backend rwkv) or the bare task prompt call_rwkv sends
    with no framing at all.

    Built 2026-08-21 after both existing rwkv-backend paths gave 0/48 on
    a checkpoint trained exclusively on
    "{system}\\n\\n{user}\\n\\n<think>\\n..." (training/tokenize_plain_cot.py)
    — with no system framing or <think> cue, EOS (now genuinely learned,
    see docs/rl-track.md item 10) fires almost immediately regardless of
    the model's real task capability. Real EOS-stopping (loop breaks the
    moment id 0 is sampled), same convention as call_rwkv/generate_rollout.

    think_prefix=False deliberately omits the "<think>\\n" cue — tests
    the model on a bare instruction with no trained scaffold at all, a
    direct check for whether full-FT toward the (narrow, 6-category)
    think-distill corpus costs general instruction-following outside
    that scaffold, not just accuracy inside it.
    """
    import torch
    full_prompt = f"{system_prompt}\n\n{prompt}\n\n"
    if think_prefix:
        full_prompt += "<think>\n"
    tok = loaded.tokenizer
    ids = tok.encode(full_prompt)
    state = loaded.new_state(batch=1)
    x = torch.tensor([ids], device=loaded.device)
    out_ids: List[int] = []
    with torch.no_grad():
        logits, state = loaded.forward_stateful(x, state)
        for _ in range(num_predict):
            nxt = int(torch.argmax(logits[0, -1]).item())
            if nxt == 0:
                break
            out_ids.append(nxt)
            x = torch.tensor([[nxt]], device=loaded.device)
            logits, state = loaded.forward_stateful(x, state)
    return tok.decode(out_ids)


def call_rwkv(model_ref: str, tokenizer, model, prompt: str,
              num_predict: int, n_passes: int = 1,
              readout_mode: str = "silent", readout_k: int = 64) -> str:
    """Greedy decode via BlinkDL rwkv package.

    H10 three axes:
      n_passes (N): cycle the same prompt through WKV N times, accumulating
        state on top of previous pass. Tests raw WKV convergence property —
        no token emission, no training required.
      readout_k (K): intermediate token budget. K tokens are decoded and fed
        back into WKV one by one, updating state. These tokens are invisible
        (not in the returned answer) — their value is entirely in how they
        shift the WKV state. Not human-readable CoT; structure should be
        WKV-optimal, not legible.
      num_predict: answer token budget (separate from readout_k).

    readout_mode — source of the K intermediate tokens:
      "silent"       : readout_k=0 forced; answer decoded directly after N passes.
      "prompt_cot"   : K tokens decoded as text continuation (free CoT tokens
                       update WKV state), then answer decoded.
      "state_readout": answer decoded directly from accumulated WKV state, no
                       intermediate tokens. Mechanically equivalent to silent;
                       kept as a distinct mode for future impl (proper version
                       would inject </think> trigger token before decoding).
                       NOTE: prior to 2026-08-12 this shared code with prompt_cot
                       — all previous state_readout results are invalid.
    """
    import torch
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    logits, state = model.forward(ids, None)
    for _ in range(n_passes - 1):
        logits, state = model.forward(ids, state)

    def _greedy(budget: int) -> List[int]:
        nonlocal logits, state
        out: List[int] = []
        for _ in range(budget):
            if logits.dim() > 1:
                logits = logits.reshape(-1)
            nxt = int(torch.argmax(logits).item())
            if nxt == 0:
                break
            out.append(nxt)
            logits, state = model.forward([nxt], state)
        return out

    if readout_mode == "state_readout":
        # Inject </think> trigger token(s) to close the reasoning block, then
        # decode. This is the correct implementation: state accumulates from the
        # prompt, </think> signals end-of-thought without visible CoT tokens,
        # and the model decodes the answer from that state alone.
        think_end_ids = tokenizer("</think>", return_tensors="pt")["input_ids"][0].tolist()
        for tid in think_end_ids:
            logits, state = model.forward([tid], state)
        out_ids = _greedy(num_predict)
    elif readout_mode == "silent":
        # No intermediate tokens — answer decoded directly from prompt state.
        out_ids = _greedy(num_predict)
    elif readout_mode == "prompt_cot":
        # Generate K free CoT tokens (update WKV state), then decode answer.
        _greedy(readout_k)
        out_ids = _greedy(num_predict)
    else:
        raise ValueError(f"Unknown readout_mode: {readout_mode!r}")
    return tokenizer.decode(out_ids)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])
    per_cat = {
        cat: {"n": len(vs), "correct": sum(vs),
              "accuracy": sum(vs) / len(vs) if vs else 0.0}
        for cat, vs in by_cat.items()
    }
    n_total = sum(x["n"] for x in per_cat.values())
    n_correct = sum(x["correct"] for x in per_cat.values())
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "overall_accuracy": n_correct / n_total if n_total else 0.0,
        "per_category": per_cat,
    }


def md_report(agg: Dict[str, Any], model_ref: str, elapsed_s: float) -> str:
    lines = [
        f"# A0.2 eval — {model_ref}",
        "",
        f"- Total tasks: {agg['n_total']}",
        f"- Correct: {agg['n_correct']}",
        f"- **Overall accuracy: {agg['overall_accuracy']:.1%}**",
        f"- Wall time: {elapsed_s:.1f}s",
        "",
        "| category | n | correct | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for cat in sorted(agg["per_category"]):
        row = agg["per_category"][cat]
        lines.append(f"| {cat} | {row['n']} | {row['correct']} | {row['accuracy']:.1%} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.2 held-out reasoning eval.")
    ap.add_argument("--tasks", default=None,
                    help="Path to tasks.jsonl (default: alongside this script).")
    ap.add_argument("--backend", choices=["ollama", "rwkv"], default="ollama")
    ap.add_argument("--host", default="http://127.0.0.1:11434",
                    help="Ollama host (ollama backend).")
    ap.add_argument("--model", required=True,
                    help="Ollama model name or path to .pth for rwkv backend.")
    ap.add_argument("--num-predict", type=int, default=64,
                    help="Answer token budget (separate from --readout-k).")
    ap.add_argument("--axis", choices=["h10", "m", "native"], default="h10",
                    help="rwkv backend only. 'h10': legacy N/K/readout_mode axes "
                         "(--n-passes/--readout-mode/--readout-k), sends the bare "
                         "task prompt with no framing at all. 'm': current "
                         "WKV-loop M axis via experiments.rl.wkv_loop.generate_rollout "
                         "(--m-max/--feed-mode) — H10's own stale-marker note says "
                         "N x K x mode collapsed to this single axis. 'native': "
                         "the model's own training-corpus template "
                         "(--system-prompt + task prompt + <think> cue unless "
                         "--no-think-prefix) — added 2026-08-21 after both h10 and "
                         "chat_wrap gave 0/48 on a checkpoint that only ever saw "
                         "this exact template in training (see call_rwkv_native).")
    ap.add_argument("--m-max", type=int, default=16,
                    help="--axis=m only: max internal WKV-loop steps before forced exit.")
    ap.add_argument("--feed-mode", choices=["discrete", "expected", "residual"],
                    default="discrete",
                    help="--axis=m only: how loop-step logits feed back into WKV. "
                         "expected/residual need --device cuda (peft backend).")
    ap.add_argument("--n-passes", type=int, default=1,
                    help="WKV cycling passes before decoding (N axis in H10). "
                         "Each pass re-feeds the prompt through WKV accumulating "
                         "state. rwkv backend only.")
    ap.add_argument("--readout-mode",
                    choices=["prompt_cot", "silent", "state_readout"],
                    default="silent",
                    help="H10 readout mode (rwkv backend only). "
                         "silent: answer directly from state after N passes (default). "
                         "prompt_cot: decode --readout-k invisible tokens first (WKV update), "
                         "then answer. Tokens are not human-readable CoT — they are "
                         "WKV-internal state updates. "
                         "state_readout: same mechanics as prompt_cot, less prompt-constrained. "
                         "(not scored), then answer.")
    ap.add_argument("--readout-k", type=int, default=64,
                    help="Intermediate (invisible) token budget for prompt_cot/state_readout modes.")
    ap.add_argument("--timeout", type=int, default=120,
                    help="Per-request timeout (seconds).")
    ap.add_argument("--out", required=True, help="Path to output JSON.")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="rwkv backend only. Was env-var-only (NOESIS_EVAL_DEVICE, "
                         "silently defaulting to cpu) — the existing G1i state_readout "
                         "result took ~29h because of this. --device wins if set; "
                         "NOESIS_EVAL_DEVICE still works as a fallback.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of tasks (for smoke tests).")
    ap.add_argument("--chat-wrap", action="store_true",
                    help="ollama backend only (silently ignored for --backend rwkv, "
                         "which call_rwkv/call_rwkv_mloop never consult) — wraps "
                         "prompt as 'User: ...\\n\\nAssistant:'. For rwkv, use "
                         "--axis native instead.")
    ap.add_argument("--system-prompt",
                    default="You are a precise reasoning assistant. Work step by step.",
                    help="--axis native only. Default matches "
                         "training/tokenize_plain_cot.py's actual training template.")
    ap.add_argument("--no-think-prefix", action="store_true",
                    help="--axis native only. Omit the '<think>\\n' cue — tests the "
                         "model on a bare instruction with no trained scaffold at "
                         "all (see call_rwkv_native docstring).")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    tasks_path = args.tasks or os.path.join(here, "tasks.jsonl")
    tasks = load_tasks(tasks_path)
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"[eval] backend={args.backend} model={args.model} tasks={len(tasks)}",
          file=sys.stderr, flush=True)

    tok = mdl = loaded = None
    device = args.device or os.environ.get("NOESIS_EVAL_DEVICE", "cpu")
    if args.backend == "rwkv" and args.axis in ("m", "native"):
        from experiments.rl.loader import load_rwkv7
        # backend=None: load_rwkv7's own auto-select (peft on cuda, blink on
        # cpu) — discrete feed_mode works on either; expected/residual need
        # peft specifically and will raise from generate_rollout if not.
        loaded = load_rwkv7(args.model, device=device)
    elif args.backend == "rwkv":
        from experiments._common.model import load_model
        mdl, tok = load_model(args.model, device=device)

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    for i, task in enumerate(tasks):
        try:
            if args.backend == "ollama":
                if args.n_passes > 1:
                    print(f"[eval] warning: --n-passes ignored for ollama backend",
                          file=sys.stderr)
                resp = call_ollama(args.host, args.model, task["prompt"],
                                   args.num_predict, args.timeout,
                                   chat_wrap=args.chat_wrap)
            elif args.axis == "m":
                resp = call_rwkv_mloop(loaded, task["prompt"], args.num_predict,
                                       args.m_max, args.feed_mode)
            elif args.axis == "native":
                resp = call_rwkv_native(loaded, task["prompt"], args.num_predict,
                                        system_prompt=args.system_prompt,
                                        think_prefix=not args.no_think_prefix)
            else:
                resp = call_rwkv(args.model, tok, mdl, task["prompt"],
                                 args.num_predict, args.n_passes,
                                 args.readout_mode, args.readout_k)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            resp = ""
            print(f"[eval]   task {i} ({task['id']}) request failed: {e}",
                  file=sys.stderr, flush=True)

        scored = score_task(task, resp)
        results.append({
            "id": task["id"],
            "category": task["category"],
            "response": resp,
            "expected": task["answer"],
            "rubric": task["rubric"],
            "correct": scored["correct"],
            "detail": scored["detail"],
        })
        mark = "OK" if scored["correct"] else "FAIL"
        print(f"[eval] {mark} {task['id']} ({task['category']})",
              file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    agg = summarise(results)

    payload = {
        "model": args.model,
        "backend": args.backend,
        "num_predict": args.num_predict,
        "axis": args.axis,
        "n_passes": args.n_passes,
        "readout_mode": args.readout_mode,
        "readout_k": args.readout_k,
        "m_max": args.m_max if args.axis == "m" else None,
        "feed_mode": args.feed_mode if args.axis == "m" else None,
        "system_prompt": args.system_prompt if args.axis == "native" else None,
        "think_prefix": (not args.no_think_prefix) if args.axis == "native" else None,
        "n_tasks": len(tasks),
        "elapsed_s": elapsed,
        "aggregate": agg,
        "results": results,
        "_summary": {"overall accuracy": f"{agg['overall_accuracy']:.3f}",
                     "n_correct/n_total": f"{agg['n_correct']}/{agg['n_total']}"},
    }
    if args.axis == "m":
        status = f"backend={args.backend} axis=m M_max={args.m_max} feed_mode={args.feed_mode}"
    elif args.axis == "native":
        status = f"backend={args.backend} axis=native think_prefix={not args.no_think_prefix}"
    else:
        status = f"backend={args.backend} N={args.n_passes} mode={args.readout_mode} K={args.readout_k}"
    out_path = save_result(
        args.out, payload, experiment="a02_eval", hypothesis=["H10"],
        model=args.model, script=__file__, status=status,
    )

    print(md_report(agg, args.model, elapsed))
    print(f"[eval] saved -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
