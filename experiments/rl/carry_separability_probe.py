#!/usr/bin/env python3
"""carry_separability_probe.py — does the deferred-ambiguity task separate?

Diagnostic on a BASE model, no gradient step anywhere. It asks whether the
`carry_ambiguity` task (experiments/A0_eval/gen_tasks.py) actually forces the
state into a different shape, before any training budget is spent on it. If
the two arms look identical, the task does not compel what it was built to
compel and training on it is pointless.

The task, in one line: identical column-addition grid, asked either for the
LAST digit of the result (control — no carry can arrive, nothing to hold) or
the FIRST (ambiguous — the digit is `base + carry_in` and the carry is
undetermined until every column to the right resolves).

## What "a decoder reads more than one candidate" means here, concretely

The answer digit factorises: `answer = (base_digit + carry_in) mod 10`. So
separability has a sharp test — fit one linear probe per FACTOR off the same
state and ask whether both are readable at once:

  base_digit   the column sum ignoring any incoming carry
  carry_in     what arrives from the right, 0..(n_addends-1)
  answer       the resolved digit, for reference

A state that holds the two factors separably reads out on both. A state that
has collapsed to a single guess reads out on `answer` at best, and `carry_in`
falls to the permutation floor. The control arm has `carry_in == 0` by
construction, so it is a built-in null: nothing there should be decodable
above floor, and if it is, the probe is picking up design leakage rather than
content.

Probes are `wkv_linear_probe.py::held_out_linear_probe` — least squares,
train-only centering, a 20-shuffle permutation floor, and the training
design's effective rank reported next to every number (d >> n here, so
"not encoded" is only a finding when the design could have carried it).

Also reported per arm: the live-direction count (mean over heads of
`numerical_rank_1pct`) at the end of the prompt, and greedy answer accuracy,
so the three questions can be answered together — does breadth rise on the
ambiguous arm, does it track correctness, and are both factors readable.

Runs on the `peft` backend on CPU, NOT `blink`. Measured 2026-09-22 on
g1d-0.4b: a 25-token prefill takes 738 ms/token through `blink` and 15 ms/token
through `peft` in one batched call — 50x. `blink` is BlinkDL's reference
inference path and steps token by token in Python; `peft` processes the whole
sequence with batched matmuls and loops only the WKV recurrence. The standing
"blink on CPU, peft on GPU" rule was about which one RUNS, and since
`_enable_peft_on_cpu` both do, so it is now the wrong default for anything
that prefills.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7
from experiments.rl.wkv_linear_probe import held_out_linear_probe
from experiments.rl.wkv_loop import _last_vec
from experiments.A0_state_probe.jlens_probe import _svd_stats


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_tasks", _ROOT / "experiments" / "A0_eval" / "gen_tasks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _factors(task: dict) -> dict | None:
    """base_digit and carry_in for the asked column, from the task's notes.

    Recomputed from the addends rather than stored, so the probe targets can
    never silently drift from what the generator rendered.
    """
    note = task["notes"]
    try:
        addends = eval(note.split("addends=")[1].split(", result=")[0])
        result = int(note.split("result=")[1])
    except Exception:
        return None
    width = max(len(str(result)), max(len(str(a)) for a in addends))
    from_right = task["hold_distance"]
    if from_right >= width:
        return None
    # carry INTO the asked column = carry produced by everything to its right
    low = sum(a % (10 ** from_right) for a in addends) if from_right else 0
    carry_in = low // (10 ** from_right) if from_right else 0
    col_digits = [(a // (10 ** from_right)) % 10 for a in addends]
    base = sum(col_digits)
    return {"base_digit": float(base % 10), "carry_in": float(carry_in),
            "answer": float(int(task["answer"]))}


def _digit_token_ids(loaded) -> dict:
    """Token id for each bare digit '0'..'9', for the restricted readout."""
    out = {}
    for d in "0123456789":
        ids = loaded.tokenizer.encode(d)
        if len(ids) == 1:
            out[d] = ids[0]
    return out


@torch.no_grad()
def _state_row(loaded, prompt: str, layers, digit_ids: dict, gen_tokens: int):
    """State at end of prompt, breadth, and two readings of the answer.

    Accuracy was measured in the first version as the argmax of the single
    next token, which returned 0.00 on both arms — an artefact, since a
    chat-tuned model does not emit a bare digit at that position. Two fairer
    reads replace it:

    (The first version also had a real bug behind that 0.00: on the peft
    backend `logits` is [B, T, V] for a T-token prefill, so `reshape(-1)`
    then argmax returned an index over the whole flattened tensor, far past
    the vocabulary. `_last_vec` takes the final position properly.)

      greedy   `gen_tokens` of greedy continuation, then the task's own
               rubric regex against the decoded text — what the model
               actually answers.
      digit    argmax restricted to the ten bare-digit tokens at the first
               answer position — free, and it asks "which digit does it
               prefer" independently of whether it chose to say a digit.
    """
    state = loaded.new_state(batch=1)
    ids = loaded.tokenizer.encode(prompt)
    inp = torch.tensor([ids]) if loaded.backend == "peft" else ids
    logits, state = loaded.forward_stateful(inp, state)
    wkv = loaded.wkv_stack(state)
    feats, live = [], {}
    for L in layers:
        s = wkv[L].float()
        s = s[0] if s.dim() == 4 else s
        feats.append(s.flatten())
        per_head = [_svd_stats(s[h]) for h in range(s.shape[0])]
        live[L] = float(sum(d["numerical_rank_1pct"] for d in per_head) / len(per_head))

    v = _last_vec(logits).float()
    best = max(digit_ids, key=lambda d: float(v[digit_ids[d]])) if digit_ids else ""

    gen_ids = []
    gl, gs = logits, state
    for _ in range(gen_tokens):
        nxt = int(_last_vec(gl).argmax())
        if nxt == 0:
            break
        gen_ids.append(nxt)
        step = torch.tensor([[nxt]]) if loaded.backend == "peft" else [nxt]
        gl, gs = loaded.forward_stateful(step, gs)
    text = loaded.tokenizer.decode(gen_ids) if gen_ids else ""
    return torch.cat(feats), live, best, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="20,21",
                     help="Layers to read the state from — use THIS "
                          "checkpoint's measured profile, never a default.")
    ap.add_argument("--n-per-arm", type=int, default=64)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--levels", default="3,4",
                     help="One matched control/ambiguous pair, e.g. 3,4.")
    ap.add_argument("--answer-stem", default=" The answer is",
                     help="Appended after 'Assistant:' to force a direct "
                          "answer. Without it a G1 model opens '<think>' and "
                          "needs hundreds of tokens before it reaches a digit, "
                          "which is unaffordable for a 192-task sweep and "
                          "measures think-length rather than knowledge. This "
                          "is a forced-choice read and is reported as such; "
                          "'' restores free generation.")
    ap.add_argument("--chat-template", default="on", choices=("on", "off"),
                     help="Wrap prompts as 'User: ...\\n\\nAssistant:' — the "
                          "same markup experiments/A0_eval/eval.py:184 uses. "
                          "Raw corpus prompts make the model open a chat turn "
                          "('\\n\\nAssistant: <think') instead of answering, "
                          "which reads as 0% accuracy for the wrong reason.")
    ap.add_argument("--gen-tokens", type=int, default=8,
                     help="Greedy continuation length for the real-generation "
                          "accuracy read. 0 disables it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    lv_ctrl, lv_amb = (int(x) for x in args.levels.split(","))

    gt = _load_generator()
    rng = random.Random(args.seed)
    pools: dict[str, list] = {"control": [], "ambiguous": []}
    tries = 0
    while min(len(v) for v in pools.values()) < args.n_per_arm and tries < 200_000:
        tries += 1
        t = gt._carry_gen(rng, tries)
        if t is None:
            continue
        want = lv_ctrl if t["arm"] == "control" else lv_amb
        if t["level"] != want or len(pools[t["arm"]]) >= args.n_per_arm:
            continue
        f = _factors(t)
        if f is None:
            continue
        t["factors"] = f
        pools[t["arm"]].append(t)
    for k, v in pools.items():
        print(f"[carry] {k}: {len(v)} tasks (level "
              f"{lv_ctrl if k == 'control' else lv_amb})", flush=True)

    print(f"[carry] loading {args.model} on {args.device} (peft, fp32)", flush=True)
    loaded = load_rwkv7(args.model, device=args.device, dtype=torch.float32,
                        backend="peft", ctx_len=2048)

    digit_ids = _digit_token_ids(loaded)
    print(f"[carry] bare-digit tokens resolved: {len(digit_ids)}/10", flush=True)

    results = {}
    for arm, tasks in pools.items():
        t0 = time.time()
        rows, lives = [], []
        acc_greedy, acc_digit = [], []
        samples = []
        for i, t in enumerate(tasks):
            text = (f"User: {t['prompt']}\n\nAssistant:{args.answer_stem}"
                    if args.chat_template == "on" else t["prompt"])
            feat, live, best_digit, gen = _state_row(
                loaded, text, layers, digit_ids, args.gen_tokens)
            rows.append(feat)
            lives.append(live)
            acc_digit.append(1.0 if best_digit == t["answer"] else 0.0)
            acc_greedy.append(
                1.0 if re.search(t["rubric"]["value"], gen) else 0.0)
            if len(samples) < 3:
                samples.append({"answer": t["answer"], "digit_argmax": best_digit,
                                "generated": gen})
            if (i + 1) % 8 == 0:
                el = time.time() - t0
                print(f"  [{arm}] {i+1}/{len(tasks)}  {el:.0f}s "
                      f"({el/(i+1):.1f}s/task)", flush=True)
        X = torch.stack(rows)
        n_train = int(len(rows) * args.train_frac)
        probes = {}
        for target in ("base_digit", "carry_in", "answer"):
            y = torch.tensor([t["factors"][target] for t in tasks])
            if float(y.std()) < 1e-8:
                probes[target] = {"constant_target": True, "value": float(y[0])}
                continue
            probes[target] = held_out_linear_probe(X, y, n_train)
        live_mean = {L: sum(d[L] for d in lives) / len(lives) for L in layers}
        results[arm] = {
            "n": len(rows), "n_train": n_train,
            "live_mean": live_mean,
            "live_mean_all": sum(live_mean.values()) / len(live_mean),
            "accuracy_greedy": sum(acc_greedy) / len(acc_greedy),
            "accuracy_digit_restricted": sum(acc_digit) / len(acc_digit),
            "samples": samples,
            "probes": probes,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[{arm}] live={results[arm]['live_mean_all']:.2f} "
              f"acc_greedy={results[arm]['accuracy_greedy']:.2f} "
              f"acc_digit={results[arm]['accuracy_digit_restricted']:.2f}",
              flush=True)
        for s in samples:
            print(f"    want={s['answer']} digit_argmax={s['digit_argmax']} "
                  f"gen={s['generated'][:60]!r}", flush=True)

    print("\n=== probes (held-out R2 vs permutation floor) ===")
    print(f"{'arm':<11}{'target':<12}{'R2_held':>9}{'floor':>9}{'margin_sd':>11}{'eff_rank':>10}{'>max':>8}")
    for arm, r in results.items():
        for target, p in r["probes"].items():
            if p.get("constant_target"):
                print(f"{arm:<11}{target:<12}{'const':>9}{'-':>9}{'-':>11}{'-':>10}")
                continue
            print(f"{arm:<11}{target:<12}{p['held_out_r2']:>9.3f}"
                  f"{p['shuffled_held_out_mean']:>9.3f}"
                  f"{p['margin_sd']:>11.2f}"
                  f"{p['design_effective_rank']:>10.1f}"
                  f"{str(p['above_shuffled_max']):>8}")

    amb, ctl = results["ambiguous"], results["control"]
    from experiments._common.results import save_result
    save_result(args.out, {
        "model": args.model, "layers": layers,
        "chat_template": args.chat_template, "gen_tokens": args.gen_tokens,
        "answer_stem": args.answer_stem,
        "levels": {"control": lv_ctrl, "ambiguous": lv_amb},
        "arms": results,
        "_summary": {
            "live ambiguous - control":
                f"{amb['live_mean_all'] - ctl['live_mean_all']:+.2f}",
            "greedy acc ambiguous / control":
                f"{amb['accuracy_greedy']:.2f} / {ctl['accuracy_greedy']:.2f}",
            "digit-restricted acc ambiguous / control":
                f"{amb['accuracy_digit_restricted']:.2f} / "
                f"{ctl['accuracy_digit_restricted']:.2f}",
            "carry_in R2 (ambiguous)":
                f"{amb['probes']['carry_in']['held_out_r2']:.3f}",
        }},
        experiment="carry_separability", hypothesis=["H25"],
        model=args.model, script="experiments/rl/carry_separability_probe.py")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
