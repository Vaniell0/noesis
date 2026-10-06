#!/usr/bin/env python3
"""train_staged_recall.py — P1 of the redefined RL track: staged-recall CE.

Trains on `gen_staged_recall.py` data: pairs written, a filler gap, then one
key queried. Loss is CE on the single answer token only (every value 0..99 is
one World token, checked), so the attachment point is the output while the task
can only be solved by keeping pairs separable in the state across the gap
(memory: project_noesis_rl_attachment_plan).

Optimizer follows the standing decision (memory: project_noesis_muon_decision):
full fine-tune, no LoRA, Muon at ~1e-4 on the hidden 2-D matrices, and the
auxiliary AdamW group (embeddings, head, token-shift vectors, ...) at its OWN
learning rate — measured 2026-09-22 that once Muon's lr is sane the token-shift
vectors in the aux group become the largest movers if they share it. Linear
warmup on both, because the collapsing runs used warmup 0.

Evaluation (before, during, after): for each eval item, one forward over the
prompt, logits at the last position restricted to the 100 candidate value
tokens, argmax. Chance = 1%. Reported per arm x n_pairs x gap:
  recall    — the trained condition
  inwindow  — ceiling (same tokens, pairs right before the query)
  blank     — floor (answer absent from the input); must stay near 1%, or the
              model is learning a value prior rather than reading the state

The training criterion is NOT recall accuracy alone. Run
`state_capacity_probe.py` and `state_readout_lens.py` on the saved checkpoint
against the base: the claim needs separable facts above 1-2 and the answer to
one key depending on FEWER state directions (less interference).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, MuonHybrid, load_weights_into


def _load(path: Path) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _candidate_ids(tok) -> torch.Tensor:
    ids = []
    for v in range(100):
        e = tok.encode(str(v))
        assert len(e) == 1, f"value {v} is not a single token"
        ids.append(e[0])
    return torch.tensor(ids)


def _prompt_ids(tok, item: dict, cand: torch.Tensor) -> list:
    """Prompt tokens, checked to be an exact prefix of prompt+answer so the
    answer really is the next token the model is asked for."""
    p = tok.encode(item["prompt"])
    full = tok.encode(item["prompt"] + item["answer"])
    assert full[:len(p)] == p and len(full) == len(p) + 1, item["id"]
    assert full[-1] == int(cand[int(item["answer"])]), item["id"]
    return p


@torch.no_grad()
def evaluate(loaded, items: list, cand: torch.Tensor, device: str) -> dict:
    cells = defaultdict(lambda: [0, 0])
    arms = defaultdict(lambda: [0, 0])
    for it in items:
        p = _prompt_ids(loaded.tokenizer, it, cand)
        st = loaded.new_state(batch=1)
        logits, _ = loaded.forward_stateful(torch.tensor([p], device=device), st)
        pred = int(logits[0, -1, cand.to(logits.device)].float().argmax())
        ok = int(pred == int(it["answer"]))
        # key style + vocab in the cell key: without them vocab 4 and vocab 8
        # pooled into one cell at n=8 and n=16 (found on the first P1 run).
        style = it.get("key_style", "distinct")
        if style == "shared":
            style += str(it.get("vocab", ""))
        k = f"{it['arm']}|{style}|n{it['n_pairs']}|g{it['gap_words']}"
        cells[k][0] += ok; cells[k][1] += 1
        arms[it["arm"]][0] += ok; arms[it["arm"]][1] += 1
    return {"by_arm": {a: round(c / t, 4) for a, (c, t) in arms.items()},
            "by_cell": {k: round(c / t, 4) for k, (c, t) in sorted(cells.items())},
            "n": len(items)}


def _subsample(items: list, per_cell: int, seed: int) -> list:
    rng, by = random.Random(seed), defaultdict(list)
    for it in items:
        by[(it["arm"], it["n_pairs"], it["gap_words"])].append(it)
    out = []
    for k in sorted(by):
        out.extend(rng.sample(by[k], min(per_cell, len(by[k]))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", type=Path,
                    default=Path("training/corpus_open/staged_recall_train.jsonl"))
    ap.add_argument("--eval", type=Path,
                    default=Path("training/corpus_open/staged_recall_eval.jsonl"))
    ap.add_argument("--out", type=Path, required=True,
                    help="Result JSON. The checkpoint goes beside it as .pth "
                         "unless --no-save.")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--accum", type=int, default=8,
                    help="Examples per optimizer step (batch 1, accumulated).")
    ap.add_argument("--muon-lr", type=float, default=1e-4)
    ap.add_argument("--aux-lr", type=float, default=1e-5,
                    help="AdamW lr for the non-Muon group — deliberately separate.")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-per-cell", type=int, default=20,
                    help="Eval items per arm x n x gap cell during training; "
                         "the final eval uses the whole eval file.")
    ap.add_argument("--final-eval-per-cell", type=int, default=0,
                    help="0 = final eval on the whole eval file (1080 rows: "
                         "~30 min on CPU, seconds on GPU). Set small for smokes.")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--weights", type=Path, default=None,
                    help="Load a saved state_dict over the base model before "
                         "anything else — with --steps 0 this re-evaluates a "
                         "trained checkpoint.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    loaded = load_rwkv7(args.model, device=args.device, dtype=dtype,
                        backend="peft", lora_r=0, ctx_len=2048)
    if args.weights is not None:
        load_weights_into(loaded, args.weights)
    tok, model = loaded.tokenizer, loaded.model
    cand = _candidate_ids(tok)
    train = _load(args.train)
    ev_all = _load(args.eval)
    ev_small = _subsample(ev_all, args.eval_per_cell, args.seed)
    print(f"[p1] train {len(train)} | eval {len(ev_all)} (periodic {len(ev_small)})",
          flush=True)

    history = []
    t0 = time.time()
    base = evaluate(loaded, ev_small, cand, args.device)
    history.append({"step": 0, **base})
    print(f"[p1] step 0 (base) {base['by_arm']}  {time.time()-t0:.0f}s", flush=True)

    if args.steps > 0:
        opt = MuonHybrid(model.named_parameters(), lr=args.muon_lr,
                         momentum_warmup_steps=0)
        aux = torch.optim.AdamW(opt.other_params, lr=args.aux_lr)
        rng = random.Random(args.seed)
        order = list(range(len(train)))
        pos, run_loss = 0, []
        for step in range(1, args.steps + 1):
            scale = min(1.0, step / max(1, args.warmup))
            # MuonHybrid is not a torch.optim.Optimizer; step() reads self.lr
            # on every call (loader.py, MuonHybrid.step), so warmup is this.
            opt.lr = args.muon_lr * scale
            for g in aux.param_groups:
                g["lr"] = args.aux_lr * scale
            opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                if pos == 0:
                    rng.shuffle(order)
                it = train[order[pos]]; pos = (pos + 1) % len(order)
                p = _prompt_ids(tok, it, cand)
                st = loaded.new_state(batch=1)
                logits, _ = loaded.forward_stateful(
                    torch.tensor([p], device=args.device), st)
                tgt = cand[int(it["answer"])].to(logits.device)
                loss = F.cross_entropy(logits[0, -1].float()[None], tgt[None]) \
                    / args.accum
                loss.backward()
                run_loss.append(float(loss) * args.accum)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); aux.step()
            if step <= 10 or step % 10 == 0:   # first ten: proof of life
                print(f"[p1] step {step} loss {sum(run_loss[-80:])/len(run_loss[-80:]):.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                r = evaluate(loaded, ev_small, cand, args.device)
                history.append({"step": step, "train_loss":
                                sum(run_loss[-80:]) / len(run_loss[-80:]), **r})
                print(f"[p1] step {step} eval {r['by_arm']}", flush=True)
                partial = {"history": history, "args": vars(args)}
                args.out.with_suffix(".partial.json").write_text(
                    json.dumps(partial, default=str))
                if not args.no_save:
                    # The laptop froze at step ~50 on 2026-10-02 and two hours
                    # of training died with it because the weights were only
                    # saved at the end. Resume: --weights <this> --steps <rest>
                    # (Muon momentum restarts; warmup re-runs at a low lr).
                    ck = args.out.with_suffix(".pth")
                    torch.save({k: v.detach().to("cpu")
                                for k, v in model.state_dict().items()},
                               ck.with_suffix(".tmp"))
                    ck.with_suffix(".tmp").replace(ck)

    ev_final = ev_all if args.final_eval_per_cell <= 0 else \
        _subsample(ev_all, args.final_eval_per_cell, args.seed + 1)
    final = evaluate(loaded, ev_final, cand, args.device)
    print(f"[p1] FINAL (all {final['n']}) {final['by_arm']}", flush=True)
    for k, v in final["by_cell"].items():
        print(f"   {k:<26} {v:.3f}", flush=True)

    if args.steps > 0 and not args.no_save:
        ck = args.out.with_suffix(".pth")
        torch.save({k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                   ck)
        print(f"[p1] checkpoint -> {ck}", flush=True)

    from experiments._common.results import save_result
    # Path objects in vars(args) are not JSON-serialisable; the first baseline
    # run died here after printing its numbers (2026-10-02).
    args_json = {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()}
    save_result(args.out, {"model": args.model, "args": args_json,
                           "history": history, "final": final,
                           "_summary": {f"final {a}": f"{v:.3f}"
                                        for a, v in final["by_arm"].items()}},
                experiment="staged_recall_p1", hypothesis=["H25"],
                model=args.model, script="experiments/rl/train_staged_recall.py")
    print(f"written: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
