#!/usr/bin/env python3
"""l_state_ablation.py — does OPTIMISING L_state raise state separability?

`L_state` (`training/state_reg.py`) is the only objective in this repo that
puts a loss on the WKV state itself rather than on the tokens. It has been
written, gradient-verified live, and left at weight 0.0 since 2026-08-19.

A free pre-check already ran on recorded trajectories: L_state scores a
content-dependent latent feed level with reading real tokens (-16.99 vs
-17.14 on g1d-0.4b) and penalises a fixed-marker chain 3.3x, and the same
ordering appears in the live-direction count. That is a CORRELATION on
trajectories that already exist. It does not show that pushing on the
objective moves separability — motion and curvature can both be increased
without opening a single new direction, which is the Goodhart case this
script exists to catch.

Three arms, identical data, seed and steps:

  ce_only      alpha = 0                      — control
  l_state      alpha > 0, lambda_delta=1, lambda_curvature=1
  motion_only  alpha > 0, lambda_curvature=0  — isolates whether the
               curvature term does any of the work, or whether plain
               state motion accounts for the whole effect

Measured before and after on held-out prompts:
  live         mean over heads of numerical_rank_1pct — the direction
               count, i.e. separability
  ce           held-out cross-entropy — did the model get wrecked

Acceptance: `l_state` raises `live` over `ce_only` AND `motion_only` does
not reproduce it. If motion alone reproduces it, the curvature term is
decorative. If neither moves `live`, the objective optimises its own
quantity without touching the one it was wanted for, and L_state is not the
lever for separability.

## Which L_state — there are two, and they behave differently

`training/state_reg.py::compute_state_reg` (the SFT stack) weights per
layer and clamps each layer's loss at **-10.0**, a cap calibrated in 2026-08
as "2x the typical pretrained baseline of ~5 per layer". On g1d-0.4b the
real per-layer value is about -17, so the clamp fires at step 0 and, since
`clamp` has zero gradient below its floor, **the term contributes no
gradient at all**. Measured here, not argued: with it, all three arms below
come out bit-identical.

`train_wkv_loop.py::_track_l_state` (the RL stack) is a different
implementation of the same idea — one norm over the whole stacked state, no
clamp, no layer weights. That is the one phase-3 would actually use, so it
is the one trained against here, reproduced in `_state_terms` below. The
clamped value is still reported alongside, so the gap between the two is
visible rather than buried.

Training therefore needs no layer choice at all (whole-state norm).
`--work-layers` only selects where separability is MEASURED, and must come
from this checkpoint's own measured profile: `DEFAULT_WORK_LAYERS` is a
32-layer artefact, and the depth fraction was measured non-portable on
2026-09-22 (four models, steepest fall at 0.500 / 0.656 / 0.875).

Runs on CPU thanks to `loader.py::_enable_peft_on_cpu`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rl.loader import load_rwkv7
from experiments.A0_state_probe.jlens_probe import _svd_stats
from training.state_reg import StateRegConfig, compute_state_reg, default_layer_weights

HELD_OUT = [
    "You are a precise reasoning assistant. Work step by step.\n\n"
    "What is the next number in this sequence: 2, 4, 6, 8, ?\n\n<think>\n",
    "You are a precise reasoning assistant. Work step by step.\n\n"
    "Compute the bitwise XOR of 1010 and 0110.\n\n<think>\n",
]


def _state_terms(wkv_seq, lambda_delta: float, lambda_curvature: float):
    """The RL stack's L_state, verbatim (`_track_l_state`, train_wkv_loop.py).

    One norm over the whole stacked state — differenced BEFORE norming, so
    opposite-signed changes in different layers cannot cancel. No clamp, no
    per-layer weights. Minimising this maximises motion and curvature.
    """
    terms = []
    for t in range(1, len(wkv_seq)):
        term = -lambda_delta * (wkv_seq[t] - wkv_seq[t - 1]).norm()
        if t >= 2 and lambda_curvature != 0.0:
            term = term - lambda_curvature * (
                wkv_seq[t] - 2 * wkv_seq[t - 1] + wkv_seq[t - 2]).norm()
        terms.append(term)
    if not terms:
        return torch.zeros((), requires_grad=True)
    return torch.stack(terms).mean()


def _states_and_logits(loaded, ids, work_layers, grad: bool):
    """Step token-by-token, collecting per-layer WKV state at each t.

    `forward_stateful` only returns the state after the whole chunk, so the
    per-timestep sequence `compute_state_reg` needs has to be built by
    stepping T=1 — the same thing `state_trajectory_probe.py` does, and no
    more expensive than one long call, since the CPU WKV op loops per
    timestep internally anyway.
    """
    state = loaded.new_state(batch=1)
    per_layer = {L: [] for L in work_layers}
    wkv_seq = []          # whole stacked state per timestep — the RL-stack term
    logits_seq = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for tid in ids:
            x = torch.tensor([[tid]], dtype=torch.long, device=loaded.device)
            logits, state = loaded.forward_stateful(x, state)
            logits_seq.append(logits[0, -1])
            wkv_seq.append(state.wkv)
            for L in work_layers:
                per_layer[L].append(state.wkv[L])
    stacked = {L: torch.stack(v, dim=1) for L, v in per_layer.items()}  # [B,T,H,h,h]
    return stacked, wkv_seq, torch.stack(logits_seq)


def _live(loaded, prompt, work_layers) -> dict:
    """Mean-over-heads numerical_rank_1pct at the end of a prompt."""
    ids = loaded.tokenizer.encode(prompt)
    stacked, _, logits = _states_and_logits(loaded, ids, work_layers, grad=False)
    out = {}
    for L in work_layers:
        s = stacked[L][:, -1]                      # [B,H,h,h]
        s = s[0] if s.dim() == 4 else s            # [H,h,h] — _svd_stats is per head
        per_head = [_svd_stats(s[h].float()) for h in range(s.shape[0])]
        out[L] = float(sum(d["numerical_rank_1pct"] for d in per_head) / len(per_head))
    tgt = torch.tensor(ids[1:], device=loaded.device)
    ce = float(F.cross_entropy(logits[:-1].float(), tgt))
    return {"live": out, "ce": ce}


def _evaluate(loaded, work_layers) -> dict:
    rows = [_live(loaded, p, work_layers) for p in HELD_OUT]
    live = {L: sum(r["live"][L] for r in rows) / len(rows) for L in work_layers}
    return {"live": live, "live_mean": sum(live.values()) / len(live),
            "ce": sum(r["ce"] for r in rows) / len(rows)}


def run_arm(model_path, arm, *, work_layers, alpha, lambda_curvature,
            steps, seq_len, lr, device, texts) -> dict:
    torch.manual_seed(0)
    loaded = load_rwkv7(model_path, device=device, dtype=torch.float32,
                        backend="peft", lora_r=8, lora_alpha=16, ctx_len=2048)
    trainable = [p for p in loaded.model.parameters() if p.requires_grad]
    print(f"[{arm}] trainable tensors: {len(trainable)}", flush=True)

    # Reported alongside the RL-stack term so the clamp gap stays visible.
    cfg_clamped = StateRegConfig(
        mode="trajectory_reg", alpha=alpha,
        lambda_delta=1.0, lambda_curvature=lambda_curvature,
        work_layers=tuple(work_layers),
        layer_weights=default_layer_weights(tuple(work_layers)))

    before = _evaluate(loaded, work_layers)
    print(f"[{arm}] before: live={before['live_mean']:.3f} ce={before['ce']:.4f}",
          flush=True)

    opt = torch.optim.Adam(trainable, lr=lr)
    log = []
    for step in range(steps):
        text = texts[step % len(texts)]
        ids = loaded.tokenizer.encode(text)[:seq_len]
        if len(ids) < 4:
            continue
        stacked, wkv_seq, logits = _states_and_logits(loaded, ids, work_layers,
                                                       grad=True)
        tgt = torch.tensor(ids[1:], device=loaded.device)
        ce = F.cross_entropy(logits[:-1].float(), tgt)
        if alpha > 0:
            sreg = _state_terms(wkv_seq, 1.0, lambda_curvature)
        else:
            sreg = torch.zeros((), requires_grad=True)
        clamped = float(compute_state_reg(stacked, cfg_clamped))
        loss = ce + alpha * sreg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        log.append({"step": step, "ce": float(ce), "l_state": float(sreg),
                    "l_state_clamped_sft": clamped, "loss": float(loss)})
        if step % 5 == 0 or step == steps - 1:
            print(f"[{arm}] step {step:3d} ce={float(ce):.4f} "
                  f"L_state={float(sreg):+.4f} (sft-clamped: {clamped:+.2f})",
                  flush=True)

    after = _evaluate(loaded, work_layers)
    print(f"[{arm}] after : live={after['live_mean']:.3f} ce={after['ce']:.4f}",
          flush=True)
    return {"arm": arm, "alpha": alpha, "lambda_curvature": lambda_curvature,
            "before": before, "after": after,
            "d_live": after["live_mean"] - before["live_mean"],
            "d_ce": after["ce"] - before["ce"], "log": log}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-layers", required=True,
                     help="Comma list, from THIS checkpoint's measured "
                          "layer_profile. Never inherited — see module docstring.")
    ap.add_argument("--tasks", type=Path,
                     default=Path("experiments/A0_eval/tasks.jsonl"))
    ap.add_argument("--n-texts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    work_layers = [int(x) for x in args.work_layers.split(",")]

    texts = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                texts.append(t["prompt"] + " " + str(t.get("answer", "")))
            if len(texts) >= args.n_texts:
                break

    arms = [("ce_only", 0.0, 1.0),
            ("l_state", args.alpha, 1.0),
            ("motion_only", args.alpha, 0.0)]
    results = []
    for name, alpha, lam_k in arms:
        results.append(run_arm(args.model, name, work_layers=work_layers,
                               alpha=alpha, lambda_curvature=lam_k,
                               steps=args.steps, seq_len=args.seq_len,
                               lr=args.lr, device=args.device, texts=texts))

    by = {r["arm"]: r for r in results}
    print("\n=== summary ===")
    print(f"{'arm':<13}{'d_live':>9}{'d_ce':>9}{'live_after':>12}{'ce_after':>10}")
    for r in results:
        print(f"{r['arm']:<13}{r['d_live']:>+9.3f}{r['d_ce']:>+9.4f}"
              f"{r['after']['live_mean']:>12.3f}{r['after']['ce']:>10.4f}")

    from experiments._common.results import save_result
    save_result(args.out, {"model": args.model, "work_layers": work_layers,
                           "steps": args.steps, "seq_len": args.seq_len,
                           "alpha": args.alpha, "lr": args.lr,
                           "arms": results,
                           "_summary": {
                               "d_live l_state vs ce_only":
                                   f"{by['l_state']['d_live'] - by['ce_only']['d_live']:+.3f}",
                               "d_live motion_only vs ce_only":
                                   f"{by['motion_only']['d_live'] - by['ce_only']['d_live']:+.3f}",
                               "d_ce l_state": f"{by['l_state']['d_ce']:+.4f}",
                           }},
                experiment="l_state_ablation", hypothesis=["H25"],
                model=args.model, script="experiments/rl/l_state_ablation.py")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
