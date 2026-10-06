#!/usr/bin/env python3
"""optimizer_geometry_probe.py — Adam vs Muon: step size and built geometry.

Two questions in one run, because they share every expensive part (load the
model, take a step) and the answer to one is meaningless without the other.

## 1. How big is the step, actually

Newton-Schulz ORTHOGONALISES the momentum, so the update's singular values
are ~1 regardless of the gradient's magnitude. The step norm is then about
`lr * sqrt(min(m, n))` and carries no information about how converged the
model is. That is a feature when pretraining and a hazard when fine-tuning a
converged checkpoint, which is exactly the regime BlinkDL declined to vouch
for ("muon works for rwkv7 pretraining ... for finetuning a trained rwkv7
model, no idea").

Measured 2026-09-22 on G1i full-FT: `--muon_lr 0.02` takes the loss from
1.72 to 27.9 in ONE step. Rather than sweep learning rates blind, this probe
reports the per-tensor RELATIVE step `‖Δw‖_F / ‖w‖_F` — from which a safe lr
follows by division instead of by five ten-minute runs.

## 2. Does the optimizer build geometry, or only bake it (H26)

H26's central criterion: "if no training procedure raises the state's
live-direction count above the pretrained baseline (13-16 of 64 on G1i,
unchanged by Adam fine-tuning), then 'builds the geometry' has no
operational content". The supporting measurement on the real model is that
G1i base and the Adam/LoRA-trained step500 carry identical per-head spectra
while their attention weights differ 0.4-1.6% — a weight change big enough
to alter behaviour that leaves the spectral envelope untouched.

What has never been run is the direct version: fine-tune ONE checkpoint under
each optimizer and measure the live-direction count before and after. This
does that. Full fine-tuning, no LoRA (dropped 2026-09-22), on a 0.4B where
both optimizers fit — Adam's two moments make the same comparison impossible
at 2.9B, which is why Muon was adopted in the first place.

Comparing rank across arms is only meaningful at comparable loss, so the
final training and held-out CE are reported next to every rank number and a
rank difference at mismatched loss must not be read as a geometry difference.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, MuonHybrid
from experiments.A0_state_probe.jlens_probe import _svd_stats

HELD_OUT = [
    "The capital of France is Paris, and the capital of Germany is Berlin.",
    "You are a precise reasoning assistant. Work step by step.\n\n"
    "What is the next number in this sequence: 2, 4, 6, 8, ?\n",
]


SUBSPACE_K = 8


@torch.no_grad()
def measure_state(loaded, layers, keep_basis: bool = False) -> dict:
    """Live directions, held-out CE, and optionally the state's top-k basis.

    The basis is the instrument for the build/bake question. Live-direction
    COUNT cannot see specialisation inside a structure whose occupancy does not
    change — and that is exactly what "baking" means, so a null on the count
    says nothing either way. What does say something is how far the state's
    principal directions ROTATE for a given drop in loss: an update rule that
    specialises within the existing structure should barely move them, one that
    rebuilds the structure should.
    """
    live_acc = {L: [] for L in layers}
    basis = {L: [] for L in layers}
    ces = []
    for text in HELD_OUT:
        state = loaded.new_state(batch=1)
        ids = loaded.tokenizer.encode(text)
        inp = torch.tensor([ids], device=loaded.device)
        logits, state = loaded.forward_stateful(inp, state)
        wkv = loaded.wkv_stack(state)
        for L in layers:
            s = wkv[L].float()
            s = s[0] if s.dim() == 4 else s
            per_head = [_svd_stats(s[h]) for h in range(s.shape[0])]
            live_acc[L].append(
                sum(d["numerical_rank_1pct"] for d in per_head) / len(per_head))
            if keep_basis:
                _, _, Vh = torch.linalg.svd(s.float(), full_matrices=False)
                basis[L].append(Vh[:, :SUBSPACE_K, :].cpu())   # [H, k, d]
        tgt = torch.tensor(ids[1:], device=loaded.device)
        ces.append(float(F.cross_entropy(logits[0, :-1].float(), tgt)))
    live = {L: sum(v) / len(v) for L, v in live_acc.items()}
    out = {"live": live, "live_mean": sum(live.values()) / len(live),
           "ce": sum(ces) / len(ces)}
    if keep_basis:
        out["_basis"] = {L: torch.stack(v) for L, v in basis.items()}
    return out


def subspace_overlap(before: dict, after: dict) -> dict:
    """Mean squared principal-angle cosine between the top-k state subspaces.

    1.0 = the principal directions did not move at all; lower = the state's
    structure was rebuilt rather than refined. Averaged over heads and prompts.
    """
    out = {}
    for L, vb in before.get("_basis", {}).items():
        va = after["_basis"][L]
        m = torch.matmul(va, vb.transpose(-1, -2))       # [P, H, k, k]
        out[L] = float((m ** 2).sum(dim=(-2, -1)).mean() / SUBSPACE_K)
    return {"per_layer": out,
            "mean": sum(out.values()) / len(out) if out else float("nan")}


def relative_step(before: dict, model) -> dict:
    """Relative step per tensor — ‖Δw‖_F / ‖w‖_F, except for LoRA.

    For a LoRA pair that ratio is undefined and explodes: PEFT initialises
    `lora_B` to zeros, so the denominator is 0 at the first step and the
    measurement returns ~1e11. Measured, not assumed — it is what the first
    run of this probe with `--lora-r 8` produced.

    The meaningful quantity for a pair is the update it actually induces on
    the weight it adapts:

        ΔW = B_after·A_after − B_before·A_before,   relative to ‖W_base‖

    which is also the product-aware form §2.4 of the write-up argues the fix
    has to take. Pairs are detected by PEFT's own naming, the same way
    `loader.py::MuonHybrid` does it, and the base weight is read live off the
    model rather than from the trainable snapshot (it is frozen under LoRA, so
    it is not in there).
    """
    params = dict(model.named_parameters())
    rows, pairs = [], {}
    for name, p in params.items():
        for role in ("A", "B"):
            tag = ".lora_%s." % role
            if tag in name:
                key = name.split(tag)[0]
                pairs.setdefault(key, {})[role] = name
    paired_names = {n for pr in pairs.values() for n in pr.values()}

    for name, p in params.items():
        if name not in before or name in paired_names:
            continue
        w0 = before[name]
        dn = float((p.detach().cpu() - w0).float().norm())
        wn = max(float(w0.float().norm()), 1e-12)
        rows.append({"name": name, "rel": dn / wn, "shape": list(p.shape),
                     "kind": "tensor"})

    for key, pr in pairs.items():
        if "A" not in pr or "B" not in pr:
            continue
        a1 = params[pr["A"]].detach().float().cpu()
        b1 = params[pr["B"]].detach().float().cpu()
        a0 = before[pr["A"]].float()
        b0 = before[pr["B"]].float()
        dW = (b1 @ a1) - (b0 @ a0)
        base = params.get(key + ".base_layer.weight",
                          params.get(key + ".weight"))
        wn = max(float(base.detach().float().norm()), 1e-12) if base is not None \
            else float("nan")
        rows.append({"name": key + " (induced dW)", "rel": float(dW.norm()) / wn,
                     "shape": list(dW.shape), "kind": "lora_pair"})

    rows.sort(key=lambda r: -r["rel"])
    rels = [r["rel"] for r in rows]
    return {"max": max(rels), "median": sorted(rels)[len(rels) // 2],
            "n_lora_pairs": sum(1 for r in rows if r["kind"] == "lora_pair"),
            "top": rows[:8]}


@torch.no_grad()
def _net_displacement(before: dict, model) -> float:
    """Median over tensors of ‖w_t − w_0‖_F / ‖w_0‖_F."""
    rels = []
    for name, p in model.named_parameters():
        if name not in before:
            continue
        w0 = before[name]
        rels.append(float((p.detach().cpu() - w0).float().norm())
                    / max(float(w0.float().norm()), 1e-12))
    return sorted(rels)[len(rels) // 2] if rels else 0.0


def run_arm(model_path, opt_name, *, lr, steps, seq_len, layers, texts,
            device, dtype=torch.float32, lora_r: int = 0,
            pair_fix: bool = True, seed: int = 0,
            path_budget: float = 0.0, shuffle_data: bool = False) -> dict:
    torch.manual_seed(seed)
    if shuffle_data:
        # Without this the run is fully deterministic — checkpoint init, fixed
        # data order, no dropout, no sampling — so `seed` has nothing to act on
        # and three "seeds" return bit-identical numbers. Found 2026-09-22 by
        # running three and getting the same digits.
        texts = list(texts)
        random.Random(seed).shuffle(texts)
    loaded = load_rwkv7(model_path, device=device, dtype=dtype, backend="peft",
                        lora_r=lora_r, lora_alpha=2 * lora_r if lora_r else 0,
                        ctx_len=2048)
    model = loaded.model
    before = measure_state(loaded, layers, keep_basis=True)
    print(f"[{opt_name}] before: live={before['live_mean']:.3f} "
          f"ce={before['ce']:.4f}", flush=True)

    if opt_name == "muon":
        opt = MuonHybrid(model.named_parameters(), lr=lr,
                          momentum_warmup_steps=0, pair_fix=pair_fix)
        aux = torch.optim.AdamW(opt.other_params, lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        aux = None

    # Kept on CPU: a full fp32 copy of the weights on the GPU would double
    # the parameter footprint for the sake of one first-step measurement.
    snap = {n: p.detach().to("cpu", copy=True)
            for n, p in model.named_parameters() if p.requires_grad}
    log, rel_first = [], None
    for step in range(steps):
        ids = loaded.tokenizer.encode(texts[step % len(texts)])[:seq_len]
        if len(ids) < 4:
            continue
        inp = torch.tensor([ids], device=device)
        state = loaded.new_state(batch=1)
        logits, _ = loaded.forward_stateful(inp, state)
        tgt = torch.tensor(ids[1:], device=device)
        loss = F.cross_entropy(logits[0, :-1].float(), tgt)
        opt.zero_grad(set_to_none=True)
        if aux is not None:
            aux.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()
        if aux is not None:
            aux.step()
        if step == 0:
            rel_first = relative_step(snap, model)
            print(f"[{opt_name}] first step relative size: "
                  f"max={rel_first['max']:.3e} median={rel_first['median']:.3e}",
                  flush=True)
        # Net displacement from the ORIGINAL weights, median over tensors.
        # This is the budget the arms are matched on when --path-budget is set.
        #
        # Why matching on it and not on steps or on lr: measured 2026-09-22,
        # the subspace-overlap metric is dominated by how far the run moved,
        # not by which rule moved it. Changing Adam's lr from 1e-5 to 1e-4
        # moves overlap by 0.33; switching optimizer at comparable quality
        # moves it by 0.008 — forty times less. Any statistic that training
        # itself drives will do this, so an optimizer comparison has to hold
        # distance travelled fixed, which is what this does.
        net = _net_displacement(snap, model)
        log.append({"step": step, "loss": float(loss), "net_disp": net})
        if step % 5 == 0 or step == steps - 1:
            print(f"[{opt_name}] step {step:3d} loss={float(loss):.4f} "
                  f"net_disp={net:.5f}", flush=True)
        if path_budget and net >= path_budget:
            print(f"[{opt_name}] path budget {path_budget} reached at step "
                  f"{step}", flush=True)
            break

    after = measure_state(loaded, layers, keep_basis=True)
    print(f"[{opt_name}] after : live={after['live_mean']:.3f} "
          f"ce={after['ce']:.4f}", flush=True)
    del model, loaded, opt
    torch.cuda.empty_cache() if device == "cuda" else None
    rot = subspace_overlap(before, after)
    print(f"[{opt_name}] top-{SUBSPACE_K} subspace overlap: {rot['mean']:.4f} "
          f"(1.0 = unchanged)", flush=True)
    for d in (before, after):
        d.pop("_basis", None)
    return {"optimizer": opt_name, "lr": lr, "lora_r": lora_r,
            "pair_fix": pair_fix, "subspace_overlap": rot,
            "before": before, "after": after,
            "d_live": after["live_mean"] - before["live_mean"],
            "d_ce": after["ce"] - before["ce"],
            "first_step_relative": rel_first,
            "final_loss": log[-1]["loss"] if log else None, "log": log}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", required=True,
                     help="From THIS checkpoint's measured profile.")
    ap.add_argument("--data", type=Path,
                     default=Path("training/corpus_open/step9b_combined_flat.jsonl"))
    ap.add_argument("--n-texts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0,
                     help="Seeds the arm. Added 2026-09-22 after three runs "
                          "labelled seed 0/1/2 turned out to be three repeats "
                          "of seed 0 — run_arm took the parameter, main never "
                          "passed it. Repeats are still useful (they give the "
                          "GPU-nondeterminism floor: d_live varied by 0.03, "
                          "subspace overlap by 0.0006) but they are not seeds.")
    ap.add_argument("--path-budget", type=float, default=0.0,
                     help="Stop the arm once median ‖w−w0‖/‖w0‖ reaches this. "
                          "Matching arms on distance travelled instead of on "
                          "steps or lr is the only way the geometry question "
                          "separates from the step-size question — see "
                          "_net_displacement. 0 disables.")
    ap.add_argument("--shuffle-data", default="off", choices=("on", "off"),
                     help="Shuffle the text order per seed. Without it the run "
                          "has no stochastic element at all and seeds are "
                          "indistinguishable.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"),
                     help="bfloat16 is required at 2.9B: fp32 params plus fp32 "
                          "grads alone are 23.6GB of a 24GB card.")
    ap.add_argument("--lora-r", type=int, default=0,
                     help="Attach a LoRA adapter of this rank. 0 = full "
                          "fine-tune. Non-zero turns this into the measurement "
                          "behind the doc's 1.2: a smaller adapter should show a "
                          "LARGER relative step, which is backwards from the "
                          "usual intuition and had never been measured.")
    ap.add_argument("--pair-fix", default="on", choices=("on", "off"),
                     help="MuonHybrid's factor-pair equalisation. 'off' "
                          "reproduces the upstream shape-coefficient behaviour, "
                          "so on/off at fixed rank measures the fix on a real "
                          "model rather than on the toy.")
    ap.add_argument("--arms", default="muon,adam",
                     help="Which optimizers to run. At 2.9B only 'muon' fits — "
                          "AdamW's two fp32 moments are 23.6GB on their own, "
                          "which is the reason Muon was adopted here at all.")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    texts = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
            if len(texts) >= args.n_texts:
                break
    print(f"[probe] {len(texts)} texts, {args.steps} steps, layers {layers}",
          flush=True)

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    results = []
    for name, lr in (("muon", args.muon_lr), ("adam", args.adam_lr)):
        if name not in wanted:
            continue
        t0 = time.time()
        r = run_arm(args.model, name, lr=lr, steps=args.steps,
                    seq_len=args.seq_len, layers=layers, texts=texts,
                    device=args.device, dtype=dtype, lora_r=args.lora_r,
                    pair_fix=args.pair_fix == "on", seed=args.seed,
                    path_budget=args.path_budget,
                    shuffle_data=args.shuffle_data == "on")
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)

    print("\n=== summary ===")
    print(f"{'opt':<7}{'lr':>9}{'rel step':>11}{'loss0':>9}{'lossN':>9}"
          f"{'d_live':>9}{'d_ce':>9}")
    for r in results:
        print(f"{r['optimizer']:<7}{r['lr']:>9.1e}"
              f"{r['first_step_relative']['max']:>11.2e}"
              f"{r['log'][0]['loss']:>9.3f}{r['final_loss']:>9.3f}"
              f"{r['d_live']:>+9.3f}{r['d_ce']:>+9.4f}")

    from experiments._common.results import save_result
    save_result(args.out, {"model": args.model, "layers": layers,
                           "steps": args.steps, "arms": results,
                           "_summary": {
                               r["optimizer"]: (
                                   f"rel_step {r['first_step_relative']['max']:.2e}, "
                                   f"loss {r['log'][0]['loss']:.3f}->{r['final_loss']:.3f}, "
                                   f"d_live {r['d_live']:+.3f}, d_ce {r['d_ce']:+.4f}")
                               for r in results
                           }},
                experiment="optimizer_geometry", hypothesis=["H26"],
                model=args.model, script="experiments/rl/optimizer_geometry_probe.py")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
