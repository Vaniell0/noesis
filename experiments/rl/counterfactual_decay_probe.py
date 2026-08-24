#!/usr/bin/env python3
"""counterfactual_decay_probe.py — H25 item 3, done for real (2026-08-24):
fleeb83's counterfactual-altered-recurrence discriminator, run on the real
`g1i_zlk_phase1_v3_step500` checkpoint instead of the toy cell. hypotheses/
H25.md's item 3 (see that section) proposed this back on 2026-08-23; what
got built same night instead was a different, weaker ablate-and-measure
test (see H25.md's "A different, substitute test..." correction) — this
script is the actual thing: does the model's real behavior follow the
value predicted by the EXACT recurrence formula under an ALTERED decay,
or does it diverge (something else compensates / the channel isn't
actually load-bearing)?

Why this is NOT tautological the way the toy version was (module
docstrings, micro_wkv.py / this project's own prior conclusion): at toy
scale there is no separate learned pathway for an alternative computation
to diverge from — the controller's only output IS the physics input, so
altering physics and recomputing the SAME formula trivially "predicts
itself". Here there is one: after the intervention (layer 12, one
channel-wide decay override, at the FINAL think-tick only), layers 13-31
process the result through their own learned, nonlinear (attention-style
gating + FFN) transforms *before* we read out the probed states at
layers 16 and 20 (`training/state_reg.py::DEFAULT_WORK_LAYERS`) — those
downstream layers COULD in principle route around/compensate for the
altered layer-12 channel. Whether they do or don't is the actual,
unknown-in-advance answer this script measures.

Design, concretely:
  1. Fit a linear probe (same held_out_linear_probe methodology as
     wkv_linear_probe.py, plain lstsq, train/test split, no leakage) for
     xor_bit1 (best held-out R²=0.314 in the earlier probe run,
     experiments/rl/results/wkv_linear_probe_v3_step500.json) on CLEAN
     final states across DEFAULT_WORK_LAYERS=(12,16,20).
  2. For held-out examples: run the prompt through prefill + entry +
     phase0 + (phase_repeat_ticks - 1) NORMAL ticks, saving that
     pre-final-tick state once (`state_pre_final`).
  3. From `state_pre_final`, branch twice into the FINAL tick only:
       (a) CLEAN: unmodified — also captures (r,k,v,a,kk,w,state_before)
           at layer 12 via a monkeypatch (same reference-forward mirror
           as state_trajectory_probe.py::_capture_rkvwag) — these
           captured tensors are valid for BOTH branches, since layers
           0-11's outputs (layer 12's own input) are IDENTICAL between
           clean and altered runs — the intervention only changes what
           happens AT layer 12 and after, never what feeds INTO it.
       (b) ALTERED: same monkeypatch, but layer 12's `w` is replaced by
           a fixed constant (`--w-altered`, default -3.0, retention
           ≈0.951 — chosen to be clearly different from whatever the
           model's own w naturally is at that channel) before calling
           RUN_RWKV7_INFCTX. Layers 13-31 process whatever real x_out
           this produces, completely unconstrained/un-patched.
  4. Three decoded values via the SAME fitted probe:
       clean_decoded            = beta @ (flatten(clean_final_states)  - x_mean) + y_mean
       actual_altered_decoded   = beta @ (flatten(altered_final_states) - x_mean) + y_mean   [REAL model behavior]
       naive_predicted_decoded  = beta @ (flatten(clean_final_states, but layer-12
                                    segment replaced by RUN_RWKV7_INFCTX(captured
                                    r,k,v,-kk,kk*a, w_altered, state_before)) - x_mean) + y_mean
                                  [layers 16/20 taken UNCHANGED from the clean run —
                                   "if nothing downstream reacts, this is what you'd
                                   get" — the null/no-compensation prediction]
  5. Report, across held-out examples: does (actual_altered - clean)
     correlate with (naive_predicted - clean)? High correlation ⇒ layers
     13-31 behave AS IF nothing special happened downstream (consistent
     with, not proof of, physics-delegation robustness). Low/no
     correlation ⇒ real compensation or unpredictable disruption exists
     downstream — the discriminator fleeb83 actually asked for.

Usage:
    experiments/rl/../../training/.venv/bin/python \\
        experiments/rl/counterfactual_decay_probe.py \\
        --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \\
        --resume experiments/rl/checkpoints/g1i_zlk_phase1_v3_step500 \\
        --lora-r 32 --lora-alpha 64 --chain-phases 1 \\
        --target-layer 12 \\
        --out experiments/rl/results/counterfactual_decay_v3_step500.json
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import load_checkpoint
from experiments.rl.train_think_distill import ThinkChain
from experiments.rl.wkv_linear_probe import PROMPT_TEMPLATE, gen_examples
from experiments._common.results import save_result
from training.state_reg import DEFAULT_WORK_LAYERS


def _prefill_to_pre_final(loaded, think_marker, example: dict, phase_repeat_ticks: int):
    """Prompt prefill + entry cue + phase0 marker + (phase_repeat_ticks-1)
    NORMAL repeat ticks — mirrors wkv_linear_probe.py::collect_states up
    to, but not including, the final tick. Returns that state, to be
    branched into clean/altered final-tick variants."""
    tok = loaded.tokenizer
    device = loaded.device
    prompt = PROMPT_TEMPLATE.format(a=example["a"], b=example["b"])
    ids = tok.encode(prompt)
    state = loaded.new_state(batch=1)
    with torch.no_grad():
        for tid in ids:
            x = torch.tensor([[tid]], device=device)
            _, state = loaded.forward_stateful(x, state)
        marker0 = think_marker.step(0).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
        _, state = loaded.forward_stateful_embeds(marker0, state)
        marker1 = think_marker.step(1).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
        for _tick in range(phase_repeat_ticks - 1):
            _, state = loaded.forward_stateful_embeds(marker1, state)
    return state, marker1


@contextmanager
def _patch_final_tick(target_layer: int, capture: dict | None, w_altered: float | None):
    """Monkeypatches RWKV_Tmix_x070_infctx.forward (reference mirror of
    rwkvt/rwkv7/att.py:235-279, same convention as
    state_trajectory_probe.py::_capture_rkvwag — resync there if that
    file changes) so that ONLY self.layer_id == target_layer is touched:
    if `capture` is a dict, records state_before/r/k/v/w/a/kk into it;
    if `w_altered` is not None, replaces w with that constant (same shape,
    every channel) before RUN_RWKV7_INFCTX — the "altered recurrence law".
    All other layers run completely unmodified."""
    import os
    from rwkvt.rwkv7.att import RWKV_Tmix_x070_infctx
    from rwkvt.infctx_module import TimeMixState
    from rwkvt.operator.rwkvop import RUN_RWKV7_INFCTX

    orig_forward = RWKV_Tmix_x070_infctx.forward

    def patched(self, x, v_first, last_state, attention_mask=None):
        B, T, C = x.size()
        H = self.n_head
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        shift_state = last_state.shift_state
        wkv_state = last_state.wkv_state.clone().contiguous()
        xx = torch.concat((shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x
        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)
        shift_state = x[:, -1, :]

        r = self.receptance(xr)
        if os.environ["WKV"] == 'cuda':
            w = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        else:
            w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        if self.layer_id == target_layer:
            if capture is not None:
                capture["state_before"] = wkv_state.clone()
                capture["r"], capture["k"], capture["v"] = r.clone(), k.clone(), v.clone()
                capture["w"], capture["a"], capture["kk"] = w.clone(), a.clone(), kk.clone()
            if w_altered is not None:
                w = torch.full_like(w, w_altered)

        x_out, wkv_state = RUN_RWKV7_INFCTX(r, k, v, w, -kk, kk * a, wkv_state)
        x_out = self.ln_x(x_out.view(B * T, C)).view(B, T, C)
        x_out = x_out + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k)
                          .sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x_out = self.output(x_out * g)
        return x_out, v_first, TimeMixState(shift_state, wkv_state)

    RWKV_Tmix_x070_infctx.forward = patched
    try:
        yield
    finally:
        RWKV_Tmix_x070_infctx.forward = orig_forward


def _flatten_state(state, layers) -> torch.Tensor:
    return torch.cat([state.wkv[L].float().flatten() for L in layers])


def fit_probe(X: torch.Tensor, y: torch.Tensor, n_train: int) -> dict:
    """Same lstsq methodology as wkv_linear_probe.py::held_out_linear_probe,
    but also returns beta/x_mean/y_mean (needed here to recompute decoded
    values under a counterfactual state, not just report R²)."""
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    x_mean = Xtr.mean(0, keepdim=True)
    Xtr_c, Xte_c = Xtr - x_mean, Xte - x_mean
    y_mean = ytr.mean()
    ytr_c, yte_c = ytr - y_mean, yte - y_mean
    beta = torch.linalg.lstsq(Xtr_c, ytr_c.unsqueeze(-1)).solution.squeeze(-1)
    pred_te = (Xte_c @ beta)
    r2_te = (1.0 - F.mse_loss(pred_te, yte_c).item() / yte_c.var().item()
             if yte_c.var().item() > 1e-8 else float("nan"))
    return {"beta": beta, "x_mean": x_mean.squeeze(0), "y_mean": y_mean.item(), "held_out_r2": r2_te}


def decode(flat_state: torch.Tensor, probe: dict) -> float:
    return ((flat_state - probe["x_mean"]) @ probe["beta"]).item() + probe["y_mean"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--chain-phases", type=int, default=1,
                     help="Must match the checkpoint's trained M — see the "
                          "same warning in wkv_linear_probe.py.")
    ap.add_argument("--phase-repeat-ticks", type=int, default=8)
    ap.add_argument("--target-layer", type=int, default=12,
                     help="Must be the SMALLEST value in --work-layers, so "
                          "the other probed layers are genuinely downstream "
                          "(feedforward stack — layers before the target are "
                          "unaffected by construction, not a real test).")
    ap.add_argument("--work-layers", default=",".join(str(x) for x in DEFAULT_WORK_LAYERS))
    ap.add_argument("--w-altered", type=float, default=-3.0,
                     help="retention = exp(-exp(w)); -3.0 -> retention~=0.951, "
                          "chosen far from typical trained values.")
    ap.add_argument("--n-fit-examples", type=int, default=200)
    ap.add_argument("--n-test-examples", type=int, default=60)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--probe-target", default="xor_bit1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.work_layers.split(","))
    if args.target_layer != min(layers):
        print(f"[counterfactual_decay_probe] WARNING: --target-layer {args.target_layer} "
              f"is not the smallest of {layers} — layers below it in the stack "
              f"would be trivially unaffected either way, but this weakens the "
              f"'genuinely downstream' property for the others.")

    loaded = load_rwkv7(args.model, device=args.device, backend="peft",
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    think_marker = ThinkChain(loaded.n_embd, args.chain_phases).to(args.device)
    if args.resume is not None:
        try:
            step = load_checkpoint(args.resume, loaded, mlp_delta=think_marker)
            print(f"[counterfactual_decay_probe] resumed base weights + trained "
                  f"think_marker from {args.resume} at step {step}")
        except RuntimeError as e:
            print(f"[counterfactual_decay_probe] ABORTING — think_marker load failed "
                  f"(shape mismatch, likely wrong --chain-phases): {e}")
            return 1

    def collect_clean(examples):
        flats = []
        for ex in examples:
            state, _ = _prefill_to_pre_final(loaded, think_marker, ex, args.phase_repeat_ticks)
            marker1 = think_marker.step(1).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
            with torch.no_grad(), _patch_final_tick(args.target_layer, capture=None, w_altered=None):
                _, state = loaded.forward_stateful_embeds(marker1, state)
            flats.append(_flatten_state(state, layers).cpu())
        return torch.stack(flats)

    print(f"[counterfactual_decay_probe] fitting probe on {args.n_fit_examples} examples...")
    fit_examples = gen_examples(args.n_fit_examples, args.n_bits, args.seed)
    X_fit = collect_clean(fit_examples)
    bit_idx = int(args.probe_target.replace("xor_bit", ""))
    y_fit = torch.tensor([ex["xor_bits"][bit_idx] for ex in fit_examples], dtype=torch.float32)
    n_train = int(len(fit_examples) * 0.8)
    probe = fit_probe(X_fit, y_fit, n_train)
    print(f"[counterfactual_decay_probe] probe held-out R²={probe['held_out_r2']:.4f} "
          f"(reference: 0.3143 from the earlier wkv_linear_probe.py run)")

    # Which probed layer carries the most decodability weight, for context —
    # doesn't change target_layer (fixed by design, see --target-layer help).
    n_layers = len(layers)
    per_layer = X_fit.shape[1] // n_layers
    beta_mass = {L: probe["beta"][i * per_layer:(i + 1) * per_layer].abs().sum().item()
                 for i, L in enumerate(layers)}
    print(f"[counterfactual_decay_probe] |beta| mass per layer: {beta_mass}")

    print(f"[counterfactual_decay_probe] running counterfactual on "
          f"{args.n_test_examples} held-out examples, target_layer={args.target_layer}, "
          f"w_altered={args.w_altered} (retention={torch.exp(-torch.exp(torch.tensor(args.w_altered))).item():.4f})...")
    test_examples = gen_examples(args.n_test_examples, args.n_bits, args.seed + 999)
    rows = []
    from rwkvt.operator.rwkvop import RUN_RWKV7_INFCTX
    for ex in test_examples:
        state_pre, marker1 = _prefill_to_pre_final(loaded, think_marker, ex, args.phase_repeat_ticks)

        cap: dict = {}
        with torch.no_grad(), _patch_final_tick(args.target_layer, capture=cap, w_altered=None):
            _, state_clean = loaded.forward_stateful_embeds(marker1, state_pre)
        with torch.no_grad(), _patch_final_tick(args.target_layer, capture=None, w_altered=args.w_altered):
            _, state_altered = loaded.forward_stateful_embeds(marker1, state_pre)

        w_alt_full = torch.full_like(cap["w"], args.w_altered)
        with torch.no_grad():
            _, naive_layer_state = RUN_RWKV7_INFCTX(
                cap["r"], cap["k"], cap["v"], w_alt_full,
                -cap["kk"], cap["kk"] * cap["a"], cap["state_before"])

        flat_clean = _flatten_state(state_clean, layers).cpu()
        flat_altered = _flatten_state(state_altered, layers).cpu()
        flat_naive = flat_clean.clone()
        target_idx = layers.index(args.target_layer)
        flat_naive[target_idx * per_layer:(target_idx + 1) * per_layer] = naive_layer_state.float().flatten().cpu()

        rows.append({
            "clean": decode(flat_clean, probe),
            "actual_altered": decode(flat_altered, probe),
            "naive_predicted": decode(flat_naive, probe),
            "true_bit": ex["xor_bits"][bit_idx],
        })

    actual_shift = torch.tensor([r["actual_altered"] - r["clean"] for r in rows])
    naive_shift = torch.tensor([r["naive_predicted"] - r["clean"] for r in rows])
    if actual_shift.std() > 1e-8 and naive_shift.std() > 1e-8:
        corr = torch.corrcoef(torch.stack([actual_shift, naive_shift]))[0, 1].item()
    else:
        corr = float("nan")
    mean_abs_actual = actual_shift.abs().mean().item()
    mean_abs_naive = naive_shift.abs().mean().item()
    residual = (actual_shift - naive_shift).abs().mean().item()

    print(f"[counterfactual_decay_probe] mean|actual shift|={mean_abs_actual:.4f}  "
          f"mean|naive-predicted shift|={mean_abs_naive:.4f}  "
          f"corr(actual, naive)={corr:.4f}  mean|residual|={residual:.4f}")
    print("[counterfactual_decay_probe] interpretation: high corr + low residual => "
          "layers 13-31 behave as if nothing compensated (consistent with physics-"
          "delegation robustness at this channel); low/no corr => real downstream "
          "compensation or unpredictable disruption (level-1, fixed-interpreter-style "
          "robustness, or something else entirely) — fleeb83's actual discriminator.")

    save_result(
        args.out,
        {"model": str(args.model), "resume": str(args.resume) if args.resume else None,
         "target_layer": args.target_layer, "work_layers": list(layers),
         "w_altered": args.w_altered, "probe_target": args.probe_target,
         "probe_held_out_r2": probe["held_out_r2"], "beta_mass_per_layer": beta_mass,
         "n_test_examples": args.n_test_examples,
         "mean_abs_actual_shift": mean_abs_actual, "mean_abs_naive_shift": mean_abs_naive,
         "corr_actual_vs_naive": corr, "mean_abs_residual": residual,
         "rows": rows},
        experiment="counterfactual_decay_probe",
        hypothesis=["H25"],
        status="done",
        summary={"corr_actual_vs_naive": f"{corr:.4f}", "mean_abs_residual": f"{residual:.4f}"},
        model=str(args.model),
        script=str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
    )
    print(f"[counterfactual_decay_probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
