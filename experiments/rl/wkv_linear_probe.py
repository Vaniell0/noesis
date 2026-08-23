#!/usr/bin/env python3
"""wkv_linear_probe.py — the toy-scale held-out linear-probe methodology
from hypotheses/H25.md's micro-WKV experiments (2026-08-23,
experiments/A0_state_probe/micro_wkv.py), applied to a REAL ThinkChain
checkpoint's post-phase WKV state. This is the "directly testable next
step, cheap, no new mechanism needed" that section names: does a real
ThinkChain phase's resulting state hold the task's numeric answer as a
cleanly, linearly-decodable feature (like the toy controller did), or
does it stay entangled with the rest of the semantic representation
(the more likely outcome, since a real marker must also serve ordinary
language modeling — see H25.md's "Honest scope limit")?

Task: bitwise XOR of two 4-bit operands (matches the style of the fixed
`xor` diagnostic prompt already used in state_trajectory_probe.py, just
parameterized over many random operand pairs instead of one fixed pair).
Ground truth: the 4-bit XOR result, as an integer 0-15 and as 4 separate
bits (probe both — an integer target forces the probe to reconstruct a
specific base-2 encoding, per-bit targets are the more natural
decomposition of what the state might actually hold).

Usage (same checkpoint-loading convention as state_trajectory_probe.py):
    python experiments/rl/wkv_linear_probe.py \\
        --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \\
        --resume experiments/rl/runs/archive/g1i_think_distill_zlk_phase1_v3/ckpt_step000500 \\
        --lora-r 32 --lora-alpha 64 --chain-phases 1 \\
        --n-examples 200 --out experiments/rl/results/wkv_linear_probe_v3_step500.json
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import load_checkpoint
from experiments.rl.train_think_distill import ThinkChain
from experiments._common.results import save_result
from training.state_reg import DEFAULT_WORK_LAYERS

PROMPT_TEMPLATE = (
    "You are a precise reasoning assistant. Work step by step.\n\n"
    "Compute the bitwise XOR of {a} and {b}.\n\n<think>\n"
)


def _random_bits(n_bits: int, rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(n_bits))


def gen_examples(n: int, n_bits: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        a_bits = _random_bits(n_bits, rng)
        b_bits = _random_bits(n_bits, rng)
        xor_val = int(a_bits, 2) ^ int(b_bits, 2)
        examples.append({
            "a": a_bits, "b": b_bits,
            "xor_int": xor_val,
            "xor_bits": [int(c) for c in format(xor_val, f"0{n_bits}b")],
        })
    return examples


def collect_states(loaded, think_marker, examples: list[dict], layers,
                    phase_repeat_ticks: int) -> torch.Tensor:
    """Runs each example's prompt through prefill + the trained ThinkChain
    entry cue + phase 0, repeated phase_repeat_ticks times (matching the
    checkpoint's own training config), returns [n_examples, len(layers)*head_size^2]
    flattened final WKV states."""
    tok = loaded.tokenizer
    device = loaded.device
    flat_states = []
    with torch.no_grad():
        for i, ex in enumerate(examples):
            prompt = PROMPT_TEMPLATE.format(a=ex["a"], b=ex["b"])
            ids = tok.encode(prompt)
            state = loaded.new_state(batch=1)
            for tid in ids:
                x = torch.tensor([[tid]], device=device)
                _, state = loaded.forward_stateful(x, state)
            marker0 = think_marker.step(0).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
            _, state = loaded.forward_stateful_embeds(marker0, state)
            marker1 = think_marker.step(1).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
            for _tick in range(phase_repeat_ticks):
                _, state = loaded.forward_stateful_embeds(marker1, state)
            layer_flats = [state.wkv[L].float().flatten() for L in layers]
            flat_states.append(torch.cat(layer_flats))
            if (i + 1) % 20 == 0:
                print(f"  [wkv_linear_probe] {i + 1}/{len(examples)} examples processed")
    return torch.stack(flat_states)


def held_out_linear_probe(X: torch.Tensor, y: torch.Tensor, n_train: int) -> dict:
    """Same methodology validated at toy scale in micro_wkv.py: fit on
    the first n_train rows, evaluate held-out R² on the rest, center
    using TRAIN statistics only (no leakage)."""
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]
    x_mean = Xtr.mean(0, keepdim=True)
    Xtr_c, Xte_c = Xtr - x_mean, Xte - x_mean
    y_mean = ytr.mean()
    ytr_c, yte_c = ytr - y_mean, yte - y_mean
    beta = torch.linalg.lstsq(Xtr_c, ytr_c.unsqueeze(-1)).solution
    pred_tr = (Xtr_c @ beta).squeeze(-1)
    pred_te = (Xte_c @ beta).squeeze(-1)
    r2_tr = 1.0 - F.mse_loss(pred_tr, ytr_c).item() / ytr_c.var().item()
    r2_te = (1.0 - F.mse_loss(pred_te, yte_c).item() / yte_c.var().item()
              if yte_c.var().item() > 1e-8 else float("nan"))
    return {"in_sample_r2": r2_tr, "held_out_r2": r2_te}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--chain-phases", type=int, default=1,
                     help="Must match the checkpoint's trained M — mismatched "
                          "shape silently falls back to random markers. Always "
                          "check the 'resumed base weights + trained think_marker' "
                          "log line, not just that --resume didn't error "
                          "(this exact mistake happened once already tonight).")
    ap.add_argument("--phase-repeat-ticks", type=int, default=8)
    ap.add_argument("--work-layers", default=",".join(str(x) for x in DEFAULT_WORK_LAYERS))
    ap.add_argument("--n-examples", type=int, default=200)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.work_layers.split(","))
    loaded = load_rwkv7(args.model, device=args.device, backend="peft",
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    think_marker = ThinkChain(loaded.n_embd, args.chain_phases).to(args.device)
    if args.resume is not None:
        try:
            step = load_checkpoint(args.resume, loaded, mlp_delta=think_marker)
            print(f"[wkv_linear_probe] resumed base weights + trained think_marker "
                  f"from {args.resume} at step {step}")
        except RuntimeError as e:
            step = load_checkpoint(args.resume, loaded, mlp_delta=None)
            print(f"[wkv_linear_probe] WARNING: think_marker NOT loaded "
                  f"(shape mismatch, likely wrong --chain-phases): {e}")
            print("[wkv_linear_probe] ABORTING — a probe against random markers "
                  "would be meaningless.")
            return 1

    examples = gen_examples(args.n_examples, args.n_bits, args.seed)
    print(f"[wkv_linear_probe] generated {len(examples)} XOR examples, "
          f"{args.n_bits}-bit operands")
    states = collect_states(loaded, think_marker, examples, layers, args.phase_repeat_ticks).cpu()
    print(f"[wkv_linear_probe] collected states: {tuple(states.shape)}")

    n_train = int(len(examples) * 0.8)
    results = {}
    xor_int = torch.tensor([ex["xor_int"] for ex in examples], dtype=torch.float32)
    results["xor_int"] = held_out_linear_probe(states, xor_int, n_train)
    print(f"  xor_int: in-sample R²={results['xor_int']['in_sample_r2']:.4f}  "
          f"HELD-OUT R²={results['xor_int']['held_out_r2']:.4f}")
    for bit_idx in range(args.n_bits):
        bit_target = torch.tensor([ex["xor_bits"][bit_idx] for ex in examples], dtype=torch.float32)
        key = f"xor_bit{bit_idx}"
        results[key] = held_out_linear_probe(states, bit_target, n_train)
        print(f"  {key}: in-sample R²={results[key]['in_sample_r2']:.4f}  "
              f"HELD-OUT R²={results[key]['held_out_r2']:.4f}")

    save_result(
        args.out,
        {"model": str(args.model), "resume": str(args.resume) if args.resume else None,
         "work_layers": list(layers), "chain_phases": args.chain_phases,
         "n_examples": args.n_examples, "n_bits": args.n_bits, "n_train": n_train,
         "results": results},
        experiment="wkv_linear_probe",
        hypothesis=["H25"],
        status="done",
        summary={"held_out_r2_xor_int": f"{results['xor_int']['held_out_r2']:.4f}",
                 "held_out_r2_bits_mean": f"{sum(results[f'xor_bit{i}']['held_out_r2'] for i in range(args.n_bits)) / args.n_bits:.4f}"},
        model=str(args.model),
        script=str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
    )
    print(f"[wkv_linear_probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
