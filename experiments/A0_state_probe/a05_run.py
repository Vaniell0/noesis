#!/usr/bin/env python3
"""A0.5 — causal state intervention CLI runner (extended methodology).

Per-seed loop:

1. Prefill prompt → initial state.
2. Nucleus-sampled decode of ``--max-new-tokens`` tokens. At k evenly-spaced
   checkpoints inside decode, snapshot ``(state, next_token, prev_state)``
   before the next forward. If ``--cross-prompt`` is set, the checkpoints
   of a reference decode of that other prompt are also snapshotted (same
   step indices, same seed) as the donor for cross-prompt state swaps.
3. At each checkpoint, for each corruption, run the paired forwards
   ``forward([tok], clean)`` vs ``forward([tok], corrupt(clean))`` and
   compute the point-metrics ``KL_next``, ``argmax_flip``, ``entropy_change``,
   ``rank_shift``. If ``--continuation-steps N > 0``, greedy-decode N further
   tokens from both states and compute the trajectory-metrics
   ``token_overlap_N`` and ``cum_KL_N``.

Corruption menu (all in ``a05_intervene``):

- ``noise_floor`` — identical clean state, two forwards. bf16 noise baseline.
- ``gauss(σ)`` — additive Gaussian per layer, sigma sweep.
- ``scale(α)`` — multiplicative rescale of all WKV.
- ``zero_layer(i)`` — zero WKV at one layer. If ``--full-layer-profile``,
  iterates every layer; else uses ``--sample-layers``.
- ``shuffle_heads(i)`` — permute head dimension at one layer.
- ``freeze_prev`` — replace WKV with the previous decode-step snapshot.
- ``cross_prompt`` — replace WKV with a donor snapshot from a different
  prompt's decode at the matching step index. Only when ``--cross-prompt``.

Output: ``<out>/seed_<i>.json`` per seed and ``<out>/summary.json`` with
the runner config. Aggregation (Cohen's d, layer profile plot, σ-response
fit) is done in a separate analysis step (or by hand in ``results.md``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

import a05_intervene as C
import prompts as PROMPT_BANK
from probe import _sample_top_p, load_model


# --------------------------------------------------------------------------- #
# Decode + snapshot
# --------------------------------------------------------------------------- #

def _checkpoint_positions(max_tokens: int, k: int) -> List[int]:
    """k evenly-spaced positions inside [1, max_tokens - 1] (avoid step 0,
    which has no prev; and the very last step, which has no continuation)."""
    positions = sorted(
        {
            max(1, min(max_tokens - 1, round((i + 1) * max_tokens / (k + 1))))
            for i in range(k)
        }
    )
    return positions


def _decode_with_checkpoints(
    model,
    tokenizer,
    prompt: str,
    ck_set: set,
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    """Prefill + decode. At each step in ``ck_set``, snapshot the state
    *before* the forward, the next token that would be fed, the logits
    just produced (starting point for a continuation), and the previous
    step's state snapshot (for freeze_prev)."""
    torch.manual_seed(seed)

    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()

    logits, state = model.forward(ids, None)

    checkpoints: List[Dict[str, Any]] = []
    prev_snapshot: Optional[List[torch.Tensor]] = None

    for step in range(max_tokens):
        if logits.dim() > 1:
            logits = logits.reshape(-1)
        next_id = _sample_top_p(logits, temperature=temperature, top_p=top_p)

        if step in ck_set:
            checkpoints.append(
                {
                    "step": step,
                    "next_id": next_id,
                    "state": C.snapshot(state),
                    "prev_state": prev_snapshot,
                    "logits_at_step": logits.detach().to(torch.float32).cpu().clone(),
                }
            )

        prev_snapshot = C.snapshot(state)
        logits, state = model.forward([next_id], state)

    return checkpoints


# --------------------------------------------------------------------------- #
# Metrics for a single (clean_state, corrupt_state) pair
# --------------------------------------------------------------------------- #

def _paired_metrics(
    model,
    next_id: int,
    clean_state: List[torch.Tensor],
    corrupt_state: List[torch.Tensor],
    temperature: float,
    continuation_steps: int,
) -> Dict[str, Any]:
    """Paired single forward → point metrics. Optional greedy continuation
    from both states → trajectory metrics."""
    # Snapshot clean before the clean forward (forward mutates state).
    clean_for_forward = C.snapshot(clean_state)
    logits_c, state_c_after = model.forward([next_id], clean_for_forward)
    logits_p, state_p_after = model.forward([next_id], corrupt_state)

    out: Dict[str, Any] = {
        "kl_next": C.kl_next(logits_c, logits_p, temperature=temperature),
        "argmax_flip": C.argmax_flip(logits_c, logits_p),
        "entropy_change": C.entropy_change(logits_c, logits_p, temperature=temperature),
        "rank_shift": C.rank_shift(logits_c, logits_p),
    }

    if continuation_steps > 0:
        # Snapshot both post-forward states before continuation (mutation).
        tokens_c, logits_seq_c = C.greedy_continue(
            model, C.snapshot(state_c_after), logits_c, continuation_steps
        )
        tokens_p, logits_seq_p = C.greedy_continue(
            model, C.snapshot(state_p_after), logits_p, continuation_steps
        )
        traj = C.trajectory_metrics(
            tokens_c, tokens_p, logits_seq_c, logits_seq_p, temperature=temperature
        )
        out.update(traj)

    return out


# --------------------------------------------------------------------------- #
# Run every corruption at one checkpoint
# --------------------------------------------------------------------------- #

def _corruptions_at_checkpoint(
    model,
    cp: Dict[str, Any],
    sigmas: List[float],
    scales: List[float],
    layer_indices: List[int],
    corrupt_seed: int,
    temperature: float,
    continuation_steps: int,
    donor_state: Optional[List[torch.Tensor]] = None,
    layer_profile_continuation: bool = False,
) -> List[Dict[str, Any]]:
    """Every corruption for one checkpoint. Layer-scale corruptions
    (zero_layer, shuffle_heads) skip the expensive continuation by default
    since ``layer_indices`` may be large (full profile of 24 layers)."""
    gen = torch.Generator().manual_seed(corrupt_seed)
    clean = cp["state"]
    next_id = cp["next_id"]
    out: List[Dict[str, Any]] = []

    layer_cont = continuation_steps if layer_profile_continuation else 0

    # (1) Noise floor — identical state, two forwards. bf16 baseline.
    m = _paired_metrics(model, next_id, clean, C.snapshot(clean),
                        temperature, continuation_steps)
    out.append({"type": "noise_floor", **m})

    # (2) Gaussian sigma sweep.
    for sigma in sigmas:
        cor = C.corrupt_gauss(clean, sigma, generator=gen)
        m = _paired_metrics(model, next_id, clean, cor, temperature, continuation_steps)
        out.append({"type": "gauss", "sigma": sigma, **m})

    # (3) Multiplicative scale sweep.
    for alpha in scales:
        cor = C.corrupt_scale(clean, alpha)
        m = _paired_metrics(model, next_id, clean, cor, temperature, continuation_steps)
        out.append({"type": "scale", "alpha": alpha, **m})

    # (4) Zero-layer profile (full 24 layers or sampled).
    for li in layer_indices:
        cor = C.corrupt_zero_layer(clean, li)
        m = _paired_metrics(model, next_id, clean, cor, temperature, layer_cont)
        out.append({"type": "zero_layer", "layer": li, **m})

    # (5) Shuffle-heads profile (same layer list).
    for li in layer_indices:
        cor = C.corrupt_shuffle_heads(clean, li, generator=gen)
        m = _paired_metrics(model, next_id, clean, cor, temperature, layer_cont)
        out.append({"type": "shuffle_heads", "layer": li, **m})

    # (6) Freeze — replace WKV with previous decode step's snapshot.
    if cp["prev_state"] is not None:
        cor = C.corrupt_freeze(clean, cp["prev_state"])
        m = _paired_metrics(model, next_id, clean, cor, temperature, continuation_steps)
        out.append({"type": "freeze_prev", **m})

    # (7) Cross-prompt state swap.
    if donor_state is not None:
        cor = C.corrupt_cross(clean, donor_state)
        m = _paired_metrics(model, next_id, clean, cor, temperature, continuation_steps)
        out.append({"type": "cross_prompt", **m})

    return out


# --------------------------------------------------------------------------- #
# Cross-prompt donor decode
# --------------------------------------------------------------------------- #

def _decode_donor(
    model,
    tokenizer,
    prompt: str,
    ck_set: set,
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> Dict[int, List[torch.Tensor]]:
    """Decode the cross-prompt with the same seed, snapshot state at each
    checkpoint step. Returns ``{step: state_snapshot}``. This gives us
    "the state a fresh model would have at step t if it had been reading
    the OTHER prompt" — the substitution material for ``corrupt_cross``.
    """
    cps = _decode_with_checkpoints(
        model, tokenizer, prompt,
        ck_set=ck_set, max_tokens=max_tokens, seed=seed,
        temperature=temperature, top_p=top_p,
    )
    return {cp["step"]: cp["state"] for cp in cps}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="RWKV-7 causal state intervention probe (A0.5, extended)."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompt", required=True, choices=list(PROMPT_BANK.ALL.keys()),
        help="Primary prompt from the shared A0 bank.",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--k-checkpoints", type=int, default=5)
    ap.add_argument(
        "--sigmas", default="0.005,0.01,0.02,0.05,0.1,0.2",
        help="Comma-separated Gaussian σ scales (fractional norm shift).",
    )
    ap.add_argument(
        "--scales", default="0.5,1.5,2.0",
        help="Comma-separated multiplicative α values. 1.0 is redundant with noise_floor.",
    )
    ap.add_argument(
        "--sample-layers", default="0,8,16",
        help="Comma-separated layer indices (ignored if --full-layer-profile).",
    )
    ap.add_argument(
        "--full-layer-profile", action="store_true",
        help="Loop over every layer for zero_layer + shuffle_heads. Enables H8-B (layer localisation) directly.",
    )
    ap.add_argument(
        "--layer-profile-continuation", action="store_true",
        help="Also compute trajectory metrics on layer-profile corruptions. Expensive.",
    )
    ap.add_argument(
        "--cross-prompt", default=None, choices=[*PROMPT_BANK.ALL.keys(), None],
        help="If set, decode this prompt as a donor and add cross_prompt corruption at each checkpoint.",
    )
    ap.add_argument(
        "--continuation-steps", type=int, default=8,
        help="Greedy-decode N tokens from clean+corrupt states for token_overlap_N + cum_KL_N. 0 disables.",
    )
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    sample_layers = [int(x) for x in args.sample_layers.split(",") if x.strip()]

    os.makedirs(args.out, exist_ok=True)
    prompt_text = PROMPT_BANK.ALL[args.prompt]

    ck_positions = _checkpoint_positions(args.max_new_tokens, args.k_checkpoints)
    ck_set = set(ck_positions)

    print(
        f"[a05] model={args.model} prompt={args.prompt} cross={args.cross_prompt} "
        f"seeds={args.seeds} tokens={args.max_new_tokens} k={args.k_checkpoints} "
        f"cont_N={args.continuation_steps} "
        f"sigmas={sigmas} scales={scales} "
        f"layers={'ALL' if args.full_layer_profile else sample_layers} "
        f"device={args.device} out={args.out}",
        file=sys.stderr, flush=True,
    )

    t0 = time.time()
    model, tokenizer = load_model(args.model, device=args.device)
    print(f"[a05] model loaded in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    # Resolve layer index list: probe the first checkpoint to count layers.
    # (n_layer = len(state) // 3; probe expects a real checkpoint.)
    if args.full_layer_profile:
        # Do a tiny probe: prefill 1 token to get initial state, count layers.
        dummy_logits, dummy_state = model.forward([0], None)
        n_layer = len(dummy_state) // 3
        layer_indices = list(range(n_layer))
        del dummy_logits, dummy_state
        print(f"[a05] full layer profile: {n_layer} layers", file=sys.stderr, flush=True)
    else:
        layer_indices = sample_layers

    # Optional cross-prompt donor decode (once, seed 0).
    donor_by_step: Dict[int, List[torch.Tensor]] = {}
    if args.cross_prompt:
        print(f"[a05] cross-prompt donor decode ({args.cross_prompt}) ...",
              file=sys.stderr, flush=True)
        dt0 = time.time()
        donor_by_step = _decode_donor(
            model, tokenizer, PROMPT_BANK.ALL[args.cross_prompt],
            ck_set=ck_set, max_tokens=args.max_new_tokens, seed=0,
            temperature=args.temperature, top_p=args.top_p,
        )
        print(
            f"[a05] donor decode done in {time.time() - dt0:.1f}s "
            f"({len(donor_by_step)} donor checkpoints)",
            file=sys.stderr, flush=True,
        )

    for seed in range(args.seeds):
        print(f"[a05] seed {seed} — decoding {args.max_new_tokens} tokens ...",
              file=sys.stderr, flush=True)
        s0 = time.time()
        cps = _decode_with_checkpoints(
            model, tokenizer, prompt_text,
            ck_set=ck_set, max_tokens=args.max_new_tokens, seed=seed,
            temperature=args.temperature, top_p=args.top_p,
        )
        decode_wall = time.time() - s0
        print(
            f"[a05] seed {seed} decode {decode_wall:.1f}s; "
            f"{len(cps)} checkpoints; running corruptions ...",
            file=sys.stderr, flush=True,
        )

        per_cp: List[Dict[str, Any]] = []
        c0 = time.time()
        for cp_idx, cp in enumerate(cps):
            donor = donor_by_step.get(cp["step"]) if donor_by_step else None
            results = _corruptions_at_checkpoint(
                model, cp,
                sigmas=sigmas,
                scales=scales,
                layer_indices=layer_indices,
                corrupt_seed=1000 * seed + cp_idx,
                temperature=args.temperature,
                continuation_steps=args.continuation_steps,
                donor_state=donor,
                layer_profile_continuation=args.layer_profile_continuation,
            )
            per_cp.append(
                {
                    "checkpoint_step": cp["step"],
                    "next_id": cp["next_id"],
                    "corruptions": results,
                }
            )
            print(
                f"[a05]   cp{cp_idx} step={cp['step']} "
                f"({len(results)} corruptions) done",
                file=sys.stderr, flush=True,
            )
        corrupt_wall = time.time() - c0

        seed_out = {
            "seed": seed,
            "decode_wall_s": decode_wall,
            "corrupt_wall_s": corrupt_wall,
            "checkpoints": per_cp,
        }
        with open(os.path.join(args.out, f"seed_{seed}.json"), "w") as f:
            json.dump(seed_out, f, indent=None, separators=(",", ":"))
        print(
            f"[a05] seed {seed} done — decode {decode_wall:.1f}s, "
            f"corruptions {corrupt_wall:.1f}s",
            file=sys.stderr, flush=True,
        )

    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "cross_prompt": args.cross_prompt,
        "seeds": args.seeds,
        "max_new_tokens": args.max_new_tokens,
        "k_checkpoints": args.k_checkpoints,
        "checkpoint_positions": ck_positions,
        "sigmas": sigmas,
        "scales": scales,
        "layer_indices": layer_indices,
        "full_layer_profile": args.full_layer_profile,
        "layer_profile_continuation": args.layer_profile_continuation,
        "continuation_steps": args.continuation_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "device": args.device,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[a05] wrote {args.out}/summary.json", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
