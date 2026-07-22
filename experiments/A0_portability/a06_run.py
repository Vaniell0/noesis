"""A0.6 — intra-model state-transplant runner.

Runs the design fixed in ``README.md``: 3 prompt pairs × 2 directions
× 3 injection depths × 2 swap modes, on a single RWKV-7 checkpoint.
Because the donor and recipient are the same model, the full state
list (including shift buffers) is compatible for transfer; A0.6 uses
full-state swap (``corrupt_cross`` / ``corrupt_cross_layers``) rather
than the WKV-only path A0.7 must use for cross-model transfer.

Directions realise the ROADMAP A0.6 spec that each pair be probed
symmetrically:

- ``AB`` — donor state = state_A (prompt A processed), recipient
  prompt = prompt_B. Tests whether A-shaped state leaks into a
  B-prompted continuation.
- ``BA`` — mirror: donor = state_B, recipient = prompt_A. Guards
  against an asymmetric artefact (e.g. prompt A's structure alone
  attracting the continuation regardless of injected state).

Per cell we produce three continuations:

- ``clean_donor`` — greedy decode from the donor's clean state.
  Computed once per pair (used across cells and both directions).
- ``clean_recipient`` — greedy decode from the recipient's clean
  state. Same reuse discipline.
- ``cross`` — the transplant continuation. Depends on direction,
  depth, and mode.

Metrics are computed against these three continuations and dumped as
JSON. A human-readable summary lands in ``results_a06.md`` after all
cells finish (or as an incremental in-progress table with ``--summary``).

Depth vocabulary (D2 in README):

- ``before_B`` — state is set to ``state_A`` **before** feeding any of
  prompt B. Model then processes prompt B tokens on top of A's state.
  Only well-defined with ``mode=full`` (hotspot mode has no baseline
  state to selectively swap into, since prompt B hasn't been processed
  yet); the ``mode=hotspot`` variant is skipped here.
- ``mid_B``    — process first ``len(prompt_B) // 2`` tokens with a
  fresh state, apply the swap, then process the remaining half.
- ``after_B``  — process all of prompt B minus its last token with a
  fresh state, apply the swap, then feed the last token so the
  post-swap logits are consistent with the swapped state.

Layer selection for hotspot mode (D3 in README):
``training/state_reg.py::DEFAULT_WORK_LAYERS`` — {12, 16, 20} on the
0.4B 24-layer checkpoints. Different checkpoints (e.g. 1.5B) would
need a matched hotspot set; A0.6 stays at 0.4B per D5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------- #
# Local imports — reach into A0_state_probe for probe / intervention utils
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
# Order matters: A0_portability's own metrics.py must win over the
# shadowing metrics.py under A0_state_probe. Insert A0_portability LAST
# so it lands at sys.path[0].
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(_REPO / "experiments" / "A0_state_probe"))
sys.path.insert(0, str(_HERE))

from a05_intervene import (  # noqa: E402
    snapshot,
    corrupt_cross,
    corrupt_cross_layers,
    greedy_continue,
    kl_next,
)
from probe import load_model  # noqa: E402
from state_reg import DEFAULT_WORK_LAYERS  # noqa: E402

from metrics import (  # noqa: E402
    alignment_vs_donor,
    first_divergence_step,
    surface_garble,
    task_lexicon_hit_rate,
    topk_jaccard,
    topk_jaccard_trajectory,
)


# --------------------------------------------------------------------------- #
# Task-pair loader
# --------------------------------------------------------------------------- #

def load_pairs(path: Path) -> "list[dict]":
    """Read ``tasks.jsonl`` — one JSON object per line."""
    pairs: "list[dict]" = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #

def prefill(model, tokenizer, prompt: str) -> "tuple[torch.Tensor, list, list[int]]":
    """Feed the whole prompt, return ``(logits, state, prompt_ids)``.

    ``logits`` is the next-token distribution the model produced after
    processing the last prompt token. ``state`` is the flat state list
    the rwkv package returned; caller may freely mutate a snapshot of
    it. Prompt ids are handed back so the runner can compute a
    consistent ``mid`` split without re-tokenising.
    """
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    logits, state = model.forward(ids, None)
    return logits, state, ids


def continue_greedy(
    model,
    state,
    initial_logits: torch.Tensor,
    n_tokens: int,
) -> "tuple[list[int], list[torch.Tensor]]":
    """Wrapper on ``greedy_continue`` returning materialised outputs.

    ``greedy_continue`` returns ``(tokens, all_logits)`` where
    ``all_logits`` is a list of length ``n_tokens + 1`` (the initial
    logits plus one per generated token). The first entry corresponds
    to the model's choice of the first generated token; subsequent
    entries are the distributions produced by each subsequent forward.
    """
    tokens, all_logits = greedy_continue(model, snapshot(state),
                                          initial_logits, n_tokens)
    return tokens, all_logits


# --------------------------------------------------------------------------- #
# Cross-state construction — the D2 depth × D3 mode matrix
# --------------------------------------------------------------------------- #

def build_cross_state(
    model,
    prompt_B_ids: "list[int]",
    donor_state,
    depth: str,
    mode: str,
    hotspot_layers: "Sequence[int]",
):
    """Return ``(logits, state)`` after applying the depth × mode
    transplant to prompt B's processing path.

    The returned pair is what the greedy decoder should feed from —
    logits are those the model produced *after* the swap has taken
    effect, so first-token selection is consistent with the swapped
    state.
    """
    if depth == "before_B":
        if mode != "full":
            raise ValueError(
                "before_B depth is only defined for mode=full — "
                "hotspot has no baseline state to selectively swap into"
            )
        state = snapshot(donor_state)
        logits, state = model.forward(prompt_B_ids, state)
        return logits, state

    if depth == "mid_B":
        mid = max(1, len(prompt_B_ids) // 2)
        _, state = model.forward(prompt_B_ids[:mid], None)
        if mode == "full":
            state = corrupt_cross(state, donor_state)
        elif mode == "hotspot":
            state = corrupt_cross_layers(state, donor_state, list(hotspot_layers))
        else:
            raise ValueError(f"unknown mode={mode!r}")
        logits, state = model.forward(prompt_B_ids[mid:], state)
        return logits, state

    if depth == "after_B":
        if len(prompt_B_ids) < 2:
            raise ValueError("after_B requires prompt B length ≥ 2")
        _, state = model.forward(prompt_B_ids[:-1], None)
        if mode == "full":
            state = corrupt_cross(state, donor_state)
        elif mode == "hotspot":
            state = corrupt_cross_layers(state, donor_state, list(hotspot_layers))
        else:
            raise ValueError(f"unknown mode={mode!r}")
        logits, state = model.forward([prompt_B_ids[-1]], state)
        return logits, state

    raise ValueError(f"unknown depth={depth!r}")


# --------------------------------------------------------------------------- #
# Metrics glue — combine tokens + logits + text into the per-cell block
# --------------------------------------------------------------------------- #

def cumulative_kl(logits_ref: "list[torch.Tensor]",
                  logits_test: "list[torch.Tensor]") -> float:
    """Sum of per-step KL(softmax(ref) || softmax(test)) over the
    trajectory. Skips the initial-logits entry so we measure divergence
    on the *decoded* portion, not on the first-token pick that both
    trajectories may share by construction."""
    n = min(len(logits_ref), len(logits_test))
    total = 0.0
    for i in range(1, n):
        total += kl_next(logits_ref[i], logits_test[i])
    return total


def compute_cell_metrics(
    prompt_donor_text: str,
    prompt_recipient_text: str,
    tokens_clean_donor: "list[int]",
    tokens_clean_recipient: "list[int]",
    tokens_cross: "list[int]",
    logits_clean_donor: "list[torch.Tensor]",
    logits_clean_recipient: "list[torch.Tensor]",
    logits_cross: "list[torch.Tensor]",
    tokenizer,
) -> "dict[str, object]":
    """Bundle all 5 portability metrics for one cell.

    Donor is the side whose state gets transplanted; recipient is the
    side whose prompt is being processed when the swap lands. In
    direction AB the donor is A and recipient is B; in BA it flips.
    All metric names are donor/recipient-relative so the JSON schema
    stays direction-agnostic.
    """
    text_clean_donor = tokenizer.decode(tokens_clean_donor)
    text_clean_recipient = tokenizer.decode(tokens_clean_recipient)
    text_cross = tokenizer.decode(tokens_cross)

    lex_hit_donor_cross = task_lexicon_hit_rate(prompt_donor_text, text_cross)
    lex_hit_recipient_cross = task_lexicon_hit_rate(prompt_recipient_text,
                                                    text_cross)
    lex_hit_donor_null = task_lexicon_hit_rate(prompt_donor_text,
                                               text_clean_recipient)
    lex_hit_recipient_null = task_lexicon_hit_rate(prompt_recipient_text,
                                                   text_clean_donor)

    cum_kl_cross_to_donor = cumulative_kl(logits_clean_donor, logits_cross)
    cum_kl_cross_to_recipient = cumulative_kl(logits_clean_recipient,
                                              logits_cross)
    alignment = alignment_vs_donor(cum_kl_cross_to_donor,
                                   cum_kl_cross_to_recipient)

    fds = first_divergence_step(tokens_clean_recipient, tokens_cross)
    jaccard_traj_k10 = topk_jaccard_trajectory(logits_clean_recipient,
                                               logits_cross, k=10)
    jaccard_traj_k5 = topk_jaccard_trajectory(logits_clean_recipient,
                                              logits_cross, k=5)

    garble_cross = surface_garble(text_cross)
    garble_clean_donor = surface_garble(text_clean_donor)
    garble_clean_recipient = surface_garble(text_clean_recipient)

    return {
        "text": {
            "clean_donor": text_clean_donor,
            "clean_recipient": text_clean_recipient,
            "cross": text_cross,
        },
        "tokens": {
            "clean_donor": tokens_clean_donor,
            "clean_recipient": tokens_clean_recipient,
            "cross": tokens_cross,
        },
        "lexicon": {
            "hit_donor_in_cross": lex_hit_donor_cross,
            "hit_recipient_in_cross": lex_hit_recipient_cross,
            "hit_donor_in_clean_recipient_null": lex_hit_donor_null,
            "hit_recipient_in_clean_donor_null": lex_hit_recipient_null,
            "delta_hit_donor_vs_null": lex_hit_donor_cross - lex_hit_donor_null,
        },
        "alignment": alignment,
        "first_divergence_step": fds,
        "topk_jaccard": {
            "k5_mean": (sum(jaccard_traj_k5) / len(jaccard_traj_k5))
                       if jaccard_traj_k5 else 0.0,
            "k10_mean": (sum(jaccard_traj_k10) / len(jaccard_traj_k10))
                        if jaccard_traj_k10 else 0.0,
            "k10_trajectory": jaccard_traj_k10,
        },
        "surface_garble": {
            "cross": garble_cross,
            "clean_donor": garble_clean_donor,
            "clean_recipient": garble_clean_recipient,
        },
    }


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

MODEL_ALIAS = {
    "world-0.4b": "BlinkDL/rwkv-7-world:RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
    "g1d-0.4b":   "BlinkDL/rwkv7-g1:rwkv7-g1d-0.4b-20260210-ctx8192.pth",
}


def run_pair(
    model,
    tokenizer,
    pair: "dict",
    n_tokens: int,
    depths: "Sequence[str]",
    modes: "Sequence[str]",
    directions: "Sequence[str]",
    hotspot_layers: "Sequence[int]",
    verbose: bool = False,
) -> "list[dict]":
    """Produce one JSON block per (direction, depth, mode) cell.

    Prefills for A and B are computed once and reused across every
    direction × depth × mode; only the cross continuation varies. The
    ``directions`` argument selects which donor→recipient combinations
    to run: ``"AB"`` (donor=A, recipient=B) and/or ``"BA"``. Symmetric
    coverage — mandated by the ROADMAP A0.6 spec — is ``["AB", "BA"]``.
    """
    prompt_A = pair["prompt_A"]
    prompt_B = pair["prompt_B"]
    pair_id = pair["pair_id"]

    t0 = time.time()
    logits_A, state_A, prompt_A_ids = prefill(model, tokenizer, prompt_A)
    if verbose:
        print(f"[{pair_id}] prefill A: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    logits_B, state_B, prompt_B_ids = prefill(model, tokenizer, prompt_B)
    if verbose:
        print(f"[{pair_id}] prefill B: {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    tokens_A, logits_A_seq = continue_greedy(model, state_A, logits_A, n_tokens)
    if verbose:
        print(f"[{pair_id}] clean_A greedy: {time.time() - t0:.1f}s",
              flush=True)

    t0 = time.time()
    tokens_B, logits_B_seq = continue_greedy(model, state_B, logits_B, n_tokens)
    if verbose:
        print(f"[{pair_id}] clean_B greedy: {time.time() - t0:.1f}s",
              flush=True)

    # direction alias → (donor prompt text, donor state, donor clean tokens,
    #                    donor clean logits, recipient prompt text,
    #                    recipient prompt ids, recipient clean tokens,
    #                    recipient clean logits)
    direction_bindings = {
        "AB": (prompt_A, state_A, tokens_A, logits_A_seq,
               prompt_B, prompt_B_ids, tokens_B, logits_B_seq),
        "BA": (prompt_B, state_B, tokens_B, logits_B_seq,
               prompt_A, prompt_A_ids, tokens_A, logits_A_seq),
    }

    results: "list[dict]" = []
    for direction in directions:
        if direction not in direction_bindings:
            raise ValueError(f"unknown direction={direction!r}")
        (donor_text, donor_state, donor_tokens, donor_logits,
         recipient_text, recipient_ids, recipient_tokens,
         recipient_logits) = direction_bindings[direction]

        for depth in depths:
            for mode in modes:
                if depth == "before_B" and mode == "hotspot":
                    continue
                t0 = time.time()
                try:
                    cross_logits, cross_state = build_cross_state(
                        model, recipient_ids, donor_state,
                        depth, mode, hotspot_layers,
                    )
                except ValueError as e:
                    if verbose:
                        print(f"[{pair_id}][{direction}/{depth}/{mode}] "
                              f"skipped: {e}", flush=True)
                    continue
                tokens_X, logits_X_seq = continue_greedy(
                    model, cross_state, cross_logits, n_tokens,
                )
                elapsed = time.time() - t0
                if verbose:
                    print(f"[{pair_id}][{direction}/{depth}/{mode}] "
                          f"cross: {elapsed:.1f}s", flush=True)

                block = compute_cell_metrics(
                    prompt_donor_text=donor_text,
                    prompt_recipient_text=recipient_text,
                    tokens_clean_donor=donor_tokens,
                    tokens_clean_recipient=recipient_tokens,
                    tokens_cross=tokens_X,
                    logits_clean_donor=donor_logits,
                    logits_clean_recipient=recipient_logits,
                    logits_cross=logits_X_seq,
                    tokenizer=tokenizer,
                )
                block["config"] = {
                    "pair_id": pair_id,
                    "direction": direction,
                    "depth": depth,
                    "mode": mode,
                    "n_tokens": n_tokens,
                    "hotspot_layers": (list(hotspot_layers)
                                       if mode == "hotspot" else None),
                    "elapsed_s": elapsed,
                }
                results.append(block)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A0.6 intra-model state-transplant runner",
    )
    parser.add_argument("--model", default="g1d-0.4b",
                        choices=sorted(MODEL_ALIAS),
                        help="Checkpoint alias (see MODEL_ALIAS in source)")
    parser.add_argument("--pair", default=None,
                        help="Restrict to a single pair_id from tasks.jsonl")
    parser.add_argument("--depth", default=None,
                        choices=["before_B", "mid_B", "after_B"],
                        help="Restrict to a single injection depth")
    parser.add_argument("--mode", default=None, choices=["full", "hotspot"],
                        help="Restrict to a single swap mode")
    parser.add_argument("--direction", default=None, choices=["AB", "BA"],
                        help="Restrict to one donor→recipient direction "
                             "(default: run both, per ROADMAP A0.6 spec)")
    parser.add_argument("--n-tokens", type=int, default=64,
                        help="Continuation length (D1 default: 64)")
    parser.add_argument("--tasks", type=Path,
                        default=_HERE / "tasks.jsonl",
                        help="Path to prompt-pair JSONL fixture")
    parser.add_argument("--output-dir", type=Path,
                        default=_HERE / "results" / "a06",
                        help="Directory for per-cell JSON dumps")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    weights = MODEL_ALIAS[args.model]
    if args.verbose:
        print(f"[a06] loading {args.model} → {weights}", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(weights, device="cpu")
    if args.verbose:
        print(f"[a06] model loaded in {time.time() - t0:.1f}s", flush=True)

    pairs = load_pairs(args.tasks)
    if args.pair is not None:
        pairs = [p for p in pairs if p["pair_id"] == args.pair]
        if not pairs:
            print(f"[a06] pair_id {args.pair!r} not found in {args.tasks}",
                  file=sys.stderr)
            return 2

    depths = [args.depth] if args.depth else ["before_B", "mid_B", "after_B"]
    modes = [args.mode] if args.mode else ["full", "hotspot"]
    directions = [args.direction] if args.direction else ["AB", "BA"]

    hotspot_layers = list(DEFAULT_WORK_LAYERS)
    if args.verbose:
        print(f"[a06] hotspot_layers = {hotspot_layers}", flush=True)
        print(f"[a06] directions = {directions}", flush=True)

    n_cells = 0
    for pair in pairs:
        pair_results = run_pair(
            model=model,
            tokenizer=tokenizer,
            pair=pair,
            n_tokens=args.n_tokens,
            depths=depths,
            modes=modes,
            directions=directions,
            hotspot_layers=hotspot_layers,
            verbose=args.verbose,
        )
        for block in pair_results:
            cfg = block["config"]
            fname = (f"{args.model}__{cfg['pair_id']}__{cfg['direction']}__"
                     f"{cfg['depth']}__{cfg['mode']}.json")
            out_path = args.output_dir / fname
            with out_path.open("w") as f:
                json.dump(block, f, indent=2, ensure_ascii=False)
            n_cells += 1
            if args.verbose:
                aln = block["alignment"]["alignment"]
                dhit = block["lexicon"]["delta_hit_donor_vs_null"]
                print(f"[a06] wrote {out_path.name}  "
                      f"alignment={aln:+.3f}  Δhit_donor={dhit:+.3f}",
                      flush=True)

    print(f"[a06] done: {n_cells} cells written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
