"""A0.7 tier-1 — cross-checkpoint WKV-state transplant runner.

Same donor/recipient shape as A0.6, but donor and recipient are
*different* RWKV-7 checkpoints of the same architecture and size
(e.g. World-0.4B ↔ G1d-0.4B). Since only the WKV entries are
architecturally aligned across a continued-pretrain — shift buffers
are model-specific rolling caches whose entries mean nothing to a
foreign model — the transfer path here is **WKV-only** via
``_extract_wkv_per_layer`` + ``load_wkv_into_state`` rather than the
full-state ``corrupt_cross`` used by A0.6.

Axes (fixed here; anything else is design negotiation before an
actual run):

- **checkpoint direction** (new for A0.7): which model is donor and
  which is recipient. ``world_to_g1d`` and ``g1d_to_world`` for the
  0.4B pair; ``world_to_g1h`` and ``g1h_to_world`` for the 1.5B pair.
  Both mandated — the verdict has to survive both directions, else the
  effect is a one-way artefact of one checkpoint's WKV geometry.

- **prompt direction** (inherited from A0.6): ``AB`` (donor state is
  produced by prompt_A, recipient consumes prompt_B) and ``BA``.

- **injection depth** (D2, README): ``before_B``, ``mid_B``,
  ``after_B``. Same semantics as A0.6.

- **swap mode** (D3, README): ``full`` (all layers) and ``hotspot``
  (``DEFAULT_WORK_LAYERS`` = {12, 16, 20}). ``before_B × hotspot`` is
  skipped for the same reason as in A0.6 — no baseline recipient state
  exists yet at that depth for the hotspot subset to swap into.

Verdict rule (per README A0.7 tier-1):

- PASS: alignment ≤ −0.3 with ``coherence_flag = 1`` at > 50 % of the
  same-checkpoint (A0.6) baseline for that pair.
- FAIL: alignment ≥ 0 or ``coherence_flag = 0`` at < 20 %.
- Caveat zone (needs per-layer analysis): 20–50 %.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

# --------------------------------------------------------------------------- #
# Local imports — reach into A0_state_probe for probe / intervention utils
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(_REPO / "experiments" / "A0_state_probe"))
sys.path.insert(0, str(_HERE))

from a05_intervene import (  # noqa: E402
    snapshot,
    greedy_continue,
    load_wkv_into_state,
    kl_next,
)
from probe import load_model, _extract_wkv_per_layer  # noqa: E402
from state_reg import DEFAULT_WORK_LAYERS  # noqa: E402

from metrics import (  # noqa: E402
    alignment_vs_donor,
    first_divergence_step,
    surface_garble,
    task_lexicon_hit_rate,
    topk_jaccard_trajectory,
)

# Reuse a06's cell-metric bundler + cumulative-KL helper to keep the
# JSON schema identical across A0.6 and A0.7 (verdict tooling later
# consumes both without a schema branch).
from a06_run import (  # noqa: E402
    compute_cell_metrics,
    cumulative_kl,
    load_pairs,
)


# --------------------------------------------------------------------------- #
# Checkpoint aliases
# --------------------------------------------------------------------------- #

MODEL_ALIAS = {
    "world-0.4b": "BlinkDL/rwkv-7-world:RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
    "g1d-0.4b":   "BlinkDL/rwkv7-g1:rwkv7-g1d-0.4b-20260210-ctx8192.pth",
    "world-1.5b": "BlinkDL/rwkv-7-world:RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
    "g1h-1.5b":   "BlinkDL/rwkv7-g1:rwkv7-g1h-1.5b-20260710-ctx10240.pth",
}

# Per-model-pair sanity: the two checkpoints must be same-size (identical
# WKV shape at each layer) for the swap to be defined.
CHECKPOINT_PAIRS = {
    "0.4b": ("world-0.4b", "g1d-0.4b"),
    "1.5b": ("world-1.5b", "g1h-1.5b"),
}


# --------------------------------------------------------------------------- #
# Cross-checkpoint state construction — the direction × depth × mode matrix
# --------------------------------------------------------------------------- #

def build_cross_state_xckpt(
    recipient_model,
    recipient_prompt_ids: "list[int]",
    donor_wkv_per_layer: "list[torch.Tensor]",
    depth: str,
    mode: str,
    hotspot_layers: "Sequence[int]",
):
    """Return ``(logits, state)`` after applying the depth × mode
    WKV-only transplant to the recipient's prompt-processing path.

    Unlike A0.6, only WKV entries are transferred — shift buffers on
    the recipient stay whatever the recipient itself produced during
    its prefill (or zero if the swap lands at ``before_B``). This is
    the entire point of A0.7: does the WKV alone carry portable state
    across a continued-pretrain, without any help from architecturally
    matched shift buffers?
    """
    layer_indices = None if mode == "full" else list(hotspot_layers)

    if depth == "before_B":
        if mode != "full":
            raise ValueError(
                "before_B depth is only defined for mode=full — hotspot "
                "has no baseline state to selectively swap into"
            )
        # Fresh recipient state, WKV overwritten from donor before any
        # of prompt B is fed. The rwkv package doesn't expose a
        # `new_state()` constructor — the only way to obtain the flat
        # state list with correct shapes is to call `forward(_, None)`
        # once. We feed a single dummy token (id=0), then zero every
        # entry (WKV + both shifts) to recover a fresh-state
        # equivalent, then overlay donor's WKV. Zeroing the shift
        # buffers is exactly what "fresh state" means for RWKV-7 —
        # they are rolling caches whose initial value is zero.
        _, state = recipient_model.forward([0], None)
        for i, t in enumerate(state):
            state[i] = torch.zeros_like(t)
        state = load_wkv_into_state(state, donor_wkv_per_layer,
                                    layer_indices=layer_indices)
        logits, state = recipient_model.forward(recipient_prompt_ids, state)
        return logits, state

    if depth == "mid_B":
        mid = max(1, len(recipient_prompt_ids) // 2)
        _, state = recipient_model.forward(recipient_prompt_ids[:mid], None)
        state = load_wkv_into_state(state, donor_wkv_per_layer,
                                    layer_indices=layer_indices)
        logits, state = recipient_model.forward(
            recipient_prompt_ids[mid:], state,
        )
        return logits, state

    if depth == "after_B":
        if len(recipient_prompt_ids) < 2:
            raise ValueError("after_B requires prompt B length ≥ 2")
        _, state = recipient_model.forward(recipient_prompt_ids[:-1], None)
        state = load_wkv_into_state(state, donor_wkv_per_layer,
                                    layer_indices=layer_indices)
        logits, state = recipient_model.forward(
            [recipient_prompt_ids[-1]], state,
        )
        return logits, state

    raise ValueError(f"unknown depth={depth!r}")


# --------------------------------------------------------------------------- #
# Per-pair runner
# --------------------------------------------------------------------------- #

def run_pair_xckpt(
    donor_model, donor_tokenizer, donor_alias: str,
    recipient_model, recipient_tokenizer, recipient_alias: str,
    pair: "dict",
    n_tokens: int,
    depths: "Sequence[str]",
    modes: "Sequence[str]",
    prompt_directions: "Sequence[str]",
    hotspot_layers: "Sequence[int]",
    verbose: bool = False,
) -> "list[dict]":
    """One pair, one checkpoint-direction. Iterates prompt directions
    (AB/BA) and the depth × mode matrix inside.

    Clean baselines used for metrics:

    - ``clean_donor``   — greedy decode from donor model on donor prompt
      (with donor's own state).
    - ``clean_recipient`` — greedy decode from **recipient** model on
      recipient prompt. This is the correct baseline for cross-model
      comparison: cross's continuation is produced by the recipient
      model, so KL-to-recipient must also use recipient-produced logits.

    Note the tokenizer choice: donor and recipient may share the same
    tokenizer (World and G1 both derive from the World tokenizer in
    2026 releases), but we do not assume it — every text↔ids
    conversion goes through the model's own tokenizer.
    """
    prompt_A = pair["prompt_A"]
    prompt_B = pair["prompt_B"]
    pair_id = pair["pair_id"]

    # -- Donor prefills. Give us the WKV to inject.
    t0 = time.time()
    ids_A_donor = donor_tokenizer(prompt_A, return_tensors="pt")["input_ids"][0].tolist()
    logits_A_donor, state_A_donor = donor_model.forward(ids_A_donor, None)
    wkv_A_donor = _extract_wkv_per_layer(state_A_donor)
    if verbose:
        print(f"[{pair_id}][{donor_alias}] prefill A + extract: "
              f"{time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    ids_B_donor = donor_tokenizer(prompt_B, return_tensors="pt")["input_ids"][0].tolist()
    logits_B_donor, state_B_donor = donor_model.forward(ids_B_donor, None)
    wkv_B_donor = _extract_wkv_per_layer(state_B_donor)
    if verbose:
        print(f"[{pair_id}][{donor_alias}] prefill B + extract: "
              f"{time.time() - t0:.1f}s", flush=True)

    # Donor clean continuations — used only for alignment-vs-donor
    # metric on the donor side of the pair. Alignment expects
    # `clean_donor` logits produced by the *donor* model (this is the
    # asymmetry vs A0.6: recipient logits come from recipient model).
    t0 = time.time()
    tokens_donor_A, logits_donor_A_seq = greedy_continue(
        donor_model, snapshot(state_A_donor), logits_A_donor, n_tokens,
    )
    if verbose:
        print(f"[{pair_id}][{donor_alias}] clean_donor_A greedy: "
              f"{time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    tokens_donor_B, logits_donor_B_seq = greedy_continue(
        donor_model, snapshot(state_B_donor), logits_B_donor, n_tokens,
    )
    if verbose:
        print(f"[{pair_id}][{donor_alias}] clean_donor_B greedy: "
              f"{time.time() - t0:.1f}s", flush=True)

    # -- Recipient prefills + clean continuations.
    t0 = time.time()
    ids_A_recipient = recipient_tokenizer(prompt_A, return_tensors="pt")["input_ids"][0].tolist()
    logits_A_recipient, state_A_recipient = recipient_model.forward(
        ids_A_recipient, None,
    )
    if verbose:
        print(f"[{pair_id}][{recipient_alias}] prefill A: "
              f"{time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    ids_B_recipient = recipient_tokenizer(prompt_B, return_tensors="pt")["input_ids"][0].tolist()
    logits_B_recipient, state_B_recipient = recipient_model.forward(
        ids_B_recipient, None,
    )
    if verbose:
        print(f"[{pair_id}][{recipient_alias}] prefill B: "
              f"{time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    tokens_recip_A, logits_recip_A_seq = greedy_continue(
        recipient_model, snapshot(state_A_recipient),
        logits_A_recipient, n_tokens,
    )
    if verbose:
        print(f"[{pair_id}][{recipient_alias}] clean_recipient_A greedy: "
              f"{time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    tokens_recip_B, logits_recip_B_seq = greedy_continue(
        recipient_model, snapshot(state_B_recipient),
        logits_B_recipient, n_tokens,
    )
    if verbose:
        print(f"[{pair_id}][{recipient_alias}] clean_recipient_B greedy: "
              f"{time.time() - t0:.1f}s", flush=True)

    # prompt_direction → (donor prompt text, donor WKV,
    #                     donor clean tokens, donor clean logits,
    #                     recipient prompt text, recipient prompt ids,
    #                     recipient clean tokens, recipient clean logits)
    binding = {
        "AB": (prompt_A, wkv_A_donor, tokens_donor_A, logits_donor_A_seq,
               prompt_B, ids_B_recipient, tokens_recip_B, logits_recip_B_seq),
        "BA": (prompt_B, wkv_B_donor, tokens_donor_B, logits_donor_B_seq,
               prompt_A, ids_A_recipient, tokens_recip_A, logits_recip_A_seq),
    }

    results: "list[dict]" = []
    for prompt_dir in prompt_directions:
        if prompt_dir not in binding:
            raise ValueError(f"unknown prompt_direction={prompt_dir!r}")
        (donor_text, donor_wkv, donor_tokens, donor_logits,
         recipient_text, recipient_ids, recipient_tokens,
         recipient_logits) = binding[prompt_dir]

        for depth in depths:
            for mode in modes:
                if depth == "before_B" and mode == "hotspot":
                    continue
                t0 = time.time()
                try:
                    cross_logits, cross_state = build_cross_state_xckpt(
                        recipient_model, recipient_ids, donor_wkv,
                        depth, mode, hotspot_layers,
                    )
                except ValueError as e:
                    if verbose:
                        print(f"[{pair_id}][{prompt_dir}/{depth}/{mode}] "
                              f"skipped: {e}", flush=True)
                    continue
                tokens_X, logits_X_seq = greedy_continue(
                    recipient_model, snapshot(cross_state),
                    cross_logits, n_tokens,
                )
                elapsed = time.time() - t0
                if verbose:
                    print(f"[{pair_id}][{prompt_dir}/{depth}/{mode}] "
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
                    tokenizer=recipient_tokenizer,
                )
                block["config"] = {
                    "pair_id": pair_id,
                    "donor_ckpt": donor_alias,
                    "recipient_ckpt": recipient_alias,
                    "prompt_direction": prompt_dir,
                    "depth": depth,
                    "mode": mode,
                    "n_tokens": n_tokens,
                    "hotspot_layers": (list(hotspot_layers)
                                       if mode == "hotspot" else None),
                    "elapsed_s": elapsed,
                }
                results.append(block)
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _iter_ckpt_directions(pair_alias: str, restrict: "str | None"):
    a, b = CHECKPOINT_PAIRS[pair_alias]
    fwd = (a, b)   # donor=world, recipient=g1x
    rev = (b, a)   # donor=g1x, recipient=world
    if restrict is None:
        return [fwd, rev]
    if restrict == "forward":
        return [fwd]
    if restrict == "reverse":
        return [rev]
    raise ValueError(f"unknown --ckpt-direction={restrict!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A0.7 tier-1 cross-checkpoint WKV-state runner",
    )
    parser.add_argument("--ckpt-pair", default="0.4b",
                        choices=sorted(CHECKPOINT_PAIRS),
                        help="Which same-size checkpoint pair to run")
    parser.add_argument("--ckpt-direction", default=None,
                        choices=["forward", "reverse"],
                        help="'forward' = world→g1x, 'reverse' = g1x→world "
                             "(default: run both)")
    parser.add_argument("--prompt-direction", default=None,
                        choices=["AB", "BA"],
                        help="Restrict to one prompt donor→recipient "
                             "direction (default: run both)")
    parser.add_argument("--pair", default=None,
                        help="Restrict to a single pair_id from tasks.jsonl")
    parser.add_argument("--depth", default=None,
                        choices=["before_B", "mid_B", "after_B"])
    parser.add_argument("--mode", default=None, choices=["full", "hotspot"])
    parser.add_argument("--n-tokens", type=int, default=64,
                        help="Continuation length (D1 default: 64)")
    parser.add_argument("--tasks", type=Path,
                        default=_HERE / "tasks.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=_HERE / "results" / "a07_tier1")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.tasks)
    if args.pair is not None:
        pairs = [p for p in pairs if p["pair_id"] == args.pair]
        if not pairs:
            print(f"[a07] pair_id {args.pair!r} not found in {args.tasks}",
                  file=sys.stderr)
            return 2

    depths = [args.depth] if args.depth else ["before_B", "mid_B", "after_B"]
    modes = [args.mode] if args.mode else ["full", "hotspot"]
    prompt_dirs = ([args.prompt_direction] if args.prompt_direction
                   else ["AB", "BA"])
    ckpt_dirs = _iter_ckpt_directions(args.ckpt_pair, args.ckpt_direction)

    hotspot_layers = list(DEFAULT_WORK_LAYERS)
    if args.verbose:
        print(f"[a07] hotspot_layers = {hotspot_layers}", flush=True)
        print(f"[a07] prompt_directions = {prompt_dirs}", flush=True)
        print(f"[a07] ckpt_directions = {ckpt_dirs}", flush=True)

    # Load-once-per-ckpt-direction discipline: (donor, recipient) swap
    # roles between the two ckpt directions, so both models end up
    # loaded once total (not once per direction). Order the iteration
    # so a load pair is reused if possible.
    loaded: "dict[str, tuple]" = {}

    def get_model(alias: str):
        if alias not in loaded:
            if args.verbose:
                print(f"[a07] loading {alias} → {MODEL_ALIAS[alias]}",
                      flush=True)
            t0 = time.time()
            m, tk = load_model(MODEL_ALIAS[alias], device="cpu")
            if args.verbose:
                print(f"[a07] {alias} loaded in {time.time() - t0:.1f}s",
                      flush=True)
            loaded[alias] = (m, tk)
        return loaded[alias]

    n_cells = 0
    for donor_alias, recipient_alias in ckpt_dirs:
        donor_model, donor_tokenizer = get_model(donor_alias)
        recipient_model, recipient_tokenizer = get_model(recipient_alias)

        for pair in pairs:
            pair_results = run_pair_xckpt(
                donor_model=donor_model,
                donor_tokenizer=donor_tokenizer,
                donor_alias=donor_alias,
                recipient_model=recipient_model,
                recipient_tokenizer=recipient_tokenizer,
                recipient_alias=recipient_alias,
                pair=pair,
                n_tokens=args.n_tokens,
                depths=depths,
                modes=modes,
                prompt_directions=prompt_dirs,
                hotspot_layers=hotspot_layers,
                verbose=args.verbose,
            )
            for block in pair_results:
                cfg = block["config"]
                fname = (f"{cfg['donor_ckpt']}__to__{cfg['recipient_ckpt']}"
                         f"__{cfg['pair_id']}__{cfg['prompt_direction']}"
                         f"__{cfg['depth']}__{cfg['mode']}.json")
                out_path = args.output_dir / fname
                with out_path.open("w") as f:
                    json.dump(block, f, indent=2, ensure_ascii=False)
                n_cells += 1
                if args.verbose:
                    aln = block["alignment"]["alignment"]
                    dhit = block["lexicon"]["delta_hit_donor_vs_null"]
                    print(f"[a07] wrote {out_path.name}  "
                          f"alignment={aln:+.3f}  Δhit_donor={dhit:+.3f}",
                          flush=True)

    # Free RAM once done.
    loaded.clear()
    gc.collect()

    print(f"[a07] done: {n_cells} cells written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
