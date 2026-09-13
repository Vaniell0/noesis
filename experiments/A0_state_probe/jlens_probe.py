#!/usr/bin/env python3
"""J-lens — WKV state spectral probe, base vs trained checkpoint comparison.

Computes SVD statistics (sigma1, stable_rank, Frobenius norm) of the raw
per-head WKV state matrix at the end of a prompt, per layer. Prediction
(H8): L_state-trained checkpoints show higher top singular values in
work_layers than base, corresponding to larger state updates accumulated
per token.

Fixed 2026-08-18 (previously mislabeled "Jacobian probe", left unmigrated
as deprecated 2026-08-17): two real bugs, both from the original version.
1. State indexing: read `state[L]` directly. State is a flat list of
   length 3*n_layer (state[3*i+0]=shift buffer, state[3*i+1]=WKV matrix,
   state[3*i+2]=FFN shift buffer) — `state[L]` mixed shift buffers, WKV
   matrices, and FFN buffers depending on L. Fixed to `state[3*L+1]`.
2. The old docstring promised an analytical WKV-7 Jacobian with a
   numeric-Jacobian sanity check. Neither was ever wired up —
   `_analytical_jacobian` was defined but never called, `_numeric_jacobian`
   raised `NotImplementedError`, and forward hooks that captured
   `time_decay` populated a dict nothing read. Removed rather than
   completed: what this probe actually measures (SVD of the raw WKV state,
   not a literal per-token Jacobian) is still a valid, useful signal for
   the H8 prediction above — the fix is describing it honestly, not
   building out an unrequested Jacobian implementation.

Usage:
    python jlens_probe.py \
        --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
        --trained ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-step7-20260807.pth \
        --work-layers 0,4,8,12,16,20,24,28 \
        --n-tokens 32 \
        --out results/jlens_base_vs_step7.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import registry
from experiments._common.model import load_model
from experiments._common.results import save_result


PROMPT = (
    "<think>\nLet me analyse the information carefully.\n"
    "The document describes a meeting on Tuesday where Alice and Bob "
    "discussed the project timeline. Key points: the deadline is March 15, "
    "the budget is $50,000, and the team lead is Carol.\n</think>"
)


def _svd_stats(J: torch.Tensor) -> Dict[str, float]:
    """Spectrum statistics for one head's WKV state matrix.

    `stable_rank` alone was misleading and is kept only for continuity with
    earlier results: it is ‖A‖²_F/σ₁², an ENERGY-CONCENTRATION measure, not a
    count of live directions. A matrix can have 32 non-negligible singular
    values and still report stable_rank≈1.3 if one of them dominates — so the
    low numbers this probe has been reporting (~1.2-1.3) never actually
    established that the state uses one direction, only that one direction
    holds most of the energy. The three added measures separate those:

    - `participation_ratio` = (Σσ)²/Σσ² — effective number of comparable
      singular values, less dominated by the largest one.
    - `effective_rank_entropy` = exp(H(p)), p_i = σ_i/Σσ — the entropy-based
      effective rank (Roy & Vetterli); the standard "how many directions are
      meaningfully in play" number.
    - `numerical_rank_1pct` = #{σ_i > 0.01·σ₁} — a plain count of directions
      above a 1% threshold, the crudest and least interpretable-away measure.

    All four come from the same svdvals call, so this costs nothing extra.
    """
    try:
        sv = torch.linalg.svdvals(J.float())
        sigma1 = float(sv[0])
        frob = float(J.float().norm())
        sv_sum = float(sv.sum())
        sv_sq_sum = float((sv ** 2).sum())
        p = sv / (sv_sum + 1e-12)
        entropy = float(-(p * (p + 1e-12).log()).sum())
        return {
            "sigma1": sigma1,
            "frob": frob,
            "stable_rank": (frob ** 2) / (sigma1 ** 2 + 1e-12),
            "participation_ratio": (sv_sum ** 2) / (sv_sq_sum + 1e-12),
            "effective_rank_entropy": float(torch.exp(torch.tensor(entropy))),
            "numerical_rank_1pct": int((sv > 0.01 * sigma1).sum()),
            "n_singular_values": int(sv.numel()),
        }
    except Exception as e:
        return {"sigma1": 0.0, "stable_rank": 0.0, "frob": 0.0, "error": str(e)}


def _dist(vals) -> Dict[str, float]:
    """min/median/max/p90 + spread of a per-head series. The point is to make
    'is this uniform across heads or driven by a few of them' answerable from
    the saved file, which it has not been in any previous run."""
    xs = sorted(float(v) for v in vals)
    if not xs:
        return {}
    n = len(xs)
    def q(f: float) -> float:
        return xs[min(n - 1, max(0, int(round(f * (n - 1)))))]
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {"min": xs[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75),
            "p90": q(0.90), "max": xs[-1], "mean": mean, "std": var ** 0.5}


def _analyze(model, tokenizer, work_layers: List[int], n_tokens: int) -> Dict:
    """Loading-free core: run the prompt, SVD the WKV state per work layer."""
    enc = tokenizer(PROMPT, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()[:n_tokens]

    with torch.no_grad():
        state = None
        for tok_id in ids:
            _, state = model.forward([tok_id], state)

    layer_stats: Dict[int, Dict] = {}
    for L in work_layers:
        idx = 3 * L + 1
        if state is None or idx >= len(state):
            continue
        try:
            s_L = state[idx].float().cpu()  # (n_head, H, H) WKV matrix
            n_head, H, _ = s_L.shape
            per_head_stats = [_svd_stats(s_L[h]) for h in range(n_head)]
            # Every earlier run of this probe computed exactly these per-head
            # stats and then threw the distribution away, keeping only the
            # mean over 40 heads. "37 heads at 1.0 plus 3 heads at 5.0" and
            # "all 40 heads at 1.3" produced an identical saved record, so no
            # existing artifact can distinguish uniform collapse from a few
            # specialised read-heads — a distinction that changes the reading
            # completely. The distribution is kept now; the mean_* keys stay
            # for continuity with the older files and the diff printer below.
            layer_stats[L] = {
                "mean_sigma1": sum(x["sigma1"] for x in per_head_stats) / n_head,
                "mean_stable_rank": sum(x["stable_rank"] for x in per_head_stats) / n_head,
                "mean_frob": sum(x["frob"] for x in per_head_stats) / n_head,
                "n_head": n_head,
                "head_size": H,
                "per_head": per_head_stats,
                "dist": {
                    key: _dist(x[key] for x in per_head_stats if key in x)
                    for key in ("sigma1", "stable_rank", "participation_ratio",
                                "effective_rank_entropy", "numerical_rank_1pct")
                },
            }
        except Exception as e:
            layer_stats[L] = {"error": str(e)}

    return {
        "n_tokens": len(ids),
        "work_layers": work_layers,
        "layer_stats": {str(k): v for k, v in layer_stats.items()},
    }


def probe_checkpoint(model_path: str, work_layers: List[int],
                     n_tokens: int, device: str) -> Dict:
    """Standalone entry point: loads its own model (for base/--trained CLI usage)."""
    model, tokenizer = load_model(model_path, device=device)
    model.eval()
    result = _analyze(model, tokenizer, work_layers, n_tokens)
    return {"model_path": str(model_path), **result}


def _add_jlens_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--work-layers", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--jlens-n-tokens", dest="jlens_n_tokens", type=int, default=32)


@registry.probe(
    "jlens", hypothesis=["H8"],
    description="SVD spectrum (sigma1/stable_rank/frob) of raw per-head WKV state at end of prompt.",
    add_args=_add_jlens_args,
)
def run(model, tokenizer, args) -> Dict:
    work_layers = [int(x) for x in args.work_layers.split(",")]
    result = _analyze(model, tokenizer, work_layers, args.jlens_n_tokens)
    return {"model": args.model, **result}


def main() -> int:
    ap = argparse.ArgumentParser(description="J-lens WKV state spectral probe.")
    ap.add_argument("--base", required=True, help="Base model .pth path")
    ap.add_argument("--trained", required=True, help="Trained model .pth path")
    ap.add_argument("--work-layers", default="0,4,8,12,16,20,24,28")
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work_layers = [int(x) for x in args.work_layers.split(",")]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Probing base: {args.base}")
    base_result = probe_checkpoint(args.base, work_layers, args.n_tokens, args.device)

    print(f"Probing trained: {args.trained}")
    trained_result = probe_checkpoint(args.trained, work_layers, args.n_tokens, args.device)

    print("\n=== J-lens WKV state spectrum comparison (base vs trained) ===")
    print(f"{'layer':>6}  {'base σ₁':>10}  {'trained σ₁':>10}  {'Δσ₁':>8}  "
          f"{'base SR':>8}  {'trained SR':>8}")
    print("-" * 60)
    for L in work_layers:
        bL = base_result["layer_stats"].get(str(L), {})
        tL = trained_result["layer_stats"].get(str(L), {})
        b1 = bL.get("mean_sigma1", float("nan"))
        t1 = tL.get("mean_sigma1", float("nan"))
        bsr = bL.get("mean_stable_rank", float("nan"))
        tsr = tL.get("mean_stable_rank", float("nan"))
        delta = t1 - b1 if (b1 == b1 and t1 == t1) else float("nan")
        print(f"{L:>6}  {b1:>10.4f}  {t1:>10.4f}  {delta:>+8.4f}  {bsr:>8.2f}  {tsr:>8.2f}")

    result = {"base": base_result, "trained": trained_result}
    save_result(
        out, result, experiment="jlens", hypothesis=["H8"],
        model=args.trained, script=__file__,
    )
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
