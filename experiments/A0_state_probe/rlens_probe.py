#!/usr/bin/env python3
"""R-lens probe — LRP stop-grad saliency + WKV spectral analysis.

Drop-in replacement for jlens_probe.py. Fixes two bugs:
  1. State indexing: uses state[3*L + 1] (WKV matrix) not state[L].
  2. Early-layer gradient noise: applies LN-rule (stop-grad on LayerNorm
     variance denominator) during backward, same as R-lens (marty1885,
     lesswrong.com/posts/nv8oedrnLXKRzNEL9). Recommended by BlinkDL for
     small-layer probing where raw J-lens gradients are incoherent.

## Metrics

Per layer L (all layers, not just work_layers):
  sigma1[L]       — largest singular value of WKV state matrix (avg over heads)
  stable_rank[L]  — (‖WKV‖_F / ‖WKV‖_2)² averaged over heads
  sv_entropy[L]   — entropy of normalised squared SVs (head specialisation)

Per token t (R-lens saliency):
  saliency[t]     — ‖∂logit/∂embed_t‖₂ with LN stop-grad applied

## Usage

    python rlens_probe.py \\
        --base  ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \\
        --trained ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-step7-20260807.pth \\
        --out results/rlens_base_vs_step7.json

    # Single model (no diff):
    python rlens_probe.py \\
        --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \\
        --out results/rlens_base.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


PROMPTS = [
    (
        "reasoning",
        "<think>\nLet me analyse carefully. The document describes a meeting on "
        "Tuesday where Alice and Bob discussed the project timeline. Deadline: March 15, "
        "budget: $50,000, team lead: Carol.\n</think>",
    ),
    (
        "math",
        "<think>\nTo solve 37 × 48: 37 × 50 - 37 × 2 = 1850 - 74 = 1776.\n</think>\nAnswer: 1776",
    ),
    (
        "narrative",
        "The old lighthouse keeper had watched over the bay for thirty years. "
        "Every evening he climbed the spiral stairs, lit the great lamp, and "
        "scanned the horizon for ships in distress.",
    ),
]


# ── LN stop-grad context manager ─────────────────────────────────────────────

@contextlib.contextmanager
def ln_stop_grad(model: nn.Module):
    """Apply LN-rule: detach variance denominator in all LayerNorm modules.

    During backward, variance is treated as a constant — no gradient flows
    through the normalisation denominator. This is the core of the R-lens
    LRP approximation that stabilises early-layer saliency.
    """
    handles = []

    def _make_hook(module):
        original_forward = module.forward

        def patched_forward(x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False).detach()  # LN-rule
            x_norm = (x - mean) / (var + module.eps).sqrt()
            if module.weight is not None:
                x_norm = x_norm * module.weight
            if module.bias is not None:
                x_norm = x_norm + module.bias
            return x_norm

        module.forward = patched_forward
        return module, original_forward

    saved = []
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod, orig = _make_hook(mod)
            saved.append((mod, orig))

    try:
        yield
    finally:
        for mod, orig in saved:
            mod.forward = orig


# ── WKV spectral analysis ─────────────────────────────────────────────────────

def wkv_spectrum(state: List[torch.Tensor]) -> Dict[str, List[float]]:
    """Extract per-layer WKV spectral metrics.

    state: RWKV state list where state[3*L + 1] = WKV matrix [n_head, H, H].
    Returns dict with lists of length n_layer:
      sigma1, stable_rank, sv_entropy
    """
    n_layers = len(state) // 3
    sigma1_all, sr_all, ent_all = [], [], []

    for L in range(n_layers):
        wkv = state[3 * L + 1].float().cpu().numpy()  # (n_head, H, H)
        n_head = wkv.shape[0]
        s1_list, sr_list, ent_list = [], [], []

        for h in range(n_head):
            sv = np.linalg.svd(wkv[h], compute_uv=False)
            s1 = float(sv[0])
            frob = float(np.sqrt((sv ** 2).sum()))
            sr = (frob / s1) ** 2 if s1 > 1e-9 else 0.0
            p = sv ** 2
            p = p / max(p.sum(), 1e-30)
            ent = float(-np.sum(p * np.log(p + 1e-30)))
            s1_list.append(s1)
            sr_list.append(sr)
            ent_list.append(ent)

        sigma1_all.append(float(np.mean(s1_list)))
        sr_all.append(float(np.mean(sr_list)))
        ent_all.append(float(np.mean(ent_list)))

    return {"sigma1": sigma1_all, "stable_rank": sr_all, "sv_entropy": ent_all}


# ── R-lens saliency ───────────────────────────────────────────────────────────

def rlens_saliency(
    model, tokenizer, prompt: str
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Compute per-token R-lens saliency.

    Returns (saliency_array [n_tok], token_strings) or None on failure.
    """
    ids = tokenizer(prompt)["input_ids"]
    if not ids:
        return None

    # Find embedding weight
    inner = model.model if hasattr(model, "model") else model
    emb_weight = None
    for name, param in inner.named_parameters():
        if "emb.weight" in name or "embedding.weight" in name.lower():
            emb_weight = param
            break
    if emb_weight is None:
        return None

    ids_t = torch.tensor(ids, dtype=torch.long)
    embeds = emb_weight[ids_t].detach().requires_grad_(True)

    try:
        with torch.enable_grad(), ln_stop_grad(inner):
            logits, _ = model.forward(ids, None)
            if isinstance(logits, torch.Tensor):
                if logits.dim() > 1:
                    logits = logits[-1]
                top_idx = int(logits.argmax())
                logits[top_idx].backward()

        if embeds.grad is None:
            return None

        sal = embeds.grad.float().norm(dim=-1).detach().numpy()
        toks = [tokenizer.decode([i]) for i in ids]  # type: ignore[arg-type]
        return sal, toks

    except Exception:
        return None


# ── Checkpoint probe ──────────────────────────────────────────────────────────

def probe_checkpoint(
    model_path: str, device: str, prompts: List[Tuple[str, str]]
) -> Dict:
    from probe import load_model, _extract_wkv_per_layer  # noqa: F401

    print(f"  loading {model_path}")
    model, tokenizer = load_model(model_path, device=device)
    model.eval()

    results: Dict[str, Dict] = {}

    for pid, prompt in prompts:
        print(f"  probe {pid} ...", end=" ", flush=True)
        ids = tokenizer(prompt)["input_ids"]

        # Forward to get final WKV state
        with torch.no_grad():
            logits, state = model.forward(ids, None)

        spectrum = wkv_spectrum(state)

        # R-lens saliency
        sal_result = rlens_saliency(model, tokenizer, prompt)
        if sal_result is not None:
            sal_arr, toks = sal_result
            saliency = sal_arr.tolist()
            tokens = toks
        else:
            saliency = []
            tokens = []

        print(f"σ₁_mid={spectrum['sigma1'][len(spectrum['sigma1'])//2]:.3f} "
              f"SR_mid={spectrum['stable_rank'][len(spectrum['stable_rank'])//2]:.2f}")

        results[pid] = {
            "n_tokens": len(ids),
            "sigma1": spectrum["sigma1"],
            "stable_rank": spectrum["stable_rank"],
            "sv_entropy": spectrum["sv_entropy"],
            "saliency": saliency,
            "tokens": tokens,
        }

    return results


# ── Diff ──────────────────────────────────────────────────────────────────────

def print_diff_table(base: Dict, trained: Dict, label: str) -> None:
    pids = [p for p in base if p in trained]
    if not pids:
        return
    print(f"\n=== R-lens σ₁ diff (trained − base) [{label}] ===")
    # Use first prompt
    pid = pids[0]
    b_s1 = base[pid]["sigma1"]
    t_s1 = trained[pid]["sigma1"]
    b_sr = base[pid]["stable_rank"]
    t_sr = trained[pid]["stable_rank"]
    n = min(len(b_s1), len(t_s1))
    step = max(1, n // 10)
    print(f"{'L':>4}  {'base σ₁':>9}  {'trained σ₁':>10}  {'Δ':>7}  "
          f"{'base SR':>8}  {'trained SR':>10}")
    print("-" * 55)
    for L in range(0, n, step):
        delta = t_s1[L] - b_s1[L]
        print(f"{L:>4}  {b_s1[L]:>9.4f}  {t_s1[L]:>10.4f}  {delta:>+7.4f}  "
              f"{b_sr[L]:>8.2f}  {t_sr[L]:>10.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="R-lens probe (LRP stop-grad saliency + WKV SVD).")
    ap.add_argument("--base", required=True, help="Base model .pth")
    ap.add_argument("--trained", default=None, help="Trained checkpoint .pth for diff")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompts", default=None, help="Plain text file, one prompt per line")
    args = ap.parse_args()

    prompts: List[Tuple[str, str]] = PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [(f"p{i}", ln.strip()) for i, ln in enumerate(f) if ln.strip()]

    print(f"[rlens] probing base: {args.base}")
    base_stats = probe_checkpoint(args.base, args.device, prompts)

    output: Dict = {"base": base_stats, "model": args.base}

    if args.trained:
        print(f"[rlens] probing trained: {args.trained}")
        trained_stats = probe_checkpoint(args.trained, args.device, prompts)
        output["trained"] = trained_stats
        output["trained_model"] = args.trained
        print_diff_table(base_stats, trained_stats, label=pathlib.Path(args.trained).stem)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"\n[rlens] saved → {out}")

    # Print sigma1 summary
    print("\n=== Base σ₁ by layer (mean across prompts) ===")
    all_sigma1 = [base_stats[p]["sigma1"] for p in base_stats if "sigma1" in base_stats[p]]
    if all_sigma1:
        mean_s1 = np.mean(all_sigma1, axis=0)
        step = max(1, len(mean_s1) // 10)
        for L in range(0, len(mean_s1), step):
            print(f"  L{L:>2}: σ₁={mean_s1[L]:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
