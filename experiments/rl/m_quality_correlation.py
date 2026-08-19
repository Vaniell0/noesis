#!/usr/bin/env python3
"""m_quality_correlation.py — does more M-loop thinking correlate with
better or worse answers?

Built 2026-08-19 specifically for the zeta (info-density reward) staged
test: accuracy alone is noisy (small effective group sizes from OOM
attrition) and doesn't say whether M is *helping* or *hurting* — the
sign of the M-vs-correctness correlation is the sharper signal. Point-
biserial correlation (Pearson between an integer M and a binary
correct/wrong) is the right tool since `correct` is 0/1.

Reads answers_log.jsonl (per-rollout M + correct, written by
train_wkv_loop.py's main loop) — no new logging needed, this only reads
what already exists.

Usage:
    python experiments/rl/m_quality_correlation.py path/to/answers_log.jsonl
    python experiments/rl/m_quality_correlation.py path/to/answers_log.jsonl --window 50
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple


def _load_rollouts(path: Path) -> List[Tuple[int, int, bool]]:
    """Returns [(step, M, correct), ...] flattened across all logged batches."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for r in d["rollouts"]:
            out.append((d["step"], r["M"], r["correct"]))
    return out


def _point_biserial(pairs: List[Tuple[int, bool]]) -> Tuple[float, int]:
    """Pearson correlation between M (int) and correct (bool->0/1).
    Returns (r, n). r is nan if n<2 or M/correct has zero variance."""
    n = len(pairs)
    if n < 2:
        return float("nan"), n
    xs = [float(m) for m, _ in pairs]
    ys = [1.0 if c else 0.0 for _, c in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan"), n
    return cov / (sx * sy), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("answers_log", type=Path)
    ap.add_argument("--window", type=int, default=0,
                    help="If >0, also report correlation per N-step window "
                         "(in addition to the overall figure).")
    args = ap.parse_args()

    rollouts = _load_rollouts(args.answers_log)
    if not rollouts:
        print(f"[m_quality_correlation] no rollouts found in {args.answers_log}")
        return 1

    overall = [(m, c) for _, m, c in rollouts]
    r, n = _point_biserial(overall)
    m_vals = [m for m, _ in overall]
    acc = sum(1 for _, c in overall if c) / n
    print(f"Overall: n={n}  mean_M={sum(m_vals)/n:.2f}  "
          f"M_range=[{min(m_vals)},{max(m_vals)}]  accuracy={acc:.3f}  "
          f"point-biserial r(M,correct)={r:+.4f}")
    if r == r:  # not nan
        direction = "POSITIVE (more M -> better)" if r > 0 else "NEGATIVE (more M -> worse)"
        print(f"  -> {direction}")
    else:
        print("  -> undefined (no variance in M or correctness this window)")

    if args.window > 0:
        steps = sorted(set(s for s, _, _ in rollouts))
        lo = steps[0]
        print(f"\nPer-{args.window}-step window:")
        while lo <= steps[-1]:
            hi = lo + args.window - 1
            window_pairs = [(m, c) for s, m, c in rollouts if lo <= s <= hi]
            if window_pairs:
                r, n = _point_biserial(window_pairs)
                m_vals = [m for m, _ in window_pairs]
                acc = sum(1 for _, c in window_pairs if c) / n
                r_str = f"{r:+.4f}" if r == r else "  n/a "
                print(f"  step {lo:>4}-{hi:<4}  n={n:>4}  mean_M={sum(m_vals)/n:.2f}  "
                      f"accuracy={acc:.3f}  r={r_str}")
            lo = hi + 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
