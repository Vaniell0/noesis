#!/usr/bin/env python3
"""Rough CPU->GPU time estimate for the execution matrix — a calculator,
not a measurement. There is no real GPU data point yet (this machine has
none), so every number here is CPU wall-clock observed tonight times an
*assumed* speedup factor — shown at three assumptions (conservative/
moderate/optimistic) rather than one falsely-precise number. Re-run once
a real GPU number exists to replace the assumption with fact.

CPU numbers below are what was actually observed 2026-08-18 on this
machine (G1d 0.4B unless noted) — see experiments/RESULTS.md and the
plan file for where each came from.
"""
from __future__ import annotations

import argparse

# (label, cpu_hours, note)
CPU_OBSERVATIONS = [
    ("ib_probe (linear), G1d, full corpus (~13.4k tok)", 11.5,
     "collection + 7-lag x 200-epoch linear decoder training"),
    ("mlp_ipc, G1i 2.9B, 256 tokens", 2.9,
     "trajectory collection + MLP training per lag x degree x layer"),
    ("lora_rank_analysis, G1h vs step9b-e1 (482 matrices)", 0.28,
     "pure SVD on weight deltas, no model forward pass"),
    ("fast battery (ipc,mlp_ipc,rlens,jlens,rich,think_geometry), G1d, 96 tok", 0.4,
     "6 probes, one shared model load"),
]

# Assumed CPU->GPU speedup for a T4 16GB. Genuinely unknown without a real
# data point — these are round-number brackets, not calibrated estimates.
# WKV recurrence is inherently sequential (can't batch across time), so
# the speedup is likely much smaller than a typical "GPU vs CPU" matmul
# benchmark would suggest — per-step compute is small either way; the win
# is mostly from not paying Python-loop + numpy/BLAS-thread overhead per
# step, not from parallelism this workload doesn't have (see
# experiments/README.md's batching note — nothing here batches across
# rollouts/prompts yet, which is the actual biggest lever, bigger than
# CPU vs GPU alone).
SPEEDUP_ASSUMPTIONS = {
    "conservative": 3.0,
    "moderate": 8.0,
    "optimistic": 20.0,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours-budget", type=float, default=None,
                    help="If set, also print how many of the CPU_OBSERVATIONS "
                         "rows fit in this many GPU-hours at each assumption.")
    args = ap.parse_args()

    print("CPU observation -> assumed GPU time (T4, unverified)\n")
    header = f"{'job':<55} {'CPU h':>7}"
    for name in SPEEDUP_ASSUMPTIONS:
        header += f"  {name+' (h)':>16}"
    print(header)
    print("-" * len(header))

    total_cpu = 0.0
    totals_gpu = {name: 0.0 for name in SPEEDUP_ASSUMPTIONS}
    for label, cpu_h, note in CPU_OBSERVATIONS:
        row = f"{label:<55} {cpu_h:>7.2f}"
        for name, factor in SPEEDUP_ASSUMPTIONS.items():
            gpu_h = cpu_h / factor
            totals_gpu[name] += gpu_h
            row += f"  {gpu_h:>16.2f}"
        total_cpu += cpu_h
        print(row)
        print(f"  ({note})")

    print("-" * len(header))
    row = f"{'TOTAL':<55} {total_cpu:>7.2f}"
    for name in SPEEDUP_ASSUMPTIONS:
        row += f"  {totals_gpu[name]:>16.2f}"
    print(row)

    print(f"\nAt T4 interruptible ~₽17.66/h: "
          f"conservative ~₽{totals_gpu['conservative']*17.66:.0f}, "
          f"moderate ~₽{totals_gpu['moderate']*17.66:.0f}, "
          f"optimistic ~₽{totals_gpu['optimistic']*17.66:.0f} "
          f"for just these 4 jobs (not the full matrix — see the plan's "
          f"execution matrix for what else runs per model).")

    if args.hours_budget:
        print(f"\nWith a {args.hours_budget}h budget:")
        for name, factor in SPEEDUP_ASSUMPTIONS.items():
            fits = totals_gpu[name] <= args.hours_budget
            print(f"  {name}: {'fits comfortably' if fits else 'likely too tight'} "
                  f"({totals_gpu[name]:.2f}h needed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
