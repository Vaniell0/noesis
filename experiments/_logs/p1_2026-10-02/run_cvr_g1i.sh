#!/usr/bin/env bash
# CoT vs silent routes on G1i — resumable: rerun after a restart, finished items are skipped.
cd /home/vaniello/Desktop/projects/noesis
M=models/rwkv7-g1i-2.9b-20260805-ctx16384.pth
nice -n 5 training/.venv/bin/python -u experiments/rl/cot_vs_routes_probe.py --model $M --families pattern --levels 1,2,3,4 --per-cell 8 \
  --out experiments/rl/results/cot_vs_routes_g1i_pattern.json >> experiments/_logs/p1_2026-10-02/cvr_g1i.log 2>&1
nice -n 5 training/.venv/bin/python -u experiments/rl/cot_vs_routes_probe.py --model $M --families arith --levels 1,2 --per-cell 8 \
  --out experiments/rl/results/cot_vs_routes_g1i_arith.json >> experiments/_logs/p1_2026-10-02/cvr_g1i.log 2>&1
