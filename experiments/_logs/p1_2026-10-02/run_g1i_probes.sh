#!/usr/bin/env bash
# G1i probe chain, resumable: rerun this file after a restart, finished items are skipped.
cd /home/vaniello/Desktop/projects/noesis
M=models/rwkv7-g1i-2.9b-20260805-ctx16384.pth
nice -n 5 training/.venv/bin/python -u experiments/rl/phase_probe.py --model $M --per-cell 10 --levels 1,3 \
  --arms latent:0,latent:2,latent:8,latent:32,const:8,argmax:8,cot:512 \
  --out experiments/rl/results/phase_probe_g1i.json >> experiments/_logs/p1_2026-10-02/phase_probe_g1i.log 2>&1
nice -n 5 training/.venv/bin/python -u experiments/rl/fork_agreement_probe.py --model $M --per-cell 20 --levels 1,2,3,4 --k 6 \
  --out experiments/rl/results/fork_agreement_g1i.json >> experiments/_logs/p1_2026-10-02/fork_agreement_g1i.log 2>&1
