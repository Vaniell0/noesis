#!/usr/bin/env bash
# Overnight run, 2026-10-01. Three stages, STRICTLY sequential — no parallel
# heavy runs on this laptop (memory: feedback_heavy_runs_supervised).
# Each stage writes a .partial.json every 4 prompts where it can, so an
# interrupted night still leaves data. No `set -e`: one stage failing must not
# cancel the next. No grep/tail in any pipeline: they buffer.
#
# What it answers: for the same states, how many directions are OCCUPIED,
# how many the model's own future output is SENSITIVE to (gradient and causal
# ablation), and — on the real 2.9B target — how many bindings are linearly
# RETRIEVABLE. Three numbers, so "occupied vs used" stops being a guess.

set -u
cd /home/vaniello/Desktop/projects/noesis
P=training/.venv/bin/python
D=experiments/_logs/overnight_2026-10-01
R=experiments/rl/results
S04=/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
G29=models/rwkv7-g1i-2.9b-20260805-ctx16384.pth

stamp () { echo "[$(date '+%F %T')] $*"; }

stamp "STAGE 1: readout lens on g1d-0.4b (gradient on 8 layers, ablation on 3)"
$P -u experiments/rl/state_readout_lens.py --model "$S04" \
  --layers 4,8,11,12,13,16,20,23 --abl-layers 8,12,20 \
  --abl-heads 16 --abl-dirs 12 --n-text 24 --n-kv 24 --cont-len 16 \
  --out $R/state_readout_lens_g1d04b.json > $D/stage1_lens_04b.log 2>&1
stamp "STAGE 1 exit $?"

stamp "STAGE 2: readout lens on G1i-2.9B (gradient on 6 layers, ablation on L21)"
$P -u experiments/rl/state_readout_lens.py --model "$G29" \
  --layers 8,15,20,21,22,28 --abl-layers 21 \
  --abl-heads 8 --abl-dirs 8 --n-text 24 --n-kv 24 --cont-len 16 \
  --out $R/state_readout_lens_g1i.json > $D/stage2_lens_g1i.log 2>&1
stamp "STAGE 2 exit $?"

stamp "STAGE 3: usable capacity on G1i-2.9B, N=1,2,4,8, 400 samples"
$P -u experiments/rl/state_capacity_probe.py --model "$G29" \
  --layers 20,21,22 --n-list 1,2,4,8 --samples 400 \
  --out $R/state_capacity_g1i_n400.json > $D/stage3_capacity_g1i.log 2>&1
stamp "STAGE 3 exit $?"

stamp "ALL DONE"
