#!/usr/bin/env bash
# Rank sweep with the step controlled (user + reviewer note, 2026-10-06).
# The earlier plan ran lr=1e-3 at r=2 and r=32: the induced step changes with the rank, so
# "concentration vs rank" would not separate capacity from "Adam spends its step faster".
# Here every rank gets an lr grid per optimizer; the analysis compares optimizers at EQUAL
# induced step (interpolated on log step) and reads top-1/erank at K=0, steps-to-new>=0.99
# and new R2 per unit of weight moved (attractor_rank_summary.py).
# alpha = 2r keeps alpha/r at the default 2. Resumable: rerun after a restart.
# Starts when the pair run (seeds 6..29) has finished: one heavy run at a time.
cd /home/vaniello/Desktop/projects/noesis
LOG=experiments/_logs/attractor_pairs_2026-10-06
R=experiments/A0_state_probe/results
while pgrep -f "attractor_depth_probe.py --seeds 8" > /dev/null; do sleep 20; done
ARMS="lora_adam:3e-4,lora_adam:1e-3,lora_adam:3e-3,lora_adam:1e-2,lora_muon_fw:3e-4,lora_muon_fw:1e-3,lora_muon_fw:3e-3,lora_muon_fw:1e-2"
for r in 2 8 32; do
  nice -n 5 training/.venv/bin/python -u experiments/A0_state_probe/attractor_depth_probe.py \
    --seeds 12 --lora-r $r --lora-alpha $((2 * r)) --arms "$ARMS" --threads 1 \
    --out $R/attractor_rank_r$r.json >> $LOG/rank_r$r.log 2>&1 &
done
wait
