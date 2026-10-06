#!/usr/bin/env bash
# Matched-step Adam vs Muon pairs of the attractor-depth toy, seeds 6..29
# (seeds 0..5 are in results/attractor_depth.partial.jsonl and are merged in
# afterwards, not recomputed). Three single-thread processes, 8 seeds each —
# resumable: rerun after a restart, finished (seed, arm) items are skipped.
#
#   pair full: full_adam 3e-3  vs full_muon 5e-3      (induced step ~0.001)
#   pair lora: lora_adam 1e-3  vs lora_muon_fw 1e-3   (induced step ~0.0007)
cd /home/vaniello/Desktop/projects/noesis
LOG=experiments/_logs/attractor_pairs_2026-10-06
R=experiments/A0_state_probe/results
ARMS="full_adam:3e-3,full_muon:0.005,lora_adam:1e-3,lora_muon_fw:0.001"
for start in 6 14 22; do
  nice -n 5 training/.venv/bin/python -u experiments/A0_state_probe/attractor_depth_probe.py \
    --seeds 8 --seed-start $start --arms "$ARMS" --threads 1 \
    --out $R/attractor_pairs_s$start.json >> $LOG/s$start.log 2>&1 &
done
wait
