#!/usr/bin/env bash
# Is Muon's usable step window wider than Adam's? (docs/muon-rwkv7-finetuning.md §2.6, 2026-10-06)
# Prediction written BEFORE the run: at induced steps of ~3e-3 .. 1e-2 Adam loses the old skill
# or fails to learn the new one in clearly more seeds than Muon does at the SAME induced step.
# If Adam holds as well as Muon at matched steps up to ~1e-2, the "wider window" reading is a
# property of the 0.4B run only and §2.6 must say so. Compare by induced step, not by lr:
# the per-arm induced steps are in the output (induced_step_mean).
# Starts after the rank sweep (one heavy run at a time). Resumable.
cd /home/vaniello/Desktop/projects/noesis
LOG=experiments/_logs/attractor_pairs_2026-10-06
R=experiments/A0_state_probe/results
while pgrep -f "attractor_depth_probe.py --seeds 12 --lora-r" > /dev/null; do sleep 30; done
ARMS="full_adam:1e-2,full_adam:3e-2,full_muon:0.05,lora_adam:1e-2,lora_adam:3e-2,lora_muon_fw:3e-3,lora_muon_fw:1e-2"
for start in 0 6; do
  nice -n 5 training/.venv/bin/python -u experiments/A0_state_probe/attractor_depth_probe.py \
    --seeds 6 --seed-start $start --arms "$ARMS" --threads 1 \
    --out $R/attractor_window_s$start.json >> $LOG/window_s$start.log 2>&1 &
done
wait
