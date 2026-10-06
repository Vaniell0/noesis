#!/usr/bin/env bash
# Replicates the 0.4B optimizer-geometry table of docs/muon-rwkv7-finetuning.md §2.6 with what the
# review asked for (2026-10-06): Muon at the SAME lr grid as AdamW (so Muon is not one point), data order
# shuffled per seed (the earlier three "seeds" were bit-identical), and both the maximum AND the median
# first-step ||dW||/||W|| kept (the maximum falls on ffn.x_k, a 1024-vector both optimizers update the same
# way, so it cannot decide anything). Compare optimizers by interpolating dCE on the MEDIAN step.
# Data: training/corpus_open/step9b_combined_flat.jsonl (same as the single run, for comparability; the
# mix contains a share derived from personal sessions — local throwaway fine-tune, not saved).
# One process at a time, 4 threads, resumable (skips finished outputs).
cd /home/vaniello/Desktop/projects/noesis
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT=experiments/rl/results/geomgrid
mkdir -p $OUT
for seed in 0 1 2; do
  for lr in 1e-5 3e-5 1e-4; do
    f=$OUT/lr${lr}_s${seed}.json
    [ -s "$f" ] && continue
    nice -n 5 training/.venv/bin/python -u experiments/rl/optimizer_geometry_probe.py \
      --model /home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
      --layers 11,12,13 --device cpu --steps 30 --arms muon,adam \
      --muon-lr $lr --adam-lr $lr --seed $seed --shuffle-data on \
      --out $f >> experiments/_logs/muon_window_2026-10-06/geom_grid.log 2>&1
  done
done
