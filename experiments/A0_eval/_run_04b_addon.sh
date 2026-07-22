#!/usr/bin/env bash
# Addon batch for the 0.4B RWKV-7 baseline @ np=2048.
#
# 2026-07-22 revision: switched from BlinkDL .pth backend (bf16, CPU,
# unusably slow — >19 min per task on 0.4B) to quantised ollama
# backend. User signed off on quant: точность нужна была в A0.5,
# для логики достаточно quant.
#
# Models:
#   1. mollysama/rwkv-7-g1d:0.4b — reasoning-tuned (G1 line), same
#      arch as g1h/g1d at the smallest pilot size, ~500 MB quantised.
#
# World-0.4B-base is skipped: not in the ollama registry (checked
# mollysama/rwkv-7-world:0.4b, rwkv7-0.4b:latest — both 404). If a
# baseline for pure World at 0.4B is needed, run the .pth path
# separately with a walltime budget.
#
# Launched 2026-07-22.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_LOG="results/_np2048_04b_addon_${TS}.log"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "=== 0.4B addon start ${TS} (pid $$) ==="
echo "=== 1 model x 48 tasks @ num_predict=2048 (ollama backend, quantised) ==="

run_one() {
    local model="$1" tag="$2"
    echo ""
    echo "=== [$(date -u +%H:%M:%SZ)] eval ${tag} (${model}) start ==="
    python3 eval.py --model "${model}" --num-predict 2048 --timeout 300 \
        --out "results/${tag}_np2048.json" 2>&1 \
        | tee "results/${tag}_np2048.log"
    echo "=== [$(date -u +%H:%M:%SZ)] eval ${tag} done ==="
}

run_one mollysama/rwkv-7-g1d:0.4b rwkv7_g1d_04b

echo ""
echo "=== [$(date -u +%H:%M:%SZ)] 0.4B ADDON DONE ==="
