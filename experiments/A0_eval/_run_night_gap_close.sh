#!/usr/bin/env bash
# Overnight batch closing the two §4.1 blocking gaps from
# ~/Desktop/noesis-arxiv-brief.md:
#   1. gemma3:4b @ np=2048 (previous batch killed by battery loss).
#   2. rwkv7-2.9b World @ np=2048 — same-size same-arch ablation vs G1h,
#      isolates G1 reasoning-tuning from parameter scale.
# Launched 2026-07-22 as an autonomous background job.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_LOG="results/_np2048_night_${TS}.log"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "=== night batch start ${TS} (pid $$) ==="
echo "=== 2 models x 48 tasks @ num_predict=2048 ==="

run_one() {
    local model="$1" tag="$2"
    echo ""
    echo "=== [$(date -u +%H:%M:%SZ)] eval ${tag} (${model}) start ==="
    python3 eval.py --model "${model}" --num-predict 2048 --timeout 300 \
        --out "results/${tag}_np2048.json" 2>&1 \
        | tee "results/${tag}_np2048.log"
    echo "=== [$(date -u +%H:%M:%SZ)] eval ${tag} done ==="
}

run_one gemma3:4b                gemma3_4b
run_one rwkv7-2.9b:latest        rwkv7_29b_world

echo ""
echo "=== [$(date -u +%H:%M:%SZ)] ALL NIGHT DONE ==="
