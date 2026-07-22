#!/usr/bin/env bash
# A0.2 re-run @ num_predict=2048 for the three models that were originally
# scored at np=256 (mollysama already re-run separately). Comparable numbers
# on the merged 48-task set (base 42 + bit_book 6). Launched 2026-07-22 as
# an ~8h autonomous background job; will finish earlier in practice.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_LOG="results/_np2048_batch_${TS}.log"
exec > >(tee -a "${BATCH_LOG}") 2>&1

echo "=== batch start ${TS} (pid $$) ==="
echo "=== 3 models x 48 tasks @ num_predict=2048 ==="

run_one() {
    local model="$1" tag="$2"
    echo ""
    echo "=== [$(date -u +%H:%M:%SZ)] eval ${tag} (${model}) start ==="
    python3 eval.py --model "${model}" --num-predict 2048 --timeout 300 \
        --out "results/${tag}_np2048.json" 2>&1 \
        | tee "results/${tag}_np2048.log"
    echo "=== [$(date -u +%H:%M:%SZ)] eval ${tag} done ==="
}

run_one qwen2.5:1.5b          qwen25_15b
run_one rwkv7-1.5b:latest     rwkv7_15b_world
run_one gemma3:4b             gemma3_4b

echo ""
echo "=== [$(date -u +%H:%M:%SZ)] ALL DONE ==="
