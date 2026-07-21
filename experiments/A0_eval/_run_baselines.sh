#!/usr/bin/env bash
# A0.2 pretrained baselines. Sequential to avoid CPU contention with
# A0.5 grid running elsewhere. Order = ascending size.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

run_one() {
    local model="$1" tag="$2"
    echo "=== [$(date +%H:%M:%S)] eval $tag ($model) ==="
    python3 eval.py --model "$model" --num-predict 256 --timeout 180 \
        --out "results/${tag}.json" 2>&1 \
        | tee "results/${tag}.log"
    echo "=== [$(date +%H:%M:%S)] done $tag ==="
}

run_one qwen2.5:1.5b                  qwen25_15b
run_one rwkv7-1.5b:latest             rwkv7_15b_world
run_one mollysama/rwkv-7-g1h:2.9b     rwkv7_29b_g1h
run_one gemma3:4b                     gemma3_4b

echo "=== [$(date +%H:%M:%S)] ALL BASELINES DONE ==="
