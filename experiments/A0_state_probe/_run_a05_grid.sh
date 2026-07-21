#!/usr/bin/env bash
# A0.5 Stage-1 grid runner. 4 cells sequentially.
# Args: sample-layers=[0,4,8,12,16,20], continuation-N=6, 3 seeds,
# 128 tokens, 4 checkpoints, cross-prompt swap enabled.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true

WORLD='BlinkDL/rwkv-7-world:RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth'
G1D='/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth'

COMMON=(--seeds 3 --max-new-tokens 128 --k-checkpoints 4
        --sigmas 0.005,0.01,0.02,0.05,0.1,0.2
        --scales 0.5,1.5,2.0
        --sample-layers 0,4,8,12,16,20
        --continuation-steps 6)

run() {
    local model="$1" prompt="$2" cross="$3" tag="$4"
    local out="results/a05_ext/${tag}"
    echo "=== [$(date +%H:%M:%S)] launching ${tag} ==="
    python3 a05_run.py --model "$model" --prompt "$prompt" \
        --cross-prompt "$cross" "${COMMON[@]}" --out "$out" 2>&1 \
        | tee "results/a05_ext/${tag}.log"
    echo "=== [$(date +%H:%M:%S)] done ${tag} ==="
}

run "$WORLD" medium    narrative world_medium
run "$WORLD" narrative medium    world_narrative
run "$G1D"   medium    narrative g1d_medium
run "$G1D"   narrative medium    g1d_narrative

echo "=== [$(date +%H:%M:%S)] ALL FOUR CELLS DONE ==="
