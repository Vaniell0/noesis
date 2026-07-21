#!/usr/bin/env bash
# A0.5 Stage-1 grid — G1d-0.4B pair only (medium, narrative).
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true

G1D='/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth'

COMMON=(--seeds 3 --max-new-tokens 128 --k-checkpoints 4
        --sigmas 0.005,0.01,0.02,0.05,0.1,0.2
        --scales 0.5,1.5,2.0
        --sample-layers 0,4,8,12,16,20
        --continuation-steps 6)

run() {
    local prompt="$1" cross="$2" tag="$3"
    local out="results/a05_ext/${tag}"
    echo "=== [$(date +%H:%M:%S)] launching ${tag} ==="
    python3 a05_run.py --model "$G1D" --prompt "$prompt" \
        --cross-prompt "$cross" "${COMMON[@]}" --out "$out" 2>&1 \
        | tee "results/a05_ext/${tag}.log"
    echo "=== [$(date +%H:%M:%S)] done ${tag} ==="
}

run medium    narrative g1d_medium
run narrative medium    g1d_narrative
echo "=== [$(date +%H:%M:%S)] G1D PAIR DONE ==="
