#!/bin/bash
# H10 full N×K×mode matrix sweep.
#
# Sweeps all three H10 axes:
#   N (--n-passes)     : {1, 2, 3, 5}
#   K (--num-predict)  : {0, 32, 128, 512}   (K=0 only meaningful for silent mode)
#   mode               : {silent, prompt_cot, state_readout}
#
# For state_readout, --readout-k controls the self-report budget.
# --num-predict is always the scored answer budget.
#
# Usage:
#   NOESIS_EVAL_DEVICE=cuda bash experiments/A0.8_refine/run_matrix_sweep.sh \
#       --model /tmp/step6_merged.pth \
#       --out   /tmp/h10_matrix_step6
#
# Approximate runtime: ~3h on RTX 4090 (144 cells × ~75 tasks × ~3s/task).
# Run after A0 baseline confirms model has non-zero signal.
set -eu

PYTHON=${PYTHON:-python3}
EVAL=experiments/A0_eval/eval.py
MODEL=""
OUT_DIR=""
READOUT_K=64   # state_readout self-report budget

N_VALUES=(1 2 3 5)
K_VALUES=(32 128 512)
MODES=(silent prompt_cot state_readout)

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)     MODEL="$2";     shift 2;;
        --out)       OUT_DIR="$2";   shift 2;;
        --readout-k) READOUT_K="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -z "$MODEL" ]] && { echo "--model required"; exit 1; }
[[ -z "$OUT_DIR" ]] && { echo "--out required"; exit 1; }
mkdir -p "$OUT_DIR"

echo "=== H10 N×K×mode matrix sweep ==="
echo "    model=$MODEL"
echo "    N=${N_VALUES[*]}, K=${K_VALUES[*]}, modes=${MODES[*]}"
echo "    readout_k=$READOUT_K"
echo ""

for N in "${N_VALUES[@]}"; do
    for MODE in "${MODES[@]}"; do
        if [[ "$MODE" == "silent" ]]; then
            # silent ignores K — run once per N
            OUT="$OUT_DIR/n${N}_silent.json"
            if [[ -f "$OUT" ]]; then
                echo "[skip] $OUT already exists"
                continue
            fi
            echo "--- N=$N mode=silent ---"
            $PYTHON "$EVAL" \
                --backend rwkv --model "$MODEL" \
                --n-passes "$N" \
                --readout-mode silent \
                --num-predict 256 \
                --out "$OUT"
        else
            for K in "${K_VALUES[@]}"; do
                if [[ "$MODE" == "state_readout" ]]; then
                    OUT="$OUT_DIR/n${N}_k${K}_readout${READOUT_K}.json"
                else
                    OUT="$OUT_DIR/n${N}_k${K}_cot.json"
                fi
                if [[ -f "$OUT" ]]; then
                    echo "[skip] $OUT already exists"
                    continue
                fi
                echo "--- N=$N K=$K mode=$MODE ---"
                $PYTHON "$EVAL" \
                    --backend rwkv --model "$MODEL" \
                    --n-passes "$N" \
                    --num-predict "$K" \
                    --readout-mode "$MODE" \
                    --readout-k "$READOUT_K" \
                    --out "$OUT"
            done
        fi
    done
done

echo ""
echo "=== Pareto analysis ==="
$PYTHON experiments/A0.8_refine/analyze_frontier.py --dir "$OUT_DIR"
