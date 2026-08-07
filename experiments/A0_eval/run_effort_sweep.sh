#!/bin/bash
# H10 effort-frontier sweep.
#
# Two sweep modes:
#   N-sweep (primary): vary WKV cycling passes at fixed K=0 (silent).
#     Tests raw WKV accumulation — no training required.
#   K-sweep (secondary): vary intermediate token budget (readout_k) at fixed N.
#     Only meaningful after model is trained with state-quality objective —
#     NOT on trace-imitation SFT. K tokens are invisible, WKV-internal.
#
# Usage:
#   # N-sweep (recommended first):
#   bash experiments/A0_eval/run_effort_sweep.sh \
#       --model /tmp/step7_merged.pth \
#       --sweep n \
#       --n-values "1 2 3 5" \
#       --out /tmp/effort_n_sweep_step7
#
#   # K-sweep at fixed N:
#   bash experiments/A0_eval/run_effort_sweep.sh \
#       --model /tmp/step7_merged.pth \
#       --sweep k \
#       --k-values "0 32 128 512" \
#       --n-passes 1 \
#       --out /tmp/effort_k_sweep_step7
#
# Requirements: rwkv backend (BlinkDL rwkv package in active venv).
set -eu

PYTHON=${PYTHON:-python3}
EVAL=experiments/A0_eval/eval.py
MODEL=""
OUT_DIR=""
SWEEP="n"
N_VALUES="1 2 3 5"
K_VALUES="0 32 128 512"
N_PASSES=1
NUM_PREDICT=64

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2";       shift 2;;
        --out)         OUT_DIR="$2";     shift 2;;
        --sweep)       SWEEP="$2";       shift 2;;
        --n-values)    N_VALUES="$2";    shift 2;;
        --k-values)    K_VALUES="$2";    shift 2;;
        --n-passes)    N_PASSES="$2";    shift 2;;
        --num-predict) NUM_PREDICT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -z "$MODEL" ]] && { echo "--model required"; exit 1; }
[[ -z "$OUT_DIR" ]] && { echo "--out required"; exit 1; }
mkdir -p "$OUT_DIR"

if [[ "$SWEEP" == "n" ]]; then
    echo "=== H10 N-sweep: model=$MODEL k=0 silent ==="
    for N in $N_VALUES; do
        OUT="$OUT_DIR/effort_n${N}_k0.json"
        echo "--- N=$N passes ---"
        $PYTHON "$EVAL" \
            --backend rwkv \
            --model "$MODEL" \
            --num-predict "$NUM_PREDICT" \
            --n-passes "$N" \
            --readout-mode silent \
            --out "$OUT"
    done
else
    echo "=== H10 K-sweep: model=$MODEL n_passes=$N_PASSES ==="
    for K in $K_VALUES; do
        MODE="state_readout"
        [[ "$K" == "0" ]] && MODE="silent"
        OUT="$OUT_DIR/effort_k${K}_n${N_PASSES}.json"
        echo "--- K=$K tokens, N=$N_PASSES passes, mode=$MODE ---"
        $PYTHON "$EVAL" \
            --backend rwkv \
            --model "$MODEL" \
            --num-predict "$NUM_PREDICT" \
            --n-passes "$N_PASSES" \
            --readout-mode "$MODE" \
            --readout-k "$K" \
            --out "$OUT"
    done
fi

echo ""
echo "=== Summary (N=$N_PASSES) ==="

# Print header dynamically from the first result file
FIRST_OUT="$OUT_DIR/effort_np${BUDGETS[0]}_n${N_PASSES}.json"
$PYTHON - "$FIRST_OUT" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
cats = sorted(d.get("aggregate", {}).get("per_category", {}).keys())
print("K    | overall | " + " | ".join(f"{c}" for c in cats))
print("-----|---------|" + "|".join("-" * (len(c) + 2) for c in cats))
EOF

for B in "${BUDGETS[@]}"; do
    OUT="$OUT_DIR/effort_np${B}_n${N_PASSES}.json"
    $PYTHON - "$OUT" "$B" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
k = sys.argv[2]
agg = d.get("aggregate", {})
overall = agg.get("overall_accuracy", 0)
cats = sorted(agg.get("per_category", {}).keys())
cat_vals = " | ".join(
    f"{agg['per_category'][c]['accuracy']:.3f} ({agg['per_category'][c]['correct']}/{agg['per_category'][c]['n']})"
    for c in cats
)
print(f"{k:4s} | {overall:.3f}   | {cat_vals}")
EOF
done
