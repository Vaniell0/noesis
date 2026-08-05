#!/bin/bash
# H10 effort-frontier sweep.
# Sweeps K (output token budget) at a fixed N (state-refinement passes).
# Produces one result JSON per K value, then prints a comparison table
# covering all task categories found in the results.
#
# Usage:
#   bash experiments/A0_eval/run_effort_sweep.sh \
#       --model /tmp/step5_merged.pth \
#       --out   /tmp/effort_sweep_step5
#
#   # N-axis: run with N=3 state-refinement passes
#   bash experiments/A0_eval/run_effort_sweep.sh \
#       --model /tmp/step5_merged.pth \
#       --out   /tmp/effort_sweep_n3 \
#       --n-passes 3
#
# Requirements: rwkv backend (BlinkDL rwkv package in active venv).
set -eu

PYTHON=${PYTHON:-python3}
EVAL=experiments/A0_eval/eval.py
BUDGETS=(0 128 512 2048)
MODEL=""
OUT_DIR=""
N_PASSES=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2;;
        --out)      OUT_DIR="$2";  shift 2;;
        --n-passes) N_PASSES="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -z "$MODEL" ]] && { echo "--model required"; exit 1; }
[[ -z "$OUT_DIR" ]] && { echo "--out required"; exit 1; }
mkdir -p "$OUT_DIR"

echo "=== H10 effort sweep: model=$MODEL n_passes=$N_PASSES ==="
for B in "${BUDGETS[@]}"; do
    OUT="$OUT_DIR/effort_np${B}_n${N_PASSES}.json"
    echo "--- K=$B tokens, N=$N_PASSES passes ---"
    $PYTHON "$EVAL" \
        --backend rwkv \
        --model "$MODEL" \
        --num-predict "$B" \
        --n-passes "$N_PASSES" \
        --out "$OUT"
done

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
