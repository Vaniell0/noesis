#!/bin/bash
# A0.6 effort-budget sweep.
# Runs the same model on the same tasks with different --num-predict budgets.
# Produces one result JSON per budget, then prints a comparison table.
#
# Usage:
#   bash experiments/A0_eval/run_effort_sweep.sh \
#       --model /tmp/step4_merged_step3500.pth \
#       --out   /tmp/effort_sweep
#
# Requirements: rwkv backend (BlinkDL rwkv package in PATH python).
set -eu

PYTHON=${PYTHON:-python3}
EVAL=experiments/A0_eval/eval.py
TASKS=experiments/A0_eval/tasks.jsonl
BUDGETS=(0 128 512 2048)
MODEL=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --out)   OUT_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

[[ -z "$MODEL" ]] && { echo "--model required"; exit 1; }
[[ -z "$OUT_DIR" ]] && { echo "--out required"; exit 1; }
mkdir -p "$OUT_DIR"

echo "=== A0.6 effort sweep: $MODEL ==="
for B in "${BUDGETS[@]}"; do
    OUT="$OUT_DIR/effort_np${B}.json"
    echo "--- budget=$B tokens ---"
    $PYTHON "$EVAL" \
        --backend rwkv \
        --model "$MODEL" \
        --tasks "$TASKS" \
        --num-predict "$B" \
        --out "$OUT"
done

echo ""
echo "=== Summary ==="
echo "budget | overall | bit_decoding | arithmetic_chain | scheduling"
for B in "${BUDGETS[@]}"; do
    OUT="$OUT_DIR/effort_np${B}.json"
    $PYTHON - <<EOF
import json
d = json.load(open("$OUT"))
agg = d.get("aggregate", {})
overall = agg.get("overall_pass_rate", 0)
cats = agg.get("by_category", {})
bd  = cats.get("bit_decoding",      {}).get("pass_rate", 0)
ac  = cats.get("arithmetic_chain",  {}).get("pass_rate", 0)
sc  = cats.get("scheduling",        {}).get("pass_rate", 0)
print(f"$B     | {overall:.3f}   | {bd:.3f}        | {ac:.3f}            | {sc:.3f}")
EOF
done
