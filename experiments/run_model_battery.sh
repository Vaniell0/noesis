#!/usr/bin/env bash
# Run the full probe battery for one model — the GPU-session execution
# matrix from the plan, as one command instead of five manually-sequenced
# ones. Written 2026-08-18, direct answer to "what's the framework still
# missing to actually run experiments."
#
# Continues past a single probe's failure (each step is independent and
# already writes its own _meta-stamped result) rather than aborting the
# whole battery — losing five probes because the sixth crashed would be
# exactly the kind of GPU-time waste this script exists to prevent.
#
# Usage:
#   experiments/run_model_battery.sh --model <path> --label <name> \
#       [--device cuda|cpu] [--n-tokens 768] [--skip-ib] [--status-port 8123]
#
# --status-port: serves experiments/_common/results/ over plain HTTP
# (stdlib `python -m http.server`, no new server code) for the duration of
# the run, so status.json / *.json results are viewable at
# http://<host>:<port>/<label>_battery/status.json from outside — a
# browser or curl, not another SSH session through a tool whose session
# can time out mid-check (see experiments/_common/heartbeat.py).
#
# Does NOT run a05_run.py/a05_analyze.py or lora_rank_analysis.py — those
# need per-model choices (prompt selection, which base to diff against)
# that don't belong in a blind loop. See the plan's execution matrix for
# those commands.
set -uo pipefail

MODEL=""
LABEL=""
DEVICE="cuda"
N_TOKENS="768"
SKIP_IB=0
STATUS_PORT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --n-tokens) N_TOKENS="$2"; shift 2 ;;
    --skip-ib) SKIP_IB=1; shift ;;
    --status-port) STATUS_PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL" || -z "$LABEL" ]]; then
  echo "usage: $0 --model <path> --label <name> [--device cuda|cpu] [--n-tokens N] [--skip-ib] [--status-port PORT]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-activate a venv if `python` doesn't already resolve to one with our
# deps — don't require the operator to remember to `source .../activate`
# first. Checks both venv names seen in this project's history
# (this machine's `.venv`, prior VMs' `.venv-pilot`) rather than assuming
# either is THE convention; first one that has torch wins.
if ! python -c "import torch" >/dev/null 2>&1; then
  for candidate in "$REPO_ROOT/training/.venv" "$REPO_ROOT/training/.venv-pilot"; do
    if [[ -f "$candidate/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$candidate/bin/activate"
      if python -c "import torch" >/dev/null 2>&1; then
        echo "[run_model_battery] activated venv: $candidate"
        break
      fi
    fi
  done
fi
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "!!! no venv with torch found (checked training/.venv, training/.venv-pilot) — activate one manually" >&2
  exit 1
fi

HTTP_PID=""
if [[ -n "$STATUS_PORT" ]]; then
  python -m http.server "$STATUS_PORT" --directory experiments/_common/results \
    > /tmp/noesis_status_server.log 2>&1 &
  HTTP_PID=$!
  echo "[run_model_battery] status server: http://0.0.0.0:${STATUS_PORT}/${LABEL}_battery/status.json (pid $HTTP_PID)"
  trap '[[ -n "$HTTP_PID" ]] && kill "$HTTP_PID" 2>/dev/null' EXIT
fi

OUT_BASE="experiments/_common/results/${LABEL}_battery"
FAILED=0

echo "=== [$LABEL] fast battery (ipc,mlp_ipc,rlens,jlens,rich,think_geometry) @ n-tokens=${N_TOKENS} ==="
python experiments/run.py --model "$MODEL" --device "$DEVICE" \
  --tests ipc,mlp_ipc,rlens,jlens,rich,think_geometry \
  --n-tokens "$N_TOKENS" --out-dir "$OUT_BASE" \
  || { echo "!!! [$LABEL] fast battery FAILED (see above) — continuing"; FAILED=1; }

if [[ "$SKIP_IB" -eq 0 ]]; then
  echo "=== [$LABEL] ib_probe (full corpus — ignores --n-tokens, see run.py --list) ==="
  python experiments/run.py --model "$MODEL" --device "$DEVICE" \
    --tests ib_probe --out-dir "${OUT_BASE}_ib" \
    || { echo "!!! [$LABEL] ib_probe FAILED (see above) — continuing"; FAILED=1; }
else
  echo "=== [$LABEL] ib_probe skipped (--skip-ib) ==="
fi

echo "=== [$LABEL] regenerating RESULTS.md index ==="
python experiments/regenerate_results.py

if [[ "$FAILED" -eq 1 ]]; then
  echo "=== [$LABEL] battery finished WITH FAILURES — check output above for which step(s) ==="
  exit 1
fi
echo "=== [$LABEL] battery finished cleanly ==="
