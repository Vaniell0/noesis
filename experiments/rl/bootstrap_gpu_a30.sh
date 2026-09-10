#!/usr/bin/env bash
# GPU bootstrap for experiments/rl/* on a rented Selectel A30 (24GB, Ampere,
# sm_80) -- NOT training/bootstrap_pilot_gpu.sh, which targets a different,
# older pilot trainer (training/train_pilot.py, G1d-0.4B) with a different,
# larger dependency set (deepspeed compiled from source, bitsandbytes).
#
# This script installs only what experiments/rl/train_think_distill.py and
# wkv_loop.py actually import (verified 2026-09-05 by grepping real imports,
# not copied from training/rwkv-peft/requirements.txt on faith) -- see
# requirements-gpu.txt alongside this script for the reasoning on what was
# deliberately left out (deepspeed, bitsandbytes, lightning, wandb, ...).
#
# What this script does NOT do:
#   * Install the NVIDIA driver -- Selectel's GPU images ship one; this only
#     verifies nvidia-smi works.
#   * Copy the model/checkpoint files. Needed before a real run --
#     MERGE ON THE VM, don't transfer the premerged file (decided
#     2026-09-05): copy only
#       - models/rwkv7-g1i-2.9b-20260805-ctx16384.pth (base, ~5.9GB)
#       - experiments/rl/checkpoints/g1i_zlk_phase1_v3_step500/ (182MB --
#         384 per-param lora_A/lora_B files + meta.pt, which also carries
#         the ThinkChain `chain` parameter separately (mlp_delta), loaded
#         via checkpoint.py::load_checkpoint(..., mlp_delta=...) AFTER
#         the merge below -- chain is a new param, not a base-weight
#         delta, so merging never touches it)
#     then run, on the VM:
#       training/.venv/bin/python experiments/rl/merge_checkpoint_dir.py
#     (defaults already point at the two paths above and write
#     models/rwkv7-g1i-2.9b-zlk_phase1_v3-step500-merged.pth, ~10.9GB --
#     verified working locally 2026-09-05, output zip-valid, 192 LoRA
#     pairs merged). This sends ~6.1GB instead of ~16.8GB (base + a
#     redundant already-merged copy) -- CPU-only, no GPU/peft/rwkvfla
#     needed for the merge itself, just torch.
#     Still NOT yet verified: whether load_checkpoint(..., mlp_delta=...)
#     works when the base model is loaded with --lora-r 0 (full-FT, no
#     LoRA slots) -- first thing to test on this VM before a real launch.
#   * Copy the training corpus -- found MISSING from this checklist
#     2026-09-05 (train_think_distill.py's own --data default pointed at a
#     file that didn't exist on disk). Both training/corpus_open/*.jsonl and
#     training/tokenised/ are gitignored, so a git clone alone will NOT
#     bring these -- must be copied (or regenerated) explicitly:
#       - training/tokenised/g1i_warmup_v3_eos_train.pt (36MB, 10529 ex.)
#       - training/tokenised/g1i_warmup_v3_eos_val.pt   (4MB, 1171 ex.)
#       - training/corpus_open/relax_v1.jsonl (692KB, 1776 ex. -- the M=0
#         anti-forgetting corpus; REQUIRED, not optional, if --m-weights'
#         w0 > 0 (train_think_distill.py asserts on this) -- today's
#         Phase 1.5 plan mixes M=0 in, so this file is load-bearing.
#         Found missing from this checklist 2026-09-05, same pass as the
#         two .pt files above -- it exists on disk (checked), just wasn't
#         listed here yet.
#     Regenerated locally 2026-09-05 from training/corpus_open/matrix_tasks.jsonl
#     (65797 rows, present) via the exact documented, seed=7 deterministic
#     recipe -- train count (10529) matches the original step500 run's
#     recorded figure exactly:
#       training/.venv/bin/python training/scripts/gen_g1i_warmup.py \
#           --tasks training/corpus_open/matrix_tasks.jsonl \
#           --out training/corpus_open/g1i_warmup_v3.jsonl --per-bucket 300
#       training/.venv/bin/python training/tokenize_plain_cot.py \
#           --input training/corpus_open/g1i_warmup_v3.jsonl \
#           --out-train training/tokenised/g1i_warmup_v3_eos_train.pt \
#           --out-val   training/tokenised/g1i_warmup_v3_eos_val.pt --val-pct 10
#     Simplest for the VM: just copy the two ready-made .pt files above
#     (40MB total, trivial next to the multi-GB model files) rather than
#     re-running the CPU-only regeneration there.
#   * Clone the repo. Assumed already present at NOESIS_DIR.
#
# Runtime: env setup ~5-10 min (no deepspeed compile, unlike the older
# pilot bootstrap -- deepspeed is stubbed by loader.py, never installed).

set -euo pipefail

NOESIS_DIR="${NOESIS_DIR:-$HOME/noesis}"
PY="${PY:-python3.11}"
VENV="${NOESIS_DIR}/experiments/rl/.venv-a30"

echo "=== noesis experiments/rl GPU bootstrap (target: Selectel A30, sm_80) ==="
echo "NOESIS_DIR = ${NOESIS_DIR}"
echo "Python     = ${PY}"
echo "venv       = ${VENV}"

if ! command -v "${PY}" >/dev/null 2>&1; then
    echo "ERROR: ${PY} not found. Install Python 3.11 first."
    echo "  Ubuntu: sudo apt install python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found -- GPU driver missing or image is wrong."
    exit 1
fi

echo "--- nvidia-smi ---"
nvidia-smi | head -20

# Swap safety net -- found 2026-09-10: this flavor's 8GB RAM has near-zero
# margin against a 5.8GB model checkpoint (merge_lora.py's CPU-RAM path
# pegged memory at 100% and had to be killed). Swap doesn't fix a script
# holding too much RAM, but it turns a hard hang/OOM-kill into graceful
# (if slow) degradation -- worth having regardless of which path a given
# script takes. 8GB file, /swapfile -- skip if swap already configured.
if [[ "$(swapon --show | wc -l)" -eq 0 ]]; then
    echo "--- adding 8GB swap (none configured) ---"
    fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
    swapon --show
else
    echo "--- swap already configured, skipping ---"
    swapon --show
fi

if [[ ! -d "${NOESIS_DIR}" ]]; then
    echo "ERROR: NOESIS_DIR=${NOESIS_DIR} not found. Copy/clone the repo there first."
    exit 1
fi

cd "${NOESIS_DIR}"

if [[ ! -d "${VENV}" ]]; then
    echo "--- creating venv at ${VENV} ---"
    "${PY}" -m venv "${VENV}"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

# --- torch -----------------------------------------------------------------
# NOT hard-pinned to one version+cu-tag: A30 is Ampere (sm_80), which every
# torch/CUDA combo released in the last several years supports natively --
# unlike training/bootstrap_pilot_gpu.sh's Pascal (sm_61) target, there is no
# narrow known-good window to pin here. Check the driver's max supported CUDA
# via nvidia-smi's own header (top-right of the table above) and pick a
# matching cu12x wheel from https://pytorch.org/get-started/locally/ if the
# default below doesn't match what's on this box.
echo "--- installing torch (cu128 wheel; adjust if the driver needs a different CUDA) ---"
# Found 2026-09-10 on the real Brenna A30 (driver reports CUDA 13.0): cu124's
# index is stale, tops out at torch 2.6.0 -- doesn't satisfy >=2.9. cu128 has
# 2.9.0 through 2.14.0. Re-check `pip index versions torch` against
# https://download.pytorch.org/whl/<tag>/ if this drifts again.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.9"

echo "--- installing experiments/rl requirements ---"
python -m pip install -r experiments/rl/requirements-gpu.txt

# Found 2026-09-10 on the real Brenna A30: without headers, rwkv-fla's Triton
# kernels fail to JIT-compile ("Python.h: No such file or directory") and
# silently fall back to CPU -- no error, just catastrophically slow on a
# real forward pass. python3-dev ships the missing Python.h.
echo "--- python3-dev (Triton JIT needs Python.h) ---"
apt-get install -y python3-dev >/dev/null 2>&1 || echo "  (apt install failed -- may need sudo, or already present)"

echo "--- python sanity ---"
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
import peft
print("peft:", peft.__version__)
import rwkvfla
print("rwkvfla: importable")
import rwkv
print("rwkv:", rwkv.__version__ if hasattr(rwkv, "__version__") else "(no __version__ attr)")
PY

echo ""
echo "=== bootstrap done. Before a real training launch: ==="
echo "  1. Copy base model + checkpoint dir + tokenised .pt data files"
echo "     (see this script's header comment for exact paths)."
echo "  2. python experiments/rl/merge_checkpoint_dir.py  # merge on the VM"
echo "  3. Test load_checkpoint(..., mlp_delta=...) on a --lora-r 0 model --"
echo "     NOT yet verified compatible, do this before any long run."
echo "  4. cd ${NOESIS_DIR} && source experiments/rl/.venv-a30/bin/activate"
echo "  5. python experiments/rl/train_think_distill.py --help  # confirm CLI loads"
