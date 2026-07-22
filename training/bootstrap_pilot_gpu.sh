#!/usr/bin/env bash
# A1 pilot GPU bootstrap — WSL2 (Ubuntu 22.04/24.04) or native Linux.
#
# Target hardware in scope: GTX 1050 4GB (Pascal, CC 6.1). Anything
# newer works too — this script does not pin sm_ arch. cu121 wheel used
# because torch 2.4/2.5 dropped cu118 for sm_61 in some builds.
#
# What this script does NOT do:
#   * Install NVIDIA driver on Windows host (WSL2) or on native Linux
#     — assumed done by user. `nvidia-smi` must work before running.
#   * Copy the .pth checkpoint. The pilot expects
#     ~/.libs/models/rwkv7/rwkv7-g1d-0.4b/rwkv7-g1d-0.4b.pth — user
#     scps it from the dev box.
#   * Clone the noesis repo. Assumed cloned to ~/noesis (adjust NOESIS_DIR).
#
# Windows-native (no WSL2) is NOT recommended: deepspeed + RWKV-PEFT
# stack is Linux-first; native Windows setup will fight with pinned
# CUDA toolchain and a bnb-windows fork that lacks 8-bit optim support
# on Pascal.
#
# Runtime: env setup ~5-15 min (torch wheel download), pilot smoke depends
# on config (default 3 epochs × ~2600 tokens × chunk_ctx=1 → tens of
# minutes on GTX 1050).

set -euo pipefail

NOESIS_DIR="${NOESIS_DIR:-$HOME/noesis}"
CKPT_PATH="${CKPT_PATH:-$HOME/.libs/models/rwkv7/rwkv7-g1d-0.4b/rwkv7-g1d-0.4b.pth}"
PY="${PY:-python3.11}"

echo "=== noesis A1 pilot GPU bootstrap ==="
echo "NOESIS_DIR = ${NOESIS_DIR}"
echo "CKPT_PATH  = ${CKPT_PATH}"
echo "Python     = ${PY}"

# --- Preflight ----------------------------------------------------------------

if ! command -v "${PY}" >/dev/null 2>&1; then
    echo "ERROR: ${PY} not found. Install Python 3.11 first."
    echo "  Ubuntu/WSL2: sudo apt install python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA driver + CUDA on"
    echo "  the host (Windows for WSL2, or Linux directly). See:"
    echo "  https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
    exit 1
fi

echo "--- nvidia-smi ---"
nvidia-smi | head -20

if [[ ! -d "${NOESIS_DIR}" ]]; then
    echo "ERROR: NOESIS_DIR=${NOESIS_DIR} not found. git clone the repo there first."
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "WARN: checkpoint not at ${CKPT_PATH}."
    echo "      scp it from the dev box before starting the actual smoke run."
    echo "      (Continuing so venv can be built independently.)"
fi

cd "${NOESIS_DIR}"

# --- venv ---------------------------------------------------------------------

VENV="${NOESIS_DIR}/training/.venv-pilot"
if [[ ! -d "${VENV}" ]]; then
    echo "--- creating venv at ${VENV} ---"
    "${PY}" -m venv "${VENV}"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel setuptools >/dev/null

# --- torch + deps -------------------------------------------------------------

# torch 2.4.x + cu121 is the last combo confirmed to work with sm_61
# (GTX 1050). Newer torch dropped some Pascal fast-paths but still
# runs. Pin conservatively.
echo "--- installing torch cu121 (this can take 5-10 min) ---"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1"

echo "--- installing pilot deps ---"
python -m pip install \
    "pytorch-lightning==2.4.0" \
    "lightning-utilities>=0.11" \
    "pyyaml" \
    "ninja" \
    "wheel" \
    "einops" \
    "packaging" \
    "peft>=0.10"

# deepspeed is required by rwkvt.rwkv7.model at import time. On WSL2/Linux
# with a recent kernel this installs cleanly from source; the wheel job
# can take 3-8 minutes on a low-end CPU.
echo "--- installing deepspeed (compile step: 3-8 min) ---"
DS_BUILD_OPS=0 python -m pip install "deepspeed>=0.14,<0.16" || {
    echo "WARN: deepspeed install failed. Options:"
    echo "  1. Retry with DS_BUILD_OPS=0 (already tried above)."
    echo "  2. Try an older release: pip install deepspeed==0.13.5"
    echo "  3. If neither works, this bootstrap is not viable on your"
    echo "     kernel — fall back to cloud burst."
    exit 1
}

# --- Sanity ------------------------------------------------------------------

echo "--- python sanity ---"
python - <<'PY'
import torch, lightning, deepspeed, yaml
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("lightning:", lightning.__version__)
print("deepspeed:", deepspeed.__version__)
PY

echo "--- vendored trainer discovery ---"
[[ -f "${NOESIS_DIR}/training/rwkv-peft/train.py" ]] \
    || { echo "ERROR: training/rwkv-peft/train.py not present"; exit 1; }
[[ -f "${NOESIS_DIR}/training/train_pilot.py" ]] \
    || { echo "ERROR: training/train_pilot.py not present"; exit 1; }

echo ""
echo "=== bootstrap done. next step: ==="
echo "  cd ${NOESIS_DIR}"
echo "  source training/.venv-pilot/bin/activate"
echo ""
echo "  # 1. build tokenised fixture (fast, CPU)"
echo "  python training/tokenize_fixture.py"
echo ""
echo "  # 2. baseline smoke (mode=off, alpha=0 already set in pilot.yaml)"
echo "  python training/train_pilot.py"
echo ""
echo "  # 3. after baseline runs, edit pilot.yaml's state_reg block to"
echo "  #    mode=trajectory_reg, alpha=0.0 (sanity), then alpha>0 sweep."
