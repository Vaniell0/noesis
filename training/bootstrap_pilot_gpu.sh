#!/usr/bin/env bash
# A1 pilot GPU bootstrap — WSL2 (Ubuntu 22.04/24.04) or native Linux.
#
# Two supported targets, selected via TARGET env var:
#
#   TARGET=wsl2_1050 (default) — GTX 1050 4GB Pascal (sm_61). torch
#       2.4.1 + cu121. No bitsandbytes (Pascal has no 8-bit optim
#       support). Intended for local WSL2 spike / smoke.
#
#   TARGET=cloud_4090 — cloud-rented 4090 spot (sm_89). torch 2.5.1 +
#       cu124. bitsandbytes 0.49.2 wheel installed from
#       ~/.libs/python/. Intended for the actual A1 pilot fine-tune
#       once compute is rented (Selectel Cloud / vast.ai / equivalent).
#       Prereq: user copies bitsandbytes-0.49.2-*.whl into
#       ~/.libs/python/ before running (curl via SOCKS5 works —
#       see docs).
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
# minutes on GTX 1050; ~2-3 min for the 100-step dry-run on a 4090).

set -euo pipefail

NOESIS_DIR="${NOESIS_DIR:-$HOME/noesis}"
CKPT_PATH="${CKPT_PATH:-$HOME/.libs/models/rwkv7/rwkv7-g1d-0.4b/rwkv7-g1d-0.4b.pth}"
PY="${PY:-python3.11}"
TARGET="${TARGET:-wsl2_1050}"

case "${TARGET}" in
    wsl2_1050|cloud_4090) ;;
    *)
        echo "ERROR: TARGET must be 'wsl2_1050' or 'cloud_4090' (got '${TARGET}')"
        exit 1
        ;;
esac

echo "=== noesis A1 pilot GPU bootstrap ==="
echo "NOESIS_DIR = ${NOESIS_DIR}"
echo "CKPT_PATH  = ${CKPT_PATH}"
echo "Python     = ${PY}"
echo "TARGET     = ${TARGET}"

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

case "${TARGET}" in
    wsl2_1050)
        # torch 2.4.x + cu121 is the last combo confirmed to work with sm_61
        # (GTX 1050). Newer torch dropped some Pascal fast-paths but still
        # runs. Pin conservatively.
        echo "--- installing torch 2.4.1 cu121 for Pascal (5-10 min) ---"
        python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
            "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1"
        ;;
    cloud_4090)
        # 4090 is sm_89 (Ada). Torch 2.5.1 + cu124 is the current stable
        # combo for RWKV-PEFT; matches the wheel bitsandbytes 0.49.2
        # was built against.
        echo "--- installing torch 2.5.1 cu124 for Ada (5-10 min) ---"
        python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
            "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1"
        ;;
esac

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

if [[ "${TARGET}" == "cloud_4090" ]]; then
    # bitsandbytes 0.49.2 supports Ada sm_89 (4090). Wheel is the safe
    # path — the pip-index build sometimes fails to link CUDA runtime
    # against the cu124 torch. Users pre-stage the wheel at
    # ~/.libs/python/bitsandbytes-0.49.2-*.whl via SOCKS5 curl. If the
    # wheel is missing, fall back to plain pip (may fail — that's OK,
    # loud is better than silent).
    BNB_WHEEL_DIR="${BNB_WHEEL_DIR:-$HOME/.libs/python}"
    if compgen -G "${BNB_WHEEL_DIR}/bitsandbytes-0.49.2-"*.whl >/dev/null; then
        echo "--- installing bitsandbytes 0.49.2 from local wheel ---"
        python -m pip install --no-index --find-links "${BNB_WHEEL_DIR}" \
            "bitsandbytes==0.49.2"
    else
        echo "WARN: no bitsandbytes wheel in ${BNB_WHEEL_DIR}, trying pip index"
        python -m pip install "bitsandbytes==0.49.2" || {
            echo "ERROR: bitsandbytes install failed. Stage the wheel at"
            echo "  ${BNB_WHEEL_DIR}/bitsandbytes-0.49.2-*.whl and retry."
            exit 1
        }
    fi
fi

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
try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except ImportError:
    print("bitsandbytes: (not installed — expected on wsl2_1050)")
PY

echo "--- vendored trainer discovery ---"
[[ -f "${NOESIS_DIR}/training/rwkv-peft/train.py" ]] \
    || { echo "ERROR: training/rwkv-peft/train.py not present"; exit 1; }
[[ -f "${NOESIS_DIR}/training/train_pilot.py" ]] \
    || { echo "ERROR: training/train_pilot.py not present"; exit 1; }

echo ""
echo "=== bootstrap done (TARGET=${TARGET}). next steps: ==="
echo "  cd ${NOESIS_DIR}"
echo "  source training/.venv-pilot/bin/activate"
echo ""
if [[ "${TARGET}" == "cloud_4090" ]]; then
    echo "  # 1. dry-run harness: slice first 100 glaive-v2 rollouts +"
    echo "  #    generate training/config/pilot_dry100.yaml."
    echo "  python training/dry_run_100.py slice"
    echo ""
    echo "  # 2. smoke the vendored trainer end-to-end (~2-3 min on 4090)."
    echo "  python training/train_pilot.py \\"
    echo "      --config training/config/pilot_dry100.yaml"
    echo ""
    echo "  # 3. verify CE trajectory monotone-decreasing. Fails loud if not."
    echo "  python training/dry_run_100.py verify \\"
    echo "      --run-dir training/runs/pilot_g1d_glaive_v2_dry100"
    echo ""
    echo "  # 4. once dry-run passes, launch the full pilot with pilot.yaml."
    echo "  python training/train_pilot.py"
else
    echo "  # 1. build tokenised fixture (fast, CPU)"
    echo "  python training/tokenize_fixture.py"
    echo ""
    echo "  # 2. baseline smoke (mode=off, alpha=0 already set in pilot.yaml)"
    echo "  python training/train_pilot.py"
    echo ""
    echo "  # 3. after baseline runs, edit pilot.yaml's state_reg block to"
    echo "  #    mode=trajectory_reg, alpha=0.0 (sanity), then alpha>0 sweep."
fi
