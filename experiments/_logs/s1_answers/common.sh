cd ~/Desktop/projects/noesis
L=experiments/_logs/s1_answers; C=training/corpus_open; R=experiments/rl/results/s1_pilot_g1d04b
M=/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
export OMP_NUM_THREADS=4 NOESIS_PROBE_THREADS=4 PYTHONUNBUFFERED=1
UV=(uv run --offline --no-project --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with 'torch==2.11.0+cpu' --with rwkv --with numpy --with einops --with rwkv-fla --with triton --with 'peft>=0.17.1' --with huggingface_hub --with pyyaml --with 'transformers==5.20.2')
# 2026-11-04: the system update moved python to 3.13.16 and uv 0.13.5 no longer finds the cached index, so
# `uv run --offline` fails ("torch was not found in the cache"). The env uv built 10-25 for exactly this --with set still
# runs (its python 3.13.15 is in the nix store): use it while it exists, else the line above.
_UVENV=~/.cache/uv/environments-v2/16d9fbc66fa765da/8b8a7f143e4f430d/bin
[ -x $_UVENV/python3 ] && UV=(env PATH=$_UVENV:$PATH)
lim() { systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=400% taskset -c $CORES "$@"; }
ITEMS=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 0)
w() { [ "$1" = base ] && return; echo "--weights $R/$1.pth"; }
