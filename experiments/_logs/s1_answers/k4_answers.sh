cd ~/Desktop/projects/noesis
while kill -0 830027 2>/dev/null; do sleep 60; done
until [ -f experiments/rl/results/s1_pilot_g1d04b/ce_T1_k4.json ]; do sleep 60; done
L=experiments/_logs/s1_answers; C=training/corpus_open; R=experiments/rl/results/s1_pilot_g1d04b
M=/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
export OMP_NUM_THREADS=4 NOESIS_PROBE_THREADS=4 PYTHONUNBUFFERED=1
UV=(uv run --offline --no-project --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with 'torch==2.11.0+cpu' --with rwkv --with numpy --with einops --with rwkv-fla --with triton --with 'peft>=0.17.1' --with huggingface_hub --with pyyaml)
lim() { systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=400% taskset -c 8-11 "$@"; }
ITEMS=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 0)
for tk in 4 0; do
  lim $UV python3 experiments/run.py --backend peft --model $M --weights $R/ce_T1_k4.pth --tests m0_pass --out-dir $R/answers_ce_T1_k4_M${tk} $ITEMS --m0-ticks $tk > $L/ce_T1_k4_M${tk}.log 2>&1
  echo "answers ce_T1_k4 M=$tk exit=$?"
done
