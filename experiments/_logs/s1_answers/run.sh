cd ~/Desktop/projects/noesis
L=experiments/_logs/s1_answers; C=training/corpus_open; R=experiments/rl/results/s1_pilot_g1d04b
M=/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
export OMP_NUM_THREADS=4 NOESIS_PROBE_THREADS=4 PYTHONUNBUFFERED=1
UV=(uv run --offline --no-project --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with 'torch==2.11.0+cpu' --with rwkv --with numpy --with einops --with rwkv-fla --with triton --with 'peft>=0.17.1' --with huggingface_hub --with pyyaml)
lim() { systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=400% taskset -c 8-11 "$@"; }
ITEMS=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 0)
for spec in base:1: ce_T1:1:$R/ce_T1.pth ce_T1:0:$R/ce_T1.pth kl_T2:1:$R/kl_T2.pth; do
  tag=${spec%%:*}; rest=${spec#*:}; tk=${rest%%:*}; w=${rest#*:}
  W=(); [ -n "$w" ] && W=(--weights $w)
  lim $UV python3 experiments/run.py --backend peft --model $M $W --tests m0_pass --out-dir $R/answers_${tag}_M${tk} $ITEMS --m0-ticks $tk > $L/${tag}_M${tk}.log 2>&1
  echo "answers $tag M=$tk exit=$?"
done
