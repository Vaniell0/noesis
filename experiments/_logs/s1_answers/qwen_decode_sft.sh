# Decode skill first (owner 2026-10-29 "ок в целом"): Qwen 0.8B decodes 0 of 60 short single-token-alphabet ciphers at M=0
# and its own CoT never does the reverse table lookup, so the ciphers have no teacher yet. CE on the gold plaintext, M=0,
# one epoch of 600 gen_ciphers decode rows; then free answers on the decode eval and transfer to the tok1 context eval.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b
big() { systemd-run --user --scope -q -p MemoryMax=14G -p CPUQuota=400% taskset -c 8-11 "$@"; }
big $UV python3 experiments/rl/train_latent_s1.py --model $Q --items $C/ciphers_decode_tok1_short_train.jsonl --target gold --loss ce --ticks 0 --steps 150 --accum 4 --eval-every 50 --eval-max 30 --threads 4 --out $RQ/sft_decode_tok1.json > $L/qwen_sft_decode_tok1.log 2>&1
echo "sft exit=$?"
for W in sft_decode_tok1_s50 sft_decode_tok1_s100 sft_decode_tok1; do
  big $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/$W.pth --tests m0_pass --m0-items $C/ciphers_decode_tok1_short_eval.jsonl --m0-per-cell 0 --out-dir $RQ/gate_$W > $L/qwen_gate_$W.log 2>&1
  echo "gate $W exit=$?"
done
big $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/sft_decode_tok1.pth --tests m0_pass --m0-items $C/ciphers_k2_tok1_context_eval.jsonl --m0-per-cell 0 --out-dir $RQ/gate_sft_decode_tok1_context > $L/qwen_gate_sft_decode_tok1_context.log 2>&1
echo "gate context exit=$?"
