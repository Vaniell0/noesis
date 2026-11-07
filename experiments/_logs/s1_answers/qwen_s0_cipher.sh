# Qwen3.5-0.8B S0 on the ciphered rungs (6 ciphers over the same who-lives-where facts), M=0, no hint.
# Which ciphers does one pass fail on - candidates for tick data (plan note 2026-10-29).
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
CI=(--m0-items $C/ciphers_k_context_eval.jsonl --m0-items $C/ciphers_k_question_eval.jsonl --m0-items $C/ciphers_k_mixed_eval.jsonl --m0-per-cell 0)
lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass $CI --out-dir $RQ/s0_cipher_hint0 > $L/qwen_s0_cipher_hint0.log 2>&1
echo "s0 cipher exit=$?"
