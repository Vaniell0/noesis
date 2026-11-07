# Own-CoT curve on short single-token-alphabet ciphers (S0 M=0: 0 of 240), cot_vs_routes_probe with --items:
# answer read every 32 CoT tokens, so first-right and right-then-lost both show. 4 rows per (file, alphabet) = 24.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
lim $UV python3 experiments/rl/cot_vs_routes_probe.py --model $Q --items $C/ciphers_k2_tok1_context_eval.jsonl --items $C/ciphers_k2_tok1_mixed_eval.jsonl --per-cell 4 --max-think 1024 --checkpoints $(seq -s, 32 32 1024) --threads 4 --out $RQ/cvr_tok1_24.json > $L/qwen_cvr_tok1.log 2>&1
echo "cvr tok1 exit=$?"
