# Owner 2026-11-04 "разберись что не так с CoT": on the plain rung, silent answer (r0) vs the own CoT read every 32 tokens -
# where CoT is first right, how often it is right and then lost, and whether the end of CoT is worse than the silent answer.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=4-7
lim $UV python3 experiments/rl/cot_vs_routes_probe.py --model $Q --items $C/ciphers_k_context_plain_eval.jsonl --items $C/ciphers_k_question_plain_eval.jsonl --items $C/ciphers_k_mixed_plain_eval.jsonl --per-cell 10 --max-think 1024 --checkpoints $(seq -s, 32 32 1024) --threads 4 --out $RQ/cvr_plain_30.json > $L/qwen_cvr_plain.log 2>&1
echo "cvr plain exit=$?"
