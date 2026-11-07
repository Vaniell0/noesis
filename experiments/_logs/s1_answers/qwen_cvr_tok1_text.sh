# Same probe, 3 context rows (one per alphabet), to read what the own CoT does with the table (cvr_tok1_24: 0 of 24, never closed).
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
lim $UV python3 experiments/rl/cot_vs_routes_probe.py --model $Q --items $C/ciphers_k2_tok1_context_eval.jsonl --per-cell 1 --max-think 512 --checkpoints 256,512 --threads 4 --out $RQ/cvr_tok1_text3.json > $L/qwen_cvr_tok1_text.log 2>&1
echo "cvr tok1 text exit=$?"
