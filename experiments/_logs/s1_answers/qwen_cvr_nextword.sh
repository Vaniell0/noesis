# Bad-sector check: per candidate from the map, silent answer vs own CoT (read every 64 tokens) on "what word comes next".
# CoT right, silent wrong = stored, not read in one pass; neither = lost or unpredictable.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=4-7
lim $UV python3 experiments/rl/cot_vs_routes_probe.py --model $Q --items $C/nextword_vital3_80.jsonl --per-cell 40 --max-think 512 --checkpoints $(seq -s, 64 64 512) --threads 4 --out $RQ/cvr_nextword_80.json > $L/qwen_cvr_nextword.log 2>&1
echo "cvr nextword exit=$?"
