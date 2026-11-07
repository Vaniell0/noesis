# Hopping-too-late check on Qwen 0.8B mixed (0/120 at M=0): back-patch later-layer residual into an earlier layer at the
# last position, every (src, dst) pair, 40 rows; control = the patch taken from another row. No training.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=4-7
lim $UV python3 experiments/run.py --backend peft --model $Q --tests backpatch --bp-items $C/ciphers_k_mixed_plain_eval.jsonl --bp-n 40 --out-dir $RQ/backpatch_mixed40 > $L/qwen_backpatch_mixed40.log 2>&1
echo "backpatch exit=$?"
