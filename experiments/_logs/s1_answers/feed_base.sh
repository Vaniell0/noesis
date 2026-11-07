# Base Qwen3.5-0.8B, untrained, fed 4 ticks: the hidden feed vs the soft-token feed (hf_loader NOESIS_HF_FEED).
# Reads whether the latent channel itself is a shock before any S1 training. Compare with s0_plain_hint0 (M=0, 104).
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B
RQ=experiments/rl/results/qwen35_08b
GATE=($ITEMS)   # the same 289 eval items as the night gates
CORES=8-11
for f in soft hidden; do
  NOESIS_HF_FEED=$f lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass "${GATE[@]}" --m0-ticks 4 --out-dir $RQ/s0_plain_M4_feed_$f > $L/qwen_s0_M4_feed_$f.log 2>&1
  echo "feed $f exit=$?"
done
echo "feed base done"
