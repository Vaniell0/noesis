# Base Qwen3.5-0.8B with the ticks inside its own think block (hf_loader NOESIS_HF_FRAME=think): does the latent
# channel keep the base answers (104 at M=0), and do 16 soft ticks make any two-hop item (mixed 0 at M=0)?
until grep -q "chat qwen done" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/bn4ub0amb.output 2>/dev/null; do sleep 60; done
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
for run in soft:4 hidden:4 soft:16; do
  f=${run%%:*}; M=${run#*:}
  NOESIS_HF_FRAME=think NOESIS_HF_FEED=$f lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass $ITEMS --m0-ticks $M --out-dir $RQ/s0_plain_M${M}_think_$f > $L/qwen_s0_M${M}_think_$f.log 2>&1
  echo "think $f M=$M exit=$?"
done
echo "frame base done"
