# Qwen S1 hint-3, 4 ticks inside the think block (NOESIS_HF_FRAME=think, soft feed): same data/steps as night_qwen.sh k4.
# Predictions: start loss near the M=0 control's 2.6 (not 6.1); free answers at M=4 close to the control's 195.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b
SEL=(--items $C/ciphers_k_mixed_plain_train.jsonl --right-from $RQ/s0_train_hint3/m0_pass.json --exclude-right-from $RQ/s0_train_hint0/m0_pass.json --limit 150)
big() { systemd-run --user --scope -q -p MemoryMax=14G -p CPUQuota=400% taskset -c 8-11 "$@"; }
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=soft
big $UV python3 experiments/rl/train_latent_s1.py --model $Q $SEL --target cache --cache $RQ/hint3_teacher_mixed150.pt --loss kl --temp 2 --ticks 4 --steps 80 --accum 2 --eval-every 40 --eval-max 40 --threads 4 --out $RQ/s1_hint3_think_k4.json > $L/qwen_s1_hint3_think_k4.log 2>&1
echo "train think_k4 exit=$?"
for M in 4 0; do
  big $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_hint3_think_k4.pth --tests m0_pass $ITEMS --m0-ticks $M --out-dir $RQ/gate_hint3_think_k4_M$M > $L/qwen_gate_hint3_think_k4_M$M.log 2>&1
  echo "gate think_k4 M=$M exit=$?"
done
echo "think k4 done"
