source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b
SEL=(--items $C/ciphers_k_mixed_plain_train.jsonl --right-from $RQ/s0_train_hint3/m0_pass.json --exclude-right-from $RQ/s0_train_hint0/m0_pass.json --limit 150)
GATE=($ITEMS)   # all 289 eval items: mixed is the target, context/question show whether the base balance is kept
big() { systemd-run --user --scope -q -p MemoryMax=14G -p CPUQuota=400% taskset -c 8-11 "$@"; }
for arm in k4:4 m0:0; do
  name=${arm%%:*}; tk=${arm#*:}
  big $UV python3 experiments/rl/train_latent_s1.py --model $Q $SEL --target cache --cache $RQ/hint3_teacher_mixed150.pt --loss kl --temp 2 --ticks $tk --steps 80 --accum 2 --eval-every 40 --eval-max 40 --threads 4 --out $RQ/s1_hint3_$name.json > $L/qwen_s1_hint3_$name.log 2>&1
  echo "train $name exit=$?"
  for M in 4 0; do
    big $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_hint3_$name.pth --tests m0_pass $GATE --m0-ticks $M --out-dir $RQ/gate_hint3_${name}_M$M > $L/qwen_gate_hint3_${name}_M$M.log 2>&1
    echo "gate $name M=$M exit=$?"
  done
done
echo "night qwen done"
