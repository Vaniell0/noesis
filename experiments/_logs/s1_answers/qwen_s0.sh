until grep -q "noise base exit=" /home/vaniello/.claude/jobs/cc005570/tmp/noise_hint3.out 2>/dev/null; do sleep 60; done
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B
RQ=experiments/rl/results/qwen35_08b
TRAIN=(--m0-items $C/ciphers_k_context_plain_train.jsonl --m0-items $C/ciphers_k_mixed_plain_train.jsonl --m0-per-cell 0)
for h in 0 3; do
  [ $h = 0 ] && CORES=4-7 || CORES=8-11
  ( lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --out-dir $RQ/s0_plain_hint$h $ITEMS --m0-hint $h > $L/qwen_s0_plain_hint$h.log 2>&1; echo "qwen eval hint=$h exit=$?"
    lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --out-dir $RQ/s0_train_hint$h $TRAIN --m0-hint $h > $L/qwen_s0_train_hint$h.log 2>&1; echo "qwen train hint=$h exit=$?" ) &
done
wait
echo "qwen s0 done"
