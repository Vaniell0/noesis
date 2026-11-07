until grep -q "chat hint3_kl_T2_k4 exit=" /home/vaniello/.claude/jobs/cc005570/tmp/gate_hint3.out 2>/dev/null; do sleep 60; done
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
NI=(--ln-items experiments/rl/results/head_ablation/sub_h22.jsonl --ln-m 1,4)
for spec in hint3_kl_T2_k4:4-7 kl_T2_k4:8-11; do
  ck=${spec%%:*}; CORES=${spec#*:}
  ( lim $UV python3 experiments/run.py --backend peft --model $M ${=$(w $ck)} --tests latent_noise $NI --out-dir $R/noise_$ck > $L/noise_$ck.log 2>&1; echo "noise $ck exit=$?" ) &
done
wait
CORES=4-7 lim $UV python3 experiments/run.py --backend peft --model $M --tests latent_noise $NI --out-dir $R/noise_base > $L/noise_base.log 2>&1
echo "noise base exit=$?"
