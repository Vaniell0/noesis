until grep -q "qwen s0 done" /home/vaniello/.claude/jobs/cc005570/tmp/qwen_s0.out 2>/dev/null; do sleep 60; done
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
NF=(--ln-free $C/ciphers_k_context_plain_eval.jsonl --ln-free $C/ciphers_k_mixed_plain_eval.jsonl --ln-limit 40 --ln-rollouts 4 --ln-sigma 0,0.5,1,2 --ln-k 16 --ln-free-ticks 4)
for spec in hint3_kl_T2_k4:4-7 kl_T2_k4:8-11; do
  ck=${spec%%:*}; CORES=${spec#*:}
  ( lim $UV python3 experiments/run.py --backend peft --model $M ${=$(w $ck)} --tests latent_noise $NF --out-dir $R/noisefree_$ck > $L/noisefree_$ck.log 2>&1; echo "noisefree $ck exit=$?" ) &
done
wait
