CORES=8-11; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
for spec in kl_T2_k4:4 kl_T2_k4:0 ce_T1_k4:4 ce_T1_k4:0; do
  ck=${spec%%:*}; tk=${spec#*:}
  lim $UV python3 experiments/run.py --backend peft --model $M ${=$(w $ck)} --tests m0_pass --out-dir $R/answers_${ck}_M$tk $ITEMS --m0-ticks $tk > $L/${ck}_M$tk.log 2>&1
  echo "answers $ck M=$tk exit=$?"
done
for ck in kl_T2_k4 ce_T1_k4; do
  lim $UV python3 experiments/run.py --backend peft --model $M ${=$(w $ck)} --tests latent_geometry --out-dir $R/gate_$ck \
    --lg-items $C/ciphers_k_context_plain_eval.jsonl --lg-items $C/ciphers_k_mixed_plain_eval.jsonl --lg-items experiments/rl/results/head_ablation/sub_h22.jsonl --lg-per-file 12 --lg-ticks 8 > $L/gate_$ck.log 2>&1
  echo "geom $ck exit=$?"
done
