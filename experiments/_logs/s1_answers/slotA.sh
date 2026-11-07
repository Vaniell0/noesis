CORES=4-7; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
for spec in base:0,1,4 kl_T2:0,1,4 kl_T2_k4:0,4 ce_T1_k4:0,4 ce_T1:0,1 kl_T1:0,1; do
  ck=${spec%%:*}; tk=${spec#*:}
  lim $UV python3 experiments/run.py --backend peft --model $M ${=$(w $ck)} --tests chat --cc-ticks $tk --out-dir $R/chat_$ck > $L/chat_$ck.log 2>&1
  echo "chat $ck exit=$?"
done
