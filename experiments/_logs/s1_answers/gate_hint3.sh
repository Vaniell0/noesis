until grep -q "train hint3 exit=" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/bu07k3j6h.output 2>/dev/null; do sleep 60; done
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
ck=hint3_kl_T2_k4
for spec in 4:4-7 0:8-11; do
  tk=${spec%%:*}; CORES=${spec#*:}
  ( lim $UV python3 experiments/run.py --backend peft --model $M --weights $R/$ck.pth --tests m0_pass --out-dir $R/answers_${ck}_M$tk $ITEMS --m0-ticks $tk > $L/${ck}_M$tk.log 2>&1; echo "answers $ck M=$tk exit=$?" ) &
done
wait
CORES=4-7 lim $UV python3 experiments/run.py --backend peft --model $M --weights $R/$ck.pth --tests chat --cc-ticks 0,4 --out-dir $R/chat_$ck > $L/chat_$ck.log 2>&1
echo "chat $ck exit=$?"
