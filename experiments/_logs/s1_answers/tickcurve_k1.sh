until grep -q "lens kl_T2_k4 exit=" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/bxtcvt28l.output 2>/dev/null; do sleep 60; done
CORES=4-7; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
SUB=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 12)
for tk in 0 1 2 4 8; do
  lim $UV python3 experiments/run.py --backend peft --model $M --weights $R/kl_T2.pth --tests m0_pass --out-dir $R/tickcurve_kl_T2/M$tk $SUB --m0-ticks $tk > $L/tickcurve_k1_M$tk.log 2>&1
  echo "tickcurve kl_T2 M=$tk exit=$?"
done
