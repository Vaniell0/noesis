while kill -0 1139329 2>/dev/null; do sleep 30; done
echo "tickcurve M=1 done"
CORES=8-11; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
HI=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 0)
# 1. reachability: base 0.4B at M=0 with hint level 1..3 (the teacher for hinted S1)
for h in 1 2 3; do
  lim $UV python3 experiments/run.py --backend peft --model $M --tests m0_pass --out-dir experiments/rl/results/s0_g1d04b_hint$h $HI --m0-hint $h > $L/reach_h$h.log 2>&1
  echo "reach hint=$h exit=$?"
done
# 2. the rest of the tick curve for kl_T2_k4
SUB=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 12)
for tk in 2 3 5 6 8; do
  lim $UV python3 experiments/run.py --backend peft --model $M --weights $R/kl_T2_k4.pth --tests m0_pass --out-dir $R/tickcurve_kl_T2_k4/M$tk $SUB --m0-ticks $tk > $L/tickcurve_M$tk.log 2>&1
  echo "tickcurve M=$tk exit=$?"
done
lim $UV python3 experiments/rl/state_readout_lens.py --model $M --out experiments/rl/results/state_readout_lens_g1d04b_L16.json --layers 4,8,11,12,13,16,20,23 --abl-layers 16 --cont-len 16 > $L/lens_base_L16.log 2>&1
echo "lens base L16 exit=$?"
