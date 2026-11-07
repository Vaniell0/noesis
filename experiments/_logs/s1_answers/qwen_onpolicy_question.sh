# Owner 2026-11-04: second pretraining repairs the read path, knowledge only through on-policy (no foreign tokens).
# Question rung split by country (A train / B held out; gen_ciphers splits by content_id, which kept 22 of 25 eval
# questions in train). Teacher = the same model reading its own CoT cut at the first right checkpoint (CoT 10/10 on
# question, one pass 17/64). Arms: on-policy M=4 (think frame, soft feed) vs on-policy M=0 control, same data and steps.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
big() { systemd-run --user --scope -q -p MemoryMax=14G -p CPUQuota=400% taskset -c 8-11 "$@"; }
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=soft
TA=$C/ciphers_k_question_plain_trainA.jsonl
EV=(--m0-items $C/ciphers_k_question_plain_evalA.jsonl --m0-items $C/ciphers_k_question_plain_evalB.jsonl --m0-per-cell 0)
lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --m0-items $TA --m0-per-cell 0 --out-dir $RQ/s0_question_trainA > $L/qwen_s0_question_trainA.log 2>&1 || exit 1
echo "s0 trainA done"
big $UV python3 experiments/rl/train_latent_s1.py --build-cache --model $Q --items $TA --exclude-right-from $RQ/s0_question_trainA/m0_pass.json --teacher-cot 768 --cot-check 32 --threads 4 --cache-out $RQ/cot_teacher_questionA.pt > $L/qwen_cot_teacher_questionA.log 2>&1 || exit 1
echo "cache done"
for arm in k4:4 m0:0; do
  n=${arm%%:*}; M=${arm##*:}
  big $UV python3 experiments/rl/train_latent_s1.py --model $Q --items $TA --target cache --cache $RQ/cot_teacher_questionA.pt --on-policy --ticks $M --steps 60 --accum 2 --eval-every 30 --eval-max 20 --threads 4 --out $RQ/s1_onpol_qA_$n.json > $L/qwen_s1_onpol_qA_$n.log 2>&1
  echo "train $n exit=$?"
done
for g in k4:4 k4:0 m0:0; do
  n=${g%%:*}; M=${g##*:}
  lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_onpol_qA_$n.pth --tests m0_pass $EV --m0-ticks $M --out-dir $RQ/gate_onpol_qA_${n}_M$M > $L/qwen_gate_onpol_qA_${n}_M$M.log 2>&1
  echo "gate $n M=$M exit=$?"
done
echo "onpolicy question done"
