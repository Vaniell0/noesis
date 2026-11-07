# Rerun of the on-policy question chain with the fixes read 2026-11-05: frozen bf16 teacher (the self-teacher drifted:
# "Answer: <answer in capital letters>", quote loops), student T 0.7, anchor = rows the base reads (frozen teacher on the
# plain prompt). Same cache (own CoT cut at first right), same 60 steps; arm M=4 vs control M=0.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
big() { systemd-run --user --scope -q -p MemoryMax=18G -p CPUQuota=400% taskset -c 8-11 "$@"; }
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=soft
TA=$C/ciphers_k_question_plain_trainA.jsonl
EV=(--m0-items $C/ciphers_k_question_plain_evalA.jsonl --m0-items $C/ciphers_k_question_plain_evalB.jsonl --m0-per-cell 0)
for arm in k4:4 m0:0; do
  n=${arm%%:*}; M=${arm##*:}
  big $UV python3 experiments/rl/train_latent_s1.py --model $Q --items $TA --target cache --cache $RQ/cot_teacher_questionA.pt --on-policy --frozen-teacher --anchor-items $C/anchor_read_qA_ctx40.jsonl --student-temp 0.7 --ticks $M --steps 60 --accum 2 --eval-every 30 --eval-max 20 --threads 4 --out $RQ/s1_onpolF_qA_$n.json > $L/qwen_s1_onpolF_qA_$n.log 2>&1
  echo "train $n exit=$?"
done
for g in k4:4 k4:0 m0:0; do
  n=${g%%:*}; M=${g##*:}
  lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_onpolF_qA_$n.pth --tests m0_pass $EV --m0-ticks $M --out-dir $RQ/gate_onpolF_qA_${n}_M$M > $L/qwen_gate_onpolF_qA_${n}_M$M.log 2>&1
  echo "gate $n M=$M exit=$?"
done
lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_onpolF_qA_k4.pth --tests chat --cc-ticks 0,4 --out-dir $RQ/chat_onpolF_qA_k4 > $L/qwen_chat_onpolF_qA_k4.log 2>&1
echo "chat exit=$?"
echo "onpolicy frozen done"
