# self-KL s40 on the class "stored, not read" (question eval, base greedy 18/64, gold first among capitals for 25 of 49
# misses): free answers at M=4 and M=0, think frame, soft feed.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=soft
for M in 4 0; do
  lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_self_think_k4_s40.pth --tests m0_pass --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-per-cell 0 --m0-ticks $M --out-dir $RQ/gate_self_s40_question_M$M > $L/qwen_gate_self_s40_question_M$M.log 2>&1
  echo "gate M=$M exit=$?"
done
lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-per-cell 0 --m0-ticks 4 --out-dir $RQ/gate_base_question_M4 > $L/qwen_gate_base_question_M4.log 2>&1
echo "gate base M=4 exit=$?"
