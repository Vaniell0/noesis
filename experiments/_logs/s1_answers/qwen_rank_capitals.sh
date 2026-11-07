# Owner 2026-11-04: "преобучение могло не записать всего в веса ... как битые сектора". Per fact: greedy answer, rank of the
# gold among the 32 capitals by teacher-forced log p (no generation), hits in 8 samples at T=1. Lost = low rank everywhere.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=${CORES:-4-7}
for t in ${TASKS:-question mixed}; do
  lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --m0-items $C/ciphers_k_${t}_plain_eval.jsonl --m0-per-cell 0 --m0-rank --m0-samples 8 --m0-temp 1.0 --out-dir $RQ/rank_capitals_$t > $L/qwen_rank_capitals_$t.log 2>&1
  echo "rank $t exit=$?"
done
