# Owner 2026-11-05 (Kolmogorov criterion): bits per byte of fresh text (UncheatableEval 2026-07, after the models'
# training) with M ticks at every 64-token boundary vs M=0, per checkpoint. Soft feed (as the arms were trained).
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=4-7
export NOESIS_HF_FEED=soft
D=~/.libs/datasets/uncheatable/2026-07/test.parquet
for W in ${ONLY:-base s1_self_think_k4_s40 s1_onpol_qA_k4}; do
  WA=(); [ "$W" = base ] || WA=(--weights $RQ/$W.pth)
  lim $UV python3 experiments/run.py --backend peft --model $Q $WA --tests compress --cp-data $D --cp-ticks 0,1,4 --out-dir $RQ/compress_$W > $L/qwen_compress_$W.log 2>&1
  echo "compress $W exit=$?"
done
