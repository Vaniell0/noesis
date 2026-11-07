# Owner 2026-11-07: how diverse are the latent ticks? latent_geometry (cross-prompt eff_rank, pc1_share, tick_cos, nn_cos)
# on Qwen in the think frame, soft feed, 8 ticks: base vs the frozen-teacher arms. One job at a time.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=4-7
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=${FEED:-soft}
IT=(--lg-items $C/ciphers_k_question_plain_evalB.jsonl --lg-items $C/ciphers_k_mixed_plain_evalB.jsonl --lg-items $C/ciphers_k_context_plain_eval.jsonl --lg-per-file 12 --lg-ticks 8)
for W in base s1_onpolF_qA_k4 s1_onpolF_mA_k4_s30; do
  WA=(); [ "$W" = base ] || WA=(--weights $RQ/$W.pth)
  lim $UV python3 experiments/run.py --backend peft --model $Q $WA --tests latent_geometry $IT --out-dir $RQ/geom_${W}_$NOESIS_HF_FEED > $L/qwen_geom_${W}.log 2>&1
  echo "geom $W exit=$?"
done
