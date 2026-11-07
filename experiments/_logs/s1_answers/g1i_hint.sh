until grep -q "qwen s0 done" /home/vaniello/.claude/jobs/cc005570/tmp/qwen_s0.out 2>/dev/null; do sleep 120; done
CORES=4-11; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
export OMP_NUM_THREADS=4 NOESIS_PROBE_THREADS=4
G=models/rwkv7-g1i-2.9b-20260805-ctx16384.pth
for h in 0 3; do
  systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=400% taskset -c 4-7 $UV python3 experiments/run.py --backend peft --model $G --tests m0_pass --out-dir experiments/rl/results/s0_g1i29_plain_hint$h     --m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 0 --m0-hint $h > $L/g1i_hint$h.log 2>&1
  echo "g1i hint=$h exit=$?"
done
