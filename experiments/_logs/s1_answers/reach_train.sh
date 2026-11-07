CORES=$1; H=$2; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
lim $UV python3 experiments/run.py --backend peft --model $M --tests m0_pass --out-dir experiments/rl/results/s0_g1d04b_train_hint$H \
  --m0-items $C/ciphers_k_context_plain_train.jsonl --m0-items $C/ciphers_k_mixed_plain_train.jsonl --m0-per-cell 0 --m0-hint $H > $L/reach_train_h$H.log 2>&1
echo "reach train hint=$H exit=$?"
