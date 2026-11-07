# Qwen3.5-0.8B S0 at M=0 on short single-token-alphabet ciphers (gen_ciphers --families script_cyrillic,script_katakana,
# script_hebrew --k 2): does one pass decode where the tokenizer does not split the symbols?
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
for sp in eval train; do
  lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --m0-items $C/ciphers_k2_tok1_context_$sp.jsonl --m0-items $C/ciphers_k2_tok1_mixed_$sp.jsonl --m0-per-cell 0 --out-dir $RQ/s0_tok1_${sp}_hint0 > $L/qwen_s0_tok1_$sp.log 2>&1
  echo "s0 tok1 $sp exit=$?"
done
