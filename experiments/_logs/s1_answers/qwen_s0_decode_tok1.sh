# Qwen3.5-0.8B S0 at M=0 on the decode rung (gen_ciphers --task decode, same single-token alphabets, 20-50 chars):
# own CoT on context/mixed never reaches per-letter decoding (cvr_tok1_24 0/24) - can it do the reverse table lookup at all?
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
lim $UV python3 experiments/run.py --backend peft --model $Q --tests m0_pass --m0-items $C/ciphers_decode_tok1_short_eval.jsonl --m0-per-cell 0 --out-dir $RQ/s0_decode_tok1 > $L/qwen_s0_decode_tok1.log 2>&1
echo "s0 decode tok1 exit=$?"
