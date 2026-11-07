# Can Qwen3.5-0.8B decode in its own CoT? 24 ciphered context items (4 per cipher, eval file - measurement only), budget
# 1024, answer read every 64 tokens; the cache's cots record first_right per item.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
python3 -c "
import json,collections
by=collections.defaultdict(list)
for l in open('$C/ciphers_k_context_eval.jsonl'):
    r=json.loads(l); by[r['view']].append(l)
open('$RQ/cot_cipher_probe_items.jsonl','w').writelines(x for v in sorted(by) for x in by[v][:4])
"
lim $UV python3 /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/scratchpad/cot_cipher.py $RQ/cot_cipher_probe_items.jsonl $RQ/cot_cipher_probe24.json > $L/qwen_cot_cipher24.log 2>&1
echo "cot cipher exit=$?"
