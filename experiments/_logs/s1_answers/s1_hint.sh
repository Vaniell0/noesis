until grep -q "reach train hint=3 exit=" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/b6pjufhct.output 2>/dev/null; do sleep 60; done
CORES=4-7; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
P=(--items $C/ciphers_k_context_plain_train.jsonl --items $C/ciphers_k_mixed_plain_train.jsonl)
SEL=(--right-from experiments/rl/results/s0_g1d04b_train_hint3/m0_pass.json --exclude-right-from experiments/rl/results/s0_g1d04b_train_hint0/m0_pass.json)
# teacher: base 0.4B at M=0 reading hint 3
lim $UV python3 experiments/rl/train_latent_s1.py --model $M $P $SEL --teacher-hint 3 --build-cache --cache-out $R/hint3_teacher_cache.pt > $L/hint_cache.log 2>&1
echo "cache exit=$?"
# student: kl_T2_k4 through 4 ticks, plain prompt
lim $UV python3 experiments/rl/train_latent_s1.py --model $M --weights $R/kl_T2_k4.pth $P $SEL --target cache --cache $R/hint3_teacher_cache.pt   --loss kl --temp 2 --phase-len 4 --steps 200 --accum 4 --eval-every 50 --threads 4 --out $R/hint3_kl_T2_k4.json > $L/train_hint3.log 2>&1
echo "train hint3 exit=$?"
