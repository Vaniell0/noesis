cd ~/Desktop/projects/noesis
O=/tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/b722vg4as.output
until grep -q "answers kl_T2 M=1 exit=" $O 2>/dev/null; do sleep 60; done
L=experiments/_logs/s1_answers; C=training/corpus_open; R=experiments/rl/results
M=/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
export OMP_NUM_THREADS=4 NOESIS_PROBE_THREADS=4 PYTHONUNBUFFERED=1
UV=(uv run --offline --no-project --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with 'torch==2.11.0+cpu' --with rwkv --with numpy --with einops --with rwkv-fla --with triton --with 'peft>=0.17.1' --with huggingface_hub --with pyyaml)
lim() { systemd-run --user --scope -q -p MemoryMax=10G -p CPUQuota=400% taskset -c 8-11 "$@"; }
P=($C/ciphers_k_context_plain_eval.jsonl $C/ciphers_k_question_plain_eval.jsonl $C/ciphers_k_mixed_plain_eval.jsonl)
for cfg in kl:2 ce:1; do
  loss=${cfg%%:*}; T=${cfg#*:}
  lim $UV python3 experiments/rl/train_latent_s1.py --model $M --items $P[1] --items $P[2] --items $P[3] \
    --right-from $R/s0_g1d04b_plain/m0_pass.json --target self --loss $loss --temp $T --phase-len 4 \
    --steps 200 --accum 4 --eval-every 50 --threads 4 \
    --out $R/s1_pilot_g1d04b/${loss}_T${T}_k4.json > $L/train_${loss}_T${T}_k4.log 2>&1
  echo "train $loss T=$T k=4 exit=$?"
done
