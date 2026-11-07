# Owner 2026-11-04: repeat the one run whose output grew (0.4B kl_T2_k4: KL T=2 to the model's OWN M=0 distribution on items it
# solves, no foreign teacher) on Qwen, think frame, soft feed. Read the output, not accuracy: chat greedy + samples at T 0.7/1.0.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b
big() { systemd-run --user --scope -q -p MemoryMax=14G -p CPUQuota=400% taskset -c 8-11 "$@"; }
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=soft
big $UV python3 experiments/run.py --backend peft --model $Q --tests chat --cc-ticks 0,4 --cc-temps 0.7,1.0 --cc-samples 3 --out-dir $RQ/chat_base_think_T > $L/qwen_chat_base_think_T.log 2>&1; echo "chat base exit=$?"
big $UV python3 experiments/rl/train_latent_s1.py --model $Q --items $C/ciphers_k_context_plain_train.jsonl --items $C/ciphers_k_mixed_plain_train.jsonl --right-from $RQ/s0_train_hint0/m0_pass.json --target self --loss kl --temp 2 --ticks 4 --steps 80 --accum 2 --eval-every 40 --eval-max 40 --threads 4 --out $RQ/s1_self_think_k4.json > $L/qwen_s1_self_think_k4.log 2>&1
echo "train self k4 exit=$?"
for W in s1_self_think_k4_s40 s1_self_think_k4; do
  big $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/$W.pth --tests chat --cc-ticks 0,4 --cc-temps 0.7,1.0 --cc-samples 3 --out-dir $RQ/chat_$W > $L/qwen_chat_$W.log 2>&1
  echo "chat $W exit=$?"
done
echo "self kl done"
