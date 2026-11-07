# Chat on the self-KL Qwen checkpoint s40 (run stopped at step 70 by memory pressure 2026-11-04): greedy + samples at T 0.7/1.0,
# M=0 and M=4, think frame, soft feed - compare with chat_base_think_T.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
export NOESIS_HF_FRAME=think NOESIS_HF_FEED=soft
lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_self_think_k4_s40.pth --tests chat --cc-ticks 0,4 --cc-temps 0.7,1.0 --cc-samples 3 --out-dir $RQ/chat_s1_self_think_k4_s40 > $L/qwen_chat_s1_self_think_k4_s40.log 2>&1
echo "chat s40 exit=$?"
