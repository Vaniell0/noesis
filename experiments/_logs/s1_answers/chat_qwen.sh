# Chat balance after the night S1: base vs the M=0 control vs the ticks arm (EN/RU everyday questions, loops).
until grep -q "feed base done" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/bb2089nxs.output 2>/dev/null; do sleep 60; done
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=8-11
lim $UV python3 experiments/run.py --backend peft --model $Q --tests chat --cc-ticks 0 --out-dir $RQ/chat_base > $L/qwen_chat_base.log 2>&1; echo "chat base exit=$?"
lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_hint3_m0.pth --tests chat --cc-ticks 0 --out-dir $RQ/chat_s1_hint3_m0 > $L/qwen_chat_m0.log 2>&1; echo "chat m0 exit=$?"
lim $UV python3 experiments/run.py --backend peft --model $Q --weights $RQ/s1_hint3_k4.pth --tests chat --cc-ticks 0,4 --out-dir $RQ/chat_s1_hint3_k4 > $L/qwen_chat_k4.log 2>&1; echo "chat k4 exit=$?"
echo "chat qwen done"
