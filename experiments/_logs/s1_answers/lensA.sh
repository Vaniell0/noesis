until grep -q "chat kl_T1 exit=" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/b2unlnn8s.output 2>/dev/null; do sleep 60; done
CORES=4-7; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
lim $UV python3 experiments/rl/state_readout_lens.py --model $M --weights $R/kl_T2_k4.pth --out experiments/rl/results/state_readout_lens_kl_T2_k4.json --layers 4,8,11,12,13,16,20,23 --abl-layers 8,12,16,20 --cont-len 16 > $L/lens_kl_T2_k4.log 2>&1
echo "lens kl_T2_k4 exit=$?"
