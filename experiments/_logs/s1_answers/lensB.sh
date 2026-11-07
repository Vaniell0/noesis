until grep -q "geom ce_T1_k4 exit=" /tmp/claude-1000/-home-vaniello/cc005570-4ce8-425a-9004-cdbb4246d71c/tasks/b1080tsww.output 2>/dev/null; do sleep 60; done
CORES=8-11; source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
lim $UV python3 experiments/rl/state_readout_lens.py --model $M --out experiments/rl/results/state_readout_lens_g1d04b_L16.json --layers 4,8,11,12,13,16,20,23 --abl-layers 16 --cont-len 16 > $L/lens_base_L16.log 2>&1
echo "lens base L16 exit=$?"
