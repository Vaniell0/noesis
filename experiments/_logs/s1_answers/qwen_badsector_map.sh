# Owner 2026-11-05: find what was written badly, to repair it pointwise. Per-token bits and rank of the real token on text
# the model read in pretraining (Wikipedia Level-3 vital articles, intro sections, 959 of 999 fetched 2026-11-05; current
# text, so a few facts may postdate the model). Expensive tokens ranked 2-5 = nearly read = bad-sector candidates.
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
Q=/home/vaniello/.libs/models/qwen/Qwen3.5-0.8B; RQ=experiments/rl/results/qwen35_08b; CORES=4-7
until [ -f $RQ/compress_s1_onpolF_qA_k4/compress.json ]; do sleep 60; done
lim $UV python3 experiments/run.py --backend peft --model $Q --tests compress --cp-jsonl ~/.libs/datasets/wiki_vital/level3_intro.jsonl --cp-docs 1000 --cp-tokens 512 --cp-window 512 --cp-ticks 0 --cp-save-tokens --out-dir $RQ/badsector_map_vital3 > $L/qwen_badsector_map.log 2>&1
echo "map exit=$?"
