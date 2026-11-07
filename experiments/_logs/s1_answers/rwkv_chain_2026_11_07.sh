# Owner 2026-11-07: back to RWKV (Qwen's tied head cannot carry a latent). A multi-day chain, one heavy job at a time.
#  1. G1L final 2.9B / 1.5B (released 2026-10-31): latent geometry on the items G1i/G1k were measured on, and the plain
#     rung at M=0. Prediction: the channel collapse seen on G1k / G1l-temp (cross-prompt cos -> 1 by tick 6-8) persists.
#  2. Short answer, long computation: plain context rows of level >= 3 ("which two people share a city").
#  3. Distil G1d 0.4B from a FROZEN G1L 2.9B teacher (same tokenizer), on-policy, on rows the teacher solves and the
#     student does not; anchor = rows the 0.4B reads; arms M=4 vs M=0. Prediction: M=4 beats M=0 of the same checkpoint
#     and the M=0 arm, and the ticks keep a higher effective rank than the base. Refuted if M=0 catches up.
#  4. Gates: free answers on the held-out level-3 rows, latent geometry, chat.
set +e
source ~/Desktop/projects/noesis/experiments/_logs/s1_answers/common.sh
cd ~/Desktop/projects/noesis
export OMP_NUM_THREADS=8 NOESIS_PROBE_THREADS=8
R=experiments/rl/results; RW=$R/rwkv_2026_11_07; mkdir -p $RW
S04=/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
L29=models/rwkv7-g1l-2.9b-20261031-ctx34304.pth
L15=models/rwkv7-g1l-1.5b-20261031-ctx34304.pth
big() { systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=800% taskset -c 4-11 "$@"; }
log() { echo "[$(date +%F' '%T)] $*"; }

for f in $L29 $L15; do                                         # downloads in the background, network only
  [ -s $f ] || (curl -sS -L -x socks5h://127.0.0.1:2080 --retry 20 --retry-delay 30 -C - -o $f.part \
     "https://huggingface.co/BlinkDL/rwkv7-g1/resolve/main/$(basename $f)" && mv $f.part $f) > $L/dl_$(basename $f).log 2>&1 &
done

# 2. data (fast, no model)
$UV python3 - <<'E'
import json, subprocess
C = 'training/corpus_open/'
for split, n in (('train', 3000), ('eval', 800)):
    out = f'{C}ciphers_k46_context_plain_{split}_all.jsonl'
    subprocess.run(['python3', 'training/scripts/gen_ciphers.py', '--task', 'context', '--families', 'plain', '--k', '4,6',
                    '--split', split, '--n', str(n), '--out', out], check=True)
    rows = [json.loads(l) for l in open(out)]
    hi = [r for r in rows if r.get('level', 0) >= 3]
    open(f'{C}ciphers_ctxL3_plain_{split}.jsonl', 'w').write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in hi[:400 if split == 'train' else 160]))
    print(split, len(rows), 'level>=3:', len(hi))
E
log "data done"
TR=$C/ciphers_ctxL3_plain_train.jsonl; EV=$C/ciphers_ctxL3_plain_eval.jsonl
GITEMS=(--lg-items experiments/rl/results/head_ablation/sub_h22.jsonl --lg-items $C/ciphers_k_context_eval.jsonl --lg-items $C/ciphers_k_mixed_eval.jsonl --lg-per-file 12 --lg-ticks 8)
PLAIN=(--m0-items $C/ciphers_k_context_plain_eval.jsonl --m0-items $C/ciphers_k_question_plain_eval.jsonl --m0-items $C/ciphers_k_mixed_plain_eval.jsonl --m0-per-cell 0)

# student side first (0.4B is local): S0 on level-3 train + eval, base geometry, anchor rows it reads
big $UV python3 experiments/run.py --backend peft --model $S04 --tests m0_pass --m0-items $TR --m0-per-cell 0 --out-dir $RW/s0_g1d04_ctxL3_train > $L/rw_s0_g1d04_tr.log 2>&1; log "s0 0.4B train $?"
big $UV python3 experiments/run.py --backend peft --model $S04 --tests m0_pass --m0-items $EV --m0-per-cell 0 --out-dir $RW/s0_g1d04_ctxL3_eval > $L/rw_s0_g1d04_ev.log 2>&1; log "s0 0.4B eval $?"
big $UV python3 experiments/run.py --backend peft --model $S04 --tests m0_pass --m0-items $EV --m0-per-cell 0 --m0-ticks 4 --out-dir $RW/s0_g1d04_ctxL3_eval_M4 > $L/rw_s0_g1d04_ev4.log 2>&1; log "s0 0.4B eval M4 $?"
big $UV python3 experiments/run.py --backend peft --model $S04 --tests m0_pass --m0-items $C/ciphers_k_context_plain_train.jsonl --m0-per-cell 0 --out-dir $RW/s0_g1d04_ctx_plain_train > $L/rw_s0_g1d04_ctxtr.log 2>&1; log "s0 0.4B ctx train $?"
$UV python3 - <<'E'
import json
C = 'training/corpus_open/'; RW = 'experiments/rl/results/rwkv_2026_11_07/'
ok = set(json.load(open(RW + 's0_g1d04_ctx_plain_train/m0_pass.json'))['right_ids'])
rows = [json.loads(l) for l in open(C + 'ciphers_k_context_plain_train.jsonl') if json.loads(l)['id'] in ok][:48]
open(C + 'anchor_read_g1d04_ctx48.jsonl', 'w').write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
print('anchor rows', len(rows))
E
big $UV python3 experiments/run.py --backend peft --model $S04 --tests latent_geometry $GITEMS --out-dir $RW/geom_g1d04_base > $L/rw_geom_g1d04.log 2>&1; log "geom 0.4B $?"

# 1. G1L final battery (waits for the download)
for M in $L29 $L15; do
  n=$(basename $M .pth)
  until [ -s $M ]; do sleep 120; done
  big $UV python3 experiments/run.py --backend peft --model $M --tests latent_geometry $GITEMS --out-dir $RW/geom_$n > $L/rw_geom_$n.log 2>&1; log "geom $n $?"
  big $UV python3 experiments/run.py --backend peft --model $M --tests m0_pass $PLAIN --out-dir $RW/s0_${n}_plain > $L/rw_s0_${n}_plain.log 2>&1; log "plain $n $?"
done

# teacher side: G1L 2.9B on the level-3 rows
big $UV python3 experiments/run.py --backend peft --model $L29 --tests m0_pass --m0-items $TR --m0-per-cell 0 --out-dir $RW/s0_g1l29_ctxL3_train > $L/rw_s0_g1l29_tr.log 2>&1; log "s0 teacher train $?"
big $UV python3 experiments/run.py --backend peft --model $L29 --tests m0_pass --m0-items $EV --m0-per-cell 0 --out-dir $RW/s0_g1l29_ctxL3_eval > $L/rw_s0_g1l29_ev.log 2>&1; log "s0 teacher eval $?"

# 3. distillation arms: frozen G1L 2.9B teacher reading the plain prompt, on-policy, student T 0.7
for arm in k4:4 m0:0; do
  n=${arm%%:*}; M=${arm##*:}
  big $UV python3 experiments/rl/train_latent_s1.py --model $S04 --items $TR --right-from $RW/s0_g1l29_ctxL3_train/m0_pass.json \
    --exclude-right-from $RW/s0_g1d04_ctxL3_train/m0_pass.json --target self --on-policy --frozen-teacher --teacher-model $L29 \
    --anchor-items $C/anchor_read_g1d04_ctx48.jsonl --student-temp 0.7 --ticks $M --steps 150 --accum 4 --eval-every 50 \
    --eval-max 20 --threads 8 --out $RW/distil_g1l29_to_g1d04_$n.json > $L/rw_distil_$n.log 2>&1; log "distil $n $?"
done

# 4. gates
for g in k4:4 k4:0 m0:0; do
  n=${g%%:*}; M=${g##*:}
  big $UV python3 experiments/run.py --backend peft --model $S04 --weights $RW/distil_g1l29_to_g1d04_$n.pth --tests m0_pass --m0-items $EV --m0-per-cell 0 --m0-ticks $M --out-dir $RW/gate_distil_${n}_M$M > $L/rw_gate_${n}_M$M.log 2>&1; log "gate $n M$M $?"
done
for n in k4 m0; do
  big $UV python3 experiments/run.py --backend peft --model $S04 --weights $RW/distil_g1l29_to_g1d04_$n.pth --tests latent_geometry,chat $GITEMS --cc-ticks 0,4 --out-dir $RW/geomchat_distil_$n > $L/rw_geomchat_$n.log 2>&1; log "geom+chat $n $?"
done
log "rwkv chain done"
