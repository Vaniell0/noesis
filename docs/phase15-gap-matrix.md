# Phase 1.5 gap matrix — Adam/Muon × markers/tokens, live tracker

**This is a live status table, not a history log — edit rows in place as
gaps close. Historical narrative (how a result was found, debug sagas)
stays in `docs/rl-track.md`'s Appendix; this file only tracks what's
tested vs. not, and the current answer where one exists.**

## Training-stability matrix (real runs, Brenna A30, 2026-09-10)

| Optimizer | LR | offload/forge | `--dynamic-phase-stop` | M-mixing+rewind | Status |
|---|---|---|---|---|---|
| Muon | 0.02 | n/a | no | no | ❌ unstable (`cos_sim`→0.6, `state_loss` clamped, `grad_norm`→31765) |
| Muon | 0.002 | n/a | no | no | ✅ stable |
| Adam | 1e-5 | none | no | yes | ❌ VRAM OOM every step, zero real updates |
| Adam | 1e-5 | offload | no | yes | ❌ host-RAM crisis, swap-thrashed |
| Adam | 1e-5 | offload+forge | yes | yes | ❌ still doesn't fit (13GB RSS need vs. 8GB box — hardware limit, not a bug) |
| Muon | 0.002 | n/a | yes | yes | ✅ **50/50 steps clean**, `m_counts={0:31,1:37,2:32}` (M=0/relax genuinely exercised) |

Full per-run numbers (`cos_sim`/`state_loss`/`grad_norm` trajectories):
`docs/rl-track.md`, Known-risks item 11.

## Eval results (real generation + exact-match, `experiments/rl/eval_thinkchain.py`, n=10 subset)

**Loading note, found 2026-09-10: the RAW base model file
(`rwkv7-g1i-2.9b-20260805-ctx16384.pth`) consistently swap-thrashed this
box on load (3 attempts, all killed); the MERGED step500 file
(`...step500-merged.pth`) loaded reliably every time. Cause not fully
isolated (cold page-cache on an untouched file vs. something about the
raw file itself) — until it is, use the merged file as the loading
recipe, don't retry the raw file blind.**

| Checkpoint | Marker | n | Accuracy | matrix | arithmetic | other |
|---|---|---|---|---|---|---|
| step500-merged (no ThinkChain phase run) | none | 10 | 50% | 1/5 | 4/4 | 0/1 |
| step500 (Phase 1) | M=1, warm-started marker | 10 | 50% | 1/5 | 4/4 | 0/1 |

**9/10 predictions were byte-identical between the two rows above** — the
M=1 think phase barely changed what the model generates on this subset.
Only 1 example flipped (a matrix wordsearch item, wrong→right). n=10 is
too small to call this a real "the phase doesn't help" finding — flagged
as worth a larger-n rerun, not yet acted on.

**phase15_muon (Phase 1.5, M=2), `--resume ckpt_step000050`: 0% accuracy (0/10) — MODE COLLAPSE, not just "wrong answers".**
Every matrix-category prediction is literally the string `"row"`; every
arithmetic-category prediction is literally `"5"` — the same token
regardless of input. step500 (M=1, same eval script, same day) answered
sensibly (50%, real per-example predictions) — the eval script isn't the
bug, the trained weights are.

**This is the single most important finding of the session: training
metrics looked completely healthy (cos_sim 0.94-0.98, state_loss falling
with no clamp hits, zero OOM, `--dynamic-phase-stop` working as
designed) while real generation quality collapsed to degenerate output.
Confirms the ICML-paper caveat wasn't hedging — full-FT Muon on this
Adam-pretrained checkpoint broke real task performance despite passing
every stability check today.**

**Root-caused same session: NOT specific to M=2/`--rewind-at-m2`/mixing.**
Control run `muon_control_m1` — identical config MINUS `--m-weights`/
`--rewind-at-m2` (plain M=1, Muon lr=0.002, dynamic-phase-stop, 50
steps, same warm-started marker) — collapses the SAME way: 10%
accuracy (1/10), every matrix prediction is the single fixed string
`"VEIL"`, every arithmetic prediction is the single fixed `"1"`.
Different collapsed tokens than phase15_muon's `"row"`/`"5"`, but the
same degenerate-single-token-per-category shape. **Conclusion: this is
plain full-FT Muon breaking the model in ~50 steps, independent of M,
mixing, or rewind — those mechanisms are not implicated.** Training
diagnostics (cos_sim/state_loss/grad_norm) did not distinguish this run
from a healthy one either.

**Practical implication: full-FT Muon on this checkpoint should be
treated as broken, not merely "unconfirmed quality," until the LoRA fix
is tried.** Muon+LoRA (gaps table below) is no longer just the ICML
paper's suggested comparison — it's now the only untried path to a
usable Muon-trained checkpoint at all.

**Muon+LoRA at lr=0.02 (the untested default) ALSO collapses — 0/10,
every prediction the single token `"2"`.** Training log shows why:
`state_loss` pinned at the clamp (100.0000) from step 34 onward,
`norm_penalty` growing to 13M+ by step 50, `cos_sim` bouncing
0.60-0.99 — this run was never actually stable, LoRA alone didn't fix
it. **Confound not yet resolved: is this "LoRA doesn't fix the
mismatch" (against the paper) or "0.02 was never recalibrated for
LoRA, same mistake as the full-FT sweep's first attempt"?** Next: rerun
LoRA+Muon at lr=0.002 (the value that WAS confirmed stable for full-FT)
before concluding anything about LoRA itself.

**Checkpoint save/load round-trip verified correct (2026-09-10) — ruling
out an eval-loading bug as the explanation.** Every collapsed eval above
used `--resume`; the one working eval (step500) used
`--warm-start-marker` — worth checking before trusting the collapse
findings. Direct test: load step500-merged + LoRA r=32, warm-start the
marker, record logits on a fixed prompt, `save_checkpoint` →
`load_checkpoint` round-trip, record logits again — **byte-identical
(`MATCH: True`, exact float match on the logit sum)**. The four
collapses are real properties of the trained weights, not a loading
artifact.

**"Clean lineage" pilot — DONE 2026-09-10, 50 steps, result is DIFFERENT
from the four collapses above, not yet conclusive.** Phase 1 from
scratch (raw base model, fresh ThinkChain marker — no warm-start, no
Adam anywhere in this checkpoint's history), LoRA r=32 + Muon lr=0.002,
50 steps, `training_stability` metrics healthy throughout (cos_sim
0.88-0.93, no clamp hits, `norm_penalty=0`). Eval: still 0% (0/10), but
**the failure shape differs from the Adam-continuation collapses**:
matrix predictions are a repetition LOOP (`"row=7 col=7 col=7 col=7
..."`), not one fixed word; arithmetic predictions are mostly `"58"`
with one `"5"` — degenerate but not a single hard-locked token across
all inputs. Repetition loops are a standard undertrained-LM symptom,
distinct from the clean single-token-regardless-of-input collapse seen
in every Adam-continuation run. **Reading, not yet confirmed:** this
looked more consistent with "50 steps from a cold, format-naive base
isn't enough exposure to the task format yet" than with the same
mismatch-driven collapse — but 0% is still 0%, and this is one data
point at one step count.

**200-step rerun (in progress, same config): the SAME instability
signature reappears, just delayed to ~step 90+.** `state_loss` pinned
at the clamp (100.0000) continuously from step 92 through at least step
100 (halfway), `norm_penalty` in the hundreds-of-thousands-to-millions
range and not recovering, `cos_sim` saturated 0.98-0.99. Steps 1-89 were
clean (matching the 50-step pilot's healthy read). **This weakens the
"it's specifically the Adam→Muon optimizer switch" hypothesis** — the
clean-lineage checkpoint has no Adam anywhere in its history and still
destabilizes under Muon, just later. Strengthens an alternative
reading: this may be a more general Muon + this project's state_loss/
clamp mechanism instability that emerges past some step-count horizon
regardless of lineage, not a mismatch-specific effect.

**Run auto-stopped at step 100 by noesis's own Kalman trend monitor** —
`[distill] [kalman] CRITICAL TREND on [state_loss] — stopping` fired
after the statistically-real upward slope (not noise) was detected,
same checkpoint-then-exit path as a manual SIGTERM. A real, working
safeguard catching exactly the degradation flagged above, independently
— worth noting as a positive finding, not just the collapse itself.

**Eval on the auto-saved step-100 checkpoint: 10% (1/10, the trivial
"other" example).** Matrix predictions mostly `"RAD"` (4/5) with one
`"KI"` — degenerate but not fully uniform; arithmetic is fully
degenerate (`"1"` on all 4). Even this early into the instability
(caught at the first Kalman-flagged step), the collapse signature is
already present. **Overall read across both clean-lineage runs: the
instability is real, reproducible, and independent of Adam ever being
in this checkpoint's history — Muon's instability on this architecture
looks like a property of the mechanism/LR/state_loss-clamp interaction
itself, not specifically an optimizer-switch mismatch.** This does not
match the ICML paper's LoRA-fixes-it story as cleanly as hoped; LoRA
alone (with either LR tried) has not produced a stable, non-collapsing
checkpoint on this architecture so far.

**`--l-state-weight 0.01` tested on LoRA+Muon (2026-09-10) — does NOT
fix the collapse.** Rationale for the test: step500 (the checkpoint
known to answer well) was itself originally trained with
`l_state_weight=0.01`, not 0 — today's earlier runs had all defaulted
to 0, so this wasn't a like-for-like comparison until now. Run:
warm-started from step500, LoRA r=32/alpha=64, Muon lr=0.002,
`--l-state-weight 0.01`, `--dynamic-phase-stop`, 50 steps, training
diagnostics healthy throughout (`cos_sim` 0.96-0.99, no clamp hits by
step 50, `grad_norm` noisy but bounded). Eval: **10% (1/10)** — matrix
0/5, arithmetic 0/4, same degenerate shape as the earlier lr=0.02
LoRA collapse (single-token-per-category pattern). **Closes the open
gap below on `--l-state-weight` as the missing factor: it isn't.**
LoRA+Muon collapses with l_state_weight either at 0 or at 0.01 — the
state-loss weight is not what distinguishes step500's success from
these runs.

## Old (Lightning/RWKV-PEFT) pipeline, G1i, real state_reg + Muon (2026-09-10)

**Context: reproducing Muon's effect through a genuinely different codebase
(not `train_think_distill.py`), with the REAL state_reg mechanism
(`training/light_rwkv_state_reg_patch.py`, previously written but never
actually driven — `training/scripts/run_state_reg_muon.py` is the first
driver for it) instead of a stripped diagnostic. Also found and fixed along
the way: `ctx_len=512/chunk_ctx=256` (T=2) silently starves L_state — this
project's own `docs/verdicts/2026-08-08-step8-step9.md` already documented
the T<3 short-circuit that broke step9b's real L_state; the fix
(`ctx_len=2048/chunk_ctx=512`, T=4) matches that doc's own recommendation.**

**LoRA r=32 + Muon (lr=0.02) + state_reg, on step9b's own DSL/tool_call
corpus (`step9b_combined_flat.jsonl`, decoded back from
`step9b_combined_train.pt`) — mechanically completed, but hit a RAM crisis
on save (`--merge 1` default remerges LoRA into base a second time,
CPU-side; killed in time, VM survived).** Not yet re-run with `--merge 0`.

**Full-FT (no LoRA) + Muon (lr=0.02) + state_reg, same corpus, ctx_len=2048
(T=4), 10 steps — completed cleanly, RAM peak 2.6GB.** This is the first
direct, immediate confirmation of Muon's core VRAM/RAM promise on this
exact box: full-FT under Adam needed ~13GB RSS (doesn't fit 8GB); full-FT
under Muon fits comfortably with room to spare, no CPU-offload tricks.
`state_loss` pinned at the reward clamp (-10.0, `state_reg.py:278`) from
step 0 through step 9 — motion-reward saturates instantly under full-rank
Muon updates, unlike LoRA's non-clamped values in earlier runs. `ce` noisy
(2.36→8.07→6.38 across 10 steps), not exploding, too few steps to call
collapse or success either way. Two real bugs found and fixed getting here:
`peft_loading.py`'s `--peft none` path never wrapped the bare model in the
Lightning `RWKV` class (crashed `trainer.fit`); `MuonWithAuxAdam` needed a
`my_lr_scale` key or the trainer's per-batch LR callback KeyErrors.

**Not yet done: the attractor-relevant question itself** (does N=3-style
multi-pass collapse behavior differ under Muon vs Adam) — everything above
is single-pass training-loop stability, not the multi-pass generation eval
the HF card's N=1/2/3 numbers (27.1/33.3/6.3%) are about. Confirmed via the
HF card + GitHub #349/#338 thread that the N=3 collapse is corpus-bound
(DSL `<tool_call>` attractor), not architecture-bound (icophy's production
data ruled out WKV-saturation) — so this training corpus (step9b's own) is
the right one to test it on, but no multi-pass eval has been run yet on
either the LoRA or full-FT Muon checkpoint from today.

## Toy-scale Muon vs. Adam, controlled (`experiments/A0_state_probe/muon_vs_adam_toy.py`, H25)

**10-seed rerun (2026-09-10) of the toy multiplication task, extending
the earlier 5-seed a_gate=0 ablation result. Clean signal, independent
of every 2.9B collapse above — no ThinkChain, no LoRA, no warm-start
from an Adam checkpoint, no state_loss.** `n_train_steps=4000`, Muon
lr=0.02 (its own published default, untuned), Adam lr=3e-3 (existing
`train_task` default), seeds 0-9, full-FT on both (there is no LoRA at
this scale).

| Metric | Adam mean±std (range) | Muon mean±std (range) |
|---|---|---|
| in-distribution R² | 0.9996±0.0004 [0.9987, 0.9999] | 0.9987±0.0005 [0.9977, 0.9994] |
| held-out (OOD) R² | 0.7580±0.0678 [0.6298, 0.8675] | **0.7763±0.0111** [0.7530, 0.7960] |
| `a_gate=0` ablation R² | -2.6186±2.2229 [-7.4574, -0.2095] | **-0.2952±0.5631** [-1.9434, -0.0169] |

Read: **id_r2 is a wash** (both ≈1.0, Adam nominally higher by a hair).
**ood_r2 has Muon slightly ahead on the mean with 6× lower seed-to-seed
variance** — Muon reaches a consistently generalizing solution, Adam's
generalization quality varies a lot by seed (worst seed 0.63, best
0.87). **The a_gate=0 ablation is the clearest signal**: forcing the
delta-rule erase/rewrite gate to 0 catastrophically breaks Adam's
solution in most seeds (mean -2.62, one seed as low as -7.46 — worse
than predicting the mean) but only mildly degrades Muon's (mean -0.30,
worst seed -1.94, several seeds barely below 0). **Reading: Adam's
solution leans harder on that one ablatable mechanism; Muon's solution
is more distributed/robust at equivalent task accuracy** — consistent
with the low-rank-implicit-bias argument (Adam concentrates capability
into fewer directions, Muon's orthogonalized updates spread it out),
not just "Muon is fine on this toy." **This is a real point in Muon's
favor, not merely "not worse"** — it does not, however, explain or
resolve why Muon collapses at 2.9B scale on this project's real
architecture/pipeline; that gap (row above) is unrelated in mechanism
until shown otherwise.

## Repeat-count isolation (`experiments/rl/state_trajectory_probe.py`, fixed M=1, step500 marker)

Sweep `--phase-repeat-ticks` (2/8/16/25), reading `delta_cos_prev` (cosine
similarity between consecutive per-tick state deltas — the direct,
per-tick measurement of whether repeating the SAME marker vector is
converging to one fixed direction, independent of how many DISTINCT
phases/M exist — see H25.md's power-iteration/affine-fixed-point theory,
which until now had no controlled per-tick measurement, only
reinterpretation of existing numbers).

`matrix_addition` prompt, layer 12, `delta_cos_prev` per tick:
- tick 1→2: **0.22** (real directional change)
- tick 3: 0.94, tick 4: 0.97 — **already near-saturated by tick 4**
- ticks 5-16: plateau, 0.97-0.99
- ticks 17-25 (only visible at T=25): mild DECLINE, 0.94-0.99 — not the
  clean monotone convergence the power-iteration theory predicts; some
  real noise/wobble in the tail

**Reading:** the same-vector-repeated-N-times collapse is real and fast
— by ~4 repeats the direction is already ~97% converged, matching
training's own independently-found `--dynamic-phase-stop` exit point
(observed `n_phase_tok` 3-10 in real Phase 1.5 training runs today) —
two different diagnostics converging on the same number is a real
cross-check, not a coincidence to wave off. The tail's non-monotone
wobble at T=25 is the one piece that doesn't match the simple theory
cleanly — worth another look if this axis gets revisited, not resolved
here.

**M-count isolation — DONE 2026-09-10.** `phase15_muon` (M=2, real
trained full-FT checkpoint, `--resume ckpt_step000050`) at the same
`--phase-repeat-ticks 8`, `matrix_addition` prompt, layer 12:

- Phase 0 (ticks 0-7): `delta_cos_prev` 0.767 → 0.994 — same fast
  within-phase collapse shape as the repeat-count sweep, just starting
  from a higher initial value.
- **Phase boundary (phase 0 end → phase 1 start): `delta_cos_prev` drops
  to 0.097** — near-orthogonal to the phase-0 direction.
- Phase 1 (ticks 0-7): 0.844 → 0.997 — re-converges via the same shape,
  from its own fresh starting direction.
- `chain_backtrack: {"backtracked": false}` — no oscillation/flip-flop.

**Conclusion: the two axes are now measured as genuinely independent.**
Within one phase, repeating the same marker vector collapses direction
fast (repeat-count sweep above). Switching to a NEW, distinct phase
resets that direction near-orthogonally (this sweep) — each phase is
doing real, different directional work, not continuing an
already-collapsed trajectory. This directly supports the ThinkChain
M-phase design (distinct learned markers per phase) over a naive
same-marker-repeated-M-times alternative, with an actual measurement
behind it instead of just the mechanism's own stated rationale.

## Open gaps — what's NOT yet done

| Gap | Needs | Status |
|---|---|---|
| Root-cause the phase15_muon mode collapse | Control run (plain M=1, no mixing/rewind) | **DONE 2026-09-10 — not M/mixing/rewind-specific, plain full-FT Muon collapses too (see above)** |
| Larger-n eval to check the 9/10-identical (step500 vs no-marker) finding isn't a small-sample artifact | Rerun at n=100+ | Not started |
| Raw base model (true "no training at all" point) | Root-cause the raw-file load failure, or accept the merged-file-as-M=0 proxy above | Not started |
| Adam-Phase1.5 eval point | A real trained Adam-Phase1.5 checkpoint | **Does not exist** — full-FT doesn't fit this box (see matrix above); only path forward is Adam+LoRA, not attempted |
| Muon+LoRA at both tried LRs (0.02, 0.002) | — | **DONE, both collapse** — 0.02 collapses immediately; 0.002 (continuation AND clean-lineage) collapses later (~step 90+), auto-caught by the Kalman monitor. LoRA has not yet produced a stable Muon checkpoint on this architecture at any tried config |
| **Root cause of the Muon instability itself** — now the top-priority gap: LR still too high even at 0.002? A bug in `MuonHybrid` (`loader.py`)? An interaction specifically with the clamped state_loss mechanism? | Needs isolating which factor actually drives it | **`l_state_weight` ruled out 2026-09-10** — LoRA+Muon collapses identically at `l_state_weight=0` and `0.01` (step500's own original value). Remaining candidates: the state_loss *clamp* mechanism itself (not the weight), `MuonHybrid`'s Newton-Schulz application to this architecture's specific matrix shapes, or LR still too high even at 0.002 over longer horizons (the 200-step clean-lineage run destabilized at step ~90+ despite looking clean through step 89) |
| Quality-at-parity comparison (Muon vs Adam, matched stability) | Blocked — no stable Muon checkpoint exists yet to compare | Not started |
| Phase 1 (step500 itself) re-verification | Was never re-checked this session — only used as a fixed starting point | Not started |
| Step 2 (discrete self-feed, no markers) | A real, non-exploitable training signal — reward-shaping has twice failed here (shortcut-collapse, echo-exploit) | Design not started, deliberately deferred |

## CORRECTION (2026-09-10, late) — the whole day's ICML framing may be backwards

**`hypotheses/H25.md` (written 2026-09-03, a week before this session) already
states: "Real G1i checkpoint was pretrained on Muon, not Adam."** Every
collapse finding above was interpreted through the ICML paper's frame
("Can Muon Fine-tune Adam-Pretrained Models?" — Adam-pretrain resists
Muon-finetune) without checking this already-recorded fact first. It's
wrong for G1i: the real lineage is **Muon (pretrain) → Adam (Phase 1 /
step500's own fine-tune, `train_think_distill.py` default optimizer) →
Muon (today's fine-tuning attempts)** — a double optimizer switch, not
the paper's single Adam→Muon case.

**Reframing that follows, consistent with today's own toy-scale finding**
(Adam's solution on the toy task depends heavily on one ablatable channel;
Muon's solution is distributed across many — `hypotheses/H25.md`'s
Track B evidence log, same mechanism): if step500's Adam fine-tuning
concentrated/narrowed the model's structure the way it did on the toy
task, THAT narrowing — not Muon's original pretrain geometry — may be
what today's Muon fine-tuning attempts actually collide with. Muon
fine-tuning directly on the RAW G1i base (skipping the Adam-narrowed
step500 entirely) is the one test that actually isolates this, and it
has never been evaluated for real generation quality:

- The `--optimizer none`/full-FT `train_think_distill.py` full-FT-Muon
  runs, and every LoRA+Muon run, ALL started from `step500` (Adam-touched)
  — none of them test "clean Muon on a base that's never seen Adam."
- Today's "clean lineage" experiments (Phase 1 from scratch via
  ThinkChain/`train_think_distill.py`, no Adam anywhere) DID use the raw
  base, but at LoRA scale and eventually destabilized past step ~90 —
  consistent with either reading (could still be Muon-general instability,
  or could be that the ThinkChain marker itself needs more steps than
  tested before Muon's updates stabilize on a truly clean base).
- The LAST run of the session (old Lightning pipeline, raw
  `rwkv7-g1i-2.9b-20260805-ctx16384.pth`, full-FT, real `state_reg`, Muon,
  step9b's own DSL corpus, 50 steps, training-loop stable, RAM stable
  2.7-2.8GB) is the cleanest version of this test to date — **but no
  checkpoint was saved (`--epoch_save 999`, set to dodge the RAM-crisis-
  on-save risk seen earlier), so there is nothing to eval yet.**

**SVD delta analysis result (2026-09-10, ran locally, no VM):** `step500 -
raw_base` (`lora_rank_analysis.py`, 482 matrices). `att_proj` (n=128):
mean top-32 energy 0.904, mean effective_rank 899.6/2560 — **consistent
with step500's own known construction** (Phase 1 was itself trained as
LoRA r=32 on attention projections), so this is confirmatory, not new
evidence about Adam's bias — of course a rank-32-LoRA-built delta is
~90% captured by its top 32 singular values. **`ffn` (n=64) is the
actually unexpected part:** mean top-32 energy only 0.827, mean
effective_rank 1642.8/2560, and 190 of 482 matrices show
effective_rank > 32 — several `ffn.value.weight` matrices at
near-FULL rank (2552-2560/2560). If Phase 1's LoRA never targeted FFN
matrices at all (attention-only `target_modules`, matching step9's own
`[receptance, key, value, output]` convention), FFN deltas should be
exactly zero, not near-full-rank with real Frobenius mass
(0.09-0.15/layer). **Resolved same session: this is bf16 round-trip noise, not a training
effect.** `target_modules=["receptance","key","value","output"]`
(`experiments/rl/train_think_distill.py`/`loader.py`, matching step9's
own convention) confirms Phase 1's LoRA never touched FFN at all —
direct check of `blocks.0.ffn.value.weight` confirms it: not
byte-identical between base and step500, but max abs diff ≈ 1.5e-4,
consistent with bf16 merge/save/load rounding, not a real weight
change. Random noise has full rank by construction (spread evenly
across all singular directions), which is exactly the "near-2560
effective rank" pattern seen — the FFN result is an artifact, not
evidence. Net: this SVD analysis doesn't actually test the reframed
hypothesis (it just confirms step500's own known LoRA construction on
attention weights, and shows FFN is untouched modulo rounding noise).
**Still need a real Adam-full-FT checkpoint to test whether Adam's
OWN bias (not LoRA's structural constraint) concentrates capability —
none exists yet, full-FT Adam has never survived to produce one on
this hardware.**

**Cheap, VM-free next step (queued, not yet run):** `experiments/A0_state_probe/lora_rank_analysis.py` already has a no-delta SVD mode (used for "Base G1i weight-SVD" — near-full-rank, `att_proj≈2391/2560`) and a delta mode (`base` vs `trained`). Running it with `--base` = raw G1i (`rwkv7-g1i-2.9b-20260805-ctx16384.pth`) and `--trained` = `step500-merged.pth` directly tests the reframed hypothesis above: is step500's own Adam delta low-rank/concentrated (matching the toy result's Adam behavior), unlike G1i's own near-full-rank base weights? Pure local SVD, no GPU/VM needed, reuses existing code.

**Next session, top priority: rerun that exact config with a real, RAM-safe
checkpoint save (LoRA-only save, or an explicit fp16-on-GPU-then-CPU-copy
save path — full-FT's plain `torch.save(state_dict)` on this 8GB box is
the same class of risk `merge_lora.py` had before its GPU-side fix earlier
today, never applied to `trainer.py`'s own save path), then run
`eval_thinkchain.py`-style real generation eval on it.** If THAT
checkpoint doesn't collapse, the whole day's "Muon breaks this model"
conclusion needs to be narrowed to "Muon fine-tuning breaks an
already-Adam-narrowed G1i" — a materially different, more specific claim.

## Caveat — what today's data does and doesn't show

**["Can Muon Fine-tune Adam-Pretrained Models?" (ICML 2026, arXiv
2605.10468, Qu/Huang/Horváth)](https://arxiv.org/abs/2605.10468)**:
full-FT Muon on an Adam-pretrained model can underperform Adam on real
task quality even when stable, via an implicit-bias mismatch; LoRA (not
a lower LR) is the paper's validated fix.

**Status after the full day's data, updated — weaker match to the paper
than the first read suggested.** The OUTCOME shape (healthy training
metrics, broken real generation) did reproduce, on the exact scenario
the paper describes (full-FT, Adam-pretrained base) — that part holds.
But the paper's specific MECHANISM claim doesn't hold up cleanly here:
(1) its predicted fix, LoRA, did not produce a stable checkpoint at
either LR tried; (2) the "clean lineage" checkpoint (no Adam anywhere
in its history at all) destabilized under Muon too, just later (~step
90+) — the paper's story is specifically about an Adam→Muon *switch*,
which this checkpoint never had. **Current best-supported reading: the
instability looks more like a property of Muon + this project's own
training mechanism (LR calibration, the clamped state_loss signal, or
`MuonHybrid`'s own implementation) than the specific Adam/Muon
implicit-bias mismatch the paper describes.** Worth remembering as a
real independent data point either way — either the mismatch mechanism
doesn't transfer cleanly to RWKV-7's structure, or there's a different,
still-undiagnosed instability being conflated with it.
