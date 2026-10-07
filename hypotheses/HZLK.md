<!--
id: HZLK
status: PROPOSED
prior: null
posterior: null
depends_on: [H0, H8, H10, H12, H17, H18, H25]
supports: []
contradicts_if:
  - "A loop trained over M>=2 phases on a verified same-model teacher set does not beat both M=1 and M=0 on held-out items of at least two task families (paired by item, effect >= 5 points and McNemar p < 0.05 on the pooled items), or it degrades the M=0 path by more than 2 points"
  - "At equal accuracy the trained loop still generates more than 80% of the teacher's mean CoT tokens on the same items (G1i, untrained teacher: 470 on 32 patterns, 546 on 16 arithmetic items)"
  - "The loop never exceeds the teacher: after the stop and search stages (C3) it is below the teacher's accuracy on every family at any token budget, i.e. distillation was all there was"
  - "The learned continue/stop action equals the best FIXED M, or collapses to the maximum M, on a mixed-family set; or a random stop with the same mean M does as well"
  - "M>1 only works when a per-phase correctness reward is added (then self-regulation is not what produced it)"
  - "A readout-level tuning (LoRA) that leaves routing unchanged nevertheless moves the state-noise basin by more than the seed-to-seed spread of the base model, i.e. output narrowness and the state are not separable"
  - "Gains persist when the teacher's content is shuffled between examples, or when the fed expectation is swapped between examples (the loop is then fitting format, not extracting)"
evidence:
  - date: 2026-10-06
    file: experiments/rl/results/cot_vs_routes_g1i_pattern.json
    call: supports
    strength: weak
    note: "G1i-2.9B untrained, 32 pattern items: 17/32 teacher CoTs run to the 768-token cap; 47% of all CoT tokens are generated after the first checkpoint at which the answer was already correct; an offline rule 'answer silently if 4 of 5 silent routes agree, else CoT' scores 0.719 at 256 tokens against CoT 0.688 at 470. A room estimate (the rule was chosen on the same items, 0.719 vs 0.688 is one item), not a mechanism."
  - date: 2026-10-06
    file: experiments/rl/results/cot_vs_routes_g1i_arith.json
    call: methodology
    strength: weak
    note: "Same probe, 16 arithmetic items: silent routes 0.00-0.06, CoT 0.312 at 546 tokens, the agreement rule saves nothing (0.25 at 339). The saving is a property of the task family; a rule we write does not transfer, which is why stop/continue has to be the model's own action."
  - date: 2026-10-06
    file: experiments/rl/results/cot_vs_routes_g1i_pattern.json
    call: supports
    strength: weak
    note: "The plain model does not use extra time: re-reading the question scores 0.469 against 0.531 for answering at once (32 patterns), 0.062 against 0.062 on arithmetic; 2 and 8 latent ticks of its own expectation 0.344. The earlier N=2 gain (H10, step8 LoRA checkpoint, 33.3 vs 27.1) is most likely the known recall effect of repeating context (arXiv 2407.05483), which this does not contradict. Pause-token work (arXiv 2310.02226) reports that extra inference compute helps only after training for it."
  - date: 2026-08-07
    file: experiments/A0.8_refine/results/step8_epoch0/h10_sweep_step8_epoch0/n2_k512_cot.json
    call: methodology
    strength: weak
    note: "H10 sweep on the step8 LoRA checkpoint, read in full 2026-10-07 (the K=512 cells were 'N/A, too large' in the summary): N=2 with 512 CoT tokens scores 15/48 = 31.2%, the best CoT cell and one task below N=2 silent (16/48 = 33.3%): 512 generated tokens buy what a second silent pass gave for free. N=2 CoT rises with K (10.4, 22.9, 31.2); N=3 does not recover with tokens (4.2, 6.3, 6.3). state_readout cells of that sweep are invalid (evaluator bug, fixed 2026-08-12)."
  - date: 2026-10-07
    file: experiments/rl/results/state_step_influence_g1d04b.json
    call: supports
    strength: weak
    note: "How much one step moves the WKV state, G1d-0.4B untrained, 6 pattern prompts, all from one post-question state (|dS|/|S| per step, mean over layers): question tokens 0.24, the question fed again (N=2) 0.20, teacher CoT tokens 0.21, latent ticks of the model's own expected embedding 0.21, ticks of a constant vector 0.04 (consecutive displacements aligned, cosine +0.95: a fixed point). Content-carrying steps all alternate direction (cosine -0.23). A latent tick moves the state as much as a real token; a constant does not. Magnitude is not usefulness: the useful component (projection on the answer gradient) is not measured yet."
  - date: 2026-09-22
    file: experiments/rl/results/state_traj_g1d04b_expected_on.md
    call: supports
    strength: weak
    note: "Earlier trajectory probe, same direction: the content-dependent `expected` feed behaves like real self-generated tokens (|dS|/|S| 0.19-0.33, cosine -0.05 to -0.29 at L12/16/20) while an untrained constant marker moves less (0.09-0.23) and aligns (cosine 0.43-0.59, to 0.88 by tick 8 at L16): the fixed point comes from feeding a constant, not from being latent. 4 prompts, n small; the 0.4B is a weak testbed for how task-dependent the write is."
  - date: 2026-10-02
    file: experiments/A0_state_probe/results/halting_chain.json
    call: supports
    strength: weak
    note: "Toy: stopping read from the state is learnable (supervised 0.953 exact stop; a state swap moves the stop to the donor's sum in 94.8%). From outcome reward with a time price only: lr 3e-3 collapses into stop-early-and-guess, lr 5e-4 reaches counter level (0.764 vs 0.758); labels first then reward keeps the labelled policy. One seed each."
  - date: 2026-10-01
    file: experiments/rl/results/state_readout_lens_g1i.json
    call: methodology
    strength: strong
    note: "The neck: per head 12-29 directions occupied (G1d L12), the answer depends on about half, a linear probe separates 1-2 facts (usable capacity 2: N=1 1/1, N=2 2/2, N=4 1/4, N=8 0/8); more stored pairs raise occupancy but not the number of separable facts - facts share directions. Readout lens checked against causal deletion (Spearman 0.92-0.98)."
  - date: 2026-10-07
    file: experiments/A0_eval/results/rwkv7_29b_g1h_np2048.json
    call: methodology
    strength: weak
    note: "Suggestive, not controlled: G1h-2.9B base through Ollama scores 7.1% (3/42) at the harness default of 64 generated tokens and 39.6% (19/48) at 2048 (it reasons before it answers); the L_state + eps-mask LoRA (step9 e0) scores 43.75% at 512 on the native harness. Backend, prompt path and budget all differ, so a gain in accuracy per generated token is suggested, not shown. No base run at 512 on the same harness exists. Our eval set, an upper bound."
  - date: 2026-09-22
    file: experiments/rl/results/l_state_ablation_g1d04b.json
    call: methodology
    strength: weak
    note: "Likelihood gets worse while task performance rises: held-out CE after 30 steps on G1d-0.4B rose +0.109 with CE only, +0.226 with motion only, +0.402 with L_state. Narrowness of tuning shows at the output."
  - date: 2026-10-02
    file: experiments/rl/results/staged_recall_p1_step100_final.json
    call: supports
    strength: weak
    note: "Precondition: binding skill is trainable. G1d-0.4B, Muon 1e-4, 100 steps: recall 0.72 -> 0.9625, in-window ceiling 0.9875, blank 0.025 (no gap effect: the skill learned is binding keys, not holding them across a gap)."
  - date: 2026-10-07
    file: experiments/rl/results/geomgrid/lr1e-4_s0.json
    call: methodology
    strength: weak
    note: "Optimizer for the training stages, G1d-0.4B full fine-tune, 30 steps, 3 shuffled data orders (experiments/rl/geomgrid_summary.py): best held-out dCE is equal for Muon (3e-5, -0.172) and AdamW (1e-5, -0.168); the zero crossing is at median step ~1.4e-4 for Muon against ~7.5e-5 for AdamW (window about 1.9x wider); Muon at 1e-4 is flat (+0.08). So fine-tuning lr 3e-5 to 1e-4 for Muon, not 0.02."
-->

## HZLK. A model that works latently and regulates itself

*Codename ZLK. The programme that the numbered hypotheses organise (H8, H10, H12, H17, H18, H25), written as one claim
with its mechanism, how the model should work, how it should be trained and why, the tests with their controls, what
changes in the code, and the constraints already paid for. Unnumbered because it is not one test. Status PROPOSED: the
evidence above is a room estimate plus instruments, not a result for the claim; thresholds in the kill conditions and in
§6 are proposals for the owner to confirm before any run.*

| sub-claim | statement | status |
|---|---|---|
| ZLK.1 | CoT spends most of its tokens moving a narrow state; much of it comes after the answer is known, by family | measured on one model, 48 items |
| ZLK.2 | The plain model cannot use extra time (re-read, ticks, pauses); training is what installs it | partly measured; recall control not run |
| ZLK.3 | Tuning narrows the output; the state survives it | partly measured (routing, CE); basin not measured |
| ZLK.4 | A content-dependent latent channel can be trained: CE on the answer through M ticks of the model's own expectation | not run |
| ZLK.5 | The model can learn to choose its own time (continue / stop) under a verifier and a price on time | toy only |
| ZLK.6 | With search over phases the loop can exceed the CoT teacher it started from; distillation alone cannot | not run |
| ZLK.7 | Two views of one content (format / paraphrase) land close in latent space without collapse | not run |
| ZLK.8 | The output side: format from the question, honest self-decisions, then the constitution | silent-mode format obedience measured |

### 1. Claim

1. RWKV-7 should think in its recurrent state, not in emitted tokens, and then say what was asked (or speak freely, as usual).
2. The plain model has the machinery - the state updates on every token and, on our reading, G1's reasoning distillation put
   logic into it (the base already routes 1.68-2.00x more WKV delta into `<think>` spans, L4/L16/L31, measured 2026-08-14) - but
   neither the time nor the control to use it.
3. What training installs is a **self-regulated latent loop**: phases in which the model feeds back its own expectation,
   decides how far to move the state, how many phases to run and when to stop. No size, count or stopping rule is a threshold we write.
4. Minimum claim: the loop matches the CoT teacher's accuracy for materially fewer generated tokens (extraction efficiency,
   КПД). Ambition (ZLK.6): it exceeds the teacher, because phases of different length and content explore the set of
   solutions the weights hold, which one fixed chain does not; plain distillation stays below its teacher, so exceeding it needs
   stimuli beyond imitation: a verifier, a price on time, and search.
5. Training attaches only to verifiable, content-bearing paths: the final answer through the phases, and the model's own
   decisions as sampled tokens. Never to the geometry of the state, and never to the correctness of an individual phase.

### 2. Definitions

- **tick**: one forward step on a fed embedding, no token emitted. Feed = softmax(logits) @ E, rescaled to the median token norm of
  the model's own table (0.376 on G1i; the largest row is 0.64), because the raw expectation shrinks toward the mean
  embedding when the model is unsure (0.93x a median token at top-1 prob 0.99, 0.09x at 0.01).
- **phase**: a run of ticks between boundaries the model chooses. **M**: number of phases. **N**: times the input is fed.
  **K**: emitted CoT tokens. **route**: one way to an answer from the same post-question state (r0 answer at once, lat-k, const-k,
  reread, cot-K).
- **waste fraction** w = CoT tokens after the first checkpoint with a correct answer / all CoT tokens. **tokens to first correct**,
  **tokens at equal accuracy** (the number that matters: loop tokens / teacher tokens at the same accuracy on the same items).
- **useful motion** of a step = projection of its state displacement on the gradient of the log-probability of the correct answer
  (readout lens machinery); **КПД** = useful motion per generated token or per tick. Magnitude of motion is not usefulness.
- **basin width**: output side = entropy of the answer distribution at small temperature T and the T* at which the answer leaves;
  state side = the noise norm at which it leaves. Temperature is an instrument here; the model works at T = 0.
- **view**: the same content under another format, paraphrase, cipher or seed (`content_id` + `view` in the dataset catalog).

### 3. How the model should work (inference)

```
input   the question only (+ ROSA tags); the harness is the question each time
read    s0 = state after the question
loop    feed own expectation  ->  state moves  ->  the model's decision token: continue | stop | (fork) | (delegate to ROSA)
        phases differ in length and content; decisions are tokens inside log pi; there is no per-phase answer reward
commit  content is fixed first; the format is rendered AFTER it, from the question (butler standard, H15) and the
        speak / tool_call / silent function (H16)
```

**What the model controls**: how far to move the state, how many phases, when to stop, what to pull from ROSA, what to write
to state, whether to fork the state into paths and merge (H18; clone identity verified). **What we control**: the environment,
the verifier, the price of time, the data. Two quantities are kept apart: state-motion (|dS|/|S|, direction) and the entropy of
the answer. Confidence is not a stop gate (a readout-confidence exit fired after one repeat on ~80% of steps because templated
answers made the first token >99.9% predictable); it enters only as optimizer pressure - more data per step, a smaller step,
no loss term. Stopping must not rely on the `expected` feed collapsing: it does not collapse (a constant does).

### 4. Why it should be so (mechanism)

- **The state is a narrow neck** (64 directions per head, 12-29 occupied, ~half read, 1-2 facts separable, facts share
  directions). A fixed amount of time cannot fit tasks that need different amounts; a chain of emitted tokens spends hundreds of
  steps to move a narrow state far enough. The combinatorial explosion is over what can be *extracted* from the weights through
  that neck, not over what is stored; phases of varied length and content are how it is searched.
- **Emitted tokens do two jobs at once**: they must stay readable as output and they must move the state. A phase has one job.
  We expect a trained loop to match the token version because the model no longer competes with itself between a stable
  surface and real work (a hypothesis; the per-step data below say a tick moves the state as much as a token).
- **Narrowness is at the output.** Tuning narrows the answer distribution (a 100% tool-call corpus made it answer in that
  format everywhere; held-out CE rises with L_state); LoRA (step9b-e1) left routing geometry unchanged (0.97-1.02x, measured 2026-08-14). LoRA has a small weight
  budget (rank 32 is 1.25% of the dimension) and learns less and forgets less than full fine-tuning (arXiv 2405.09673).
- **Early convergence is the enemy of the loop.** Rewarding a correct answer at every phase teaches the model to be right at
  phase 1 and M>1 disappears. Time has a price; per-phase correctness has none.
- **A constant cannot search.** Repeating one vector is an affine fixed-point iteration: the state slides to that vector's
  fixed point by tick 3 (cosine of consecutive displacements 0.04 -> 0.97 at L20) and, past the verified-safe horizon,
  diverges toward it (H25). A new marker resets direction (cosine 0.10), a content-dependent feed does not collapse. This is
  why the phase channel is the model's own expectation, and why markers are only a boundary and an analytical miniature.
- **Why not plain distillation**: in the default warm-up data 10529/10529 think spans restate the answer (median 18 tokens), so
  it contains no reasoning teacher; state targets from such a teacher pulled phases toward "already knowing the answer"; and a
  state loss tracks nothing about answer quality (state_loss pinned at its clamp while answer CE improved monotonically).
  The teacher must be the model's own verified CoT (same-model rule), and the target is the answer.

### 5. Sub-hypotheses: prediction, control, kill

- **ZLK.1 waste.** Prediction: w >= 30% on at least one family and tokens-to-first-correct well below total tokens; the saving
  is family-dependent. *Measured:* w = 47%, 17/32 at the cap (patterns); nothing to save on arithmetic. *Control:* paraphrase
  floor for any format effect; the same item set for every route. *Kill:* w < 10% on all families.
- **ZLK.2 time is not free.** Prediction: untrained re-read, ticks and constants do not beat answering at once on reasoning
  items; re-read helps recall items (just-read-twice). *Control:* repeat vs neutral filler of equal length; recall items vs
  reasoning items. *Run next:* N=2 on the staged-recall eval vs the pattern eval, 0.4B, CPU. *Kill:* re-read helps reasoning items
  beyond the filler control (then time is usable and C1 is not needed for it).
- **ZLK.3 narrowness at the output.** Prediction: after readout-level tuning, answer entropy at fixed small T and T* shift by
  more than the base model's seed-view spread while the state-noise basin and routing do not; full fine-tuning at matched
  budget moves both less. *Control:* neutral-filler and random-word prefixes (any prefix shifts the greedy path), human-like vs
  uniform digits. *Kill:* the state basin moves with the readout (see frontmatter).
- **ZLK.4 the channel.** Prediction: after C1, accuracy at M=2 (own expectation) > M=1 > M=0 on held-out content x format cells
  of >= 2 families, the per-tick answer curve rises, phases decode to distinct things, and M=0 does not degrade. *Controls:*
  constant-feed arm (must stay at the fixed point), feed swapped between examples (must break the answer), teacher content
  shuffled. *Kill:* in the frontmatter.
- **ZLK.5 self-chosen time.** Prediction: a learned stop reaches the accuracy of the best fixed M with <= 85% of its ticks on a
  mixed-family set and moves with family. *Controls:* random stop with equal mean M; best fixed M per family; price sweep.
  *Kill:* equals fixed M or collapses to M_max.
- **ZLK.6 beyond the teacher.** Prediction: at <= 60% of the teacher's tokens, accuracy >= teacher + 3 points (paired by item)
  on at least one family, none worse than -3 elsewhere. *Room estimate:* the offline agreement rule 0.719 @ 256 tokens vs CoT
  0.688 @ 470 (54% of the tokens, one item on 32). *Kill:* in the frontmatter.
- **ZLK.7 views.** Prediction: a representation term over two views of one content (stop-gradient target, across-example
  contrastive anti-collapse such as CLIPO, never a within-state breadth term - H26: breadth is buildable and worthless) raises
  the similarity of post-question states across views without collapsing the spread across examples, and raises accuracy on
  held-out formats. *Control:* shuffled-view pairs. *Kill:* collapse, or no gain over the answer-only loss.
- **ZLK.8 output side.** Prediction: format is read from the question (JSON and sentence formats obeyed 100%, bare number 0% in
  silent mode on G1i - the gap to close); self-decisions are trained with a proper scoring rule (nanodiff: same accuracy, ECE 0.333
  -> 0.036). The constitution is a short training later, mandatory, not now.

### 6. Training protocol (capabilities, each with its gate)

Common: **full fine-tuning, no LoRA; Muon, lr 3e-5 to 1e-4 (0.02 is the pretraining default and was our collapse), aux AdamW on its own
lr; work layers from the measured profile by depth fraction (0.656 on G1i = L20-22) and never from a copied layer list; the
teacher is the model's own CoT, kept only when verified; every run names its result path, dataset names + SHA-256 and a
written prediction first; at least three data orders per claim and the spread reported; checkpoints are never deleted.**

| capability | what is trained | loss | gate (measured by) | state |
|---|---|---|---|---|
| C0 binding | key->value over a filler gap | CE on the answer | recall / in-window / blank arms; readout lens | done on 0.4B (P1) |
| C1 channel | answer through M ticks of the model's own expectation; mixture over M with M=0 kept (Birdie-style `--m-weights`); 8-token phase budget (the 1-token budget diverged at M=2) | CE on the answer only; teacher-state term at most 0.001 | per-tick answer curve; phases decode distinctly; swapped feed breaks the answer; M=0 not degraded | not run; trainer does not exist for CE through `expected` ticks |
| C2 teacher | model's own CoT on matrices (content x format) and ciphers, verified; M taken from the teacher's own step count | - (data) | КПД vs teacher at equal accuracy per family | generator built (`gen_teacher_cot.py`) |
| C3 time | continue/stop (and fork) as sampled tokens inside log pi; labels = the smallest M at which the model's own sweep is right, then outcome reward with a price on time | CE on labels, then policy gradient | stop accuracy; ticks per family; no collapse to M_max | toy only (halting) |
| C4 format | required format read from the question; honest self-decisions; constitution later | CE / proper scoring rule | format obedience per format; ECE | silent obedience measured |

Hardware record: P1 and the probes run on the 0.4B on the laptop CPU; Phase 1 (LoRA r=32, 500 steps) and the full-FT attempts ran on
GPU VMs (16 GB T4: full-FT batch 2 only, an out-of-memory guard and a host-RAM leak found there; 24 GB A30: Muon peaks at 19.8 GB on 2.9B
where AdamW's two fp32 moments alone are 23.6 GB; bf16 is required at 2.9B). A sitting is budgeted in hours, so C1 is the first GPU job and
everything cheaper is done on the CPU first.

**What changes relative to what was run** (the numbered Phases 1 / 1.5 / 2 / 3 are retired as labels):

| was | becomes | why |
|---|---|---|
| Phase 1: ThinkChain with constant learned markers, LoRA r=32, M=1, teacher states as targets | C1 with the model's own expectation, full FT, answer CE | a constant writes the same thing whatever the task and sits at a fixed point by tick 3; the marker is 2.70x a median token and did not move from its init; LoRA out |
| Phase 1.5: full-FT smoothing, M in {0,1,2} mixture, on the templated warm-up data | C1 mixture over M on the verified teacher set (C2) | 10529/10529 thinks restate the answer: M>1 would split one sentence |
| Phase 2: a constant rewind marker, `--rewind-at-m2`, L_tPC | the capability (return and rewrite) as the model's own re-think / fork-merge inside the loop | the constant-vector family is falsified; the function stays (the owner's correction), the implementation goes |
| Phase 3: GRPO on word-search, binary reward, log pi on answer tokens only | C3/C4: decisions as sampled tokens, verifier + price of time | the loop was deterministic, so beta/gamma/delta/zeta were provably inert (advantage change 6e-8) |
| H10 N x K x mode grid as the test of thinking | instrument only; T-recall and T-gap | N=2 is most likely a recall effect; state_readout cells were an evaluator bug |
| L_state / teacher-state matching as the objective | a shaping term at most (0.001), never a target | state loss pinned while answer CE improved; do-not-latch: beta/gamma/delta/zeta, L_state, stable_rank, retention_range, DE |
| word-search / crossword families as the main RL data | matrices as the platform (content x format), arithmetic and pattern first, ciphers and an invented language for search | character search measures tokenisation as much as reasoning; 36% of tasks, ~60% of tokens |

### 7. Tests (changed) with controls

| test | instrument | control | decision rule | state |
|---|---|---|---|---|
| T-gap | `cot_vs_routes_probe.py` | same items for every route; paraphrase floor for format effects | w, tokens to first correct, per family | done once (32 + 16 items) |
| T-recall | N=2 on recall vs reasoning items | neutral filler of equal length | re-read helps recall only | not run |
| T-influence | `state_step_influence_probe.py` + useful motion (projection on the answer gradient) | constant-feed arm; shuffled answer | useful motion per step: token vs tick vs re-read | magnitude done, usefulness not |
| T-basin | seed views at T=0, temperature sweep, state-noise (A0.5-style) | neutral-filler / random-word prefix; human-like vs uniform digits | output vs state basin, before/after K ticks, LoRA vs full-FT | not run |
| T-efficiency | accuracy-vs-tokens frontier per family | base on the SAME harness at 512 and 2048 | tokens at equal accuracy | base rerun missing |
| T-channel | C1 gate above | constant arm, swapped feed, shuffled content | ZLK.4 kill | after C1 |
| T-stop | C3 gate | fixed-M, random stop at equal mean M | ZLK.5 kill | after C3 |
| T-views | similarity across views + collapse check | shuffled views | ZLK.7 kill | after C1 |
| T-data | hold the method fixed, vary composition (narrow / many families / many formats) | same token budget | does composition matter once the method is rich | not run |
| T-search | unknown-key ciphers, invented language | shuffled key | accuracy vs M | generator built (ciphers); language planned |

### 8. Code and data

| status | where |
|---|---|
| exists | `experiments/rl/`: `train_think_distill.py` (`--student-feed expected`, `--phase-budget`, `--phase-split`, `--work-layers-from`), `gen_teacher_cot.py`, `cot_vs_routes_probe.py`, `state_step_influence_probe.py`, `state_trajectory_probe.py`, `state_readout_lens.py`, `state_capacity_probe.py`, `optimizer_geometry_probe.py`, `loader.py` (`MuonHybrid`, `extend_vocab_for_marker`, `load_weights_into`), `wkv_loop.py`; toys `experiments/A0_state_probe/` (`micro_wkv.py`, `halting_chain.py`, `halting_rl.py`, `attractor_depth_probe.py`); data: `training/datasets.py` + catalog, `gen_ciphers.py`, `gen_staged_recall.py`, matrices |
| must change | the C1 trainer: answer-only CE through `expected` ticks, M from the teacher's step count, M=0 preserved, teacher-state term at most 0.001; every run writes a manifest (dataset names + SHA-256, prediction, result path) |
| to write | decision tokens inside log pi (continue/stop/fork) and their trainer; a verifier registry shared by reward, teacher filter and eval; useful-motion probe; seed/hash view generator (functional and nuisance seeds); invented-language generator; the view term (CLIPO exists, unused) |

### 9. Constraints already paid for (do not re-derive)

LoRA is out. Markers are a boundary, not the substrate; discrete vocabulary tokens are a test rig, the model's own space is the
vocabulary. A constant feed collapses, the expectation does not, but its norm must be matched to a token's. GRPO's log pi covers
answer tokens only, so any shaping on a deterministic loop is inert. Do not reward per-phase correctness. State targets from a
templated teacher are not CoT distillation. Generate teacher data with the SAME model. Personal session data never enters a corpus or
git. Muon at lr 0.02 on a trained checkpoint moves the worst tensor 349% of its norm in one step.

### 10. What exists in the literature, and what is ours

Recurrent-depth latent reasoning with emergent backtracking (Huginn, arXiv 2602.08100) and per-depth decoded process supervision
(LSRL, EMNLP Findings 2025); pause tokens help only after training (arXiv 2310.02226); just-read-twice for recurrent LMs (arXiv
2407.05483); overthinking, with the redundant part after the first correct answer the safe one to cut (arXiv 2412.21187 and
follow-ups); LoRA learns less and forgets less (arXiv 2405.09673); JEPA-style objectives for LLMs (arXiv 2509.14252) and selective
decoding (arXiv 2512.10942). Not found elsewhere: a latent loop on a fixed-size recurrent state whose continue/stop decision is the
model's own action, trained against a verifier and a price on time, with the efficiency measured against that model's own CoT.
(A quick search, not a survey: "not found" is not "does not exist".)

### 11. Not claimed

Nothing about pretraining. No claim that the loop beats the best CoT model, only this teacher on tasks where time helps. No priority
claims. The Muon write-up (`docs/muon-rwkv7-finetuning.md`) is a separate piece.
