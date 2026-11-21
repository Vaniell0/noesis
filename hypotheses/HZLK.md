<!--
id: HZLK
status: PROPOSED
prior: null
posterior: null
depends_on: [H0, H7, H8, H9, H10, H12, H17, H18, H19, H25]
supports: []
contradicts_if:
  - "A loop trained over M>=2 ticks on a verified same-model teacher set does not beat both M=1 and M=0 on held-out items of at least two task families (paired by item, effect >= 5 points and McNemar p < 0.05 on the pooled items)"
  - "At equal accuracy the trained loop still generates more than 80% of the teacher's mean CoT tokens on the same items (G1i, untrained teacher: 470 on 32 patterns, 546 on 16 arithmetic items)"
  - "The loop never exceeds the teacher: after the stop and search stages (S3) it is below the teacher's accuracy on every family at any token budget, i.e. distillation was all there was"
  - "The learned continue/stop action equals the best FIXED M, or collapses to the maximum M, on a mixed-family set; or a random stop with the same mean M does as well"
  - "M>1 only works when a per-phase correctness reward is added (then self-regulation is not what produced it)"
  - "A readout-level tuning (LoRA) that leaves routing unchanged nevertheless moves the state-noise basin by more than the seed-to-seed spread of the base model, i.e. output narrowness and the state are not separable"
  - "Gains persist when the teacher's content is shuffled between examples, or when the fed expectation is swapped between examples (the loop is then fitting format, not extracting)"
evidence:
  - date: 2026-11-12
    file: hypotheses/HZLK.md
    call: methodology
    strength: strong
    note: "Rebuilt under the frame of hypotheses/README.md: the owner's interpretation of the record is stated first (section 1), then every sub-claim as objects, statement, estimator, refuter and what it rests on (section 4), with the extraction claims added (ZLK.11-ZLK.14: stored-not-read, a reader that raises usable capacity, denser packing under a trained reader, derived relations) and the training claims (ZLK.15-ZLK.19). S2 is redefined as knowledge widening through a fixed teacher; the latent-noise policy moves to S3. The pilot narrative is compressed into a run table (section 6b) with its files. Nothing was re-measured for the rebuild."
  - date: 2026-11-12
    file: experiments/A0_state_probe/results/a05_2.9b_h9_verdict.md
    call: supports
    strength: weak
    note: "Correction of how the World-vs-G1 difference was quoted in conversation on 2026-11-12: at 0.4B World's next-token KL under state noise is 2-3x below G1d's (sigma 0.1: 0.108 / 0.033 vs 0.258 / 0.249, a05_ext/verdict.md), but at 2.9B World3 is 40x below G1h (0.0055 vs 0.23) while holding the prompt's identity in the state more tightly (cross-prompt KL 9.64 vs 7.92). A plain pretrain routes little through its state at scale. The measurement is dependence of the output on the state, not computation; it supports starting the loop from World (ZLK.16) as a base whose read path has not settled."
  - date: 2026-11-12
    file: external:docs/community-map.md, Scarletwolf readout corrector 2026-08-12
    call: supports
    strength: weak
    note: "A post-hoc reranker on the WKV-state readout, no retraining, raises a tool-calling bench by +7 at 1.5B, +6 at 2.9B, +2 at 7.2B; 1.5B tuned plus the corrector equals 2.9B tuned (64 vs 63 of 82). A better reader of the same state buys most at the smallest size: the shape ZLK.12 predicts. External, one bench, not our estimator."
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
    note: "The neck: per head 12-29 directions occupied (G1d L12), the answer depends on about half, a linear probe was read as separating 1-2 facts (N=1 1/1, N=2 2/2, N=4 1/4, N=8 0/8), but that probe's solver (lstsq gelsy) was erratic on this design; with ridge (lambda by CV) the L11-13 state reads 3.6 / 5.5 / 6.7 / 6.0 bound values (sum of held-out R2) at N = 4 / 8 / 16 / 32, against about 3 recalled at 8 (2026-11-12). Readout lens checked against causal deletion (Spearman 0.92-0.98)."
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
# ZLK sub-claims: the source of the table in 'Sub-claims at a glance' (python experiments/regenerate_hyp_index.py).
# regime = the data and the model's state the status was measured in (pretrained vs from scratch).
claims:
- id: ZLK.1
  claim: CoT spends many tokens after the answer is already there
  status: measured on one model, 48 items
  regime: 'G1i 2.9B pretrained, 32 pattern items'
- id: ZLK.2
  claim: untrained time (re-read, ticks, pauses) is not used
  status: regular; recall control not run
  regime: 'G1i 2.9B pretrained, untrained ticks and re-read'
- id: ZLK.3
  claim: tuning narrows the output; the state survives it
  status: partly measured
  regime: 'pretrained, LoRA readout tuning'
- id: ZLK.4
  claim: a channel of the model's own latent can be trained without erasing the answer
  status: '0.4B: tick no longer erases; every arm a pass-through'
  regime: G1d 0.4B pretrained
- id: ZLK.5
  claim: the model can learn to choose its own number of ticks
  status: 'CONFIRMED on the toy 11-20 (2 of 3 seeds, sL_ramp1): own stop without a head equals fixed M8 at 62-69% of the ticks; seed 6 lost 0.10 to training'
  regime: 'toy from scratch (chase), start from a trained C model; pretrained not run'
- id: ZLK.6
  claim: with search the loop exceeds its teacher's one-pass reading
  status: not run; bounded by T1/T2
  regime: ''
- id: ZLK.7
  claim: two views of one content land close without collapse
  status: 'precondition measured 11-20: trained ticks raise cross-script state CKA without collapse (toy loop, 3 seeds, 0.469 -> 0.625 at width 48, fading with width; 0.4B kl_T2_k4 +0.07 / +0.05; base-model M4 readings are untrained ticks, not evidence; other text 0.15-0.22); the trained term not run'
  regime: 'toy from scratch (two scripts); 0.4B pretrained, inference only'
- id: ZLK.8
  claim: content first, format from the question; honest self-decisions
  status: obedience measured, near 0
  regime: 'G1i 2.9B pretrained, silent answers'
- id: ZLK.9
  claim: a persistent state absorbs the configuration once
  status: H17, contested
  regime: "none measured (the owner's sessions)"
- id: ZLK.10
  claim: an exact-recall branch queried at every tick keeps the past the settling state forgets
  status: not started
  regime: ''
- id: ZLK.11
  claim: a pretrained model holds facts it does not read in one pass
  status: measured on Qwen3.5 0.8B; not on RWKV
  regime: Qwen3.5 0.8B pretrained
- id: ZLK.12
  claim: a trained reader raises usable capacity toward what is held
  status: 'supported in part 11-20: after 4 trained ticks (kl_T2_k4) a linear reader gets the next word at 0.177 where the head gets 0.101 (head ahead without ticks, 0.373 vs 0.254): what the state holds exceeds what the head reads; T15 earlier: ticks +2% readable, training adds nothing'
  regime: G1d 0.4B pretrained (state readout; head vs reader, same target)
- id: ZLK.13
  claim: storage and reading trained together pack more at equal readability
  status: proposed; proved part for the state
  regime: ''
- id: ZLK.14
  claim: relations derived from held facts are usable knowledge the loop can add
  status: proposed
  regime: ''
- id: ZLK.15
  claim: knowledge enters through a fixed teacher on the student's own tokens without breaking the rest
  status: partly measured (Qwen)
  regime: 'Qwen3.5 0.8B pretrained'
- id: ZLK.16
  claim: a loop learned from a plain pretrain beats the same chain grafted on a finished model
  status: 'toy form REFUTED 11-20 (1o: ticks grafted onto a one-pass toy pass k6 in 2 of 3 seeds, as reliable as grown); pretrained form not run'
  regime: toy from scratch (chase); pretrained not run
- id: ZLK.17
  claim: World trained by the chain against G1d compares two training methods from one base
  status: proposed
  regime: ''
- id: ZLK.18
  claim: trained ticks widen the attractor instead of collapsing it, and the output grows more expressive
  status: 'one checkpoint supports it (kl_T2_k4); counter-case 11-21: 1t (100 full-weight steps on 160 Caesar rows, KL + GRPO, no pretraining CE beside it) broke the chat through its ticks (loops, refusals): the widening holds only with the pretraining term kept'
  regime: G1d 0.4B pretrained; long-distance regime
- id: ZLK.19
  claim: separability is installed by tasks that need it, not by an answer loss alone
  status: 'P1 negative; toy: a task one pass cannot solve made the ticks a register, a per-hop target made the step iterable'
  regime: toy from scratch; P1 on G1d 0.4B
- id: ZLK.20
  claim: the neck of one-pass reading is the late fall of rank, where key interference is highest
  status: rank fall measured; interference not
  regime: G1d 0.4B and 2.9B pretrained
- id: ZLK.22
  claim: loop training on a mix of domains folds them into one system that transfers to a held-out domain,
    below a specialist but smoother
  status: 'toy 11-20: no transfer in bpc (1n plain/Cyrillic: joint +0.46 bpc vs Cyrillic-only, REFUTED; 1j plain+cipher -0.014), but the loop raises deep-layer CKA between scripts 0.44 -> 0.71; cs: the context step did not reach stored facts'
  regime: 'toy from scratch, 55K (1n) and 271K (1j), one seed each; pretrained not run'
  regime: 'toy from scratch, 55K (1n) and 271K (1j), one seed each; pretrained not run'
-->

## HZLK. Second pretraining: make what the weights hold readable, and read it through a self-regulated latent loop

*Codename ZLK. The programme that the numbered hypotheses organise, written as the owner's interpretation of
everything the project has measured, then as claims in the form of the frame (`hypotheses/README.md`): objects,
statement, estimator, refuter, and what each rests on today. Rebuilt 2026-11-12; the version before it is in git
(`af4da08`). Status PROPOSED: no sub-claim of the programme has passed a registered test; thresholds below are
proposals for the owner to confirm before the run that tests them.*

How to read this file. §1 is the interpretation, in short numbered theses (13), each pointing at the claims that make
it testable. §1b is what is proved and what is regular. §1c is reach per growth (`S`) as the direction of
training where there is no geometry. §2 defines the objects. §3 says how the model should
work. §4 is the claims. §5 keeps the positions that are still argued. §6 is the training chain derived from the
claims, with the record of every run in §6b. §7-§11 are tests, code, paid constraints, literature, and what is
not claimed.

### Sub-claims at a glance

<!-- ZLK TABLE: generated from the claims records above by experiments/regenerate_hyp_index.py -->

| id | statement (short) | status | regime | evidence files | linked |
|---|---|---|---|---|---|
| ZLK.1 | CoT spends many tokens after the answer is already there | measured on one model, 48 items | G1i 2.9B pretrained, 32 pattern items | 0 | <-H24 |
| ZLK.2 | untrained time (re-read, ticks, pauses) is not used | regular; recall control not run | G1i 2.9B pretrained, untrained ticks and re-read | 0 | <-H21 |
| ZLK.3 | tuning narrows the output; the state survives it | partly measured | pretrained, LoRA readout tuning | 0 | - |
| ZLK.4 | a channel of the model's own latent can be trained without erasing the answer | 0.4B: tick no longer erases; every arm a pass-through | G1d 0.4B pretrained | 0 | - |
| ZLK.5 | the model can learn to choose its own number of ticks | CONFIRMED on the toy 11-20 (2 of 3 seeds, sL_ramp1): own stop without a head equals fixed M8 at 62-69% of the ticks; seed 6 lost 0.10 to training | toy from scratch (chase), start from a trained C model; pretrained not run | 0 | <-H24 |
| ZLK.6 | with search the loop exceeds its teacher's one-pass reading | not run; bounded by T1/T2 | - | 0 | ZLK.1 |
| ZLK.7 | two views of one content land close without collapse | precondition measured 11-20: trained ticks raise cross-script state CKA without collapse (toy loop, 3 seeds, 0.469 -> 0.625 at width 48, fading with width; 0.4B kl_T2_k4 +0.07 / +0.05; base-model M4 readings are untrained ticks, not evidence; other text 0.15-0.22); the trained term not run | toy from scratch (two scripts); 0.4B pretrained, inference only | 0 | - |
| ZLK.8 | content first, format from the question; honest self-decisions | obedience measured, near 0 | G1i 2.9B pretrained, silent answers | 0 | - |
| ZLK.9 | a persistent state absorbs the configuration once | H17, contested | none measured (the owner's sessions) | 0 | H17 |
| ZLK.10 | an exact-recall branch queried at every tick keeps the past the settling state forgets | not started | - | 0 | <-H23 |
| ZLK.11 | a pretrained model holds facts it does not read in one pass | measured on Qwen3.5 0.8B; not on RWKV | Qwen3.5 0.8B pretrained | 0 | - |
| ZLK.12 | a trained reader raises usable capacity toward what is held | supported in part 11-20: after 4 trained ticks (kl_T2_k4) a linear reader gets the next word at 0.177 where the head gets 0.101 (head ahead without ticks, 0.373 vs 0.254): what the state holds exceeds what the head reads; T15 earlier: ticks +2% readable, training adds nothing | G1d 0.4B pretrained (state readout; head vs reader, same target) | 0 | ZLK.11 |
| ZLK.13 | storage and reading trained together pack more at equal readability | proposed; proved part for the state | - | 0 | - |
| ZLK.14 | relations derived from held facts are usable knowledge the loop can add | proposed | - | 0 | H12 |
| ZLK.15 | knowledge enters through a fixed teacher on the student's own tokens without breaking the rest | partly measured (Qwen) | Qwen3.5 0.8B pretrained | 0 | H7 |
| ZLK.16 | a loop learned from a plain pretrain beats the same chain grafted on a finished model | toy form REFUTED 11-20 (1o: ticks grafted onto a one-pass toy pass k6 in 2 of 3 seeds, as reliable as grown); pretrained form not run | toy from scratch (chase); pretrained not run | 0 | H9, ZLK.12 |
| ZLK.17 | World trained by the chain against G1d compares two training methods from one base | proposed | - | 0 | - |
| ZLK.18 | trained ticks widen the attractor instead of collapsing it, and the output grows more expressive | one checkpoint supports it (kl_T2_k4); counter-case 11-21: 1t (100 full-weight steps on 160 Caesar rows, KL + GRPO, no pretraining CE beside it) broke the chat through its ticks (loops, refusals): the widening holds only with the pretraining term kept | G1d 0.4B pretrained; long-distance regime | 0 | - |
| ZLK.19 | separability is installed by tasks that need it, not by an answer loss alone | P1 negative; toy: a task one pass cannot solve made the ticks a register, a per-hop target made the step iterable | toy from scratch; P1 on G1d 0.4B | 0 | - |
| ZLK.20 | the neck of one-pass reading is the late fall of rank, where key interference is highest | rank fall measured; interference not | G1d 0.4B and 2.9B pretrained | 1 | H12, ZLK.18 |
| ZLK.22 | loop training on a mix of domains folds them into one system that transfers to a held-out domain, below a specialist but smoother | toy 11-20: no transfer in bpc (1n plain/Cyrillic: joint +0.46 bpc vs Cyrillic-only, REFUTED; 1j plain+cipher -0.014), but the loop raises deep-layer CKA between scripts 0.44 -> 0.71; cs: the context step did not reach stored facts | toy from scratch, 55K (1n) and 271K (1j), one seed each; pretrained not run | 0 | - |

<!-- END ZLK TABLE -->

### 1. The interpretation (the owner's reading of the record, in my words)

Each thesis is dated by when he stated it and points at the claims that test it. Where the record contradicts a
thesis, §5 says so.

1. **A model is weights plus harness.** The weights are the constant and hold what the model knows and can do;
   the state and the activations are the present moment. Output is an interface, not the thought. Training that
   attaches only to output funnels the model to one target and never lets it use its environment for itself.
   (2026-10-01/02; ZLK.3, ZLK.8)
2. **The explosion is in extraction, not in storage.** The heads' directions collapse, the model cannot bring
   most of what it holds to the output in one pass, and it converges instantly on the most probable answer: the
   attractor. Time in the loop is how what is held gets searched. The neck is the fall of rank toward the end
   of the model, where interference is highest (2026-11-12); the first image models built themselves a memory
   there, and the loop needs work instead (2026-09-22). (2026-10-02; ZLK.2, ZLK.11, ZLK.12, ZLK.18,
   ZLK.20)
3. **Efficiency of extraction is the measure.** A CoT model spends hundreds of tokens to move a narrow state far
   enough; efficiency is information gained about the answer per tick per unit of state motion (R_density,
   2026-08-19). Success is a better answer than the CoT teacher for less compute. (ZLK.1, ZLK.6)
4. **The model regulates itself.** It decides how far and where to move its state, how many ticks, and when to
   stop, by what it generates: an unreadable latent tick continues, a vocabulary token speaks. No threshold is
   written by us; confidence is optimizer pressure, not a stop rule. Reflection is the model taking apart and
   reorganising what it already holds. (2026-10-06, 10-22; ZLK.5, ZLK.18)
5. **The attractor is the enemy.** However large, it is bounded; the chain should widen both its capacity and
   the way it forms, moving the distribution of the latent space, not the distribution of the output.
   (2026-10-19, 10-22; ZLK.18; refined in 14 on 2026-11-21: the enemy is the global or premature attractor, while
   a per-answer attractor reached after the steps is where the program halts)
6. **Pretraining asks the state to be sufficient; the loop needs it separable.** Design the task, not the
   objective: a task where keeping items apart is the only way to answer makes the answer reward the
   separability reward. (2026-09-22; ZLK.19)
7. **Pretraining stored facts with losses to interference, like bad sectors.** Our training is a second
   pretraining that repairs the reading of what is stored and finds what was badly written so that the repair
   can be targeted. (2026-11-04/05; ZLK.11, ZLK.12)
8. **A reading policy refills what interference blocked, and the model grows in usable size without more
   parameters.** A reader that untangles allows denser packing, and relations the model derives between
   scattered facts are new usable understanding, which is what the processor needs. (2026-11-12; ZLK.12,
   ZLK.13, ZLK.14)
13. **Neuron count does not set intelligence; depth does, and depth can come from time.** Model growth paid off
    because it brought depth. Image models went through this once: a new channel (recurrence, memory) brought
    memory and then thinking without a wider interface. A tick is the same layers run again on the task, so
    each pass works it further instead of letting it slip through: mixture of experts in time, not in weights.
    Depth grows through the policy and through data the model generates for itself. What weights can store
    still grows with size; skills, reading and work with context need not. (2026-09-22, 2026-11-12; ZLK.12,
    ZLK.16, ZLK.20). The vision record this rests on: a shallow RNN with weights shared across steps is
    exactly a very deep ResNet with weights shared across layers and performs close to it with orders of
    magnitude fewer parameters (Liao & Poggio 2016, arXiv 1604.03640); deep classifiers end in a compressing
    "tunnel" (ZLK.20).
9. **New knowledge enters only on the model's own tokens.** Anything else breaks the rest; after S1 the model
   should be filled with knowledge as in pretraining, so the channel saturates and the model stays plastic,
   learning facts and the linking of pieces from different sources: weights times context, plus the latent
   channel that operates on both. (2026-10-22, 11-04; ZLK.15)
10. **The loop is learned early, not grafted on.** Late checkpoints have a settled read path; a plain pretrain
    (World) trained by our chain, against G1d (World after Peng's G1 training), compares two training methods
    from one base. (2026-11-11/12; ZLK.16, ZLK.17)
11. **The exact past lives beside the state.** A QKV-ROSA branch queried at every tick keeps the exact history
    while the state settles and forgets; that last stage makes the model autonomous on weak hardware, where long
    context is the main pain. (2026-10-28; ZLK.10)
12. **Tools adapt to the information, not the reverse.** Vocabulary tokens appear only where the model itself
    starts to speak; the loop runs on its own latent. (2026-10-21; ZLK.4)
14. **Freedom is in the choice of path; the end of the path is the answer's point, not a global attractor.**
    Freedom is freedom from the interface: the model explores its own latent space and what it can do there,
    and uses what it finds to make the work simpler. In the loop the end of a trajectory is set by the model's
    own choices along it, not by its start alone: the states reachable from one post-question state within `M`
    ticks have volume, and the model picks where in it to go. Refined 2026-11-21 by the toy's attractors
    (`attractor_probe.py`): a trained loop steps k times and then settles into a region of the answer, shared
    across contexts, and its own stop is that settling. So convergence is the program halting, not a loss of
    freedom; what kills freedom is one point for all prompts (no computation) or a per-prompt point at tick 1
    (`kl_T2_k4`: nothing left to choose). The volume belongs to the transient before the halt; at the halt the
    map should contract, or the answer would not be stable.
    The ticks are the scratchpad: re-reading rewrites the model's own record more densely (proved for the
    state, T6), which frees room to turn. RL that binds heterogeneous information more tightly raises the
    bits each part of the model carries, the loop decodes them, and later stages shorten the loop and order
    the knowledge unevenly for a cheaper answer. In H25's terms the weights are the processor (policies, and
    the knowledge that classifies objects and picks the policy) and the WKV state is the memory the program
    runs on. Freedom is not noise: rollouts that differ by the model's choices must still answer right.
    (2026-11-15; ZLK.18, ZLK.5, ZLK.13; `H25.md` A)

### 1b. Theory: what is proved, what is regular, and which runs are false before they start

Proved from the setup (proofs in `docs/state-and-reasoning.md`):

- **T1. The loop adds usable information, never information.** With the input fixed, the state after any number
  of ticks is a function of the post-question state `S_0`, so `I(S_t; Y) <= I(S_0; Y)`. What can grow is the
  usable information `I_V` that a bounded reader extracts (Xu et al. 2020), the reader being the model's own
  remaining layers and head. The gain `G_M = log p_M(y*) - log p_0(y*)` of one checkpoint estimates it.
  "Decompression" is raising `I_V` toward `I`; a bad sector is `I > I_V` for one pass.
- **T2. New information enters only through the input.** A fact absent from the input and the weights is out of
  reach for any `M`. A reward with `|R|` outcomes carries at most `log2 |R|` bits per episode, so reinforcement
  reweights what the model holds; new knowledge comes from a teacher that has it, in its context or its weights,
  passed on the student's own tokens with the teacher fixed. What is derivable from the weights by computation
  counts as held (`H12.md` §4): T2 bounds facts, not derived relations.
- **T6. Repeated delta-rule passes converge where one pass interferes.** A head's step is a generalised SGD step
  on a linear key-to-value map (`docs/state-and-reasoning.md` §2). One pass of pure writes reads back each
  binding plus crosstalk from every overlapping key; the delta step repeated over the same bindings converges to
  the least-squares map, exact for up to `n` linearly independent keys. Within a context, re-reading writes the
  residual the first pass left. This is proved for the state only; for the weights it is the analogy behind
  ZLK.13, not a theorem.

Regular (every run agrees; one counterexample refutes):

- **T3. Pass-through is the optimum unless the task is deeper than one pass.** An output-level loss is lowered
  more cheaply by repairing the one-pass read than by computing in the ticks whenever the computation fits one
  pass plus the answer's own tokens. Ticks are needed by construction only for short answers whose computation is
  long: k-hop chains with k >= 3, "which two share a city", tracking, iteration, counting.
- **T4. The loop needs a channel that is not collapsed.** If ticks carry intermediate results they differ across
  prompts and ticks (effective rank > 1, step_c > 0). Every breakage on record narrowed the channel (Muon step
  size; CE sharpening, P1; one attractor for all prompts, G1k / G1l; one format; a self-teacher drifting with the
  student); every success kept it wide (KL at T=2 to the own distribution; a frozen teacher on the own tokens).
- **T5. The channel needs an architecture that lets a vector be a non-word.** With a head tied to the input
  embeddings and a soft feed, a fed vector is read as the next word (Qwen3.5: ticks are word embeddings, one
  trajectory for every prompt). RWKV-7 has a separate head and ln0 on its input, so only the direction arrives.

**Pre-run gate.** A run is false, and is not started, unless all five answers are yes:
1. Is the needed information in the input plus the weights of the model or of its fixed teacher? (T1, T2)
2. If the run claims ticks: is the answer short while the computation is longer than one pass? (T3)
3. Is every target a teacher's distribution, with the teacher fixed, and no foreign token string as a target?
   (T2, T4; the one exception is §6 W0, form for a base that cannot answer at all, named there)
4. Is it gated by `G_M` at `M = k` vs `M = 0` of the same checkpoint and vs an arm trained at `M = 0`, on facts
   split from training, plus the channel's spread and the output read in chat? (T1, T3, T4)
5. Can the architecture carry a non-word vector? (T5)

Applied to the record, the runs that would have been stopped: every Qwen ticks arm (5); every S1 on one-pass
tasks (2, 4); gold-string decode CE (3); the self-teacher on-policy run (3); fresh-text compression as a repair
gate (1: its high bits are mostly absent information); the next-word chat game (the instrument failed).

### 1c. Reach per growth: the training direction where there is no geometry

The owner's rule for code (2026-11-04) is `S = T / (N⁺ + 1)`: a change is better the more of the existing
description it re-expresses (`T`, reach) and the less it grows it (`N⁺`, net growth). He reads it (2026-11-12)
not as a number but as a policy, an optimizer and a gradient for the places where training has no geometry, and
names the Kolmogorov criterion of compression as the base of the whole project (2026-11-05). In the frame's
terms:

*Objects.* A description `D` (code, weights, the context or state the model writes, the data it makes for
itself, a rollout of the loop), its behaviour `F(D)`, and a change `Δ`. Reach `T(Δ)`: how much of `D` the change
re-expresses, measured where the behaviour is read (lines touched in code; for the loop, how far the ticks moved
the answer: `G_M`, or `KL(p_M || p_0)`). Growth `N⁺(Δ)`: what the change adds to the description (lines added;
for the loop, ticks, emitted tokens, state motion `Σ‖ΔS‖`).

- **C1 (identity: one criterion at every level).** КПД = `ΔI_V / (M · Σ‖ΔS‖)` is `S` for the loop: usable
  information gained per motion spent. Accuracy per emitted word (DE) is `S` for the output; facts read per
  parameter (ZLK.13) is `S` for storage; `S` itself is the criterion for code. The programme measures one thing
  at four levels: reach per growth.
- **C2 (proved: a gradient needs no geometry of the outcome space).** For a policy `π_θ` that proposes outcomes
  `a` in any space (an edit, what to write to ROSA, continue or stop, which data to make), `∇_θ E[u(a)] =
  E[u(a) ∇_θ log π_θ(a)]` (Williams 1992). The geometry used is that of `θ`; the outcome space needs only a
  score `u`.
- **C3 (proved: order is enough).** Replacing `u` by its rank within a group of rollouts leaves the direction
  invariant to every monotone transform of the score; GRPO's group normalisation is invariant to affine ones
  only. So `S` never has to be calibrated as a number: a judge that orders two rollouts gives the same
  direction as one that scores them.
- **C4 (identity: the geometry the model induces).** Where the parameters carry no metric suited to the task,
  the model's own outputs supply one: `KL(p_θ || p_{θ+δ}) = ½ δᵀ F δ + O(δ³)`, `F` the Fisher information. The
  losses of the chain (KL at T=2 to the own distribution, reverse KL to a frozen teacher on the student's own
  tokens) step in the geometry of the model's outputs; the natural policy gradient (Kakade 2001) is the same on
  the policy side. It is the output-side analogue of H26's strong form (the step metric induced by what the
  weights write rather than by their coordinates).
- **C5 (proved: compression is not a matter of taste).** For a hypothesis described in `L` bits that fits `n`
  independent samples, with probability `1 − δ` its error is at most `(L ln 2 + ln(1/δ)) / n` (Occam bound,
  Blumer et al. 1987, realisable case). Among changes that keep the behaviour on the data, the one that shortens
  the description has the tighter bound; `S` ranks changes by that.
- **C6 (conjecture: the curriculum is the slope of S).** A task is worth training on while the model's `S` on it
  is still rising; where `S` has saturated it adds nothing, where `S` is zero it is out of reach (Schmidhuber's
  compression progress as the drive of curiosity, 2009, cited from memory). This is the owner's "saturate the
  loop with work", with no threshold of ours: items are drawn by the slope of their `S` over training.
  *Prior work (read 2026-11-12).* Absolute Zero (arXiv 2505.03335) is the nearest recipe: one model proposes
  and solves, an executor verifies, and the proposer is paid 1 - solve rate, 0 at solve rate 0; it gains most
  on larger bases, an untrained proposer costs only 1.4 points, and stripping hints (comments, leaked
  variables) from the tasks hurt, because hints make unsolvable tasks solvable and the solver climbs out of
  them. MATES (arXiv 2406.06046) scores data by the drop of a reference loss after one training step on it and
  refits a small scorer as the model's preferences shift. Both rank by the effect on loss or accuracy; C6 ranks
  by reach per growth of the loop, which neither measures. The cipher generators can play the executor (exact
  gold), so a learned critic only orders and cannot be gamed into inventing truth.

*Where the chain has no geometry* and takes its direction from C2-C3: the continue/stop decision (S3), what a
tick asks of ROSA (S5), the data the model generates for itself, which items enter the curriculum, and edits
to code and context outside the model. Each needs a policy whose log-probability carries the gradient and an
order on outcomes; nothing else.

*What it makes easier.* The S3 reward needs no calibrated value: ordering rollouts by the right answer, then by
fewer ticks, is the whole reward. A critic for open tasks only has to compare two rollouts, which is easier to
keep fixed and honest than a scorer; anchor items with a known answer keep it honest, and it adds only what it
knows itself (T2). The curriculum comes from the slope of `S`, not from thresholds we write (thesis 4).

*Estimator and refuter of the conjecture part.* C1 as a regularity: across checkpoints, reach per growth at the
loop level (R_density, tokens at equal accuracy) predicts held-out gain better than the training loss does;
refuted if it does not correlate. C6: a curriculum drawn by the slope of `S` against a uniform one at equal
steps, gain on held-out deep items; refuted if equal. *My caveat:* reach must be read on the answer (`G_M`,
`KL`), not on the motion, or ticks that move the state far and change nothing would count as reach; the zero-
motion guard of R_density applies to `S` in the same way.

### 2. Definitions

- **own latent**: the model's last-layer vector (before the head) fed back as the next input, Coconut-style
  (`loader.py::_peft_forward_embeds`). Not a vocabulary row, not the word hull, not a constant. RL stochasticity
  comes from a noisy continuous feed (`z = h + sigma * eps`, a Gaussian log pi), not from discreteness.
- **tick**: one forward step on a fed vector, no token emitted. **M**: the number of ticks before the answer, one
  number (owner 2026-10-27). A **phase** exists only where a boundary does something; the natural boundary is the
  model emitting a vocabulary token between runs of ticks (S3). Feeding the input again (**N**) is not an axis of
  the loop; inside it, re-reading the exact history is what ROSA gives a tick (ZLK.10). **K**: emitted CoT
  tokens. **route**: one way to an answer from the same post-question state.
- **I, I_V**: the information the state after the question holds about the answer, and the part the model's own
  reader extracts. **G_M**: the per-item gain of §1b T1.
- **fact classes** at `M = 0`: **read** (greedy right); **stored, not read** (greedy wrong, the gold ranks high
  under teacher forcing or appears among samples at T = 1); **lost** (neither). Estimators: `m0_pass_probe.py
  --m0-rank`, `--m0-samples`. The **transition table** is the classes of the same items before and after a stage.
- **derived relation**: an answer computed from two or more held facts and never stated as one (who shares a
  city, the capital of the country where X lives).
- **separable facts per head**: stored bindings a held-out linear probe tells apart on one head's state
  (`state_capacity_probe.py`), read beside the design's effective rank; with `--ticks M` after `M` own-latent
  ticks.
- **effective size**: the size of the plain model of the same family that matches a checkpoint (with its ticks)
  on the same item set. The measure of "bigger without parameters".
- **КПД (efficiency)** = R_density = `ΔI_V / (M · Σ‖ΔS‖)`: usable information gained per tick per unit of state
  motion (owner 2026-08-19). Proxy for `ΔI_V`: `G_M`, or the drop of answer entropy on the gold. Per tick:
  `(H_{t-1} - H_t) / rel_motion_t`, undefined when the motion is below 1e-3 (`noesis-tick-yield-probe` spec).
- **channel spread**: effective rank, PC1 share and centred step size `step_c` of the fed latents across a prompt
  set per tick; settle tick; cross-prompt cosine on the state `S`, not only on `h`.
- **waste fraction** w = CoT tokens after the first correct checkpoint / all CoT tokens; **tokens at equal
  accuracy** (loop tokens / teacher tokens at matched accuracy on the same items).
- **basin width**: entropy of the answer at small T and the T* at which it leaves (output side); the noise norm at
  which it leaves (state side).
- **re-read ratio** (system level): cache-read input tokens / output tokens over a session.
- **view**: the same content under another format, paraphrase, cipher or seed (`content_id` + `view`).

### 3. How the model should work (inference)

```
input   the question only (+ ROSA tags); the harness is the question each time
read    S_0 = state after the question
loop    own latent (h -> next input) -> state moves -> the model's decision, in one softmax:
        an unreadable latent tick continues | a vocabulary token speaks | (fork) | (query ROSA)
        ticks differ in content; there is no per-tick answer reward
commit  content is fixed first; the format is rendered after it, from the question (butler standard, H15)
```

What the model controls: how far to move the state, how many ticks, when to stop, what to pull from ROSA,
whether to fork and merge (H18). What we control: the environment, the verifier, the price of time, the data.
State motion (|dS|/|S|, direction) and answer entropy are kept apart; confidence is never a stop gate (a
readout-confidence exit fired after one tick on ~80% of steps because templated answers made the first token
predictable).

### 4. The claims

Each: statement; estimator; refuter; what it rests on; and the reading (the owner's and mine where they differ).

#### A. What one pass leaves behind

**ZLK.1 Waste.** A CoT teacher spends a family-dependent share of its tokens after its answer is already right.
*Estimator:* w and tokens to first correct per family (`cot_vs_routes_probe.py`). *Refuter:* w < 10% on every
family. *Rests on:* G1i 2.9B, w = 47%, 17 of 32 at the 768 cap on patterns; nothing to save on arithmetic. The
Qwen3.5 0.8B own CoT finds the capital at 192-256 tokens and then keeps going (71 of 73 kept CoTs never close in
768).

**ZLK.2 Untrained time is not used.** Re-reading, latent ticks and pauses do not beat answering at once on
reasoning items of an untrained model; re-reading helps recall items. *Estimator:* the same items at once, read
twice, with `M` ticks; recall vs reasoning items; neutral filler of equal length. *Refuter:* re-reading helps
reasoning items beyond the filler control. *Rests on:* G1i 0.469 vs 0.531 (patterns), ticks 0.344; one untrained
latent tick erases the answer (G1i 0.85 -> 0.50, G1k 0.95 -> 0.50).

**ZLK.3 Narrowness is at the output.** Readout-level tuning shifts answer entropy and T* by more than the base
model's seed spread while the state-noise basin and routing do not move. *Estimator:* T-basin on output and
state, before and after; LoRA vs full FT at matched budget. *Refuter:* the state basin moves with the readout.
*Rests on:* LoRA left routing at 0.97-1.02x (2026-08-14); a 100% tool-call corpus made every answer a tool call;
held-out CE rises while task scores rise (+0.109 to +0.402 on 0.4B). Basin not measured.

**ZLK.11 Stored, not read (bad sectors).** For a pretrained model, a measurable share of the facts answerable from
its weights is in the class stored-not-read; the greedy misses land on neighbours of the gold or on a common
attractor, not on random answers. *Estimator:* fact classes on an entity battery with type-matched alternatives
(the capital probe generalised). *Refuter:* stored-not-read under 5% of misses on a battery of at least 200 facts;
or misses uncorrelated with the gold's rank. *Rests on:* Qwen3.5 0.8B, 49 capital questions: greedy right 17, gold
ranked first among the 32 capitals while greedy misses 25; misses are neighbours (ZURICH for Bern) or a generic
LONDON; own CoT reads the question rung 1 -> 10 of 10. Two hops: the capital asked directly ranks first (BERN rank
1), inside a two-hop item rank 22. Not measured on RWKV: the battery is not built; a per-token rank map of 959
Wikipedia intros exists (`badsector_map_vital3`, Qwen) and cannot separate a badly read fact from a free
continuation without a question form.

**ZLK.20 The neck is the late fall of rank.** The bottleneck of one-pass reading is the narrowing of the state
before the readout: from a depth fraction near 0.656 to the head, heads hold fewer live directions and fewer
separable facts than mid-stack, and the erase keys of distinct items overlap more. *Estimator:* live directions
per head at stride 1 over layers (`jlens_probe.py`); separable facts per layer (`state_capacity_probe.py --per-layer`,
ridge with lambda by CV); the mean overlap `|κ̂_i · κ̂_j|` of the erase keys written for distinct items,
per layer (`H12.md` estimator 4). *Refuter:* late layers separate as many facts as mid-stack, or their key
overlap is not higher (then the narrowing is compression toward the answer, not interference). *Rests on:* G1i
and G1k 2.9B fall from 16.0 to 10.9 live directions at L21 -> L22 (depth 0.656, both corpora) and hold 8-11 at
L24-L28; on G1d 0.4B the directions the answer depends on are 8.7 / 8.2 / 6.3 / 6.4 at L8 / L12 / L16 / L20,
and late layers depend on directions below the occupancy threshold; P1's sharpening hit L20 hardest (6.4 ->
2.6). The 0.4B cliff, predicted at L15 -> L16, is measured (2026-11-12, `experiments/rl/results/state_capacity_profile_g1d04b_M0.json`,
one layer = its 16 heads, CV ridge): facts read, as the sum of held-out R2 over N = 8 bound values, are 4.2 / 3.4 /
1.6 / 0.7 at L14-L17 and 0.0-0.4 at L18-L23, against 0.4-2.2 at L2-L11; at N = 16 L15 holds 0.5 and L16 none.
L12 alone keeps rising with stored pairs (3.6 / 5.5 / 6.8 at N = 4 / 8 / 16; its design rank is also the
highest). The capacity half of the refuter did not fire on 0.4B; key overlap is not measured anywhere. *Reading
for the loop:* a tick feeds back the vector that has passed the neck and sends it through every layer again, so
the gain of ticks should be looked for at the late layers. *History it rests on* (image classifiers; the owner's
remark of 2026-09-22 that the first image models built themselves a memory, and that we need work instead): deep
classifiers split into an extractor that builds linearly separable representations and a "tunnel" that
compresses them, with the numerical rank of the representations falling toward the number of classes; the
tunnel forms early in training, its depth grows with capacity relative to the task, and it hurts
out-of-distribution transfer (Masarczyk et al., NeurIPS 2023, arXiv 2305.19753; not universal, arXiv
2405.15018). At the end of training the last-layer features collapse onto their class means and the classifier
becomes nearest-mean (neural collapse, Papyan, Han, Donoho, PNAS 2020): the attractor of ZLK.18 in classifier
form. Memorisation of randomly labelled examples sits mostly in the deeper layers, where object manifolds shrink
in radius and dimension, and rewinding the last layers restores generalisation (Stephenson et al., ICLR 2021,
arXiv 2105.14602; later work finds it spread wider). The transfer to RWKV is not direct: these are residual
representations of classifiers with few classes, our numbers are per-head state ranks of a language model. The
same estimator (numerical rank of the residual stream per layer) applied to G1d 0.4B and World 0.4B makes the
comparison exact, and the per-layer weight deltas of P1 and `kl_T2_k4` say whether our training moved the
tunnel.

#### B. What a reader can bring out

**ZLK.12 A trained reader raises usable capacity.** On one checkpoint, reading through `M` ticks makes more of
what is held usable than one pass: more separable facts per head, and stored-not-read facts read, without any new
fact in the data. *Estimator:* separable facts per head at `M = 0` and `M = 4` on the same checkpoint
(`state_capacity_probe.py --ticks`); the transition stored-not-read -> read at `M = k` vs `M = 0` on held-out
facts; effective size. *Refuter:* ticks never read more separable facts than one pass on a trained checkpoint, and
the stored-not-read class does not shrink at `M = k` beyond the `M = 0` arm. *Bound:* `I_V <= I` (T1); the claim is
a rising readable share, never knowledge from nothing. *Rests on:* externally, a post-hoc state-readout corrector
gains +7 / +6 / +2 points at 1.5B / 2.9B / 7.2B, and 1.5B with it equals 2.9B without it (Scarletwolf, 2026-08):
a better reader pays most at the smallest size. Internally, own CoT reads facts one pass misses (ZLK.11); the
one-pass baseline is at least 4 separable facts per head on G1d 0.4B (L11-13: 1/1, 2/2, 4/4 at R2 1.00 / 0.98 /
0.89 with a ridge fit; the earlier "2" came from an erratic lstsq driver, 2026-11-12). *Measured (T15,
2026-11-12, `results/state_capacity_g1d04b_{base,klT2k4}_M{0,4}.cv_ridge.json`, L11-13, N = 1..32, same items
at `M = 0` and `M = 4`):* four ticks raise held-out R2 per bound value by about +0.005 at N = 16 and 32 (paired:
26/32 slots up on the base, 23/32 on `kl_T2_k4`), about 2% more read; the gain is the same on the untrained
base as on the S1 checkpoint. So the loop reads marginally more, and S1 training did not make it a better
reader of stored bindings. The first half of the refuter does not fire as written; the claim is not supported
either. The stored-not-read class was not measured here.

**ZLK.13 Denser packing under a trained reader.** When storage and reading are trained together, the model holds
more items in superposition at equal readability than a model trained to be read in one pass. *Estimator:* after
the same training budget, with ticks (arm B) and at `M = 0` (arm C, §6): the number of facts of a fixed battery in
class read at the arm's own `M`; separable facts per head with ticks minus without, compared across the two arms.
*Refuter:* the arm trained with ticks reads no more facts at its `M` than the `M = 0` arm reads at 0, or the
ticks-minus-one-pass gap is the same in both arms. *Rests on:* the proved part T6 for the state (iterated delta
steps converge to the least-squares map where one pass leaves crosstalk) and compressed sensing as an analogy (an
iterative decoder recovers sparse signals from fewer measurements than one linear read separates); nothing
measured. *My reading:* plausible as mathematics; the weights side is untested, and an arm C is required, since
any distillation also raises what one pass reads.

**ZLK.14 Derived relations are usable knowledge the loop can add.** A trained loop answers relation items whose
component facts the model reads at `M = 0` but whose relation it does not, and this gain transfers to relations
never seen in training. *Estimator:* held-out relation items (two-hop, shared city, comparisons) built only from
facts in class read; accuracy at `M` vs `M = 0` of the same checkpoint vs arm C. *Refuter:* the `M = 0` arm
reaches the same accuracy (then it is a one-pass skill, T3), or the gain is only on trained relations. *Rests on:*
`H12.md` §4 (a state that holds a relation answers queries about bindings it never stored). Qwen two hops distil
into one pass when the facts are held (control 76 of 120 mixed); the 0.4B, which does not hold the capitals,
stays at 0-2 of 120. *Reading:* the owner calls this new understanding; I agree it is new usable knowledge (I_V of
the relation), and it is not new information about the world (T1, T2). The paper must say which "new" it means.

#### C. The channel

**ZLK.4 The own-latent channel can be trained.** After S1, accuracy through the ticks is not below the untrained
model at `M = 0` on held-out cells, and ticks decode to prompt-specific content. *Estimator:* free answers at `M`
and `M = 0`, content and form apart; `chat`; `latent_geometry`. *Controls:* a constant feed must stay at its fixed
point; a feed swapped between examples must break the answer; teacher content shuffled. *Refuter:* frontmatter
item 7. *Rests on:* 0.4B: one tick no longer erases (held 0.11 -> 0.56-0.89); every trained checkpoint a
pass-through (44 / 46 of 304; 75 / 81 of 289; Qwen 197 / 204 of 289); KL T=2 keeps the tick path fluent where CE
narrows it.

**ZLK.18 Trained ticks widen the attractor.** After training, ticks keep a point per prompt (never one point for
all), settle later than tick 1 on deep items, and the output grows more expressive at equal content. *Estimator:*
channel spread over ticks and cross-prompt cosine on `S`; settle tick; dMI of the output (distinct words per
answer, temperature split of sharpening vs new support, a shuffled-key control); disclaimers and loops in chat;
the volume of thesis 14: effective rank of the Jacobian of the state with respect to the model's own latent
choices (noise in the k~16 subspace), taken on the transient before the halt (at the halt it should contract),
counted only where those rollouts keep accuracy >= `M = 0` - 0.1; the program form of the end (toy, 2026-11-21,
`attractor_probe.py`): past the last step the latent holds the answer (0.89-0.99) and not the next node (chance),
its per-tick move drops 2-6x, the end falls into a per-answer region shared across tables (nearest answer centroid
0.88-0.94), and the own stop coincides with that settling within one tick on 0.69 / 0.75 of examples.
*Refuter:* collapse to one point for all prompts (cross-prompt cosine on `S` >= 0.95 for two consecutive gated
checkpoints), or expressiveness rising only with loops or disclaimers. *Rests on:* `kl_T2_k4` (0.4B): effective
rank over ticks 3.2 -> 7.9, a point per prompt; chat distinct words 48.5 vs base 42.3 (`M = 0`) / 44.8 (base
`M = 4`), n = 6 questions, no disclaimer; but its latent sits at a per-prompt fixed point from tick 1. G1k and G1l
collapse to one point for all prompts by tick 6-8 (read on ticks they were never trained for, so no weight here;
dropped from the paper 2026-11-21); G1i spreads (0.88); at 2.9B a shared component holds PC1 at
0.999 already at tick 0 in all three, so the per-file breakdown has to confirm "one point". The behavioural side,
measured by the community on G1l 7B preview (No-22, 568-question agent exam, 22 720 passes, shared by BlinkDL
2026-10-24): the dominant failure is inertia, 53% of questions fall into repeat, rejection, forced close; 56% of
repeats copy the immediately preceding step, usually a call that had succeeded; prompt, nudge and sampling
changes cut it by at most about 20%, and the author concludes only data fixes it. The owner's note that reflection needs a
refusal to converge is this claim's settle-tick half: by the tick lemma a tick at fixed controls settles at
`ρ(A)`, so ticks that keep reading must keep changing the controls.

**ZLK.10 The exact past beside the settling state.** A QKV-ROSA branch queried at every tick lets the loop recall
a fact from an earlier turn after `M` ticks as well as at `M = 0`, without a common attractor across turns.
*Estimator:* multi-turn recall at `M` vs `M = 0`; cross-turn cosine. *Refuter:* recall after ticks below `M = 0`
with ROSA trained; or ROSA read copies the previous turn instead of the needed one. *Rests on:* nothing; no trained
ROSA exists here (`rosa_soft` not installed). States install dispositions, not procedures: multi-turn recall 0 of 2
on every model measured externally. With ROSA a tick can re-read exact history: the write-side counterpart of
reading (T6), which the retired axis N approximated by re-feeding the prompt.

#### D. Training

**ZLK.15 Knowledge through a fixed teacher on the student's own tokens.** On-policy distillation (reverse KL on
tokens the student sampled through its ticks) from a frozen teacher that holds the fact, in its context or its
weights, lifts the trained facts and the held-out ones, with read -> lost under 5% of read items and chat intact.
A target that is a foreign string, or a teacher that moves with the student, breaks the rest. *Estimator:* the
transition table on trained and held-out facts, split by fact; chat; fresh-text bpb at `M = 0` not worse than the
base by more than 0.01. *Refuter:* read -> lost outnumbers stored-not-read -> read on held-out facts (H7's
refuter), or the on-policy arm loses more than an off-policy arm on the same data. *Rests on:* Qwen 0.8B,
frozen-teacher on-policy: trained countries 6 -> 16 of 28, held-out 11 -> 13 of 21, form and chat kept, bpb +0.003;
the self-teacher run collapsed into a placeholder ("Answer: <answer in capital letters>"); gold-string CE on a
decode task learned the English payload prior (0 of 60); a hint teacher's answers taught "<answer>" tags. *Reading:*
the owner's "only through RL" is this mechanism: an on-policy dense per-token signal from a teacher that has the
fact, not a 1-bit reward (T2).

**ZLK.19 Separability comes from tasks that need it.** A stage whose items can only be answered by keeping several
bindings apart raises separable facts per head and spreads the answer over fewer shared directions; an answer loss
on items that do not need it sharpens the readout instead. *Estimator:* the pair separable facts per head +
ablation-read directions per answer, before and after (fewer directions with more separable facts = taking apart;
fewer with fewer = sharpening). *Refuter:* the task-designed stage also sharpens. *Rests on:* P1 staged recall:
recall 0.72 -> 0.96 while ablation-read directions fell 40-60% and separable facts 7/32 -> 3/32 (sharpening);
`kl_T2_k4` held or raised ablation-read at L8-L20 (no P1 dissociation). The level-3 context rows ("which two share
a city") are the first items built so that separability is load-bearing; the 0.4B solves 0 of 160 of them.

**ZLK.16 Early loop.** The same chain run from a plain pretrain (World 0.4B) yields a wider channel and a larger
tick gain than when grafted on a finished model (G1d 0.4B), at equal data and steps. *Estimator:* arms B and D of
§6: `G_M` and tick-only items at `M = 4` vs `M = 0` on deep held-out items; channel spread; ZLK.12 separable facts.
*Refuter:* D >= B on tick gain and on effective rank. *Rests on:* nothing measured with ticks. World routes far less
through its state than G1 at 2.9B (KL under state noise 0.0055 vs 0.23) and 2-3x less at 0.4B, while holding the
prompt's identity more tightly (`H9.md`); G1k / G1l collapse to one attractor, a settled read path.

**ZLK.17 Two training methods from one base.** World 0.4B trained by this chain is not below G1d 0.4B with our best
tuning (`kl_T2_k4`) on deep held-out items and on the fact classes of a common battery, with fresh-text bpb at
`M = 0` within 0.01 of G1d's. *Estimator:* arms A and B of §6 on the same battery. *Refuter:* A beats B with
p < 0.05 on both the deep items and the fact classes. *Rests on:* nothing. G1d shares World's trained weights
(lineage cosine 0.743), so what each method moved is readable matrix by matrix. *Reading:* G1's data budget is far
larger than ours, so "not below" is the claim, not "above"; thresholds are the owner's to set.

#### E. Self-regulation and the interface

**ZLK.5 Self-chosen time.** A learned stop reaches the best fixed `M` per family with <= 85% of its ticks on a
mixed-family set, and moves with family. *Controls:* random stop at equal mean `M`; best fixed `M`; price sweep.
*Refuter:* frontmatter item 4. *Rests on:* toy only (labels first, then reward; lr 3e-3 collapses into stop-early).
Huginn learns backtracking from variable recurrence depth alone (arXiv 2602.08100): a reason to train at random `M`.
*Measured 11-20 (toy, `sL_ramp1_seed{4,5}`, plan 1l):* no stop head; the answer head is read after each tick and
the loop stops at the first confident tick; trained by imitating a frozen snapshot at every tick, weight (t/8)^1.
Own stop 0.905 / 0.912 equals fixed M8 on the same model, within 0.03 of the start (0.923 / 0.936), at 64% / 62%
of the ticks; a random stop at the same ticks 0.60 / 0.58; ticks used rise with the hop count (1.2-1.6 to 8.0).

**ZLK.6 Beyond the teacher.** At <= 60% of the teacher's tokens, accuracy >= teacher + 3 points on at least one
family, none worse than -3. Bounded by T1/T2: beyond the teacher's one-pass reading, never beyond what it holds.
*Rests on:* the room estimate of ZLK.1 only.

**ZLK.7 Views without collapse.** A representation term over two views of one content (stop-gradient target,
contrast across examples) raises the similarity of post-question states across views without collapsing the spread
across examples, and raises accuracy on held-out formats. *Refuter:* collapse, or no gain over the answer loss.
*Rests on:* format changes the content of the answer in 32 of 48 grids (G1i; paraphrase floor not measured).

**ZLK.8 Content first, format from the question.** Format is read from the question; self-decisions are trained
with a proper scoring rule. *Rests on:* JSON and sentence formats obeyed 100%, a bare number 0% (G1i, silent);
nanodiff: same accuracy, ECE 0.333 -> 0.036.

**ZLK.9 Absorption.** After the configuration is fed once, answers that depend on it are as accurate as with the
history re-injected, at a fraction of the tokens read. *Refuter:* H17's. *Rests on:* nothing; the owner's sessions
re-read every input token about 79 times (98.8% of input tokens were cache reads), which motivates it.

**ZLK.22 One system across domains.** A loop trained on a mix of domains without domain D solves D better than
the same model trained by CE on the same mix, and below a model specialised on D; its accuracy falls more slowly with
distance from the training mix, and the tick latents of different domains share directions. *Estimator:* held-out
family accuracy (e.g. 5 of 25 cipher families, after the cipher control shows the cipher is read) for loop, CE-mix
and specialist; the slope of accuracy over distance; cross-domain similarity of tick latents before and after.
*Refuter:* no gain on the held-out family over CE-mix beyond seed spread, or gains only on families in training.
*Rests on:* the owner's question 11-19; outside: RL transfers to unseen variants better than imitation (Chu et al.
2025), representations converge across models (Huh et al. 2024), both without a latent loop. Toy so far: one step
generalises within a task (1b) and did not carry from context to stored facts (cs). *My reading:* plausible because
shared tick weights make one general step shorter than one path per domain; the real test is on World 0.4B.

### 5. Positions still argued

- **What A0.5 says about World.** The owner reads it as World's WKV doing no computation. It measures how much the
  next token depends on the state, not computation. At 0.4B World depends 2-3x less than G1d; at 2.9B 40x less,
  while keeping the prompt's identity more tightly. On the 2.9B pair his reading holds in the form "a plain pretrain
  routes little through its state"; whether its state computes is ZLK.16's question.
- **Self-distillation of World.** Agreed: World's own answers carry nothing to re-read (T1), so its teacher is
  foreign and frozen.
- **A trained initial state teaches form, adds no facts.** Mine; consistent with the external measurement that a
  LoRA recalls injected identity facts about 100% and state tuning about 30% (RWKV-LM #338).
- **Weight edits remove an obstacle, they do not add an ability.** Mine (Sparks of Weight-Space Programming,
  2026-11-02, not re-read since).
- **"New" in ZLK.14.** See the claim; both readings are kept and the paper separates them.
- **S2 teaches knowledge.** The earlier text said S2 and later optimise a policy only. The owner's word of
  2026-11-12 puts knowledge widening into S2; the reading that keeps both true is ZLK.15: knowledge enters by
  on-policy distillation from a teacher that has it, and from S3 on the stages are policy.

### 6. Training chain

Common: full fine-tuning, no LoRA; Muon lr 3e-5 to 1e-4, aux AdamW on its own lr; the teacher is fixed (a cache or
a frozen copy, never the weights being trained); targets are teacher distributions on the student's own tokens;
splits by fact; every run names its result path, dataset names + SHA-256 and a written prediction first; at least
three data orders per claim; a checkpoint per eval, none deleted; every set has a no-prior control (random payload,
shuffled key); every run is read on its output (chat, samples), not only on accuracy.

Arms for the comparison of ZLK.12-ZLK.17:

| arm | start | trained by | role |
|---|---|---|---|
| A | G1d 0.4B | `kl_T2_k4` (KL T=2 to its own `M = 0`, `M = 4`, step 100) | our best on the G1 side |
| B | World 0.4B | W0, then S1-S2 with ticks (`M = 4` until S3) | the claim |
| C | World 0.4B | the same data and steps at `M = 0` | separates the teacher from the ticks |
| D | `kl_T2_k4` | S1-S2 of the same chain | the same chain on a finished model |
| refs | World, G1d, the teacher at `M = 0` | none | floor and ceiling |

| stage | what is trained | signal | gate | tests | state |
|---|---|---|---|---|---|
| S0 data | the items, sorted by fact class on the student and on the teacher | - | every family has items the teacher solves and the student does not; the level-3 rows and k-hop chains have a teacher | ZLK.11 | plain and cipher rungs measured; level-3 rows: 0.4B 0 of 160; teacher pass and k-hop generator not run |
| W0 form (a plain base only) | answer form for a base that cannot answer: off-policy KD on the teacher's sampled answers (full-vocabulary KL, T=2), then a growing on-policy share | KL to the frozen teacher | chat fluent EN/RU, no loops, format obedience per format, fresh-text bpb at `M = 0` within 0.01 of the base | - | not run; the one place where the teacher's token string enters the context (rule 3 of the gate), until the on-policy share takes over |
| S1 channel | the answer through `M` own-latent ticks, on-policy reverse KL to the frozen teacher (own `M = 0` for a base that answers, a bigger teacher otherwise), anchors on items the student reads | KL | the tick does not erase (free answers at `M` >= `M = 0` base); same checkpoint at `M = 0` vs `M` on deep held-out items; ticks prompt-specific; chat fluent; channel gate on `S` | ZLK.4, ZLK.18 | 0.4B and Qwen pilots: pass-through (§6b) |
| S2 knowledge widening | facts and relations the teacher holds: knowledge text (Wikipedia vital intros, extendable) with the teacher's per-token distribution as target and ticks at random boundaries with random `M` in 0..4; the fact battery and relation items on-policy, the teacher reading the fact in context or from its weights; saturated, contradictory sequences as a share of the rows | KL, on-policy | transition table on held-out facts (stored-not-read -> read up, read -> lost < 5%); `G_M` on derived relations vs arm C; bpb on seen text at `M = k` below `M = 0`; fresh-text bpb at `M = 0` within 0.01 | ZLK.12-ZLK.16 | not run |
| S3 time and policy | continue = an unreadable latent tick in the answer softmax, speak = a vocabulary token; noise in a k~16 subspace of the latent inside log pi; verified answer or continuation-gain reward minus a price per tick | policy gradient | learned stop reaches the best fixed `M` with <= 85% of its ticks; random stop worse; the noise window opens (rollouts disagree at acc >= `M = 0` - 0.1); settle tick later than 1 with distinct ticks | ZLK.5, ZLK.6, ZLK.18 | toy only |
| S4 format and honesty | required format read from the question; honest self-decisions; the constitution later | CE / proper scoring rule | obedience per format; ECE | ZLK.8 | obedience measured |
| S5 ROSA | a QKV-ROSA branch beside WKV (output zero-initialised, Q/K/V trained), queried at every tick | S4 loss + recall of earlier turns | ZLK.10 | ZLK.10 | not started |

Order is strict; a stage opens when the previous gate is measured. Every gate is one `experiments/run.py --backend
peft [--weights CKPT] --tests m0_pass,chat,latent_geometry,latent_noise` run plus the stage's own number;
`state_capacity_probe --ticks` and `state_readout_lens` read the neck as a pair. M>1 is opened by the decision
rule of the tick-yield spec (`~/.claude/plans/noesis-tick-yield-probe-2026-10-26.md`), not by accuracy alone.

#### 6b. Record of the runs

| run (date) | model | data and loss | result | file |
|---|---|---|---|---|
| P1 staged recall (10-02) | G1d 0.4B | key-value pairs over a gap, CE, Muon 1e-4, 100 steps | recall 0.72 -> 0.96; ablation-read -40-60%; separable 7/32 -> 3/32: sharpening | `results/staged_recall_p1_step100_final.json` |
| S1 pilot (10-27) | G1d 0.4B | 45 items it solves, CE / KL T=1 / KL T=2, `M` = 1, 200 steps | tick no longer erases (held 0.11 -> 0.56-0.89); CE `M = 1` 75 and `M = 0` 73 with the same 30 new: pass-through; KL T=2 tick path fluent | `results/s1_pilot_g1d04b/` |
| S1 `M = 4` (10-28) | G1d 0.4B | KL T=2, `M = 4`, step 100 (`kl_T2_k4`) | 44 at `M = 4` vs 46 at `M = 0` of 304; latent fixed from tick 1; flat over `M` = 1..8; lens held | same |
| hint distillation (10-28) | `kl_T2_k4` | base 0.4B reading hint 3 as teacher, 211 hint-only items | rescored: base 46 -> 75 (`M = 4`) / 81 (`M = 0`) of 289; teacher answers carried "<answer>" tags (77 of 211), loops 16 -> 81 | `results/s1_pilot_g1d04b/hint3_*` |
| latent noise (10-28) | 0.4B | k = 16, sigma 0.5-2 | after S1 the ticks damp noise (flips 0.45 -> 0); window unreadable on H22 | `results/s1_pilot_g1d04b/` |
| Qwen hint-3 (10-29) | Qwen3.5 0.8B | hint-3 teacher, KL T=2, ticks vs `M = 0` control | control 195, ticks `M = 4` 119 / `M = 0` 105 of 289; tied head spells the answer in ticks; think frame: 197 / 204 / 195 | `results/qwen35_08b/` |
| Qwen ciphers (10-29/30) | Qwen3.5 0.8B | cipher rungs, own CoT, decode SFT on gold | 0 of 585, 0 of 240, CoT 0 of 24; decode SFT 0 of 60 (payload prior) | `results/qwen35_08b/` |
| Qwen self-KL (11-04) | Qwen3.5 0.8B | KL T=2 to own `M = 0`, think frame | chat = base; stored-not-read not repaired (3 new, 8 lost) | `results/qwen35_08b/` |
| Qwen on-policy (11-05) | Qwen3.5 0.8B | self-teacher, then frozen teacher, anchors | self-teacher collapsed; frozen: trained 6 -> 16 of 28, held-out 11 -> 13 of 21, form kept | `results/qwen35_08b/s1_onpolF_*` |
| compression (11-05) | Qwen3.5 0.8B | UncheatableEval 2026-07, ticks every 64 tokens | ticks in raw text cost ~0.08 bpb on every checkpoint; frozen arm +0.003 at `M = 0` | `results/qwen35_08b/compress_*` |
| RWKV chain (11-07) | G1d 0.4B | level-3 context rows, S0 and geometry | 0 of 160 eval, 2 of 400 train; plain context 158 of 600; stopped after 0.4B geometry, before the G1L teacher | `results/rwkv_2026_11_07/` |
| cipher ladder (11-13) | G1d 0.4B | key given; Caesar / katakana / cyrillic x words / random x 6-24 letters x 0-75% of the plaintext shown; greedy `M = 0` | 0 of 576; no hidden letter right in any cell: the answer copies the ciphertext or echoes the shown prefix | `results/ladder_g1d04b_M0/` |
| cipher lookup (11-13) | G1d 0.4B | same rows (none shown), gold teacher-forced; key as plain->cipher (as generated) vs cipher->plain; answer as plaintext vs "symbol=letter" pairs | pairs, first occurrence of a symbol, p(letter) > 0.5: cipher->plain 0.57 Caesar, 0.76-0.81 scripts; plain->cipher 0.002-0.004 (repeats 0.3-0.57 are copies of its own earlier pairs). First pair of the answer ~0 in both: lookup starts after 1-3 worked pairs. Plaintext format 0.01-0.27 per token | `results/cipher_lookup_g1d04b/` |
| ticks vs inverse lookup (11-13) | G1d 0.4B, `kl_T2_k4` | same probe through one embeds route for every `M`; `M = 0` vs `M = 4`; key before or after the message. Refuter written before the run: ticks open the inverse lookup unless `M = 4` stays <= 0.01 on both models in all views | refuted: plain->cipher stays 0.000-0.009 at `M = 4` on both; cipher->plain control 0.45-0.82, `M = 4` within +-0.05 except Caesar +0.05-0.08; message first does not help. A tick re-reads the state with the same reader and does not transpose an association. Key repeated after the message (read-twice): inverse still <= 0.009, forward +0.03-0.15 | `results/cipher_lookup_g1d04b_{base,klT2k4}_M{0,4}/`, `..._base_twice_M{0,4}/` |
| cipher ladder v2 (11-13) | G1d 0.4B | cipher->plain key, answer as "symbol=letter" pairs, 4 / 2 / 1 / 0 pairs of the answer written for the model; greedy `M = 0` | 1 of 576. Pair form kept; letter right for its own symbol 0.67-0.96, next message symbol right 0.02-0.04: the answer walks the key table in order. The missing step is the position in the message (symbols repeat) | `results/ladder_trace_g1d04b_M0/` |
| cipher ladder v3 (11-13) | G1d 0.4B | v2 with positions on the message and the answer ("1:ハ" / "1:ハ=T") | 28 of 576; 6 letters with pairs shown 1-6 of 16 per cell, 6-13 of 16 if the answer had stopped at the last position (most misses run past the end). Per pair at 6 letters: position 0.97-1.0, symbol 0.8-0.98, letter 0.5-0.89; 24 letters: symbol 0.34-0.76. No pair shown: the answer never starts the pairs (0 everywhere). First zone with solve rate between 0 and 1 | `results/ladder_indexed_g1d04b_M0/` |
| training with ticks vs inverse lookup (11-13; order 0 trained, read pending) | G1d 0.4B | v3 rows with the key plain->cipher ("A=ク", `--ladder-table plain_cipher`), whole answers (no pairs shown), seed 11, 1607 rows (every 20th held: 1526 / 81); CE on the gold pairs, full weights, Muon 1e-4, 300 steps x 4 rows, CPU WKV chunked (logits within 0.01 of the per-token reference, so the base is read again on the same code); arm A trained at `M = 0`, arm B at `M = 4`, same rows and steps, three data orders (seeds 0-2) per arm; the rule reads the mean over orders and needs the same sign in all three. Read: `cipher_lookup --cl-formats row` on the 96 rung-0 rows of a separate set (seed 7), first_p50, each arm at its own `M` and at the other; base at both. Refuter written before the run, for the claim "training with ticks opens the inverse lookup beyond the same training without them": confirmed if B at `M = 4` minus A at `M = 0` >= +0.10 overall and in 2 of 3 ciphers; refuted if <= +0.03 overall; between: a second seed; both arms <= 0.05: the budget decides nothing. My prediction: A opens it (the reverse pair can be written while "A=ク" is read), B about equal | Order 0, in-training eval on the 81 held rows (exact answer): A at `M = 0` 0 / 0 / 0.63 / 0.98 / 1.0 at steps 0 / 50 / 100 / 150 / 200; B 0 / 0 / 0.65 / 0.94 / 0.98, the same at `M = 0` and `M = 4` at every step. B stopped at step 215 (the machine froze 11-14); both arms keep a checkpoint at step 200 on the same order, so the read compares them there. The inverse lookup is learned without ticks from ~400-600 rows and holds on new keys; B trained through ticks and reads equally without them. Blind spot of the rule, found after training and before the read: the read set has the training cells (6 and 12 letters, rung 0), A sits at the ceiling, so "B - A >= +0.10" cannot happen; the read can refute only "ticks help where one pass already suffices" | `~/tmp/inv/{A_M0,B_M4}_s0*` (logs, checkpoints); reads -> `results/train_inv_g1d04b_*_read_M*/` |
| closed-loop toy from scratch, stage 1 (11-15) | RWKV-7, 2 layers, width 48, 49.8K params, trained from scratch | pointer chasing in context over a 10-node cycle, k = 1..8 hops, query last; curriculum k_max 1 -> 8 over 4500 of 6000 steps; CE on the answer only; A at `M = 0`, B-E at `M = 8` from step 0 (C per-tick JEPA target, D SIGReg 0.05, E both); seeds 0-2 | first run on record where ticks are not a pass-through: B 0.97 / 0.93 / 0.87 at k = 1 / 2 / 3 against A 0.78 / 0.16 / 0.13; B read at `M = 0` gives 0.00 at k = 1 (the computation lives in the ticks). k >= 4 at chance in both (B stalls at 3 hops though 8 ticks x 2 layers would allow more). Refuter 1 ("ticks buy depth", max(B..E) - A at k* + 2 = 3 >= +0.20, same sign in 3 seeds): +0.74 / +0.74 / +0.78, CONFIRMED. B is stable and stalls at 3 hops in all seeds (k = 3 0.85-0.93, k = 4 0.13-0.21). The per-tick terms break the wall but bimodally: C seeds 1-2 reach 0.92 / 0.88 at k = 4 and 0.83 / 0.72 at k = 8 / 7, C seed 0 only 0.35 at k = 3; D and E 1 of 3 seeds past the wall (D s0 0.55 at k = 7, E s0 0.50 at k = 8), the other seeds never pass k = 1 (D s2 at chance even at k = 1). C seeds 3-6 (run by the spatium session): k = 6 0.70 / 0.92 / 0.91 / 0.83, so 6 of 7 C seeds pass k = 6 >= 0.5 (seed 0 the only failure: its JEPA loss never falls, 0.46 -> 0.52, while seed 1's falls 0.50 -> 0.12). Refuter 2 (C - B >= +0.10 at k > k* in all seeds, shuffle costs C >= 2x B): mean over k = 2..8 -0.10 / +0.51 / +0.38 against B; shuffle cost (tick input = another example's latent, own state kept) over k = 2..8 is B 0.26 / 0.25 / 0.27, C 0.20 / 0.80 / 0.66 (seeds 4-6 0.83 / 0.82 / 0.80): 2x holds in every C seed but 0. Not confirmed by the letter (seed 0), not refuted. Reading: at k = 2..3, where both solve, the shuffle costs B 0.78-0.82 and C 0.81-0.87, so the per-example tick latent carries the computation in both arms (a register); the per-tick target does not create the register, it makes the step iterable past 3 hops. The 2x clause is confounded by B's wall. Bits per parameter (quant_sweep.py, per-row absmax rounding, `quant_s1.json`): at k = 1..3 B and C tolerate the same rounding (5 bits lossless, 4 bits -0.03 to -0.20, 3 bits about half lost); at k = 8 C loses more at 4 bits (0.82 -> 0.64, 0.89 -> 0.30), the rounding error accumulates over the ticks; the strict per-k minimum is B 5, C 5-8. The iterated step costs no more weight precision per step than the unrolled one, long chains need about one bit more. State probe (toy_state_probe.py, `state_probe_s1.json`; linear decode of every pair x -> next(x) from the WKV state after the pairs, held out): untrained 0.53, A 0.73 / 0.85 / 0.86, B 0.91 / 0.93 / 0.93, C 0.73 (failed seed 0) / 0.81 / 0.90 and seeds 4-6 0.96 / 0.96 / 0.95. Gate written before the run (tick arms >= +0.10 over M = 0 means the loop changed writing): B +0.11, C on its passing seeds +0.10, seeds 4-6 +0.13. The loop changed what is written, not only how it is read. Effective rank of the WKV heads (16 x 16, 10 pairs): untrained ~7-8, A 6-8 in layer 0, C seeds 4-6 3.5-4.6 in layer 0 with the best decode: the pairs are held more decodably in fewer dominant directions (ZLK.13 in miniature). After 8 ticks the rank stays 3.5-6.8, no collapse toward 1 as in the G1d attractor. Refuter 3 (SIGReg keeps the channel free without losing > 0.05): D loses > 0.05 at k <= 3 in 2 of 3 seeds, so at weight 0.05 it cannot pass; geometry not measured | `experiments/A0_state_probe/results/closed_loop_toy/s1_*` |
| closed-loop toy, stage 1b: affine cipher seen through runs (11-16) | same toy, C recipe (M = 8, per-tick target), A (M = 0), and a chase + affine mix | 4 runs x -> a x + b mod 16 in context, query not among them; seeds 0-1 of 3 | not learned: k = 1 0.27-0.29 (C) and 0.35-0.48 (A), above chance 0.06 but far from the rule (parity of the runs explains part). A design flaw: odd a mod 16 has order <= 4, so E^4 is the identity in 64 of 128 ciphers and E^8 in 96; A's 1.00 at k = 8 and 0.75 at k = 4 are that shortcut, not depth. In the mix the affine half stays at chance and drags the chase half (seed 0 walls at 3 hops). Refuter "the loop applies a rule inferred from runs" not met (k = 3 at M = 8 0.25-0.28 < 0.5): inferring a and b from runs is out of reach at 50K params and 6000 steps; the iterated depth was not tested because of the shortcut (seed 2 the same: k = 1 0.25-0.33). Shift rung (11-16, E(x) = x + b mod 13, b != 0, no power shortcut): the rule is inferred from the runs perfectly, C at M = 8 1.00 at every k = 1..8 in all three seeds (M = 0 read of it 0.00-0.40); A at M = 0 1.00 up to k = 5, then 0.97 / 0.52 / 0.97 at k = 6 and 0.52 / 0.12 / 0.72 at k = 8. Refuter as written not met on its M = 0 side (A 1.00 at k = 3): a shift has a closed form, x + k b, which one pass computes without iterating; the ticks win only where the closed form's arithmetic fails (k >= 6). Depth tasks for the loop need no closed form (tables, derivations) | `results/closed_loop_toy/s1b_*` |
| closed-loop toy, stage 1f: time against parameters (11-17) | one-pass A (M = 0) with 4 / 8 layers vs C (2 layers, 8 ticks, 49.8K) | chase, curriculum, 6000 steps; seeds 0-2 | A with 4 layers (94.7K params, 1.9x), seed 0: 1.00 / 0.99 / 0.98 / 0.75 / 0.36 at k = 1..5, 0.22-0.26 at k = 6-8: depth per pass grows about one hop per layer (2 layers: 1 hop). Seeds 1 / 2: k = 3 0.83 / 0.14, k >= 5 0.12-0.18. Three seeds of the 4-layer pass: 2-4 hops, chance at k = 6-8 (0.12-0.26). C with half the parameters holds 0.72-0.92 at k = 6-8 on its passing seeds (6 of 7), +0.5-0.7 over the 4-layer pass. 8 layers (184K, 3.7x): k = 6-8 0.16-0.28 on all 3 seeds: more layers do not buy the depth time gives | `results/closed_loop_toy/s1f_*` |
| closed-loop toy, phase A: stabilising the recipe (11-17) | chase C recipe (8 ticks, per-tick node target, curriculum), one knob at a time, 5 seeds | lr 1e-3 (base 3e-3) on the laptop; batch 256 (base 128) on the Aspire; gate: k = 6 >= 0.5 in 5 of 5 seeds | k = 6 >= 0.5 by knob: base (lr 3e-3, batch 128) 6 of 7; lr 1e-3 1 of 5 (no seed collapses, k = 1 >= 0.81, but 6000 steps undertrain: 0.59 / 0.47 / 0.47 / 0.40 / 0.33); batch 256 3 of 4 (seed 0, the base's failure, now 0.95 at k = 6 and 0.96 at k = 8; seed 2 fails instead). No knob gives 5 of 5: the failure is a bad basin that moves between seeds. It is visible early: the per-tick loss at step 1500 is 0.43-0.52 in every failing or weak run (base s0, b256 s2, lr1e-3 s4) and 0.20-0.38 in every passing one, and the failing ones never fall after. A stall rule (per-tick loss > 0.40 at step 1500 -> restart with the next seed, report seeds tried) would have kept only passing runs here; it was a retrospective fit on 8 runs. Prospective test A3 (base recipe, seeds 10-14 so far): all 5 pass k = 6 (0.81-0.99), but seeds 10 and 12 had loss 0.43 / 0.49 at step 1500 and recovered: the rule is wrong on 2 of 5, its refuter (>= 7 of 8 right) cannot be met: REFUTED. The base recipe itself now passes 11 of 12 seeds. A3 seeds 15-17 pass too (k = 6 0.84-0.90): base 14 of 15. A2, lr 1e-3 x 12000 steps: 4 of 5 (seed 4 fails, 0.13): the slow rate was undertraining, not a fix for the bad basin | `results/closed_loop_toy/sA_*` |
| closed-loop toy, stage 1e: spatium rewrite derivations (11-17) | same toy, vocab +30 symbols; CE vs CE + owner's ZLK term (tick t -> EMA latent of hop t's term) | 33,985 items (29,987 train), 8 premises #1..#8 in context, ~140 tokens, depth 1..8, true/false 50/50 with near-miss falses; 6000 steps x 64, curriculum; seed 0 | On val_laws (held-out laws), M = 8: true items 0.90-0.98, false items 0.84-0.96 at every depth 1-8, both branches (exact BFS 0.99-1.00 at 55-4,347 terms expanded per item; the model spends 8 ticks). Mechanism (deriv_breakdown.py): falses are partly a shortcut, M = 0 already 0.72-0.82 on them and shuffled ticks keep 0.79-0.97; trues need the ticks, M = 0 0.61-0.74 (CE) / 0.26-0.42 (ZLK) and shuffled ticks 0.45-0.73. ZLK refuter (ZLK - CE at depth 4-8 on val_laws >= +0.10): +0.004, refuted on seed 0, but at a 0.91 ceiling, so it decides little; ZLK moves more of the work into the ticks (its M = 0 trues 0.26-0.42 vs CE 0.61-0.74). Seed 1 replicates (depth 4-8 balanced at M = 8: CE 0.913, ZLK 0.938; seed 0: 0.914, 0.918): Seed 2: CE 0.951, ZLK 0.933. ZLK - CE on the truth answer +0.004 / +0.025 / -0.018 over 3 seeds: refuted by the letter, at a ceiling. last_rule variant (answer = local id of the last hop's rule, true items only, chance 0.125), CE seed 0 final: val_laws M = 8 0.20-0.41 (depth 4-8 mean 0.27), M = 0 0.14-0.27, shuffled 0.17-0.29; val (seen laws) 0.19-0.40. The CE model names the last rule at about twice chance, far below its 0.91 truth accuracy: the truth answer rests mostly on a consistency check, not on a derivation it can report. ZLK last_rule seed 0: val_laws M = 8 0.37-0.50 (depth 4-8 mean 0.448 vs CE 0.270), M = 0 0.07-0.17, shuffled 0.23-0.36; val 0.41-0.57. ZLK refuter (ZLK - CE at depth 4-8 on val_laws >= +0.10) on the shortcut-proof answer: +0.18, CONFIRMED on seed 0; but CE seed 1 reaches 0.48 at depth 4-8 (seed 0 was a weak CE seed), and ZLK seed 1 ends at 0.439: ZLK - CE = +0.18 / -0.04 on seeds 0 / 1, not confirmed by the letter. Seed 2: CE 0.434, ZLK 0.295. Over 3 seeds ZLK - CE = +0.18 / -0.04 / -0.14, means 0.394 vs 0.395: the ZLK term (hop-term latent target) is REFUTED on the last_rule answer as on the truth answer; the seed-0 gain was seed noise (CE spread 0.27-0.48). Both arms name the last rule on held-out laws at about 3x chance (0.39 vs 0.125), so part of a derivation is there, with large seed variance. Label-free goal target (every tick toward the EMA latent of the goal term): seeds 0 / 1 0.46 / 0.362, mean 0.411 vs CE 0.395 (+0.016, paired +0.19 / -0.12): refuted (<= +0.03). self (tick t toward the EMA copy's tick t+1): 0.28 / 0.242, mean 0.261, below CE by 0.13: refuted, and it hurts (a target that rewards reaching the later state sooner pulls toward a fixed point). both (goal + self): 0.285 / 0.246, mean 0.266: refuted, self drags it down. Anonymised CE seed 0: 0.287 (plain CE seed 0: 0.27); seeds 1-2 running. The owner's term (tick t predicts the latent of hop t's term) is what lets the model name the rule it applied; on the truth answer it was hidden by the surface cue. cs seed 1 repeats seed 0 (chase 0.84-0.98 to k = 8, storage one-hop 0.24). | `results/closed_loop_toy/s1e_*`, `~/tmp/toy/breakdown_s0.txt` |
| closed-loop toy, stage 1c: facts in the weights (11-16) | same toy, vocab +64 id tokens, 55,968 params | K fixed random 16-cycles named by two id tokens, no context; one-hop over all edges (capacity by Fano) and k-hop on held-out vs trained starts; 20000 steps x 256; seed 0 | One-pass capacity (A1, one-hop only): K 512 all edges (0.57 bit/param), K 1024 all edges, 64 kbit = 1.14 bit/param, K 2048 0.81 of edges = 1.46 bit/param (highest), K 4096 0.37 = 0.66 (under-trained at ~80 exposures per edge). Composition: A with the curriculum (M = 0) answers k = 2 on TRAINED starts 0.97 but on held-out starts 0.06 (chance): it memorises composites, the two-hop curse reproduced. C (8 ticks + per-hop target) at K 1024 stores less (one-hop 0.60, 0.41 bit/param: the curriculum dilutes one-hop exposure) and also composes only trained starts (k = 2 0.99 seen, 0.06 held out). With facts in the state the loop composes (stage 1); with facts in the weights it did not, at this K and schedule. Refuter not applicable (one-hop < 0.9). C at K 2048 (105 min) stores almost nothing: one-hop 0.16 at M = 8, 0.04 bit/param, k = 2 on trained starts 0.72: with the curriculum and the per-hop target from step 0, storage starves as K grows. A at K 512 with the curriculum stores EVERY edge (0.9994) and still composes only trained starts (k = 2: 1.00 seen, 0.06 held out): for one pass the two-hop curse is not a storage shortfall. C at K 512 (8 ticks + per-hop target) also stores every edge (one-hop 0.999 at M = 8) and composes trained starts (k = 2 1.00, k = 3 0.99) but not held-out ones (k = 2 0.07): refuter (ticks - M = 0 at k = 2 held out, one-hop >= 0.9) = +0.01, REFUTED on seed 0. A with a one-hop warm phase (Aw K 512): every edge, seen k = 2 1.00, held out 0.07. Reading: at K 512 the 8192 trained composites are memorisable, so the loop memorises them instead of iterating; in chase nothing repeats, so iteration is the only route. C with the one-hop warm phase (Cw K 512, 28000 steps) the same: every edge, seen k = 2 1.00, k = 3 0.90, held out 0.07. Storing first does not change it. cs (C, K 512, 20000 x 256, half chase half one-hop store, 276 min): chase perfect to k = 8 (0.98-1.00, M = 0 0.00 at k >= 2), but the stored facts barely land: one-hop 0.32 at M = 8 (0.06 bit/param) against 1.00 for A1 at K 1024 with ~300 exposures per edge here; k-hop over stored facts chance. Transfer untestable on seed 0 (one-hop < 0.9): the in-context task and storage in the weights compete in a 50K net. Control cs A (M = 0, 196 min): chase 0.90 / 0.84 at k = 1 / 2 then chance; one-hop storage 0.41 (0.10 bit/param): storage is starved by sharing the net with the context task with or without ticks, so the competition is between the tasks, not caused by the loop. | `results/closed_loop_toy/s1c_*` |

| closed-loop toy, 1e variants (11-17/18) | last_rule on derivations, depth 4-8 on val_laws at M = 8 | anonymised symbols CE; label-free tick targets goal (EMA latent of the goal term), self (EMA copy's next tick), both; seeds 0-2 / 0-1 | anon CE 0.287 / 0.237 / 0.287 (mean 0.270 vs 0.395 named); goal 0.460 / 0.362 (0.411), self 0.280 / 0.242 (0.261), both 0.285 / 0.246 (0.266) against CE 0.395: no gain on clean derivations (the regime where CE already carries the signal; their own regime is untested) | `results/closed_loop_toy/s1e_lr_*` |
| closed-loop toy, 1d ciphers from scratch (11-17/20) | width 96, 271K params; character-level decryption, key or table shown, 25 ciphers, HH-RLHF texts | M0, CE M4, + goal, + self (seed 0, 6000 steps); control: each answer paired with another row's prompt | char acc 0.605-0.608 in every arm, exact 0; control 0.590 vs 0.601 at step 4000: about one point read from the ciphertext, the rest is the English prior | `results/closed_loop_toy/s1d_*` |
| closed-loop toy, 1g distillation (11-19) | teacher: base-recipe C at M = 8 (seeds 14 / 10 / 15) | student on the teacher's final distribution only (KL T = 2): born-again 2 layers M = 8; one-pass mimic 2 and 8 layers | born-again k = 6 0.74 / 0.90 / 0.18: CONFIRMED (5-6 hops past B's 3); mimic L8 k = 6-8 <= 0.24 every seed (L2 1-2 hops, L8 2-4): the depth is not reachable by one pass even from the loop's soft answers | `results/closed_loop_toy/sG_*` |
| closed-loop toy, 1h width and 1i doubt (11-19/20) | chase C recipe at widths 12 / 24 / 96; doubt: K stored tables, 8 pairs in context, the queried pair sometimes a lie | 2 seeds per width; doubt K 64 / 256 at widths 24 / 48 / 96, M = 0, seed 0 | k = 6 >= 0.5: w12 0/2 (5.1K), w24 0/2 (15K), w48 6/7 (50K), w96 2/2 (178K). Doubt K 64: lie overruled 1.00, lie repeated 0.00, new tables from context 0.89-1.00, stored edges 0.95-0.99 | `results/closed_loop_toy/sH_*`, `sI_*` |
| closed-loop toy, 1k GRPO on the latent (11-19/20) | from arm B (3-hop wall) | exact-answer reward, Gaussian policy on each tick latent (sigma 0.1 / 0.3), G 8, KL 0.02 to the start, 3000 steps, 3 seeds (Aspire) | k = 4-8 0.12-0.20 in all 6 runs: REFUTED, an outcome reward with latent exploration does not break the wall; sigma 0 and the CE control stopped unread | `~/tmp/toyk/results/sK_*` (Aspire `noesis-head`) |
| closed-loop toy, 1l own stop without a head (11-19/20) | from C (fixed M8 0.92-0.94) | the answer head read after every tick, stop at the first max p >= 0.9; every tick's answer imitates a frozen snapshot's M8 answer, weight (t/8)^1, no label, no reward; 3000 steps | seeds 4 / 5: own 0.905 / 0.912 = fixed M8, start 0.923 / 0.936, random stop at the same ticks 0.60 / 0.58, tick share 0.64 / 0.62, ticks 1.2-1.6 at one hop to 8.0 at eight: ZLK.5 toy CONFIRMED. A self-target that moves with the model drifts (fixed 0.93 -> 0.74); GRPO beside it degraded too | `results/closed_loop_toy/sL_*` |
### 6c. The closed-loop toy as the bench (11-16)

One small RWKV-7 trained from scratch (`experiments/A0_state_probe/closed_loop_toy.py`, plan
`noesis-closed-loop-toy-2026-11-15`) tests the claims where the 0.4B cannot separate them. Claim -> stage -> estimator:

| claim | toy stage | estimator |
|---|---|---|
| thesis 13, depth from time | 1 (done) | accuracy by hop count at M = 8 vs M = 0, same parameters |
| ZLK.16, ZLK.19 | 1 (done) | M = 0 read of a tick-trained model; tick-latent shuffle cost |
| ZLK.13, ZLK.20 (write vs read; rank) | 1 probe | linear decode of every pair from the WKV state; effective rank after pairs, prompt, ticks |
| rule from runs (owner 11-16) | 1b | affine cipher seen only through runs, query unseen; M = 8 vs M = 0 |
| ZLK.11, ZLK.12, H25 bits per parameter | 1c | facts in the weights: capacity in bits per parameter at M = 0; k-hop use of the same facts with ticks |
| ZLK.5, thesis 14 | 2 | own stop; noise in log pi; ticks used vs best fixed M |
| deep derivation, the owner's ZLK term (RSC student) | 1e | spatium rewrite items, depth 1..8, held-out laws: CE vs CE + per-tick term latent (EMA target); cost vs exact BFS |
| §1c S, R_density, owner's rule | 3 | order reward vs CE only on unseen compositions |

Dropped from the toy line: rewarding motion (L_state, refuted), SIGReg at 0.05 (costs accuracy), weight rounding
as a capacity measure (it measures precision, not information; kept only as the cost of a long chain).

### 6d. Methodology from the toy: the recipe, why it works, what not to retry (11-17)

Settled on the closed-loop toy (2-layer RWKV-7, ~50K parameters, from scratch; numbers in §6b). Each item says what
was measured; the scale is the toy's, the transfer to trained models is §6d.5.

**6d.1 Recipe.** (R1) A tick feeds the last layer's normalised output back through `ln0` as the next input; weights
are shared across ticks; M fixed during training (8). (R2) Ticks from the first step: grown, not grafted. (R3)
Tasks one pass cannot solve (depth beyond what the layers give) and where nothing repeats (fresh context per
example). (R4) A curriculum on depth, k_max 1 -> 8 over 3/4 of the steps; uniform depth from the start drowns the
gradient. (R5) Where the intermediate step is exact and atomic (one node), a per-tick target: tick t predicts a
frozen embedding of step t. (R6) CE on the answer always.

**6d.2 Why it works (mechanism, each measured).** (W1) The tick latent becomes a register holding the current value:
feeding each tick another example's latent costs 0.78-0.87 at the depths solved, with or without R5. (W2) Without R5
the model learns an unrolled program of fixed depth (3 hops in every seed, then chance); with R5 the same weights
apply one step per tick (to 8 hops in 6 of 7 seeds): the per-step target turns a circuit into an iterable
instruction. (W3) The loop changes the write side: pairs written to the WKV state decode better (0.81 -> 0.92-0.96)
in fewer dominant directions; no collapse toward rank 1 over ticks (3.5-6.8). (W4) Time substitutes for depth: one
pass gains about one hop per layer (4 layers, 1.9x parameters: chance at 6-8 hops); 2 layers with 8 ticks hold
0.72-0.92 there. (W5) The iterated step costs the same weight precision per step as the unrolled one; long chains
need about one bit more (rounding error accumulates over ticks). (W6) On derivations over held-out laws the loop
names the rule applied at the last step at ~3x chance (0.39 vs 0.125), M = 0 at chance: partial derivation.

**6d.3 Limits found.** (L1) Facts stored in the weights are not composed on unseen starts, with or without ticks,
even when every edge is stored: trained composites are memorised (the two-hop curse holds under the loop). (L2) In
50K parameters a context task and storage in the weights compete (stored edge accuracy 0.16-0.41 vs 1.00 alone, 0.01-0.10 bit/param; the M = 0 control
starves too). (L3) Seed
instability: 1 of 7 C seeds never passes one hop; derivation CE ranges 0.27-0.48 across seeds.

**6d.4 Do not retry (each with the reason it failed).** (N1) Grafting ticks on a finished model with tasks one pass
solves: pass-through (T3, T20) - the task never asked for the ticks. (N2) Rewarding state motion: motion is not
use (held-out CE +0.40). (N3) Ungated shaping in policy gradient near zero accuracy: shaping takes over (run6).
(N4) SIGReg against collapse: there is no collapse to prevent (W3) and it costs accuracy. (N5) Latent auxiliary
targets on composite terms over CE (the hop-term target, goal, self): no gain over 3 seeds; R5 pays only when the
intermediate is exact and atomic. (N6) Rule tasks with a closed form in k (shift; affine mod 16, whose powers are
the identity) as depth tests: one pass computes them. (N7) Judging an add-on by final accuracy at the trained depths
at a ceiling: inside the seed spread; measure what the idea predicts (extrapolation in depth or M, data
efficiency, per-tick decode). (N8) Storing facts first, then the curriculum, to make the loop compose them: no
change (L1). Caution on N5: the label-free targets were tested on clean derivations only, where CE
already carries the signal; their claim is about data where the structure must be found (stage 1d, text
enciphered), not tested yet.

**6d.5 Transfer.** General RWKV-7 models: the recipe asks for R3 data (anonymised derivations, chase) and trained
ticks with CE; ZLK.16 predicts a model closer to raw pretraining (World) learns them more readily than one tuned
further (G1d); the laptop precursor (plan noesis-world-a30 §0) tests it. RWKV-8: RWKV-8 keeps the RWKV-7 time mix
and adds ROSA, an exact-match suffix-automaton block over the history, trained with the rest (community map; v8 code
not read yet); its public checkpoints are toy-scale (39.6K and ~1M parameters), the scale of this toy. L1 and ZLK.10
mark where ROSA belongs: the exact past is fetched from history at each tick instead of being stored and composed
in the state or the weights. Extension, not built: the same recipe with a ROSA block queried per tick, measured on
chase with more pairs than the WKV state holds (n beyond state capacity) and on composition over facts in context
far back; read BlinkDL's v8 code whole before writing it.

### 7. Tests with controls

| test | instrument | control | decision rule | claims | state |
|---|---|---|---|---|---|
| T-gap | `cot_vs_routes_probe.py` | same items every route; paraphrase floor | w, tokens to first correct per family | ZLK.1 | done once |
| T-recall | re-read on recall vs reasoning items | neutral filler | re-read helps recall only | ZLK.2 | not run |
| T-classes | `m0_pass --m0-rank --m0-samples` on an entity battery | type-matched alternatives; held-out facts | transition table per stage | ZLK.11, 12, 15 | capitals only (Qwen) |
| T-capacity | `state_capacity_probe --ticks`, `state_readout_lens` | design eff-rank; multiplicity; P1 as negative | separable facts with ticks vs one pass, per arm | ZLK.12, 13, 19 | T15 done on G1d 0.4B (CV ridge) |
| T-relations | held-out relation items from read facts | arm C | `G_M` and tick-only items vs arm C | ZLK.14 | not built |
| T-yield | tick-yield probe | constant feed; shuffled answer; zero-motion guard | R_density(t) CI > 0 and paired acc rise | ZLK.4, 5 | spec only |
| T-channel | `latent_geometry` + S-side cosine | constant arm; swapped feed | collapse criterion of ZLK.18 | ZLK.4, 18 | measured per checkpoint |
| T-basin | temperature sweep; state noise | neutral-filler / random-word prefix | output vs state basin | ZLK.3 | not run |
| T-text | `compress_probe` | fresh text (nothing broken) vs seen text (repair) | bpb at `M = k` vs `M = 0` | ZLK.12, 15 | Qwen only |
| T-source | three conditions (`H19.md`) | unrelated passage | context gap grows on in-context items | ZLK.15 | not run |
| T-stop | S3 gate | fixed `M`; random stop at equal mean | ZLK.5 refuter | ZLK.5 | after S3 |
| T-views | similarity across views | shuffled views | ZLK.7 refuter | ZLK.7 | after S1 |
| T-data | method fixed, composition varied | same token budget | does composition matter once the method is rich | - | not run |

### 8. Code and data

| status | where |
|---|---|
| exists | `experiments/rl/`: `train_latent_s1.py` (self / cache / gold targets, KL T, `--on-policy`, `--frozen-teacher`, `--teacher-model`, `--anchor-items`, `--teacher-hint`, `--teacher-cot`, checkpoint per eval, `--device/--dtype`), `loader.py` (`forward_stateful_embeds(..., return_hidden=True)`, `MuonHybrid`), `hf_loader.py`, probes registered in `experiments/run.py` (`m0_pass` with `--m0-ticks --m0-rank --m0-samples --m0-hint`, `chat`, `latent_geometry`, `latent_noise`, `known_heads`), `state_capacity_probe.py` (`--ticks`, uncommitted), `state_readout_lens.py`, `compress_probe.py`, `cot_vs_routes_probe.py`; data: `training/scripts/gen_ciphers.py` (decode / context / question / mixed, levels, hint ladder, formats), the catalog, Wikipedia vital intros (959, outside the repo), UncheatableEval 2026-07 |
| must change | `train_latent_s1.py`: an off-policy teacher-sample mode with an on-policy share schedule (W0); a text mode (teacher KL per token on passages, ticks at random boundaries, random `M`) (S2); a run manifest; `gen_ciphers`: name and city pools widened (22 / 16 / 32 now), fact-split |
| to write | the entity fact battery from Wikidata triples of the vital entities, with type-matched alternatives; a k-hop generator (k = 3..5); relation items built from read facts; the tick-yield probe; the continue class in the answer softmax and its trainer (S3); k-subspace latent noise in log pi; a QKV-ROSA branch (S5) |

### 9. Constraints already paid for

LoRA is out. Markers and constant feeds collapse to a fixed point by tick 3-4; distinct inputs reset the direction
(0.097). Discrete word tokens are a test rig; the channel is the own latent. RL does not start on all-wrong groups.
GRPO's log pi over answer tokens only makes shaping on a deterministic loop inert (advantage change 6e-8). No reward
for per-tick correctness. A teacher state is not a target. The teacher is fixed. Gold strings are not targets.
Muon at 0.02 moves the worst tensor 349% of its norm in one step. A training corpus without an end-of-answer token
teaches loops. Personal session data never enters a corpus or git.

### 10. Literature, and what is ours

Continuous latent reasoning by feeding back the last hidden state (Coconut, arXiv 2412.06769); recurrent-depth
reasoning with backtracking from variable depth (Huginn, arXiv 2602.08100) and per-depth process rewards (LSRL,
Findings EMNLP 2025); pause tokens help only after training (arXiv 2310.02226); just-read-twice for recurrent LMs
(arXiv 2407.05483); overthinking after the first correct answer (arXiv 2412.21187); LoRA learns less and forgets
less (arXiv 2405.09673); JEPA objectives for LLMs (arXiv 2509.14252); usable information (Xu et al. 2020);
on-policy distillation from a student's own samples (GKD, Agarwal et al. 2024, cited from memory); a state-readout
corrector that pays most at small size (Scarletwolf, 2026-08); the tunnel effect in deep classifiers (arXiv
2305.19753), neural collapse (PNAS 2020), memorisation in deep layers (arXiv 2105.14602), read for ZLK.20;
weight-shared recurrence as depth in vision (Liao & Poggio 2016, arXiv 1604.03640), for §1 thesis 13;
REINFORCE (Williams 1992), the natural policy gradient (Kakade 2001), the Occam bound (Blumer et al. 1987) and
compression progress (Schmidhuber 2009, from memory), for §1c. Not found elsewhere: a latent loop on a fixed-size
recurrent state whose continue/stop decision is the model's own action, trained against a verifier and a price on
time, measured by the readable share of what the model holds. (A search, not a survey.)

### 11. Not claimed

No claim that the loop beats the best CoT model, only this teacher on tasks where time helps. No claim about
pretraining from scratch: "second pretraining" is a post-training chain on a pretrained base; a latent loop inside
pretraining (an offer to the RWKV-8 work) is a proposal outside these tests. No priority claims. The Muon write-up
(`docs/muon-rwkv7-finetuning.md`) is a separate piece.
