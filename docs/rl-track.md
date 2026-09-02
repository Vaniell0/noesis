# RL Track — Matrix Task Curriculum (A1.5)

## Track status (read first — the 4-stage plan, current as of 2026-08-23)

This file covers two related sub-tracks that share one document because Phase 3
consumes Phase 1/1.5/2's output: **ThinkChain latent-refinement** (Phases
1/1.5/2, `experiments/rl/train_think_distill.py`, matrix/arithmetic/xor/
crossword corpus) and **word-search GRPO** (Phase 3, `experiments/rl/
train_wkv_loop.py`, this file's own §RL design below). The corpus mismatch
between them (matrix/arithmetic vs. word-search) is flagged in Phase 3 below,
not resolved — read it before assuming the two stages share a curriculum.

1. **Phase 1 — the cycle is born (done, shipped).** ThinkChain (`M+1`
   explicit, distinct trainable embeddings — one shared entry cue, one per
   phase — fed via `forward_stateful_embeds`) replaces the self-feed loop,
   which homogenised into a fixed point instead of doing `M` genuinely
   different units of work (mechanism explained in the appendix below). Bar:
   existence + stability, not richness. Real run
   `g1i_think_distill_zlk_phase1_v3` (500 steps, LoRA r=32/alpha=64, M=1):
   free generation with the trained phase went from actively harmful at step
   100 to correct on both diagnostic prompts at step 500 — full reversal,
   confirming the mechanism was undertrained, not broken. Bare M=0 path
   degraded over the same training (LoRA reshaping weights toward "expect the
   marker") — a real risk carried into Phase 1.5, which removes the LoRA
   bottleneck that was limiting how far that drift could go.

2. **Phase 1.5 — robustness + full-FT smoothing (next, not started).** Merge
   the LoRA delta and continue on full-FT; goal is a mechanism that doesn't
   break and an M=0 path that doesn't degrade, across M=0..3, not just the one
   trained M. Plumbing, not meaning, yet — but see the 2026-08-23 addition to
   Phase 2 below: this stage may already be doing more than plumbing.
   **Curriculum design, 2026-08-23 (concern raised, not yet implemented):**
   the M=0 degradation already observed on v3 (LoRA reshaping weights toward
   "expect the phase marker") is a real risk for full-FT, which has no LoRA
   bottleneck limiting how far it can go — worse, not better, without a fix.
   Proposed fix, Birdie-style (arXiv 2411.01030, already in
   `docs/community-map.md`'s §"Aligned techniques"): extend
   `_CategoryBatcher` with a second stratification axis — M/difficulty, not
   just task category — so every batch keeps a fraction of plain, no-marker,
   simple-response examples (M=0, not task-specific) alongside the M=1..3
   harder-task examples, continuously through training, not as an early
   curriculum stage that gets left behind. Not "train easy first, then
   hard" (that's exactly the sequential pattern that would let M=0 be
   forgotten again) — concurrent mixing, every batch.

3. **Phase 2 — give M>1 real meaning (designed, revised 2026-08-23, not
   built).** Original framing (kept in the appendix below for the reasoning
   that led here): a dedicated rewind marker, trained to pull state back
   toward `state_after_prompt` via L2, so a different subsequent phase-marker
   could explore a distinct continuation from near the same origin. **Revised
   design, following a direct discussion of what `M>1` is actually for** — not
   just boundary-balancing between fixed phases, but genuine movement:
   advance, then *smoothly, gradually* retreat, then advance again,
   potentially several times, not a single snap back to a fixed target.

   - **Mechanism reuses phase-marker plumbing, doesn't add new plumbing.** A
     rewind marker is just another constant learned embedding, repeated over
     several ticks via the same `chunk_lens`/dynamic-phase-stop machinery
     phase markers already use (repeating a constant embedding is a fixed
     contraction applied T times — not the self-feed loop's homogenisation
     failure, since R/K/V/decay's time-shift delta is exactly zero from the
     second repeat onward). The "smooth" in "smoothly retreat" is this:
     several ticks of contraction, not one L2-driven jump.
   - **Loss: mirror L_state, don't repeat the Dreamer mistake.** The original
     framing's L2-to-a-fixed-target approach is exactly the strict-target
     mistake already diagnosed for teacher-state matching generally (see
     `docs/community-map.md`'s "Dreaming" entry: answer/EOS correctness should
     be primary, state-matching a *softer shaping term*, not a strict target —
     confirmed the hard way once already, appendix item 10 below). Proposed
     instead: use `L_tPC` (parked in this file's §State metrics as "post-GPU,
     ablation required" — InfoNCE between WKV states at `t` and `t+k`,
     rewards *predictable* motion) as the rewind marker's training signal, as
     the direct mirror of `L_state` (rewards *large, curved* motion) used for
     phase markers. Explore gets the curvature-maximising loss; retreat gets
     the smoothness-maximising loss — genuinely opposite objectives for
     genuinely opposite roles, not two flavors of the same mechanism.
     **Complicated by real data, 2026-08-23 (task #12's run, below): the
     CURRENT explore marker (M=1, no curvature loss active — `l_state_weight`
     is already de-prioritized per standing project decisions) already shows
     `delta_cos_prev` drifting strongly toward +1 across its own 8 repeat-
     ticks (0.64-0.68 → 0.97-0.98, all 4 diagnostic prompts) — i.e. it
     *already* contracts smoothly, the exact signature this bullet wanted to
     reserve for rewind specifically. Explore and rewind may not be
     distinguishable by dynamics *shape* (curvy vs. smooth) at all — a
     repeated constant marker seems to contract smoothly regardless of
     content, by construction (fixed transformation applied T times, see
     appendix's "why the self-feed loop was replaced"). The real
     distinguishing factor is more likely *where* each marker's fixed point
     sits (task-relevant content vs. `state_after_prompt`), not how it gets
     there. This doesn't kill the L_state/L_tPC pairing outright, but it's
     not the clean "opposite dynamics" story as written — revisit before
     implementing either loss.**
   - **Marker sequences become interleaved, not monotonic.** `[entry,
     phase_A, rewind, phase_B, rewind, phase_C, ...]` instead of `[entry,
     phase_1, phase_2, ...]` — the model explores several distinct directions
     from approximately the same near-origin point, which is the actual
     operationalisation of "advance and retreat," not a single mechanism
     bolted onto the old design.
   - **Architecture caveat, stated explicitly so nobody expects more than
     this buys:** WKV's decay + outer-product update is not invertible
     (contrast the so(3)/Cayley PoC in `docs/community-map.md`, where
     composition is a rotation and trivially reversible by construction). A
     rewind marker is always a *trained approximation* to an inverse, never
     an exact undo.
   - **Real budget-safety finding, 2026-09-02 (`hypotheses/H25.md`'s
     "third follow-up" — user-corrected math, not the naive power-iteration
     reading first assumed): a trained rewind-style marker looped past its
     intended horizon does not stay safely near where it was trained to
     land — it diverges.** `pseudo_inverse_ceiling`'s T=8-optimized marker,
     traced past T=8 (`experiments/A0_state_probe/micro_wkv.py::
     verify_power_iteration_prediction`, `--extend-ticks`), heads
     monotonically toward its operator's TRUE fixed point — `‖x*-S0‖/‖S0‖
     ≈252`, nothing like S0 — not toward a plateau near the target:
     `t=8→0.558, t=16→0.821, t=32→1.690, t=64→3.318, t=128→6.071,
     t=1024→39.01` (worse than the 1.4654 starting drift past t≈32).
     T=8's low error is a transient pass-near-target on the way elsewhere,
     not proximity to a stable attractor — consistent with, and explained
     by, the near-degenerate top eigenvalue pair (0.9998, 0.9998) from
     `jacobian_spectral_sampling`'s spectral-gap analysis: barely `<1`,
     not a comfortable safety margin. **Direct implication for `−β·M`
     budget design (§RL design below): a marker running past `M_max` due
     to a bug, an off-by-one, or a resumed checkpoint with a mismatched
     `--chain-phases` (already a documented real failure mode, task #12
     above) is not merely wasted compute capped by the reward penalty —
     it can actively degrade state, with the degradation compounding, not
     saturating.** Whether this toy-scale finding transfers to the real
     32-layer×32-head G1i substrate is untested; treat as a reason to
     bound M_max/chain-phases defensively (fail closed, not just
     expensively) rather than assume "more ticks than needed" is merely
     inefficient.
   - **New validation step, ahead of building any of this (2026-08-23,
     arXiv 2602.08100, "Emergent Search and Backtracking in Latent Reasoning
     Models," Huginn-0125):** backtracking in looped latent reasoning showed
     up *without* being explicitly trained for — it emerged purely from
     training at variable recurrence depth, exactly what Phase 1.5 already
     plans (M=0..3 robustness). Their detection method is nearly free to
     port: decode the answer-readout distribution after every internal step,
     watch for the majority-vote answer to flip (A dominant ≥3 steps, then
     B≠A dominant ≥3 steps) — no probe, no new training. Their result: 32% of
     instances backtrack, +34% accuracy when they do, smooth (not
     discontinuous) entropy transitions, 72% of backtracks abandon the
     semantically-closest wrong answer (a real recalibration signature, not
     noise). **Done, 2026-08-23 (task #12) — real trained M=1 markers
     (`--chain-phases 1`, matching the checkpoint's own training config; a
     first attempt at `--chain-phases 3` silently fell back to random,
     untrained markers on a shape mismatch and was discarded — same failure
     mode already documented for the step200 data above, worth watching for
     every time `--resume` and `--chain-phases` aren't both checked
     together).** Result: chain backtracked in 1 of 4 diagnostic prompts
     (wordsearch: readout flips from token 869 to token 25427 between tick 2
     and tick 5 of the 8-tick phase); `loop` backtracked in 0 of 4. A real,
     directional signal — chain can and loop didn't, on this small sample —
     but `n=4` fixed prompts is nowhere near Huginn's 260-instance,
     32%-of-cases claim; treat as "worth tracking as Phase 1.5 trains more
     M," not as confirmation. Full data:
     `experiments/rl/results/state_trajectory_backtrack_v3_step500_m1.json`.
     Again after Phase 1.5 (M=0..3 robustness) before building a dedicated
     rewind marker — if backtracking rate rises with M-variance training
     alone, Phase 2's first job stays measuring and reinforcing it, not
     constructing a mechanism from scratch. Tracked in memory
     `project_noesis_rl_track` / TaskCreate #12.
   - Ceiling test, unchanged: `state_trajectory_probe.py`'s
     `delta_norm`/`delta_cos_prev`, applied *across* phases — stop growing
     `M` where an added phase contributes no distinct work.

4. **Phase 3 — RL teaches autonomous use (this file's own §RL design
   below).** Once Phase 2 has given the cycle real, designed meaning (or
   established that it emerges from Phase 1.5 alone), RL is where the model
   learns to *autonomously decide* how to use it — when to rewind, how much
   `M` — rather than following a fixed schedule. **Two flagged, unresolved
   gaps, unchanged since 2026-08-22:** (a) this file's own §RL design trains
   word-search grid tasks; Phase 1/1.5/2 train matrix/arithmetic/xor/
   crossword — different corpora, intentional split or oversight, not
   decided. (b) the mandatory R-lens-on-non-task-axes shortcut check this
   file already specifies (§Step10 SFT below) has never been run against any
   ThinkChain checkpoint — do this before trusting Phase 1/1.5's output as
   clean groundwork. **New, from the same 2026-08-23 discussion:** once
   ThinkChain markers replace `generate_rollout`'s feed modes (the still-
   pending port, §Switch-GRPO below), and if Phase 2's rewind marker or its
   Huginn-style emergent equivalent is real, Switch-GRPO's `<swi>`/`</swi>`
   boundary-token machinery likely needs a third boundary type (a tagged
   "revert" transition) to keep the policy ratio well-defined across a
   rewind event too, not just entry/exit — flag for whoever does that port,
   not designed yet.

5. **Phase 4 (speculative, not started, gated on an external signal) — ROSA
   as a delegate marker, not an architecture merge.** Standing decision
   (2026-08-22, `reference_rwkv_lucas_community_2026_08.md`): don't
   integrate ROSA into RL without a stronger signal it works at LM scale,
   not just synthetic tasks — unchanged. If that signal ever lands, the
   design direction discussed 2026-08-23 is NOT "merge the ROSA block into
   the architecture and retrain end-to-end" — it's a third ThinkChain
   marker type alongside explore (Phase 1-3) and rewind (Phase 2): a
   **delegate** marker whose fixed point is a call into a ROSA register for
   an exact-precision sub-operation (digit sequences, verbatim copy, exact
   lookup — H25's controllability bottleneck), with the result read back
   into WKV via the same repeat-tick mechanism already built for the other
   two marker types. Ties to the parked "WKV as programmable coprocessor"
   idea (memory `project_noesis_reservoir_computing_idea`). Not
   implementable today — ROSA has no LM-scale checkpoint to delegate to.
   **Scope narrowed by the toy-computer result below (hypotheses/H25.md,
   2026-08-23): WKV's own delta-rule physics was shown to perform real
   bilinear computation (multiplication) unaided, at toy scale, once
   given the right control signals.** If that holds up at real-model
   scale (not yet tested), a delegate-to-ROSA marker isn't the default
   path for exact operations in general — it becomes a reserve for
   whatever WKV's own precision/capacity genuinely can't cover (longer
   chains, larger magnitudes, operations past what per-channel decay +
   rank-1 write can represent), not a replacement for training WKV
   itself to compute.

Mechanism-level detail (recurrence math, decay derivation) lives in
`docs/rwkv7-mechanics.md`. Full run-by-run narrative for Phases 1/1.5/2 is the
appendix below and memory `project_noesis_think_distill_experiments`/
`project_noesis_rl_track` (read the memory first in a fresh session — this
file's appendix is the detailed backing, not the entry point).

---

## Motivation

The RL track trains the model on a unified curriculum of matrix tasks — all
structured as ASCII grids with verifiable answers. The design goals:

- **Verifiable reward without a judge.** Every task has a rubric (regex or
  exact match) checked deterministically. GRPO works on binary ±1 signal.
- **Gradient diversity.** Nine task types (wordsearch, bits, arithmetic,
  pattern, crossword, sudoku, ARC…) activate orthogonal WKV channels.
  A single task type causes gradient collapse to one learned pattern.
- **Internal iteration (M-loop) — mechanism exists, "self-reflection" as a
  capability is not yet demonstrated.** The model runs M internal steps
  without emitting tokens; the entropy trajectory over those steps acts as
  the exit criterion (`plateau`/`commit`/`M_max`, see §RL design). No
  `<think>`-span tokens or `r_entropy`-on-text-span reward exist in the
  current design — that framing is gone, not just renamed. What's verified:
  the mechanism runs and the exit criteria fire as designed. What's *not*
  verified, and as of 2026-08-19 actively in question: whether training
  shapes this into anything resembling reflection rather than just an
  internal loop that sometimes helps and sometimes doesn't — real
  pathologies found the same day (unconditional reward-shaping collapsing
  the loop to 100% M_max-saturation between step20 and step50; a rubric
  scoring bug letting the model get "correct" credit for echoing its input
  verbatim) show concrete ways this can go wrong before it goes right. A
  clean, bug-fixed baseline was only just established (see §Mechanisms and
  their relationships) — not yet run long enough to claim more than that.
- **Bidirectional and diagonal attention.** Wordsearch L3–L7 forces
  right-to-left, bottom-to-top, and diagonal reading — breaking the
  left-to-right pretraining bias. WKV state must encode directional context.
- **Bootstrap signal from L0.** `wordsearch_name` (find the word, not its
  position) gives the model its first positive rewards. L1–L2 appear rarely
  (~5% each) for initial signal; L3–L7 dominate to prevent guessing.
  Curriculum cuts L1–L2 once success rate stabilises.
- **State utilisation measurement.** Grid tasks require accumulating
  evidence across many positions before committing an answer — a direct
  test of H8 (state-as-computation) and H10 (effort frontier).

---

## Difficulty Levels

| Level | Grid | Orientations allowed | Word length |
|-------|------|----------------------|-------------|
| 1 | 5×5  | H_LR only            | 3–4 |
| 2 | 6×6  | H_LR, V_TD           | 4–5 |
| 3 | 7×7  | H_LR, V_TD, H_RL     | 4–5 |
| 4 | 8×8  | H_LR, H_RL, V_TD, V_BU | 5–6 |
| 5 | 9×9  | all 4 axes ± reverse | 5–6 |
| 6 | 10×10 | all 4 axes + diagonals | 5–7 |
| 7 | 12×12 | all 8 directions     | 6–8 |

Orientations: H_LR (→), H_RL (←), V_TD (↓), V_BU (↑),
D_DR (↘), D_DL (↙), D_UR (↗), D_UL (↖).

Within each level the orientation is drawn uniformly at random from the
allowed set, so the eval file mixes types. This prevents the model from
pattern-matching on the prompt phrasing alone.

---

## Task Format

```json
{
  "id": "wsearch_L3_7x7_H_RL_00_04",
  "category": "matrix_wordsearch",
  "level": 3,
  "orientation": "H_RL",
  "prompt": "Below is a 7×7 letter matrix ...\n\nThe word STONE appears exactly once, placed horizontally (right-to-left). Find it. Output only the position of the word's first letter in the form 'row=R col=C'.",
  "answer": "row=2 col=6",
  "rubric": {"type": "regex", "value": "row\\s*=\\s*2\\b[^0-9]*col\\s*=\\s*6\\b"}
}
```

The rubric anchors on the **first letter of the word** (where placement
begins), regardless of reading direction. The model must understand that
for H_RL the first letter is the rightmost one visually.

---

## Baseline (G1h base, 2026-08-13)

Word-search eval on G1h-2.9B-q5_1 via HTTP shim: **0% on all levels**.
G1h has not been trained for spatial scanning — this is the expected zero-shot
floor. All future RL checkpoints should be compared against this baseline.

Results: `experiments/A0_eval/results/g1h_wordsearch_baseline.json`

## Generator

```
experiments/A0_eval/gen_matrix_wordsearch.py
```

```bash
# All 7 levels, 8 tasks per level (56 tasks)
python3 experiments/A0_eval/gen_matrix_wordsearch.py \
    --all-levels --per-level 8 \
    --out experiments/A0_eval/tasks_matrix_wordsearch.jsonl

# Single level for targeted eval
python3 experiments/A0_eval/gen_matrix_wordsearch.py \
    --level 5 --per-level 20 \
    --out /tmp/wsearch_L5.jsonl

# Custom tier spec (backward-compat)
python3 experiments/A0_eval/gen_matrix_wordsearch.py \
    --tiers "7x7:4:6:H_LR+V_TD+H_RL" --per-tier 12 \
    --out /tmp/custom.jsonl
```

---

## Training corpus — matrix_tasks.jsonl

**Generator:** `experiments/A0_eval/gen_tasks.py`  
**Current file:** `training/corpus_open/matrix_tasks.jsonl` — 65 797 tasks, ~38 MB  
(SHA-256 in PROVENANCE.md; not committed — regenerate with `--seed 42 --n-tokens 20_000_000`
once Sudoku CSV and ARC-AGI data are available; current 66k is the base run without them)

**No `.pt` file.** GRPO is an online algorithm: each step samples a prompt from `.jsonl`,
generates G=8 rollouts live, computes rewards, updates weights. Pre-tokenization to `.pt`
is for SFT (fixed batches). The RL loop tokenizes on the fly inside `rollout.py`.

**Format: sp only (space-separated).** nsp eval on G1d 0.4B → 3.6% (see ByteAdapter section).
All training tasks use WorldTokenizer sp format. ByteAdapter is a separate GPU experiment.

### Task types — what the model is trained and tested on

| Category | Tasks | Level dial | Model action | Rubric | Cognitive load |
|---|---|---|---|---|---|
| `matrix_wordsearch` | 13 098 | L1–L7 (grid size, orientations) | Find word → `row=R col=C` | `regex row=\d+ col=\d+` | Directional spatial scanning |
| `matrix_wordsearch_name` | 6 679 | L1–L7 | Name the word in grid | `exact \bWORD\b` | Grid reading, recognition (L0 warmup) |
| `arithmetic_matrix` | 12 988 | L1–L7 (digit count, op, task type) | Compute sum / find missing digit / find error | `regex \bN\b` | Column-position arithmetic |
| `bits_matrix` | 13 126 | L1–L7 (op: XOR/AND/OR/NOT, task: result/missing/error/key) | Find result bit / missing input / correct bit | `regex \b[01]\b` | Bitwise computation, reverse lookup |
| `pattern_matrix` | 13 312 | L1–L7 (arithmetic/geometric/modular/Fibonacci) | Find missing value in sequence | `regex \bN\b` | Rule induction, sequence extrapolation |
| `crossword_enum` | 3 297 | L8–L11 | Enumerate intersecting words | `regex` | Multi-word grid constraint |
| `crossword_fill` | 3 297 | L8–L11 | Fill blank in crossing word | `regex` | Constraint propagation |
| `sudoku_matrix` | — (needs CSV) | fixed 9×9 | Find blank cell value | `regex \b[1-9]\b` | Row/col/box constraint reasoning |
| `arc_matrix` | — (needs dir) | 1–5 (train examples) | Apply transformation, output one cell | `regex \b[0-9]\b` | Pattern induction from examples |

All nine types share the same `{id, category, level, prompt, answer, rubric}` format —
`compute_rewards` evaluates any of them via the rubric field unchanged.

### Eval protocol

No pre-built held-out split. Eval is run separately with a different seed:

```bash
# word-search only (primary benchmark, 56 tasks)
python3 experiments/A0_eval/eval.py \
    --tasks experiments/A0_eval/tasks_matrix_wordsearch.jsonl \
    --backend rwkv --model <checkpoint_path>

# full curriculum sample (2 000 tasks, seed ≠ training seed)
python3 experiments/A0_eval/gen_tasks.py \
    --out /tmp/matrix_eval_2k.jsonl --n-tokens 400_000 --seed 99
python3 experiments/A0_eval/eval.py \
    --tasks /tmp/matrix_eval_2k.jsonl --backend rwkv --model <checkpoint_path>
```

Run full-curriculum eval after checkpoint 1 to catch category-level collapse
(a model gaming wordsearch via shortcut will still score ~chance on bits/pattern).

---

## Eval integration

Word-search tasks are scored by `eval.py` via the existing `regex` rubric
type. Run them through the same Ollama backend:

```bash
python3 experiments/A0_eval/eval.py \
    --backend ollama --host http://127.0.0.1:11435 \
    --model rwkv7-g1i-2.9b-q5_1 \
    --tasks experiments/A0_eval/tasks_matrix_wordsearch.jsonl \
    --out experiments/A0_eval/results/g1i_wordsearch_baseline.json
```

Report by level: filter `results.json` on the `level` field in each task
record to get per-level accuracy.

---

## Full FT vs LoRA for latent token training

**Decision: full FT, not a runtime choice.** `experiments/rl/loader.py::_load_peft`
hardcodes `args.peft = "none"` — there is currently no LoRA toggle in the WKV-loop
stack at all (unlike the earlier `train_wordsearch.py`, which had `--lora-r`). The
reasoning behind that choice still holds and predates this rewrite: LoRA modifies
projection matrices within a low-rank subspace, and the LoRA-rank analysis
(`experiments/A0_state_probe/lora_rank_analysis.py`, 2026-08-14, see `hypotheses/README.md`
§H16) found LoRA r=32 spans only ~1.25% of a 2560×2560 weight matrix — the routing
geometry that decides emit-vs-hold (H16) lives in the other 98.75%, which LoRA never
touches. GRPO training the M-step exit behavior is exactly this kind of routing
decision, so full FT is the only path expected to move it.

**Cost consequence — RESOLVED 2026-08-18, this paragraph was stale until
2026-08-23's pass caught it.** Full FT on G1i 2.9B needs the whole optimizer
state in VRAM — naive estimate ~18GB. The originally-planned fix (`dk4248/FORGE`
fusing the optimizer *into* the backward pass) turned out structurally
incompatible with a WKV-loop's BPTT (fused-into-backward assumes each layer's
weight is touched by `backward()` at most once per step; the loop's weight-reuse
across timesteps violates that). Real fix that shipped instead:
`experiments/rl/loader.py::Int8AdamW` — don't fuse anything into backward, run
ordinary full-BPTT `backward()`, then apply FORGE's standalone
`optimizer_only_adamw_int8state()` kernel to the resulting `.grad` tensors.
Verified on G1d, 5 real steps, no crash. VRAM measured (not the informal
~18→~12-13GB estimate above): 13713 MiB without `--forge`, 13655 MiB with, at
G=4/batch=2/M_max=4 — essentially no difference at *this* scale, because the
WKV-loop's own unbatched per-rollout activation memory dominates total usage
here, not optimizer state (see §Known risks #2 below — that's the actual lever
for a bigger run, more than the optimizer path). Full writeup: memory
`project_noesis_forge_bptt`.

The old ε-mask / two-phase-SFT framing that used to live in this section is gone —
SFT was skipped entirely (see §Step10 SFT below), and ε-mask doesn't exist in the
WKV-loop design (`−β·M` replaces it, see §RL design).

**Adam vs. Muon (hand-integration written 2026-09-02, not yet run on the
real model/GPU — synthesis started from a conversation whose session state
was lost before it reached a commit).** The
"Muon doesn't work well on LoRA factors" objection no longer applies now that
Phase 1.5 merged the LoRA delta into base weights and moved to full FT (see
above). Muon's actual draw for this project: it carries only a momentum buffer,
no second-moment (`v`) state — a direct answer to the fixed-cost VRAM wall in
§Known risks #9 (weights 5.896GB + grad 5.895GB + Adam's own ~5.8GB int8
second-moment buffers is exactly what `Int8AdamW(offload_state=True)` exists to
work around; Muon would shrink that third term at the source instead of
offloading it). Real blocker: trainable state in this stack lives in the
`.z`/state-dict path `Int8AdamW` was hand-integrated against
(`experiments/rl/loader.py`), not as ordinary `nn.Parameter`s a stock Muon
implementation expects — it needs the same kind of manual integration work
`Int8AdamW` already got, not a drop-in swap. Cheap test before touching a real
run: reuse the frozen-readout `micro_wkv.py` toy from H25 (`hypotheses/H25.md`)
to compare OOD R² and VRAM profile between Int8AdamW and a hand-integrated
Muon, before spending a real GPU session on it.

**Cheap test run, 2026-09-02** (`experiments/A0_state_probe/muon_vs_adam_toy.py`,
results in `experiments/A0_state_probe/results/muon_vs_adam_toy.json`) —
CPU-only, so this answers only "does Muon train this architecture family at
all," not the VRAM half (needs the real model + a GPU). Canonical
`SingleDeviceMuon` (github.com/KellerJordan/Muon) on the hidden `net.*.weight`
matrices, plain AdamW on everything else (embedding/readout/biases), same
seed/protocol as the existing Adam baseline. Result: Muon trains without
collapsing (`id_r2=0.9989`, `ood_r2=0.7782`) but underperforms Adam
(`id_r2=0.9999`, `ood_r2=0.8675`) at Muon's generic published default
learning rate (0.02) — never tuned for this loss landscape. Reads as
"viable, not yet competitive under a default LR," not a verdict either
way — a learning-rate sweep on the toy is the next cheap step before
deciding whether the real `Int8AdamW`-style hand-integration is worth
building.

**LR sweep, same day** (`--lr-sweep`, `experiments/A0_state_probe/results/
muon_vs_adam_lr_sweep.json`, single seed per point — a direction, not a
statistically robust result): `id_r2` falls monotonically as `muon_lr`
rises (0.9997 @ 0.005 → 0.9995 @ 0.01 → 0.9989 @ 0.02 → 0.9966 @ 0.04 →
0.9725 @ 0.08, the last not even converged by step 4000). At `lr=0.005`
(4x below Muon's generic default) the gap to Adam nearly closes:
`id_r2=0.9997`/`ood_r2=0.8419` vs. Adam's `0.9999`/`0.8675` — the earlier
"underperforms" reading was an artifact of the untuned default, not a
real ceiling on what Muon can reach here.

**5-seed check, same day** (`--n-seeds 5 --muon-lr 0.005`, `experiments/
A0_state_probe/results/muon_vs_adam_multiseed.json`) — the single-seed
"nearly closes the gap" reading above was itself an artifact: seed=0
happened to be one of Adam's *better* draws on `ood_r2`. Across 5 seeds,
`id_r2` is near-identical and tight for both (Adam 0.9996±0.0005, Muon
0.9997±0.0001); `ood_r2` is if anything slightly higher for Muon on
average (Adam 0.7449±0.0921, Muon 0.8173±0.0636) — no real quality gap
either direction once seed noise is accounted for. **The one real,
seed-robust difference is mechanistic, not quality**: forcing `a_gate=0`
(H25's delta-rule-necessity ablation) collapses Adam's solutions hard,
every seed (mean -1.376±1.326, range -0.21 to -3.30), but barely touches
Muon's, every seed (mean -0.029±0.032, range -0.076 to +0.003) — see
`hypotheses/H25.md`'s "second follow-up" for the full table and the
reframe this implies for H25's own necessity claim (solution-dependent,
not task-inherent).

**Decision (user, 2026-09-02): proceed with the real hand-integration,
scoped to Phase 1.5/2 only, not Phase 3.** Two independent reasons stack:
(1) the memory case never depended on this toy's R² at all — Muon drops
`Int8AdamW`'s ~5.8GB int8 second-moment buffer entirely (§Known risks
#9's actual fixed-cost problem), which is real headroom regardless of
how the LR sweep turned out; (2) Phase 1.5/2 are supervised distillation
(teacher/student CE + state-distillation), not RL — an unfamiliar
optimizer's quirks are far cheaper to diagnose there than inside GRPO's
own policy-gradient noise, so this is the lower-risk place to spend the
integration effort first. Phase 3's optimizer choice is explicitly left
open/deferred — RL's loss landscape and Muon's interaction with
advantage-weighted updates are untested, and a new VM before Phase 3 is
likely anyway, so nothing about Phase 1.5/2's choice commits Phase 3.
**Hand-integration done (2026-09-02): `MuonHybrid` in `experiments/rl/loader.py`**,
same interface as `Int8AdamW` (`.other_params`, `.zero_grad()`, `.step()`) so it
drops into the same call sites — `--muon` flag added alongside `--forge` in both
`train_think_distill.py` and `train_wkv_loop.py` (mutually exclusive with
`--forge`; separate `--muon-lr`/`--muon-momentum`/`--muon-weight-decay` since
Muon's LR scale has nothing to do with Adam's `--lr`). Selection is by
parameter *name*, not `Int8AdamW`'s dim()==2 catch-all: only
`blocks.N.att.*.weight` / `blocks.N.ffn.*.weight` go to Muon (matches Muon's
own usage guidance — embeddings and output heads are excluded on purpose,
`emb.weight` is a lookup table not a matrix-product participant, and `head`
was already carved out of FusedLinear wrapping for an unrelated reason
upstream). Newton-Schulz coefficients and the momentum update are copied
near-verbatim from github.com/KellerJordan/Muon, same reasoning as the toy
script. Verified so far, CPU-only (no GPU on this machine): a unit test
against a toy module with real RWKV7 parameter names confirms the
muon/other split lands exactly on the intended params (embedding, head,
LayerNorm all correctly excluded) and that a few real optimizer steps move
every trainable parameter without error, including through the rectangular
`ffn.key`/`ffn.value` matrices (not just square `att.*` ones). This checks
the integration's *mechanics* only — it says nothing about the real model's
training dynamics, VRAM profile, or whether the leak (§Known risks #11)
goes away; that needs an actual GPU run.

**First real GPU run, 2026-09-02 (Alberta, T4 16GB): OOMs on backward every
single step, even at the most minimal config.** `--muon --grad-cp --batch 1
--think-marker --M 1`, raw `rwkv7-g1i-2.9b-20260805-ctx16384.pth` (full-FT,
no LoRA) — the smallest full-FT config this stack supports. Ran all 20
smoke-test steps without crashing (the existing `_is_oom_error` try/except
around `micro_total.backward()` caught it and skipped every time), but
`grad_norm=0.0000` and the `(partial: a micro-batch OOM'd)` tag appear on
**20/20 steps** — forward succeeds (loss values are real and change per
step), backward never completes even once. No real gradient was ever
applied; the checkpoint this run "completed" to is untrained noise, deleted
rather than kept. `[distill] Muon enabled: 192 hidden matrices on Muon,
871 params on AdamW` confirms the split ran (matches the toy-verified
selection logic).

**Reading, not yet independently profiled**: the CUDA allocator's own OOM
message at the moment of failure reports as little as ~12MB free out of the
~16GB card (`free: 12189696, total: 16704405504`) on the very first step —
i.e. something already consumes essentially the entire card before the
backward pass gets far, the same *fixed-cost* shape as `Int8AdamW`
(without `offload_state`) hit before (§Known risks #9, measured ~17.4GB
fixed cost alone on this card). Plausible arithmetic for why: weights
(~5.9GB bf16) + first-backward grad buffers (~5.9GB) + `MuonHybrid`'s
momentum buffer, allocated eagerly in `__init__` for all 192 muon_params
(bf16, likely ~4.5-5GB given att/ffn dominate parameter count) already
sums close to or past 16GB, before `other_params`' AdamW state or any
activation memory. Not confirmed by a direct `torch.cuda.memory_allocated()`
snapshot (skipped to save GPU time given the allocator's own free-memory
number already makes the fixed-cost explanation the only one consistent
with failing on the very first step at minimum batch) — if this needs
re-litigating later, that snapshot is the next thing to add, not another
blind config change.

**The tension this creates, worth flagging plainly rather than patching
around silently**: the whole draw of Muon over `Int8AdamW` was carrying
only one buffer with *no* CPU-offload dance needed — removing the exact
mechanism suspected in the still-open host-RAM leak (§Known risks #11).
If `MuonHybrid` also needs an `Int8AdamW`-style `offload_state` to fit this
16GB card, that hoped-for side benefit doesn't hold: an offloaded Muon
would reintroduce the same per-step CPU↔GPU `.to()` swap pattern that is
the #1 remaining leak suspect, just wrapped around a different optimizer.
Not decided here — this is a real fork (add offload to `MuonHybrid` and
accept the leak risk returns; or find a smaller `MuonHybrid` footprint some
other way; or accept LoRA rather than full-FT for the Muon path, reversing
Phase 1.5's own move away from LoRA) rather than a bug with one obvious fix.

**BlinkDL's answer, Discord #state-and-finetuning, 2026-09-02 (asked
in response to this session's Muon-vs-Adam finding): "muon works for
rwkv7 pretraining, but for finetuning a trained rwkv7 model, no idea.
please let us know."** Confirms Muon is architecturally compatible with
RWKV-7's update mechanics at least for pretraining — no fundamental
incompatibility with the per-channel decay / rank-1 write structure. Our
specific case (full-FT on an already-pretrained G1i checkpoint, not
pretraining from scratch) is explicitly unknown territory, including to
him — real motivation to actually finish the hand-integration and report
back, not just a nice-to-have. A separate community member ("Tomeno")
raised a skeptical question in the same exchange about state/decay
averaging under Muon relative to RWKV4-style decay — not yet parsed
carefully enough to answer here, flagged rather than guessed at.

**Backlog idea, not started (user, 2026-09-02): once the real
hand-integration exists and has run for real, consider upstreaming it as
a FORGE PR** (`dk4248/FORGE`, same repo `Int8AdamW`'s approach already
has a merged PR in, `project_noesis_forge_bptt` memory) — a
`fused_muon_update`-style kernel/wrapper mirroring how
`optimizer_only_adamw_int8state` is exposed today. Explicitly contingent
on the integration actually working and being worth sharing — "if it
even makes sense" was the user's own framing, not a commitment.

---

## RL design

**No `<think>` tokens.** The design below (rewritten 2026-08-17, superseding the
think-token/CLIPO/ε-mask design this section used to describe) replaces visible
CoT with **M silent internal WKV-refinement steps**: the model runs the WKV
recurrence forward up to `M_max` times with no token emitted, then commits to an
answer. Implementation: `experiments/rl/wkv_loop.py::generate_rollout` (loop) +
`experiments/rl/rewards.py::compute_wkv_loop_rewards` (reward) +
`experiments/rl/train_wkv_loop.py` (GRPO training loop, entry point).

**The M-step loop and its exit criteria.** Each step feeds the model's own
current-position output back in (mode depends on `--feed-mode`: `discrete` samples
a token and feeds its embedding; `expected`/`residual` feed the softmax-weighted
embedding directly, differentiable, peft-backend only) and re-runs the WKV
recurrence. The loop exits — and the model commits to the actual answer — on
whichever of three conditions fires first:

| Exit reason | Condition | Meaning |
|---|---|---|
| `plateau` | `\|H_t − H_{t-1}\| < eps_plateau` (default 0.05) | Entropy stopped moving — another step isn't changing what the model "thinks" |
| `commit` | `max(softmax) > tau_commit` (default 0.9) | Model is already confident — no need to keep refining |
| `M_max` | step count reached the cap (CLI default 16) | Budget exhausted, forced exit |

`H_t` is the entropy of the next-token distribution at loop step `t` — same
quantity as the old think-span entropy metric, just measured over internal
WKV steps instead of emitted tokens. `exit_reason` and `M` (steps actually used)
are recorded per rollout (`WKVLoopRollout.exit_reason`/`.M`) and drive both the
reward and `experiments/rl/monitor.py`'s health checks (STATE_COL flags when
`exit_reason` is almost always `M_max` — the model isn't learning to stop early).

**Reward signal.** `r = r_correct − β·M − γ·Σ_t ReLU(H_t − H_{t-1})`
(`compute_wkv_loop_rewards`, defaults β=0.02, γ=0.1):
- `r_correct`: rubric match against the task answer (binary).
- `−β·M`: step-count penalty — same role as the old ε-mask's "every emitted
  token has a cost," just counting internal steps instead of think-tokens.
  This is what teaches the model to prefer `plateau`/`commit` exits over
  riding out to `M_max` on every rollout — the emergent quality gate the old
  ε-mask section described, now via a direct penalty on step count rather
  than an indirect loss on non-think-span tokens.
- `−γ·Σ ReLU(ΔH_t)`: penalises entropy going *up* between consecutive steps
  (only the increases; `ReLU` zeroes out decreases) — a step that makes the
  model *less* sure is actively penalised, not just uncompensated.

There is no CLIPO term and no separate `r_clipo`/`r_entropy`-on-think-span
computation in the current implementation — see §Deferred below for where
CLIPO/L_KVB actually stand.

**Curriculum schedule** (`experiments/rl/corpus.py::CorpusScheduler`,
confirmed matches this description 2026-08-17):
- L0 (bootstrap, pinned): `matrix_wordsearch_name` never leaves level 1,
  regardless of accuracy — `_L0_ANCHOR` in `corpus.py` explicitly skips it in
  both the advance and drop loops. Gives the model early positive reward
  before it can do positional wordsearch at all.
- All other categories: advance one level when a 5-batch rolling average
  accuracy ≥ 80% (`advance_thresh`), drop one level below 50%
  (`drop_thresh`). Matches the original design's numbers exactly; the
  5-batch averaging window is new detail worth having on record.

**Training algorithm.** GRPO, `G=8` rollouts per prompt, `batch=4` prompts per
update (CLI defaults), PPO-clip surrogate (`clip_eps=0.2`) + KL penalty
(`kl_coef=0.01`) in `train_wkv_loop.py::wkv_grpo_loss`. Temperature 0.7 during
rollout (`wkv_loop.py` default `answer_temperature`), greedy for eval.

**Corpus mix.** `training/corpus_open/matrix_tasks.jsonl` (65 797 tasks, ~20M
tokens, see §Training corpus above for the up-to-date category counts).
No SFT — pure RL on verifiable matrix tasks (see §Step10 SFT decision below).

**Connection to hypotheses.**

| Hypothesis | What word-search measures |
|-----------|--------------------------|
| H8 (state-as-computation) | Does WKV state retain directional context row-by-row? |
| H10 (multi-pass convergence) | Does the learned M distribution shift with task difficulty? |
| H12b.i (slot utilisation) | Do slot-entropy losses improve diagonal scanning? |
| H16 (gated externalisation, RETRACTED as a separate head) | The `−β·M` mechanism is a *direct* implementation of "every step has a cost" — if it works, H16's emergent-gate story is confirmed via M-step commit behavior rather than a dedicated MLP head, consistent with H16's retraction. |
| H19 (weight-knowledge contamination) | Word-search is a pure context task — the grid is in the prompt, model weights cannot help. Accuracy on L6–L7 is a direct proxy for context-read vs. weight-recall. |
| H23 (spatial state) | Level 6–7 performance before vs after RL step |

---

## Deferred / not implemented

These were explored as design directions before or during the WKV-loop rewrite
but are **not** in `rewards.py`/`train_wkv_loop.py` today. Kept here as a
reference for what to revisit, not as a description of current behavior.

**CLIPO contrastive reward** (arXiv 2603.10101) — push WKV states of correct
rollouts together, pull correct vs. incorrect apart, as a reward term
(InfoNCE on a WKV-state projection) on top of binary correctness. Made moot in
its original form: it was keyed on WKV state at `</think>`, and there is no
`</think>` position in the M-step loop anymore. Still not in `rewards.py`/
`train_wkv_loop.py` (this section's own RL loop) — but the re-adaptation this
used to say "not designed yet" now exists one layer over, in
`experiments/rl/train_think_distill.py`'s `_clipo_contrastive_loss`
(2026-08-21/22): keyed on `i == M_eff - 1` (the last/commit phase's
student/teacher representations), in-batch InfoNCE across examples instead of
correct-vs-incorrect rollouts (distillation has no incorrect rollout to
contrast against). Porting this from the distillation script into the actual
RL loop is part of the generate_rollout ThinkChain port this doc's Phase 3
section flags as not yet done — not a separate open design question anymore.

**L_KVB auxiliary SFT loss** (arXiv 2602.21204) — teach think-span tokens to
carry KV structure the WKV state absorbs cleanly. Doesn't apply without an SFT
phase (see §Step10 SFT — skipped entirely), and doesn't have an obvious
M-step analog (there's no token to attach a KV-consistency loss to during
silent steps). Not revisited since the WKV-loop rewrite.

**Entropy-weighted per-token GRPO advantage** (arXiv 2508.04349, GTPO/GRPO-S) —
weight gradient by per-step entropy inside the loop, focusing updates on
high-uncertainty steps. Compatible in principle with the current M-step loop
(steps replace think-tokens as the unit), not implemented.

---

## Token orientation: the primary curriculum axis

N* (number of state-refinement passes) is a derived observable, not the
primary design axis. The curriculum is actually organised by **token
orientation complexity** — the set of directions each grid-cell token must
push the WKV state to encode its spatial position:

| Orientation | WKV update direction requirement |
|-------------|----------------------------------|
| H_LR | left→right monotone; simplest (matches pre-training direction) |
| V_TD | must modulate per-row; orthogonal to left→right sweep |
| H_RL | right→left; reverses normal causal ordering |
| V_BU, diagonals | compound: both row AND column counter-flow simultaneously |

The J-lens `stable_rank` metric (measured at ~1.15–1.30 on G1h base, see
`results/jlens_base_vs_step8.json`) shows WKV heads are near rank-1 —
almost all update energy concentrates in a single direction. Word-search RL
should force stable_rank upward for mid-to-deep layers as the model learns
to encode multiple spatial directions. Track stable_rank at L4, L16, L28
before and after RL as the primary "orientation diversity" signal.

Trajectory diversity reward (proposed by icophy 2026-08-13) is the natural
GRPO complement: reward rollouts where consecutive WKV update directions are
distinct (cosine distance > threshold), penalise rollouts where every token
update collapses to the same direction. This prevents narrow-casting at the
RL level, not just at the SFT level (where H12b.i slot-entropy operates).

---

## ByteAdapter — tokenizer-free word-search (closed, not "planned" —
## header corrected 2026-08-23 to match this section's own body below)

**Motivation.** WorldTokenizer creates two problems for grid tasks (see below).
ByteAdapter eliminates both by replacing the 65k-vocab embedding with a 256-entry
byte-level embedding trained from scratch, keeping the WKV backbone frozen.

**Architecture.** `experiments/byte_adapter/byte_adapter.py`:
- `ByteEmbed`: `nn.Embedding(256, model_dim)` — trainable, ~262K params
- Frozen G1d/G1i WKV backbone (patched via `emb.weight[:256]` replacement)
- `ByteHead`: `nn.Linear(model_dim, 256)` — trainable, ~262K params
- Total trainable: ~524K (vs 410M backbone)

**Task format:** `--format nsp` (no-spaces) + byte tokenizer.
`STONE` → `[83, 84, 79, 78, 69]` — 1 byte per letter, zero asymmetry.
Compare: WorldTokenizer gives `['ST', 'ONE']` (positions destroyed).

**Probe results (G1d 0.4B, random ByteEmbed, 2026-08-16):**

| Layer | World norm | Byte norm | Ratio |
|-------|-----------|-----------|-------|
| L0    | 3.6–7.6   | 5.4–13.4  | 1.0–3.7× |
| L4    | 2.7–13.8  | 1.7–2.7   | **0.14–0.26×** |
| L16   | 12.6–35.1 | 12.2–17.0 | 0.35–1.35× |
| L23   | 31.8–60.4 | 30.6–35.2 | 0.53–0.96× |

L4 is the bottleneck: random byte embeddings activate it at only 14–26% of
World norm. L4 is the G1 think-routing layer (think_geometry L4=1.68×). The
ByteAdapter training objective: bring L4 byte/World ratio to ~1.0 on nsp grids.
L16/L23 already converge (0.5–1.0×) even with random embeddings.

**RL integration:** `--byte-adapter` flag in `train_wordsearch.py`.
THINK_CLOSE detection switches to byte sequence `[60,47,116,104,105,110,107,62]`
(`</think>` UTF-8). See `experiments/rl/rollout.py:ByteTokenizer`.

**Status:** code ready, training blocked on GPU.
**nsp eval result (G1d 0.4B, 2026-08-16): 3.6%. Closed.** Random ByteEmbed without
training converges at chance level on word-search — L4 activation deficit confirmed
(14–26% of World norm). RL proceeds on sp (space-separated) format exclusively.
ByteAdapter training is a separate GPU experiment, not on the RL critical path.

Note: `rollout.py` references in this section are the old design. Current rollout
lives in `experiments/rl/wkv_loop.py`; no `</think>` token detection needed.

---

## WorldTokenizer asymmetry in grid tasks

The RWKV WorldTokenizer creates a systematic positional asymmetry in
word-search grids that the model must overcome:

```
'A B C D E' → ['A', ' B', ' C', ' D', ' E']
```

Column 0 of every row receives a **bare letter token** (`A`). All other
columns receive a **space-prefixed token** (` B`, ` C`, …). Row boundaries
reset this: the first cell of each new row is again bare.

Consequences:
1. **Column-0 vs. column-N asymmetry.** The same letter `B` has two
   tokenisations depending on its column. The model must learn that `B` and
   ` B` refer to the same alphabet letter at different positions — a
   non-trivial induction for a model that reads left-to-right.
2. **BPE merging in prompts.** The target word in the prompt
   (`Find STONE`) tokenises as `['ST', 'ONE']`. In the grid, `S` appears as
   `S` or ` S` depending on column. The model must bridge BPE-merged
   mention → per-letter grid scan.
3. **H_RL (right-to-left) penalty.** For rightward words the first letter
   is in the rightmost column — it carries a space prefix — while the last
   letter is in column 0 — bare. This inverts the token-type gradient
   relative to reading order, which may be a key source of L3 difficulty.

These are not bugs to fix (WorldTokenizer is fixed); they are structural
constraints the RL curriculum must accommodate. The H_LR→H_RL progression
in levels 1→3 is partly a token-type-reversal training, not only a
directional-attention training.

---

## Switch-GRPO (arXiv 2606.13106) — reopened by ThinkChain, was "role unclear"

Source: arXiv 2606.13106 "Switchable Latent Reasoning."

**This section's original verdict (below, kept for the reasoning) is now
reopened, not settled, by the ThinkChain port this doc's Phase 3 section
flags as pending.** The verdict was: Switch-GRPO's contribution — a
well-defined policy ratio at *explicit block boundaries* — "solves a problem
the M-step design doesn't have," because the old self-feed loop's boundary is
just "the loop exited," not a tagged pair. **ThinkChain's boundaries are the
opposite of that**: `M` explicitly-distinct, discretely-indexed phase markers,
each with a definite start/end — exactly the kind of explicit block structure
Switch-GRPO's ratio mechanism attaches to. Once `generate_rollout` uses
ThinkChain markers instead of the old feed modes (the not-yet-done port), this
paper's boundary-token machinery becomes the natural candidate for keeping
GRPO's policy ratio well-defined over the (still non-emitting) phase
positions — read it properly before that port, not filed as background any
more.

Original obsolete framing, for the historical record: this section was
written as "Phase 4 extends Phase 3's visible `<think>` tokens to latent
blocks" — true when Phase 3 emitted visible think-tokens, false once the
self-feed WKV-loop replaced that with no emitted tokens at all (nothing to
progressively replace with `<latent>`). The mechanism sketch below still
describes the paper's own `<swi>`/`</swi>`/`<latent>` vocabulary — read it as
the paper's mechanism, not as a noesis-vocabulary mapping (there isn't one
yet).

**Three-token vocabulary extension:**
- `<swi>` — enter latent block
- `</swi>` — exit latent block
- `<latent>` — latent placeholder (no embedding; previous hidden state h_{t-1} is the input)

**Switch-SFT (prerequisite, after word-search RL convergence):**

Phase 1 SFT: tag high-entropy sub-spans of existing think-span outputs with
`<swi>`/`</swi>`. Use Shannon entropy of token distribution to identify
uncertain positions — these are the natural candidate latent positions.
Train with standard CE on the annotated corpus.

Phase 2 latent curriculum: progressively replace text inside `<swi>` blocks
with `<latent>` placeholders. Parallel schedule:
```
n_m(k) = c · min(k, |span_m|, K_max)
```
per-span latent count grows each stage. One-shot replacement kills the block
(model exits in 1 step without K_min constraint).

**Switch-GRPO objective:**

Hidden-state injection is deterministic given preceding text → rollout
likelihood factors over visible positions only. Policy ratio defined at
`<swi>`, `</swi>`, visible answer tokens. `<latent>` positions contribute
no policy-gradient term. Gradient propagates through latent computation via
segmented backward.

Reward = ±1 correctness + ±1 tag-format (valid `<swi>`/`</swi>` pairs) +
{0,1} latent-usage (bonus when correct answer used the latent path).

**Noesis mapping — not built yet, now a real candidate (see above), not n/a.**
The table this section used to have mapped `<swi>`/`</swi>`/`<latent>` onto
`<think>`/`</think>` tokens that don't exist in the WKV-loop vocabulary —
removed rather than patched, since nothing consumed it at the time. The
mapping to build now: `<swi>`/`</swi>` onto ThinkChain's entry-cue/phase-marker
boundaries (each phase already IS a discrete, indexed block — the exact shape
Switch-GRPO's ratio mechanism wants), not against the old self-feed loop's
fuzzy exit-reason/M mechanics.

K_min (minimum latent dwell) is still a real design point if a boundary-token
mechanism gets revisited: without it, a trained latent-block model exits in
one step. Start K_min = 4; ablate (arXiv 2606.13106 Appendix I).

---

## Connection to A4 (truth-system heads)

After step-10 RL locks the backbone, A4 trains small detector heads on
top of the frozen weights. Word-search RL contributes two things to A4:

1. **H21 / H22 re-run.** The step-9 premise-validity (H21, F1=1.00) and
   attribution (H22, F1=0.947) results are on the G1h base. Re-run on the
   step-10 G1i backbone to confirm the signal survives RL.
2. **H16 gate head (A4 step).** If word-search RL confirms the model is
   using think-phase for spatial scanning, train the gated-emit MLP head
   (H16) on the RL-tuned backbone rather than the SFT-only backbone.
   The RL backbone is a better substrate because it has already learned
   when think-space is useful.

---

## Step10 SFT — skip decision (icophy, 2026-08-14)

Source: icophy Discord message, 6 months production RWKV-6 state work.

**Conclusion: skip SFT warm-up, go direct RL on G1i.**

Reason: SFT before RL reduces variance on *which behavior* RL reinforces — it does not
improve the model's ability to do the task. For G1i at 0/56 on word-search there is no
existing behavior to exploit as shortcut. This is exactly the case where skipping SFT is cleaner.

**Mechanism of SFT-less RL risk:** GRPO on binary reward latches onto surface shortcuts early. State encodes the shortcut, not the reasoning pathway. The shortcut's verifiable output is indistinguishable from genuine task completion in reward signal.

**Why 0/56 baseline protects us:** no spurious partial-success correlation → nothing to pre-remove. Icophy: "The counterargument for skipping SFT is stronger here than in tasks where the model already achieves non-trivial baseline performance."

**Mandatory checkpoint after first RL epoch:**
R-lens probe on non-task axes (e.g. narrative, arithmetic) before continuing curriculum. If R-lens returns near-chance probe accuracy on non-task axes while word-search performance is high → state is encoding shortcut geometry, not reasoning. Stop and investigate before advancing curriculum.

**L_KVB:** skip for now (SFT-only loss, no SFT phase). Revisit if full-FT SFT is added later.

---

## Integrated RL loop — instrument map

Actual current implementation (rewritten 2026-08-17). No `verl`, no
TransformerLens, no external CLIPO repo — this is a custom loop built directly
on RWKV-PEFT's model class, not the tool stack the original plan assumed:

```
G1i base checkpoint — models/rwkv7-g1i-2.9b-20260805-ctx16384.pth (2026-08-05)
│
├── [NO SFT warm-up — skip, see §Step10 SFT decision]
│
└── train_wkv_loop.py — main entry point, GRPO training loop
    │
    ├── loader.py::load_rwkv7 — dual-mode: peft (GPU, differentiable,
    │     args.peft="none" hardcoded → full FT only, no LoRA toggle) or
    │     blink (CPU, inference-only, used for smoke tests)
    │
    ├── Per batch (design default batch=4; real 16GB-T4 runs use batch=2,
    │   │     G=8 rollouts each, T=0.7 — see §Known risks #9 for why):
    │   ├── wkv_loop.py::generate_rollout — M-step loop per rollout,
    │   │     exits on plateau/commit/M_max (see §RL design)
    │   ├── rewards.py::compute_wkv_loop_rewards —
    │   │     r = r_correct − β·M − γ·Σ ReLU(ΔH_t) [+ δ·stability, off]
    │   │         [+ ζ·info-density, grafted 2026-08-19, off by default]
    │   │     gate_on_correct (default on): β/γ/δ/ζ only apply when
    │   │     r_correct > 0 — see §Mechanisms and their relationships
    │   ├── grpo.py::compute_advantages — group-relative advantage from
    │   │     the 8 rollouts per prompt (this file predates the WKV-loop
    │   │     rewrite, reused unchanged — GRPO's advantage math didn't
    │   │     need to change when the reward did)
    │   └── train_wkv_loop.py::wkv_grpo_loss — PPO-clip surrogate
    │         (clip_eps=0.2) + KL penalty (kl_coef=0.01)
    │         [+ L_state motion/curvature term, grafted 2026-08-19, off
    │           by default — see §Mechanisms and their relationships];
    │         replays prefill+loop+answer per rollout to get gradients
    │         (_recompute_wkv_log_probs — see §Known risks, this is the
    │         "32 forward passes per update" cost point)
    │
    ├── monitor.py::TrainingMonitor — per-batch health flags: SHORTCUT,
    │     NO_COMMIT, ECHO (both added 2026-08-19), HACKING (sliding
    │     10-batch window), STATE_COL, MODE_COL
    │
    ├── checkpoint.py::save_checkpoint/load_checkpoint — directory-based,
    │     one file per trainable tensor (extracted from train_wkv_loop.py
    │     2026-08-19; see §Known risks #9 for the OOM-kill this replaced)
    │
    ├── corpus.py::CorpusScheduler — curriculum advance/drop (see §RL design)
    │
    ├── probes.py — inline stable_rank + IPC, run periodically without a
    │     separate process (shares the already-loaded model)
    │
    └── vm_watchdog.py::VMWatchdog — Selectel 24h deadline safety net,
          checkpoints and exits cleanly before forced VM termination
```

No R-lens/TransformerLens hook is wired into this loop — `monitor.py`'s
STATE_COL/MODE_COL flags are the current stand-in for "is the model
collapsing to a shortcut," not a stable_rank probe on a held-out axis. The
R-lens-after-checkpoint-1 check the old diagram described is still a good
idea; it just isn't automated here yet.

---

## Mechanisms and their relationships

Living section, started 2026-08-19 — fill in as mechanisms are added or
their interactions get understood, not a one-time writeup.

**Reward-shaping terms** (`rewards.py::compute_wkv_loop_rewards`) —
β (effort/M penalty), γ (entropy-rise penalty), δ (stability bonus,
default 0), ζ (info-density reward, default 0, grafted 2026-08-19). All
four gated by `gate_on_correct` (default on): shaping only ranks *among*
already-correct rollouts, never lets a wrong-but-"efficient" rollout
outscore a wrong-but-verbose one. Found necessary the hard way — an
*unconditional* shaping term dominates GRPO's within-group gradient
whenever a whole group scores identically on raw correctness (near-zero
accuracy, common early in training), because that's the only source of
reward variance left for the group. This is the actual mechanism behind
`g1i_real_run6`'s collapse (100% commit-at-M=2 at step20 → 100%
M_max-saturated boilerplate by step50) — see `docs/rl-track.md`'s own
history and `project_noesis_forge_bptt` memory for the full bisection.

**δ vs. L_state — opposite directions, don't stack blindly.** δ rewards
LOW WKV state motion (`mean_stability < threshold` → bonus). L_state
(`train_wkv_loop.py::_recompute_wkv_log_probs`'s
`l_state_delta_weight`/`l_state_kappa_weight`, grafted 2026-08-19, both
default 0) rewards HIGH motion and curvature (inverted-SFA, revived from
the old A1-pilot SFT stack's loss design). Turning both on together
without resolving which direction is wanted would fight itself.

**ζ inherits the same correctness-trust dependency as β/γ/δ.** It's
gated by `gate_on_correct` the same way, which means it's only as honest
as `r_correct` is. The 2026-08-19 rubric echo-exploit (a model echoing
its input verbatim got scored correct=True whenever the target digit
happened to appear in the copy — 45/153, ~29%, of correct=True rollouts
in one run) would have let ζ reward "efficiency" on rollouts that never
actually solved anything, same as β/γ/δ would have. Partially mitigated
by the rubric fix (`_score_correct` now anchors to the first number
token, not "anywhere in the text") and the new `ECHO` monitor flag
(structural, rubric-independent — catches the behavior even for rubric
families the anchor fix doesn't cover, e.g. word-lookahead rubrics where
a genuinely correct answer can legitimately overlap the prompt).

**gate_on_correct ↔ NO_COMMIT.** Gating shaping is the fix that (so far,
pending the run8 step150-200 verification gate) prevents the M_max-
saturation collapse `NO_COMMIT` was added to catch. Not proven causally
independent of other same-day changes yet — both landed close together.

**Open question, not yet tested — gate_on_correct as a "know vs. don't
know" signal.** Because shaping only activates on already-correct
rollouts, the gradient a rollout receives differs qualitatively depending
on whether the model got the task right: correct rollouts get a
secondary, effort-shaped signal; wrong rollouts get pure ±1 with no
further discrimination. Whether this actually trains the model to
distinguish "I know this" from "I don't know this" in any usable,
measurable sense is a real prediction, not a confirmed finding — closest
existing hypotheses are H21 (premise-validity readout — different axis,
detects malformed *questions*, not answer confidence) and H19 (weight-
knowledge contamination — different question, about corpus-leaked facts,
not calibration). Neither is a clean match. Candidate for its own
hypothesis once there's a stable checkpoint worth probing this on.

## Appendix: ThinkChain run history (Phases 1/1.5/2 debugging log)

**Current status and the 4-stage plan now live in §Track status at the top of
this file — this section is the detailed run-by-run history behind it, not
the entry point.**

**Word-search RL (Phase 3) training is paused, by explicit decision, not by crash or budget.**
The M-loop content decoder (a one-off diagnostic, not kept in the repo)
showed that even a directly-verified-clean G1i RL checkpoint's internal
M-step was never real task content — just chat-template scaffolding
(`"\n\nAnswer"`, `"\n\nAssistant"`), and this degraded *further* toward
contest-boilerplate under only 17 steps of further M_max=1-constrained
training. `mean_M` was also found frozen at exactly 2.0 across every
step of one full run — no variance to correlate "amount of internal
work" against quality at all. Conclusion: RL has nothing to select for
inside the M-loop when the loop has no content to begin with — tighter
M_max or reward tuning treats a symptom, not the cause. "RL запускать
рано, модель не готова" — RL resumes once the loop reliably carries
genuine, task-relevant content, not on a fixed calendar date.

**Prerequisite: `experiments/rl/train_think_distill.py`** — not RL (no
GRPO, no reward, no sampling). Teacher pass reads real explicit
`<think>` text and its resulting WKV state (layers L12/L16/L20, weights
from `training/state_reg.py`'s A0.5 profile) becomes the target; student
pass reads only the prompt, then self-feeds (argmax → feed back,
repeated) to approach that target, before being scored on the real
answer via CE. β/γ/δ/ζ all stay at their existing off defaults
throughout this phase.

**"Latent overshooting" (M phases, arXiv 2007.14535's J^k_KL).** The
teacher's think span is sliced into `M` chunks; student phase `i` is
scored against teacher-state-after-chunk-`i`, not one distant endpoint.
Each phase self-feeds a *fixed* `--max-phase-tokens` (default 8,
deliberately generous, not squeezed — compression/efficiency is already
RL's job via β·M and the entropy-plateau exit; this phase's job is
correctness and stability, not budget). `M=1` collapses to a single
phase.

**Real experimental history, in order, all on
`training/corpus_open/matrix_tasks.jsonl`-derived data (65k tasks, 7
balanced categories) — not guessed, each run's actual failure inspected
before the next attempt:**

1. **Teacher text from `step9_combined_train.pt` (existing step9 SFT
   corpus) — mode collapse.** Mechanically stable (a state-loss runaway,
   25→4493 over ~10 steps, was fixed with a per-layer clamp at 100,
   `training/state_reg.py`-style — same convention as its existing
   `layer_loss.clamp(min=-10.0)`), but the trained self-fed token
   converged to `"Binary: ..."` on *every* prompt tested, including
   wordsearch/arithmetic/XOR with zero relation to binary content.
   Root cause confirmed directly: 235/269 (87%) of that corpus mentions
   "Binary" — the model took the cheapest average-loss shortcut across
   an 87%-skewed target distribution, not a genuine per-task response.
2. **step9b-e1 as a balanced-corpus generator — abandoned by design,
   not by failure.** Planned to regenerate a balanced self-CoT corpus
   via step9b-e1 (a G1h-lineage checkpoint confirmed capable of genuine
   `<think>` reasoning) over `matrix_tasks.jsonl`. Overridden: WKV state
   differs materially model-to-model, so borrowing another checkpoint's
   generated text imports a real distortion into the distillation, not
   just a style difference. Separately, the real pre-RL DE table (§H24
   below) showed step9b-e1 has the *worst* DE (0.15, 270 tok/correct)
   of anything measured — G1i's own raw chatwrap output is already more
   concise (DE 0.27, 154 tok/correct) — so borrowing step9b-e1's text
   would also import a specifically bad-DE habit, not a neutral one.
3. **G1i as its own generator, SFT-style `<think>` prompting —
   confirmed non-functional.** G1i was never SFT'd on explicit
   `<think>` spans (only the step9/step9b, G1h lineage, was); prompted
   the same way, it produced **zero** `<think>...</think>` output in a
   tiny test (5/5 "no `<think>` found," not garbage-but-present —
   structurally absent, since it never saw the convention in training).
4. **Procedural G1i-native warm-up corpus, M=1, one self-fed token per
   example — diverged at step ~287/585.** Built 585 short (1-sentence,
   2-3 varied phrasings per category, not one universal template)
   `<think>` examples directly from each task's own ground-truth answer
   — zero foreign-model text. `state_loss` ran clean to ~step 280 (even
   *dropping* early on), then pinned at the 100.0 clamp ceiling for
   13+ consecutive steps while `answer_ce` climbed in lockstep
   (2.5→3.5→8.7→11.8) — real degradation, not noise. Working diagnosis:
   short think content means the teacher's post-think state is
   dominated by the *prompt's* structural diversity (7 very different
   task shapes: letter grid vs. crossword vs. arithmetic table) rather
   than smoothed by long homogeneous think text (as in run 1) — high-
   variance targets against a 1-token student channel.
5. **M=2 chunked overshooting, still 1 self-fed token per phase —
   diverged later (~step 321) but still diverged.** Confirmed the
   divergence isn't purely about target distance (M=2's chunked
   checkpoints are closer than M=1's single endpoint) — it's also a
   token-budget mismatch: each teacher chunk represents several real
   tokens' worth of state update; one self-fed token is a much smaller
   update, asking the student to close the same distance in far fewer
   steps.
6. **Fixed 8-token phase budget, M=1 — completed clean, 2026-08-20
   (`overnight2` run, LoRA r=32/alpha=64).** Ran the full 1950/1950
   steps, no divergence, no clamp-pinning at any point. First run to
   survive past the ~280-330 step range where every prior attempt
   broke. `answer_ce` dropped from ~9.9 (step 1) to a ~2.5-3 plateau
   and stayed there for the remaining ~1600 steps — real early
   adaptation, no further improvement after.

7. **Diagnostic decode (4 fixed prompts — matrix addition, wordsearch,
   arithmetic sequence, XOR; `_diag_think_content.py`, one-off, same
   convention as `_debug_modecol.py`) reached for the first time,
   2026-08-20 — real qualitative shift, but no convergence.** Baseline
   (pre-`overnight2`) M-loop think-tokens are pure chat-template
   scaffolding ("\n\nAnswer", "\n\nAssistant") on every prompt, never
   task content — matches the earlier finding in the "Latent
   imagination" section of `docs/community-map.md`. Post-`overnight2`,
   think-tokens are genuinely task-domain (digits on numeric tasks,
   letters on the word task; one case, the arithmetic sequence, landed
   on the mathematically correct missing digit). But the model doesn't
   converge to a clean single answer — it repeats/loops instead
   ("STARSTSTSTAR", "000000000000", "12800128012801280..."). LoRA
   moved *what* the loop thinks about, not *whether it stops*.

8. **Full-FT continuation attempt #1 — immediate regression, not a
   slow drift, 2026-08-20.** Merged `overnight2`'s LoRA delta into the
   base weights (`_merge_lora.py`, PEFT's own per-layer `.merge()`,
   verified key-for-key identical to the un-merged checkpoint's
   behaviour on the 4-prompt diagnostic). Continued full-FT at
   `--lr 3.3e-5` (1/3 of the LoRA run's rate — full-FT here had
   previously diverged "regardless of M/phase-budget tuning" per this
   script's own `--lora-r` doc string, and LR itself was never part of
   that earlier sweep). Result: `answer_ce` never improved past its
   starting ~6 across 238 steps (LoRA had reached ~3.3), and
   `state_loss` broke cleanly at step ~180 (stable 28-47 for 150+
   steps, then a sharp climb to the 80-90s and clamp-touches at 100).
   Isolated, no-gradient comparison on the same first 5 corpus examples
   (`_check_ex5.py`-adjacent script) confirmed the merge itself was not
   the cause — LoRA and merged-model losses matched to 3-4 decimal
   places. The actual break: `answer_ce` spiked 2.5→18.0 at step 6,
   nothing to do with the later state_loss clamp event. That example
   was a 9x9 wordsearch grid (176-token prompt, 2-token answer
   "GRAZE") — a task category the model has ~0% baseline accuracy on
   (`WorldTokenizer` column-position asymmetry, already documented
   above). No gradient clipping existed anywhere in this script.
   LoRA's adapter bottleneck (1.58% of parameters) had been *implicitly*
   capping how far any one example's gradient could move the model;
   full-FT has no such bottleneck.

9. **`--grad-clip` added (`clip_grad_norm_`, default max-norm 1.0),
   full-FT continuation attempt #2, 2026-08-20 — holding.** Same
   wordsearch example at step 6: raw grad_norm=1519 (confirms it really
   was that large), `answer_ce` only reached 5.7 and *recovered to 2.2
   the very next step* — contrast attempt #1, where the same event
   never recovered. Raw grad_norm routinely reaches 1000-10000+ per
   step even on ordinary examples (step 1, alone, was 2938) — clipping
   is doing real, constant work, not a rare edge-case correction.
   Survived a VM reboot (Selectel's 24h reclaim) via the existing
   SIGTERM checkpoint handler with zero lost steps, resumed cleanly
   from `ckpt_step000429` — `state_loss` had been sitting at 60-69
   (elevated vs. the ~40 baseline, but nowhere near the 100 clamp)
   through the entire previously-fatal step-150-300 window and beyond.
   Not yet run to completion; the retroactive implication worth
   flagging: run 5 above (the failed M=2 attempt, 1-token phase budget)
   never had gradient clipping either, so that divergence may have had
   two compounding causes (budget mismatch *and* unclipped outlier
   gradients), not only the one already diagnosed.

10. **`full2` closed at step 1000 (flat Kalman plateau), eval reveals a
    real root cause, 2026-08-20 — likely explains item 7's "repeats/
    loops" finding too, not just a new bug.** M-sweep was deferred in
    favor of a plain chatwrap eval first: **0/48**, every category,
    every response degenerates into a repeated-token loop
    (`"000000000..."`, `"932932932..."`, digit/word cycles) regardless
    of the task's actual expected answer type. A native-format (no
    chat-wrap, no M-loop, greedy) control ruled out a harness/template
    artifact: on a wordsearch prompt the *first* generated token was the
    genuinely correct answer (`"CAT"`), then the model cycled through
    unrelated words indefinitely, never emitting EOS in 80 tokens; on a
    bit-decoding prompt it collapsed to `"0000..."` from the first
    token. **Root cause, confirmed by direct inspection (not inferred):**
    `training/tokenize_plain_cot.py::_render()` never emits an EOS
    segment after `answer` (`segs.append((answer, 1, 0))` is the last
    segment) — checked against 5 real rollouts in
    `training/tokenised/g1i_warmup_v2_train.pt`, none end in token 0.
    **The model was never given a single training example of "stop
    after the answer."** State-distillation quality (state_loss/
    answer_ce trending flat-but-not-diverging) said nothing about this —
    a plateau in those metrics is consistent with a stable *but
    degenerate* attractor, not necessarily a healthy one. **Take for the
    record:** state distillation without terminal (EOS) supervision on
    the answer produces looping in the visible output even when the
    model's first tokens are genuinely correct — greedy/low-temperature
    decode past a corpus's trained-answer-length boundary is undefined
    territory for any autoregressive LM, and degenerates the same way
    regardless of whether the underlying reasoning/state mechanism is
    otherwise sound. This reframes item 7 above: the M-loop's own
    "repeats/loops instead" finding was very likely the same missing-EOS
    problem showing up internally first, not independent evidence that
    the M-loop specifically lacks genuine content.

11. **EOS fix verified end-to-end, 2026-08-20/21 — `tokenize_plain_cot.py`
    now emits a CE-supervised EOS segment after `answer`. From-scratch
    LoRA run (r=32/alpha=64, base model, corrected corpus, 1950/1950
    steps, no divergence) confirms the fix: `answer_ce` drops from 6.82
    (step 1) to a stable ~1 level and stays flat (Kalman slope
    -0.014±0.018/step, 1σ CI includes zero) — genuinely healthy, not
    just non-diverging. `state_loss` climbs toward the 100.0 clamp
    (level ~89 by the end, real sustained slope +0.24 → +0.15/step
    through steps 600-1000, not noise) — same shape as the already-
    documented Run 6 pattern (`project_noesis_think_distill_experiments`
    memory): state distance from the teacher target grows while answer
    quality stays good, "a genuinely different, self-consistent way to
    solve tasks, not failure." Native-format sanity check on the final
    checkpoint: wordsearch → `"CAT"` + clean EOS (exact match to the
    200-step control run); bit-decoding → `"01"` + EOS — terser than the
    control's `"01 00 10 00"` + EOS, not yet the correct full decode
    (`"HI"`), but still a clean stop on real content, no loop.

12. **First real per-token state-trajectory data
    (`experiments/rl/state_trajectory_probe.py`, new — records WKV
    state-norm at every individual token, prefill and self-feed alike,
    unlike training's batched prefill which only exposes the end-of-read
    state) — answers the open "how does the whole system change per
    token, read vs generate" question raised earlier this session, and
    finds something unexpected.** Same 4 fixed prompts as item 7's
    diagnostic decoder, run against both base G1i (untrained) and the
    item-11 checkpoint, layers L12/L16/L20 (the A0.5 set). Base model:
    state norm during the 40-token read phase stays in a modest, stable
    band (e.g. L20 ≈ 135-138 across all 4 prompts) and drifts up
    *slowly and smoothly* during a forced 40-token generate phase (never
    reaches EOS in that budget — confirms base has no stop behavior at
    all, consistent with item 10's fix being training-induced, not
    innate). **The item-11 checkpoint is dramatically different at the
    exact same read position, same input tokens:** end-of-prompt state
    norm is 3-4x the base model's, on every prompt and every tracked
    layer (L20: 135→512-647; L16: 165-180→342-464). Then the generate
    phase (only 2-3 tokens before EOS, matching item 11's terse-answer
    finding) shows the norm barely moving further (e.g. 646.6→646.8) —
    the state is already near its endpoint by the time generation
    starts, not still travelling toward it. **This directly explains
    item 11's climbing `state_loss`, not as a separate mystery:**
    `state_loss` is a raw L2 distance, and a 3-4x growth in the whole
    state's magnitude scale mechanically inflates any absolute distance
    measured against it, independent of whether the *direction* the
    model is moving is coherent (which the healthy `answer_ce` suggests
    it is). **Also a direct, now-answered instance of the open
    instrumentation gap `docs/community-map.md` §3.1 flagged**
    ("instrument state_norm_layer_L_step_t... if any layer's norm grows
    >10x over baseline, add an explicit norm-anchor loss or clip") — at
    3-4x this isn't yet at that alarm threshold, but it's a real,
    substantial, training-induced amplification worth tracking across
    future runs (especially the planned full-FT continuation and any
    M≥2 Dreaming-cycle work), not something to wave off because answer
    quality currently looks fine.

13. **Per-token *delta* (not just magnitude), computed post-hoc from
    item 12's already-collected trajectories — the growth isn't diffuse,
    it's a sharp late-read acceleration.** `‖S_t‖` alone can't
    distinguish "big but static" from "still actively moving"; `‖S_t -
    S_{t-1}‖` per token can. Base model: read-phase deltas oscillate
    around a small, stable band the entire prompt (`|mean|≈2.0-2.2`,
    `max_abs≈7.95` across all 4 prompts — the first ~12 deltas are
    literally identical across prompts, since they're computed over the
    shared system-prompt prefix). The item-11 checkpoint's read-phase
    deltas are elevated from the very first shared-prefix token (18.82
    vs. base's -5.15, same tokens) and then, specifically in the last
    ~4-6 tokens before the `<think>` marker (i.e. right at the
    read→generate transition, not spread evenly through the prompt),
    spike hard: arithmetic_sequence hits 66.46 in one single-token step;
    xor hits 35.12; wordsearch's last 5 deltas alone are
    12.0/22.3/2.9/11.8/17.8. The 1-3 generate-phase tokens that follow
    continue at this same elevated magnitude, not a return to the calm
    mid-prompt regime. Reads as the model concentrating a burst of
    state-writing right at the moment it must commit to an answer,
    rather than uniformly inflating state throughout the read — a much
    more specific, testable claim than "training amplified state norm"
    (item 12's framing), and the first real answer (not yet a full one —
    4 prompts, 2 checkpoints) to the standing open question about what
    each individual token actually does to the system. Not yet checked:
    whether this late-read spike is present in exactly this shape
    across a larger prompt sample, or specific to these 4 fixed
    diagnostics.

**Superseded 2026-08-21/22/23 — the self-feed loop above was replaced,
not just re-tuned, and the "Dreaming cycle" plan below was overtaken by
a locked 4-stage structure.** Kept verbatim above for the debugging
history (the self-feed loop's actual failure mode is real and
instructive — see below), not because it's still the design.

**Why the self-feed loop was replaced, not fixed.** All N self-feed
steps apply the SAME transformation to state, and the input at each
step (a self-generated token) has no signal telling the model which
phase it's in. Since a token's R/K/V/decay are derived from that
token's own embedding, and self-generated tokens grow more similar as
the model gets confident, the per-step transformation homogenises and
the loop converges toward a fixed point instead of doing M genuinely
different units of work — the exact mechanism behind two things this
section already documented as separate bugs (dynamic-phase-stop's
entropy-plateau firing increasingly early; the need for an explicit
norm anchor against unbounded state drift).

**This paragraph is mechanics-reasoned, not yet confirmed by the direct
measurement built for it — checked 2026-08-23, worth being honest about.**
`delta_cos_prev` (`state_trajectory_probe.py`, exactly built to test
"does the loop's per-step delta converge toward a single direction, cos
→ 1") was computed against the one existing local run
(`experiments/_common/results/state_trajectory_step200.json`, 2026-08-22,
loop branch on the step200 checkpoint the old mechanism actually trained
on — the right checkpoint to ask this on). Result: inconclusive, not
confirming. 3 of 4 prompts hit EOS after only 2-3 self-feed steps (too
short a trace to show any trend); the one long trace (matrix_addition,
13 steps) sits roughly flat around −0.08 to −0.11, not drifting toward
+1 — consecutive deltas pointing in *somewhat opposing* directions on
average, not the same one. Doesn't refute the mechanism either (mostly a
sample-length problem, and a genuine fixed-point contraction doesn't
strictly require cos→1 the way a naive reading suggests — WKV's `v^T k`
write term keeps injecting content a pure decay-only contraction
wouldn't have). That same file's `chain` branch is not a usable
comparison point: step200 predates ThinkChain, so `--resume` fell back
to random, untrained markers there.

**Second data point, task #12's run on v3/step500, 2026-08-23 — still not
confirming.** `loop`'s `delta_cos_prev` stays negative on all 4 prompts
here too (e.g. xor: −0.086 → −0.195), same as step200. Caveat this time
runs the other way: step500 was trained *under ThinkChain*, not the old
self-feed mechanism, so `loop` here is genuinely off-distribution for
this checkpoint (there is no single checkpoint that is both
self-feed-trained and available to test cleanly — step200 is the
self-feed-trained one, step500 is ThinkChain-trained). Between the two
checkpoints tested so far, the specific `cos_prev → 1` prediction has
never shown up, on-distribution or off. Doesn't mean the qualitative
"self-feed is less stable" finding is wrong (n_phase_tok/state_loss/
grad_norm variance still clearly favor ThinkChain, measured separately —
see the run-6-vs-v3 comparison above) — it means this one specific
predicted *signature* of why isn't the right measurement, or the
homogenisation happens through a channel this metric doesn't see (e.g.
concentrated in specific channels/layers rather than the whole-state
vector cosine). Not chased further; flagging so a future session doesn't
treat this paragraph's causal story as measured fact.

**ThinkChain**
(`ThinkChain` class, `experiments/rl/train_think_distill.py`,
2026-08-21) replaces the self-feed loop with `M+1` explicitly distinct,
directly-learned embeddings (one shared entry cue, one per phase) fed
straight into WKV via `forward_stateful_embeds` — no self-feed token
loop at all. Mirrors what the teacher already has for free: `M_eff`
real, naturally-distinct text chunks, instead of asking the student to
manufacture distinctness from a homogeneous loop.

**Two bugs found and fixed in the rewrite itself (2026-08-22), same
day, before trusting a real run:**
- *Budget-matching silently dropped.* The first ThinkChain cut fed each
  phase's marker for exactly one step regardless of the teacher
  chunk's real length (up to 141 tokens in `g1i_warmup_v3`, median 65)
  — this repo's own history already needed this fix once before for
  the self-feed loop (see the M=2/1-token-budget divergence above).
  Fixed: each phase repeats its marker up to `chunk_lens[i]` times (a
  ceiling, not mandatory). Repeating a *constant* embedding isn't the
  loop-collapse failure mode above — R/K/V/decay depend on the input
  and its time-shift delta, and that delta is exactly zero from the
  second repeat onward (identical consecutive inputs), so this is a
  fixed transformation applied T times to the evolving state, not a
  self-referential one whose own output drifts.
- *Wrong stop criterion.* `--dynamic-phase-stop` was re-added reusing
  `wkv_loop.py::generate_rollout`'s readout-confidence check
  (`max_p`/entropy on the phase's logits) verbatim. A real run showed
  it firing after exactly 1 repeat on ~80% of steps — traced to
  `g1i_warmup_v3`'s templated answers (many start "Decimal: ...") making
  the *first answer token* >99.9% predictable regardless of whether the
  phase did any real work; the readout-confidence check answers a
  question that only makes sense for a real generated token stream,
  which a phase's internal readout isn't. Replaced with a state-delta
  criterion (exits once the WKV state itself stops moving, relative to
  its own norm) — see `experiments/rl/train_think_distill.py`'s
  `--dynamic-phase-stop` docstring for the exact formula.

**Corpus was also wrong, independently.** `--data`'s old default
(`step9_combined_train.pt`, 268 examples) has 85% of examples sharing
one answer-template first token ("Decimal") — confirmed, not a sampling
artifact. Switched default to `g1i_warmup_v3_eos_train.pt` (10 529
examples, 1409 unique first words) with a new best-effort
category-diverse batch sampler (`_CategoryBatcher`) so a batch doesn't
land in one category on a skewed corpus.

**Real run result (`g1i_think_distill_zlk_phase1_v3`, 500 steps, LoRA
r=32/alpha=64, M=1): the fixes worked, on a delay.** At step 100, free
generation (M=1, trained marker) was *worse* than the bare LoRA weights
with no marker at all (M=0) — e.g. XOR(1010,0110) answered `1010`
(wrong) at M=1 vs. a coherent-if-incomplete reasoning trace at M=0; "is
a flower or car bigger" answered `apples` at M=1 vs. a correct answer
at M=0. By step 500, this fully reversed: M=1 answered XOR correctly
(`1100`) and the flower/car question correctly (`car`), while M=0's own
answers had *degraded* over the same training (plausible cause: LoRA
reshaping weights toward "expect the phase marker" as normal operation,
making the bare M=0 path increasingly out-of-distribution for this
specific fine-tune — a real risk to watch on the planned full-FT
continuation below, where there is no LoRA bottleneck limiting how far
that drift can go). Kalman (standalone, full 500-step log): answer_ce
and cos_sim both plateaued by step 500 (further steps at this config
unlikely to buy more quality) while state_loss/norm_penalty/grad_norm
kept rising (real, not noise) — stop-and-reconfigure point, not a
stop-and-declare-done point.

**The 4-stage structure this section used to detail here now lives at the
top of the file (§Track status), including the 2026-08-23 revisions to Phase
2** (graduated advance/retreat instead of a one-shot snap-to-origin, a
smoothness-loss/curvature-loss mirror pair for rewind vs. phase markers, and
a Huginn-backtracking-style validation step ahead of building anything).
Not duplicated here to avoid two copies of a living design drifting apart —
this appendix stays a historical record of how the taxonomy was arrived at,
the top of the file is what to edit when the plan changes again.

---

## Known risks — unverified, found by code review 2026-08-17

None of these are known bugs; they're places the design leans on an
assumption that hasn't been checked. Listed here instead of silently, so the
first GPU session treats them as things to verify, not things already
settled.

1. **`_peft_forward_embeds` numerical equivalence — RESOLVED, verified
   2026-08-18.** `experiments/rl/test_forward_embeds_equivalence.py` run for
   real on a T4: single-token and multi-token (T=5) sub-tests both pass with
   **exact** match (`max_abs_diff=0`). `_peft_forward_embeds` does not diverge
   from `forward_infctx` — `expected`/`residual` feed-mode gradients flow
   through the correct computation. (Two real, unrelated bugs found and
   fixed getting the peft backend to load at all on this run —
   `TrainingArgs` missing `my_testing`, and `RWKV_TRAIN_TYPE` defaulting to
   `""` instead of `"infctx"` — see `loader.py`'s `_prime_env`.)

2. **`_recompute_wkv_log_probs` is not batched — confirmed real, still
   unbatched.** Called once per rollout inside `wkv_grpo_loss`'s nested loop
   — at `G=8`, `batch=4` that's 32 sequential replays of prefill + M-loop +
   answer per training update, each one token-at-a-time (no batching across
   the M steps or across rollouts). Confirmed as a real wall-clock cost
   2026-08-18 running the first actual gradient-update step on a T4 (G1d
   0.4B, G=4/batch=2/M_max=4 — 8 rollouts, each already several minutes of
   CPU-bound sequential replay before the first `optimizer.step()`). Not
   fixed — the actual lever for a first real training run, more than model
   size or GPU choice.

3. **`head_size = 64` hardcoded in `loader.py::_load_blink`** — correct for
   every checkpoint currently in use (G1d/G1h/G1i, all RWKV-7 "Goose"), not
   derived from the checkpoint. Could be read from `blocks.0.att.r_k.shape[1]`
   instead (verified 2026-08-17: G1d r_k is `(16, 64)`, G1i is `(40, 64)` —
   `(n_head, head_size)` in both). Low priority — would only matter for a
   differently-shaped checkpoint, none exists yet.
   ~~`dim_ffn` formula in `_load_peft`~~ **verified 2026-08-17: not actually a
   risk.** `RWKV_CMix_x070` (`rwkvt/rwkv7/ffn.py`) hardcodes its hidden layer
   as `args.n_embd * 4` directly — it never reads `args.dim_ffn` at all. The
   `int(((n_embd * 3.5) // 32) * 32)` formula (copied from `train.py`'s own
   CLI default, itself inherited from RWKV-5/6's `ffn.py`, which *do* use
   `dim_ffn`) computes a value that doesn't even match either G1d (3584 vs.
   actual 4096) or G1i (8960 vs. actual 10240) — and it doesn't matter,
   because nothing in the RWKV-7 path reads it. Dead code, safe to leave or
   delete; not a source of silent misconfiguration.

4. **No unit tests for the RL stack.** `training/tests/` and
   `experiments/H18_merge/`, `experiments/byte_adapter/` have real test
   files; `experiments/rl/` has only the `_smoke()` functions in
   `loader.py`/`wkv_loop.py` (manual `--model` invocation, not automated,
   not run in CI). For code this size and this correctness-sensitive
   (reward shaping, GRPO advantage, gradient path), refactoring without
   tests risks silently breaking the reward or the gradient without any
   signal until a training run produces visibly wrong behavior.

5. **byte_adapter's parameter-registration gap — still open, `train_wordsearch.py`
   itself deleted 2026-08-18** (superseded per its own successor's
   docstring, unrelated to this gap). `RWKV_x070.parameters()` still
   returns 0 (weights live in a `.z` dict, never registered as
   `nn.Parameter`), confirmed again 2026-08-18 — byte_adapter's own
   parameters have never been wired into any optimizer. Only
   *inference-time* embedding-patching (`eval_byte_wordsearch.py`, run for
   real 2026-08-18 on G1i: 0/56 = 0.0% with a random-init adapter, the
   expected floor before training) works; *training* the adapter is still
   not implemented. Needs a real design decision (how to expose `.z`
   tensors as trainable `nn.Parameter`s for this one component without
   touching the frozen backbone), not a patch.

6. **RESOLVED 2026-08-18 — `--forge` now works, via a different mechanism
   than originally planned.** FORGE's fused-into-backward optimizer update
   assumes each layer's weight is touched by `backward()` at most once per
   step; `wkv_grpo_loss` builds a differentiable graph across the *whole*
   rollout (prefill → M-loop → answer tokens) so gradient flows through the
   WKV-loop's reasoning state, and BPTT weight-reuse across timesteps
   touches every FusedLinear layer more than once per backward() call —
   confirmed structural (not per-layer) by excluding `head` from the fused
   wrapping and watching the crash reappear on an `ffn` layer instead.
   Truncated BPTT would satisfy FORGE's assumption but zeroes gradient into
   the M-loop entirely, defeating the state-carries-reasoning design —
   rejected. **Real fix: `experiments/rl/loader.py::Int8AdamW`** — don't
   fuse anything into backward at all; run ordinary `backward()` (full
   BPTT, unmodified) and apply FORGE's standalone
   `optimizer_only_adamw_int8state()` kernel to the resulting `.grad`
   tensors afterward, same as any other optimizer.step(). Verified on G1d,
   5 real training steps, no crash, checkpoint-diffed 16/20 params changed
   (max diff 0.00049) — genuine gradient flow through the int8-quantized
   optimizer path. `wrap_rwkv7_excluding_head`/`FusedOptimizerManager`
   (the fused-into-backward approach) are unused by `--forge` now, kept in
   `loader.py` as a documented dead end, not deleted.

   **VRAM measurement, G1i, real not estimated (2026-08-18):** single-step
   at G=4/batch=2/M_max=4 — 13713 MiB without `--forge`, 13655 MiB with.
   **Essentially no difference at this scale** — the WKV-loop's own
   unbatched per-rollout activation memory (risk #2 above) dominates total
   usage at this batch size, not optimizer state. This does not confirm
   the earlier informal "~18GB→~12-13GB for G1i full-FT" estimate from
   before any real measurement existed — that number should be treated as
   unverified/likely wrong until tested at real training scale
   (G=8/batch=4/M_max=16), not retested tonight (each step already took
   several minutes at the tiny scale above). Full writeup: memory
   `project_noesis_forge_bptt.md`.

7. **RESOLVED 2026-08-18 — `feed_mode="expected"` verified end-to-end,
   real gradient flow confirmed.** Previously zero coverage beyond the
   isolated `_peft_forward_embeds` equivalence test. Full
   `generate_rollout(feed_mode="expected")` → `wkv_grpo_loss` →
   per-rollout `backward()` → `optimizer.step()` run on GPU (G1d,
   G=8/batch=2/M_max=4, lr=1e-4): step 1 loss=0.4337 acc=12.5%, step 2
   loss=-0.4337 acc=0%, both nonzero/no-NaN. Checkpoint-diffed 20
   parameters — 18/20 changed, max diff 0.00037 (sane for lr=1e-4).
   `residual` mode (needs `mlp_delta`) still untested — lower priority,
   same code path as `expected` plus one extra module. Step 3 triggered
   `TrainingMonitor`'s emergency stop (`SHORTCUT` flag, mean_M=0 — every
   rollout in the batch committed to an answer with zero internal WKV-loop
   steps) — the safety mechanism working exactly as designed, not a bug;
   real, worth noting that `expected` mode hit a degenerate high-confidence
   collapse this quickly on an undertrained 0.4B model with random tasks.

8. **RESOLVED 2026-08-18 — first real (non-`--no-update`, non-`--forge`)
   gradient-update run, confirmed with actual nonzero signal.** First
   attempt (G1d, G=4/batch=2/M_max=4, lr=1e-5, 3 steps) ran clean but hit
   a degenerate all-identical-reward batch on every step (loss=0.0 exactly,
   confirmed via zero-diff checkpoint comparison — not a bug, just bad
   luck with such a small batch). Second attempt (G=8/batch=2/M_max=4,
   lr=1e-4, 5 steps) landed real signal: step 1 loss=0.4337, accuracy=6.25%
   (1/16 rollouts correct). Checkpoint-diffed 20 parameters against the
   original weights: **18/20 changed**, max diff 0.00049 (sane magnitude
   for lr=1e-4 over a few steps). The full pipeline — forward → per-rollout
   backward → optimizer.step() → checkpoint — is confirmed working
   end-to-end with real gradient flow, not just "doesn't crash." This was
   the single largest open item before trusting a real training launch.

9. **RESOLVED 2026-08-18 — G1i (2.9B) full-FT's real memory ceiling on a
   16GB T4 root-caused: a FIXED cost (weights + first-backward grad
   buffers), not activation/BPTT memory.** Direct `torch.cuda.
   memory_allocated()` tracing (not the earlier informal estimates):
   weights 5.896GB + grad buffers 5.895GB (bf16, allocated in full on the
   *first* `backward()` regardless of workload — one `.grad` tensor per
   trainable param, sized by the param) = ~11.8GB baseline. Confirmed via
   an extreme minimal-chain test (G=2/batch=1/M_max=1/max-answer=2) that
   OOM'd at the *same* ~14.7GB mark as every larger config tested that
   day — chain length wasn't the driver across the whole 8–36 step range.
   **Fix: `Int8AdamW(offload_state=True)`** — keeps the ~5.8GB int8
   optimizer-state buffers in CPU RAM, staging one parameter's state onto
   GPU at a time inside the existing per-param `step()` loop. Measured
   fixed cost with offload: 11.8GB (matches weights+grad only). Combined
   with per-timestep `torch.utils.checkpoint` (re-added — see below) and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, a real G=8/batch=2
   /M_max=4 training step can complete, though typically only 1–9 of 16
   rollouts survive per step (curriculum task-length variance against a
   thin remaining margin) — accepted as the real operating ceiling for
   this card/model/method, not a bug left to chase. Full narrative
   (including two separate `gc.collect()` hypotheses tested and
   disproven, and why the earlier "checkpointing measured worse" finding
   was an artifact of testing it before the fixed-cost fix existed) in
   memory `project_noesis_forge_bptt.md`.

   **OOM-catch-and-skip resilience added** to `wkv_grpo_loss`'s
   micro-step loop (`try/except torch.cuda.OutOfMemoryError` around each
   rollout's recompute+backward) — an hours-long run no longer crashes
   over one unlucky rollout. Advantages are computed from the *full*
   no-grad-generated group before any OOM filtering (`generate_rollout`
   always succeeds — see next item), so GRPO's per-prompt advantage
   estimate is never starved by the skip rate; only which rollouts get to
   *contribute gradient* is affected.

   **`generate_rollout()` (`wkv_loop.py`) was missing `torch.no_grad()`**
   — independent real bug, fixed same day. Every rollout generation was
   building a full, live, unused BPTT autograd graph (thrown away — GRPO
   recomputes log π_θ separately, with grad, in
   `_recompute_wkv_log_probs`), paying the BPTT cost twice per rollout
   for nothing. Confirmed post-fix: generation now adds ~9MB, not
   gigabytes.

   **Second real bug found via a live `MODE_COL` emergency stop**
   (real training run, step 8, `<20%` unique decoded texts in the group —
   `TrainingMonitor` correctly caught and cleanly `break`'d out, not a
   crash): the OLD `_save_checkpoint` built the entire trainable-param
   state_dict as one CPU-resident dict comprehension before writing
   anything — a full second ~5.9GB CPU copy on top of Int8AdamW's
   already-resident ~5.8GB offloaded state. Confirmed via `dmesg`/
   `journalctl -k`: Linux OOM-killer SIGKILLed the process (anon-rss=
   14.9GB, no swap) at exactly that moment — silently, no Python
   traceback, no checkpoint file. Would have broken checkpointing on
   *every* save, not just this one. **Fixed**: rewrote to a
   `ckpt_step{N}/` directory, one `.pt` file per trainable parameter,
   written in the existing per-param loop — verified via isolated RAM
   trace: checkpoint save now costs ~109MB, not ~5.9GB. Also added:
   on any monitor flag, print up to 5 sample rollout texts to stderr —
   the MODE_COL collapse's actual output text was never captured
   anywhere before this fix, so there was nothing to diagnose *why* it
   collapsed once the process (and its weights) were gone.

10. **RESOLVED 2026-08-18, same evening — the "MODE_COL collapse" above
    was a false alarm: `.text` construction bug, not a real model
    collapse.** The flagged-batch sample logging (item 9) immediately
    showed the cause: answers decoded to `'�'` (U+FFFD). Root cause,
    confirmed via `experiments/rl/_debug_modecol.py` replaying the exact
    prompt against both base weights and `ckpt_step000007`:
    `tok.decode([eos_id])` is `'�'` on its own (not a real vocabulary
    token), and `wkv_loop.py`'s answer-decode loop appended the sampled
    `eos_id` to `answer_ids` *then* decoded the whole list into `.text`
    — so every rollout that correctly and tersely answered (exactly what
    "Output only 0 or 1"-style prompts ask for) and stopped got its
    `.text` corrupted, while `answer_ids`/`answer_log_probs` were always
    fine. Two real consequences from one bug: (a) `TrainingMonitor`'s
    MODE_COL diversity check saw every terse-correct answer collapse to
    the same `'�'` string — a guaranteed false trigger, confirmed by
    re-running the identical prompt pre/post fix on the *same*
    `ckpt_step000007` weights: real answers were `'0','1','0','0','1'`
    (2 unique, 40% diversity, well above the 20% threshold — would never
    have fired), 3/5 *correct* (true answer for that prompt is `0`).
    (b) `rewards.py::compute_wkv_loop_rewards` scores
    `_score_correct(r.text, rubric)` against this same corrupted
    `.text` via regex — reward computation itself was silently zeroing
    correct-but-terse answers all session, plausibly a meaningful
    contributor to every low/noisy accuracy number logged that day, not
    just the two emergency-stop incidents. **Fix**: `generate_rollout`
    now decodes `text` from `answer_ids` with any trailing `eos_id`
    stripped (`answer_ids` itself unchanged — log-prob replay still
    needs the full sequence including eos). Real lesson: the WKV-loop
    mechanism and the model itself were working correctly and giving
    sensible answers the whole session, base weights included — every
    "collapse" signal traced back to text construction, not training
    instability.

11. **STILL OPEN, 2026-08-23 — real system-RAM leak (not VRAM) killed the
    Phase 1.5 full-FT run (`train_think_distill.py`) twice around step
    ~57-59, host memory only, undocumented here until now (was only in
    memory, not repo canon).** `dmesg`: kernel OOM-killer SIGKILLed the
    process (`anon-rss:15647628kB`) on a 15GB-RAM VM — not a CUDA/VRAM
    error, so none of `_is_oom_error`'s guards (item 9's descendants)
    could ever catch it, a kernel SIGKILL isn't a Python exception at
    all. Relaunched `--resume` with `--batch 1 --grad-accum-steps 2`
    (same effective batch=2 statistics, lower peak VRAM per micro-batch)
    specifically to test whether batch/VRAM-shaped memory was the cause
    — it wasn't: RAM climbed to the same ~15GB ceiling by step 59 again,
    nearly identical timing. **Growth is not proportional to batch
    size** — something accumulates roughly per-step regardless. Original
    (2026-08-23) suspects were the CUDA allocator
    (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was set) or
    `Int8AdamW(offload_state=True)`'s CPU↔GPU staging `.to()` swap
    (`loader.py::Int8AdamW.step()`).

    **The CUDA-allocator half of that was never a coherent suspect for
    THIS symptom (caught 2026-09-02, category error carried forward
    uncritically for over a week): `PYTORCH_CUDA_ALLOC_CONF` governs the
    CUDA driver's DEVICE (VRAM) segment-mapping strategy — it has no
    mechanism that touches host RAM at all.** The leak is confirmed host
    RSS (`anon-rss` in the kernel OOM-killer log above, not a CUDA/VRAM
    error), so this was never the right shape of explanation regardless
    of what review turned up. The `Int8AdamW` CPU-offload `.to()` swap
    remains the one suspect that actually touches host memory by
    construction and is still open — see the FORGE-source review below,
    which narrows but doesn't close it. Not anything in this project's
    own code changed that night.

    **2026-09-02 follow-up (no GPU available to reproduce, local-only
    work): one candidate ruled out, instrumentation added instead of
    another blind config change.** Reviewed FORGE's actual source
    (`github.com/dk4248/FORGE`, `src/fused_grad_optimizer/{state.py,
    autograd.py,kernel.py}`) — `optimizer_only_adamw_int8state` is a
    `@triton.autotune`'d kernel, which raised the obvious suspicion that
    varying call shapes could grow Triton's host-side compiled-kernel
    cache indefinitely (a well-known host-RAM leak pattern with dynamic
    shapes). **Ruled out**: `Int8AdamW.step()` calls this kernel once per
    *parameter*, and every parameter's shape is fixed for the entire run
    (weight matrices don't change shape) — the same handful of shapes
    recur every step, so Triton's autotune cache should saturate
    immediately and never grow. `state.py::FusedOptimizerState.
    ensure_buffers()` also only allocates once (`if self.m_q is None`
    guard), not on every `step()` call. Nothing in the reviewed FORGE
    source shows an obvious per-step host allocation that isn't freed.
    **Added instead**: `train_think_distill.py::_rss_mb()` (reads
    `VmRSS` from `/proc/self/status`, same convention as
    `vm_watchdog.py`'s `/proc/uptime` parsing, no new dependency) logged
    every step in both the console line and `distill_log.jsonl`'s
    `rss_mb` field — the actual growth curve (linear vs. step-like, and
    whether it lines up with `--ckpt-every`) is still unmeasured; this
    makes it visible on the very next launch instead of needing a
    dedicated debugging pass. Next real step still needs a GPU/VM:
    launch with this logging active and read the curve before touching
    config again.

    **2026-09-02, real GPU run with the RSS logging active — reproduced
    the exact same kill, but in ~3 steps instead of ~57-59, on a
    freshly-booted 15GB VM (uptime 44min, 15GB free at launch, no
    leftover state from anything else).** `dmesg`: `Out of memory: Killed
    process (python3) ... anon-rss:15627920kB` — essentially identical
    RSS at time of kill to the original (`15647628kB`), reached in a small
    fraction of the steps. The `_rss_mb()` logging never got a chance to
    print even once: every micro-batch this run (`--forge
    --forge-offload-state --batch 1 --grad-accum-steps 2 --grad-cp`,
    `PYTORCH_CUDA_ALLOC_CONF` NOT set this time, unlike whatever the
    original run had) was ALSO hitting CUDA VRAM OOM and retrying
    internally (`CUDACachingAllocator.cpp` warnings, 8+ in the first 3
    steps) before the host-level kill happened — the run never completed
    a single clean step. **New candidate mechanism, not yet confirmed**:
    the two OOM failure modes may not be independent. Repeated failed CUDA
    allocation attempts inside a backward pass that is ALSO doing
    `Int8AdamW`'s per-parameter CPU↔GPU `.to()` staging could be leaving
    orphaned host-side buffers behind on each failed/retried transfer,
    which would explain both observations at once: slow accumulation
    across many *clean* steps if VRAM pressure is low (the original run),
    and fast accumulation to the same ceiling if VRAM pressure triggers
    OOM-retry storms from the start (this run) — same host-RAM sink, two
    different feed rates depending on how much CUDA-side retrying is
    happening per step. If true, this reframes the debugging strategy:
    reproducing the crash for real diagnosis (heap snapshots, `tracemalloc`,
    or watching `_rss_mb()` climb during a deliberately VRAM-constrained
    run) no longer needs a slow 57-step wait — a VRAM-OOM-heavy config
    reproduces it in minutes. Not yet tested in isolation (a VRAM-safe
    config with zero OOM retries, run long enough to see whether RSS still
    climbs without any retry storms, would falsify or confirm this).

---

## State metrics — what to measure before and during A1.5

### IPC (Information Processing Capacity) — DONE (2026-08-16)

Linear IPC (Dambre et al. 2012) measured on G1i with held-out R² (20% split):

| Layer | IPC_total (held-out) |
|-------|----------------------|
| L0    | 0.53                 |
| L4–L31 | ≈ 0.0              |

Result: `experiments/A0_state_probe/results/ipc_g1i_fixed.json`.

**WKV state is not linearly decodable.** Earlier in-sample measurements (IPC≈12)
were ridge regression overfitting artifacts — n_proj=128 random projections with
~200 training points memorise noise. Held-out R²≈0 is the correct reading.

This is a positive result: IPC≈0 means the state encodes information nonlinearly,
i.e. it is doing computation rather than acting as a linear token buffer. Consistent
with H8. To measure actual state content after RL, use a nonlinear probe (2-layer MLP).

**⚠ Unresolved (2026-08-16):** fleeb83's independently-reported IPC numbers on the
same checkpoints (G1h/G1i base, step9b-e1) are nonzero (17–48% of ceiling), not ≈0.
Not reconciled — possibly an in-sample-vs-held-out split difference, possibly sample
size (256 vs 512 tokens). See hypotheses/H10.md (IPC section) for detail before treating
IPC≈0 as settled.

**H10 gap** = requires nonlinear probe, not ridge regression. Run MLP probe after RL
checkpoint 1 to quantify how much task-relevant content accumulates in WKV state.

### MLP probe — RUN, 2026-08-18 (corrected 2026-08-23; the command below
### is a dead path, the probe shipped a different way)

Nonlinear IPC via 2-layer MLP replacing ridge regression. **Not actually
"never launched" as this section previously said** — it shipped by
migrating into the `experiments/_common` shared battery framework rather
than through the standalone invocation below, closing the open TODO.
Real results across G1d/G1i/step9b-e1 (`hypotheses/H8.md`'s 2026-08-18
evidence entry): same size ordering as linear IPC (G1d 6.9-10.5/16 >>
G1i 0.46-3.6/16 ≈ step9b-e1 3.9/16) — G1i stays low even under a strictly
more expressive nonlinear estimator, so the near-zero-past-L0 pattern
isn't a linear-probe blind spot. Result files:
`experiments/_common/results/{mlp_ipc_g1d_64,mlp_ipc_g1i_256,g1d_battery_gpu_smoke,g1i_battery_gpu,step9b_e1_battery_gpu}/mlp_ipc.json`.
Linear IPC≈0 (held-out) on G1i was the original motivation, but that
result is itself flagged as unreconciled (see §State metrics above) —
worth keeping in mind when reading these numbers too.

Dead invocation path, kept only as a historical note — the file this
pointed at directly (not through `_common`) was never actually the one
that ran:
```bash
python3 experiments/A0_state_probe/mlp_probe.py \
    --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \
    --n-tokens 256 --layers 0,4,8,16,24,31 \
    --out experiments/A0_state_probe/results/mlp_ipc_g1i_base.json
```

Re-run after RL checkpoint 1 on same trajectory (`--trajectory-in`) to measure
how much task-relevant content accumulates in WKV state after training.

### L_mem — SKIP (IPC verdict 2026-08-16)

Linear memory capacity ≈ 0 on held-out data. L_mem would penalise the model for
not retaining tokens linearly — but the state encodes nonlinearly by design (H8).
Adding L_mem would fight the architecture. Skipped permanently.

```
L_mem = −Σ_{k=1}^{5} R²(x(t), u(t−k))   # skipped — linear IPC ≈ 0, wrong metric
```

### L_tPC — post-GPU, ablation required

InfoNCE between WKV states at t and t+k. Keeps state motion on a
predictable manifold (Predictive Coding framing). Theoretical anchor:
arXiv 2506.00580 (SFA = variational inference objective — closes the gap
under L_state). Risk: β tuning may collapse 43.8% baseline — defer until
full ablation budget exists on Selectel 4090.

### SNN/STDP — closed

Checked 2026-08-15. STDP pushes toward temporal smoothness (same direction
as SFA), opposite of L_state. ε-mask covered the reward-modulation intuition
in the old design; ε-mask is now removed (WKV-loop). Conclusion unchanged:
no new objective needed — entropy-plateau exit in wkv_loop.py serves the
same gating role without a separate loss term.

### ARC-AGI probe — spatial reasoning ceiling

400 eval tasks, cell-query format. Running on step9b-e1 (PID 491788).
Result will bound what spatial RL can realistically achieve on novel grids.

---

## Open questions

- **Prompting direction hint.** The current prompt names the orientation
  explicitly ("placed horizontally right-to-left"). A harder variant
  omits the direction hint and asks the model to find the word anywhere.
  Reserve for level 8 (future).
- **Multi-word grids.** Level 8+ could hide multiple words and ask for
  all of them. Requires a different rubric (set match, not position).
- **State readout probe.** After a full grid pass, can `state_readout`
  mode (inject `</think>`, decode answer) match `prompt_cot`? If yes,
  the grid has been fully absorbed into WKV state — direct H8 evidence.
