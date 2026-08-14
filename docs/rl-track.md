# RL Track — Word-Search Curriculum

## Motivation

Bit-decoding (the prior RL task candidate) requires the model to learn a
lookup-table encoding entirely absent from its pre-training distribution.
This produces out-of-distribution signal that trains a narrow procedural
behaviour rather than the spatial reasoning and bidirectional attention
we actually want to strengthen.

Word-search puzzles are a better RL task family because:

- **Verifiable reward.** The answer is `row=R col=C` — checked by regex,
  no judge model required.
- **Spatial reasoning.** The model must scan the grid, not memorise a
  mapping. The state accumulates directional context as it reads each row.
- **Natural progression.** Difficulty is a single continuous dial: grid
  size, orientation count, and word length. The curriculum starts at what
  the base model should already solve (short horizontal word, 5×5 grid)
  and ends at what it demonstrably cannot yet (8-direction 12×12 with
  longer reversed words).
- **Reverse and diagonal perception.** The user hypothesis: models trained
  only on left-to-right text have weak right-to-left and diagonal
  attention. Word-search at levels 3–7 forces the model to develop these
  pathways — and whether WKV state retains directional orientation across
  rows is a direct test of H8 (state-as-computation).
- **State utilisation signal.** Unlike single-token tasks, a word-search
  query requires integrating evidence across many grid positions before
  committing an answer. Multi-pass (N > 1) sweeps of the same grid are
  a direct handle on H10 (multi-pass state convergence).

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

LoRA alone cannot achieve a fundamental shift toward latent-token behavior.
LoRA modifies projection matrices within a low-rank subspace; the model's
core decision about whether to emit or hold (the quality gate, H16) lives
in the full weight geometry. To train the model to allocate WKV state
budget toward latent computation rather than output generation, the full
parameter space must be updated.

**Variant A — Full FT (default, latent token track):**
- All weights updated (full_ft_parts: all, lora.enabled: false)
- L_state mandatory: trajectory regularization + ε-mask (0.05 outside think)
- H12b.i slot entropy optional (separate run for ablation)
- Two-phase: (1) SFT to establish latent token style, (2) GRPO RL to shape it
- Loss on memory: L_state penalises low-rank WKV updates, forcing the model
  to use its state budget for reasoning before committing to output

**Variant B — LoRA (ablation only):**
- LoRA rank 32 on attention weights, full FT on time/ln
- Cheaper; useful to quantify what LoRA alone achieves vs full FT
- Expected: partial latent behavior, insufficient depth shift

H12b.i is optional in both variants — run as a separate checkpoint to
isolate the slot entropy effect from the full-FT effect.

---

## RL design (step 10+)

**Reward signal.** Binary: +1 for exact match on `row=R col=C`, 0 otherwise.
No partial credit — the position is either right or wrong.

**Curriculum schedule.** Start with level 1–2 only. Advance to the next
level when accuracy on the current level crosses 80% within a batch.
Drop back one level if accuracy falls below 50%.

**Training algorithm.** GRPO (Group Relative Policy Optimisation) with
`n_samples=8` per prompt. Temperature 0.7 during rollout, greedy for eval.

**Corpus mix.** Word-search tasks are a supplement to the step-10 base
corpus (RFC QA + selfcot + hh-rlhf), not a replacement. Target fraction:
15–20% of training tokens in each epoch. See `training/config/pilot_step10.yaml`
for the full mix.

**Connection to hypotheses.**

| Hypothesis | What word-search measures |
|-----------|--------------------------|
| H8 (state-as-computation) | Does WKV state retain directional context row-by-row? |
| H10 (multi-pass convergence) | Does N=2 sweep improve level 4–7 accuracy? |
| H12b.i (slot utilisation) | Do slot-entropy losses improve diagonal scanning? |
| H16 (gated externalisation) | Word-search RL trains latent think-phase use — precursor to H16 gate head. If the model learns to scan the grid in `<think>` tokens via GRPO, that is H16-without-gate. The gate head (emit vs. hold) is a separate A4 step. |
| H19 (weight-knowledge contamination) | Word-search is a pure context task — the grid is in the prompt, model weights cannot help. Accuracy on L6–L7 is a direct proxy for context-read vs. weight-recall. A model scoring well on L7 demonstrably reads the context window. |
| H23 (spatial state) | Level 6–7 performance before vs after RL step |

**ε-mask as H16 bootstrap — the key insight.**

The ε=0.05 outside-think loss is not just an implementation parameter.
It is the training signal that bootstraps the model's quality gate:

- With ε=0, the model learns "emit anything outside think, it costs nothing."
- With ε=0.05, even tokens outside the think span carry a small loss signal.
  The model learns that *every emitted token has a cost*. It learns to be
  selective — to prefer think-span computation over low-confidence emission.
- After GRPO on verifiable tasks, this selectivity becomes a dynamic gate:
  the model routes uncertain reasoning into `<think>` and emits only when
  confident. This is H16 (gated externalisation) emerging from the training
  signal, not from an explicit gating head.

In other words: **ε-mask teaches quality awareness → GRPO shapes it into a
self-regulated gate → H16 emerges without a dedicated MLP head.**

The explicit H16 gate head (A4 step) would measure and sharpen this
emergent behavior, not create it from scratch.

**Entropy reward shaping.** During GRPO rollout, reward term:
`α * Δentropy_reduction` (entropy before vs. after think span). Encourages
resolving uncertainty in think-space. Tune α separately from binary reward.

Both activate at step 10+ RL phase; irrelevant to the SFT baseline.

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
