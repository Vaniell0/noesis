# RL Track — Matrix Task Curriculum (A1.5)

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

## RL design

**Reward signal.** Binary: +1 for exact match on `row=R col=C`, 0 otherwise.
No partial credit — the position is either right or wrong.

**Curriculum schedule.** Start with level 1–2 only. Advance to the next
level when accuracy on the current level crosses 80% within a batch.
Drop back one level if accuracy falls below 50%.

**Training algorithm.** GRPO (Group Relative Policy Optimisation) with
`n_samples=8` per prompt. Temperature 0.7 during rollout, greedy for eval.

**Corpus mix.** Unified matrix task curriculum via `experiments/A0_eval/gen_tasks.py`
→ `training/corpus_open/matrix_tasks.jsonl` (65 899 tasks, ~20M tokens).
Five types: wordsearch 30%, crossword 10%, arithmetic 20%, pattern 20%, bits 20%.
No SFT base corpus mixed in — pure RL on verifiable matrix tasks.
SFT phase explicitly skipped (see decision record below).

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
See also arXiv 2508.04349 (GTPO/GRPO-S): entropy-weight per-token advantage
directly inside the GRPO update — weight WKV-write tokens inside `<think>`
by their token-level entropy, focusing gradient on high-uncertainty state
writes. Complement to the span-level entropy term above.

**CLIPO contrastive reward** (arXiv 2603.10101). Add as a reward term on top
of binary correctness:

```python
# for each prompt group of G rollouts:
# z_i = WKV state at end of <think> span, passed through small MLP head
r_con_i = -λ * InfoNCE(z_i, positives={z_j: correct_j}, negatives={z_k: ~correct_k})
r_con_i = max(r_con_i, -0.5)   # clamp
# add r_con_i to verifiable reward before computing GRPO advantages
```

τ = 0.05, λ tuned per run (start 0.01). Projection dim 512. Push WKV states
of correct rollouts together; pull correct vs. incorrect apart — geometry-level
signal that correct answers route through similar WKV trajectories.

**L_KVB auxiliary SFT loss** (arXiv 2602.21204). Add during SFT phase (before
RL) to teach model to emit think-span tokens whose KV structure the WKV state
can absorb cleanly:

```python
# inside think span forward pass, per WKV layer l:
# W_{t-1}: WKV state before token t
# k_t, v_t: key/value projections of current token
L_KVB += mean(‖W_{t-1} @ φ(k_t) - v_t‖²)   # over think-span tokens
# add to SFT loss: L_total = L_CE + λ_kvb * L_KVB,  λ_kvb ≈ 0.01
```

This is the inner-loop loss the delta rule already minimises per step; training
L_KVB end-to-end teaches the outer loop to issue self-consistent KV pairs so
the state becomes a good compressor of think-span content.

Both activate in the A1.5 RL phase; irrelevant to the SFT baseline (steps 1–9b).

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

## ByteAdapter — tokenizer-free word-search (planned, GPU required)

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

## Phase 4 — Switch-GRPO path to H16 (latent computation, no emitted tokens)

Source: arXiv 2606.13106 "Switchable Latent Reasoning."

Current word-search RL (Phase 3) uses visible `<think>` tokens — GRPO policy
ratio is well-defined over all emitted positions. H16 (gated externalisation)
requires the model to eventually compute silently: no visible think tokens,
pure WKV state accumulation. At that point, policy density is undefined over
latent positions — Switch-GRPO fixes this.

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

**Noesis mapping:**

| Switch paper | noesis equivalent |
|---|---|
| `<swi>`/`</swi>` | `<think>`/`</think>` (already in vocab) |
| `<latent>` placeholder | new token (add to WordTokenizer) |
| High-entropy CoT segment | Any think-span content after word-search RL |
| Phase 1 SFT | Annotate think spans from RL checkpoint with entropy probe |
| Phase 2 curriculum | Progressive: think tokens → `<latent>` placeholders |
| Phase 3 Switch-GRPO | GRPO with latent-usage reward on word-search tasks |

K_min (minimum latent dwell) must be enforced; without it the trained model
exits the latent block in one step. Start K_min = 4; ablate (arXiv 2606.13106
Appendix I).

**Prerequisite:** word-search RL (Phase 3) must first converge on visible
think tokens before Phase 4 is useful. Do not start Phase 4 cold.

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

Reason: SFT before RL reduces variance on *which behavior* RL reinforces — it does not improve the model's ability to do the task. For G1i at 0/56 on word-search there is no existing behavior to exploit as shortcut. This is exactly the case where skipping SFT is cleaner.

**Mechanism of SFT-less RL risk:** GRPO on binary reward latches onto surface shortcuts early. State encodes the shortcut, not the reasoning pathway. The shortcut's verifiable output is indistinguishable from genuine task completion in reward signal.

**Why 0/56 baseline protects us:** no spurious partial-success correlation → nothing to pre-remove. Icophy: "The counterargument for skipping SFT is stronger here than in tasks where the model already achieves non-trivial baseline performance."

**Mandatory checkpoint after first RL epoch:**
R-lens probe on non-task axes (e.g. narrative, arithmetic) before continuing curriculum. If R-lens returns near-chance probe accuracy on non-task axes while word-search performance is high → state is encoding shortcut geometry, not reasoning. Stop and investigate before advancing curriculum.

**L_KVB:** skip for now (SFT-only loss, no SFT phase). Revisit if full-FT SFT is added later.

---

## Integrated RL loop — instrument map

All instruments and where they plug in:

```
G1i base checkpoint
│
├── [NO SFT warm-up — skip]
│
└── Phase 3: Direct RL (GRPO, word-search L1→L7)
    │
    ├── Per batch:
    │   ├── Sample G=8 rollouts per prompt (T=0.7)
    │   ├── r_correct  = ±1 binary (regex match on row=R col=C)
    │   ├── r_clipo    = -λ · InfoNCE(WKV_state_end_think) clamped -0.5
    │   │                 [CLIPO, 2603.10101 — separates reasoning from shortcut]
    │   ├── r_entropy  = α · Δentropy_reduction (think span entry→exit)
    │   │                 [GTPO, 2508.04349 — weight WKV-write tokens by entropy]
    │   └── GRPO update on combined reward
    │
    ├── After checkpoint 1 (mandatory):
    │   └── R-lens probe: stable_rank on task + non-task axes
    │       → if non-task near-chance: shortcut alert, stop curriculum
    │       → if non-task intact: proceed
    │
    ├── Curriculum advance: L_k → L_{k+1} when acc > 80% on current level
    │
    └── Phase 4 (after convergence):
        Switch-GRPO → latent tokens (<swi>/<latent></swi>)
        [2606.13106 — boundary tokens make policy ratio well-defined]
```

**Resolved — use existing tools:**
1. **Training stack**: RWKV-PEFT (reliable, already used in step9/9b). GRPO via `verl`
   (same stack CLIPO uses — `verl/trainer/ppo/ray_trainer.py`).
2. **WKV state extraction**: TransformerLens hook on `state[3*L+1]` at `</think>` position.
   Canonical RWKV-7 support confirmed (Lucas, 2026-08-12). No custom backward needed.
3. **LoRA r=32** on att weights, full FT on time/ln — same as step9b. RWKV-PEFT handles this.
4. **CLIPO projection head**: use `Qwen-Applications/CLIPO` repo directly. Their MLP head
   (2-layer, dim 512, InfoNCE) plugs into verl reward computation unchanged.

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

**H10 gap** = requires nonlinear probe, not ridge regression. Run MLP probe after RL
checkpoint 1 to quantify how much task-relevant content accumulates in WKV state.

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
as SFA), opposite of L_state. ε-mask already covers the reward-modulation
intuition. No new objective.

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
