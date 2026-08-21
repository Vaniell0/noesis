# RL Track — Matrix Task Curriculum (A1.5)

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

**Cost consequence, not yet addressed.** Full FT on G1i 2.9B needs the whole
optimizer state in VRAM — naive estimate ~18GB. **FORGE** (`dk4248/FORGE`, fuses the
optimizer into the backward pass, tile-by-tile) was researched as the fix
(~18GB → ~12–13GB) but was never actually wired in — no `wrap_rwkv7()` helper, no
integration with `_load_peft`/`train_wkv_loop.py`. This is a real gap, not a
"someday": at 24GB (4090) the naive full-FT VRAM budget leaves little room for
`G × batch` rollout parallelism; FORGE's headroom directly trades against how many
GRPO rollouts can run per update. Needs doing before the next GPU session, not
during it.

The old ε-mask / two-phase-SFT framing that used to live in this section is gone —
SFT was skipped entirely (see §Step10 SFT below), and ε-mask doesn't exist in the
WKV-loop design (`−β·M` replaces it, see §RL design).

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
`</think>` position in the M-step loop anymore. A re-adaptation would key on
state at the commit step instead — not designed yet.

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

## Switch-GRPO (arXiv 2606.13106) — role unclear post-WKV-loop, kept for reference

Source: arXiv 2606.13106 "Switchable Latent Reasoning."

**This section's original framing is obsolete, not just its terminology.** It
was written as "Phase 4 extends Phase 3's visible `<think>` tokens to latent
blocks" — but current Phase 3 (WKV-loop) *already* has no emitted reasoning
tokens at all (M internal steps, nothing decoded). There is no "text →
`<latent>` placeholder" curriculum to run, because there was never text to
replace. Switch-GRPO's actual contribution — a well-defined policy ratio at
explicit block boundaries — solves a problem the M-step design doesn't have
(WKV-loop's boundary is just "the loop exited," not a tagged token pair).

What might still be worth revisiting from the paper: if a future design wants
*partial* visibility (some reasoning surfaced as text, some kept in state),
Switch-GRPO's boundary-token mechanism is the right reference. Not scheduled;
no prerequisite chain currently points at it. The mechanism sketch below is
left as-is for that future reference, not as a near-term plan.

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

**Noesis mapping — n/a.** The table this section used to have mapped
`<swi>`/`</swi>`/`<latent>` onto `<think>`/`</think>` tokens that don't exist
in the WKV-loop vocabulary — removed rather than patched, since there's no
current design that would consume it. If the partial-visibility idea above
ever gets picked up, the mapping needs to be rebuilt against `wkv_loop.py`'s
actual exit-reason/M mechanics, not against a think-token vocabulary.

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

## RL status: PAUSED — think-loop state distillation prerequisite (2026-08-19)

**RL training is paused, by explicit decision, not by crash or budget.**
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

**Next phase — "Dreaming cycle" (user's naming, 2026-08-20), M>1,
sits between the current M=1 phase and RL resuming.** Not a new
mechanism to build and not a reframing of RL itself (both considered,
both ruled out on discussion) — it's this same latent-overshooting
curriculum, run at M≥2, now with the 8-token phase budget (this run,
not run 5's 1-token budget) *and* gradient clipping (new this round).
Why "Dreaming": as M grows, each teacher chunk thins — less real
observation grounds each individual phase, so the mechanism leans
increasingly on the model's own prior rollout rather than dense
teacher supervision, the same posterior-to-prior shift the Dreamer
paper (arXiv 2007.14535) describes. Why this has to happen before RL,
not after: RL can only shape/select among behaviours the model can
already produce — an earlier RL run already found `mean_M` frozen at
exactly 2.0 with zero variance to correlate quality against, concluding
"RL has nothing to select for inside the M-loop when the loop has no
content to begin with." A model that has never coherently run an M>1
self-feed can't have that regulated by RL, only trained into it first.
Full plan: `/home/vaniello/.claude/plans/twinkly-questing-ullman.md`
(local machine only, not part of this repo).

Full narrative, including the "why 1 token per step is too little"
reasoning and the DE-table correction (an earlier claim that G1i's own
output was *more* verbose than step9b-e1's got the real numbers
backwards until checked), in memory `project_noesis_think_distill_experiments`.

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

8. **RESOLVED 2026-08-18 — `feed_mode="expected"` verified end-to-end,
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

7. **RESOLVED 2026-08-18 — first real (non-`--no-update`, non-`--forge`)
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

### MLP probe — NOT YET RUN (was mislabeled RUNNING; corrected 2026-08-17)

Nonlinear IPC via 2-layer MLP replacing ridge regression. Written 2026-08-16,
never actually launched — was queued behind the H10 state_readout eval (CPU
contention), then behind an IPC re-verification run. Linear IPC≈0 (held-out)
on G1i was the original motivation, but that result is itself now flagged as
unreconciled (see §State metrics above) — worth keeping in mind when reading
whatever this probe eventually reports.

```bash
python3 experiments/A0_state_probe/mlp_probe.py \
    --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth \
    --n-tokens 256 --layers 0,4,8,16,24,31 \
    --out experiments/A0_state_probe/results/mlp_ipc_g1i_base.json
```

Result: `experiments/A0_state_probe/results/mlp_ipc_g1i_base.json` (pending).
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
