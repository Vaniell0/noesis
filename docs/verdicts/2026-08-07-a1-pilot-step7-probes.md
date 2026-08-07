# Verdict: A1 pilot Step 7 + probe battery (2026-08-07)

## Context

Step 7 checkpoint (`pilot_g1h_step7_action/rwkv-0.pth`, 3.6 GB bf16)
trained G1h-2.9B on action-chain corpus, ctx_len=16384, ~53% through epoch 1.
Battery: A0 end-task eval + H12b multi-slot behavioral probe + H21
premise-validity N-sweep + H21 Variant A cross-N head transfer.

---

## A0 end-task eval (48 tasks, G1h-2.9B step7)

**Raw: 8/48 = 16.7%. Genuine after FP audit: 4/48 = 8.3%.**

Baseline (G1h-2.9B base, no fine-tune): 3/48 genuine = 6.2%.

False positives stripped:
- `sched_05` — regex `\b1\b` trivially matches any "1"
- `sched_06` — known FP from prior audit
- `str_op_02` — tool_output hallucination accepted by checker

Genuine solves: `sym_alg_02`, `sym_alg_03`, `sym_alg_04`, `arith_03`.

Notable: model discovered `<think>` tag format from action-chain corpus
(not explicit in corpus). sym_alg tasks solved via reasoning inside
`<think>` block. This is expected — G1h is a reasoning model; the
action-chain training surfaced rather than introduced that capability.

**`bit_decoding` category: 0/16 across all modes.** Indicates a
fundamental gap in multi-step bitwise reasoning. Not closed by step7.

**Limitation.** `tasks.jsonl` was authored by Claude Code and some
tasks may overlap with the action-chain training corpus (same tool_use
session format). Results for tool-call-style tasks should be treated as
upper-bound estimates until regenerated from a clean algorithmic
generator. `sym_alg` and `arith` tasks are less likely to be
contaminated (mathematical structure).

---

## H12b behavioral probe (multi-slot working memory)

Two checkpoints; K∈{2,4,8} parallel colour-name tracks, P=1 retrieval.

| Model | K=2 | K=4 | K=8 | Verdict |
|-------|-----|-----|-----|---------|
| G1d-0.4B base | 50% | 40% | 5% | CONTAMINATION |
| G1h-2.9B step7 | 20% | 55% | 52.5% | NO CONTAMINATION |

Step7 maintains multi-slot retention at K=8 (52.5%). The base model
collapses at K=8 (5% ≈ chance). Action-chain training appears to have
stabilised multi-track state independently of the LoRA intervention
H12b predicts. Note: this is a behavioral baseline probe on existing
state, not H12b's architectural treatment (LoRA-expanded multi-slot).

Unexplained: step7 accuracy at K=2 (20%) is *lower* than base (50%).
Possible cause: step7 representation reorganised state for longer-range
retention (K=8) at the cost of short-range slot density. Warrants
investigation before H12b intervention.

---

## H21 N-sweep (WKV cycling, G1d-0.4B, 40 items, LOO)

N = number of WKV forward passes of same prompt before readout.

| N | LOO F1 | LOO acc |
|---|--------|---------|
| 1 | 0.811 | 0.825 |
| 2 | 0.811 | 0.825 |
| 3 | 0.789 | 0.800 |
| 5 | 0.769 | 0.775 |

F1 monotonically decreases with N beyond N=2. Additional WKV cycling
adds representational noise for this probe; the base model encodes
premise validity maximally in a single pass. Does not rule out N>1
being useful for other state dimensions (H10-class reasoning).

---

## H21 Variant A (cross-N head transfer)

N=1 trained head applied to N=2/3/5 state features (no retraining).

| N (features) | Variant A F1 |
|--------------|-------------|
| 2 | 0.800 |
| 3 | 0.800 |
| 5 | 0.800 |

The N=1 projection direction transfers stably to N-cycled states.
F1=0.800 is below fresh N=1 head (0.811) but above fresh N=3 (0.789)
and N=5 (0.769) heads. Interpretation: state geometry changes slowly
under WKV cycling — the validity-separating direction at N=1 persists
with modest decay. This validates that premise-validity is encoded in
a robust, cycling-stable subspace of the state.

---

## Open items

1. **Clean eval tasks** — regenerate `tasks.jsonl` algorithmically to
   remove Claude Code authorship bias. Priority before step8 verdict.
2. **H12b K=2 regression** — why does step7 perform *worse* than base
   at K=2? Check if it's a scoring artefact or genuine.
3. **bit_decoding gap** — 0/16 across all models/modes. Either a
   reasoning-depth issue or a tokenisation mismatch. Investigate before
   claiming the task category is fair.
4. **H21 N>1 training** — the N-sweep shows single-pass is optimal for
   this probe, but trained N>1 (with L_state reward) might differ from
   zero-shot cycling. Defer to Phase 2.

---

## Step 8 candidates (dataset extensions)

- DSL-native action chains (tool calls → DSL syntax, reduces token noise)
- ctx_len 16384→32768 (requires re-initialising positional patterns)
- Algorithmic task corpus (bit_decoding coverage, clean eval overlap)
- L_state supervision on reasoning traces (latent CoT training)
