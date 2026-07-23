# Hypotheses

This file is the intellectual audit trail of noesis. Every serious design
decision either tests one of these claims or takes one for granted — be
explicit about which.

The file is not a wishlist. Every claim listed is falsifiable, and the
criterion for rejection is spelled out. If a claim cannot be stated in a
form that could be shown wrong, it does not belong here.

## Evaluation philosophy

**What counts as evidence.**
- Numbers on the user's real held-out eval set (from A0.2 in ROADMAP.md),
  not on standard benchmarks. Benchmarks are references, not verdicts.
- Sustained-operation metrics (RAM, CPU, wall-clock, quality decay over
  N days) for anything claiming background viability.
- Blind comparison where possible: same task, multiple backbones,
  LLM-as-judge scoring, spot-checked by the user.

**What does not count.**
- Improvement on a benchmark that was in the training corpus.
- One-shot demos on cherry-picked prompts.
- Philosophical elegance.
- Alignment with prior claims made in this file.

**Sources of bias to name and mitigate.**
- *Confirmation bias.* It is emotionally expensive to reject H4 after
  months of work. Pre-commit the refutation criterion before running the
  experiment, not after.
- *Goodhart's Law.* Any single metric will get gamed. Use at least three
  disjoint metrics per hypothesis where the task allows.
- *Sunk cost.* If Gate 1 refutes the RWKV wager on target tasks, honour
  the pre-commitment to re-open the backbone decision. Do not rescue
  with post-hoc reframing.

**What NOT to optimise.**
- Do not optimise for benchmark scores that were not agreed on in
  advance. If you find a metric that noesis happens to win, log it as
  interesting; do not promote it to primary.

**Reporting cadence.**
- Every gate produces a short honest write-up: what was tested, what the
  numbers say, what the interpretation is, what is left unresolved.
  Failure to report a negative result is worse than the negative result
  itself.

---

## H1. Constant-cost background operation

**Claim.** An RWKV-7-G1 2.9B model, quantised for local hardware, can
run as a persistent background reasoner on the user's stack (GTX 1050 or
CPU-only) with resource consumption low enough not to disrupt foreground
work.

**Prediction (tightened 2026-07-22).** Under a realistic 24-hour
background workload (event-stream ingestion + retrieval on demand +
periodic composer/reflection bursts), CPU usage falls into one of two
disjoint regimes and never outside them:

- **Steady:** < 1 % CPU. Model resident (Ollama child, `keep_alive: -1`)
  but idle; only event collectors and the scheduler run.
- **Burst:** up to ~20 % CPU for episodes of *tens of seconds* at
  roughly minute-scale periodicity. Every LLM job (composer,
  incremental digest, reflection) is a burst; long-running jobs are
  fragmented into burst chunks, never allowed to sustain.

Alongside this: resident RAM stays < 3 GB (backbone + memory system +
supervisor), and battery life at idle is not measurably degraded
beyond ~10 %.

**Falsification.** Sustained operation for 7 days at Gate 2 stays
inside both regimes and inside the RAM and battery caps. If any LLM
job exceeds tens of seconds in a burst, if steady CPU drifts above
1 %, or if RAM crosses 3 GB, the prediction fails. Response order:
scheduler/budget accounting, quantisation, inference framework,
backbone.

**Design note.** The two-regime rule replaces the earlier "< 10 %
average CPU" line, which averaged over the interesting behaviour.
Enforcement lives in the `noesis-scheduler` module (Rust runtime): a
budget accountant caps burst duration and defers jobs that overrun
into the next burst window.

**Related.** Track C (C1, C2), Gate 2.

**Status.** Untested.

---

## H2. Reasoning-first outperforms knowledge-first at this scale

**Claim.** A small model (≤ 3B) fine-tuned exclusively on reasoning
supervision, given equivalent runtime retrieval access, will match or
exceed a same-size model trained on mixed corpora (reasoning + domain
knowledge in weights) on the user's real held-out tasks.

**Prediction.** After A1, noesis + retrieval scores at least on-par with
the strongest reference model (Qwen-2.5-3B-Instruct or Phi-4-mini) +
retrieval on the A0.2 eval set.

**Falsification.** If noesis + retrieval trails the reference by more
than the noise floor across three independent metrics, the reasoning-
first thesis is at least materially weakened, and the corpus strategy
for A3 must be re-opened.

**Related.** Track A (A1), Gate 2.

**Status.** Untested.

---

## H3. Learned memory policy trumps heuristic memory at small scale

**Claim.** Following Memory-R1 (Yan et al., ACL 2026), an RL-trained
Memory Manager on top of noesis can outperform vanilla RAG / heuristic
memory pipelines on long-horizon recall.

**Prediction.** After A2, noesis with the RL-trained memory manager
scores materially better on multi-session recall tasks than the same
noesis with a vanilla top-K retrieval baseline. Memory-R1 reports ~+28 %
F1 on LLaMA-3.1-8B against Mem0; adjusted for our smaller backbone the
target is a clear, statistically meaningful improvement over baseline.

**Falsification.** If the RL-trained policy fails to beat vanilla RAG by
a meaningful margin, either the data pipeline is wrong or the
hypothesis is wrong. Diagnose in that order.

**Related.** Track A (A2), Track B (B2), Gate 3.

**Status.** Untested.

---

## H4a. RWKV-7-G1 2.9B reaches parity with same-size Transformer

**Claim.** RWKV-7-G1 2.9B — correctly variant-selected, correctly
quantised, and reasoning-tuned per A1 — performs within a defined
margin of the strongest Transformer reference of similar size
(Qwen-2.5-3B-Instruct, Phi-4-mini) on the A0.2 eval set.

**Prediction.** After A1, on the primary metric of the A0.2 eval set,
RWKV-7-G1 achieves a score no worse than 0.7× the score of the
strongest reference model.

**Falsification.** If RWKV-7-G1 trails the strongest reference by
more than ~1.4× on the primary metric after A1, and the gap cannot
be closed by budget-realistic additional training, *this specific
implementation* is refuted. Note the tightly-scoped subject: this
specific model, this specific tuning, this specific eval.

**What this does NOT test.** Whether state-evolution architectures
are fundamentally weaker than attention architectures for reasoning.
That is H4b — a broader wager which this specific comparison can
only weaken or strengthen, not settle.

**Related.** Track A (A0, A1), Gate 1, Gate 2.

**Status.** Untested.

---

## H4b. State-evolution architectures are viable for reasoning
### *(wager, not directly falsifiable at this project's scale)*

**Claim.** Recurrent state-evolution architectures (RWKV-family,
Mamba-family) are not fundamentally weaker than attention
architectures for reasoning on the noesis target task distribution.
Differences in observed capability at similar parameter counts are
attributable to training data, tuning effort, and ecosystem
maturity — not to an architectural capability ceiling.

**Why this is a wager, not a hypothesis.** No single experiment can
distinguish "RWKV lost because state-evolution is worse" from "RWKV
lost because it was under-trained / mis-quantised / wrong-tuned /
disadvantaged by an eval bias / behind on ecosystem tooling". The
confounds are inseparable at our scale of experimentation.

**How H4a evidence updates H4b.**
- If H4a is *supported*, H4b is meaningfully strengthened.
- If H4a is *refuted*, H4b is *weakened but not refuted* — one of the
  confounds may explain the specific loss.
- To make H4b truly falsifiable would require controlled experiments
  well beyond this project's budget (matched architectures, matched
  training data at scale, matched compute, held-out evals designed
  to be architecture-neutral).

**How to act on H4b.** Treat it as the wager underlying the RWKV
backbone choice. If accumulated H4a-style evidence across multiple
G1 generations, multiple training runs, and multiple eval sets
consistently disfavours RWKV *without a plausible confound story*,
the backbone decision reopens under P8 (empirical over
philosophical). Any single failure of H4a is insufficient to force a
reopen; a *pattern* is.

**Mechanism sub-questions tested separately.** H8 (state-as-
computation) and H9 (G1 amplifies state utilisation) address *why*
RWKV would or would not close the gap — the mechanism, not the score.
See those entries.

**Related.** All of Track A across the project lifetime.

**Status.** Perpetually under provisional evaluation. Not a
checkpoint hypothesis — a stance to be corroborated or eroded over
time.

---

## H5. Inter-model state transfer via compact structured summary

**Claim.** Task-state handoff from noesis (background) to remote Claude
(heavy) can be mediated by a compact structured representation (task
graph + condensed reasoning trace) that preserves task continuity
better than either (a) raw context dump or (b) unaided cold-start.

**Prediction.** For a matched set of handoff tasks, Claude receiving
the structured summary completes them at least as accurately as Claude
receiving the full raw context, while using materially fewer input
tokens.

**Falsification.** If the structured summary underperforms raw context
by more than a small margin on accuracy, or if the token savings are
trivial, the protocol design must be reconsidered.

**Related.** Track C (C3). Long-horizon.

**Status.** Untested. Depends on C3, which depends on C1/C2.

---

## H6. Cognitive layer on modest hardware

**Claim.** The full noesis stack (backbone inference + memory system +
event ingestion + summary generation) runs sustainably on the user's
current hardware — GTX 1050 + laptop CPU — without cloud dependency
for the everyday loop. Cloud is required only for occasional training
bursts.

**Prediction.** Steady-state operation on user hardware maintains
< 50 % overall system load, does not hit thermal limits, and remains
responsive (< 2 s for typical query completions on a warm cache) — such
that the user actually keeps it running.

**Falsification.** If sustained operation forces the user to disable
noesis during real work, the hypothesis is refuted. Response: either
simplification, model downsizing, or cloud-serving, in that order.

**Related.** Track C (C1, C2), Gate 2.

**Status.** Untested.

---

## H7. Understanding in weights, knowledge in context

**Claim.** For a personal assistant of this scope, keeping general
reasoning competence in the model's weights and delivering fresh
knowledge through the context window (via retrieval and tool
observations) is a strictly better allocation than baking domain
knowledge into weights.

**Rationale.** Weights are expensive to update; context is cheap to
refresh. Knowledge decays or changes; reasoning does not. A model that
reasons well over given facts generalises to new facts; a model that
memorised facts does not automatically reason about them.

**Prediction.** For A0.2 tasks that require both reasoning and current
information, noesis + retrieval beats a same-size model with domain
data baked into weights but without retrieval, on both accuracy and
freshness.

**Falsification.** If, at any point, the required behaviour of noesis
cannot be achieved through in-context knowledge but only through
weight-baked knowledge, this hypothesis is at least partially refuted
and A3 must lean toward fine-tune rather than retrieval.

**Related.** Track A (A1, A3), Track B (B1). This hypothesis is the
justification for the Phase-1 corpus discipline.

**Status.** Untested. Directly tied to H2 but distinct — H2 is *"is
reasoning-first enough?"*, H7 is *"where should knowledge live once
integration starts?"*.

---

## H8. State-as-computation in RWKV-7

**Claim.** During autoregressive generation, RWKV-7's hidden WKV state
does substantive *computational* work — not merely rolling-summary
memory. On reasoning-flavoured prompts, the state trajectory shows
qualitatively different dynamics than on non-reasoning prompts of
matched length and vocabulary distribution, in a way that is not
attributable to prompt-content confounds alone.

**Motivation.** RWKV-7 paper §2 (Background, p. 4) frames the delta-
rule update as "equivalent to a single step of stochastic gradient
descent, training the state S_t at test time to output the desired
values v_t for the keys k_t as inputs" (arXiv 2503.14456v2). That is a
*per-step* framing. The cumulative-sequence version — "the state
evolves as if learning during generation" — is stronger, is what
noesis's backbone choice is philosophically staked on (see P4, H4b),
and is empirically open.

**Prediction (qualitative — quantitative thresholds TBD after first
probe run, A0.4 step 5).** Across three disjoint metrics of state
dynamics (delta-norm `‖s_t − s_{t-1}‖`, trajectory curvature `κ_t`,
and stable rank `SR(s_t) = (‖s_t‖_F / ‖s_t‖_2)^2` — the latter
matching the paper's Appendix J probe), the effect size between
reasoning-prompt and non-reasoning-prompt trajectories on the same
model exceeds the baseline noise floor (measured over 10 seeds) with
a consistent sign across metrics.

**Thresholds (locked from A0.4 pilot, 2026-07-21).** Pilot: 3 seeds ×
128 decode tokens on `World-0.4B × medium`, CPU bf16
(`experiments/A0_state_probe/results/pilot/`).

| metric                          | pilot mean | between-seed SD | `Δ_min = 3·SD` |
|---------------------------------|------------|-----------------|----------------|
| `delta_pooled`                  | 39.91      | 2.97            | **8.90**       |
| `curvature_pooled`              | 62.52      | 4.50            | **13.49**      |
| `stable_rank` (per-step std)    | 0.571      | 0.163           | **0.49**       |

- **Effect-size lock:** `d = |mean_reasoning − mean_narrative| /
  pooled_sd`; H8 support requires **`d ≥ 1.0` on ≥ 2 of the 3 metrics**
  with Welch's t-test **`p < 0.05 / 3`** (Bonferroni).
- **Scale caveat.** Pilot ran on 0.4B (World-0.4B / G1d-0.4B), not the
  planned 2.9B pair. CPU-only throughput on i5-1235U made 2.9B bf16
  infeasible (~54 h wall for full sweep). H8/H9 verdicts therefore
  bind to the **small-model regime**; 2.9B re-run is a `ROADMAP`
  follow-up conditional on GPU access.

*Placeholder rationale (retained for history).* Thresholds were
intentionally left as a placeholder in the pre-pilot version because
locking them earlier would have risked formulating a criterion on the
wrong mental model. See `docs/state-and-reasoning.md` for the
calibration reference (RWKV-7 paper Appendix J).

**Falsification (staged — placeholder thresholds).** Refutation of a
claim this load-bearing cannot rest on a single run. Staged flow:

1. *First failure.* If, across all three metrics, the
   reasoning-vs-non-reasoning contrast lies within the noise floor
   measured across seeds and prompt-content matched pairs, the
   default response is **not** to declare H8 refuted. First: verify
   the metric implementations against the paper (Appendix J for SR;
   cross-check delta-norm and curvature against a synthetic sanity
   trajectory with known dynamics), and verify the state-extraction
   hooks are capturing the intended tensor at the intended point in
   the forward pass.
2. *Repeat under adjusted probe.* Re-run with instrumentation
   corrections and, if needed, an alternative prompt pair to rule out
   a prompt-content confound. Record the pilot noise floor
   independently on each run.
3. *Sustained failure ⇒ H8 refuted at this scale.* If, after (1) and
   (2), all three metrics still show null contrast on independent
   replications, H8 is refuted for this architecture at this
   parameter count. This is the point at which state-as-computation
   moves from "empirically open" to "metaphor rather than mechanism"
   in the file's audit trail.
4. *Consequence for P4 and backbone choice.* A sustained-refutation
   H8 result weakens (does not by itself overturn) P4's
   constant-cost-over-peak-capability wager — the throughput/RSS
   half of P4's justification remains, supported by A0.1.
   Reopening the backbone decision on the strength of H8 alone is
   possible only *after* stage 3, and even then requires pairing
   with the H4a/H4b evidence (see ROADMAP Gate 1).

**What refutation does *not* imply.** H4a and H4b remain independently
testable — RWKV could still win on end-task quality via other
mechanisms (e.g. training-data quality, tokenizer choice) even if H8
falls.

**Frontier adjacency (Transformer side, 2026-07-23).** Anthropic's
MyTHOS line and the OpenMythos open follow-up couple Recurrent-Depth
Transformers with MoE and memory compression. Recurrent-Depth
Transformer = looped forward pass over the same block stack = *depth-
side* computation-in-forward, the Transformer-flavoured answer to
the same underlying question H8 asks in *width-side* (state-per-token)
form. This is not a claim of equivalence, and the two mechanisms are
not interchangeable; but the frontier's decision to invest in
computation-inside-forward-pass rather than more parameters or more
tokens strengthens the *class* of bet noesis is on. Useful marker for
framing H8's significance in any public write-up. **Not evidence for
H8.** The frontier converging does not mean the RWKV-side version
works — that is what A0.4/A0.5 exists to measure.

**Related.** Track A (A0.4). Feeds into A1 loss-formulation decision
(see ROADMAP Gate 1 exit criteria).

**Status.** Untested; probe designed in `experiments/A0_state_probe/`;
execution deferred to the next session.

---

## H9. G1-line training amplifies state utilisation

**Claim.** RWKV-7-G1h — reasoning-tuned via the G1 curriculum on top
of the World3 base — shows *measurably different* state dynamics from
the World3 base on the same reasoning-flavoured prompts, in the
direction of larger delta-norm, higher curvature, and/or greater
stable-rank variance. That is, G1 training does not merely change the
distribution of *output tokens* (which would be visible only at the
logits level); it changes the way the model *uses its state* during
generation.

**Motivation.** From `docs/state-and-reasoning.md`: no G1 training
documentation is present in RWKV-LM at commit `846b08c1`, so the
mechanism of G1's contribution is not publicly specified. Two
distinguishable hypotheses:

- *Amplification:* G1 supervision teaches the model to route more
  computation through state evolution during the `<think>` phase.
- *Output-only:* G1 supervision changes token distributions without
  altering the underlying state dynamics — the model just emits more
  reasoning-tokens without doing more state-work per token.

A0.4 discriminates these by running paired probes on World3 and G1h.

**Prediction (qualitative — thresholds TBD).** On the same reasoning
prompt with matched seeds, at least one of the three A0.4 metrics
(delta-norm, curvature, stable rank) shows a statistically significant
G1h-vs-World3 difference (Welch's t-test, α = 0.05, corrected for
three metrics via Bonferroni or equivalent), with the direction
consistent with "G1h uses state more actively".

**Falsification (placeholder).** If G1h and World3 are
statistically indistinguishable across all three state metrics on
matched reasoning prompts and seeds, H9 is refuted. G1 would then be
credited only with an output-distribution shift, not a mechanism
change.

**Consequence of refutation on A1 design.** If H9 fails, A1 training
should not attempt to induce different state dynamics via a
state-regularised loss (the paper's own §2 delta-rule framing gives
no leverage without empirical support); standard SFT on reasoning
traces remains the only defensible approach. This closes off the
"train against state trajectory" branch of the A0.4→A1 decision gate
(plan step 7, branch A).

**Related.** Track A (A0.4 → A1). Interacts with H4a (specific quality
match) and H2 (reasoning-first thesis). Independent of H4b (wager)
but a positive H9 would strengthen H4b's mechanism story.

**Status.** Untested; probe designed in `experiments/A0_state_probe/`;
execution deferred to the next session.

---

## H10. Test-time compute frontier — state × tokens × readout

**Claim.** The RWKV-7 backbone exposes three orthogonal knobs before
final answer decode:

- **N** — state-refinement passes over the prompt (each pass updates
  WKV, no tokens emitted).
- **K** — CoT-token budget: how many think-tokens are decoded before
  the answer.
- **readout_mode** — how those think-tokens are produced:
  - `silent` — no think-tokens, K=0 (pure state refinement).
  - `prompt_cot` — classic. Think-tokens decoded as continuation of
    the prompt (each token re-ingested via state update).
  - `state_readout` — after N refinement passes, K tokens decoded
    directly from the refined state (no CoT-prompt scaffolding);
    the tokens are a self-report on the state, then the answer
    decodes from the state-after-readout.

The `(N, K, mode)` matrix has a non-trivial Pareto frontier on the
A0.2 rubric set — i.e. the current AI-industry default
`(N=1, K=large, mode=prompt_cot)` is not necessarily optimal for this
architecture. Mapping the frontier lets noesis define its own effort
levels rather than copying Transformer conventions.

**Prediction.** Sweep `N ∈ {0, 1, 2, 3, 5}` × `K ∈ {0, 32, 128, 512}`
× `mode ∈ {silent, prompt_cot, state_readout}` at greedy decode on
the A0.2 rubric set with G1d-0.4B. Two concrete claims on the
resulting frontier:

- **Non-degenerate frontier.** At least one non-default cell (N > 1
  OR mode ≠ prompt_cot) achieves ≥ +0.05 rubric points at ≤ 1.0× the
  compute cost of the default cell `(N=1, K_default, prompt_cot)`.
- **Readout carries signal.** `state_readout` at K > 0 beats `silent`
  at the same N by ≥ +0.02 rubric — the readout tokens carry
  information back into the final decode, not just noise.

Auxiliary signal: between-step state motion `‖state_N − state_{N-1}‖_2`
is monotone non-increasing with N (refinement converges, not
diverges).

**Falsification (per-claim).**
- If the default cell is Pareto-dominant (nothing beats it at ≤ 1.0×
  compute), all knobs collapse to Transformer conventions → the
  effort registry has no distinguishing content; drop the matrix
  back to N-only refinement scope.
- If `state_readout` ≈ `silent` at the same N (Δ < 0.02 rubric),
  readout tokens are non-load-bearing → keep matrix, drop the readout
  axis.
- If rubric decreases with N (state destabilises on re-feed),
  refinement itself is refuted — supersedes the matrix conclusion;
  register in `FAILED.md`.

**Related.** Track A (A0.8, extended 2026-07-22 from N-only sweep to
3D matrix). Directly follows H8-causal PASS: if state does work per
token, more state work should compound. Deliverable: a runtime
`effort` registry with noesis-specific presets (fast / normal / deep)
selected from the measured Pareto frontier, not copy-pasted from
Transformer effort levels. Design draft: `docs/effort-frontier.md`.
Competing axis: H12 asks whether the ceiling being probed here is set
by *single-state capacity* rather than by test-time compute per token.
Both are legitimate frontier directions but the tests answer disjoint
questions — H10 measures how far one state can be pushed, H12 asks
whether one state is the right unit at all.

**Status.** Untested; runner and eval design in
`experiments/A0.8_refine/` (pending). Scheduled after A0.6/A0.7 —
their verdicts may narrow the design space (e.g. if state turns out
not to survive re-feed at N > 1, the readout-mode axis collapses).

---

## H11. Zone-typed lenses beat monolithic text-bottleneck handoff

**Claim.** Cross-model handoff via **per-zone lenses** (DSL blocks for
`insights`/`vault`/`events` + DSL-rendered scratch-lens from the
incumbent model, paraphrased to prose only at the foreign-model edge)
preserves task success within 10 % of a full raw-log handoff while
using under 10 % of the tokens. Refinement of H5 — H5's "compact
structured summary" is generalised into a zone-typed DSL protocol that
covers resident-model swaps, not only remote-Claude escalation. The
runtime owns the wire format end-to-end; Ollama supplies token I/O.

**Prediction.** On ≥ 30 multi-turn tasks from an extended A0.2 pool
that require a mid-task model handoff:
- Task success on M_B under the lens bundle ≥ 0.9 × success under raw
  log
- Input-token cost of the lens bundle ≤ 0.1 × raw log cost

**Falsification.** If task success drops more than 10 % *or* token
cost exceeds 10 % of raw, per-zone ablation identifies which lens is
under-designed. If the *scratch* lens is the culprit specifically,
the model cannot reliably describe its own reasoning state — a much
stronger negative result that closes off runtime-owned memory as an
architectural bet and pushes noesis toward a text-only handoff
protocol (H5's original form).

**Related.** Track B, Track C (C3). Depends on Phase B/D of runtime
plan + A1 checkpoint + at least one alternative Ollama-servable
model. Design frozen in `docs/memory-lenses.md`. Interacts with
H5 (which becomes a special case: scratch-only lens, remote Claude
as M_B).

**Status.** Untested. Runs after A1 lands and Phase B/D seedling is
online. Not in Phase 1 critical path.

---

## H12. Working-memory bottleneck vs decay-rate bottleneck in WKV

**Claim.** RWKV-7's failure mode on cross-domain reasoning is
dominated by **active-representation width** — how many *distinct*
concepts the state can hold simultaneously — not by decay-rate over
distance. That is, the model *knows* the pieces (weights carry them)
but cannot hold enough of them active at once to discover cross-links.
If true, a multi-slot state extension (K parallel WKV slots per layer
with input-dependent gating and cross-slot read) buys more accuracy
than an equivalent-parameter widening of a single-slot state, at
comparable FLOPs/token.

**Motivation.** User intuition 2026-07-22: "модель работает как
процессор ... не хватает не знаний, а возможности собирать более
обширные представления". The multi-core analogy is misleading (CPUs
went multi-core against a thermal wall; models don't have one), but
the underlying observation — that working-memory width, not knowledge
count, may be the binding constraint — is empirically open. Prior art
in the direction: RetNet (multi-retention), Griffin (linear recurrence
+ sliding-window attention), Titans (learned long-term memory slot).

**Two disjoint failure modes to distinguish first (H12a).**

- *Decay-mode.* Error rate scales with token-distance to the referent.
  Close is remembered, far is forgotten.
- *Width-mode.* Error rate scales with the *number* of simultaneously
  active concepts required to answer, at *small* token-distance.

**Prediction (H12a — bottleneck attribution).** Construct a
cross-linking probe: N triples `(entity → property)` in a short
context (≤ 512 tokens), question requires finding all entity pairs
sharing a property. Sweep `N ∈ {4, 8, 16, 32, 64}` at fixed context
length on G1d-0.4B.

- If accuracy falls sharply with N at N ≪ context-length capacity,
  width is the bottleneck → H12b becomes worth running.
- If accuracy is flat in N but falls with mean triple-to-question
  distance, decay is the bottleneck → H12b drops; retrieval / longer
  effective context are the right fixes.

**Probe-design gap (registered 2026-07-23 after G1d 0.4B n=30 run).**
The v1 width sweep as-implemented in
`experiments/A0_H12a_working_memory/gen_triples.py` grows both **N**
and **mean gap-to-question distance** together (each triple adds
tokens, so N=64 lands with gap≈107 while N=4 lands with gap≈8). This
**confounds width with decay** — a fall in accuracy across N cannot
be attributed to either axis alone. The v1 distance sweep (fixed
N=8, gap ∈ {50, 200, 500, 1000}) is uncontaminated and did show a
clean decay signal (recall 0.40 → 0.02 between gap 14 and 229).
Verdict of the current data: **decay proven; width unresolved.**
A v2 probe design is required before H12a can gate H12b — one that
sweeps N at **fixed** gap by padding distractor filler between the
last triple and the question. Until v2 lands, treat H12a as decay-
positive-only, not width-attributive.

**Prediction (H12b — multi-slot fix, gated on H12a = width).**
LoRA-add `K = 4` parallel WKV slots per layer, input-dependent gating
routes each incoming token's contribution across slots, simple learned
merge (weighted sum with per-slot query) at readout. Retest H12a's
probe.

- If the largest N with ≥ 0.9 baseline accuracy grows by ≥ 2× under
  the K=4 variant at ≤ 1.5× FLOPs/token, multi-slot is validated.
- Ablation: equivalent-parameter widening of a single slot (same
  parameter budget, K=1) as a control — multi-slot must beat this,
  not just beat vanilla, to earn the architectural cost.

**H12b.i — utilisation regularizer (impl-detail sub-protocol,
added 2026-07-23).** MoE and multi-expert prior art (Switch
Transformer, Mixtral, GShard) consistently show that
input-dependent gating **collapses to a single slot** in the absence
of an explicit utilisation loss. If H12b is run without such a
regularizer and shows "K slots ≈ K=1 baseline", the negative result
may be a **training failure**, not an architectural refutation. The
sub-protocol below is therefore mandatory for H12b:

- **Slot-usage entropy loss.** Compute per-batch gating distribution
  across slots; add `−λ · H(p_slot)` to the objective. Encourages
  uniform slot usage. λ tuned to keep entropy above `log₂(K) − 0.5`
  at convergence.
- **Cross-slot dissimilarity.** Cosine-distance penalty between
  learned per-slot read-out projections; forces slots to carry
  distinguishable content.
- **Read-out coverage.** During eval, decode from each slot's
  read-out head separately and require that the K decodings not
  reduce to K identical outputs (measured by n-gram overlap ≤ τ).

**Motivation for H12b.i.** User observation 2026-07-23: "может нам
стоит учить модель использовать не просто одно представление а
специально обучать использовать множество таких областей". Correct
diagnosis of the training-vs-architecture confound. Without
utilisation regularization, H12b's "fail" branch cannot be
distinguished from "architecture correct but under-trained".

**Falsification (H12b + H12b.i).**
- Multi-slot with regularizer PASS on the H12a probe **and** slot
  entropy stays ≥ `log₂(K) − 0.5` at convergence ⇒ H12b validated,
  utilisation matters.
- Multi-slot with regularizer PASS **but** entropy is at ceiling
  and read-out coverage is degenerate ⇒ entropy loss forced formal
  spread while functional collapse persists; H12b.i design failed,
  redesign the loss.
- Multi-slot with regularizer FAIL ⇒ architectural refutation is
  now real; multi-slot is not the mechanism.

**Frontier note.** Anthropic's Claude Fable 5 (June 2026, Mythos
tier) is a public data point that dense-Transformer architectures
can sustain long autonomous work sessions — the exact "hold many
concepts active for a long horizon" capability H12 wagers is the
missing piece for RWKV-line models. The frontier's mechanism there
is unknown to us (likely a mix of scale, MoE routing, and internal
scratchpad protocols). H12b is our architectural bet for how the
recurrent-state family closes that gap; utilisation regularization
(H12b.i) is the training-protocol bet that makes the architecture
land instead of collapse.

**Falsification.**
- H12a v2 (fixed-gap width sweep) shows flat accuracy across N ⇒
  decay dominates; H12b as *multi-slot fix* is not worth running,
  because K slots each still decay. Fixes shift to retrieval /
  longer-window / different decay schedule / per-slot decay-rate
  learning (a *different* H12b variant, worth splitting out then).
- H12b fails despite H12a v2 pass ⇒ width is the constraint but
  multi-slot is not the right mechanism. Register in `FAILED.md`;
  falls back to widening single-slot state (dumber but cheaper).

**Frontier adjacency (Transformer side, 2026-07-23).** OpenMythos and
the MyTHOS-line MoE + memory-compression stack address the same
underlying question by *routing* rather than *widening*: expert
selection per token acts as a discrete cousin of multi-slot state
where different experts hold different sub-representations active in
parallel. Not equivalence — MoE routing operates on FFN blocks, not
on the recurrent state; the analogy is by function (parallel
sub-representations) not by mechanism. Useful marker: the frontier is
independently converging on "one dense representation is not enough,"
which is what H12b bets on for the RWKV state specifically.

**Related.** Track A, deferred from Phase 1 (H7 lock keeps logic
in weights, knowledge in context; multi-slot state is an
architectural change, not a Phase 1 lever). Adjacent to H8 (state-as-
computation) and H10 (test-time compute) — those probe *how* the
single state works; H12 probes *whether one is enough*. Adjacent to
A0.6/A0.7 verdict: if state is not portable between instances, any
multi-slot design must live inside one forward-pass, not across model
copies.

**Status.** H12a v1 partially run (G1d 0.4B n=30, 2026-07-23): decay
axis proven, width axis blocked by v1 probe design (see gap above).
H12a v2 (fixed-gap width sweep) pending. H12b LoRA blocked on H12a
v2; unchanged budget estimate (< 24 GPU-hours at 0.4B). Phase 2
architectural probe.

---

## H13a. State compresses geometry, not just token distributions
### *(wager, precedent-informed but not yet directly tested inside noesis)*

**Claim.** The state-as-computation dynamics that H8 probes for text
generalise to **visual patch streams** — the WKV state can absorb
2D geometric structure (patch tokens flattened in a fixed
raster / spatial curve order) and produce useful downstream
representations without needing an attention operator over the whole
image. If true, RWKV-7 becomes a natural **multimodal substrate**:
one architecture, one state format, text ⊕ image ⊕ (possibly) audio
under the same delta-rule update.

**Motivation.** User intuition 2026-07-22: image is a *representation
of geometry*, and RWKV's state — evolved per-token by a delta-rule
update — is a plausible place for geometric structure to compress.
Precedent (user-cited): the VisualRWKV line of work (BlinkDL /
academic follow-ups) already shows RWKV variants absorbing visual
tokens; this hypothesis is that the same phenomenon extends to the
G1-line state dynamics noesis is staked on. If P4 / H4b hold, they
should hold for image tokens too — the model does not "know" the
tokens are visual.

**Why it matters for noesis.** noesis observes a Linux session — the
richest single sensor is the framebuffer, not the keystrokes. If
RWKV-7 can absorb visual patches through the same state mechanism,
the runtime can eventually feed screenshots, wallpaper regions, video
frames straight into the model without a bolted-on vision head. This
is the difference between "noesis reads about what happened on the
screen" and "noesis saw the screen."

**Prediction (small-scale probe, before any noesis-side integration).**
Take G1d-0.4B, feed a patch-tokenised image (raster order, standard
patch size) as a prompt, then decode. Two disjoint claims:

1. **State-dynamics parity.** On a matched-length text prompt and a
   matched-length visual patch prompt, the state-motion metrics from
   H8 (delta-norm, curvature, stable rank) are within one order of
   magnitude of each other. The state is *doing something* with the
   visual input, not going flat.
2. **Task carry-through.** Fine-tune a small readout head on top of
   the final state for a coarse visual task (e.g. CIFAR-10 or
   MNIST-scale classification) and reach ≥ 0.7 accuracy at very
   modest data budget (< 100 k examples). Baseline is a random-init
   RWKV of matched parameter count fine-tuned on the same data.

**Falsification (per-claim).**
- If state metrics on visual prompts collapse to noise (delta-norm
  drops by an order of magnitude vs text at matched length), the
  state does not engage with patch tokens — visual generalisation of
  state-as-computation is refuted; multimodal support has to come
  from a bolted-on vision encoder, not from state alone.
- If the readout head cannot beat the random-init baseline at any
  data budget, geometry does not compress into the state usefully.

**Related.** Phase 3+ direction; not on Phase 1 or Phase 2 critical
path. Depends on H8 verdict (state must first do work for text).
Adjacent to VisualRWKV literature — this hypothesis is the *noesis-
side reason* to care about that literature, not a claim of novelty
over it. If PASS, ROADMAP Track B expands to include a passive
visual observation collector (screenshot cadence, framebuffer
snapshot) as first-class alongside keyboard / journal input.

**Status.** Untested. Speculative wager. Recorded 2026-07-22 as a
future-Phase direction rather than a near-term probe.

---

## H13b. Image-in-context beats text-digest for screen-content tasks
### *(near-term, well-supported by precedent — cheap to test)*

**Claim.** For tasks where the input is a rendered screen, a
vision-capable model that receives the **screenshot itself as
context** (patch tokens or native-vision channel) outperforms an
otherwise-identical text-only pipeline that receives a
carefully-digested textual summary of the same screen. The precedent
is broad: Claude Vision, GPT-4V, Gemini, and the MyTHOS-line vision
reconstruction demos all show frontier models routinely reasoning
about layout, whitespace, colour cues, and iconography that no
practical OCR-plus-digest pipeline captures without hand-tuning.

**Distinction from H13a.** H13a is the *architectural* wager (state
absorbs geometry). H13b is the *pipeline* wager (feeding the raw
pixels into a context window is already yielding, today, on
mainstream vision-capable models). H13a says "the WKV state can be
the vision head"; H13b says "wherever the vision head lives, don't
throw away the image before you reason." H13a's outcome is
independent of H13b — H13b can hold with a bolted-on encoder just
fine.

**Motivation (2026-07-23).** User push-back against a split
perception-backend + reasoning-backend architecture: coordination,
format translation, and latency across two models are real costs;
the frontier is investing in unified multimodal models
(vision-language, and MyTHOS-line where Recurrent-Depth Transformers
absorb visual input through the same forward pass) rather than
gluing two backends together. H13b captures the near-term evidence
that image-in-context is the dominant strategy in practice.

**Prediction.** On a coarse screen-content classification benchmark
(≥ 30 held-out screenshots from the user's real Linux session, five
labels: `code_editor / terminal / browser_docs / video_media /
other`), a vision-capable model with the screenshot in context
outperforms the same-class text-only model with a
carefully-digested textual description of the same screen by
**≥ 2× accuracy** (measured as either overall accuracy on a
class-balanced set or F1 macro on a class-imbalanced one).

**Falsification.**
- If the text-digest baseline reaches within 0.5× accuracy of the
  vision-in-context path, the practical case for image-in-context on
  *this* task class is weak — text digestion is enough. Would push
  noesis toward keeping the vision channel out of the critical path
  and re-investigate for finer-grained tasks (UI-element extraction,
  spatial reasoning) instead.
- If the vision-in-context model is confused by the raster order or
  tokeniser choice (accuracy at chance), the pipeline is broken, not
  the hypothesis. Fix and re-run before drawing conclusions.

**Related.** Track B (visual observation collector, gated on
verdict) and Track C (screenshot-in-context handoff, C2/C3 side).
Cheap to test — needs a small labelled screenshot set and a
vision-capable Ollama-servable model, both attainable in-week.
Interacts with H11 (lens design) — if H13b holds, `screen` becomes
a first-class zone alongside `events` / `insights` / `vault`.

**Status.** Untested. Near-term candidate for the next probe cycle
after A0.3 completes.

---

## H14. Domain competence via targeted Phase-2 SFT, not Phase-1 weights
### *(deferred, Phase 2; sits behind H7 lock)*

**Claim.** Once the Phase-1 logic-fine-tune (A1) has landed and shown
that reasoning competence is present in weights, a **narrow domain
SFT** — RFC corpus (~9500 RFCs, rfc-editor.org), CLI tooling docs
(man pages, tldr, `--help` dumps), technical spec material — can
lift the model's ability to *act* on domain tasks without violating
the H7 knowledge-in-context wager. The distinction is subtle but
load-bearing: general knowledge stays in the runtime context via
retrieval; **formal-IT vocabulary, protocol structure, and the "shape"
of technical prose** enter through weights so the model can *parse*
retrieved documents fluently rather than treating them as foreign
text.

**Prediction.** On a Phase-2 A0.2-successor eval that requires acting
on RFC-adjacent tasks (e.g. reasoning about a protocol message given
the retrieved RFC excerpt, filling in a `curl` invocation from a
retrieved API doc):
- Post-H14 SFT model beats pure-A1 baseline by ≥ 10 pp task success
  when retrieval is available.
- Without retrieval, post-H14 model does *not* meaningfully beat A1
  baseline. If it does, the SFT leaked *knowledge* rather than
  *structure*, which is a data-curation failure to file in FAILED.md.

**Falsification.**
- H14 fails ⇒ RFC-shaped structure did not transfer to task acting;
  either the SFT recipe is wrong (data cleaning, prompt format) or
  the H7 lock is too narrow — technical vocabulary genuinely needs
  retrieval-only handling. Log the ablation in FAILED.md and consider
  a retrieval-heavier alternative.
- Without-retrieval gain ⇒ knowledge leaked into weights; either
  tighten the corpus filter or drop H14 as incompatible with H7.

**Data-curation constraint (locked with H14).** RFC corpus goes in
verbatim; personal chat logs and Compilerium contents remain
**excluded** from H14 SFT — the exclusion is not lifted by the H15
persona carve-out below (H15 uses chat logs for register only,
H14 for structural competence).

**Related.** H7 (understanding in weights, knowledge in context) is
the Phase-1 lock this hypothesis sits behind — H14 is worth running
only after A1 confirms reasoning transfer. Adjacent to A3 in ROADMAP.

**Status.** Untested. Phase 2. Budget: ~8 GPU-hours QLoRA on 0.4B
for the ablation, ~24 GPU-hours on 2.9B for the production run.

---

## H15. Persona-SFT to a dry butler/secretary register beats default helpful-assistant tone
### *(Phase 2 style probe; complements H14)*

**Claim.** A short persona-SFT pass (a few thousand curated turns) on
the "peer Linux user" register — short, factual, task-oriented,
minimal hedging, no gratuitous restatement of the user's question —
produces **higher task density per turn** than the default
helpful-assistant register the G1 line ships with, while consuming
fewer tokens per interaction. Data source: **user's own chat traces**
(the CLAUDE.md carve-out for narrow persona use — never knowledge or
reasoning).

**Motivation.** noesis is framed in CLAUDE.md as a *peer Linux user*
running continuously alongside the human, effectively as another
tenant on the machine — not a conversational assistant. Its output
should read like a butler or a secretary: concise, formal-neutral,
task-completed-not-narrated. The default G1d tone (chatty, hedging,
restating context) is a mismatch for daily co-existence and
inflates token spend on background loops.

**Prediction.** On an A0.2-style task-density eval with matched
retrieval and matched CoT budget:
- Persona-SFT variant produces ≥ 20 % fewer output tokens per task
  than baseline G1d at equal task success.
- Persona-SFT variant produces higher task success at equal token
  budget (i.e. the register itself is compute-efficient, not just
  compact).
- Ablation: persona-SFT should not degrade H12a-style working-memory
  probes — if it does, the persona training is chewing capacity that
  reasoning needs.

**Falsification.**
- Tokens/task not measurably lower ⇒ the register was cosmetic, not
  compute. Log in FAILED.md.
- Task success drops ⇒ persona-SFT damaged reasoning; the "butler"
  register is stylistically appealing but semantically expensive.
  Choose one of: (a) mix in more logic examples during persona-SFT,
  (b) drop persona-SFT and accept default tone.

**Data governance (locked with H15).** Personal chat traces enter H15
supervision **only as persona/style signal** — the corpus filter must
strip factual content, project names, and any task-specific detail
before SFT ingest. The residual signal is *register*, not
*information*. If any downstream eval shows H15 leaked user-specific
knowledge back into weights, treat as data-curation failure and
retrain with a tighter filter.

**Related.** H7 (knowledge in context) — H15 must not violate H7 by
smuggling knowledge under the persona flag. H10 (test-time compute
per token) — if the butler register genuinely compresses output, it
shifts the effective H10 budget upward for the same wall-clock cost.

**Status.** Untested. Phase 2. Budget: ~4 GPU-hours QLoRA on 0.4B
after H14 SFT completes; the two SFT passes can be merged or
sequenced.

---

## H16. Gated externalisation from a rate-limited silent think-stream
### *(wager, Phase 3+, informs runtime architecture)*

**Claim.** A production-grade "peer" model — one that *lives*
alongside the user rather than answers when polled — must be able to
**self-initiate output**. RWKV-7 as shipped is strictly autoregressive
and needs an external token to fire generation. If the noesis runtime
runs a **rate-limited silent think-stream** (each generated token
updates WKV and is discarded rather than externalised, at a target
rate ≤ R tokens/sec chosen so package CPU stays inside the H1
envelope) and a small **gated-emit head** classifies each such token
as *keep silent* vs *emit*, then the model self-initiates speech
from within its own state dynamics — not from a supervisor-driven
polling loop.

**CPU-budget grounding (measured 2026-07-23).** Burst generation on
0.4B G1d **via Ollama's llama-server** on i5-1235U measured at 18.6
tok/s, consuming 0.106 CPU-seconds per token, ≈ 190 % of one core ≈
15.8 % of package (12 threads). H1 caps steady-state package CPU at
< 1 %. Continuous-burst think-stream therefore breaks H1 by ~16×.
Analytical extrapolation (linear in R, since per-token cost is
constant at fixed batch=1):

| R (tok/s) | package CPU % | one-core equiv | fits H1<1% |
|----------:|--------------:|---------------:|:----------:|
|      0.10 |         0.089 |          1.06  |     YES    |
|      0.25 |         0.221 |          2.66  |     YES    |
|      0.50 |         0.443 |          5.31  |     YES    |
|      1.00 |         0.885 |         10.62  |     YES    |
|      1.13 |         1.000 |         12.00  |    edge    |
|      2.00 |         1.771 |         21.25  |     NO     |

**H1 ceiling: R_max ≈ 1.13 tok/s** (single instance, current
backend). At R_max: latency to first drip token ≈ 0.89 s. When we
move from Ollama's llama-server to direct rwkv.cpp bindings (C0
verified at ~30 tok/s on Q8_0), per-token cost drops ~1.6×, so
R_max rises to ~1.8 tok/s at same H1 ceiling. The trade sits on a
supervisor-tunable knob; guarantees are analytical, no 24 h probe
needed to bind R to a CPU% guarantee.

**Motivation.** User framing 2026-07-23: "он в любой момент срывается
отвечать по собственному усмотрению также из think-токенов
порождённых wkv" — the runtime's job is not to poll the model at N Hz
asking "should you speak now?", but to give the model an outlet
that fires when its own dynamics cross a threshold. This is what a
peer would do; a supervisor-polled model is closer to a servant. It
also composes with H13a: if visual patches stream into the WKV state
per-frame and the model's silent think-stream integrates them, the
gate can fire in response to a visual event (e.g. an error appears on
screen) with no polling loop needed.

**Two candidate architectures.**

- *Poll-mode (baseline, currently implemented).* Supervisor loop runs
  every T seconds, issues a synthetic "should-speak?" query, reads the
  response head, decides whether to externalise. Cheap; not
  self-initiating. **This is what noesis-runtime does today** — H16
  is the ambition to replace it.
- *Gated-emit (the H16 wager).* A small MLP head over the WKV state
  produces `p(emit | state)` after each internally-generated (drip)
  token. Training signal: transcripts of "when the user actually
  wanted a response" vs "when silence was correct" — collectable
  from noesis's own runtime traces (see H15 carve-out) plus
  synthesized negative examples. Fires generation into a real
  output stream when `p(emit) > τ` for `k` consecutive tokens; on
  fire, the runtime switches the same forward-pass loop from drip
  rate to burst rate for the reply, then returns to drip.

**Prediction.** With a well-trained gate at drip rate R = 0.5 tok/s
(Ollama backend, measured):
- **False-fire rate** ≤ 1 per 8 h of idle screen time. Higher rate ⇒
  runtime spam, gate is under-trained.
- **Latency to fire** on a triggering visual event (H13a integration)
  ≤ 5 s median at R = 0.5 tok/s (⇔ ≤ 2–3 drip tokens between event
  and threshold cross). If unmet at R = 0.5, R is a supervisor-tuned
  knob — the gate is scored on latency-per-CPU-percent, not absolute
  latency.
- **Package CPU** of the drip stream ≤ 1 % steady state (H1 ceiling).
  At R = 0.5: 0.44 % package (measured 2026-07-23). Ceiling gives
  ~2.3× headroom on R before H1 breaks. When runtime moves to direct
  rwkv.cpp bindings (~30 tok/s Q8_0), headroom rises to ~3.6×.

**Falsification.**
- False-fire rate above ceiling ⇒ gate under-trained or the runtime
  observation traces are too sparse to teach silence. Register in
  FAILED.md and revert to poll-mode with a heavier salience filter.
- CPU ceiling requires R so low that latency > 30 s median on
  triggering events ⇒ drip cadence can't buy useful reactivity on
  Alder Lake; H16 becomes a cloud-bursted / bigger-headroom-hardware
  probe only.

**Related.** H1 (constant-cost background operation) — H1 sets the
drip-rate ceiling; the whole rate-limited framing exists specifically
to fit inside H1. H10 (test-time compute per token) — H16
fundamentally converts *test-time compute* into *decision to speak*,
so the two share metrics; in particular, H10's `readout_mode = silent`
(N-refinement without emitting tokens) is the natural cousin of the
drip stream, and a fused runtime would pick between "N passes then
gate" (one shot) and "drip stream + gate" (persistent) as two modes
of the same knob. H13a (state absorbs visual patches) — H16 gains
sensory grounding for its fires when H13a lands. H15 (dry-formal
register) — the emit-gate's training data comes from persona-labeled
traces; H15 must land first so the register is stable when H16
learns *when* to use it.

**Status.** Untested. Phase 3+ (post A1, post H13a probe, post H15
persona pass). No budget estimate yet — depends on whether the
gate can be trained with a few-K examples (~2 GPU-hours) or requires
online RL from runtime traces (>> that).

---

## Architectural note — unified multimodal RWKV, not split backends
### *(locked 2026-07-23)*

**Decision.** noesis targets a *unified* multimodal backbone (one
model, one state format, text ⊕ image ⊕ (possibly) audio through
the same delta-rule update) rather than a split perception-backend
+ reasoning-backend architecture. If, at any point, adding a vision
head means introducing a second model with a serialised handoff
protocol between the two, the architectural drift needs to be
challenged before committing.

**Why.** Split backends carry real costs:
- **Coordination overhead** — two schedulers, two lifecycles, two
  memory footprints resident.
- **Format translation** — perception-side output has to be
  serialised into text (or a synthetic embedding format) that the
  reasoning-side model can consume; the serialisation itself is
  lossy and slow.
- **Latency stack-up** — inference on both models in sequence, plus
  the translation step, dominates any per-step wins from
  specialising each backend.
- **Frontier signal.** MyTHOS-line and OpenMythos work
  (Recurrent-Depth Transformer + MoE + memory-compression) is
  investing in *state-side* computation and multimodal-in-context,
  not in inter-model orchestration. If the frontier is unifying, a
  small research project should not be gluing.

**How this shapes near-term work.** H13a and H13b are the two probes
that inform the unified-substrate wager. H13b is the cheap
near-term test (does image-in-context yield with any vision-capable
substrate?); H13a is the deep wager (does the *RWKV* state itself
carry that yield without a bolted-on encoder?). Both are worth
running; neither justifies introducing a second local reasoning
model to service perception.

**Escape hatch.** If H13a fails clearly (state cannot absorb visual
tokens) *and* H13b holds (image-in-context yield is real, but only
via an external vision head), the escape hatch is a *fused*
architecture where the vision head produces tokens or embeddings
consumed inside the same forward pass of the reasoning backbone —
not a split-backends handoff protocol. This preserves the
single-cognitive-engine constraint from `CLAUDE.md`.

**Recorded from.** User push-back 2026-07-23 in response to a design
sketch that proposed a split perception/reasoning stack. Recorded to
prevent architectural drift over the next 3–6 months while H13
probes are pending.

---

## H17. State-substrate absorption substitutes for message-history re-injection
### *(wager, Phase 2, empirical arm of H4b)*

**Claim.** For a state-substrate reasoning model (RWKV-7 G1 2.9B),
prompts that carry `tail_turns = K` recent user turns + retrieval +
current query achieve response quality equivalent to prompts that
carry full message history, **provided** the substrate's WKV state
has already absorbed the older context. Below an absorption
threshold `T_absorb`, the K-tail transform loses signal that
full-history would preserve.

**Why this matters.** The context-management transform in
`noesis-http` (see runtime plan §10: K=4 tail + retrieval + preamble,
older messages dropped) is currently justified by the
state-evolution framing (H4b): substrate already holds absorbed
context, re-injecting duplicates state and burns CPU (H1 violation).
This is a *wager*, not a measurement. If wrong, response quality
degrades below full-history baseline exactly in the cases where
substrate has not yet absorbed the relevant material — cold start,
freshly-hydrated lens, sharp topic shift, or content that was in a
message but never made it into a WKV update (short session with
early state save).

**Prediction.**
- At high runtime age (state has absorbed ≥ 100k tokens of context
  since lens hydration): `quality(K=4) ≈ quality(full)`, both
  within LLM-judge noise.
- At low runtime age (state absorbed < 10k tokens): `quality(full)
  > quality(K=4)` by a measurable margin on retrieval-heavy or
  history-dependent queries (co-reference, "as I said earlier",
  topic follow-up).
- Transition at some `T_absorb` between these two regimes — the
  measurable quantity of interest.

**Regulation mechanism (if hypothesis holds).** Adaptive
`tail_turns` based on a state-saturation signal. Open sub-question:
what signal?
- **Explicit token counter**
  (`state_absorbed_tokens_since_lens_hydrate`). Cheap, unambiguous.
  Downside: does not know if absorbed tokens were *relevant* to the
  current query.
- **Entropy of hidden state** — higher entropy = less-informed
  state = need more history. Requires a probe pass, adds cost.
- **Composer-scored novelty of `last_user_query`** vs recent
  retrieval hits — query for material not yet in insights = needs
  more history. Reuses composer output.

Not decided; test with the simplest (token counter) first, escalate
only if it misses.

**Falsification.**
- `quality(K=4) < quality(full)` even at high runtime age →
  substrate does not effectively absorb history for reasoning
  purposes; §10 transform is wrong; revert to fuller history or
  hybrid.
- `T_absorb` so high (e.g. > 1M tokens) that noesis rarely reaches
  it in real deployment → K-tail transform is a Phase-3 idea
  masquerading as a Phase-2 default; ship with full history until
  substrate proves it can catch up.
- Quality curve flat across `K ∈ {0, 2, 4, 8, full}` → tail turns
  irrelevant; retrieval + query alone suffices; `K=0` is the right
  default (further simplification, retrieval carries all the load).

**Measurement setup.**
- **Eval set.** Retrieval-heavy queries + history-dependent queries
  (co-reference, topic-shift-followup). Ideally the same set used
  for A1 SFT eval so results feed back.
- **Runtime age control.** Run against fresh lens hydration for
  low-age; run against lens with N minutes of prior conversation
  for intermediate/high age.
- **Metric.** Response quality via LLM-judge (cheaper) or
  held-out human-scored set (higher trust). Start with LLM-judge;
  escalate to human on any surprise.
- **Variables.** `tail_turns ∈ {0, 2, 4, 8, full}`, `runtime_age
  ∈ {0k, 10k, 100k, 1M} absorbed tokens`. 4 × 5 = 20 conditions.
- **Compare.** `quality(K, age)` matrix vs baseline
  `quality(full, any_age)`.

**Recorded from.** User push-back 2026-07-24: the K=4 transform in
runtime plan §10 was framed as decision; user correctly noted it is
an untested wager on state-substrate absorption, "an obvious
double-edged sword" that could either fix H1 waste or bleed context
in early-runtime sessions. Elevated to falsifiable hypothesis so
metrics govern the transform, not an untested assumption.

**Status.** Untested. Phase 2 (blocked on: composer + tool-
dispatcher landing so §10 transform can be measured against a
baseline).

**Related.** H4b (state-evolution wager) — H17 is the empirical
arm. H1 (CPU budget) — the whole motivation for K-tail is H1
compliance. H7 (understanding-in-weights) — H17 tests whether
*runtime context in state* substitutes for *prompt-injected
history*; H7 is the parallel claim about weights.


