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
