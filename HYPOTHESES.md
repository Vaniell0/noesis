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

**Prediction.** Under a realistic 24-hour background workload
(event-stream ingestion + hourly summary + retrieval on demand), the
process consumes < 3 GB resident RAM, < 10 % average CPU on a mid-range
laptop, and does not measurably degrade battery life beyond ~10 % at
idle.

**Falsification.** Sustained operation for 7 days at Gate 2 either meets
these numbers or does not. If it does not, investigate quantisation,
inference framework, and backbone in that order before concluding the
architecture itself is wrong.

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
