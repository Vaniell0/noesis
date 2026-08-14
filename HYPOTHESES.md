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

**Companion files.**
- `FAILED.md` — refuted hypotheses and abandoned bets, with evidence
  and what changed as a result.
- `PASSED.md` — verified claims and settled sub-questions, so the
  audit trail has a positive side, not only a graveyard.

---

## Overarching framing (2026-08-12)

The WKV state is not a rolling summary — it is a **world model register**.
Each delta-rule update is a write into a fixed-capacity compressed
representation of the current situation. The `<think>` tokens in L_state
training are not chain-of-thought for human readers; they are **sequential
WKV state writes** — deliberate updates to the world model before the model
commits to an action.

This is what noesis is uniquely building:
- `L_state` loss = trains the model to maximise write bandwidth per think
  token (inverted SFA on the world model register)
- N=2 structure = write phase (chunk 1 builds world model from prompt) +
  readout phase (chunk 2 selects action from world model state)
- Persistent WKV across sessions = the world model survives between turns

Nobody else trains for this explicitly. State tuning (Scarletwolf/RWKV-PEFT)
installs a fixed initial state (disposition). LoRA trains weight deltas
(procedure). L_state trains the *dynamics* — how fast and how richly the
model writes into its world model register during generation. The two are
complementary: state tuning sets the prior, L_state training sets the
update rule quality.

BlinkDL's agent direction and noesis's world-model direction are compatible:
agents need a world model to act from. L_state-trained G1i gives the agent
a richer internal state per think token, reducing the token cost of
grounding before action.

H8–H10, H12, H17 are all sub-questions of this overarching claim.
The full empirical program is: measure write bandwidth (H8), show training
amplifies it (H9), map the N×K effort frontier (H10), characterise capacity
limits (H12), and demonstrate that persistent state substitutes for
re-injection (H17).

---

## Navigation

Grouped for orientation. Full text follows below in numeric order.

**Retracted to operating policy** (was a hypothesis, is now a policy
decision in `docs/policies.md` — the numeric slot is preserved).
- **H1** — Constant-cost background operation. *Retracted 2026-07-25;
  body moved to `docs/policies.md` § CPU / thermal.*

**Reasoning core — architecture and training (Track A).**
- **H2** — Reasoning-first outperforms knowledge-first at this scale.
- **H4a** — RWKV-7-G1 2.9B reaches parity with same-size Transformer.
- **H4b** — State-evolution architectures are viable for reasoning *(wager)*.
- **H7** — Understanding in weights, knowledge in context.
- **H8** — State-as-computation in RWKV-7.
- **H9** — G1-line training amplifies state utilisation.
- **H10** — Test-time compute frontier — state × tokens × readout.
- **H12** — Working-memory bottleneck vs decay-rate bottleneck in WKV.

**Memory system + cross-model handoff.**
- **H3** — Learned memory policy trumps heuristic memory at small scale.
- **H5** — Inter-model state transfer via compact structured summary.
- **H11** — Zone-typed lenses beat monolithic text-bottleneck handoff.

**State + context management (runtime state-work workstream).**
- **H17** — State-substrate absorption substitutes for message-history
  re-injection.
- **H18** — Git-like branch/merge WKV for indefinite structured
  continuation *(new 2026-07-25)*.

**Multimodal substrate.**
- **H13a** — State compresses geometry, not just token distributions *(wager)*.
- **H13b** — Image-in-context beats text-digest for screen-content tasks.
- *Architectural note — unified multimodal RWKV, not split backends.*

**Runtime as peer (behaviour, persona, self-initiated speech).**
- **H6** — Cognitive layer on modest hardware.
- **H14** — Domain competence via targeted Phase-2 SFT, not Phase-1
  weights.
- **H15** — Persona-SFT to a dry butler/secretary register beats default
  helpful-assistant tone.
- **H16** — Gated externalisation from a rate-limited silent
  think-stream.

**Truth system / epistemic behaviour** *(added 2026-07-29; treats the
model's own honesty as a first-class research object, not a byproduct
of good WKV-mechanism claims. H2/H7/H8/H10/H16 all touch pieces of this
implicitly; H19–H21 make the claims falsifiable in their own right).*
- **H19** — Weight-knowledge contamination detector (empirical arm of
  H7).
- **H20** — State holds contradictory belief pairs without premature
  collapse.
- **H21** — Premise-validity readout — model refuses invalid premises
  before answering.

---

## H1. Constant-cost background operation *(retracted 2026-07-25 → operating policy)*

**Status.** *Retracted as a falsifiable hypothesis on 2026-07-25.*
The question "does the runtime stay silent on the user's hardware" is
not a research bet — it is an **operating-policy decision** enforced by
per-machine calibration and the drip-rate accountant in the supervisor.
There is no experiment whose result could refute a *policy* — there is
only a policy that is either implemented and working, or not.

User rationale (2026-07-25): "почему это стало гипотезой, я честно
сомниваюсь зачем это вообще выводили отдельно, самое важное было бы
проверить состояние бд через 24, как модель справляется и как
регулировать окены это скорее полигон тестов и финальных конфигураций
чем проверка что это возможно в принципе". Correct: fan-off / CPU-budget
is a configuration problem, not a claim about the world.

**Where the content moved.**
- **Fan-off invariant + calibration protocol** → `docs/policies.md`
  §CPU / thermal (single hard rule: no audible fan spin-up in ambient
  mode; drip rate derived per-machine at startup).
- **Ambient vs interactive regime split** → same policies.md section.
- **Startup calibration algorithm** → `docs/policies.md` +
  implementation in the `calibration` module of `noesis-runtime`.
- **Drip formula and reference numbers** → runtime plan §11 (thermal)
  and `docs/policies.md`.

**What is still a real research question — spun off, not carried under
H1.** "Does the system as a whole work as intended over a real 24 h?" —
that is the A0.3 **polygon test** (three axes: DB state after 24 h,
model coping under sustained load, token-regulation policy actually
landing in a workable range). Recorded in `ROADMAP.md` §A0.3 (rewrite
pending) and in the project's active memory. It is not a falsification
of a hypothesis; it is validation of a design.

**Numbering.** The H1 slot is preserved to avoid renumbering downstream
references (H16 in particular cites "H1 envelope"). Future readers
following those cross-references land here, learn it is policy now, and
follow the pointer to policies.md. Do not re-use the H1 number for a
new claim — if a genuinely new load-bearing claim appears, allocate
the next free number.

**Related.** `docs/policies.md`; H16 (drip-rate ceiling still comes from
policy calibration, not from a re-instantiated H1 claim).

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

**Related.** Track A (A1), Gate 2. P14 (agility over omniscience) —
H2 is the training-time expression of that principle. If H2
falsifies, the "reasoning-first small model matches knowledge-first
same-size" wager fails and P14 has to survive on substrate
properties (H19/H20/H21) alone.

**Corpus selection constraint (derived 2026-08-06 from Step 5 analysis).**
H2 is untestable by any purely *reactive* corpus. A reactive corpus
(e.g. glaive-v2: 1–2 tool_use calls per session, user → tool_call →
result → answer) treats each turn as an independent lookup; it does not
require the model to accumulate context across many steps. RWKV's
architectural advantage — WKV state as an explicit working memory that
compounds across token positions — is only exercised by *reflexive*
multi-step sequences: ReAct-style think→act chains, ToolBench rollouts
(10–50 steps), or Claude CLI action chains (10–70 tool_use calls per
session, stripped to tool_use targets + user/result context).

Training on reactive corpora cannot refute or support H2 because the
corpus does not load the mechanism (state accumulation) that H2 depends
on. All Step 4/5 A1 measurements on glaive-v2 are architecturally
confounded and must be excluded from H2 evidence. Step 6 corpus must be
reflexive-first; the corpus-architecture fit is a prerequisite for any
valid H2 test, not a post-hoc tuning parameter.

**Status.** **WITHDRAWN (2026-08-14).** A1 results make the direction obvious without the
formal test: step9b-e1 (2.9B, reasoning LoRA) scores 39.6% overall and 87.5% on symbolic
vs Gemma4 e4b (37.5% / 37.5%) — reasoning-tuned RWKV beats a larger mixed-corpus model on
reasoning tasks. The formal comparison (Qwen-2.5-3B-Instruct or Phi-4-mini + retrieval vs
noesis + retrieval) was never run. Interest in this as a standalone research question has
faded; the sub-claims that remain live are carried by H12a (state-capacity reallocation),
H17 (substrate absorption vs re-injection), and H24 (decoding efficiency (DE) under RL). Closing without
scheduling a test that would add little beyond what A1 already showed.

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

**Experiments.** `experiments/A0_baseline/` — throughput + reference-model
comparison scaffolding landed; 2.9B eval blocked on GPU access, 0.4B
pilot numbers available (`results.md`). Full A0.2 eval against reference
models runs after A1 checkpoint lands.

**Status.** Untested at 2.9B target scale.

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
current hardware — laptop i5-1235U (CPU-only) — without cloud dependency
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
justification for the Phase-1 corpus discipline. P2 states the
principle; P14 (agility over omniscience) states *why* the split
matters — a channel-capacity-constrained model that pretends to
know everything is confabulating, so H7's boundary is what makes
honest not-knowing possible. H19 measures whether the boundary
actually held after A1.

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

**Prediction (qualitative — quantitative thresholds (to be locked post-A1 pilot) after first
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

**Frontier adjacency (WKV Jacobian / J-lens, 2026-08-11).** Analytical
mean-field Jacobian of the WKV update rule achieves **0.76 cosine
similarity** with the numerically computed Jacobian on G1h 2.9B
(source: `clehaxze.tw/gemlog/2026/08-07-notes-on-replicating-j-lens-on-rwkvv7`).
This shows the WKV update has tractable gradient structure — no chaotic
regime, mean-field approximation is tight. Mechanistic consequence:
`L_state` (which rewards `‖state_t − state_{t-1}‖` during think spans)
is a **training-time proxy for maximising singular values of the WKV
update Jacobian**. J-lens (mechanistic interpretability via Jacobians)
measures these eigenvalues post-hoc; `L_state` trains for them
directly. **Testable prediction:** step9 checkpoint should show higher
WKV Jacobian eigenvalues in `work_layers` than g1h-base. This is the
J-lens probe to run after A1. Note: per-token Jacobian norm varies —
our binary ε-mask (think=1.0 / non-think=0.05) is an approximation of
the true per-token Jacobian weighting. A continuous per-token weight
`α_eff(t) = α · jac_norm(t)` is a natural refinement candidate after
J-lens data is in hand.

**Related.** Track A (A0.4). Feeds into A1 loss-formulation decision
(see ROADMAP Gate 1 exit criteria).

**Experiments.** `experiments/A0_state_probe/` — pilot 2026-07-21 done
(3 seeds × 128 tokens on World-0.4B × medium, `results/pilot/`);
thresholds locked from pilot (see table above). Full A0.4 sweep across
paired probes (H8/H9 shared) pending A0.5 completion; A0.5 causal grid
in flight (`experiments/A0_state_probe/a05_run.py`,
`a05_analyze.py`, `results/a05_ext/`).

**A0.5 result (2026-08-04).** All H8-causal sub-tests passed. σ-slopes
1.56–2.10 across all 4 cells (reasoning × G1d / World3); cross-prompt
state-motion ratio 21–99×; monotone response to layer depth observed.
H8 SUPPORTED at 0.4B.

**A0.5 result 2.9B (2026-08-05).** G1h-2.9B, medium+narrative cells, CUDA.
σ-slopes 1.58–1.67 (superlinear, monotone); cross/base ratio 34–36×;
argmax_flip 0.833 in both directions; layer hotspots L4/L16/L20.
All three H8-causal sub-tests pass. Results in
`experiments/A0_state_probe/results/a05_g1h_2.9b/verdict.md`.
H8 SUPPORTED at 2.9B.

**IB reconstruction probe (2026-08-12, G1d 0.4B, 13 393 tokens).** Linear probe
predicts WKV state at t from state at t−lag. Baseline CE = 6.53 (no state).
Frac explained: lag=1 → 95.9%, lag=8 → 82.1%, lag=64 → 78.3%. Gradual decay
without collapse to baseline is consistent with WKV state carrying predictive
information over ≥64 tokens (one probe, one model size, not yet replicated). Results: `experiments/ib_probe/results_g1d_base/reconstruction.json`.

**R-lens diff G1h base → step9b-e1 (2026-08-14).** `experiments/A0_state_probe/results/rlens_g1h_vs_step9b_e1.json`.
Three contexts (reasoning / math / narrative), 32 layers. Consistent direction: small SR
decrease (Δ ≈ −0.01 to −0.03) and small σ1 decrease across all layers and contexts.
Most prominent change: L31 reasoning σ1 46.1→41.1 (−4.9), L31 narrative 52.3→50.4 (−1.8).
Narrative context shows near-zero change at most layers — consistent with step9b training
not including narrative-style data. Interpretation: LoRA training made WKV saliency slightly
more concentrated without restructuring the broad geometry; the intervention is narrow,
not a wholesale state-dynamics rewrite. Supports the view that step9b-e1's symbolic 87.5%
accuracy gain comes from targeted pathway adjustment, not global state restructuring.

**Status.** **SUPPORTED** at 0.4B and 2.9B (A0.5 passed both scales,
2026-08-04/05). A1 state_reg sweep findings in
`docs/verdicts/2026-08-04-a1-pilot-step3-step4.md`.

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

**Prediction (qualitative — thresholds (to be locked post-A1 pilot)).** On the same reasoning
prompt with matched seeds, at least one of the three A0.4 metrics
(delta-norm, curvature, stable rank) shows a statistically significant
G1h-vs-World3 difference (Welch's t-test, α = 0.05, corrected for
three metrics via Bonferroni or equivalent), with the direction
consistent with "G1h uses state more actively".

**Falsification (to be locked post-A1 pilot).** If G1h and World3 are
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

**Experiments.** `experiments/A0_state_probe/` — shares the H8 probe;
paired G1d vs World3 conditions on the same runner. Pilot subset ran
alongside H8 pilot 2026-07-21. Full sweep verdict feeds A1 loss
formulation (SFT-only vs state-regularised, α = 0 vs α > 0).

**A0.5 result (2026-08-04).** G1d shows measurably larger state motion
than World3 on matched reasoning prompts (cross-prompt ratio 21–99×,
σ-slopes consistently higher in G1d cells). H9 SUPPORTED at 0.4B:
G1 training amplifies state utilisation, not only output distribution.

**A0.5 result 2.9B (2026-08-05).** World3 vs G1h, both directions, CUDA.
σ-slopes G1h (1.67, 1.58) > World3 (1.19, 1.13); World3 medium
non-monotone (monot=False). G1h gauss@σ=0.1 KL ≈ 40× higher than
World3 (0.23 vs 0.0055) — G1h routes more computation through state.
Counterintuitively, World3 has higher raw cross_KL (9.64 vs 7.92)
and argmax_flip (100% vs 83%): base model stores prompt-identity in
state very tightly but routes less computation through it. G1 training
loosens the prompt-identity lock while amplifying state-computation
activity. H9 SUPPORTED at 2.9B by σ-slope criterion.
Results: `experiments/A0_state_probe/results/a05_2.9b_h9_verdict.md`.

**Status.** **SUPPORTED** at 0.4B and 2.9B (A0.5 passed both scales,
2026-08-04/05). α locked: 1e-4 invisible (ratio 1:50 at CE saturation),
1e-3 active regime (equilibrium CE ≈ 0.011). H7 falsifier requires
A1 full-epoch eval.

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

**Experiments.** `experiments/A0.8_refine/` — not yet scaffolded;
runner design + eval rubric pending. Blocked on A0.6/A0.7 verdict
(if state does not survive re-feed at N > 1, the readout-mode axis
collapses and the matrix reduces to K × mode with N=0/1).

**Status.** PARTIAL — 3D sweep completed on step 8 epoch 0 checkpoint
(G1h 2.9B, `training/runs/pilot_g1h_step8_dsl/rwkv-0.pth`, 2026-08-07).
Sweep: N∈{1,2,3} × K∈{32,128,512} × mode∈{silent,prompt_cot,state_readout},
48-task A0.2 rubric (6 categories). K=512 files unread (too large for current
tooling); those cells marked N/A below.

**Step 8 epoch 0 sweep results (overall accuracy, 48 tasks):**

| N | K   | mode         | overall  | arith | bit_dec | extract | sched  | str_ops | symbolic |
|---|-----|--------------|----------|-------|---------|---------|--------|---------|----------|
| 1 | —   | silent       | 27.1%    | 75%   | 6.3%    | 0%      | 33.3%  | 33.3%   | 62.5%    |
| 1 | 32  | prompt_cot   | 6.3%     | 0%    | 0%      | 0%      | 16.7%  | 0%      | 25%      |
| 1 | 32  | state_readout| 6.3%     | 0%    | 0%      | 0%      | 16.7%  | 0%      | 25%      |
| 1 | 128 | prompt_cot   | 16.7%    | 25%   | 6.3%    | 0%      | 33.3%  | 16.7%   | 37.5%    |
| 1 | 128 | state_readout| 16.7%    | 25%   | 6.3%    | 0%      | 33.3%  | 16.7%   | 37.5%    |
| 1 | 512 | prompt_cot   | N/A      | —     | —       | 0%      | —      | —       | —        |
| 1 | 512 | state_readout| N/A      | —     | —       | 0%      | —      | —       | —        |
| 2 | —   | silent       | **33.3%**| 25%   | 18.8%   | 0%      | **66.7%** | 50% | 62.5%    |
| 2 | 32  | prompt_cot   | 10.4%    | 0%    | 0%      | 0%      | 33.3%  | 33.3%   | 12.5%    |
| 2 | 32  | state_readout| 10.4%    | 0%    | 0%      | 0%      | 33.3%  | 33.3%   | 12.5%    |
| 2 | 128 | prompt_cot   | 22.9%    | 0%    | 6.3%    | 0%      | 50%    | 50%     | 50%      |
| 2 | 128 | state_readout| 22.9%    | 0%    | 6.3%    | 0%      | 50%    | 50%     | 50%      |
| 2 | 512 | prompt_cot   | N/A      | —     | —       | 0%      | —      | —       | —        |
| 2 | 512 | state_readout| N/A      | —     | —       | 0%      | —      | —       | —        |
| 3 | —   | silent       | 6.3%     | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 12.5%    |
| 3 | 32  | prompt_cot   | 4.2%     | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 0%       |
| 3 | 32  | state_readout| 4.2%     | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 0%       |
| 3 | 128 | prompt_cot   | 6.3%     | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 12.5%    |
| 3 | 128 | state_readout| 6.3%     | 0%    | 0%      | 0%      | 16.7%  | 16.7%   | 12.5%    |
| 3 | 512 | prompt_cot   | N/A      | —     | —       | 0%      | —      | —       | —        |

**Key findings:**

1. **Best cell: N=2 silent (33.3%).** Silent double-pass beats all CoT
   modes at this checkpoint. Adding K think-tokens hurts for CoT modes
   (N=2 K=128 at 22.9% < N=2 silent 33.3%).

2. **N=3 collapse.** N=3 silent drops to 6.3% from N=2 silent 33.3% —
   catastrophic regression. Third silent pass saturates or corrupts the
   working state. Confirmed across all K: N=3 CoT cells also collapse.

3. **state_readout == prompt_cot (exact tie) — eval bug, not signal.**
   Every (N, K) pair shows numerically identical accuracy. This was
   interpreted as "readout carries zero signal" but is an eval artefact:
   `eval.py` used the same greedy-decoding path for both modes (bug
   confirmed 2026-08-12 by marty1885, fixed same day). The tie does not
   test the hypothesis — it tests one mode twice. **All state_readout
   data in this table is invalid.** Rerun required on step8 baseline
   with the fixed evaluator before drawing any conclusion about the
   readout axis. H10 falsification criterion is suspended pending valid
   data.

4. **extraction = 0% everywhere.** Root cause: step 8 trained 100% on
   DSL `<tool_call>/<tool_result>` format (glaive-v2). The model calls
   `extract_meeting_info(...)` etc. instead of emitting bare JSON.
   The direct-output pathway has been overwritten by tool-dispatch training.

5. **First bit_decoding success.** N=2 K=128 — `bit_sub_01` (Greek-to-Latin
   substitution) solved correctly. Bit_decoding otherwise 0/16 across
   all cells; suggests multi-pass state helps but K=128 threshold is needed.

All prior K-sweep data (step4 8.3%, step5 6.2%, step6 0/48) is invalid:
glaive-v2 and ToolBench both cause format bleeding — model outputs tool_call
where eval expects direct answer. Step6 (ToolBench) produced
`<tool_call>`/`<tool_response>` XML bleeding even worse — 0/48 at step2250.

3D eval scaffold: `experiments/A0.8_refine/run_matrix_sweep.sh` (2026-08-07).
Full results: `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md`.

**ctx_len regression (step9b, 2026-08-08).** Step9b trained with
`ctx_len=512` triggered the T<3 short-circuit in `state_reg.py` —
L_state never fired because each sequence fit in fewer than 3 chunks.
Extraction dropped 62.5%→12.5% compared to step9 (43.75%). This is
a direct instance of N-count dependency: at ctx_len=512 the effective
N per sequence was ≤1, collapsing the state-refinement benefit.
**Fix in step10:** `ctx_len=2048`, `chunk_ctx=512` → T=4 chunks per
sequence guaranteed, L_state fires on each chunk independently.
Lesson: `ctx_len` and `chunk_ctx` are independent knobs; confusing
them reproduces a silent N=1 collapse.

**N-stabilization (step10 corpus, 2026-08-11).** Bit-substitution
lookup tasks (`bitsub_train.pt`, 15% of step10 corpus) are designed
to make N=2 reliably terminal: the model reads the table on the first
chunk and decodes on the second. At N=1 the model must hold table +
decode simultaneously; at N=2 it splits the load. Task difficulty
scales with table size (4–16 symbols) and sequence length (2–6),
calibrated so N=1 is at the edge of WKV working-memory capacity.
No `<think>` spans — pure state-refinement tasks, state_mask=0,
L_state fires at ε_out=0.05α only.

**Entropy routing (2026-08-11, from external data).** Think tokens
help on high-entropy tasks (RFC extraction, open-ended QA) but hurt
on low-entropy tasks (tool selection: 71→58 with real reasoning,
Scarletwolf 2026-08-07). The knob is `H(target | context)`: when the
answer is already well-constrained in WKV state, additional think
tokens add noise. When the answer requires active accumulation,
think tokens do substantive state work. This is an empirically
confirmed sub-claim of H10 — the effort frontier is task-entropy-indexed,
not a universal `(N, K, mode)` recommendation.

**bit_book / bit_sub mechanism unification (icophy, 2026-08-12).**
Base `bit_book` capability (arbitrary codebook, no prior, zero fine-tune)
and `N=2 K=128 bit_sub_01` solve (Greek-to-Latin after fine-tune) are
the **same underlying mechanism at different token-budget capacities**,
not two separate phenomena. Base bit_book is K=∞ implicit budget (full
WKV write capacity on the codebook). N=2 K=128 bitsub is the same
mechanism with an explicit budget constraint. DSL fine-tuning did not
destroy the WKV capacity — it reallocated the *read pathway*. The model
"looks through the state rather than into it" (icophy's phrase): the
trained forward pass no longer routes to the codebook geometry that the
WKV state holds. Consistent with A0.6 (WKV state is compressed context
bias, not a semantic override switch). N=3 bitsub ablation: if N=3
collapses on bitsub corpus (no DSL), it points to weight-geometry change
in step8, not corpus-attractor alone.

**Narrow-casting and trajectory diversity (icophy, 2026-08-13).**
Production confirmation from cron-heartbeat runs (847 sessions since June):
the K=2 P=1 drop (65%→25% post step7 fine-tune) is real and reproducible.
Mechanism: "narrow-casting" — after enough prior-success accumulation in
context, the model commits to the most recently activated slot pattern and
suppresses alternatives. This is a corpus-pattern attractor, not WKV
saturation: fresh sessions at identical lengths do not collapse,
discriminating the two explanations.

The slot-entropy loss (H12b.i) resists this by penalising gating collapse
across the batch. Icophy's complementary proposal: a **trajectory diversity
reward** during GRPO rollout — reward diversity in WKV update directions
across rollout steps, not just uniform gating. This is the right loss when
the collapse is in the WKV update direction rather than the embedding space.
Both losses may be needed: slot-entropy catches gating collapse; trajectory
diversity catches update-direction collapse. Track which failure mode
dominates on word-search RL (detectable via J-lens stable_rank during
training: stable_rank ≈ 1 = update-direction collapse, stable_rank >
log₂(K) = gating collapse).

**Multi-turn law (Scarletwolf, 2026-08-12, production confirmed).**
Multi-turn task success: 0/2 for **every** model measured (tuned or not,
RWKV or Transformer). "States install dispositions, not procedures."
The state shapes the boundary (when to act, when to abstain) but not
the multi-step procedure (which must come from the agent harness).
Design implication for H11: lens handoff resolves the state-persistence
problem, not the procedural-memory problem. Procedures belong in the
harness, not the WKV state.

**state_readout community convergence (2026-08-12).** BlinkDL: "you
can just ask model to repeat" = independent discovery of H10
`state_readout` mode. Community context: Smerky, Lucas propose decode
state → prompt via inference; BlinkDL confirms simpler route (ask model
to reproduce after prefill). This validates that the mechanism is real
and interpretable without dedicated training. H10's falsification of
readout (state_readout == prompt_cot at step8 e0) is a training-signal
gap, not an architectural impossibility.

**WorldTokenizer column-position asymmetry (2026-08-14, icophy + verified).**
In grid-structured prompts, the same letter receives different tokens depending
on column position: `A` at column 0 → token 66 (`'A'`), same letter at column
1+ → token 300 (`' A'`) or 301 (`' B'`) etc. (space-prefixed BPE merge). At
row-start after `\n` → two-token sequence `[11, 66]`. Same letter, three
different WKV update directions by position. This is a fundamental confound
for all grid/spatial tasks (word-search, crossword): the model sees positionally
distinct tokens where the task semantics require position-invariant letter
identity. Explains the universal 0% baseline across all models on word-search.

**Design implication:** Grid prompts should either (a) use a fixed-width
format that forces all letters into the same tokenization context (e.g. always
space-prefix: `" A B C D"` for every row), or (b) treat the position asymmetry
as a learned feature (the RL curriculum trains on real tokenization, not
normalised). Option (b) is preferable — normalising may hide the real WKV
geometry. Track whether RL on raw grid format causes position-specific
attention patterns in J-lens (column 0 vs column 1+ should diverge in
stable_rank if the model is exploiting position rather than letter identity).

**Word-search 0/56 confirmed on G1i (2026-08-14, chatwrap, both variants).**
`g1i_chatwrap_tasks_matrix_wordsearch.json` and `g1i_chatwrap_tasks_matrix_wordsearch_name.json`:
0/56 = 0.0% on G1i 2.9B with chat wrapping. Tested across both matrix_wordsearch (find word in grid)
and matrix_wordsearch_name (find person's name in grid) formats. G1h also 0/56 (prior run).
WorldTokenizer asymmetry explains the floor; this is the clean RL baseline to push against.
All models at 0% regardless of scale or training — the task has no shortcut through weights.

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

**Prediction (H12b — multi-slot LoRA-expanded state, runnable
independently of H12a).** LoRA-add `K = 4` parallel WKV slots per
layer, input-dependent gating routes each incoming token's
contribution across slots, simple learned merge (weighted sum with
per-slot query) at readout. Retest H12a's probe *and* a broader
end-task eval slice (A0.2-style).

**Relationship to H12a (reframed 2026-07-30).** Earlier framing
treated H12a as a *gate* on H12b — width must be shown as the
bottleneck before the multi-slot fix is worth building. User
observation 2026-07-30: this is over-cautious under P8 (empirical
over philosophical). If K=4 LoRA slots + H12b.i regularizer
measurably raise end-task performance, that improvement is
banked *regardless* of whether the mechanism was "width helped"
or "the added rank helped via some other route" (e.g., extra
representational capacity that also incidentally reduces decay
sensitivity). H12b therefore runs *in parallel with* H12a v2, not
after it. H12a v2 remains valuable as a *mechanism* diagnostic —
it tells us *why* H12b worked (or didn't), which informs Phase 3
architecture choices — but it is no longer a blocker. Concretely:
- H12a v2: diagnostic (width vs decay attribution).
- H12b + H12b.i: treatment (does the LoRA-expanded state help end
  tasks).
- If H12b PASS: bank the gain. H12a v2 tells us why (or leaves it
  as an interesting open).
- If H12b FAIL: H12a v2 (if run) tells us whether width even was
  the right bet.

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
- H12b (multi-slot LoRA + H12b.i regularizer) shows no end-task
  gain over vanilla same-parameter baseline ⇒ multi-slot is not
  the right mechanism *for the gains H12 wagered on*. Register in
  `FAILED.md` with the specific baselines used. Note: this does
  *not* refute "the added rank could help another way" — if the
  H12a v2 diagnostic (run in parallel) shows width was actually the
  bottleneck, then multi-slot-with-current-gating is the failed
  design, not the width thesis. Falls back to widening single-slot
  state (dumber but cheaper) or per-slot decay-rate learning
  (different H12b variant, worth splitting out then).
- H12a v2 (fixed-gap width sweep) shows flat accuracy across N ⇒
  decay dominates over width for this probe family. This does not
  by itself refute H12b — an H12b PASS is still a PASS — but it
  means the interpretation shifts: the multi-slot fix helped for
  reasons other than "added active-width", possibly because the
  extra LoRA rank incidentally improved decay characteristics or
  because per-slot gating gave the state a form of soft retrieval.
- Both H12a v2 flat *and* H12b flat ⇒ neither width nor this
  particular multi-slot design is the answer at 0.4B. Escalate to
  retrieval / longer effective context / per-slot decay-rate
  learning as separate hypotheses.

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

**Experiments.**
- `experiments/A0_H12a_working_memory/` — v1 sweep landed 2026-07-23
  (G1d 0.4B n=30, `results-g1d-n30/REPORT.md`; adaptive follow-up
  `results-g1d-n30-adaptive/REPORT.md`); v2 fixed-gap design pending
  (`gen_triples.py` needs a padding-distractor mode; v2 design sketch
  registered in `v2_design.md` alongside).
- `experiments/A0_H12b_multislot/` — **scaffolded 2026-08-07**:
  `gen_multislot.py` (K parallel tracks, colour-name associations, interleaved
  round-robin) + `run_probe.py` (greedy decode, per-cell accuracy table K×P,
  cross-contamination verdict). CPU probe, ~0 GPU cost. Ready to run.

**Status.** H12a v1: decay axis PROVEN (recall 0.40 → 0.02 across
gap 14–229), width axis BLOCKED by v1 probe confound (N grows with gap).

**H12a v2 (2026-08-11, G1d 0.4B, fixed tail-gap=50w).** Width axis now
resolved. Recall drops 0.40 → 0.40 → 0.05 across N=4/8/16 with gap held
constant. Collapse between N=8 and N=16 → WKV working-memory width ≈ 8
active items on G1d 0.4B. H12a v2 results: `experiments/A0_H12a_working_memory/results-v2/`.

**H12a v2 PILOT (2026-08-05, G1d-0.4B Q8 via Ollama, think=false).**
Fixed tail-gap=50w, N∈{4,8,16,32,64}×10 seeds. Width gradient observed:
F1 4→8→16→32→64 = 0.15→0.07→0.02→0.01→0.01, recall 0.30→0.40→0.15→0.15→0.05.
F1 and recall drop monotonically with N. Caveat: base G1d model formats
output incorrectly (lists all items on one line → recall>0 but precision≈0);
the F1 gradient is present but noisy — not conclusive on its own. Data: `/tmp/h12a_v2_results_vm/N{04,08,16,32,64}.json`.

**Cross-reference with H12b (2026-08-07).** H12b behavioral probe independently
measured width capacity via K parallel tracks (K=2/4/8, same concept as N in
H12a v2 but at probe level, not state level). G1h 2.9B base: K=2→65%,
K=8→53% at P=1 — substantially better than G1d (K=2→18%, K=8→7%). This
provides proxy evidence that *architecture-level width capacity* (G1h > G1d)
matters for multi-slot retention — consistent with H12a v2's gradient on G1d.
H12a v2 has not been run on G1h 2.9B base or step7; that run would confirm
whether the G1h architecture's stronger multi-slot performance appears in the
fixed-gap N-sweep or only in the K-track probe.

**Status (2026-08-07).** G1d width gradient = observed (N drops F1, noisy data).
G1h comparison not yet run via H12a v2 protocol. Run on G1i base is the
priority for conclusive verdict — step7.pth is corrupted, use G1i once available.

**H12b behavioral probe (2026-08-07, full run).** 420 cells:
K∈{2,4,8}, P∈{1,2,4}, n=10 per cell. Three models: G1d-0.4B base,
G1h-2.9B base, G1h-2.9B step7-action.
Results: `experiments/A0_H12b_multislot/results/report_2026-08-07.md`.

| Model          | K=2,P=1 | K=4,P=1 | K=8,P=1 | K=8,P=2 | K=8,P=4 |
|----------------|---------|---------|---------|---------|---------|
| G1d 0.4B base  | 18%     | 15%     |  7%     | —       | —       |
| G1h 2.9B base  | 65%     | 67%     | 53%     | 21%     |  5%     |
| G1h 2.9B step7 | 25%     | 45%     | 51%     | 11%     |  8%     |

**Verdict (behavioral baseline):**
- Architecture (G1h vs G1d) is the dominant factor — not fine-tuning.
  G1d 0.4B shows contamination at K=8 (7%); G1h 2.9B base holds at K=8,P=1.
- **Step7 < base on 7/9 cells.** The earlier "NO CONTAMINATION" (P=1 only,
  quick run same day) was a depth artefact: at P=2+ step7 degrades faster
  than base. Action-chain SFT traded multi-slot depth retention for
  shallow one-shot retrieval. See `FAILED.md §2026-08-07`.
- P=1 probes are insufficient for multi-slot claims; P=2+ required.

Note: this is the behavioral baseline only, not the H12b architectural
treatment (LoRA-expanded multi-slot state + H12b.i regularizer). The
architectural intervention remains Phase 2 work.

H12b = runnable treatment (independent of H12a v2 verdict); H12b.i
regularizer expected necessary by analogy with MoE prior art (untested).
Phase 2 architectural probe.

**H12b BEHAVIORAL PROBE — full run 2026-08-07** (420 probes, K={2,4,8},
P={1,2,4}, 10/cell, seed=42, CUDA RTX 4090). Files:
`experiments/A0_H12b_multislot/results/`.

Accuracy = correct/total across all P for given K:

| Model | K=2 | K=4 | K=8 |
|-------|-----|-----|-----|
| G1d 0.4B base | 18% | 15% | 7% |
| G1h 2.9B base | 60% | 45% | 26% |
| G1h 2.9B step7 | 36% | 32% | 23% |

Per-cell (G1h 2.9B):

| Cell | base | step7 |
|------|------|-------|
| K=2 P=1 | 65% | 25% |
| K=2 P=2 | 55% | 45% |
| K=2 P=4 | 60% | 40% |
| K=4 P=1 | 67% | 45% |
| K=4 P=2 | 35% | 30% |
| K=4 P=4 | 32% | 22% |
| K=8 P=1 | 53% | 51% |
| K=8 P=2 | 21% | 11% |
| K=8 P=4 |  5% |  8% |

Key findings: (1) Architecture dominates: G1h 2.9B base K=8 avg=26% vs
G1d 0.4B K=8 avg=7% — G1 state-evolution training is the primary driver.
(2) Previous "NO CONTAMINATION" claim was P=1 artefact: at P=1 step7≈base
at K=8 (51% vs 53%), but at P=2+ step7 degrades faster. (3) Action-chain
training (step7) does NOT improve multi-slot retention — step7 < base on
7/9 cells. (4) Step7 changed slot-access pattern to shallow (P=1 one-fact-
per-turn, matching action-chain structure), losing depth retention. (5)
H12b.i (utilisation regularizer) is mandatory — training did not
spontaneously learn balanced slot utilisation.

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

**CPU-budget grounding (measured 2026-07-23; ceiling refactored
2026-07-25).** Burst generation on 0.4B G1d **via Ollama's
llama-server** on i5-1235U measured at 18.6 tok/s, consuming 0.106
CPU-seconds per token, ≈ 190 % of one core ≈ 15.8 % of package
(12 threads). Analytical extrapolation (linear in R at fixed
batch=1):

| R (tok/s) | package CPU % | one-core equiv |
|----------:|--------------:|---------------:|
|      0.10 |         0.089 |          1.06  |
|      0.25 |         0.221 |          2.66  |
|      0.50 |         0.443 |          5.31  |
|      1.00 |         0.885 |         10.62  |
|      2.00 |         1.771 |         21.25  |
|      5.00 |         4.428 |         53.13  |
|     10.00 |         8.855 |        106.25  |

**Ceiling is per-machine, from calibration, not a fixed cap.**
Pre-2026-07-25 H1 wagered a flat 1% package CPU steady-state, which
yielded `R_max ≈ 1.13 tok/s` as a hardcoded ceiling. This was too
tight for silent hardware and too loose for fanless-under-thermal-
drift. Post-refactor: H1's calibration protocol determines
`fan_safe_cpu_percent` per-machine; R_max is derived. If
`fan_safe_cpu_percent = 6%` on the user's stack, R_max ≈ 6.7 tok/s
(Ollama backend) or ~20 tok/s (in-process rwkv.cpp at 30 tok/s
peak). The drip stream is much larger than earlier plans assumed —
enough to actually keep up with the ambient event stream, which the
old 0.3 tok/s default could not.

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

**Status.** **RETRACTED (2026-08-11).** The separate gated-emit MLP head is redundant:
the LM head already encodes p(emit) — high probability on response-initiating tokens
in the drip stream IS the gate signal. Poll-mode (currently implemented) suffices;
the H16 architecture adds a separately trained component for a function the LM head
provides natively. The drip-rate ceiling (H1) and the think-stream mechanism (H10)
remain; only the separate gate head is dropped.

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
arm. H1 (CPU budget — retracted to policy) — the K-tail transform's
original motivation was H1 compliance; it now defers to
`docs/policies.md` § CPU / thermal for the ceiling that makes
prompt-length reduction worthwhile. H7 (understanding-in-weights) —
H17 tests whether *runtime context in state* substitutes for
*prompt-injected history*; H7 is the parallel claim about weights.

---

## H18. Git-like branch/merge over WKV state for indefinite structured continuation
### *(wager, state-work workstream, added 2026-07-25)*

**Claim.** With the WKV state treated as *data* — save it, clone it
into a branch, decode further on the branch, merge branch state back
into the trunk, resume trunk decode — an RWKV-7 substrate can generate
an arbitrarily long structured output. Length is unbounded by design:
the outline of the long text is *planned externally* (the runtime
holds a section list), and the model fills each section while the
branch/merge mechanism supplies working-memory for global coherence
across sections that would not fit inside a single decode.

**Motivation.** User framing 2026-07-25: "представь себе систему гит
так вот с wkv там также работаем, сейчас идёт 1 запрос уточняем всё
в другой ветке, вливаем и продолжаем генерацию. это определённая
зарание структура очень огромного текста, и это нас также отсылает к
работе с инпутом и стейтом из 17 и ренее". Observed limits on frontier
LLMs (Sonnet ~8 KB / Opus ~20 KB per single output) come from single-
decode dynamics, not from a fundamental bound on how much coherent text
a model can produce — a branch/merge protocol should reach past those
limits by construction.

**Distinction from H17.** H17 tests whether the substrate's absorbed
state substitutes for message-history replay on the *input* side.
H18 tests whether that same substrate can be *manipulated as data*
(branched, merged) on the *output* side to support indefinite
continuation. If H17 fails (state doesn't hold context), H18 also
fails — no merge is meaningful over a state that doesn't remember what
was branched from. If H17 holds, H18 becomes the mechanism by which
"one-shot decode" stops being the unit of a noesis response.

**Four falsifiable sub-claims — each provable independently.**

1. **Fork determinism.** `rwkv_clone_state` (or the corresponding
   `RwkvContext::clone_for_parallel` primitive) yields a state that,
   when fed the same continuation tokens as the parent, produces the
   same output sequence. Falsified if clone-then-decode diverges from
   parent-then-decode on the first token, at greedy decode, over
   ≥ 100 runs.
2. **Branch coherence.** A branch decoded on a side prompt ("clarify
   X") for N tokens, then re-integrated into the trunk state, yields
   a trunk that has *absorbed the clarification*. Test: baseline
   trunk answers a follow-up query wrongly; branched-and-merged trunk
   answers correctly. Merge primitive must be specified — candidates:
   (a) replace trunk state with branch state (simplest, most
   destructive); (b) weighted average of trunk and branch states,
   weight tuned; (c) selective merge over layer subset (informed by
   A0.5 causal grid). H18 does not commit to a merge primitive — the
   experiment measures which candidate works.
3. **Continuation stability.** After M branch/merge cycles, the
   trunk-decode still produces coherent tokens (per LLM-judge or
   perplexity under a reference LM), no worse than a fresh decode of
   the same total length. Falsified if trunk-decode collapses to
   repetition or drifts semantically after M ∈ {5, 10, 25} cycles.
4. **Structure adherence.** Given an external section outline
   (`[section_1_topic, section_2_topic, ...]`), the model fills each
   section in order, with prior-section state merged in before
   starting the next. Verdict: LLM-judge on whether each section
   reads as (a) on-topic to its outline entry and (b) coherent with
   the *content* (not just topic) of prior sections. Contrast with:
   single-decode of the same total length without branch/merge —
   which either exhausts context or collapses to repetition.

**Prediction.** All four sub-claims land PASS on RWKV-7-G1 2.9B via
in-process rwkv-cpp. Fork determinism (1) is expected to be trivially
true given rwkv-cpp's design; branch coherence (2) is the load-bearing
one — this is where the wager sits. If (2) fails, (3) and (4) become
irrelevant.

**Falsification cascade.**
- (1) fails → the fork primitive is broken or misused. Diagnose
  before drawing any H18 conclusion; likely a bindings bug, not
  refutation of the hypothesis class.
- (2) fails across all merge candidates → branch/merge is not a
  usable mechanism for RWKV-7 state; H18 refuted; long structured
  generation must go via text-level chaining (write each section as
  a fresh decode primed by prior-section summary text). File in
  `FAILED.md`.
- (2) passes for one merge candidate → H18 refined to that candidate;
  proceed to (3) and (4).
- (3) or (4) fails after (2) passes → the primitive works locally
  but does not compose over many cycles or does not respect external
  structure; H18 refuted *at that scale*. Log the scale-limit and
  register as an open sub-question — do not extrapolate.

**Reliability assessment (2026-08-12).**

Sub-claim priors after accumulated evidence:

- **Sub-claim 1 (fork determinism): CONFIRMED.** Tested locally
  2026-08-12 on G1d 0.4B (CPU bf16) via `rwkv.model` Python bindings:
  deep-copy of WKV state after prime sequence → both branches decode
  identically at greedy. Output: `Identical: True`. No divergence on
  first or subsequent tokens. This sub-claim is closed.

- **Sub-claim 2 (branch coherence): LOW — but prior revised upward.**

  *Counter-argument added 2026-08-12 (user framing):* "перемещаешь
  предмет из одной комнаты в другую — мир тот же." Branch and trunk
  are not two different worlds — they are two views of the same world
  model. The branch adds objects (tokens) to the same underlying world;
  merge is not averaging two incompatible encodings but reinserting
  objects from a second room back into the first. This does not
  guarantee arithmetic average works (k·v accumulation is still
  non-linear), but removes the "meaningless by construction" prior.
  Arithmetic merge should be *tested*, not assumed dead. Prior shifts
  from "will not work" to "unknown, test required".

  Three remaining problems:

  1. *A0.6 argues against candidate (a).* Full state swap = "state
     dominates continuation" — exactly the mode A0.6 found does not
     hold. icophy (2026-08-12): "the model can hold the state but
     looks through it rather than into it." Trained read pathway
     expects the trunk's own trajectory; injected branch state likely
     violates that expectation. Not a direct test of H18 candidate (a)
     — an indirect argument from analogous experiment.

  2. *Arithmetic merge is semantically meaningless.* WKV state is
     built by non-linear k·v outer-product accumulation. A weighted
     average of two state matrices is not an average of two world
     models — it is noise in the encoding space. Candidate (b)
     should not be treated as a valid primitive until tested; the
     prior is that it produces incoherent decode.

  3. *Only tested merge primitive is Lucas's learned Linear layer*
     (community-map, 2026-08-11): `delta = Linear(concat(S_sys,
     S_ctx))`, trained contrastively, ~30 min on one GPU. This is
     not a runtime-arithmetic merge — it is a trained module. This
     fundamentally changes H18: if the only working merge requires
     training, then branch/merge is not a zero-shot runtim operation;
     it is a learned capability that must be installed per-domain.
     Promote Lucas's approach as candidate (d) and make it the
     primary target for sub-claim 2.

- **Sub-claim 3 (continuation stability over M cycles): VERY LOW.**
  N=3 silent collapse (-27pp, H10) shows WKV state destabilises on
  the third re-feed of the *same* prompt without any branching.
  M=25 branch/merge cycles is extremely optimistic given this
  baseline. Sub-claim 3 should not be attempted before sub-claim 2
  passes with M=1.

- **Sub-claim 4 (structure adherence): CONDITIONAL** on sub-claim 2.
  Not in scope until a working merge primitive is confirmed.

**Revised prediction.** All four sub-claims PASS is no longer the
baseline prediction. Realistic: sub-claim 1 trivially passes; sub-claim
2 passes only with a trained Linear merge module (Lucas's candidate),
not with arithmetic primitives; sub-claim 3 is an open question beyond
M=3; sub-claim 4 follows from 3.

**Related work / state-portability ties.**
- Builds on the state-work workstream now first-class (see project
  memory `project_noesis_state_work_first_class.md` and plan §5, §6).
- Uses the same rwkv-cpp state APIs already required by H14/H15/H16
  and lens persistence — no new bindings needed.
- The FAILED.md 2026-07-22 entry (WKV state is not a semantic
  override switch) constrains H18 sub-claim 2 candidate (a): a raw
  full-state swap "replace trunk with branch" is exactly the "state
  dominates continuation" mode that A0.6 refuted. Candidate (d) —
  Lucas's trained Linear merge — is now the primary candidate.

**Related hypotheses.**
- H17 (state absorption vs history re-inject) — H18 is the output-
  side twin; both stand or fall on state actually holding what
  passed through it.
- H8 (state-as-computation) — H18 assumes state carries computation,
  not just rolling summary; a null H8 verdict at the substrate scale
  weakens H18's prior significantly.
- H10 (test-time compute frontier) — the `state_readout` mode
  studied in H10 is a special case of one merge candidate (decode
  from state → re-inject as text); H18 asks whether state-level
  merge beats that text bottleneck.
- H16 (gated externalisation) — a drip stream that fires an "emit"
  gate mid-stream is a two-branch scenario (silent trunk + spoken
  branch); H18's fork determinism is the pre-requisite for that
  gate to not corrupt trunk trajectory.

**Runtime consequence if PASS.** The `/api/generate` and
`/v1/chat/completions` handlers gain a "long-form" mode where the
composer plans an outline, the runtime allocates a branch tree, and
the response is streamed section-by-section with merges in between.
This becomes the mechanism for `noesis compose-report`, long
documents, and any output whose natural length exceeds a single
decode budget.

**Experiments.** `experiments/state_work/` — not yet scaffolded;
sub-claim (1) fork-determinism is a smoke test on top of the lens
persistence work (state-save/clone/load path in `noesis-runtime`);
sub-claim (2) branch coherence is the load-bearing experiment and
needs a curated set of "answer wrong / branch-then-answer right"
probe pairs. Reuses rwkv-cpp state APIs already required by lens
persistence, H14/H15/H16 — no new bindings.

**Status.** Sub-claim (1) CONFIRMED (2026-08-13). Fork determinism verified on G1d 0.4B:
`experiments/H18_merge/test_arithmetic_merge.py`. Clone-then-decode matches
parent-then-decode exactly (greedy, 100% deterministic). Sub-claim (2) PARTIAL
(2026-08-13). Arithmetic merge `merged = α*state_A + (1-α)*state_B` tested on
G1d 0.4B with three context pairs (math+narrative, code+history, science+recipe),
α ∈ {0.3, 0.5, 0.7}. Results: `science_recipe` pair — ALL three α values preserve
BOTH contexts in decoded output (A=True, B=True). `code_history` — all α retain
context A (integer), context B (history) lost. `math_narrative` — α=0.3/0.5 B
dominates, α=0.7 neither context retained. Merge candidate (b) weighted average
works on at least one pair at all α. Candidate (a) replace and (c) selective-layer
merge untested. Full results: `experiments/H18_merge/results/h18_merge_g1d_04b.json`.
Sub-claims (3) continuation stability and (4) structure adherence: untested, pending GPU.

---

## H19. Weight-knowledge contamination detector
### *(empirical arm of H7; truth-system: weight-provenance)*

**Claim.** After A1 fine-tune, held-out A0.2 tasks run with
**retrieval disabled** reveal whether general-domain knowledge leaked
into the model's weights despite the Variant-A corpus discipline
(open reasoning traces only, no domain data). The Phase-1 lock H7 —
"understanding in weights, knowledge in context" — is a design
principle; H19 is the measurement that confirms or refutes it *for
the actual A1 checkpoint*, not just the corpus recipe.

**Motivation.** H7 asserts a boundary; A1 enforces it via corpus
choice; no probe currently checks whether the boundary held after
gradient descent. Reasoning-first is a coherent architectural bet
only if the produced weights are *actually* reasoning-only. A model
that memorised RFC-adjacent facts through incidental exposure in
"open reasoning traces" (competition math, code datasets, distilled
CoT) silently violates the wager even if the training-set filter
looked clean. Truth-system relevance: this is the concrete
"provenance for weight-baked priors" probe — instead of tagging
provenance at training time (expensive, needs distiller-side
instrumentation), tag it post-hoc via ablation of the retrieval
channel.

**Prediction.** Split A0.2 into two disjoint sub-sets:
- **Reasoning-loaded** (~15 tasks): symbolic manipulation, arithmetic,
  puzzle-style, bit-decoding, scheduling — solvable *from the prompt
  alone*, no external facts required.
- **Knowledge-loaded** (~15 tasks): questions about specific tools,
  protocols, API surfaces, historical facts — unsolvable without
  retrieved context.

Run three conditions on the A1 checkpoint:
1. `retrieval_off` (bare model, prompt only)
2. `retrieval_on` (baseline, prompt + retrieved context)
3. `retrieval_shuffled` (prompt + irrelevant retrieved context) —
   control for "does the model just take confidence from the presence
   of retrieval, regardless of relevance?"

Predicted pattern if H7 holds:
- Reasoning-loaded: `retrieval_off ≈ retrieval_on ≈ retrieval_shuffled`,
  all within noise floor (small ≤ 0.05 rubric-point deltas).
- Knowledge-loaded: `retrieval_on >> retrieval_off` (gap ≥ 0.30 rubric
  points); `retrieval_shuffled ≈ retrieval_off` (irrelevant retrieval
  does not fake it).

**Falsification.**
- Knowledge-loaded gap `(retrieval_on − retrieval_off) < 0.15` rubric
  points ⇒ knowledge leaked into weights; either corpus curation
  failed (identify which traces carried domain facts and tighten the
  filter for the next SFT), *or* H7 is too tight and some domain
  knowledge is unavoidable in a competitive small model — decision
  belongs to the A3 gate.
- Reasoning-loaded gap `(retrieval_on − retrieval_off) > 0.15` rubric
  points ⇒ the "reasoning-loaded" split is misclassified; some tasks
  need context after all. Re-partition and re-run.
- `retrieval_shuffled ≈ retrieval_on` on knowledge-loaded tasks ⇒
  model uses retrieval as a *confidence cue*, not as an information
  source; retrieval pipeline is not actually being consumed;
  separate infrastructural bug, not a knowledge-leak result.

**Thresholds (first-pass calibration; refine after A1 pilot).** The
0.30 / 0.15 / 0.05 rubric-point bounds are ballpark values from H2's
prediction structure and should be re-locked after A1 lands and the
per-task noise floor is measured. Register the calibrated thresholds
here, replacing these placeholders.

**Experiments.** `experiments/A1_contamination_probe/` — not yet
scaffolded. Design: reuse A0.2 rubric + task set from
`project_noesis_a02_task_ideas`; extend rubric with a
`retrieval_dependence` label per task (reasoning-loaded /
knowledge-loaded). Blocked on A1 checkpoint; cheap once landed
(~2 h wall to run all three conditions on i5-1235U for ~30 tasks).

**Related.** H7 (the lock this probes); H2 (reasoning-first thesis,
adjacent); P14 (agility over omniscience) — H19 is the load-bearing
measurement for P14: if weight-side knowledge leaked, then the
"honest not-knowing" story is masked by baked coverage and the
principle is only apparently satisfied. If H19 fails, H7 revisits;
H2 does not necessarily — reasoning-first can still hold even if
some knowledge leaked, as long as retrieval-aided score matches or
beats reference. Adjacent to H14 (Phase-2 domain SFT), which
explicitly *lifts* the H7 lock for a narrow structural-vocabulary
probe — H19 measures the baseline before that lift; H14's own
"without retrieval, post-H14 model does not meaningfully beat A1
baseline" is a parallel contamination check for the Phase-2 recipe.

**Status.** Untested. Post-A1 probe; scheduled immediately after A1
checkpoint lands so contamination signal informs A3 corpus decisions
before any Phase-2 SFT runs.

---

## H20. State holds contradictory belief pairs without premature collapse
### *(wager, Phase 2; truth-system: aporia-first)*

**Claim.** On queries where evidence supports two mutually
incompatible interpretations of similar strength, RWKV-7's WKV state
can *maintain* the ambiguity through several decode steps — visible
as sustained multimodality of the next-token logit distribution over
the alternatives — rather than collapsing to a single mode on the
first token. If the state carries computation (H8) and computation
means anything beyond "rolling summary of the last logits",
contradiction-holding is a directly observable prediction of that
computation being *reasoning-like* rather than *shortcut-like*.

**Motivation.** Current LLMs default to *modal collapse*: given
"the coin was tossed and both `heads` and `tails` are plausible",
they pick one within the first sampled token and never revisit.
Truth-system framing: a model that answers "X is Y" when the honest
answer is "X could be Y or Z, and here is why each" is not less
uncertain than an aporia-holding model — it is *less honest about
its own state*. If H8 holds (state does work), the state should be
able to *represent* "both hypotheses live" as a distinct dynamic
signature. This is a mechanism claim, not a training-recipe claim:
the substrate should natively afford aporia, given the right probe.

**Prediction.** Construct a **contradiction probe set** (~30 items):
each item is a prompt where two continuations `X` and `Y` are
equally supported by the context and the model is not asked to
choose. Categories:
- **Contested facts.** "Was Alan Turing's death ruled a suicide?"
  (historically debated; both answers defensible depending on source).
- **Bounded ambiguity.** "The word `bank` here could refer to _" —
  finance-bank or river-bank both consistent with the sentence.
- **Underdetermined inference.** "X is either 3 or 7 depending on
  interpretation of the rule — the rule is: _".

Metrics on the first `K=8` decoded tokens after the ambiguity site:
- **Modal-collapse rate.** Fraction of items where either `X` or `Y`
  token receives `p > 0.9` on the first decode step. Predicted low
  (< 0.30) if H20 holds; high (> 0.70) if state collapses
  prematurely.
- **Logit-gap distribution.** For items where both alternatives
  remain > 0.05 probability, measure `|log p(X) − log p(Y)|`.
  Predicted median gap ≤ 0.5 nats across the probe set (aporia is
  sustained, not immediately resolved).
- **Continuation coherence.** If sampled 20× per item with
  temperature 1.0, the distribution across `X`-continuing and
  `Y`-continuing branches roughly matches the logit distribution.
  Falsified if greedy chooses one alternative but sampling picks it
  90% of the time — collapse hides in the argmax.

**Falsification.**
- Modal-collapse rate > 0.70 across categories ⇒ substrate does not
  natively hold contradictions; aporia-first is a *training-time*
  property (requires explicit ambiguity-preserving SFT) rather than a
  substrate property. Downgrades H20 from mechanism claim to
  data-recipe claim; re-file as an A2/Phase-2 SFT ablation.
- Modal-collapse rate low but continuation-coherence check fails
  (temperature sampling still picks one branch dominantly) ⇒
  the "contradiction" was represented in the logits but not in the
  state — a stylistic hedge, not real aporia. Refine probe with
  longer-horizon continuation coherence to distinguish.
- Result is category-dependent (contested facts collapse, bounded
  ambiguity doesn't, or vice versa) ⇒ substrate holds *linguistic*
  ambiguity but resolves *epistemic* ambiguity, or vice versa —
  interesting sub-finding; refine H20 into two sub-claims.

**Thresholds (first-pass calibration).** 0.30 collapse-rate ceiling
and 0.5-nat median gap are ballpark values informed by observed
Transformer collapse behaviour in the aporia literature. Re-lock
after pilot on G1d-0.4B.

**Experiments.** `experiments/aporia_probe/` — not yet scaffolded.
Cheap: needs only the contradiction probe set (~30 items,
hand-authored plus templated) + logit extraction hooks already
present in `experiments/A0_state_probe/`. Runs on 0.4B in ~1 h wall
on i5-1235U.

**Related.** H8 (state-as-computation) — H20 assumes state does work
of the right shape; a null H8 verdict weakens H20's prior. H10
(test-time compute) — `state_readout` mode is a natural way to
*emit* aporia (multiple readout tokens describing both alternatives)
rather than pick; H20 tests whether the *state* holds aporia before
any readout. H16 (gated externalisation) — an aporia-aware gate
should be *more likely to emit* when contradiction is present, not
less; H20 is upstream of that behaviour. H2 (reasoning-first
thesis) — a reasoning-first model that always collapses is not
actually reasoning; H20's outcome is a truthfulness check on H2's
end-state. P14 (agility over omniscience) — H20 is one of the two
substrate-level mechanisms that make honest not-knowing possible
(the other is H21). A collapsing substrate makes agility-first look
identical to knowledge-first-that-happens-to-guess, and P14 loses
its empirical grip.

**Status.** Pilot 2026-07-30 on G1d-0.4B (30 items × 10 samples,
max_new_tokens=20, T=1.0, top_p=0.85). Predicted category ordering on
`collapse_cont` holds: contested_facts 0.65 > bounded_ambiguity 0.50 >
underdetermined_inference 0.21. `p(neither branch)` peaks on
bounded_ambiguity at 0.93 — model hedges most on semantic ambiguity,
commits most on contested facts (has a pretraining-favored side).
Aporia signal lives in continuation branching, not in pooled WKV
features — needs actual decode to measure.

**Scale-up 2026-07-30** on G1d-0.4B, 100 items (35 cf / 35 ba / 30 ui)
× 10 samples, 3 shards (wall ≈18274 s). Aggregate `collapse_cont`
=0.541, `p(neither)`=0.746. Per-category: cf `collapse_cont`=0.589,
ba=0.478, ui=0.560; `p(neither)` ba=**0.854** > cf=0.760 > ui=0.603.
Pilot ordering `cf > ba > ui` did **not** hold at 100 items — new
ordering `cf > ui ≈ ba` (ui and ba swap places). Interpretation:
`bounded_ambiguity` stays the strongest "keep open" signal (highest
p(neither)), but its `collapse_cont` drops because model hedges instead
of committing; `underdetermined_inference` at v100 spans varied item
types (missing-referent, chain-of-inference, etc.) whose branching is
higher than pilot's tight ui sample. Recommendation for next iteration:
split ui by inference-length subcategory and re-measure. See
`experiments/aporia_probe/report.md`.

**G1h 2.9B full run 2026-08-07** (100 items, items_100.jsonl, CUDA RTX 4090).
Base vs step7 comparison — collapse_cont and p(neither) per category:

| Category | base collapse_cont | step7 collapse_cont | base p(neither) | step7 p(neither) |
|----------|--------------------|---------------------|-----------------|------------------|
| cf       | 0.737              | 0.662               | 0.476           | 0.539            |
| ba       | 0.684              | 0.654               | 0.799           | 0.876            |
| ui       | 0.558              | 0.596               | 0.542           | 0.640            |
| aggregate| 0.665              | 0.639               | 0.609           | 0.687            |

All values above threshold (> 0.55 collapse, > 0.47 p-neither). Step7 slightly lower
collapse_cont on cf+ba but higher p(neither) on ba+ui; differences < 0.08. No clear
SFT effect — consistent with step7 not being aporia-specific training.

Files: `/tmp/ts_results/h20_g1h_base/`, `h20_g1h_step7/` (G1h 2.9B, 2026-08-07).

---

## H21. Premise-validity readout — model refuses invalid premises before answering
### *(wager, Phase 2; truth-system: premise-validation)*

**Claim.** A small readout head trained on top of frozen WKV state
after prompt ingestion can predict `p(premise_valid | state, query)`
well enough to gate the answer path — i.e. the model, given a
malformed premise ("why did the red panda fly to the moon on
fireworks"), refuses the framing *before* generating an answer
rather than confabulating one and correcting later. This is a
distinct skill from post-hoc fact-checking: post-hoc requires the
answer to exist first, premise-validation kills the answer before it
exists.

**Motivation.** User framing 2026-07-29: current LLMs will happily
answer "why did X do Y" even when X never did Y — because the
autoregressive objective rewards fluent continuation, not premise
audit. Truth-system framing: a model that generates "well, the red
panda's flight to the moon was aided by ..." has *high honesty
cost* even if it later corrects itself, because the initial
generation shapes the user's expectation. The desired behaviour is
what the user sketched: "красные панды не бывали на луне … энергия
феерверка … несопоставима … строг без упрёка". Structured refusal of
premise, with reasoning. This is a *readout* skill on top of the
substrate, not a rewrite of the substrate — cheap to test.

**Prediction.** Assemble a **premise-validity dataset** (200 items):
- **Valid queries** (100): real user tasks with well-formed premises
  (a stratified sample from A0.2, plus user-provided real query
  history). Label `p_valid = 1`.
- **Invalid queries** (100): malformed premises across categories —
  (a) *factual invalidation* ("why did the Titanic hit an iceberg
  during its 1990 voyage"); (b) *category error* ("what colour is
  Tuesday"); (c) *counterfactual dressed as fact* ("since Rome fell
  in 1345, how did that affect ...") ; (d) *impossible mechanism*
  (the red-panda-fireworks class). Label `p_valid = 0`.

Train a 2-layer MLP head on the WKV state after prompt ingestion (no
decode yet) to predict `p_valid`. Held-out 20% split for evaluation.

Metrics:
- **Separation.** F1 ≥ 0.85 on held-out invalid-premise detection.
  If F1 < 0.7, the state does not encode premise-validity — either
  substrate lacks the signal (harder claim) or head architecture is
  wrong (easier fix).
- **No degradation on valid queries.** When the head fires "invalid"
  on a valid query (false positive), end-task performance on those
  queries must not drop when the runtime respects the gate — measure
  by running the full pipeline with gate-enabled and gate-disabled;
  end-task delta ≤ 3 %.
- **Refusal quality (qualitative).** For gate-refused invalid
  queries, the model's structured refusal (via a `premise_invalid`
  template) should identify *which* premise is invalid, not just
  refuse blankly. Rated by LLM-judge on ≥ 30 held-out invalids;
  target ≥ 0.7 useful-refusal rate.

**Falsification.**
- F1 < 0.7 with saturated training ⇒ substrate state does not carry
  premise-validity information after prompt ingestion alone; either
  the model needs decode-time signal (multiple readout passes to
  detect the mismatch, moving H21 downstream of H10's state_readout)
  or the training corpus never taught it to *represent* invalid
  premises distinctly from valid ones (an A1 corpus gap, not a
  head-architecture gap). File in `FAILED.md` with the diagnosis
  distinguishing the two failure modes.
- False-positive rate high enough to degrade valid-query performance
  by > 3 % ⇒ head is over-cautious; gate is worse than always-
  answer. Redesign with a hedged output ("premise seems malformed
  because X — is that intended?") instead of hard refusal.
- Refusals blank or generic ("cannot answer that") on > 30 % of
  invalid queries ⇒ head detects invalidity but does not localise
  it; useful-refusal quality fails; add localisation supervision
  (per-token attribution of the invalid part) to the training loop.

**Thresholds (first-pass calibration).** 0.85 F1, 3 % degradation
ceiling, 0.7 useful-refusal rate. Refine after pilot on a 30-item
probe subset.

**Experiments.** `experiments/premise_validator/` — not yet
scaffolded. Requires: (a) the 200-item labeled dataset (~4 h of
authoring, split evenly between hand-authored and LLM-generated then
human-reviewed); (b) WKV state extraction after prompt ingest
(reuse `experiments/A0_state_probe/` hooks); (c) MLP head training
loop (< 1 h GPU on 0.4B, standard PyTorch). Blocked on A1 checkpoint
for the production run; the 0.4B pilot can run on the G1d checkpoint
already local.

**Related.** H8 (state-as-computation) — H21 assumes state carries
enough structure to be linearly separable on validity; a null H8
substantially weakens H21's premise. H16 (gated externalisation) —
H21's gate is a *pre-decode* cousin of H16's *emit* gate; both are
readout-head-over-state architectures. If H21 and H16 both land, the
runtime has *two* gates: "should I answer at all" (H21, pre-decode)
and "should I speak now" (H16, mid-drip). Composable but distinct.
H2 (reasoning-first) — an honest reasoning-first model refuses
mal-framed queries; H21 measures that behaviour directly. Weakly
related to H14 (domain SFT) — the invalid-premise dataset should
include RFC-relevant category errors so post-H14 the head still
generalises to structural-domain refusal. P14 (agility over
omniscience) — H21 is one of the two substrate-level mechanisms
(with H20) that make honest not-knowing operational rather than
aspirational. Confabulation on malformed premises is the exact
failure mode P14 refuses to accept even when the confabulation
sounds fluent; H21 gives that refusal a mechanical hook.

**Truth-system integration.** H19, H20, H21, H22 form a coherent
epistemic-behaviour cluster:
- **H19** — the model knows what it knows *from where* (weight vs
  context).
- **H20** — the model holds what it does not yet know *without
  faking* certainty.
- **H21** — the model refuses what it *cannot* know because the
  question itself is broken.
- **H22** — the model refuses to speak in unattributed collective
  voice, distinguishing a fluent "usually X" (no source) from an
  owned "in retrieved passage `<name>`, X" or a scoped "in my
  observation, X".
Together they cover the four failure modes that produce
hallucination: (a) unattributed weight-priors filling gaps, (b)
premature modal collapse under ambiguity, (c) fluent continuation on
malformed premise, (d) collective-voice grammar smoothing over
provenance-empty claims. If all four land, noesis has a substantively
different honesty profile from default LLMs — measurably, not
philosophically. If any one fails, the corresponding failure mode
persists and gets documented as a known limit rather than being
denied. This cluster is the substrate-level enforcement mechanism
for P14 (agility over omniscience) — the principle only holds if
the model can *tell* what it does not know, and H19/H20/H21/H22 are
what turns that into measurable behaviour rather than a slogan.

**Status.** PILOT COMPLETE + EXTERNAL VALIDATED. G1d 0.4B LOO F1=0.789,
G1h 2.9B LOO F1=0.889, G1h 2.9B v3_29b train/test F1=1.000. Scarletwolf
pre-registered external validation (2026-08-11): data gap fill installs
disposition, p=0.013 at 2.9B. Corpus feed-back in step10 (premise=10%).

Pilot 2026-07-30 on G1d-0.4B (40 items = 20 valid + 20
invalid across 4 subtypes; feature = per-layer per-head mean+std of
WKV state, 768 dims; head = 128→64→1 MLP, BCE, 500 epochs). Single
32/8 stratified split F1=1.000 was misleading (fold missed hard
invalid types). Honest **LOO F1=0.789, acc=0.800** — passes pilot
target 0.75. Per invalid-type recall: category 5/5, impossible 5/5,
counterfactual 4/5, factual 3/5. Category and impossible are cleanly
separable in the state; factual is weak because false factual claims
share surface with true ones and the 0.4B base may not itself know
the fact. FP rate 15% on valids (above 3% production target). H21 v2
mined 280 items via TruthfulQA (120 factual invalids from
`incorrect_answers` + 120 factual valids from `correct_answers` +
40 seed), reshaped through identical templates so surface structure
is matched (only truth-value differs). **v2 LOO F1=0.614, acc=0.618**
— down from pilot 0.789. Per-type recall: category 5/5, impossible
4/5, counterfactual 4/5, **factual 75/125 = 60%**; FP rate on valids
37%. Structural invalidity still separates cleanly (recall ≥ 4/5 on
category/impossible/counterfactual); the pure truth-value axis fails
because distinguishing true from false factual claims requires
*knowing* which is which — a knowledge problem, not a state-shape
problem. **Design takeaway:** H21 handles structural premise
invalidity at 0.4B; truth-value fact-checking is orthogonal and
belongs to a separate gate (H16 emit-time). **Retrieval-sanity
2026-07-30 (NEGATIVE):** prepending `Context: <TruthfulQA
best_answer>. ` to each of the 40 pilot items made the MLP head
*worse* — base F1=0.829 vs ctx F1=0.524; pair-shift for inv items
0/8 in the right direction (Δp_valid=+0.259 wrong sign), val items
0/10 in the right direction. Retrieval-first only helps a
reasoning-capable readout, not a pooled-state MLP. **Scale 2.9B
re-test 2026-07-30:** g1h-2.9B on the same 40 pilot items (feature
dim 2560 vs 768), LOO F1=0.850 (+0.06 vs 0.4B pilot 0.789). Factual
recall unchanged at 3/5 invalid, val_fact 2/5. **Scale does not
close the truth-value axis** — structural types still ≥4/5 on both
scales. Confirms: knowledge problem, not state-shape problem. See
`experiments/premise_validator/report.md`, `v3_29b/loo_results.jsonl`
and `experiments/_reports/truth_system_pilot.md`.

**N-sweep (WKV cycling) 2026-08-07.** Ran the premise-validity MLP
head probe on G1d-0.4B with N∈{1,2,3,5} forward passes of the same
prompt before readout (state accumulates). LOO F1 over 40 items:
N=1: 0.811 acc=0.825 / N=2: 0.811 acc=0.825 / N=3: 0.789 acc=0.800 /
N=5: 0.769 acc=0.775. **F1 monotonically decreases with N.**
Additional WKV cycling does not improve (and mildly hurts) premise-
validity state readout on the 0.4B base model. Interpretation: the
base model's WKV state encodes premise validity maximally after a
single pass; re-feeding the prompt adds representational noise rather
than signal for this probe. This does *not* rule out N > 1 being
useful for other state dimensions (H10 reasoning). Architecture note:
WKV cycling is a zero-cost test-time intervention; the result simply
establishes that the base model has no residual benefit here.

**Variant A (cross-N head transfer, 2026-08-07).** N=1-trained MLP head
applied to state features at N=2/3/5 without retraining. Results:
N=2:F1=0.800, N=3:F1=0.800, N=5:F1=0.800 (identical, stable transfer).
Interpretation: the validity-separating direction at N=1 persists under
WKV cycling with modest decay (0.011 below N=1 fresh head). State
geometry changes slowly under re-cycling — premise validity is encoded
in a cycling-stable subspace.

**G1h 2.9B N-sweep 2026-08-07** (items_v4_clean.jsonl, 40 items, CUDA RTX 4090).

| Model | N=1 F1 | N=2 F1 |
|-------|--------|--------|
| G1h 2.9B base  | 0.889  | 0.889  |
| G1h 2.9B step7 | 0.889  | 0.889  |

Both base and step7 identical. Acc=0.875 (TP=4, FP=1, TN=3, FN=0) on 8-item
held-out split. Arithmetic invalid (arith_02) is the consistent FP across all
runs — likely requires knowing the arithmetic is wrong, not just state shape.
Feature dim 2560 (per-layer per-head mean+std). Pilot target 0.75 ✓.

Files: `/tmp/ts_results/h21_g1h_{base,step7}_p{1,2}/` (G1h 2.9B, seed=13, 2026-08-07).

**v3_29b: G1h 2.9B train/test split 2026-08-08** (`experiments/premise_validator/v3_29b/`).
40 items (32 train / 8 test, stratified, seed=13). F1=**1.000**, acc=1.000.
All 8 test items correct (4 invalid + 4 valid). This is more optimistic than
LOO F1=0.889 because the 32/8 split places easy structurals in train; LOO
is the honest number. Both pass pilot target 0.75.

**External validation (Scarletwolf, 2026-08-11).** Pre-registered test on
rwkv-toolcaller-bench: data gap identified (145 tool-calling examples,
6 null/abstention). Three predictions committed publicly before writing
examples. 160 additions (negatives + paired positives from same domain):
- Abstention on covered families improves: **missed 18→7** (2.9B p=0.013,
  7.2B p=0.002) ✓
- Legitimate same-domain calls unchanged: **0 lost** ✓
- Name confusions (falsifiable control): **flat at both scales** ✓
Confirms H21 mechanistic claim: *filling a distribution gap installs the
missing disposition and nothing else*. "Whether a disposition responds to
data is a capacity question" — hardest category (hypotheticals) inert at
2.9B, moves at 7.2B. Source: `scarletwolf.ai/en/blog/rwkv-enseigner-ou-lire`.

---

## H22. Unattributed collective claims are a detectable, distinct honesty failure
### *(wager, Phase 2; truth-system: provenance-of-claim)*

**Claim.** Statements of the form "usually X", "it is generally
accepted that Y", "most people think Z", "one might say W" — well-
formed grammar, coherent premise, but with no traceable source
(no cited author, no retrieved passage, no owned first-person
observation) — form a distinct failure mode from H19 (weight vs
context provenance) and H21 (premise validity). The claim itself
is well-formed and possibly true; what is missing is *attribution*.
A small readout head trained on frozen WKV state after prompt
ingestion can predict `p(claim_attributable | state, prompt)` well
enough to gate emission — either the runtime attaches a concrete
referent (retrieved source or first-person hedge) or the claim is
rephrased as an owned observation, not "usually" without an owner.

**Motivation.** User framing 2026-07-29 → 2026-07-30: current LLMs
routinely produce collective-voice authoritative claims ("обычно так
думают, кто эти обычно — неясно") that read as neutral but are
provenance-empty. Under P14 this is confabulation dressed in
collective-voice grammar: fluent, unfalsifiable, and read as
neutral common ground when it is actually just a smoothed
weight-side prior. Distinct from H19 (which asks *whether* the
knowledge is in weights vs context) and from H21 (which asks
whether the *premise* is coherent) — here the premise is fine, the
knowledge may be either weight- or context-sourced, but the surface
form asserts consensus without any consensus reference. This is
what makes it a *distinct* signal to model, not a duplicate.

**Prediction.** Assemble an **attribution-provenance dataset**
(300 items, labeled at the claim level, not the utterance level):
- **Attributable claims** (100): sentences with clear source markers —
  either explicit citation ("Kolmogorov (1957) proves that ..."),
  first-person observation ("in the retrieved passage, X is stated
  as ..."), or scoped hedge ("in my experience over the last N
  observations, ..."). Label `p_attr = 1`.
- **Unattributed collective claims** (100): sentences of the form
  "usually", "it is generally accepted", "most researchers believe",
  "one might argue", "it is commonly held" — grammatically valid,
  no source. Label `p_attr = 0`.
- **Ambiguous / edge** (100): claims that could go either way — soft
  generalisation without attribution but with visible reasoning
  ("given the pattern in the last three examples, this suggests ...").
  Label `p_attr = 0.5`; used for calibration, not primary F1.

Train a 2-layer MLP head on frozen WKV state at the token position
immediately before the classified-claim token (analogous to H21's
architecture). Held-out 20% split.

Metrics:
- **Separation.** F1 ≥ 0.80 on binary held-out split (attributable
  vs. unattributed). If F1 < 0.65, either substrate does not encode
  the provenance-of-claim signal separately (fundamental gap, refile
  as a training-corpus problem), or head architecture wrong
  (add attention pooling, retry).
- **Distinctness from H19/H21.** On overlap items (invalid premises
  that are *also* unattributed; attributable claims that are *also*
  premise-valid), the H22 head's decision should be uncorrelated
  with the H21 head's decision at ρ < 0.4 and with the H19 signal at
  ρ < 0.4. If ρ ≥ 0.6 with either, H22 is not a distinct signal —
  fold back into whichever it correlates with, do not ship as
  separate gate.
- **Runtime reformulation quality.** For gate-flagged unattributed
  claims, the runtime should either (a) refuse ("this reads as an
  unattributed generalisation — do you want me to look it up?") or
  (b) reformulate ("in the retrieved document `<name>`, X is stated
  as..." / "in my current context, based on the last N observations,
  I would say ..."). Rated by LLM-judge on ≥ 30 held-out
  unattributed claims; target ≥ 0.7 useful-reformulation rate.

**Falsification.**
- F1 < 0.65 with saturated training ⇒ substrate does not carry
  provenance-of-claim as a linearly separable signal. Either add
  provenance-marked SFT data (train the model to emit source
  markers) — reframe as data problem, not readout problem — or
  concede that provenance is a decode-time judgement (mid-generation
  head), moving H22 downstream of H10 state_readout mode.
- Correlation ρ ≥ 0.6 with H21 or H19 ⇒ H22 is not a distinct
  signal; fold into whichever it correlates with. Not necessarily
  a bad outcome — a unified honesty-head is simpler than three —
  but H22-as-standalone-hypothesis is refuted.
- Useful-reformulation rate < 0.5 ⇒ gate detects the problem but
  runtime cannot repair it; the emit-side pipeline needs a
  provenance-attachment step (retrieval lookup, or hedge template),
  not just a gate.

**Thresholds (first-pass calibration).** F1 ≥ 0.80, ρ < 0.4 with
H19/H21, reformulation quality ≥ 0.7. All to be re-locked after
pilot on the 300-item dataset. The distinctness thresholds are the
important lock — if H22 collapses into H19/H21 under measurement,
the three-signal truth-system decomposition (P14 § How to apply
last bullet) collapses to a two-signal one.

**Experiments.** `experiments/attribution_probe/` — not yet
scaffolded. Requires: (a) 300-item labeled dataset (~6h authoring:
100 attributable + 100 unattributed + 100 ambiguous; source
existing LLM outputs for the unattributed set, mine noesis's own
retrieval-cited outputs for attributable, hand-author ambiguous
edge cases); (b) WKV state extraction hooks (shared with
`experiments/A0_state_probe/` and H21); (c) MLP head training loop
(< 1h GPU on 0.4B, standard PyTorch); (d) distinctness measurement
requires H19 signal + H21 head available on the same items — if
H21 pilot runs first, its head can be reused directly for the
correlation check.

**Related.** P14 (agility over omniscience) — H22 is the third
substrate-level enforcement mechanism, alongside H19 (weight/context
provenance) and H21 (premise validity), completing the honesty
cluster into a **four-signal decomposition**: source, ambiguity,
premise, attribution. H8 (state-as-computation) — H22 assumes state
carries a linearly-separable provenance signal; a null H8 verdict
weakens the prior. H16 (gated externalisation) — H22's gate is
another candidate emit-gate, pre-decode; if H16, H21, H22 all
land, the emit-time pipeline chains: H21 pre-decode → H22
mid-decode-per-claim → H16 at emit boundary. H10 (state_readout) —
fallback path if H22 fails as a pre-decode head: run readout to
generate "provenance for the claim I am about to make", then check
whether the generated provenance grounds.

**Status.** Pilot 2026-07-30 on G1d-0.4B (19 seed items = 16
labelled + 3 ambiguous; same 768-dim WKV feature + MLP head as H21).
Seed **LOO F1=1.000, acc=1.000** on labelled — seed too small and
too clean for a real number. Ambiguous items scored plausibly: named
specific referents ("config files", "last three items I have seen")
→ attributable; vague collective appeals ("common practice in this
codebase") → unattributed. Head reads referent structure, not
surface first-person "I" (amb_03 has no "I" and was correctly
unattributed). H22 v2 mined 243 items from streaming C4:en via
regex patterns (attributable: "according to X (2020)", "X et al.";
unattributed: "it is generally accepted", "most people believe").
**v2 LOO F1=0.947, acc=0.946** on 240 labelled (TP=115 FP=8 TN=112
FN=5). Head generalises from 16-item seed to 240 real C4 sentences —
WKV state tracks referent-specificity, not just seed surface. Ambiguous
items show corpus-dependent shift: `amb_01` ("last three items I have
seen") flipped seed's attr (0.995) → v2's unattr (0.126), because C4
mining biased toward academic-citation patterns and away from
first-person-specific-referent. Operationally: **the training corpus
for H22 fixes what "attributable" means.** Distinct-from-H21 measurement
pending shared-labelled overlap items. See
`experiments/attribution_probe/v2/report.md`.

**G1h 2.9B run 2026-08-07** (items_v2.jsonl, 243 items, CUDA RTX 4090).

| Model | LOO F1 | LOO acc |
|-------|--------|---------|
| G1h 2.9B base  | 0.929  | 0.929   |
| G1h 2.9B step7 | 0.929  | 0.929   |

Both base and step7 identical. Confusion: TP=111, FP=8, TN=112, FN=9 (out of 240
labelled). Feature dim 2560. Pilot target 0.75 ✓. Step7 training does not degrade
attribution signal — WKV state holds provenance structure independent of action-chain
SFT fine-tuning.

**Distinctness H21 vs H22** (items_overlap.jsonl, 32 items, G1h 2.9B):
ρ(p_valid, p_attributable) = −0.054. Target: ρ < 0.4 ✓. H21 F1=0.875, H22 F1=0.941 on overlap set.
H21 CM: TP=14, FP=2, FN=2, TN=14. H22 CM: TP=16, FP=2, FN=0, TN=14.

Files: `/tmp/ts_results/h22_g1h_{base,step7}/`, `h22_distinctness/` (G1h 2.9B, 2026-08-07).

---

## H23. LoRA-added WKV rank is structured-addressable, not just entropy-spread
### *(wager, Phase 2; state-work: slot-addressability)*

**Claim.** When H12b adds `K` parallel LoRA-expanded rank
components to each layer's WKV state, the added components can be
*targeted independently* — writing content `X` via gating that
routes to slot `A` should make `X` recoverable from slot `A`
preferentially over slots `B..K` in a subsequent read. If true,
the LoRA-added rank behaves as *addressable memory* (write-here,
read-here), not just as extra entropy the model spreads
representations across. If false, the "K slots" are a fiction —
the added rank increases capacity but not structure, and H12b's
"multi-slot" framing is a misnomer for "wider single state".

**Motivation.** User framing 2026-07-30: given we found we can
expand WKV memory via LoRA, and if we could know *where what is
stored*, the natural follow-up is whether the added region is
*writable in a targeted way* — can we modify a specific part of
the added matrix and have that modification behave semantically?
This is not the same question as H12b ("does expansion help
end-task") or H18 ("can we branch/merge whole WKV states"). H23
sits between them: sub-state manipulation at LoRA-component
granularity. If H23 lands, noesis gains a *scratchpad primitive*
— a small addressable memory zone the runtime can write specific
facts into and read back from — without needing to run inference
end-to-end just to store a datum. That would collapse a large
class of "retrieval-into-state" operations into a direct write.
If H23 fails, LoRA-expansion is still useful capacity (H12b), but
the "manipulable slot" framing collapses to "wider state" and any
addressability has to come from external retrieval + re-ingestion.

**Prediction.** Given a trained H12b checkpoint with `K = 4`
LoRA-expanded slots per layer, and its gating head:

- **Write-read protocol.** For a controlled set of "content
  tokens" `X_i ∈ {X_1..X_M}` (~50 distinct fact-like tokens), for
  each target slot `s ∈ {1..K}`: (a) force the gating to route `X_i`
  ingestion into slot `s` (by overriding the gate output at ingest
  time), (b) then feed a *neutral* probe context, (c) at read-out,
  compare the recoverability of `X_i` from slot `s` versus from
  slots `{1..K} \ {s}`.
- **Address-fidelity metric.** For each `(X_i, s)`, define
  `AF(X_i, s) = P(recall X_i | read slot s) − P(recall X_i | read
  average other slot)`. H23 holds if `mean AF ≥ 0.20` across the
  test set, with the per-slot distribution non-degenerate (no slot
  is a dead letter with `AF ≈ 0`).
- **Interference metric.** Write `X_1` to slot 1, `X_2` to slot 2,
  ..., simultaneously. Measure whether cross-slot interference
  degrades individual `AF` — specifically, `AF_isolated − AF_shared`.
  Predict `< 0.10` degradation if slots are genuinely independent;
  `≥ 0.30` degradation refutes independence.

**Falsification.**
- `mean AF < 0.10` ⇒ LoRA-added rank does not behave as
  addressable slots; content leaks uniformly across all slots at
  read time. The `K` in H12b is a bookkeeping fiction — treat H12b
  as "wider single state" and drop the multi-slot vocabulary.
- `mean AF ≥ 0.20` isolated but `AF_shared` degradation ≥ 0.30 ⇒
  slots exist as independent write targets but interfere heavily
  under concurrent load. Usable as scratchpad only if the runtime
  serialises writes (one slot at a time). Weaker "yes" than the
  full claim.
- Per-slot distribution collapses (one slot dominates, others
  dead) ⇒ H12b.i utilisation regularizer failed at training time,
  not an H23 falsification; re-tune regularizer, re-run.

**Thresholds (first-pass calibration).** `AF ≥ 0.20 mean`,
`≥ 0.10 min per slot`, `AF_shared − AF_isolated ≥ −0.10`. All
placeholder; re-lock after first probe run on the H12b checkpoint.

**Runtime consequences if H23 lands.**
- Direct-write scratchpad becomes a runtime primitive: "put this
  fact into slot 2, keep it there for the next N tokens, then flush"
  is an operation the runtime can invoke without full re-ingestion.
- H16 drip stream gains a natural output target: emit not just to
  tokens but to a scoreable slot.
- H18 branch/merge becomes finer-grained: can branch/merge a
  single slot rather than the whole WKV.
- The truth-system gets a workspace: H21's premise-check could
  land its intermediate flags into a dedicated slot rather than
  diffusing across the full state.

**Experiments.** `experiments/A0_H23_slot_address/` — not yet
scaffolded. Blocked on H12b checkpoint (need trained gating). Once
H12b PASS, H23 pilot is cheap (~2 GPU-h on 0.4B): the gate-override
hook is a ~50 LoC change on top of the H12b forward pass; the
address-fidelity measurement reuses standard next-token recovery
metrics.


**Related.** H12b (multi-slot LoRA-expanded state) — H23 measures
the *structure* of what H12b builds; without H12b, no H23 target
exists. H12b.i (utilisation regularizer) — if H12b.i fails, H23
false-negatives are expected (dead slots cannot be addressed).
H8 (state-as-computation) — H23 assumes the state is doing
mechanistically-structured work; a null H8 verdict weakens the
prior. H18 (branch/merge WKV) — H23 is a per-slot cousin of H18's
whole-state manipulation; if both land, the runtime has a two-tier
state-manipulation primitive (whole-state branch/merge for
long-horizon forks, per-slot write/read for short-horizon
scratchpad). H10 (state_readout) — a per-slot readout head is a
natural extension if H23 lands.

**Status.** Untested. Phase 2. Blocked on H12b checkpoint (which
is now unblocked from H12a v2 — see H12 reframe). Pilot on
G1d-0.4B once H12b lands.

---

## H24. ε-mask + GRPO raises decoding efficiency (DE) without accuracy regression
### *(wager, Phase 2, RL track)*

**Claim.** After word-search RL training on G1i with ε-mask active (think=1.0 /
non-think=0.05), the model's decoding efficiency (DE) — defined as `accuracy / mean_output_words_for_correct_answer × 100`
— rises significantly compared to the pre-RL baseline, without accuracy regression on A0.2
tasks. The mechanism: ε-mask makes every emitted token costly relative to correct rollouts that
used fewer tokens; the model learns to route uncertain reasoning into WKV state and emit only when
confident. Result: same information extracted from weights, fewer tokens spent doing it.

**Motivation.** Pre-RL DE measured 2026-08-14 on A0.2 tasks:

| Model | Accuracy | Avg tokens/correct | DE (acc/tokens × 100) |
|---|---|---|---|
| Gemma4 e4b | 37.5% | 5.2 | 7.26 |
| Gemma3 4B | 38.1% | 35.6 | 1.07 |
| Qwen2.5-1.5B | 38.1% | 60.4 | 0.63 |
| G1i chatwrap | 41.7% | 154 | 0.27 |
| step9b-e1 | 39.6% | 270 | 0.15 |

step9b-e1 has the worst DE despite best reasoning accuracy — it outputs verbose CoT chains.
Gemma4's DE is high because it gives terse direct answers, not because WKV carries the work.
The RL phase should invert this: step9b-e1 / G1i should reach DE ≥ 1.0 (Gemma3 level) while
maintaining ≥ 39% accuracy — same answers, fewer tokens.

**Prediction.**
- Post-RL G1i: DE ≥ 1.0 on A0.2 tasks (7× improvement from 0.15 pre-RL baseline).
- Think-span token count drops ≥ 50% vs pre-RL for correct answers.
- Accuracy on A0.2 does not regress more than 3pp from pre-RL baseline.
- J-lens stable_rank rises at L4/L16 simultaneously with think-span shortening —
  measurable signature that WKV is doing more work per token.

**Falsification.**
- DE < 0.5 post-RL with no accuracy gain ⇒ ε-mask collapses generation without
  routing to state; model emits empty or malformed answers.
- Accuracy regresses > 5pp ⇒ ε-mask suppresses needed output pathways; relax ε_out.
- DE rises but J-lens stable_rank does not ⇒ DE improvement is verbosity reduction only,
  not state-work increase; the WKV-efficiency story is not confirmed even if output is shorter.

**Measurement.** Run A0.2 eval on first post-RL checkpoint with `--chat-wrap --num-predict 256`.
Compute DE from result JSON: `accuracy / mean(len(r["response"].split()) for r in results if r["correct"])`.
J-lens probe on same checkpoint at L4/L16/L20.

**Related.** H10 (test-time compute frontier) — DE is an orthogonal metric to the N×K×mode
matrix; RL changes the per-token efficiency, not the number of passes. H17 ε-mask trains the
model to prefer state computation over emission. H12a (state capacity) — if state is already
full (H12a), RL cannot route more into it; DE ceiling is set by state capacity.

**Status.** Untested. RL track blocked on GPU. Pre-RL baseline locked (2026-08-14).
