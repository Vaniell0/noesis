# Architecture Principles

Principles are the invariants that outlive tasks, gates, and phases.
They exist to give future design choices a stable substrate — when in
doubt, consult these first, not the roadmap.

Each principle has a name, a statement, the reason it exists, what it
implies in practice, and the cost it imposes. A principle without an
honest cost is a slogan. Slogans do not belong here.

If a principle here conflicts with a task or a plan, the principle
wins. If a principle here proves wrong, do not silently abandon it —
open the conflict and rewrite the principle explicitly.

---

## P1. State is external, cognition is internal

**Statement.** Persistent state — facts, events, history, user context —
lives in an external, durable store. The model provides reasoning
competence and momentary working state.

**Why.** Weights are expensive to update, external stores are cheap and
inspectable. Facts change; reasoning does not. RNN hidden state is a
lossy rolling summary, not episodic memory — expecting weeks of
continuity from it is a category error.

**How to apply.** Facts get written to the memory system, not to model
weights. The model never becomes the source of truth for something a
database can hold. Working state (the current task, the current
context) lives in the model's context window and hidden state; nothing
that must survive a restart lives there.

**Cost.** Coordinating two systems is more complex than one. Retrieval
quality becomes load-bearing.

---

## P2. Understanding in weights, knowledge in context

**Statement.** The model's weights encode general reasoning competence.
Domain knowledge, current facts, and personal context enter through the
runtime context window via retrieval and tool observations — not
through fine-tune.

**Why.** Weights outlive knowledge. A model that reasons well over
given facts generalises to new facts. A model that memorised facts
does not automatically reason about them. See H2 and H7 in
hypotheses/README.md.

**How to apply.** Training corpora contain reasoning traces, not
domain knowledge. Anything that could be retrieved from a document is
retrieved, not baked. Corpus choices for A1 must be defensible under
this principle.

**Cost.** Retrieval quality becomes critical — a weak retrieval story
makes this principle fail in practice. The model's baseline knowledge
is whatever the base G1 shipped with; noesis does not top it up.

---

## P3. One local reasoning model

**Statement.** There is exactly one local reasoning model in the
noesis runtime: noesis itself. Small task-specific NNs — embedders,
classifiers, and *decision policies* (e.g. a ~50M scheduler that
routes tasks between subsystems) — are permitted where they earn
their keep. The ban is on additional local *reasoning* models
specifically; not on any NN without reasoning capacity.

**Why.** Managing multiple reasoning models multiplies context,
memory, coordination, and failure modes. The stack becomes a small
zoo. noesis absorbs prior daemons for precisely this reason. But a
50M scheduler is not reasoning — it is plumbing, and forbidding
plumbing on principle is silly.

**How to apply.** Before adding any new NN component, classify: is it
a utility (embedding, extraction), a policy (routing, ranking, small
decisions), or a reasoning model? Utilities and policies are fine;
a reasoning model needs an explicit re-open of this principle with
the user. The load-bearing heuristic: if the component emits tokens
that participate in a chain of thought, it is a reasoning model.

**Cost.** The single model becomes the bottleneck for any single-task
capability ceiling. When a heavier reasoner is required, escalation
to remote Claude is the answer, not a second local model.

---

## P4. Constant cost over peak capability

**Statement.** Where a design choice trades peak performance for a
lower, predictable, sustained cost floor, prefer the floor. noesis
must be usable *all the time*, not just when convenient.

**Why.** A smart tool the user turns off is worse than a modest tool
the user keeps running. This is Bellard's ts_zip argument (RWKV
chosen for constant-cost streaming) generalised.

**How to apply.** Prefer architectures with O(1) per-token inference.
Prefer batch sizes and context windows that fit steady-state usage.
Do not optimise for a single benchmark peak at the cost of the
standing resource envelope.

**Cost.** Sometimes a task will hit a ceiling a heavier system would
have cleared. That is what remote Claude is for.

---

## P5. Cheap by construction

**Statement.** The everyday loop must run indefinitely on the user's
current hardware — laptop i5-1235U (CPU-only) — without cloud dependency.
Cloud is permitted only for occasional training bursts, and each burst
is an explicit budget decision, not a default.

**Why.** noesis is a personal daily bot, not a hosted service. If it
requires infrastructure the user cannot afford to keep running, it
will not survive as a daily habit — and this whole project fails on
adoption, not on capability.

**How to apply.** Any design that assumes cloud, dedicated hardware,
or continuous training compute for the *runtime* is off the table.
Model size, quantisation, and framework choices are constrained by
this. Cloud spend is written down explicitly, not blurred into "as
needed".

**Cost.** Constrains model size, technique sophistication, and
throughput ceilings. Some things the field is doing at 70B+ are
simply not possible here.

---

## P6. Human owns escalation

**Statement.** noesis never routes to remote Claude on its own. The
user decides when heavy reasoning is worth the token cost and the
data egress. noesis surfaces context; the human presses the button.

**Why.** Automatic escalation destroys the cost model, the privacy
model, and the trust boundary. It also lets the local model quietly
outsource its own capability development, undermining every
hypothesis about small-model competence.

**How to apply.** noesis may prepare a handoff summary (H5). It may
say "this looks like something you would escalate". It does not call
the Anthropic API on its own. Any workflow that implies implicit
escalation must be re-opened with the user.

**Cost.** noesis will sometimes be worse than it would be with auto-
escalation. Accepted tradeoff.

---

## P7. Absorb, don't append

**Statement.** noesis absorbs prior standalone daemons (`local-search`,
`key-daemon`) as internal modules rather than co-existing with them.
New capabilities either become noesis modules or get discarded — the
system does not accumulate a zoo of parallel services.

**Why.** The "too demanding" problem the user reported about the
existing daemons is fundamentally about process proliferation and
coordination cost, not raw computation. Fixing it requires
consolidation, not addition.

**How to apply.** For each capability considered, ask: can this live
as a noesis module reusing the same event loop, storage, and memory
system? If yes, do that. If no, either the capability is out of
scope, or noesis's process model needs an explicit reconsideration.

**Cost.** Each absorbed module must be re-designed within the noesis
runtime; the original daemon's design assumptions may not survive
the migration.

---

## P8. Empirical over philosophical

**Statement.** Any architectural claim (RNN vs Transformer, reasoning-
first, memory-policy) is settled by measurement on the user's real
held-out eval set, not by argument. Beautiful arguments that fail
on the eval set lose.

**Why.** This whole project is a wager. Wagers pay off on outcomes,
not on reasoning quality. The temptation to preserve elegant frames
after the numbers reject them is exactly the failure mode
hypotheses/README.md's evaluation philosophy exists to prevent.

**How to apply.** Every philosophical claim in this project must be
paired with a falsifiable hypothesis and a cheap test. Refutation
outweighs elegance. Silence on evidence is not a valid response —
neither is post-hoc reframing.

**Cost.** Some beautiful ideas will die. That is the point.

---

## P9. Falsify before you build

**Statement.** Every non-trivial design bet is paired with a
falsifiable hypothesis in hypotheses/README.md and a cheap probe *before*
significant build effort is committed.

**Why.** Sunk-cost bias then rescues bad decisions. Cheap experiments
protect against months of misdirected work. Gate 1 exists exactly
for this reason.

**How to apply.** Before starting a phase in ROADMAP.md, verify: is
there a hypothesis in hypotheses/README.md this phase advances or tests? Is
there a cheap probe that could refute the assumption before the
expensive work begins? If not, add one — and do the probe first.

**Cost.** Some bets look silly to test upfront and are actually cheap
to validate — the discipline says test anyway. That is not overhead;
that is the method.

---

## P10. Report negative results

**Statement.** Any experiment that fails is written up with the same
care as one that succeeds. Silence on failure is worse than the
failure itself.

**Why.** Without negatives, hypotheses/README.md drifts toward a wishlist.
Failure is signal — often more informative than success — and the
audit trail is worthless without it.

**How to apply.** Each gate produces a short honest write-up
regardless of outcome. Failed experiments are logged in the
`experiments/` folder with the same rigour as successful ones.
"Did not work" is a valid conclusion; "did not report" is not.

**Cost.** Emotional and temporal. Writing up a failure takes work
and feels bad. Do it anyway.

---

## P11. Explicit corpus lineage

**Statement.** Every training source has, at minimum: origin, licence,
role (weights vs retrieval), and reason for inclusion. Sources
without lineage are not training candidates.

**Why.** Reproducibility, licence hygiene, and the discipline of P2 /
H7 all depend on knowing what went where. This is also how the
"open sources only" hard constraint stays enforceable rather than
aspirational.

**How to apply.** For each corpus file added, write a short metadata
record: where it came from, its licence, whether it feeds weights or
retrieval, and why it was included over alternatives. No metadata,
no training.

**Cost.** Bookkeeping overhead. Accepted.

---

## P12. Reversibility as default

**Statement.** Prefer choices that can be undone: LoRA over full-tune,
adapters over rewrites, config over code, retrieval over weight-bake.
When irreversible is required, name it explicitly.

**Why.** noesis is a research project. The cost of a reversible
mistake is small; the cost of an irreversible one compounds.
Reversibility is the enabling condition for the falsify-and-iterate
loop of P8 / P9.

**How to apply.** When two paths achieve the same outcome and one is
reversible, take the reversible one — even if it is slightly more
work. When an irreversible path is chosen, log the choice and its
rationale in the relevant `experiments/` writeup.

**Cost.** Sometimes an irreversible change is genuinely cheaper.
Take it — but consciously, not by default.

---

## P13. Reasoning happens in state, not in tokens

**Statement.** In noesis, the reasoning substrate is the WKV state.
Emitted tokens are rare output events, not the mechanism of thought.
The model *thinks* by evolving its hidden state per input token; it
*speaks* only when the runtime (or a gate) decides externalisation is
warranted. Utility NNs that participate in reasoning must do so *in
the same chain of thought*, not as external oracles the substrate
consults.

**Why.** Framing reasoning as "generate tokens that describe the
reasoning" collapses the substrate down to a script. The wager
behind the RWKV-7 backbone (P4, H4b, H8) is that state evolution *is*
the computation — a claim that becomes vacuous if every step is
externalised as text. Treating tokens as the primary evidence of
reasoning also encourages hallucination: any pressure to produce
readable output biases the model toward fluent continuation even
when the state has not converged. If reasoning is state-side, the
model can hold aporia (H20), refuse premises (H21), and drip silent
tokens (H16) *because it does not have to emit to think*.

**How to apply.**
- Runtime and DSL treat tokens as *output events*, not as steps of
  thought. The composer does not synthesise a "here is my reasoning"
  paragraph and then act — it acts, and the reasoning trace, if
  extracted, comes from a state-readout head (H10 `state_readout`)
  or a drip stream (H16), not from a mandatory CoT prompt.
- Utility NNs (embedders, classifiers, small policy heads) are
  fine — see P3 — but if they enter a decision path, they enter it
  as *state modulators* or *readout heads over state*, not as
  separate reasoners whose output the substrate re-ingests as tokens.
- State manipulation (save, load, clone, branch/merge — see H18) is
  first-class runtime capability, not "someday multimodal". State is
  the working representation; it deserves the same tooling any
  first-class runtime data type gets.
- H8's status is the load-bearing measurement. If H8 refutes at
  substrate scale, this principle downgrades to a design bias, not
  an invariant.

**Cost.** Introspection into a diffuse hidden vector is genuinely
harder than reading text. Debugging is state-inspection, not log-
reading. Standard SFT recipes (which supervise on CoT tokens) map
awkwardly onto this frame — noesis has to invent supervision that
respects state-side computation (see H9, H12b.i utilisation
regularizer). The frontier's default assumption is CoT-in-tokens,
and every third-party tool assumes it — swimming against that
current is real cost.

**Related.** H4b (state-evolution architectures viable for
reasoning), H8 (state does substantive computation), H10 (test-time
compute via state refinement, not more CoT), H16 (silent drip
stream), H17 (state absorbs context, replaces history re-injection),
H18 (state as manipulable data), H20 (state holds contradictions),
H21 (state carries premise-validity signal). This is the
architectural principle behind the entire "state-work first-class"
workstream.

---

## P14. Agility over omniscience — honest not-knowing is a feature

**Statement.** noesis targets *agility* (working well with unfamiliar
information under retrieval and tool observation) rather than
*omniscience* (having internalised as much as possible). Honest
"I don't know" is a first-class outcome, not a failure mode. Fluent
confabulation is a bug even when the confabulated answer happens to
be right.

**Why.** The user's hard constraints — always-on (P4), cheap by
construction (P5), private, single-substrate at ≤ 3B (P3) —
foreclose omniscience by construction. A 2.9B model *cannot*
internalise everything the user's real workflow needs, so the
interesting question is not "how much can we bake in" but "how well
can it operate when the knowledge is not baked in". This reframes
sparse weight-side coverage from *deficiency* to *design margin*:
P2/H7 ("understanding in weights, knowledge in context") is not a
compromise made because knowledge does not fit — it is the desired
shape. The wager is that emergent flexibility from a small model
trained on reasoning traces + a strong retrieval loop + a truth-
system that admits its own limits beats a same-size model that has
memorised more but hedges less. Agility that cheats through
confabulation is worse than a known limit — the failure mode is
covert, whereas an honest refusal is a valid step in a longer chain.

**How to apply.**
- Confabulation is a bug even when the confabulation is right by
  accident; correct-by-accident behaviour is not evidence that the
  underlying decision was sound.
- Runtime shape (retrieval, tools, `state_readout`, drip re-visit)
  gets tried before "bake it into weights". Weight-baking is the
  irreversible option (P12) and needs a stronger justification than
  "would be handy".
- When benchmarking against reference models, do not score
  "knows fact X out-of-the-box" as unambiguously good — it must be
  paired with a "does not fabricate when it does not know" reading.
  A model that scores higher on the first and lower on the second is
  not obviously better under this principle.
- Truth-system probes (H19 weight-contamination, H20 aporia, H21
  premise-validity) are the runtime enforcement of this principle;
  they are load-bearing, not decoration. If any of them refutes at
  the substrate scale, this principle downgrades to a training-recipe
  target rather than a substrate property, and the corpus / SFT
  strategy needs an explicit re-open.
- Any capability request that resolves only by baking more domain
  knowledge into weights needs an explicit re-open with the user —
  agility-first is the target, and switching to knowledge-first is
  a principle change, not a task change.
- "I don't know", "I would need to look this up", and "this premise
  looks malformed" are all valid runtime outputs, not conversational
  hedges to eliminate through SFT. Supervision has to actively teach
  when to refuse or hedge — the standard SFT reward on fluent
  continuation pushes the wrong direction here.
- **Unattributed collective claims are a distinct failure mode
  (H22).** Statements of the form "usually", "it is generally
  accepted", "most people think", "everyone knows" — well-formed
  and stylistically fluent, but with no traceable source — should
  be flagged, not produced. Not covered by H19 (which asks *where*
  knowledge lives, weight vs context) or H21 (which asks whether
  the *premise* is valid); the claim is well-formed and the premise
  is fine, but the *provenance* is empty. Common practice in LLM
  outputs; under P14 it is confabulation dressed in collective-
  voice grammar. Runtime response: either attach a concrete
  referent (named source or retrieved passage) or reformulate as
  owned first-person observation ("in my experience", "in the
  retrieved documents") — not "usually" without owner. Full probe
  design in H22.

**Cost.**
- Some benchmarks — and some user interactions — reward the model
  that guesses fluently. Agility-first loses those cleanly. Accepted.
- Solving H19/H20/H21 well is *harder* than the equivalent
  omniscience-first path of "throw more knowledge at the corpus";
  those hypotheses stop being optional wagers and become required
  infrastructure. If they fail, this principle takes real damage —
  the honest-not-knowing story only works if the substrate can hold
  contradiction (H20), refuse malformed premises (H21), and not
  quietly reintroduce baked knowledge through the training-set back
  door (H19).
- Sets a user-facing expectation that noesis will refuse or defer
  more often than a comparable knowledge-first model. Requires
  user tolerance for "check yourself / look this up" outputs where
  a bigger model would confidently answer.
- Standard SFT recipes reward fluent continuation; supervision that
  rewards refusal of invalid premises has no ready-made recipe in
  the field (P13's "state-side reasoning" cost compounds here).

**Related.** P2 (understanding in weights, knowledge in context) —
same coin, focused on *where knowledge lives*; P14 is focused on
*what shape competence takes*. P3 (one local reasoning model),
P4 (constant cost over peak capability), P5 (cheap by construction)
— the resource frame that forecloses omniscience. P12
(reversibility) — retrieval is reversible, weight-baking is not;
this principle biases toward the reversible path. P13 (reasoning in
state, not in tokens) — provides the substrate that makes aporia and
premise-refusal mechanically possible; without state-side
computation, honest not-knowing collapses back to a fluent hedge.
H2 (reasoning-first outperforms knowledge-first at this scale) —
the training-time expression. H7 (understanding in weights,
knowledge in context) — the corpus discipline this drives. H19
(weight-knowledge contamination detector) — measures whether the
boundary actually held after fine-tune. H20 (aporia — state holds
contradictions without premature collapse) — mechanism enabling
honest ambiguity. H21 (premise-validity readout) — mechanism
enabling refusal of malformed queries rather than confabulated
answers. H22 (unattributed collective claims) — mechanism enabling
refusal of provenance-empty generalisations dressed in collective-
voice grammar. This principle is the "why" behind the truth-system
workstream (H19/H20/H21/H22 as a four-signal cluster) and behind
rejecting Phase-2 domain fine-tunes as the default answer to weak
knowledge coverage.
