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
HYPOTHESES.md.

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
current hardware — GTX 1050 + laptop CPU — without cloud dependency.
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
HYPOTHESES.md's evaluation philosophy exists to prevent.

**How to apply.** Every philosophical claim in this project must be
paired with a falsifiable hypothesis and a cheap test. Refutation
outweighs elegance. Silence on evidence is not a valid response —
neither is post-hoc reframing.

**Cost.** Some beautiful ideas will die. That is the point.

---

## P9. Falsify before you build

**Statement.** Every non-trivial design bet is paired with a
falsifiable hypothesis in HYPOTHESES.md and a cheap probe *before*
significant build effort is committed.

**Why.** Sunk-cost bias then rescues bad decisions. Cheap experiments
protect against months of misdirected work. Gate 1 exists exactly
for this reason.

**How to apply.** Before starting a phase in ROADMAP.md, verify: is
there a hypothesis in HYPOTHESES.md this phase advances or tests? Is
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

**Why.** Without negatives, HYPOTHESES.md drifts toward a wishlist.
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
