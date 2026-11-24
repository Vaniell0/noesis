## H4b. State-evolution architectures are viable for reasoning
### *(wager, not directly falsifiable at this project's scale)*

Record: `hypotheses/records/H4b.yml`.

**Estimator.** None single: the accumulated H4a-style comparisons over generations, training runs and eval sets.

**Refuter.** A pattern across them disfavours RWKV without a plausible confound.

**What the status rests on (2026-11-24).** `WAGER`: a wager by its own words, never tested. A pattern, not one run, reopens the choice of backbone; nothing here moves a status. Outside the research tree (README section 7): written before the programme had estimators, in the vocabulary of the runtime of that time.

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
