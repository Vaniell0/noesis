## H4a. RWKV-7-G1 2.9B reaches parity with same-size Transformer

Record: `hypotheses/records/H4a.yml`.

**Estimator.** The score of RWKV-7-G1 2.9B after A1 on the A0.2 set against the strongest reference of similar size.

**Refuter.** It trails the strongest reference by more than 1.4x on the primary metric after A1.

**What the status rests on (2026-11-24).** `PROPOSED`: never run at 2.9B. `experiments/A0_baseline/README.md` holds the throughput of two RWKV models and says the older Qwen and Gemma figures are not comparable to it; the 0.4B pilot numbers the file names are not in that directory. Outside the research tree (README section 7): written before the programme had estimators, in the vocabulary of the runtime of that time.

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
