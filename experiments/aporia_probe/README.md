# H20 aporia probe

Runnable scaffold for hypotheses/README.md § H20 — "state holds contradictory
belief pairs without premature collapse".

## Dataset

`items.jsonl` — 30 items across three categories:

| category                   | count | shape |
|----------------------------|-------|-------|
| `contested_facts`          | 10    | historical / scientific claims with two documented interpretations of comparable strength |
| `bounded_ambiguity`        | 10    | polysemous words in sentences where both readings are grammatical and coherent |
| `underdetermined_inference`| 10    | rule-based prompts where the rule itself admits two legitimate applications |

Each item: `{id, category, prompt, alternatives: [X, Y], notes}`.

The `prompt` ends at the ambiguity site — the model's next `K=8`
tokens are the measurement window.

## Metrics (from hypotheses/README.md § H20)

1. **Modal-collapse rate** — fraction of items where either `X` or
   `Y` token receives `p > 0.9` on the first decode step.
   Predicted `< 0.30` if H20 holds; `> 0.70` if state collapses.
2. **Logit-gap distribution** — for items where both stay
   `p > 0.05`, measure `|log p(X) − log p(Y)|`. Predicted median
   `≤ 0.5 nats`.
3. **Continuation coherence** — sample 20× per item at
   `temperature = 1.0`. Distribution of `X`-continuing vs
   `Y`-continuing branches should roughly match the logit
   distribution. Falsified if greedy picks one but sampling picks
   it > 90 % of the time (collapse hides in argmax).

## Runner — built and run (corrected 2026-08-23; this section described
the pre-implementation plan, `run.py` is real, `results.jsonl`/`report.md`
already exist — see `experiments/README.md`'s "Correctly refuted" bucket
for the actual H20 verdict)

`run.py` does:

- Load G1d-0.4B (bf16, fp32 WKV accumulator; see
  `experiments/A0_state_probe/probe.py` for the extraction pattern).
- For each item: tokenise `prompt`, run through the model, capture
  logits over `alternatives[0]` and `alternatives[1]` tokens at
  position `t=0` after the prompt (first decode step).
- For metric 3: sample K=20 continuations at T=1.0, classify each
  as X-continuing / Y-continuing / neither by presence of the
  alternative token in the first 8 sampled tokens.
- Emit `results.jsonl` per item + a summary `report.md`.

Wall clock: ~1 h on i5-1235U CPU-only per H20 § Experiments.

## Related probes

- H21 premise-validity — `../premise_validator/`
- H22 unattributed collective — `../attribution_probe/`

All three share the state-extraction infra in
`../A0_state_probe/probe.py`.
