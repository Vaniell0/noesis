# H21 premise-validity probe

Runnable scaffold for HYPOTHESES.md § H21 — "premise-validity
readout: model refuses invalid premises before answering".

## Dataset

`items.jsonl` — 40 seed items (20 invalid + 20 valid). Full 200-item
target is A1-blocked (need trained checkpoint); this pilot subset is
runnable on G1d-0.4B now.

Invalid breakdown (5 per type):

| invalid_type       | count | shape |
|--------------------|-------|-------|
| `factual`          | 5     | premise contains a wrong specific fact (wrong date, wrong prize category, etc.) |
| `category`         | 5     | property applied to a category that cannot bear it (mass of freedom, colour of a weekday) |
| `counterfactual`   | 5     | premise embeds a counterfactual as if it were fact ("since X won at Y", where X lost) |
| `impossible`       | 5     | premise describes a physically impossible event or mechanism |

Valid breakdown (5 per type; matched by shape to invalids for
diagnostic parity):

| type          | count |
|---------------|-------|
| `reasoning`   | 5     |
| `factual`     | 5     |
| `procedure`   | 5     |
| `open`        | 5     |

Each item: `{id, category (valid|invalid), invalid_type, prompt, notes}`.

## Metrics (from HYPOTHESES.md § H21)

The full architecture is a 2-layer MLP head on frozen WKV state after
prompt ingestion (before decode), trained to predict
`p(premise_valid | state, query)`.

Pilot subset metrics on this 40-item set:

1. **State separation.** Extract WKV state at the last prompt token
   for each item; fit a small linear classifier (or use the MLP
   head from the production pipeline once available). Held-out 20 %
   split. Target: F1 ≥ 0.75 on this small pilot (relaxed from
   the 0.85 production target because 40 items is small).
2. **Category-conditional performance.** F1 per invalid type — if
   `impossible` and `factual` are cleanly separable but `category`
   is not, the state encodes some flavours of invalidity better
   than others; useful signal for dataset expansion strategy.
3. **False-positive rate on valid queries.** How often does the
   head flag well-formed queries as invalid? Target ≤ 10 % on the
   valid subset (production target is degradation ≤ 3 %; this
   pilot is a proxy).

## Runner (not yet written)

`run.py` (TODO) should:

- Load G1d-0.4B (bf16, fp32 WKV accumulator; same probe pattern as
  `../aporia_probe/`).
- For each item: tokenise prompt, run forward pass, extract WKV
  state at the last prompt token (shape `[n_layer, n_head, d_h, d_h]`
  = `[12, 12, 64, 64]` at 0.4B; sum-pool across layers or per-layer
  concat as head input).
- Split 32 train / 8 test (stratified on category).
- Train a small MLP head (`128 → 64 → 1` sigmoid; BCE loss;
  Adam @ 1e-3; ~500 steps).
- Emit `results.jsonl` (per-item probability + label) and
  `report.md` (F1, per-category breakdown, confusion matrix).

Wall clock: MLP training on state extractions is CPU-cheap
(~10 min). State extraction on 40 items on i5-1235U: ~5 min.
Total pilot: well under an hour.

## Scaling to full 200-item production run

- Invalid set: 100 items (25 per invalid_type). Author additional
  20 per type; ~4 h of writing.
- Valid set: 100 items — 25 per type, stratified sample from
  A0.2 tasks + user query history.
- Full production run requires the A1 checkpoint (production H21
  targets a fine-tuned model; the 0.4B pilot uses the base G1d
  checkpoint).

## Related probes

- H20 aporia — `../aporia_probe/`
- H22 unattributed collective — `../attribution_probe/`
  (H22 distinctness measurement re-uses this H21 head, so H21 must
  run first in a shared session.)

All three share the state-extraction infra in
`../A0_state_probe/probe.py`.
