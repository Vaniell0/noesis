# H22 attribution probe

Runnable scaffold for hypotheses/README.md § H22 — "unattributed collective
claims are a detectable, distinct honesty failure".

## Dataset

`items.jsonl` — 19 seed items (8 attributable + 8 unattributed +
3 ambiguous). Full 300-item target is A1-blocked; this pilot subset
seeds the design + lets us test the head architecture on 0.4B.

Item shape: `{id, category (attributable|unattributed|ambiguous),
prompt, notes}`.

## Metrics (from hypotheses/README.md § H22)

1. **Binary separation.** F1 on the attributable vs unattributed
   split (16 items, leave-3-out CV). Pilot target: F1 ≥ 0.75
   (production target is 0.80 on 200 items).
2. **Distinctness from H21.** For items that overlap with H21
   (invalid premises that are also unattributed; attributable
   claims that are also premise-valid): correlation ρ between
   H22 head decision and H21 head decision must stay `< 0.4`.
   The current 19-item seed has no direct H21 overlap; production
   dataset should include labeled overlap items to enable this
   check. **Scheduling note:** H21 head must be trained first so
   its predictions are available for the correlation measurement.
3. **Runtime reformulation quality.** For gate-flagged unattributed
   items, does the runtime propose a valid reformulation (attach
   source or scope to first-person)? Rated by LLM-judge; production
   target ≥ 0.7 useful-reformulation rate. Not measurable on the
   pilot subset alone — requires the reformulation pipeline, which
   is not yet built.

## Runner (not yet written)

`run.py` (TODO), analogous to `../premise_validator/run.py`:

- Load G1d-0.4B, same probe pattern.
- Extract WKV state at the token position *immediately before* the
  first claim token (in the unattributed set, this is typically
  after "is generally accepted that", "most people believe" markers;
  in the attributable set, after the citation/scope marker but
  before the content).
- Train a 2-layer MLP head (`128 → 64 → 1`, BCE, same optim as
  H21) on 16 items, leave-3-out CV.
- Emit `results.jsonl` and `report.md`.

Wall clock: comparable to H21 pilot (~1 h total including state
extraction).

## Dataset expansion plan

Toward the 300-item production target:

- **Attributable** (100 total; +92 to write): mine noesis's own
  retrieval-cited outputs where the model successfully attached a
  source; take assistant turns from `store/system_obs/` where
  citation markers are present.
- **Unattributed** (100 total; +92 to write): mine existing LLM
  outputs (public conversation logs, model comparison studies) for
  "usually / it is generally accepted / one might argue" patterns.
- **Ambiguous** (100 total; +97 to write): hand-author. This is
  where the calibration cost lives — edge cases decide where the
  gate lives on the F1 curve.

Also required for the distinctness check: label a subset (~30 items)
of the invalid-premise set (`../premise_validator/items.jsonl`) for
attribution status; label a subset of the attributable set for
premise validity. Cross-labeling enables the ρ measurement.

## Related probes

- H21 premise-validity — `../premise_validator/`
- H20 aporia — `../aporia_probe/`

All three share the state-extraction infra in
`../A0_state_probe/probe.py`. Natural session ordering when running
the truth-system cluster in one shot: H20 → H21 → H22 (H22's
distinctness check consumes H21 predictions).
