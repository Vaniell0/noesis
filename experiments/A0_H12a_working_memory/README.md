# A0.H12a — working-memory bottleneck attribution

Probe for hypothesis H12 (see `HYPOTHESES.md`). Distinguishes whether
RWKV-7's cross-domain reasoning failures are dominated by
**active-representation width** (number of concepts held simultaneously)
or by **decay-rate** over token distance.

## Design

Cross-linking task: N triples `(entity_i, property_i, value_i)` packed
into a single short prompt (≤ 512 tokens, well within the model's
nominal context). The question requires the model to enumerate all
entity pairs sharing a property — a strict cross-domain lookup that
cannot be answered from any single triple alone.

Two sweeps produce disjoint failure signatures:

- **Width sweep** — vary `N ∈ {4, 8, 16, 32, 64}` at fixed context
  length. Increasing N raises the number of *simultaneously active*
  concepts required to answer.
- **Distance sweep** — fix `N = 16` and vary the mean token gap
  between the query and the target triples in `{50, 200, 500, 1000}`.
  Increases decay pressure while holding representation-width constant.

## Falsification decision tree

- Accuracy falls sharply in the width sweep but stays flat in the
  distance sweep ⇒ **width bottleneck** ⇒ H12b (multi-slot LoRA) is
  worth building.
- Accuracy falls in the distance sweep but stays flat in width ⇒
  **decay bottleneck** ⇒ H12b is not worth building; the right fix is
  retrieval / longer effective context / different decay schedule.
- Both fall together ⇒ inconclusive ⇒ need finer-grained probes to
  disentangle.
- Neither falls ⇒ probe is too easy; increase difficulty (more
  distractor triples, subtler property overlap) and rerun.

## Status

Design draft only. Runner + task generator to be written after A0.8
lands and the corresponding Phase 2 architectural window opens. Budget
< 24 GPU-hours on G1d-0.4B at greedy decode.

## Files (planned)

- `gen_triples.py` — task generator (deterministic seed).
- `run_probe.py` — Ollama-driven runner over the two sweeps.
- `results/` — per-config accuracy dumps + summary table.
- `README.md` — this file.
