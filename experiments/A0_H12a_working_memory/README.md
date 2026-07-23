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

Implemented 2026-07-23. Runner (`run_probe.py`) drives Ollama or the
in-process rwkv.cpp HTTP shim; generator (`gen_triples.py`) supports
seed count and dist-N override; summariser (`summarise.py`) applies
the falsification decision tree.

Runs so far (all CPU-only, i5-1235U Alder Lake):

- `results-q8_0/` — RWKV-7-World-0.4B Q8_0 baseline via the noesis-runtime
  HTTP shim on :11435. Base model, no instruction tuning; hits the
  "enumerate-all IDs on one line" failure mode from N=4 onward.
- `results-g1d/` — mollysama/rwkv-7-g1d:0.4b via system Ollama. Tuned
  reasoning model; runner patched to concat `thinking + response` since
  G1d is CoT-formatted. Sanity: n=3 per config.
- `results-g1d-n30/` — same G1d 0.4B but n=30 per config and dist-N=8
  fixed (dist sweep at N=16 had floored below task-comprehension
  threshold in the sanity run).
- `results-g1d-n30-adaptive/` — adaptive `num_predict = min(3000,
  max(512, 128 × N))` variant of the n=30 run. Tests whether giving
  the model more test-time compute at large N produces a *damped-search*
  degradation curve rather than divergence into code-mode. Follow-up
  informed by the H10 axis.

Budget for the follow-on H12b LoRA (multi-slot WKV) remains
< 24 GPU-hours on G1d-0.4B, but H12b is gated on H12a producing a
clean width-verdict — see `results-*/REPORT.md` for the current gate
status.

## Files

- `gen_triples.py` — task generator (deterministic seed).
- `run_probe.py` — Ollama-shape runner; supports `--num-predict-per-n`
  and `--dist-n` for the follow-on sweeps.
- `summarise.py` — decision-tree verdict against the width/distance
  drop thresholds.
- `tasks/` — n=3 sanity task set (design default).
- `tasks-n30/` — n=30 clean-rerun task set (dist-N=8).
- `results-q8_0/`, `results-g1d/`, `results-g1d-n30/` — per-run
  results, each with `SUMMARY.txt` and `REPORT.md`.
- `README.md` — this file.
