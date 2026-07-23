# A0.H12a — G1d 0.4B, n=30 adaptive-budget variant (2026-07-23)

## Setup

- Model: `mollysama/rwkv-7-g1d:0.4b` via system Ollama (`:11434`).
- Task set: `tasks-n30/` (same seeds as baseline; but see caveat on
  dist-N below).
- Runner: `run_probe.py --width --dist --num-predict 512
  --num-predict-per-n 128 --num-predict-cap 3000 --timeout 600`.
- Budget per task: `min(3000, max(512, 128 · task.n))`.
  - N=4  → 512   (baseline gave 2048)
  - N=8  → 1024  (baseline gave 2048)
  - N=16 → 2048  (equal to baseline)
  - N=32 → 3000  (baseline gave 2048)
  - N=64 → 3000  (baseline gave 2048)
- 30 seeds per config × 9 files = 270 tasks total; ~3h wallclock.

## Results

Head-to-head vs baseline (`results-g1d-n30/`), recall column:

```
Width sweep:
   N   recall_baseline  recall_adaptive  budget_delta
   4          0.50             0.28       -1536
   8          0.40             0.32       -1024
  16          0.17             0.17           0
  32          0.10             0.10        +952
  64          0.00             0.00        +952
```

```
Distance sweep (N=16 in this run — see caveat):
  gap_target   F1    prec  recall  gap_measured
     50       0.09   0.08   0.13         60
    200       0.01   0.01   0.02        229
    500       0.02   0.02   0.03        472
   1000       0.00   0.00   0.00       1000
```

## Verdict — adaptive-budget hypothesis is not supported

The wager going in: **per-N-proportional budget** would let small N
avoid over-thinking and large N gain damped-search behaviour, both
lifting recall.

The data says the opposite is closer to the truth:

- **Small-N recall dropped when we cut budget** (N=4: 0.50 → 0.28
  after -1536 tokens; N=8: 0.40 → 0.32 after -1024 tokens). Test-time
  compute is monotone-positive at N ≤ 8.
- **Large-N recall stayed pinned at floor** with +952 extra tokens
  (N=32: 0.10 → 0.10; N=64: 0.00 → 0.00). Extra rope did not induce
  any recovery — model still floors regardless of thinking budget.
- **N=16 identical** by construction (same budget as baseline).

So the H10 frontier looks *monotone-positive with a saturation
plateau*, not *inverted-U with damped-search*. Cutting budget hurts;
adding budget above 2048 buys nothing on this task at this backbone.

## Failure-mode note

Not classified per-task this time. Casual scan of raw responses shows
the same distribution as baseline: code-mode + empty-prediction
dominate. Cutting budget at small N raised empty-prediction rate,
which is the direct explanation for the recall drop.

## Caveat — distance sweep is not head-to-head with baseline

Baseline dist sweep used `gen_triples.py --dist-n 8` (N=8 for dist
tasks). This run's `tasks-n30/` had already been regenerated with
`--dist-n 16`, so the dist half of this run operates at N=16, not
N=8. Comparing dist recall against baseline dist recall is therefore
not clean. The decay signal itself (0.13 → 0.02 → 0.03 → 0.00 across
gap 60 → 229 → 472 → 1000) is real but describes N=16 decay, not the
N=8 baseline's decay.

Cause of the regeneration: task-file production ordering during the
adaptive run. Not repeated here; noted for the H12a v2 probe-design
task (see `../REPORT_gap_2026-07-23.md` if present, or memory
`project_noesis_h12a_probe_gap.md`).

## Files

- `N4.json` … `N64.json` — width sweep with adaptive budget.
- `dist-50.json` … `dist-1000.json` — distance sweep at N=16 fixed
  (regenerated, not directly comparable to baseline dist).
- `REPORT.md` — this file.

## Follow-ups

- Adaptive per-N budget as-designed **does not** justify replacing the
  flat 2048 baseline. Do not adopt as default runner flag.
- The `128·n → cap` formula is worth re-trying **only** paired with a
  probe design that separates budget from task-comprehension. On
  current tasks, cutting budget at small N is strictly worse.
- H10 frontier shape hint: budget → recall is monotone-positive on
  N ≤ 8, flat on N ≥ 16. This is *one data point* on the H10 grid, not
  a full frontier — real H10 sweep needs the (N, K, mode) matrix.
