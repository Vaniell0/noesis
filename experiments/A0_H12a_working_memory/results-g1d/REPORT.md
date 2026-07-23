# A0.H12a — G1d (tuned 0.4B) run report (2026-07-23)

## Setup

- Model: `mollysama/rwkv-7-g1d:0.4b` served by system Ollama at
  `127.0.0.1:11434`. G1d is BlinkDL's instruction-tuned RWKV-7 line
  (G = general, d = distilled). Reasoning-model output: chain-of-thought
  goes to `thinking`, final answer to `response`.
- Runner: `run_probe.py --host http://127.0.0.1:11434
  --model mollysama/rwkv-7-g1d:0.4b --width --dist --num-predict 2048
  --timeout 600`.
- **Runner patch (this run):** `call_ollama` now concatenates
  `thinking + response`. The pair extractor is regex-based, so it can
  pick up IDs the model mentions during reasoning. Without this patch
  every response would be empty because the CoT phase alone consumed
  the token budget (`done_reason = "length"`, `response = ""`).
- 3 tasks per config (design default). Real interpretation still needs
  n ≥ 30 per config.

## Results

```
Width sweep (N ↑, mean_word_gap ~ N):
   N   F1    prec  recall  gap_w
   4   0.17  0.11  0.33       10
   8   0.09  0.05  0.67       13
  16   0.01  0.01  0.33       22
  32   0.00  0.00  0.00       72
  64   0.00  0.00  0.00       90

Distance sweep (target gap ↑, N=16 fixed):
  gap   F1    prec  recall  gap_measured
   50   0.00  0.00  0.00          40
  200   0.00  0.00  0.00         314
  500   0.00  0.00  0.00         498
 1000   0.00  0.00  0.00         860
```

`acc_exact = 0` for every config — the model never produced exactly the
right pair set.

## Comparison with base World Q8_0 sanity run

| axis | base World Q8_0 | G1d |
|------|-----------------|-----|
| N=4 F1 | 0.50 | 0.17 |
| N=4 recall | 1.00 | 0.33 |
| N=8 F1 | 0.13 | 0.09 |
| N=16 F1 | 0.03 | 0.01 |
| N=32 F1 | 0.01 | 0.00 |
| N=64 F1 | 0.00 | 0.00 |
| dist-50 F1 | 0.15 | 0.00 |

Counter-intuitively, base World scored **higher F1** at N=4, but that
was a formatting artifact: base World dumped all 4 IDs on one line, the
extractor generated C(4,2)=6 pairs, of which 2 were correct → recall
1.00, precision 0.33. G1d actually reasons — sometimes correctly, mostly
wrongly. Its recall at N=8 (0.67) is real signal, not enumeration
artifact.

## What this establishes

1. **G1d attempts the task.** Sample from N=4-s0: the model emits
   properly-formatted `item-alpha-00, item-beta-01\nitem-gamma-02,
   item-delta-03` pair lines. It just paired the wrong items (positional
   adjacency 00↔01 and 02↔03 instead of colour-based 00↔02 and 01↔03).
   That is a *reasoning* failure, not a *representation* failure.
2. **G1d writes code sometimes.** N=4-s2 response is ~8 KB of Python
   scaffolding building a dictionary — the extractor still hits IDs in
   the code and generates C(N,2) pairs, giving spurious "signal". This
   is the same enumeration artifact base World had. Rate is task-random
   (1/3 in this window).
3. **Width drop is real.** Recall goes 0.33 → 0.67 → 0.33 → 0.00 → 0.00
   from N=4 to N=64. The step from N=16 to N=32 is where the model
   stops emitting any correct pair at all — this is closer to the
   "working memory bottleneck" signal H12a was designed to detect.
4. **Distance sweep is uninterpretable at n=3.** All 4 gap conditions
   flatlined at F1=0, but N=16 in the width sweep also only scored
   F1=0.01 — the dist axis inherits the width-axis failure at N=16 and
   can't show independent decay. Need to either (a) lower N to 8 for
   the dist sweep or (b) use a much more capable model.

## Verdict on H12a design

- **Reasoning-model output format matters.** `thinking` vs `response`
  split, num_predict pressure, code-writing behaviour all interfere
  with the pair-extraction pipeline. Any future run must either patch
  around this (as done here) or use a non-reasoning tag.
- **n=3 is not enough for a decay curve.** The width axis shows an
  interpretable trend (recall drops sharply between N=16 and N=32) but
  a single s2 that happens to write code can move the aggregate by
  ±0.15 F1. The full run needs n ≥ 30.
- **The task hits a floor before the distance axis bites.** dist-50
  should be strictly easier than N=16 in the width sweep (same N,
  smaller gap), but both sit at ~0 F1. This says the model is bottlenecked
  by *task comprehension*, not distance. A cleaner distance probe needs
  N below the comprehension bottleneck — for G1d 0.4B, that's N ≤ 8.

## Follow-ups (not this run)

1. **Re-run distance sweep with N=8 fixed** (not N=16). Currently the
   dist axis measures nothing because it starts below the model's
   comprehension threshold.
2. **Raise n per config to ≥ 30** before treating any F1 delta as a
   robust signal.
3. **Add a `--strip-code` filter to `run_probe.py`** so responses
   containing Python-like `pair = (…)` scaffolding are either
   normalised or excluded from pair extraction — currently they inflate
   `predicted` size to `C(N,2)` and destroy precision.
4. **Try G1d 1.5B or larger** — the 0.4B tag hits floor before the
   H12a decay axes engage. Larger G1 tag with more instruction capacity
   would give a cleaner curve.
5. **H12b (multi-slot LoRA)** — gated on H12a producing a clean
   "width, not decay" verdict. Current H12a on 0.4B G1d does *not*
   satisfy that gate (the flat verdict is a floor effect, not a
   confirmation of width bottleneck). Do not launch H12b yet.

## Files

- `N4.json` … `N64.json` — width sweep raw + aggregate.
- `dist-50.json` … `dist-1000.json` — distance sweep raw + aggregate.
- `SUMMARY.txt` — `summarise.py` output.
- `REPORT.md` — this file.
