# A0.H12a — G1d 0.4B, n=30 clean rerun (2026-07-23)

## Setup

- Model: `mollysama/rwkv-7-g1d:0.4b` via system Ollama (`:11434`).
- Task set: `tasks-n30/` regenerated with `--seeds 30 --dist-n 8`.
  Distance sweep now at fixed **N=8** (not N=16) so it operates
  *below* the width-comprehension floor observed in the sanity run.
- Runner: `run_probe.py --width --dist --num-predict 2048 --timeout 600`
  with the earlier `thinking + response` concat patch preserved.
- 30 seeds per config × 9 files = 270 tasks total; ~3h 24m wallclock.
- systemd-inhibit held sleep off for the duration.

## Results

```
Width sweep (n=30 each):
   N   F1    prec  recall  gap_w
   4   0.27  0.20  0.50       8
   8   0.07  0.05  0.40      14
  16   0.01  0.01  0.17      27
  32   0.02  0.03  0.10      57
  64   0.00  0.00  0.00     107

Distance sweep (N=8 fixed, n=30 each):
  gap   F1    prec  recall  gap_measured
   50   0.09  0.08  0.25       60
  200   0.01  0.01  0.02      229
  500   0.02  0.02  0.03      472
 1000   0.00  0.00  0.00     1000
```

`acc_exact = 0` everywhere. `summarise.py` still verdicts "floor
effect" against exact-match; the real signal is in recall.

## Verdict — both width AND decay are real

The n=3 sanity was inconclusive because dist at N=16 was already at
task-comprehension floor. Dropping dist to N=8 unblocks the axis.

- **Width recall** (at their natural growing gap): 0.50 → 0.40 → 0.17
  → 0.10 → 0.00 across N=4→64.
- **Distance recall** (at fixed N=8, gap growing): 0.40 (width baseline
  gap≈14) → **0.25 at gap≈60** → 0.02 at gap≈229 → 0.03 at gap≈472 →
  0.00 at gap≈1000.

Head-to-head at N=8:
| gap_words | recall |
|-----------|--------|
| 14        | 0.40 (width-N=8 baseline) |
| 60        | 0.25 |
| 229       | 0.02 |
| 1000      | 0.00 |

Distance alone crushes recall by a factor of 20 (0.40 → 0.02) between
gap 14 and gap 229 at unchanged N. Width alone crushes recall by a
factor of 3 (0.50 → 0.17) between N=4 and N=16 at comparable gaps.
**Both bottlenecks are present.** This is the "inconclusive — both
fall" branch of the H12a falsification tree.

## Failure-mode breakdown (N=8, n=30)

Ran a small classifier over the raw responses. Of 30 tasks:

- **code-mode**: 13 (43 %) — model writes Python scaffolding
  (`pair = (item['x'], item['y'])`) instead of listing pairs.
- **empty prediction**: 12 (40 %) — CoT emits no item IDs at all in
  either `thinking` or `response`.
- **partial pairs**: 3 (10 %) — some correct IDs surface.
- **wrong pairs**: 2 (7 %) — IDs present but paired positionally
  instead of by shared property.
- **exact**: 0.

The code-mode and empty-prediction failure modes together account for
83 % of losses at N=8. Neither is architectural — they are
**output-format failures** the SFT recipe should be able to correct.

## What this means for H12b gate

The formal H12b gate in HYPOTHESES.md §H12 requires:

> Accuracy falls sharply in the width sweep but stays flat in the
> distance sweep ⇒ width bottleneck ⇒ H12b (multi-slot LoRA) is worth
> building.

**Gate not passed.** The distance sweep at N=8 is not flat — recall
drops from 0.40 to 0.00 across gap. H12b would attack the width axis,
but the distance axis also carries real signal, so the "multi-slot"
fix would leave decay on the table.

Two things could change the verdict:

1. **Persona-SFT (H15) first.** The 43 % code-mode + 40 % empty
   failure at N=8 is exactly the "dry butler register" gap that H15
   targets. A model that reliably emits "item-X, item-Y\n" pairs
   would reveal the underlying working-memory signal, not this noisy
   surface.
2. **Bigger backbone (G1d 1.5B / 2.9B).** Recall floor at N=8 with
   gap≈14 is already only 0.40 — pushing to 1.5B may lift baseline
   enough that the width axis separates cleanly.

Both are cheap next steps. H12b (LoRA training, cloud burst) is not
justified by this data.

## Follow-ups actively planned

1. **Adaptive-budget variant** (in flight next): rerun with
   `--num-predict-per-n 128` — smaller N gets less rope (may reduce
   code-mode), larger N gets more test-time compute (may induce
   damped-search degradation rather than divergence). Tests the H10
   axis inside the H12a task.
2. **G1d 1.5B same sweep** (task #19). Bigger comprehension ceiling.
3. **Persona-SFT filter design** (H15 — see HYPOTHESES.md §H15). Fixes
   the underlying output-format failure mode this run exposed.
4. **H12b deferred** (task #20). Gate not passed on current evidence.

## Files

- `N4.json` … `N64.json` — width sweep raw + aggregate.
- `dist-50.json` … `dist-1000.json` — distance sweep at N=8 fixed.
- `SUMMARY.txt` — `summarise.py` output (verdict there is stale
  against exact-match; see recall table above).
- `REPORT.md` — this file.
