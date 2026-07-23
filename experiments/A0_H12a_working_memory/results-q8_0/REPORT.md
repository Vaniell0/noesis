# A0.H12a — Q8_0 sanity run report (2026-07-23)

## Setup

- Model: `noesis-model-rwkv7-world-0.4b` at Q8_0 (591 MB `.bin`), built via
  the new two-stage nix path (`nix/noesis-model.nix` → FP16 intermediate
  → `rwkv-quantize` from `rwkv-cpp` extras).
- Backend: `noesis-runtime` supervisor with `inference_backend = "rwkv-cpp"`
  and `rwkv_cpp.http_bind = "127.0.0.1:11435"` (Ollama-shape shim).
- Runner: `run_probe.py --host http://127.0.0.1:11435
  --model noesis-rwkv7-0.4b --width --dist --num-predict 512 --timeout 600`.
- Config file: `/tmp/noesis-h12a/config.toml`, 4 threads, heartbeat
  deferred (`heartbeat_secs = 3600`) so it never contended with the sweep.
- 3 tasks per config (design default). Real full-scale sweep will need
  n≥30 to be worth interpreting.

## Results

```
Width sweep (N ↑, mean_word_gap ~ N):
   N   F1    prec  recall  gap_w
   4   0.50  0.33  1.00       10
   8   0.13  0.07  1.00       13
  16   0.03  0.02  1.00       22
  32   0.01  0.00  0.67       72
  64   0.00  0.00  0.17       90

Distance sweep (target gap ↑, N=16 fixed):
  gap   F1    prec  recall  gap_measured
   50   0.15  0.10  0.33          40
  200   0.00  0.00  0.00         314
  500   0.00  0.00  0.17         498
 1000   0.00  0.00  0.17         860
```

`acc_exact` is 0 for every config — no configuration ever produced a
strictly correct pair set.

## What this sanity run establishes

1. **Runner + HTTP shim pipeline is end-to-end functional.** 27 requests
   went out via `/api/generate`, 27 returned scored JSON, all files landed
   in `results-q8_0/`.
2. **Q8_0 model loads and decodes.** Load `506 ms`; steady-state ≈ 30
   tok/s on 4 Alder Lake threads.
3. **Supervisor is stable.** RSS held 1.16 GB ± 15 MB across 100 s of
   continuous eval (see `#15` note in the session log). No leak.

## What this run does *not* decide

The `summarise.py` verdict is **"flat but at low accuracy → floor
effect."** This is the correct read for `acc_exact` (0.00 everywhere),
but the real information is in F1:

- Width F1 falls sharply (0.50 → 0.00, Δ = 0.50).
- Distance F1 also falls (0.15 → 0.00, Δ = 0.15) but from a much lower
  ceiling — dist-50 is already worse than N=4 despite the same lookup
  structure. That's not a clean bottleneck signal, it's the model
  failing before the sweep axis even bites.

The root cause is model choice: `RWKV-7-World-0.4B` is **base** (no
instruction tuning). Inspecting responses in `N4.json` shows the model
enumerates all 4 IDs on a single line and then wanders off into task
description or unrelated text ("Detailed Instructions: convert to
Hindi…"). The pair extractor then generates all `C(N,2)` pairs, which is
why precision decays as `2/(N(N−1))` — a mathematical artifact of the
flat-enumeration failure mode, not a working-memory diagnosis.

## Follow-ups (not this run)

1. **Rerun against G1d-0.4B** — the original design target
   (`mollysama/rwkv-7-g1d:0.4b` via Ollama, or an in-process G1d weight
   swap once it exists in `noesis-model.nix`). G1d has SFT on
   instruction-following, so it can actually attempt the task and the
   sweep signal becomes interpretable.
2. **Raise task count to n ≥ 30 per config** before treating any drop as
   robust; the current n = 3 is only good enough for pipeline sanity.
3. **Split the F1 verdict path in `summarise.py`** — add a
   `--metric f1|accuracy_exact` flag so the tree can classify when
   `acc_exact` is at the floor.

## Files

- `N4.json`, `N8.json`, `N16.json`, `N32.json`, `N64.json` — width sweep
  raw + aggregate.
- `dist-50.json`, `dist-200.json`, `dist-500.json`, `dist-1000.json` —
  distance sweep raw + aggregate.
- `SUMMARY.txt` — `summarise.py` output.
- `REPORT.md` — this file.
