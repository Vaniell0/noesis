# experiments/

Measurements and evaluations against RWKV-7 checkpoints (state-geometry
probes, causal-intervention tests, RL training) — the empirical half of
noesis, feeding `HYPOTHESES.md`. This file is the map; each subdirectory
that needs more detail has its own `README.md`.

## Quick start

```bash
# What's available?
python experiments/run.py --list

# Run the fast probe battery on one model (loads it once, shares across probes)
python experiments/run.py --model <checkpoint.pth> --device cpu \
  --tests ipc,mlp_ipc,rlens,jlens,rich,think_geometry --n-tokens 768 \
  --out-dir experiments/_common/results/<label>

# ...or the whole thing (battery + ib_probe) as one command:
experiments/run_model_battery.sh --model <checkpoint.pth> --label <name> --device cpu

# Check on a long-running job without SSHing into wherever it's running:
python experiments/monitor.py <out_dir>/status.json --watch
# or serve it over HTTP for a browser view (see run_model_battery.sh --status-port)

# After any batch of runs, refresh the results index:
python experiments/regenerate_results.py
```

`experiments/RESULTS.md` is the single flat index of numeric results —
check there before re-running something to see if it's already been
measured. `experiments/_common/README.md` is the framework-internals
guide (how to add a new probe, what each `_common/` module does).

## What's active

**Registered probes** (`python experiments/run.py --list` for the live
list, descriptions included) — all measure WKV-state geometry, all
share one model load, all write `_meta`-stamped results:
`ipc`, `mlp_ipc`, `rlens`, `jlens`, `rich`, `think_geometry`, `ib_probe`.
Live in `A0_state_probe/` and `ib_probe/`.

**Standalone but current** (not in the shared battery — different shape,
usually needs a per-run choice like which prompt or which base checkpoint
to diff against):
- `A0_state_probe/a05_run.py` + `a05_analyze.py` — causal state
  intervention (zero/shuffle/corrupt WKV, measure the effect) — the
  *causal*, not decodability, test. See the plan/HYPOTHESES.md H8 section
  for why this matters more than another probe when a decodability
  result is ambiguous.
- `A0_state_probe/lora_rank_analysis.py` — SVD of a base/trained weight
  delta. Diff against the *actual* training base (check lineage — a
  fine-tune's base isn't always the model you'd guess).
- `A0_eval/eval.py` — the A0.2 rubric harness (ollama or direct-rwkv
  backend), including the N/K/readout_mode axes (see its own docstring
  for the state_readout mechanism).
- `A0_H12b_multislot/run_probe.py` — multi-slot K×P accuracy sweep, needs
  its own `gen_multislot.py`-generated probe file.

**RL stack** — `rl/train_wkv_loop.py`, the WKV-loop GRPO trainer.
`--no-update` runs rollouts + reward without a gradient step (CPU-safe
smoke test / M-distribution measurement, no GPU needed). Real training
needs the peft backend (`--device cuda`). Supports `--resume <ckpt>` to
continue after an interruption — matters on a preemptible instance, which
can be reclaimed without much warning. Writes a heartbeat every step.

## What's parked, and why (full audit 2026-08-18)

Cross-referenced every directory against `HYPOTHESES.md`/`ROADMAP.md`/
`FAILED.md` — nothing here is unaccounted-for, but not everything is
live:

- **Correctly refuted, not broken:** `A0_portability` (H8 cross-model
  transfer, 0/6 pass, properly filed in `FAILED.md`), `aporia_probe`
  (H20 pilot ordering claim refuted, substrate mechanism itself not
  refuted).
- **Blocked/deferred by design:** `A0_baseline` (2.9B eval needs GPU),
  `rosa_probe` (closed 2026-08-08, needs a full RWKV-8 forward pass).
- **Superseded:** `A0_3_sustained_idle` — tested H1, which was retracted
  2026-07-25; its successor is ROADMAP.md's "A0.3 24h runtime polygon."
- **Out of scope:** `byte_adapter/` — training infrastructure, not
  test/eval code.
- **Real remaining standardization debt, deprioritized 2026-08-18:**
  `attribution_probe` (H22), `premise_validator` (H21), `H18_merge`
  (H18) — all still have the loader sys.path hack + raw `json.dump`
  pattern fixed everywhere else. Known, not urgent per current priority.
- **Data generators** (`gen_*.py`, `mine_*.py`, `build_ctx_items.py` —
  9 scripts across several directories) — a distinct category from
  probes; a possible future second framework module, not started.

## Conventions

- CPU inference: `experiments._common.model.load_model`. GPU/training:
  `experiments.rl.loader.load_rwkv7` — different job, don't conflate.
- Layer defaults: `experiments._common.layers.default_layers(n_layer)`,
  not a hardcoded list — every probe with a hardcoded layer default has
  broken on a checkpoint whose depth didn't match the default's
  assumption at some point.
- Every script here should work both as `python path/to/script.py` (bare)
  and as `experiments.X.Y` module import (what `run.py` needs) — see
  `_common/README.md`'s sys.path bootstrap snippet.
