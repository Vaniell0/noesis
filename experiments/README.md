# experiments/

Measurements and evaluations against RWKV-7 checkpoints (state-geometry
probes, causal-intervention tests, RL training) — the empirical half of
noesis, feeding `hypotheses/README.md`. This file is the map; each subdirectory
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

*Updated 2026-11-11. Every result here feeds a claim in `hypotheses/README.md`; the claim's file says
which estimator a number is and what would refute it.*

**Gate battery for the latent loop** (`python experiments/run.py --backend peft [--weights CKPT] --tests ...`,
one model load; `--list` for descriptions): `m0_pass` (answers at `M = 0` and at `M`, fact classes with
`--m0-rank` / `--m0-samples`; the H7 and H19 instrument), `latent_geometry` (per-tick direction, fixed
point, cross-prompt spread), `latent_noise` (rollout disagreement under latent noise, the RL window),
`known_heads` (zeroing known heads), `chat` (everyday questions EN/RU, what each tick decodes to),
`compress` (bits per byte of fresh text with ticks), `backpatch` (two-hop answers after patching a later
layer into an earlier one; HF models). Stages and gates: `hypotheses/HZLK.md` §6.

**State-geometry probes** (`--backend blink`, CPU): `ipc`, `mlp_ipc`, `rlens`, `jlens`, `rich`,
`think_geometry`, `ib_probe`. Live in `A0_state_probe/` and `ib_probe/`. Readout lens and capacity:
`rl/state_readout_lens.py`, `rl/state_capacity_probe.py` (`hypotheses/H25.md` §5).

**Trainers.**
- `rl/train_latent_s1.py` — stage S1: the answer through `M` ticks of the model's own latent, KL to the
  model's own `M = 0` answers (`--loss kl|ce --temp`), full fine-tune. `rl/hf_loader.py` is the
  transformers bridge (Qwen3.5).
- `rl/train_staged_recall.py` — P1 staged recall (a measured result in `HZLK.md` §6, kept in the S0 mix).
- Retired, kept runnable: `rl/train_think_distill.py` (ThinkChain markers, replaced 2026-10-22) and
  `rl/train_wkv_loop.py` (self-feed GRPO, paused 2026-08-19; `--no-update` still measures rollouts).

**Toys** (CPU, minutes): `A0_state_probe/micro_wkv.py` — the single-head cell; `--verify` runs five
checks including the additivity and silent-step identities behind the open-loop lemma
(`docs/state-and-reasoning.md` §3.2). Its controls never read the state, so it cannot show computation
that needs the closed loop. Also `halting_chain.py` / `halting_rl.py` (state-dependent stopping),
`attractor_depth_probe.py` (fine-tuning and concentration), the optimizer probes of `hypotheses/H26.md`.

**Standalone but current:**
- `A0_state_probe/a05_run.py` + `a05_analyze.py` — causal state intervention (zero / shuffle / corrupt
  the state, measure the effect on the output).
- `A0_state_probe/lora_rank_analysis.py` — SVD of a base/trained weight delta. Diff against the *actual*
  training base.
- `A0_eval/eval.py` — the 48-task rubric harness; its `N` / `K` / readout-mode axes belong to a retired
  design (`docs/effort-frontier.md` §1).

## What's parked, and why (full audit 2026-08-18)

Cross-referenced every directory against `hypotheses/README.md`/`ROADMAP.md`
— nothing here is unaccounted-for, but not everything is
live:

- **Correctly refuted, not broken:** `A0_portability` (H8 cross-model
  transfer, 0/6 pass, `A0_portability/results_a06_summary.md`), `aporia_probe`
  (H20 pilot ordering claim refuted, substrate mechanism itself not
  refuted).
- **Blocked/deferred by design:** `A0_baseline` (2.9B eval needs GPU),
  `rosa_probe` (closed 2026-08-08, needs a full RWKV-8 forward pass).
- **Superseded:** `A0_3_sustained_idle` — tested H1, which was retracted
  2026-07-25; its successor is ROADMAP.md's "A0.3 24h runtime polygon."
- **Out of scope, but not just infra:** `byte_adapter/` — mostly
  training infrastructure (adapter never wired into an optimizer, see
  Known gaps below), but `eval_byte_wordsearch.py` did produce a real,
  closed eval result (0/56 = 0%, G1d 0.4B, random-init adapter — cited
  in `docs/rl-track.md`'s Known risks #5) — a filed result, not pure infra.
- **`A0_H12a_working_memory`, `A0_H12b_multislot`** — the width and decay sweeps of H12. Recall falls with
  distance and drops between 8 and 16 items, but exact match is 0 in every cell, so the sweeps are at the
  task's floor (`hypotheses/H12.md` §3). The multi-slot state they were built for is retired.
- **Missing from this list until 2026-08-23: `A0.8_refine`** —
  `generate_cot_corpus.py` is the explicitly-cited ancestor of
  `experiments/rl/gen_selfcot_fast.py` (same docstring), feeding the
  ThinkChain training corpus in `docs/rl-track.md`.
- **Real remaining standardization debt, deprioritized 2026-08-18:**
  `attribution_probe` (H22), `premise_validator` (H21), `H18_merge`
  (H18) — all still have the loader sys.path hack + raw `json.dump`
  pattern fixed everywhere else. Known, not urgent per current priority.
- **Data generators** (`gen_*.py`, `mine_*.py`, `build_ctx_items.py` —
  11 scripts across several directories as of `gen_selfcot_fast.py`,
  2026-08-19; was 9 at this section's original 2026-08-18 count) — a
  distinct category from probes; a possible future second framework
  module, not started.

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

## Known gaps (found during the 2026-08-18 GPU-VM session, not yet fixed)

- **No schema version on probe output formats.** `a05_run.py`'s
  `summary.json`/`seed_N.json` shape changed (old: `delta_pooled`/
  `curvature_pooled`/`sr_step_mean`; current: `checkpoints`) with no
  version marker anywhere — `a05_analyze.py` just crashes with a raw
  `KeyError` on an old-format cell instead of a clear "this cell predates
  format vX" message. Same risk applies to any other probe whose JSON
  shape might drift over time; nothing here currently guards against it.
- **`_TokenizerAdapter` has two incompatible definitions.** The one in
  `experiments/rl/loader.py` has a real `.encode()`; the one in
  `experiments/_common/model.py` only implements `__call__`/`.decode()`
  (HF-style). Not a bug today (each is only ever used through its own
  loader), but a trap for anyone assuming the two are interchangeable.
- ~~FORGE's `wrap_rwkv7()` has no exclude-list~~ **RESOLVED 2026-08-18,
  same day, via a different mechanism than an exclude-list.** The
  fused-into-backward approach (`wrap_rwkv7_excluding_head`) turned out
  structurally incompatible with the WKV-loop's BPTT regardless of which
  layers get excluded — real fix is `experiments/rl/loader.py::Int8AdamW`
  (don't fuse into backward at all, run ordinary `backward()`, apply
  FORGE's standalone `optimizer_only_adamw_int8state()` afterward). See
  `docs/rl-track.md`'s Known risks #6 for the full writeup;
  `wrap_rwkv7_excluding_head` is kept as a documented dead end.
- **`ib_probe/run.py`'s nonlinear (`--nonlinear`) decoder needed real
  hyperparameter tuning, not just wiring — and the first attempted fix
  was wrong.** Reusing the linear decoder's `epochs=200`/`lr=1e-3` gave
  held-out CE worse than the unigram baseline, diverging at longer lags.
  Bumping epochs 10x and lowering lr (this section's claim until
  2026-08-23) made it *worse* — more time to overfit — and was reverted.
  The actual fix: `weight_decay=0.05` on the nonlinear path's optimizer
  (`experiments/ib_probe/run.py::probe_lag`), epochs/lr left at the
  linear decoder's defaults. Re-verify against a second corpus before
  trusting the numbers either way.
- **No system-load-aware job scheduling for the GPU VM.** Every
  multi-job GPU session tonight was manual (watch `nvidia-smi`/`uptime`,
  decide by eye when to launch the next probe). Fine for a few jobs,
  won't scale to the full execution matrix across more models — worth a
  small script that queues jobs against a VRAM/CPU-load budget instead
  of eyeballing it.
