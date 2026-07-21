# A0.4 — state-utilisation probe

**Status: skeleton — not yet run.** This directory currently contains a
scaffolded probe with unimplemented function bodies. Execution is
scheduled for a dedicated session. Do not read results here — there
are none.

## Purpose

Directly test whether RWKV-7's WKV state does *computational* work
during autoregressive generation, or only *memory* work — and whether
the G1-line reasoning training measurably changes state dynamics
compared to the base World3 checkpoint.

The A0.1 throughput baseline said nothing about this. It answered
"can noesis run cheaply?" (yes) and "does G1h decode slower than
World?" (no — identical tok/s). It did *not* answer "does the state
evolution mean anything?" — which is the philosophical wager
underneath the RWKV backbone choice (see `docs/state-and-reasoning.md`
for the paper's own framing of the WKV update as an SGD step at test
time).

## Hypotheses under test

- **H8** — state-as-computation in RWKV-7 (`HYPOTHESES.md`).
- **H9** — G1-line training amplifies state utilisation
  (`HYPOTHESES.md`).

Both prediction / falsification thresholds are placeholders in
`HYPOTHESES.md` and are locked at the *start* of the execution
session, after a pilot run establishes the noise floor. Do not lock
thresholds by reading this README.

## Design

### Metrics — three disjoint measures of state activity

Chosen per P8 Goodhart-mitigation (no single metric decides). All
three are computed from the per-layer WKV state tensors extracted via
torch hooks on each RWKV-7 block forward call.

- **`delta_norm(s_t, s_{t-1})`** — L2 norm of the state's change
  between consecutive tokens, per layer and pooled. "How much did
  the state move?"
- **`curvature(s_{t-2}, s_{t-1}, s_t)`** — L2 norm of the second
  difference. "Is the trajectory a straight line (memory update) or
  a curve (computation)?"
- **`stable_rank(s_t) = (‖s_t‖_F / ‖s_t‖_2)^2`** — effective rank of
  each WKV head's state matrix. Matches the SR metric authors use in
  RWKV-7 paper Appendix J (State Inspections); direct comparability
  to their published probing is worth more than an ad-hoc entropy
  measure.

### Experimental cells

Four cells: `{World3, G1h} × {medium, narrative}`.

- `medium` — event-stream digest, reasoning-flavoured (from
  `../A0_baseline/prompts.py`).
- `narrative` — descriptive prose of matched length, no reasoning
  demand (from the same file, added for this experiment).

Per cell: 10 seeds, 256 tokens generated, all three metrics captured
per generated token per layer.

### Statistics

- Test H8 by comparing metric distributions between reasoning
  (`medium`) and non-reasoning (`narrative`) trajectories on a fixed
  model — Welch's t-test with Bonferroni correction across three
  metrics.
- Test H9 by comparing metric distributions between World3 and G1h on
  the reasoning prompt — same test, same correction.
- Report effect sizes and 95 % CIs alongside p-values; small p on a
  tiny effect does not clear the bar. Noise floor pre-registered
  before conclusions.

## Stack

- Python 3, PyTorch, HuggingFace `transformers`.
- Weights: BlinkDL RWKV-7 World3 2.9B (bf16 native, from HF
  `RWKV/` org). G1h availability is currently unresolved — see
  `docs/state-and-reasoning.md` §3 and the open question in the plan.
  If native G1h weights are not obtainable, drop to a single-model
  H8-only run and document the omission.
- **No GGUF, no Ollama.** Q4 quantisation distorts state dynamics
  enough to confound H8/H9. Bf16 weights + fp32 WKV accumulator, per
  paper §8.

## How to run (once implemented)

```bash
# Pilot: one seed, one prompt, verify plumbing.
python3 run.py --model RWKV/rwkv-7-world --prompt medium --seeds 1 --out results/pilot/

# Full sweep (planned).
python3 run.py --model RWKV/rwkv-7-world     --prompt medium    --seeds 10 --out results/world_medium/
python3 run.py --model RWKV/rwkv-7-world     --prompt narrative --seeds 10 --out results/world_narrative/
python3 run.py --model RWKV/rwkv-7-g1h-2.9b  --prompt medium    --seeds 10 --out results/g1h_medium/
python3 run.py --model RWKV/rwkv-7-g1h-2.9b  --prompt narrative --seeds 10 --out results/g1h_narrative/
```

Currently `run.py` prints a "not yet implemented" notice and exits 0
— the CLI shape is fixed so the execution session begins with the
plumbing already in place.

## Files

- `probe.py` — state extraction: model loading + torch hooks on RWKV
  blocks. Stubs, `NotImplementedError`.
- `metrics.py` — the three metric functions. Stubs with math in
  docstrings, `NotImplementedError`.
- `run.py` — CLI wrapper. Argparse works; body is a stub.
- `prompts.py` — thin shim that re-exports `ALL`, `SHORT`, `MEDIUM`,
  `LONG`, `NARRATIVE`, `word_count` from `../A0_baseline/prompts.py`.

## Follow-ups (out of scope for skeleton)

- Verify G1h HF weight availability before the execution session.
- Threshold-lock for H8 / H9 after pilot noise-floor measurement.
- Decision on adding a fourth metric (paper Appendix J's RMS) if SR
  alone doesn't discriminate.
- Gate 1 exit-criteria discussion once H8/H9 numbers are in.
- **Server unavailable for the execution session** — plan for laptop
  CPU only. HF weight download and probe run both happen locally;
  budget the download time (RWKV/rwkv-7-world ≈ 5.8 GB bf16).

## Related

- `../A0_baseline/` — throughput baseline (done, one commit).
- `../../docs/state-and-reasoning.md` — literature notes underpinning
  metric choice and stack decision.
- `../../HYPOTHESES.md` — H8, H9 (placeholders until execution
  session pilots).
- `../../ROADMAP.md` — Track A, A0.4 slot; Gate 1 exit criteria.
