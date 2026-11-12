# Probe gap queue

Unfinished measurement cells, one job at a time, resumable. `jobs.json` is the
list; `queue_run.py` is the worker; `out` is the idempotency key (an existing
non-empty output means the job is done and is skipped).

## Why it exists

The 2026-10-25/26 queue closed the 0.4B gates and the G1k/G1l capacity cells, then
died inside the G1k lens. The remaining cells are independent measurements, not a
training run: capacity and readout lens on the three S1 checkpoints, lens on G1l,
geometry to 16 ticks on the three 2.9B bases (8 was the window edge), latent_noise
on 2.9B, useful motion on 2.9B, and step influence on G1i.

## Run

From a host shell (not the agent sandbox — the agent's processes die with its call):

    cd ~/Desktop/projects/noesis
    setsid nohup python3 experiments/queue/queue_run.py \
        >> experiments/_logs/queue/runner.log 2>&1 &

The runner finds the torch/rwkv/peft environment in `~/.cache/uv/archive-v0` by
itself (PYTHONPATH is set for each job), waits until nothing under
`experiments/rl/results` has changed for 30 min, then runs on E-cores 4-7.

Status: `experiments/queue/state/status.json`; per-job logs:
`experiments/_logs/queue/<id>.log`.

Stop after the current job: `touch experiments/queue/STOP`.
Resume: delete STOP and run the same line. Completed cells are skipped, so this is
safe after a crash.

Dry run / selection:

    python3 experiments/queue/queue_run.py --dry-run
    python3 experiments/queue/queue_run.py --only cap-ce,lens-g1l --no-idle-wait

## Not in the queue (needs code or a decision first)

- `tick_yield_probe` (M>1 criterion) — spec in
  `~/.claude/plans/noesis-tick-yield-probe-2026-10-26.md`, not written.
- capacity / lens after M ticks — depend on `tick_yield_probe`'s tick path.
- `known_heads` on G1k/G1l — the known-head list is G1i-specific; needs a fresh
  per-model profile before it means anything.
- `cot_vs_routes` on G1k/G1l — heavy (up to 768 tokens per item) and not a
  capability cell; run when the card is free.
- 2.9B S1 gates — no 2.9B S1 checkpoint exists yet.
