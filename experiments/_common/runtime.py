"""Process-level hygiene shared by the CPU toy probes: thread budget and
output flushing. Both exist because of one concrete incident.

2026-09-14: five probes were launched in parallel on a 12-core machine. Torch
sizes its intra-op thread pool from the core count, so each process spawned 21
threads — 84 runnable threads on 12 cores. Everything slowed roughly 2x from
contention alone: a run that takes ~1h alone was still going at 1h39m. The
machine was not short of cores; the processes were fighting over them.

The same incident exposed the second problem. Each job's stdout went through a
pipe (`... | tail -30`), Python block-buffers a pipe, and the probes print one
line per run — so nothing appeared at all until the process exited. From the
outside a healthy 90-minute run and a hung one look identical, which is exactly
the situation this project already has a rule about (kill a run at the
swap-thrash signature, not at a guess).

Neither of these is a probe's scientific content, which is why they live here
rather than being re-solved in each file.
"""
from __future__ import annotations

import os
import sys


def limit_threads(n: int | None = None) -> int:
    """Cap torch's intra-op threads for this process and return the value set.

    Resolution order: explicit `n`, then `$NOESIS_PROBE_THREADS`, then 4 —
    enough that a single probe run is not artificially slow, small enough that
    three or four concurrent probes still fit on a typical desktop core count.
    Call this BEFORE the first tensor op.

    Note this does not reach OpenMP's own pool if it has already been
    initialised, so a caller that wants a hard guarantee should also export
    `OMP_NUM_THREADS` in the environment before launching python. Pinning with
    `taskset -acp <cores> <pid>` is the fallback for a process already running.
    """
    if n is None:
        n = int(os.environ.get("NOESIS_PROBE_THREADS", "4"))
    n = max(1, n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:                       # torch absent or already fixed
        pass
    return n


def progress(*args, **kwargs) -> None:
    """`print` that always flushes, for the one-line-per-run progress output.

    Probes are routinely run in the background with stdout on a pipe, where
    Python buffers ~8KB — long enough that a 40-run battery shows nothing until
    it finishes. Progress lines are low-volume by construction (one per run),
    so flushing each is free.
    """
    kwargs.setdefault("flush", True)
    kwargs.setdefault("file", sys.stdout)
    print(*args, **kwargs)


__all__ = ["limit_threads", "progress"]
