"""One definition of "this run actually learned the task", shared by every
toy probe, because getting it wrong silently corrupted three separate results
in a single day (2026-09-14).

The failure is always the same shape and is never visible in the headline
number. A toy arm that diverges still produces a full set of perfectly
well-formed statistics — a state spectrum, an update norm, a correlation, an
ablation score — and those statistics go into a mean alongside the runs that
worked. What comes out looks like a measurement and is not one.

The three incidents, kept here rather than in a commit message because the
point is that this is a *recurring* mistake, not three unrelated slips:

1. `breadth_growth_probe.py` — plain Adam at T=16 converged on 1 seed of 3;
   the other two sat at `id_r2 ~ 0` with exactly `head_size/2` live
   directions. The reported mean of 6.59 live directions was two dead runs
   plus one real one at 3.76, and it was written into H26 as "plain training
   reaches 41-46% of reachable rank".

2. `lora_muon_probe.py` — 2 of 5 *pretrained bases* never learned the
   pretrain task. Every arm shared the cached base per seed, so all five arm
   means were dragged to ~0.60 and the comparison had to be recomputed by
   hand. A base that never learned the task cannot answer "did the finetune
   destroy what it knew".

3. `step_geometry_probe.py` — a T-sweep appeared to show Adam's correlation
   between state displacement and fixed weight norms RISING with sequence
   length (0.546 -> 0.585 -> 0.828), which was reported as evidence bearing
   on H26's compounding criterion. Adam's `id_r2` at those points was 0.9995,
   0.9991 and **0.334**. On the converged points the trend was flat
   (0.683 -> 0.629). The apparent trend was the divergence itself.

A diverging run's statistics are not noise to be averaged away — they are
systematically different from a solution's, so including them biases rather
than widens. The rule is therefore to exclude, and to say how many were
excluded, never to quietly average.

Related guard, different failure: `step_geometry_probe.MIN_POINTS_FOR_CORR`
refuses a correlation computed from too few measurement points.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeVar

# In-distribution R^2 below which a toy run is treated as not having learned
# the task at all. These substrates reach 0.99+ when they work and ~0.0 when
# they do not, so anything in between is already pathological; 0.5 sits in the
# empty middle rather than being a tuned threshold.
CONVERGED_ID_R2 = 0.5

T = TypeVar("T")


def is_converged(run: dict, key: str = "id_r2",
                 threshold: float = CONVERGED_ID_R2) -> bool:
    """True if `run` learned the task. NaN counts as not converged."""
    v = run.get(key)
    if v is None or v != v:          # missing or NaN
        return False
    return float(v) > threshold


def split_converged(runs: Iterable[dict], key: str = "id_r2",
                    threshold: float = CONVERGED_ID_R2
                    ) -> tuple[list[dict], list[dict]]:
    """(converged, dead). Callers should aggregate over the first and REPORT
    the size of the second — a result whose dead fraction is not stated cannot
    be read correctly later."""
    ok, dead = [], []
    for r in runs:
        (ok if is_converged(r, key, threshold) else dead).append(r)
    return ok, dead


def converged_note(ok: Sequence[T], dead: Sequence[T]) -> str:
    """One-line provenance string for a summary or verdict. Empty when nothing
    was excluded, so it adds no noise to a clean run."""
    if not dead:
        return ""
    total = len(ok) + len(dead)
    if not ok:
        return (f"[!] NOT EVALUABLE — 0/{total} runs learned the task; every "
                f"statistic here describes diverging runs and must not be "
                f"scored as a pass or a failure")
    return f"[{len(ok)}/{total} runs converged; {len(dead)} excluded]"


def mean_over_converged(runs: Iterable[dict], key: str,
                        conv_key: str = "id_r2",
                        threshold: float = CONVERGED_ID_R2) -> float:
    ok, _ = split_converged(runs, conv_key, threshold)
    vals = [r[key] for r in ok if key in r and r[key] == r[key]]
    return (sum(vals) / len(vals)) if vals else float("nan")


__all__ = [
    "CONVERGED_ID_R2",
    "is_converged",
    "split_converged",
    "converged_note",
    "mean_over_converged",
]
