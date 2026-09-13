"""Guards for jlens_probe.py's spectrum statistics.

The load-bearing test here is `test_stable_rank_hides_live_directions`. Every
state-rank reading this project has made rested on `stable_rank` ~1.2-1.3 being
read as "the state uses essentially one direction". That inference does not
follow: stable_rank is ‖A‖²_F/σ₁², an energy-concentration ratio, and a matrix
with one dominant singular value plus thirty-one genuinely non-zero ones scores
~1.03 on it. The test pins that down with a constructed spectrum, so nobody
(including a future me) re-derives the "near rank-1" conclusion from the same
statistic without noticing what it can and cannot distinguish.

The distribution helpers exist because the probe used to average forty heads
into one number and discard the rest — see `_analyze`'s comment.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experiments.A0_state_probe.jlens_probe import _dist, _svd_stats


def _with_spectrum(sv: torch.Tensor) -> torch.Tensor:
    """A square matrix whose singular values are exactly `sv`."""
    n = sv.numel()
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(0)))
    return q @ torch.diag(sv) @ q.T


def test_rank_one_reads_as_rank_one_everywhere() -> None:
    m = torch.randn(16, 1) @ torch.randn(1, 16)
    s = _svd_stats(m)
    assert math.isclose(s["stable_rank"], 1.0, rel_tol=1e-3)
    assert math.isclose(s["participation_ratio"], 1.0, rel_tol=1e-3)
    assert math.isclose(s["effective_rank_entropy"], 1.0, rel_tol=1e-2)
    assert s["numerical_rank_1pct"] == 1


def test_flat_spectrum_reads_as_full_rank_everywhere() -> None:
    n = 32
    s = _svd_stats(torch.eye(n))
    assert math.isclose(s["stable_rank"], n, rel_tol=1e-3)
    assert math.isclose(s["participation_ratio"], n, rel_tol=1e-3)
    assert math.isclose(s["effective_rank_entropy"], n, rel_tol=1e-2)
    assert s["numerical_rank_1pct"] == n


def test_stable_rank_hides_live_directions() -> None:
    """The whole reason the other three measures were added.

    One dominant direction (σ=100) plus 31 real ones (σ=3) — a state carrying
    32 usable directions — scores stable_rank ≈ 1.03, i.e. indistinguishable
    from genuine rank-1 by that statistic alone. The project's measured
    1.19-1.30 sits in exactly this regime, so "near rank-1" was never
    established by it.
    """
    sv = torch.cat([torch.tensor([100.0]), torch.full((31,), 3.0), torch.zeros(32)])
    s = _svd_stats(_with_spectrum(sv))

    assert s["stable_rank"] < 1.1, "stable_rank should look ~rank-1 here"
    assert s["numerical_rank_1pct"] == 32, "but 32 directions are above 1% of sigma1"
    assert s["effective_rank_entropy"] > 5.0, "and entropy rank sees several of them"
    assert s["participation_ratio"] > s["stable_rank"]


def test_dist_reports_spread_not_just_mean() -> None:
    """'37 heads at 1.0 plus 3 at 5.0' must be distinguishable from
    'all 40 at 1.3' — the thing the old mean-only record could not do."""
    concentrated = _dist([1.0] * 37 + [5.0] * 3)
    uniform = _dist([1.3] * 40)
    assert math.isclose(concentrated["median"], 1.0)
    assert math.isclose(concentrated["max"], 5.0)
    assert concentrated["std"] > uniform["std"]
    assert math.isclose(uniform["min"], uniform["max"])


def test_dist_empty_is_safe() -> None:
    assert _dist([]) == {}
