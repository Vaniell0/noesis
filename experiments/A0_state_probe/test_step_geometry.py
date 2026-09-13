"""Guards for step_geometry_probe.py's measurement primitives (H26).

Two of these exist because of a bug this file's subject actually had, not
because a test file ought to have tests in it:

- `MIN_POINTS_FOR_CORR` — the probe's first smoke run used 60 training steps
  with measure_every=20, i.e. 3 measurement points, and printed a confident
  "H26 strong-version premise REFUTED" off |r|=0.996. Three points are nearly
  collinear by chance; the number was an artifact, and it was one step away
  from being written into a hypothesis record as evidence. The guard is what
  stops that; this test is what stops the guard from being removed later.
- `_update_norms` / `_trace_displacement` — everything H26 claims rests on
  these two being the quantities they say they are, so they are checked
  against hand-computable cases rather than assumed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experiments.A0_state_probe.step_geometry_probe import (
    MIN_POINTS_FOR_CORR,
    _cv,
    _pearson,
    _trace_displacement,
    _update_norms,
)


def test_pearson_refuses_small_n() -> None:
    """The actual bug: too few points must yield NaN, not a confident r."""
    xs = [1.0, 2.0, 3.0]
    ys = [2.0, 4.0, 6.0]  # perfectly correlated, and meaningless at n=3
    assert math.isnan(_pearson(xs, ys))


def test_pearson_correct_above_threshold() -> None:
    n = MIN_POINTS_FOR_CORR + 5
    xs = [float(i) for i in range(n)]
    assert _pearson(xs, [2.0 * x + 1.0 for x in xs]) == 1.0
    assert _pearson(xs, [-3.0 * x for x in xs]) == -1.0
    assert math.isnan(_pearson(xs, [7.0] * n))  # zero variance -> undefined


def test_cv_shape() -> None:
    assert _cv([5.0] * 10) == 0.0          # constant series: no variation
    assert _cv([1.0, 3.0]) == 0.5          # std 1, mean 2
    assert math.isnan(_cv([]))


def test_update_norms_against_hand_computed_case() -> None:
    """Diagonal delta with known singular values: spectral = largest |entry|,
    Frobenius over the whole list = sqrt(sum of squares), inf = max |entry|."""
    before = [torch.zeros(3, 3), torch.zeros(2)]
    after = [torch.diag(torch.tensor([3.0, -4.0, 1.0])), torch.tensor([0.5, -2.0])]
    n = _update_norms(before, after)
    assert math.isclose(n["dW_spec_max"], 4.0, rel_tol=1e-5)
    assert math.isclose(n["dW_inf_max"], 4.0, rel_tol=1e-5)
    expected_fro = math.sqrt(9 + 16 + 1 + 0.25 + 4)
    assert math.isclose(n["dW_fro_total"], expected_fro, rel_tol=1e-5)
    # The 1-D tensor must not contribute a spectral norm — only matrices do.
    assert math.isclose(n["dW_spec_mean"], 4.0, rel_tol=1e-5)


def test_trace_displacement_zero_when_unchanged() -> None:
    trace = [torch.ones(4, 2, 2) * 3.0 for _ in range(5)]
    abs_d, rel_d = _trace_displacement(trace, [t.clone() for t in trace])
    assert abs_d == 0.0
    assert rel_d == 0.0


def test_trace_displacement_relative_normalisation() -> None:
    """Doubling every state should give relative displacement 1.0 — the point
    of the relative measure is that it does not grow just because the state's
    own magnitude grew."""
    before = [torch.ones(1, 2, 2) for _ in range(3)]
    after = [t * 2.0 for t in before]
    abs_d, rel_d = _trace_displacement(before, after)
    assert math.isclose(abs_d, 2.0, rel_tol=1e-5)   # ||[1,1,1,1]||_F = 2
    assert math.isclose(rel_d, 1.0, rel_tol=1e-5)
