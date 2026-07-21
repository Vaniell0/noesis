"""Sanity check for metrics.py using synthetic trajectories.

Fabricates three-state sequences with known dynamics and verifies:
- Zero curvature on a linear trajectory.
- Positive curvature on a bending trajectory.
- Stable rank = 1 on a rank-1 matrix, = h on the identity.

Run:
    .venv/bin/python _sanity.py
"""
import torch
from metrics import delta_norm, curvature, stable_rank


def _fp32(*shape):
    return torch.zeros(*shape, dtype=torch.float32)


def test_delta_and_curvature_linear():
    n_head, h = 4, 8
    s0 = _fp32(n_head, h, h)
    s1 = s0 + 1e-3 * torch.eye(h).unsqueeze(0).expand(n_head, h, h)
    s2 = s0 + 2e-3 * torch.eye(h).unsqueeze(0).expand(n_head, h, h)

    d01, _ = delta_norm([s0], [s1])
    d12, _ = delta_norm([s1], [s2])
    c012, _ = curvature([s0], [s1], [s2])

    print(f"[linear] delta 0→1 = {d01:.6g}")
    print(f"[linear] delta 1→2 = {d12:.6g}  (should match delta 0→1)")
    print(f"[linear] curvature 0,1,2 = {c012:.6g}  (should be ≈ 0)")
    assert abs(d01 - d12) < 1e-6, "linear delta should be constant"
    assert c012 < 1e-6, "linear trajectory should have zero curvature"


def test_curvature_bending():
    n_head, h = 4, 8
    s0 = _fp32(n_head, h, h)
    s1 = s0 + 1e-3 * torch.eye(h).unsqueeze(0).expand(n_head, h, h)
    s2 = s1  # trajectory stops
    c012, _ = curvature([s0], [s1], [s2])
    print(f"[bent] curvature (accel = -delta) = {c012:.6g}  (should be > 0)")
    assert c012 > 1e-6, "bent trajectory should have non-zero curvature"


def test_stable_rank_rank1():
    n_head, h = 1, 4
    v = torch.arange(1, h + 1, dtype=torch.float32)
    A = torch.outer(v, v).unsqueeze(0)  # rank 1
    sr = stable_rank([A])[0]
    print(f"[rank1] SR = {sr[0]:.6g}  (should be ≈ 1)")
    assert abs(sr[0] - 1.0) < 1e-4, f"rank-1 SR should be 1, got {sr[0]}"


def test_stable_rank_identity():
    n_head, h = 1, 4
    I = torch.eye(h, dtype=torch.float32).unsqueeze(0)
    sr = stable_rank([I])[0]
    print(f"[eye]   SR = {sr[0]:.6g}  (should be = h = {h})")
    assert abs(sr[0] - float(h)) < 1e-4, f"identity SR should be {h}, got {sr[0]}"


if __name__ == "__main__":
    test_delta_and_curvature_linear()
    test_curvature_bending()
    test_stable_rank_rank1()
    test_stable_rank_identity()
    print("all sanity checks passed")
