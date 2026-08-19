"""Regression tests for compute_wkv_loop_rewards — same plain-assertion/
no-pytest convention as test_checkpoint.py / experiments/_common/test_core.py.

Covers `gate_on_correct` (added 2026-08-19): found via a real bisected
training collapse (g1i_real_run6 went from 100% commit-at-M=2 at step20 to
100% M_max-saturated boilerplate by step50) that unconditional beta*M/
gamma*entropy shaping dominates the GRPO gradient whenever a whole group
is equally wrong — gating shaping on r_correct>0 means a wrong-but-brief
rollout can never outscore a wrong-but-verbose one, only correct rollouts
get ranked by effort. Verified by hand with an ad-hoc python -c snippet
before shipping to a live run; this persists that check as an actual test.

Run: `python experiments/rl/test_rewards.py`
(CPU-only, no model — WKVLoopRollout built directly with fixed fields.)
"""
from __future__ import annotations

import pathlib
import sys

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.rl.rewards import compute_wkv_loop_rewards, _score_correct
from experiments.rl.wkv_loop import WKVLoopRollout


_EXACT_RUBRIC = {"type": "exact", "value": "42"}


def _mk(text: str, M: int, n_entropy_steps: int = None) -> WKVLoopRollout:
    """Minimal rollout: flat entropy trajectory (no ReLU(dH) penalty) and
    flat wkv_stability, so tests can isolate the effort (beta*M) term
    without also reasoning about entropy-penalty arithmetic."""
    n = n_entropy_steps if n_entropy_steps is not None else M + 1
    return WKVLoopRollout(
        prompt_ids=[1], answer_ids=[2], M=M,
        entropy_trajectory=[1.0] * n, wkv_stability=[0.0] * n,
        exit_reason="commit", text=text,
    )


def test_score_correct_exact_and_regex_and_partial_format():
    assert _score_correct("the answer is 42", _EXACT_RUBRIC) == 1.0
    assert _score_correct("the answer is 7", _EXACT_RUBRIC) == -1.0

    regex_rubric = {"type": "regex", "value": r"row\s*=\s*3.*col\s*=\s*1"}
    assert _score_correct("row=3 col=1", regex_rubric) == 1.0
    # right format ("row=X col=Y"), wrong value → 0.0, not -1.0 — avoids
    # punishing a model that has learned the output structure but not yet
    # the task, which would otherwise look identical to total noise.
    assert _score_correct("row=9 col=9", regex_rubric) == 0.0
    # no row=/col= pattern at all → genuinely wrong, -1.0
    assert _score_correct("banana", regex_rubric) == -1.0


def test_gate_on_correct_shields_wrong_rollouts_from_shaping():
    # Two wrong rollouts with very different M — under gating, both must
    # score exactly r_correct (-1.0), since shaping never applies to them.
    rollouts = [_mk("wrong", M=1), _mk("also wrong", M=8)]
    rewards, diag = compute_wkv_loop_rewards(
        rollouts, _EXACT_RUBRIC, beta=0.1, gamma=0.1, gate_on_correct=True)
    assert torch.allclose(rewards, torch.tensor([-1.0, -1.0])), rewards.tolist()
    assert torch.equal(diag["r_effort"], torch.zeros(2))
    # M is still logged for diagnostics even though it didn't affect reward
    assert diag["M"].tolist() == [1, 8]


def test_gate_on_correct_still_ranks_among_correct_rollouts():
    # Two correct rollouts, different M — shaping should still separate
    # them (efficient correct answer scores higher than verbose one).
    rollouts = [_mk("42", M=1), _mk("42", M=5)]
    rewards, diag = compute_wkv_loop_rewards(
        rollouts, _EXACT_RUBRIC, beta=0.1, gamma=0.0, gate_on_correct=True)
    assert rewards[0] > rewards[1], (
        f"low-M correct rollout ({rewards[0]}) should outscore "
        f"high-M correct rollout ({rewards[1]})"
    )
    assert torch.allclose(rewards, torch.tensor([0.9, 0.5]))


def test_ungated_shaping_reproduces_pre_2026_08_19_behavior():
    # Same two wrong rollouts as above, but gate_on_correct=False: this is
    # the exact pathology that caused the training collapse — a "wrong but
    # brief" rollout (M=1) scores better than a "wrong and verbose" one
    # (M=8), teaching the model to minimize effort while still wrong.
    rollouts = [_mk("wrong", M=1), _mk("also wrong", M=8)]
    rewards, diag = compute_wkv_loop_rewards(
        rollouts, _EXACT_RUBRIC, beta=0.1, gamma=0.0, gate_on_correct=False)
    assert rewards[0] > rewards[1], "ungated: low-M wrong rollout should score higher (the bug)"
    assert torch.allclose(rewards, torch.tensor([-1.1, -1.8]))


def test_score_correct_rejects_input_echo_exploit():
    # Found live 2026-08-19 (g1i_real_run8, step106): a model that echoes
    # the input matrix verbatim gets scored correct whenever the target
    # digit happens to appear isolated in that echo — the bare-digit
    # rubric (?<!\d)N(?!\d) matches anywhere, and matrix rows are
    # themselves sequences of isolated 0/1 digits. 45/153 correct=True
    # rollouts in that run showed this exact pattern.
    rubric = {"type": "regex", "value": r"(?<!\d)1(?!\d)"}
    echoed_input = (
        "= 0 1 1 1\nB           = 1 1 1 1\n────────────────────"
    )
    assert _score_correct(echoed_input, rubric) == -1.0, (
        "echoing the input matrix must not score as correct just because "
        "the target digit happens to appear in the copied text"
    )
    # Genuine terse answers (with reasonable padding) must still pass —
    # the fix must not overcorrect into rejecting real short answers.
    for text in ["1", "= 1", "The answer is 1", "1\n", " 1 "]:
        assert _score_correct(text, rubric) == 1.0, f"{text!r} should score correct"
    assert _score_correct("0", rubric) == -1.0


def test_zeta_density_reward_rewards_efficient_entropy_drop():
    # zeta=0 (default) must be a true no-op — same rewards as without it.
    rollouts = [_mk("42", M=2)]
    r_off, diag_off = compute_wkv_loop_rewards(
        rollouts, _EXACT_RUBRIC, beta=0.0, gamma=0.0, zeta=0.0)
    assert diag_off["r_density"].tolist() == [0.0]

    # Two correct rollouts, same M, but rollout A's entropy drops a lot
    # (efficient — resolved uncertainty fast) with little state motion,
    # rollout B's entropy barely drops despite the same amount of motion.
    # A should score higher under zeta>0.
    efficient = WKVLoopRollout(
        prompt_ids=[1], answer_ids=[2], M=2,
        entropy_trajectory=[3.0, 1.0, 0.5], wkv_stability=[0.0, 1.0, 1.0],
        exit_reason="commit", text="42",
    )
    wasteful = WKVLoopRollout(
        prompt_ids=[1], answer_ids=[2], M=2,
        entropy_trajectory=[3.0, 2.9, 2.8], wkv_stability=[0.0, 1.0, 1.0],
        exit_reason="commit", text="42",
    )
    rewards, diag = compute_wkv_loop_rewards(
        [efficient, wasteful], _EXACT_RUBRIC,
        beta=0.0, gamma=0.0, zeta=1.0, gate_on_correct=True)
    assert diag["r_density"][0] > diag["r_density"][1] > 0, (
        f"efficient rollout should score higher density reward than "
        f"wasteful one: {diag['r_density'].tolist()}"
    )

    # A wrong rollout with the same efficient trajectory must get zero
    # density reward under gate_on_correct — same shielding as beta/gamma.
    wrong_but_efficient = WKVLoopRollout(
        prompt_ids=[1], answer_ids=[2], M=2,
        entropy_trajectory=[3.0, 1.0, 0.5], wkv_stability=[0.0, 1.0, 1.0],
        exit_reason="commit", text="wrong",
    )
    _, diag_wrong = compute_wkv_loop_rewards(
        [wrong_but_efficient], _EXACT_RUBRIC,
        beta=0.0, gamma=0.0, zeta=1.0, gate_on_correct=True)
    assert diag_wrong["r_density"].tolist() == [0.0]


def test_zero_beta_gamma_gating_is_a_no_op():
    # beta=gamma=0 (the g1i_real_run7 ablation config) must give identical
    # rewards regardless of gate_on_correct — nothing to gate when the
    # shaping weights are already zero.
    rollouts = [_mk("42", M=1), _mk("wrong", M=5)]
    r_gated, _ = compute_wkv_loop_rewards(
        rollouts, _EXACT_RUBRIC, beta=0.0, gamma=0.0, gate_on_correct=True)
    r_ungated, _ = compute_wkv_loop_rewards(
        rollouts, _EXACT_RUBRIC, beta=0.0, gamma=0.0, gate_on_correct=False)
    assert torch.equal(r_gated, r_ungated)
    assert torch.allclose(r_gated, torch.tensor([1.0, -1.0]))


def main() -> int:
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed.append(t.__name__)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
