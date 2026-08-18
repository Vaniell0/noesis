"""Regression tests for checkpoint save/resume — the RL stack's own core.py-
style guard, same plain-assertion/no-pytest convention as
experiments/_common/test_core.py.

Covers the resume path added 2026-08-18: before that date
`_save_checkpoint` existed with no corresponding load path at all (write
without ever reading back), and separately `CorpusScheduler` held no
serializable state at all — both real gaps, found by hand-verification
with throwaway `python -c` snippets during that session, never
persisted as an actual test until now. Given the training VM is
interruptible (can be reclaimed anytime), a broken resume path is not a
theoretical risk — this is the thing standing between "one preemption"
and "start the run over."

Run: `python experiments/rl/test_checkpoint.py`
(CPU-only, no model download or GPU needed — uses a tiny mock nn.Module
in place of a real LoadedModel, and a tiny synthetic task list for the
curriculum scheduler.)
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.rl.train_wkv_loop import _save_checkpoint, _load_checkpoint
from experiments.rl.corpus import CorpusScheduler


class _FakeTrainableModel(nn.Module):
    """Frozen backbone + trainable head — mimics a peft-loaded model's
    parameter split (only `requires_grad=True` params get checkpointed)."""
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)
        self.trainable = nn.Linear(4, 4)


def test_checkpoint_roundtrip_restores_trainable_weights():
    with tempfile.TemporaryDirectory() as td:
        m = _FakeTrainableModel()
        loaded = SimpleNamespace(backend="peft", model=m, device="cpu")

        _save_checkpoint(pathlib.Path(td), loaded, step=42)

        original = m.trainable.weight.detach().clone()
        with torch.no_grad():
            m.trainable.weight.add_(1.0)
        assert not torch.equal(m.trainable.weight, original), "mutation didn't happen — test is broken"

        step = _load_checkpoint(pathlib.Path(td) / "ckpt_step000042.pt", loaded)
        assert step == 42, f"expected step 42, got {step}"
        assert torch.equal(m.trainable.weight, original), "load did not restore the trainable weight"


def test_checkpoint_on_blink_backend_has_no_weights_but_does_not_crash():
    # blink (CPU inference-only) backend has nothing trainable — the
    # checkpoint should still write (with a warning), just with no
    # "model" key. Loading it back must fail loudly, not silently.
    with tempfile.TemporaryDirectory() as td:
        m = _FakeTrainableModel()
        loaded = SimpleNamespace(backend="blink", model=m, device="cpu")
        _save_checkpoint(pathlib.Path(td), loaded, step=1)

        try:
            _load_checkpoint(pathlib.Path(td) / "ckpt_step000001.pt", loaded)
            raise AssertionError("loading a weight-less checkpoint should have raised ValueError")
        except ValueError:
            pass


def _make_tasks():
    return [{"category": "wordsearch", "level": l, "rubric": {}, "prompt": f"p{l}"}
            for l in range(1, 6)]


def test_curriculum_state_roundtrip():
    sched = CorpusScheduler(_make_tasks(), rng_seed=1)
    for _ in range(6):
        sched.update_accuracy(1, 0.9)
    level_before = sched.current_level
    assert level_before > 1, f"expected the curriculum to advance past level 1, got {level_before}"

    saved = sched.state_dict()

    fresh = CorpusScheduler(_make_tasks(), rng_seed=1)
    assert fresh.current_level == 1, "a fresh scheduler must start at level 1"
    fresh.load_state_dict(saved)
    assert fresh.current_level == level_before, (
        f"resumed scheduler landed on level {fresh.current_level}, expected {level_before} "
        "— curriculum progress was not actually restored"
    )


def test_checkpoint_and_curriculum_together_via_save_checkpoint():
    """The actual integration point: _save_checkpoint(sched=...) /
    _load_checkpoint(sched=...) — not just the two pieces tested in isolation."""
    with tempfile.TemporaryDirectory() as td:
        m = _FakeTrainableModel()
        loaded = SimpleNamespace(backend="peft", model=m, device="cpu")
        sched = CorpusScheduler(_make_tasks(), rng_seed=1)
        for _ in range(6):
            sched.update_accuracy(1, 0.9)
        level_before = sched.current_level

        _save_checkpoint(pathlib.Path(td), loaded, step=7, sched=sched)

        fresh_sched = CorpusScheduler(_make_tasks(), rng_seed=1)
        step = _load_checkpoint(pathlib.Path(td) / "ckpt_step000007.pt", loaded, sched=fresh_sched)
        assert step == 7
        assert fresh_sched.current_level == level_before, (
            "curriculum state did not survive the real _save_checkpoint/_load_checkpoint path "
            "(as opposed to calling sched.state_dict()/load_state_dict() directly)"
        )


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
