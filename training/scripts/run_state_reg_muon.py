#!/usr/bin/env python3
"""Driver: launch the Lightning/RWKV-PEFT trainer with the real state_reg
(L_state) patch applied and an arbitrary set of train.py CLI args, e.g.
`--optimizer muon`.

Exists because `light_rwkv_state_reg_patch.py`'s own docstring says a
driver must set env vars and call `apply()` *before* Lightning constructs
the RWKV module — no such driver existed before 2026-09-10 (the patch was
real and working, just never actually invoked by anything). This is that
driver, not a new training mechanism: everything downstream is
`train.py`'s own existing argparse + Lightning flow.

Usage:
    python training/scripts/run_state_reg_muon.py \\
        --state-reg-yaml training/config/pilot_g1i_muon_lora.yaml \\
        -- \\
        --load_model models/....pth --data_file ...jsonl --data_type jsonl \\
        --optimizer muon --peft lora --lora_config '{...}' \\
        --train_type infctx --chunk_ctx 512 ...

Everything after `--` is passed to train.py verbatim as sys.argv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _stub_deepspeed_if_missing() -> None:
    """rwkvt/rwkv7/model.py does `import deepspeed` unconditionally at
    module scope (only actually calling into it when --grad_cp=1), and
    light_rwkv.py separately does `from deepspeed.ops.adam import
    DeepSpeedCPUAdam, FusedAdam` guarded by `importlib.util.find_spec`.
    A bare `sys.modules["deepspeed"] = types.ModuleType(...)` injection
    (the trick used in experiments/rl/loader.py, which only needs the
    bare top-level import to succeed) is NOT enough here: find_spec
    looks at sys.modules first, sees our fake entry, and light_rwkv.py
    then tries `from deepspeed.ops.adam import ...` against a module
    with no `.ops` submodule. Using a real importable stub PACKAGE on
    sys.path instead makes both import forms behave like a genuinely
    (if minimally) installed package."""
    try:
        import deepspeed  # noqa: F401
        return
    except ImportError:
        pass
    stub_root = str(Path(__file__).parent / "_deepspeed_stub")
    if stub_root not in sys.path:
        sys.path.insert(0, stub_root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-reg-yaml", required=True)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    train_args = args.train_args
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    # Only what light_rwkv_state_reg_patch.apply() itself reads *before*
    # train.py's own argparse runs (RWKV_TRAIN_TYPE, RWKV_MY_TESTING etc.
    # get re-set from --train_type/--my_testing inside train.py right
    # after — pass those as real CLI args in train_args, not here).
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ["NOESIS_STATE_REG_YAML"] = str(Path(args.state_reg_yaml).resolve())
    os.environ["NOESIS_LOG_STEPS"] = str(args.log_steps)
    # Several vendored modules read these at IMPORT time (rwkvt/rwkv7/*.py,
    # rwkvt/operator/rwkvop.py) — model-class selection, WKV backend choice,
    # fused-kernel flag, head size — all of which patch.apply()'s import
    # chain reaches below, before train.py's own argparse has a chance to
    # set the same env vars from the matching --flag. Mirror whatever's
    # passed in train_args (train.py re-sets these to the same values
    # right after anyway, so this is just making import-time equal
    # runtime-time, not overriding anything).
    def _flag(name: str, default: str) -> str:
        return train_args[train_args.index(name) + 1] if name in train_args else default

    os.environ["RWKV_MY_TESTING"] = _flag("--my_testing", "x052")
    os.environ["WKV"] = _flag("--op", "cuda")
    os.environ["FUSED_KERNEL"] = "1" if "--fused_kernel" in train_args else "0"
    os.environ["RWKV_HEAD_SIZE_A"] = _flag("--head_size_a", "64")
    os.environ["RWKV_CTXLEN"] = _flag("--ctx_len", "1024")

    _stub_deepspeed_if_missing()

    training_dir = str(_REPO_ROOT / "training")
    peft_dir = str(_REPO_ROOT / "training" / "rwkv-peft")
    for p in (training_dir, peft_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    import light_rwkv_state_reg_patch as patch
    status = patch.apply()
    print(f"[run_state_reg_muon] {status}")

    sys.argv = ["train.py"] + train_args
    os.chdir(peft_dir)
    train_py = Path(peft_dir) / "train.py"
    exec(compile(train_py.read_text(), str(train_py), "exec"), {"__name__": "__main__"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
