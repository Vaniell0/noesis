"""CPU-runnable smoke test for `light_rwkv_state_reg_patch.apply`.

Verifies the *pre-active* half of the patch — the parts that don't need
CUDA + deepspeed + a real RWKV checkpoint:

  (a) missing RWKV_TRAIN_TYPE / wrong value → RuntimeError.
  (b) missing NOESIS_STATE_REG_YAML → RuntimeError.
  (c) mode='off' path → returns INACTIVE status, does NOT import
      `rwkvt.lightning_train.light_rwkv`.
  (d) idempotent: second apply() call is a no-op.

The *active* path (mode=trajectory_reg, alpha>0 → monkey-patch installed
on `RWKV.training_step`) cannot be exercised without the deepspeed +
CUDA stack; the smoke test in `test_state_reg_hookup.py` covers the
underlying `compute_state_reg` + `StateCapture` interface on a mock.

Run:
    /home/vaniello/Desktop/projects/noesis/training/.venv/bin/python \\
        training/tests/test_light_rwkv_patch.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)


def _fresh_patch_module():
    """Re-import patch module so each test starts with _PATCH_APPLIED=False."""
    if "light_rwkv_state_reg_patch" in sys.modules:
        del sys.modules["light_rwkv_state_reg_patch"]
    return importlib.import_module("light_rwkv_state_reg_patch")


def _write_pilot_yaml(mode: str, alpha: float) -> str:
    body = textwrap.dedent(f"""
    state_reg:
      mode: {mode!r}
      alpha: {alpha}
      lambda_delta: 1.0
      lambda_curvature: 1.0
      work_layers: []
    """).strip()
    fd, path = tempfile.mkstemp(suffix=".yaml", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return path


def _clean_env() -> None:
    for k in ("RWKV_TRAIN_TYPE", "NOESIS_STATE_REG_YAML"):
        os.environ.pop(k, None)


# --------------------------------------------------------------------------- #

def test_wrong_train_type_raises() -> None:
    _clean_env()
    os.environ["RWKV_TRAIN_TYPE"] = "default"  # not 'infctx'
    os.environ["NOESIS_STATE_REG_YAML"] = _write_pilot_yaml("off", 0.0)
    patch = _fresh_patch_module()
    try:
        patch.apply()
    except RuntimeError as e:
        assert "infctx" in str(e), f"expected 'infctx' in error, got: {e}"
        print("(a) wrong RWKV_TRAIN_TYPE raises: OK")
        return
    raise AssertionError("apply() did not raise on RWKV_TRAIN_TYPE=default")


def test_missing_yaml_raises() -> None:
    _clean_env()
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    patch = _fresh_patch_module()
    try:
        patch.apply()
    except RuntimeError as e:
        assert "NOESIS_STATE_REG_YAML" in str(e), f"unexpected error: {e}"
        print("(b) missing NOESIS_STATE_REG_YAML raises: OK")
        return
    raise AssertionError("apply() did not raise on missing yaml env var")


def test_off_mode_inactive_and_no_light_rwkv_import() -> None:
    _clean_env()
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ["NOESIS_STATE_REG_YAML"] = _write_pilot_yaml("off", 0.0)
    # Ensure light_rwkv is NOT already imported from a prior test.
    for k in list(sys.modules):
        if k.startswith("rwkvt.lightning_train"):
            del sys.modules[k]
    patch = _fresh_patch_module()
    status = patch.apply()
    assert "INACTIVE" in status, f"expected INACTIVE status, got: {status}"
    assert patch.is_applied(), "is_applied() should be True after apply()"
    assert "rwkvt.lightning_train.light_rwkv" not in sys.modules, (
        "INACTIVE branch should NOT import light_rwkv (deepspeed/CUDA-hostile)"
    )
    print(f"(c) off mode INACTIVE, light_rwkv untouched: OK — {status}")


def test_idempotent() -> None:
    _clean_env()
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ["NOESIS_STATE_REG_YAML"] = _write_pilot_yaml("off", 0.0)
    patch = _fresh_patch_module()
    s1 = patch.apply()
    s2 = patch.apply()
    assert "no-op" in s2, f"second apply() should no-op, got: {s2}"
    print(f"(d) idempotent: OK — first={s1!r}, second={s2!r}")


def main() -> int:
    tests = [
        test_wrong_train_type_raises,
        test_missing_yaml_raises,
        test_off_mode_inactive_and_no_light_rwkv_import,
        test_idempotent,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except BaseException as e:  # noqa: BLE001
            failed.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)}/{len(tests)} FAILED")
        return 1
    print(f"\n{len(tests)}/{len(tests)} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
