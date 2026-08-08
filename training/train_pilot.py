"""A1 pilot driver — applies the state_reg monkey-patch, then delegates
to the vendored ``training/rwkv-peft/train.py`` argparse-driven trainer.

Reads ``training/config/pilot.yaml`` for model / LoRA / state_reg config,
translates it into the CLI flags ``train.py`` expects, and invokes
``train.py`` via ``runpy.run_path`` so the monkey-patch installed here
is live in that module's ``sys.modules`` cache.

Env vars set before invoking ``train.py``:
    RWKV_MY_TESTING=7     — select RWKV-7 model class
    RWKV_TRAIN_TYPE=infctx — required by state_reg patch
    RWKV_JIT_ON=1         — vendored default
    RWKV_FLOAT_MODE=bf16  — from pilot.yaml model.dtype
    RWKV_HEAD_SIZE=64     — RWKV-7 head size (arch-fixed)
    NOESIS_STATE_REG_YAML=<absolute path>

Usage (from repo root, on a machine with CUDA):
    python training/train_pilot.py [--config training/config/pilot.yaml]

The pilot bring-up plan (from training/README.md and pilot.yaml docstring):
    step 1: mode=off, alpha=0.0            — baseline CE
    step 2: mode=trajectory_reg, alpha=0.0 — sanity, CE unchanged
    step 3: mode=trajectory_reg, alpha>0   — sweep

Change the ``state_reg`` block in pilot.yaml to switch between steps.
This driver does not add its own CLI beyond ``--config``; everything
else lives in the YAML for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import runpy
import signal
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
PEFT_DIR = TRAINING_DIR / "rwkv-peft"


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _install_sigterm_handler(run_dir: Path) -> None:
    """Write .noesis_stop sentinel on SIGTERM so the training loop saves and exits."""
    sentinel = run_dir / ".noesis_stop"

    def _handler(signum, frame):
        print(f"\n[train_pilot] SIGTERM received — writing {sentinel}", flush=True)
        try:
            sentinel.touch()
        except Exception as e:
            print(f"[train_pilot] sentinel write failed: {e}", flush=True)

    signal.signal(signal.SIGTERM, _handler)


def _build_argv(cfg: dict, yaml_path: Path) -> list[str]:
    model = cfg["model"]
    lora = cfg["lora"]
    optim = cfg["optimizer"]
    train = cfg["training"]
    corpus = cfg["corpus"]
    logging = cfg["logging"]

    raw_ckpt = os.path.expanduser(model["checkpoint"])
    ckpt = raw_ckpt if os.path.isabs(raw_ckpt) else str(REPO_ROOT / raw_ckpt)
    data = str(REPO_ROOT / corpus["tokenized_pt"])
    run_dir = str(REPO_ROOT / logging["run_dir"] / logging["run_name"])

    n_layer = str(model.get("n_layer", 24))
    n_embd = str(model.get("n_embd", 1024))
    argv = [
        str(PEFT_DIR / "train.py"),
        "--load_model", ckpt,
        "--proj_dir", run_dir,
        "--data_file", data,
        "--data_type", "sft",
        "--vocab_size", "65536",
        "--n_layer", n_layer,
        "--n_embd", n_embd,
        "--my_testing", "x070",
        "--ctx_len", str(model["ctx_len"]),
        "--chunk_ctx", str(train.get("chunk_ctx", 512)),
        # vendored train.py:191 does os.environ["RWKV_TRAIN_TYPE"] = args.train_type,
        # so we must pass infctx explicitly — otherwise the default 'none' resets the
        # env after our patch already captured the infctx training_step, and model
        # forward routes to forward_normal (misinterprets shift_states as attn_mask).
        "--train_type", "infctx",
        # vendored train.py:196 overrides WKV env from args.op (default 'cuda').
        # For x070+infctx we MUST use fla — see WKV comment above.
        "--op", "fla",
        # num_workers=8 (vendored default) forks 8 worker processes; each
        # inherits the full 6GB in-RAM dataset from noesis_dataset_patch
        # (copy-on-write breaks under index access). Caused OOM on Step 3.
        "--num_workers", "1",
        "--micro_bsz", str(optim["batch_size"]),
        "--accumulate_grad_batches", str(optim["grad_accum_steps"]),
        "--lr_init", str(optim.get("lr_init", optim.get("lr", 2e-5))),
        "--lr_final", str(optim.get("lr_final", optim.get("lr", 2e-5) * 0.1)),
        "--warmup_steps", str(optim["warmup_steps"]),
        "--weight_decay", str(optim["weight_decay"]),
        "--epoch_count", str(train["epochs"]),
        "--epoch_save", "1",
        "--precision", model["dtype"],
        "--accelerator", "gpu",
        "--devices", "1",
        "--strategy", "deepspeed_stage_1",
        "--grad_cp", "1",
        "--peft", "lora",
        "--peft_config", (
            f'{{"r":{lora["rank"]},'
            f'"lora_alpha":{lora["alpha"]},'
            f'"lora_dropout":{lora["dropout"]}}}'
        ),
    ]
    return argv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(TRAINING_DIR / "config" / "pilot.yaml"))
    args = ap.parse_args()
    yaml_path = Path(args.config).resolve()
    cfg = _load_yaml(yaml_path)

    # Env vars must be set *before* light_rwkv is imported.
    # RWKV_MY_TESTING must be 'x070' (not '7') because rwkvop.py checks
    # `'x070' in os.environ["RWKV_MY_TESTING"]` at module-import time to
    # decide whether to define RUN_RWKV7_INFCTX. If '7' at import time,
    # RWKV-7 op branches never run, and the stub (raise NotImplementedError)
    # is what our patched training_step hits.
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ["RWKV_FLOAT_MODE"] = cfg["model"]["dtype"]
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
    os.environ.setdefault("RWKV_CTXLEN", str(cfg["model"]["ctx_len"]))
    # Vendored rwkvop.py hard-imports os.environ["WKV"] at module-import
    # time; must be set BEFORE the state_reg patch triggers that import.
    # Must be 'fla' for x070+infctx: vendored rwkvop.py only defines
    # RUN_RWKV7_INFCTX inside the FLA branch (RWKV_MY_TESTING=x070,
    # WKV=fla, RWKV_TRAIN_TYPE=infctx). The CUDA branch for x070 has no
    # infctx op — Step 1 worked only because RWKV_TRAIN_TYPE was 'none'
    # then and the non-infctx path used RUN_CUDA_RWKV7g.
    os.environ.setdefault("WKV", "fla")
    os.environ.setdefault("FUSED_KERNEL", "0")
    os.environ["NOESIS_STATE_REG_YAML"] = str(yaml_path)

    # Intermediate checkpoint interval (0 = disabled).
    save_steps = cfg.get("logging", {}).get("save_steps", 0)
    if save_steps:
        os.environ["NOESIS_SAVE_STEPS"] = str(save_steps)

    # Dataset resume: skip first N rollouts (for manual restart after interruption).
    skip_rollouts = cfg.get("training", {}).get("skip_rollouts", 0)
    if skip_rollouts:
        os.environ["NOESIS_SKIP_ROLLOUTS"] = str(skip_rollouts)

    sys.path.insert(0, str(TRAINING_DIR))
    sys.path.insert(0, str(PEFT_DIR))

    # chdir before the state_reg patch fires — light_rwkv_state_reg_patch.apply()
    # imports rwkvt.lightning_train.light_rwkv, which cascades into rwkvop.py
    # and loads CUDA kernels via relative paths (`cuda/wkv7_op.cpp` etc.). Those
    # paths only resolve from PEFT_DIR.
    os.chdir(PEFT_DIR)

    # SIGTERM handler must be installed after run_dir is known.
    run_dir = REPO_ROOT / cfg["logging"]["run_dir"] / cfg["logging"]["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    _install_sigterm_handler(run_dir)

    import light_rwkv_state_reg_patch
    status = light_rwkv_state_reg_patch.apply()
    print(f"[train_pilot] {status}")

    import noesis_dataset_patch
    ds_status = noesis_dataset_patch.apply()
    print(f"[train_pilot] {ds_status}")

    train_argv = _build_argv(cfg, yaml_path)
    print(f"[train_pilot] invoking vendored train.py with {len(train_argv)-1} args")
    sys.argv = train_argv
    runpy.run_path(train_argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
