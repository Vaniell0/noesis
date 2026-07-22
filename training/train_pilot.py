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
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
PEFT_DIR = TRAINING_DIR / "rwkv-peft"


def _load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _build_argv(cfg: dict, yaml_path: Path) -> list[str]:
    model = cfg["model"]
    lora = cfg["lora"]
    optim = cfg["optimizer"]
    train = cfg["training"]
    corpus = cfg["corpus"]
    logging = cfg["logging"]

    ckpt = os.path.expanduser(model["checkpoint"])
    data = str(REPO_ROOT / corpus["tokenized_pt"])
    run_dir = str(REPO_ROOT / logging["run_dir"] / logging["run_name"])

    argv = [
        str(PEFT_DIR / "train.py"),
        "--load_model", ckpt,
        "--proj_dir", run_dir,
        "--data_file", data,
        "--data_type", "sft",
        "--ctx_len", str(model["ctx_len"]),
        "--chunk_ctx", "1",  # per-token trajectory for state_reg
        "--micro_bsz", str(optim["batch_size"]),
        "--accumulate_grad_batches", str(optim["grad_accum_steps"]),
        "--lr_init", str(optim["lr"]),
        "--lr_final", str(optim["lr"] * 0.1),
        "--warmup_steps", str(optim["warmup_steps"]),
        "--weight_decay", str(optim["weight_decay"]),
        "--epoch_count", str(train["epochs"]),
        "--epoch_save", "1",
        "--precision", model["dtype"],
        "--accelerator", "gpu",
        "--devices", "1",
        "--strategy", "auto",
        "--peft", "lora",
        "--lora_config", (
            f'{{"lora_load":"","lora_r":{lora["rank"]},'
            f'"lora_alpha":{lora["alpha"]},"lora_dropout":{lora["dropout"]},'
            f'"lora_parts":"{",".join(lora["target_modules"])}"}}'
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
    os.environ.setdefault("RWKV_MY_TESTING", "7")
    os.environ["RWKV_TRAIN_TYPE"] = "infctx"
    os.environ.setdefault("RWKV_JIT_ON", "1")
    os.environ["RWKV_FLOAT_MODE"] = cfg["model"]["dtype"]
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("FUSED_KERNEL", "0")
    os.environ["NOESIS_STATE_REG_YAML"] = str(yaml_path)

    sys.path.insert(0, str(TRAINING_DIR))
    sys.path.insert(0, str(PEFT_DIR))

    import light_rwkv_state_reg_patch
    status = light_rwkv_state_reg_patch.apply()
    print(f"[train_pilot] {status}")

    train_argv = _build_argv(cfg, yaml_path)
    print(f"[train_pilot] invoking vendored train.py with {len(train_argv)-1} args")
    sys.argv = train_argv
    runpy.run_path(train_argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
