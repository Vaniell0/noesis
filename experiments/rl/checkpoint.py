"""checkpoint.py — directory-based checkpoint save/load for the RL stack.

Extracted from train_wkv_loop.py 2026-08-19: self-contained, no dependency
on the training loop itself, already covered end-to-end by
test_checkpoint.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from experiments.rl.loader import LoadedModel


def save_checkpoint(out_dir: Path, loaded: LoadedModel, step: int,
                     mlp_delta: Optional[torch.nn.Module] = None,
                     sched=None, monitor=None) -> None:
    """Saves to a `ckpt_step{step:06d}/` directory, one file per trainable
    parameter, instead of one `.pt` with a single in-memory dict of all
    CPU-copied tensors.

    The old single-dict-comprehension version (`{n: p.detach().cpu() for
    n, p in trainable.items()}`) built a full second CPU-resident copy of
    every trainable parameter (~5.9GB for G1i 2.9B) before `torch.save`
    even started writing — on top of `Int8AdamW(offload_state=True)`'s
    already-resident ~5.8GB of CPU-side optimizer state, this pushed
    system RAM (15GB, no swap) over the edge. Confirmed 2026-08-18 via
    `dmesg`: kernel oom-killer SIGKILLed the training process
    (anon-rss=14.9GB) at exactly this point, silently — no Python
    traceback, no checkpoint file, process just vanished. Writing one
    parameter at a time bounds peak checkpoint-save memory to the single
    largest parameter (tens–hundreds of MB), not the whole model.
    """
    ckpt_dir = out_dir / f"ckpt_step{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ckpt_dir / "model"
    n_saved = 0
    if loaded.backend == "peft":
        # Save only trainable params (LoRA A/B or full diff)
        model_dir.mkdir(exist_ok=True)
        for n, p in loaded.model.named_parameters():
            if not p.requires_grad:
                continue
            torch.save(p.detach().cpu(), model_dir / f"{n}.pt")
            n_saved += 1
    else:
        # blink (CPU) backend has no trainable params — --no-update smoke
        # runs are the only thing that should ever hit this path. Flagged
        # loudly rather than silently: a checkpoint with no model/ dir
        # looks identical to a real one until someone tries to resume from
        # it and finds nothing to load.
        print(f"[train] WARNING: checkpoint at step {step} has no model "
              f"weights (backend={loaded.backend!r}, not 'peft') — only "
              f"step count is saved, nothing to resume from")
    meta = {"step": step, "param_names": None}
    if loaded.backend == "peft":
        meta["param_names"] = [n for n, p in loaded.model.named_parameters()
                               if p.requires_grad]
    if mlp_delta is not None:
        meta["mlp_delta"] = {n: p.detach().cpu()
                             for n, p in mlp_delta.named_parameters()}
    if sched is not None:
        # Curriculum progress (per-category level + accuracy history) —
        # without this, a resume restores model weights correctly but
        # silently resets every category back to start_level=1, re-earning
        # curriculum progress from scratch. Added 2026-08-18 alongside the
        # model resume path, same reasoning: interruptible instance means
        # resumes are the expected case, not a rare edge case.
        meta["sched"] = sched.state_dict()
    if monitor is not None:
        # TrainingMonitor's HACKING check uses a 10-batch trailing window
        # of reward/accuracy — without this, every resume starts that
        # window empty, and a freshly-filling small-sample window is prone
        # to spurious early HACKING triggers unrelated to real reward
        # hacking. Found 2026-08-19: two separate resumes from the same
        # checkpoint both tripped HACKING within ~17-18 steps of
        # restarting. Same reasoning as `sched` above.
        meta["monitor"] = monitor.state_dict()
    torch.save(meta, ckpt_dir / "meta.pt")
    print(f"[train] checkpoint → {ckpt_dir} ({n_saved} tensors)")


def load_checkpoint(path: Path, loaded: LoadedModel,
                     mlp_delta: Optional[torch.nn.Module] = None,
                     sched=None, monitor=None) -> int:
    """Resume from a checkpoint directory written by `save_checkpoint`.
    Returns the saved step (caller should continue from `step + 1`).

    Written 2026-08-18 — `save_checkpoint` existed with no corresponding
    load path at all; a checkpoint could be written but never read back.
    Matters specifically because the planned GPU rental is a preemptible
    (interruptible) instance, which the provider can reclaim mid-run —
    without this, that would mean restarting training from scratch on
    every preemption, not just resuming.
    """
    meta = torch.load(path / "meta.pt", map_location=loaded.device)
    if meta["param_names"] is None:
        raise ValueError(
            f"{path} has no saved model weights — it was saved on a "
            f"non-'peft' backend and has nothing to resume from (see the "
            f"WARNING printed when it was saved)"
        )
    state_dict = {
        n: torch.load(path / "model" / f"{n}.pt", map_location=loaded.device)
        for n in meta["param_names"]
    }
    missing, unexpected = loaded.model.load_state_dict(state_dict, strict=False)
    # `strict=False` is required: only trainable params were saved, so the
    # frozen backbone is expected to show up as "missing" here — that's
    # correct, not an error. Genuinely unexpected keys would still be real.
    if unexpected:
        raise ValueError(f"{path}: unexpected keys in checkpoint not present "
                          f"in model: {unexpected}")
    if mlp_delta is not None and "mlp_delta" in meta:
        mlp_delta.load_state_dict(meta["mlp_delta"], strict=True)
    if sched is not None and "sched" in meta:
        sched.load_state_dict(meta["sched"])
    if monitor is not None and "monitor" in meta:
        monitor.load_state_dict(meta["monitor"])
    step = meta["step"]
    print(f"[train] resumed from {path} at step {step} "
          f"({len(state_dict)} trainable tensors loaded)")
    return step
