"""Monkey-patch RWKV-PEFT's ``light_rwkv.RWKV.training_step`` to add state_reg.

Non-invasive to the vendored tree at ``training/rwkv-peft/``. Import this
module **before** Lightning constructs the ``RWKV`` module and the
``infctx``-branch ``training_step`` is swapped for a wrapper that:

  1. Runs the same chunked infctx forward as the original.
  2. Captures per-chunk final ``new_wkv_states`` into a trajectory
     (**before** the ``.clone().detach()`` that blocks cross-chunk grad).
  3. Slices to ``cfg.work_layers`` and calls ``compute_state_reg``.
  4. Adds ``cfg.alpha * L_state`` to the returned total loss.

If ``mode='off'`` or ``alpha==0.0``, the wrapper degrades to a straight
delegation of the original method (no state capture, zero overhead).

Contract: state trajectory is *per-chunk* state, not per-token. Set
``chunk_ctx=1`` in the args to get per-token trajectory; larger chunks
yield a sparser trajectory (still valid for ``compute_state_reg`` — it
treats the time axis as opaque).

Env vars (set by driver script before importing this module):
  ``RWKV_TRAIN_TYPE``        — must be ``infctx`` (else the patch no-ops).
  ``NOESIS_STATE_REG_YAML``  — path to ``training/config/pilot.yaml``.

Not applicable outside the vendored ``light_rwkv``: this is a targeted
integration shim, not a general trainer. Smoke-tested on Linux/CPU via
``training/tests/test_light_rwkv_patch.py``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_PATCH_APPLIED = False


def _resolve_paths() -> None:
    """Ensure ``training/`` and ``training/rwkv-peft/`` are on ``sys.path``."""
    here = os.path.dirname(os.path.abspath(__file__))
    peft = os.path.join(here, "rwkv-peft")
    for p in (here, peft):
        if p not in sys.path:
            sys.path.insert(0, p)


def apply() -> str:
    """Install the state_reg wrapper. Returns a short human-readable status.

    Idempotent: second call is a no-op. Raises ``RuntimeError`` if
    ``RWKV_TRAIN_TYPE`` is not ``infctx`` (state_reg needs per-chunk
    ``wkv_states`` from the infctx branch).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return "state_reg patch already applied (no-op)"

    _resolve_paths()

    train_type = os.environ.get("RWKV_TRAIN_TYPE", "")
    if train_type != "infctx":
        raise RuntimeError(
            f"state_reg patch requires RWKV_TRAIN_TYPE=infctx, got {train_type!r}. "
            "Set the env var *before* importing this module."
        )

    yaml_path = os.environ.get("NOESIS_STATE_REG_YAML")
    if not yaml_path:
        raise RuntimeError(
            "NOESIS_STATE_REG_YAML must point to pilot.yaml before applying patch"
        )

    from lora_train import load_state_reg_config
    from state_reg import compute_state_reg

    cfg = load_state_reg_config(yaml_path)

    if cfg.mode == "off" or cfg.alpha == 0.0:
        # No import of light_rwkv → patch works as a documentation
        # / config-load smoke on CPU without deepspeed/CUDA deps.
        _PATCH_APPLIED = True
        return (
            f"state_reg patch loaded but INACTIVE "
            f"(mode={cfg.mode!r}, alpha={cfg.alpha}). "
            "Original training_step runs unchanged."
        )

    import math
    import torch
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    from rwkvt.infctx_module import BlockStateList
    from rwkvt.lightning_train import light_rwkv as _lr

    def training_step_with_state_reg(self, batch, batch_idx):  # type: ignore[no-redef]
        args = self.args
        T_train = args.chunk_ctx
        idx, targets = batch
        B, T = idx.shape
        C = args.n_embd
        H = args.dim_att // args.head_size_a
        assert C == H * args.head_size_a
        states = BlockStateList.create(
            args.n_layer, B, C, H, idx.device, self.model.emb.weight.dtype
        )

        from rwkvt.lightning_train.light_rwkv import L2Wrap

        def checkpointed_step(idx_c, targets_c, prev_loss,
                              last_shift_states, last_wkv_states,
                              prev_token_amount):
            logits, new_shift_states, new_wkv_states = self(
                idx_c, last_shift_states, last_wkv_states
            )
            current_token_amount = idx_c.shape[1]
            loss = self.criterion(
                logits.view(-1, logits.size(-1)), targets_c.reshape(-1)
            )
            if current_token_amount != 0:
                loss = L2Wrap.apply(loss, logits, current_token_amount)
            new_token_amount = prev_token_amount + current_token_amount
            if new_token_amount > 0:
                new_loss = prev_loss * (prev_token_amount / new_token_amount) + \
                    loss * (current_token_amount / new_token_amount)
            else:
                new_loss = prev_loss
            return new_loss, new_shift_states, new_wkv_states, new_token_amount

        total_loss = torch.tensor(
            0.0, dtype=self.model.emb.weight.dtype
        ).requires_grad_()
        token_amount = 0
        traj_per_L: dict[int, list[torch.Tensor]] = {L: [] for L in cfg.work_layers}

        for i in range(math.ceil(T / T_train)):
            total_loss, new_shift_states, new_wkv_states, token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, i * T_train:(i + 1) * T_train],
                targets[:, i * T_train:(i + 1) * T_train],
                total_loss,
                states.shift_states,
                states.wkv_states,
                token_amount,
                use_reentrant=False,
            )
            # Capture per-chunk wkv_state for state_reg BEFORE detach.
            # new_wkv_states shape: [N_layer, B, H, h, h]. Grab work layers.
            for L in cfg.work_layers:
                traj_per_L[L].append(new_wkv_states[L].unsqueeze(1))  # [B,1,H,h,h]
            states = BlockStateList(
                new_shift_states.clone().detach(),
                new_wkv_states.clone().detach(),
            )

        # Concatenate per-layer trajectories: dict[int -> [B, T_chunks, H, h, h]].
        lookup = {L: torch.cat(traj_per_L[L], dim=1) for L in cfg.work_layers}
        state_loss = compute_state_reg(lookup, cfg)
        return total_loss + cfg.alpha * state_loss

    _lr.RWKV.training_step = training_step_with_state_reg
    _PATCH_APPLIED = True
    return (
        f"state_reg patch APPLIED — mode={cfg.mode!r}, alpha={cfg.alpha}, "
        f"work_layers={cfg.work_layers}, "
        f"lambdas=(δ={cfg.lambda_delta}, κ={cfg.lambda_curvature})"
    )


def is_applied() -> bool:
    return _PATCH_APPLIED


__all__ = ["apply", "is_applied"]
