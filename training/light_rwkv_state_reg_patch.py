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

Contract: state trajectory is *per-chunk* state, not per-token. Larger
``chunk_ctx`` yields a sparser trajectory (still valid for
``compute_state_reg`` — it treats the time axis as opaque). Do NOT set
``chunk_ctx=1`` — it causes OOM on 4096-ctx sequences.

Env vars (set by driver script before importing this module):
  ``RWKV_TRAIN_TYPE``        — must be ``infctx`` (else the patch no-ops).
  ``NOESIS_STATE_REG_YAML``  — path to ``training/config/pilot.yaml``.
  ``NOESIS_LOG_STEPS``       — log state_loss every N opt-steps (default 100).
  ``NOESIS_DBG``             — if set, enables verbose per-chunk diagnostics.
"""

from __future__ import annotations

import json
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
    ``RWKV_TRAIN_TYPE`` is not ``infctx``.
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
<<<<<<< Updated upstream
    from state_reg import compute_state_reg
=======
    from state_reg import compute_state_reg, compute_h12bi_aux
>>>>>>> Stashed changes
    import yaml as _yaml

    cfg = load_state_reg_config(yaml_path)
    # ε-mask params (step 9): read outside_epsilon from yaml
    _raw_cfg = _yaml.safe_load(open(yaml_path))
    _sr_raw = _raw_cfg.get("state_reg", {})
    _eps_out = float(_sr_raw.get("outside_epsilon", 0.0))
    _eps_in  = float(_sr_raw.get("inside_epsilon",  1.0))
<<<<<<< Updated upstream
=======
    # H12b.i aux_loss config
    _h12bi_enabled = bool(_raw_cfg.get("h12b_i_enabled", False))
    _h12bi_weight  = float(_raw_cfg.get("h12b_i_aux_weight", 1e-4))
>>>>>>> Stashed changes

    if cfg.mode == "off" or cfg.alpha == 0.0:
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

    _log_steps = int(os.environ.get("NOESIS_LOG_STEPS", "100"))
    _dbg = bool(os.environ.get("NOESIS_DBG"))
    # Track last logged opt-step to avoid duplicate logging within the same
    # accumulation window (training_step fires once per micro-batch).
    _last_logged = [-1]

    def training_step_with_state_reg(self, batch, batch_idx):  # type: ignore[no-redef]
        args = self.args
        T_train = args.chunk_ctx
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            idx, targets, _mask, _state_mask = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            idx, targets, _mask = batch
            _state_mask = None
        else:
            idx, targets = batch
            _mask = None
            _state_mask = None
        B, T = idx.shape
        C = args.n_embd
        H = args.dim_att // args.head_size_a
        assert C == H * args.head_size_a
        states = BlockStateList.create(
            args.n_layer, B, C, H, idx.device, self.model.emb.weight.dtype
        )

        from rwkvt.lightning_train.light_rwkv import L2Wrap

        _dbg_printed = [False]

        def checkpointed_step(idx_c, targets_c, prev_loss,
                              last_shift_states, last_wkv_states,
                              prev_token_amount):
            logits, new_shift_states, new_wkv_states = self(
                idx_c, last_shift_states, last_wkv_states
            )
            current_token_amount = int((targets_c != -100).sum().item())
            if current_token_amount == 0:
                _raw_loss = torch.zeros(
                    (), dtype=prev_loss.dtype, device=idx_c.device
                )
            else:
                _raw_loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets_c.reshape(-1)
                )
            if _dbg and not _dbg_printed[0]:
                _dbg_printed[0] = True
                print(
                    f"[state_reg_dbg] step={self.global_step} "
                    f"raw_ce={_raw_loss.detach().float().item():.4f} "
                    f"logits_nan={logits.isnan().any().item()} "
                    f"wkv_nan={new_wkv_states.isnan().any().item()} "
                    f"supervised={current_token_amount}"
                )
            loss = _raw_loss
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
        # ε-mask: per-chunk alpha scalar based on <think> token fraction
        chunk_alphas: list[float] = []

        for i in range(math.ceil(T / T_train)):
            chunk_slice = slice(i * T_train, (i + 1) * T_train)
            total_loss, new_shift_states, new_wkv_states, token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, chunk_slice],
                targets[:, chunk_slice],
                total_loss,
                states.shift_states,
                states.wkv_states,
                token_amount,
                use_reentrant=False,
            )
            for L in cfg.work_layers:
                traj_per_L[L].append(new_wkv_states[L].unsqueeze(1))
            states = BlockStateList(
                new_shift_states.clone().detach(),
                new_wkv_states.clone().detach(),
            )
            # Compute ε-scaled alpha for this chunk
            if _eps_out > 0.0 and _state_mask is not None:
                chunk_sm = _state_mask[:, chunk_slice].float()
                think_frac = chunk_sm.mean().item()
                # effective_mask formula: span * (1 - ε_out) + ε_out
                chunk_alpha = cfg.alpha * (think_frac * (1.0 - _eps_out) + _eps_out)
            else:
                chunk_alpha = cfg.alpha
            chunk_alphas.append(chunk_alpha)

        lookup = {L: torch.cat(traj_per_L[L], dim=1) for L in cfg.work_layers}
        state_loss = compute_state_reg(lookup, cfg)
        # Apply mean ε-scaled alpha across chunks (approximation: one alpha per seq)
        mean_alpha = sum(chunk_alphas) / max(len(chunk_alphas), 1)

        # Periodic state_loss logging — written once per opt-step (not per micro-batch).
        _step = self.global_step
        if _log_steps > 0 and _step % _log_steps == 0 and _step != _last_logged[0]:
            _last_logged[0] = _step
            _sl = state_loss.detach().float().item()
            _tl = (total_loss + mean_alpha * state_loss).detach().float().item()
            _ce = total_loss.detach().float().item()
            print(
                f"[state_reg] step={_step} "
                f"ce={_ce:.4f} state_loss={_sl:.4f} alpha={mean_alpha:.4f} total={_tl:.4f}"
            )
            _log_path = os.path.join(args.proj_dir, "state_loss.jsonl")
            with open(_log_path, "a") as _f:
                json.dump({
                    "step": _step, "ce": _ce, "state_loss": _sl,
                    "alpha": mean_alpha, "total": _tl,
                }, _f)
                _f.write("\n")

<<<<<<< Updated upstream
        return total_loss + mean_alpha * state_loss
=======
        final_loss = total_loss + mean_alpha * state_loss

        if _h12bi_enabled:
            _lora_A = {n[:-len("lora_A")]: p for n, p in self.model.named_parameters() if n.endswith("lora_A")}
            _lora_B = {n[:-len("lora_B")]: p for n, p in self.model.named_parameters() if n.endswith("lora_B")}
            lora_pairs = [(a, _lora_B[k]) for k, a in _lora_A.items() if k in _lora_B]
            if lora_pairs:
                final_loss = final_loss + _h12bi_weight * compute_h12bi_aux(lora_pairs)

        return final_loss
>>>>>>> Stashed changes

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
