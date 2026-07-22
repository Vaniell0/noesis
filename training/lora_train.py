"""A1 pilot LoRA training driver — state_reg-aware.

Skeleton driver that wires the state-regularization loss (``state_reg.py``)
into a per-step training call. The vendored ``RWKV-PEFT`` (Lightning) loop
in ``training/rwkv-peft/rwkvt/lightning_train/light_rwkv.py`` remains the
production trainer; this file provides the two integration primitives
Lightning is missing:

- ``StateCapture`` — a context manager that installs forward hooks on
  every RWKV-7 attention module and collects per-timestep WKV state
  into ``wkv_per_layer`` tensors of shape ``[B, T, H, h, h]``. Same
  read pattern as ``experiments/A0_state_probe/probe.py::_extract_wkv_per_layer``.
- ``train_step`` — a single forward+state-reg+backward step, isolated
  so it can be unit-tested against a mock model without needing the
  full Lightning stack or CUDA kernels.

## RWKV-PEFT hookup (documentation, not code here)

The vendored trainer runs in one of four modes selected by
``RWKV_TRAIN_TYPE``: default / ``state`` / ``infctx`` / ``fullstate``
(see ``rwkvt/rwkv7/att.py``).

Only the ``infctx`` path exposes the per-chunk WKV state at the
Python layer — ``RWKV_Tmix_x070_infctx.forward`` returns
``TimeMixState(shift_state, wkv_state)`` and Lightning threads them
across chunks in ``light_rwkv.py::training_step`` around line 195.

For A1 pilot with ``state_reg`` we run:

    RWKV_TRAIN_TYPE=infctx  chunk_size=1

so that each chunk covers exactly one token — the returned ``wkv_state``
after chunk ``t`` is the WKV state at timestep ``t``. Concatenating
those across the sequence gives the ``[B, T, H, h, h]`` trajectory
``compute_state_reg`` expects.

**Concrete patch (apply in ``light_rwkv.py::training_step``, infctx
branch):**

1. Import ``from training.state_reg import compute_state_reg, StateRegConfig``.
2. Instantiate ``StateRegConfig`` once from the pilot YAML (mode, alpha,
   work_layers, layer_weights).
3. Around the ``torch_checkpoint`` chunk loop, install a
   ``StateCapture(model, cfg.work_layers)`` context manager (this file).
4. After the chunked forward completes, call
   ``L_state = compute_state_reg(capture.per_layer(), cfg)``.
5. ``L_total = L_CE_SFT + cfg.alpha * L_state`` before the existing
   ``.backward()``. Preserve Lightning's gradient-accumulation
   averaging by dividing ``L_total`` by the same factor Lightning
   already divides ``L_CE_SFT`` by.

This driver deliberately does NOT modify vendored ``rwkv-peft/`` code
(the tree is a pinned commit, treated as external). The patch above
is the intended integration point; the smoke test in
``tests/test_state_reg_hookup.py`` validates the same interface on a
mock micro-model that doesn't require CUDA kernels.

## What this driver is NOT

- Not a Lightning replacement. If you want to train A1 for real, run
  the Lightning trainer under ``training/rwkv-peft/train.py`` after
  applying the patch above.
- Not GPU-runnable here (CUDA kernels compile on first use; laptop
  CPU-only). The smoke test is the only currently-runnable exercise.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn

from state_reg import StateRegConfig, compute_state_reg


# --------------------------------------------------------------------------- #
# State capture
# --------------------------------------------------------------------------- #

class StateCapture:
    """Forward-hook harness that collects per-timestep WKV state.

    Two capture protocols are supported (auto-detected):

    - ``attr`` — the attention module writes ``self._captured_wkv``
      (Tensor of shape ``[B, T, H, h, h]``) during forward. Used by
      the mock model in the smoke test and by any wrapper that
      pre-materialises the trajectory.
    - ``infctx`` — the attention module's forward returns a tuple
      whose last element is a ``TimeMixState`` (or any object with a
      ``.wkv_state`` attribute of shape ``[B, H, h, h]``). Used with
      ``RWKV_TRAIN_TYPE=infctx`` at ``chunk_size=1``: each chunk yields
      one timestep of state; we stack them across chunks.

    The layer discovery walks the model tree looking for modules whose
    class name contains ``Tmix`` or that expose a ``layer_id`` attribute
    matching a member of ``work_layers``. Non-work layers are skipped
    to avoid the memory cost of tracking irrelevant trajectories.

    Usage::

        with StateCapture(model, work_layers=(12, 16, 20)) as cap:
            logits = model(input_ids)          # or chunked forward
        wkv = cap.per_layer()                  # list[Tensor], one per work layer
        state_loss = compute_state_reg(wkv, state_reg_cfg)
    """

    def __init__(self, model: nn.Module, work_layers: Iterable[int]) -> None:
        self.model = model
        self.work_layers = tuple(sorted(set(work_layers)))
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # Per-layer state stack — list of Tensor chunks; concatenated on read.
        self._chunks: dict[int, list[torch.Tensor]] = {L: [] for L in self.work_layers}

    def _discover_work_modules(self) -> dict[int, nn.Module]:
        found: dict[int, nn.Module] = {}
        for _, mod in self.model.named_modules():
            layer_id = getattr(mod, "layer_id", None)
            if layer_id is None:
                continue
            if layer_id in self.work_layers and "Tmix" in type(mod).__name__:
                found[layer_id] = mod
        return found

    def _make_hook(self, layer_id: int) -> Callable:
        chunks = self._chunks[layer_id]

        def hook(module: nn.Module, inputs: tuple, output) -> None:
            # Protocol A: module set _captured_wkv on self.
            cap = getattr(module, "_captured_wkv", None)
            if cap is not None:
                chunks.append(cap)
                return

            # Protocol B: TimeMixState in output tuple.
            if isinstance(output, tuple):
                for item in reversed(output):
                    wkv = getattr(item, "wkv_state", None)
                    if wkv is not None:
                        # wkv shape [B, H, h, h] per chunk; add T=1 axis for
                        # concatenation across chunks.
                        chunks.append(wkv.unsqueeze(1))
                        return

        return hook

    def __enter__(self) -> "StateCapture":
        modules = self._discover_work_modules()
        for L, mod in modules.items():
            self._handles.append(mod.register_forward_hook(self._make_hook(L)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def per_layer(self) -> list[torch.Tensor]:
        """Return concatenated per-layer trajectories, ordered by
        ``config.work_layers`` (the ``StateRegConfig`` iteration order).

        Each entry is a Tensor of shape ``[B, T, H, h, h]``. If the
        model wrote a single pre-materialised trajectory (protocol A),
        the list holds one tensor per work layer directly. If the
        model emitted per-chunk states (protocol B), the chunks are
        concatenated along ``dim=1``.
        """
        out: list[torch.Tensor] = []
        for L in self.work_layers:
            chunks = self._chunks[L]
            if not chunks:
                raise RuntimeError(
                    f"StateCapture: no state captured for layer {L}. "
                    "Verify the attention module exposes _captured_wkv "
                    "or returns a TimeMixState in its output tuple."
                )
            if len(chunks) == 1:
                out.append(chunks[0])
            else:
                out.append(torch.cat(chunks, dim=1))
        return out


# --------------------------------------------------------------------------- #
# Per-step trainer primitive
# --------------------------------------------------------------------------- #

@dataclass
class StepOutput:
    ce_loss: torch.Tensor
    state_reg_loss: torch.Tensor
    total_loss: torch.Tensor


def train_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    ce_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    state_reg_cfg: StateRegConfig,
    forward: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> StepOutput:
    """One forward + state-reg + backward-ready step.

    Args:
      model: The RWKV(-like) module. Must expose per-timestep WKV
        state via ``StateCapture`` (see docstring above).
      input_ids: ``[B, T]`` long tensor.
      labels: ``[B, T]`` long tensor; same shape as input_ids.
      ce_loss_fn: ``(logits, labels) -> scalar Tensor``. Loss masking
        is the caller's responsibility.
      state_reg_cfg: config from ``state_reg.py``. If ``mode="off"``
        or ``alpha == 0``, state capture is skipped entirely (no hook
        overhead) and ``state_reg_loss`` is a zero tensor.
      forward: optional custom forward callable
        ``(model, input_ids) -> logits``. Defaults to
        ``model(input_ids)``. Chunked-forward callers pass their own
        driver here.

    Returns:
      ``StepOutput`` with ``total_loss`` ready for ``.backward()``.
      Caller owns the optimizer step and grad clearing.
    """
    fwd = forward or (lambda m, x: m(x))

    if state_reg_cfg.mode == "off" or state_reg_cfg.alpha == 0.0:
        logits = fwd(model, input_ids)
        ce = ce_loss_fn(logits, labels)
        zero = torch.zeros((), device=ce.device, dtype=ce.dtype)
        return StepOutput(ce_loss=ce, state_reg_loss=zero, total_loss=ce + zero)

    with StateCapture(model, state_reg_cfg.work_layers) as cap:
        logits = fwd(model, input_ids)
    ce = ce_loss_fn(logits, labels)

    wkv_per_layer_all = cap.per_layer()
    # Reorder into a full list-of-n_layer where non-work slots are
    # placeholders; compute_state_reg only indexes work_layers.
    # Cheaper: pass a dict-like whose __getitem__ resolves work indices.
    class _LayerLookup:
        def __init__(self, work_layers, tensors):
            self._m = dict(zip(work_layers, tensors))

        def __getitem__(self, L: int) -> torch.Tensor:
            return self._m[L]

    lookup = _LayerLookup(state_reg_cfg.work_layers, wkv_per_layer_all)
    state_loss = compute_state_reg(lookup, state_reg_cfg)
    total = ce + state_reg_cfg.alpha * state_loss
    return StepOutput(ce_loss=ce, state_reg_loss=state_loss, total_loss=total)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_state_reg_config(pilot_yaml_path: str) -> StateRegConfig:
    """Read ``training/config/pilot.yaml``'s ``state_reg`` block.

    Minimal YAML parser (no PyYAML dependency): we only need to
    extract a small handful of keys, all at fixed indentation. If the
    file layout drifts, fall back to raising with a clear message so
    misconfig can't silently disable the regularizer.
    """
    import yaml  # PyYAML is a transitive dep of RWKV-PEFT; safe assumption

    with open(pilot_yaml_path) as f:
        cfg = yaml.safe_load(f)

    sr = cfg.get("state_reg", {}) or {}
    kwargs: dict = {}
    if "mode" in sr:
        kwargs["mode"] = str(sr["mode"])
    if "alpha" in sr:
        kwargs["alpha"] = float(sr["alpha"])
    if "lambda_delta" in sr:
        kwargs["lambda_delta"] = float(sr["lambda_delta"])
    if "lambda_curvature" in sr:
        kwargs["lambda_curvature"] = float(sr["lambda_curvature"])
    if "work_layers" in sr and sr["work_layers"]:
        kwargs["work_layers"] = tuple(int(x) for x in sr["work_layers"])
    return StateRegConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Not a main. The Lightning trainer under rwkv-peft/train.py is the
# production entry point; this file exposes the reusable primitives.
# --------------------------------------------------------------------------- #

__all__ = [
    "StateCapture",
    "StepOutput",
    "train_step",
    "load_state_reg_config",
]
