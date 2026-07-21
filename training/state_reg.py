"""State-regularization loss hook — STUB implementation.

The concrete L_state formula is gated on A0.5 causal-intervention results
(see `experiments/A0_state_probe/A05_intervention_plan.md` and
`plans/cosmic-purring-cocke.md` §Step 8). Until that verdict is in,
this module returns 0.0 unconditionally and every training run is
effectively pure action-cloning with α=0.

We keep the module wired into the pipeline anyway (per P12 —
reversibility): switching state-reg on later is one config edit.

When A0.5 lands, replace `compute_state_reg` with one of:
  * encourage_motion — `mean_over_work_layers(‖s_t − s_{t-1}‖ · direction)`
  * penalize_still  — `max(0, δ_min − ‖s_t − s_{t-1}‖)` on tool_use steps
  * per_layer_weighted — layer-mask from A0.5-identified work layers

See design decision §7 in the cosmic plan for the branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Config values accepted; enforced downstream in `lora_train.py`.
VALID_MODES: tuple[str, ...] = (
    "off",
    "encourage_motion",
    "penalize_still",
    "per_layer_weighted",
)


@dataclass
class StateRegConfig:
    mode: str = "off"
    alpha: float = 0.0
    delta_min: float = 0.0  # penalize_still branch only
    layer_mask: tuple[int, ...] = ()  # per_layer_weighted branch only

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"state_reg_mode={self.mode!r} not in {VALID_MODES}"
            )


def compute_state_reg(
    wkv_per_layer: Any,
    tool_use_mask: Any,
    config: StateRegConfig,
) -> float:
    """Return scalar L_state (float, pre-alpha).

    STUB: returns 0.0 for every branch. Real implementation waits on A0.5.

    Args:
      wkv_per_layer: tensor / list-of-tensors of state activations per
          layer, per timestep — same shape convention as
          `experiments/A0_state_probe/probe.py:_extract_wkv_per_layer`.
      tool_use_mask: boolean tensor selecting positions inside
          `<tool_use>…</tool_use>` spans (Design §6 loss mask).
      config: StateRegConfig from `training/config/pilot.yaml`.

    Returns:
      Scalar loss contribution *before* multiplying by α. Callers do
      `L_total = L_ce + config.alpha * L_state`.
    """
    if config.mode == "off":
        return 0.0
    # Non-off modes reserved for post-A0.5 implementation; today they still
    # return 0.0 so a config typo doesn't silently activate a non-existent
    # loss and confuse post-mortems.
    return 0.0
