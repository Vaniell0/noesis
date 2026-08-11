"""G1h + ROSA additive block for pseudo-RWKV-8 experiments.

Adds a small ROSA branch alongside WKV in each G1h block.
Output weights start at zero → zero initial contribution → safe to add
to any existing G1h checkpoint without changing behaviour.

Architecture per block:
    x_wkv  = WKV(LN1(x))          # existing G1h time-mixing
    x_rosa = ROSA(LN_rosa(x))      # new additive branch
    x      = x + x_wkv + x_rosa   # combined
    x      = x + FFN(LN2(x))      # existing G1h channel-mixing

ROSA projections use n_rosa << n_embd to keep memory budget manageable.
Default n_rosa=256 → 4-bit ROSA has 64 heads (vs 640 for full 2560).

Usage (inference / eval):
    from experiments.rosa_probe.g1h_rosa_block import RosaAddon, attach_rosa
    addons = attach_rosa(g1h_model, n_rosa=256)
    # addons is a list of RosaAddon modules, one per layer
    # forward hook already installed — just run model normally

Usage (training):
    # addons parameters are the only trainables (output_proj zeroed)
    optimizer = torch.optim.AdamW(
        [p for a in addons for p in a.parameters()],
        lr=1e-4,
    )
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


# ── ROSA operator selection ───────────────────────────────────────────────────

def _get_rosa_fn():
    try:
        from rosa_soft import rosa_soft, BUILD_CAPABILITIES
        if BUILD_CAPABILITIES.rosa_soft_cuda:
            return rosa_soft
    except ImportError:
        pass
    from rosa_soft import rosa_soft_reference
    return rosa_soft_reference


_rosa_fn = _get_rosa_fn()


# ── ROSA addon module ─────────────────────────────────────────────────────────

class RosaAddon(nn.Module):
    """Trainable ROSA branch to be added alongside one G1h block's WKV output.

    n_embd   : hidden size of G1h (2560 for 2.9B)
    n_rosa   : inner dim for ROSA projections (default 256; 4-bit → 64 heads)
    layer_id : used for parameter initialization scaling
    n_layer  : total layers (used for init scaling)
    qk_bits  : bits per head in 4-bit ROSA (default 4)
    """

    def __init__(
        self,
        n_embd: int,
        n_rosa: int,
        layer_id: int,
        n_layer: int,
        qk_bits: int = 4,
        max_suffix_length: int = 32,
    ) -> None:
        super().__init__()
        assert n_rosa % qk_bits == 0, "n_rosa must be divisible by qk_bits"
        self.n_embd   = n_embd
        self.n_rosa   = n_rosa
        self.qk_bits  = qk_bits
        self.n_heads  = n_rosa // qk_bits
        self.max_suffix_length = max_suffix_length

        self.ln = nn.LayerNorm(n_embd)

        # time-shift (same pattern as WKV)
        self.x_q = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_k = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.x_v = nn.Parameter(torch.zeros(1, 1, n_embd))

        self.proj_q = nn.Linear(n_embd, n_rosa, bias=False)
        self.proj_k = nn.Linear(n_embd, n_rosa, bias=False)
        self.proj_v = nn.Linear(n_embd, n_rosa, bias=False)
        self.out    = nn.Linear(n_rosa, n_embd, bias=False)

        self._init_weights(layer_id, n_layer)

    def _init_weights(self, layer_id: int, n_layer: int) -> None:
        c = self.n_embd
        ratio = 1.0 - (layer_id / n_layer)
        ddd = torch.arange(c, dtype=torch.float32).div_(c).view(1, 1, c)
        self.x_q.data.copy_(1.0 - torch.pow(ddd, 0.9 * ratio))
        self.x_k.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio))
        self.x_v.data.copy_(1.0 - torch.pow(ddd, 0.7 * ratio))
        inv_sqrt = 1.0 / math.sqrt(c)
        self.proj_q.weight.data.uniform_(-0.5 * inv_sqrt,  0.5 * inv_sqrt)
        self.proj_k.weight.data.uniform_(-0.05 * inv_sqrt, 0.05 * inv_sqrt)
        self.proj_v.weight.data.uniform_(-0.5 * inv_sqrt,  0.5 * inv_sqrt)
        # Zero output → starts contributing nothing to existing model behaviour
        self.out.weight.data.zero_()

    def forward(self, x: Tensor, prev_x: Tensor | None = None) -> Tensor:
        """
        x      : [B, T, n_embd]
        prev_x : [B, n_embd] last token from previous chunk (optional)
        returns: [B, T, n_embd]
        """
        B, T, C = x.shape
        xn = self.ln(x)

        if prev_x is None:
            shifted = torch.cat([xn[:, :1, :], xn[:, :-1, :]], dim=1)
        else:
            shifted = torch.cat([prev_x.unsqueeze(1), xn[:, :-1, :]], dim=1)
        delta = shifted - xn

        q = self.proj_q(xn + delta * self.x_q)  # [B, T, n_rosa]
        k = self.proj_k(xn + delta * self.x_k)
        v = self.proj_v(xn + delta * self.x_v)

        # reshape to [B, T, H, D] for rosa_fn
        q = q.view(B, T, self.n_heads, self.qk_bits)
        k = k.view(B, T, self.n_heads, self.qk_bits)
        v = v.view(B, T, self.n_heads, self.qk_bits)

        y = _rosa_fn(q, k, v, max_suffix_length=self.max_suffix_length)  # [B, T, H, D]
        y = y.view(B, T, self.n_rosa)
        return self.out(y)


# ── Attach ROSA to a loaded G1h model ────────────────────────────────────────

def attach_rosa(
    model: nn.Module,
    n_rosa: int = 256,
    qk_bits: int = 4,
    max_suffix_length: int = 32,
    device: str | torch.device | None = None,
) -> List[RosaAddon]:
    """Register RosaAddon hooks on every block of a G1h model.

    Detects n_embd and n_layer from the model's embedding weight.
    Returns the list of addon modules (also accessible as model._rosa_addons).

    The forward hook adds ROSA output after WKV inside each block's forward.
    IMPORTANT: hooks intercept `blocks[i].forward`, not the full model forward.
    """
    # Detect dims
    emb = (model.emb if hasattr(model, "emb")
           else getattr(model, "embedding", None))
    if emb is None:
        raise ValueError("Cannot find embedding layer in model")
    n_embd = emb.weight.shape[1]

    blocks = model.blocks
    n_layer = len(blocks)

    if device is None:
        device = next(model.parameters()).device

    addons: List[RosaAddon] = []
    for i, block in enumerate(blocks):
        addon = RosaAddon(
            n_embd=n_embd,
            n_rosa=n_rosa,
            layer_id=i,
            n_layer=n_layer,
            qk_bits=qk_bits,
            max_suffix_length=max_suffix_length,
        ).to(device=device, dtype=next(model.parameters()).dtype)
        addons.append(addon)

        # Store on the block for hook access
        block._rosa_addon = addon

        # Forward hook: intercepts block output and adds ROSA
        def _make_hook(idx: int):
            def _hook(module, args, output):
                # output may be tensor or tuple
                if isinstance(output, tuple):
                    x_out = output[0]
                    rest = output[1:]
                else:
                    x_out = output
                    rest = None
                # ROSA input = block's input hidden state
                # args[0] is the input tensor to the block
                x_in = args[0]
                rosa_out = module._rosa_addon(x_in)
                x_out = x_out + rosa_out
                return (x_out, *rest) if rest is not None else x_out
            return _hook

        block.register_forward_hook(_make_hook(i))

    model._rosa_addons = addons
    print(f"[rosa] attached {n_layer} RosaAddon blocks "
          f"(n_rosa={n_rosa}, heads={n_rosa//qk_bits}, "
          f"params/block={sum(p.numel() for p in addons[0].parameters()):,})")
    return addons
