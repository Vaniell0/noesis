#!/usr/bin/env python3
"""clipo_head.py — MLP projection head for CLIPO contrastive reward.

Projects flattened WKV state (n_layers × H × h × h) into a fixed-dim
embedding space for InfoNCE computation. Trained from scratch alongside
the policy (gradients flow through GRPO reward).

Input dim is inferred from the first batch (lazy init) so it works
regardless of model size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CLIPOHead(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self._built = False
        self._layers: Optional[nn.Sequential] = None

    def _build(self, in_dim: int) -> None:
        self._layers = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, self.out_dim),
        )
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[-1])
            self._layers = self._layers.to(x.device)
        return self._layers(x)
