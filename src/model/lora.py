"""
lora.py
=======
Minimal LoRA (low-rank adaptation) wrapper for ``nn.Linear`` layers, used by the
few-shot cross-subject trainer to unfreeze the *last* trunk block(s) cheaply.

A wrapped layer computes ``base(x) + (x @ Aᵀ @ Bᵀ) * (alpha/rank)`` where the
frozen ``base`` keeps its pretrained weights and only ``A`` (rank, in) and
``B`` (out, rank) are trainable. ``B`` starts at zero so the adapted model is
identical to the pretrained one at step 0 (no warm-up shock).

Isolated on purpose: nothing here touches single-subject or trunk training.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B stays 0 → Δ=0 at init
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (self.drop(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + delta * self.scaling


def _inject(parent: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    """Recursively replace every ``nn.Linear`` under *parent* with LoRALinear."""
    n = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, nn.Linear):
            setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
            n += 1
        else:
            n += _inject(child, rank, alpha, dropout)
    return n


def apply_lora_to_blocks(blocks: nn.ModuleList, n_last: int,
                         targets: Iterable[str] = ("attn", "mlp"),
                         rank: int = 8, alpha: float = 16.0,
                         dropout: float = 0.0) -> int:
    """Inject LoRA into the last ``n_last`` trunk blocks.

    Only the sub-modules named in *targets* (default the self-attention and the
    feed-forward) are adapted — conditioning (adaLN) and norms are left frozen.
    Returns the number of Linear layers wrapped.
    """
    depth = len(blocks)
    n_last = max(1, min(n_last, depth))
    total = 0
    for blk in blocks[depth - n_last:]:
        for tname in targets:
            sub = getattr(blk, tname, None)
            if sub is not None:
                total += _inject(sub, rank, alpha, dropout)
    return total
