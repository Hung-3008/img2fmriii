"""
training_utils.py
=================
Training helpers: optimizer / scheduler construction.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer
# ═══════════════════════════════════════════════════════════════════════════


def _as_float_tuple(values: Any, length: int = 2) -> Tuple[float, ...]:
    if isinstance(values, (list, tuple)):
        if len(values) != length:
            raise ValueError(f"Expected {length} values, got {len(values)}")
        return tuple(float(v) for v in values)
    return tuple(float(values) for _ in range(length))


def _build_adamw(
    parameters: Iterable[nn.Parameter],
    opt_cfg: Dict[str, Any],
    base_lr: float,
) -> Tuple[Optimizer, str]:
    betas = _as_float_tuple(opt_cfg.get("betas", (0.9, 0.95)))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    eps = float(opt_cfg.get("eps", 1e-8))
    optimizer = torch.optim.AdamW(
        parameters, lr=base_lr, betas=betas, weight_decay=wd, eps=eps,
    )
    msg = f"Optimizer: AdamW  lr={base_lr}  betas={betas}  wd={wd}  eps={eps}"
    return optimizer, msg


# ═══════════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════════


def _build_lr_lambda(
    schedule_type: str,
    warmup_steps: int,
    decay_end_steps: int,
    final_ratio: float,
):
    """Return a ``lr_lambda`` callable for :class:`LambdaLR`.

    Supports ``"linear"``, ``"cosine"``, and ``"constant"`` schedules,
    each with an optional warmup plateau.
    """
    total_decay = max(decay_end_steps - warmup_steps, 1)

    if schedule_type == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay
            return 1.0 - (1.0 - final_ratio) * progress

    elif schedule_type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 1.0
            if step >= decay_end_steps:
                return final_ratio
            progress = (step - warmup_steps) / total_decay
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine

    elif schedule_type == "constant":
        def lr_lambda(step: int) -> float:
            return 1.0

    else:
        raise ValueError(
            f"Unsupported schedule '{schedule_type}'. "
            "Choose from ['linear', 'cosine', 'constant']."
        )

    return lr_lambda


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def build_optimizer_and_scheduler(
    parameters: Iterable[nn.Parameter],
    train_cfg: Dict[str, Any],
    steps_per_epoch: int,
    epochs: int,
) -> Tuple[Optimizer, LambdaLR, str, str]:
    """Build AdamW optimizer and LR scheduler from *train_cfg*.

    Returns ``(optimizer, scheduler, opt_msg, sched_msg)``.
    """
    opt_cfg = dict(train_cfg.get("optimizer", {}))
    base_lr = float(opt_cfg.get("lr", train_cfg.get("base_lr", 2e-4)))
    final_lr = float(train_cfg.get("final_lr", base_lr))
    final_ratio = final_lr / base_lr if base_lr > 0 else 1.0

    optimizer, opt_msg = _build_adamw(parameters, opt_cfg, base_lr)

    # Warmup steps
    warmup_steps = int(train_cfg.get("warmup_steps", 500))

    # Decay end
    decay_end_steps = int(
        train_cfg.get("decay_end_steps", epochs * steps_per_epoch)
    )
    warmup_steps = max(warmup_steps, 0)
    decay_end_steps = max(decay_end_steps, warmup_steps)

    schedule_type = train_cfg.get("schedule_type", "cosine").lower()
    lr_lambda = _build_lr_lambda(
        schedule_type, warmup_steps, decay_end_steps, final_ratio,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    sched_msg = (
        f"Scheduler: {schedule_type}  warmup={warmup_steps}  "
        f"decay_end={decay_end_steps}  final_lr={final_lr}"
    )
    return optimizer, scheduler, opt_msg, sched_msg
