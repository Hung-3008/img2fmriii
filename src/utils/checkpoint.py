"""
checkpoint.py
=============
Checkpoint save / load / rolling-last utilities.

Checkpoint dict layout::

    {
        "model":       wrapper.state_dict(),
        "opt":         optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "train_steps": int,
        "epoch":       int,
        "best_val_pcc": float,
        **extra_kwargs,
    }
"""

from __future__ import annotations

import logging
import os
from glob import glob
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str,
    wrapper: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_steps: int,
    epoch: int,
    best_val_pcc: float,
    **extra: Any,
) -> None:
    """Save a training checkpoint to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "model": wrapper.state_dict(),
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_steps": train_steps,
        "epoch": epoch,
        "best_val_pcc": best_val_pcc,
    }
    payload.update(extra)
    torch.save(payload, path)
    logger.info("Saved checkpoint: %s", path)


def load_checkpoint(
    path: str,
    wrapper: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint and restore model / optimizer / scheduler state.

    Returns a dict with ``train_steps``, ``epoch``, ``best_val_pcc``.
    Legacy checkpoints that still contain an ``"ema"`` key are accepted —
    the EMA weights are simply ignored.
    """
    ckpt = torch.load(path, map_location=device)
    wrapper.load_state_dict(ckpt["model"])
    if optimizer is not None and "opt" in ckpt:
        optimizer.load_state_dict(ckpt["opt"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    info = {
        "train_steps": int(ckpt.get("train_steps", 0)),
        "epoch": int(ckpt.get("epoch", 0)),
        "best_val_pcc": float(ckpt.get("best_val_pcc", -1.0)),
    }
    logger.info(
        "Loaded checkpoint from %s  (step=%d, epoch=%d, best_pcc=%.4f)",
        path, info["train_steps"], info["epoch"], info["best_val_pcc"],
    )
    del ckpt
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return info


def save_rolling_last(
    ckpt_dir: str,
    wrapper: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_steps: int,
    epoch: int,
    best_val_pcc: float,
) -> str:
    """Delete old ``last-*.pt`` files, then save a new one.

    Returns the path of the newly saved checkpoint.
    """
    for old in glob(os.path.join(ckpt_dir, "last-*.pt")):
        os.remove(old)
    path = os.path.join(ckpt_dir, f"last-{train_steps}.pt")
    save_checkpoint(
        path, wrapper, optimizer, scheduler,
        train_steps, epoch, best_val_pcc,
    )
    return path


def find_last_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Find the most recent ``last-*.pt`` checkpoint, or *None*."""
    candidates = glob(os.path.join(ckpt_dir, "last-*.pt"))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda p: int(
            os.path.basename(p).replace("last-", "").replace(".pt", "")
        ),
    )
