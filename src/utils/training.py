"""
Training utilities for NeuroFlow-H.

Includes EarlyStopping, LR schedulers, checkpoint management, and logging.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


class EarlyStopping:
    """
    Early stopping to terminate training when a monitored metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        mode: "min" (loss) or "max" (accuracy/PCC).
    """
    
    def __init__(
        self, 
        patience: int = 20, 
        min_delta: float = 1e-4, 
        mode: str = "max"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value.
            epoch: Current epoch number.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    LR linearly increases during warmup, then follows cosine decay.
    
    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps (or epochs).
        total_steps: Total number of steps (or epochs).
        min_lr: Minimum learning rate at end of cosine decay.
        last_epoch: The index of last epoch.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    scheduler: Optional[_LRScheduler] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        metrics: Dict of metric values.
        path: Save path.
        scheduler: Optional LR scheduler.
        extra: Optional extra data to save.
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if extra is not None:
        checkpoint.update(extra)
    
    torch.save(checkpoint, save_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = "cpu",
) -> Tuple[int, Dict[str, float]]:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path.
        model: PyTorch model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional scheduler to restore state.
        device: Device to map tensors to.
        
    Returns:
        Tuple of (epoch, metrics).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    
    return epoch, metrics


def setup_logger(
    name: str, 
    log_dir: Optional[str] = None, 
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a structured logger.
    
    Args:
        name: Logger name.
        log_dir: Optional directory for log file output.
        level: Logging level.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers if already configured
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all frameworks."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_tau(epoch: int, total_epochs: int, tau_start: float = 0.1, tau_end: float = 0.01) -> float:
    """Cosine-scheduled temperature for SoftCLIP."""
    progress = epoch / max(1, total_epochs)
    return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * progress))
