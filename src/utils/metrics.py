"""
metrics.py
==========
Evaluation metrics for fMRI synthesis.

Two levels of API:

* **Torch-based** (fast, GPU-friendly) — ``masked_mse``, ``pearson_corr_per_sample``
  Used during training for inline validation.

* **NumPy/SciPy-based** — ``compute_full_metrics``
  Used in post-training evaluation for comprehensive per-voxel and per-image
  Pearson correlations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════════
# Torch metrics (for training / inline eval)
# ═══════════════════════════════════════════════════════════════════════════


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE computed only on real (non-padded) voxels.

    Args:
        pred, target: ``(B, C, H, W)`` — predicted and ground-truth fMRI.
        pad_mask: ``(V_pad,)`` boolean — ``True`` for real voxels.

    Returns:
        Scalar MSE tensor.
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    mask = pad_mask.to(pred.device).float()
    diff_sq = (pred_flat - target_flat) ** 2
    return (diff_sq * mask.unsqueeze(0)).sum() / (mask.sum() * B)


@torch.no_grad()
def pearson_corr_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample (profile) Pearson *r* between predicted and target fMRI.

    Args:
        pred, target: ``(B, C, H, W)``
        pad_mask: ``(V_pad,)`` boolean

    Returns:
        ``(B,)`` tensor of Pearson *r* values.
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)[:, pad_mask]
    target_flat = target.reshape(B, -1)[:, pad_mask]
    p = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    t = target_flat - target_flat.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1))
    return num / (den + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# NumPy / SciPy metrics (for full evaluation)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EvalMetrics:
    """Container for all evaluation results."""

    # Single-trial
    voxel_r: np.ndarray           # (V,) per-voxel Pearson r
    profile_r: np.ndarray         # (N,) per-sample profile Pearson r
    mse: float

    # Derived scalars
    mean_voxel_r: float
    median_voxel_r: float
    mean_profile_r: float

    # Image-level (rep-averaged)
    img_voxel_r: Optional[np.ndarray] = None
    img_profile_r: Optional[np.ndarray] = None
    mean_img_voxel_r: Optional[float] = None
    mean_img_profile_r: Optional[float] = None


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns 0.0 on degenerate input."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    r, _ = stats.pearsonr(a, b)
    return r if np.isfinite(r) else 0.0


def compute_full_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    n_voxels: int,
    n_reps: int = 1,
    n_images: Optional[int] = None,
) -> EvalMetrics:
    """Compute comprehensive evaluation metrics.

    Args:
        preds:    ``(N, V)`` predicted voxels (already unpadded).
        targets:  ``(N, V)`` ground-truth voxels.
        n_voxels: Number of real voxels (should equal ``V``).
        n_reps:   Repetitions per image in the test set.
        n_images: Number of unique images; inferred as ``N // n_reps`` if
                  *None*.

    Returns:
        :class:`EvalMetrics` dataclass.
    """
    n_samples, V = preds.shape
    assert V == n_voxels

    # --- Per-voxel Pearson r ---
    voxel_r = np.array(
        [_safe_pearsonr(preds[:, v], targets[:, v]) for v in range(V)],
        dtype=np.float64,
    )

    # --- Profile Pearson r ---
    profile_r = np.array(
        [_safe_pearsonr(preds[i], targets[i]) for i in range(n_samples)],
        dtype=np.float64,
    )

    # --- MSE ---
    mse = float(np.mean((preds - targets) ** 2))

    metrics = EvalMetrics(
        voxel_r=voxel_r,
        profile_r=profile_r,
        mse=mse,
        mean_voxel_r=float(np.mean(voxel_r)),
        median_voxel_r=float(np.median(voxel_r)),
        mean_profile_r=float(np.mean(profile_r)),
    )

    # --- Image-level (rep-averaged) ---
    if n_images is None:
        n_images = n_samples // n_reps

    if n_reps > 1 and n_images * n_reps == n_samples:
        preds_img = preds.reshape(n_images, n_reps, V).mean(axis=1)
        targets_img = targets.reshape(n_images, n_reps, V).mean(axis=1)

        metrics.img_profile_r = np.array(
            [_safe_pearsonr(preds_img[i], targets_img[i]) for i in range(n_images)],
            dtype=np.float64,
        )
        metrics.img_voxel_r = np.array(
            [_safe_pearsonr(preds_img[:, v], targets_img[:, v]) for v in range(V)],
            dtype=np.float64,
        )
        metrics.mean_img_profile_r = float(np.mean(metrics.img_profile_r))
        metrics.mean_img_voxel_r = float(np.mean(metrics.img_voxel_r))

    return metrics
