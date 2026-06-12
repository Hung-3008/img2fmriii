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
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE computed only on real (non-padded) voxels.

    Args:
        pred, target: ``(B, C, H, W)`` — predicted and ground-truth fMRI.
        pad_mask: boolean — ``True`` for real voxels. Either ``(V_pad,)``
            (shared across the batch) or ``(B, V_pad)`` (per-sample, e.g.
            multi-subject where each item has a different real-voxel count).
        weight: optional ``(V_pad,)`` per-voxel weight (e.g. noise-ceiling
            reliability). When given, the loss becomes a weighted average —
            voxels with low reliability contribute less, focusing capacity on
            learnable voxels. Should be ~0 on padded positions.

    Returns:
        Scalar MSE tensor.
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    mask = pad_mask.to(pred.device).float()
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)               # (1, V_pad) → broadcast over batch
    if weight is not None:
        mask = mask * weight.to(pred.device).float()
    diff_sq = (pred_flat - target_flat) ** 2   # (B, V_pad)
    mask = mask.expand_as(diff_sq)             # (B, V_pad); 1-D mask → real·B count
    return (diff_sq * mask).sum() / (mask.sum() + 1e-8)


def compute_voxel_reliability(
    fmri_data: np.ndarray,
    n_reps: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-voxel noise-ceiling reliability from repeated presentations.

    Estimates, for each voxel, the fraction of variance in the
    ``n_reps``-averaged response that is explainable signal (NSD-style noise
    ceiling). Voxels dominated by trial noise get a weight ≈ 0; reliable
    voxels get a weight ≈ 1.

    Args:
        fmri_data: ``(N_images, n_reps, V)`` raw (un-averaged) betas.
        n_reps:    repetitions per image.
        eps:       numerical floor.

    Returns:
        ``(V,)`` reliability in ``[0, 1]`` (noise ceiling for the rep-mean).
    """
    x = np.asarray(fmri_data, dtype=np.float32)        # (N, K, V)
    rep_mean = x.mean(axis=1)                          # (N, V)
    noise_var = x.var(axis=1, ddof=1).mean(axis=0)     # (V,) within-image noise
    signal_var = rep_mean.var(axis=0) - noise_var / n_reps
    signal_var = np.clip(signal_var, 0.0, None)
    nc = signal_var / (signal_var + noise_var / n_reps + eps)
    return nc.astype(np.float64)


@torch.no_grad()
def pearson_corr_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-sample (profile) Pearson *r* between predicted and target fMRI.

    Args:
        pred, target: ``(B, C, H, W)``
        pad_mask: boolean — ``(V_pad,)`` (shared) or ``(B, V_pad)`` (per-sample,
            e.g. multi-subject with different real-voxel counts per item).

    Returns:
        ``(B,)`` tensor of Pearson *r* values.
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)
    mask = pad_mask.to(pred.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand(B, -1)
    mask = mask.float()
    n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    pm = (pred_flat * mask).sum(dim=1, keepdim=True) / n
    tm = (target_flat * mask).sum(dim=1, keepdim=True) / n
    p = (pred_flat - pm) * mask
    t = (target_flat - tm) * mask
    num = (p * t).sum(dim=1)
    den = torch.sqrt((p ** 2).sum(dim=1) * (t ** 2).sum(dim=1))
    return num / (den + 1e-8)


@torch.no_grad()
def voxel_pearson(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Per-voxel (encoding) Pearson *r* across the sample axis.

    The standard NSD encoding-model metric: for each voxel, the correlation
    between predicted and measured responses across all images. Unlike the
    per-sample profile r, it is **not** inflated by anatomical mean structure
    — a constant predictor (per-voxel mean) scores r ≈ 0.

    Args:
        preds, targets: ``(N, V)`` — unpadded voxels for all N samples.

    Returns:
        ``(V,)`` tensor of per-voxel Pearson *r*.
    """
    p = preds - preds.mean(dim=0, keepdim=True)
    t = targets - targets.mean(dim=0, keepdim=True)
    num = (p * t).sum(dim=0)
    den = torch.sqrt((p ** 2).sum(dim=0) * (t ** 2).sum(dim=0))
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
