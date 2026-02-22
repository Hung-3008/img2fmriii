"""
Evaluation metrics for NeuroFlow-H.

All functions operate on PyTorch tensors and support batched computation.
"""

import torch
from typing import Dict, Optional


def pearson_correlation(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient.
    
    Args:
        pred: Predicted values, shape (..., N) or (B, N).
        target: Ground truth values, same shape as pred.
        dim: Dimension along which to compute correlation.
        
    Returns:
        Correlation coefficient(s). Shape is pred.shape with `dim` removed.
    """
    pred = pred.float()
    target = target.float()
    
    pred_mean = pred.mean(dim=dim, keepdim=True)
    target_mean = target.mean(dim=dim, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    numerator = (pred_centered * target_centered).sum(dim=dim)
    
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=dim).clamp(min=1e-8))
    target_std = torch.sqrt((target_centered ** 2).sum(dim=dim).clamp(min=1e-8))
    
    corr = numerator / (pred_std * target_std)
    return corr.clamp(-1.0, 1.0)


def mse_score(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Mean Squared Error.
    
    Args:
        pred: Predicted values.
        target: Ground truth values.
        reduction: "mean", "sum", or "none".
        
    Returns:
        MSE value(s).
    """
    diff = (pred.float() - target.float()) ** 2
    if reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    else:
        return diff


def r2_score(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    dim: int = 0
) -> torch.Tensor:
    """
    Compute R² (coefficient of determination).
    
    Computes per-voxel R² when dim=0 (across samples).
    
    Args:
        pred: Predicted values, shape (N_samples, N_voxels).
        target: Ground truth values, same shape.
        dim: Dimension of samples.
        
    Returns:
        R² values, shape (N_voxels,) if dim=0.
    """
    pred = pred.float()
    target = target.float()
    
    ss_res = ((target - pred) ** 2).sum(dim=dim)
    ss_tot = ((target - target.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)
    
    r2 = 1.0 - ss_res / ss_tot.clamp(min=1e-8)
    return r2


def noise_normalized_r2(
    pred: torch.Tensor,
    target: torch.Tensor,
    noise_ceiling: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """
    Compute noise-normalized R² = R² / noise_ceiling.
    
    This measures how close the model is to the theoretical limit 
    of predictability given the noise in the data.
    
    Args:
        pred: Predicted values, shape (N_samples, N_voxels).
        target: Ground truth values, same shape.
        noise_ceiling: Per-voxel noise ceiling, shape (N_voxels,).
        dim: Dimension of samples.
        
    Returns:
        Noise-normalized R² values, shape (N_voxels,).
    """
    r2 = r2_score(pred, target, dim=dim)
    return r2 / noise_ceiling.float().clamp(min=1e-8)


def roi_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    roi_masks: Dict[str, torch.Tensor],
    sample_dim: int = 0
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-ROI evaluation metrics.
    
    Args:
        pred: Predicted values, shape (N_samples, N_voxels).
        target: Ground truth values, same shape.
        roi_masks: Dict mapping ROI names to boolean masks of shape (N_voxels,).
        sample_dim: Dimension of samples (0 for batch-first).
        
    Returns:
        Dict mapping ROI names to dicts of {pcc, mse, r2}.
    """
    results = {}
    for roi_name, mask in roi_masks.items():
        mask = mask.bool()
        if mask.sum() == 0:
            continue
        
        pred_roi = pred[:, mask] if sample_dim == 0 else pred[mask, :]
        target_roi = target[:, mask] if sample_dim == 0 else target[mask, :]
        
        # Per-voxel PCC averaged across voxels
        pcc_per_voxel = pearson_correlation(pred_roi, target_roi, dim=0)
        mean_pcc = pcc_per_voxel.mean().item()
        
        # MSE for this ROI
        roi_mse = mse_score(pred_roi, target_roi).item()
        
        # Mean R² across voxels
        r2_per_voxel = r2_score(pred_roi, target_roi, dim=sample_dim)
        mean_r2 = r2_per_voxel.mean().item()
        
        results[roi_name] = {
            "pcc": mean_pcc,
            "mse": roi_mse,
            "r2": mean_r2,
            "n_voxels": int(mask.sum().item()),
        }
    
    return results
