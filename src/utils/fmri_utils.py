"""
fmri_utils.py
=============
fMRI-specific helpers: pad-mask creation, latent-size computation.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf


def create_pad_mask(
    n_voxels: int,
    pad_to: int,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Boolean mask — ``True`` for real voxels, ``False`` for padding.

    Shape: ``(pad_to,)``
    """
    mask = torch.zeros(pad_to, dtype=torch.bool, device=device)
    mask[:n_voxels] = True
    return mask


def get_latent_size(data_cfg: Union[DictConfig, Dict]) -> Tuple[int, int, int]:
    """Return ``(C, H, W)`` for the pseudo-2D fMRI representation.

    Reads ``fmri_channels`` and ``fmri_spatial`` from *data_cfg*.
    """
    if isinstance(data_cfg, DictConfig):
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    c = int(data_cfg["fmri_channels"])
    s = int(data_cfg["fmri_spatial"])
    return (c, s, s)
