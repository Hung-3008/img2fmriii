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


def get_latent_size(data_cfg: Union[DictConfig, Dict]) -> Tuple[int, ...]:
    """Return latent tensor shape for fMRI representation.

    - 1D mode (default): ``(1, pad_to)``
    - 2D mode (legacy):  ``(C, H, W)``

    Reads ``fmri_channels``, ``fmri_spatial``, and ``pad_to`` from *data_cfg*.
    """
    if isinstance(data_cfg, DictConfig):
        data_cfg = OmegaConf.to_container(data_cfg, resolve=True)
    s = data_cfg.get("fmri_spatial", None)
    if s is not None and int(s) > 0:
        # 2D legacy mode
        c = int(data_cfg["fmri_channels"])
        s = int(s)
        return (c, s, s)
    else:
        # 1D native mode
        return (1, int(data_cfg["pad_to"]))
