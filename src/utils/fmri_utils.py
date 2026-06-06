"""
fmri_utils.py
=============
fMRI-specific helpers: pad-mask creation, latent-size computation,
per-subject native sizing (auto-pad).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf


def auto_size_config(cfg: DictConfig) -> Optional[str]:
    """Size the model to a subject's voxel count, eliminating dead padding.

    When ``cfg.data.auto_pad`` is truthy, derive::

        pad_to      = ceil(n_voxels / L) * L      with  L = lcm(patch_size, out_channels)
        seq_len     = pad_to                       (DiT1D input length)
        num_queries = pad_to // out_channels       (PerceiverVE source tokens)

    ``L`` is the least common multiple of the DiT patch size and the source
    encoder's ``out_channels`` so that ``pad_to`` is divisible by both — the
    Conv1d patch embedding needs ``pad_to % patch_size == 0`` and the source
    reshape ``(B, Q, out_ch) → (B, 1, pad_to)`` needs ``Q * out_ch == pad_to``.
    This leaves at most ``L - 1`` padded voxels (<0.5 %), versus the ~4–23 %
    wasted by a fixed ``pad_to=16384`` across subjects.

    Mutates *cfg* in place. Returns a human-readable log message, or ``None``
    when auto-pad is disabled (config left untouched).
    """
    data = cfg.get("data", {})
    if not bool(data.get("auto_pad", False)):
        return None

    n_voxels = int(data["n_voxels"])
    patch_size = int(cfg.stage_2.params.patch_size)

    # Source encoder couples num_queries × out_channels = pad_to; include
    # out_channels in the LCM so the reshape stays exact. Skip when there is
    # no variational source (pure-noise flow) — then only patch_size matters.
    has_source = "source_encoder" in cfg and bool(cfg.get("use_source_encoder", True))
    if has_source:
        out_channels = int(cfg.source_encoder.params.out_channels)
        L = math.lcm(patch_size, out_channels)
    else:
        out_channels = None
        L = patch_size

    pad_to = math.ceil(n_voxels / L) * L

    old_pad_to = int(data.get("pad_to", 0))
    data.pad_to = pad_to
    cfg.stage_2.params.seq_len = pad_to
    if has_source:
        cfg.source_encoder.params.num_queries = pad_to // out_channels

    # The transport's time-shift base is set equal to pad_to so the effective
    # shift = sqrt(prod(latent)/base) = sqrt(pad_to/pad_to) = 1.0 — matching the
    # baseline (which hardcoded base = 16384 = old pad_to). Without this, a
    # smaller pad_to makes the ratio < 1 and the transport asserts shift >= 1.0.
    if "transport" in cfg and "params" in cfg.transport:
        cfg.transport.params.time_dist_shift = pad_to

    pad_voxels = pad_to - n_voxels
    msg = (
        f"auto_pad: n_voxels={n_voxels} → pad_to={pad_to} "
        f"(L=lcm({patch_size},{out_channels})={L}, pad={pad_voxels} voxels "
        f"= {100 * pad_voxels / pad_to:.2f}%; was pad_to={old_pad_to}); "
        f"seq_len={pad_to}"
    )
    if has_source:
        msg += f", num_queries={pad_to // out_channels}"
    return msg


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
