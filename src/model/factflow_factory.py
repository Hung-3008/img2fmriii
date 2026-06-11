"""
factflow_factory.py
===================
Factory functions for the FactFlow fMRI stack:

1. **Model**: a velocity network (DiT1D) + optional SourceEncoder, wrapped in
   ``FactFlowWrapper`` for unified state-dict management.
2. **Transport**: flow-matching transport with configurable path type and
   time-distribution shift.
3. **Sampler**: ODE sampler for inference.

When ``source_encoder.enabled: true`` is set in the config, x₀ is sampled from
a learned image-conditioned distribution N(μ_θ(c), σ_θ(c)²) instead of N(0, I).
Otherwise the baseline pure-Gaussian source is used.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .source_encoder import SourceEncoder
from .transport import create_transport, Sampler
from utils.config_utils import instantiate_from_config

logger = logging.getLogger(__name__)


class FactFlowWrapper(nn.Module):
    """Container around the velocity DiT (+ optional SourceEncoder)."""

    def __init__(self, dit: nn.Module, source_encoder: SourceEncoder | None = None) -> None:
        super().__init__()
        self.dit = dit
        self.source_encoder = source_encoder

    def predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        contexts=None,
        subject_ids=None,
        roi_profile=None,
    ) -> torch.Tensor:
        """Run the velocity network.

        Args:
            x:           ``(B, 1, L)`` noisy fMRI signal.
            t:           ``(B,)`` timestep.
            y:           ``(B, D_pool)`` conditioning (CLIP pooled, AdaLN).
            contexts:    list of ``(B, Mᵢ, Dᵢ)`` cross-attention streams.
            subject_ids: ``(B,)`` long — 0-indexed subject ID (learned mode).
            roi_profile: ``(B, n_buckets)`` float — ROI profile (zero-shot mode).

        Returns:
            ``(B, 1, L)`` predicted velocity.
        """
        return self.dit(
            x=x, t=t, y=y, contexts=contexts,
            subject_ids=subject_ids, roi_profile=roi_profile,
        )

    def sample_source(
        self,
        z_pool: torch.Tensor,
        C_dino: torch.Tensor | None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample x₀ from the conditioned source distribution.

        Delegates to ``self.source_encoder.sample()``.  The returned tensor is
        **always detached** — the velocity DiT loss never backpropagates through
        the source encoder.

        Args:
            z_pool:      (B, D_pool) CLIP pooled embedding.
            C_dino:      (B, T, D_dino) DINOv2 tokens; may be None for CLIP-only.
            noise_scale: 0.0 = deterministic (μ only); 1.0 = full stochastic.

        Returns:
            x₀: (B, 1, L) detached source sample.

        Raises:
            RuntimeError: if called when no source_encoder was built.
        """
        if self.source_encoder is None:
            raise RuntimeError(
                "sample_source() called but source_encoder is None. "
                "Set source_encoder.enabled: true in the config."
            )
        return self.source_encoder.sample(z_pool, C_dino, noise_scale)


# ═══════════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════════


def build_models(cfg: DictConfig, device: str = "cpu") -> FactFlowWrapper:
    """Instantiate the velocity DiT (+ optional SourceEncoder) and wrap them.

    The SourceEncoder is built when ``source_encoder.enabled: true`` appears in
    the config.  All other keys in the ``source_encoder`` block are forwarded as
    keyword arguments to :class:`SourceEncoder`.
    """
    dit = instantiate_from_config(cfg.stage_2)
    dit_params = sum(p.numel() for p in dit.parameters())
    logger.info("DiT params: %.2fM", dit_params / 1e6)

    source_enc: SourceEncoder | None = None
    _se_raw = cfg.get("source_encoder", OmegaConf.create({}))
    if not isinstance(_se_raw, DictConfig):
        _se_raw = OmegaConf.create(_se_raw if isinstance(_se_raw, dict) else {})
    se_cfg = OmegaConf.to_container(_se_raw, resolve=True)
    if se_cfg.get("enabled", False):
        # n_voxels: use pad_to from data config (authoritative after auto_pad)
        n_voxels = int(cfg.data.get("pad_to", se_cfg.get("n_voxels", 15744)))

        # dino_dim: prefer context_dims[0] injected by the dataset at runtime
        # (e.g. dinov2_vitg14_multilayer4p: 4×257 tokens, dim=1536 per token).
        # Fall back to se_cfg['dino_dim'] only when context_dims is not yet set.
        context_dims = list(cfg.stage_2.params.get("context_dims") or [])
        if context_dims:
            dino_dim = int(context_dims[0])  # first stream = DINO
        else:
            dino_dim = int(se_cfg.get("dino_dim", 1024))

        source_enc = SourceEncoder(
            clip_dim   = int(se_cfg.get("clip_dim",   1280)),
            dino_dim   = dino_dim,
            hidden_dim = int(se_cfg.get("hidden_dim", 512)),
            n_voxels   = n_voxels,
            patch_size = int(se_cfg.get("patch_size",
                             cfg.stage_2.params.get("patch_size", 32))),
            use_dino   = bool(se_cfg.get("use_dino",  True)),
        )
        se_params = sum(p.numel() for p in source_enc.parameters())
        logger.info(
            "SourceEncoder enabled  params: %.2fM  use_dino=%s  dino_dim=%d  n_voxels=%d",
            se_params / 1e6, se_cfg.get("use_dino", True), dino_dim, n_voxels,
        )
    else:
        logger.info("SourceEncoder disabled — using pure Gaussian x₀ ~ N(0, I)")

    return FactFlowWrapper(dit=dit, source_encoder=source_enc).to(device)


def build_transport(cfg: DictConfig, latent_size: Tuple[int, int, int]) -> Any:
    """Create a flow-matching Transport object.

    Computes ``time_dist_shift`` from *latent_size* automatically
    (sqrt(dim / shift_base)). Returns a ``Transport`` instance.
    """
    transport_cfg: Dict[str, Any] = OmegaConf.to_container(
        cfg.transport.get("params", {}), resolve=True,
    )
    shift_base = transport_cfg.pop("time_dist_shift", 4096)
    shift_dim = math.prod(latent_size)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    transport = create_transport(**transport_cfg, time_dist_shift=time_dist_shift)
    logger.info(
        "Transport: path=%s  time_dist_shift=%.4f  (dim=%d, base=%d)",
        transport_cfg.get("path_type", "Linear"),
        time_dist_shift, shift_dim, shift_base,
    )
    return transport


def build_sampler(
    transport: Any,
    sampler_cfg: DictConfig | Dict[str, Any],
) -> Callable:
    """Create an ODE sampling function from a Transport instance.

    Returns a callable ``sample_fn(x0, model_fn, **kwargs) → trajectory``.
    """
    if isinstance(sampler_cfg, DictConfig):
        sampler_cfg = OmegaConf.to_container(sampler_cfg, resolve=True)

    sampler = Sampler(transport)
    mode = sampler_cfg.get("mode", "ODE").upper()
    params = dict(sampler_cfg.get("params", {}))

    if mode == "ODE":
        return sampler.sample_ode(**params)
    raise NotImplementedError(f"Sampler mode '{mode}' not supported")
