"""
factflow_factory.py
===================
Factory functions for the FactFlow fMRI stack (no-source flow matching):

1. **Model**: a single velocity network (DiT1D) wrapped in ``FactFlowWrapper``
   for unified state-dict management.
2. **Transport**: flow-matching transport with configurable path type and
   time-distribution shift.
3. **Sampler**: ODE sampler for inference.

The source x₀ is pure Gaussian noise (standard flow matching) — there is no
learned source encoder.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .transport import create_transport, Sampler
from utils.config_utils import instantiate_from_config

logger = logging.getLogger(__name__)


class FactFlowWrapper(nn.Module):
    """Thin container around the velocity DiT for unified state-dict handling."""

    def __init__(self, dit: nn.Module) -> None:
        super().__init__()
        self.dit = dit

    def predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        contexts=None,
    ) -> torch.Tensor:
        """Run the velocity network.

        Args:
            x: ``(B, C, H, W)`` noisy latent.
            t: ``(B,)`` timestep.
            y: ``(B, D_pool)`` conditioning (CLIP pooled, AdaLN).
            contexts: list of ``(B, Mᵢ, Dᵢ)`` cross-attention streams (DINOv2,
                     Gabor, …); ignored unless the DiT was built with
                     ``use_cross_attn``.

        Returns:
            ``(B, C, H, W)`` predicted velocity.
        """
        return self.dit(x=x, t=t, y=y, contexts=contexts)


# ═══════════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════════


def build_models(cfg: DictConfig, device: str = "cpu") -> FactFlowWrapper:
    """Instantiate the velocity DiT and wrap it, returning it on *device*."""
    dit = instantiate_from_config(cfg.stage_2)
    dit_params = sum(p.numel() for p in dit.parameters())
    logger.info("DiT params: %.2fM", dit_params / 1e6)
    return FactFlowWrapper(dit=dit).to(device)


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
