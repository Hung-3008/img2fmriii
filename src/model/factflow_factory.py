"""
factflow_factory.py
===================
Factory functions for building the FactFlow model stack:

1. **Models**: LightningDiT (velocity network) + PerceiverVE (source encoder),
   wrapped in a unified ``nn.Module`` for joint state-dict management.
2. **Transport**: Flow-matching transport with configurable path type
   and time distribution shift.
3. **Sampler**: ODE sampler for inference.

Architecture notes
------------------
We import the model classes (``LightningDiT``, ``PerceiverVE``) from
the local modules. The *wiring* and *factory logic* is fully owned here.
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .transport import create_transport, Sampler
from utils.config_utils import instantiate_from_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Unified wrapper — replaces original generic ``Wrapper``
# ═══════════════════════════════════════════════════════════════════════════


class FactFlowWrapper(nn.Module):
    """Unified container for the velocity DiT and the source encoder.

    Advantages over the generic ``Wrapper``:
    * Typed attributes (``self.dit``, ``self.source_encoder``) for IDE
      auto-completion and static analysis.
    * Direct method calls instead of string-based dispatch.
    """

    def __init__(self, dit: nn.Module, source_encoder: nn.Module) -> None:
        super().__init__()
        self.dit = dit
        self.source_encoder = source_encoder

    def encode_source(self, clip_tokens: torch.Tensor):
        """Run the source encoder (PerceiverVE).

        Args:
            clip_tokens: ``(B, T, D)`` CLIP spatial tokens.

        Returns:
            ``(x0_tok, mu, log_var)`` where ``x0_tok`` is
            ``(B, num_queries, out_channels)``.
        """
        return self.source_encoder(text_tokens=clip_tokens)

    def predict_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Run the velocity network (DiT).

        Args:
            x: ``(B, C, H, W)`` noisy latent.
            t: ``(B,)`` timestep.
            y: ``(B, D_pool)`` conditioning (CLIP pooled).

        Returns:
            ``(B, C, H, W)`` predicted velocity.
        """
        return self.dit(x=x, t=t, y=y)


# ═══════════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════════


def build_models(
    cfg: DictConfig,
    device: str = "cpu",
) -> Tuple[FactFlowWrapper, FactFlowWrapper]:
    """Instantiate DiT + SourceEncoder, wrap, and create an EMA copy.

    Returns ``(wrapper, ema)`` — both on *device*.
    """
    dit = instantiate_from_config(cfg.stage_2)
    source_encoder = instantiate_from_config(cfg.source_encoder)

    dit_params = sum(p.numel() for p in dit.parameters())
    se_params = sum(p.numel() for p in source_encoder.parameters())
    logger.info("DiT params: %.2fM", dit_params / 1e6)
    logger.info("SourceEncoder params: %.2fM", se_params / 1e6)
    logger.info("Total trainable: %.2fM", (dit_params + se_params) / 1e6)

    wrapper = FactFlowWrapper(dit=dit, source_encoder=source_encoder).to(device)
    ema = deepcopy(wrapper).to(device)
    return wrapper, ema


def build_transport(
    cfg: DictConfig,
    latent_size: Tuple[int, int, int],
) -> Any:
    """Create a flow-matching Transport object.

    Computes the ``time_dist_shift`` from *latent_size* automatically
    (matching the convention of sqrt(dim / shift_base)).

    Returns a ``Transport`` instance.
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
    """Create an ODE/SDE sampling function from a Transport instance.

    Returns a callable ``sample_fn(x0, model_fn, **kwargs) → trajectory``.
    """
    if isinstance(sampler_cfg, DictConfig):
        sampler_cfg = OmegaConf.to_container(sampler_cfg, resolve=True)

    sampler = Sampler(transport)
    mode = sampler_cfg.get("mode", "ODE").upper()
    params = dict(sampler_cfg.get("params", {}))

    if mode == "ODE":
        return sampler.sample_ode(**params)
    elif mode == "SDE":
        return sampler.sample_sde(**params)
    else:
        raise NotImplementedError(f"Sampler mode '{mode}' not supported")
