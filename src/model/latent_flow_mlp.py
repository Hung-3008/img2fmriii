"""
1D Latent Flow MLP — Backbone for Stage 2.

MLP-based architecture for flow matching on flat latent vectors.
Uses ADDITIVE context injection + adaLN modulation for deep conditioning.

Key change vs pure adaLN-Zero: context is ADDED to hidden state at every block,
not just used as modulation parameters. This prevents the model from learning to
ignore the conditioning (which happens when all adaLN gates start at 0).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ─── Helpers ──────────────────────────────────────────────────────────────────

def modulate_2d(x, shift, scale):
    """Apply adaLN modulation for 2D tensors [B, D]."""
    return x * (1 + scale) + shift


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ─── Blocks ───────────────────────────────────────────────────────────────────

class MLPResBlock(nn.Module):
    """
    Residual MLP block with adaLN conditioning.

    Conditioning is injected via:
      1. adaLN modulation (shift + scale from time+context)
      2. Additive context bias (context projected directly added to hidden state)

    adaLN gate is initialized to SMALL NON-ZERO (0.1) instead of zero,
    so the residual path is active from epoch 1.
    """

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        # adaLN: shift, scale, gate from conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )
        # Initialize gate to small positive value (not zero!)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        with torch.no_grad():
            bias = torch.zeros(3 * dim)
            bias[2 * dim:] = 0.1  # gate bias = 0.1 → residual active from start
            self.adaLN_modulation[1].bias.copy_(bias)

    def forward(self, x, c):
        """
        Args:
            x: [B, D] hidden state
            c: [B, cond_dim] conditioning (time + context)
        """
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
        return x + gate * self.net(modulate_2d(self.norm(x), shift, scale))


# ─── Model ────────────────────────────────────────────────────────────────────

@dataclass
class FlowMLPConfig:
    latent_dim: int = 1024
    hidden_dim: int = 4096
    depth: int = 8
    context_dim: int = 1024
    dropout: float = 0.1


class LatentFlowMLP(nn.Module):
    """
    MLP for flow matching: predicts velocity field v(z_t, t, context).

    Context injection strategy (fixes pure-adaLN-Zero context ignoring):
      1. Context is CONCATENATED with z_t at input (forces early feature mixing)
      2. Time + context conditioning via adaLN at every block (with non-zero gate init)
      3. Final layer uses adaLN for output modulation
    """

    def __init__(self, config: Optional[FlowMLPConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = FlowMLPConfig(**kwargs)
        self.config = config

        self.latent_dim = config.latent_dim
        hidden = config.hidden_dim

        # Input: concat [z_t, context_proj] → project to hidden
        self.context_proj = nn.Linear(config.context_dim, config.latent_dim)
        self.input_proj = nn.Linear(self.latent_dim * 2, hidden)  # z_t + ctx

        # Timestep embedder
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Context embedder for adaLN conditioning (separate from input injection)
        self.cond_proj = nn.Linear(config.context_dim, hidden)

        # Conditioning dim for adaLN = time_dim + cond_dim = hidden + hidden
        cond_dim = hidden * 2

        # MLP blocks with adaLN conditioning
        self.blocks = nn.ModuleList([
            MLPResBlock(hidden, cond_dim, config.dropout) for _ in range(config.depth)
        ])

        # Final output
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_proj = nn.Linear(hidden, self.latent_dim)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden))

        self._init_weights()

    def _init_weights(self):
        # Xavier init for projections
        for module in [self.context_proj, self.cond_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Input proj: xavier (takes concat of z_t and context)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        for layer in self.t_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Final adaLN: zero init (standard)
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        # Final projection: zero init (outputs zero velocity at start)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, t, z_t, context):
        """
        Predict velocity field v(z_t, t, context).

        Args:
            t: [B] timestep ∈ [0, 1]
            z_t: [B, D] noisy latent
            context: [B, C] or [B, M, C] features (CLS token used if 3D)
        """
        # Truncate to CLS token if 3D
        if context.ndim == 3:
            context = context[:, 0, :]

        # ── Input: concat z_t with projected context ──
        ctx_input = self.context_proj(context)       # [B, latent_dim]
        x = self.input_proj(torch.cat([z_t, ctx_input], dim=1))  # [B, hidden]

        # ── Build conditioning vector for adaLN ──
        t_scaled = t * 1000.0
        t_emb = timestep_embedding(t_scaled, self.config.hidden_dim)
        c_t = self.t_embedder(t_emb)                # [B, hidden]
        c_ctx = self.cond_proj(context)              # [B, hidden]
        c = torch.cat([c_t, c_ctx], dim=1)           # [B, hidden*2]

        # ── MLP blocks ──
        for block in self.blocks:
            x = block(x, c)

        # ── Final layer ──
        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = modulate_2d(self.norm_final(x), shift, scale)
        return self.final_proj(x)

    def forward_with_cfg(self, t, z_t, context, cfg_scale=1.0):
        """Forward with classifier-free guidance."""
        if cfg_scale == 1.0:
            return self.forward(t, z_t, context)
        v_cond = self.forward(t, z_t, context)
        v_uncond = self.forward(t, z_t, torch.zeros_like(context))
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": total / 1e6}
