"""
Simple AdaLN Flow MLP — DINOv2 CLS token conditioning.

Key difference from ResidualFlowSiT:
    - No multi-layer cross-attention (flow ignored it anyway)
    - Uses DINOv2 layer 12 CLS token, pooled to a vector
    - Conditioning = concat(pool(CLS), t_emb) → AdaLN modulation
    - Architecture: simple ResMLP with AdaLN, flat latent (no tokenize)

This forces the flow to use conditioning directly (no shortcut through xt).
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class SimpleFlowConfig:
    latent_dim: int = 768
    context_dim: int = 768           # DINOv2 token dim
    hidden_dim: int = 1024           # MLP hidden dim
    depth: int = 8                   # number of ResMLP blocks
    dropout: float = 0.1
    cond_dim: int = 512              # conditioning vector dim (pool + t)
    dino_layer_idx: int = 3          # which DINOv2 layer to use (0-indexed, 3 = layer 12)
    pool_method: str = "cls"         # "cls" or "mean"


# ─── AdaLN ResMLP Block ──────────────────────────────────────────────────────


class AdaLNResBlock(nn.Module):
    """Residual MLP block with AdaLN conditioning.
    
    cond → adaLN params (γ, β, gate)
    z → LN → (γ·z + β) → MLP → gate · residual
    """

    def __init__(self, dim, cond_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        # AdaLN: cond → (γ, β, gate)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )
        # Init: gate bias → small positive for stable residuals
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)
        with torch.no_grad():
            self.adaln[1].bias.data[2*dim:] = 0.1  # gate init

    def forward(self, x, cond):
        """x: (B, D), cond: (B, cond_dim)"""
        params = self.adaln(cond)
        D = x.shape[-1]
        gamma, beta, gate = (
            params[:, :D],
            params[:, D:2*D],
            params[:, 2*D:].sigmoid()
        )

        h = self.norm(x) * (1 + gamma) + beta
        h = self.mlp(h)
        return x + gate * h


# ─── Simple Flow Model ───────────────────────────────────────────────────────


class SimpleAdaLNFlow(nn.Module):
    """Simple flow model: DINOv2 CLS → pool → concat(t) → AdaLN ResMLP.
    
    Much simpler than ResidualFlowSiT — no tokenization, no cross-attention.
    Conditioning is directly injected via AdaLN, forcing the model to use it.
    """

    def __init__(self, config: SimpleFlowConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.cond_dim

        # ── Context encoder: DINOv2 CLS token → cond vector ──
        self.ctx_proj = nn.Sequential(
            nn.Linear(config.context_dim, C),
            nn.GELU(approximate='tanh'),
            nn.Linear(C, C),
        )

        # ── Time encoder ──
        self.t_embed = nn.Sequential(
            nn.Linear(C, C),
            nn.SiLU(),
            nn.Linear(C, C),
        )

        # ── Conditioning merge: concat(ctx, t) → cond ──
        self.cond_merge = nn.Sequential(
            nn.Linear(C * 2, C),
            nn.SiLU(),
        )

        # ── Flow backbone: input → hidden → blocks → output ──
        self.input_proj = nn.Linear(config.latent_dim, D)
        self.blocks = nn.ModuleList([
            AdaLNResBlock(D, C, mlp_ratio=4.0, dropout=config.dropout)
            for _ in range(config.depth)
        ])
        self.output_norm = nn.LayerNorm(D)
        self.output_proj = nn.Linear(D, config.latent_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for output
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def _get_cls_token(self, dino_multilayer):
        """Extract CLS token from specified DINOv2 layer.
        
        dino_multilayer: (B, L, 257, ctx_dim)
        Returns: (B, ctx_dim) — CLS token
        """
        idx = self.config.dino_layer_idx
        if self.config.pool_method == "cls":
            return dino_multilayer[:, idx, 0, :]   # CLS token (index 0)
        else:  # mean pool
            return dino_multilayer[:, idx, :, :].mean(dim=1)

    def forward(self, t, z_t, dino_multilayer):
        """Predict velocity v(z_t, t | DINOv2).
        
        t: (B,), z_t: (B, latent_dim), dino_multilayer: (B, L, 257, ctx_dim)
        Returns: (B, latent_dim)
        """
        # Context: CLS token → project
        cls_token = self._get_cls_token(dino_multilayer)

        # Optional noise augmentation during training
        if self.training:
            cls_token = cls_token + 0.05 * torch.randn_like(cls_token)

        ctx = self.ctx_proj(cls_token)             # (B, C)

        # Time embedding
        t_emb = self.t_embed(
            timestep_embedding(t * 1000, self.config.cond_dim))  # (B, C)

        # Merge: concat(ctx, t) → single conditioning vector
        cond = self.cond_merge(
            torch.cat([ctx, t_emb], dim=-1))       # (B, C)

        # Flow backbone
        h = self.input_proj(z_t)                   # (B, D)
        for block in self.blocks:
            h = block(h, cond)
        h = self.output_norm(h)
        return self.output_proj(h)                  # (B, latent_dim)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total_M": total / 1e6}
