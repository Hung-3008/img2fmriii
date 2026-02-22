"""
Cross-Attention Regressor: DINOv2 → fMRI latent (z₁).

Architecture:
    Learnable queries (n_queries, hidden_dim)
        → Cross-Attention to DINOv2 patches (257, 1024) × depth layers
        → Flatten → Linear → z_pred ∈ ℝ^latent_dim

No time conditioning, no flow. Pure regression.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossAttnRegressorConfig:
    latent_dim: int = 1024        # output dim (VAE latent)
    context_dim: int = 1024       # DINOv2 token dim
    n_context_tokens: int = 257   # DINOv2: CLS + 256 patches
    n_queries: int = 16           # learnable query tokens
    hidden_dim: int = 512         # transformer hidden dim
    depth: int = 6                # number of transformer layers
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1


class CrossAttnBlock(nn.Module):
    """Single transformer block: Cross-Attention + Self-Attention + FFN."""

    def __init__(self, hidden_dim, context_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Cross-attention: queries attend to DINOv2 tokens
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout,
            kdim=context_dim, vdim=context_dim)

        # Self-attention among query tokens
        self.norm_sa = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout)

        # FFN
        self.norm_ff = nn.LayerNorm(hidden_dim)
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context):
        """
        x: (B, n_queries, hidden_dim) — learnable queries
        context: (B, n_ctx, context_dim) — DINOv2 tokens
        """
        # Cross-attention
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]

        # Self-attention
        sa_in = self.norm_sa(x)
        x = x + self.self_attn(sa_in, sa_in, sa_in, need_weights=False)[0]

        # FFN
        x = x + self.ffn(self.norm_ff(x))
        return x


class CrossAttnRegressor(nn.Module):
    """Direct regression: DINOv2 patches → z_pred (fMRI latent)."""

    def __init__(self, cfg: CrossAttnRegressorConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable query tokens
        self.queries = nn.Parameter(
            torch.randn(1, cfg.n_queries, cfg.hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttnBlock(
                cfg.hidden_dim, cfg.context_dim, cfg.num_heads,
                cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth)
        ])

        # Output: flatten queries → z_pred
        self.norm_out = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.n_queries * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, dino_tokens):
        """
        dino_tokens: (B, 257, 1024) — DINOv2 full tokens
        Returns: z_pred (B, latent_dim)
        """
        B = dino_tokens.shape[0]
        x = self.queries.expand(B, -1, -1)  # (B, n_queries, hidden_dim)

        for block in self.blocks:
            x = block(x, dino_tokens)

        x = self.norm_out(x)
        x = x.reshape(B, -1)               # (B, n_queries * hidden_dim)
        z_pred = self.head(x)               # (B, latent_dim)
        return z_pred

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": total / 1e6}
