"""
ResidualFlowSiT — Residual Flow Matching with multi-layer DINOv2.

Two-stage architecture:
    1. Regression Head: Multi-layer DINOv2 → z̄ (deterministic mean prediction)
       - Learnable queries cross-attend to multi-layer DINOv2 features
       - Per-block learned layer mixing (like MultiLayerFlowSiT)
    2. Flow Network: Learns residual Δz = z_true − z̄
       - Multi-layer SiT blocks with cross-attention
       - Same architecture as MultiLayerFlowSiT but for residuals

The regression head captures the deterministic ~36.5% variance while
the flow network only needs to model the remaining residual distribution
(lower variance → easier to learn, less overfitting).
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.aligned_flow_mlp import (
    timestep_embedding, modulate, get_1d_sincos_pos_embed, FinalLayer,
)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class ResidualFlowSiTConfig:
    # ── Shared ──
    latent_dim: int = 1024
    context_dim: int = 1024        # DINOv2 token dim per layer
    n_context_tokens: int = 257    # CLS + 256 patches
    n_dino_layers: int = 4         # number of DINOv2 layers extracted

    # ── Regression Head ──
    reg_n_queries: int = 16        # learnable query tokens
    reg_depth: int = 4             # regression transformer depth
    reg_hidden_dim: int = 512
    reg_num_heads: int = 8
    reg_mlp_ratio: float = 4.0
    reg_dropout: float = 0.1

    # ── Flow Network ──
    n_latent_tokens: int = 16
    hidden_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.25


# ─── Multi-Layer Cross-Attention Block (shared by regression + flow) ─────────

class MultiLayerCrossBlock(nn.Module):
    """
    Transformer block with per-block learned layer mixing for multi-layer DINOv2.

    Learnable softmax weights α ∈ R^n_layers mix the projected DINOv2 layers
    before cross-attention. This enables automatic discovery of which DINOv2
    layer is most useful at each processing stage.
    """

    def __init__(self, dim, context_dim, num_heads, n_layers,
                 mlp_ratio=4.0, dropout=0.0, use_adaln=False):
        super().__init__()
        self.use_adaln = use_adaln

        # Learnable layer mixing weights (initialized uniform)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

        # Cross-attention: queries attend to mixed multi-layer DINOv2
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=not use_adaln,
                                   eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, elementwise_affine=not use_adaln,
                                    eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout)

        # Self-attention among tokens
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=not use_adaln,
                                    eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout)

        # FFN
        self.norm_ff = nn.LayerNorm(dim, elementwise_affine=not use_adaln,
                                    eps=1e-6)
        ffn_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        # adaLN modulation (only for flow blocks, not regression)
        if use_adaln:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def get_layer_weights(self):
        """Return softmax weights α for analysis."""
        return F.softmax(self.layer_logits, dim=0)

    def forward(self, x, multi_ctx, c=None):
        """
        Args:
            x: (B, T, D)                    tokens (queries or latent)
            multi_ctx: list of (B, T_ctx, D)  per-layer projected contexts
            c: (B, D) optional               time conditioning (flow only)
        """
        # Mix layers: α = softmax(logits), ctx = Σ αᵢ * ctxᵢ
        alpha = F.softmax(self.layer_logits, dim=0)  # (n_layers,)
        ctx = sum(a * ctx_l for a, ctx_l in zip(alpha, multi_ctx))

        if self.use_adaln and c is not None:
            params = self.adaLN(c).chunk(9, dim=-1)
            s_sa, sc_sa, g_sa = params[0], params[1], params[2]
            s_ca, sc_ca, g_ca = params[3], params[4], params[5]
            s_ff, sc_ff, g_ff = params[6], params[7], params[8]

            # Self-Attention with adaLN
            h = modulate(self.norm_sa(x), s_sa, sc_sa)
            sa_out, _ = self.self_attn(h, h, h)
            x = x + g_sa.unsqueeze(1) * sa_out

            # Cross-Attention with adaLN
            q = modulate(self.norm_q(x), s_ca, sc_ca)
            kv = self.norm_kv(ctx)
            ca_out, _ = self.cross_attn(q, kv, kv)
            x = x + g_ca.unsqueeze(1) * ca_out

            # FFN with adaLN
            h = modulate(self.norm_ff(x), s_ff, sc_ff)
            x = x + g_ff.unsqueeze(1) * self.ffn(h)
        else:
            # Standard pre-norm (for regression head)
            # Self-Attention
            sa_in = self.norm_sa(x)
            x = x + self.self_attn(sa_in, sa_in, sa_in,
                                    need_weights=False)[0]

            # Cross-Attention
            q = self.norm_q(x)
            kv = self.norm_kv(ctx)
            x = x + self.cross_attn(q, kv, kv, need_weights=False)[0]

            # FFN
            x = x + self.ffn(self.norm_ff(x))

        return x


# ─── Regression Head ─────────────────────────────────────────────────────────

class MultiLayerRegressor(nn.Module):
    """
    Multi-layer DINOv2 → z̄ (deterministic mean prediction).

    Learnable queries cross-attend to multi-layer DINOv2 features with
    per-block learned layer mixing. No time conditioning.
    """

    def __init__(self, config: ResidualFlowSiTConfig):
        super().__init__()
        D = config.reg_hidden_dim
        L = config.n_dino_layers

        # Learnable query tokens
        self.queries = nn.Parameter(
            torch.randn(1, config.reg_n_queries, D) * 0.02)

        # Per-layer DINOv2 projections (separate from flow network)
        self.context_projs = nn.ModuleList([
            nn.Linear(config.context_dim, D) for _ in range(L)
        ])

        # Transformer blocks with multi-layer mixing
        self.blocks = nn.ModuleList([
            MultiLayerCrossBlock(
                D, config.context_dim, config.reg_num_heads, L,
                config.reg_mlp_ratio, config.reg_dropout, use_adaln=False)
            for _ in range(config.reg_depth)
        ])

        # Output: flatten queries → z̄
        self.norm_out = nn.LayerNorm(D)
        self.head = nn.Sequential(
            nn.Linear(config.reg_n_queries * D, D),
            nn.GELU(approximate='tanh'),
            nn.Dropout(config.reg_dropout),
            nn.Linear(D, config.latent_dim),
        )

    def forward(self, dino_multilayer):
        """
        dino_multilayer: (B, L, 257, 1024) → z̄ (B, latent_dim)
        """
        B = dino_multilayer.shape[0]
        L = dino_multilayer.shape[1]

        # Project each DINOv2 layer
        multi_ctx = []
        for i in range(L):
            layer_tokens = dino_multilayer[:, i, :, :].float()
            ctx = self.context_projs[i](layer_tokens)
            multi_ctx.append(ctx)

        # Learnable queries attend to multi-layer context
        x = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, multi_ctx)

        x = self.norm_out(x)
        x = x.reshape(B, -1)
        z_bar = self.head(x)
        return z_bar

    def get_layer_mixing_weights(self):
        """Return (reg_depth, n_layers) matrix."""
        weights = []
        for block in self.blocks:
            weights.append(block.get_layer_weights().detach().cpu())
        return torch.stack(weights)


# ─── Main Model ──────────────────────────────────────────────────────────────

class ResidualFlowSiT(nn.Module):
    """
    Residual Flow Matching with multi-layer DINOv2.

    Two modules:
        1. regression: DINOv2 → z̄ (deterministic mean, ~36.5% variance)
        2. flow: learns Δz = z_true − z̄ via conditional flow matching

    At inference:
        z_gen = z̄ + ODE(flow, x0, DINOv2)

    Key methods:
        forward_regression(dino)  → z̄
        forward_flow(t, z_t, dino) → velocity (for residual Δz)
        get_layer_mixing_weights() → dict with 'reg' and 'flow' keys
    """

    def __init__(self, config: Optional[ResidualFlowSiTConfig] = None,
                 **kwargs):
        super().__init__()
        if config is None:
            config = ResidualFlowSiTConfig(**kwargs)
        self.config = config

        D = config.hidden_dim
        L = config.n_dino_layers
        n_lat = config.n_latent_tokens
        token_dim = config.latent_dim // n_lat

        # ── Regression Head ──
        self.regressor = MultiLayerRegressor(config)

        # ── Flow Network ──
        # Patchify / Depatchify
        self.latent_proj_in = nn.Linear(token_dim, D)
        self.latent_proj_out = nn.Linear(D, token_dim)
        self.n_latent_tokens = n_lat
        self.token_dim = token_dim

        self.register_buffer('latent_pos_embed',
            torch.from_numpy(
                get_1d_sincos_pos_embed(D, n_lat)).unsqueeze(0))

        # Time embedder
        self.t_embed = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))

        # Per-layer DINOv2 projections (separate from regressor)
        self.flow_context_projs = nn.ModuleList([
            nn.Linear(config.context_dim, D) for _ in range(L)
        ])
        self.register_buffer('context_pos_embed',
            torch.from_numpy(
                get_1d_sincos_pos_embed(D, config.n_context_tokens)
            ).unsqueeze(0))

        # Flow SiT blocks with multi-layer mixing + adaLN
        self.flow_blocks = nn.ModuleList([
            MultiLayerCrossBlock(
                D, config.context_dim, config.num_heads, L,
                config.mlp_ratio, config.dropout, use_adaln=True)
            for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(D)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.apply(_basic_init)

        # Gate biases → 0.1 so residual paths are active
        D = self.config.hidden_dim
        for block in self.flow_blocks:
            if block.use_adaln:
                with torch.no_grad():
                    bias = block.adaLN[1].bias.data.view(9, D)
                    bias[2, :] = 0.1   # g_sa
                    bias[5, :] = 0.1   # g_ca
                    bias[8, :] = 0.1   # g_ff

        # Output proj: small init
        nn.init.normal_(self.latent_proj_out.weight, std=0.02)
        nn.init.zeros_(self.latent_proj_out.bias)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

        # Time embedder
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)

    def _patchify(self, z):
        return z.view(z.shape[0], self.n_latent_tokens, self.token_dim)

    def _depatchify(self, tokens):
        return tokens.reshape(tokens.shape[0], -1)

    # ── Forward methods ──

    def forward_regression(self, dino_multilayer):
        """Deterministic prediction: DINOv2 → z̄.
        dino_multilayer: (B, L, 257, 1024) → (B, latent_dim)"""
        return self.regressor(dino_multilayer)

    def forward_flow(self, t, z_t, dino_multilayer):
        """
        Predict velocity v(z_t, t | DINOv2) for residual Δz.

        Args:
            t: (B,) timestep ∈ [0, 1]
            z_t: (B, latent_dim) noisy residual
            dino_multilayer: (B, L, 257, 1024) multi-layer features
        """
        B = z_t.shape[0]
        L = self.config.n_dino_layers

        # Patchify z_t (residual)
        lat_tokens = self._patchify(z_t)
        lat_tokens = self.latent_proj_in(lat_tokens)
        lat_tokens = lat_tokens + self.latent_pos_embed

        # Project each DINOv2 layer separately
        multi_ctx = []
        for i in range(L):
            layer_tokens = dino_multilayer[:, i, :, :].float()
            ctx = self.flow_context_projs[i](layer_tokens)
            ctx = ctx + self.context_pos_embed[:, :ctx.shape[1], :]
            multi_ctx.append(ctx)

        # Time conditioning
        t_emb = self.t_embed(
            timestep_embedding(t * 1000, self.config.hidden_dim))

        # Flow SiT blocks
        for block in self.flow_blocks:
            lat_tokens = block(lat_tokens, multi_ctx, t_emb)

        # Final layer + depatchify
        lat_tokens = self.final_layer(lat_tokens, t_emb)
        lat_tokens = self.latent_proj_out(lat_tokens)
        return self._depatchify(lat_tokens)

    def forward_flow_with_cfg(self, t, z_t, dino_multilayer, cfg_scale=1.0):
        """Classifier-free guidance for flow network."""
        if cfg_scale == 1.0:
            return self.forward_flow(t, z_t, dino_multilayer)
        v_cond = self.forward_flow(t, z_t, dino_multilayer)
        v_uncond = self.forward_flow(
            t, z_t, torch.zeros_like(dino_multilayer))
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    # ── Analysis ──

    def get_layer_mixing_weights(self):
        """Return dict of mixing weights for regression and flow blocks.
        'reg': (reg_depth, n_layers), 'flow': (depth, n_layers)"""
        return {
            'reg': self.regressor.get_layer_mixing_weights(),
            'flow': torch.stack([
                block.get_layer_weights().detach().cpu()
                for block in self.flow_blocks
            ]),
        }

    def param_count(self):
        reg_p = sum(p.numel() for p in self.regressor.parameters())
        total = sum(p.numel() for p in self.parameters())
        flow_p = total - reg_p
        return {"reg_M": reg_p / 1e6, "flow_M": flow_p / 1e6,
                "total_M": total / 1e6}
