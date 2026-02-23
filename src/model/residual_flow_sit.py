"""
ResidualFlowSiT V2 — Residual Flow Matching with multi-layer DINOv2.

Two-stage architecture:
    1. Regression Head: Multi-layer DINOv2 → z̄ (deterministic mean)
    2. Flow Network: Learns residual Δz = z_true − z̄

Improvements over V1:
    - Attention Pooling: 257 → n_pool_tokens per DINOv2 layer
    - Shared context projections between reg + flow
    - Stochastic Depth for regularization
    - Context noise augmentation
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
    latent_dim: int = 768
    context_dim: int = 768         # DINOv2 token dim per layer
    n_context_tokens: int = 257    # CLS + 256 patches
    n_dino_layers: int = 4
    n_pool_tokens: int = 32        # pool 257 → 32 tokens per layer

    # ── Regression Head ──
    reg_n_queries: int = 16
    reg_depth: int = 3
    reg_hidden_dim: int = 512
    reg_num_heads: int = 8
    reg_mlp_ratio: float = 4.0
    reg_dropout: float = 0.15

    # ── Flow Network ──
    n_latent_tokens: int = 12      # 768 / 64 = 12
    hidden_dim: int = 512
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.15

    # ── Regularization ──
    stochastic_depth_rate: float = 0.2   # max drop rate (linearly scaled)
    context_noise_std: float = 0.1       # noise augmentation on DINOv2


# ─── Stochastic Depth ────────────────────────────────────────────────────────

def drop_path(x, drop_prob: float, training: bool):
    """Drop entire samples (batch dimension) for stochastic depth."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x * random_tensor / keep_prob


# ─── Cross-Attention Block with Layer Mixing ──────────────────────────────────

class MultiLayerCrossBlock(nn.Module):
    """
    Transformer block with per-block learned layer mixing.
    Supports: adaLN (flow) or standard pre-norm (regression).
    Supports: stochastic depth.
    """

    def __init__(self, dim, num_heads, n_layers,
                 mlp_ratio=4.0, dropout=0.0, use_adaln=False,
                 drop_path_rate=0.0):
        super().__init__()
        self.use_adaln = use_adaln
        self.drop_path_rate = drop_path_rate

        # Learnable layer mixing weights
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

        # Cross-attention
        self.norm_q = nn.LayerNorm(dim, elementwise_affine=not use_adaln,
                                    eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, elementwise_affine=not use_adaln,
                                     eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout)

        # Self-attention
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

        if use_adaln:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def get_layer_weights(self):
        return F.softmax(self.layer_logits, dim=0)

    def forward(self, x, multi_ctx, c=None):
        alpha = F.softmax(self.layer_logits, dim=0)
        ctx = sum(a * ctx_l for a, ctx_l in zip(alpha, multi_ctx))

        if self.use_adaln and c is not None:
            params = self.adaLN(c).chunk(9, dim=-1)
            s_sa, sc_sa, g_sa = params[0], params[1], params[2]
            s_ca, sc_ca, g_ca = params[3], params[4], params[5]
            s_ff, sc_ff, g_ff = params[6], params[7], params[8]

            # Self-Attention with adaLN
            h = modulate(self.norm_sa(x), s_sa, sc_sa)
            sa_out, _ = self.self_attn(h, h, h)
            x = x + g_sa.unsqueeze(1) * drop_path(
                sa_out, self.drop_path_rate, self.training)

            # Cross-Attention with adaLN
            q = modulate(self.norm_q(x), s_ca, sc_ca)
            kv = self.norm_kv(ctx)
            ca_out, _ = self.cross_attn(q, kv, kv)
            x = x + g_ca.unsqueeze(1) * drop_path(
                ca_out, self.drop_path_rate, self.training)

            # FFN with adaLN
            h = modulate(self.norm_ff(x), s_ff, sc_ff)
            x = x + g_ff.unsqueeze(1) * drop_path(
                self.ffn(h), self.drop_path_rate, self.training)
        else:
            # Standard pre-norm (regression)
            sa_in = self.norm_sa(x)
            sa_out, _ = self.self_attn(sa_in, sa_in, sa_in)
            x = x + drop_path(sa_out, self.drop_path_rate, self.training)

            q = self.norm_q(x)
            kv = self.norm_kv(ctx)
            ca_out, _ = self.cross_attn(q, kv, kv)
            x = x + drop_path(ca_out, self.drop_path_rate, self.training)

            x = x + drop_path(
                self.ffn(self.norm_ff(x)), self.drop_path_rate, self.training)

        return x


# ─── Regression Head ─────────────────────────────────────────────────────────

class MultiLayerRegressor(nn.Module):
    """Multi-layer DINOv2 → z̄ (deterministic mean prediction)."""

    def __init__(self, config: ResidualFlowSiTConfig):
        super().__init__()
        D = config.reg_hidden_dim
        L = config.n_dino_layers
        sd_rate = config.stochastic_depth_rate

        # Learnable query tokens
        self.queries = nn.Parameter(
            torch.randn(1, config.reg_n_queries, D) * 0.02)

        # Transformer blocks with stochastic depth
        depth = config.reg_depth
        self.blocks = nn.ModuleList([
            MultiLayerCrossBlock(
                D, config.reg_num_heads, L,
                config.reg_mlp_ratio, config.reg_dropout, use_adaln=False,
                drop_path_rate=sd_rate * i / max(depth - 1, 1))
            for i in range(depth)
        ])

        # Output: flatten queries → z̄
        self.norm_out = nn.LayerNorm(D)
        self.head = nn.Sequential(
            nn.Linear(config.reg_n_queries * D, D),
            nn.GELU(approximate='tanh'),
            nn.Dropout(config.reg_dropout),
            nn.Linear(D, config.latent_dim),
        )

    def forward(self, multi_ctx):
        """multi_ctx: list of (B, n_pool, D) → z̄ (B, latent_dim)"""
        B = multi_ctx[0].shape[0]

        x = self.queries.expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, multi_ctx)

        x = self.norm_out(x)
        x = x.reshape(B, -1)
        return self.head(x)

    def get_layer_mixing_weights(self):
        weights = []
        for block in self.blocks:
            weights.append(block.get_layer_weights().detach().cpu())
        return torch.stack(weights)


# ─── Main Model ──────────────────────────────────────────────────────────────

class ResidualFlowSiT(nn.Module):
    """
    Residual Flow Matching V2.

    Architecture:
        1. Shared context: DINOv2 → project → attention pool → pooled tokens
        2. Regression: pooled tokens → z̄ (deterministic)
        3. Flow: learns Δz = z_true − z̄ via conditional flow matching

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
        sd_rate = config.stochastic_depth_rate

        # ── Shared Context Processing ──
        # Project each DINOv2 layer (shared between reg + flow)
        self.context_projs = nn.ModuleList([
            nn.Linear(config.context_dim, D) for _ in range(L)
        ])
        self.register_buffer('context_pos_embed',
            torch.from_numpy(
                get_1d_sincos_pos_embed(D, config.n_context_tokens)
            ).unsqueeze(0))

        # Attention pooling: 257 → n_pool_tokens per layer
        self.pool_queries = nn.Parameter(
            torch.randn(L, config.n_pool_tokens, D))
        self.pool_attn = nn.ModuleList([
            nn.MultiheadAttention(
                D, config.num_heads, batch_first=True,
                dropout=config.dropout)
            for _ in range(L)
        ])
        nn.init.normal_(self.pool_queries, std=0.02)

        # ── Regression Head ──
        self.regressor = MultiLayerRegressor(config)

        # ── Flow Network ──
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

        # Flow SiT blocks with stochastic depth
        flow_depth = config.depth
        self.flow_blocks = nn.ModuleList([
            MultiLayerCrossBlock(
                D, config.num_heads, L,
                config.mlp_ratio, config.dropout, use_adaln=True,
                drop_path_rate=sd_rate * i / max(flow_depth - 1, 1))
            for i in range(flow_depth)
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

        # Gate biases → 0.1
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

    def _pool_context(self, dino_multilayer):
        """Shared context: project + pool 257 → n_pool_tokens per layer.
        Returns list of (B, n_pool, D)."""
        B = dino_multilayer.shape[0]
        L = self.config.n_dino_layers

        # Optional noise augmentation during training
        if self.training and self.config.context_noise_std > 0:
            dino_multilayer = dino_multilayer + (
                torch.randn_like(dino_multilayer) *
                self.config.context_noise_std)

        multi_ctx = []
        for i in range(L):
            layer_tokens = dino_multilayer[:, i, :, :].float()
            ctx = self.context_projs[i](layer_tokens)   # (B, 257, D)
            ctx = ctx + self.context_pos_embed[:, :ctx.shape[1], :]

            # Attention pool: 257 → n_pool_tokens
            q = self.pool_queries[i].unsqueeze(0).expand(B, -1, -1)
            ctx_pooled, _ = self.pool_attn[i](q, ctx, ctx)
            multi_ctx.append(ctx_pooled)

        return multi_ctx

    # ── Forward methods ──

    def forward_regression(self, dino_multilayer):
        """Deterministic prediction: DINOv2 → z̄.
        dino_multilayer: (B, L, 257, ctx_dim) → (B, latent_dim)"""
        multi_ctx = self._pool_context(dino_multilayer)
        return self.regressor(multi_ctx)

    def forward_flow(self, t, z_t, dino_multilayer):
        """Predict velocity v(z_t, t | DINOv2) for residual Δz."""
        B = z_t.shape[0]

        # Patchify z_t (residual)
        lat_tokens = self._patchify(z_t)
        lat_tokens = self.latent_proj_in(lat_tokens)
        lat_tokens = lat_tokens + self.latent_pos_embed

        # Shared context
        multi_ctx = self._pool_context(dino_multilayer)

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
        if cfg_scale == 1.0:
            return self.forward_flow(t, z_t, dino_multilayer)
        v_cond = self.forward_flow(t, z_t, dino_multilayer)
        v_uncond = self.forward_flow(
            t, z_t, torch.zeros_like(dino_multilayer))
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    # ── Analysis ──

    def get_layer_mixing_weights(self):
        """Return dict: 'reg' → (reg_depth, L), 'flow' → (depth, L)."""
        return {
            'reg': self.regressor.get_layer_mixing_weights(),
            'flow': torch.stack([
                block.get_layer_weights().detach().cpu()
                for block in self.flow_blocks
            ]),
        }

    def param_count(self):
        reg_p = sum(p.numel() for p in self.regressor.parameters())
        shared_p = (sum(p.numel() for p in self.context_projs.parameters()) +
                    sum(p.numel() for p in self.pool_attn.parameters()) +
                    self.pool_queries.numel())
        total = sum(p.numel() for p in self.parameters())
        flow_p = total - reg_p - shared_p
        return {"reg_M": reg_p / 1e6, "shared_M": shared_p / 1e6,
                "flow_M": flow_p / 1e6, "total_M": total / 1e6}

    def freeze_regression(self):
        """Freeze shared context pooling and regression head for Phase 2."""
        # 1. Shared Context
        self.context_projs.eval()
        for p in self.context_projs.parameters():
            p.requires_grad = False
            
        self.pool_queries.requires_grad = False
        self.pool_attn.eval()
        for p in self.pool_attn.parameters():
            p.requires_grad = False
            
        # 2. Regression Head
        self.regressor.eval()
        for p in self.regressor.parameters():
            p.requires_grad = False
