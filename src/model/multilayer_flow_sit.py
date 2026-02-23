"""
MultiLayerFlowSiT — Flow Matching with multi-layer DINOv2 conditioning.

Supports three context modes:
  1. "cross_attention" (original): Each DiT block cross-attends to all 257
     DINOv2 tokens with per-block learned layer mixing weights α.
  2. "attention_pool": Like cross_attention, but pools the 257 tokens down
     to `n_pool_tokens` (e.g. 32) per layer using learnable queries before
     the SiT blocks. Prevents overfitting while preserving spatial info.
  3. "cls_concat": Extract CLS token from each DINOv2 layer, concat,
     project via MLP → conditioning vector. No cross-attention.

After training, the mixing/α weights can be analyzed:
    - Early blocks attending to early DINOv2 layers → V1/V2 alignment
    - Late blocks attending to late DINOv2 layers → FFA/PPA alignment
    → Interpretable neuroscience insight.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.aligned_flow_mlp import (
    timestep_embedding, modulate, get_1d_sincos_pos_embed,
    FinalLayer,
)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class MultiLayerFlowSiTConfig:
    latent_dim: int = 1024
    context_dim: int = 1024        # DINOv2 token dim per layer
    n_context_tokens: int = 257    # CLS + 256 patches
    n_dino_layers: int = 4         # number of DINOv2 layers extracted
    n_latent_tokens: int = 16
    hidden_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    context_mode: str = "attention_pool"  # "cross_attention", "attention_pool" or "cls_concat"
    n_pool_tokens: int = 32        # Number of tokens to pool 257 down to


# ─── SiT Block (supports both modes) ─────────────────────────────────────────

class MultiLayerCrossSiTBlock(nn.Module):
    """
    SiT block with optional cross-attention.

    When use_cross_attn=True: SA → CA (to mixed multi-layer ctx) → FFN
    When use_cross_attn=False: SA → FFN (lighter, for cls_concat mode)
    """

    def __init__(self, dim, num_heads, n_layers, mlp_ratio=4.0, dropout=0.0,
                 use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout)

        if use_cross_attn:
            # Learnable layer mixing weights (initialized uniform)
            self.layer_logits = nn.Parameter(torch.zeros(n_layers))

            # Cross-attention (to mixed multi-layer context)
            self.norm_cross_q = nn.LayerNorm(dim, elementwise_affine=False,
                                              eps=1e-6)
            self.norm_cross_kv = nn.LayerNorm(dim, elementwise_affine=False,
                                               eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                dim, num_heads, batch_first=True, dropout=dropout)

            # adaLN: 9 vectors (shift/scale/gate for SA, CA, FFN)
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        else:
            # adaLN: 6 vectors (shift/scale/gate for SA, FFN only)
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        ffn_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def get_layer_weights(self):
        """Return softmax weights α for analysis (cross_attention mode only)."""
        if self.use_cross_attn:
            return F.softmax(self.layer_logits, dim=0)
        return None

    def forward(self, x, c, multi_ctx=None):
        """
        Args:
            x: (B, T_lat, D)       latent tokens
            c: (B, D)              conditioning (time + optional context)
            multi_ctx: list of (B, T_ctx, D) — only used if use_cross_attn
        """
        if self.use_cross_attn:
            return self._forward_cross_attn(x, c, multi_ctx)
        else:
            return self._forward_sa_only(x, c)

    def _forward_cross_attn(self, x, c, multi_ctx):
        # Mix layers: α = softmax(logits), ctx = Σ αᵢ * ctxᵢ
        alpha = F.softmax(self.layer_logits, dim=0)
        ctx = sum(a * ctx_l for a, ctx_l in zip(alpha, multi_ctx))

        # adaLN modulation params (9 vectors)
        params = self.adaLN(c).chunk(9, dim=-1)
        s_sa, sc_sa, g_sa = params[0], params[1], params[2]
        s_ca, sc_ca, g_ca = params[3], params[4], params[5]
        s_ff, sc_ff, g_ff = params[6], params[7], params[8]

        # 1. Self-Attention
        h = modulate(self.norm1(x), s_sa, sc_sa)
        sa_out, _ = self.self_attn(h, h, h)
        x = x + g_sa.unsqueeze(1) * sa_out

        # 2. Cross-Attention to mixed context
        q = modulate(self.norm_cross_q(x), s_ca, sc_ca)
        kv = self.norm_cross_kv(ctx)
        ca_out, _ = self.cross_attn(q, kv, kv)
        x = x + g_ca.unsqueeze(1) * ca_out

        # 3. FFN
        h = modulate(self.norm2(x), s_ff, sc_ff)
        x = x + g_ff.unsqueeze(1) * self.ffn(h)
        return x

    def _forward_sa_only(self, x, c):
        # adaLN modulation params (6 vectors: SA + FFN)
        params = self.adaLN(c).chunk(6, dim=-1)
        s_sa, sc_sa, g_sa = params[0], params[1], params[2]
        s_ff, sc_ff, g_ff = params[3], params[4], params[5]

        # 1. Self-Attention
        h = modulate(self.norm1(x), s_sa, sc_sa)
        sa_out, _ = self.self_attn(h, h, h)
        x = x + g_sa.unsqueeze(1) * sa_out

        # 2. FFN (no cross-attention)
        h = modulate(self.norm2(x), s_ff, sc_ff)
        x = x + g_ff.unsqueeze(1) * self.ffn(h)
        return x


# ─── Main Model ──────────────────────────────────────────────────────────────

class MultiLayerFlowSiT(nn.Module):
    """
    Multi-Layer DINOv2 Flow Matching with learned layer mixing.

    Supports three context modes:
      - "cross_attention": Full 257-token cross-attention per layer
      - "attention_pool": Pool 257 tokens → n_pool_tokens via Cross-Attention
      - "cls_concat": CLS tokens concatenated → MLP → adaLN conditioning

    forward_flow(t, z_t, dino_multilayer) → velocity
    get_layer_mixing_weights() → (depth, n_layers) matrix for analysis
    """

    def __init__(self, config: Optional[MultiLayerFlowSiTConfig] = None,
                 **kwargs):
        super().__init__()
        if config is None:
            config = MultiLayerFlowSiTConfig(**kwargs)
        self.config = config

        n_lat = config.n_latent_tokens
        token_dim = config.latent_dim // n_lat
        D = config.hidden_dim
        L = config.n_dino_layers
        self.use_cross_attn = config.context_mode in ["cross_attention", "attention_pool"]

        # ── Patchify / Depatchify ──
        self.latent_proj_in = nn.Linear(token_dim, D)
        self.latent_proj_out = nn.Linear(D, token_dim)
        self.n_latent_tokens = n_lat
        self.token_dim = token_dim

        self.register_buffer('latent_pos_embed',
            torch.from_numpy(
                get_1d_sincos_pos_embed(D, n_lat)).unsqueeze(0))

        # ── Time embedder ──
        self.t_embed = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))

        # ── Context processing ──
        if self.use_cross_attn:
            # Per-layer DINOv2 projections (full tokens)
            self.context_projs = nn.ModuleList([
                nn.Linear(config.context_dim, D) for _ in range(L)
            ])
            self.register_buffer('context_pos_embed',
                torch.from_numpy(
                    get_1d_sincos_pos_embed(D, config.n_context_tokens)
                ).unsqueeze(0))

            if config.context_mode == "attention_pool":
                # Learnable queries to pool 257 tokens -> n_pool_tokens per layer
                self.pool_queries = nn.Parameter(
                    torch.randn(L, config.n_pool_tokens, D))
                self.pool_attn = nn.ModuleList([
                    nn.MultiheadAttention(
                        D, config.num_heads, batch_first=True,
                        dropout=config.dropout)
                    for _ in range(L)
                ])
                # Small init for queries
                nn.init.normal_(self.pool_queries, std=0.02)
        else:
            # CLS concat: 4 CLS tokens → concat → MLP → D
            self.context_mlp = nn.Sequential(
                nn.Linear(L * config.context_dim, D * 2),
                nn.GELU(approximate='tanh'),
                nn.Dropout(config.dropout),
                nn.Linear(D * 2, D),
            )

        # ── SiT Blocks ──
        self.blocks = nn.ModuleList([
            MultiLayerCrossSiTBlock(
                D, config.num_heads, L, config.mlp_ratio, config.dropout,
                use_cross_attn=self.use_cross_attn)
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
        n_ada = 9 if self.use_cross_attn else 6
        gate_indices = [2, 5, 8] if self.use_cross_attn else [2, 5]

        for block in self.blocks:
            with torch.no_grad():
                bias = block.adaLN[1].bias.data.view(n_ada, D)
                for gi in gate_indices:
                    bias[gi, :] = 0.1

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

    def forward_flow(self, t, z_t, dino_multilayer):
        """
        Predict velocity v(z_t, t | multi-layer DINOv2).

        Args:
            t: (B,) timestep ∈ [0, 1]
            z_t: (B, latent_dim) noisy latent
            dino_multilayer: (B, L, 257, context_dim) multi-layer features
        """
        B = z_t.shape[0]
        L = self.config.n_dino_layers

        # Patchify z_t
        lat_tokens = self._patchify(z_t)
        lat_tokens = self.latent_proj_in(lat_tokens)
        lat_tokens = lat_tokens + self.latent_pos_embed

        # Time conditioning
        t_emb = self.t_embed(
            timestep_embedding(t * 1000, self.config.hidden_dim))

        if self.use_cross_attn:
            # Project each DINOv2 layer
            multi_ctx = []
            for i in range(L):
                layer_tokens = dino_multilayer[:, i, :, :].float()
                ctx = self.context_projs[i](layer_tokens)
                ctx = ctx + self.context_pos_embed[:, :ctx.shape[1], :]

                if self.config.context_mode == "attention_pool":
                    # Pool 257 tokens -> n_pool_tokens
                    # Q = learnable queries (expanded to batch), K=V=ctx
                    q = self.pool_queries[i].unsqueeze(0).expand(B, -1, -1)
                    ctx_pooled, _ = self.pool_attn[i](q, ctx, ctx)
                    ctx = ctx_pooled
                
                multi_ctx.append(ctx)

            # Pass through blocks with cross-attention
            for block in self.blocks:
                lat_tokens = block(lat_tokens, t_emb, multi_ctx)
        else:
            # CLS concat: extract CLS from each layer → concat → MLP
            cls_tokens = []
            for i in range(L):
                cls_tokens.append(dino_multilayer[:, i, 0, :].float())
            cls_cat = torch.cat(cls_tokens, dim=-1)  # (B, L * context_dim)
            ctx_emb = self.context_mlp(cls_cat)       # (B, D)

            # Combine time + context as conditioning
            c = t_emb + ctx_emb

            # Pass through blocks (SA + FFN only, no cross-attention)
            for block in self.blocks:
                lat_tokens = block(lat_tokens, c)

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

    def get_layer_mixing_weights(self):
        """Return (depth, n_layers) matrix of learned mixing weights.
        Returns None if context_mode is cls_concat."""
        if not self.use_cross_attn:
            return None
        weights = []
        for block in self.blocks:
            weights.append(block.get_layer_weights().detach().cpu())
        return torch.stack(weights)

    def param_count(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total_M": total/1e6}
