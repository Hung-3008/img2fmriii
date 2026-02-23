"""
MultiLayerFlowSiT — Flow Matching with multi-layer DINOv2 conditioning.

Extends AlignedFlowSiT: instead of attending to a single DINOv2 layer,
each DiT block has learnable mixing weights α ∈ R⁴ (softmax) over 4
DINOv2 layers, allowing the model to automatically discover which DNN
layers are most useful at each processing stage.

After training, the α weights can be analyzed:
    - Early blocks attending to early DINOv2 layers → V1/V2 alignment
    - Late blocks attending to late DINOv2 layers → FFA/PPA alignment
    → Interpretable neuroscience insight.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.aligned_flow_mlp import (
    timestep_embedding, modulate, get_1d_sincos_pos_embed,
    AlignmentMLP, AttentionPoolAlignment, FinalLayer,
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
    # Alignment
    align_hidden: int = 1024
    align_layers: int = 2
    align_type: str = 'mlp'
    align_n_queries: int = 4


# ─── Multi-Layer Cross-Attention SiT Block ───────────────────────────────────

class MultiLayerCrossSiTBlock(nn.Module):
    """
    SiT block with per-block learned layer mixing for multi-layer DINOv2.

    Instead of a fixed context, this block receives n_layers projected
    contexts and mixes them using learnable softmax weights α.

    The α weights are the key interpretable output of this architecture.
    """

    def __init__(self, dim, num_heads, n_layers, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        # Learnable layer mixing weights (initialized uniform)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout)

        # Cross-attention (to mixed multi-layer context)
        self.norm_cross_q = nn.LayerNorm(dim, elementwise_affine=False,
                                          eps=1e-6)
        self.norm_cross_kv = nn.LayerNorm(dim, elementwise_affine=False,
                                           eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout)

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

        # adaLN: 9 vectors (shift/scale/gate for SA, CA, FFN)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))

    def get_layer_weights(self):
        """Return softmax weights α for analysis."""
        return F.softmax(self.layer_logits, dim=0)

    def forward(self, x, multi_ctx, c):
        """
        Args:
            x: (B, T_lat, D)       latent tokens
            multi_ctx: list of (B, T_ctx, D)  per-layer projected contexts
            c: (B, D)              time conditioning
        """
        # Mix layers: α = softmax(logits), ctx = Σ αᵢ * ctxᵢ
        alpha = F.softmax(self.layer_logits, dim=0)  # (n_layers,)
        ctx = sum(a * ctx_l for a, ctx_l in zip(alpha, multi_ctx))

        # adaLN modulation params
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


# ─── Main Model ──────────────────────────────────────────────────────────────

class MultiLayerFlowSiT(nn.Module):
    """
    Multi-Layer DINOv2 Flow Matching with learned layer mixing.

    Key difference from AlignedFlowSiT:
        - Takes (B, n_layers, 257, 1024) multi-layer DINOv2 features
        - Each DiT block learns its own layer mixing weights α
        - Pure flow matching (no auxiliary ROI loss)

    forward_align(dino_multilayer) → z_approx
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

        # ── Alignment (uses final layer CLS / all-layer pooling) ──
        if config.align_type == 'attn_pool':
            self.align = AttentionPoolAlignment(
                token_dim=config.context_dim, latent_dim=config.latent_dim,
                hidden=config.align_hidden, n_queries=config.align_n_queries,
                n_heads=config.num_heads, dropout=config.dropout)
            self._align_type = 'attn_pool'
        else:
            self.align = AlignmentMLP(
                config.context_dim, config.latent_dim,
                config.align_hidden, config.align_layers, config.dropout)
            self._align_type = 'mlp'

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

        # ── Per-layer DINOv2 projections ──
        self.context_projs = nn.ModuleList([
            nn.Linear(config.context_dim, D) for _ in range(L)
        ])
        self.register_buffer('context_pos_embed',
            torch.from_numpy(
                get_1d_sincos_pos_embed(D, config.n_context_tokens)
            ).unsqueeze(0))

        # ── Multi-Layer SiT Blocks ──
        self.blocks = nn.ModuleList([
            MultiLayerCrossSiTBlock(
                D, config.num_heads, L, config.mlp_ratio, config.dropout)
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
        for block in self.blocks:
            with torch.no_grad():
                bias = block.adaLN[1].bias.data.view(9, D)
                bias[2, :] = 0.1
                bias[5, :] = 0.1
                bias[8, :] = 0.1

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

    def forward_align(self, dino_multilayer):
        """Use final layer for alignment.
        dino_multilayer: (B, L, 257, 1024) → z_approx (B, 1024)"""
        # Use last layer's tokens for alignment
        final_tokens = dino_multilayer[:, -1, :, :]  # (B, 257, 1024)
        if self._align_type == 'attn_pool':
            return self.align(final_tokens)
        else:
            return self.align(final_tokens[:, 0, :])  # CLS

    def forward_flow(self, t, z_t, dino_multilayer):
        """
        Predict velocity v(z_t, t | multi-layer DINOv2).

        Args:
            t: (B,) timestep ∈ [0, 1]
            z_t: (B, 1024) noisy latent
            dino_multilayer: (B, L, 257, 1024) multi-layer features
        """
        B = z_t.shape[0]
        L = self.config.n_dino_layers

        # Patchify z_t
        lat_tokens = self._patchify(z_t)
        lat_tokens = self.latent_proj_in(lat_tokens)
        lat_tokens = lat_tokens + self.latent_pos_embed

        # Project each DINOv2 layer separately
        multi_ctx = []
        for i in range(L):
            layer_tokens = dino_multilayer[:, i, :, :].float()  # (B, 257, D_in)
            ctx = self.context_projs[i](layer_tokens)  # (B, 257, D)
            ctx = ctx + self.context_pos_embed[:, :ctx.shape[1], :]
            multi_ctx.append(ctx)

        # Time conditioning
        t_emb = self.t_embed(
            timestep_embedding(t * 1000, self.config.hidden_dim))

        # Multi-layer SiT blocks
        for block in self.blocks:
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

    def get_layer_mixing_weights(self):
        """Return (depth, n_layers) matrix of learned mixing weights."""
        weights = []
        for block in self.blocks:
            weights.append(block.get_layer_weights().detach().cpu())
        return torch.stack(weights)  # (depth, n_layers)

    def param_count(self):
        align_p = sum(p.numel() for p in self.align.parameters())
        total = sum(p.numel() for p in self.parameters())
        flow_p = total - align_p
        return {"align_M": align_p/1e6, "flow_M": flow_p/1e6,
                "total_M": total/1e6}
