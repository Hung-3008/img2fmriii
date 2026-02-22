"""
Aligned Flow SiT — Stage 2 backbone with Cross-Attention.

Architecture:
    1. Alignment MLP: DINOv2 CLS → z_approx (1024) — rough fMRI latent estimate
    2. Patchify: z_t (1024) → (n_latent_tokens, token_dim) latent tokens
    3. SiT Blocks: Self-Attention among latent tokens +
                   Cross-Attention to DINOv2 patch tokens (257, 1024)
    4. Depatchify: latent tokens → flat velocity (1024)

Inspired by SynBrain's SiT but adapted for:
    - Image→Brain direction (CLIP→fMRI vs fMRI→CLIP)
    - Cross-attention (SynBrain only does self-attention since input=output dim)
    - Patchified 1D latent (SynBrain operates on native 2D patches)
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helpers ──────────────────────────────────────────────────────────────────

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def modulate(x, shift, scale):
    """adaLN modulation: x * (1 + scale) + shift. Supports 2D [B,D] and 3D [B,T,D]."""
    if shift.ndim == 2 and x.ndim == 3:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def get_1d_sincos_pos_embed(dim, length):
    """Generate 1D sinusoidal positional embedding (length, dim)."""
    pos = np.arange(length, dtype=np.float64)
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class AlignedFlowSiTConfig:
    latent_dim: int = 1024        # VAE latent dim (flat)
    context_dim: int = 1024       # DINOv2 token dim
    n_context_tokens: int = 257   # DINOv2: 1 CLS + 256 patches
    n_latent_tokens: int = 16     # patchify: 1024 / 16 = 64 dim per token
    hidden_dim: int = 512         # SiT hidden dim
    depth: int = 6                # number of SiT blocks
    num_heads: int = 8            # attention heads
    mlp_ratio: float = 4.0       # FFN expansion
    dropout: float = 0.1
    # Alignment
    align_hidden: int = 1024
    align_layers: int = 2
    align_type: str = 'mlp'       # 'mlp' (CLS only) or 'attn_pool' (full tokens)
    align_n_queries: int = 4      # number of learnable queries for attn_pool


# ─── Alignment MLP ────────────────────────────────────────────────────────────

class AlignmentMLP(nn.Module):
    """DINOv2 CLS → z_approx (1024). Small to avoid overfitting."""

    def __init__(self, in_dim, out_dim, hidden, n_layers, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionPoolAlignment(nn.Module):
    """DINOv2 full tokens (257, 1024) → z_approx (1024) via learned attention pooling.

    Architecture:
        1. Learnable queries cross-attend to all DINOv2 tokens (spatial selection)
        2. FFN refines pooled representation
        3. Flatten + MLP maps to z_approx
    """

    def __init__(self, token_dim=1024, latent_dim=1024, hidden=1024,
                 n_queries=4, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_queries = n_queries

        # Learnable queries: "what spatial features matter for fMRI?"
        self.queries = nn.Parameter(torch.randn(1, n_queries, token_dim) * 0.02)

        # Cross-attention: queries attend to DINOv2 tokens
        self.norm_q = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            token_dim, n_heads, batch_first=True, dropout=dropout)

        # FFN after attention
        self.norm_ff = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
            nn.Dropout(dropout),
        )

        # Final projection: flatten queries → latent_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(n_queries * token_dim),
            nn.Linear(n_queries * token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, dino_tokens):
        """(B, 257, 1024) → (B, 1024)"""
        B = dino_tokens.shape[0]
        q = self.queries.expand(B, -1, -1)        # (B, n_queries, 1024)

        # Cross-attention
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(dino_tokens)
        attn_out, _ = self.cross_attn(q_normed, kv_normed, kv_normed)
        q = q + attn_out                          # residual

        # FFN
        q = q + self.ffn(self.norm_ff(q))          # (B, n_queries, 1024)

        # Flatten and project
        pooled = q.reshape(B, -1)                  # (B, n_queries * 1024)
        return self.out_proj(pooled)                # (B, latent_dim)


# ─── SiT Block ────────────────────────────────────────────────────────────────

class CrossSiTBlock(nn.Module):
    """
    SiT block with:
        1. Self-Attention among latent tokens
        2. Cross-Attention from latent tokens → DINOv2 tokens
        3. FFN
    All with adaLN conditioning (shift/scale/gate × 3 sub-layers).
    Uses Xavier init (NOT zero-init) for immediate gradient flow.
    """

    def __init__(self, dim, num_heads, context_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)

        # Cross-attention
        self.norm_cross_q = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_cross_kv = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True,
                                                 dropout=dropout, kdim=context_dim, vdim=context_dim)

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

        # adaLN: 9 vectors (shift/scale/gate for self-attn, cross-attn, ffn)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        # Xavier init (applied by _basic_init), gate biases init to 0.1
        # so residual paths are active from epoch 1

    def forward(self, x, context, c):
        """
        x: (B, T_lat, D)       latent tokens
        context: (B, T_ctx, C) DINOv2 tokens
        c: (B, D)               conditioning (time embedding)
        """
        params = self.adaLN(c).chunk(9, dim=-1)
        s_sa, sc_sa, g_sa = params[0], params[1], params[2]      # self-attn
        s_ca, sc_ca, g_ca = params[3], params[4], params[5]      # cross-attn
        s_ff, sc_ff, g_ff = params[6], params[7], params[8]      # ffn

        # 1. Self-Attention
        h = modulate(self.norm1(x), s_sa, sc_sa)
        sa_out, _ = self.self_attn(h, h, h)
        x = x + g_sa.unsqueeze(1) * sa_out

        # 2. Cross-Attention (latent queries, DINOv2 keys/values)
        q = modulate(self.norm_cross_q(x), s_ca, sc_ca)
        kv = self.norm_cross_kv(context)
        ca_out, _ = self.cross_attn(q, kv, kv)
        x = x + g_ca.unsqueeze(1) * ca_out

        # 3. FFN
        h = modulate(self.norm2(x), s_ff, sc_ff)
        x = x + g_ff.unsqueeze(1) * self.ffn(h)

        return x


# ─── Final Layer ──────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        # No zero-init — Xavier applied by _basic_init

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ─── Main Model ──────────────────────────────────────────────────────────────

class AlignedFlowSiT(nn.Module):
    """
    Aligned Flow Matching with SiT cross-attention backbone.

    forward_align(dino_tokens) → z_approx
    forward_flow(t, z_t, dino_tokens) → velocity
    """

    def __init__(self, config: Optional[AlignedFlowSiTConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = AlignedFlowSiTConfig(**kwargs)
        self.config = config

        n_lat = config.n_latent_tokens
        token_dim = config.latent_dim // n_lat  # e.g. 1024/16 = 64
        D = config.hidden_dim

        # ── Alignment ──
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
        self.latent_proj_in = nn.Linear(token_dim, D)   # 64 → D
        self.latent_proj_out = nn.Linear(D, token_dim)  # D → 64
        self.n_latent_tokens = n_lat
        self.token_dim = token_dim

        # Positional embeddings (frozen, sincos)
        self.register_buffer('latent_pos_embed',
            torch.from_numpy(get_1d_sincos_pos_embed(D, n_lat)).unsqueeze(0))

        # ── Time embedder ──
        self.t_embed = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))

        # ── DINOv2 context projection (1024 → D) ──
        self.context_proj = nn.Linear(config.context_dim, D)
        self.register_buffer('context_pos_embed',
            torch.from_numpy(get_1d_sincos_pos_embed(D, config.n_context_tokens)).unsqueeze(0))

        # ── SiT Blocks ──
        self.blocks = nn.ModuleList([
            CrossSiTBlock(D, config.num_heads, D, config.mlp_ratio, config.dropout)
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

        # Init gate biases to 0.1 so residual paths are active from epoch 1
        for block in self.blocks:
            # adaLN outputs 9*D: [shift,scale,gate]×3
            # gate indices are at positions 2,5,8 (every 3rd starting from idx 2)
            D = self.config.hidden_dim
            with torch.no_grad():
                bias = block.adaLN[1].bias.data.view(9, D)
                bias[2, :] = 0.1   # gate_self_attn
                bias[5, :] = 0.1   # gate_cross_attn
                bias[8, :] = 0.1   # gate_ffn

        # Output projection: small init (not zero, not too large)
        nn.init.normal_(self.latent_proj_out.weight, std=0.02)
        nn.init.zeros_(self.latent_proj_out.bias)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

        # Time embedder
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)

    def _patchify(self, z):
        """(B, latent_dim) → (B, n_tokens, token_dim)"""
        return z.view(z.shape[0], self.n_latent_tokens, self.token_dim)

    def _depatchify(self, tokens):
        """(B, n_tokens, token_dim) → (B, latent_dim)"""
        return tokens.reshape(tokens.shape[0], -1)

    def forward_align(self, dino_tokens):
        """DINOv2 tokens (B, 257, 1024) or CLS (B, 1024) → z_approx (B, 1024)."""
        if self._align_type == 'attn_pool':
            # Attention pooling uses all tokens
            if dino_tokens.ndim == 2:
                dino_tokens = dino_tokens.unsqueeze(1)  # fallback
            return self.align(dino_tokens)
        else:
            # MLP uses CLS only
            if dino_tokens.ndim == 3:
                cls = dino_tokens[:, 0, :]
            else:
                cls = dino_tokens
            return self.align(cls)

    def forward_flow(self, t, z_t, dino_tokens):
        """
        Predict velocity v(z_t, t | dino_tokens).

        Args:
            t: (B,) timestep
            z_t: (B, 1024) noisy/interpolated latent
            dino_tokens: (B, 257, 1024) full DINOv2 features
        """
        if dino_tokens.ndim == 2:
            dino_tokens = dino_tokens.unsqueeze(1)  # (B, 1, D) fallback

        B = z_t.shape[0]

        # Patchify z_t → latent tokens
        lat_tokens = self._patchify(z_t)                        # (B, 16, 64)
        lat_tokens = self.latent_proj_in(lat_tokens)            # (B, 16, D)
        lat_tokens = lat_tokens + self.latent_pos_embed          # + pos embed

        # Project DINOv2 context
        ctx = self.context_proj(dino_tokens.float())            # (B, 257, D)
        ctx = ctx + self.context_pos_embed[:, :ctx.shape[1], :] # + pos embed

        # Time conditioning
        t_emb = self.t_embed(timestep_embedding(t * 1000, self.config.hidden_dim))  # (B, D)

        # SiT blocks: self-attn + cross-attn + FFN
        for block in self.blocks:
            lat_tokens = block(lat_tokens, ctx, t_emb)

        # Final layer
        lat_tokens = self.final_layer(lat_tokens, t_emb)        # (B, 16, D)

        # Depatchify → velocity
        lat_tokens = self.latent_proj_out(lat_tokens)           # (B, 16, 64)
        return self._depatchify(lat_tokens)                      # (B, 1024)

    def forward_flow_with_cfg(self, t, z_t, dino_tokens, cfg_scale=1.0):
        if cfg_scale == 1.0:
            return self.forward_flow(t, z_t, dino_tokens)
        v_cond = self.forward_flow(t, z_t, dino_tokens)
        v_uncond = self.forward_flow(t, z_t, torch.zeros_like(dino_tokens))
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def param_count(self):
        align_p = sum(p.numel() for p in self.align.parameters())
        total = sum(p.numel() for p in self.parameters())
        flow_p = total - align_p
        return {"align_M": align_p/1e6, "flow_M": flow_p/1e6, "total_M": total/1e6}
