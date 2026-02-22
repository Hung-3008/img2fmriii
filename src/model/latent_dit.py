"""
1D Latent DiT — Flow Matching backbone for Stage 2.

Reshape flat fMRI latent [B, 1024] into token sequence [B, N, D],
then apply Transformer blocks with:
  - adaLN-Zero modulated Self-Attention (latent tokens attend each other)
  - Cross-Attention (latent tokens attend CLIP patch tokens)
  - adaLN-Zero modulated FFN

Inspired by:
  - DiT (Peebles & Xie 2023) — adaLN-Zero conditioning
  - LFM (NVIDIA) — latent flow matching with DiT
  - Stable Diffusion — cross-attention for continuous conditioning
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helpers ──────────────────────────────────────────────────────────────────


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaLN modulation: scale * x + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (same as DiT/LFM)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ─── Blocks ───────────────────────────────────────────────────────────────────


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q from latent tokens, K/V from context (CLIP)."""

    def __init__(self, dim: int, context_dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] latent tokens
            context: [B, M, C] CLIP tokens (M=257, C=1024)
        Returns:
            [B, N, D]
        """
        B, N, _ = x.shape
        M = context.shape[1]
        H = self.n_heads

        q = self.q_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, M, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, M, H, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QKV projection."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.n_heads

        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class DiTBlock(nn.Module):
    """
    DiT block with cross-attention:

    1. adaLN-Zero → Self-Attention (gated residual)
    2. Cross-Attention → CLIP context (gated residual)
    3. adaLN-Zero → FFN (gated residual)
    """

    def __init__(self, hidden_size: int, context_dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(hidden_size, n_heads, dropout)

        # Cross-attention
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, context_dim, n_heads, dropout)
        self.cross_gate = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size))

        # FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        # adaLN-Zero: produces (shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Init gate projections to zero → residual blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.cross_gate[1].weight)
        nn.init.zeros_(self.cross_gate[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] latent tokens
            c: [B, D] timestep conditioning embedding
            context: [B, M, C] CLIP tokens
        """
        shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(6, dim=1)

        # 1. Self-Attention with adaLN-Zero
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))

        # 2. Cross-Attention to CLIP context (with learned gate)
        cross_gate = self.cross_gate(c).unsqueeze(1)  # [B, 1, D]
        x = x + cross_gate * self.cross_attn(self.norm_cross(x), context)

        # 3. FFN with adaLN-Zero
        x = x + gate_ffn.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))

        return x


class FinalLayer(nn.Module):
    """Final adaLN-Zero modulated layer → linear projection."""

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ─── Main Model ──────────────────────────────────────────────────────────────


@dataclass
class LatentDiTConfig:
    """Configuration for LatentDiT."""

    latent_dim: int = 1024       # Total flat latent dimension
    n_tokens: int = 16           # Number of latent tokens (latent_dim / token_dim)
    hidden_size: int = 512       # Transformer hidden dimension
    depth: int = 8               # Number of DiT blocks
    n_heads: int = 8             # Attention heads
    mlp_ratio: float = 4.0      # FFN expansion ratio
    context_dim: int = 1024      # CLIP feature dimension
    dropout: float = 0.1         # Dropout rate


class LatentDiT(nn.Module):
    """
    1D Latent DiT for conditional flow matching.

    Reshapes flat latent into token sequence, applies DiT blocks with
    cross-attention to CLIP features, then reshapes back.

    Args:
        config: LatentDiTConfig (or pass kwargs directly)
    """

    def __init__(self, config: LatentDiTConfig = None, **kwargs):
        super().__init__()

        if config is None:
            config = LatentDiTConfig(**kwargs)
        self.config = config

        latent_dim = config.latent_dim
        n_tokens = config.n_tokens
        token_dim = latent_dim // n_tokens
        hidden = config.hidden_size

        assert latent_dim % n_tokens == 0, f"latent_dim {latent_dim} must be divisible by n_tokens {n_tokens}"

        self.n_tokens = n_tokens
        self.token_dim = token_dim

        # ── Embeddings ──
        self.input_proj = nn.Linear(token_dim, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, hidden))

        # Timestep embedding
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Context projection (CLIP → hidden)
        self.context_proj = nn.Linear(config.context_dim, hidden)

        # ── Transformer blocks ──
        self.blocks = nn.ModuleList([
            DiTBlock(hidden, hidden, config.n_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])

        # ── Output ──
        self.final_layer = FinalLayer(hidden, token_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        # Initialize pos_embed
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize projections
        for module in [self.input_proj, self.context_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        # Initialize t_embedder
        for layer in self.t_embedder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """Reshape flat latent to token sequence: [B, 1024] → [B, N, token_dim]."""
        return z.reshape(z.shape[0], self.n_tokens, self.token_dim)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape token sequence to flat latent: [B, N, token_dim] → [B, 1024]."""
        return tokens.reshape(tokens.shape[0], -1)

    def forward(self, t: torch.Tensor, z_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field v(z_t, t, context).

        Args:
            t: [B] timestep ∈ [0, 1]
            z_t: [B, D] flat latent OR [B, N, token_dim] pre-tokenized
            context: [B, M, C] CLIP features

        Returns:
            v_pred: same shape as z_t
        """
        input_is_3d = z_t.ndim == 3

        # Timestep embedding
        t_emb = timestep_embedding(t, self.config.hidden_size)
        c = self.t_embedder(t_emb)

        # Project CLIP context
        ctx = self.context_proj(context)

        # Tokenize if flat
        if input_is_3d:
            x = self.input_proj(z_t) + self.pos_embed
        else:
            x = self.patchify(z_t)
            x = self.input_proj(x) + self.pos_embed

        # DiT blocks
        for block in self.blocks:
            x = block(x, c, ctx)

        # Final projection
        x = self.final_layer(x, c)

        # Return same shape as input
        if input_is_3d:
            return x
        else:
            return self.unpatchify(x)

    def forward_with_cfg(
        self, t: torch.Tensor, z_t: torch.Tensor, context: torch.Tensor, cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        if cfg_scale == 1.0:
            return self.forward(t, z_t, context)

        # Conditional
        v_cond = self.forward(t, z_t, context)
        # Unconditional (zero context)
        v_uncond = self.forward(t, z_t, torch.zeros_like(context))
        # CFG interpolation
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": total / 1e6}
