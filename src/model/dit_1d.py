"""
dit_1d.py
=========
1D Diffusion Transformer for fMRI voxel synthesis.

Native 1D architecture — no 2D reshape needed.
Reuses LightningDiTBlock (AdaLN-Zero, SwiGLU, RMSNorm, QKNorm)
with 1D Conv1d patch embedding and 1D RoPE.

Input:  (B, 1, seq_len)  e.g. (B, 1, 16384)
Output: (B, 1, seq_len)  predicted velocity
"""

import torch
import torch.nn as nn
import numpy as np

from .lightning_dit import LightningDiTBlock, VectorEmbedder
from .model_utils import (
    GaussianFourierEmbedding,
    RMSNorm,
    RotaryEmbedding1D,
    get_1d_sincos_pos_embed,
    modulate,
)


# ─── 1D Patch Embedding ──────────────────────────────────────────────

class PatchEmbed1D(nn.Module):
    """Tokenize a 1D signal into non-overlapping patches via Conv1d."""

    def __init__(self, seq_len: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, num_patches, embed_dim)"""
        x = self.proj(x)        # (B, D, N)
        x = x.transpose(1, 2)   # (B, N, D)
        return x


# ─── 1D Final Layer ──────────────────────────────────────────────────

class DiT1DFinalLayer(nn.Module):
    """Project tokens back to patch space with AdaLN modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = False):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─── DiT1D ───────────────────────────────────────────────────────────

class DiT1D(nn.Module):
    """
    1D Diffusion Transformer for fMRI voxel synthesis.

    Replaces the 2D LightningDiT — operates directly on flat voxel
    vectors without artificial 2D reshape.

    Architecture:
        Conv1d(patch_size) → [LightningDiTBlock × depth] → Linear → unpatchify
        Conditioning: t (timestep) + y (CLIP pool) via AdaLN in each block.
    """

    def __init__(
        self,
        seq_len: int = 16384,
        patch_size: int = 32,
        in_channels: int = 1,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        y_in_channels: int = 1280,
        learn_sigma: bool = False,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        wo_shift: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.learn_sigma = learn_sigma
        self.depth = depth

        num_patches = seq_len // patch_size

        # ── Input stem ────────────────────────────────────────────────
        self.x_embedder = PatchEmbed1D(seq_len, patch_size, in_channels, hidden_size)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = VectorEmbedder(y_in_channels, hidden_size)

        # 1D positional embedding (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # 1D Rotary Position Embedding
        if use_rope:
            head_dim = hidden_size // num_heads
            self.feat_rope = RotaryEmbedding1D(dim=head_dim, seq_len=num_patches)
        else:
            self.feat_rope = None

        # ── Transformer blocks (reuse LightningDiTBlock) ─────────────
        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
            ) for _ in range(depth)
        ])

        # ── Output head ──────────────────────────────────────────────
        self.final_layer = DiT1DFinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init for all linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 1D sin-cos positional embedding
        num_patches = self.seq_len // self.patch_size
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning embedding MLP
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN layers → identity init for stable start
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_size * out_channels) → (B, out_channels, seq_len)
        """
        B, N, _ = x.shape
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(B, N, p, c)     # (B, N, P, C)
        x = x.permute(0, 3, 1, 2)     # (B, C, N, P)
        x = x.reshape(B, c, N * p)    # (B, C, N*P) = (B, C, seq_len)
        return x

    def forward(self, x, t=None, y=None):
        """
        Forward pass of DiT1D.

        Args:
            x: (B, C, L) — 1D fMRI signal (C=1 typically)
            t: (B,) — diffusion timestep
            y: (B, D_pool) — CLIP pooled conditioning

        Returns:
            (B, C, L) — predicted velocity
        """
        x = self.x_embedder(x) + self.pos_embed   # (B, N, D)
        t = self.t_embedder(t)                      # (B, D)
        y = self.y_embedder(y)                      # (B, D)
        c = t + y                                    # (B, D)

        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope)

        x = self.final_layer(x, c)   # (B, N, P*C)
        x = self.unpatchify(x)       # (B, C, L)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x
