"""
regression_net.py
=================
Direct CLIP+DINOv2 → fMRI regression baseline.

Purpose: bound the conditional-mean ceiling of the visual features, to decide
whether the FactFlow plateau (~0.37 voxel_r) is a *feature* limit or a *decoder*
limit. This is the SAME backbone the flow uses (cross-attention over CLIP + DINO
tokens, LightningDiTBlock capacity) but trained as a plain regressor — no flow,
no timestep, no source encoder, no ODE. The target is predicted in a single
forward pass and trained with masked MSE, which is exactly the conditional-mean
objective that maximises the per-voxel encoding correlation on z-scored fMRI.

Input:
    clip_pool:   (B, D_pool)         — global conditioning (AdaLN)
    context:     (B, M1, D_tok)      — CLIP spatial tokens   (cross-attention)
    context2:    (B, M2, D_tok2)     — DINOv2 tokens         (cross-attention)
Output:
    (B, 1, seq_len) — predicted fMRI voxels (padded)
"""

import numpy as np
import torch
import torch.nn as nn

from .dit_1d import DiT1DFinalLayer
from .lightning_dit import LightningDiTBlock, VectorEmbedder
from .model_utils import RMSNorm, RotaryEmbedding1D, get_1d_sincos_pos_embed


class CrossAttnRegressor(nn.Module):
    """Learnable voxel-patch queries cross-attend to CLIP+DINOv2 tokens and are
    decoded straight to fMRI voxels. Structurally DiT1D minus the diffusion
    timestep and the noised input — a deterministic feature→fMRI map.
    """

    def __init__(
        self,
        seq_len: int = 16384,
        patch_size: int = 32,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        y_in_channels: int = 1280,
        y_token_channels: int = 1664,
        y_token_channels_2: int = None,
        context_dims=None,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        wo_shift: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.out_channels = 1
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        num_patches = seq_len // patch_size
        self.num_patches = num_patches

        # Learnable query tokens (the "what to predict" slots) + fixed pos-embed.
        self.query = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # Global conditioning (CLIP pooled) → AdaLN vector c.
        self.y_embedder = VectorEmbedder(y_in_channels, hidden_size)

        # Context stems: one embedder per cross-attention stream (CLIP, DINOv2,
        # multi-layer DINOv2, Gabor, …) → hidden, concatenated along tokens.
        if context_dims is None:
            context_dims = [y_token_channels]
            if y_token_channels_2 is not None:
                context_dims.append(y_token_channels_2)
        self.context_dims = list(context_dims)
        self.context_embedders = nn.ModuleList([
            nn.Sequential(nn.Linear(int(d), hidden_size), RMSNorm(hidden_size))
            for d in self.context_dims
        ])

        if use_rope:
            self.feat_rope = RotaryEmbedding1D(dim=hidden_size // num_heads, seq_len=num_patches)
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
                use_cross_attn=True,
            ) for _ in range(depth)
        ])

        self.final_layer = DiT1DFinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.query, std=0.02)

        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Zero-init AdaLN gates, final layer and cross-attn proj → stable start.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        c, p = self.out_channels, self.patch_size
        x = x.reshape(B, N, p, c).permute(0, 3, 1, 2).reshape(B, c, N * p)
        return x

    def forward(self, clip_pool, contexts):
        """clip_pool: (B, D_pool); contexts: list of (B, Mᵢ, Dᵢ) cross-attn streams."""
        B = clip_pool.shape[0]
        x = self.query.expand(B, -1, -1) + self.pos_embed
        c = self.y_embedder(clip_pool)

        embedded = [emb(ctx) for emb, ctx in zip(self.context_embedders, contexts)]
        ctx = torch.cat(embedded, dim=1) if len(embedded) > 1 else embedded[0]

        for block in self.blocks:
            x = block(x, c, feat_rope=self.feat_rope, context=ctx)
        x = self.final_layer(x, c)
        return self.unpatchify(x)  # (B, 1, seq_len)
