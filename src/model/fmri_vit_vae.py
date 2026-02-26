"""
fMRI Vision Transformer (BrainViT) VAE — Stage 1.

Replaces the massive MLP with a patch-based Transformer.
Architecture:
    Encoder: fMRI (15724) → Pad(15872) → Reshape(128 patches of size 124) 
             → Linear(512) → PosEmb → TransformerEncoder(x layers) → [CLS] token → μ, logvar (768)
    Decoder: Latent (768) → Linear(128 * 512) → Reshape(128, 512)
             → PosEmb → TransformerDecoder(x layers) → Linear(124) 
             → Reshape(15872) → Crop(15724)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class FmriViTVAEConfig:
    n_voxels: int = 15724
    patch_size: int = 124
    embed_dim: int = 512
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    latent_dim: int = 768
    dropout: float = 0.1
    # Controls KL and PCC weighting in loss computation (managed by config mostly)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class FmriViTVAE(nn.Module):
    """
    BrainViT-based VAE for fMRI data mapping to a global latent representation.
    """

    def __init__(self, config: Optional[FmriViTVAEConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = FmriViTVAEConfig(**kwargs)
        self.config = config

        self.n_voxels = config.n_voxels
        self.patch_size = config.patch_size
        
        # Calculate padding needed to make n_voxels divisible by patch_size
        if self.n_voxels % self.patch_size != 0:
            self.padded_voxels = ((self.n_voxels // self.patch_size) + 1) * self.patch_size
        else:
            self.padded_voxels = self.n_voxels
            
        self.num_patches = self.padded_voxels // self.patch_size
        self.pad_len = self.padded_voxels - self.n_voxels

        # ─── Encoder ───
        self.patch_embed = nn.Linear(self.patch_size, config.embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

        self.fc_mu = nn.Linear(config.embed_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.embed_dim, config.latent_dim)

        # ─── Decoder ───
        self.dec_embed = nn.Linear(config.latent_dim, self.num_patches * config.embed_dim)
        self.dec_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, config.embed_dim))
        self.dec_pos_drop = nn.Dropout(p=config.dropout)
        
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])
        self.dec_norm = nn.LayerNorm(config.embed_dim)
        
        # Project back to patch size
        self.patch_recover = nn.Linear(config.embed_dim, self.patch_size)

        self._init_weights()

    def _init_weights(self):
        # Initialize pos embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dec_pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
        # Zero init for logvar to start with posterior ~ N(0,1)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def encode(self, x: torch.Tensor, sample_posterior: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        # 1. Pad input
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))

        # 2. Patchify: (B, padded_voxels) -> (B, num_patches, patch_size)
        x = rearrange(x, 'b (n p) -> b n p', p=self.patch_size)

        # 3. Embed patches
        x = self.patch_embed(x) # (B, num_patches, embed_dim)

        # 4. Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches + 1, embed_dim)

        # 5. Add pos embed
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 6. Apply Transformer Blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 7. Extract CLS token representation
        cls_out = x[:, 0] # (B, embed_dim)

        # 8. Get latent stats
        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]

        # 1. Project to seq length * embed dim
        x = self.dec_embed(z) # (B, num_patches * embed_dim)
        x = rearrange(x, 'b (n d) -> b n d', n=self.num_patches)

        # 2. Add Pos Embed
        x = x + self.dec_pos_embed
        x = self.dec_pos_drop(x)

        # 3. Apply Decoder Blocks
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)

        # 4. Project sequence back to patch size
        x = self.patch_recover(x) # (B, num_patches, patch_size)

        # 5. Un-patchify: (B, num_patches, patch_size) -> (B, padded_voxels)
        x = rearrange(x, 'b n p -> b (n p)')

        # 6. Crop padding
        if self.pad_len > 0:
            x = x[:, :-self.pad_len]

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.01,
        lambda_pcc: float = 0.5,
    ) -> dict:
        # MSE reconstruction
        mse = F.mse_loss(x_recon, x)

        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # PCC loss (sample-wise)
        x_zm = x - x.mean(dim=1, keepdim=True)
        r_zm = x_recon - x_recon.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(x_zm, r_zm, dim=1).mean()
        pcc_loss = 1.0 - pcc

        # Total loss
        loss = mse + beta * kl + lambda_pcc * pcc_loss

        return {
            "loss": loss,
            "mse": mse,
            "kl": kl,
            "pcc_loss": pcc_loss,
            "pcc": pcc,
        }

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_mb": total * 4 / 1024 / 1024,
        }

# Factory
def create_fmri_vit_vae(**kwargs) -> FmriViTVAE:
    config = FmriViTVAEConfig(**kwargs)
    return FmriViTVAE(config)
