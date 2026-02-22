"""
fMRI Conv2D VAE for Latent Flow Matching (Stage 1).

Architecture:
    fMRI (15724,) → pad (16384,) → reshape (1, 128, 128)
        → Encoder → z ~ N(μ, σ²) (4, 16, 16)
        → Decoder → (1, 128, 128) → unpad → (15724,)

The latent space (4, 16, 16) = 1024 dims is designed for Stage 2 Flow Matching.

Usage:
    model = FmriVAE(n_voxels=15724)
    fmri_2d = pad_and_reshape(fmri_flat)  # (B, 1, 128, 128)
    recon, mu, logvar = model(fmri_2d)
    loss = model.compute_loss(fmri_2d, recon, mu, logvar, beta=0.001)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helper constants ─────────────────────────────────────────────────────────
N_VOXELS = 15724
TARGET_SIZE = 128  # 128 × 128 = 16384


# ─── Helper functions ─────────────────────────────────────────────────────────

def pad_fmri_to_2d(fmri_flat: torch.Tensor, target_size: int = TARGET_SIZE) -> torch.Tensor:
    """
    Pad and reshape flat fMRI vector to 2D pseudo-image.
    
    Args:
        fmri_flat: (B, n_voxels) flat fMRI data
        target_size: spatial size H = W
        
    Returns:
        (B, 1, H, W) pseudo-image
    """
    B, n_voxels = fmri_flat.shape
    padded_size = target_size * target_size
    if n_voxels < padded_size:
        fmri_flat = F.pad(fmri_flat, (0, padded_size - n_voxels), value=0.0)
    return fmri_flat.view(B, 1, target_size, target_size)


def unpad_fmri_from_2d(fmri_2d: torch.Tensor, n_voxels: int = N_VOXELS) -> torch.Tensor:
    """
    Reshape 2D pseudo-image back to flat fMRI vector and remove padding.
    
    Args:
        fmri_2d: (B, 1, H, W) pseudo-image
        n_voxels: number of real voxels
        
    Returns:
        (B, n_voxels) flat fMRI data
    """
    B = fmri_2d.shape[0]
    flat = fmri_2d.view(B, -1)
    return flat[:, :n_voxels]


def create_voxel_mask(n_voxels: int = N_VOXELS, target_size: int = TARGET_SIZE) -> torch.Tensor:
    """
    Create a binary mask for real voxels (exclude padding).
    
    Returns:
        (1, 1, H, W) boolean mask
    """
    padded_size = target_size * target_size
    mask = torch.zeros(padded_size, dtype=torch.bool)
    mask[:n_voxels] = True
    return mask.view(1, 1, target_size, target_size)


# ─── Building blocks ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    
    def __init__(self, channels: int, groups: int = 32):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(groups, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SelfAttention2d(nn.Module):
    """Self-attention for 2D feature maps."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, dim, HW)
        
        # Transpose for attention: (B, heads, HW, dim)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        return x + self.proj(attn)


class DownBlock(nn.Module):
    """Downsampling block: Conv(stride=2) → ResBlock × 2."""
    
    def __init__(self, in_ch: int, out_ch: int, use_attn: bool = False):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res1 = ResBlock(out_ch)
        self.res2 = ResBlock(out_ch)
        self.attn = SelfAttention2d(out_ch) if use_attn else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: Upsample → Conv → ResBlock × 2."""
    
    def __init__(self, in_ch: int, out_ch: int, use_attn: bool = False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.res1 = ResBlock(out_ch)
        self.res2 = ResBlock(out_ch)
        self.attn = SelfAttention2d(out_ch) if use_attn else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        return x


# ─── VAE Model ────────────────────────────────────────────────────────────────

@dataclass
class FmriVAEConfig:
    """Configuration for FmriVAE."""
    n_voxels: int = N_VOXELS
    target_size: int = TARGET_SIZE
    latent_channels: int = 4
    base_channels: int = 64
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)  # → 64, 128, 256, 512
    num_res_blocks: int = 2
    attn_at_bottleneck: bool = True
    dropout: float = 0.0


class FmriVAE(nn.Module):
    """
    Convolutional VAE for fMRI data.
    
    Encodes fMRI pseudo-images (1, 128, 128) into compact latent codes
    (latent_channels, 16, 16) for downstream Flow Matching.
    
    Architecture:
        Encoder: (1, 128, 128) → Down×4 → (512, 8, 8) → quant_conv → (4, 16, 16) [mu, logvar]
        Decoder: (4, 16, 16) → post_quant_conv → (512, 8, 8) → Up×4 → (1, 128, 128)
        
    Note: The encoder produces (512, 8, 8) from 4 downsampling stages (128→64→32→16→8),
    but we use an intermediate approach with the last DownBlock going to 16×16 spatial, 
    then quant_conv to latent channels at 16×16.
    """
    
    def __init__(self, config: Optional[FmriVAEConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = FmriVAEConfig(**kwargs)
        self.config = config
        
        ch = config.base_channels
        ch_mult = config.channel_mult
        latent_ch = config.latent_channels
        
        # Channel progression: 1 → 64 → 128 → 256 → 512
        channels = [ch * m for m in ch_mult]
        
        # ── Encoder ──
        # Input conv: (1, 128, 128) → (64, 128, 128)
        self.encoder_input = nn.Conv2d(1, channels[0], 3, padding=1)
        
        # Downsampling blocks
        # Stage 0: (64, 128, 128) → (64, 64, 64)   — no extra down, just res blocks at 128×128 first
        # Stage 1: (64, 64, 64) → (128, 32, 32)
        # Stage 2: (128, 32, 32) → (256, 16, 16)
        # Stage 3: this is the bottleneck res blocks at 16×16
        # Actually let's do 3 downsampling stages: 128→64→32→16
        
        self.encoder_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels[1:]):
            use_attn = (i == len(channels) - 2) and config.attn_at_bottleneck
            self.encoder_blocks.append(DownBlock(in_ch, out_ch, use_attn=use_attn))
            in_ch = out_ch
        
        # Bottleneck at 16×16
        self.encoder_mid = nn.Sequential(
            ResBlock(channels[-1]),
            SelfAttention2d(channels[-1]) if config.attn_at_bottleneck else nn.Identity(),
            ResBlock(channels[-1]),
        )
        
        # To latent: (512, 16, 16) → (2*latent_ch, 16, 16) for mu and logvar
        self.encoder_norm = nn.GroupNorm(32, channels[-1])
        self.quant_conv = nn.Conv2d(channels[-1], 2 * latent_ch, 1)
        
        # ── Decoder ──
        # From latent: (latent_ch, 16, 16) → (512, 16, 16)
        self.post_quant_conv = nn.Conv2d(latent_ch, channels[-1], 1)
        
        # Bottleneck at 16×16
        self.decoder_mid = nn.Sequential(
            ResBlock(channels[-1]),
            SelfAttention2d(channels[-1]) if config.attn_at_bottleneck else nn.Identity(),
            ResBlock(channels[-1]),
        )
        
        # Upsampling blocks (reverse order)
        self.decoder_blocks = nn.ModuleList()
        in_ch = channels[-1]
        for i, out_ch in enumerate(reversed(channels[:-1])):
            use_attn = (i == 0) and config.attn_at_bottleneck
            self.decoder_blocks.append(UpBlock(in_ch, out_ch, use_attn=use_attn))
            in_ch = out_ch
        
        # Output conv: (64, 128, 128) → (1, 128, 128)
        self.decoder_output = nn.Sequential(
            nn.GroupNorm(min(32, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], 1, 3, padding=1),
        )
        
        # Voxel mask for masked loss
        self.register_buffer(
            'voxel_mask',
            create_voxel_mask(config.n_voxels, config.target_size),
            persistent=False,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for Conv2d, zero for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Zero-init last conv for residual blocks
        for m in self.modules():
            if isinstance(m, ResBlock):
                nn.init.zeros_(m.block[-1].weight)
    
    def encode(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: (B, 1, 128, 128) pseudo-image
            sample_posterior: if True, sample z ~ N(mu, sigma²); else z = mu
            
        Returns:
            (z, mu, logvar) where z is (B, latent_ch, 16, 16)
        """
        h = self.encoder_input(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_mid(h)
        h = self.encoder_norm(h)
        h = F.silu(h)
        h = self.quant_conv(h)
        
        # Split into mu and logvar
        mu, logvar = h.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        
        return z, mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pseudo-image.
        
        Args:
            z: (B, latent_ch, 16, 16) latent code
            
        Returns:
            (B, 1, 128, 128) reconstructed pseudo-image
        """
        h = self.post_quant_conv(z)
        h = self.decoder_mid(h)
        for block in self.decoder_blocks:
            h = block(h)
        h = self.decoder_output(h)
        return h
    
    def forward(
        self, x: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode → sample → decode.
        
        Args:
            x: (B, 1, 128, 128) pseudo-image
            sample_posterior: if True, sample z; else z = mu
            
        Returns:
            (x_recon, mu, logvar)
        """
        z, mu, logvar = self.encode(x, sample_posterior)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor, 
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.001,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss with masked MSE (only on real voxels).
        
        Args:
            x: (B, 1, 128, 128) original
            x_recon: (B, 1, 128, 128) reconstruction
            mu: (B, latent_ch, 16, 16) mean
            logvar: (B, latent_ch, 16, 16) log variance
            beta: KL weight
            
        Returns:
            Dict with total_loss, recon_loss, kl_loss
        """
        # Masked MSE: only compute on real voxels (exclude padding)
        mask = self.voxel_mask.to(x.device)  # (1, 1, H, W)
        diff = (x - x_recon) ** 2
        recon_loss = (diff * mask).sum() / mask.sum() / x.shape[0]
        
        # KL divergence: D_KL(q(z|x) || p(z))
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
    
    def param_count(self) -> Dict[str, int]:
        """Count parameters by component."""
        enc_params = sum(
            p.numel() for n, p in self.named_parameters() 
            if 'encoder' in n or 'quant_conv' in n
        )
        dec_params = sum(
            p.numel() for n, p in self.named_parameters() 
            if 'decoder' in n or 'post_quant_conv' in n
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "total": total,
            "total_mb": total / 1024**2,
        }


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_fmri_vae(
    n_voxels: int = N_VOXELS,
    latent_channels: int = 4,
    base_channels: int = 64,
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
    **kwargs,
) -> FmriVAE:
    """Create an FmriVAE with the given configuration."""
    config = FmriVAEConfig(
        n_voxels=n_voxels,
        latent_channels=latent_channels,
        base_channels=base_channels,
        channel_mult=channel_mult,
        **kwargs,
    )
    return FmriVAE(config)
