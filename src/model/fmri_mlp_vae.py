"""
fMRI MLP VAE — Stage 1 for Latent Flow Matching.

Architecture:
    fMRI (15724) → MLP Encoder → z ~ N(μ, σ²) (latent_dim)
    z (latent_dim) → MLP Decoder → fMRI (15724)

Key improvements over Conv2D VAE:
    1. Direct 1D processing — no artificial 2D reshape/padding
    2. MLP with residual blocks — correct inductive bias for fMRI
    3. Higher β — enforces Gaussian prior for flow matching
    4. Dropout — prevents latent collapse

Inspired by MindEye2's MLP backbone (Linear→ResBlock×4).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class MLPResBlock(nn.Module):
    """Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─── Model ────────────────────────────────────────────────────────────────────

@dataclass
class FmriMLPVAEConfig:
    """Configuration for FmriMLPVAE."""
    n_voxels: int = 15724
    hidden_dim: int = 2048
    latent_dim: int = 1024
    n_res_blocks: int = 4
    dropout: float = 0.1


class FmriMLPVAE(nn.Module):
    """
    MLP-based VAE for fMRI data.

    Encodes flat fMRI vectors (n_voxels,) into compact latent codes (latent_dim,)
    suitable for downstream Flow Matching.

    Architecture:
        Encoder: Linear(n_voxels → hidden) → ResBlock × N → Linear(hidden → 2*latent)
        Decoder: Linear(latent → hidden) → ResBlock × N → Linear(hidden → n_voxels)
    """

    def __init__(self, config: Optional[FmriMLPVAEConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = FmriMLPVAEConfig(**kwargs)
        self.config = config

        n_voxels = config.n_voxels
        hidden = config.hidden_dim
        latent = config.latent_dim
        n_blocks = config.n_res_blocks
        dropout = config.dropout

        # ── Encoder ──
        encoder_layers = [
            nn.Linear(n_voxels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
        for _ in range(n_blocks):
            encoder_layers.append(MLPResBlock(hidden, dropout))

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

        # ── Decoder ──
        decoder_layers = [
            nn.Linear(latent, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
        for _ in range(n_blocks):
            decoder_layers.append(MLPResBlock(hidden, dropout))
        decoder_layers.append(nn.Linear(hidden, n_voxels))

        self.decoder = nn.Sequential(*decoder_layers)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(
        self, x: torch.Tensor, sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode fMRI → latent.

        Args:
            x: (B, n_voxels) flat fMRI data.
            sample_posterior: If True, sample z ~ N(μ, σ²); else z = μ.

        Returns:
            z: (B, latent_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent → fMRI.

        Args:
            z: (B, latent_dim)

        Returns:
            x_recon: (B, n_voxels)
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: (B, n_voxels) flat fMRI data.

        Returns:
            x_recon: (B, n_voxels)
            z: (B, latent_dim)
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
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
        """
        Compute VAE loss: MSE + β·KL + λ_pcc·(1-PCC).

        Args:
            x: (B, n_voxels) original
            x_recon: (B, n_voxels) reconstruction
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
            beta: KL weight
            lambda_pcc: PCC loss weight

        Returns:
            dict with 'loss', 'mse', 'kl', 'pcc_loss', 'pcc'
        """
        # MSE reconstruction
        mse = F.mse_loss(x_recon, x)

        # KL divergence: -0.5 * Σ(1 + logvar - μ² - σ²)
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


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_fmri_mlp_vae(**kwargs) -> FmriMLPVAE:
    """Create FmriMLPVAE from keyword arguments."""
    config = FmriMLPVAEConfig(**kwargs)
    return FmriMLPVAE(config)
