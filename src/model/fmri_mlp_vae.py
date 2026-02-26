"""
fMRI MLP VAE — Stage 1 for Latent Flow Matching.

Architecture (v2 — progressive):
    fMRI (15724) → Linear(4096) → ResBlock×N → Linear(2048) → ResBlock×N → μ,logvar (768)
    z (768) → Linear(2048) → ResBlock×N → Linear(4096) → ResBlock×N → Linear(15724)

Key improvements over v1:
    1. Progressive compression — gradual dim reduction, not abrupt
    2. Wider ResBlocks — expansion ratio for more expressivity
    3. Backward compatible — single hidden_dim still works
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────

class MLPResBlock(nn.Module):
    """Residual MLP block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout + skip.
    
    Supports expansion ratio: dim → dim*expansion → dim.
    """

    def __init__(self, dim: int, expansion: int = 1, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# ─── Model ────────────────────────────────────────────────────────────────────

@dataclass
class FmriMLPVAEConfig:
    """Configuration for FmriMLPVAE."""
    n_voxels: int = 15724
    hidden_dim: int = 2048           # used if hidden_dims is empty (backward compat)
    hidden_dims: List[int] = field(default_factory=list)  # progressive: [4096, 2048]
    latent_dim: int = 768
    n_res_blocks: int = 4            # per hidden level (v1) or per stage (v2)
    expansion: int = 1               # ResBlock expansion ratio (1 = original, 2 = wider)
    dropout: float = 0.1


class FmriMLPVAE(nn.Module):
    """
    MLP-based VAE for fMRI data.

    Encodes flat fMRI vectors (n_voxels,) into compact latent codes (latent_dim,)
    suitable for downstream Flow Matching.

    Architecture (progressive mode, hidden_dims=[4096, 2048]):
        Encoder: 15724 → 4096 → ResBlock×N(4096) → 2048 → ResBlock×N(2048) → μ,logvar(768)
        Decoder: 768 → 2048 → ResBlock×N(2048) → 4096 → ResBlock×N(4096) → 15724

    Architecture (legacy mode, hidden_dim=2048):
        Encoder: 15724 → 2048 → ResBlock×N(2048) → μ,logvar(768)
        Decoder: 768 → 2048 → ResBlock×N(2048) → 15724
    """

    def __init__(self, config: Optional[FmriMLPVAEConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = FmriMLPVAEConfig(**kwargs)
        self.config = config

        n_voxels = config.n_voxels
        latent = config.latent_dim
        n_blocks = config.n_res_blocks
        dropout = config.dropout
        expansion = config.expansion

        # Determine hidden dim progression
        if config.hidden_dims:
            dims = config.hidden_dims  # e.g. [4096, 2048]
        else:
            dims = [config.hidden_dim]  # e.g. [2048] — backward compat

        # ── Encoder: n_voxels → dims[0] → dims[1] → ... → latent ──
        encoder_layers = []
        prev_dim = n_voxels
        for d in dims:
            encoder_layers.append(nn.Linear(prev_dim, d))
            encoder_layers.append(nn.LayerNorm(d))
            encoder_layers.append(nn.GELU())
            for _ in range(n_blocks):
                encoder_layers.append(MLPResBlock(d, expansion, dropout))
            prev_dim = d

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent)
        self.fc_logvar = nn.Linear(prev_dim, latent)

        # ── Decoder: latent → dims[-1] → ... → dims[0] → n_voxels ──
        decoder_layers = []
        rev_dims = list(reversed(dims))  # e.g. [2048, 4096]
        prev_dim = latent
        for d in rev_dims:
            decoder_layers.append(nn.Linear(prev_dim, d))
            decoder_layers.append(nn.LayerNorm(d))
            decoder_layers.append(nn.GELU())
            for _ in range(n_blocks):
                decoder_layers.append(MLPResBlock(d, expansion, dropout))
            prev_dim = d
        decoder_layers.append(nn.Linear(prev_dim, n_voxels))

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
        # Bug fix: fc_logvar should start near 0 so initial posterior ≈ N(0,1)
        # Xavier init could produce large logvar → exp(logvar) blows up → KL/NaN
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

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

        # Bug fix: clamp logvar to prevent exp() overflow/underflow
        # logvar=30 → std=e^15 ≈ 3.3M (gradient explosion)
        # logvar=-30 → std≈0 (vanishing gradients, posterior collapse)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

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
