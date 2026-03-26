"""fMRI VAE for NSD — Static MLP-based VAE for 15724-D fMRI.

Compresses 15724 raw voxels into a compact latent space (default 256-D).

Architecture:
    Encoder: Linear(V→H) → ResBlocks(H)×N → Linear→(μ, logσ²) = latent_dim
    Decoder: Linear(Z→H) → ResBlocks(H)×N → Linear→V
    Loss: MSE + λ_pcc * PCC_loss + β * KL (with free-bits and β-annealing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Helpers
# =============================================================================

def beta_schedule(epoch: int, beta_max: float = 0.01, warmup_epochs: int = 20) -> float:
    """Linear β-KL annealing: 0 → beta_max over warmup_epochs."""
    if warmup_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / warmup_epochs)


class ResBlock(nn.Module):
    """MLP residual block: LN → GELU → Linear → LN → GELU → Dropout → Linear."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# =============================================================================
# fMRI VAE for NSD
# =============================================================================

class fMRI_VAE_NSD(nn.Module):
    """Static MLP-based VAE for NSD fMRI data.

    Input:  (B, V)  where V = 15724 voxels
    Latent: (B, Z)  where Z = latent_dim
    Output: (B, V)  reconstructed fMRI
    """

    def __init__(
        self,
        n_voxels: int = 15724,
        latent_dim: int = 256,
        hidden_dim: int = 2048,
        num_res_blocks: int = 4,
        dropout: float = 0.1,
        free_bits: float = 0.5,
        lambda_pcc: float = 1.0,
        pcc_warmstart_epochs: int = 0,
        **kwargs,  # absorb unused args
    ):
        super().__init__()
        self.n_voxels = n_voxels
        self.latent_dim = latent_dim
        self.free_bits = free_bits
        self.lambda_pcc = lambda_pcc
        self.pcc_warmstart_epochs = pcc_warmstart_epochs
        self._current_epoch = 0

        # Encoder: V → hidden → ... → (μ, logvar)
        self.enc_in = nn.Sequential(
            nn.Linear(n_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.enc_blocks = nn.Sequential(*[
            ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: Z → hidden → ... → V
        self.dec_in = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.dec_blocks = nn.Sequential(*[
            ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        self.dec_out = nn.Linear(hidden_dim, n_voxels)

        # Learnable output scale and bias (per-voxel)
        self.output_scale = nn.Parameter(torch.ones(1, n_voxels))
        self.output_bias = nn.Parameter(torch.zeros(1, n_voxels))

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def encode(self, fmri):
        """fmri: (B, V) → z, mu, logvar: (B, Z)"""
        h = self.enc_in(fmri)
        h = self.enc_blocks(h)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h).clamp(-10.0, 10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        """z: (B, Z) → recon: (B, V)"""
        h = self.dec_in(z)
        h = self.dec_blocks(h)
        out = self.dec_out(h)
        return out * self.output_scale + self.output_bias

    def forward(self, fmri):
        z, mu, logvar = self.encode(fmri)
        recon = self.decode(z)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def pcc_loss(recon, target):
        """Per-sample spatial PCC loss: 1 - mean(PCC across voxels)."""
        r = recon - recon.mean(dim=1, keepdim=True)
        t = target - target.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(r, t, dim=1).mean()
        return 1.0 - pcc

    def loss(self, fmri, target, beta=1.0):
        recon, mu, logvar = self.forward(fmri)
        l_recon = F.mse_loss(recon, target, reduction="mean")
        l_pcc = self.pcc_loss(recon, target)

        # KL with free-bits
        kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        kl = kl_per_dim.mean()
        if self.free_bits > 0:
            kl = torch.max(kl, torch.tensor(self.free_bits, device=kl.device))

        # PCC warmstart: train PCC only first, then add MSE + KL
        if self._current_epoch < self.pcc_warmstart_epochs:
            total = self.lambda_pcc * l_pcc
        else:
            total = l_recon + self.lambda_pcc * l_pcc + beta * kl

        return {
            "loss": total,
            "recon": l_recon,
            "spatial_pcc": 1.0 - l_pcc,
            "kl": kl,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent(self, fmri):
        """Encode to latent mean (no sampling): (B, V) → (B, Z)."""
        h = self.enc_blocks(self.enc_in(fmri))
        return self.enc_mu(h)

    @torch.no_grad()
    def reconstruct(self, fmri):
        return self.decode(self.get_latent(fmri))

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"fMRI_VAE_NSD(n_voxels={self.n_voxels}, latent_dim={self.latent_dim}, "
            f"lambda_pcc={self.lambda_pcc}, params={n:,})"
        )
