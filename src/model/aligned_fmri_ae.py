"""
Aligned fMRI Autoencoder — Stage 1 for MindSimulator-inspired pipeline.

Architecture (inspired by MindSimulator §3.2):
    Encoder: fMRI (15724) → MLP bottleneck (256) → factored projection → [257, repr_dim]
    Decoder: [257, repr_dim] → factored projection → MLP → fMRI (15724)

The representation [B, 257, repr_dim] is aligned with projected CLIP features
[B, 257, repr_dim] via SoftCLIP contrastive loss during training.

This alignment ensures:
    1. fMRI representation lives near CLIP representation in shared space
    2. Flow matching in Stage 2 only needs to bridge a small gap
    3. Information from all 257 CLIP tokens (CLS + patches) is leveraged
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Building Blocks ──────────────────────────────────────────────────────────


class ResBlock(nn.Module):
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


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class AlignedAEConfig:
    """Configuration for AlignedFmriAutoencoder."""

    n_voxels: int = 15724        # Number of fMRI voxels
    bottleneck: int = 256        # Bottleneck dimension (like MindSimulator)
    n_tokens: int = 257          # Number of representation tokens (match CLIP: CLS + 256 patches)
    token_inner: int = 64        # Intermediate token dim (before expansion to repr_dim)
    repr_dim: int = 256          # Representation token dimension (aligned with CLIP projection)
    clip_dim: int = 1024         # Native CLIP feature dimension
    n_blocks: int = 4            # Number of residual blocks
    dropout: float = 0.1        # Dropout rate


# ─── Model ────────────────────────────────────────────────────────────────────


class AlignedFmriAutoencoder(nn.Module):
    """
    fMRI Autoencoder with CLIP-aligned representation space.

    Produces [B, 257, repr_dim] representations aligned with projected CLIP tokens
    via contrastive loss. The 257-token structure matches CLIP ViT-L/14 output.

    Encoder pathway:
        fMRI [B, 15724]
        → Linear → bottleneck [B, 256]
        → ResBlock × N
        → Linear → [B, 257*64] → reshape [B, 257, 64]
        → per-token Linear → [B, 257, repr_dim]

    Decoder pathway (reverse):
        [B, 257, repr_dim]
        → per-token Linear → [B, 257, 64]
        → flatten [B, 257*64]
        → Linear → bottleneck [B, 256]
        → ResBlock × N
        → Linear → fMRI [B, 15724]
    """

    def __init__(self, config: Optional[AlignedAEConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = AlignedAEConfig(**kwargs)
        self.config = config

        nv = config.n_voxels
        bn = config.bottleneck
        nt = config.n_tokens
        ti = config.token_inner
        rd = config.repr_dim
        nb = config.n_blocks
        dp = config.dropout

        # ── Encoder backbone ──
        enc_layers = [nn.Linear(nv, bn), nn.LayerNorm(bn), nn.GELU()]
        for _ in range(nb):
            enc_layers.append(ResBlock(bn, dp))
        self.encoder = nn.Sequential(*enc_layers)

        # ── Encoder token projection (factored: bottleneck → tokens × inner → tokens × repr) ──
        self.to_tokens = nn.Linear(bn, nt * ti)
        self.token_up = nn.Linear(ti, rd)  # per-token expansion

        # ── Decoder token projection (reverse) ──
        self.token_down = nn.Linear(rd, ti)  # per-token compression
        self.from_tokens = nn.Linear(nt * ti, bn)

        # ── Decoder backbone ──
        dec_layers = [nn.LayerNorm(bn), nn.GELU()]
        for _ in range(nb):
            dec_layers.append(ResBlock(bn, dp))
        dec_layers.append(nn.Linear(bn, nv))
        self.decoder = nn.Sequential(*dec_layers)

        # ── CLIP projection (for alignment loss, frozen during Stage 2) ──
        self.clip_proj = nn.Linear(config.clip_dim, rd)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, fmri: torch.Tensor) -> torch.Tensor:
        """
        Encode fMRI → aligned representation.

        Args:
            fmri: [B, n_voxels]

        Returns:
            repr: [B, 257, repr_dim]
        """
        h = self.encoder(fmri)  # [B, bottleneck]
        tokens = self.to_tokens(h)  # [B, n_tokens * token_inner]
        tokens = tokens.reshape(fmri.shape[0], self.config.n_tokens, self.config.token_inner)
        repr = self.token_up(tokens)  # [B, n_tokens, repr_dim]
        return repr

    def decode(self, repr: torch.Tensor) -> torch.Tensor:
        """
        Decode representation → fMRI.

        Args:
            repr: [B, 257, repr_dim]

        Returns:
            fmri_recon: [B, n_voxels]
        """
        tokens = self.token_down(repr)  # [B, n_tokens, token_inner]
        flat = tokens.reshape(repr.shape[0], -1)  # [B, n_tokens * token_inner]
        h = self.from_tokens(flat)  # [B, bottleneck]
        return self.decoder(h)  # [B, n_voxels]

    def project_clip(self, clip_features: torch.Tensor) -> torch.Tensor:
        """
        Project CLIP features to alignment space.

        Args:
            clip_features: [B, 257, clip_dim]

        Returns:
            clip_proj: [B, 257, repr_dim]
        """
        return self.clip_proj(clip_features)  # per-token linear

    def forward(self, fmri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward: encode → decode.

        Returns:
            fmri_recon: [B, n_voxels]
            repr: [B, 257, repr_dim]
        """
        repr = self.encode(fmri)
        fmri_recon = self.decode(repr)
        return fmri_recon, repr

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "total_M": total / 1e6}


# ─── Loss Functions ───────────────────────────────────────────────────────────


def softclip_loss(
    fmri_repr: torch.Tensor,
    clip_repr: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    SoftCLIP contrastive loss between fMRI and CLIP representations.

    Flattens token representations and computes bidirectional contrastive loss.
    Aligns fMRI sample i with CLIP sample i (positive pair).

    Args:
        fmri_repr: [B, N, D] fMRI representation tokens
        clip_repr: [B, N, D] projected CLIP tokens
        temperature: scaling factor for logits

    Returns:
        loss: scalar contrastive loss
    """
    B = fmri_repr.shape[0]

    # Flatten tokens → single vector per sample
    fmri_flat = fmri_repr.reshape(B, -1)  # [B, N*D]
    clip_flat = clip_repr.reshape(B, -1)  # [B, N*D]

    # L2 normalize
    fmri_norm = F.normalize(fmri_flat, dim=-1)
    clip_norm = F.normalize(clip_flat, dim=-1)

    # Similarity matrix [B, B]
    logits = fmri_norm @ clip_norm.T / temperature

    # Bidirectional cross-entropy
    labels = torch.arange(B, device=logits.device)
    loss_f2c = F.cross_entropy(logits, labels)
    loss_c2f = F.cross_entropy(logits.T, labels)

    return (loss_f2c + loss_c2f) / 2


def compute_ae_loss(
    fmri: torch.Tensor,
    fmri_recon: torch.Tensor,
    fmri_repr: torch.Tensor,
    clip_proj: torch.Tensor,
    lambda_align: float = 1.0,
    temperature: float = 0.07,
) -> dict:
    """
    Combined autoencoder loss: MSE reconstruction + SoftCLIP alignment.

    Args:
        fmri: [B, V] original fMRI
        fmri_recon: [B, V] reconstructed fMRI
        fmri_repr: [B, N, D] fMRI representation
        clip_proj: [B, N, D] projected CLIP features
        lambda_align: weight for alignment loss
        temperature: SoftCLIP temperature

    Returns:
        dict with 'loss', 'mse', 'align', 'cosine_sim'
    """
    mse = F.mse_loss(fmri_recon, fmri)
    align = softclip_loss(fmri_repr, clip_proj, temperature)

    # Monitoring: average cosine similarity between paired fmri/clip reprs
    with torch.no_grad():
        B = fmri_repr.shape[0]
        fmri_flat = F.normalize(fmri_repr.reshape(B, -1), dim=-1)
        clip_flat = F.normalize(clip_proj.reshape(B, -1), dim=-1)
        cos_sim = (fmri_flat * clip_flat).sum(dim=-1).mean().item()

    loss = mse + lambda_align * align

    return {
        "loss": loss,
        "mse": mse.item(),
        "align": align.item(),
        "cosine_sim": cos_sim,
    }
