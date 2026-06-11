"""
source_encoder.py
=================
Image-conditioned source distribution for Source-Conditioned Flow Matching.

Replaces the fixed x₀ ~ N(0, I) in standard flow matching with a learned
per-image distribution x₀ ~ N(μ_θ(c), σ_θ(c)²), where c is the visual
conditioning (CLIP-pool + DINOv2 tokens).  The flow matching velocity objective
is unchanged — only the source x₀ changes.

Architecture (mean-pool variant, ~3.5M params):
    z_pool  (B, 1280)     ──► clip_proj  ──► h_clip  (B, D)
    C_dino  (B, T, D_in)  ──► dino_proj  ──► mean  ──► h_dino  (B, D)
                                              fuse([h_clip, h_dino]) ──► h (B, D)
                                              to_patches ──► (B, N_patches × 2)
                                              unpatchify ──► μ, log_var (B, 1, L)

Training signals (auxiliary, independent of the velocity loss):
    L_mu  = MSE(μ_θ(c), rep_mean(x₁))   — mean matches average fMRI response
    L_kl  = mean(exp(log_var) + μ² − 1 − log_var) × 0.5  — KL to N(0,1) prior

Both losses are optional and controlled by the caller (factflow_trainer.py).
The DiT velocity loss **never** flows gradients through the source encoder
(x₀ is always detached before passing to the flow path).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_utils import RMSNorm


class SourceEncoder(nn.Module):
    """Predict image-conditioned source distribution μ and log-var in voxel space.

    Args:
        clip_dim:      Dimension of z_pool (CLIP pooled). Default 1280 (SDXL-CLIP).
        dino_dim:      Dimension of each DINOv2 token. Default 1024 (ViT-g/14).
        hidden_dim:    Internal bottleneck width. Default 512 (matches DiT hidden_size).
        n_voxels:      Padded sequence length (pad_to). E.g. 15744.
        patch_size:    Same patch size as DiT1D (must divide n_voxels). Default 32.
        use_dino:      Whether to incorporate DINO tokens (False = CLIP-only ablation).
    """

    def __init__(
        self,
        clip_dim: int = 1280,
        dino_dim: int = 1024,
        hidden_dim: int = 512,
        n_voxels: int = 15744,
        patch_size: int = 32,
        use_dino: bool = True,
    ) -> None:
        super().__init__()
        self.use_dino = use_dino
        self.n_voxels = n_voxels
        self.patch_size = patch_size
        assert n_voxels % patch_size == 0, (
            f"n_voxels ({n_voxels}) must be divisible by patch_size ({patch_size})"
        )
        self.n_patches = n_voxels // patch_size

        # ── CLIP branch ──────────────────────────────────────────────────
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.SiLU(),
            RMSNorm(hidden_dim),
        )

        # ── DINO branch (mean-pool over tokens) ─────────────────────────
        fuse_in = hidden_dim
        if use_dino:
            self.dino_proj = nn.Sequential(
                nn.Linear(dino_dim, hidden_dim),
                nn.SiLU(),
                RMSNorm(hidden_dim),
            )
            fuse_in = hidden_dim * 2  # concat [h_clip, h_dino]

        # ── Fusion MLP ───────────────────────────────────────────────────
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
        )

        # ── Decode → patch-level μ and log_var ───────────────────────────
        # Output: 2 × n_patches scalars (first half = μ, second half = log_var)
        self.to_patches = nn.Linear(hidden_dim, self.n_patches * 2)

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Safe zero-init for output head → source starts as N(0, 1) at epoch 0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init output head: μ=0, log_var=0 at init → N(0,1) source (safe start)
        nn.init.zeros_(self.to_patches.weight)
        nn.init.zeros_(self.to_patches.bias)

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        z_pool: torch.Tensor,
        C_dino: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict μ and log_var in native voxel space.

        Args:
            z_pool: (B, clip_dim) — CLIP pooled embedding.
            C_dino: (B, T, dino_dim) — DINOv2 token sequence; ignored if
                    ``use_dino=False`` or ``None``.

        Returns:
            mu:      (B, 1, n_voxels) — predicted mean of source Gaussian.
            log_var: (B, 1, n_voxels) — predicted log-variance (clamped to [-4, 4]).
        """
        # Encode CLIP
        h_clip = self.clip_proj(z_pool)            # (B, D)

        # Encode DINO (mean-pool over token axis)
        if self.use_dino and C_dino is not None:
            h_dino = self.dino_proj(C_dino)         # (B, T, D)
            h_dino = h_dino.mean(dim=1)             # (B, D)
            h = torch.cat([h_clip, h_dino], dim=-1) # (B, 2D)
        else:
            h = h_clip                              # (B, D)

        # Fuse
        h = self.fuse(h)                            # (B, D)

        # Project to patch-level (μ, log_var)
        out = self.to_patches(h)                    # (B, N_patches * 2)
        mu_p, lv_p = out.chunk(2, dim=-1)           # (B, N_patches) each

        # Clamp log_var to avoid numerical instability (σ ∈ [e^-2, e^2] ≈ [0.14, 7.4])
        lv_p = lv_p.clamp(-4.0, 4.0)

        # Unpatchify: broadcast each patch scalar to patch_size voxels
        # (B, N) → (B, N * P) via repeat_interleave → (B, 1, L)
        mu = mu_p.repeat_interleave(self.patch_size, dim=1).unsqueeze(1)      # (B,1,L)
        log_var = lv_p.repeat_interleave(self.patch_size, dim=1).unsqueeze(1) # (B,1,L)
        return mu, log_var

    # ─────────────────────────────────────────────────────────────────────
    # Sampling
    # ─────────────────────────────────────────────────────────────────────

    def sample(
        self,
        z_pool: torch.Tensor,
        C_dino: torch.Tensor | None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample x₀ from the conditioned source distribution.

        x₀ = μ_θ(c) + noise_scale · σ_θ(c) · ε,   ε ~ N(0, I)

        Special cases:
            noise_scale = 0.0  → x₀ = μ_θ (fully deterministic ceiling)
            noise_scale = 0.01 → x₀ ≈ μ_θ (near-deterministic eval)
            noise_scale = 1.0  → full stochastic sampling

        Returns:
            x₀: (B, 1, n_voxels) — sampled source point, detached from graph.
        """
        mu, log_var = self.forward(z_pool, C_dino)
        if noise_scale == 0.0:
            return mu.detach()
        std = torch.exp(0.5 * log_var).clamp(max=3.0)  # σ ∈ (0, e^1.5 ≈ 4.5]
        eps = torch.randn_like(mu)
        return (mu + noise_scale * std * eps).detach()

    # ─────────────────────────────────────────────────────────────────────
    # Auxiliary losses (called from trainer, not from velocity path)
    # ─────────────────────────────────────────────────────────────────────

    def loss_mu(
        self,
        z_pool: torch.Tensor,
        C_dino: torch.Tensor | None,
        x1: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted μ and the observed fMRI (rep-averaged).

        Teaches the source encoder to predict the mean neural response.
        x₀ starting near the data mean → shorter ODE trajectories at inference.

        Args:
            x1:       (B, 1, L) — rep-averaged fMRI target (training x₁).
            pad_mask: (L,) — boolean True for real voxels (excludes padding).

        Returns:
            Scalar loss.
        """
        mu, _ = self.forward(z_pool, C_dino)
        # Apply pad mask: only real voxels contribute
        mu_real = mu[:, :, pad_mask]     # (B, 1, V)
        x1_real = x1[:, :, pad_mask]     # (B, 1, V)
        return F.mse_loss(mu_real, x1_real)

    def loss_kl(
        self,
        z_pool: torch.Tensor,
        C_dino: torch.Tensor | None,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence KL[N(μ, σ²) || N(0, 1)] averaged over real voxels.

        Prevents σ from collapsing to 0 and keeps the source distribution
        close to a standard Gaussian (useful regularization for out-of-distribution
        images at inference).

        Returns:
            Scalar loss: 0.5 * mean(exp(log_var) + μ² - 1 - log_var)
        """
        mu, log_var = self.forward(z_pool, C_dino)
        mu_real  = mu[:, :, pad_mask]       # (B, 1, V)
        lv_real  = log_var[:, :, pad_mask]  # (B, 1, V)
        kl = 0.5 * (lv_real.exp() + mu_real.pow(2) - 1.0 - lv_real)
        return kl.mean()
