"""
fMRI MoE VAE — Stage 1 with Mixture-of-Experts.

Based on V1 MLP VAE but replaces standard ResBlocks with MoE blocks.
Each MoE block has N expert MLPs + a router that selects top-k experts
per sample. Different fMRI patterns (visual, motor, etc.) route to
different experts → better specialization with same compute.

Architecture:
    Encoder: Linear(V→H) → MoEResBlock×N → μ,logvar(latent)
    Decoder: Linear(latent→H) → MoEResBlock×N → Linear(H→V)

Key: load balancing loss ensures all experts are used.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── MoE Building Blocks ─────────────────────────────────────────────────────


class ExpertMLP(nn.Module):
    """Single expert: Linear → GELU → Linear."""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with top-k routing.

    Router: Linear(dim → n_experts) → top-k softmax → weighted sum.
    Load balancing loss encourages uniform expert usage.
    """

    def __init__(
        self, dim: int, n_experts: int = 4, top_k: int = 2,
        expansion: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            ExpertMLP(dim, expansion, dropout)
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(dim, n_experts, bias=False)

        # For logging
        self.last_load_balance_loss = 0.0

    def forward(self, x):
        """
        Args:
            x: (B, D)

        Returns:
            output: (B, D) — weighted sum of top-k expert outputs
        """
        B, D = x.shape

        # Router logits → top-k selection
        logits = self.router(x)                    # (B, n_experts)
        top_k_logits, top_k_indices = logits.topk(
            self.top_k, dim=-1)                    # (B, k), (B, k)
        top_k_weights = F.softmax(
            top_k_logits, dim=-1)                  # (B, k) normalized

        # Compute load balancing loss (differentiable)
        # fraction of tokens routed to each expert
        router_probs = F.softmax(logits, dim=-1)   # (B, n_experts)
        # Average routing probability per expert
        avg_prob = router_probs.mean(dim=0)         # (n_experts,)
        # Fraction of tokens where expert is in top-k
        mask = torch.zeros_like(logits)
        mask.scatter_(1, top_k_indices, 1.0)
        avg_mask = mask.mean(dim=0)                 # (n_experts,)
        # Loss = n_experts * Σ(avg_prob * avg_mask) — encourages uniform
        self.last_load_balance_loss = (
            self.n_experts * (avg_prob * avg_mask).sum()
        )

        # Compute expert outputs (only for selected experts)
        output = torch.zeros_like(x)
        for k_idx in range(self.top_k):
            expert_idx = top_k_indices[:, k_idx]    # (B,)
            weight = top_k_weights[:, k_idx:k_idx+1]  # (B, 1)

            for e in range(self.n_experts):
                mask_e = (expert_idx == e)
                if mask_e.any():
                    expert_input = x[mask_e]
                    expert_out = self.experts[e](expert_input)
                    output[mask_e] += weight[mask_e] * expert_out

        return output


class MoEResBlock(nn.Module):
    """Residual block with MoE: LayerNorm → MoE → skip connection."""

    def __init__(
        self, dim: int, n_experts: int = 4, top_k: int = 2,
        expansion: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.moe = MoELayer(dim, n_experts, top_k, expansion, dropout)

    def forward(self, x):
        return x + self.moe(self.norm(x))

    @property
    def load_balance_loss(self):
        return self.moe.last_load_balance_loss


class MLPResBlock(nn.Module):
    """Standard residual block (for compatibility / mixing)."""

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

    def forward(self, x):
        return x + self.net(x)


# ─── Model ────────────────────────────────────────────────────────────────────


@dataclass
class FmriMoEVAEConfig:
    """Configuration for FmriMoEVAE."""
    n_voxels: int = 15724
    hidden_dim: int = 2048
    latent_dim: int = 768
    n_res_blocks: int = 4           # number of MoE blocks
    n_experts: int = 4              # experts per MoE layer
    top_k: int = 2                  # active experts per sample
    expansion: int = 2              # expert MLP expansion ratio
    dropout: float = 0.1
    load_balance_weight: float = 0.01  # weight for load balancing loss

    # V4 Asymmetric Decoder
    n_res_blocks_decoder: Optional[int] = None
    decoder_expansion_dims: List[int] = field(default_factory=list)


class FmriMoEVAE(nn.Module):
    """
    MoE-based VAE for fMRI data.

    Same structure as V1 MLP VAE but with MoE blocks.
    Different fMRI patterns route to different expert sub-networks,
    enabling better specialization while keeping compute per-sample moderate.

    Architecture:
        Encoder: Linear(V→H) → LN → GELU → MoEResBlock×N → Linear→μ, Linear→logvar
        Decoder: Linear(latent→H) → LN → GELU → MoEResBlock×N → Linear(H→V)
    """

    def __init__(self, config: Optional[FmriMoEVAEConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = FmriMoEVAEConfig(**kwargs)
        self.config = config

        n_voxels = config.n_voxels
        hidden = config.hidden_dim
        latent = config.latent_dim
        dropout = config.dropout

        # ── Encoder ──
        encoder_layers = [
            nn.Linear(n_voxels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
        for _ in range(config.n_res_blocks):
            encoder_layers.append(MoEResBlock(
                hidden, config.n_experts, config.top_k,
                config.expansion, dropout,
            ))

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

        # ── Decoder ──
        decoder_layers = [
            nn.Linear(latent, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        ]
        
        n_dec_blocks = config.n_res_blocks_decoder if config.n_res_blocks_decoder is not None else config.n_res_blocks
        
        for _ in range(n_dec_blocks):
            decoder_layers.append(MoEResBlock(
                hidden, config.n_experts, config.top_k,
                config.expansion, dropout,
            ))
            
        prev_dim = hidden
        for exp_dim in config.decoder_expansion_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, exp_dim),
                nn.LayerNorm(exp_dim),
                nn.GELU(),
                MoEResBlock(
                    exp_dim, config.n_experts, config.top_k,
                    config.expansion, dropout,
                )
            ])
            prev_dim = exp_dim
            
        decoder_layers.append(nn.Linear(prev_dim, n_voxels))

        self.decoder = nn.Sequential(*decoder_layers)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_load_balance_loss(self):
        """Sum load balancing loss from all MoE blocks."""
        total = 0.0
        for m in self.modules():
            if isinstance(m, MoEResBlock):
                total = total + m.load_balance_loss
        return total

    def encode(
        self, x: torch.Tensor, sample_posterior: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
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
        """VAE loss + load balancing loss."""
        # MSE
        mse = F.mse_loss(x_recon, x)

        # KL
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # PCC (sample-wise)
        x_zm = x - x.mean(dim=1, keepdim=True)
        r_zm = x_recon - x_recon.mean(dim=1, keepdim=True)
        pcc = F.cosine_similarity(x_zm, r_zm, dim=1).mean()
        pcc_loss = 1.0 - pcc

        # Load balancing
        lb_loss = self._get_load_balance_loss()

        # Total
        loss = (mse + beta * kl + lambda_pcc * pcc_loss +
                self.config.load_balance_weight * lb_loss)

        return {
            "loss": loss,
            "mse": mse,
            "kl": kl,
            "pcc_loss": pcc_loss,
            "pcc": pcc,
            "lb_loss": lb_loss if isinstance(lb_loss, float) else lb_loss.item(),
        }

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_mb": total * 4 / 1024 / 1024,
        }

    def expert_usage_stats(self) -> dict:
        """Get routing statistics from last forward pass."""
        stats = {}
        for i, m in enumerate(self.modules()):
            if isinstance(m, MoELayer):
                probs = F.softmax(
                    m.router.weight.data.mean(dim=1), dim=0)
                stats[f"moe_{i}"] = probs.cpu().tolist()
        return stats


# ─── Factory ──────────────────────────────────────────────────────────────────

def create_fmri_moe_vae(**kwargs) -> FmriMoEVAE:
    config = FmriMoEVAEConfig(**kwargs)
    return FmriMoEVAE(config)
