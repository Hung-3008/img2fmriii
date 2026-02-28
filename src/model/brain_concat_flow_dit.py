"""
BrainConcatFlowDiT — Flow Matching with Concatenation Conditioning.

Architecture:
    - Flow Matching: learns to map noise z_0 ~ N(0, I) → fMRI latent z_1
      via ODE: dz/dt = v(z_t, t | c), where v = z_1 - z_0.
    - Conditioning mechanism: CONCAT (not cross-attention).
      The DINOv2 context tokens are prepended to the z_t tokens,
      and the combined sequence is processed via standard self-attention.
    - Timestep t is injected via AdaLN on every TransformerBlock.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Utilities ────────────────────────────────────────────────────────────────

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal timestep embeddings. t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    # Scale t to [0, 1000] for sinusoidal embedding
    args = t[:, None].float() * 1000 * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift, scale):
    """AdaLN modulation. x: (B, N, D), shift/scale: (B, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BrainConcatFlowDiTConfig:
    # fMRI latent
    latent_dim: int = 768           # total latent size
    n_latent_tokens: int = 12       # 768 / 12 = 64 per token

    # Context (DINOv2)
    context_dim: int = 768          # DINOv2 feature dim per layer
    n_dino_layers: int = 4          # Number of DINOv2 layers used
    n_context_tokens: int = 257     # 1 CLS + 256 patches (ViT-B/14)

    # Transformer backbone
    hidden_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1


# ─── AdaLN Transformer Block ──────────────────────────────────────────────────

class ConcatFlowBlock(nn.Module):
    """
    Standard Self-Attention Transformer Block with AdaLN for timestep conditioning.

    The input sequence is [context_tokens | z_t_tokens], and attention is
    computed over the entire joint sequence. The AdaLN modulation is conditioned
    on the timestep embedding only (no global class conditioning).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, drop_path_rate: float = 0.0):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.dim = dim

        # Self-Attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation from timestep embedding.
        # 6 = 2 (shift+scale for norm1) + 2 (shift+scale for norm2) + 2 (gate for attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # Zero-init so block starts as identity at initialization
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, t_cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N_total, D)  — full sequence [context | z_t]
        t_cond: (B, D)      — timestep embedding
        """
        mods = self.adaLN_modulation(t_cond)  # (B, 6*D)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)

        # Self-Attention sub-layer
        h = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(h, h, h)
        x = x + drop_path(gate1.unsqueeze(1) * attn_out, self.drop_path_rate, self.training)

        # FFN sub-layer
        h = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(h)
        x = x + drop_path(gate2.unsqueeze(1) * mlp_out, self.drop_path_rate, self.training)

        return x


# ─── Main Model ───────────────────────────────────────────────────────────────

class BrainConcatFlowDiT(nn.Module):
    """
    Concat-conditioned Flow Matching DiT for fMRI latent generation.

    Flow Matching formulation:
        - Samples: z_0 ~ N(0, I),  z_1 = VAE-encoded fMRI latent
        - Interpolation:  z_t = (1 - t) * z_0 + t * z_1
        - Target velocity: v = z_1 - z_0
        - Model predicts: v_hat = f(z_t, t ; DINOv2)
        - Loss: MSE(v_hat, v)

    At inference, integrate ODE:
        z_{t+dt} = z_t + dt * v_hat(z_t, t)
    from t=0 → t=1 using Euler solver.
    """

    def __init__(self, config: BrainConcatFlowDiTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.context_dim

        assert config.latent_dim % config.n_latent_tokens == 0, \
            f"latent_dim ({config.latent_dim}) must be divisible by n_latent_tokens ({config.n_latent_tokens})"
        self.token_dim = config.latent_dim // config.n_latent_tokens

        # ─── 1. Timestep Embedding ────────────────────────────────────────────
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D),
        )

        # ─── 2. DINOv2 Context Processing ────────────────────────────────────
        # Learnable layer mixing weights (one weight per DINOv2 layer)
        self.layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))

        # Project context tokens from context_dim → hidden_dim
        self.context_proj = nn.Linear(C, D)
        self.context_pos_embed = nn.Parameter(
            torch.randn(1, config.n_context_tokens, D) * 0.02
        )

        # ─── 3. fMRI Latent Tokenization ─────────────────────────────────────
        # Project each token from token_dim → hidden_dim
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.latent_pos_embed = nn.Parameter(
            torch.randn(1, config.n_latent_tokens, D) * 0.02
        )

        # ─── 4. Transformer Backbone ──────────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            ConcatFlowBlock(
                dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path_rate=dpr[i],
            ) for i in range(config.depth)
        ])

        # ─── 5. Output Head ───────────────────────────────────────────────────
        # Applied only to the fMRI tokens (last n_latent_tokens of the sequence)
        self.final_norm = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.final_adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D),
        )
        nn.init.zeros_(self.final_adaLN_mod[1].weight)
        nn.init.zeros_(self.final_adaLN_mod[1].bias)

        # Project hidden tokens → velocity predictions (token_dim each)
        self.output_proj = nn.Linear(D, self.token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _process_context(self, dino_multilayer: torch.Tensor) -> torch.Tensor:
        """
        Compute the weighted sum of multi-layer DINOv2 features.

        dino_multilayer: (B, L, N_ctx, C)
        Returns: (B, N_ctx, C)
        """
        w = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)  # (1, L, 1, 1)
        return (dino_multilayer * w).sum(dim=1)  # (B, N_ctx, C)

    def forward(
        self,
        t: torch.Tensor,
        z_t: torch.Tensor,
        dino_multilayer: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(z_t, t | DINOv2).

        Args:
            t:               (B,)            — timestep in [0, 1]
            z_t:             (B, latent_dim) — noisy fMRI latent
            dino_multilayer: (B, L, N_ctx, C) — multi-layer DINOv2 features

        Returns:
            v_hat: (B, latent_dim) — predicted velocity
        """
        B = z_t.shape[0]
        N = self.config.n_latent_tokens

        # 1. Timestep embedding
        t_emb = timestep_embedding(t, self.config.hidden_dim)  # (B, D)
        t_cond = self.t_embedder(t_emb)                         # (B, D)

        # 2. DINOv2 context tokens: (B, N_ctx, D)
        ctx = self._process_context(dino_multilayer)            # (B, N_ctx, C)
        ctx = self.context_proj(ctx) + self.context_pos_embed   # (B, N_ctx, D)

        # 3. fMRI latent tokens: (B, N, D)
        z_seq = z_t.view(B, N, self.token_dim)                  # (B, N, token_dim)
        lat = self.latent_proj(z_seq) + self.latent_pos_embed   # (B, N, D)

        # 4. Concatenate: [context | latent]
        x = torch.cat([ctx, lat], dim=1)                        # (B, N_ctx + N, D)

        # 5. Transformer blocks (all attend over the full joint sequence)
        for block in self.blocks:
            x = block(x, t_cond)

        # 6. Extract fMRI token outputs (last N tokens)
        x_lat = x[:, -N:, :]                                    # (B, N, D)

        # Final AdaLN modulation
        mod = self.final_adaLN_mod(t_cond)                      # (B, 2*D)
        shift, scale = mod.chunk(2, dim=-1)
        x_lat = modulate(self.final_norm(x_lat), shift, scale)

        # Project to velocity
        v_hat = self.output_proj(x_lat)                         # (B, N, token_dim)
        return v_hat.view(B, -1)                                 # (B, latent_dim)

    @torch.no_grad()
    def sample(
        self,
        dino_multilayer: torch.Tensor,
        n_steps: int = 20,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Generate fMRI latent from noise via Euler ODE integration.

        Args:
            dino_multilayer: (B, L, N_ctx, C)
            n_steps: number of Euler integration steps

        Returns:
            z_1: (B, latent_dim) — generated fMRI latent
        """
        B = dino_multilayer.shape[0]
        if device is None:
            device = dino_multilayer.device

        # Start from pure noise
        z = torch.randn(B, self.config.latent_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = i / n_steps
            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            v = self.forward(t, z, dino_multilayer)
            z = z + dt * v

        return z

    def param_count(self) -> dict:
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_p,
            'total_M': total_p / 1e6,
        }
