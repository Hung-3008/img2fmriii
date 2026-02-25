"""
BrainFlowDiT — Diffusion Transformer for Brain Latent Flow Matching.

References Internet Best Practices for Conditional Flow Matching (CFM):
1. DiT (Diffusion Transformer) backbone with AdaLN-Zero for stable scaling.
2. Tokenize the flat 768-D fMRI latent into a sequence (e.g., 12 tokens of 64 dims)
   to enable self-attention and cross-attention.
3. Learnable layer mixing for multi-layer DINOv2 features instead of hardcoded slices.
4. Dual Conditioning:
   - Global Conditioning (AdaLN): t_emb + DINOv2 CLS token
   - Dense Conditioning (Cross-Attention): DINOv2 spatial patch tokens.
5. Stochastic Depth & LayerNorm for stable training.

This replaces the fragmented/overcomplex ResidualFlowSiT and SimpleAdaLNFlow.
"""

from dataclasses import dataclass
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─── Configuration ────────────────────────────────────────────────────────────


@dataclass
class BrainFlowDiTConfig:
    # Latent configuration
    latent_dim: int = 768
    n_latent_tokens: int = 12       # 768 = 12 * 64
    
    # Context (DINOv2) configuration
    context_dim: int = 768
    n_dino_layers: int = 4
    
    # Transformer backbone
    hidden_dim: int = 512
    depth: int = 6                  # Best practice for ~8.5K samples: avoid too deep
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    
    # Informed Prior Regressor
    use_regressor: bool = True
    regressor_depth: int = 2


# ─── DiT Block ────────────────────────────────────────────────────────────────


class BrainDiTBlock(nn.Module):
    """DiT block with Self-Attention, Cross-Attention, and AdaLN-Zero."""
    def __init__(self, hidden_dim, context_dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_path_rate = drop_path
        
        # Self-Attention
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-Attention to Dense Context
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # FFN
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # AdaLN-Zero params (for norm1, norm2, norm3)
        # 6 params for shift/scale, + 3 for gate values (alpha)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim + 3)
        )
        # Initialize AdaLN-Zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, global_cond, dense_cond):
        """
        x: (B, N, D)
        global_cond: (B, D) from t_emb + CLS
        dense_cond: (B, N_ctx, C) mapped to D
        """
        mod_params = self.adaLN_modulation(global_cond)
        
        # Split params: 6 scale/shift, 3 gates
        splits = mod_params.split(self.hidden_dim, dim=-1)
        shift1, scale1, shift2, scale2, shift3, scale3 = splits[:6]
        gate1, gate2, gate3 = mod_params[..., -3:].split(1, dim=-1)
        
        gate1 = gate1.unsqueeze(1)
        gate2 = gate2.unsqueeze(1)
        gate3 = gate3.unsqueeze(1)

        # 1. Self-Attention
        h1 = modulate(self.norm1(x), shift1, scale1)
        # MHA returns (output, weights)
        attn_out, _ = self.attn(h1, h1, h1)
        x = x + drop_path(attn_out * gate1, self.drop_path_rate, self.training)
        
        # 2. Cross-Attention
        h2 = modulate(self.norm2(x), shift2, scale2)
        cross_out, _ = self.cross_attn(h2, dense_cond, dense_cond)
        x = x + drop_path(cross_out * gate2, self.drop_path_rate, self.training)
        
        # 3. FFN
        h3 = modulate(self.norm3(x), shift3, scale3)
        mlp_out = self.mlp(h3)
        x = x + drop_path(mlp_out * gate3, self.drop_path_rate, self.training)
        
        return x


# ─── Main Model ───────────────────────────────────────────────────────────────


class BrainFlowDiT(nn.Module):
    """
    Brain Flow Diffusion Transformer.
    Integrates completely redesigned Flow Matching for fMRI.
    """
    def __init__(self, config: BrainFlowDiTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.context_dim
        
        # ─── 1. DINOv2 Layer Mixing ───
        # Learnable sum of the specified DINOv2 layers
        self.layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))
        
        # ─── 2. Conditional Encoders ───
        # Global Conditioning: t_emb + CLS
        self.t_embedder = nn.Sequential(
            nn.Linear(D, D),
            nn.SiLU(),
            nn.Linear(D, D)
        )
        self.cls_embedder = nn.Sequential(
            nn.Linear(C, D),
            nn.GELU(approximate='tanh'),
            nn.Linear(D, D)
        )
        # Dense Conditioning: Spatial Patches
        self.patch_embedder = nn.Sequential(
            nn.Linear(C, D),
            nn.GELU(approximate='tanh'),
            nn.Linear(D, D)
        )
        
        # ─── 3. Latent Tokenization ───
        assert config.latent_dim % config.n_latent_tokens == 0
        self.token_dim = config.latent_dim // config.n_latent_tokens
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.pos_embed = nn.Parameter(torch.randn(1, config.n_latent_tokens, D) * 0.02)
        
        # ─── 4. Backbone DiT Blocks ───
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            BrainDiTBlock(
                hidden_dim=D,
                context_dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path=dpr[i]
            ) for i in range(config.depth)
        ])
        
        # ─── 5. Final Output Head ───
        self.final_layer_norm = nn.LayerNorm(D, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D, 2 * D)
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.output_proj = nn.Linear(D, self.token_dim)
        
        # ─── 6. Lightweight Informed Regressor (Optional) ───
        if config.use_regressor:
            # Multi-layer MLP
            reg_layers = []
            in_dim = C
            for _ in range(config.regressor_depth):
                reg_layers.extend([
                    nn.Linear(in_dim, D),
                    nn.GELU(approximate='tanh'),
                    nn.Dropout(config.dropout)
                ])
                in_dim = D
            reg_layers.append(nn.Linear(D, config.latent_dim))
            
            self.regressor = nn.Sequential(*reg_layers)
            
        self._init_weights()

    def _init_weights(self):
        # Output proj zero init
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        if self.config.use_regressor:
            nn.init.zeros_(self.regressor[-1].weight)
            nn.init.zeros_(self.regressor[-1].bias)

    def _process_context(self, dino_multilayer):
        """
        Process the multi-layer DINOv2 features.
        dino_multilayer: (B, L, 257, C)
        Returns:
            cls_token: (B, C) - CLS token
            spatial_patches: (B, 256, C) - Spatial patches
        """
        # Softmax over layer weights ensures stability
        w = F.softmax(self.layer_weights, dim=0) # (L,)
        w = w.view(1, -1, 1, 1)                  # (1, L, 1, 1)
        
        # Weighted sum of layers
        dino_mixed = (dino_multilayer * w).sum(dim=1) # (B, 257, C)
        
        cls_token = dino_mixed[:, 0, :]               # (B, C)
        spatial_patches = dino_mixed[:, 1:, :]        # (B, 256, C)
        
        return cls_token, spatial_patches

    def forward_regression(self, dino_multilayer):
        """
        Predict the conditional mean z_bar. Lightweight approach.
        Returns: z_bar (B, latent_dim)
        """
        assert self.config.use_regressor, "Regressor not enabled in config."
        cls_token, _ = self._process_context(dino_multilayer)
        return self.regressor(cls_token)

    def forward_flow(self, t, z_t, dino_multilayer):
        """
        Predict the velocity v(z_t, t | DINOv2).
        t: (B,)
        z_t: (B, latent_dim)
        dino_multilayer: (B, L, 257, context_dim)
        """
        B = z_t.shape[0]
        
        # 1. Context embedding
        cls_token, spatial_patches = self._process_context(dino_multilayer)
        
        t_emb = timestep_embedding(t * 1000, self.config.hidden_dim)
        global_cond = self.t_embedder(t_emb) + self.cls_embedder(cls_token) # (B, D)
        dense_cond = self.patch_embedder(spatial_patches)                   # (B, 256, D)
        
        # 2. Tokenize Latent
        # Reshape: (B, 768) -> (B, 12, 64)
        N = self.config.n_latent_tokens
        T_dim = self.token_dim
        z_seq = z_t.view(B, N, T_dim)
        
        x = self.latent_proj(z_seq) + self.pos_embed # (B, 12, D)
        
        # 3. Backbone
        for block in self.blocks:
            x = block(x, global_cond, dense_cond)
            
        # 4. Output Head
        mod_params = self.final_adaLN(global_cond)
        shift, scale = mod_params.chunk(2, dim=-1)
        x = modulate(self.final_layer_norm(x), shift, scale)
        
        x = self.output_proj(x) # (B, 12, 64)
        
        # Flatten back to (B, latent_dim)
        return x.view(B, -1)

    def forward_flow_with_cfg(self, t, z_t, dino_multilayer, cfg_scale=1.0):
        if cfg_scale == 1.0:
            return self.forward_flow(t, z_t, dino_multilayer)
        v_cond = self.forward_flow(t, z_t, dino_multilayer)
        v_uncond = self.forward_flow(t, z_t, torch.zeros_like(dino_multilayer))
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    def get_layer_mixing_weights(self):
        """Returns the layer weights dict for consistent logging format."""
        with torch.no_grad():
            w = F.softmax(self.layer_weights, dim=0).unsqueeze(0).cpu() # (1, L)
        # We simulate block weights format for compatibility: shape (blocks, layers)
        return {
            'reg': w.repeat(self.config.regressor_depth, 1),
            'flow': w.repeat(self.config.depth, 1)
        }

    def param_count(self):
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        reg_p = sum(p.numel() for p in self.regressor.parameters() if p.requires_grad) if self.config.use_regressor else 0
        return {
            'reg_M': reg_p / 1e6,
            'flow_M': (total_p - reg_p) / 1e6,
            'shared_M': 0.0,
            'total_M': total_p / 1e6
        }
