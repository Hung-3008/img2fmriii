"""
BrainMaskedDiT — Masked Brain Modeling via DINOv2 Prefix Tokens.

Replaces the bottlenecked Informed Flow Matching approach by casting fMRI 
latent generation as a Masked Sequence Modeling problem (similar to MAE logic).
"""

from dataclasses import dataclass
import math

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

@dataclass
class BrainMaskedDiTConfig:
    latent_dim: int = 768
    n_latent_tokens: int = 12       # 768 = 12 * 64
    context_dim: int = 768
    n_dino_layers: int = 4
    hidden_dim: int = 768
    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1

class TransformerBlock(nn.Module):
    """Standard Transformer Block without temporal AdaLN conditioning."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.drop_path_rate = drop_path
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + drop_path(attn_out, self.drop_path_rate, self.training)
        
        h = self.norm2(x)
        mlp_out = self.mlp(h)
        x = x + drop_path(mlp_out, self.drop_path_rate, self.training)
        return x

class BrainMaskedDiT(nn.Module):
    def __init__(self, config: BrainMaskedDiTConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        C = config.context_dim
        
        # 1. DINOv2 Layer Mixing
        self.layer_weights = nn.Parameter(torch.ones(config.n_dino_layers))
        
        # 2. Context (DINOv2) Embedder
        self.context_proj = nn.Linear(C, D)
        self.context_pos_embed = nn.Parameter(torch.randn(1, 257, D) * 0.02)
        
        # 3. Target (fMRI) Embedder
        assert config.latent_dim % config.n_latent_tokens == 0
        self.token_dim = config.latent_dim // config.n_latent_tokens
        self.latent_proj = nn.Linear(self.token_dim, D)
        self.latent_pos_embed = nn.Parameter(torch.randn(1, config.n_latent_tokens, D) * 0.02)
        
        # 4. Mask Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
        
        # 5. Backbone
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=D,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                drop_path=dpr[i]
            ) for i in range(config.depth)
        ])
        
        # 6. Output Head
        self.norm = nn.LayerNorm(D)
        self.output_proj = nn.Linear(D, self.token_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _process_context(self, dino_multilayer):
        w = F.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        dino_mixed = (dino_multilayer * w).sum(dim=1)
        return dino_mixed # (B, 257, C)

    def forward(self, z_t, dino_multilayer, mask_ratio=0.75):
        """
        Forward pass for Masked Brain Modeling.
        z_t: (B, latent_dim) target fMRI sequence
        dino_multilayer: (B, 4, 257, 768)
        mask_ratio: float, ratio of fMRI tokens to replace with [MASK]
        Returns:
            predicted sequence matching original latent_dim, mask array
        """
        B = z_t.shape[0]
        N = self.config.n_latent_tokens
        
        # 1. Process Context
        context = self._process_context(dino_multilayer)
        context = self.context_proj(context) + self.context_pos_embed # (B, 257, D)
        
        # 2. Process Target
        z_seq = z_t.view(B, N, self.token_dim)
        target = self.latent_proj(z_seq) # (B, N, D)
        
        # 3. Masking
        if self.training and mask_ratio > 0:
            noise = torch.rand(B, N, device=z_t.device)
            mask = noise < mask_ratio
            mask_expanded = mask.unsqueeze(-1).expand_as(target)
            target = torch.where(mask_expanded, self.mask_token.expand_as(target), target)
        elif mask_ratio == 1.0:
            target = self.mask_token.expand(B, N, -1)
            mask = torch.ones(B, N, dtype=torch.bool, device=z_t.device)
        else:
            mask = torch.zeros(B, N, dtype=torch.bool, device=z_t.device)
            
        target = target + self.latent_pos_embed # (B, N, D)
        
        # 4. Sequence Fusion
        x = torch.cat([context, target], dim=1) # (B, 257+N, D)
        
        # 5. Model execution
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 6. Extract fMRI predictions
        target_out = x[:, -N:, :] # (B, N, D)
        target_pred = self.output_proj(target_out) # (B, N, token_dim)
        
        return target_pred.view(B, -1), mask

    def get_layer_mixing_weights(self):
        with torch.no_grad():
            w = F.softmax(self.layer_weights, dim=0).unsqueeze(0).cpu()
        return { 'mix': w }

    def param_count(self):
        total_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_M': total_p / 1e6
        }
