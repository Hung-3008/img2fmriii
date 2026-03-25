"""BrainFlow NSD — BrainDiT Architecture (Flow Matching + Regression for fMRI).

Inspired by RNACG, this uses a Joint-Attention multimodal Diffusion Transformer (mm-DiT)
with AdaLN-Zero to condition on visual tokens, plus an auxiliary MLP Regression head
directly extracting continuous fMRI components to maximize Pearson Correlation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver


# =============================================================================
# Building Blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t * 1000.0
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


# =============================================================================
# LinearFusion — Per-modality projection + concat
# =============================================================================

class MultiTokenFusion(nn.Module):
    """Projects multiple variable-length modalities into a unified sequence."""

    def __init__(self, in_dims: list[int], hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Project each modality to the same hidden dim
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            ) for d in in_dims
        ])

        # Global sequence projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, context_dict: dict) -> torch.Tensor:
        keys = ['dino', 'clip', 'qwen']
        projected = []
        for i, k in enumerate(keys):
            if k in context_dict:
                x = context_dict[k]  
                # If pool feature missing sequence dim: (B, D) -> (B, 1, D)
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                p = self.projs[i](x) 
                projected.append(p)
        
        # Concatenate along sequence dimension (dim=1)
        x = torch.cat(projected, dim=1) # (B, L_total, hidden)
        return self.output_proj(x)


class AdaLNZeroBlock(nn.Module):
    """DiT block: Self-Attention + FFN with AdaLN-Zero modulation."""
    
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Zero-initialize the final layers
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x is (B, L, D), c is time emb (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention
        x_mod = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # FFN MLP
        x_mod2 = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        ffn_out = self.ffn(x_mod2)
        x = x + gate_mlp.unsqueeze(1) * ffn_out
        
        return x


# =============================================================================
# BrainDiT_Net — Joint Attention DiT Network
# =============================================================================

class BrainDiT_Net(nn.Module):
    """
    Architecture:
      1. Project contexts into Sequence of Tokens
      2. Project x_t into 1 Token
      3. Concat X and Context tokens -> feed into DiT ADA-LN Blocks
      4. Extract X_token out of the blocks and map to fMRI dims.
    """

    def __init__(
        self,
        output_dim: int = 15724,
        hidden_dim: int = 768,
        modality_dims: list[int] = None,
        proj_dim: int = 256,
        n_blocks: int = 6,
        n_heads: int = 12,
        dropout: float = 0.1,
        modality_dropout: float = 0.3,
        **kwargs,  # Absorb unused params
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [4352] 

        # Context Fusion
        self.fusion_block = MultiTokenFusion(
            in_dims=self.modality_dims,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # X Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
        )

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # DiT Blocks
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_blocks)
        ])

        # Output Modulators
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Zero-init output
        nn.init.constant_(self.final_modulation[-1].weight, 0)
        nn.init.constant_(self.final_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.weight, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def encode_context(self, cond: dict) -> torch.Tensor:
        return self.fusion_block(cond)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: dict = None,
        pre_encoded_context: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if pre_encoded_context is not None:
            context_encoded = pre_encoded_context
        elif cond is not None:
            context_encoded = self.encode_context(cond)
        else:
            context_encoded = torch.zeros(
                x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype
            )

        # Embeddings
        t_emb = self.time_mlp(SinusoidalPosEmb(self.hidden_dim)(t))

        # Joint Attention Input
        x_token = self.input_proj(x).unsqueeze(1) # (B, 1, hidden)
        h = torch.cat([x_token, context_encoded], dim=1) # (B, 1+L, hidden)

        # Forward
        for block in self.blocks:
            h = block(h, t_emb)

        # Extract only the X token head
        x_out = h[:, 0, :] # (B, hidden)

        shift_c, scale_c = self.final_modulation(t_emb).chunk(2, dim=1)
        x_out = modulate(self.final_norm(x_out), shift_c, scale_c)
        return self.output_layer(x_out)


# =============================================================================
# BrainFlowNSD — Top-level Model
# =============================================================================

class BrainFlowNSD(nn.Module):
    """Flow matching with auxiliary regression for direct 15k outputs."""

    def __init__(
        self,
        output_dim: int = 15724,
        dit_net_params: dict = None,
        n_subjects: int = 1,
        reg_weight: float = 1.0,
        contrastive_weight: float = 0.1,
        contrastive_temp: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.reg_weight = reg_weight
        self.contrastive_weight = contrastive_weight
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / contrastive_temp)))

        dn_cfg = dict(dit_net_params or {})
        dn_cfg.setdefault("output_dim", output_dim)
        self.dit_net = BrainDiT_Net(**dn_cfg)
        
        hidden_dim = dn_cfg.get("hidden_dim", 768)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        # OT-CFM path
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Log
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BrainFlowNSD] BrainDiT: {params:,} params (RegWeight={reg_weight})")

    def forward(
        self,
        context: dict[str, torch.Tensor],
        target: torch.Tensor,
        subject_ids: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        # 1. Encode context
        context_encoded = self.dit_net.encode_context(context)
        
        # 2. Regression Branch (Crucial for PCC)
        context_pooled = context_encoded.mean(dim=1)
        fmri_reg = self.regression_head(context_pooled)
        reg_loss = F.mse_loss(fmri_reg, target)

        # 3. Contrastive Loss Branch (InfoNCE)
        if self.contrastive_weight > 0.0:
            fmri_norm = F.normalize(fmri_reg, dim=-1)
            target_norm = F.normalize(target, dim=-1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (fmri_norm @ target_norm.t())
            labels = torch.arange(logits.size(0), device=logits.device)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)
            cont_loss = (loss_i + loss_t) / 2
        else:
            cont_loss = torch.tensor(0.0, device=target.device)

        # 4. Flow matching Branch
        x_1 = target
        x_0 = torch.randn_like(x_1)
        
        with torch.no_grad():
            x_1_sq = x_1.pow(2).sum(dim=1, keepdim=True)
            x_0_sq = x_0.pow(2).sum(dim=1).unsqueeze(0)
            C = x_1_sq + x_0_sq - 2 * torch.mm(x_1, x_0.t())
            from scipy.optimize import linear_sum_assignment
            _, col_ind = linear_sum_assignment(C.cpu().numpy())
            x_0 = x_0[col_ind]

        t = torch.rand(x_1.shape[0], device=x_1.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        v_pred = self.dit_net(
            x=sample_info.x_t,
            t=sample_info.t,
            pre_encoded_context=context_encoded,
        )
        flow_loss = F.mse_loss(v_pred, sample_info.dx_t)
        
        total_loss = flow_loss + self.reg_weight * reg_loss + self.contrastive_weight * cont_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "reg_loss": reg_loss,
            "cont_loss": cont_loss,
        }

    @torch.no_grad()
    def predict_regression(self, context: dict) -> torch.Tensor:
        """Directly predict conditional mean fMRI for Val PCC testing."""
        self.eval()
        context_encoded = self.dit_net.encode_context(context)
        context_pooled = context_encoded.mean(dim=1)
        return self.regression_head(context_pooled)

    @torch.no_grad()
    def synthesise(
        self,
        context: dict,
        n_timesteps: int = 50,
        solver_method: str = "euler",
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        B = context["dino"].shape[0] if isinstance(context, dict) else 1

        x_init = torch.randn(B, self.output_dim, device=device, dtype=dtype)
        context_encoded = self.dit_net.encode_context(context)

        if cfg_scale > 0:
            uncond_context = {k: torch.zeros_like(v) for k, v in context.items()}
            uncond_encoded = self.dit_net.encode_context(uncond_context)

            class CFGWrapper(nn.Module):
                def __init__(self, dit_net, context_encoded, uncond_encoded, cfg_scale):
                    super().__init__()
                    self.dit_net = dit_net
                    self.context_encoded = context_encoded
                    self.uncond_encoded = uncond_encoded
                    self.cfg_scale = cfg_scale

                def forward(self, x, t, **kwargs):
                    B = x.shape[0]
                    t_batch = t.expand(B) if t.dim() == 0 else t
                    v_cond = self.dit_net(x=x, t=t_batch, pre_encoded_context=self.context_encoded)
                    v_uncond = self.dit_net(x=x, t=t_batch, pre_encoded_context=self.uncond_encoded)
                    return v_uncond + self.cfg_scale * (v_cond - v_uncond)

            cfg_model = CFGWrapper(self.dit_net, context_encoded, uncond_encoded, cfg_scale)
            solver = ODESolver(velocity_model=cfg_model)
        else:
            solver = ODESolver(velocity_model=self.dit_net)

        T = torch.linspace(0, 1, n_timesteps, device=device, dtype=dtype)
        return solver.sample(
            time_grid=T, x_init=x_init,
            method=solver_method,
            step_size=1.0 / n_timesteps,
            return_intermediates=False,
            pre_encoded_context=context_encoded if cfg_scale == 0 else None,
        )
