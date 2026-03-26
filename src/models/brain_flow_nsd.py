"""BrainFlow NSD — CSFM-Style Conditional Source Flow Matching.

Uses CSFM's transport module (path interpolation, time sampling, ODE solver)
to ensure correct gradient flow from flow loss → SourceEncoder.

Key CSFM conventions:
  - Path: xt = (1-t)·x₁ + t·x₀  (alpha_t*x1 + sigma_t*x0)
  - Time: t goes from 1→0 during sampling (source → target)
  - Velocity target: ut = d_alpha·x₁ + d_sigma·x₀ = -x₁ + x₀ = x₀ - x₁
  - detach_ut=True: velocity target detached, but x₀ gradient flows through xt
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# CSFM transport (our adapted copy)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transport import create_transport, Sampler


# =============================================================================
# Building Blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
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
# SourceEncoder — CSFM-style Perceiver Variational Encoder
# =============================================================================

class SourceEncoderLayer(nn.Module):
    """Cross-Attention (queries ← context) + Self-Attention + FFN."""
    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop_cross = nn.Dropout(dropout)
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.drop_sa = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )

    def forward(self, queries, context):
        q_norm, kv_norm = self.norm_q(queries), self.norm_kv(context)
        ca_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        queries = queries + self.drop_cross(ca_out)
        sa_norm = self.norm_sa(queries)
        sa_out, _ = self.self_attn(sa_norm, sa_norm, sa_norm)
        queries = queries + self.drop_sa(sa_out)
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class SourceEncoder(nn.Module):
    """Perceiver-style variational encoder: multimodal features → (μ, log_var).
    
    Like CSFM's PerceiverVE but adapted for 256-D output.
    """
    def __init__(self, modality_dims, hidden_dim=512, output_dim=256,
                 n_queries=8, n_layers=2, n_heads=8, dropout=0.1,
                 init_logvar=0.0, n_modalities=3, max_layers_per_modality=8, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.n_queries = n_queries
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            for d in modality_dims
        ])
        self.modality_emb = nn.Embedding(n_modalities, hidden_dim)
        self.layer_pos_emb = nn.Embedding(max_layers_per_modality, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            SourceEncoderLayer(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_var_head = nn.Linear(hidden_dim, output_dim)
        nn.init.normal_(self.log_var_head.weight, std=1e-4)
        nn.init.constant_(self.log_var_head.bias, init_logvar)

        params = sum(p.numel() for p in self.parameters())
        print(f"[SourceEncoder] {params:,} params (hidden={hidden_dim}, queries={n_queries}, layers={n_layers})")

    def _project_context(self, ctx):
        keys = ['dino', 'clip', 'qwen']
        projected = []
        for i, k in enumerate(keys):
            if k in ctx:
                x = ctx[k]
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                p = self.projs[i](x)
                L_i = p.shape[1]
                p = p + self.modality_emb.weight[i].unsqueeze(0).unsqueeze(0)
                layer_ids = torch.arange(L_i, device=p.device)
                p = p + self.layer_pos_emb(layer_ids).unsqueeze(0)
                projected.append(p)
        return torch.cat(projected, dim=1)

    def forward(self, ctx):
        B = next(iter(ctx.values())).shape[0]
        context = self._project_context(ctx)
        queries = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            queries = layer(queries, context)
        pooled = self.out_norm(queries.mean(dim=1))
        mu = self.mean_head(pooled)
        log_var = self.log_var_head(pooled)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(mu) * std


# =============================================================================
# MultiTokenFusion — Per-modality projection for DiT context
# =============================================================================

class MultiTokenFusion(nn.Module):
    def __init__(self, in_dims, hidden_dim, dropout=0.1,
                 n_modalities=3, max_layers_per_modality=8):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            for d in in_dims
        ])
        self.modality_emb = nn.Embedding(n_modalities, hidden_dim)
        self.layer_pos_emb = nn.Embedding(max_layers_per_modality, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, ctx):
        keys = ['dino', 'clip', 'qwen']
        projected = []
        for i, k in enumerate(keys):
            if k in ctx:
                x = ctx[k]
                if x.ndim == 2:
                    x = x.unsqueeze(1)
                p = self.projs[i](x)
                L_i = p.shape[1]
                p = p + self.modality_emb.weight[i].unsqueeze(0).unsqueeze(0)
                layer_ids = torch.arange(L_i, device=p.device)
                p = p + self.layer_pos_emb(layer_ids).unsqueeze(0)
                projected.append(p)
        return self.output_proj(torch.cat(projected, dim=1))


# =============================================================================
# AdaLN-Zero DiT Block
# =============================================================================

class AdaLNZeroBlock(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_mod = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x_mod2 = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_mod2)
        return x


# =============================================================================
# BrainDiT_Net — Velocity Network (for DiT)
# =============================================================================

class BrainDiT_Net(nn.Module):
    def __init__(self, output_dim=256, hidden_dim=768, modality_dims=None,
                 n_blocks=6, n_heads=12, dropout=0.1, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.modality_dims = modality_dims or [4352]
        self.fusion_block = MultiTokenFusion(self.modality_dims, hidden_dim, dropout)
        self.input_proj = nn.Sequential(nn.Linear(output_dim, hidden_dim), nn.GELU())
        self.time_pos_emb = SinusoidalPosEmb(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([AdaLNZeroBlock(hidden_dim, n_heads, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.constant_(self.final_modulation[-1].weight, 0)
        nn.init.constant_(self.final_modulation[-1].bias, 0)
        nn.init.constant_(self.output_layer.weight, 0)
        nn.init.constant_(self.output_layer.bias, 0)

    def encode_context(self, cond):
        return self.fusion_block(cond)

    def forward(self, x, t, cond=None, pre_encoded_context=None, **kwargs):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if pre_encoded_context is not None:
            ctx = pre_encoded_context
        elif cond is not None:
            ctx = self.encode_context(cond)
        else:
            ctx = torch.zeros(x.shape[0], 1, self.hidden_dim, device=x.device, dtype=x.dtype)
        t_emb = self.time_mlp(self.time_pos_emb(t))
        h = torch.cat([self.input_proj(x).unsqueeze(1), ctx], dim=1)
        for block in self.blocks:
            h = block(h, t_emb)
        x_out = h[:, 0, :]
        shift_c, scale_c = self.final_modulation(t_emb).chunk(2, dim=1)
        return self.output_layer(modulate(self.final_norm(x_out), shift_c, scale_c))


# =============================================================================
# BrainFlowNSD — CSFM-Style
# =============================================================================

class BrainFlowNSD(nn.Module):
    """CSFM-style flow matching with learned conditional source distribution.
    
    Uses CSFM's transport module for correct gradient flow.
    Source: x₀ = SourceEncoder(features) → (μ, log_var) → reparameterize
    Target: x₁ = target_latent
    
    CSFM path convention:
      xt = (1-t)·x₁ + t·x₀   (alpha_t*x1 + sigma_t*x0)
      ut = -x₁ + x₀ = x₀ - x₁
      Sampling: t goes 1→0
    """

    def __init__(
        self,
        output_dim: int = 256,
        dit_net_params: dict = None,
        source_encoder_params: dict = None,
        transport_params: dict = None,
        kld_weight: float = 1e-3,
        align_weight: float = 0.1,
        detach_ut: bool = True,
        vae: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.kld_weight = kld_weight
        self.align_weight = align_weight
        self.detach_ut = detach_ut

        # Frozen VAE
        self.vae = vae
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        # Source Encoder (separate from DiT)
        se_cfg = dict(source_encoder_params or {})
        se_cfg.setdefault("output_dim", output_dim)
        self.source_encoder = SourceEncoder(**se_cfg)

        # Velocity network (DiT)
        dn_cfg = dict(dit_net_params or {})
        dn_cfg.setdefault("output_dim", output_dim)
        self.dit_net = BrainDiT_Net(**dn_cfg)

        # CSFM Transport (path interpolation, time sampling)
        tp_cfg = dict(transport_params or {})
        tp_cfg.setdefault("path_type", "Linear")
        tp_cfg.setdefault("prediction", "velocity")
        self.transport = create_transport(**tp_cfg)

        # CSFM Sampler (for ODE inference)  
        self.transport_sampler = Sampler(self.transport)

        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        se_params = sum(p.numel() for p in self.source_encoder.parameters())
        dit_params = sum(p.numel() for p in self.dit_net.parameters())
        print(f"[BrainFlowNSD] CSFM: {params:,} total "
              f"(SourceEnc={se_params:,}, DiT={dit_params:,})")

    @torch.no_grad()
    def _encode_to_latent(self, fmri):
        if self.vae is not None:
            return self.vae.get_latent(fmri)
        return fmri

    @torch.no_grad()
    def _decode_from_latent(self, z):
        if self.vae is not None:
            return self.vae.decode(z)
        return z

    def forward(
        self,
        context: dict[str, torch.Tensor],
        target_fmri: torch.Tensor,
        context_dropped: dict[str, torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """CSFM-style training step.
        
        Gradient flow:
          flow_loss → v_pred → DiT (direct)
          flow_loss → v_pred → DiT input xt → x₀ → SourceEncoder (through xt)
          kld_loss + align_loss → μ, log_var → SourceEncoder (direct)
        """
        # 0. Encode fMRI target → latent
        target_latent = self._encode_to_latent(target_fmri)  # (B, 256)
        x1 = target_latent  # x₁ = target (CSFM convention)

        # 1. Source Encoder → (μ, log_var) → x₀ (NOT detached!)
        mu, log_var = self.source_encoder(context)
        log_var = torch.clamp(log_var, min=-4.0, max=2.0)  # min σ ≈ 0.14
        x0 = self.source_encoder.reparameterize(mu, log_var)  # x₀ = source

        # 2. KLD loss: regularize source toward N(0,1)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # 3. Alignment loss: encourage μ → target
        align_loss = F.mse_loss(mu, target_latent)

        # 4. CSFM Flow Matching (using CSFM's transport)
        # Time sampling (CSFM-style, with optional logit-normal + shift)
        t = self.transport.sample_timestep(x1)

        # Path interpolation: xt = (1-t)·x₁ + t·x₀, ut = -x₁ + x₀
        # x₀ maintains gradient → SourceEncoder gets gradient through xt
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)

        # DiT context encoding (separate from source encoder)
        if context_dropped is not None:
            flow_ctx = self.dit_net.encode_context(context_dropped)
        else:
            flow_ctx = self.dit_net.encode_context(context)

        # Velocity prediction
        v_pred = self.dit_net(x=xt, t=t, pre_encoded_context=flow_ctx)

        # Flow loss with detach_ut (like CSFM)
        ut_target = ut.detach() if self.detach_ut else ut
        flow_loss = F.mse_loss(v_pred, ut_target)

        total_loss = flow_loss + self.kld_weight * kld_loss + self.align_weight * align_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "kld_loss": kld_loss,
            "align_loss": align_loss,
            "mu_norm": mu.detach().norm(dim=-1).mean(),
            "std_mean": torch.exp(0.5 * log_var).detach().mean(),
        }

    @torch.no_grad()
    def predict_regression(self, context: dict, decode: bool = True):
        """Predict fMRI via source encoder μ (coarse estimate)."""
        self.eval()
        mu, _ = self.source_encoder(context)
        return self._decode_from_latent(mu) if decode else mu

    @torch.no_grad()
    def synthesise(
        self,
        context: dict,
        n_timesteps: int = 50,
        solver_method: str = "dopri5",
        cfg_scale: float = 0.0,
        decode: bool = True,
    ) -> torch.Tensor:
        """CSFM inference: SourceEncoder μ → ODE solve (t=1→0) → decode.
        
        Uses CSFM's ODE sampler with torchdiffeq.
        """
        self.eval()
        device = next(self.parameters()).device

        # Start from source encoder μ (deterministic at inference)
        mu, _ = self.source_encoder(context)
        x_init = mu

        # Pre-encode DiT context
        context_encoded = self.dit_net.encode_context(context)

        if cfg_scale > 0:
            uncond_context = {k: torch.zeros_like(v) for k, v in context.items()}
            uncond_encoded = self.dit_net.encode_context(uncond_context)

            def model_fn(x, t, **kwargs):
                B = x.shape[0]
                tb = t.expand(B) if t.dim() == 0 else t
                vc = self.dit_net(x=x, t=tb, pre_encoded_context=context_encoded)
                vu = self.dit_net(x=x, t=tb, pre_encoded_context=uncond_encoded)
                return vu + cfg_scale * (vc - vu)
        else:
            def model_fn(x, t, **kwargs):
                B = x.shape[0]
                tb = t.expand(B) if t.dim() == 0 else t
                return self.dit_net(x=x, t=tb, pre_encoded_context=context_encoded)

        # CSFM ODE sampler (t goes from 1→0)
        ode_sampler = self.transport_sampler.sample_ode(
            sampling_method=solver_method,
            num_steps=n_timesteps,
        )
        traj = ode_sampler(x_init, model_fn)
        latent_pred = traj[-1]  # Final state at t≈0 is the target

        return self._decode_from_latent(latent_pred) if decode else latent_pred
