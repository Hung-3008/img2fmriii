"""
dit_1d.py
=========
1D Diffusion Transformer for fMRI voxel synthesis.

Native 1D architecture — no 2D reshape needed.
Reuses LightningDiTBlock (AdaLN-Zero, SwiGLU, RMSNorm, QKNorm)
with 1D Conv1d patch embedding and 1D RoPE.

Input:  (B, 1, seq_len)  e.g. (B, 1, 16384)
Output: (B, 1, seq_len)  predicted velocity
"""

import torch
import torch.nn as nn
import numpy as np

from .lightning_dit import LightningDiTBlock, VectorEmbedder
from .model_utils import (
    GaussianFourierEmbedding,
    RMSNorm,
    RotaryEmbedding1D,
    StreamRouter,
    get_1d_sincos_pos_embed,
    modulate,
)
from .subject_embedder import SubjectEmbedder, ZeroShotSubjectEmbedder


# ─── 1D Patch Embedding ──────────────────────────────────────────────

class PatchEmbed1D(nn.Module):
    """Tokenize a 1D signal into non-overlapping patches via Conv1d."""

    def __init__(self, seq_len: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, L) → (B, num_patches, embed_dim)"""
        x = self.proj(x)        # (B, D, N)
        x = x.transpose(1, 2)   # (B, N, D)
        return x


# ─── 1D Final Layer ──────────────────────────────────────────────────

class DiT1DFinalLayer(nn.Module):
    """Project tokens back to patch space with AdaLN modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool = False):
        super().__init__()
        if use_rmsnorm:
            self.norm_final = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─── DiT1D ───────────────────────────────────────────────────────────

class DiT1D(nn.Module):
    """
    1D Diffusion Transformer for fMRI voxel synthesis.

    Replaces the 2D LightningDiT — operates directly on flat voxel
    vectors without artificial 2D reshape.

    Architecture:
        Conv1d(patch_size) → [LightningDiTBlock × depth] → Linear → unpatchify
        Conditioning: t (timestep) + y (CLIP pool) via AdaLN in each block.
    """

    def __init__(
        self,
        seq_len: int = 16384,
        patch_size: int = 32,
        in_channels: int = 1,
        hidden_size: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        y_in_channels: int = 1280,
        learn_sigma: bool = False,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        wo_shift: bool = False,
        use_cross_attn: bool = False,
        y_token_channels: int = 1664,
        y_token_channels_2: int = None,
        context_dims=None,
        # ── ROI-Stratified Feature Routing ───────────────────
        use_roi_routing: bool = False,
        n_roi_buckets: int = 3,
        roi_emb_dim: int = 64,
        # ── Subject conditioning ──────────────────────────────
        use_subject_cond: bool = False,
        n_subjects: int = 8,
        subject_dropout: float = 0.1,
        subject_cond_mode: str = "learned",  # "learned" | "zero_shot"
        n_roi_buckets_profile: int = 3,       # only for zero_shot mode
    ):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.learn_sigma = learn_sigma
        self.depth = depth
        self.use_cross_attn = use_cross_attn
        self.use_roi_routing = use_roi_routing and use_cross_attn

        num_patches = seq_len // patch_size

        # ── Input stem ────────────────────────────────────────────────
        self.x_embedder = PatchEmbed1D(seq_len, patch_size, in_channels, hidden_size)
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = VectorEmbedder(y_in_channels, hidden_size)

        # ── Context stems ───────────────────────────────────────
        # One embedder per context stream (CLIP, DINOv2, multi-layer DINOv2,
        # Gabor, …). When use_roi_routing is False, results are concatenated
        # along the token axis into one shared context sequence (legacy).
        # When use_roi_routing is True, streams stay separate so each block
        # can attend to them independently with learned per-patch gates.
        self.context_embedders = None
        if use_cross_attn:
            if context_dims is None:
                context_dims = [y_token_channels]
                if y_token_channels_2 is not None:
                    context_dims.append(y_token_channels_2)
            self.context_dims = list(context_dims)
            self.context_embedders = nn.ModuleList([
                nn.Sequential(nn.Linear(int(d), hidden_size), RMSNorm(hidden_size))
                for d in self.context_dims
            ])

        # ── ROI-Stratified Feature Routing ──────────────────────
        self.stream_router = None
        if self.use_roi_routing:
            n_streams = len(self.context_dims)
            self.stream_router = StreamRouter(
                n_streams=n_streams,
                n_buckets=n_roi_buckets,
                emb_dim=roi_emb_dim,
            )
            # Placeholder; will be filled by set_roi_buckets()
            self.register_buffer(
                "bucket_ids",
                torch.zeros(1, num_patches, dtype=torch.long),
                persistent=True,
            )
        else:
            self.bucket_ids = None

        # ── Subject conditioning ──────────────────────────────────
        self.subject_embedder = None
        if use_subject_cond:
            if subject_cond_mode == "learned":
                self.subject_embedder = SubjectEmbedder(
                    n_subjects=n_subjects,
                    hidden_size=hidden_size,
                    dropout_prob=subject_dropout,
                )
            elif subject_cond_mode == "zero_shot":
                self.subject_embedder = ZeroShotSubjectEmbedder(
                    profile_dim=n_roi_buckets_profile,
                    hidden_size=hidden_size,
                )

        # 1D positional embedding (fixed sin-cos)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # 1D Rotary Position Embedding
        if use_rope:
            head_dim = hidden_size // num_heads
            self.feat_rope = RotaryEmbedding1D(dim=head_dim, seq_len=num_patches)
        else:
            self.feat_rope = None

        # ── Transformer blocks ───────────────────────────────────
        n_streams_routed = len(context_dims) if (use_cross_attn and self.use_roi_routing) else 0
        self.blocks = nn.ModuleList([
            LightningDiTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                wo_shift=wo_shift,
                use_cross_attn=use_cross_attn,
                n_streams_routed=n_streams_routed,
            ) for _ in range(depth)
        ])

        # ── Output head ──────────────────────────────────────────────
        self.final_layer = DiT1DFinalLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init for all linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 1D sin-cos positional embedding
        num_patches = self.seq_len // self.patch_size
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embed like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning embedding MLP
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN layers → identity init for stable start
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-out cross-attention output proj → block starts as pooled-only
        # DiT, then learns to use context.  Handles both legacy (cross_attn)
        # and routed (cross_attns) paths.
        for block in self.blocks:
            if getattr(block, "use_cross_attn", False):
                if block.cross_attn is not None:
                    nn.init.constant_(block.cross_attn.proj.weight, 0)
                    nn.init.constant_(block.cross_attn.proj.bias, 0)
                if hasattr(block, "cross_attns"):
                    for ca in block.cross_attns:
                        nn.init.constant_(ca.proj.weight, 0)
                        nn.init.constant_(ca.proj.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_size * out_channels) → (B, out_channels, seq_len)
        """
        B, N, _ = x.shape
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(B, N, p, c)     # (B, N, P, C)
        x = x.permute(0, 3, 1, 2)     # (B, C, N, P)
        x = x.reshape(B, c, N * p)    # (B, C, N*P) = (B, C, seq_len)
        return x

    def set_roi_buckets(
        self,
        voxel_bucket_ids: np.ndarray,
        pad_to: int,
        n_roi_buckets: int,
    ) -> None:
        """Precompute and cache patch-level ROI bucket IDs.

        Called once after the model is built, before training begins.
        Aggregates voxel-level bucket ids to patch level by majority-vote
        (mean → round), then stores as a ``(1, N)`` buffer on the model device.

        Args:
            voxel_bucket_ids: int array (n_voxels,) — ROI bucket per voxel
                              in the sorted (roi_order) space.
            pad_to:           padded sequence length (must match self.seq_len)
            n_roi_buckets:    total number of buckets (clips values to [0, n-1])
        """
        if self.stream_router is None:
            return
        P = self.patch_size
        # Pad to seq_len with bucket 0 (early visual — conservative default)
        padded = np.zeros(pad_to, dtype=np.float32)
        padded[: len(voxel_bucket_ids)] = voxel_bucket_ids.astype(np.float32)
        # Reshape to (N, P) → mean per patch → round → clip
        n_patches = pad_to // P
        patch_buckets = padded.reshape(n_patches, P).mean(axis=1).round().astype(np.int64)
        patch_buckets = np.clip(patch_buckets, 0, n_roi_buckets - 1)
        buf = torch.from_numpy(patch_buckets).long().unsqueeze(0)  # (1, N)
        # Store on same device as model
        self.bucket_ids = buf.to(self.pos_embed.device)
        self.register_buffer("bucket_ids", self.bucket_ids, persistent=True)

    def _embed_contexts(self, contexts):
        """Project each context stream to hidden dim.

        Returns:
            If use_roi_routing: list of (B, M_s, D) tensors (one per stream)
            Else: single (B, M_total, D) tensor (legacy concatenation)
        """
        if not self.use_cross_attn or contexts is None or self.context_embedders is None:
            return None
        embedded = [emb(c) for emb, c in zip(self.context_embedders, contexts)]
        if self.use_roi_routing:
            return embedded               # keep separate for per-stream cross-attn
        return torch.cat(embedded, dim=1) if len(embedded) > 1 else embedded[0]

    def forward(self, x, t=None, y=None, contexts=None,
                subject_ids=None, roi_profile=None):
        """
        Forward pass of DiT1D.

        Args:
            x:           (B, C, L) — 1D fMRI signal (C=1 typically)
            t:           (B,) — diffusion timestep
            y:           (B, D_pool) — CLIP pooled conditioning
            contexts:    list of (B, Mᵢ, Dᵢ) cross-attention streams.
            subject_ids: (B,) long — 0-indexed subject ID (for learned mode).
            roi_profile: (B, n_buckets) float — normalised ROI profile (for zero_shot mode).

        Returns:
            (B, C, L) — predicted velocity
        """
        B = x.shape[0]
        x = self.x_embedder(x) + self.pos_embed   # (B, N, D)
        t = self.t_embedder(t)                      # (B, D)
        y = self.y_embedder(y)                      # (B, D)
        c = t + y                                    # (B, D)

        # Subject conditioning: c = t + y + s_emb
        if self.subject_embedder is not None:
            if isinstance(self.subject_embedder, SubjectEmbedder):
                # Learned: needs subject_ids
                if subject_ids is None:
                    # Fallback: unconditional (null token) — safe for single-subject configs
                    subject_ids = torch.full(
                        (B,), self.subject_embedder.null_id,
                        dtype=torch.long, device=x.device,
                    )
                s_emb = self.subject_embedder(subject_ids, self.training)  # (B, D)
            else:
                # Zero-shot: needs roi_profile
                assert roi_profile is not None, "roi_profile required for zero_shot subject embedder"
                s_emb = self.subject_embedder(roi_profile)                 # (B, D)
            c = c + s_emb

        ctx = self._embed_contexts(contexts)

        if self.use_roi_routing and self.stream_router is not None:
            # Expand bucket_ids to batch dimension: (1, N) → (B, N)
            bids = self.bucket_ids.expand(B, -1)          # (B, N)
            stream_gates = self.stream_router(bids)        # (B, N, S)
            for block in self.blocks:
                x = block(x, c, feat_rope=self.feat_rope,
                          contexts_list=ctx, stream_gates=stream_gates)
        else:
            for block in self.blocks:
                x = block(x, c, feat_rope=self.feat_rope, context=ctx)

        x = self.final_layer(x, c)   # (B, N, P*C)
        x = self.unpatchify(x)       # (B, C, L)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x
