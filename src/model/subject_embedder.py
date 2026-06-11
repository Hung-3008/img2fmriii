"""
subject_embedder.py
===================
Subject-specific conditioning modules for cross-subject flow matching.

Two variants:

1. **SubjectEmbedder** (few-shot, learnable):
   - ``nn.Embedding(n_subjects + 1, hidden_size)`` — one vector per subject.
   - Trained jointly with the DiT during multi-subject training.
   - For a **new subject**: initialise a new embedding and fine-tune ONLY
     ``SubjectEmbedder.emb[new_id]`` + ``StreamRouter.gate_proj`` (~512 + 3K params)
     with ~30 min of scan data.  All other DiT params stay frozen.
   - Subject dropout during training (replace subject_id → null_id with prob p)
     gives the model an unconditional fallback.

2. **ZeroShotSubjectEmbedder** (zero-shot, anatomical):
   - Projects a normalised ROI bucket count profile to hidden_size.
   - No per-subject parameters → works out-of-box for unseen subjects whose
     roi_meta_subS.npz is available (always the case after NSD preprocessing).
   - Weaker but requires zero scan time.

Usage in DiT1D.forward()::

    c = t_emb + y_emb                   # baseline
    if subject_embedder is not None:
        s = subject_embedder(subject_ids, self.training)   # (B, D)
        c = c + s                        # subject-conditioned AdaLN
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_utils import RMSNorm


class SubjectEmbedder(nn.Module):
    """Learnable per-subject embedding for AdaLN conditioning.

    Args:
        n_subjects:    Number of subjects in the training pool.
        hidden_size:   DiT hidden dimension (must match y_embedder output).
        dropout_prob:  Probability of replacing subject_id with the null token
                       during training → classifier-free guidance style fallback.
    """

    def __init__(
        self,
        n_subjects: int,
        hidden_size: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_subjects = n_subjects
        self.null_id = n_subjects          # index for "unknown / null" subject
        self.dropout_prob = dropout_prob

        # +1 for null/unconditional token
        self.emb = nn.Embedding(n_subjects + 1, hidden_size)
        self.norm = RMSNorm(hidden_size)

        # Zero-init: start as identity (s_emb = 0), model learns from scratch
        nn.init.zeros_(self.emb.weight)

    def forward(
        self,
        subject_ids: torch.Tensor,
        training: bool = False,
    ) -> torch.Tensor:
        """Return subject embeddings for a batch.

        Args:
            subject_ids: (B,) long tensor — 0-indexed subject index.
            training:    If True, applies subject dropout.

        Returns:
            (B, hidden_size) subject embeddings.
        """
        if training and self.dropout_prob > 0.0:
            drop = torch.rand(subject_ids.shape, device=subject_ids.device) < self.dropout_prob
            subject_ids = torch.where(drop, self.null_id, subject_ids)
        return self.norm(self.emb(subject_ids))   # (B, D)

    # ── Fine-tune helpers ────────────────────────────────────────────────────

    def get_finetune_params(self, new_subject_id: int):
        """Return parameters to unfreeze when adapting to a new subject.

        For new subjects: only the new embedding row needs gradient.
        Caller should freeze everything else in the model.

        Usage::

            params = subject_emb.get_finetune_params(new_id)
            optimizer = AdamW(params, lr=1e-4)
        """
        return [self.emb.weight[new_subject_id:new_subject_id + 1]]


class ZeroShotSubjectEmbedder(nn.Module):
    """Anatomical fingerprint → subject embedding (no per-subject parameters).

    Uses the ROI bucket profile (normalised voxel count per bucket) as a
    subject descriptor.  Works for any subject whose roi_meta is available —
    even without any fMRI data.

    Args:
        profile_dim:  Number of ROI buckets (e.g. 3: early/mid/high).
        hidden_size:  DiT hidden dimension.
    """

    def __init__(self, profile_dim: int = 3, hidden_size: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(profile_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            RMSNorm(hidden_size),
        )
        # Zero-init output layer → s_emb = 0 at init
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)

    def forward(self, roi_profile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_profile: (B, profile_dim) — normalised voxel fraction per bucket.
                         e.g. [0.90, 0.10, 0.004] for Sub1.

        Returns:
            (B, hidden_size) subject embedding.
        """
        return self.proj(roi_profile)
