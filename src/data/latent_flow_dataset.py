"""
Dataset for Stage 2: Flow Matching on aligned representation space.

Pairs pre-extracted aligned autoencoder representations with CLIP features
and raw fMRI (for validation).

Data layout:
    - Aligned representations: results/stage1_aligned/latents/subj01_{split}_avg_repr.npy [N, 257, 256]
    - CLIP features: NSD/data/features/clip_vitl14_openai_subj01.npy [10000, 257, 1024]
    - Raw fMRI: NSD/data/mindeye_nsd/processed/subj01_{split}_avg.npy [N, 15724]
    - feat_idx: NSD/data/mindeye_nsd/processed/subj01_{split}_avg_feat_idx.npy [N]
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LatentFlowDataset(Dataset):
    """
    Dataset for Stage 2 flow matching on aligned representations.

    Returns dict with keys:
        - z1:      [257, 256]    aligned fMRI representation (flow target)
        - context: [257, 1024]   ALL CLIP tokens (for cross-attention conditioning)
        - fmri:    [15724]       raw fMRI (for validation decode comparison)
    """

    def __init__(
        self,
        latents_path: str,
        fmri_path: str,
        feat_idx_path: str,
        clip_path: str,
        clip_layer: int = 0,
        split: str = "train",
    ):
        super().__init__()
        print(f"LatentFlowDataset [{split}]: Loading...")

        # ---- Aligned representations [N, 257, 256] or [N, 1024] ----
        self.latents = np.load(latents_path, mmap_mode="r")
        self.n_samples = self.latents.shape[0]
        print(f"  Latents: {self.latents.shape}")

        # ---- Raw fMRI [N, 15724] ----
        self.fmri = np.load(fmri_path, mmap_mode="r")
        if self.fmri.shape[0] > self.n_samples:
            # Use first n_samples (latents may be from a subset)
            self.fmri = self.fmri[:self.n_samples]
        assert self.fmri.shape[0] == self.n_samples, (
            f"fMRI {self.fmri.shape[0]} < latents {self.n_samples}"
        )
        print(f"  fMRI: {self.fmri.shape}")

        # ---- Feature index mapping [N] ----
        self.feat_idx = np.load(feat_idx_path)
        if len(self.feat_idx) > self.n_samples:
            self.feat_idx = self.feat_idx[:self.n_samples]
        assert len(self.feat_idx) == self.n_samples

        # ---- CLIP features [10000, 257, 1024] or [10000, L, 257, 1024] ----
        raw_feat = np.load(clip_path, mmap_mode="r")
        if raw_feat.ndim == 3:
            self.clip_tokens = raw_feat
        elif raw_feat.ndim == 4:
            self.clip_tokens = raw_feat[:, clip_layer, :, :]
        else:
            raise ValueError(f"Unexpected feature shape: {raw_feat.shape}")
        print(f"  CLIP: {self.clip_tokens.shape}")
        print(f"  Samples: {self.n_samples}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        z1 = torch.from_numpy(self.latents[idx].copy()).float()
        feat_i = self.feat_idx[idx]
        context = torch.from_numpy(self.clip_tokens[feat_i].copy()).float()
        fmri = torch.from_numpy(self.fmri[idx].copy()).float()
        return {"z1": z1, "context": context, "fmri": fmri}
