"""
factflow_fmri_dataset.py
========================
Dataset for FactFlow-based fMRI synthesis.

Loads pre-extracted SDXL CLIP features and NSD fMRI betas,
returning them in a format ready for flow matching training.

fMRI is either:
- 1D mode: returned as (1, pad_to) — native flat voxel vector
- 2D mode: padded and reshaped to (C, H, W) for 2D DiT (legacy)
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FactFlowfMRIDataset(Dataset):
    """Dataset pairing CLIP visual features with fMRI betas.

    Each sample returns:
        fmri:        (1, V_pad) or (C, H, W) — fMRI beta-weights
        clip_tokens: (T, D)    — CLIP spatial tokens (for PerceiverVE)
        clip_pool:   (D_pool,) — CLIP pooled embedding (for DiT y-conditioning)
        pad_mask:    (V_pad,)  — boolean, True for real voxels
    """

    def __init__(
        self,
        data_dir: str,
        subject: int,
        mode: str = "train",
        fmri_mode: str = "scale",
        clip_feature: str = "sdxl_clip",
        n_voxels: int = 15724,
        pad_to: int = 16384,
        fmri_channels: int = 1,
        fmri_spatial: int = None,
    ):
        super().__init__()
        self.mode = mode
        self.subject = subject
        self.n_voxels = n_voxels
        self.pad_to = pad_to
        self.fmri_channels = fmri_channels
        self.fmri_spatial = fmri_spatial

        # Determine reshape mode
        if fmri_spatial is not None and fmri_spatial > 0:
            self.reshape_2d = True
            assert fmri_channels * fmri_spatial * fmri_spatial == pad_to, (
                f"fmri_channels * fmri_spatial² must equal pad_to: "
                f"{fmri_channels}×{fmri_spatial}² = {fmri_channels * fmri_spatial**2} ≠ {pad_to}"
            )
        else:
            self.reshape_2d = False

        subj_dir = os.path.join(data_dir, f"subj0{subject}")

        # --- fMRI: (N_images, 3, V) ---
        fmri_path = os.path.join(
            subj_dir, f"nsd_{mode}_fmri_{fmri_mode}_sub{subject}.npy"
        )
        self.fmri_data = np.load(fmri_path, mmap_mode="r")  # (N_img, 3, V)
        self.n_images = self.fmri_data.shape[0]
        self.n_reps = self.fmri_data.shape[1]
        self.n_samples = self.n_images * self.n_reps

        # --- CLIP tokens: (N_images, T, D) ---
        clip_tok_path = os.path.join(
            subj_dir, f"nsd_{clip_feature}_{mode}_sub{subject}.npy"
        )
        self.clip_tokens = np.load(clip_tok_path, mmap_mode="r")

        # --- CLIP pooled: (N_images, D_pool) ---
        clip_pool_path = os.path.join(
            subj_dir, f"nsd_{clip_feature}_pool_{mode}_sub{subject}.npy"
        )
        self.clip_pool = np.load(clip_pool_path, mmap_mode="r")

        assert self.clip_tokens.shape[0] == self.n_images
        assert self.clip_pool.shape[0] == self.n_images

        # --- Pad mask: True for real voxels ---
        self.pad_mask = np.zeros(pad_to, dtype=np.bool_)
        self.pad_mask[:n_voxels] = True

        shape_str = (
            f"({fmri_channels},{fmri_spatial},{fmri_spatial})"
            if self.reshape_2d else f"(1,{pad_to})"
        )
        logger.info(
            "FactFlowfMRIDataset: subj=%d, mode=%s, images=%d, reps=%d, "
            "samples=%d, voxels=%d→%d, shape=%s, "
            "clip_tokens=%s, clip_pool=%s",
            subject, mode, self.n_images, self.n_reps,
            self.n_samples, n_voxels, pad_to, shape_str,
            self.clip_tokens.shape, self.clip_pool.shape,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image_idx = idx // self.n_reps
        rep_idx = idx % self.n_reps

        # --- fMRI: load, pad, reshape ---
        raw_fmri = self.fmri_data[image_idx, rep_idx].astype(np.float32)  # (V,)
        padded = np.zeros(self.pad_to, dtype=np.float32)
        padded[: self.n_voxels] = raw_fmri
        if self.reshape_2d:
            fmri_out = padded.reshape(self.fmri_channels, self.fmri_spatial, self.fmri_spatial)
        else:
            fmri_out = padded[np.newaxis, :]  # (1, pad_to)

        # --- CLIP features: same for all reps of same image ---
        clip_tok = self.clip_tokens[image_idx].astype(np.float32)
        clip_p = self.clip_pool[image_idx].astype(np.float32)

        return {
            "fmri": torch.from_numpy(fmri_out),
            "clip_tokens": torch.from_numpy(clip_tok), # (T, D)
            "clip_pool": torch.from_numpy(clip_p),     # (D_pool,)
            "pad_mask": torch.from_numpy(self.pad_mask.copy()),  # (V_pad,)
        }

    @property
    def voxel_count(self) -> int:
        return self.n_voxels

    @property
    def clip_token_dim(self) -> int:
        return self.clip_tokens.shape[-1]

    @property
    def clip_pool_dim(self) -> int:
        return self.clip_pool.shape[-1]
