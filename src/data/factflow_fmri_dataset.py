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
        avg_reps: bool = False,
        dino_feature: str = None,
        use_roi: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.subject = subject
        self.n_voxels = n_voxels
        self.pad_to = pad_to
        self.fmri_channels = fmri_channels
        self.fmri_spatial = fmri_spatial
        self.avg_reps = avg_reps
        self.dino_feature = dino_feature
        self.use_roi = use_roi

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
        # avg_reps: treat each image as a single sample (mean across reps)
        self.n_samples = self.n_images if avg_reps else self.n_images * self.n_reps

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

        # --- Optional DINOv2 tokens: (N_images, T2, D2) ---
        # Second visual-feature stream for cross-attention (early-visual cortex).
        self.dino_tokens = None
        if dino_feature is not None:
            dino_path = os.path.join(
                subj_dir, f"nsd_{dino_feature}_{mode}_sub{subject}.npy"
            )
            self.dino_tokens = np.load(dino_path, mmap_mode="r")
            assert self.dino_tokens.shape[0] == self.n_images

        assert self.clip_tokens.shape[0] == self.n_images
        assert self.clip_pool.shape[0] == self.n_images

        # --- Pad mask: True for real voxels ---
        self.pad_mask = np.zeros(pad_to, dtype=np.bool_)
        self.pad_mask[:n_voxels] = True

        # --- Optional ROI reordering: cluster same-ROI voxels contiguously ---
        # ``sorted_fmri[i] = raw_fmri[sort_idx[i]]`` — applied to every sample so
        # the velocity field operates on an ROI-grouped sequence (patches become
        # ROI-pure). The metric is permutation-invariant (pred & GT both sorted),
        # so no un-sort is needed for training/eval; ``unsort_idx`` is exposed for
        # restoring anatomical order when exporting predictions.
        self.sort_idx = None
        self.unsort_idx = None
        self.roi_labels = None
        self.n_roi = None
        if use_roi:
            roi_path = os.path.join(subj_dir, f"roi_meta_sub{subject}.npz")
            meta = np.load(roi_path)
            self.sort_idx = meta["sort_idx"].astype(np.int64)
            self.unsort_idx = meta["unsort_idx"].astype(np.int64)
            self.roi_labels = meta["roi_labels"].astype(np.int64)
            self.n_roi = int(meta["n_roi"])
            assert self.sort_idx.shape[0] == n_voxels, (
                f"roi_meta sort_idx has {self.sort_idx.shape[0]} voxels, "
                f"expected n_voxels={n_voxels}"
            )
            logger.info(
                "ROI reorder ON: subj=%d, n_roi=%d, voxels=%d (atlas=streams)",
                subject, self.n_roi, n_voxels,
            )

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
        if self.avg_reps:
            # One sample per image: average all reps to reduce session-level noise
            image_idx = idx
            raw_fmri = self.fmri_data[image_idx].astype(np.float32)  # (3, V)
            raw_fmri = raw_fmri.mean(axis=0)  # (V,)
        else:
            image_idx = idx // self.n_reps
            rep_idx = idx % self.n_reps
            raw_fmri = self.fmri_data[image_idx, rep_idx].astype(np.float32)  # (V,)
        if self.sort_idx is not None:
            raw_fmri = raw_fmri[self.sort_idx]  # reorder voxels → ROI-grouped
        padded = np.zeros(self.pad_to, dtype=np.float32)
        padded[: self.n_voxels] = raw_fmri
        if self.reshape_2d:
            fmri_out = padded.reshape(self.fmri_channels, self.fmri_spatial, self.fmri_spatial)
        else:
            fmri_out = padded[np.newaxis, :]  # (1, pad_to)

        # --- CLIP features: same for all reps of same image ---
        clip_tok = self.clip_tokens[image_idx].astype(np.float32)
        clip_p = self.clip_pool[image_idx].astype(np.float32)

        sample = {
            "fmri": torch.from_numpy(fmri_out),
            "clip_tokens": torch.from_numpy(clip_tok), # (T, D)
            "clip_pool": torch.from_numpy(clip_p),     # (D_pool,)
            "pad_mask": torch.from_numpy(self.pad_mask.copy()),  # (V_pad,)
        }
        if self.dino_tokens is not None:
            dino_tok = self.dino_tokens[image_idx].astype(np.float32)
            sample["dino_tokens"] = torch.from_numpy(dino_tok)  # (T2, D2)
        return sample

    @property
    def voxel_count(self) -> int:
        return self.n_voxels

    @property
    def clip_token_dim(self) -> int:
        return self.clip_tokens.shape[-1]

    @property
    def clip_pool_dim(self) -> int:
        return self.clip_pool.shape[-1]
