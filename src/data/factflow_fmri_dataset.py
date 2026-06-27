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
        fmri:      (1, V_pad) or (C, H, W) — fMRI beta-weights
        clip_pool: (D_pool,) — CLIP pooled embedding (for DiT AdaLN conditioning)
        pad_mask:  (V_pad,)  — boolean, True for real voxels
        contexts:  list of (Tᵢ, Dᵢ) — cross-attention image-feature streams
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
        roi_order: bool = False,
        voxel_order: str = "default",
        voxel_perm_seed: int = 0,
        context_features=None,
        subdirs=None,
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
        self.roi_order = roi_order

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
        self.subj_dir = subj_dir
        # Config-driven sub-directory layout. ``data.subdirs`` maps a category
        # (``"fmri"`` or a feature basename like ``sdxl_clip``) to its sub-folder
        # under the subject dir. The final path is a single relative join:
        #   data_dir / subj0{S} / subdirs[key] / filename
        # A key absent from the map resolves to the subject root ("").
        self.subdirs = dict(subdirs or {})

        # --- fMRI: (N_images, 3, V) ---
        fmri_path = self._resolve(f"nsd_{mode}_fmri_{fmri_mode}_sub{subject}.npy", "fmri")
        self.fmri_data = np.load(fmri_path, mmap_mode="r")  # (N_img, 3, V)
        self.n_images = self.fmri_data.shape[0]
        self.n_reps = self.fmri_data.shape[1]
        # avg_reps: treat each image as a single sample (mean across reps)
        self.n_samples = self.n_images if avg_reps else self.n_images * self.n_reps

        # --- CLIP pooled: (N_images, D_pool) --- (AdaLN conditioning) ---
        clip_pool_path = self._resolve(
            f"nsd_{clip_feature}_pool_{mode}_sub{subject}.npy", clip_feature
        )
        self.clip_pool = np.load(clip_pool_path, mmap_mode="r")

        # --- Cross-attention context streams (one or more token sequences) ---
        # Each stream is a separate image-feature source (CLIP, DINOv2, multi-layer
        # DINOv2, Gabor energy, …). ``context_features`` is a list of feature names;
        # if unset it falls back to the legacy [clip_feature (+ dino_feature)] pair.
        # Features stored as (N, T, D) are token sequences; (N, L, T, D) multi-layer
        # features are flattened to (L*T, D) per sample.
        if context_features is None:
            context_features = [clip_feature]
            if dino_feature is not None:
                context_features.append(dino_feature)
        self.context_features = list(context_features)
        self.context_mmaps = []
        for feat in self.context_features:
            p = self._resolve(f"nsd_{feat}_{mode}_sub{subject}.npy", feat)
            m = np.load(p, mmap_mode="r")
            assert m.shape[0] == self.n_images, f"{feat}: {m.shape[0]} != {self.n_images}"
            self.context_mmaps.append(m)
        # Per-stream feature dim (last axis) — used to size the model embedders.
        self.context_dims = [int(m.shape[-1]) for m in self.context_mmaps]

        assert self.clip_pool.shape[0] == self.n_images

        # --- Pad mask: True for real voxels ---
        self.pad_mask = np.zeros(pad_to, dtype=np.bool_)
        self.pad_mask[:n_voxels] = True

        # --- Optional ROI reordering: cluster same-ROI voxels contiguously ---
        # ``sorted_fmri[i] = raw_fmri[sort_idx[i]]`` — applied to every sample so
        # the velocity field operates on an ROI-grouped sequence (patches become
        # ROI-coherent). The metric is permutation-invariant (pred & GT both
        # sorted), so no un-sort is needed for training/eval; ``unsort_idx`` is
        # exposed for restoring anatomical order when exporting predictions.
        self.sort_idx = None
        self.unsort_idx = None
        if roi_order:
            roi_path = os.path.join(subj_dir, f"roi_meta_sub{subject}.npz")
            meta = np.load(roi_path)
            self.sort_idx = meta["sort_idx"].astype(np.int64)
            self.unsort_idx = meta["unsort_idx"].astype(np.int64)
            assert self.sort_idx.shape[0] == n_voxels, (
                f"roi_meta sort_idx has {self.sort_idx.shape[0]} voxels, "
                f"expected n_voxels={n_voxels}"
            )
            logger.info(
                "ROI voxel ordering ON: subj=%d, voxels=%d (atlas=streams)",
                subject, n_voxels,
            )

        shape_str = (
            f"({fmri_channels},{fmri_spatial},{fmri_spatial})"
            if self.reshape_2d else f"(1,{pad_to})"
        )
        logger.info(
            "FactFlowfMRIDataset: subj=%d, mode=%s, images=%d, reps=%d, "
            "samples=%d, voxels=%d→%d, shape=%s, "
            "contexts=%s dims=%s",
            subject, mode, self.n_images, self.n_reps,
            self.n_samples, n_voxels, pad_to, shape_str,
            self.context_features, self.context_dims,
        )

    def _resolve(self, filename: str, key: str) -> str:
        """Join the config-declared sub-folder for ``key`` into one relative path:
        ``data_dir / subj0{S} / subdirs[key] / filename`` (root if ``key`` unmapped).
        """
        return os.path.join(self.subj_dir, self.subdirs.get(key, ""), filename)

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

        # --- CLIP pooled: same for all reps of same image ---
        clip_p = self.clip_pool[image_idx].astype(np.float32)

        # --- Cross-attention context streams ---
        contexts = []
        for m in self.context_mmaps:
            arr = np.asarray(m[image_idx]).astype(np.float32)
            if arr.ndim == 3:                    # (L, T, D) multi-layer → (L*T, D)
                arr = arr.reshape(-1, arr.shape[-1])
            contexts.append(torch.from_numpy(arr))

        return {
            "fmri": torch.from_numpy(fmri_out),
            "clip_pool": torch.from_numpy(clip_p),     # (D_pool,) — AdaLN conditioning
            "pad_mask": torch.from_numpy(self.pad_mask.copy()),  # (V_pad,)
            "contexts": contexts,              # list of (Tᵢ, Dᵢ) cross-attn streams
        }

    @property
    def voxel_count(self) -> int:
        return self.n_voxels

    @property
    def clip_pool_dim(self) -> int:
        return self.clip_pool.shape[-1]
