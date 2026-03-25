"""NSD Dataset — Simple PyTorch Dataset for NSD discrete image→fMRI pairs.

Loads pre-extracted pooled features (DINOv2, CLIP, Qwen3-VL) and fMRI data.
fMRI has 3 repetitions per image, averaged to get a single target per sample.
PCA is fitted on training fMRI and applied to both train/test.

Data shapes (subj01):
  - fmri_train: (9000, 3, 15724) float64 → avg → (9000, 15724)
  - fmri_test:  (1000, 3, 15724) float64 → avg → (1000, 15724)
  - dinov2_pool: (N, 1024)   float16
  - clip_pool:   (N, 1280)   float32
  - qwen_pool:   (N, 2048)   float16
"""

import logging
import os

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

logger = logging.getLogger("nsd_dataset")


class NSDDataset(Dataset):
    """NSD dataset for discrete image→fMRI mapping.

    Each sample returns:
        context:     dict of features
        fmri:        (n_voxels,)     raw fMRI (averaged reps)
        subject_idx: int             subject index (0 for single-subject)
    """

    def __init__(
        self,
        data_dir: str,
        subject: int,
        mode: str = "train",         # "train" or "test"
    ):
        super().__init__()
        self.mode = mode
        self.subject = subject

        subj_dir = os.path.join(data_dir, f"subj0{subject}")
        logger.info("Loading NSD data from %s (mode=%s)", subj_dir, mode)

        # --- Load fMRI: (N, 3, 15724) → average reps → (N, 15724) ---
        fmri_path = os.path.join(subj_dir, f"nsd_{mode}_fmri_scale_sub{subject}.npy")
        fmri_raw = np.load(fmri_path)  # (N, 3, V)
        logger.info("  fMRI raw: %s %s", fmri_raw.shape, fmri_raw.dtype)

        # Average across repetitions
        self.fmri = torch.from_numpy(fmri_raw.mean(axis=1).astype(np.float32))  # (N, V)
        del fmri_raw
        logger.info("  fMRI averaged: %s", self.fmri.shape)

        def load_feature(mod_name, default_prefix):
            multi_path = os.path.join(subj_dir, f"nsd_{default_prefix}_multi_{mode}_sub{subject}.npy")
            pool_path = os.path.join(subj_dir, f"nsd_{default_prefix}_pool_{mode}_sub{subject}.npy")
            
            if os.path.exists(multi_path):
                arr = np.load(multi_path)
            elif os.path.exists(pool_path):
                arr = np.load(pool_path)
                # Ensure sequence dimension exists: (N, D) -> (N, 1, D)
                if arr.ndim == 2:
                    arr = np.expand_dims(arr, axis=1)
            else:
                raise FileNotFoundError(f"Missing {mod_name} feature file for {mode} split")
            
            return torch.from_numpy(arr.astype(np.float32))

        self.feat_dinov2 = load_feature("dinov2", "dinov2_vitl14")
        self.feat_clip = load_feature("clip", "sdxl_clip")
        self.feat_qwen = load_feature("qwen", "qwen3vl")

        logger.info("  DINOv2: %s, CLIP: %s, Qwen: %s", self.feat_dinov2.shape, self.feat_clip.shape, self.feat_qwen.shape)



        self.n_samples = self.fmri.shape[0]
        assert self.feat_dinov2.shape[0] == self.n_samples
        assert self.feat_clip.shape[0] == self.n_samples
        assert self.feat_qwen.shape[0] == self.n_samples

        logger.info("  Dataset ready: %d samples", self.n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = self.fmri[idx]  # (V=15724,)

        return {
            "context": {
                "dino": self.feat_dinov2[idx], # (L1, 1024)
                "clip": self.feat_clip[idx],   # (L2, 1280)
                "qwen": self.feat_qwen[idx],   # (L3, 2048)
            },
            "fmri": fmri,             # (15724,)
            "subject_idx": 0,          # single-subject
        }
