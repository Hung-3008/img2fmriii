"""
Dataset for fMRI 2D pseudo-image representation.

Loads flat fMRI vectors (15724,) and reshapes to (1, 128, 128) 
pseudo-images for Conv2D-based models.

Usage:
    dataset = Fmri2DDataset("path/to/subj01_train_avg.npy")
    sample = dataset[0]
    # sample["fmri_2d"]: (1, 128, 128)
    # sample["fmri_flat"]: (15724,) — for loss on real voxels
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Fmri2DDataset(Dataset):
    """
    fMRI dataset that reshapes flat voxel data to 2D pseudo-images.
    
    Args:
        fmri_path: Path to .npy file with shape (N, n_voxels)
        target_size: Spatial size for the pseudo-image (H = W = target_size)
        split: Dataset split name for logging
    """
    
    def __init__(
        self,
        fmri_path: str,
        target_size: int = 128,
        split: str = "train",
    ):
        super().__init__()
        
        fmri_path = Path(fmri_path)
        if not fmri_path.exists():
            raise FileNotFoundError(f"fMRI file not found: {fmri_path}")
        
        self.fmri = np.load(str(fmri_path), mmap_mode='r')
        self.n_samples = self.fmri.shape[0]
        self.n_voxels = self.fmri.shape[1]
        self.target_size = target_size
        self.padded_size = target_size * target_size
        
        print(
            f"Fmri2DDataset [{split}]: "
            f"fMRI shape={self.fmri.shape}, "
            f"target_size={target_size}×{target_size}={self.padded_size}, "
            f"padding={self.padded_size - self.n_voxels} voxels"
        )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        # Load flat fMRI
        fmri_flat = torch.from_numpy(self.fmri[idx].copy()).float()
        
        # Pad to target_size^2 and reshape to (1, H, W)
        fmri_padded = F.pad(fmri_flat, (0, self.padded_size - self.n_voxels), value=0.0)
        fmri_2d = fmri_padded.view(1, self.target_size, self.target_size)
        
        return {
            "fmri_2d": fmri_2d,       # (1, 128, 128) — input to VAE
            "fmri_flat": fmri_flat,    # (15724,) — for masked loss computation
        }
