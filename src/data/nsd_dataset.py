import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional


class NSDDataset(Dataset):
    """
    Dataset for Natural Scenes Dataset (NSD).
    Loads preprocessed fMRI .npy files from the MindEye data layout.

    Expected directory structure:
        {data_root}/processed/{subject}_{split}_avg.npy      (averaged trials)
        {data_root}/processed/{subject}_{split}_single.npy   (single trials)
        {data_root}/embeddings/{embedding_file}               (optional image embeddings)

    Args:
        data_root:       Root directory (e.g. 'NSD/data/mindeye_nsd')
        split:           'train' or 'test'
        mode:            'averaged' or 'single_trial'
        subject:         Subject id, e.g. 'subj01'
        embedding_file:  Filename inside {data_root}/embeddings/ (optional)
        normalize_fmri:  Whether to z-score normalize fMRI data
        roi_file:        ROI atlas name (currently unused, reserved for future)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        mode: str = "averaged",
        subject: str = "subj01",
        embedding_file: Optional[str] = None,
        normalize_fmri: bool = False,
        roi_file: Optional[str] = None,
        images_path: Optional[str] = None,
        indices_path: Optional[str] = None,
        transform = None,
        features_path: Optional[str] = None,
        latents_path: Optional[str] = None,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.subject = subject
        self.images_path = images_path
        self.indices_path = indices_path
        self.transform = transform

        # --- Load fMRI data ---
        suffix = "avg" if mode == "averaged" else "single"
        fmri_path = self.data_root / "processed" / f"{subject}_{split}_{suffix}.npy"
        if not fmri_path.exists():
            raise FileNotFoundError(f"fMRI file not found: {fmri_path}")

        self.fmri = np.load(str(fmri_path), mmap_mode='r')  # [N, V] (Memory mapped)
        print(f"NSDDataset: loaded {fmri_path.name} (mmap) shape={self.fmri.shape}")

        if normalize_fmri:
            mean = self.fmri.mean(axis=0, keepdims=True)
            std = self.fmri.std(axis=0, keepdims=True) + 1e-8
            self.fmri = (self.fmri - mean) / std
            print(f"  fMRI z-score normalized")

        # --- Optionally load embeddings ---
        self.embeddings = None
        if embedding_file is not None:
            emb_path = self.data_root / "embeddings" / embedding_file
            if emb_path.exists():
                self.embeddings = np.load(str(emb_path))
                print(f"  Embeddings loaded: {emb_path.name}  shape={self.embeddings.shape}")
            else:
                print(f"  Warning: embedding file not found at {emb_path}, skipping.")

        # --- Optionally load DINOv2 patch features from HDF5 ---
        self.features_path = features_path
        self.trial_to_feature_idx = None
        if features_path is not None:
            fp = Path(features_path)
            if fp.exists():
                with h5py.File(str(fp), 'r') as f:
                    self.trial_to_feature_idx = f['trial_to_feature_idx'][:]
                    feat_shape = f['features'].shape
                print(f"  DINOv2 features: {fp.name}  shape={feat_shape}")
                print(f"  trial_to_feature_idx loaded: {self.trial_to_feature_idx.shape}")
            else:
                print(f"  Warning: features file not found at {fp}, skipping.")
                self.features_path = None

        # Store roi_file for potential future use
        self.roi_file = roi_file

        # --- Optionally load pre-extracted VAE latents (mu) ---
        self.vae_latents = None
        if latents_path is not None:
            lp = Path(latents_path)
            if lp.exists():
                # Load fully into RAM for fast access (cast float16→float32)
                raw = np.load(str(lp))
                self.vae_latents = raw.astype(np.float32) if raw.dtype == np.float16 else raw
                print(f"  VAE latents loaded into RAM: {lp.name}  shape={self.vae_latents.shape}  "
                      f"dtype={self.vae_latents.dtype}  ({self.vae_latents.nbytes / 1024**3:.1f} GB)")
                assert len(self.vae_latents) == len(self.fmri), (
                    f"Latents/fMRI length mismatch: {len(self.vae_latents)} vs {len(self.fmri)}"
                )
            else:
                print(f"  Warning: latents file not found at {lp}, will encode on-the-fly")

    def __len__(self) -> int:
        return self.fmri.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fmri = torch.from_numpy(self.fmri[idx].copy()).float()

        out = {
            "fmri": fmri,
            "trial_idx": torch.tensor(idx).long(),
        }

        # Pre-extracted VAE latents
        if self.vae_latents is not None:
            out["vae_mu"] = torch.from_numpy(self.vae_latents[idx].copy()).float()

        # Load image if paths are provided
        if self.images_path and self.indices_path:
            try:
                # Map trial index -> image index
                with h5py.File(self.indices_path, 'r') as f_idx:
                    image_idx = int(f_idx[self.subject][idx])
                
                out["image_idx"] = torch.tensor(image_idx).long()

                # Load image
                with h5py.File(self.images_path, 'r') as f_img:
                    if 'images' in f_img:
                        dset = f_img['images']
                    else:
                        key = list(f_img.keys())[0]
                        dset = f_img[key]
                    
                    image_data = dset[image_idx] # [H, W, C]
                    image = torch.from_numpy(image_data).float()
                    
                    if image.shape[-1] == 3:
                        image = image.permute(2, 0, 1) # [3, 224, 224]
                    
                    # Normalize to [-1, 1] if needed (assuming [0, 1] float16 input)
                    if image.max() > 1.0:
                        image = image / 255.0
                    image = (image - 0.5) / 0.5

                if self.transform:
                    image = self.transform(image)
                
                out["image"] = image
            except Exception as e:
                print(f"Error loading image for index {idx}: {e}")

        if self.embeddings is not None:
            # For averaged mode the mapping may differ; use idx directly
            emb_idx = idx if idx < len(self.embeddings) else idx % len(self.embeddings)
            out["embedding"] = torch.from_numpy(self.embeddings[emb_idx]).float()

        # Load DINOv2 patch features (lazy from HDF5)
        if self.features_path is not None and self.trial_to_feature_idx is not None:
            feat_idx = int(self.trial_to_feature_idx[idx])
            with h5py.File(self.features_path, 'r') as f:
                features = f['features'][feat_idx]  # [3, 256, 1024] float16
            out["dino_features"] = torch.from_numpy(features.astype(np.float32))

        return out
