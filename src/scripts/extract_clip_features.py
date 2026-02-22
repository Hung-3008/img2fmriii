"""
Extract CLIP ViT-L/14 image features for Stage 2 conditioning.

Extracts features from the 10,000 NSD images (same order as DINOv2 features)
using OpenAI CLIP ViT-L/14. CLIP was trained on 400M image-text pairs (including COCO),
providing semantically richer and more diverse features than DINOv2 CLS tokens.

Output:
    - features: (10000, 257, 1024) float16 — CLS token + 256 patch tokens
    - Uses internal 1024-dim representation (NOT the projected 768-dim CLIP embedding)
    - Format matches DINOv2 feature structure for drop-in replacement

Usage:
    python -m src.scripts.extract_clip_features
    python -m src.scripts.extract_clip_features --model ViT-L-14 --pretrained openai
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import open_clip


# ─── Dataset for COCO images from HDF5 ───────────────────────────────────────

class COCOImagesHDF5Dataset(Dataset):
    """Load COCO images from HDF5 file for specific image indices."""

    def __init__(self, hdf5_path: str, image_indices: np.ndarray, transform=None):
        self.hdf5_path = hdf5_path
        self.image_indices = image_indices
        self.transform = transform
        self._hdf5 = None

    def _get_hdf5(self):
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.hdf5_path, "r")
        return self._hdf5

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        f = self._get_hdf5()
        coco_idx = int(self.image_indices[idx])

        # Image: (3, 224, 224) float16 → PIL Image → CLIP transform
        img = f["images"][coco_idx].astype(np.float32)  # [3, 224, 224] in [0, 1]
        img_uint8 = (img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        from PIL import Image
        pil_img = Image.fromarray(img_uint8)

        if self.transform is not None:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = torch.from_numpy(img)

        return img_tensor, idx


# ─── Feature Extraction using forward_intermediates ──────────────────────────

@torch.no_grad()
def extract_features(model, dataloader, device, n_images, hidden_dim, n_tokens):
    """Extract CLIP vision features using forward_intermediates API."""

    all_features = np.zeros((n_images, n_tokens, hidden_dim), dtype=np.float16)

    for batch_imgs, batch_indices in tqdm(dataloader, desc="Extracting CLIP features"):
        batch_imgs = batch_imgs.to(device)

        # Use forward_intermediates to get pre-projection features
        # indices=[-1] → last transformer layer output
        # output_extra_tokens=True → returns CLS token separately
        result = model.visual.forward_intermediates(
            batch_imgs,
            indices=[-1],           # last layer
            output_fmt='NLC',       # (batch, seq_len, channels)
            output_extra_tokens=True,
        )

        # CLS token: [B, 1, hidden_dim]
        cls_tokens = result['image_intermediates_prefix'][0]  # first (and only) layer

        # Patch tokens: [B, n_patches, hidden_dim]
        patch_tokens = result['image_intermediates'][0]  # first (and only) layer

        # Concatenate: [B, 1+n_patches, hidden_dim]
        features = torch.cat([cls_tokens, patch_tokens], dim=1)
        features = features.cpu().float().numpy().astype(np.float16)

        for i, global_idx in enumerate(batch_indices):
            all_features[global_idx] = features[i]

    return all_features


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP features from NSD/COCO images")
    parser.add_argument("--model", type=str, default="ViT-L-14",
                        help="CLIP model name (default: ViT-L-14)")
    parser.add_argument("--pretrained", type=str, default="openai",
                        help="Pretrained weights (default: openai)")
    parser.add_argument("--coco_hdf5", type=str,
                        default="NSD/data/coco_images_224_float16.hdf5")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--output_dir", type=str, default="NSD/data/features")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Model: {args.model} / {args.pretrained}")

    # ── Load CLIP model ──
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device).eval()

    # Get model info
    vision = model.visual
    hidden_dim = vision.ln_pre.weight.shape[0]  # internal hidden dim (1024 for ViT-L)
    patch_size = vision.patch_size
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    grid_size = 224 // patch_size
    n_patches = grid_size * grid_size
    n_tokens = 1 + n_patches  # CLS + patches

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Internal hidden dim: {hidden_dim}")
    print(f"  Output dim (proj): {vision.output_dim}")
    print(f"  Patch size: {patch_size}, Grid: {grid_size}x{grid_size} = {n_patches} patches")
    print(f"  Tokens: {n_tokens} (CLS + {n_patches} patches)")
    print(f"  Total params: {total_params:,}")

    # ── Determine which images to extract ──
    print(f"\nDetermining image indices for {args.subject}...")
    data_dir = Path("NSD/data/mindeye_nsd/processed")
    train_imgids = np.load(data_dir / f"{args.subject}_train_avg_imgids.npy")
    test_imgids = np.load(data_dir / f"{args.subject}_test_avg_imgids.npy")
    all_imgids = np.sort(np.concatenate([train_imgids, test_imgids]))
    print(f"  Total unique images: {len(all_imgids)}")

    # Verify against DINOv2 reference
    dino_ref = Path("NSD/data/features/dinov2_vitl14_combined_L4_8_12_subj01.hdf5")
    if dino_ref.exists():
        with h5py.File(dino_ref, "r") as f:
            if "image_indices" in f:
                if np.array_equal(all_imgids, f["image_indices"][:]):
                    print("  ✅ Image indices match DINOv2 reference")

    # ── Create dataset ──
    dataset = COCOImagesHDF5Dataset(args.coco_hdf5, all_imgids, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"\n  Dataset: {len(dataset)} images")
    print(f"  Expected output: ({len(dataset)}, {n_tokens}, {hidden_dim}) float16")

    # ── Extract features ──
    print("\nExtracting features...")
    features = extract_features(model, dataloader, device, len(dataset), hidden_dim, n_tokens)
    print(f"  Features shape: {features.shape}, dtype: {features.dtype}")

    # ── Save ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("-", "").replace("/", "_").lower()
    output_name = f"clip_{model_tag}_{args.pretrained}_{args.subject}"

    npy_path = output_dir / f"{output_name}.npy"
    np.save(npy_path, features)
    print(f"  Saved NPY: {npy_path} ({features.nbytes / 1024**2:.1f} MB)")

    hdf5_path = output_dir / f"{output_name}.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("features", data=features, compression="gzip", compression_opts=4)
        f.create_dataset("image_indices", data=all_imgids)
        f.attrs["model"] = args.model
        f.attrs["pretrained"] = args.pretrained
        f.attrs["hidden_dim"] = hidden_dim
        f.attrs["n_tokens"] = n_tokens
    print(f"  Saved HDF5: {hdf5_path}")

    # ── Quality Check ──
    print("\n── Quality Check ──")
    cls_tokens = features[:, 0, :].astype(np.float32)  # [10000, hidden_dim]
    print(f"  CLS token shape: {cls_tokens.shape}")

    norms = np.linalg.norm(cls_tokens, axis=1)
    print(f"  L2 norm: {norms.mean():.4f} ± {norms.std():.4f}")
    print(f"  Std (per-dim): {cls_tokens.std(axis=0).mean():.6f}")

    # Pairwise cosine similarity
    from scipy.spatial.distance import pdist
    N = min(2000, len(cls_tokens))
    subset = cls_tokens[:N]
    normed = subset / (np.linalg.norm(subset, axis=1, keepdims=True) + 1e-8)
    cos_sims = 1.0 - pdist(normed, metric="cosine")
    print(f"  Cosine sim (mean): {cos_sims.mean():.4f} (DINOv2 L12 was 0.968)")
    print(f"  Cosine sim (std):  {cos_sims.std():.4f}")
    print(f"  Cosine sim (min):  {cos_sims.min():.4f}")

    # PCA for effective dimensionality
    centered = cls_tokens[:N] - cls_tokens[:N].mean(axis=0)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    explained = (S**2) / (S**2).sum()
    cumvar = np.cumsum(explained)
    n90 = np.searchsorted(cumvar, 0.90) + 1
    n95 = np.searchsorted(cumvar, 0.95) + 1
    print(f"  PCA dims for 90%: {n90}/{hidden_dim}")
    print(f"  PCA dims for 95%: {n95}/{hidden_dim}")

    print(f"\n✅ Done! To use in training, update config:")
    print(f"    dino_path: \"{npy_path}\"")
    print(f"    cond_dim: {hidden_dim}")


if __name__ == "__main__":
    main()
