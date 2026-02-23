"""
extract_features_dinov2_multilayer.py
=====================================
Extracts multi-layer DINOv2 ViT-B/14 features from PNG images.

Uses `model.get_intermediate_layers()` to extract tokens from
layers [3, 6, 9, 12] of DINOv2 ViT-B/14 (12 layers total).

Output shape: (N, 4, 257, 768)  float16
  - 4 layers × (1 CLS + 256 patches) × 768-dim

Storage: ~14 GB per train split, ~1.5 GB per test split.

Usage:
  python src/data/extract_features_dinov2_multilayer.py
  python src/data/extract_features_dinov2_multilayer.py --subjects 1
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Config ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../Data'))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'nsd')

DEFAULT_SUBJECTS = [1, 2, 5, 7]
DEFAULT_BATCH_SIZE = 128  # ViT-B is smaller, can use larger batch
DINOV2_EMB_DIM = 768
DINOV2_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Layers to extract (0-indexed block indices for DINOv2 API)
# ViT-B/14 has 12 blocks (0-11). These correspond to layers 3,6,9,12 (1-indexed)
EXTRACT_LAYERS = [2, 5, 8, 11]


class DINOv2_Image_Dataset(Dataset):
    def __init__(self, image_paths, transform):
        self.img_data = image_paths
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.img_data[idx]).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.img_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multi-layer DINOv2 ViT-B/14 features")
    parser.add_argument('--subjects', type=int, nargs='+',
                        default=DEFAULT_SUBJECTS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--layers', type=int, nargs='+',
                        default=EXTRACT_LAYERS,
                        help='Layer indices to extract (1-indexed)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    layers = args.layers
    n_layers = len(layers)

    print(f"Device: {device}")
    print(f"Extracting layers: {layers}")

    torch.backends.cuda.matmul.allow_tf32 = True

    # ── Load DINOv2 ──
    print("Loading DINOv2 ViT-B/14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    model = model.to(device).eval()
    print(f"DINOv2 loaded. {n_layers} layers to extract.")

    transform = transforms.Compose([
        transforms.Resize(DINOV2_INPUT_SIZE,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(DINOV2_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    for subj in args.subjects:
        save_path = os.path.join(PROCESSED_DATA_DIR, f'subj0{subj}')
        if not os.path.exists(save_path):
            print(f"\nSubject {subj} not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Subject {subj}...")
        print(f"{'='*60}")

        for mode in ['train', 'test']:
            image_dir = os.path.join(save_path, f'{mode}_img')
            if not os.path.exists(image_dir):
                print(f"  {mode}_img not found, skipping...")
                continue

            num_images = len(
                [f for f in os.listdir(image_dir) if f.endswith('.png')])
            image_paths = np.array([
                os.path.join(image_dir, f'{i}.png')
                for i in range(num_images)
            ])
            print(f'  {mode.capitalize()} images: {num_images}')

            dataset = DINOv2_Image_Dataset(image_paths, transform)
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size,
                drop_last=False, num_workers=4, pin_memory=True)

            all_features = []

            for batch_i, images in enumerate(dataloader):
                if batch_i % 10 == 0:
                    print(f"  Batch {batch_i}/{len(dataloader)} / {mode}...")

                with torch.no_grad(), torch.amp.autocast(
                        'cuda', enabled=(device == 'cuda')):
                    # get_intermediate_layers returns list of (B, N_patches, D)
                    # n= specifies which layers (1-indexed)
                    outputs = model.get_intermediate_layers(
                        images.to(device), n=layers,
                        reshape=False, return_class_token=True)
                    # outputs: list of tuples (patch_tokens, cls_token)
                    # patch_tokens: (B, 256, 768)
                    # cls_token: (B, 768)

                    layer_features = []
                    for patch_tokens, cls_token in outputs:
                        # Concat CLS + patches → (B, 257, 768)
                        tokens = torch.cat([
                            cls_token.unsqueeze(1),
                            patch_tokens
                        ], dim=1)
                        layer_features.append(tokens)

                    # Stack layers → (B, n_layers, 257, 768)
                    stacked = torch.stack(layer_features, dim=1)

                all_features.append(stacked.cpu().half())
                torch.cuda.empty_cache()

            if all_features:
                all_features = torch.cat(all_features, dim=0).numpy()
                out_path = os.path.join(
                    save_path,
                    f'nsd_dinov2_vitb14_multilayer_{mode}_sub{subj}.npy')
                np.save(out_path, all_features)
                print(f"  ✅ Saved: {out_path}")
                print(f"     Shape: {all_features.shape}, "
                      f"Dtype: {all_features.dtype}")
                print(f"     Size: {all_features.nbytes / 1e9:.1f} GB")
                del all_features

    print(f"\n{'='*60}")
    print("Done. Multi-layer DINOv2 extraction completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
