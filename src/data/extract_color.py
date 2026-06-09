"""
extract_color.py
================
Low-level **chromatic** features for early / ventral colour-selective cortex
(V1 blobs, V4, VO) — the orthogonal partner to the (grayscale) Gabor stream.

Motivation (per the per-ROI breakdown, src/analyze_roi.py): the encoder captures
only ~79% of the noise ceiling and the largest *recoverable* mass sits in early
visual cortex. CLIP/DINO are semantic and Gabor is computed on **grayscale**, so
there is no colour signal anywhere in the conditioning. Human early vision carries
two cone-opponent chromatic channels (L−M red/green, S−(L+M) blue/yellow); CIELAB
``a*``/``b*`` are their perceptual proxies. This stream supplies exactly that.

Pipeline (per subject, train + test stimuli already aligned to the fMRI axis):
  RGB image → sRGB→linear→XYZ→CIELAB (D65) → per-grid-cell statistics on a
  G×G retinotopic grid → tokens.

Per grid cell (7 channels): mean L*, mean a*, mean b*, std L*, std a*, std b*,
mean chroma √(a*²+b*²). a*/b* means already encode chroma-weighted hue; std
captures within-cell chromatic texture; chroma adds saturation that survives
spatial averaging. Values are normalised (L/100, a,b /110) to ~[-1,1].

Output: nsd_color_{mode}_sub{S}.npy of shape (N, G*G, 7) float16 — a token
sequence ready for cross-attention (token = grid location, channels = colour
statistics), dropped into ``data.context_features`` exactly like Gabor.

Usage::
    python src/data/extract_color.py --subjects 1
    python src/data/extract_color.py --subjects 1 2 5 7 --grid 8 --share_test_from 1
"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── sRGB (0..1) → CIELAB (D65) ───────────────────────────────────────────────

# Linear-sRGB → XYZ (D65), row-major
_RGB2XYZ = torch.tensor([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=torch.float32)
_XYZ_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)  # D65


def srgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) sRGB in [0,1] → (B,3,H,W) CIELAB (L in [0,100], a/b ~[-128,127])."""
    # sRGB → linear
    lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    # linear RGB → XYZ
    m = _RGB2XYZ.to(x.device, x.dtype)
    xyz = torch.einsum("ij,bjhw->bihw", m, lin)
    xyz = xyz / _XYZ_WHITE.to(x.device, x.dtype).view(1, 3, 1, 1)
    # XYZ → Lab
    delta = 6.0 / 29.0
    f = torch.where(xyz > delta ** 3, xyz.clamp_min(1e-8) ** (1.0 / 3.0),
                    xyz / (3 * delta ** 2) + 4.0 / 29.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=1)


@torch.no_grad()
def extract(stim, grid, device, batch=64, img_size=None):
    """stim: (N,H,W,3) uint8 → (N, grid*grid, 7) float16 colour statistics."""
    out = np.empty((stim.shape[0], grid * grid, 7), dtype=np.float16)
    norm = torch.tensor([100.0, 110.0, 110.0], device=device).view(1, 3, 1, 1)
    for i in tqdm(range(0, stim.shape[0], batch), desc="color", unit="batch", dynamic_ncols=True):
        chunk = torch.from_numpy(np.ascontiguousarray(stim[i:i + batch])).to(device)
        x = chunk.permute(0, 3, 1, 2).float() / 255.0          # (B,3,H,W) sRGB
        if img_size is not None:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        lab = srgb_to_lab(x) / norm                            # normalised L,a,b  (B,3,H,W)

        mean = F.adaptive_avg_pool2d(lab, (grid, grid))        # (B,3,grid,grid)
        mean_sq = F.adaptive_avg_pool2d(lab * lab, (grid, grid))
        std = (mean_sq - mean * mean).clamp_min(0.0).sqrt()    # (B,3,grid,grid)

        chroma = (lab[:, 1:3] ** 2).sum(1, keepdim=True).sqrt()  # √(a²+b²) per pixel
        chroma = F.adaptive_avg_pool2d(chroma, (grid, grid))     # (B,1,grid,grid)

        feat = torch.cat([mean, std, chroma], dim=1)           # (B,7,grid,grid)
        tok = feat.flatten(2).transpose(1, 2).contiguous()     # (B, grid*grid, 7)
        out[i:i + batch] = tok.cpu().half().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CIELAB chromatic feature extraction")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--data_root", type=str, default="NSD/data/nsd")
    ap.add_argument("--subdir", type=str, default="color",
                    help="output sub-folder under subj0{S}/ (matches data.subdirs)")
    ap.add_argument("--grid", type=int, default=8, help="output spatial grid (tokens = grid²)")
    ap.add_argument("--img", type=int, default=None,
                    help="optional resize before pooling (default: native resolution)")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--share_test_from", type=int, default=None,
                    help="Reuse test features from this subject (NSD test stimuli are "
                         "identical across subjects); others copy its test file instead "
                         "of recomputing. The reference subject is processed first.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Colour (CIELAB) features: grid={args.grid} → tokens=({args.grid**2}, 7) "
          f"[meanL,a,b, stdL,a,b, chroma]  device={device}")

    # Process the reference subject first so its test feature exists before copying.
    subjects = list(args.subjects)
    if args.share_test_from is not None and args.share_test_from in subjects:
        subjects.remove(args.share_test_from)
        subjects.insert(0, args.share_test_from)

    for sub in subjects:
        subj_dir = os.path.join(args.data_root, f"subj{sub:02d}")
        out_dir = os.path.join(subj_dir, args.subdir) if args.subdir else subj_dir
        os.makedirs(out_dir, exist_ok=True)
        for mode in ("train", "test"):
            out_path = os.path.join(out_dir, f"nsd_color_{mode}_sub{sub}.npy")

            # Reuse the reference subject's test feature (shared NSD test set).
            if (mode == "test" and args.share_test_from is not None
                    and sub != args.share_test_from):
                ref = args.share_test_from
                ref_dir = os.path.join(args.data_root, f"subj{ref:02d}", args.subdir) \
                    if args.subdir else os.path.join(args.data_root, f"subj{ref:02d}")
                src = os.path.join(ref_dir, f"nsd_color_test_sub{ref}.npy")
                if not os.path.exists(src):
                    print(f"  ⚠️  ref test file missing, skipped: {src}"); continue
                shutil.copyfile(src, out_path)
                print(f"  subj{sub:02d} test: 📋 copied from subj{ref:02d} → {out_path}")
                continue

            stim_path = os.path.join(subj_dir, f"nsd_{mode}_stim_sub{sub}.npy")
            if not os.path.exists(stim_path):
                print(f"  [skip] {stim_path} not found"); continue
            stim = np.load(stim_path, mmap_mode="r")           # (N,425,425,3) uint8
            feats = extract(stim, args.grid, device, args.batch, args.img)
            np.save(out_path, feats)
            print(f"  subj{sub:02d} {mode}: {stim.shape[0]} imgs → {feats.shape} {feats.dtype}  "
                  f"[mean={feats.astype(np.float32).mean():.3f}] → {out_path}")


if __name__ == "__main__":
    main()
