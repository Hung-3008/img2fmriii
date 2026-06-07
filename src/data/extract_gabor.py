"""
extract_gabor.py
================
Low-level Gabor-energy features for the early visual cortex (V1-V4).

The FactFlow encoder plateaus at ~0.37 voxel_r because CLIP and DINOv2-last-layer
are both *high-level / semantic* — there is no good *low-level* source for early
visual cortex, which is retinotopic and orientation/spatial-frequency tuned.
Gabor energy (a classic V1 model) fills that gap.

Pipeline (per subject, train + test stimuli already aligned to the fMRI axis):
  RGB image → grayscale → resize → bank of complex Gabor filters
  (n_orient × n_freq) → magnitude (phase-invariant, complex-cell-like) energy
  maps → average-pool to a G×G retinotopic grid → tokens.

Output: nsd_gabor_{mode}_sub{S}.npy of shape (N, G*G, n_orient*n_freq) float16,
a token sequence ready for cross-attention (token = grid location, channels =
orientation×frequency energy).

Usage::
    python src/data/extract_gabor.py --subjects 1
    python src/data/extract_gabor.py --subjects 1 2 5 7 --grid 16 --img 160
"""

import argparse
import math
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F


def gabor_bank(orientations, wavelengths, ksize, gamma=0.5, device="cpu"):
    """Return (even, odd) conv weights of shape (n_orient*n_freq, 1, k, k).

    even = cos (symmetric) phase, odd = sin (antisymmetric) phase; the per-filter
    magnitude sqrt(even² + odd²) is the phase-invariant Gabor energy.
    """
    half = ksize // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    even, odd = [], []
    for lam in wavelengths:
        sigma = 0.56 * lam
        for th in orientations:
            xr = xs * math.cos(th) + ys * math.sin(th)
            yr = -xs * math.sin(th) + ys * math.cos(th)
            envelope = torch.exp(-(xr ** 2 + (gamma ** 2) * yr ** 2) / (2 * sigma ** 2))
            carrier_c = torch.cos(2 * math.pi * xr / lam)
            carrier_s = torch.sin(2 * math.pi * xr / lam)
            ec = envelope * carrier_c
            es = envelope * carrier_s
            # zero-mean the even filter (remove DC response)
            ec = ec - ec.mean()
            even.append(ec)
            odd.append(es)
    even = torch.stack(even).unsqueeze(1).to(device)   # (F,1,k,k)
    odd = torch.stack(odd).unsqueeze(1).to(device)
    return even, odd


@torch.no_grad()
def extract(stim, even, odd, ksize, grid, img_size, device, batch=64):
    """stim: (N, H, W, 3) uint8 → (N, grid*grid, n_filters) float16 Gabor energy."""
    n_filters = even.shape[0]
    gray_w = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    out = np.empty((stim.shape[0], grid * grid, n_filters), dtype=np.float16)
    pad = ksize // 2
    for i in range(0, stim.shape[0], batch):
        chunk = torch.from_numpy(np.ascontiguousarray(stim[i:i + batch])).to(device)
        x = chunk.permute(0, 3, 1, 2).float() / 255.0          # (B,3,H,W)
        x = (x * gray_w).sum(1, keepdim=True)                  # grayscale (B,1,H,W)
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        e = F.conv2d(x, even, padding=pad)                     # (B,F,H,W)
        o = F.conv2d(x, odd, padding=pad)
        energy = torch.sqrt(e * e + o * o + 1e-12)             # phase-invariant
        energy = F.adaptive_avg_pool2d(energy, (grid, grid))   # (B,F,grid,grid)
        tok = energy.flatten(2).transpose(1, 2).contiguous()   # (B, grid*grid, F)
        out[i:i + batch] = tok.cpu().half().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Gabor-energy feature extraction")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--data_root", type=str, default="NSD/data/nsd")
    ap.add_argument("--orient", type=int, default=8, help="number of orientations")
    ap.add_argument("--wavelengths", type=float, nargs="+", default=[3.0, 6.0, 12.0],
                    help="Gabor wavelengths in px (on the resized image)")
    ap.add_argument("--img", type=int, default=160, help="resized image size")
    ap.add_argument("--grid", type=int, default=16, help="output spatial grid (tokens = grid²)")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--share_test_from", type=int, default=None,
                    help="Reuse test features from this subject (NSD test stimuli are "
                         "identical across subjects); the others copy its test file "
                         "instead of recomputing. The reference subject is processed first.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    orientations = [k * math.pi / args.orient for k in range(args.orient)]
    sigma_max = 0.56 * max(args.wavelengths)
    ksize = 2 * int(round(3 * sigma_max)) + 1
    even, odd = gabor_bank(orientations, args.wavelengths, ksize, device=device)
    n_filters = even.shape[0]
    print(f"Gabor bank: {args.orient} orient × {len(args.wavelengths)} freq = {n_filters} filters, "
          f"ksize={ksize}, img={args.img}, grid={args.grid} → tokens=({args.grid**2},{n_filters})")

    # Process the reference subject first so its test feature exists before copying.
    subjects = list(args.subjects)
    if args.share_test_from is not None and args.share_test_from in subjects:
        subjects.remove(args.share_test_from)
        subjects.insert(0, args.share_test_from)

    for sub in subjects:
        subj_dir = os.path.join(args.data_root, f"subj{sub:02d}")
        for mode in ("train", "test"):
            out_path = os.path.join(subj_dir, f"nsd_gabor_{mode}_sub{sub}.npy")

            # Reuse the reference subject's test feature (shared NSD test set).
            if (mode == "test" and args.share_test_from is not None
                    and sub != args.share_test_from):
                ref = args.share_test_from
                src = os.path.join(args.data_root, f"subj{ref:02d}", f"nsd_gabor_test_sub{ref}.npy")
                if not os.path.exists(src):
                    print(f"  ⚠️  ref test file missing, skipped: {src}"); continue
                shutil.copyfile(src, out_path)
                print(f"  subj{sub:02d} test: 📋 copied from subj{ref:02d} → {out_path}")
                continue

            stim_path = os.path.join(subj_dir, f"nsd_{mode}_stim_sub{sub}.npy")
            if not os.path.exists(stim_path):
                print(f"  [skip] {stim_path} not found"); continue
            stim = np.load(stim_path, mmap_mode="r")           # (N,425,425,3) uint8
            feats = extract(stim, even, odd, ksize, args.grid, args.img, device, args.batch)
            np.save(out_path, feats)
            print(f"  subj{sub:02d} {mode}: {stim.shape[0]} imgs → {feats.shape} {feats.dtype}  "
                  f"[mean={feats.astype(np.float32).mean():.3f}] → {out_path}")


if __name__ == "__main__":
    main()
