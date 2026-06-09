"""
extract_depth.py
================
Monocular **depth** features from DINOv2's pretrained depth head — the dorsal /
parietal partner to the (chromatic) Color and (orientation) Gabor streams.

Motivation (per src/analyze_roi.py): the worst-captured ROI is **parietal**
(captured% ~74.8%), a dorsal-stream region that codes spatial layout / 3-D
structure — a cue absent from every current stream (CLIP/DINO = semantic, Gabor
= grayscale orientation, Color = chroma). DINOv2 ships a depth head (DPT or
linear, trained on NYUd/KITTI) that reuses the SAME backbone we already run, so
depth comes "for free" without a separate MiDaS/DepthAnything model.

The hub depther (``dinov2_vitg14_dd`` DPT / ``dinov2_vitg14_ld`` linear) is the
*mmcv-free* path (``dinov2/hub/depth`` does not import mmcv/mmseg, unlike
``dinov2/eval/depth``). ``model.forward_dummy(img)`` returns a clamped depth map
at the input resolution.

Pipeline (per subject, train + test stimuli aligned to the fMRI axis):
  RGB → ImageNet-norm, resize to a ×14 grid → DINOv2 depther → dense depth map →
  per-image robust [0,1] normalisation (2nd/98th pct) → per-grid-cell statistics.

Per grid cell (3 channels): mean relative depth, std depth (within-cell 3-D
complexity), mean |∇depth| (depth-gradient = occlusion / figure-ground edges).
Per-image normalisation keeps *relative* layout (robust to NYU→NSD domain shift;
absolute metric scale from NYU is unreliable on natural scenes).

Output: nsd_depth_{mode}_sub{S}.npy of shape (N, G*G, 3) float16 — a token
sequence for cross-attention, dropped into ``data.context_features`` like Gabor.

Usage::
    python src/data/extract_depth.py --subjects 1
    python src/data/extract_depth.py --subjects 1 2 5 7 --share_test_from 1 --head dpt
"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_ARCHES = ("vitg14", "vitl14", "vitb14", "vits14")


def load_depther(arch: str, head: str, weights: str, repo: str, device: str):
    """Load a DINOv2 depther (mmcv-free hub path). head ∈ {dpt, linear}.

    Hub entrypoints are named by the short arch, e.g. ``dinov2_vitg14_dd`` (DPT)
    / ``dinov2_vitg14_ld`` (linear).
    """
    suffix = "dd" if head == "dpt" else "ld"
    entry = f"dinov2_{arch}_{suffix}"
    source = "local" if os.path.isdir(repo) else "github"
    repo_or_dir = repo if source == "local" else "facebookresearch/dinov2"
    print(f"Loading {entry} (weights={weights}) from {repo_or_dir} [{source}] ...")

    # DINOv2's hub loader forces torch.load(weights_only=True), which rejects the
    # numpy scalars stored in the official depth-head checkpoints under PyTorch
    # ≥2.6. These are trusted Meta weights → load with weights_only=False.
    _orig_load = torch.load
    def _trusted_load(*a, **k):
        k["weights_only"] = False
        return _orig_load(*a, **k)
    torch.load = _trusted_load
    try:
        model = torch.hub.load(repo_or_dir, entry, source=source, weights=weights)
    finally:
        torch.load = _orig_load
    return model.to(device).eval()


@torch.no_grad()
def extract(stim, model, grid, img_size, device, batch=8):
    """stim: (N,H,W,3) uint8 → (N, grid*grid, 3) float16 depth statistics."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = np.empty((stim.shape[0], grid * grid, 3), dtype=np.float16)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
        if device.startswith("cuda") else torch.autocast(device_type="cpu", enabled=False)

    for i in tqdm(range(0, stim.shape[0], batch), desc="depth", unit="batch", dynamic_ncols=True):
        chunk = torch.from_numpy(np.ascontiguousarray(stim[i:i + batch])).to(device)
        x = chunk.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with autocast:
            depth = model.forward_dummy(x).float()             # (B,1,H,W), metric, clamped

        # Per-image robust [0,1] normalisation (2nd/98th percentile) → relative depth
        B = depth.shape[0]
        flat = depth.reshape(B, -1)
        q = torch.quantile(flat, torch.tensor([0.02, 0.98], device=device), dim=1)  # (2,B)
        lo, hi = q[0].view(B, 1, 1, 1), q[1].view(B, 1, 1, 1)
        d = ((depth - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)   # (B,1,H,W) relative

        # Depth-gradient magnitude (occlusion / figure-ground edges)
        gx = F.pad(d[:, :, :, 1:] - d[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(d[:, :, 1:, :] - d[:, :, :-1, :], (0, 0, 0, 1))
        grad = (gx * gx + gy * gy).sqrt()

        m = F.adaptive_avg_pool2d(d, (grid, grid))             # mean rel depth
        msq = F.adaptive_avg_pool2d(d * d, (grid, grid))
        sd = (msq - m * m).clamp_min(0.0).sqrt()               # depth std
        g = F.adaptive_avg_pool2d(grad, (grid, grid))          # mean |∇depth|

        feat = torch.cat([m, sd, g], dim=1)                    # (B,3,grid,grid)
        tok = feat.flatten(2).transpose(1, 2).contiguous()     # (B, grid*grid, 3)
        out[i:i + batch] = tok.cpu().half().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DINOv2 monocular depth feature extraction")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--data_root", type=str, default="NSD/data/nsd")
    ap.add_argument("--subdir", type=str, default="depth",
                    help="output sub-folder under subj0{S}/ (matches data.subdirs)")
    ap.add_argument("--arch", type=str, default="vitg14", choices=list(_ARCHES),
                    help="DINOv2 backbone (reuses the cached vitg14 by default)")
    ap.add_argument("--head", type=str, default="dpt", choices=["dpt", "linear"],
                    help="depth head: dpt (sharper) or linear (lighter)")
    ap.add_argument("--weights", type=str, default="NYU", choices=["NYU", "KITTI"],
                    help="NYU = indoor (denser near-field); KITTI = driving/outdoor")
    ap.add_argument("--repo", type=str, default="NSD/notes/dinov2",
                    help="local DINOv2 repo (falls back to github hub if absent)")
    ap.add_argument("--grid", type=int, default=8, help="output spatial grid (tokens = grid²)")
    ap.add_argument("--img", type=int, default=448, help="model input size (multiple of 14)")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--share_test_from", type=int, default=None,
                    help="Reuse test features from this subject (NSD test stimuli are "
                         "identical across subjects); others copy its test file instead.")
    args = ap.parse_args()

    if args.img % 14 != 0:
        print(f"  ⚠️  --img {args.img} is not a multiple of 14; the backbone will "
              f"centre-pad it. Consider 448 (=14×32) or 392 (=14×28).")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_depther(args.arch, args.head, args.weights, args.repo, device)
    print(f"Depth features: arch={args.arch} head={args.head} weights={args.weights} "
          f"img={args.img} grid={args.grid} → tokens=({args.grid**2}, 3) "
          f"[mean_rel_depth, std, grad]  device={device}")

    subjects = list(args.subjects)
    if args.share_test_from is not None and args.share_test_from in subjects:
        subjects.remove(args.share_test_from)
        subjects.insert(0, args.share_test_from)

    for sub in subjects:
        subj_dir = os.path.join(args.data_root, f"subj{sub:02d}")
        out_dir = os.path.join(subj_dir, args.subdir) if args.subdir else subj_dir
        os.makedirs(out_dir, exist_ok=True)
        for mode in ("train", "test"):
            out_path = os.path.join(out_dir, f"nsd_depth_{mode}_sub{sub}.npy")

            if (mode == "test" and args.share_test_from is not None
                    and sub != args.share_test_from):
                ref = args.share_test_from
                ref_dir = os.path.join(args.data_root, f"subj{ref:02d}", args.subdir) \
                    if args.subdir else os.path.join(args.data_root, f"subj{ref:02d}")
                src = os.path.join(ref_dir, f"nsd_depth_test_sub{ref}.npy")
                if not os.path.exists(src):
                    print(f"  ⚠️  ref test file missing, skipped: {src}"); continue
                shutil.copyfile(src, out_path)
                print(f"  subj{sub:02d} test: 📋 copied from subj{ref:02d} → {out_path}")
                continue

            stim_path = os.path.join(subj_dir, f"nsd_{mode}_stim_sub{sub}.npy")
            if not os.path.exists(stim_path):
                print(f"  [skip] {stim_path} not found"); continue
            stim = np.load(stim_path, mmap_mode="r")           # (N,425,425,3) uint8
            feats = extract(stim, model, args.grid, args.img, device, args.batch)
            np.save(out_path, feats)
            print(f"  subj{sub:02d} {mode}: {stim.shape[0]} imgs → {feats.shape} {feats.dtype}  "
                  f"[mean={feats.astype(np.float32).mean():.3f}] → {out_path}")


if __name__ == "__main__":
    main()
