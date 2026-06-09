"""
fit_ridge_encoding.py
=====================
Per-voxel **kernel ridge** encoding baseline on the rich stimulus features —
the decisive test of *routing vs stimulus-ceiling* for the ~0.39 plateau.

FactFlow shares one DiT decoder across all voxels (a flat, global cross-attn
conditioning). The per-ROI gate showed added features (color/depth) are NOT
routed to their functional voxels. A per-voxel ridge gives **each voxel its own
weights over the features** — exactly the routing FactFlow lacks. So:

  * ridge voxel_r  >  0.39  ⇒  FactFlow UNDER-USES the features per-voxel
                              → the lever is a per-voxel / RF readout (routing).
  * ridge voxel_r  ≈  0.39  ⇒  stimulus / data ceiling
                              → no feature or readout helps; need more data.

Dual (kernel) ridge so arbitrarily high-dim spatial features are tractable
(only the N×N Gram is formed):
    α = (K + λI)⁻¹ Y_train,   Ŷ = K_test α          K = Σ_streams Xₛ Xₛᵀ
Each stream is z-scored per-dim and scaled by 1/√Dₛ so streams of very different
dimensionality contribute comparably (multi-kernel average). λ is picked on a
held-out split of the train images.

Usage::
    python src/fit_ridge_encoding.py --subject 1
    python src/fit_ridge_encoding.py --subject 1 --streams gabor color depth
"""

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from utils.metrics import voxel_pearson

# stream name → (subdir, file_prefix). filename = nsd_{prefix}_{mode}_sub{S}.npy
STREAMS = {
    "clip":   ("clip",  "sdxl_clip"),
    "clippool": ("clip", "sdxl_clip_pool"),
    "dino":   ("dino",  "dinov2_vitg14_multilayer4p"),
    "gabor":  ("gabor", "gabor"),
    "color":  ("color", "color"),
    "depth":  ("depth", "depth"),
}


def _path(base, subdir, prefix, mode, sub):
    return os.path.join(base, f"subj{sub:02d}", subdir, f"nsd_{prefix}_{mode}_sub{sub}.npy")


@torch.no_grad()
def stream_gram(path_tr, path_te, dev, chunk=20000):
    """Return (K_tr, K_te) contributions of one stream: z-scored, /√D features.

    K_tr = X_tr X_trᵀ (Ntr,Ntr); K_te = X_te X_trᵀ (Nte,Ntr). Accumulated over
    feature-dim chunks so the (N, D) matrix is never fully materialised.
    """
    Xtr = np.load(path_tr, mmap_mode="r"); Xte = np.load(path_te, mmap_mode="r")
    Ntr, Nte = Xtr.shape[0], Xte.shape[0]
    Dtr = int(np.prod(Xtr.shape[1:]))
    Xtr = Xtr.reshape(Ntr, Dtr); Xte = Xte.reshape(Nte, Dtr)

    # Pass 1: per-dim mean/std from train (chunked)
    mean = torch.zeros(Dtr, device=dev); sq = torch.zeros(Dtr, device=dev)
    for c in range(0, Dtr, chunk):
        x = torch.from_numpy(np.ascontiguousarray(Xtr[:, c:c + chunk])).to(dev).float()
        mean[c:c + chunk] = x.mean(0); sq[c:c + chunk] = (x * x).mean(0)
    std = (sq - mean * mean).clamp_min(1e-8).sqrt()
    scale = 1.0 / (std * (Dtr ** 0.5))                      # z-score then /√D

    # Pass 2: accumulate Gram
    Ktr = torch.zeros(Ntr, Ntr, device=dev); Kte = torch.zeros(Nte, Ntr, device=dev)
    for c in range(0, Dtr, chunk):
        sl = slice(c, c + chunk)
        xtr = (torch.from_numpy(np.ascontiguousarray(Xtr[:, sl])).to(dev).float() - mean[sl]) * scale[sl]
        xte = (torch.from_numpy(np.ascontiguousarray(Xte[:, sl])).to(dev).float() - mean[sl]) * scale[sl]
        Ktr += xtr @ xtr.t(); Kte += xte @ xtr.t()
        del xtr, xte
    return Ktr, Kte


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-voxel kernel-ridge encoding baseline")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--data_root", type=str, default="NSD/data/nsd")
    ap.add_argument("--fmri_mode", type=str, default="zscore")
    ap.add_argument("--n_voxels", type=int, default=15724)
    ap.add_argument("--streams", type=str, nargs="+",
                    default=["clip", "clippool", "dino", "gabor", "color", "depth"])
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[1e1, 1e2, 1e3, 1e4, 1e5])
    ap.add_argument("--val_n", type=int, default=1000, help="train images held out to pick λ")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sub, base = args.subject, args.data_root
    print(f"Kernel ridge encoding | subj {sub} | streams={args.streams}")

    # ── fMRI target: rep-mean, real voxels, anatomical order ──
    fdir = os.path.join(base, f"subj{sub:02d}", "fmri")
    ftr = np.load(os.path.join(fdir, f"nsd_train_fmri_{args.fmri_mode}_sub{sub}.npy"), mmap_mode="r")
    fte = np.load(os.path.join(fdir, f"nsd_test_fmri_{args.fmri_mode}_sub{sub}.npy"), mmap_mode="r")
    Ytr = torch.from_numpy(np.asarray(ftr).mean(1)[:, :args.n_voxels]).float().to(dev)  # (Ntr,V)
    Yte = torch.from_numpy(np.asarray(fte).mean(1)[:, :args.n_voxels]).float().to(dev)  # (Nte,V)
    Ntr = Ytr.shape[0]
    print(f"  fMRI: train {tuple(Ytr.shape)}  test {tuple(Yte.shape)}")

    # ── Build the summed Gram over streams ──
    Ktr = torch.zeros(Ntr, Ntr, device=dev); Kte = torch.zeros(Yte.shape[0], Ntr, device=dev)
    for s in args.streams:
        subdir, prefix = STREAMS[s]
        ptr = _path(base, subdir, prefix, "train", sub)
        pte = _path(base, subdir, prefix, "test", sub)
        kt, ke = stream_gram(ptr, pte, dev)
        Ktr += kt; Kte += ke
        print(f"  +stream {s:9s} diag(K_tr).mean={kt.diag().mean():.3f}")

    # ── λ selection on a held-out train split ──
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(Ntr, generator=g)
    val_idx, fit_idx = perm[:args.val_n].to(dev), perm[args.val_n:].to(dev)
    Kff = Ktr[fit_idx][:, fit_idx]; Kvf = Ktr[val_idx][:, fit_idx]
    Yf = Ytr[fit_idx]; Yf_mu = Yf.mean(0, keepdim=True); Yfc = Yf - Yf_mu
    eye = torch.eye(Kff.shape[0], device=dev)
    best = (-1.0, None)
    for lam in args.lambdas:
        alpha = torch.linalg.solve(Kff + lam * eye, Yfc)
        pv = Kvf @ alpha + Yf_mu
        r = voxel_pearson(pv, Ytr[val_idx]).mean().item()
        print(f"  λ={lam:>8.0f}  val voxel_r={r:.4f}")
        if r > best[0]:
            best = (r, lam)
    lam = best[1]
    print(f"  → best λ={lam:.0f} (val {best[0]:.4f}); refit on all {Ntr} train")

    # ── Refit on full train, evaluate on test ──
    Ymu = Ytr.mean(0, keepdim=True)
    alpha = torch.linalg.solve(Ktr + lam * torch.eye(Ntr, device=dev), Ytr - Ymu)
    pred = Kte @ alpha + Ymu
    vr = voxel_pearson(pred, Yte)                          # (V,)
    print(f"\n  TEST per-voxel r: mean={vr.mean().item():.4f}  median={vr.median().item():.4f}")

    # ── Per-ROI breakdown ──
    meta = np.load(os.path.join(base, f"subj{sub:02d}", f"roi_meta_sub{sub}.npz"), allow_pickle=True)
    lab = torch.from_numpy(meta["roi_labels"].astype(np.int64)).to(dev)
    names = [str(x) for x in meta["roi_names"]]
    vrc = vr.detach().cpu()
    print(f"  {'ROI':<13}{'nvox':>7}{'ridge_r':>9}")
    print("  " + "-" * 29)
    for rid, nm in zip(meta["roi_ids"].astype(int), names):
        m = (lab == int(rid)).cpu()
        if m.sum() == 0:
            continue
        print(f"  {nm:<13}{int(m.sum()):>7}{vrc[m].mean().item():>9.4f}")


if __name__ == "__main__":
    main()
