#!/usr/bin/env python3
"""Ridge regression baseline on the SAME few-shot data splits (memory-efficient).

Uses kernel (dual) ridge with stream-wise Gram accumulation to avoid
materializing the full (N, D) feature matrix. Processes one subject at a time.

Usage:
    .venv/bin/python src/utils/compute_ridge_fewshot_baseline.py
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
NSD = ROOT / "NSD" / "data" / "nsd"

SUBJ_MAP = {1: "subj01", 2: "subj02", 5: "subj05", 7: "subj07"}
TRIALS_PER_HOUR = 750

LOO_CONFIGS = [
    {"held": 1, "trunk": [2, 5, 7]},
    {"held": 2, "trunk": [1, 5, 7]},
    {"held": 5, "trunk": [1, 2, 7]},
    {"held": 7, "trunk": [1, 2, 5]},
]

STREAMS = {
    "clip_pool": ("clip", "sdxl_clip_pool"),
    "dino": ("dino", "dinov2_vitg14_multilayer4p"),
}


def get_fewshot_split(n_images, n_reps, hours, seed=42, n_val=250):
    """Reproduce the exact data split from factflow_fewshot_trainer.py."""
    rng = np.random.RandomState(seed)
    img_order = rng.permutation(n_images)
    n_val_img = min(n_val, n_images)
    adapt_images_all = img_order[n_val_img:].tolist()
    n_adapt_trials = int(round(hours * TRIALS_PER_HOUR))
    n_adapt_images = min(len(adapt_images_all),
                         int(np.ceil(n_adapt_trials / n_reps)))
    return adapt_images_all[:n_adapt_images]


@torch.no_grad()
def stream_gram(path_tr, path_te, adapt_idx, dev, chunk=10000):
    """Gram contribution from one stream, subsetted to adapt_idx for train."""
    Xtr_full = np.load(path_tr, mmap_mode="r")
    Xte_full = np.load(path_te, mmap_mode="r")

    # Subset train to adapt images
    Xtr = Xtr_full[adapt_idx]
    Ntr, Nte = len(adapt_idx), Xte_full.shape[0]
    D = int(np.prod(Xtr.shape[1:]))
    Xtr = Xtr.reshape(Ntr, D)
    Xte = Xte_full.reshape(Nte, D)

    # Per-dim mean/std from train (chunked)
    mean = torch.zeros(D, device=dev)
    sq = torch.zeros(D, device=dev)
    for c in range(0, D, chunk):
        x = torch.from_numpy(np.ascontiguousarray(Xtr[:, c:c+chunk])).to(dev).float()
        mean[c:c+chunk] = x.mean(0)
        sq[c:c+chunk] = (x * x).mean(0)
    std = (sq - mean * mean).clamp_min(1e-8).sqrt()
    scale = 1.0 / (std * (D ** 0.5))

    # Accumulate Gram
    Ktr = torch.zeros(Ntr, Ntr, device=dev)
    Kte = torch.zeros(Nte, Ntr, device=dev)
    for c in range(0, D, chunk):
        sl = slice(c, c + chunk)
        xtr = (torch.from_numpy(np.ascontiguousarray(Xtr[:, sl])).to(dev).float()
               - mean[sl]) * scale[sl]
        xte = (torch.from_numpy(np.ascontiguousarray(Xte[:, sl])).to(dev).float()
               - mean[sl]) * scale[sl]
        Ktr += xtr @ xtr.t()
        Kte += xte @ xtr.t()
        del xtr, xte
    return Ktr, Kte


def profile_r_numpy(preds, targets):
    """Per-image Pearson r across voxels."""
    N = preds.shape[0]
    rs = np.zeros(N)
    for i in range(N):
        p = preds[i] - preds[i].mean()
        t = targets[i] - targets[i].mean()
        den = np.sqrt(np.dot(p, p) * np.dot(t, t))
        rs[i] = np.dot(p, t) / (den + 1e-12)
    return rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[1e1, 1e2, 1e3, 1e4, 1e5])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_voxels", type=int, default=15724)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "ridge_fewshot"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "ridge_fewshot_results.csv"

    print(f"\n{'Held':>6s} {'Hours':>6s} {'N_img':>6s} {'λ':>8s} {'profile_r':>10s}")
    print("-" * 42)

    all_results = []

    for cfg in LOO_CONFIGS:
        held = cfg["held"]
        sdir = NSD / SUBJ_MAP[held]

        # Load fMRI
        fmri_train = np.load(
            sdir / "fmri" / f"nsd_train_fmri_zscore_sub{held}.npy",
            mmap_mode="r")
        fmri_test = np.load(
            sdir / "fmri" / f"nsd_test_fmri_zscore_sub{held}.npy",
            mmap_mode="r")
        n_images, n_reps = fmri_train.shape[0], fmri_train.shape[1]
        V = min(args.n_voxels, fmri_train.shape[2])

        Y_train_avg = np.asarray(fmri_train).mean(axis=1)[:, :V]  # (N, V)
        Y_test_avg = np.asarray(fmri_test).mean(axis=1)[:, :V]

        for hours in args.hours:
            adapt_idx = get_fewshot_split(n_images, n_reps, hours, args.seed)

            # Build summed Gram over streams (memory-efficient)
            Ntr, Nte = len(adapt_idx), Y_test_avg.shape[0]
            Ktr = torch.zeros(Ntr, Ntr, device=dev)
            Kte = torch.zeros(Nte, Ntr, device=dev)

            for sname, (subdir, prefix) in STREAMS.items():
                ptr = sdir / subdir / f"nsd_{prefix}_train_sub{held}.npy"
                pte = sdir / subdir / f"nsd_{prefix}_test_sub{held}.npy"
                kt, ke = stream_gram(str(ptr), str(pte), adapt_idx, dev)
                Ktr += kt; Kte += ke
                del kt, ke
                torch.cuda.empty_cache() if dev == "cuda" else None

            # Y subset
            Y_tr = torch.from_numpy(Y_train_avg[adapt_idx]).float().to(dev)
            Y_te_np = Y_test_avg

            # Lambda selection
            rng = torch.Generator().manual_seed(0)
            perm = torch.randperm(Ntr, generator=rng)
            n_val = max(int(Ntr * 0.15), 10)
            val_idx, fit_idx = perm[:n_val], perm[n_val:]

            Kff = Ktr[fit_idx][:, fit_idx]
            Kvf = Ktr[val_idx][:, fit_idx]
            Yf = Y_tr[fit_idx]
            Yf_mu = Yf.mean(0, keepdim=True)
            eye_f = torch.eye(Kff.shape[0], device=dev)

            best_r, best_lam = -1.0, args.lambdas[0]
            for lam in args.lambdas:
                alpha = torch.linalg.solve(Kff + lam * eye_f, Yf - Yf_mu)
                pred_val = Kvf @ alpha + Yf_mu
                pr = profile_r_numpy(pred_val.cpu().numpy(),
                                     Y_tr[val_idx].cpu().numpy())
                r = pr.mean()
                if r > best_r:
                    best_r, best_lam = r, lam

            # Refit on all adapt data
            Y_mu = Y_tr.mean(0, keepdim=True)
            alpha = torch.linalg.solve(
                Ktr + best_lam * torch.eye(Ntr, device=dev), Y_tr - Y_mu)
            pred = (Kte @ alpha + Y_mu).cpu().numpy()

            pr = profile_r_numpy(pred, Y_te_np)
            mean_pr = pr.mean()

            all_results.append({
                "held_out": held, "hours": hours,
                "n_images": Ntr, "best_lambda": best_lam,
                "profile_r": mean_pr,
            })
            print(f"{held:>6d} {hours:>6.0f} {Ntr:>6d} {best_lam:>8.0f} {mean_pr:>10.4f}")

            del Ktr, Kte, Y_tr, alpha, pred
            torch.cuda.empty_cache() if dev == "cuda" else None

        del Y_train_avg, Y_test_avg
        gc.collect()
        torch.cuda.empty_cache() if dev == "cuda" else None

    # Averages
    print("-" * 42)
    for h in args.hours:
        rows = [r for r in all_results if r["hours"] == h]
        avg = np.mean([r["profile_r"] for r in rows])
        print(f"{'avg':>6s} {h:>6.0f} {'--':>6s} {'--':>8s} {avg:>10.4f}")

    # Save
    with open(out_csv, "w") as f:
        f.write("held_out,hours,n_images,best_lambda,profile_r\n")
        for r in all_results:
            f.write(f"{r['held_out']},{r['hours']},{r['n_images']},"
                    f"{r['best_lambda']},{r['profile_r']:.6f}\n")
    print(f"\n-> Saved to {out_csv}")


if __name__ == "__main__":
    main()
