#!/usr/bin/env python3
"""Top-K reconstruction comparison canvas for StimFlow semantic eval.

Selects the K test images whose StimFlow (T=1) reconstruction is most similar
to the seen stimulus (CLIP image-embedding cosine), and lays them out as a grid:

    stimulus (GT) | recon from GT fMRI | recon from StimFlow T=1 | recon from T=5

Image sources (1000 shared-test images each):
    stimulus            results/semantic_eval/<sub>_T1/gt/NNNN.png
    GT-fMRI recon       results/semantic_eval/<sub>_gt/recons/NNNN.png   (DIFFERENT order!)
    StimFlow T=1 recon  results/semantic_eval/<sub>_T1/recons/NNNN.png
    StimFlow T=5 recon  results/semantic_eval/<sub>_T5/recons/NNNN.png

The <sub>_gt run uses a different image ordering, so its column is realigned to
the T1/T5 order by matching the GT thumbnails (exact permutation, verified).

Usage:
    .venv/bin/python src/plot_recon_comparison.py --sub sub1 --topk 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
SEM = ROOT / "results" / "semantic_eval"
CLIP_ID = "openai/clip-vit-large-patch14"
N = 1000


def folder(sub, run, kind):
    return SEM / f"{sub}_{run}" / kind


def load_imgs(d: Path, n=N):
    return [Image.open(d / f"{i:04d}.png").convert("RGB") for i in range(n)]


@torch.no_grad()
def clip_embed(imgs, model, proc, device, bs=64):
    feats = []
    for i in range(0, len(imgs), bs):
        batch = proc(images=imgs[i:i + bs], return_tensors="pt").to(device)
        f = model.get_image_features(**batch)
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(feats).numpy()


def thumb_feats(imgs, s=16):
    X = np.stack([np.asarray(im.resize((s, s)), np.float32).ravel() for im in imgs])
    X -= X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub1")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--start", type=int, default=0,
                    help="rank offset (0 = best); e.g. --start 10 --topk 20 -> ranks 11..30")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="explicit image indices (overrides ranking); kept in given order")
    ap.add_argument("--transpose", action="store_true",
                    help="rows = methods (GT/GT-recon/T=1/T=5), columns = images")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "viz"))
    args = ap.parse_args()
    sub, K, START = args.sub, args.topk, args.start
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load the four image sets -----------------------------------------
    gt = load_imgs(folder(sub, "T1", "gt"))
    rec_t1 = load_imgs(folder(sub, "T1", "recons"))
    rec_t5 = load_imgs(folder(sub, "T5", "recons"))
    gt_g = load_imgs(folder(sub, "gt", "gt"))          # different order
    rec_g = load_imgs(folder(sub, "gt", "recons"))

    # realign the GT-fMRI run to the T1/T5 order by matching GT thumbnails
    A, B = thumb_feats(gt), thumb_feats(gt_g)
    gmap = (A @ B.T).argmax(1)                          # T1 index -> gt-run index
    rec_g = [rec_g[j] for j in gmap]

    # methods: (long title, short label, image list, sim key)
    methods = [("Stimulus (GT)", "GT", gt, None),
               ("Recon | GT fMRI", "GT recons", rec_g, "GT fMRI"),
               ("Recon | StimFlow T=1", "StimFlow T=1", rec_t1, "T=1"),
               ("Recon | StimFlow T=5", "StimFlow T=5", rec_t5, "T=5")]

    # --- CLIP similarity to stimulus --------------------------------------
    print(f"loading CLIP {CLIP_ID} on {device} ...")
    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)

    if args.ids is not None:                            # explicit selection
        sel = list(args.ids)
        e_gt = clip_embed([gt[i] for i in sel], model, proc, device)
        sims = {key: (clip_embed([imgs[i] for i in sel], model, proc, device)
                      * e_gt).sum(1)
                for _, _, imgs, key in methods if key}
        print("selected ids:", sel)
        tag, sub_lbl = "selected", f"{len(sel)} selected images"
    else:                                               # rank by StimFlow T=1
        e_gt = clip_embed(gt, model, proc, device)
        full = {key: (clip_embed(imgs, model, proc, device) * e_gt).sum(1)
                for _, _, imgs, key in methods if key}
        order = np.argsort(-full["T=1"])[START:START + K]
        sel = order.tolist()
        sims = {key: full[key][np.array(sel)] for key in full}
        tag = f"top{K}" if START == 0 else f"rank{START+1}-{START+K}"
        sub_lbl = "top-%d" % K if START == 0 else "ranks %d-%d" % (START+1, START+K)
        print("ranks: %s" % sel)

    sim_pos = {key: i for i, key in enumerate(sims)}    # row->col index in sims arrays

    def cell(ax, imgs, idx, red, key, sims_idx):
        ax.imshow(imgs[idx]); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#C8102E" if red else "#888"); sp.set_linewidth(1.4)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- canvas -----------------------------------------------------------
    if args.transpose:                                  # rows = methods, cols = images
        n = len(sel)
        fig, axes = plt.subplots(4, n, figsize=(2.5 * n + 0.6, 2.5 * 4))
        axes = np.atleast_2d(axes)
        for r, (long, short, imgs, key) in enumerate(methods):
            for c, idx in enumerate(sel):
                ax = axes[r, c]
                cell(ax, imgs, idx, red=r >= 2, key=key, sims_idx=c)
                if r == 0:
                    ax.set_title(f"#{idx}", fontsize=11, pad=4)
                if c == 0:
                    ax.set_ylabel(short, fontsize=13, rotation=90, labelpad=8,
                                  color="#C8102E" if r >= 2 else "#222")
        orient = "_T"
    else:                                               # rows = images, cols = methods
        K2 = len(sel)
        fig, axes = plt.subplots(K2, 4, figsize=(8.4, 2.1 * K2))
        axes = np.atleast_2d(axes)
        for r, idx in enumerate(sel):
            for c, (long, short, imgs, key) in enumerate(methods):
                ax = axes[r, c]
                cell(ax, imgs, idx, red=c >= 2, key=key, sims_idx=r)
                if r == 0:
                    ax.set_title(long, fontsize=11, pad=6)
                if c == 0:
                    ax.set_ylabel(f"#{idx}", fontsize=9)
        orient = ""

    if not args.transpose:
        fig.suptitle(f"StimFlow reconstructions ({sub}, {sub_lbl})", fontsize=13, y=1.005)
    fig.tight_layout()
    if args.transpose:
        fig.subplots_adjust(wspace=0.03, hspace=0.08)
    out = outdir / f"09_recon_{tag}{orient}_{sub}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
