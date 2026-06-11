"""
Plot first 20 training images + captions for subjects 1, 2, 5, 7.
Layout: 4 rows (subjects) x 20 columns (images).

Caption source: NSD/data/annots/COCO_73k_annots_curated.npy  (73000, 5)
Index mapping:  NSD/data/COCO_73k_subj_indices.hdf5
  → dataset 'subj01' etc. holds the 73k-image indices for each subject's
    train/test trials (in NSD trial order).
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECTS   = [1, 2, 5, 7]
N_IMAGES   = 20
DATA_ROOT  = Path("NSD/data/nsd")
ANNOT_PATH = Path("NSD/data/annots/COCO_73k_annots_curated.npy")
INDEX_PATH = Path("NSD/data/COCO_73k_subj_indices.hdf5")
OUT_PATH   = Path("results/plot_first20_images.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CAP_WIDTH  = 28   # chars per caption line
FONT_SIZE  = 5.2

# ── Load global captions & index map ─────────────────────────────────────────
print("Loading global COCO 73k annotations (mmap)...")
all_caps = np.load(ANNOT_PATH, mmap_mode="r")          # (73000, 5) <U250
print(f"  all_caps shape: {all_caps.shape}")

print("Loading COCO_73k_subj_indices.hdf5 ...")
with h5py.File(INDEX_PATH, "r") as hf:
    print(f"  HDF5 keys: {list(hf.keys())}")
    subj_indices = {k: hf[k][:] for k in hf.keys()}   # load all into RAM (small)

# ── Load images + captions per subject ───────────────────────────────────────
print("Loading train stimuli (mmap, first 20 samples each)...")

images_all   = {}
captions_all = {}

for s in SUBJECTS:
    subj_dir  = DATA_ROOT / f"subj0{s}"
    stim_path = subj_dir / f"nsd_train_stim_sub{s}.npy"

    imgs = np.load(stim_path, mmap_mode="r")           # (9000, 425, 425, 3) u8
    images_all[s] = np.array(imgs[:N_IMAGES])          # copy 20 frames to RAM

    # Try to get per-subject caption file first
    cap_path = subj_dir / f"nsd_train_cap_sub{s}.npy"
    if cap_path.exists():
        caps = np.load(cap_path, mmap_mode="r")        # (9000, 5)
        captions_all[s] = [str(caps[i, 0]) for i in range(N_IMAGES)]
        print(f"  subj0{s}: imgs={images_all[s].shape}  caps from per-subject file")
    else:
        # Fall back to global index mapping
        key = f"subj0{s}"
        if key in subj_indices:
            idx73k = subj_indices[key][:N_IMAGES]      # first 20 global indices
            captions_all[s] = [str(all_caps[idx, 0]) for idx in idx73k]
            print(f"  subj0{s}: imgs={images_all[s].shape}  caps via index map key='{key}'")
        else:
            # Last resort: no caption available
            captions_all[s] = [f"(no caption)" for _ in range(N_IMAGES)]
            print(f"  subj0{s}: imgs={images_all[s].shape}  ⚠️  no caption found")

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Rendering figure...")

BG_COLOR   = "#0f0f1a"
TITLE_CLR  = "#e8e8ff"
CAP_CLR    = "#c8c8dd"
LABEL_CLR  = "#e2c275"
IDX_CLR    = "#7ec8e3"

fig, axes = plt.subplots(
    nrows=len(SUBJECTS),
    ncols=N_IMAGES,
    figsize=(N_IMAGES * 2.0, len(SUBJECTS) * 3.5),
    facecolor=BG_COLOR,
)
fig.subplots_adjust(hspace=0.05, wspace=0.04)

for row_idx, s in enumerate(SUBJECTS):
    for col_idx in range(N_IMAGES):
        ax = axes[row_idx, col_idx]
        ax.set_facecolor(BG_COLOR)

        # ── Image ──
        img = images_all[s][col_idx]
        ax.imshow(img, interpolation="bilinear", aspect="equal")
        ax.axis("off")

        # Thin border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#3a3a5c")
            spine.set_linewidth(0.5)

        # ── Caption ──
        cap_raw = captions_all[s][col_idx]
        cap_wrapped = "\n".join(textwrap.wrap(cap_raw, CAP_WIDTH))
        ax.set_title(
            cap_wrapped,
            fontsize=FONT_SIZE,
            color=CAP_CLR,
            pad=3,
            loc="center",
            linespacing=1.3,
        )

        # ── Image index badge ──
        ax.text(
            0.03, 0.97,
            f"#{col_idx}",
            transform=ax.transAxes,
            fontsize=5,
            color=IDX_CLR,
            va="top", ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#0f0f1a", alpha=0.7, ec="none"),
        )

    # ── Row / subject label ──
    axes[row_idx, 0].set_ylabel(
        f"subj{s:02d}\n{images_all[s].shape[1]}px",
        fontsize=8,
        color=LABEL_CLR,
        fontweight="bold",
        rotation=90,
        labelpad=6,
        va="center",
    )

fig.suptitle(
    "NSD — First 20 Training Images per Subject  (caption #0 of 5)",
    fontsize=11,
    color=TITLE_CLR,
    fontweight="bold",
    y=1.002,
)

fig.savefig(
    OUT_PATH,
    dpi=150,
    bbox_inches="tight",
    facecolor=BG_COLOR,
)
print(f"\n✅  Saved → {OUT_PATH}")
plt.close(fig)
