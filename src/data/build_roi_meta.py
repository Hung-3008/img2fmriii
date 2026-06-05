"""
build_roi_meta.py
=================
Build per-subject ROI metadata for ROI-aware FactFlow training.

For each subject, derive a per-voxel ROI label aligned to the fMRI voxel axis,
plus a stable sort that groups voxels of the same ROI contiguously (so the
DiT1D's 32-voxel patches become ROI-pure and an ROI-block attention mask is
possible).

ROI atlas: **streams** (``streams.nii.gz``) — covers ~99.6 % of nsdgeneral
voxels with 7 labels (1=early … 7=parietal). Voxels with label <= 0 (outside
the streams atlas / unknown) are folded into a single "other" bucket = 0, so
the label set is ``0..7`` ⇒ ``n_roi = 8``.

Voxel ordering
--------------
The fMRI arrays are saved as ``betas[nsdgeneral>0].T`` (see
``prepare_nsddata_scale.py``), i.e. C-order boolean indexing of the 3-D mask.
Indexing any ROI volume the same way — ``roi_vol[nsdgeneral>0]`` — yields a
label per voxel in exactly the fMRI voxel order. No resampling needed.

Output (per subject)::

    NSD/data/nsd/subj0{S}/roi_meta_sub{S}.npz
        roi_labels   (n_voxels,) int   — ROI id per voxel, original fMRI order
        sort_idx     (n_voxels,) int   — sorted[i] = raw[sort_idx[i]] (groups ROIs)
        unsort_idx   (n_voxels,) int   — inverse of sort_idx (restore original order)
        roi_ids      (R,) int          — distinct ROI ids present
        roi_sizes    (R,) int          — voxel count per ROI id
        roi_names    (8,) str          — name per ROI id 0..7
        n_roi        int               — 8

Usage::

    python src/data/build_roi_meta.py                 # all of [1,2,5,7]
    python src/data/build_roi_meta.py --subjects 1 7
"""

import argparse
import os

import numpy as np
import nibabel as nib

# ROI id → name (streams atlas; id 0 = "other" = folded label <= 0)
ROI_NAMES = [
    "other",        # 0
    "early",        # 1
    "midventral",   # 2
    "midlateral",   # 3
    "midparietal",  # 4
    "ventral",      # 5
    "lateral",      # 6
    "parietal",     # 7
]
N_ROI = len(ROI_NAMES)  # 8


def build_subject(sub: int, data_root: str) -> dict:
    roi_dir = os.path.join(
        data_root, "nsddata", "ppdata", f"subj{sub:02d}", "func1pt8mm", "roi"
    )
    mask = nib.load(os.path.join(roi_dir, "nsdgeneral.nii.gz")).get_fdata()
    sel = mask > 0                          # canonical voxel selector (C-order)
    n_voxels = int(sel.sum())

    streams = nib.load(os.path.join(roi_dir, "streams.nii.gz")).get_fdata()
    raw = streams[sel]                      # (n_voxels,) values in {-1,0,1..7}
    roi_labels = np.where(raw > 0, raw, 0).astype(np.int64)  # <=0 → 0 "other"

    # Cross-check against the saved fMRI voxel count.
    fmri_path = os.path.join(
        data_root, "nsd", f"subj{sub:02d}", "fmri",
        f"nsd_train_fmri_zscore_sub{sub}.npy",
    )
    if os.path.exists(fmri_path):
        v_fmri = np.load(fmri_path, mmap_mode="r").shape[-1]
        assert v_fmri == n_voxels, (
            f"subj{sub}: fMRI has {v_fmri} voxels but nsdgeneral mask has "
            f"{n_voxels} — voxel ordering would be inconsistent."
        )

    # Stable sort → contiguous ROI blocks; argsort of that = inverse permutation.
    sort_idx = np.argsort(roi_labels, kind="stable")
    unsort_idx = np.argsort(sort_idx, kind="stable")

    roi_ids, roi_sizes = np.unique(roi_labels, return_counts=True)
    return dict(
        roi_labels=roi_labels,
        sort_idx=sort_idx.astype(np.int64),
        unsort_idx=unsort_idx.astype(np.int64),
        roi_ids=roi_ids.astype(np.int64),
        roi_sizes=roi_sizes.astype(np.int64),
        roi_names=np.array(ROI_NAMES),
        n_roi=np.int64(N_ROI),
        n_voxels=np.int64(n_voxels),
    )


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    default_root = os.path.join(repo_root, "NSD", "data")

    ap = argparse.ArgumentParser(description="Build per-subject ROI metadata")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 5, 7])
    ap.add_argument("--data_root", type=str, default=default_root,
                    help="Path to NSD/data (contains nsddata/ and nsd/)")
    args = ap.parse_args()

    for sub in args.subjects:
        meta = build_subject(sub, args.data_root)
        out_dir = os.path.join(args.data_root, "nsd", f"subj{sub:02d}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"roi_meta_sub{sub}.npz")
        np.savez(out_path, **meta)

        # Report coverage / block structure.
        sizes = {int(i): int(c) for i, c in zip(meta["roi_ids"], meta["roi_sizes"])}
        n_vox = int(meta["n_voxels"])
        other = sizes.get(0, 0)
        print(
            f"subj{sub:02d}: n_voxels={n_vox}  "
            f"other(0)={other} ({100*other/n_vox:.2f}%)  "
            f"ROIs present={list(sizes.keys())}"
        )
        named = "  ".join(
            f"{ROI_NAMES[i]}={sizes.get(i, 0)}" for i in range(N_ROI)
        )
        print(f"          {named}")
        print(f"          → {out_path}")


if __name__ == "__main__":
    main()
