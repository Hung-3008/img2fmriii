"""
Authoritative Data Preparation Script for img2fmri.

Generates ALL processed data from MindEye2 pre-processed betas.
Cross-references against DINOv2 feature file for correct index mapping.

Inputs:
    - betas_all_subj01_fp32_renorm.hdf5  (30000, 15724) — MindEye2 betas
    - COCO_73k_subj_indices.hdf5         (30000,)       — trial → COCO 73K ID
    - shared1000.npy                     (73000,)       — test image flag
    - DINOv2 features HDF5               (10000, ...)   — with image_indices

Outputs (in processed_dir):
    subj01_train_avg.npy           (9000, 15724)   — averaged train fMRI
    subj01_train_avg_imgids.npy    (9000,)          — COCO 73K IDs
    subj01_train_avg_feat_idx.npy  (9000,)          — index into DINOv2 features
    subj01_test_avg.npy            (982, 15724)     — averaged test fMRI
    subj01_test_avg_imgids.npy     (982,)           — COCO 73K IDs
    subj01_test_avg_feat_idx.npy   (982,)           — index into DINOv2 features
    subj01_train_single.npy        (27000, 15724)   — single-trial train
    subj01_train_single_feat_idx.npy (27000,)       — index into DINOv2 features
    subj01_test_single.npy         (~2770, 15724)   — single-trial test
    subj01_test_single_feat_idx.npy  (~2770,)       — index into DINOv2 features

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --verify_only
"""

import argparse
import os
import sys
import numpy as np
import h5py
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for img2fmri")
    parser.add_argument(
        "--data_root", type=str, default="NSD/data",
        help="Root directory containing all NSD data"
    )
    parser.add_argument(
        "--subject", type=str, default="subj01",
        help="Subject ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="NSD/data/mindeye_nsd/processed",
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--verify_only", action="store_true",
        help="Only verify existing files, don't regenerate"
    )
    return parser.parse_args()


def load_sources(data_root: str, subject: str):
    """Load all source data files."""
    print("=" * 60)
    print("Loading source data...")
    print("=" * 60)

    # 1. Betas
    subj_num = subject.replace("subj", "")
    betas_path = os.path.join(
        data_root, "mindeye_nsd", f"subject_{int(subj_num)}",
        f"betas_all_{subject}_fp32_renorm.hdf5"
    )
    print(f"\n[1/4] Loading betas from {betas_path}...")
    with h5py.File(betas_path, 'r') as f:
        betas_all = f['betas'][:]  # (30000, 15724)
    print(f"  betas_all: shape={betas_all.shape}, dtype={betas_all.dtype}")

    # 2. COCO 73K indices (trial → image ID)
    indices_path = os.path.join(data_root, "COCO_73k_subj_indices.hdf5")
    print(f"\n[2/4] Loading COCO indices from {indices_path}...")
    with h5py.File(indices_path, 'r') as f:
        coco_indices = f[subject][:]  # (30000,)
    print(f"  coco_indices: shape={coco_indices.shape}, range=[{coco_indices.min()}, {coco_indices.max()}]")

    # 3. shared1000 flag
    shared1000_path = os.path.join(data_root, "shared1000.npy")
    print(f"\n[3/4] Loading shared1000 from {shared1000_path}...")
    shared1000 = np.load(shared1000_path)  # (73000,)
    print(f"  shared1000: shape={shared1000.shape}, sum={shared1000.sum():.0f}")

    # 4. DINOv2 features (just image_indices for mapping)
    features_dir = os.path.join(data_root, "features")
    dino_files = [f for f in os.listdir(features_dir) if f.endswith('.hdf5')]
    assert len(dino_files) >= 1, f"No DINOv2 features HDF5 found in {features_dir}"
    dino_path = os.path.join(features_dir, dino_files[0])
    print(f"\n[4/4] Loading DINOv2 image_indices from {dino_path}...")
    with h5py.File(dino_path, 'r') as f:
        dino_image_indices = f['image_indices'][:]  # (10000,)
        dino_feat_shape = f['features'].shape
    print(f"  image_indices: shape={dino_image_indices.shape}, range=[{dino_image_indices.min()}, {dino_image_indices.max()}]")
    print(f"  features shape: {dino_feat_shape}")

    return betas_all, coco_indices, shared1000, dino_image_indices


def build_coco_to_feat_idx(dino_image_indices: np.ndarray) -> dict:
    """Build mapping: COCO 73K image ID → position in DINOv2 features array."""
    coco_to_feat = {}
    for feat_pos, coco_id in enumerate(dino_image_indices):
        coco_to_feat[int(coco_id)] = feat_pos
    return coco_to_feat


def prepare_data(betas_all, coco_indices, shared1000, dino_image_indices,
                 subject, output_dir):
    """Generate all processed data files."""
    os.makedirs(output_dir, exist_ok=True)
    n_trials = len(betas_all)
    n_voxels = betas_all.shape[1]

    # Build COCO → DINOv2 feature index mapping
    coco_to_feat = build_coco_to_feat_idx(dino_image_indices)

    # Determine train/test for each trial
    is_test = np.array([shared1000[int(coco_indices[t])] == 1.0 for t in range(n_trials)])
    is_train = ~is_test

    train_trial_indices = np.where(is_train)[0]
    test_trial_indices = np.where(is_test)[0]

    print(f"\n{'='*60}")
    print(f"Trial split: {len(train_trial_indices)} train, {len(test_trial_indices)} test")
    print(f"{'='*60}")

    # ─── Averaged Data ─────────────────────────────────────────────────

    def create_averaged_data(trial_indices, split_name):
        """Group trials by image, average, sort by COCO ID."""
        print(f"\n--- Creating {split_name}_avg ---")

        # Group trial indices by COCO image ID
        image_to_trials = defaultdict(list)
        for t in trial_indices:
            coco_id = int(coco_indices[t])
            image_to_trials[coco_id].append(t)

        # Sort by COCO ID
        sorted_coco_ids = sorted(image_to_trials.keys())
        n_images = len(sorted_coco_ids)
        print(f"  {n_images} unique images")

        # Average trials per image
        avg_betas = np.zeros((n_images, n_voxels), dtype=np.float32)
        imgids = np.zeros(n_images, dtype=np.int64)
        feat_idx = np.zeros(n_images, dtype=np.int64)
        missing_feat = 0

        for i, coco_id in enumerate(sorted_coco_ids):
            trials = image_to_trials[coco_id]
            avg_betas[i] = betas_all[trials].mean(axis=0)
            imgids[i] = coco_id

            if coco_id in coco_to_feat:
                feat_idx[i] = coco_to_feat[coco_id]
            else:
                feat_idx[i] = -1
                missing_feat += 1

        if missing_feat > 0:
            print(f"  WARNING: {missing_feat} images have no DINOv2 features!")
            # Remove entries with no features
            valid_mask = feat_idx >= 0
            avg_betas = avg_betas[valid_mask]
            imgids = imgids[valid_mask]
            feat_idx = feat_idx[valid_mask]
            n_images = len(avg_betas)
            print(f"  After filtering: {n_images} images with valid features")

        # Verify trials per image
        trials_per_img = [len(image_to_trials[cid]) for cid in sorted_coco_ids]
        print(f"  Trials per image: mean={np.mean(trials_per_img):.2f}, "
              f"min={min(trials_per_img)}, max={max(trials_per_img)}")

        # Save
        prefix = f"{subject}_{split_name}_avg"
        np.save(os.path.join(output_dir, f"{prefix}.npy"), avg_betas)
        np.save(os.path.join(output_dir, f"{prefix}_imgids.npy"), imgids)
        np.save(os.path.join(output_dir, f"{prefix}_feat_idx.npy"), feat_idx)

        print(f"  Saved: {prefix}.npy  shape={avg_betas.shape}")
        print(f"  Saved: {prefix}_imgids.npy  shape={imgids.shape} range=[{imgids.min()}, {imgids.max()}]")
        print(f"  Saved: {prefix}_feat_idx.npy  shape={feat_idx.shape} range=[{feat_idx.min()}, {feat_idx.max()}]")

        return avg_betas, imgids, feat_idx

    train_avg, train_imgids, train_feat_idx = create_averaged_data(train_trial_indices, "train")
    test_avg, test_imgids, test_feat_idx = create_averaged_data(test_trial_indices, "test")

    # ─── Single-Trial Data ─────────────────────────────────────────────

    def create_single_trial_data(trial_indices, split_name):
        """Keep trials in chronological order, build feat_idx."""
        print(f"\n--- Creating {split_name}_single ---")

        # trial_indices are already in chronological order (sorted)
        sorted_trials = np.sort(trial_indices)
        n_trials_split = len(sorted_trials)

        single_betas = betas_all[sorted_trials]  # (N, V)
        single_feat_idx = np.array(
            [coco_to_feat.get(int(coco_indices[t]), -1) for t in sorted_trials],
            dtype=np.int64
        )

        missing = (single_feat_idx < 0).sum()
        if missing > 0:
            print(f"  WARNING: {missing} trials have no DINOv2 features!")

        # Save
        prefix = f"{subject}_{split_name}_single"
        np.save(os.path.join(output_dir, f"{prefix}.npy"), single_betas)
        np.save(os.path.join(output_dir, f"{prefix}_feat_idx.npy"), single_feat_idx)

        print(f"  Saved: {prefix}.npy  shape={single_betas.shape}")
        print(f"  Saved: {prefix}_feat_idx.npy  shape={single_feat_idx.shape} range=[{single_feat_idx.min()}, {single_feat_idx.max()}]")

        return single_betas, single_feat_idx

    train_single, train_single_fidx = create_single_trial_data(train_trial_indices, "train")
    test_single, test_single_fidx = create_single_trial_data(test_trial_indices, "test")

    return {
        "train_avg": (train_avg, train_imgids, train_feat_idx),
        "test_avg": (test_avg, test_imgids, test_feat_idx),
        "train_single": (train_single, train_single_fidx),
        "test_single": (test_single, test_single_fidx),
    }


def verify_data(output_dir, subject, dino_image_indices):
    """Verify all processed data files."""
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    coco_to_feat = build_coco_to_feat_idx(dino_image_indices)
    errors = []

    # ─── Check averaged data ───
    for split in ["train", "test"]:
        prefix = f"{subject}_{split}_avg"
        fmri_path = os.path.join(output_dir, f"{prefix}.npy")
        imgids_path = os.path.join(output_dir, f"{prefix}_imgids.npy")
        feat_idx_path = os.path.join(output_dir, f"{prefix}_feat_idx.npy")

        for p in [fmri_path, imgids_path, feat_idx_path]:
            if not os.path.exists(p):
                errors.append(f"MISSING: {p}")
                continue

        fmri = np.load(fmri_path, mmap_mode='r')
        imgids = np.load(imgids_path)
        feat_idx = np.load(feat_idx_path)

        print(f"\n--- {prefix} ---")
        print(f"  fmri: shape={fmri.shape}, dtype={fmri.dtype}")
        print(f"  imgids: shape={imgids.shape}, dtype={imgids.dtype}, range=[{imgids.min()}, {imgids.max()}]")
        print(f"  feat_idx: shape={feat_idx.shape}, dtype={feat_idx.dtype}, range=[{feat_idx.min()}, {feat_idx.max()}]")

        # Shape consistency
        if fmri.shape[0] != len(imgids):
            errors.append(f"{prefix}: fmri.shape[0]={fmri.shape[0]} != imgids.len={len(imgids)}")
        if fmri.shape[0] != len(feat_idx):
            errors.append(f"{prefix}: fmri.shape[0]={fmri.shape[0]} != feat_idx.len={len(feat_idx)}")

        # Sorted by COCO ID
        if not np.all(imgids[:-1] <= imgids[1:]):
            errors.append(f"{prefix}: imgids not sorted ascending!")

        # feat_idx consistency
        for i in range(min(10, len(imgids))):
            coco_id = int(imgids[i])
            expected_feat = coco_to_feat.get(coco_id, -1)
            if feat_idx[i] != expected_feat:
                errors.append(f"{prefix}[{i}]: feat_idx={feat_idx[i]} but expected={expected_feat} for COCO {coco_id}")

        # Spot-check a few random samples
        rng = np.random.RandomState(42)
        check_indices = rng.choice(len(imgids), min(5, len(imgids)), replace=False)
        print(f"  Spot-check (random indices {check_indices}):")
        for i in check_indices:
            coco_id = int(imgids[i])
            expected = coco_to_feat.get(coco_id, -1)
            match = "✅" if feat_idx[i] == expected else "❌"
            print(f"    [{i}] COCO={coco_id} → feat_idx={feat_idx[i]} (expected={expected}) {match}")

    # ─── Check single-trial data ───
    for split in ["train", "test"]:
        prefix = f"{subject}_{split}_single"
        fmri_path = os.path.join(output_dir, f"{prefix}.npy")
        feat_idx_path = os.path.join(output_dir, f"{prefix}_feat_idx.npy")

        for p in [fmri_path, feat_idx_path]:
            if not os.path.exists(p):
                errors.append(f"MISSING: {p}")
                continue

        fmri = np.load(fmri_path, mmap_mode='r')
        feat_idx = np.load(feat_idx_path)

        print(f"\n--- {prefix} ---")
        print(f"  fmri: shape={fmri.shape}, dtype={fmri.dtype}")
        print(f"  feat_idx: shape={feat_idx.shape}, dtype={feat_idx.dtype}, range=[{feat_idx.min()}, {feat_idx.max()}]")

        if fmri.shape[0] != len(feat_idx):
            errors.append(f"{prefix}: fmri.shape[0]={fmri.shape[0]} != feat_idx.len={len(feat_idx)}")

        if (feat_idx < 0).any():
            n_missing = (feat_idx < 0).sum()
            errors.append(f"{prefix}: {n_missing} entries have feat_idx=-1 (no DINOv2 features)")

    # ─── Cross-check: train ∪ test = all images with features ───
    train_imgids = np.load(os.path.join(output_dir, f"{subject}_train_avg_imgids.npy"))
    test_imgids = np.load(os.path.join(output_dir, f"{subject}_test_avg_imgids.npy"))
    all_imgids = set(train_imgids.tolist()) | set(test_imgids.tolist())
    dino_ids = set(dino_image_indices.tolist())
    overlap = all_imgids & dino_ids
    only_fmri = all_imgids - dino_ids
    only_dino = dino_ids - all_imgids

    print(f"\n--- Coverage ---")
    print(f"  Train images: {len(train_imgids)}")
    print(f"  Test images:  {len(test_imgids)}")
    print(f"  Total:        {len(all_imgids)}")
    print(f"  DINOv2 has:   {len(dino_ids)} images")
    print(f"  Overlapping:  {len(overlap)}")
    print(f"  fMRI only:    {len(only_fmri)}")
    print(f"  DINOv2 only:  {len(only_dino)}")

    if only_fmri:
        errors.append(f"{len(only_fmri)} images in fMRI but not in DINOv2 features!")

    # ─── Summary ───
    print(f"\n{'='*60}")
    if errors:
        print(f"❌ VERIFICATION FAILED — {len(errors)} errors:")
        for e in errors:
            print(f"  • {e}")
        return False
    else:
        print("✅ ALL CHECKS PASSED")
        return True


def main():
    args = parse_args()

    # Load DINOv2 image_indices for verification
    features_dir = os.path.join(args.data_root, "features")
    dino_files = [f for f in os.listdir(features_dir) if f.endswith('.hdf5')]
    dino_path = os.path.join(features_dir, dino_files[0])
    with h5py.File(dino_path, 'r') as f:
        dino_image_indices = f['image_indices'][:]

    if args.verify_only:
        success = verify_data(args.output_dir, args.subject, dino_image_indices)
        sys.exit(0 if success else 1)

    # Full pipeline
    betas_all, coco_indices, shared1000, dino_image_indices = load_sources(
        args.data_root, args.subject
    )

    data = prepare_data(
        betas_all, coco_indices, shared1000, dino_image_indices,
        args.subject, args.output_dir
    )

    # Verify
    success = verify_data(args.output_dir, args.subject, dino_image_indices)

    if success:
        print(f"\n🎉 Data preparation complete! Files saved to {args.output_dir}")
    else:
        print(f"\n⚠️  Data preparation finished but verification found issues!")
        sys.exit(1)


if __name__ == "__main__":
    main()
