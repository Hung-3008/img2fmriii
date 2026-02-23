"""
Noise Ceiling & Performance Ceiling Analysis for NSD fMRI Data.

Checks:
1. fMRI data statistics (mean, std, range)
2. VAE reconstruction ceiling: encode→decode test set, measure PCC/MSE
3. Test-set repetition noise ceiling (if 3-rep data available)
4. Split-half reliability on training data (proxy noise ceiling)
5. Direct regression baseline: Ridge(DINOv2 CLS → fMRI)

Usage:
    python scripts/check_ceiling.py
"""

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import os
import sys

sys.path.insert(0, os.getcwd())

from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig


def pearson_corr_voxelwise(pred, target):
    """Mean voxelwise PCC across all voxels."""
    pred_zm = pred - pred.mean(0, keepdim=True)
    tgt_zm = target - target.mean(0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


def pearson_corr_samplewise(pred, target):
    """Mean samplewise PCC across all samples."""
    pred_zm = pred - pred.mean(1, keepdim=True)
    tgt_zm = target - target.mean(1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(1)
    den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    data_root = "Data/nsd/subj01"
    train_fmri_path = os.path.join(data_root, "nsd_train_fmri_zscore_sub1.npy")
    test_fmri_path = os.path.join(data_root, "nsd_test_fmri_zscore_sub1.npy")

    print("\n" + "="*70)
    print("1. DATA STATISTICS")
    print("="*70)

    raw_train = np.load(train_fmri_path)
    raw_test = np.load(test_fmri_path)
    print(f"Train fMRI shape: {raw_train.shape}, dtype: {raw_train.dtype}")
    print(f"Test  fMRI shape: {raw_test.shape}, dtype: {raw_test.dtype}")

    # Average reps if ndim==3
    if raw_train.ndim == 3:
        n_reps = raw_train.shape[1]
        print(f"\nTrain has {n_reps} repetitions per image")
        train_fmri = raw_train.mean(axis=1).astype(np.float32)
    else:
        n_reps = 0
        train_fmri = raw_train.astype(np.float32)

    if raw_test.ndim == 3:
        n_reps_test = raw_test.shape[1]
        print(f"Test  has {n_reps_test} repetitions per image")
        test_fmri = raw_test.mean(axis=1).astype(np.float32)
    else:
        n_reps_test = 0
        test_fmri = raw_test.astype(np.float32)

    print(f"\nTrain (averaged): {train_fmri.shape}")
    print(f"Test  (averaged): {test_fmri.shape}")
    print(f"Train mean: {train_fmri.mean():.6f}, std: {train_fmri.std():.6f}")
    print(f"Test  mean: {test_fmri.mean():.6f}, std: {test_fmri.std():.6f}")
    print(f"Train range: [{train_fmri.min():.4f}, {train_fmri.max():.4f}]")
    print(f"Test  range: [{test_fmri.min():.4f}, {test_fmri.max():.4f}]")

    n_voxels = train_fmri.shape[-1]
    print(f"N voxels: {n_voxels}")

    # ── Noise Ceiling from repetitions ──
    print("\n" + "="*70)
    print("2. NOISE CEILING (from repetitions)")
    print("="*70)

    if raw_test.ndim == 3 and n_reps_test >= 2:
        print(f"\nUsing test set ({n_reps_test} reps) for noise ceiling...")
        # Split-half noise ceiling:
        # For each voxel, correlate the mean of odd reps with mean of even reps
        reps = raw_test.astype(np.float32)
        n_splits = 100
        nc_voxel_all = []
        for _ in range(n_splits):
            perm = np.random.permutation(n_reps_test)
            half1 = reps[:, perm[:n_reps_test//2], :].mean(axis=1)
            half2 = reps[:, perm[n_reps_test//2:], :].mean(axis=1)
            # Voxelwise correlation
            h1_zm = half1 - half1.mean(axis=0, keepdims=True)
            h2_zm = half2 - half2.mean(axis=0, keepdims=True)
            num = (h1_zm * h2_zm).sum(axis=0)
            den = np.sqrt((h1_zm**2).sum(axis=0) * (h2_zm**2).sum(axis=0) + 1e-8)
            nc_voxel = num / den
            nc_voxel_all.append(nc_voxel)

        nc_voxel_mean = np.mean(nc_voxel_all, axis=0)

        # Spearman-Brown correction for full data (all reps)
        # r_full = (n_reps * r_half) / (1 + (n_reps - 1) * r_half)
        r_half = nc_voxel_mean
        r_corrected = (n_reps_test * r_half) / (1 + (n_reps_test - 1) * np.abs(r_half))

        print(f"  Split-half voxelwise PCC (raw):       mean={r_half.mean():.4f}, median={np.median(r_half):.4f}")
        print(f"  Spearman-Brown corrected (full data):  mean={r_corrected.mean():.4f}, median={np.median(r_corrected):.4f}")
        print(f"  Voxels with r > 0.3: {(r_corrected > 0.3).sum()} / {n_voxels} ({100*(r_corrected > 0.3).mean():.1f}%)")
        print(f"  Voxels with r > 0.5: {(r_corrected > 0.5).sum()} / {n_voxels} ({100*(r_corrected > 0.5).mean():.1f}%)")
        print(f"  Voxels with r > 0.7: {(r_corrected > 0.7).sum()} / {n_voxels} ({100*(r_corrected > 0.7).mean():.1f}%)")

        # Samplewise noise ceiling
        h1_zm_s = half1 - half1.mean(axis=1, keepdims=True)
        h2_zm_s = half2 - half2.mean(axis=1, keepdims=True)
        num_s = (h1_zm_s * h2_zm_s).sum(axis=1)
        den_s = np.sqrt((h1_zm_s**2).sum(axis=1) * (h2_zm_s**2).sum(axis=1) + 1e-8)
        nc_sample = num_s / den_s
        print(f"\n  Split-half samplewise PCC: mean={nc_sample.mean():.4f}, median={np.median(nc_sample):.4f}")
    elif raw_train.ndim == 3 and n_reps >= 2:
        print(f"\nUsing train set ({n_reps} reps) for noise ceiling...")
        reps = raw_train.astype(np.float32)
        half1 = reps[:, :n_reps//2, :].mean(axis=1)
        half2 = reps[:, n_reps//2:, :].mean(axis=1)
        h1_zm = half1 - half1.mean(axis=0, keepdims=True)
        h2_zm = half2 - half2.mean(axis=0, keepdims=True)
        num = (h1_zm * h2_zm).sum(axis=0)
        den = np.sqrt((h1_zm**2).sum(axis=0) * (h2_zm**2).sum(axis=0) + 1e-8)
        nc_voxel = num / den
        r_corrected = (n_reps * nc_voxel) / (1 + (n_reps - 1) * np.abs(nc_voxel))
        print(f"  Split-half voxelwise PCC (raw):       mean={nc_voxel.mean():.4f}, median={np.median(nc_voxel):.4f}")
        print(f"  Spearman-Brown corrected:              mean={r_corrected.mean():.4f}, median={np.median(r_corrected):.4f}")
    else:
        print("  No repetitions available. Cannot compute noise ceiling from reps.")
        print("  Using inter-sample variance as proxy...")
        # Compute per-voxel variance as a proxy
        test_t = torch.from_numpy(test_fmri).float()
        voxel_std = test_t.std(dim=0)
        print(f"  Per-voxel std: mean={voxel_std.mean():.4f}, median={voxel_std.median():.4f}")
        print(f"  Voxels with std > 0.5: {(voxel_std > 0.5).sum().item()} / {n_voxels}")

    # ── VAE Ceiling ──
    print("\n" + "="*70)
    print("3. VAE RECONSTRUCTION CEILING")
    print("="*70)

    vae_ckpt_path = "results/fmri_mlp_vae/best.pt"
    vae_config_path = "results/fmri_mlp_vae/config.yaml"

    if os.path.exists(vae_ckpt_path):
        with open(vae_config_path) as f:
            vae_cfg = yaml.safe_load(f)
        vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
        ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
        vae.load_state_dict(ckpt["model_state_dict"])
        print(f"VAE loaded from {vae_ckpt_path}")

        test_t = torch.from_numpy(test_fmri).float().to(device)

        with torch.no_grad():
            # Deterministic encode → decode (no sampling)
            z_det, _, _ = vae.encode(test_t, sample_posterior=False)
            recon_det = vae.decode(z_det)

            print(f"\n  Latent z shape: {z_det.shape}")
            print(f"  Latent z stats: mean={z_det.mean():.4f}, std={z_det.std():.4f}")
            print(f"    range: [{z_det.min():.4f}, {z_det.max():.4f}]")

            # VAE deterministic reconstruction quality
            mse_det = F.mse_loss(recon_det, test_t).item()
            vpcc_det = pearson_corr_voxelwise(recon_det, test_t)
            spcc_det = pearson_corr_samplewise(recon_det, test_t)

            print(f"\n  [Deterministic] encode(mean) → decode:")
            print(f"    MSE:  {mse_det:.6f}")
            print(f"    vPCC: {vpcc_det:.4f}")
            print(f"    sPCC: {spcc_det:.4f}")

            # VAE with sampling (posterior)
            z_samp, mu, logvar = vae.encode(test_t, sample_posterior=True)
            recon_samp = vae.decode(z_samp)
            mse_samp = F.mse_loss(recon_samp, test_t).item()
            vpcc_samp = pearson_corr_voxelwise(recon_samp, test_t)
            spcc_samp = pearson_corr_samplewise(recon_samp, test_t)

            print(f"\n  [Stochastic] encode(sample) → decode:")
            print(f"    MSE:  {mse_samp:.6f}")
            print(f"    vPCC: {vpcc_samp:.4f}")
            print(f"    sPCC: {spcc_samp:.4f}")

            # What's the VAE latent variance?
            print(f"\n  Posterior stats:")
            print(f"    mu:     mean={mu.mean():.4f}, std={mu.std():.4f}")
            print(f"    logvar: mean={logvar.mean():.4f}, std={logvar.std():.4f}")
            print(f"    sigma:  mean={logvar.div(2).exp().mean():.4f}")

            # ── Latent-level ceiling ──
            print(f"\n  Latent-level analysis:")
            z_true = z_det  # this is the 'ground truth' latent
            print(f"    z_true inter-sample variance: {z_true.var(dim=0).mean():.6f}")
            print(f"    z_true cross-sample std: {z_true.std():.4f}")

            # If flow matching produces perfect z_gen = z_true, what's the fMRI quality?
            # → Already computed above as deterministic recon
            print(f"\n  ★ CEILING: Perfect flow → z_gen == z_true → fMRI:")
            print(f"    MSE:  {mse_det:.6f}")
            print(f"    vPCC: {vpcc_det:.4f}")
            print(f"    sPCC: {spcc_det:.4f}")
    else:
        print(f"  VAE checkpoint not found: {vae_ckpt_path}")

    # ── Direct Regression Baseline ──
    print("\n" + "="*70)
    print("4. DIRECT REGRESSION BASELINE (DINOv2 CLS → fMRI)")
    print("="*70)

    dino_train_path = os.path.join(data_root,
        "nsd_dinov2_vitl14_multilayer_train_sub1.npy")
    dino_test_path = os.path.join(data_root,
        "nsd_dinov2_vitl14_multilayer_test_sub1.npy")

    if os.path.exists(dino_train_path):
        print("Loading DINOv2 features...")
        dino_train = np.load(dino_train_path, mmap_mode='r')
        dino_test = np.load(dino_test_path, mmap_mode='r')
        print(f"  DINOv2 train: {dino_train.shape}")
        print(f"  DINOv2 test:  {dino_test.shape}")

        # Use last layer CLS token
        print("  Using last layer CLS token for regression...")
        X_train = np.array(dino_train[:, -1, 0, :]).astype(np.float32)  # (N, 1024)
        X_test = np.array(dino_test[:, -1, 0, :]).astype(np.float32)

        # Also try concatenating all layer CLS tokens
        X_train_all = np.array(dino_train[:, :, 0, :]).reshape(
            dino_train.shape[0], -1).astype(np.float32)  # (N, 4096)
        X_test_all = np.array(dino_test[:, :, 0, :]).reshape(
            dino_test.shape[0], -1).astype(np.float32)

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        for name, Xtr, Xte in [
            ("Last-layer CLS (1024d)", X_train, X_test),
            ("All-layers CLS (4096d)", X_train_all, X_test_all),
        ]:
            print(f"\n  Ridge Regression: {name}")
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            for alpha in [1.0, 10.0, 100.0, 1000.0]:
                reg = Ridge(alpha=alpha)
                reg.fit(Xtr_s, train_fmri)
                pred = reg.predict(Xte_s)

                pred_t = torch.from_numpy(pred.astype(np.float32))
                test_t_cpu = torch.from_numpy(test_fmri)
                mse = F.mse_loss(pred_t, test_t_cpu).item()
                vpcc = pearson_corr_voxelwise(pred_t, test_t_cpu)
                spcc = pearson_corr_samplewise(pred_t, test_t_cpu)
                print(f"    α={alpha:>6.0f} → MSE={mse:.4f}, vPCC={vpcc:.4f}, sPCC={spcc:.4f}")
    else:
        print(f"  DINOv2 features not found: {dino_train_path}")

    # ── Summary ──
    print("\n" + "="*70)
    print("5. SUMMARY")
    print("="*70)
    print("""
    Your current best results (Multi-Layer V3, Ep 210):
      val_fmri_mse:  0.601
      val_fmri_pcc:  0.301
      val_fmri_spcc: 0.351

    Compare against the ceilings above to understand headroom.
    """)


if __name__ == "__main__":
    main()
