"""
Stage 1 Training: Aligned fMRI Autoencoder.

Trains an autoencoder that produces fMRI representations aligned with CLIP features.
Uses MSE reconstruction loss + SoftCLIP contrastive alignment loss.

After training:
    1. Encoder produces [B, 257, repr_dim] representations near CLIP in shared space
    2. Extract representations for all training/test fMRI → save as .npy
    3. Use these representations as targets for Stage 2 flow matching
"""

import argparse
import copy
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.model.aligned_fmri_ae import (
    AlignedFmriAutoencoder,
    AlignedAEConfig,
    compute_ae_loss,
)


# ─── Dataset ──────────────────────────────────────────────────────────────────


class AlignedAEDataset(Dataset):
    """Dataset for training the aligned autoencoder. Returns fMRI + CLIP features."""

    def __init__(self, fmri_path: str, clip_path: str, feat_idx_path: str, split: str = "train"):
        print(f"AlignedAEDataset [{split}]: Loading...")

        self.fmri = np.load(fmri_path, mmap_mode="r")
        self.n_samples = self.fmri.shape[0]
        print(f"  fMRI: {self.fmri.shape}")

        # CLIP features [10000, 257, 1024] or [10000, L, 257, 1024]
        raw_feat = np.load(clip_path, mmap_mode="r")
        if raw_feat.ndim == 3:
            self.clip = raw_feat  # [N_img, 257, 1024]
        elif raw_feat.ndim == 4:
            self.clip = raw_feat[:, 0, :, :]  # first layer
        else:
            raise ValueError(f"Unexpected CLIP shape: {raw_feat.shape}")
        print(f"  CLIP: {self.clip.shape}")

        self.feat_idx = np.load(feat_idx_path)
        assert len(self.feat_idx) == self.n_samples
        print(f"  Samples: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx].copy()).float()
        clip = torch.from_numpy(self.clip[self.feat_idx[idx]].copy()).float()
        return {"fmri": fmri, "clip": clip}


# ─── Utilities ────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=1e-6):
    import math
    if epoch < warmup_epochs:
        lr = base_lr * epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def pearson_corr_voxelwise(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_zm = pred - pred.mean(dim=0, keepdim=True)
    tgt_zm = target - target.mean(dim=0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(dim=0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser("Stage 1: Aligned fMRI Autoencoder")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    num_epochs = 2 if args.debug else train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]

    output_dir = cfg.get("output_dir", "results/stage1_aligned")
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──
    print("\n=== Loading Data ===")
    train_ds = AlignedAEDataset(
        fmri_path=data_cfg["train_fmri"], clip_path=data_cfg["clip_path"],
        feat_idx_path=data_cfg["train_feat_idx"], split="train",
    )
    val_ds = AlignedAEDataset(
        fmri_path=data_cfg["test_fmri"], clip_path=data_cfg["clip_path"],
        feat_idx_path=data_cfg["test_feat_idx"], split="test",
    )

    if args.debug:
        train_ds = torch.utils.data.Subset(train_ds, range(min(128, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(64, len(val_ds))))
        batch_size = min(batch_size, 32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=(not args.debug))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    # ── Model ──
    print("\n=== Creating Model ===")
    ae_config = AlignedAEConfig(**model_cfg)
    model = AlignedFmriAutoencoder(ae_config).to(device)
    print(f"  AlignedFmriAutoencoder: {model.param_count()['total_M']:.1f}M params")
    print(f"  Config: {ae_config}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=train_cfg.get("weight_decay", 0.01))

    lambda_align = train_cfg.get("lambda_align", 1.0)
    temperature = train_cfg.get("temperature", 0.07)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    # ── History ──
    history_path = os.path.join(output_dir, "history.csv")
    fields = ["epoch", "lr", "train_loss", "train_mse", "train_align", "train_cos_sim",
              "val_loss", "val_mse", "val_align", "val_cos_sim", "val_pcc"]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_loss = float("inf")

    # ── Training ──
    print(f"\n=== Training ({num_epochs} epochs) ===")
    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(optimizer, epoch - 1, num_epochs, warmup_epochs, lr)

        running = {"loss": 0, "mse": 0, "align": 0, "cosine_sim": 0}
        n_steps = 0

        for batch in train_loader:
            fmri = batch["fmri"].to(device)
            clip = batch["clip"].to(device)

            fmri_recon, fmri_repr = model(fmri)
            clip_proj = model.project_clip(clip)

            losses = compute_ae_loss(fmri, fmri_recon, fmri_repr, clip_proj,
                                      lambda_align=lambda_align, temperature=temperature)

            optimizer.zero_grad()
            losses["loss"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for k in running:
                running[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
            n_steps += 1

        for k in running:
            running[k] /= max(n_steps, 1)

        # ── Validation ──
        model.eval()
        val_running = {"loss": 0, "mse": 0, "align": 0, "cosine_sim": 0}
        all_pred, all_true = [], []
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                fmri = batch["fmri"].to(device)
                clip = batch["clip"].to(device)

                fmri_recon, fmri_repr = model(fmri)
                clip_proj = model.project_clip(clip)
                losses = compute_ae_loss(fmri, fmri_recon, fmri_repr, clip_proj,
                                          lambda_align=lambda_align, temperature=temperature)

                for k in val_running:
                    val_running[k] += losses[k] if isinstance(losses[k], float) else losses[k].item()
                n_val += 1
                all_pred.append(fmri_recon)
                all_true.append(fmri)

        for k in val_running:
            val_running[k] /= max(n_val, 1)

        val_pcc = pearson_corr_voxelwise(torch.cat(all_pred), torch.cat(all_true))

        # ── Logging ──
        row = {
            "epoch": epoch, "lr": f"{current_lr:.2e}",
            "train_loss": f"{running['loss']:.6f}", "train_mse": f"{running['mse']:.6f}",
            "train_align": f"{running['align']:.6f}", "train_cos_sim": f"{running['cosine_sim']:.4f}",
            "val_loss": f"{val_running['loss']:.6f}", "val_mse": f"{val_running['mse']:.6f}",
            "val_align": f"{val_running['align']:.6f}", "val_cos_sim": f"{val_running['cosine_sim']:.4f}",
            "val_pcc": f"{val_pcc:.4f}",
        }
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        is_best = val_running["loss"] < best_loss
        if is_best:
            best_loss = val_running["loss"]
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg, "best_loss": best_loss,
            }, os.path.join(output_dir, "best.pt"))

        print(
            f"  Epoch {epoch:3d}/{num_epochs} | lr={current_lr:.2e} | "
            f"mse={running['mse']:.5f} align={running['align']:.4f} cos={running['cosine_sim']:.3f} | "
            f"v_mse={val_running['mse']:.5f} v_align={val_running['align']:.4f} "
            f"v_cos={val_running['cosine_sim']:.3f} v_pcc={val_pcc:.4f} {'★' if is_best else ''}"
        )

    # ── Extract representations ──
    print(f"\n=== Extracting representations ===")
    model.eval()
    latent_dir = os.path.join(output_dir, "latents")
    os.makedirs(latent_dir, exist_ok=True)

    for split, loader in [("train", train_loader), ("test", val_loader)]:
        all_repr = []
        with torch.no_grad():
            for batch in loader:
                fmri = batch["fmri"].to(device)
                repr = model.encode(fmri)  # [B, 257, repr_dim]
                all_repr.append(repr.cpu().numpy())

        all_repr = np.concatenate(all_repr, axis=0)
        save_path = os.path.join(latent_dir, f"subj01_{split}_avg_repr.npy")
        np.save(save_path, all_repr)
        print(f"  Saved {split}: {all_repr.shape} → {save_path}")

    print(f"\n=== Done! Best val loss: {best_loss:.6f} ===")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
