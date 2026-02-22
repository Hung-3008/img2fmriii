"""
Stage 2: Direct Regression — DINOv2 → fMRI latent.

Cross-Attention Regressor: learnable queries cross-attend to DINOv2 patches.
No flow matching, no ODE. Pure regression with MSE + optional contrastive loss.

Usage:
    python -m src.train_stage2_regressor --config src/configs/stage2_regressor.yaml
    python -m src.train_stage2_regressor --config src/configs/stage2_regressor.yaml --debug
"""

import argparse
import copy
import csv
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.model.cross_attn_regressor import CrossAttnRegressor, CrossAttnRegressorConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig


# ─── Dataset ──────────────────────────────────────────────────────────────────


class FmriFeatureDataset(Dataset):
    """Dataset pairing fMRI with DINOv2 full tokens (257, 1024)."""

    def __init__(self, fmri_path, dino_path, split="train", max_samples=0):
        print(f"\nFmriFeatureDataset [{split}]: Loading...")
        raw_fmri = np.load(fmri_path)
        self.dino_mmap = np.load(dino_path, mmap_mode='r')
        n_images = self.dino_mmap.shape[0]

        if raw_fmri.ndim == 3:
            fmri = raw_fmri.mean(axis=1).astype(np.float32)
        elif raw_fmri.ndim == 2:
            fmri = raw_fmri.astype(np.float32)
        else:
            raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")
        del raw_fmri

        assert fmri.shape[0] == n_images, \
            f"Mismatch: fMRI {fmri.shape[0]} vs DINOv2 {n_images}"
        self.fmri = fmri
        self.n_samples = fmri.shape[0]

        if max_samples > 0:
            self.fmri = self.fmri[:max_samples]
            self.n_samples = min(max_samples, self.n_samples)
        print(f"  {split}: {self.n_samples} samples, fMRI {self.fmri.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()
        dino = torch.from_numpy(np.array(self.dino_mmap[idx])).float()
        return fmri, dino


# ─── Utilities ────────────────────────────────────────────────────────────────


def pearson_corr_voxelwise(pred, target):
    pred_zm = pred - pred.mean(0, keepdim=True)
    tgt_zm = target - target.mean(0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


def pearson_corr_samplewise(pred, target):
    pred_zm = pred - pred.mean(1, keepdim=True)
    tgt_zm = target - target.mean(1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(1)
    den = (pred_zm.norm(dim=1) * tgt_zm.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def ema_update(source, target, decay):
    with torch.no_grad():
        for s, t in zip(source.parameters(), target.parameters()):
            t.data.mul_(decay).add_(s.data, alpha=1 - decay)


def cosine_lr(optimizer, epoch, total, warmup, base_lr, min_lr=1e-6):
    if epoch < warmup:
        lr = base_lr * epoch / max(warmup, 1)
    else:
        p = (epoch - warmup) / max(total - warmup, 1)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def info_nce_loss(z_a, z_b, temperature=0.07):
    """Symmetric InfoNCE (CLIP-style)."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, acc


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, vae, val_loader, device):
    model.eval()
    all_z_pred, all_z_true = [], []
    all_pred, all_true = [], []

    for fmri, dino in val_loader:
        fmri, dino = fmri.to(device), dino.to(device)
        z1, _, _ = vae.encode(fmri, sample_posterior=False)
        z_pred = model(dino)

        fmri_pred = vae.decode(z_pred)

        all_z_pred.append(z_pred)
        all_z_true.append(z1)
        all_pred.append(fmri_pred)
        all_true.append(fmri)

    model.train()

    z_preds = torch.cat(all_z_pred)
    z_trues = torch.cat(all_z_true)
    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)

    return {
        "val_latent_mse": F.mse_loss(z_preds, z_trues).item(),
        "val_latent_pcc": pearson_corr_samplewise(z_preds, z_trues),
        "val_fmri_mse": F.mse_loss(preds, trues).item(),
        "val_fmri_pcc": pearson_corr_voxelwise(preds, trues),
        "val_fmri_spcc": pearson_corr_samplewise(preds, trues),
        "val_zpred_std": z_preds.std().item(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser("Stage 2: Cross-Attention Regressor")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    num_epochs = 2 if args.debug else train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    contrastive_weight = train_cfg.get("contrastive_weight", 0.0)
    contrastive_temp = train_cfg.get("contrastive_temp", 0.07)

    output_dir = cfg.get("output_dir", "results/stage2_regressor")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Logger ──
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode='w')],
    )
    logger = logging.getLogger('stage2_reg')
    logger.info(f"Config: {cfg}")

    # ── Data ──
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    root = data_cfg["root"]

    debug_n = 128 if args.debug else 0
    train_ds = FmriFeatureDataset(
        os.path.join(root, subject, f"nsd_train_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_dinov2_vitl14_train_sub{sub_num}.npy"),
        split="train", max_samples=debug_n)
    val_ds = FmriFeatureDataset(
        os.path.join(root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        os.path.join(root, subject, f"nsd_dinov2_vitl14_test_sub{sub_num}.npy"),
        split="test", max_samples=debug_n // 4 if args.debug else 0)

    if args.debug:
        batch_size = min(batch_size, 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=(not args.debug))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Frozen VAE ──
    vae_ckpt = data_cfg["vae_checkpoint"]
    with open(os.path.join(os.path.dirname(vae_ckpt), "config.yaml")) as f:
        vae_cfg = yaml.safe_load(f)
    vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
    ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    for p in vae.parameters():
        p.requires_grad = False
    logger.info(f"VAE loaded from {vae_ckpt}")

    # ── Model ──
    model = CrossAttnRegressor(CrossAttnRegressorConfig(**model_cfg)).to(device)
    ema_model = copy.deepcopy(model)
    pc = model.param_count()
    logger.info(f"CrossAttnRegressor: {pc['total_M']:.2f}M params")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=train_cfg.get("weight_decay", 0.01))

    # ── History ──
    history_path = os.path.join(output_dir, "history.csv")
    fields = [
        "epoch", "train_loss", "mse_loss", "nce_loss", "nce_acc", "lr",
        "grad_avg",
        "val_latent_mse", "val_latent_pcc",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
        "val_zpred_std",
    ]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = -1.0
    patience_counter = 0
    patience = train_cfg.get("patience", 100)

    # ── Training ──
    logger.info(f"Training {num_epochs} epochs, eval every {eval_interval}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(optimizer, epoch - 1, num_epochs, warmup_epochs, lr)
        ep_loss, ep_mse, ep_nce, ep_acc, n_steps = 0, 0, 0, 0, 0
        grads = []
        t0 = time.time()

        for fmri, dino in train_loader:
            fmri, dino = fmri.to(device), dino.to(device)

            with torch.no_grad():
                z1, _, _ = vae.encode(fmri, sample_posterior=False)

            z_pred = model(dino)
            loss_mse = F.mse_loss(z_pred, z1)

            if contrastive_weight > 0:
                loss_nce, nce_acc = info_nce_loss(z_pred, z1.detach(),
                                                   temperature=contrastive_temp)
                loss = loss_mse + contrastive_weight * loss_nce
            else:
                loss_nce = torch.tensor(0.0)
                nce_acc = 0.0
                loss = loss_mse

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 grad_clip if grad_clip > 0 else float('inf'))
            optimizer.step()
            ema_update(model, ema_model, ema_decay)

            ep_loss += loss.item()
            ep_mse += loss_mse.item()
            ep_nce += loss_nce.item()
            ep_acc += nce_acc
            grads.append(gn.item())
            n_steps += 1

        avg_loss = ep_loss / max(n_steps, 1)
        avg_mse = ep_mse / max(n_steps, 1)
        avg_nce = ep_nce / max(n_steps, 1)
        avg_acc = ep_acc / max(n_steps, 1)
        avg_grad = sum(grads) / len(grads)
        ep_time = time.time() - t0

        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) | "
            f"loss={avg_loss:.5f} mse={avg_mse:.5f} "
            f"nce={avg_nce:.4f} acc={avg_acc:.3f} "
            f"lr={current_lr:.2e} grad={avg_grad:.4f}")

        # ── Eval ──
        if epoch % eval_interval == 0 or epoch == 1:
            val = validate(ema_model, vae, val_loader, device)

            row = {"epoch": epoch,
                   "train_loss": f"{avg_loss:.6f}",
                   "mse_loss": f"{avg_mse:.6f}",
                   "nce_loss": f"{avg_nce:.6f}",
                   "nce_acc": f"{avg_acc:.4f}",
                   "lr": f"{current_lr:.2e}",
                   "grad_avg": f"{avg_grad:.4f}",
                   **{k: f"{v:.6f}" for k, v in val.items()}}
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_pcc
            logger.info(
                f"  VAL | l_mse={val['val_latent_mse']:.4f} "
                f"l_pcc={val['val_latent_pcc']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} "
                f"f_vpcc={val['val_fmri_pcc']:.4f} "
                f"f_spcc={spcc:.4f} | "
                f"z_std={val['val_zpred_std']:.4f}"
                f"{' ★' if is_best else ''}")

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                torch.save({"epoch": epoch,
                             "model_state_dict": ema_model.state_dict(),
                             "optimizer_state_dict": optimizer.state_dict(),
                             "best_pcc": best_pcc, "config": cfg},
                            os.path.join(output_dir, "best_model.pt"))
                logger.info(f"  ★ Saved best (PCC={best_pcc:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        if epoch % train_cfg.get("save_every", 50) == 0:
            torch.save({"epoch": epoch,
                         "model_state_dict": model.state_dict(),
                         "ema_state_dict": ema_model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "best_pcc": best_pcc, "config": cfg},
                        os.path.join(output_dir, "latest.pt"))

    logger.info(f"Done! Best PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()
