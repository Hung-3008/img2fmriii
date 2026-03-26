"""Train fMRI VAE for NSD — Stage 1 latent space learning.

Trains a static MLP-based VAE to compress 15724 voxels → 256-D latent.
The learned encoder/decoder will be used by Stage-2 BrainFlow flow matching.

Uses existing NSDDataset (only fmri field needed).

Usage:
    python src/train_fmri_vae_nsd.py --config src/configs/fmri_vae_nsd.yaml --fast_dev_run
    python src/train_fmri_vae_nsd.py --config src/configs/fmri_vae_nsd.yaml
    python src/train_fmri_vae_nsd.py --config src/configs/fmri_vae_nsd.yaml --resume
"""

import argparse
import copy
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_fmri_vae")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Metrics
# =============================================================================

def pearson_corr_per_voxel(pred, target):
    """Per-voxel PCC across samples. Input: (N, V). Returns (V,)."""
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cov = (pred * target).sum(dim=0)
    std = torch.sqrt((pred ** 2).sum(dim=0) * (target ** 2).sum(dim=0))
    return cov / (std + 1e-8)


# =============================================================================
# EMA
# =============================================================================

@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


# =============================================================================
# Training
# =============================================================================

def train(args):
    cfg = load_config(args.config)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Output dir
    out_dir = PROJECT_ROOT / cfg.get("output_dir", "outputs/fmri_vae_nsd")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    # Data
    data_dir = str(PROJECT_ROOT / cfg["data_dir"])
    subject = cfg["subject"]

    from src.data.nsd_dataset import NSDDataset

    fmri_mode = cfg.get("fmri_mode", "scale")

    logger.info("Loading training data...")
    train_set = NSDDataset(data_dir=data_dir, subject=subject, mode="train", fmri_mode=fmri_mode)

    logger.info("Loading test data...")
    test_set = NSDDataset(data_dir=data_dir, subject=subject, mode="test", fmri_mode=fmri_mode)

    tr_cfg = cfg["training"]
    batch_size = tr_cfg.get("batch_size", 256)
    val_batch_size = tr_cfg.get("val_batch_size", 512)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # Model
    from src.models.fmri_vae import fMRI_VAE_NSD, beta_schedule

    vae_cfg = cfg["vae"]
    model = fMRI_VAE_NSD(
        n_voxels=cfg["n_voxels"],
        latent_dim=vae_cfg.get("latent_dim", 256),
        hidden_dim=vae_cfg.get("hidden_dim", 2048),
        num_res_blocks=vae_cfg.get("num_res_blocks", 4),
        dropout=vae_cfg.get("dropout", 0.1),
        free_bits=vae_cfg.get("free_bits", 0.5),
        lambda_pcc=vae_cfg.get("lambda_pcc", 1.0),
        pcc_warmstart_epochs=vae_cfg.get("pcc_warmstart_epochs", 0),
    ).to(device)

    ema_model = copy.deepcopy(model)
    logger.info("%s", model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # Optimizer & Scheduler
    n_epochs = tr_cfg["n_epochs"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg.get("weight_decay", 0.01),
    )

    total_steps = len(train_loader) * n_epochs
    if args.fast_dev_run:
        n_epochs = 1
        total_steps = 2

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=tr_cfg["lr"],
        total_steps=total_steps,
        pct_start=tr_cfg.get("warmup_ratio", 0.05),
        anneal_strategy="cos",
    )

    use_amp = tr_cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Hyperparams
    beta_max = vae_cfg.get("beta_max", 0.01)
    beta_warmup = vae_cfg.get("beta_warmup_epochs", 20)
    noise_std = vae_cfg.get("noise_std", 0.0)
    ema_decay = tr_cfg.get("ema_decay", 0.999)
    grad_clip = tr_cfg.get("grad_clip", 1.0)
    val_every = tr_cfg.get("val_every_n_epochs", 5)
    log_every = tr_cfg.get("log_every_n_steps", 10)

    # Resume
    start_epoch = 1
    best_pearson = -1.0
    if args.resume:
        resume_path = out_dir / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            ema_model.load_state_dict(ckpt["ema_model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_pearson = ckpt.get("best_pearson", -1.0)
            logger.info("Resumed from epoch %d (best_pcc=%.4f)",
                        ckpt["epoch"], best_pearson)
            del ckpt
        else:
            logger.warning("--resume but no last.pt found. Starting fresh.")

    # History
    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        with open(history_file, "w") as f:
            f.write("epoch,train_loss,train_recon,train_pcc,train_kl,val_loss,val_recon,val_pcc,val_kl,val_pearson,beta,lr\n")

    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info("=" * 60)
    logger.info("fMRI VAE: %d epochs | latent_dim=%d | β_max=%.4f",
                n_epochs, vae_cfg["latent_dim"], beta_max)
    logger.info("=" * 60)

    for epoch in range(start_epoch, n_epochs + 1):
        t_epoch = time.time()
        beta = beta_schedule(epoch, beta_max=beta_max, warmup_epochs=beta_warmup)
        model._current_epoch = epoch
        model.train()

        epoch_losses = {"loss": [], "recon": [], "spatial_pcc": [], "kl": []}

        pbar = tqdm(train_loader, desc=f"VAE Epoch {epoch}/{n_epochs}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            fmri = batch["fmri"].to(device)  # (B, V=15724)
            target_fmri = fmri.clone()

            # Gaussian noise augmentation
            if noise_std > 0 and model.training:
                fmri = fmri + noise_std * torch.randn_like(fmri)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                losses = model.loss(fmri, target_fmri, beta=beta)

            if use_amp:
                scaler.scale(losses["loss"]).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss"].backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            ema_update(model, ema_model, ema_decay)

            for k in epoch_losses:
                v = losses[k]
                epoch_losses[k].append(v.item() if torch.is_tensor(v) else float(v))

            if (batch_idx + 1) % log_every == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses['loss'][-50:]):.4f}",
                    "pcc": f"{np.mean(epoch_losses['spatial_pcc'][-50:]):.4f}",
                    "β": f"{beta:.4f}",
                })

        # =====================================================================
        # Validation
        # =====================================================================
        val_metrics = {"loss": 0, "recon": 0, "spatial_pcc": 0, "kl": 0, "pearson": 0}
        do_val = (epoch % val_every == 0) or args.fast_dev_run

        if do_val:
            ema_model.eval()
            all_recon = []
            all_target = []
            val_losses = {"loss": [], "recon": [], "spatial_pcc": [], "kl": []}

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break
                    fmri = batch["fmri"].to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                        losses = ema_model.loss(fmri, fmri, beta=beta)
                        recon = ema_model.reconstruct(fmri)

                    for k in val_losses:
                        v = losses[k]
                        val_losses[k].append(v.item() if torch.is_tensor(v) else float(v))

                    all_recon.append(recon.cpu().float())
                    all_target.append(fmri.cpu().float())

            if all_recon:
                all_recon = torch.cat(all_recon, dim=0)
                all_target = torch.cat(all_target, dim=0)
                pcc = pearson_corr_per_voxel(all_recon, all_target)
                val_pearson = float(pcc.mean().item())
            else:
                val_pearson = 0.0

            val_metrics = {k: float(np.mean(v)) for k, v in val_losses.items()}
            val_metrics["pearson"] = val_pearson

        # Log
        elapsed = time.time() - t_epoch
        lr = optimizer.param_groups[0]["lr"]
        train_means = {k: float(np.mean(v)) for k, v in epoch_losses.items()}

        logger.info(
            "Epoch %d/%d | β=%.4f | "
            "Train [loss=%.4f recon=%.4f pcc=%.4f kl=%.4f] | "
            "Val [loss=%.4f recon=%.4f pcc=%.4f pearson=%.4f] | "
            "LR=%.2e | %.1fs",
            epoch, n_epochs, beta,
            train_means["loss"], train_means["recon"],
            train_means["spatial_pcc"], train_means["kl"],
            val_metrics["loss"], val_metrics["recon"],
            val_metrics.get("spatial_pcc", 0), val_metrics["pearson"],
            lr, elapsed,
        )

        if do_val:
            with open(history_file, "a") as f:
                f.write(
                    f"{epoch},{train_means['loss']:.6f},{train_means['recon']:.6f},"
                    f"{train_means['spatial_pcc']:.6f},{train_means['kl']:.6f},"
                    f"{val_metrics['loss']:.6f},{val_metrics['recon']:.6f},"
                    f"{val_metrics.get('spatial_pcc', 0):.6f},{val_metrics['kl']:.6f},"
                    f"{val_metrics['pearson']:.6f},{beta:.5f},{lr:.2e}\n"
                )

        # Checkpoint
        if do_val and not args.fast_dev_run:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_pearson": best_pearson,
                "config": cfg,
            }
            torch.save(ckpt, out_dir / "last.pt")

            if val_metrics["pearson"] > best_pearson:
                best_pearson = val_metrics["pearson"]
                torch.save(ckpt, out_dir / "best.pt")
                logger.info("  ✅ New best Pearson: %.4f", best_pearson)

    logger.info("Done! Best Pearson: %.4f | Saved to: %s", best_pearson, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fMRI VAE training for NSD")
    parser.add_argument("--config", type=str, default="src/configs/fmri_vae_nsd.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt")
    args = parser.parse_args()
    train(args)
