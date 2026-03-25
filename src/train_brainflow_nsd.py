"""Train BrainFlow NSD — Flow Matching for Discrete Image→fMRI.

Simplified from train_brainflow_direct.py:
  - No H5, no video clips, no temporal windowing
  - Loads NSD .npy files directly
  - PCA fitted on train set, applied to both train/test
  - Validation: PCA inverse transform → PCC vs raw fMRI

Usage:
    python src/train_brainflow_nsd.py --config src/configs/brainflow_nsd.yaml --fast_dev_run
    python src/train_brainflow_nsd.py --config src/configs/brainflow_nsd.yaml
    python src/train_brainflow_nsd.py --config src/configs/brainflow_nsd.yaml --resume
"""

import argparse
import logging
import math as pymath
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_brainflow_nsd")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Metrics
# =============================================================================

def pearson_corr_per_dim(pred, target):
    """Per-voxel PCC across samples. Input: (N, D)."""
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cov = (pred * target).sum(dim=0)
    std = torch.sqrt((pred ** 2).sum(dim=0) * (target ** 2).sum(dim=0))
    return cov / (std + 1e-8)


# =============================================================================
# EMA
# =============================================================================

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


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
    out_dir = PROJECT_ROOT / cfg.get("output_dir", "outputs/brainflow_nsd")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "config.yaml")

    # Data
    data_dir = str(PROJECT_ROOT / cfg["data_dir"])
    subject = cfg["subject"]

    from src.data.nsd_dataset import NSDDataset

    logger.info("Loading training data...")
    train_set = NSDDataset(
        data_dir=data_dir,
        subject=subject,
        mode="train",
    )

    logger.info("Loading test data...")
    test_set = NSDDataset(
        data_dir=data_dir,
        subject=subject,
        mode="test",
    )

    tr_cfg = cfg["training"]
    batch_size = tr_cfg.get("batch_size", 512)
    val_batch_size = tr_cfg.get("val_batch_size", 1000)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Data is preloaded in RAM as tensors
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    from src.models.brain_flow_nsd import BrainFlowNSD

    nf_cfg = cfg["brainflow"]
    output_dim = nf_cfg.get("output_dim", 15724)

    dn_params = dict(nf_cfg.get("dit_net", {}))
    dn_params["modality_dims"] = [
        train_set.feat_dinov2.shape[-1],
        train_set.feat_clip.shape[-1],
        train_set.feat_qwen.shape[-1],
    ]

    model = BrainFlowNSD(
        output_dim=output_dim,
        dit_net_params=dn_params,
        n_subjects=nf_cfg.get("n_subjects", 1),
        reg_weight=nf_cfg.get("reg_weight", 1.0),
        contrastive_weight=nf_cfg.get("contrastive_weight", 0.1),
        contrastive_temp=nf_cfg.get("contrastive_temp", 0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )

    total_steps = len(train_loader) * tr_cfg["n_epochs"]
    if args.fast_dev_run:
        tr_cfg["n_epochs"] = 1
        tr_cfg["val_every_n_epochs"] = 1
        total_steps = 2

    warmup_steps = int(total_steps * tr_cfg.get("warmup_ratio", 0.05))
    min_lr = tr_cfg.get("min_lr", 1e-6)
    base_lr = tr_cfg["lr"]

    def cosine_with_warmup(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + pymath.cos(pymath.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)

    # EMA
    ema = EMAModel(model, decay=tr_cfg.get("ema_decay", 0.999))

    # Resume
    start_epoch = 1
    best_val_corr = -1.0
    global_step = 0

    if args.resume:
        resume_path = out_dir / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "ema" in ckpt:
                ema.load_state_dict(ckpt["ema"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            best_val_corr = ckpt.get("best_val_corr", -1.0)
            logger.info("Resumed from epoch %d (step=%d, best_pcc=%.4f)",
                        ckpt["epoch"], global_step, best_val_corr)
            del ckpt
        else:
            logger.warning("--resume but no last.pt found. Starting fresh.")

    history_file = out_dir / "history.csv"
    if start_epoch == 1:
        with open(history_file, "w") as f:
            f.write("epoch,total_loss,flow_loss,reg_loss,cont_loss,val_pcc,lr\n")

    # Solver config
    solver_cfg = cfg.get("solver_args", {})
    val_n_timesteps = solver_cfg.get("time_points", 50)
    val_solver_method = solver_cfg.get("method", "midpoint")
    val_cfg_scale = solver_cfg.get("cfg_scale", 0.0)

    # =====================================================================
    # Training loop
    # =====================================================================
    logger.info("Starting training: %d epochs, %d steps/epoch", tr_cfg["n_epochs"], len(train_loader))

    for epoch in range(start_epoch, tr_cfg["n_epochs"] + 1):
        model.train()
        epoch_losses = {"total": [], "flow": [], "reg": [], "cont": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tr_cfg['n_epochs']}")
        for batch_idx, batch in enumerate(pbar):
            if args.fast_dev_run and batch_idx >= 2:
                break

            context = {k: v.to(device) for k, v in batch["context"].items()}
            target = batch["fmri"].to(device)            # (B, 15724)
            subject_ids = batch["subject_idx"].to(device)

            # CFG: 10% context dropout
            if random.random() < 0.1:
                # If context is a dict, apply dropout to each tensor value
                context = {k: torch.zeros_like(v) for k, v in context.items()}

            with torch.amp.autocast("cuda", enabled=tr_cfg["use_amp"], dtype=torch.bfloat16):
                losses = model(context, target, subject_ids=subject_ids)
                loss = losses["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            epoch_losses["total"].append(loss.item())
            epoch_losses["flow"].append(losses["flow_loss"].item())
            epoch_losses["reg"].append(losses["reg_loss"].item())
            epoch_losses["cont"].append(losses["cont_loss"].item())
            global_step += 1
            ema.update(model)

            if global_step % tr_cfg["log_every_n_steps"] == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses['total'][-50:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # =====================================================================
        # Validation
        # =====================================================================
        mean_pcc = 0.0
        if epoch % tr_cfg["val_every_n_epochs"] == 0 or args.fast_dev_run:
            ema.apply_shadow(model)
            model.eval()

            all_fmri_pred = []
            all_fmri_target = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Val")):
                    if args.fast_dev_run and batch_idx >= 2:
                        break
                    context = {k: v.to(device) for k, v in batch["context"].items()}
                    fmri_target = batch["fmri"]  # (B, 15724) — raw fMRI

                    # Directly predict Regression for PCC calculation
                    fmri_pred = model.predict_regression(context)  # (B, 15724)

                    all_fmri_pred.append(fmri_pred.cpu())
                    all_fmri_target.append(fmri_target)

            if all_fmri_pred:
                all_fmri_pred = torch.cat(all_fmri_pred, dim=0).float()
                all_fmri_target = torch.cat(all_fmri_target, dim=0).float()

                pcc = pearson_corr_per_dim(all_fmri_pred, all_fmri_target)
                mean_pcc = float(pcc.mean().item())
                median_pcc = float(pcc.median().item())

            logger.info("Epoch %d | Val PCC: mean=%.4f, median=%.4f (PCA→fMRI)",
                        epoch, mean_pcc, median_pcc)

            if mean_pcc > best_val_corr:
                best_val_corr = mean_pcc
                torch.save(model.state_dict(), out_dir / "best.pt")
                logger.info("✅ New best model (PCC=%.4f)", best_val_corr)

            # Log
            mean_total = float(np.mean(epoch_losses["total"]))
            mean_flow = float(np.mean(epoch_losses["flow"]))
            mean_reg = float(np.mean(epoch_losses["reg"]))
            mean_cont = float(np.mean(epoch_losses["cont"]))
            lr = scheduler.get_last_lr()[0]
            with open(history_file, "a") as f:
                f.write(f"{epoch},{mean_total:.6f},{mean_flow:.6f},{mean_reg:.6f},{mean_cont:.6f},{mean_pcc:.6f},{lr:.2e}\n")

            ema.restore(model)

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "global_step": global_step,
            "best_val_corr": best_val_corr,
        }, out_dir / "last.pt")

    logger.info("Training complete. Best val PCC: %.4f", best_val_corr)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/configs/brainflow_nsd.yaml")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 2 batches to test pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt")
    args = parser.parse_args()
    train(args)
