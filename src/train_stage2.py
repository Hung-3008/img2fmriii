"""
Stage 2 Training: Flow Matching on fMRI latent space.

Architecture:
    fMRI (15724) → Frozen MLP VAE Encoder → z (1024) ← flow target
    DINOv2 CLS token (1024) → LatentFlowMLP condition
    Flow: noise ~ N(0,I)(1024) → z_pred (1024) → VAE Decoder → fMRI_recon

Data mapping (SynBrain convention):
    Train fMRI: (27000, V) — 3 reps × 9000 images, interleaved
    DINOv2:     (9000, 257, 1024) → CLS token → duplicated 3× → (27000, 1024)
    Test fMRI:  (3000, V) → reshape (1000, 3, V) → average → (1000, V)
    DINOv2:     (1000, 257, 1024) → CLS token → (1000, 1024)

Usage:
    python -m src.train_stage2 --config src/configs/stage2.yaml
    python -m src.train_stage2 --config src/configs/stage2.yaml --debug
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

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    ConditionalFlowMatcher,
)

from src.model.latent_flow_mlp import LatentFlowMLP, FlowMLPConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig


# ─── Dataset (SynBrain-style mapping) ────────────────────────────────────────


class FmriFeatureDataset(Dataset):
    """
    Dataset pairing fMRI samples with DINOv2 CLS tokens.

    SynBrain data mapping:
        Train: fMRI has 3 reps per image (interleaved), DINOv2 has 1 per image.
               → Duplicate DINOv2 3× to match fMRI samples.
        Test:  fMRI has 3 reps per image → average to 1 per image.
               → DINOv2 stays 1-to-1.

    Args:
        fmri_path:  path to fMRI .npy file
                    train: (N_images*3, V) flat OR (N_images, 3, V) structured
                    test:  (N_images*3, V) flat OR (N_images, 3, V) structured
        dino_path:  path to DINOv2 features .npy (N_images, 257, 1024)
        split:      'train' or 'test'
        max_samples: optional, limit samples for debug
    """

    def __init__(self, fmri_path: str, dino_path: str, split: str = "train",
                 max_samples: int = 0):
        print(f"\nFmriFeatureDataset [{split}]: Loading...")

        # ── Load fMRI ──
        raw_fmri = np.load(fmri_path)
        print(f"  Raw fMRI: {raw_fmri.shape} dtype={raw_fmri.dtype}")

        # ── Load DINOv2 CLS token ──
        raw_dino = np.load(dino_path, mmap_mode='r')
        n_images = raw_dino.shape[0]
        # Extract CLS token (index 0) → (N_images, 1024)
        dino_cls = np.array(raw_dino[:, 0, :], dtype=np.float32)
        print(f"  DINOv2 raw: {raw_dino.shape} → CLS: {dino_cls.shape}")
        del raw_dino

        if split == "train":
            # ── Train: SynBrain convention ──
            # fMRI: (N_images, 3, V) or already flat (N_images*3, V)
            if raw_fmri.ndim == 3:
                n_img, n_reps, n_voxels = raw_fmri.shape
                # Flatten: each rep is a separate sample
                fmri = raw_fmri.reshape(-1, n_voxels).astype(np.float32)
            elif raw_fmri.ndim == 2:
                fmri = raw_fmri.astype(np.float32)
                n_reps = fmri.shape[0] // n_images
            else:
                raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")

            # Duplicate DINOv2 CLS 3× to match each fMRI rep
            # [img0_cls, img0_cls, img0_cls, img1_cls, img1_cls, ...]
            dino_cls_expanded = np.repeat(dino_cls, n_reps, axis=0)

            assert fmri.shape[0] == dino_cls_expanded.shape[0], \
                f"Mismatch: fMRI {fmri.shape[0]} vs DINOv2×{n_reps} {dino_cls_expanded.shape[0]}"

            self.fmri = fmri
            self.dino = dino_cls_expanded
            print(f"  Train: {self.fmri.shape[0]} samples "
                  f"({n_images} images × {n_reps} reps)")

        elif split == "test":
            # ── Test: average 3 reps (SynBrain convention) ──
            if raw_fmri.ndim == 3:
                n_img, n_reps, n_voxels = raw_fmri.shape
                fmri = raw_fmri.mean(axis=1).astype(np.float32)
            elif raw_fmri.ndim == 2:
                # Already flat — reshape and average
                n_voxels = raw_fmri.shape[1]
                fmri = raw_fmri.reshape(-1, 3, n_voxels).mean(axis=1).astype(np.float32)
            else:
                raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")

            assert fmri.shape[0] == dino_cls.shape[0], \
                f"Mismatch: fMRI {fmri.shape[0]} vs DINOv2 {dino_cls.shape[0]}"

            self.fmri = fmri
            self.dino = dino_cls
            print(f"  Test: {self.fmri.shape[0]} samples (averaged)")

        else:
            raise ValueError(f"Unknown split: {split}")

        del raw_fmri

        # Optional sample limit
        if max_samples > 0:
            self.fmri = self.fmri[:max_samples]
            self.dino = self.dino[:max_samples]
            print(f"  Debug: limited to {max_samples} samples")

        print(f"  Final: fMRI {self.fmri.shape}, DINOv2 {self.dino.shape}")

    def __len__(self):
        return self.fmri.shape[0]

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()
        dino = torch.from_numpy(self.dino[idx]).float()
        return fmri, dino


# ─── Utilities ────────────────────────────────────────────────────────────────


def pearson_corr_voxelwise(pred, target):
    pred_zm = pred - pred.mean(dim=0, keepdim=True)
    tgt_zm = target - target.mean(dim=0, keepdim=True)
    num = (pred_zm * tgt_zm).sum(dim=0)
    den = (pred_zm.norm(dim=0) * tgt_zm.norm(dim=0)).clamp(min=1e-8)
    return (num / den).mean().item()


def pearson_corr_samplewise(pred, target):
    pred_zm = pred - pred.mean(dim=1, keepdim=True)
    tgt_zm = target - target.mean(dim=1, keepdim=True)
    num = (pred_zm * tgt_zm).sum(dim=1)
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


# ─── ODE Wrapper ──────────────────────────────────────────────────────────────


class ODEWrapper(torch.nn.Module):
    def __init__(self, model, context, cfg_scale=1.0):
        super().__init__()
        self.model = model
        self.context = context
        self.cfg_scale = cfg_scale

    def forward(self, t, z):
        B = z.shape[0]
        t_batch = t.expand(B)
        if self.cfg_scale == 1.0:
            return self.model(t_batch, z, self.context)
        else:
            return self.model.forward_with_cfg(t_batch, z, self.context, self.cfg_scale)


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, vae, val_loader, fm, device, ode_steps=50, cfg_scale=1.0,
             num_trials=1):
    """
    Validate: encode fMRI online → CFM loss + ODE solve → decode → PCC.

    Args:
        num_trials: number of ODE generations to sum together (default: 1).
    """
    from torchdiffeq import odeint

    model.eval()
    total_loss = 0
    n_batches = 0
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []

    for fmri, dino in val_loader:
        fmri = fmri.to(device)
        dino = dino.to(device)

        # Online VAE encode (deterministic: z = mu)
        z1, mu, _ = vae.encode(fmri, sample_posterior=False)

        # Use raw DINOv2 features (no L2 norm — preserves 45× more signal)
        context = dino

        # CFM loss
        x0 = torch.randn_like(z1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
        v_pred = model(t, xt, context)
        loss = F.mse_loss(v_pred, ut)
        total_loss += loss.item()
        n_batches += 1

        # ODE solve: noise → z_gen (sum over num_trials)
        ode_fn = ODEWrapper(model, context, cfg_scale)
        t_span = torch.linspace(0, 1, ode_steps, device=device)
        z_gen = torch.zeros_like(z1)
        for _ in range(num_trials):
            noise = torch.randn_like(z1)
            traj = odeint(ode_fn, noise, t_span, method="euler")
            z_gen = z_gen + traj[-1]
        if num_trials > 1:
            z_gen = z_gen / num_trials  # average for stable decoding

        # Decode → fMRI
        fmri_pred = vae.decode(z_gen)

        all_z_gen.append(z_gen)
        all_z_true.append(z1)
        all_pred.append(fmri_pred)
        all_true.append(fmri)

    model.train()

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    z_gens = torch.cat(all_z_gen)
    z_trues = torch.cat(all_z_true)

    return {
        "val_cfm_loss": total_loss / max(n_batches, 1),
        "val_latent_mse": F.mse_loss(z_gens, z_trues).item(),
        "val_latent_pcc": pearson_corr_voxelwise(z_gens, z_trues),
        "val_fmri_mse": F.mse_loss(preds, trues).item(),
        "val_fmri_pcc": pearson_corr_voxelwise(preds, trues),
        "val_fmri_sample_pcc": pearson_corr_samplewise(preds, trues),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser("Stage 2: Flow Matching")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    num_epochs = 2 if args.debug else train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["lr"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    cfg_drop_prob = train_cfg.get("cfg_drop_prob", 0.1)
    cfg_scale = train_cfg.get("cfg_scale", 1.0)
    cond_noise_std = train_cfg.get("cond_noise_std", 0.0)
    ode_steps = train_cfg.get("ode_steps", 50)
    num_trials = train_cfg.get("num_trials", 1)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    log_interval = train_cfg.get("log_interval", 10)  # log batch stats every N batches

    output_dir = cfg.get("output_dir", "results/stage2_mlp")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── Logger ──
    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ],
    )
    logger = logging.getLogger('stage2')
    logger.info(f"Config: {cfg}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Eval every {eval_interval} epochs")

    # ── Data (SynBrain-style mapping) ──
    print("\n=== Loading Data ===")
    subject = data_cfg.get("subject", "subj01")
    sub_num = int(subject.replace("subj", "").lstrip("0"))
    data_root = data_cfg["root"]

    train_fmri_path = os.path.join(data_root, subject,
                                    f"nsd_train_fmri_zscore_sub{sub_num}.npy")
    test_fmri_path = os.path.join(data_root, subject,
                                   f"nsd_test_fmri_zscore_sub{sub_num}.npy")
    train_dino_path = os.path.join(data_root, subject,
                                    f"nsd_dinov2_vitl14_train_sub{sub_num}.npy")
    test_dino_path = os.path.join(data_root, subject,
                                   f"nsd_dinov2_vitl14_test_sub{sub_num}.npy")

    debug_samples = 128 if args.debug else 0

    train_ds = FmriFeatureDataset(train_fmri_path, train_dino_path,
                                   split="train", max_samples=debug_samples)
    val_ds = FmriFeatureDataset(test_fmri_path, test_dino_path,
                                 split="test", max_samples=debug_samples // 4 if args.debug else 0)

    if args.debug:
        batch_size = min(batch_size, 32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True,
                              drop_last=(not args.debug))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    logger.info(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")

    # ── Frozen MLP VAE (for online encoding + decoding) ──
    print("\n=== Loading Frozen MLP VAE ===")
    vae_ckpt_path = data_cfg["vae_checkpoint"]
    vae_config_path = os.path.join(os.path.dirname(vae_ckpt_path), "config.yaml")
    with open(vae_config_path) as f:
        vae_cfg = yaml.safe_load(f)

    vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device)
    ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"  FmriMLPVAE loaded from {vae_ckpt_path}")
    print(f"  Params: {vae.param_count()['total']:,}")

    # ── Flow Matching Model ──
    print("\n=== Creating Flow Model ===")
    mlp_config = FlowMLPConfig(**model_cfg)
    model = LatentFlowMLP(mlp_config).to(device)
    ema_model = copy.deepcopy(model)
    print(f"  LatentFlowMLP: {model.param_count()['total_M']:.1f}M params")

    # ── Flow Matcher ──
    sigma = train_cfg.get("sigma", 0.0)
    cfm_type = train_cfg.get("cfm_type", "otcfm")
    if cfm_type == "otcfm":
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    else:
        fm = ConditionalFlowMatcher(sigma=sigma)
    print(f"  Flow Matcher: {cfm_type} (sigma={sigma})")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=train_cfg.get("weight_decay", 0.01))

    # ── History ──
    history_path = os.path.join(output_dir, "history.csv")
    fields = [
        "epoch", "train_loss", "lr", "val_cfm_loss",
        "val_latent_mse", "val_latent_pcc",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_sample_pcc"
    ]
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = -1.0
    patience_counter = 0
    patience = train_cfg.get("patience", 100)

    # ── Training ──
    logger.info(f"Training for {num_epochs} epochs, eval every {eval_interval}...")
    logger.info(f"  batch_size={batch_size} lr={lr} grad_clip={grad_clip}")
    logger.info(f"  ema_decay={ema_decay} cfg_drop={cfg_drop_prob} cond_noise={cond_noise_std}")
    logger.info(f"  cfm={cfm_type} sigma={sigma} ode_steps={ode_steps} num_trials={num_trials}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(optimizer, epoch - 1, num_epochs, warmup_epochs, lr)
        epoch_loss = 0
        n_steps = 0
        batch_grad_norms = []
        t0 = time.time()

        for batch_idx, (fmri, dino) in enumerate(train_loader):
            fmri = fmri.to(device)
            dino = dino.to(device)
            B = fmri.shape[0]

            # ── Online VAE encode (frozen, deterministic) ──
            with torch.no_grad():
                z1, mu, _ = vae.encode(fmri, sample_posterior=False)

            # ── Prepare conditioning (raw DINOv2, no L2 norm) ──
            context = dino

            # CFG dropout
            if cfg_drop_prob > 0:
                drop_mask = torch.rand(B, device=device) < cfg_drop_prob
                if drop_mask.any():
                    context = context.clone()
                    context[drop_mask] = 0.0

            # Condition noise injection
            if cond_noise_std > 0:
                noise = torch.randn_like(context) * cond_noise_std
                if cfg_drop_prob > 0:
                    noise[drop_mask] = 0.0
                context = context + noise

            # ── Flow matching ──
            x0 = torch.randn_like(z1)
            t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
            v_pred = model(t, xt, context)
            loss = F.mse_loss(v_pred, ut)

            optimizer.zero_grad()
            loss.backward()

            # Track gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0 else float('inf'))

            optimizer.step()
            ema_update(model, ema_model, ema_decay)

            epoch_loss += loss.item()
            batch_grad_norms.append(grad_norm.item())
            n_steps += 1

            # ── Batch-level debug logging ──
            if batch_idx == 0 and epoch <= 3:
                # First batch of first few epochs: detailed diagnostics
                with torch.no_grad():
                    v_cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
                logger.info(
                    f"  [Ep{epoch} B0 diag] z1: mean={z1.mean():.4f} std={z1.std():.4f} | "
                    f"ctx: mean={context.mean():.4f} std={context.std():.4f} norm={context.norm(dim=-1).mean():.2f} | "
                    f"v_pred: std={v_pred.std():.4f} | ut: std={ut.std():.4f} | "
                    f"v·ut_cos={v_cos:.4f} | t: mean={t.mean():.3f}"
                )

        avg_loss = epoch_loss / max(n_steps, 1)
        avg_grad = sum(batch_grad_norms) / len(batch_grad_norms)
        max_grad = max(batch_grad_norms)
        ep_time = time.time() - t0

        # ── Always log train stats ──
        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) | "
            f"loss={avg_loss:.5f} lr={current_lr:.2e} | "
            f"grad: avg={avg_grad:.4f} max={max_grad:.4f}"
        )

        # ── Eval + history (every eval_interval epochs) ──
        if epoch % eval_interval == 0 or epoch == 1:
            val = validate(ema_model, vae, val_loader, fm, device, ode_steps, cfg_scale,
                           num_trials=num_trials)

            row = {
                "epoch": epoch, "train_loss": f"{avg_loss:.6f}", "lr": f"{current_lr:.2e}",
                "val_cfm_loss": f"{val['val_cfm_loss']:.6f}",
                "val_latent_mse": f"{val['val_latent_mse']:.6f}",
                "val_latent_pcc": f"{val['val_latent_pcc']:.4f}",
                "val_fmri_mse": f"{val['val_fmri_mse']:.6f}",
                "val_fmri_pcc": f"{val['val_fmri_pcc']:.4f}",
                "val_fmri_sample_pcc": f"{val['val_fmri_sample_pcc']:.4f}",
            }
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            vpcc = val["val_fmri_pcc"]
            spcc = val["val_fmri_sample_pcc"]
            is_best = spcc > best_pcc
            logger.info(
                f"  VAL | cfm={val['val_cfm_loss']:.5f} "
                f"l_mse={val['val_latent_mse']:.4f} l_pcc={val['val_latent_pcc']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} f_vpcc={vpcc:.4f} f_spcc={spcc:.4f}"
                f"{' ★' if is_best else ''}"
            )

            if is_best:
                best_pcc = spcc
                patience_counter = 0
                torch.save({
                    "epoch": epoch, "model_state_dict": ema_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_pcc": best_pcc, "config": cfg,
                }, os.path.join(output_dir, "best_model.pt"))
                logger.info(f"  ★ Saved best model (PCC={best_pcc:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

        if epoch % train_cfg.get("save_every", 25) == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_pcc": best_pcc, "config": cfg,
            }, os.path.join(output_dir, "latest.pt"))

    logger.info(f"Done! Best PCC: {best_pcc:.4f}")


if __name__ == "__main__":
    main()
