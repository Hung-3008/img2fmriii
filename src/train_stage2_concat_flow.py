"""
Training script for Stage 2 — Flow Matching Baseline with Concat Conditioning.

Architecture:
    Noise z_0 ~ N(0, I) → BrainConcatFlowDiT → fMRI latent z_1 → VAE Decoder → fMRI

Flow Matching:
    z_t = (1 - t) * z_0 + t * z_1       [linear interpolation]
    v   = z_1 - z_0                       [target velocity]
    loss = MSE(model(z_t, t, DINOv2), v)

Usage:
    python -m src.train_stage2_concat_flow --config src/configs/subj01/stage2_concat_flow.yaml
    python -m src.train_stage2_concat_flow --config src/configs/subj01/stage2_concat_flow.yaml --debug
"""

import os
import argparse
import yaml
import logging
import csv
import time
import copy
import math

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset

from src.model.brain_concat_flow_dit import BrainConcatFlowDiT, BrainConcatFlowDiTConfig
from src.model.fmri_vit_vae import FmriViTVAE, create_fmri_vit_vae
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FmriDinoDataset(Dataset):
    """Dataset pairing fMRI with multi-layer DINOv2 features."""

    def __init__(self, fmri_path: str, dino_path: str, split: str = "train", max_samples: int = 0):
        print(f"\nFmriDinoDataset [{split}]: Loading...")

        raw_fmri = np.load(fmri_path)
        self.dino_mmap = np.load(dino_path, mmap_mode='r')

        if raw_fmri.ndim == 3:
            fmri = raw_fmri.mean(axis=1).astype(np.float32)
        elif raw_fmri.ndim == 2:
            fmri = raw_fmri.astype(np.float32)
        else:
            raise ValueError(f"Unexpected fMRI shape: {raw_fmri.shape}")
        del raw_fmri

        assert fmri.shape[0] == self.dino_mmap.shape[0], \
            f"Mismatch: fMRI {fmri.shape[0]} vs DINOv2 {self.dino_mmap.shape[0]}"

        self.fmri = fmri
        self.n_samples = fmri.shape[0]

        if max_samples > 0:
            self.fmri = self.fmri[:max_samples]
            self.n_samples = min(max_samples, self.n_samples)

        print(f"  fMRI: {self.fmri.shape}, DINOv2: {self.dino_mmap.shape} (mmap)")
        print(f"  {split}: {self.n_samples} samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()
        dino = torch.from_numpy(np.array(self.dino_mmap[idx])).float()
        return fmri, dino


# ─── Utilities ────────────────────────────────────────────────────────────────

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, max_lr):
    if epoch < warmup_epochs:
        lr = max_lr * (epoch + 1) / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = max_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def ema_update(model, ema_model, decay):
    with torch.no_grad():
        for mp, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)


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


# ─── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, vae, val_loader, device, ode_steps: int = 20):
    """
    Validate using the ODE solver: integrate from noise z_0 to fMRI latent z_1.
    """
    model.eval()
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []
    total_flow_loss = 0.0
    n_batches = 0

    for fmri, dino in val_loader:
        fmri, dino = fmri.to(device), dino.to(device)
        B = fmri.shape[0]

        # Encode target fMRI to latent z1
        z1, _, _ = vae.encode(fmri, sample_posterior=False)

        # ── Compute flow matching loss using random t ──
        z0 = torch.randn_like(z1)
        t = torch.rand(B, device=device)
        z_t = (1 - t[:, None]) * z0 + t[:, None] * z1
        v_target = z1 - z0
        v_pred = model(t, z_t, dino)
        total_flow_loss += F.mse_loss(v_pred, v_target).item()

        # ── ODE sampling from noise → z1 ──
        z_gen = model.sample(dino, n_steps=ode_steps, device=device)
        fmri_pred = vae.decode(z_gen)

        all_z_gen.append(z_gen.cpu())
        all_z_true.append(z1.cpu())
        all_pred.append(fmri_pred.cpu())
        all_true.append(fmri.cpu())
        n_batches += 1

    model.train()

    preds = torch.cat(all_pred)
    trues = torch.cat(all_true)
    z_gens = torch.cat(all_z_gen)
    z_trues = torch.cat(all_z_true)

    return {
        "val_flow_loss": total_flow_loss / max(n_batches, 1),
        "val_latent_mse": F.mse_loss(z_gens, z_trues).item(),
        "val_latent_pcc": pearson_corr_samplewise(z_gens, z_trues),
        "val_zgen_std": z_gens.std().item(),
        "val_fmri_mse": F.mse_loss(preds, trues).item(),
        "val_fmri_pcc": pearson_corr_voxelwise(preds, trues),
        "val_fmri_spcc": pearson_corr_samplewise(preds, trues),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2: Flow Matching Baseline (Concat)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # ── Data ──
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    subject = data_cfg["subject"]
    dino_suffix = data_cfg.get("dino_suffix", "dinov2_vitb14_multilayer")
    sub_num = int(subject.replace("subj", "").lstrip("0"))

    debug_n = 128 if args.debug else 0
    debug_test_n = 32 if args.debug else 0

    train_ds = FmriDinoDataset(
        fmri_path=os.path.join(root, subject, f"nsd_train_fmri_zscore_sub{sub_num}.npy"),
        dino_path=os.path.join(root, subject, f"nsd_{dino_suffix}_train_sub{sub_num}.npy"),
        split="train", max_samples=debug_n,
    )
    test_ds = FmriDinoDataset(
        fmri_path=os.path.join(root, subject, f"nsd_test_fmri_zscore_sub{sub_num}.npy"),
        dino_path=os.path.join(root, subject, f"nsd_{dino_suffix}_test_sub{sub_num}.npy"),
        split="test", max_samples=debug_test_n,
    )

    train_cfg = cfg["training"]
    bs = train_cfg["batch_size"]

    if args.debug:
        train_ds = Subset(train_ds, range(min(128, len(train_ds))))
        test_ds = Subset(test_ds, range(min(32, len(test_ds))))
        bs = min(bs, 16)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    logger.info(f"Train: {len(train_ds)}, Val: {len(test_ds)}")

    # ── Load frozen VAE ──
    vae_ckpt_path = data_cfg["vae_checkpoint"]
    vae_dir = os.path.dirname(vae_ckpt_path)
    config_path = os.path.join(vae_dir, "config.yaml")
    if not os.path.exists(config_path):
        for alt in [
            f"src/configs/{subject}/stage1_vit_vae.yaml",
            f"src/configs/exp/fmri_mlp_vae_768_{subject}.yaml",
        ]:
            if os.path.exists(alt):
                config_path = alt
                break

    with open(config_path) as f:
        vae_cfg = yaml.safe_load(f)

    if vae_cfg.get("model_type", "mlp") == "vit":
        vae = create_fmri_vit_vae(**vae_cfg["model"]).to(device).eval()
        logger.info("VAE type: ViT")
    else:
        vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
        logger.info("VAE type: MLP")

    ckpt = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    for p in vae.parameters():
        p.requires_grad = False
    logger.info(f"VAE loaded from {vae_ckpt_path}")

    # ── Build Flow Model ──
    model_cfg_dict = cfg.get("model", {})
    model_config = BrainConcatFlowDiTConfig(**model_cfg_dict)
    model = BrainConcatFlowDiT(model_config).to(device)

    param_info = model.param_count()
    logger.info(f"BrainConcatFlowDiT: {param_info['total_M']:.2f}M params")

    # ── EMA ──
    use_ema = train_cfg.get("use_ema", True)
    ema_decay = train_cfg.get("ema_decay", 0.999)
    if use_ema:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False
        logger.info("EMA enabled")
    else:
        ema_model = None

    # ── Optimizer ──
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    eval_interval = train_cfg.get("eval_interval", 5)
    ode_steps = train_cfg.get("ode_steps", 20)
    patience = train_cfg.get("patience", 0)

    # ── Output dir + CSV ──
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    fields = [
        "epoch", "train_loss", "lr",
        "val_flow_loss", "val_latent_mse", "val_latent_pcc",
        "val_zgen_std", "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
    ]
    history_path = os.path.join(output_dir, "history.csv")
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_spcc = -1.0
    patience_counter = 0

    # ── Training Loop ──
    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(optimizer, epoch - 1, num_epochs, warmup_epochs, lr)

        ep_loss = 0.0
        n_steps = 0
        t0 = time.time()

        for batch_idx, (fmri, dino) in enumerate(train_loader):
            fmri, dino = fmri.to(device), dino.to(device)
            B = fmri.shape[0]

            # Encode fMRI to latent (frozen VAE)
            with torch.no_grad():
                z1, _, _ = vae.encode(fmri, sample_posterior=False)

            # ─── Flow Matching ─────────────────────────────────────────────────
            # Sample noise and timestep
            z0 = torch.randn_like(z1)                          # z0 ~ N(0, I)
            t_val = torch.rand(B, device=device)               # t  ~ U(0, 1)

            # Linear interpolation: z_t = (1-t)*z0 + t*z1
            z_t = (1 - t_val[:, None]) * z0 + t_val[:, None] * z1

            # Target velocity (Conditional Flow Matching)
            v_target = z1 - z0

            # Forward pass: predict velocity
            v_pred = model(t_val, z_t, dino)

            # Loss
            loss = F.mse_loss(v_pred, v_target)
            # ───────────────────────────────────────────────────────────────────

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if use_ema:
                ema_update(model, ema_model, ema_decay)

            ep_loss += loss.item()
            n_steps += 1

            if batch_idx == 0 and epoch <= 5:
                logger.info(f"  [Ep{epoch} B0] flow_loss={loss.item():.6f}")

        avg_loss = ep_loss / max(n_steps, 1)
        ep_time = time.time() - t0
        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) | "
            f"flow_loss={avg_loss:.5f} | lr={current_lr:.2e}"
        )

        # ── Validation ──
        if epoch % eval_interval == 0 or epoch == 1:
            eval_model = ema_model if use_ema else model
            val = validate(eval_model, vae, val_loader, device, ode_steps=ode_steps)

            row = {
                "epoch": epoch,
                "train_loss": f"{avg_loss:.6f}",
                "lr": f"{current_lr:.2e}",
                **{k: f"{v:.6f}" for k, v in val.items()},
            }
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_spcc
            logger.info(
                f"  VAL | flow_loss={val['val_flow_loss']:.5f} "
                f"l_mse={val['val_latent_mse']:.4f} l_pcc={val['val_latent_pcc']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} f_spcc={spcc:.4f}{'  ★' if is_best else ''}"
            )

            if is_best:
                best_spcc = spcc
                patience_counter = 0
                save_dict = {
                    "epoch": epoch,
                    "model_state_dict": eval_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_spcc": best_spcc,
                    "config": cfg,
                }
                torch.save(save_dict, os.path.join(output_dir, "best.pt"))
            else:
                patience_counter += eval_interval

            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % train_cfg.get("save_every", 50) == 0:
            save_model = ema_model if use_ema else model
            torch.save(
                {"epoch": epoch, "model_state_dict": save_model.state_dict()},
                os.path.join(output_dir, f"epoch_{epoch}.pt"),
            )

    logger.info(f"Done! Best fMRI sPCC: {best_spcc:.4f}")


if __name__ == "__main__":
    main()
