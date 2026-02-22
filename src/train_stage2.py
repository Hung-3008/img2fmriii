"""
Stage 2 Training: Conditional Flow Matching on fMRI latent space.

Architecture:
    x₀ ~ N(0, σ)  →  CrossSiT v_θ(t, x_t | DINOv2 patches)  →  z_fmri
    z_fmri → Frozen VAE Decoder → fMRI_recon

Pure conditional flow matching — no alignment module.
DINOv2 patches provide all image conditioning via cross-attention.

Data: Average 3 reps for training (one-to-one: 1 DINOv2 → 1 fMRI).

Usage:
    python -m src.train_stage2 --config src/configs/stage2_mlp.yaml
    python -m src.train_stage2 --config src/configs/stage2_mlp.yaml --debug
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

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from src.model.aligned_flow_mlp import AlignedFlowSiT, AlignedFlowSiTConfig
from src.model.fmri_mlp_vae import FmriMLPVAE, FmriMLPVAEConfig
from src.utils.roi_utils import ROIDecomposer


# ─── Dataset ──────────────────────────────────────────────────────────────────


class FmriFeatureDataset(Dataset):
    """
    Dataset pairing fMRI with DINOv2 full tokens (257, 1024).

    Both train and test: average 3 fMRI reps → one-to-one mapping.
    DINOv2 tokens include CLS (idx 0) + 256 patch tokens.
    """

    def __init__(self, fmri_path, dino_path, split="train", max_samples=0):
        print(f"\nFmriFeatureDataset [{split}]: Loading...")

        raw_fmri = np.load(fmri_path)
        print(f"  Raw fMRI: {raw_fmri.shape} dtype={raw_fmri.dtype}")

        # Load full DINOv2 tokens (N, 257, 1024)
        self.dino_mmap = np.load(dino_path, mmap_mode='r')
        n_images = self.dino_mmap.shape[0]
        print(f"  DINOv2: {self.dino_mmap.shape} (full tokens, mmap)")

        # Average 3 reps for both train and test
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
        print(f"  {split}: {self.n_samples} samples (1-to-1, averaged reps)")

        if max_samples > 0:
            self.fmri = self.fmri[:max_samples]
            self.n_samples = min(max_samples, self.n_samples)
            print(f"  Debug: limited to {max_samples}")

        print(f"  Final: fMRI {self.fmri.shape}, DINOv2 ({self.n_samples}, 257, 1024)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        fmri = torch.from_numpy(self.fmri[idx]).float()
        dino = torch.from_numpy(np.array(self.dino_mmap[idx])).float()
        return fmri, dino


# ─── Utilities ────────────────────────────────────────────────────────────────


def info_nce_loss(z_a, z_b, temperature=0.07):
    """Symmetric InfoNCE (CLIP-style) between z_a and z_b.
    z_a[i] should match z_b[i] (positive pair), all other j are negatives."""
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature    # (B, B)
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    # Retrieval accuracy for logging
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, acc


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
            return self.model.forward_flow(t_batch, z, self.context)
        else:
            return self.model.forward_flow_with_cfg(
                t_batch, z, self.context, self.cfg_scale)


# ─── Validation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def validate(model, vae, val_loader, fm, device, ode_steps=50,
             cfg_scale=1.0, num_trials=3, prior_sigma=1.0,
             decomposer=None):
    from torchdiffeq import odeint

    model.eval()
    total_flow_loss = 0
    n_batches = 0
    all_pred, all_true = [], []
    all_z_gen, all_z_true = [], []
    all_v_cos = []

    for fmri, dino in val_loader:
        fmri, dino = fmri.to(device), dino.to(device)

        # VAE encode (target)
        z1, _, _ = vae.encode(fmri, sample_posterior=False)

        # Flow loss (Gaussian prior)
        x0 = prior_sigma * torch.randn_like(z1)
        t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
        v_pred = model.forward_flow(t, xt, dino)
        flow_loss = F.mse_loss(v_pred, ut)
        total_flow_loss += flow_loss.item()
        n_batches += 1

        # Velocity cosine similarity
        cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
        all_v_cos.append(cos)

        # ODE solve: N(0,σ) → z_gen (multi-trial)
        ode_fn = ODEWrapper(model, dino, cfg_scale)
        t_span = torch.linspace(0, 1, ode_steps, device=device)
        z_gen = torch.zeros_like(z1)
        for _ in range(num_trials):
            x0_trial = prior_sigma * torch.randn_like(z1)
            traj = odeint(ode_fn, x0_trial, t_span, method="euler")
            z_gen = z_gen + traj[-1]
        z_gen = z_gen / num_trials

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

    # z_gen distribution stats
    z_gen_std = z_gens.std().item()
    z_gen_cross_var = z_gens.var(dim=0).mean().item()
    z_true_cross_var = z_trues.var(dim=0).mean().item()

    metrics = {
        "val_flow_loss": total_flow_loss / max(n_batches, 1),
        "val_v_cos": sum(all_v_cos) / len(all_v_cos),
        "val_latent_mse": F.mse_loss(z_gens, z_trues).item(),
        "val_latent_pcc": pearson_corr_samplewise(z_gens, z_trues),
        "val_zgen_std": z_gen_std,
        "val_zgen_crossvar_ratio": z_gen_cross_var / max(z_true_cross_var, 1e-8),
        "val_fmri_mse": F.mse_loss(preds, trues).item(),
        "val_fmri_pcc": pearson_corr_voxelwise(preds, trues),
        "val_fmri_spcc": pearson_corr_samplewise(preds, trues),
    }

    # Per-ROI metrics
    if decomposer is not None:
        for roi in decomposer.rois:
            if roi.n_voxels > 10:
                p = preds[:, roi.indices]
                t = trues[:, roi.indices]
                metrics[f"roi_{roi.name}_spcc"] = pearson_corr_samplewise(p, t)
            else:
                metrics[f"roi_{roi.name}_spcc"] = 0.0

    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser("Stage 2: Aligned Flow Matching")
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
    cfg_drop_prob = train_cfg.get("cfg_drop_prob", 0.1)
    cfg_scale = train_cfg.get("cfg_scale", 1.0)
    ode_steps = train_cfg.get("ode_steps", 50)
    num_trials = train_cfg.get("num_trials", 1)
    eval_interval = 1 if args.debug else train_cfg.get("eval_interval", 5)
    prior_sigma = train_cfg.get("prior_sigma", 1.0)

    output_dir = cfg.get("output_dir", "results/stage2_aligned_flow")
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
    logger = logging.getLogger('stage2')
    logger.info(f"Config: {cfg}")

    # ── ROI Decomposer ──
    roi_dir = data_cfg.get("roi_dir",
                           "Data/nsddata/ppdata/subj01/func1pt8mm/roi")
    decomposer = ROIDecomposer(roi_dir)
    roi_names = decomposer.get_roi_names()
    logger.info(f"\n{decomposer.summary()}")

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
    logger.info(f"Train: {len(train_ds)} samples (1-to-1, averaged)")
    logger.info(f"Val:   {len(val_ds)} samples")

    # ── Frozen VAE ──
    vae_ckpt = data_cfg["vae_checkpoint"]
    with open(os.path.join(os.path.dirname(vae_ckpt), "config.yaml")) as f:
        vae_cfg = yaml.safe_load(f)
    vae = FmriMLPVAE(FmriMLPVAEConfig(**vae_cfg["model"])).to(device).eval()
    ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    for p in vae.parameters():
        p.requires_grad = False
    logger.info(f"VAE loaded from {vae_ckpt} (epoch {ckpt.get('epoch','?')})")

    # ── Model ──
    model = AlignedFlowSiT(AlignedFlowSiTConfig(**model_cfg)).to(device)
    ema_model = copy.deepcopy(model)
    pc = model.param_count()
    logger.info(f"AlignedFlowSiT: align={pc['align_M']:.1f}M flow={pc['flow_M']:.1f}M total={pc['total_M']:.1f}M")

    # ── Flow Matcher ──
    fm = ConditionalFlowMatcher(sigma=train_cfg.get("sigma", 0.0))

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=train_cfg.get("weight_decay", 0.01))

    # ── History ──
    history_path = os.path.join(output_dir, "history.csv")
    roi_fields = [f"roi_{n}_spcc" for n in roi_names]
    fields = [
        "epoch", "train_loss", "lr",
        "grad_avg", "grad_max",
        "val_flow_loss", "val_v_cos",
        "val_latent_mse", "val_latent_pcc",
        "val_zgen_std", "val_zgen_crossvar_ratio",
        "val_fmri_mse", "val_fmri_pcc", "val_fmri_spcc",
    ] + roi_fields
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_pcc = -1.0
    patience_counter = 0
    patience = train_cfg.get("patience", 100)

    # ── Training ──
    logger.info(f"Training {num_epochs} epochs, eval every {eval_interval}")
    logger.info(f"  prior_sigma={prior_sigma} cfg_drop={cfg_drop_prob}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        current_lr = cosine_lr(optimizer, epoch - 1, num_epochs, warmup_epochs, lr)
        ep_flow, n_steps = 0, 0
        grads_all, grads_max_all = [], []
        t0 = time.time()
        ep_v_cos, ep_ut_norm = [], []

        for batch_idx, (fmri, dino) in enumerate(train_loader):
            fmri, dino = fmri.to(device), dino.to(device)
            B = fmri.shape[0]

            # ── VAE encode (frozen) → z₁ target ──
            with torch.no_grad():
                z1, _, _ = vae.encode(fmri, sample_posterior=False)

            # ── Flow: N(0,σ) → z₁, conditioned on DINOv2 ──
            context = dino

            # CFG dropout
            if cfg_drop_prob > 0:
                drop = torch.rand(B, device=device) < cfg_drop_prob
                if drop.any():
                    context = context.clone()
                    context[drop] = 0.0

            x0 = prior_sigma * torch.randn_like(z1)   # Gaussian prior
            t, xt, ut = fm.sample_location_and_conditional_flow(x0, z1)
            v_pred = model.forward_flow(t, xt, context)
            loss = F.mse_loss(v_pred, ut)

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                 grad_clip if grad_clip > 0 else float('inf'))
            optimizer.step()
            ema_update(model, ema_model, ema_decay)

            ep_flow += loss.item()
            grads_all.append(gn.item())
            grads_max_all.append(gn.item())
            n_steps += 1

            with torch.no_grad():
                v_cos = F.cosine_similarity(v_pred, ut, dim=-1).mean().item()
                ep_v_cos.append(v_cos)
                ep_ut_norm.append(ut.norm(dim=-1).mean().item())

            # Diagnostics for early epochs
            if batch_idx == 0 and epoch <= 5:
                with torch.no_grad():
                    z1_norm = z1.norm(dim=-1).mean().item()
                    v_pred_norm = v_pred.norm(dim=-1).mean().item()
                    ut_norm = ut.norm(dim=-1).mean().item()
                logger.info(
                    f"  [Ep{epoch} B0] z1_norm={z1_norm:.2f} | "
                    f"v_pred_norm={v_pred_norm:.2f} ut_norm={ut_norm:.2f} | "
                    f"v·ut_cos={v_cos:.4f} flow={loss.item():.4f}")

        avg_flow = ep_flow / max(n_steps, 1)
        avg_grad = sum(grads_all) / len(grads_all)
        max_grad = max(grads_max_all)
        ep_time = time.time() - t0

        logger.info(
            f"Ep {epoch:4d}/{num_epochs} ({ep_time:.1f}s) | "
            f"loss={avg_flow:.5f} lr={current_lr:.2e} | "
            f"grad: avg={avg_grad:.4f} max={max_grad:.4f} | "
            f"v_cos={sum(ep_v_cos)/len(ep_v_cos):.4f} "
            f"ut_norm={sum(ep_ut_norm)/len(ep_ut_norm):.2f}")

        # ── Eval ──
        if epoch % eval_interval == 0 or epoch == 1:
            val = validate(ema_model, vae, val_loader, fm, device,
                           ode_steps, cfg_scale, num_trials, prior_sigma,
                           decomposer=decomposer)

            row = {"epoch": epoch,
                   "train_loss": f"{avg_flow:.6f}",
                   "lr": f"{current_lr:.2e}",
                   "grad_avg": f"{avg_grad:.4f}",
                   "grad_max": f"{max_grad:.4f}",
                   **{k: f"{v:.6f}" if ('loss' in k or 'mse' in k or 'std' in k or 'ratio' in k)
                      else f"{v:.4f}" for k, v in val.items()}}
            with open(history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)

            spcc = val["val_fmri_spcc"]
            is_best = spcc > best_pcc
            logger.info(
                f"  VAL | v_cos={val['val_v_cos']:.4f} | "
                f"l_mse={val['val_latent_mse']:.4f} l_pcc={val['val_latent_pcc']:.4f} | "
                f"f_mse={val['val_fmri_mse']:.4f} f_vpcc={val['val_fmri_pcc']:.4f} "
                f"f_spcc={spcc:.4f} | "
                f"zgen: std={val['val_zgen_std']:.4f} var_ratio={val['val_zgen_crossvar_ratio']:.4f}"
                f"{' ★' if is_best else ''}")
            # Log per-ROI PCC
            roi_str = " | ".join(
                f"{n}={val.get(f'roi_{n}_spcc', 0):.3f}" for n in roi_names)
            logger.info(f"  ROI | {roi_str}")

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
