"""
eval_csfm_fmri.py
=================
Evaluation script for CSFM-based fMRI synthesis.

Loads a trained checkpoint, runs ODE inference on the test set,
and computes metrics: per-voxel Pearson r, profile Pearson r, MSE.
"""

import argparse
import logging
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import stats
from torch import autocast
from torch.utils.data import DataLoader

# --- CSFM imports ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CSFM_SRC = os.path.join(PROJECT_ROOT, "reproduces", "CSFM", "src")
if CSFM_SRC not in sys.path:
    sys.path.insert(0, CSFM_SRC)

from stage2.transport import create_transport, Sampler, ModelType
from stage2 import Wrapper

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from data.csfm_fmri_dataset import CSFMfMRIDataset

logging.basicConfig(
    level=logging.INFO,
    format="[\033[34m%(asctime)s\033[0m] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_obj_from_str(string):
    module_path, cls_name = string.rsplit(".", 1)
    return getattr(__import__(module_path, fromlist=[cls_name]), cls_name)


def instantiate_from_config(config):
    target = config["target"]
    params = OmegaConf.to_container(config.get("params", {}), resolve=True)
    return get_obj_from_str(target)(**params)


@torch.no_grad()
def evaluate(args):
    # --- Config ---
    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    transport_cfg = OmegaConf.to_container(cfg.transport.get("params", {}), resolve=True)
    sampler_cfg = OmegaConf.to_container(cfg.sampler, resolve=True)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = cfg.training.get("precision", "fp32") == "bf16"
    autocast_kwargs = dict(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=use_bf16)

    # --- Data ---
    test_ds = CSFMfMRIDataset(
        data_dir=data_cfg["data_dir"],
        subject=data_cfg["subject"],
        mode="test",
        fmri_mode=data_cfg["fmri_mode"],
        clip_feature=data_cfg["clip_feature"],
        n_voxels=data_cfg["n_voxels"],
        pad_to=data_cfg["pad_to"],
        fmri_channels=data_cfg["fmri_channels"],
        fmri_spatial=data_cfg["fmri_spatial"],
    )
    n_voxels = data_cfg["n_voxels"]
    pad_to = data_cfg["pad_to"]

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --- Pad mask ---
    pad_mask = torch.zeros(pad_to, dtype=torch.bool, device=device)
    pad_mask[:n_voxels] = True

    # --- Model ---
    dit = instantiate_from_config(cfg.stage_2)
    source_encoder = instantiate_from_config(cfg.source_encoder)
    wrapper = Wrapper(dit=dit, source_encoder=source_encoder).to(device)

    # Load checkpoint (use EMA weights)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_key = "ema" if "ema" in ckpt else "model"
    wrapper.load_state_dict(ckpt[state_key], strict=False)
    logger.info(f"Loaded {state_key} from {args.ckpt}")
    del ckpt
    if device == "cuda":
        torch.cuda.empty_cache()

    wrapper.eval()

    # --- Transport & sampler ---
    fmri_spatial = data_cfg["fmri_spatial"]
    fmri_channels = data_cfg["fmri_channels"]
    latent_size = (fmri_channels, fmri_spatial, fmri_spatial)
    shift_dim = math.prod(latent_size)
    shift_base = transport_cfg.pop("time_dist_shift", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    transport = create_transport(**transport_cfg, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    if sampler_mode == "ODE":
        eval_sampler_fn = transport_sampler.sample_ode(**sampler_params)
    else:
        raise NotImplementedError(f"Sampler mode {sampler_mode}")

    # --- Inference ---
    all_preds = []
    all_targets = []

    logger.info(f"Running inference on {len(test_ds)} test samples...")

    for batch in test_loader:
        clip_tokens = batch["clip_tokens"].to(device)
        clip_pool = batch["clip_pool"].to(device)
        fmri_gt = batch["fmri"].to(device)
        B = clip_tokens.shape[0]

        # Source from PerceiverVE
        x0_tok, _, _ = wrapper.source_encoder(text_tokens=clip_tokens)
        x0 = x0_tok.permute(0, 2, 1).contiguous().view(B, *latent_size)

        # ODE sampling
        with autocast(**autocast_kwargs):
            traj = eval_sampler_fn(
                x0,
                wrapper.dit.forward,
                y=clip_pool,
            )
        pred = traj[-1].float()  # (B, C, H, W)

        # Flatten and unpad
        pred_flat = pred.reshape(B, -1)[:, :n_voxels]   # (B, V)
        gt_flat = fmri_gt.reshape(B, -1)[:, :n_voxels]  # (B, V)

        all_preds.append(pred_flat.cpu().numpy())
        all_targets.append(gt_flat.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)    # (N, V)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, V)

    logger.info(f"Predictions: {all_preds.shape}, Targets: {all_targets.shape}")

    # --- Metrics ---
    # 1. Per-voxel Pearson r: for each voxel, correlate across samples
    n_samples, V = all_preds.shape
    voxel_r = np.zeros(V, dtype=np.float64)
    for v in range(V):
        r, _ = stats.pearsonr(all_preds[:, v], all_targets[:, v])
        voxel_r[v] = r if np.isfinite(r) else 0.0

    mean_voxel_r = np.mean(voxel_r)
    median_voxel_r = np.median(voxel_r)

    # 2. Profile Pearson r: for each sample, correlate across voxels
    profile_r = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        r, _ = stats.pearsonr(all_preds[i], all_targets[i])
        profile_r[i] = r if np.isfinite(r) else 0.0

    mean_profile_r = np.mean(profile_r)

    # 3. MSE
    mse = np.mean((all_preds - all_targets) ** 2)

    # 4. Image-level metrics (average reps for same image)
    n_reps = test_ds.n_reps
    n_images = test_ds.n_images
    preds_img = all_preds.reshape(n_images, n_reps, V).mean(axis=1)   # (N_img, V)
    targets_img = all_targets.reshape(n_images, n_reps, V).mean(axis=1)

    img_profile_r = np.zeros(n_images, dtype=np.float64)
    for i in range(n_images):
        r, _ = stats.pearsonr(preds_img[i], targets_img[i])
        img_profile_r[i] = r if np.isfinite(r) else 0.0

    mean_img_profile_r = np.mean(img_profile_r)

    img_voxel_r = np.zeros(V, dtype=np.float64)
    for v in range(V):
        r, _ = stats.pearsonr(preds_img[:, v], targets_img[:, v])
        img_voxel_r[v] = r if np.isfinite(r) else 0.0

    mean_img_voxel_r = np.mean(img_voxel_r)

    # --- Report ---
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Single-trial metrics (N={n_samples}):")
    logger.info(f"    Per-voxel Pearson r (mean):   {mean_voxel_r:.4f}")
    logger.info(f"    Per-voxel Pearson r (median): {median_voxel_r:.4f}")
    logger.info(f"    Profile Pearson r (mean):     {mean_profile_r:.4f}")
    logger.info(f"    MSE:                          {mse:.6f}")
    logger.info(f"  Image-level metrics (N={n_images}, rep-averaged):")
    logger.info(f"    Per-voxel Pearson r (mean):   {mean_img_voxel_r:.4f}")
    logger.info(f"    Profile Pearson r (mean):     {mean_img_profile_r:.4f}")
    logger.info("=" * 60)

    # --- Save ---
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        np.savez(
            args.output,
            preds=all_preds,
            targets=all_targets,
            voxel_r=voxel_r,
            profile_r=profile_r,
            img_profile_r=img_profile_r,
            img_voxel_r=img_voxel_r,
        )
        logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSFM fMRI Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output", type=str, default=None, help="Path to save .npz results")
    args = parser.parse_args()

    evaluate(args)
