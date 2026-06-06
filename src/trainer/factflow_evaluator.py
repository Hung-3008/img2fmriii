"""
factflow_evaluator.py
=====================
Full evaluation for FactFlow-based fMRI synthesis.

Supports three inference scenarios:

  1. **deterministic** — PerceiverVE mean μ → ODE solver (fully deterministic).
  2. **perceiver_stochastic** — PerceiverVE samples x₀ = μ + ε·σ → ODE solver.
  3. **flow_stochastic** — PerceiverVE mean μ → SDE solver (noise-injected flow).

For stochastic scenarios, runs ``max_trials`` forward passes (default 10),
saves each individual pass, then computes metrics for each K in ``k_values``
(default [1, 5, 10]) by averaging the first K passes.  This avoids redundant
computation — 10 passes total instead of 1+5+10 = 16.

Metrics:
  - Per-voxel Pearson r (encoding metric, mean & median)
  - Profile Pearson r (mean)
  - MSE

Results can be saved to ``.npz`` and/or appended to a CSV file.
"""

from __future__ import annotations

import csv
import logging
import os
from argparse import Namespace
from typing import Dict, List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models, build_transport, build_sampler
from model.transport import Sampler
from utils.checkpoint import load_checkpoint
from utils.fmri_utils import auto_size_config, get_latent_size
from utils.logging_utils import create_logger
from utils.metrics import compute_full_metrics

logger = logging.getLogger(__name__)


class FactFlowEvaluator:
    """Evaluate a trained FactFlow fMRI model on the test set."""

    def __init__(self, args: Namespace) -> None:
        # ── Config ────────────────────────────────────────────────────
        self.cfg = OmegaConf.load(args.config)
        # Per-subject native sizing — must match training (no-op unless
        # data.auto_pad is set), so the model matches the checkpoint shapes.
        _autosize_msg = auto_size_config(self.cfg)
        if _autosize_msg:
            logger.info(_autosize_msg)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.loss_cfg = OmegaConf.to_container(
            self.cfg.get("losses", {}), resolve=True,
        )
        self.args = args

        # ── Scenario ──────────────────────────────────────────────────
        self.scenario = getattr(args, "scenario", "deterministic")

        # For stochastic scenarios: run max_trials passes, then compute
        # metrics for each K in k_values by averaging the first K passes.
        self.max_trials = getattr(args, "max_trials", 10)
        k_str = getattr(args, "k_values", "1,5,10")
        self.k_values: List[int] = sorted(
            int(x) for x in str(k_str).split(",")
        )

        if self.scenario == "deterministic":
            # Deterministic: only 1 pass, only K=1 makes sense
            self.max_trials = 1
            self.k_values = [1]

        # Ensure max_trials >= max(k_values)
        self.max_trials = max(self.max_trials, max(self.k_values))

        # ── Device ────────────────────────────────────────────────────
        self.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = self.cfg.training.get("precision", "fp32") == "bf16"
        self.autocast_kwargs = dict(
            device_type=self.device.split(":")[0],
            dtype=torch.bfloat16,
            enabled=use_bf16,
        )

        self.logger = create_logger(name="factflow_eval")

        # ── Data ──────────────────────────────────────────────────────
        # Always use avg_reps=True for evaluation: 1000 images with
        # rep-averaged GT (mean of 3 trials).  This matches the trainer's
        # _validate() and published NSD benchmarks.
        dc = self.data_cfg
        self.n_voxels = dc["n_voxels"]
        # ROI voxel ordering? Then the dataset reorders voxels by ROI; predictions
        # are un-sorted back to anatomical order before metrics/saving so output
        # is identical in layout to non-reordered runs.
        self.roi_order = bool(dc.get("roi_order", False))
        self.test_ds = FactFlowfMRIDataset(
            data_dir=dc["data_dir"],
            subject=dc["subject"],
            mode="test",
            fmri_mode=dc["fmri_mode"],
            clip_feature=dc["clip_feature"],
            n_voxels=dc["n_voxels"],
            pad_to=dc["pad_to"],
            fmri_channels=dc.get("fmri_channels", 1),
            fmri_spatial=dc.get("fmri_spatial", None),
            dino_feature=dc.get("dino_feature", None),
            avg_reps=True,
            roi_order=self.roi_order,
            context_features=dc.get("context_features", None),
        )
        # Index that maps ROI-sorted voxels back to anatomical order (or None).
        self.unsort_idx = self.test_ds.unsort_idx if self.roi_order else None
        # Context streams define the per-stream embedder dims (matches training).
        self.cfg.stage_2.params.context_dims = list(self.test_ds.context_dims)
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # ── Geometry ──────────────────────────────────────────────────
        self.latent_size = get_latent_size(self.data_cfg)
        self.use_cross_attn = bool(
            self.cfg.stage_2.get("params", {}).get("use_cross_attn", False)
        )

        # ── Source encoder config ─────────────────────────────────────
        self.clip_logvar_min = float(self.loss_cfg.get("clip_logvar_min", -10.0))
        self.clip_logvar_max = float(self.loss_cfg.get("clip_logvar_max", 10.0))

        # ── Model ─────────────────────────────────────────────────────
        self.wrapper = build_models(self.cfg, self.device)
        self.transport = build_transport(self.cfg, self.latent_size)

        # Build ODE sampler (always needed for scenarios 1 & 2)
        self.sample_fn_ode = build_sampler(self.transport, self.cfg.sampler)

        # Build SDE sampler (for scenario 3: flow_stochastic)
        self.sample_fn_sde = None
        if self.scenario == "flow_stochastic":
            sde_num_steps = getattr(args, "sde_num_steps", 250)
            sde_diffusion_norm = getattr(args, "sde_diffusion_norm", 1.0)
            sde_sampler = Sampler(self.transport)
            self.sample_fn_sde = sde_sampler.sample_sde(
                sampling_method="euler",
                diffusion_form="SBDM",
                diffusion_norm=sde_diffusion_norm,
                last_step="Mean",
                last_step_size=0.04,
                num_steps=sde_num_steps,
            )

        # Load checkpoint
        ckpt = torch.load(args.ckpt, map_location="cpu")
        self.wrapper.load_state_dict(ckpt["model"], strict=False)
        self.logger.info("Loaded model weights from %s", args.ckpt)
        del ckpt
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        self.wrapper.eval()

    # ──────────────────────────────────────────────────────────────────
    # Source encoding helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_x0_deterministic(
        self, clip_tokens: torch.Tensor, B: int,
    ) -> torch.Tensor:
        """Scenario 1 & 3: use PerceiverVE mean μ (no sampling)."""
        _, mu, _ = self.wrapper.encode_source(clip_tokens)
        return mu.permute(0, 2, 1).contiguous().view(B, *self.latent_size)

    def _get_x0_stochastic(
        self, clip_tokens: torch.Tensor, B: int,
    ) -> torch.Tensor:
        """Scenario 2: sample x₀ = μ + ε·σ from PerceiverVE."""
        _, mu, log_var = self.wrapper.encode_source(clip_tokens)
        if log_var is not None:
            log_var = torch.clamp(
                log_var,
                min=self.clip_logvar_min,
                max=self.clip_logvar_max,
            )
            std = (0.5 * log_var).exp()
            x0_tok = mu + torch.randn_like(mu) * std
        else:
            x0_tok = mu
        return x0_tok.permute(0, 2, 1).contiguous().view(B, *self.latent_size)

    # ──────────────────────────────────────────────────────────────────
    # Single-pass inference
    # ──────────────────────────────────────────────────────────────────

    def _run_single_pass(self) -> np.ndarray:
        """Run one complete inference pass over the test set.

        Returns:
            ``(N, V)`` predicted voxels (unpadded).
        """
        all_preds = []

        sample_fn = (
            self.sample_fn_sde
            if self.scenario == "flow_stochastic"
            else self.sample_fn_ode
        )

        for batch in tqdm(self.test_loader, desc="Inference", dynamic_ncols=True):
            clip_tokens = batch["clip_tokens"].to(self.device)
            clip_pool = batch["clip_pool"].to(self.device)
            contexts = [c.to(self.device) for c in batch["contexts"]]
            B = clip_tokens.shape[0]

            # Source x₀
            if self.scenario == "perceiver_stochastic":
                x0 = self._get_x0_stochastic(clip_tokens, B)
            else:
                # deterministic or flow_stochastic: use μ
                x0 = self._get_x0_deterministic(clip_tokens, B)

            # Flow sampling (ODE or SDE)
            ctx = contexts if self.use_cross_attn else None
            with autocast(**self.autocast_kwargs):
                traj = sample_fn(
                    x0, self.wrapper.dit.forward, y=clip_pool, contexts=ctx,
                )
            pred = traj[-1].float()
            # Free all intermediate ODE/SDE steps immediately
            del traj, x0, clip_tokens, clip_pool, contexts, ctx

            pred_flat = pred.reshape(B, -1)[:, : self.n_voxels].cpu().numpy()
            if self.unsort_idx is not None:
                pred_flat = pred_flat[:, self.unsort_idx]  # → anatomical order
            del pred
            all_preds.append(pred_flat)

        # Release GPU memory before returning to the next pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.concatenate(all_preds, axis=0)

    # ──────────────────────────────────────────────────────────────────
    # Metrics helper
    # ──────────────────────────────────────────────────────────────────

    def _compute_and_report(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        k: int,
    ) -> Dict[str, float]:
        """Compute metrics for a given K-averaged prediction and log results."""
        metrics = compute_full_metrics(
            preds, targets,
            n_voxels=self.n_voxels,
            n_reps=1,
            n_images=self.test_ds.n_images,
        )

        self.logger.info("  ── K=%d ──", k)
        self.logger.info("    Per-voxel Pearson r (mean):   %.4f", metrics.mean_voxel_r)
        self.logger.info("    Per-voxel Pearson r (median): %.4f", metrics.median_voxel_r)
        self.logger.info("    Profile Pearson r (mean):     %.4f", metrics.mean_profile_r)
        self.logger.info("    MSE:                          %.6f", metrics.mse)

        return {
            "mean_voxel_r": metrics.mean_voxel_r,
            "median_voxel_r": metrics.median_voxel_r,
            "mean_profile_r": metrics.mean_profile_r,
            "mse": metrics.mse,
            "voxel_r": metrics.voxel_r,
            "profile_r": metrics.profile_r,
        }

    def _append_csv(
        self, k: int, result: Dict[str, float],
    ) -> None:
        """Append one CSV row for a given K."""
        csv_path = getattr(self.args, "csv_out", None)
        if not csv_path:
            return
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        header = [
            "scenario", "k_trials",
            "mean_voxel_r", "median_voxel_r",
            "mean_profile_r", "mse",
            "ckpt",
        ]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                self.scenario,
                k,
                f"{result['mean_voxel_r']:.6f}",
                f"{result['median_voxel_r']:.6f}",
                f"{result['mean_profile_r']:.6f}",
                f"{result['mse']:.6f}",
                self.args.ckpt,
            ])
        self.logger.info("    → CSV row appended to %s", csv_path)

    # ──────────────────────────────────────────────────────────────────
    # Main evaluation
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run inference and compute all metrics.

        For stochastic scenarios, runs ``max_trials`` forward passes, saves
        each individual pass to ``.npz``, then computes metrics for each K
        in ``k_values`` by averaging the first K passes.
        """
        self.logger.info("=" * 60)
        self.logger.info(
            "Scenario: %s  |  max_trials: %d  |  k_values: %s",
            self.scenario, self.max_trials, self.k_values,
        )
        self.logger.info(
            "Running inference on %d test images (avg_reps=True)...",
            len(self.test_ds),
        )
        self.logger.info("=" * 60)

        # ── Collect ground truth (once) ───────────────────────────────
        all_targets = []
        for batch in self.test_loader:
            fmri_gt = batch["fmri"]
            B = fmri_gt.shape[0]
            gt_flat = fmri_gt.reshape(B, -1)[:, : self.n_voxels].numpy()
            if self.unsort_idx is not None:
                gt_flat = gt_flat[:, self.unsort_idx]  # → anatomical order
            all_targets.append(gt_flat)
        targets = np.concatenate(all_targets, axis=0)  # (N, V)

        # ── Run max_trials forward passes ─────────────────────────────
        # Store each pass individually: list of (N, V) arrays
        all_passes: List[np.ndarray] = []
        out_dir = getattr(self.args, "output", None)

        for t in range(self.max_trials):
            self.logger.info(
                "Pass %d / %d ...", t + 1, self.max_trials,
            )
            preds_t = self._run_single_pass()  # (N, V)
            all_passes.append(preds_t)

            # Save individual pass
            if out_dir:
                pass_dir = os.path.join(out_dir, "passes")
                os.makedirs(pass_dir, exist_ok=True)
                pass_path = os.path.join(pass_dir, f"pass_{t:02d}.npy")
                np.save(pass_path, preds_t)
                self.logger.info("  Saved pass %d → %s", t + 1, pass_path)

        # ── Compute metrics for each K ────────────────────────────────
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS  (scenario=%s)", self.scenario)
        self.logger.info("=" * 60)

        last_result = {}
        for k in self.k_values:
            # Average first K passes
            preds_avg = np.stack(all_passes[:k], axis=0).mean(axis=0).astype(np.float32)
            result = self._compute_and_report(preds_avg, targets, k)
            self._append_csv(k, result)

            # Save averaged prediction .npz
            if out_dir:
                avg_path = os.path.join(out_dir, f"avg_k{k:02d}.npz")
                np.savez(
                    avg_path,
                    preds=preds_avg,
                    targets=targets,
                    voxel_r=result["voxel_r"],
                    profile_r=result["profile_r"],
                )
                self.logger.info("    → Saved avg_k%d → %s", k, avg_path)

            last_result = result

        self.logger.info("=" * 60)

        return {
            "mean_voxel_r": last_result.get("mean_voxel_r", 0.0),
            "median_voxel_r": last_result.get("median_voxel_r", 0.0),
            "mean_profile_r": last_result.get("mean_profile_r", 0.0),
            "mse": last_result.get("mse", 0.0),
        }
