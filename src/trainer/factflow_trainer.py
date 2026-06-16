"""
factflow_trainer.py
===================
Single-subject trainer for FactFlow fMRI synthesis (no-source flow matching).

Pipeline:  Gaussian noise x₀  →  Flow Matching (velocity matching),
           conditioned on CLIP-pooled (AdaLN) + cross-attention image features
           (DINOv2, Gabor, …)  →  fMRI voxels.

  - Standard flow matching: x₀ ~ N(0, I), masked (optionally SNR-weighted)
    velocity-MSE objective — no source encoder, no auxiliary losses.
  - Validation against rep-averaged GT with noise_scale=0 (deterministic ceiling);
    best.pt selected by profile_r (comparable to SynBrain / MindSimulator).

Shared flow-matching / eval / loop machinery lives in ``factflow_core``.
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Dict

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models
from utils.fmri_utils import auto_size_config, create_pad_mask, get_latent_size
from utils.logging_utils import create_logger
from utils.training_utils import build_optimizer_and_scheduler

from trainer.factflow_core import (
    FactFlowBase,
    TrainerLoopMixin,
    build_ds_kwargs,
    build_voxel_weight,
    evaluate_metrics,
    flow_matching_loss,
    setup_roi_routing,
)


class FactFlowTrainer(FactFlowBase, TrainerLoopMixin):
    """End-to-end single-subject trainer for FactFlow fMRI synthesis."""

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        # Per-subject native sizing (no-op unless data.auto_pad); must run before
        # the config snapshot is saved and before geometry is derived.
        self._autosize_msg = auto_size_config(self.cfg)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.train_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        self.roi_order = bool(self.data_cfg.get("roi_order", False))

        self._init_device_seed(int(self.train_cfg.get("global_seed", 42)))

        exp_name = args.exp_name or f"factflow_fmri_sub{self.data_cfg['subject']}"
        logger = create_logger(
            os.path.join(args.exps_dir, exp_name), name="factflow_trainer")
        self._init_exp_dir(exp_name, "factflow_trainer", logger)
        if self._autosize_msg:
            self.logger.info(self._autosize_msg)

        # ── Data ──
        self._build_datasets()

        # ── Geometry ──
        self.pad_mask = create_pad_mask(
            self.data_cfg["n_voxels"], self.data_cfg["pad_to"], self.device,
        )
        self._init_flow(get_latent_size(self.data_cfg), self.train_cfg)

        # ── Model ──
        self.wrapper = build_models(self.cfg, self.device)
        setup_roi_routing(
            dit=self.wrapper.dit, cfg=self.cfg, data_cfg=self.data_cfg,
            subjects=[self.data_cfg["subject"]], datasets=[self.train_ds],
            roi_order=self.roi_order, pad_to=self.data_cfg["pad_to"],
            logger=self.logger,
        )

        # ── Optimizer & scheduler ──
        self.grad_accum = int(self.train_cfg.get("grad_accum_steps", 1))
        self.steps_per_epoch = max(len(self.train_loader) // self.grad_accum, 1)
        self.epochs = int(self.train_cfg.get("epochs", 200))
        self.optimizer, self.scheduler, opt_msg, sched_msg = (
            build_optimizer_and_scheduler(
                self.wrapper.parameters(), self.train_cfg,
                self.steps_per_epoch, self.epochs,
            )
        )
        self.logger.info(opt_msg)
        self.logger.info(sched_msg)

        # ── State + resume ──
        self.train_steps = 0
        self.start_epoch = 0
        self.best_metric = -1.0   # best val_profile_r
        self._maybe_resume()

        # ── Loop hyper-params ──
        self.clip_grad = float(self.train_cfg.get("clip_grad", 1.0))
        self.log_every = int(self.train_cfg.get("log_every", 50))
        self.ckpt_every = int(self.train_cfg.get("ckpt_every", 5000))
        self.sample_every = int(self.train_cfg.get("sample_every", 2000))
        self.val_every = int(self.train_cfg.get("val_every", 1))

        # Per-voxel SNR (noise-ceiling) weight for the velocity loss.
        self.voxel_weight = (
            build_voxel_weight(
                ds=self.train_ds, pad_to=self.data_cfg["pad_to"],
                n_voxels=self.data_cfg["n_voxels"], roi_order=self.roi_order,
                device=self.device,
            )
            if self.cfg.get("losses", {}).get("use_snr_weight", False) else None
        )
        if self.voxel_weight is not None:
            self.logger.info("SNR-weighted loss ON (noise-ceiling per voxel).")

        self.wrapper.train()

    # ── Datasets ───────────────────────────────────────────────────────
    def _build_datasets(self) -> None:
        dc = self.data_cfg
        ds_kwargs = build_ds_kwargs(
            dc, pad_to=dc["pad_to"], roi_order=self.roi_order,
            subject=dc["subject"], n_voxels=dc["n_voxels"],
            avg_reps=dc.get("avg_reps", False),
        )
        self.train_ds = FactFlowfMRIDataset(mode="train", **ds_kwargs)
        self.test_ds = FactFlowfMRIDataset(mode="test", **ds_kwargs)
        # Per-stream embedder dims come from the dataset's loaded context streams.
        self.cfg.stage_2.params.context_dims = list(self.train_ds.context_dims)
        # Validate against rep-averaged GT (removes measurement noise).
        self.val_ds = FactFlowfMRIDataset(
            mode="test", **{**ds_kwargs, "avg_reps": True})

        batch_size = int(self.train_cfg.get("batch_size", 64))
        num_workers = int(self.train_cfg.get("num_workers", 4))
        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers,
            pin_memory=bool(self.train_cfg.get("pin_memory", True)),
            drop_last=True, persistent_workers=num_workers > 0,
        )
        self.logger.info(
            "Train: %d samples, Test: %d samples (val_avg: %d images)",
            len(self.train_ds), len(self.test_ds), len(self.val_ds),
        )
        self.logger.info(
            "Batch size: %d, Steps/epoch: %d", batch_size, len(self.train_loader))

    # ── Loop hooks ─────────────────────────────────────────────────────
    def _loss_for_batch(self, batch):
        x1 = batch["fmri"].to(self.device)
        clip_pool = batch["clip_pool"].to(self.device)
        contexts = ([c.to(self.device) for c in batch["contexts"]]
                    if self.use_cross_attn else None)
        loss = flow_matching_loss(
            transport=self.transport, wrapper=self.wrapper, x1=x1,
            clip_pool=clip_pool, contexts=contexts, pad_mask=self.pad_mask,
            voxel_weight=self.voxel_weight,
        )
        return loss, None

    def _maybe_inline_eval(self) -> bool:
        if self.train_steps % self.sample_every == 0 and self.train_steps > 0:
            self.logger.info("Running inline evaluation...")
            n_eval = min(64, len(self.test_ds))
            m = evaluate_metrics(
                wrapper=self.wrapper, sample_fn=self.sample_fn,
                ds=Subset(self.test_ds, list(range(n_eval))),
                pad_mask=self.pad_mask, latent_size=self.latent_size,
                device=self.device, autocast_kwargs=self.autocast_kwargs,
                use_cross_attn=self.use_cross_attn,
                noise_scale=self.eval_noise_scale, batch_size=n_eval,
            )
            self.logger.info(
                "  [Eval step=%d] voxel_r=%.4f  profile_r=%.4f  mse=%.5f",
                self.train_steps, m["voxel_r"], m["profile_r"], m["mse"],
            )
        return True

    def _history_header(self):
        return ["epoch", "step", "train_loss",
                "val_mse", "val_voxel_r", "val_profile_r", "lr"]

    def _history_row(self, epoch, train_loss, val):
        lr = self.scheduler.get_last_lr()[0]
        return [epoch + 1, self.train_steps, f"{train_loss:.6f}",
                f"{val['mse']:.6f}", f"{val['voxel_r']:.6f}",
                f"{val['profile_r']:.6f}", f"{lr:.2e}"]

    def _best_ckpt_extra(self, val: Dict) -> Dict:
        return {"val_mse": val["mse"]}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Full validation over rep-averaged test set (deterministic, noise=0)."""
        m = evaluate_metrics(
            wrapper=self.wrapper, sample_fn=self.sample_fn, ds=self.val_ds,
            pad_mask=self.pad_mask, latent_size=self.latent_size,
            device=self.device, autocast_kwargs=self.autocast_kwargs,
            use_cross_attn=self.use_cross_attn, noise_scale=0.0,
        )
        self.logger.info(
            "  [Val epoch=%d] voxel_r=%.4f  profile_r=%.4f  mse=%.5f  n=%d  "
            "(noise_scale=0, deterministic)",
            epoch + 1, m["voxel_r"], m["profile_r"], m["mse"], m["n"],
        )
        return m
