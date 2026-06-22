"""
factflow_fewshot_trainer.py
===========================
Few-shot cross-subject adaptation as a *real* training loop (not a fixed-budget
one-shot adapt). Given a shared-trunk multi-subject checkpoint, adapt to a
HELD-OUT subject:

  - Add one extra subject slot (the held-out subject at the last index).
  - Load trunk + trained adapters (strict=False); warm-start the new adapter
    from the mean of the trained adapters.
  - Freeze EVERYTHING except the held-out subject's input patch-embed + readout.
  - Split the held-out subject's data into 750 adaptation trials (1 hour) and
    250 disjoint validation trials (single-rep; no test leakage).
  - Train a small number of epochs, validating EVERY epoch and keeping the BEST
    adapter (by profile_r).
  - Finally evaluate the BEST adapter on the real test set (rep-averaged),
    averaging Trials in {1, 5}.

Dispatched from train_factflow_multisubject.py via ``--fewshot_held_out``.
Shared flow-matching / eval / adapter machinery lives in ``factflow_core``.
"""

from __future__ import annotations

import copy
import csv
import os
from argparse import Namespace
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import autocast
from torch.utils.data import DataLoader, Subset

from data.factflow_fmri_dataset import FactFlowfMRIDataset
from model.factflow_factory import build_models
from model.lora import apply_lora_to_blocks
from utils.checkpoint import save_checkpoint
from utils.fmri_utils import create_pad_mask, get_latent_size
from utils.logging_utils import create_logger

from trainer.factflow_core import (
    FactFlowBase,
    build_ds_kwargs,
    evaluate_metrics,
    flow_matching_loss,
    freeze_to_adapter,
    warm_start_adapter,
)

TRIALS_PER_HOUR = 750   # 1 NSD session ≈ 1 hour ≈ 750 stimulus trials


class FactFlowFewShotTrainer(FactFlowBase):
    """Few-shot adaptation to a held-out subject (shared trunk frozen)."""

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.train_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)

        self.held = int(args.fewshot_held_out)
        self.trunk_subjects: List[int] = list(self.data_cfg["subjects"])
        self.held_idx = len(self.trunk_subjects)          # new adapter slot
        n_voxels_map = {int(k): int(v)
                        for k, v in self.data_cfg["n_voxels_map"].items()}
        self.n_voxels = n_voxels_map[self.held]
        self.pad_to = int(self.data_cfg["pad_to"])
        self.roi_order = bool(self.data_cfg.get("roi_order", False))

        seed = int(args.seed)
        self._init_device_seed(seed)

        trunk_tag = "".join(str(s) for s in self.trunk_subjects)
        exp_name = args.exp_name or f"fewshot_sub{self.held}_from{trunk_tag}"
        logger = create_logger(
            os.path.join(args.exps_dir, exp_name), name="factflow_fewshot")
        self._init_exp_dir(exp_name, "factflow_fewshot", logger)

        # ── Hyper-params ──
        self.epochs = int(args.fewshot_epochs)
        self.batch_size = int(args.adapt_bs)
        self.adapt_lr = float(args.adapt_lr)
        self.noise_scale = float(args.noise_scale)
        self.trials = list(args.trials)
        self.n_val = int(args.fewshot_val_trials)

        # ── Data ──
        self._build_datasets(seed)

        # ── Model: trunk + new adapter, load pretrained, freeze ──
        self._build_model()

        # ── Transport / sampler / geometry / autocast ──
        self._init_flow(
            get_latent_size({**self.data_cfg, "pad_to": self.pad_to}),
            self.train_cfg,
        )
        self.pad_mask = create_pad_mask(self.n_voxels, self.pad_to, self.device)

        # ── Optimizer (adapter params only) + cosine schedule ──
        trainable = [p for p in self.wrapper.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.adapt_lr, betas=(0.9, 0.95),
            weight_decay=float(args.adapt_wd),
        )
        self.steps_per_epoch = max(len(self.train_loader), 1)
        total_steps = max(self.steps_per_epoch * self.epochs, 1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.adapt_lr * 0.05,
        )

    # ── Datasets ───────────────────────────────────────────────────────
    def _ds_kwargs(self, avg_reps: bool) -> dict:
        return build_ds_kwargs(
            self.data_cfg, pad_to=self.pad_to, roi_order=self.roi_order,
            subject=self.held, n_voxels=self.n_voxels, avg_reps=avg_reps,
        )

    def _build_datasets(self, seed: int) -> None:
        # Adaptation uses SINGLE-REP trials (1 hour = 750 trials). Validation uses
        # REP-AVERAGED samples on a DISJOINT set of images (split by image, not
        # trial → no rep leakage), for a clean low-noise selection signal.
        full_single = FactFlowfMRIDataset(mode="train", **self._ds_kwargs(False))
        full_avg = FactFlowfMRIDataset(mode="train", **self._ds_kwargs(True))
        self.context_dims = list(full_single.context_dims)
        n_images, n_reps = full_single.n_images, full_single.n_reps

        rng = np.random.RandomState(seed)
        img_order = rng.permutation(n_images)
        n_val_img = min(self.n_val, n_images)
        val_images = img_order[:n_val_img].tolist()
        adapt_images = set(img_order[n_val_img:].tolist())

        n_adapt_target = int(round(self.args.fewshot_hours * TRIALS_PER_HOUR))
        trial_order = rng.permutation(len(full_single))
        adapt_trials = [int(t) for t in trial_order
                        if (int(t) // n_reps) in adapt_images][:n_adapt_target]

        self.train_ds = Subset(full_single, adapt_trials)
        self.val_ds = Subset(full_avg, val_images)            # rep-averaged
        self.n_adapt = len(self.train_ds)
        self.n_val = len(self.val_ds)
        self.test_ds = FactFlowfMRIDataset(mode="test", **self._ds_kwargs(True))

        self.train_loader = DataLoader(
            self.train_ds, batch_size=min(self.batch_size, len(self.train_ds)),
            shuffle=True, num_workers=0, drop_last=False,
        )
        self.logger.info(
            "Few-shot subj=%d (trunk=%s): adapt=%d single-rep trials (%.2fh) | "
            "val=%d rep-avg images (disjoint, noise=0) | test=%d images",
            self.held, self.trunk_subjects, self.n_adapt,
            self.n_adapt / TRIALS_PER_HOUR, self.n_val, len(self.test_ds),
        )

    def _build_model(self) -> None:
        self.cfg.stage_2.params.n_subjects = self.held_idx + 1
        self.cfg.stage_2.params.seq_len = self.pad_to
        self.cfg.stage_2.params.context_dims = self.context_dims
        self.wrapper = build_models(self.cfg, self.device)

        ckpt = torch.load(self.args.fewshot_pretrained, map_location=self.device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        missing, unexpected = self.wrapper.load_state_dict(state, strict=False)
        new_keys = [k for k in missing if f".{self.held_idx}." in k]
        self.logger.info(
            "[load] %s  strict=False missing=%d (new adapter=%d) unexpected=%d",
            self.args.fewshot_pretrained, len(missing), len(new_keys), len(unexpected),
        )

        if not self.args.no_warm_start:
            warm_start_adapter(self.wrapper.dit, self.held_idx)

        # Optional LoRA on the last trunk block(s): lets the frozen shared trunk
        # specialise to the held-out subject (the linear readout alone saturates
        # with ~1h of data). Applied AFTER load + warm-start, BEFORE freezing.
        self.use_lora = bool(getattr(self.args, "lora", False))
        if self.use_lora:
            n_wrapped = apply_lora_to_blocks(
                self.wrapper.dit.blocks,
                n_last=int(getattr(self.args, "lora_blocks", 1)),
                rank=int(getattr(self.args, "lora_rank", 8)),
                alpha=float(getattr(self.args, "lora_alpha", 16.0)),
                dropout=float(getattr(self.args, "lora_dropout", 0.0)),
            )
            self.wrapper.to(self.device)   # new LoRA params → device

        trainable = freeze_to_adapter(self.wrapper, self.held_idx)
        n_params = sum(p.numel() for p in trainable)
        if self.use_lora:
            n_lora = 0
            for name, p in self.wrapper.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.requires_grad = True
                    n_lora += p.numel()
            self.logger.info(
                "[lora] wrapped %d Linear layers in last %d block(s) "
                "(rank=%d alpha=%.1f) — +%.3fM trainable",
                n_wrapped, int(getattr(self.args, "lora_blocks", 1)),
                int(getattr(self.args, "lora_rank", 8)),
                float(getattr(self.args, "lora_alpha", 16.0)), n_lora / 1e6,
            )
        self.logger.info(
            "[freeze] trainable adapter params: %.3fM (held_idx=%d)",
            n_params / 1e6, self.held_idx,
        )

    # ── Loss / eval ────────────────────────────────────────────────────
    def _loss(self, x1, clip_pool, contexts) -> torch.Tensor:
        return flow_matching_loss(
            transport=self.transport, wrapper=self.wrapper, x1=x1,
            clip_pool=clip_pool, contexts=contexts, pad_mask=self.pad_mask,
            subject_id=self.held_idx,
        )

    @torch.no_grad()
    def _evaluate(self, ds, n_inf_trials: int,
                  noise_scale: Optional[float] = None) -> Dict[str, float]:
        return evaluate_metrics(
            wrapper=self.wrapper, sample_fn=self.sample_fn, ds=ds,
            pad_mask=self.pad_mask, latent_size=self.latent_size,
            device=self.device, autocast_kwargs=self.autocast_kwargs,
            use_cross_attn=self.use_cross_attn,
            noise_scale=self.noise_scale if noise_scale is None else noise_scale,
            subject_id=self.held_idx, n_inf_trials=n_inf_trials,
        )

    def _adapter_state(self) -> dict:
        # Keep the held-out readout AND (when enabled) the LoRA residuals, so the
        # best-epoch restore captures everything that was trained.
        return copy.deepcopy({
            k: v.detach().cpu() for k, v in self.wrapper.state_dict().items()
            if f"x_embedders.{self.held_idx}." in k
            or f"final_layers.{self.held_idx}." in k
            or "lora_A" in k or "lora_B" in k
        })

    # ── Train ──────────────────────────────────────────────────────────
    def train(self) -> None:
        hist_path = os.path.join(self.exp_dir, "history.csv")
        hist = open(hist_path, "w", newline="")
        writer = csv.writer(hist)
        writer.writerow(["epoch", "train_loss", "val_voxel_r", "val_profile_r",
                         "val_mse", "lr"])
        hist.flush()

        best_metric = -1.0
        best_state = None
        self.logger.info(
            "Starting few-shot adaptation: %d epochs, lr=%.1e, noise_scale=%.2f",
            self.epochs, self.adapt_lr, self.noise_scale,
        )

        for epoch in range(self.epochs):
            self.wrapper.train()
            ep_loss, n = 0.0, 0
            for batch in self.train_loader:
                x1 = batch["fmri"].to(self.device)
                clip_pool = batch["clip_pool"].to(self.device)
                contexts = ([c.to(self.device) for c in batch["contexts"]]
                            if self.use_cross_attn else None)
                with autocast(**self.autocast_kwargs):
                    loss = self._loss(x1, clip_pool, contexts)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                ep_loss += loss.item()
                n += 1

            # Validate EVERY epoch on the disjoint rep-averaged val set,
            # deterministically (noise=0 → clean, low-variance ranking).
            val = self._evaluate(self.val_ds, n_inf_trials=1, noise_scale=0.0)
            lr = self.scheduler.get_last_lr()[0]
            self.logger.info(
                "[ep %02d/%d] train_loss=%.5f | val voxel_r=%.4f profile_r=%.4f "
                "mse=%.4f | lr=%.2e", epoch + 1, self.epochs, ep_loss / max(n, 1),
                val["voxel_r"], val["profile_r"], val["mse"], lr,
            )
            writer.writerow([epoch + 1, f"{ep_loss / max(n, 1):.6f}",
                             f"{val['voxel_r']:.6f}", f"{val['profile_r']:.6f}",
                             f"{val['mse']:.6f}", f"{lr:.2e}"])
            hist.flush()

            if val["profile_r"] > best_metric:
                best_metric = val["profile_r"]
                best_state = self._adapter_state()
                save_checkpoint(
                    os.path.join(self.ckpt_dir, "best.pt"),
                    self.wrapper, self.optimizer, self.scheduler,
                    epoch + 1, epoch, best_metric,
                )
                self.logger.info("  new best val profile_r=%.4f (epoch %d)",
                                 best_metric, epoch + 1)

        # ── Restore best adapter, final TEST eval (Trials 1 & 5) ──
        if best_state is not None:
            sd = self.wrapper.state_dict()
            sd.update({k: v.to(self.device) for k, v in best_state.items()})
            self.wrapper.load_state_dict(sd)
            self.logger.info("Restored best adapter (val profile_r=%.4f) for test.",
                             best_metric)

        self.logger.info("=== FINAL TEST (subj=%d, noise_scale=%.2f) ===",
                         self.held, self.noise_scale)
        self.logger.info("%7s  %8s  %9s  %8s  %8s",
                         "Trials", "MSE", "profile_r", "voxel_r", "cosine")
        for t in self.trials:
            m = self._evaluate(self.test_ds, n_inf_trials=t)
            self.logger.info("%7d  %8.4f  %9.4f  %8.4f  %8.4f",
                             t, m["mse"], m["profile_r"], m["voxel_r"], m["cosine"])
        hist.close()
        self.logger.info("Few-shot done. exp_dir=%s", self.exp_dir)
