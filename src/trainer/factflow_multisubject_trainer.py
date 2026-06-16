"""
factflow_multisubject_trainer.py
================================
Multi-subject trainer for FactFlow fMRI synthesis (shared trunk + per-subject
adapters; Option B / MindEye2-style).

  - One shared DiT1D trunk; per-subject input patch-embed + output readout,
    selected by ``subject_id``.
  - All subjects padded to the same ``pad_to``; flow matching runs in native
    voxel space, so metrics stay comparable to published baselines.
  - Subject-homogeneous batches (SubjectBatchSampler); grad-accum mixes subjects.
  - Per-subject pad_mask + SNR weight + validation; best.pt by MEAN profile_r.

Shared flow-matching / eval / loop machinery lives in ``factflow_core``.
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.multi_subject_dataset import (
    MultiSubjectDataset,
    SubjectBatchSampler,
    build_subject_datasets,
)
from model.factflow_factory import build_models
from utils.fmri_utils import create_pad_mask, get_latent_size
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


class FactFlowMultiSubjectTrainer(FactFlowBase, TrainerLoopMixin):
    """Shared-trunk, per-subject-adapter trainer for FactFlow fMRI synthesis."""

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.data_cfg = OmegaConf.to_container(self.cfg.data, resolve=True)
        self.train_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)

        self.subjects: List[int] = list(self.data_cfg["subjects"])
        self.n_subjects = len(self.subjects)
        self.n_voxels_map = {int(k): int(v)
                             for k, v in self.data_cfg["n_voxels_map"].items()}
        self.pad_to = int(self.data_cfg["pad_to"])
        self.roi_order = bool(self.data_cfg.get("roi_order", False))

        self._init_device_seed(int(self.train_cfg.get("global_seed", 42)))

        subj_tag = "".join(str(s) for s in self.subjects)
        exp_name = args.exp_name or f"factflow_ms_sub{subj_tag}"
        logger = create_logger(
            os.path.join(args.exps_dir, exp_name), name="factflow_ms_trainer")
        self._init_exp_dir(exp_name, "factflow_ms_trainer", logger)
        self.logger.info(
            "Multi-subject: subjects=%s  n_voxels=%s  pad_to=%d",
            self.subjects, [self.n_voxels_map[s] for s in self.subjects], self.pad_to,
        )

        # ── Data ──
        self._build_datasets()

        # ── Geometry ──
        self.pad_masks = {
            sidx: create_pad_mask(self.n_voxels_map[s], self.pad_to, self.device)
            for sidx, s in enumerate(self.subjects)
        }
        self._init_flow(
            get_latent_size({**self.data_cfg, "pad_to": self.pad_to}),
            self.train_cfg,
        )

        # ── Model (inject multi-subject + per-stream dims before build) ──
        self.cfg.stage_2.params.n_subjects = self.n_subjects
        self.cfg.stage_2.params.seq_len = self.pad_to
        self.cfg.stage_2.params.context_dims = list(self.train_dss[0].context_dims)
        self.wrapper = build_models(self.cfg, self.device)
        setup_roi_routing(
            dit=self.wrapper.dit, cfg=self.cfg, data_cfg=self.data_cfg,
            subjects=self.subjects, datasets=self.train_dss,
            roi_order=self.roi_order, pad_to=self.pad_to, logger=self.logger,
        )

        # ── Optimizer & scheduler ──
        self.grad_accum = int(self.train_cfg.get("grad_accum_steps", 1))
        self.steps_per_epoch = max(len(self.batch_sampler) // self.grad_accum, 1)
        self.epochs = int(self.train_cfg.get("epochs", 50))
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
        self.best_metric = -1.0   # best MEAN val_profile_r
        self._maybe_resume()

        # ── Loop hyper-params ──
        self.clip_grad = float(self.train_cfg.get("clip_grad", 1.0))
        self.log_every = int(self.train_cfg.get("log_every", 50))
        self.ckpt_every = int(self.train_cfg.get("ckpt_every", 5000))
        self.val_every = int(self.train_cfg.get("val_every", 1))

        # Per-subject SNR weights (contiguous index → (pad_to,) tensor or None).
        self.voxel_weights = self._build_voxel_weights()
        self.wrapper.train()

    # ── Datasets ───────────────────────────────────────────────────────
    def _base_kwargs(self) -> dict:
        # Fresh dict each call: build_subject_datasets pops keys (n_voxels_map),
        # mutating its argument, so train/val must each get their own copy.
        return build_ds_kwargs(
            self.data_cfg, pad_to=self.pad_to, roi_order=self.roi_order,
            n_voxels_map=self.n_voxels_map,
        )

    def _build_datasets(self) -> None:
        avg_reps = bool(self.data_cfg.get("avg_reps", False))
        self.train_dss = build_subject_datasets(
            self.subjects, self._base_kwargs(), mode="train", avg_reps=avg_reps)
        self.train_ds = MultiSubjectDataset(self.train_dss)
        # Val: always rep-averaged, kept per-subject (native-space metrics).
        self.val_dss = build_subject_datasets(
            self.subjects, self._base_kwargs(), mode="test", avg_reps=True)

        batch_size = int(self.train_cfg.get("batch_size", 32))
        num_workers = int(self.train_cfg.get("num_workers", 2))
        self.batch_sampler = SubjectBatchSampler(
            self.train_ds.boundaries, batch_size=batch_size,
            shuffle=True, drop_last=True,
        )
        self.train_loader = DataLoader(
            self.train_ds, batch_sampler=self.batch_sampler,
            num_workers=num_workers,
            pin_memory=bool(self.train_cfg.get("pin_memory", False)),
            persistent_workers=num_workers > 0,
        )
        self.logger.info(
            "Train: %d samples over %d subjects, %d batches/epoch (bs=%d)",
            len(self.train_ds), self.n_subjects, len(self.batch_sampler), batch_size,
        )

    def _build_voxel_weights(self) -> Dict[int, Optional[torch.Tensor]]:
        use_snr = bool(self.cfg.get("losses", {}).get("use_snr_weight", False))
        if not use_snr:
            return {sidx: None for sidx in range(self.n_subjects)}
        weights = {
            sidx: build_voxel_weight(
                ds=ds, pad_to=self.pad_to, n_voxels=ds.n_voxels,
                roi_order=self.roi_order, device=self.device,
            )
            for sidx, ds in enumerate(self.train_dss)
        }
        self.logger.info("SNR-weighted loss ON (per-subject noise-ceiling).")
        return weights

    # ── Loop hooks ─────────────────────────────────────────────────────
    def _on_epoch_start(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)

    def _loss_for_batch(self, batch):
        sidx = int(batch["subject_id"][0])
        x1 = batch["fmri"].to(self.device)
        clip_pool = batch["clip_pool"].to(self.device)
        contexts = ([c.to(self.device) for c in batch["contexts"]]
                    if self.use_cross_attn else None)
        loss = flow_matching_loss(
            transport=self.transport, wrapper=self.wrapper, x1=x1,
            clip_pool=clip_pool, contexts=contexts,
            pad_mask=self.pad_masks[sidx], subject_id=sidx,
            voxel_weight=self.voxel_weights[sidx],
        )
        return loss, {"subj": self.subjects[sidx]}

    def _history_header(self):
        return ["epoch", "step", "train_loss", "val_voxel_r", "val_profile_r", "lr"]

    def _history_row(self, epoch, train_loss, val):
        lr = self.scheduler.get_last_lr()[0]
        return [epoch + 1, self.train_steps, f"{train_loss:.6f}",
                f"{val['voxel_r']:.6f}", f"{val['profile_r']:.6f}", f"{lr:.2e}"]

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        per_subj = {}
        for sidx, ds in enumerate(self.val_dss):
            m = evaluate_metrics(
                wrapper=self.wrapper, sample_fn=self.sample_fn, ds=ds,
                pad_mask=self.pad_masks[sidx], latent_size=self.latent_size,
                device=self.device, autocast_kwargs=self.autocast_kwargs,
                use_cross_attn=self.use_cross_attn, noise_scale=0.0,
                subject_id=sidx,
            )
            per_subj[self.subjects[sidx]] = m
            self.logger.info(
                "  [Val ep=%d subj=%d] voxel_r=%.4f profile_r=%.4f mse=%.5f n=%d",
                epoch + 1, self.subjects[sidx],
                m["voxel_r"], m["profile_r"], m["mse"], m["n"],
            )
        mean_profile_r = float(np.mean([v["profile_r"] for v in per_subj.values()]))
        mean_voxel_r = float(np.mean([v["voxel_r"] for v in per_subj.values()]))
        self.logger.info(
            "  [Val ep=%d MEAN] voxel_r=%.4f profile_r=%.4f",
            epoch + 1, mean_voxel_r, mean_profile_r,
        )
        return {"profile_r": mean_profile_r, "voxel_r": mean_voxel_r,
                "per_subj": per_subj}
