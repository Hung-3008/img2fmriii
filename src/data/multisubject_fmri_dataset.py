"""
multisubject_fmri_dataset.py
============================
Joint dataset combining multiple subjects' FactFlowfMRIDataset instances.

Each sample includes the original fields plus:
    subject_id:  (,) int — 0-indexed subject index (for SubjectEmbedder)
    roi_profile: (n_buckets,) float — normalised ROI voxel fraction per bucket
                 (for ZeroShotSubjectEmbedder; zeros if roi_meta not available)

Voxel-count differences across subjects are handled by per-subject pad_mask
and a shared ``common_seq_len`` (= largest pad_to among all subjects, aligned
to patch_size). Shorter sequences are zero-padded to common_seq_len; their
pad_mask marks only real voxels as True.

Usage::

    ds = MultiSubjectfMRIDataset(
        subjects=[1, 2, 5, 7],
        data_dir="NSD/data/nsd",
        mode="train",
        patch_size=32,
        common_seq_len=15744,   # or None → auto-computed
        per_subject_kwargs={    # overrides applied per subject
            1: {"n_voxels": 15724, "avg_reps": True},
            2: {"n_voxels": 14278, "avg_reps": True},
            5: {"n_voxels": 13039, "avg_reps": True},
            7: {"n_voxels": 12682, "avg_reps": True},
        },
        **common_kwargs,        # applied to all subjects
    )
"""

from __future__ import annotations

import logging
import os
from bisect import bisect_right

import numpy as np
import torch
from torch.utils.data import Dataset

from .factflow_fmri_dataset import FactFlowfMRIDataset

logger = logging.getLogger(__name__)

# Number of ROI buckets used to compute roi_profile (early / mid / high)
_N_ROI_BUCKETS = 3


def _load_roi_profile(subj_dir: str, subject: int, n_voxels: int) -> torch.Tensor:
    """Load roi_meta and compute normalised bucket count vector.

    Returns a (n_buckets,) float tensor of [early_frac, mid_frac, high_frac].
    Falls back to uniform distribution if roi_meta is not found.
    """
    roi_path = os.path.join(subj_dir, f"roi_meta_sub{subject}.npz")
    if not os.path.exists(roi_path):
        logger.warning(
            "roi_meta not found for sub%d at %s; roi_profile will be uniform.",
            subject, roi_path,
        )
        return torch.full((_N_ROI_BUCKETS,), 1.0 / _N_ROI_BUCKETS)

    meta = np.load(roi_path)
    bucket_ids = meta["bucket_ids"][:n_voxels]  # (n_voxels,) int

    counts = np.bincount(bucket_ids.astype(int), minlength=_N_ROI_BUCKETS).astype(np.float32)
    total = counts.sum()
    profile = counts / total if total > 0 else np.ones(_N_ROI_BUCKETS) / _N_ROI_BUCKETS
    return torch.from_numpy(profile)


class MultiSubjectfMRIDataset(Dataset):
    """Concatenated multi-subject dataset for joint flow matching training.

    Args:
        subjects:          Ordered list of NSD subject IDs (e.g. [1, 2, 5, 7]).
        data_dir:          Root NSD data directory.
        mode:              "train" | "test".
        patch_size:        DiT patch size (default 32) — used to align pad_to.
        common_seq_len:    Shared padded sequence length. If None, auto-computed
                           as the largest per-subject pad_to, aligned to patch_size.
        per_subject_kwargs: Dict mapping subject_id → kwargs override for
                           FactFlowfMRIDataset (must include at least n_voxels).
        **common_kwargs:   Forwarded to every FactFlowfMRIDataset (e.g. mode,
                           clip_feature, context_features, roi_order, …).
    """

    def __init__(
        self,
        subjects: list[int],
        data_dir: str,
        mode: str = "train",
        patch_size: int = 32,
        common_seq_len: int | None = None,
        per_subject_kwargs: dict | None = None,
        **common_kwargs,
    ) -> None:
        super().__init__()
        self.subjects = list(subjects)
        self.subject_to_idx = {s: i for i, s in enumerate(self.subjects)}
        per_subject_kwargs = per_subject_kwargs or {}

        # ── Build per-subject datasets ───────────────────────────────────────
        self.datasets: list[FactFlowfMRIDataset] = []
        max_pad_to = 0
        for s in self.subjects:
            kw = dict(common_kwargs)
            kw.update(per_subject_kwargs.get(s, {}))
            kw["subject"] = s
            kw["data_dir"] = data_dir
            kw["mode"] = mode
            # Temporarily set pad_to to per-subject value; we'll re-pad later
            ds = FactFlowfMRIDataset(**kw)
            self.datasets.append(ds)
            max_pad_to = max(max_pad_to, ds.pad_to)

        # ── Compute common_seq_len ───────────────────────────────────────────
        if common_seq_len is None:
            # Round up to nearest multiple of patch_size
            common_seq_len = ((max_pad_to + patch_size - 1) // patch_size) * patch_size
        self.common_seq_len = common_seq_len
        self.patch_size = patch_size

        # ── Per-subject pad_masks aligned to common_seq_len ──────────────────
        self.subject_pad_masks: list[torch.Tensor] = []
        for ds in self.datasets:
            mask = torch.zeros(common_seq_len, dtype=torch.bool)
            mask[: ds.n_voxels] = True
            self.subject_pad_masks.append(mask)

        # ── ROI profiles ─────────────────────────────────────────────────────
        self.roi_profiles: list[torch.Tensor] = []
        for ds in self.datasets:
            subj_dir = ds.subj_dir
            profile = _load_roi_profile(subj_dir, ds.subject, ds.n_voxels)
            self.roi_profiles.append(profile)

        # ── Cumulative lengths for __getitem__ ───────────────────────────────
        self._cumlen: list[int] = []
        running = 0
        for ds in self.datasets:
            running += len(ds)
            self._cumlen.append(running)

        # ── Context dims (must be same across subjects) ───────────────────────
        self.context_dims = self.datasets[0].context_dims
        self.clip_pool_dim = self.datasets[0].clip_pool_dim

        logger.info(
            "MultiSubjectfMRIDataset: subjects=%s  mode=%s  common_seq_len=%d  "
            "total_samples=%d  context_dims=%s",
            self.subjects, mode, self.common_seq_len,
            len(self), self.context_dims,
        )
        for i, (s, ds) in enumerate(zip(self.subjects, self.datasets)):
            logger.info(
                "  Sub%d (idx=%d): n_voxels=%d → pad_to=%d → common=%d  samples=%d  "
                "roi_profile=%s",
                s, i, ds.n_voxels, ds.pad_to, common_seq_len,
                len(ds),
                [f"{x:.3f}" for x in self.roi_profiles[i].tolist()],
            )

    def __len__(self) -> int:
        return self._cumlen[-1]

    def __getitem__(self, idx: int) -> dict:
        # Identify which subject dataset this index falls into
        ds_idx = bisect_right(self._cumlen, idx)
        local_idx = idx - (self._cumlen[ds_idx - 1] if ds_idx > 0 else 0)

        sample = self.datasets[ds_idx][local_idx]

        # ── Re-pad fMRI to common_seq_len ────────────────────────────────────
        # sample["fmri"] is (1, per_subject_pad_to) — extend with zeros
        fmri = sample["fmri"]  # (1, pad_to_s)
        if fmri.shape[-1] < self.common_seq_len:
            pad_len = self.common_seq_len - fmri.shape[-1]
            fmri = torch.cat(
                [fmri, torch.zeros(*fmri.shape[:-1], pad_len, dtype=fmri.dtype)],
                dim=-1,
            )
            sample["fmri"] = fmri

        # ── Replace pad_mask with common-space mask ───────────────────────────
        sample["pad_mask"] = self.subject_pad_masks[ds_idx]

        # ── Add subject_id and roi_profile ────────────────────────────────────
        sample["subject_id"] = torch.tensor(
            self.subject_to_idx[self.subjects[ds_idx]], dtype=torch.long
        )
        sample["roi_profile"] = self.roi_profiles[ds_idx]  # (n_buckets,)

        return sample

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def voxel_count(self) -> int:
        """Returns common_seq_len (the padded, shared sequence length)."""
        return self.common_seq_len
