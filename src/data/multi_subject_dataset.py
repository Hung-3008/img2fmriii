"""
multi_subject_dataset.py
========================
Multi-subject wrapper for FactFlow fMRI synthesis (shared trunk + per-subject
adapters).

Every subject is loaded with the *same* ``pad_to`` (e.g. 16384) but its own
``n_voxels``, so the patch grid is identical across subjects while the native
voxel content (and therefore the per-sample ``pad_mask``) stays subject-specific.

Two pieces:

  * ``MultiSubjectDataset`` — concatenates one ``FactFlowfMRIDataset`` per
    subject and injects a contiguous ``subject_id`` (0..S-1) into every sample.
  * ``SubjectBatchSampler`` — yields **subject-homogeneous** batches so each
    optimizer micro-batch touches a single subject's input/output adapter.
    Gradient accumulation across batches mixes subjects.
"""

from __future__ import annotations

import logging
import random
from typing import List, Tuple

from torch.utils.data import Dataset, Sampler

from data.factflow_fmri_dataset import FactFlowfMRIDataset

logger = logging.getLogger(__name__)


def build_subject_datasets(
    subjects: List[int],
    base_kwargs: dict,
    mode: str,
    avg_reps: bool,
) -> List[FactFlowfMRIDataset]:
    """Instantiate one ``FactFlowfMRIDataset`` per subject.

    ``base_kwargs`` carries the shared config (data_dir, pad_to, fmri_mode,
    features, subdirs, …); ``subject`` and ``n_voxels`` are injected per subject
    from the ``n_voxels_map``.
    """
    n_voxels_map = base_kwargs.pop("n_voxels_map")
    datasets = []
    for s in subjects:
        ds = FactFlowfMRIDataset(
            mode=mode,
            subject=s,
            n_voxels=int(n_voxels_map[s]),
            avg_reps=avg_reps,
            **base_kwargs,
        )
        datasets.append(ds)
    return datasets


class MultiSubjectDataset(Dataset):
    """Concatenate per-subject datasets and tag each sample with ``subject_id``.

    ``subject_id`` is the *contiguous index* (position in ``subject_datasets``),
    which is what the model's per-subject adapters are indexed by — not the raw
    NSD subject number. ``subject_nums`` keeps the mapping for logging/eval.
    """

    def __init__(self, subject_datasets: List[FactFlowfMRIDataset]):
        super().__init__()
        self.subject_datasets = subject_datasets
        self.subject_nums = [ds.subject for ds in subject_datasets]

        # Global-index → (subject_idx, local_idx) layout via cumulative bounds.
        self.boundaries: List[Tuple[int, int]] = []
        start = 0
        for ds in subject_datasets:
            end = start + len(ds)
            self.boundaries.append((start, end))
            start = end
        self.total = start

        logger.info(
            "MultiSubjectDataset: subjects=%s  sizes=%s  total=%d",
            self.subject_nums, [len(d) for d in subject_datasets], self.total,
        )

    def __len__(self) -> int:
        return self.total

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        for sidx, (start, end) in enumerate(self.boundaries):
            if start <= global_idx < end:
                return sidx, global_idx - start
        raise IndexError(global_idx)

    def __getitem__(self, global_idx: int):
        sidx, local_idx = self._locate(global_idx)
        sample = self.subject_datasets[sidx][local_idx]
        sample["subject_id"] = sidx          # contiguous adapter index
        return sample


class SubjectBatchSampler(Sampler):
    """Yield batches whose indices all belong to a single subject.

    Each subject's index range is split into fixed-size batches; the order of
    batches (across all subjects) is shuffled every epoch so the optimizer
    alternates subjects. ``drop_last`` keeps batch shapes uniform.
    """

    def __init__(
        self,
        boundaries: List[Tuple[int, int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.boundaries = boundaries
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._num_batches = sum(
            self._count(start, end) for (start, end) in boundaries
        )

    def _count(self, start: int, end: int) -> int:
        n = end - start
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for (start, end) in self.boundaries:
            idxs = list(range(start, end))
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return self._num_batches
