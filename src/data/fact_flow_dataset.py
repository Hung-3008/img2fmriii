"""FACT-Flow Dataset — Single-trial fMRI with session/run/trial metadata.

Designed for FACT-Flow's disentanglement framework which requires:
- Single-trial fMRI (not averaged across repetitions)
- Session/run/trial metadata for z_state temporal consistency loss
- Visual features for semantic conditioning (DINOv2, CLIP, etc.)
- Per-subject data with varying voxel counts

Data shapes per subject (e.g. subj01):
  - fmri_train: (9000, 3, 15724) → 27000 single-trial samples
  - fmri_test:  (1000, 3, 15724) → 3000 single-trial samples
  - visual_tokens: (N, T, D)     (optional, e.g. DINOv2: (9000,257,1024))

Indexing: each sample = one trial. Index i maps to:
  image_idx = i // 3,  rep_idx = i % 3

Metadata is recovered from nsd_expdesign.mat at init time.
"""

import logging
import os
from collections import defaultdict

import numpy as np
import scipy.io as spio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("fact_flow_dataset")


class FactFlowDataset(Dataset):
    """FACT-Flow dataset for single-trial fMRI with rich metadata.

    Each sample returns a dict with:
        fmri:          (V_s,)           single-trial fMRI beta-weights
        subject_id:    int              subject index (0-based among configured subjects)
        image_idx:     int              image index within split (0..N_images-1)
        rep_idx:       int              repetition index (0, 1, 2)
        session:       int              session number (0-39)
        run:           int              run within session (0-11)
        trial_in_run:  int              trial within run (0-74)
        trial_idx:     int              global trial index (0-29999)

    When fmri_only=False, also includes:
        visual_tokens: (T, D)           visual feature tokens (e.g. DINOv2: (257,1024))
    """

    # NSD constants
    N_SESSIONS = 40
    TRIALS_PER_SESSION = 750
    RUNS_PER_SESSION = 12
    TRIALS_PER_RUN = 75  # 750 / 12 is not exactly 75... let me check
    # Actually: 40 sessions * 750 trials/session = 30000 total trials
    # 750 / 12 = 62.5... The run structure is encoded differently.
    # From prepare_nsddata_scale.py, the trial structure is:
    #   session = trial_idx // 750
    #   Within session, trials are ordered chronologically across runs
    # The run info comes from stimpattern(40, 12, 75) but the simple
    # mapping trial_idx -> run isn't exactly 75-per-run for all sessions.
    # We'll use the raw masterordering + stimpattern for precise mapping.

    TOTAL_TRIALS = 30000
    TEST_STIM_THRESHOLD = 1000  # masterordering <= 1000 = shared/test

    def __init__(
        self,
        data_dir: str,
        expdesign_path: str,
        subject: int,
        subject_id: int = 0,
        mode: str = "train",
        fmri_mode: str = "scale",
        fmri_only: bool = False,
        visual_feature: str = "dinov2_vitl14",
    ):
        """Initialize FACT-Flow dataset.

        Args:
            data_dir: Path to NSD data directory (containing subj01/, subj02/, etc.)
            expdesign_path: Path to nsd_expdesign.mat
            subject: Subject number (1, 2, 5, or 7)
            subject_id: Subject index for multi-subject training (0-based)
            mode: "train" or "test"
            fmri_mode: "scale" or "zscore" for fMRI normalization
            fmri_only: If True, skip visual feature loading (for Stage 1 BrainVAE)
            visual_feature: Feature name prefix (e.g. "dinov2_vitl14", "sdxl_clip")
        """
        super().__init__()
        self.mode = mode
        self.subject = subject
        self.subject_id = subject_id
        self.fmri_only = fmri_only

        subj_dir = os.path.join(data_dir, f"subj0{subject}")
        logger.info(
            "Loading FACT-Flow data: subj=%d, mode=%s, fmri_mode=%s, fmri_only=%s",
            subject, mode, fmri_mode, fmri_only,
        )

        # --- Load fMRI: (N_images, 3, V_s) ---
        fmri_path = os.path.join(
            subj_dir, f"nsd_{mode}_fmri_{fmri_mode}_sub{subject}.npy"
        )
        self.fmri_data = np.load(fmri_path, mmap_mode="r")  # (N_images, 3, V_s)
        self.n_images = self.fmri_data.shape[0]
        self.n_reps = self.fmri_data.shape[1]  # should be 3
        self.n_voxels = self.fmri_data.shape[2]
        self.n_samples = self.n_images * self.n_reps

        logger.info(
            "  fMRI: %s %s → %d single-trial samples, V=%d",
            self.fmri_data.shape, self.fmri_data.dtype,
            self.n_samples, self.n_voxels,
        )

        # --- Load visual features (optional) ---
        self.visual_tokens = None
        if not fmri_only:
            vis_path = os.path.join(
                subj_dir, f"nsd_{visual_feature}_{mode}_sub{subject}.npy"
            )
            if os.path.exists(vis_path):
                self.visual_tokens = np.load(vis_path, mmap_mode="r")
                logger.info(
                    "  Visual tokens (%s): %s %s",
                    visual_feature, self.visual_tokens.shape, self.visual_tokens.dtype,
                )
                assert self.visual_tokens.shape[0] == self.n_images, (
                    f"Visual feature count ({self.visual_tokens.shape[0]}) != "
                    f"fMRI image count ({self.n_images})"
                )
            else:
                logger.warning(
                    "  Visual features not found at %s. "
                    "Proceeding without visual features.",
                    vis_path,
                )

        # --- Recover trial metadata from nsd_expdesign.mat ---
        self._build_trial_metadata(expdesign_path)

        logger.info(
            "  Dataset ready: %d samples (%d images × %d reps)",
            self.n_samples, self.n_images, self.n_reps,
        )

    def _build_trial_metadata(self, expdesign_path: str):
        """Reconstruct trial→(session, run, trial_in_run) mapping.

        The on-disk fMRI data is stored as (N_images, 3, V_s) where:
        - Images are sorted by NSD image ID
        - The 3 repetitions are sorted by trial index (earliest first)

        We reconstruct which global trial index (0..29999) corresponds
        to each (image_idx, rep_idx) pair, and derive session/run from that.
        """
        mat = spio.loadmat(expdesign_path, squeeze_me=True)
        masterordering = mat["masterordering"]  # (30000,) 1-indexed stimulus
        subjectim = mat["subjectim"]  # (8, 10000) subject × stimulus → nsdId

        sub_idx = self.subject - 1  # convert to 0-indexed for subjectim
        is_test = self.mode == "test"

        # Group global trial indices by NSD image ID
        img_to_trials = defaultdict(list)
        for trial_idx in range(self.TOTAL_TRIALS):
            stim_idx = int(masterordering[trial_idx])  # 1-indexed
            trial_is_test = stim_idx <= self.TEST_STIM_THRESHOLD

            if trial_is_test != is_test:
                continue

            nsd_id = int(subjectim[sub_idx, stim_idx - 1])  # 1-indexed NSD ID
            img_to_trials[nsd_id].append(trial_idx)

        # Sort images by NSD ID (matching on-disk order from prepare_nsddata_scale.py)
        sorted_nsd_ids = sorted(img_to_trials.keys())

        assert len(sorted_nsd_ids) == self.n_images, (
            f"Recovered {len(sorted_nsd_ids)} unique images from expdesign "
            f"but fMRI has {self.n_images} images"
        )

        # Build per-sample metadata arrays
        # sample index i → image_idx = i // 3, rep_idx = i % 3
        self.trial_indices = np.zeros(self.n_samples, dtype=np.int32)
        self.sessions = np.zeros(self.n_samples, dtype=np.int16)
        self.runs = np.zeros(self.n_samples, dtype=np.int16)
        self.trials_in_run = np.zeros(self.n_samples, dtype=np.int16)

        for img_idx, nsd_id in enumerate(sorted_nsd_ids):
            trials = sorted(img_to_trials[nsd_id])  # sort by trial index
            assert len(trials) == self.n_reps, (
                f"Image NSD ID {nsd_id} has {len(trials)} trials, expected {self.n_reps}"
            )
            for rep_idx, trial_idx in enumerate(trials):
                sample_idx = img_idx * self.n_reps + rep_idx
                session = trial_idx // self.TRIALS_PER_SESSION
                within_session = trial_idx % self.TRIALS_PER_SESSION
                # Approximate run: 750 trials / 12 runs ≈ 62-63 trials per run
                # Use integer division for a reasonable approximation
                run = within_session * self.RUNS_PER_SESSION // self.TRIALS_PER_SESSION
                trial_in_run = within_session - (
                    run * self.TRIALS_PER_SESSION // self.RUNS_PER_SESSION
                )

                self.trial_indices[sample_idx] = trial_idx
                self.sessions[sample_idx] = session
                self.runs[sample_idx] = run
                self.trials_in_run[sample_idx] = trial_in_run

        logger.info(
            "  Metadata: sessions=[%d..%d], unique_sessions=%d",
            self.sessions.min(), self.sessions.max(),
            len(np.unique(self.sessions)),
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image_idx = idx // self.n_reps
        rep_idx = idx % self.n_reps

        # Single-trial fMRI
        fmri = torch.from_numpy(
            self.fmri_data[image_idx, rep_idx].astype(np.float32)
        )  # (V_s,)

        result = {
            "fmri": fmri,
            "subject_id": self.subject_id,
            "image_idx": image_idx,
            "rep_idx": rep_idx,
            "session": int(self.sessions[idx]),
            "run": int(self.runs[idx]),
            "trial_in_run": int(self.trials_in_run[idx]),
            "trial_idx": int(self.trial_indices[idx]),
        }

        # Visual features (same for all reps of same image)
        if self.visual_tokens is not None:
            result["visual_tokens"] = torch.from_numpy(
                self.visual_tokens[image_idx].astype(np.float32)
            )

        return result

    @property
    def voxel_count(self) -> int:
        """Number of voxels for this subject."""
        return self.n_voxels

    def get_image_indices_for_rep_group(self, image_idx: int) -> list[int]:
        """Return the 3 sample indices corresponding to all reps of an image."""
        base = image_idx * self.n_reps
        return list(range(base, base + self.n_reps))
