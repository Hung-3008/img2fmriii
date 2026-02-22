"""
verify_mapping.py
=================
Verify data mapping by reconstructing 5 random train samples via MindEye2.
Plots original images with captions alongside raw + enhanced reconstructions.

Usage:
  python src/utils/verify_mapping.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add MindEye2 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from infer_mindeye2 import reconstruct

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../Data'))
NSD_DIR = os.path.join(BASE_DIR, 'nsd', 'subj01')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evals', 'verify_mapping')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5
SEED = 42
np.random.seed(SEED)

# ==============================================================================
# Load data
# ==============================================================================
print("Loading train data...")
stim = np.load(os.path.join(NSD_DIR, 'nsd_train_stim_sub1.npy'))        # (9000, 425, 425, 3)
fmri = np.load(os.path.join(NSD_DIR, 'nsd_train_fmri_zscore_sub1.npy')) # (9000, 3, 15724)
caps = np.load(os.path.join(NSD_DIR, 'nsd_train_cap_sub1.npy'))         # (9000, 5)

print(f"  Stimuli: {stim.shape}")
print(f"  fMRI:    {fmri.shape}")
print(f"  Captions:{caps.shape}")

# Random 5 indices
indices = np.random.choice(stim.shape[0], size=NUM_SAMPLES, replace=False)
indices = np.sort(indices)
print(f"\nSelected indices: {indices}")

# Extract subsets
selected_stim = stim[indices]           # (5, 425, 425, 3)
selected_fmri = fmri[indices]           # (5, 3, 15724)
selected_caps = [caps[i, 0] for i in indices]  # first caption for each

print("\nCaptions for selected samples:")
for i, idx in enumerate(indices):
    print(f"  [{idx}] {selected_caps[i]}")

# ==============================================================================
# Reconstruct via MindEye2 (providing captions to skip Stage 1.5)
# ==============================================================================
print("\n" + "="*60)
print("Running MindEye2 reconstruction...")
print("="*60)

result = reconstruct(
    fmri_data=selected_fmri,
    captions=selected_caps,
    output_dir=OUTPUT_DIR,
    cache_dir=os.path.join(BASE_DIR, 'checkpoints'),
    seed=SEED,
)

# ==============================================================================
# Plot: Original | Raw Recon | Enhanced Recon
# ==============================================================================
print("\nPlotting results...")

fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))

for i in range(NUM_SAMPLES):
    idx = indices[i]
    caption = selected_caps[i]

    # Column 0: Original image
    axes[i, 0].imshow(selected_stim[i])
    axes[i, 0].set_title(f"Original [idx={idx}]\n\"{caption[:60]}...\"" if len(caption) > 60
                         else f"Original [idx={idx}]\n\"{caption}\"",
                         fontsize=10, wrap=True)
    axes[i, 0].axis('off')

    # Column 1: Raw reconstruction (Stage 2 output)
    axes[i, 1].imshow(result['raw_images'][i])
    axes[i, 1].set_title(f"Raw Recon (unCLIP)", fontsize=10)
    axes[i, 1].axis('off')

    # Column 2: Enhanced reconstruction (Stage 3 output)
    axes[i, 2].imshow(result['enhanced_images'][i])
    axes[i, 2].set_title(f"Enhanced (SDXL)", fontsize=10)
    axes[i, 2].axis('off')

plt.suptitle("MindEye2 Mapping Verification — Train Set (Subject 1)", fontsize=14, y=1.01)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "verify_mapping_train.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved comparison plot to: {output_path}")
print("Done!")
