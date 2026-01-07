import os
from huggingface_hub import hf_hub_download
import shutil

# Target directory
models_dir = "/workspace/sdb1/img2fmri/models"
os.makedirs(models_dir, exist_ok=True)

print(f"Downloading models to {models_dir}...")

# 1. Download CLIP-ViT-bigG-14-laion2B-39B-b160k
# Expecting 'open_clip_pytorch_model.bin'
print("Downloading CLIP-ViT-bigG-14-laion2B-39B-b160k...")
clip_dest = os.path.join(models_dir, "open_clip_pytorch_model.bin")
if os.path.exists(clip_dest):
    print(f"File already exists: {clip_dest}. Skipping download.")
else:
    try:
        clip_path = hf_hub_download(
            repo_id="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            filename="open_clip_pytorch_model.bin",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded: {clip_path}")
    except Exception as e:
        print(f"Error downloading CLIP model: {e}")

# 2. Download unclip6_epoch0_step110000.ckpt from pscotti/mindeyev2
print("Downloading unclip6_epoch0_step110000.ckpt...")
try:
    # Check if file exists in the repo root or subfolder. 
    # Usually checkpoints are in root.
    ckpt_path = hf_hub_download(
        repo_id="pscotti/mindeyev2",
        filename="unclip6_epoch0_step110000.ckpt",
        repo_type="dataset",
        local_dir=models_dir,
        local_dir_use_symlinks=False
    )
    print(f"Successfully downloaded: {ckpt_path}")
except Exception as e:
    print(f"Error downloading unclip checkpoint: {e}")

print("Download complete.")
