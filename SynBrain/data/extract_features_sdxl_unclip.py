import sys
import os

# Add SynBrain/src to path to find utils.py
sys.path.append('/workspace/sdb1/img2fmri/SynBrain/src')

import torch
import utils
import numpy as np
from torchvision import transforms
from PIL import Image

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
# NOTE: You may need to adjust this path to match your current workspace structure
sys.path.append('/workspace/sdb1/img2fmri/SynBrain/src/sdxl')
try:
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
except ImportError:
    print("Warning: Could not import FrozenOpenCLIPImageEmbedder. Check the sys.path.append path.")

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

device = 'cuda'

# Initialize Embedder
print("Initializing Embedder...")
# NOTE: Check if the version path exists in your environment
try:
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        # version="laion2b_s39b_b160k",
        version="/workspace/sdb1/img2fmri/NSD/data/open_clip_pytorch_model.bin",
        output_tokens=True,
        only_tokens=False,
    )
    clip_img_embedder.to(device).eval()
except Exception as e:
    print(f"Error initializing embedder: {e}")

clip_seq_dim = 256
clip_emb_dim = 1664

# Dataset and DataLoader
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from IPython.display import display
import torchvision
from torchvision import transforms

class CLIP_Image_Dataset(Dataset):
    def __init__(self, image_path):
        self.img_data = image_path   #图像path

    def __getitem__(self, idx):
        img = Image.open(self.img_data[idx])  # 一张图像对应1个fmri
        img = TF.to_tensor(img).float()
        return img

    def __len__(self):
        return len(self.img_data)


data_path="/workspace/sdb1/img2fmri/NSD/data/nsd"
batch_size=1000

print("Starting processing...")
# Processing loop
# Note: Indented to run if paths are valid, otherwise this might fail if paths don't exist
if os.path.exists(data_path):
    for subj in [1]:
        save_path = os.path.join("/workspace/sdb1/img2fmri/NSD/data/nsd", 'subj0{}'.format(subj))
        
        train_image_path = os.path.join(data_path, 'subj0{}/train_img'.format(subj))
        if not os.path.exists(train_image_path):
             print(f"Path not found: {train_image_path}")
             continue

        train_image = np.array([os.path.join(train_image_path, f'{i}.png') for i in range(len(os.listdir(train_image_path)))])
        
        print(f'Train Image Sub{subj}: {train_image.shape}')
        train_dataset = CLIP_Image_Dataset(train_image)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False)
        
        test_image_path = os.path.join(data_path, 'subj0{}/test_img'.format(subj))
            
        test_image = np.array([os.path.join(test_image_path, f'{i}.png') for i in range(len(os.listdir(test_image_path)))])
        print(f'Test Image Sub{subj}: {test_image.shape}')
        test_dataset = CLIP_Image_Dataset(test_image)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False)
        mode = 'test'
        for test_i, image in enumerate(test_dataloader):
            print(f"Test batch {test_i}")
            
            with torch.no_grad():
                z, z_pool = clip_img_embedder(image.to(device))
            
            if test_i == 0:
                zs = z.detach().cpu()
                zs_pool = z_pool.detach().cpu()
            else:
                zs = torch.cat((zs, z.detach().cpu()), dim=0)
                zs_pool = torch.cat((zs_pool, z_pool.detach().cpu()), dim=0)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        np.save(os.path.join(save_path, f'nsd_{mode}_clip_sub{subj}.npy'), zs.numpy())
        np.save(os.path.join(save_path, f'nsd_{mode}_clip_pool_sub{subj}.npy'), zs_pool.numpy())
        del z, zs, zs_pool, image
        
        mode = 'train'
        for train_i, image in enumerate(train_dataloader):
            print(f"Train batch {train_i}")
            
            with torch.no_grad():
                z, z_pool = clip_img_embedder(image.to(device))
            
            if train_i == 0:
                zs = z.detach().cpu()
                zs_pool = z_pool.detach().cpu()
            else:
                zs = torch.cat((zs, z.detach().cpu()), dim=0)
                zs_pool = torch.cat((zs_pool, z_pool.detach().cpu()), dim=0)

        np.save(os.path.join(save_path, f'nsd_{mode}_clip_sub{subj}.npy'), zs.numpy())
        np.save(os.path.join(save_path, f'nsd_{mode}_clip_pool_sub{subj}.npy'), zs_pool.numpy())
        del z, zs, zs_pool, image
else:
    print(f"Data path not found: {data_path}. Skipping processing loop.")


# Single batch check
subj = 1
# Redefine to avoid errors if loop skipped
train_image_path = os.path.join(data_path, 'subj0{}/train_img'.format(subj))
if os.path.exists(train_image_path):
    train_image_path = np.array([os.path.join(train_image_path, f'{i}.png') for i in range(len(os.listdir(train_image_path)))])

    transform = transforms.ToTensor()

    tensors = []
    for path in train_image_path[:10]:
        img = Image.open(path).convert('RGB')   # 保证RGB三通道
        tensor = transform(img)                 # 转为 tensor [3, H, W], float32
        tensors.append(tensor)

    image_batch = torch.stack(tensors)

    if 'clip_img_embedder' in locals():
        clip_target = clip_img_embedder(image_batch.to(device)) #!处理IMAGE
else:
    print("Train image path for single batch check not found.")


# Diffusion Engine Setup
try:
    from models import *
    from omegaconf import OmegaConf
    from generative_models.sgm.models.diffusion import DiffusionEngine

    # prep unCLIP
    config_path = "/workspace/sdb1/img2fmri/SynBrain/src/sdxl/generative_models/configs/unclip6.yaml"
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        unclip_params = config["model"]["params"]
        network_config = unclip_params["network_config"]
        denoiser_config = unclip_params["denoiser_config"]
        first_stage_config = unclip_params["first_stage_config"]
        conditioner_config = unclip_params["conditioner_config"]
        sampler_config = unclip_params["sampler_config"]
        scale_factor = unclip_params["scale_factor"]
        disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
        offset_noise_level = unclip_params["loss_fn_config"]["params"]["offset_noise_level"]

        first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
        sampler_config['params']['num_steps'] = 38

        diffusion_engine = DiffusionEngine(network_config=network_config,
                            denoiser_config=denoiser_config,
                            first_stage_config=first_stage_config,
                            conditioner_config=conditioner_config,
                            sampler_config=sampler_config,
                            scale_factor=scale_factor,
                            disable_first_stage_autocast=disable_first_stage_autocast)
        # set to inference
        diffusion_engine.eval().requires_grad_(False)
        diffusion_engine.to(device)

        ckpt_path = f'/workspace/sdb1/img2fmri/NSD/data/unclip6_epoch0_step110000.ckpt'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            diffusion_engine.load_state_dict(ckpt['state_dict'])

            batch={"jpg": torch.randn(1,3,1,1).to(device), # jpg doesnt get used, it's just a placeholder
                "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
                "crop_coords_top_left": torch.zeros(1, 2).to(device)}
            out = diffusion_engine.conditioner(batch)
            vector_suffix = out["vector"].to(device)
            print("vector_suffix", vector_suffix.shape)
        else:
            print(f"Checkpoint not found: {ckpt_path}")
    else:
         print(f"Config path not found: {config_path}")

except ImportError:
    print("Could not import diffusion models components.")
