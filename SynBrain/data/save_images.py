import numpy as np
from PIL import Image
import os

# Load npy format Stimuli
data_dir = "/workspace/sdb1/img2fmri/NSD/data/nsd"
subjects = [1]
modes = ['test', 'train']

for sub in subjects:
    for mode in modes:
        npy_path = f'{data_dir}/subj0{sub}/nsd_{mode}_stim_sub{sub}.npy'
        if not os.path.exists(npy_path):
            print(f"File not found: {npy_path}, skipping...")
            continue
            
        print(f"Processing Subject {sub}, Mode {mode}...")
        data = np.load(npy_path)
        print(f"Data shape: {data.shape}")

        output_dir = f'{data_dir}/subj0{sub}/{mode}_img'
        os.makedirs(output_dir, exist_ok=True)

        for i in range(data.shape[0]):
            img = Image.fromarray(data[i].astype(np.uint8))  
            img.save(os.path.join(output_dir, f"{i}.png"))

        print(f"Saved images to: {output_dir}")