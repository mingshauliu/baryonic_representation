import torch
import numpy as np
from tqdm import tqdm
from module import FlowMatchingModel

def _to_tensor(x, dtype=torch.float32):
    # no device here — move later in batch
    x = torch.as_tensor(x, dtype=dtype)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return x

# --- setup ---
base_dir = '/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data'
model_name, dataset = 'IllustrisTNG', 'LH'
checkpoint_path = '/mnt/home/mliu1/FMbaseline_final/lightning_logs/kzcdk2ei/checkpoints/best-model-epoch=266-val_loss=0.026007.ckpt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FlowMatchingModel.load_from_checkpoint(checkpoint_path).to(device).eval()
# load data
cdm_mass = np.load(f'{base_dir}/{model_name}/Grids_Mcdm_{model_name}_{dataset}_128_z=0.0.npy', mmap_mode='r')
gas_maps = np.load(f'{base_dir}/{model_name}/Grids_Mgas_{model_name}_{dataset}_128_z=0.0.npy', mmap_mode='r')
vcdm_maps = np.load(f'{base_dir}/{model_name}/Grids_Vcdm_{model_name}_{dataset}_128_z=0.0.npy', mmap_mode='r')
params = np.loadtxt(f'{base_dir}/{model_name}/params_{dataset}_{model_name}.txt')
norm = np.load('norm3d.npy', allow_pickle=True).item()

# normalize once
dm_mean, dm_std = norm['dm']['mean'], norm['dm']['std']
gas_mean, gas_std = norm['gas']['mean'], norm['gas']['std']
vcdm_mean, vcdm_std = norm['vcdm']['mean'], norm['vcdm']['std']

cdm_maps = (np.log1p(cdm_mass) - dm_mean) / dm_std
vcdm_maps = (np.log1p(vcdm_maps) - vcdm_mean) / vcdm_std
cosmo_params = torch.as_tensor(params, dtype=torch.float32)

num_samples = len(cdm_mass)
batch_size = 4
num_steps = 100

outputs = []

# --- batched loop ---
for start in tqdm(range(0, num_samples, batch_size), desc="Generating predictions", ncols=80):
    end = min(start + batch_size, num_samples)

    # batch tensors
    cdm_batch = torch.stack([_to_tensor(cdm_maps[i]) for i in range(start, end)])
    vcdm_batch = torch.stack([_to_tensor(vcdm_maps[i]) for i in range(start, end)])
    cosmo_batch = cosmo_params[start:end]

    cdm_batch = cdm_batch.to(device, non_blocking=True)
    vcdm_batch = vcdm_batch.to(device, non_blocking=True)
    cosmo_batch = cosmo_batch.to(device, non_blocking=True)

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        pred = model.sample(
            cdm_mass=cdm_batch,
            vcdm_map=vcdm_batch,
            cosmo_params=cosmo_batch,
            num_steps=num_steps,
            method='euler'
        ).cpu().numpy()

    print(pred[:, 0].shape)
    outputs.append(pred[:, 0])

# --- save once ---
outputs = np.concatenate(outputs, axis=0)
np.save(f'Synth_Mgas_{model_name}_{dataset}_128.npy', outputs)