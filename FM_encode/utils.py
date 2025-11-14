import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl

class AstroMapDataset(Dataset):
    def __init__(self, total_mass_maps: np.ndarray, gas_maps: np.ndarray, vcdm_maps:np.ndarray, params: np.ndarray, transform=None):
        self.total_mass_maps = torch.FloatTensor(total_mass_maps)
        self.vcdm_maps = torch.FloatTensor(vcdm_maps)
        
        # Stack star and gas maps into 2-channel target
        gas_tensor = torch.FloatTensor(gas_maps)
        self.params = torch.FloatTensor(params)
        self.transform = transform
        
        # Normalize
        norm = np.load('../norm3d.npy',allow_pickle=True).item()
        tot_mean, tot_std = norm['dm']['mean'], norm['dm']['std']
        gas_mean, gas_std = norm['gas']['mean'], norm['gas']['std']
        vcdm_mean, vcdm_std = norm['vcdm']['mean'], norm['vcdm']['std']
        print(f"Normalising tot log maps, mean: {tot_mean}, std: {tot_std}")
        print(f"Normalising gas log maps, mean: {gas_mean}, std: {gas_std}")
        print(f"Normalising vcdm log maps, mean: {vcdm_mean}, std: {vcdm_std}")
        self.total_mass_maps = (self.total_mass_maps - tot_mean) / tot_std
        self.vcdm_maps = (self.vcdm_maps - vcdm_mean) / vcdm_std
        self.target_maps = (gas_tensor - gas_mean) / gas_std
        
    def __len__(self):
        return len(self.total_mass_maps)
    
    def __getitem__(self, idx):
        total_mass = self.total_mass_maps[idx].unsqueeze(0)
        target_map = self.target_maps[idx].unsqueeze(0)
        vcdm_map = self.vcdm_maps[idx].unsqueeze(0)
        cosmo_param = self.params[idx]
        
        if self.transform is not None:
            # Apply same transformation to all tensors at once
            total_mass, target_map, vcdm_map = self.transform(total_mass, target_map, vcdm_map)
        
        return total_mass, target_map, vcdm_map, cosmo_param