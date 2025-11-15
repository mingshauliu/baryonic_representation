# MSE loss version
# Three encoder/decoer levels 

import numpy as np
import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader
import gc

from utils import AstroMapDataset
from models import UNetScalarField, ResNetBranch
import Pk_library as PKL

def _xcorr_metric(delta1, delta2):
    BoxSize = 25
    threads = 1
    axis    = 0
    delta1 = (delta1-delta1.mean())/delta1.std()
    delta2 = (delta2-delta2.mean())/delta2.std()
    Pk = PKL.XPk([delta1,delta2], BoxSize, axis, MAS=['CIC','CIC'], threads=1)
    k , xpk = Pk.k1D, Pk.PkX1D[:,0]  / np.sqrt(Pk.Pk1D[:,0] * Pk.Pk1D[:,1])
    mask = k <= 15
    k_cut = k[mask]
    xpk_cut = xpk[mask]
    return np.trapz(xpk_cut, k_cut) / (k_cut.max() - k_cut.min())

 
class RandomRotateFlip3D:
    """Apply random 90° rotation + random flips along chosen dims.
       Works on both CPU and GPU tensors.
    """
    def __init__(self, dims=(2,3,4)):
        # dims correspond to spatial dims for 5D tensors: (B,C,D,H,W)
        self.dims = dims
        self.axis_pairs = [(2,3), (3,4), (2,4)]

    def __call__(self, *tensors):
        # tensors are already on CPU or GPU depending on where __call__ is executed
        
        # ---- Random rotation ----
        k = torch.randint(0, 4, (1,)).item()
        axes = self.axis_pairs[torch.randint(0, len(self.axis_pairs), (1,)).item()]

        tensors = tuple(torch.rot90(t, k, axes) for t in tensors)

        # ---- Random flips ----
        for dim in self.dims:
            if torch.rand(1).item() < 0.5:
                tensors = tuple(torch.flip(t, [dim]) for t in tensors)

        return tensors


class AstroFlowMatchingDataModule(pl.LightningDataModule):
    """flow matching data pairs"""
    
    def __init__(self, 
                 cdm_mass_maps: np.ndarray,
                 gas_maps: np.ndarray,
                 vcdm_maps: np.ndarray,
                 cosmo_params: np.ndarray,
                 batch_size: int = 8,
                 val_split: float = 0.2,
                 num_workers: int = 4
                ):
        super().__init__()
        self.cdm_mass_maps = cdm_mass_maps.astype(np.float32)
        self.gas_maps = gas_maps.astype(np.float32)
        self.vcdm_maps = vcdm_maps.astype(np.float32)
        self.cosmo_params = cosmo_params.astype(np.float32)
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.train_transform = RandomRotateFlip3D()
        
    def setup(self, stage: Optional[str] = None):
        # Split data
        n_samples = len(self.cdm_mass_maps)
        n_val = int(n_samples * self.val_split)
        n_train = n_samples - n_val
        
        indices = np.random.permutation(n_samples)
        # indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_cdm_mass = self.cdm_mass_maps[train_indices]
        train_gas_maps = self.gas_maps[train_indices]
        train_vcdm_maps = self.vcdm_maps[train_indices]
        train_cosmo_params = self.cosmo_params[train_indices]
        
        val_cdm_mass = self.cdm_mass_maps[val_indices]
        val_gas_maps = self.gas_maps[val_indices]
        val_vcdm_maps = self.vcdm_maps[val_indices]
        val_cosmo_params = self.cosmo_params[val_indices]
        
        self.train_dataset = AstroMapDataset(
            train_cdm_mass,
            train_gas_maps,
            train_vcdm_maps,
            params=train_cosmo_params,
            # transform=self.train_transform,
        )
        self.val_dataset = AstroMapDataset(
            val_cdm_mass,
            val_gas_maps,   
            val_vcdm_maps,
            params=val_cosmo_params,
            # transform=None,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  
            persistent_workers=True if self.num_workers > 0 else False,  
            drop_last=True,  # Don't drop incomplete batches,
            prefetch_factor=1 if self.num_workers > 0 else 2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,  
            persistent_workers=True if self.num_workers > 0 else False,  
            drop_last=False,  # Don't drop incomplete batches
            prefetch_factor=1 if self.num_workers > 0 else 2
        )

class FlowMatchingModel(pl.LightningModule):
    """Flow Matching model for transforming total mass maps to star maps"""
    
    def __init__(self, 
                 architecture='unet',
                 noise_std=0.0,
                 learning_rate=1e-4,
                 beta = 0.02):
        super().__init__()
        self.save_hyperparameters()
        self.weight_decay= 1e-3
        self.learning_rate = learning_rate
        self.beta = beta
        self.noise_std = noise_std
        self.scalar_field = UNetScalarField(in_channels=3, base_channels=128, out_channels=1)
        self.resnet_branch = ResNetBranch(in_channels=1, embedding_dim=16)
        self.train_transform = RandomRotateFlip3D()
        
        if hasattr(self.scalar_field, 'enable_gradient_checkpointing'):
            self.scalar_field.enable_gradient_checkpointing()
            
        # Enable memory efficient training
        self.automatic_optimization = True
        self.automatic_logging = True
            
    def sample_time(self, batch_size, device):
        """Sample random times for flow matching"""
        return torch.rand(batch_size, device=device, dtype=torch.float32)
    
    def forward(self, x, t, params, embed):
        return self.scalar_field(x, t, params, embed)
    
    def integrate(self, cdm, vcdm, params, zbar, num_steps=50):
        """Integrate the flow from t=0 to t=1 using Euler method"""
        device = cdm.device
        batch_size = cdm.size(0)
        dt = torch.tensor(1.0 / num_steps, dtype=torch.float32, device=device)
        
        x = cdm.clone()
        if self.noise_std>0:
            noise = torch.randn_like(x)*self.noise_std
            x = x + noise
            del noise
            
        for i in range(num_steps):
            tt = torch.full((batch_size,), i * dt, device=device, dtype=torch.float32)
            input_field = torch.cat([x, cdm, vcdm], dim=1)  # Shape: (batch, 3, D, H, W)
            field_change = self(input_field, tt, params, zbar)  # Predict scalar field
            x = x + dt * field_change  # Euler update

        return x
    
    def training_step(self, batch, batch_idx):

        cdm_mass, target_maps, vcdm_maps, params = batch

        batch_size = cdm_mass.size(0)
        device = cdm_mass.device
        
        cdm_mass = cdm_mass.to(device)
        target_maps = target_maps.to(device)
        vcdm_maps = vcdm_maps.to(device)
        params = params.to(device)

        # ---- APPLY GPU AUGMENTATION ----
        if self.trainer.training:  # only train, not val
            cdm_mass, target_maps, vcdm_maps = self.train_transform(cdm_mass, target_maps, vcdm_maps)
        
        t = self.sample_time(batch_size, device)
        x0 = cdm_mass
        x1 = target_maps

        if self.noise_std>0:
            noise = torch.randn_like(x0)*self.noise_std
            x0 = x0 + noise
            del noise
             
        # Interpolate between x0 and x1
        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Compute target scalar field
        target_velocity = x1-x0
        del x0, x1

        # Compute ResNet embedding here (maintains same training behavior)
        resnet_embed = self.resnet_branch(target_maps)  # Shape: (batch, 8)
        embed_mean = resnet_embed[:, :8]
        embed_log_std = resnet_embed[:, 8:]  # Directly use log std for numerical stability
        eps = torch.randn_like(embed_mean)
        embed = embed_mean + torch.exp(embed_log_std) * eps
        
        input_field = torch.cat([x_t, cdm_mass, vcdm_maps], dim=1)  # Shape: (batch, 3, D, H, W)
        
        # Predict scalar field
        predicted_velocity = self(input_field, t, params, embed)
        kl_loss = 0.5 * (torch.exp(2*embed_log_std) + embed_mean**2 - 1 - 2*embed_log_std).mean()
        loss = F.mse_loss(predicted_velocity, target_velocity) + self.beta * kl_loss
                
        self.log('train_kl_loss', kl_loss, on_step=False, on_epoch=True)                
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        del predicted_velocity, target_velocity, x_t
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        cdm_mass, target_maps, vcdm_map, params = batch

        batch_size = cdm_mass.size(0)
        device = cdm_mass.device

        t = self.sample_time(batch_size, device)

        x0 = cdm_mass
        x1 = target_maps                              # make sure this is (B,1,D,H,W); see §2

        if self.noise_std > 0:
            noise = torch.randn_like(x0) * self.noise_std
            x0 = x0 + noise
            del noise

        t_expanded = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Compute ResNet embedding here (maintains same training behavior)
        resnet_embed = self.resnet_branch(target_maps)  # Shape: (batch, 8)
        embed_mean = resnet_embed[:, :8]
        embed_log_std = resnet_embed[:, 8:]  # Directly use log std for numerical stability
        eps = torch.randn_like(embed_mean)
        embed = embed_mean + torch.exp(embed_log_std) * eps
        
        input_field = torch.cat([x_t, cdm_mass, vcdm_map], dim=1)  # Shape: (batch, 3, D, H, W)
        target_velocity = x1 - x0
        
        # Predict scalar field
        predicted_velocity = self(input_field, t, params, embed)
        kl_loss = 0.5 * (torch.exp(2*embed_log_std) + embed_mean**2 - 1 - 2*embed_log_std).mean()
        loss = F.mse_loss(predicted_velocity, target_velocity) + self.beta * kl_loss
                
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)                
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if self.trainer.sanity_checking:
            return loss
        else:
            if (self.current_epoch % 10 == 0) and (batch_idx == 0):
                # compute xcorr only once per 5 epochs on the first batch
                with torch.no_grad():

                    x = self.integrate(cdm_mass, vcdm_map, params, embed, num_steps=100)
                    
                    target_np = target_maps[0, 0].detach().cpu().numpy()
                    np.save(f'sample/epoch{self.current_epoch}_target.npy', target_np)
                    field_np = x[0, 0].detach().cpu().numpy()
                    np.save(f'sample/epoch{self.current_epoch}_sample.npy', field_np)
                    xcorr_val = _xcorr_metric(
                        field_np,
                        target_np
                    )
                    
                    self.log("xcorr", xcorr_val, prog_bar=True, on_step=False, on_epoch=True)

            del predicted_velocity, target_velocity, x_t
            return loss

        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
        
    
    def sample(self, cdm_mass, vcdm_map, cosmo_params, embed=None, resnet_input=None, num_steps=100, method='euler'):
        self.eval()
        device = next(self.parameters()).device
        batch_size = cdm_mass.size(0)
        noise_std = 0.0
        noise_std = self.noise_std

        if embed is None:
            if resnet_input is not None:
                embed = self.resnet_branch(resnet_input)
                embed_mean = embed[:, :8]
                embed_std  = F.softplus(embed[:, 8:])  # ensure std > 0
                
                eps = torch.randn_like(embed_mean)
                zbar = embed_mean + embed_std * eps
            else: 
                print("Missing ResNet inputs")
                return None
        else:
            zbar = embed   
            
        with torch.no_grad():
            x = self.integrate(cdm_mass, vcdm_map, cosmo_params, zbar, num_steps=num_steps)

        return x