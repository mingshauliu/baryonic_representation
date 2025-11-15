import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import numpy as np

from module import AstroFlowMatchingDataModule, FlowMatchingModel

def train_flow_matching_model(cdm_mass_maps, gas_maps, vcdm_maps, cosmo_params,
                            architecture='unet',
                            noise_std=0.0,
                            max_epochs=300,
                            batch_size=16,
                            patience=20):
    
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
    
    data_module = AstroFlowMatchingDataModule(
        cdm_mass_maps=cdm_mass_maps,
        gas_maps=gas_maps,
        vcdm_maps=vcdm_maps,
        cosmo_params=cosmo_params,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=6
    )
    
    model = FlowMatchingModel(
        architecture=architecture,
        noise_std=noise_std,
        learning_rate=1e-4
    )
    
    # early_stop = EarlyStopping(
    #     monitor='val_loss',
    #     verbose=True,
    #     mode='min'
    # )
    
    checkpoint = ModelCheckpoint(
        monitor='val_loss',
        filename='best-model-{epoch:02d}-{val_loss:.6f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    # ckpt_path = None
    # ckpt_dir = '/mnt/home/mliu1/FMbaseline_final2/lightning_logs/o6p5mo2p/checkpoints/'
    # if os.path.isdir(ckpt_dir):
    #     ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    #     if ckpts:
    #         ckpt_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])  # load latest checkpoint
    #         print(f"Resuming from checkpoint: {ckpt_path}")
    #     else:
    #         print("No checkpoint found. Training from scratch.")


    logger = WandbLogger(log_model="False")
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        precision='16-mixed',
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        check_val_every_n_epoch=5,
        log_every_n_steps=50,
        callbacks=[checkpoint, lr_cb],
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        detect_anomaly=False,
        use_distributed_sampler=False,
        limit_train_batches=1.0,
        limit_val_batches=1.0,  
        num_sanity_val_steps=2,
    )
    
    # trainer.fit(model, data_module, ckpt_path=ckpt_path)
    trainer.fit(model, data_module)
    
    print(f"Best model saved at: {checkpoint.best_model_path}")
    
    return model, trainer

if __name__ == "__main__":

    np.random.seed(42)
    
    config={
        'models': ['IllustrisTNG', 'SIMBA', 'Astrid', 'EAGLE'],  # List of models to include
        'samples_per_model': 1000,  # Number of samples to load from each model
        'noise_std': 0.2,
        'architecture': 'unet',
        'max_epochs': 1000,
        'batch_size': 4,  # Reduce batch size to fit in memory
        'patience': None 
    }
    print('Configurations:',config)
    
    # Load data from multiple models
    all_cdm_mass_maps = []
    all_gas_maps = []
    all_vcdm_maps = []
    all_cosmo_params = []
    
    for model_name in config['models']:
        
        dataset='LH'
            
        print(f"Loading data from {model_name} {dataset} set...")

        if model_name == 'EAGLE':
            base_dir = '/mnt/home/mliu1/CAMELS-cube' # EAGLE-Swift is handmade
        else:
            base_dir = f'/mnt/ceph/users/camels/PUBLIC_RELEASE/CMD/3D_grids/data/{model_name}'

        cdm_mass = np.load(f'{base_dir}/Grids_Mcdm_{model_name}_{dataset}_128_z=0.0.npy',mmap_mode='r')
        gas_maps = np.load(f'{base_dir}/Grids_Mgas_{model_name}_{dataset}_128_z=0.0.npy',mmap_mode='r')
        vcdm_maps = np.load(f'{base_dir}/Grids_Vcdm_{model_name}_{dataset}_128_z=0.0.npy',mmap_mode='r')
        params = np.loadtxt(f'{base_dir}/params_{dataset}_{model_name}.txt')[:,:2] # Cosmological constants only: Omega_m, sigma_8
        
        n_samples = min(config['samples_per_model'], len(cdm_mass))
        if len(cdm_mass) > n_samples:
            # Randomly sample indices
            indices = np.random.choice(len(cdm_mass), n_samples, replace=False)
            cdm_mass = cdm_mass[indices]
            gas_maps = gas_maps[indices]
            vcdm_maps = vcdm_maps[indices]
            params = params[indices] 

        print(f"  Loaded {len(cdm_mass)} samples from {model_name}")
        
        # Apply log1p transformation
        cdm_mass = np.log1p(cdm_mass)
        gas_maps = np.log1p(gas_maps)
        # Vcdm DOES NOT NEED LOG TRANSFORM
        # vcdm_maps = np.log1p(vcdm_maps)
        
        all_cdm_mass_maps.append(cdm_mass)
        all_gas_maps.append(gas_maps)
        all_vcdm_maps.append(vcdm_maps)
        all_cosmo_params.append(params)
    
    # Combine all data
    cdm_mass_maps = np.concatenate(all_cdm_mass_maps, axis=0)
    gas_maps = np.concatenate(all_gas_maps, axis=0)
    vcdm_maps = np.concatenate(all_vcdm_maps, axis=0)
    cosmo_params = np.concatenate(all_cosmo_params, axis=0)
    
    # Memory cleanup before training
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Combined dataset sizes:")
    print(f"  Total mass maps: {cdm_mass_maps.shape}")
    print(f"  Gas maps: {gas_maps.shape}")
    print(f"  Astro params: {cosmo_params.shape}")
    print(f"  Training on {len(config['models'])} models: {config['models']}")
    
    # Estimate memory usage
    total_samples = cdm_mass_maps.shape[0]
    if len(cdm_mass_maps.shape) == 4:  # (N, D, H, W)
        map_size = cdm_mass_maps.shape[1] * cdm_mass_maps.shape[2] * cdm_mass_maps.shape[3]
    else:  # (N, C, D, H, W)
        map_size = cdm_mass_maps.shape[2] * cdm_mass_maps.shape[3] * cdm_mass_maps.shape[4]
    estimated_memory_gb = (total_samples * map_size * 4 * 5) / (1024**3) # for 3 data types
    print(f"Estimated memory usage: ~{estimated_memory_gb:.2f} GB")
    print(f"Batch size: {config['batch_size']}, Effective batch size: {config['batch_size']} (no gradient accumulation)")
    print(f"Samples per model: {config['samples_per_model']}, Total samples: {total_samples}")
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory available: {gpu_memory:.2f} GB")
        if estimated_memory_gb > gpu_memory * 0.8:
            print("⚠️  WARNING: Estimated memory usage is high relative to GPU memory!")
            print("   Consider reducing samples_per_model or batch_size further.")
        else:
            print("✅ Memory usage looks good for training!")

    print("Training U-Net Flow Matching Model on multiple models...")
    model_unet, trainer_unet = train_flow_matching_model(
        cdm_mass_maps, gas_maps, vcdm_maps, cosmo_params,
        noise_std=config['noise_std'],
        architecture=config['architecture'],
        max_epochs=config['max_epochs'],
        batch_size=config['batch_size'],
        patience=config['patience']
    )
    
    print("Training complete!")