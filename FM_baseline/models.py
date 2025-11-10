import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMLayer(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.linear = nn.Linear(condition_dim, feature_dim * 2)
    
    def forward(self, features, condition_embed):
        scale_shift = self.linear(condition_embed)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.reshape(-1, features.size(1), 1, 1, 1)
        shift = shift.reshape(-1, features.size(1), 1, 1, 1)
        return features * (1 + scale) + shift

def sinusoidal_time_embedding(timesteps, dim, max_period=10000):
    device = timesteps.device
    half = dim // 2
    if half == 0:
        raise ValueError('sinusoidal_time_embedding requires dim >= 2')
    freq_positions = torch.arange(half, dtype=torch.float32, device=device)
    denom = max(half - 1, 1)
    freqs = torch.exp(-math.log(max_period) * freq_positions / denom)
    angles = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim=128, down=True):
        super().__init__()
        self.down = down

        self.norm1 = nn.GroupNorm(1, in_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act = nn.SiLU()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.film1 = FiLMLayer(condition_dim, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.film2 = FiLMLayer(condition_dim, out_channels)

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1)

        if self.down:
            self.pool = nn.MaxPool3d(2)

    def forward(self, x, condition_embed):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.film1(h, condition_embed)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.film2(h, condition_embed)

        h = h + residual
        h = self.act(h)

        if self.down:
            return h, self.pool(h)
        else:
            return h

class UNetScalarField(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, out_channels=1):
        super().__init__()

        self.upconv3 = nn.ConvTranspose3d(base_channels*2, base_channels*2, 2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.upconv1 = nn.ConvTranspose3d(base_channels, base_channels//2, 2, stride=2)
        self.output = nn.Conv3d(base_channels//2, out_channels, 1)

        self.time_embed_dim = 64
        time_hidden_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, 32)
        )

        self.param_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        
        self.dm_encoder = nn.Sequential(
            nn.Conv3d(1, base_channels//4, 3, padding=1), 
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels//4, 32)
        )
        
        self.vdm_encoder = nn.Sequential(
            nn.Conv3d(1, base_channels//4, 3, padding=1), 
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels//4, 32)
        )

        condition_dim = 32 + 32  # time + parameter
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim)
        )

        self.enc1 = UNetBlock(in_channels, base_channels, condition_dim, down=True)
        self.enc2 = UNetBlock(base_channels, base_channels, condition_dim, down=True)
        self.enc3 = UNetBlock(base_channels, base_channels*2, condition_dim, down=True)
        
        self.bottleneck = nn.Sequential(
            nn.GroupNorm(1, base_channels*2),
            nn.SiLU(),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
            nn.GroupNorm(1, base_channels*2),
            nn.SiLU(),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1)
        )
        self.bottleneck_film = FiLMLayer(condition_dim, base_channels*2)
        
        self.dec3 = UNetBlock(base_channels*2 + base_channels*2, base_channels*2, condition_dim, down=False) 
        self.dec2 = UNetBlock(base_channels + base_channels, base_channels, condition_dim, down=False)     
        self.dec1 = UNetBlock(base_channels + base_channels//2, base_channels//2, condition_dim, down=False) 

    def forward(self, x, t, params):
        
        time_embed = sinusoidal_time_embedding(t, self.time_embed_dim)
        time_embed = self.time_mlp(time_embed)
        param_embed = self.param_encoder(params)
        combined_embed = torch.cat([time_embed, param_embed], dim=1)
        combined_embed = self.condition_mlp(combined_embed)

        # Encoder
        skip1, x = self.enc1(x, combined_embed)
        skip2, x = self.enc2(x, combined_embed)
        skip3, x = self.enc3(x, combined_embed)
        
        # Bottleneck with FiLM
        x = self.bottleneck(x)
        x = self.bottleneck_film(x, combined_embed)
        
        # Decoder
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.dec3(x, combined_embed)
        
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x, combined_embed)
        
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, combined_embed)
        
        return self.output(x)
