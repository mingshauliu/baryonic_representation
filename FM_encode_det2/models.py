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

def _make_gn(num_channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    g = min(preferred_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return nn.GroupNorm(g, num_channels)

class ResNetBlock(nn.Module):
    """
    Pre-activation ResNet basic block with GroupNorm + SiLU.
    Layout: x -> GN -> SiLU -> Conv (stride) -> GN -> SiLU -> Conv -> add skip
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Pre-activation norm/act
        self.gn1 = _make_gn(in_channels, groups)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False, padding_mode='circular')

        self.gn2 = _make_gn(out_channels, groups)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')

        # Shortcut for downsample or channel change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False, padding_mode='circular')
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Pre-activation
        h = self.act1(self.gn1(x))
        out = self.conv1(h)

        out = self.act2(self.gn2(out))
        out = self.conv2(out)

        skip = self.shortcut(x)
        out = out + skip
        return out

class ResNetBranch(nn.Module):
    """
    Same interface as your original ResNetBranch, but:
    - initial stem uses GN+SiLU
    - blocks use GN+SiLU pre-activation
    """
    def __init__(self, in_channels=1, embedding_dim=16, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.groups = groups

        # Stem
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='circular')
        self.gn1 = _make_gn(64, groups)
        self.act1 = nn.SiLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(64,   64,  2, stride=1)
        self.layer2 = self._make_layer(64,   128, 2, stride=2)
        self.layer3 = self._make_layer(128,  256, 2, stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResNetBlock(in_ch, out_ch, stride=stride, groups=self.groups)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_ch, out_ch, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(self.gn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        return x

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim=128, down=True, groups=8):
        super().__init__()
        self.down = down

        self.norm1 = _make_gn(in_channels, groups)
        self.norm2 = _make_gn(out_channels, groups)
        
        self.act = nn.SiLU()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, padding_mode='circular')
        self.film1 = FiLMLayer(condition_dim, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, padding_mode='circular')
        self.film2 = FiLMLayer(condition_dim, out_channels)

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, 1, padding_mode='circular')

        if self.down:
            self.downsample = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1, padding_mode='circular')

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
        # h = self.act(h)

        if self.down:
            return h, self.downsample(h)
        else:
            return h

def upsample_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode='circular')
    )

class UNetScalarField(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, out_channels=1, groups=8):
        super().__init__()

        self.upconv3 = upsample_block(base_channels*2, base_channels*2)
        self.upconv2 = upsample_block(base_channels*2, base_channels)
        self.upconv1 = upsample_block(base_channels, base_channels//2)
        self.output = nn.Conv3d(base_channels//2, out_channels, 1, padding_mode='circular')

        # self.time_embed_dim = 64
        # time_hidden_dim = 128
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(self.time_embed_dim, time_hidden_dim),
        #     nn.SiLU(),
        #     nn.Linear(time_hidden_dim, 32)
        # )

        self.time_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        
        self.param_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        ) # Cosmological parameters only: Omega_m, sigma_8

        condition_dim = 32 + 32 + 8 # time + parameter
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Linear(condition_dim * 2, condition_dim)
        )

        self.enc1 = UNetBlock(in_channels, base_channels, condition_dim, down=True)
        self.enc2 = UNetBlock(base_channels, base_channels, condition_dim, down=True)
        self.enc3 = UNetBlock(base_channels, base_channels*2, condition_dim, down=True)
        
        self.bottleneck = nn.Sequential(
            _make_gn(base_channels*2, groups),
            nn.SiLU(),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1, padding_mode='circular'),
            _make_gn(base_channels*2, groups),
            nn.SiLU(),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1, padding_mode='circular')
        )
        self.bottleneck_film = FiLMLayer(condition_dim, base_channels*2)
        
        self.dec3 = UNetBlock(base_channels*2 + base_channels*2, base_channels*2, condition_dim, down=False) 
        self.dec2 = UNetBlock(base_channels + base_channels, base_channels, condition_dim, down=False)     
        self.dec1 = UNetBlock(base_channels + base_channels//2, base_channels//2, condition_dim, down=False) 

    def forward(self, x, t, params, zbar):
        
        time_embed = self.time_encoder(t.unsqueeze(-1)) 
        param_embed = self.param_encoder(params)
        combined_embed = torch.cat([time_embed, param_embed, zbar], dim=1)
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
