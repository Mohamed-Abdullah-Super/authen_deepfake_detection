import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange, repeat
import math

class FrequencyAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_freq = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Learnable frequency filters
        self.freq_filters = nn.Parameter(torch.randn(channels, 8, 8))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DCT transform
        x_freq = torch.fft.rfft2(x, norm='ortho')
        magnitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        
        # Apply learnable frequency filters
        filtered_mag = magnitude * F.interpolate(
            self.freq_filters.unsqueeze(0),
            size=magnitude.shape[-2:],
            mode='bilinear'
        )
        
        # Reconstruct and combine
        freq_features = torch.cat([filtered_mag, phase], dim=1)
        attention_weights = self.conv_freq(freq_features)
        
        return x * attention_weights

class TemporalConsistencyModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ReLU(),
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(dim),
            nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        # Extract motion features
        motion = self.motion_encoder(x)
        
        # Apply temporal attention
        features = rearrange(motion, 'b c t h w -> t (b h w) c')
        attended_features, _ = self.attention(features, features, features)
        attended_features = rearrange(attended_features, 't (b h w) c -> b c t h w', h=H, w=W)
        
        return x + attended_features

class SpatialAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(dim // 8, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return residual + x
