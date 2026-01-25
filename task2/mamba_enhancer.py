import torch
import torch.nn as nn
import warnings

# Try to import Mamba, provide fallback if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except (ImportError, OSError, AttributeError) as e:
    MAMBA_AVAILABLE = False
    warnings.warn(f"Mamba library not available or broken ({e}). If you want to use Mamba backend, please install mamba-ssm.")

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, backend="mamba"):
        super().__init__()
        self.backend = backend
        
        if backend == "mamba":
            if not MAMBA_AVAILABLE:
                raise ImportError("Mamba backend requested but mamba_ssm not installed.")
            self.norm = nn.LayerNorm(dim)
            self.mamba = Mamba(d_model=dim, d_state=d_state)
            # FFN is often useful after Mamba mixer
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            # Fallback: Transformer Encoder Layer
            self.block = nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=4, 
                dim_feedforward=dim*4, 
                dropout=0.1, 
                activation='gelu',
                batch_first=True
            )

    def forward(self, x):
        """
        x: (B, L, C)
        """
        if self.backend == "mamba":
            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            x = x + residual
            
            # FFN part
            residual = x
            x = self.norm(x)
            x = self.ffn(x)
            x = x + residual
            return x
        else:
            return self.block(x)

class MambaFeatureEnhancer(nn.Module):
    def __init__(self, in_channels=128, d_state=16, n_layers=2, backend="mamba"):
        super().__init__()
        self.in_channels = in_channels
        self.backend = backend
        
        # 1x1 conv to embed features if needed, or just use as is
        # Here we keep dimensions same as input
        self.embed = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            MambaBlock(dim=in_channels, d_state=d_state, backend=backend)
            for _ in range(n_layers)
        ])
        
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: Enhanced features (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x
        
        x = self.embed(x) # (B, C, H, W)
        
        # Flatten for Sequence Modeling: (B, C, H*W) -> (B, H*W, C)
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        
        for blk in self.blocks:
            x_flat = blk(x_flat)
            
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_flat = x_flat.permute(0, 2, 1).view(B, C, H, W)
        
        x_enh = self.out_proj(x_flat)
        
        # Residual Connection
        return residual + x_enh
