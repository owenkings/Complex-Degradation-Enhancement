import torch
import torch.nn as nn
import warnings

# Try to import Mamba, provide fallback if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except (ImportError, OSError, AttributeError) as e:
    MAMBA_AVAILABLE = False
    # Don't warn here, check in __init__

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, backend="mamba"):
        super().__init__()
        self.backend = backend
        
        if backend == "mamba":
            if not MAMBA_AVAILABLE:
                raise ImportError(
                    "Mamba backend requested but mamba_ssm not installed or failed to load. "
                    "Please install it using: pip install mamba_ssm"
                )
            self.norm = nn.LayerNorm(dim)
            self.mamba = Mamba(d_model=dim, d_state=d_state)
            # FFN is often useful after Mamba mixer
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        else:
            raise ValueError(f"Backend '{backend}' is not allowed for Task 2. You must use 'mamba'.")

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
            # Should not be reached given strict check in __init__
            raise ValueError(f"Backend {self.backend} not supported")

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
        self.gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

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
            y_f = blk(x_flat)
            y_b = blk(torch.flip(x_flat, dims=[1]))
            y_b = torch.flip(y_b, dims=[1])
            x_flat = y_f + y_b
            
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x_flat = x_flat.permute(0, 2, 1).view(B, C, H, W)
        
        x_enh = self.out_proj(x_flat)
        gate = torch.sigmoid(self.gate(x))
        return residual + gate * x_enh
