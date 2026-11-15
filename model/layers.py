import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedLayerNorm2d(nn.Module):
    def __init__(self, num_channels, groups=1, eps=1e-6):
        super().__init__()
        assert num_channels % groups == 0, "Channels must be divisible by groups"
        self.groups = groups
        self.eps = eps
        
        # 가중치와 편향은 여전히 채널별로 가짐
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        G = self.groups
        
        # 1. 그룹별로 분리: (B, G, C//G, H, W)
        x = x.view(B, G, C // G, H, W)
        
        # 2. 각 그룹 내에서(dim=2) 평균과 분산 계산
        mean = x.mean(2, keepdim=True)
        var = x.var(2, keepdim=True, unbiased=False)
        
        # 3. 정규화
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # 4. 원래 형상으로 복원: (B, C, H, W)
        x = x.view(B, C, H, W)
        
        # 5. 채널별 가중치/편향 적용
        x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=96, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = GroupedLayerNorm2d(embed_dim, groups=1)

    def forward(self, x):
        # 1. Conv 수행 (B, C, H, W)
        x = self.proj(x)
        x = self.ln(x)

        return x
    
class PatchReconstruct(nn.Module):
    def __init__(self, dim=96, patch_size=2):
        super().__init__()
        # 업샘플링 전에 LayerNorm으로 한 번 다듬어줌 (SwinIR 등의 방식)
        self.ln = GroupedLayerNorm2d(dim, groups=1)
        self.up = nn.Conv2d(dim, (patch_size**2) * 3, kernel_size=1)
        self.ps = nn.PixelShuffle(patch_size)

    def forward(self, x):
        x = self.ln(x)
        x = self.up(x)
        x = self.ps(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim=None, groups=1):
        super().__init__()
        out_dim = out_dim or 2 * dim
        self.unshuffle = nn.PixelUnshuffle(2) # H, W -> H/2, W/2 및 C -> 4C
        self.ln = GroupedLayerNorm2d(4 * dim, groups=groups)
        self.reduction = nn.Conv2d(4 * dim, out_dim, kernel_size=1, bias=False, groups=groups)

    def forward(self, x):
        # (B,C,H,W) 형태
        x = self.unshuffle(x)
        x = self.ln(x)
        x = self.reduction(x)
        return x

class PatchReverse(nn.Module):
    def __init__(self, dim, out_dim=None, groups=1):
        super().__init__()
        out_dim = out_dim or dim // 2
        self.ln = GroupedLayerNorm2d(dim, groups=groups)
        self.expand = nn.Conv2d(dim, 4 * out_dim, kernel_size=1, bias=False, groups=groups)
        self.ps = nn.PixelShuffle(2) # H, W -> 2H, 2W 및 4C -> C

    def forward(self, x):
        # (B,C,H,W) 형태
        x = self.ln(x)
        x = self.expand(x)
        x = self.ps(x)
        return x
    
class ResNetBlock(nn.Module):
    def __init__(self, dim:int, groups:int=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim//2, kernel_size=1, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim//2,dim//2, kernel_size=3,padding=1, groups=groups)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim//2, dim, kernel_size=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) 형태
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out
    
def channel_shuffle_nhwc(x, groups: int):
    # x: [B, H, W, C]
    B, H, W, C = x.shape
    assert C % groups == 0

    x = x.view(B, H, W, groups, C // groups)
    x = x.transpose(3, 4).contiguous()
    x = x.view(B, H, W, C)
    return x
    
class GroupedConvNeXtBlock(nn.Module):
    def __init__(self, dim, groups=1, mlp_ratio=2, shuffle=True, normgroup=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = GroupedLayerNorm2d(dim, groups=groups if normgroup else 1)

        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, groups=groups)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, groups=groups)

        self.groups = groups
        self.shuffle=shuffle

    def forward(self, x):
        # x: [B, C, H, W]
        shortcut = x

        x = self.dwconv(x)

        # ConvNeXt-style LN over channel
        x = self.norm(x)

        x = self.pw1(x)
        x = self.act(x)

        # channel shuffle on BCHW
        if self.shuffle and self.groups > 1:
            B, C, H, W = x.shape
            G = self.groups
            x = x.view(B, G, C // G, H, W)
            x = x.transpose(1, 2).contiguous()
            x = x.view(B, C, H, W)

        x = self.pw2(x)

        return shortcut + x