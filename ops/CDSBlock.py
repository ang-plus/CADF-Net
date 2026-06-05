import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.ChannelAttention import CA_layer
from ops.layernorm import LayerNorm2d

# CDS注意力子模块
class CDSWindowMSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    def forward(self, x):
        return self.conv(x)

class CDSShiftedWindowMSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    def forward(self, x):
        return self.conv(x)

class CDSGridMSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    def forward(self, x):
        return self.conv(x)

class CDSMixAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ca = CA_layer(dim)
        self.grid_msa = CDSGridMSA(dim)
        self.w_msa = CDSWindowMSA(dim)
        self.sw_msa = CDSShiftedWindowMSA(dim)
        self.norm = LayerNorm2d(dim)
        # 动态通道分割mask
        self.channel_mask = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        # 动态加权融合
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(dim*4, 4, 1),
        )
    def forward(self, x):
        B, C, H, W = x.shape
        # 动态通道mask
        M = self.channel_mask(x)  # [B, C, H, W], 取通道均值做全局mask
        M_pool = M.mean(dim=[2,3], keepdim=True)  # [B, C, 1, 1]
        # 动态分割
        F_G = x * M_pool
        F_W = x * (1 - M_pool)
        # 四路注意力
        X_W1 = self.w_msa(F_W)
        X_W2 = self.sw_msa(F_W)
        X_G = self.grid_msa(F_G)
        X_C = self.ca(x)
        # 拼接后动态加权
        X_cat = torch.cat([X_W1, X_W2, X_G, X_C], dim=1)  # [B, 4C, H, W]
        # 预测权重
        w = self.weight_predictor(X_cat)  # [B, 4, H, W]
        w = w.mean(dim=[2,3])  # [B, 4]
        w = F.softmax(w, dim=1)  # [B, 4]
        # 加权融合
        Xs = [X_W1, X_W2, X_G, X_C]
        X_sum = sum(w[:,i].view(B,1,1,1) * Xs[i] for i in range(4))
        out = self.norm(X_sum + x)
        return out

class CDSBlock(nn.Module):  # Cross-domain attention aggregation Block
    def __init__(self, dim):
        super().__init__()
        self.mal1 = CDSMixAttentionLayer(dim)
        self.norm1 = LayerNorm2d(dim)
        self.mal2 = CDSMixAttentionLayer(dim)
        self.norm2 = LayerNorm2d(dim)
    def forward(self, x):
        F_M = self.norm1(self.mal1(x)) + x
        F_M = self.norm2(self.mal2(F_M)) + F_M
        return F_M 