import math

import torch
import torch.nn as nn


class PerVariableEmbedding(nn.Module):
    """每个变量独立 1x1 升维 (标量 → d_model)，避免不同量纲互相干扰。
    输入 (B, T, D, 1) 或 (B, T, D) → 输出 (B, T, D, d_model)。
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.proj = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.proj, a=math.sqrt(5))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T, D) bool, True = masked
        # 输出 (B, T, D, d_model)
        emb = x.unsqueeze(-1) * self.proj + self.bias  # (B, T, D, d_model)
        # 用 mask_token 替换被 mask 位置
        m = mask.unsqueeze(-1)
        emb = torch.where(m, self.mask_token.expand_as(emb), emb)
        return emb


class LocalConv(nn.Module):
    """轻量级 depthwise 1D 卷积，在 Transformer 之前注入局部时序上下文。

    输入  (B, T, D, d_model)
    输出  (B, T, D, d_model)（同形状，残差）

    设计：
    - Depthwise conv（每个 embedding 通道独立的 K-长卷积核），只在 T 维卷积
    - 卷积参数 **跨变量共享**：局部时序模式（趋势、二阶差分）跟"是哪个变量"无关，
      共享参数 = 强归纳偏置 + 极少参数
    - 参数量 = d_model × kernel_size（d_model=128, K=5 → 640 params）
    - 残差 + LayerNorm 保证训练稳定，且关闭时退化为单位映射
    """

    def __init__(self, d_model: int, n_features: int = None, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数（same padding）"
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,  # 真·depthwise: 每个 channel 一个独立小 kernel
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D, d_model)
        B, T, D, E = x.shape
        # 把 (B, D) 折进 batch，每个变量当独立样本走 (E, T) 的 1D 卷积
        h = x.permute(0, 2, 3, 1).reshape(B * D, E, T)
        # cuDNN 已在 scripts/train.py 里全局禁用以规避 depthwise+fp16 兼容问题；
        # 仍 cast 到 fp32 算 conv，避免极少数 fp16 下 AMP 数值不稳定
        h = self.conv(h.float()).to(x.dtype)
        h = h.reshape(B, D, E, T).permute(0, 3, 1, 2)  # (B, T, D, E)
        return self.norm(x + h)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., T, d_model)
        T = x.size(-2)
        return x + self.pe[:T]
