import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """标准 Transformer encoder block (Pre-LN)，用于**时间维** attention。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class VarAttention(nn.Module):
    """**变量维**多头自注意力（把 T 折进 batch，对 D 个变量做 self-attention）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D, E)  N=B*T, D=变量数, E=d_model
        N, D, E = x.shape
        qkv = self.qkv(x).reshape(N, D, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # 各 (N, h, D, dk)
        attn = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)  # (N, h, D, D)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        o = attn @ v                               # (N, h, D, dk)
        o = o.transpose(1, 2).reshape(N, D, E)
        return self.out(o)


class VarEncoderBlock(nn.Module):
    """变量维 encoder block (Pre-LN)。"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = VarAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.attn(h)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class DualAxisEncoder(nn.Module):
    """时间维 attention 与变量维 attention 交替堆叠。
    输入 (B, T, D, d_model) → 输出同形状。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 dropout: float, attn_mode: str = "alternate"):
        super().__init__()
        assert attn_mode in {"alternate", "time_only", "feature_only"}
        self.attn_mode = attn_mode
        self.axes = []          # 每层是 'time' 还是 'var'，由 attn_mode 决定（确定性，便于 ckpt 对齐）
        blocks = []
        for i in range(n_layers):
            if attn_mode == "time_only":
                is_time = True
            elif attn_mode == "feature_only":
                is_time = False
            else:  # alternate
                is_time = (i % 2 == 0)
            self.axes.append("time" if is_time else "var")
            blocks.append(
                EncoderBlock(d_model, n_heads, d_ff, dropout) if is_time
                else VarEncoderBlock(d_model, n_heads, d_ff, dropout)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D, d_model)
        B, T, D, E = x.shape
        for axis, blk in zip(self.axes, self.blocks):
            if axis == "time":
                # 时间维 attention：把 D 折进 batch
                h = x.permute(0, 2, 1, 3).reshape(B * D, T, E)
                h = blk(h)
                x = h.reshape(B, D, T, E).permute(0, 2, 1, 3)
            else:
                # 变量维 attention：把 T 折进 batch
                h = x.reshape(B * T, D, E)
                h = blk(h)
                x = h.reshape(B, T, D, E)
        return x


class CausalTCNBlock(nn.Module):
    """因果 dilated depthwise-separable 卷积块（per-variable 共享，沿 T 因果）。

    输入/输出 (N, E, T)，N=B*D。左侧 padding = (k-1)*dilation 保证因果：
    位置 t 的输出只依赖 <=t 的输入（不偷看未来）。depthwise（时序混合）+ pointwise
    （通道混合）+ 残差 + LayerNorm。多个块按 dilation=1,2,4,... 叠 → 长程因果感受野。
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, E, T)。cuDNN 已在 train.py 全局禁用以规避 depthwise+fp16 兼容问题，
        # 这里仍 cast 到 fp32 算卷积，避免 AMP 下数值不稳定（与 LocalConv 一致）。
        dt = x.dtype
        h = F.pad(x.float(), (self.pad, 0))   # 仅左 pad → 因果
        h = self.dw(h)
        h = self.pw(h)
        h = self.act(h).to(dt)
        h = self.drop(h)
        y = (x + h).transpose(1, 2)           # (N, T, E)
        y = self.norm(y).transpose(1, 2)      # (N, E, T)
        return y


class StagedEncoder(nn.Module):
    """分阶段编码器：先 per-variable **因果时序**（dilated TCN），再 per-timestep **跨变量** attention。

    输入/输出 (B, T, D, d_model)。
    - Stage 1：每个变量单独过共享的因果 TCN（折 D 入 batch），得到 h[t,d]——变量 d 截至 t 的
      时序摘要，不偷看未来 → 对持续型异常不会被"自己的未来"带平。
    - Stage 2：每个时刻把 D 个变量摊开做双向 attention（折 B,T 入 batch），用其他变量的 h[t,≠d]
      重构被 mask 的 d。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 temporal_layers: int, temporal_kernel: int, var_layers: int):
        super().__init__()
        self.temporal = nn.ModuleList([
            CausalTCNBlock(d_model, temporal_kernel, dilation=2 ** i, dropout=dropout)
            for i in range(temporal_layers)
        ])
        self.var_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(var_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D, E = x.shape
        # Stage 1：时间（per-variable，折 D 入 batch，沿 T 因果）
        h = x.permute(0, 2, 3, 1).reshape(B * D, E, T)   # (B*D, E, T)
        for blk in self.temporal:
            h = blk(h)
        h = h.reshape(B, D, E, T).permute(0, 3, 1, 2)     # (B, T, D, E)
        # Stage 2：跨变量（per-timestep，折 B,T 入 batch，双向）
        h = h.reshape(B * T, D, E)
        for blk in self.var_blocks:
            h = blk(h)
        return h.reshape(B, T, D, E)
