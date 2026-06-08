import torch
import torch.nn as nn

from .embedding import (
    LocalConv,
    PerVariableEmbedding,
    SinusoidalPositionalEncoding,
)
from .transformer import DualAxisEncoder, StagedEncoder


class MaskPredictModel(nn.Module):
    """Mask-then-Predict 主模型。

    forward 输入:
        x:    (B, T, D) 归一化后的窗口
        mask: (B, T, D) bool, True 表示被 mask 的位置
    返回:
        x_hat: (B, T, D) 对每个位置的重构预测值
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, attn_mode: str,
                 local_conv_kernel: int = 0, encoder: str = "dualaxis",
                 temporal_layers: int = 5, temporal_kernel: int = 3, var_layers: int = 2):
        super().__init__()
        self.embed = PerVariableEmbedding(n_features, d_model)
        self.encoder_kind = encoder
        if encoder == "staged":
            # 分阶段：Stage1 因果 TCN 自带时序卷积 → 不再单独 local_conv
            self.local_conv = None
            self.encoder = StagedEncoder(d_model, n_heads, d_ff, dropout,
                                         temporal_layers, temporal_kernel, var_layers)
        else:
            # local_conv_kernel=0 关闭；>=3 启用 per-variable 1D conv 局部上下文
            if local_conv_kernel and local_conv_kernel >= 3:
                self.local_conv = LocalConv(d_model, n_features, kernel_size=local_conv_kernel)
            else:
                self.local_conv = None
            self.encoder = DualAxisEncoder(d_model, n_heads, d_ff, n_layers, dropout, attn_mode)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T, D) bool
        h = self.embed(x, mask)            # (B, T, D, d_model)，mask 位置已被 mask_token 替换
        if self.local_conv is not None:
            h = self.local_conv(h)         # 注入局部时序上下文（残差）
        if self.encoder_kind != "staged":
            # 注意：现有 pos 实际沿 size(-2)=D（变量轴）加，非时间轴（疑似 bug，保留不动）。
            # staged 的 Stage1 因果卷积本身 position-aware，跳过绝对位置编码。
            h = self.pos(h)                # 加位置编码
        h = self.encoder(h)                # (B, T, D, d_model)
        out = self.head(h).squeeze(-1)     # (B, T, D)
        return out


def build_model(cfg) -> MaskPredictModel:
    return MaskPredictModel(
        n_features=cfg.data.n_features,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        attn_mode=cfg.model.attn_mode,
        local_conv_kernel=int(getattr(cfg.model, "local_conv_kernel", 0)),
        encoder=str(getattr(cfg.model, "encoder", "dualaxis")),
        temporal_layers=int(getattr(cfg.model, "temporal_layers", 5)),
        temporal_kernel=int(getattr(cfg.model, "temporal_kernel", 3)),
        var_layers=int(getattr(cfg.model, "var_layers", 2)),
    )
