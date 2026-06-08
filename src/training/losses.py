import torch


def masked_l2_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """仅在 mask=True 的位置计算 L2，再做平均。
    pred, target: (B, T, D); mask: (B, T, D) bool
    """
    diff = (pred - target) ** 2
    diff = diff * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return diff.sum() / denom
