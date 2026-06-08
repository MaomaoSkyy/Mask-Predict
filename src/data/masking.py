"""Mask 策略：
- 训练（时间维）：BERT 风格的随机 mask。点 mask + span mask 混合。
- 训练（变量维）：随机选 k 个变量，把它们在整个窗口内全部 mask 掉。模型只能从
  其他 D-k 个变量推断这些被 mask 变量 —— 强迫学习变量间的物理关联（如 P=UI）。
- 训练时按 var_mask_prob 在两种模式间随机切换。
- 推理（时间维）：棋盘 mask，多次前向覆盖全窗。
- 推理（变量维）：逐变量轮换 mask，D 次前向得到每个变量的"被其他变量解释"误差。
"""
from typing import Tuple

import torch


def random_mask(
    batch_shape: Tuple[int, int, int],
    ratio: float,
    span_prob: float,
    span_min: int,
    span_max: int,
    device: torch.device,
) -> torch.Tensor:
    """返回 bool mask, shape = (B, T, D), True 表示被 mask 的位置。"""
    B, T, D = batch_shape
    mask = torch.rand(B, T, D, device=device) < ratio  # 初始随机点 mask

    if span_prob > 0:
        # 决定哪些 batch×variable 切片要做 span 扩展
        do_span = torch.rand(B, D, device=device) < span_prob
        if do_span.any():
            # 对每个被选中的 (b, d) 序列做形态学膨胀：把每个 True 向右扩展 [0, span_max-1]
            # 简化实现：用 max-pool 模拟随机 span
            span_len = torch.randint(span_min, span_max + 1, (1,), device=device).item()
            # mask shape: (B, T, D) -> (B*D, 1, T) for 1d pooling
            m = mask.permute(0, 2, 1).reshape(B * D, 1, T).float()
            m_dilated = torch.nn.functional.max_pool1d(
                m, kernel_size=span_len, stride=1, padding=span_len // 2
            )[:, :, :T]
            m_dilated = m_dilated.reshape(B, D, T).permute(0, 2, 1).bool()
            # 仅在被选 (b, d) 上替换为膨胀后的 mask
            sel = do_span.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
            mask = torch.where(sel, m_dilated, mask)
    return mask


def variable_mask(
    batch_shape: Tuple[int, int, int],
    k_min: int,
    k_max: int,
    device: torch.device,
    time_span: int = 0,
) -> torch.Tensor:
    """变量维 mask：每个样本随机选 k ∈ [k_min, k_max] 个变量。
    - time_span=0：把这些变量的整列（所有 T）都 mask（最激进，硬要求从其他变量重构）
    - time_span>0：每个被选变量只 mask 长度 ≈ time_span 的随机一段（保留其他时刻的自身上下文，
      只在局部强迫利用变量间关联；更稳定，DDMT 风格）
    返回 (B, T, D) bool。
    """
    B, T, D = batch_shape
    k = torch.randint(k_min, k_max + 1, (B,), device=device)
    scores = torch.rand(B, D, device=device)
    sorted_scores, _ = scores.sort(dim=1, descending=True)
    idx = (k - 1).clamp(min=0).unsqueeze(1)
    thr = sorted_scores.gather(1, idx)
    var_sel = scores >= thr  # (B, D)

    if time_span <= 0 or time_span >= T:
        # 整列 mask
        return var_sel.unsqueeze(1).expand(B, T, D).contiguous()

    # 只 mask 长度为 time_span 的随机一段
    starts = torch.randint(0, T - time_span + 1, (B, D), device=device)  # (B, D)
    arange = torch.arange(T, device=device).view(1, T, 1)  # (1, T, 1)
    starts = starts.unsqueeze(1)  # (B, 1, D)
    in_span = (arange >= starts) & (arange < starts + time_span)  # (B, T, D)
    mask = in_span & var_sel.unsqueeze(1)
    return mask


def mixed_mask(
    batch_shape: Tuple[int, int, int],
    ratio: float,
    span_prob: float,
    span_min: int,
    span_max: int,
    var_mask_prob: float,
    var_k_min: int,
    var_k_max: int,
    device: torch.device,
    var_time_span: int = 0,
) -> torch.Tensor:
    """每个样本独立按 var_mask_prob 选择 mask 模式（per-sample 而非 per-batch）。
    var_time_span: 见 variable_mask；0=整列 mask，>0=只 mask 一段。
    """
    B, T, D = batch_shape
    time_m = random_mask(batch_shape, ratio, span_prob, span_min, span_max, device)
    if var_mask_prob <= 0:
        return time_m
    var_m = variable_mask(batch_shape, var_k_min, var_k_max, device, time_span=var_time_span)
    use_var = (torch.rand(B, device=device) < var_mask_prob).view(B, 1, 1)
    return torch.where(use_var, var_m, time_m)


def variable_rotation_masks(D: int, device: torch.device) -> torch.Tensor:
    """推理用：(D, D) bool；第 d 行只 mask 变量 d。
    推理时对每个 batch 做 D 次前向，每次只 mask 一个变量，
    收集该变量在所有 t 上的重构误差。
    """
    return torch.eye(D, dtype=torch.bool, device=device)


def checkerboard_masks(T: int, n_strides: int, device: torch.device) -> torch.Tensor:
    """生成推理用的棋盘 mask 集合。
    返回 (n_strides, T) 的 bool 张量，每行 mask 一个 phase；所有 phase 的并集是全 1。
    在变量维上广播 → 推理时一次前向 mask 一整个时间步的所有变量。
    """
    masks = torch.zeros(n_strides, T, dtype=torch.bool, device=device)
    for k in range(n_strides):
        masks[k, k::n_strides] = True
    return masks
