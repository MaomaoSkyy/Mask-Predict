"""推理时的 mask 方案：

- time_checkerboard: 经典棋盘 mask。把 T 分 n_strides 份，每次 mask 一个 phase 的所有变量，
  n_strides 次前向覆盖整窗。强调"该时刻整体偏离"。
- variable_rotation: 逐变量轮换。每次只 mask 一个变量整列，让模型从其他 D-1 个变量
  重构它。强调"该变量与其他变量的物理/统计关联是否被破坏"——对 P=UI 这种强耦合
  组特别有效。
- both: 上面两种各跑一遍，分数按 score_combine 融合。
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.masking import checkerboard_masks, variable_rotation_masks


@torch.no_grad()
def _score_time_checkerboard(model, batch: torch.Tensor, n_strides: int, score_reduce: str) -> torch.Tensor:
    B, T, D = batch.shape
    device = batch.device
    time_masks = checkerboard_masks(T, n_strides, device)  # (n_strides, T)

    if score_reduce == "max":
        agg = torch.full((B, T, D), float("-inf"), device=device)
    else:
        agg = torch.zeros(B, T, D, device=device)
        cnt = torch.zeros(B, T, D, device=device)

    for k in range(n_strides):
        m = time_masks[k].view(1, T, 1).expand(B, T, D)
        pred = model(batch, m)
        err = (pred - batch) ** 2
        if score_reduce == "max":
            agg = torch.where(m, torch.maximum(agg, err), agg)
        else:
            agg = agg + torch.where(m, err, torch.zeros_like(err))
            cnt = cnt + m.float()
    if score_reduce == "mean":
        agg = agg / cnt.clamp(min=1.0)
    return agg


@torch.no_grad()
def _score_variable_rotation(model, batch: torch.Tensor) -> torch.Tensor:
    """每个变量被 mask 整列后的逐点误差。返回 (B, T, D)。
    每个 (t, d) 位置的分数 = 当只 mask 变量 d 时，模型在 t 时刻对 d 的预测误差。
    """
    B, T, D = batch.shape
    device = batch.device
    var_masks = variable_rotation_masks(D, device)  # (D, D)
    out = torch.zeros(B, T, D, device=device)
    for d in range(D):
        m = var_masks[d].view(1, 1, D).expand(B, T, D)  # 只 mask 变量 d
        pred = model(batch, m)
        err = (pred[..., d] - batch[..., d]) ** 2  # (B, T)
        out[..., d] = err
    return out


@torch.no_grad()
def score_series(model, dataset, cfg, device):
    """返回值：
    - mode='time_checkerboard' 或 'variable_rotation'：ndarray (T_total, D)
    - mode='both'：dict {'time': (T,D), 'var': (T,D)}，把融合留给 sweep 脚本离线做
      （避免在 batch 内做 z-score 这种会污染段级异常分数的操作）
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    mode = getattr(cfg.inference, "mode", "time_checkerboard")
    n_strides = cfg.inference.mask_strides
    score_reduce = cfg.inference.score_reduce

    var_prob = float(getattr(cfg.mask, "var_mask_prob", 0.0))
    if mode in ("variable_rotation", "both") and var_prob <= 0.0:
        raise ValueError(
            f"inference.mode='{mode}' 需要 mask.var_mask_prob > 0 "
            f"（当前 {var_prob}），否则模型从未学过变量 mask。"
            f"改 inference.mode='time_checkerboard'，或重训时设 var_mask_prob>0。"
        )
    if mode == "time_checkerboard" and var_prob >= 1.0:
        raise ValueError(
            "inference.mode='time_checkerboard' 但 mask.var_mask_prob=1.0，"
            "模型从未学过时间 mask。请改 inference.mode='variable_rotation'。"
        )

    t_chunks, v_chunks = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if mode in ("time_checkerboard", "both"):
            t_chunks.append(_score_time_checkerboard(
                model, batch, n_strides, score_reduce).cpu().numpy())
        if mode in ("variable_rotation", "both"):
            v_chunks.append(_score_variable_rotation(model, batch).cpu().numpy())

    def _stack(chunks):
        a = np.concatenate(chunks, axis=0)
        return a.reshape(-1, a.shape[-1])

    if mode == "time_checkerboard":
        return _stack(t_chunks)
    if mode == "variable_rotation":
        return _stack(v_chunks)
    return {"time": _stack(t_chunks), "var": _stack(v_chunks)}
