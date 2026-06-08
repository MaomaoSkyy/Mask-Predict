from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float32)


def fill_sentinel(x: np.ndarray, threshold: float = 1e6) -> np.ndarray:
    """对训练数据做 sentinel 值前向填充。

    SMD 等数据集中常用 1e8 作为缺失/故障的 sentinel；若混在 train/val 段会污染
    归一化与 val_loss。检测 ≥ threshold 的位置，用前一个有效值替换；首个有效值
    出现前用该列中位数兜底。

    test 数据不应调用此函数 —— 真实异常可能恰好以 sentinel 形式呈现，
    保留原值经过 scaler+clip 后可作为异常信号被 POT 抓到。
    """
    if threshold <= 0:
        return x  # 0 / 负数视为关闭
    x = x.copy()
    sentinel = x >= threshold
    if not sentinel.any():
        return x
    T, D = x.shape
    for j in range(D):
        col = x[:, j]
        s = sentinel[:, j]
        if not s.any():
            continue
        valid_vals = col[~s]
        if valid_vals.size == 0:
            x[:, j] = 0.0
            continue
        last_valid = float(np.median(valid_vals))
        for i in range(T):
            if s[i]:
                x[i, j] = last_valid
            else:
                last_valid = float(x[i, j])
    return x


class MinMaxScaler:
    """朴素 min-max。保留作 baseline，但默认改用 RobustScaler。"""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.lo = None
        self.hi = None

    def fit(self, x: np.ndarray):
        self.lo = x.min(axis=0)
        self.hi = x.max(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / (self.hi - self.lo + self.eps)


class RobustScaler:
    """基于分位数的归一化 + clip，专治 SMD 这种带 sentinel 值（1e8）的数据。

    流程：
      1. fit: 用 train 的 q_low/q_high 分位数（默认 1%/99%）当 lo/hi，不被 sentinel 主导
      2. transform: (x - lo) / (hi - lo) → 正常数据落在 ~[0, 1]
      3. clip: 截到 [-clip_pad, 1+clip_pad]（默认 [-4, 5]）→ sentinel 1e8 被压扁，
         但仍保留显著的"out-of-range"信号（model 预测 ~0.5，clip 后真值 5，平方误差 ≈ 20，
         足以被 POT 抓到）
    """

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99,
                 clip_pad: float = 4.0, min_range: float = 1e-3):
        self.q_low = q_low
        self.q_high = q_high
        self.clip_pad = clip_pad
        self.min_range = min_range  # 低于此值视为常数列，分母用 1，避免数值爆炸
        self.lo = None
        self.hi = None

    def fit(self, x: np.ndarray):
        self.lo = np.quantile(x, self.q_low, axis=0)
        self.hi = np.quantile(x, self.q_high, axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        range_ = self.hi - self.lo
        # 对常数/近常数列（range < min_range），用 1 作分母，输出 = x - lo，等价于"只居中不缩放"
        # 这避免了 SMD 里某些 sensor 几乎不变时，val/test 微小波动除以接近 0 的分母变 1e8
        range_safe = np.where(range_ < self.min_range, 1.0, range_)
        y = (x - self.lo) / range_safe
        return np.clip(y, -self.clip_pad, 1.0 + self.clip_pad)


def load_smd_entity(root: str, entity: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(root)
    train = _load_txt(root / "train" / f"{entity}.txt")
    test = _load_txt(root / "test" / f"{entity}.txt")
    label = np.loadtxt(root / "test_label" / f"{entity}.txt", dtype=np.int64)
    return train, test, label


class SlidingWindow(Dataset):
    """从一段连续时间序列上滑窗。
    返回 tensor shape: (T, D), float32。
    """

    def __init__(self, series: np.ndarray, window: int, stride: int = 1):
        assert series.ndim == 2, f"expect (T, D), got {series.shape}"
        self.series = torch.from_numpy(series.astype(np.float32))
        self.window = window
        self.stride = stride
        self.n = max(0, (series.shape[0] - window) // stride + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> torch.Tensor:
        s = i * self.stride
        return self.series[s:s + self.window]


def build_smd_datasets(cfg) -> Tuple[SlidingWindow, SlidingWindow, SlidingWindow, np.ndarray, MinMaxScaler]:
    """返回 (train_ds, val_ds, test_ds, test_label, scaler)。
    val 从训练集尾部切，避免泄露未来；scaler 仅在训练数据上拟合。
    """
    train_raw, test_raw, label = load_smd_entity(cfg.data.root, cfg.data.entity)

    n_train_all = train_raw.shape[0]
    # Sentinel 前向填充（仅训练数据；test 保留原值用于异常检测）
    sentinel_thr = float(getattr(cfg.data, "sentinel_threshold", 1e6))
    if sentinel_thr > 0:
        n_before = int((train_raw >= sentinel_thr).sum())
        train_raw = fill_sentinel(train_raw, threshold=sentinel_thr)
        n_after = int((train_raw >= sentinel_thr).sum())
        if n_before:
            print(f"[data] {cfg.data.entity}: forward-filled {n_before} sentinel "
                  f"values in train_raw (≥ {sentinel_thr:g})")

    n_val = int(n_train_all * cfg.data.val_ratio)
    train_part = train_raw[: n_train_all - n_val]
    val_part = train_raw[n_train_all - n_val:]

    clip_pad = float(getattr(cfg.data, "clip_pad", 4.0))
    scaler = RobustScaler(clip_pad=clip_pad).fit(train_part)
    train_norm = scaler.transform(train_part)
    val_norm = scaler.transform(val_part)
    test_norm = scaler.transform(test_raw)  # test 不做 sentinel 填充

    val_stride = getattr(cfg.data, "val_stride", cfg.data.window)
    train_ds = SlidingWindow(train_norm, cfg.data.window, cfg.data.train_stride)
    val_ds = SlidingWindow(val_norm, cfg.data.window, val_stride)
    test_ds = SlidingWindow(test_norm, cfg.data.window, cfg.data.test_stride)

    return train_ds, val_ds, test_ds, label, scaler
