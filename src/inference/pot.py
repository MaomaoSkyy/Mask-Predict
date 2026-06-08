"""Peaks-Over-Threshold 阈值估计 + 鲁棒化。

跨 entity 时 GPD 拟合常常失败（heavy tail / 重复值过多 / 拟合后 xi 大），
导致阈值变成负数、NaN 或巨大值。这里加三道防线：
1. 计算前 sanity-check（足够样本、有方差、非全零）
2. 计算后 sanity-check（阈值落在合理范围 [min, max] 内）
3. 任何一步失败都 fallback 到 (1 - level) 分位数
"""
import numpy as np
from scipy import stats


def _quantile_fallback(scores: np.ndarray, level: float) -> float:
    return float(np.quantile(scores, 1.0 - level))


def pot_threshold(scores: np.ndarray, q: float = 0.99, level: float = 0.02,
                  verbose: bool = False) -> float:
    """
    scores: 1D 异常分数（建议用验证集分数估计阈值）
    q:      初始尾部分位数，作为 GPD 拟合的截断
    level:  目标风险水平（越小阈值越保守，召回越低）
    返回最终阈值。
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    scores = scores[np.isfinite(scores)]

    # 防线 0: 数值病态离群点清理。
    # 数据已 min-max 归一化到 [0,1]，平方重构误差自然在 [0, ~10] 量级；
    # 若出现 >= 1e6 的值，几乎可以肯定是 AMP/fp16 训练时偶发的数值溢出尖刺，
    # 不是真实信号 —— 直接丢弃，否则会污染 GPD 拟合与分位数计算。
    if scores.size:
        sane = scores < 1e6
        n_drop = int((~sane).sum())
        if n_drop and sane.sum() >= 50:
            if verbose:
                print(f"[POT] dropped {n_drop} pathological scores "
                      f"(>= 1e6, max={scores.max():.3g})")
            scores = scores[sane]
        elif n_drop:
            # 病态值太多 → 数据无救
            if verbose:
                print(f"[POT] fallback: {n_drop}/{scores.size} pathological scores")
            sane_scores = scores[sane]
            return _quantile_fallback(sane_scores if sane_scores.size else scores, level)

    # 防线 1: 输入 sanity-check
    if scores.size < 50:
        if verbose: print(f"[POT] fallback: too few samples ({scores.size})")
        return _quantile_fallback(scores, level) if scores.size else 0.0
    if scores.std() < 1e-10:
        if verbose: print("[POT] fallback: zero variance")
        return float(scores.max())

    init = float(np.quantile(scores, q))
    peaks = scores[scores > init] - init
    if peaks.size < 10:
        if verbose: print(f"[POT] fallback: too few peaks ({peaks.size})")
        return _quantile_fallback(scores, level)

    # GPD 拟合
    try:
        xi, _, beta = stats.genpareto.fit(peaks, floc=0)
    except Exception as e:
        if verbose: print(f"[POT] fallback: GPD fit raised {e!r}")
        return _quantile_fallback(scores, level)

    if not (np.isfinite(xi) and np.isfinite(beta)) or beta <= 0:
        if verbose: print(f"[POT] fallback: invalid GPD params xi={xi} beta={beta}")
        return _quantile_fallback(scores, level)

    n = scores.size
    Nt = peaks.size
    r = level * n / max(1, Nt)
    if r <= 0 or not np.isfinite(r):
        return _quantile_fallback(scores, level)

    try:
        if abs(xi) < 1e-8:
            thr = init + beta * (-np.log(r))
        else:
            base = r ** (-xi)
            if not np.isfinite(base) or base > 1e12:
                if verbose: print(f"[POT] fallback: r^(-xi) overflow, xi={xi}")
                return _quantile_fallback(scores, level)
            thr = init + (beta / xi) * (base - 1.0)
    except Exception as e:
        if verbose: print(f"[POT] fallback: formula error {e!r}")
        return _quantile_fallback(scores, level)

    # 防线 2: 输出合理性。
    # 病态离群点已在防线 0 清理，这里只保 finite + 不低于 score 最小值。
    # 不再做"分位数硬上界"——那会把 GPD 在轻尾分布上的合理外推（如 1-6 上的 1.58）压回去。
    if not np.isfinite(thr) or thr < float(scores.min()):
        if verbose:
            print(f"[POT] fallback: thr={thr:.4g} not finite or below s_min")
        return _quantile_fallback(scores, level)

    return float(thr)
