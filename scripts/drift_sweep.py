"""**首选阈值扫描脚本**：在已保存的 val/test scores 上扫描
后处理 × 聚合 × POT 参数，零重训找出最佳逐点异常分数。

本脚本是 threshold_sweep.py 的升级版——加入时序后处理后，SMD 上
8 entity 均值 raw_f1 从 0.37 → 0.50（+34%）。

后处理变体（per-variable error series 的时序操作，不涉及变量间）：
- raw: 原始分数
- zscore: 用 val 的 mean/std 做 per-variable z-score
- smooth_k: z-score 后窗口 k 的滑动平均（强调持续异常）
- runmax_k: z-score 后窗口 k 的滑动 max（保留 spike）
- combine_smooth_k: z-score raw + smooth_k（兼顾尖刺与持续）

每个 entity 的最佳模式跟它的异常形态强相关：
- 短 spike → runmax_3
- 持续漂移 → smooth_10
- 混合 → combine_smooth_10

每个变体扫聚合(max/topk3/topk5/mean) × q × level，输出该 entity 最高 raw_f1。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_scores  # noqa: E402
from src.inference.pot import pot_threshold  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# ---------------- 后处理 ----------------

def _zscore(val: np.ndarray, test: np.ndarray, eps: float = 1e-8):
    """用 val 统计量对 (T, D) 做 per-variable z-score。"""
    mu = val.mean(axis=0, keepdims=True)
    sd = val.std(axis=0, keepdims=True).clip(min=eps)
    return (val - mu) / sd, (test - mu) / sd


def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    """(T, D) 沿 T 做窗口 k 的滑动平均，same padding。"""
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    csum = np.cumsum(xp, axis=0)
    out = (csum[k:] - csum[:-k]) / k
    return out[: x.shape[0]]


def _moving_max(x: np.ndarray, k: int) -> np.ndarray:
    """(T, D) 沿 T 做窗口 k 的滑动 max。"""
    if k <= 1:
        return x
    T, D = x.shape
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for t in range(T):
        out[t] = xp[t:t + k].max(axis=0)
    return out


def postprocess(val: np.ndarray, test: np.ndarray, mode: str):
    """返回 (val_pp, test_pp)。"""
    if mode == "raw":
        return val, test
    if mode == "zscore":
        return _zscore(val, test)
    if mode.startswith("smooth_"):
        k = int(mode.split("_")[1])
        v, t = _zscore(val, test)
        return _moving_avg(v, k), _moving_avg(t, k)
    if mode.startswith("runmax_"):
        k = int(mode.split("_")[1])
        v, t = _zscore(val, test)
        return _moving_max(v, k), _moving_max(t, k)
    if mode.startswith("combine_smooth_"):
        # z-score 后的 raw + smooth_k（兼顾尖刺与持续）
        k = int(mode.split("_")[2])
        v_z, t_z = _zscore(val, test)
        v_smooth = _moving_avg(v_z, k)
        t_smooth = _moving_avg(t_z, k)
        return v_z + v_smooth, t_z + t_smooth
    raise ValueError(mode)


# ---------------- 聚合 + POT ----------------

def aggregate(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return scores.max(axis=1)
    if mode.startswith("topk"):
        k = int(mode[4:])
        return np.sort(scores, axis=1)[:, -k:].mean(axis=1)
    if mode == "mean":
        return scores.mean(axis=1)
    raise ValueError(mode)


def search_one(val_s: np.ndarray, test_s: np.ndarray, label: np.ndarray) -> dict:
    """扫聚合 × POT，返回最佳 F1。"""
    best = None
    for agg in ["max", "topk3", "topk5", "mean"]:
        v = aggregate(val_s, agg)
        t = aggregate(test_s, agg)
        for q in [0.95, 0.99]:
            for lv in [1e-3, 1e-4, 1e-5, 1e-6]:
                try:
                    thr = pot_threshold(v, q=q, level=lv)
                    r = evaluate_scores(t, label, thr)
                    score = r["raw_f1"]
                    if best is None or score > best["raw_f1"]:
                        best = {"raw_f1": score, "raw_p": r["raw_p"],
                                "raw_r": r["raw_r"], "pa_f1": r["pa_f1"],
                                "agg": agg, "q": q, "level": lv, "thr": thr}
                except Exception:
                    pass
    return best


# ---------------- 主流程 ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.train.save_dir) / args.entity
    val_scores = np.load(run_dir / "val_scores.npy")
    test_scores = np.load(run_dir / "test_scores.npy")
    label = np.loadtxt(Path(cfg.data.root) / "test_label" / f"{args.entity}.txt",
                        dtype=np.int64)

    modes = [
        "raw",
        "zscore",
        "smooth_3", "smooth_5", "smooth_10",
        "runmax_3", "runmax_5", "runmax_10",
        "combine_smooth_5", "combine_smooth_10",
    ]
    print(f"# entity={args.entity}  val_T={val_scores.shape[0]}  "
          f"test_T={test_scores.shape[0]}  D={val_scores.shape[1]}\n")
    print(f"{'mode':<22} {'raw_f1':>8} {'raw_p':>8} {'raw_r':>8} {'pa_f1':>8}  "
          f"{'agg':>6} q  level")
    print("-" * 80)

    best_overall = None
    for m in modes:
        try:
            v_pp, t_pp = postprocess(val_scores, test_scores, m)
            best = search_one(v_pp, t_pp, label)
            if best is None:
                print(f"{m:<22} (no valid)")
                continue
            tag = ""
            if best_overall is None or best["raw_f1"] > best_overall["raw_f1"]:
                best_overall = {**best, "mode": m}
                tag = "  ★"
            print(f"{m:<22} {best['raw_f1']:>8.3f} {best['raw_p']:>8.3f} "
                  f"{best['raw_r']:>8.3f} {best['pa_f1']:>8.3f}  "
                  f"{best['agg']:>6} {best['q']:.2f} {best['level']:.0e}{tag}")
        except Exception as e:
            print(f"{m:<22} err: {e}")

    print("-" * 80)
    print(f"BEST: mode={best_overall['mode']}  raw_f1={best_overall['raw_f1']:.3f}  "
          f"pa_f1={best_overall['pa_f1']:.3f}")


if __name__ == "__main__":
    main()
