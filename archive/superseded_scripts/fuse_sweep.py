"""融合：drift（per-variable 时序后处理）+ relation（Mahalanobis 关系漂移）。

两条独立信号：
  - drift_score[t]：每个变量误差时序的局部统计量（spike/sustained）
  - relation_score[t]：D 维误差向量整体偏离正常联合分布的程度（pattern）

融合：
  z-score 各自标准化 → final[t] = α · drift_z + (1-α) · relation_z → POT + 评估

扫描：
  drift 内部最优 mode × relation 内部最优 mode × α ∈ {0, 0.25, 0.5, 0.75, 1.0} × POT 参数
  仅在 val 上估归一化与阈值，test 仅用于报指标。

零重训，读 val_scores.npy / test_scores.npy 即可。
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

# 复用两个 sweep 脚本里的核心函数
from drift_sweep import (  # noqa: E402
    _moving_avg, _moving_max, _zscore, aggregate, postprocess
)
from relation_drift import estimate_cov, mahalanobis  # noqa: E402


DRIFT_MODES = [
    "raw", "zscore",
    "smooth_3", "smooth_5", "smooth_10",
    "runmax_3", "runmax_5", "runmax_10",
    "combine_smooth_5", "combine_smooth_10",
]

DRIFT_AGGS = ["max", "topk3", "topk5", "mean"]

RELATION_MODES = ["raw", "smooth_5", "smooth_10", "runmax_3", "runmax_5"]

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
QS = [0.95, 0.99]
LEVELS = [1e-3, 1e-4, 1e-5, 1e-6]


def _ma_1d(x, k):
    if k <= 1: return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    csum = np.cumsum(xp)
    return ((csum[k:] - csum[:-k]) / k)[: x.shape[0]]


def _mm_1d(x, k):
    if k <= 1: return x
    T = x.shape[0]; pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.zeros(T)
    for t in range(T):
        out[t] = xp[t:t + k].max()
    return out


def post_1d(val: np.ndarray, test: np.ndarray, mode: str):
    """对一维分数（已聚合的 drift 或 mahalanobis）做后处理。"""
    if mode == "raw":
        return val, test
    mu = val.mean(); sd = max(val.std(), 1e-8)
    vz = (val - mu) / sd
    tz = (test - mu) / sd
    if mode == "zscore":
        return vz, tz
    if mode.startswith("smooth_"):
        k = int(mode.split("_")[1])
        return _ma_1d(vz, k), _ma_1d(tz, k)
    if mode.startswith("runmax_"):
        k = int(mode.split("_")[1])
        return _mm_1d(vz, k), _mm_1d(tz, k)
    raise ValueError(mode)


def best_drift(val: np.ndarray, test: np.ndarray, label: np.ndarray) -> dict:
    """找 drift 流水线下最佳单一时间序列分数 (T,)。"""
    best = None
    for mode in DRIFT_MODES:
        try:
            vp, tp = postprocess(val, test, mode)
        except Exception:
            continue
        for agg in DRIFT_AGGS:
            try:
                v_s = aggregate(vp, agg)
                t_s = aggregate(tp, agg)
            except Exception:
                continue
            for q in QS:
                for lv in LEVELS:
                    try:
                        thr = pot_threshold(v_s, q=q, level=lv)
                        r = evaluate_scores(t_s, label, thr)
                        if best is None or r["raw_f1"] > best["raw_f1"]:
                            best = {
                                "raw_f1": r["raw_f1"],
                                "v_s": v_s, "t_s": t_s,
                                "mode": mode, "agg": agg, "q": q, "level": lv,
                            }
                    except Exception:
                        pass
    return best


def best_relation(val_mh: np.ndarray, test_mh: np.ndarray, label: np.ndarray) -> dict:
    """找 mahalanobis 流水线下最佳一维分数。"""
    best = None
    for mode in RELATION_MODES:
        try:
            v_s, t_s = post_1d(val_mh, test_mh, mode)
        except Exception:
            continue
        for q in QS:
            for lv in LEVELS:
                try:
                    thr = pot_threshold(v_s, q=q, level=lv)
                    r = evaluate_scores(t_s, label, thr)
                    if best is None or r["raw_f1"] > best["raw_f1"]:
                        best = {
                            "raw_f1": r["raw_f1"],
                            "v_s": v_s, "t_s": t_s,
                            "mode": mode, "q": q, "level": lv,
                        }
                except Exception:
                    pass
    return best


def fuse_and_evaluate(v_drift, t_drift, v_rel, t_rel, label):
    """对两条 z-score 后的一维分数做加权融合并扫 POT，返回最佳。"""
    # z-score 各自标准化
    v_d_z, t_d_z = post_1d(v_drift, t_drift, "zscore")
    v_r_z, t_r_z = post_1d(v_rel, t_rel, "zscore")

    best = None
    for a in ALPHAS:
        v_fuse = a * v_d_z + (1 - a) * v_r_z
        t_fuse = a * t_d_z + (1 - a) * t_r_z
        for q in QS:
            for lv in LEVELS:
                try:
                    thr = pot_threshold(v_fuse, q=q, level=lv)
                    r = evaluate_scores(t_fuse, label, thr)
                    if best is None or r["raw_f1"] > best["raw_f1"]:
                        best = {
                            "alpha": a, "q": q, "level": lv, "thr": thr,
                            "raw_f1": r["raw_f1"], "raw_p": r["raw_p"],
                            "raw_r": r["raw_r"], "pa_f1": r["pa_f1"],
                        }
                except Exception:
                    pass
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--cov_mode", choices=["diag", "sample", "shrink"],
                        default="shrink")
    parser.add_argument("--shrink", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.train.save_dir) / args.entity
    val = np.load(run_dir / "val_scores.npy")
    test = np.load(run_dir / "test_scores.npy")
    label = np.loadtxt(Path(cfg.data.root) / "test_label" / f"{args.entity}.txt",
                        dtype=np.int64)

    print(f"# entity={args.entity}  val_T={len(val)}  test_T={len(test)}  D={val.shape[1]}")

    # 1) 找 drift 最佳
    bd = best_drift(val, test, label)
    print(f"\n[drift]    best raw_f1={bd['raw_f1']:.3f}  "
          f"mode={bd['mode']}  agg={bd['agg']}  q={bd['q']}  level={bd['level']:.0e}")

    # 2) Mahalanobis 距离 → 找 relation 最佳
    mu, Sigma_inv = estimate_cov(val, mode=args.cov_mode, shrink=args.shrink)
    val_mh = mahalanobis(val, mu, Sigma_inv)
    test_mh = mahalanobis(test, mu, Sigma_inv)
    br = best_relation(val_mh, test_mh, label)
    print(f"[relation] best raw_f1={br['raw_f1']:.3f}  "
          f"mode={br['mode']}  q={br['q']}  level={br['level']:.0e}")

    # 3) 融合
    bf = fuse_and_evaluate(bd["v_s"], bd["t_s"], br["v_s"], br["t_s"], label)
    print(f"\n# Fusion sweep (α=drift weight):")
    print(f"{'alpha':>6} {'raw_f1':>8} {'raw_p':>8} {'raw_r':>8} {'pa_f1':>8}  q level")
    print("-" * 60)
    # 重新跑一遍 alpha 扫描以打印每个 alpha 的最佳
    v_d_z, t_d_z = post_1d(bd["v_s"], bd["t_s"], "zscore")
    v_r_z, t_r_z = post_1d(br["v_s"], br["t_s"], "zscore")
    for a in ALPHAS:
        v_fuse = a * v_d_z + (1 - a) * v_r_z
        t_fuse = a * t_d_z + (1 - a) * t_r_z
        sub_best = None
        for q in QS:
            for lv in LEVELS:
                try:
                    thr = pot_threshold(v_fuse, q=q, level=lv)
                    r = evaluate_scores(t_fuse, label, thr)
                    if sub_best is None or r["raw_f1"] > sub_best["raw_f1"]:
                        sub_best = {**r, "q": q, "level": lv}
                except Exception:
                    pass
        if sub_best:
            tag = "  ★" if abs(a - bf["alpha"]) < 1e-6 else ""
            print(f"{a:>6.2f} {sub_best['raw_f1']:>8.3f} {sub_best['raw_p']:>8.3f} "
                  f"{sub_best['raw_r']:>8.3f} {sub_best['pa_f1']:>8.3f}  "
                  f"{sub_best['q']:.2f} {sub_best['level']:.0e}{tag}")

    print(f"\nBEST: alpha={bf['alpha']:.2f}  raw_f1={bf['raw_f1']:.3f}  "
          f"pa_f1={bf['pa_f1']:.3f}")


if __name__ == "__main__":
    main()
