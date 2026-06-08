"""关系漂移检测：Mahalanobis 距离 on per-timestep 误差向量。

核心思想（vs drift_sweep 的 per-variable 时序平滑）：
- drift_sweep 看的是"每个变量自己的误差时序"——单变量信号
- relation_drift 看的是"D 维误差向量整体偏离正常联合分布的程度"——多变量信号

数学：
  val 阶段估计 μ ∈ R^D, Σ ∈ R^{D×D}（样本均值与协方差，加 shrinkage 保正定）
  test 时刻 t: relation_score[t] = (E[t,:] - μ)ᵀ Σ⁻¹ (E[t,:] - μ)

什么时候 relation_score 比 drift_score 高：
  - 正常时：哪些变量同时出现高误差有固定模式，被 Σ 编码
  - 异常时：变量出错的"组合"不寻常 → Mahalanobis 距离飙升
  - 即使每个变量单看都正常，组合不寻常也会被检测到

零重训。读 val_scores.npy / test_scores.npy 即可。
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


# ---------------- Σ 估计 ----------------

def estimate_cov(val: np.ndarray, mode: str = "shrink", shrink: float = 0.1) -> tuple:
    """估计 (μ, Σ_inv)。返回 inv 而不是 Σ 是为了避免每次推理都 solve。

    mode:
      - 'diag'：对角线 Σ（等价于 per-variable z-score）
      - 'sample'：朴素样本协方差 + small ε·I 保正定
      - 'shrink'：Ledoit-Wolf 风格 shrinkage（推荐）
                  Σ_shrunk = (1-α) Σ_sample + α · (tr(Σ)/D) · I
    """
    T, D = val.shape
    mu = val.mean(axis=0)
    if mode == "diag":
        var = val.var(axis=0).clip(min=1e-8)
        Sigma_inv = np.diag(1.0 / var)
        return mu, Sigma_inv

    centered = val - mu
    S = centered.T @ centered / max(T - 1, 1)  # (D, D) 样本协方差

    if mode == "sample":
        S_reg = S + 1e-6 * np.eye(D)
    elif mode == "shrink":
        target = np.trace(S) / D * np.eye(D)
        S_reg = (1.0 - shrink) * S + shrink * target
    else:
        raise ValueError(mode)

    # 计算逆
    try:
        Sigma_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(S_reg)
    return mu, Sigma_inv


def mahalanobis(x: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
    """x: (T, D), 返回 (T,) Mahalanobis 距离平方。"""
    centered = x - mu
    # (T, D) @ (D, D) → (T, D), 然后 row-wise dot with centered
    out = np.einsum("ti,ij,tj->t", centered, Sigma_inv, centered)
    return out


# ---------------- 后处理（复用 drift_sweep 的思路） ----------------

def _moving_avg(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    csum = np.cumsum(xp)
    out = (csum[k:] - csum[:-k]) / k
    return out[: x.shape[0]]


def _moving_max(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    T = x.shape[0]
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.zeros(T)
    for t in range(T):
        out[t] = xp[t:t + k].max()
    return out


# ---------------- 评估 ----------------

def sweep_pot(val_s: np.ndarray, test_s: np.ndarray, label: np.ndarray) -> dict:
    best = None
    for q in [0.95, 0.99]:
        for lv in [1e-3, 1e-4, 1e-5, 1e-6]:
            try:
                thr = pot_threshold(val_s, q=q, level=lv)
                r = evaluate_scores(test_s, label, thr)
                if best is None or r["raw_f1"] > best["raw_f1"]:
                    best = {
                        "raw_f1": r["raw_f1"], "raw_p": r["raw_p"],
                        "raw_r": r["raw_r"], "pa_f1": r["pa_f1"],
                        "q": q, "level": lv, "thr": thr,
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
    parser.add_argument("--shrink", type=float, default=0.1,
                        help="Ledoit-Wolf shrinkage 系数 (0~1)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.train.save_dir) / args.entity
    val = np.load(run_dir / "val_scores.npy")     # (T_val, D)
    test = np.load(run_dir / "test_scores.npy")   # (T_test, D)
    label = np.loadtxt(Path(cfg.data.root) / "test_label" / f"{args.entity}.txt",
                        dtype=np.int64)
    T, D = val.shape
    print(f"# entity={args.entity}  D={D}  val_T={T}  test_T={test.shape[0]}\n")

    # 1) 估 μ, Σ_inv
    mu, Sigma_inv = estimate_cov(val, mode=args.cov_mode, shrink=args.shrink)

    # 2) Mahalanobis 距离
    val_m = mahalanobis(val, mu, Sigma_inv)
    test_m = mahalanobis(test, mu, Sigma_inv)

    print(f"# Sigma mode={args.cov_mode}  shrink={args.shrink}")
    print(f"# val mahalanobis stats: mean={val_m.mean():.2f}  "
          f"std={val_m.std():.2f}  max={val_m.max():.2f}")
    print(f"# test mahalanobis stats: mean={test_m.mean():.2f}  "
          f"std={test_m.std():.2f}  max={test_m.max():.2f}\n")

    # 3) 扫不同后处理（沿用 drift_sweep 的套路）
    variants = {
        "mahala_raw": (val_m, test_m),
        "mahala_smooth_5": (_moving_avg(val_m, 5), _moving_avg(test_m, 5)),
        "mahala_smooth_10": (_moving_avg(val_m, 10), _moving_avg(test_m, 10)),
        "mahala_runmax_3": (_moving_max(val_m, 3), _moving_max(test_m, 3)),
        "mahala_runmax_5": (_moving_max(val_m, 5), _moving_max(test_m, 5)),
    }

    print(f"{'mode':<22} {'raw_f1':>8} {'raw_p':>8} {'raw_r':>8} {'pa_f1':>8}  q level")
    print("-" * 72)
    best_overall = None
    for name, (v_s, t_s) in variants.items():
        b = sweep_pot(v_s, t_s, label)
        if b is None:
            print(f"{name:<22} (no valid)")
            continue
        tag = ""
        if best_overall is None or b["raw_f1"] > best_overall["raw_f1"]:
            best_overall = {**b, "mode": name}
            tag = "  ★"
        print(f"{name:<22} {b['raw_f1']:>8.3f} {b['raw_p']:>8.3f} "
              f"{b['raw_r']:>8.3f} {b['pa_f1']:>8.3f}  "
              f"{b['q']:.2f} {b['level']:.0e}{tag}")

    print("-" * 72)
    print(f"BEST: mode={best_overall['mode']}  raw_f1={best_overall['raw_f1']:.3f}  "
          f"pa_f1={best_overall['pa_f1']:.3f}")


if __name__ == "__main__":
    main()
