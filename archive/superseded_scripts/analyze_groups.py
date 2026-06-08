"""变量关系分析：每个变量的 top-K 相关邻居。

SMD 上变量没有清晰的分簇结构（一团弱相关 + 少数强相关对），硬聚类总会留下
giant cluster + singletons。改用 GDN 风格的**非对称邻接**：对每个变量 d 单独取
|corr| 最大的 K 个邻居作为它的"组内上下文"。

输出：data/groups/{entity}.json
  - neighbors_per_var: {var_id: [neighbor_ids]}  长度均为 K
  - corr_strength_per_var: {var_id: [|corr| with each neighbor]}（用于加权）
  - constant_vars: 常数列（std 接近 0），它们的 neighbors 为空

使用：
  python scripts/analyze_groups.py --config configs/smd.yaml --k 8
  python scripts/analyze_groups.py --config configs/smd.yaml --sweep    # 看不同 K 的统计
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.smd_dataset import load_smd_entity  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def compute_corr(x: np.ndarray, method: str = "spearman") -> np.ndarray:
    """返回 (D, D) 相关性矩阵。NaN/常数列对应位置置 0。"""
    if method == "pearson":
        c = np.corrcoef(x.T)
    elif method == "spearman":
        c, _ = stats.spearmanr(x, axis=0)
        if np.isscalar(c) or c.ndim == 0:
            c = np.array([[1.0]])
    else:
        raise ValueError(method)
    return np.nan_to_num(c, nan=0.0)


def topk_neighbors(corr: np.ndarray, k: int) -> tuple:
    """对每行取 |corr| 最大的 top-K 列（不含自己）。
    返回 (neighbors_idx, neighbors_strength)，shape 都是 (D, K)。
    """
    D = corr.shape[0]
    abs_corr = np.abs(corr).copy()
    np.fill_diagonal(abs_corr, -1.0)  # 排除自身
    k = min(k, D - 1)
    # argsort 降序取前 K
    order = np.argsort(-abs_corr, axis=1)
    idx = order[:, :k]
    strength = np.take_along_axis(abs_corr, idx, axis=1)
    return idx, strength


def analyze_entity(root: str, entity: str, val_ratio: float,
                   method: str = "spearman", k: int = 8,
                   eps_constant: float = 1e-3) -> dict:
    train_raw, _, _ = load_smd_entity(root, entity)
    n = len(train_raw)
    train_part = train_raw[: n - int(n * val_ratio)]
    D = train_part.shape[1]

    # 识别常数列
    stds = train_part.std(axis=0)
    is_constant = stds < eps_constant
    var_idx = np.where(~is_constant)[0]
    const_idx = np.where(is_constant)[0]

    # 对非常数列计算 corr 和 top-K
    neighbors_per_var = {}
    strength_per_var = {}
    if len(var_idx) >= 2:
        sub = train_part[:, var_idx]
        corr = compute_corr(sub, method=method)
        idx_local, str_local = topk_neighbors(corr, k=k)
        # 映射回原始变量索引
        for local_i, original_i in enumerate(var_idx):
            neighbors_per_var[int(original_i)] = [
                int(var_idx[j]) for j in idx_local[local_i]
            ]
            strength_per_var[int(original_i)] = [
                float(s) for s in str_local[local_i]
            ]

    # 常数列邻居为空（它们没有信号，推理时跳过即可）
    for c in const_idx:
        neighbors_per_var[int(c)] = []
        strength_per_var[int(c)] = []

    # 统计信息
    strengths_all = [s for vs in strength_per_var.values() for s in vs]
    mean_strength = float(np.mean(strengths_all)) if strengths_all else 0.0
    min_strength = float(np.min(strengths_all)) if strengths_all else 0.0
    max_strength = float(np.max(strengths_all)) if strengths_all else 0.0

    return {
        "entity": entity,
        "n_features": D,
        "method": method,
        "k_neighbors": k,
        "n_constant_vars": int(is_constant.sum()),
        "constant_vars": [int(c) for c in const_idx],
        "neighbors_per_var": {str(k_): v for k_, v in neighbors_per_var.items()},
        "corr_strength_per_var": {str(k_): v for k_, v in strength_per_var.items()},
        "stats": {
            "mean_neighbor_strength": mean_strength,
            "min_neighbor_strength": min_strength,
            "max_neighbor_strength": max_strength,
        },
    }


def sweep_k(root: str, entity: str, val_ratio: float, method: str,
            ks=(4, 6, 8, 10, 12, 16)) -> dict:
    """对单个 entity 跑多个 K，返回每个 K 下的邻居相关强度统计。"""
    out = {}
    for k in ks:
        info = analyze_entity(root, entity, val_ratio, method=method, k=k)
        out[k] = info["stats"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--entities", nargs="+", default=None,
                        help="entity 列表；缺省 = 训练集目录下所有")
    parser.add_argument("--method", choices=["pearson", "spearman"],
                        default="spearman", help="相关性度量")
    parser.add_argument("--k", type=int, default=8,
                        help="每个变量的邻居数")
    parser.add_argument("--out_dir", default="data/groups")
    parser.add_argument("--sweep", action="store_true",
                        help="对每个 entity 扫多个 K，看邻居相关强度变化")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg.data.root
    val_ratio = cfg.data.val_ratio

    if args.entities:
        entities = args.entities
    else:
        train_dir = Path(root) / "train"
        entities = sorted([p.stem for p in train_dir.glob("*.txt")])

    # 模式 1：扫 K，看邻居相关强度
    if args.sweep:
        ks = [4, 6, 8, 10, 12, 16]
        print(f"# K sweep on {len(entities)} entities, method={args.method}")
        print(f"# 数字 = K 个邻居中最弱那个的 |corr|（越高说明 K 越小就够了）\n")
        header = f"{'entity':<15}  " + "  ".join(f"K={k:<3d}" for k in ks)
        print(header)
        print("-" * len(header))
        for e in entities:
            s = sweep_k(root, e, val_ratio, args.method, ks)
            cells = []
            for k in ks:
                cells.append(f"{s[k]['min_neighbor_strength']:>5.2f}")
            print(f"{e:<15}  " + "  ".join(cells))
        print("\n# 选 K：找到 |corr| 还 ≥ 0.3 的最大 K（再大邻居就是噪声了）")
        return

    # 模式 2：用指定 K 保存 JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# Analyzing {len(entities)} entities  method={args.method}  k={args.k}\n")
    print(f"{'entity':<15} {'D':>4} {'n_const':>8} "
          f"{'mean_str':>10} {'min_str':>9}")
    print("-" * 50)

    for e in entities:
        info = analyze_entity(root, e, val_ratio, method=args.method, k=args.k)
        out_path = out_dir / f"{e}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        s = info["stats"]
        print(f"{e:<15} {info['n_features']:>4d} "
              f"{info['n_constant_vars']:>8d} "
              f"{s['mean_neighbor_strength']:>10.3f} "
              f"{s['min_neighbor_strength']:>9.3f}")

    print(f"\n# Saved {len(entities)} JSON files to {out_dir}/")


if __name__ == "__main__":
    main()
